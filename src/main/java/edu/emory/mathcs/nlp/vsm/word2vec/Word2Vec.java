/**
 * Copyright 2015, Emory University
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package edu.emory.mathcs.nlp.vsm.word2vec;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Function;

import org.kohsuke.args4j.Option;

import edu.emory.mathcs.nlp.common.BinUtils;
import edu.emory.mathcs.nlp.common.FileUtils;
import edu.emory.mathcs.nlp.common.MathUtils;
import edu.emory.mathcs.nlp.common.collection.atomic.AtomicDouble;
import edu.emory.mathcs.nlp.common.constant.StringConst;

/**
 * @author Jinho D. Choi ({@code jinho.choi@emory.edu})
 * http://arxiv.org/pdf/1301.3781.pdf
 * http://www-personal.umich.edu/~ronxin/pdf/w2vexp.pdf
 */
public class Word2Vec
{
	@Option(name="-train", usage="path to the training file or the directory containig the training files.", required=true, metaVar="<filepath>")
    String train_path = null;
	@Option(name="-output", usage="output file to save the resulting word vectors.", required=true, metaVar="<filename>")
    String output_file = null;
	@Option(name="-ext", usage="extension of the training files (default: \"*\").", required=false, metaVar="<string>")
    String train_ext = "*";
	@Option(name="-size", usage="size of word vectors (default: 100).", required=false, metaVar="<int>")
    int vector_size = 100;
	@Option(name="-window", usage="max-window of contextual words (default: 5).", required=false, metaVar="<int>")
    int max_skip_window = 5;
	@Option(name="-sample", usage="threshold for occurrence of words (default: 1e-3). Those that appear with higher frequency in the training data will be randomly down-sampled.", required=false, metaVar="<double>")
    double sample_threshold = 1e-3;
	@Option(name="-negative", usage="number of negative examples (default: 5; common values are 3 - 10). If negative = 0, use Hierarchical Softmax instead of Negative Sampling.", required=false, metaVar="<int>")
    int num_negative = 5;
	@Option(name="-threads", usage="number of threads (default: 12).", required=false, metaVar="<int>")
    int num_threads = 12;
	@Option(name="-iter", usage="number of training iterations (default: 5).", required=false, metaVar="<int>")
    int train_iteration = 5;
	@Option(name="-min-count", usage="min-count of words (default: 5). This will discard words that appear less than <int> times.", required=false, metaVar="<int>")
    int min_count = 5;
	@Option(name="-alpha", usage="initial learning rate (default: 0.025 for skip-gram; use 0.05 for CBOW).", required=false, metaVar="<double>")
	double alpha_init = 0.025;
	@Option(name="-binary", usage="If set, save the resulting vectors in binary moded.", required=false, metaVar="<boolean>")
	boolean binary = false;
	@Option(name="-cbow", usage="If set, use the continuous bag-of-words model instead of the skip-gram model.", required=false, metaVar="<boolean>")
	boolean cbow = false;
	
	final int VOCAB_REDUCE_SIZE = 21000000;
	final int MAX_SENTENCE_LENGTH = 1000;
	final int MAX_CODE_LENGTH = 40;
	
	volatile long word_count_global;	// word count dynamically updated by all threads
	volatile double alpha_global;		// learning rate dynamically updated by all threads
	volatile float[] syn0, syn1;
	
	
	Vocabulary vocab;
	NegativeSampling negative;
	Sigmoid sigmoid;
	long word_count_train;
	
	public Word2Vec(String[] args)
	{
		BinUtils.initArgs(args, this);
		sigmoid = new Sigmoid();

		try
		{
			train(FileUtils.getFileList(train_path, "*", false));
		}
		catch (Exception e) {e.printStackTrace();}
	}
	
//	=================================== Training ===================================
	
	public void train(List<String> trainFiles) throws Exception
	{
		BinUtils.LOG.info("Reading vocabulary:");
		vocab = new Vocabulary();
		learnVocabulary(trainFiles);
		BinUtils.LOG.info(String.format("\n- types = %d, words = %d\n", vocab.size(), word_count_train));
		
		BinUtils.LOG.info("Initializing neural network.\n");
		initNeuralNetwork();
		
		if (num_negative > 0)
		{
			BinUtils.LOG.info("Initializing negative sampling.\n");
			negative = new NegativeSampling();
		}

		BinUtils.LOG.info("Training word vectors:");
		ExecutorService executor = Executors.newFixedThreadPool(num_threads);
		word_count_global = 0;
		alpha_global = alpha_init;
		
		
//		for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
//		for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);

		BinUtils.LOG.info("Saving word vectors.\n");
		saveModel();
	}
	
	class TrainTask implements Runnable
	{
		@Override
		public void run()
		{
			
		}
		
	}
	
	void TrainModelThread(String trainFile, int id) throws IOException
	{
		int local_iter = train_iteration;
		long random = (long)id;
		
		int sentence_length = 0, word_position = 0, word, window, cw, i, k, c;
		long word_count = 0;
		double d, e;
		
		
		
		int[] sen = new int[MAX_SENTENCE_LENGTH];
		long label;
		float[] neu1  = new float[vector_size];
		float[] neu1e = new float[vector_size];
		
		WordReader in = new WordReader(new BufferedInputStream(new FileInputStream(trainFile)));
		int target, l1, l2, a, b;
		
		while (true)
		{
			// read next sentence
			if (sentence_length == 0)
			{
				while (true)
				{
					word = readWordIndex(in);
					if (word == Vocabulary.OOV) continue;
					if (word == Vocabulary.EOL || word == Vocabulary.EOF) break;
					word_count++;
					
					// sub-sampling: randomly discards frequent words
					if (sample_threshold > 0)
					{
						d = sample_threshold * word_count_train;
						e = (Math.sqrt(MathUtils.divide(vocab.get(word).count, d)) + 1) * (d / vocab.get(word).count);
						if (e < MathUtils.divide(nextRandom(random) & 0xFFFF, 65536)) continue;
					}
					
					sen[sentence_length++] = word;
					if (sentence_length >= MAX_SENTENCE_LENGTH) break;
				}
				
				word_position = 0;
			}
			
			// no more context
			if (sentence_length == 0)
			{
				word_count_global += word_count;
				alpha_global = alpha_init * (1 - MathUtils.divide(word_count_global, train_iteration * word_count_train + 1));
				if (alpha_global < alpha_init * 0.0001) alpha_global = alpha_init * 0.0001;
				if (--local_iter == 0) break;
				
				word_count = 0;
				sentence_length = 0;
				in.close(); in.init(new BufferedInputStream(new FileInputStream(trainFile)));
				continue;
			}
		
			Arrays.fill(neu1 , 0);
			Arrays.fill(neu1e, 0);
			window = (int)(nextRandom(random) % max_skip_window);
			word   = sen[word_position];
			
			if (cbow)	// continuous bag-of-words
			{
				// in -> hidden
				cw = 0;
				
				for (i=-window; i<=window; i++)
				{
					if (i == 0) continue;
					c = word_position + i;
					if (sentence_length <= c || c < 0) continue;
					k = sen[c] * vector_size;
					for (c=0; c<vector_size; c++) neu1[c] += syn0[c + k];
					cw++;
				}
				
				if (cw > 0)
				{
					for (c=0; c<vector_size; c++) neu1[c] /= cw;
					
					if (num_negative > 0) for (k = 0; k < num_negative + 1; k++)	// negative sampling
					{
						if (k == 0)
						{
							target = word;
							label  = 1;
						}
						else
						{
							random = nextRandom(random);
							target = unigram_table[(int)((long)(random >> 16) % table_size)];
							if (target == 0) target = (int)(random % (vocab.size() - 1)) + 1;
							if (target == word) continue;
							label = 0;
						}
						
						l2 = target * vector_size;
						d = 0;
						for (c = 0; c < vector_size; c++) d += neu1[c] * syn1[c + l2];
						if (d > MAX_EXP) e = (label - 1) * alpha_global;
						else if (d < -MAX_EXP) e = (label - 0) * alpha_global;
						else e = (label - sigmoid[(int)((d + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha_global;
						for (c = 0; c < vector_size; c++) neu1e[c] += e * syn1[c + l2];
						for (c = 0; c < vector_size; c++) syn1[c + l2] += e * neu1[c];
					}
					
					
					
					
					
					
					
					
					else for (k=0; k < vocab.get(word).codeLength(); k++)		// hierarchical softmax
					{
						d = 0;
						l2 = vocab.get(word).point[k] * vector_size;
						// Propagate hidden -> output
						for (c = 0; c < vector_size; c++) d += neu1[c] * syn1[c + l2];
						if (d <= -MAX_EXP) continue;
						else if (d >= MAX_EXP) continue;
						else d = sigmoid[(int)((d + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
						// 'g' is the gradient multiplied by the learning rate
						e = (1 - vocab.get(word).code[k] - d) * alpha_global;
						// Propagate errors output -> hidden
						for (c = 0; c < vector_size; c++) neu1e[c] += e * syn1[c + l2];
						// Learn weights hidden -> output
						for (c = 0; c < vector_size; c++) syn1[c + l2] += e * neu1[c];
					}
						
					// NEGATIVE SAMPLING
					
					
					// hidden -> in
					for (a = b; a < max_skip_window * 2 + 1 - b; a++) if (a != max_skip_window)
					{
						c = word_position - max_skip_window + a;
						if (c < 0) continue;
						if (c >= sentence_length) continue;
						context = sen[c];
						if (context == -1) continue;
						for (c = 0; c < vector_size; c++) syn0[c + context * vector_size] += neu1e[c];
					}
				}
			}
			else	// continuous skip-gram
			{
				for (a = b; a < max_skip_window * 2 + 1 - b; a++) if (a != max_skip_window)
				{
					c = word_position - max_skip_window + a;
					if (c < 0) continue;
					if (c >= sentence_length) continue;
					context = sen[c];
					if (context == -1) continue;
					l1 = context * vector_size;
					Arrays.fill(neu1e, 0);
					
					// HIERARCHICAL SOFTMAX
					if (hs) for (k = 0; k < vocab.get(word).codeLength(); k++)
					{
						d = 0;
						l2 = vocab.get(word).point[k] * vector_size;
						// Propagate hidden -> output
						for (c = 0; c < vector_size; c++) d += syn0[c + l1] * syn1[c + l2];
						if (d <= -MAX_EXP) continue;
						else if (d >= MAX_EXP) continue;
						else d = sigmoid[(int)((d + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
						// 'g' is the gradient multiplied by the learning rate
						e = (1 - vocab.get(word).code[k] - d) * alpha_global;
						// Propagate errors output -> hidden
						for (c = 0; c < vector_size; c++) neu1e[c] += e * syn1[c + l2];
						// Learn weights hidden -> output
						for (c = 0; c < vector_size; c++) syn1[c + l2] += e * syn0[c + l1];
					}
					
					// NEGATIVE SAMPLING
					if (num_negative > 0) for (k = 0; k < num_negative + 1; k++)
					{
						if (k == 0)
						{
							target = word;
							label = 1;
						}
						else
						{
							random = nextRandom(random);
							target = unigram_table[(int)((long)(random >> 16) % table_size)];
							if (target == 0) target = (int)(random % (vocab.size() - 1)) + 1;
							if (target == word) continue;
							label = 0;
						}
						
						l2 = target * vector_size;
						d = 0;
						for (c = 0; c < vector_size; c++) d += syn0[c + l1] * syn1neg[c + l2];
						if (d > MAX_EXP) e = (label - 1) * alpha_global;
						else if (d < -MAX_EXP) e = (label - 0) * alpha_global;
						else e = (label - sigmoid[(int)((d + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha_global;
						for (c = 0; c < vector_size; c++) neu1e[c] += e * syn1neg[c + l2];
						for (c = 0; c < vector_size; c++) syn1neg[c + l2] += e * syn0[c + l1];
					}
					
					// Learn weights input -> hidden
					for (c = 0; c < vector_size; c++) syn0[c + l1] += neu1e[c];
				}
			}
			
			if (++word_position >= sentence_length)
			{
				sentence_length = 0;
				continue;
			}
		}
		
		in.close();
	}
	
	void readSentence(String[] sen)
	{
		
	}
	
//	=================================== Helper Methods ===================================
	
	void initNeuralNetwork()
	{
		int size = vocab.size() * vector_size;
		long random = 1;
		
		syn0 = new float[size];
		syn1 = new float[size];
		
		for (int i=0; i<size; i++)
		{
			random = Word2Vec.nextRandom(random);
			syn0[i] = (float)((MathUtils.divide(random & 0xFFFF, 65536) - 0.5) / vector_size);
		}
	}
	
	
	
	/** @return the index of the next word in the reader ; if the word is not in the vocabulary, {@link Vocabulary#OOV}; if no word, {@link Vocabulary#EOF}. */
	int readWordIndex(WordReader in) throws IOException
	{
		String word = in.read();
		return (word == null) ? Vocabulary.EOF : StringConst.NEW_LINE.equals(word) ? Vocabulary.EOF : vocab.indexOf(word);
	}
	
	void saveModel() throws IOException
	{
		PrintStream out = new PrintStream(new BufferedOutputStream(new FileOutputStream(output_file)));
		network.save(out, binary);
		out.close();
	}
	
	static long nextRandom(long prev)
	{
		return prev * 25214903917L + 11;
	}
	
//	=================================== Neural Network ===================================

	class NegativeSampling
	{
		final int TABLE_SIZE = (int)1e8;
		final double DIST_POWER = 0.75;
		int[] dist_table;
		int num_samples;
		
		public NegativeSampling(int numSamples)
		{
			num_samples = numSamples;
			initDistributionTable();
		}
		
		void initDistributionTable()
		{
			double d, Z = vocab.list().stream().mapToDouble(v -> Math.pow(v.count, DIST_POWER)).sum();
			int i = 0, j;
			
			dist_table = new int[TABLE_SIZE];
			d = nextDistribution(i, Z);
			
			for (j=0; j<TABLE_SIZE && i<vocab.size(); j++)
			{
				dist_table[j] = i;
				
				if (MathUtils.divide(j, TABLE_SIZE) > d)
					d += nextDistribution(++i, Z);
			}
			
			if (j < TABLE_SIZE)
				Arrays.fill(dist_table, j, TABLE_SIZE, vocab.size()-1);
		}
		
		double nextDistribution(int index, double Z)
		{
			return Math.pow(vocab.get(index).count, DIST_POWER) / Z;
		}
		
		public void sample(int word)
		{
			int target;
			
			for (int k=0; k<=num_negative; k++)
			{
				if (k == 0)
				{
					target = word;
					label = 1;
				}
				else
				{
					random = nextRandom(random);
					target = unigram_table[(int)((long)(random >> 16) % table_size)];
					if (target == 0) target = (int)(random % (vocab.size() - 1)) + 1;
					if (target == word) continue;
					label = 0;
				}
				
				
			}
		}
		
		void sample(int target, int label, boolean cbow, float[] neu1)
		{
			int i, l2 = target * vector_size;
			double d = 0;
			
			for (i=0; i<vector_size; i++)
				d += cbow ? neu1[i] * syn1[i+l2] : syn0[i + l1] * syn1[i+l2];
			
			
			
			if (d > MAX_EXP) e = (label - 1) * alpha_global;
			else if (d < -MAX_EXP) e = (label - 0) * alpha_global;
			else e = (label - sigmoid[(int)((d + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha_global;
			for (i = 0; i < vector_size; i++) neu1e[i] += e * syn1neg[i + l2];
			for (i = 0; i < vector_size; i++) syn1neg[i + l2] += e * neu1[i];
			
			
			
			
			
			for (i = 0; i < vector_size; i++) d += syn0[i + l1] * syn1neg[i + l2];
			if (d > MAX_EXP) e = (label - 1) * alpha_global;
			else if (d < -MAX_EXP) e = (label - 0) * alpha_global;
			else e = (label - sigmoid[(int)((d + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha_global;
			for (i = 0; i < vector_size; i++) neu1e[i] += e * syn1neg[i + l2];
			for (i = 0; i < vector_size; i++) syn1neg[i + l2] += e * syn0[i + l1];
		}
	}
}
