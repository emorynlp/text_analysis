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
import java.io.InputStream;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.kohsuke.args4j.Option;

import edu.emory.mathcs.nlp.common.BinUtils;
import edu.emory.mathcs.nlp.common.FileUtils;
import edu.emory.mathcs.nlp.common.MathUtils;

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
	@Option(name="-sample", usage="threshold for occurrence of words (default: 1e-3). Those that appear with higher frequency in the training data will be randomly down-sampled.", required=false, metaVar="<float>")
    float sample_threshold = 1e-3f;
	@Option(name="-negative", usage="number of negative examples (default: 5; common values are 3 - 10). If negative = 0, use Hierarchical Softmax instead of Negative Sampling.", required=false, metaVar="<int>")
    int num_negative = 5;
	@Option(name="-threads", usage="number of threads (default: 12).", required=false, metaVar="<int>")
    int num_threads = 12;
	@Option(name="-iter", usage="number of training iterations (default: 5).", required=false, metaVar="<int>")
    int train_iter = 5;
	@Option(name="-min-count", usage="min-count of words (default: 5). This will discard words that appear less than <int> times.", required=false, metaVar="<int>")
    int min_count = 5;
	@Option(name="-alpha", usage="starting learning rate (default: 0.025 for skip-gram; use 0.05 for CBOW).", required=false, metaVar="<float>")
	float alpha = 0.025f;
	@Option(name="-binary", usage="If set, save the resulting vectors in binary moded.", required=false, metaVar="<boolean>")
	boolean binary = false;
	@Option(name="-cbow", usage="If set, use the continuous bag-of-words model instead of the skip-gram model.", required=false, metaVar="<boolean>")
	boolean cbow = false;
	
	final int VOCAB_REDUCE_SIZE = 21000000;
	final int MAX_SENTENCE_LENGTH = 1000;
	final int MAX_CODE_LENGTH = 40;
	final float ALPHA_INIT;
	
	volatile long word_count;	// total word count from all threads
	volatile NeuralNetwork network;
	Vocabulary vocab;
	NegativeSampling negative;
	SigmoidTable sigmoid;
	long train_words;
	
	public Word2Vec(String[] args)
	{
		BinUtils.initArgs(args, this);
		sigmoid = new SigmoidTable();
		ALPHA_INIT = alpha;

		try
		{
			train(FileUtils.getFileList(train_path, "*", false));
		}
		catch (Exception e) {e.printStackTrace();}
	}
	
//	=================================== Training ===================================
	
	public void train(List<String> trainFiles) throws Exception
	{
		vocab = new Vocabulary();
		
		BinUtils.LOG.info("Reading vocabulary:");
		learnVocabulary(trainFiles);
		BinUtils.LOG.info(String.format("\n- types = %d, words = %d\n", vocab.size(), train_words));
		
		BinUtils.LOG.info("Initializing neural network.\n");
		network = new NeuralNetwork(vocab, vector_size);
		
		if (num_negative > 0)
		{
			BinUtils.LOG.info("Initializing negative sampling.\n");
			negative = new NegativeSampling();
		}

		BinUtils.LOG.info("Training word vectors:");
		ExecutorService executor = Executors.newFixedThreadPool(num_threads);
		word_count = 0;
//		for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
//		for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);

		BinUtils.LOG.info("Saving word vectors.\n");
		saveModel();
	}
	
	void TrainModelThread(String trainFile, int id) throws IOException
	{
		long cw;
		long wc = 0, lwc = 0;
		int[] sen = new int[MAX_SENTENCE_LENGTH];
		long label, local_iter = train_iter;
		long random = (long)id;
		float f, g, ran;
		float[] neu1  = new float[vector_size];
		float[] neu1e = new float[vector_size];
		
		InputStream fi = new BufferedInputStream(new FileInputStream(trainFile));
		int word = Const.EOF, target, l1, l2, last_word, sentence_length = 0, sentence_position = 0, c, a, b, d;
		
		while (true)
		{
			// TODO: weighting later instances less; could be a issue when samples are not randomly drawn.
			if (wc - lwc > 10000)
			{
				word_count += wc - lwc;
				lwc = wc;
				alpha = (float)(ALPHA_INIT * (1 - MathUtils.divide(word_count, train_iter * train_words + 1)));
				if (alpha < ALPHA_INIT * 0.0001) alpha = ALPHA_INIT * 0.0001f;
		    }
			
			if (sentence_length == 0)
			{
				while (true)
				{
					word = ReadWordIndex(fi);
					if (word == EOF) break;		// end-of-file
					if (word == OOV) continue;	// out-of-vocab
					wc++;
					if (word == EOL) break;		// end-of-line
					
					// sub-sampling randomly discards frequent words while keeping the ranking same
					if (sample_threshold > 0)
					{
						ran = (Math.sqrt((float)vocab.get(word).count / (sample_threshold * train_words)) + 1) * (sample_threshold * train_words) / vocab.get(word).count;
						random = nextRandom(random);
						if (ran < MathUtils.divide(random & 0xFFFF, 65536)) continue;
					}
					
					sen[sentence_length++] = word;
					if (sentence_length >= MAX_SENTENCE_LENGTH) break;
				}
				
				sentence_position = 0;
			}
			
			if (word == EOF)
			{
				word_count += wc - lwc;
				if (--local_iter == 0) break;
				wc = 0;
				lwc = 0;
				sentence_length = 0;
				fi.close();
				fi = new BufferedInputStream(new FileInputStream(trainFile));
				continue;
			}
		
			word = sen[sentence_position];
			if (word == -1) continue;
			random = nextRandom(random);
			b = (int)(random % max_skip_window);
			
			if (cbow)	// continuous bag-of-words architecture
			{
				// in -> hidden
				cw = 0;
				
				for (a=b; a < max_skip_window*2+1-b; a++)
				{
					if (a == max_skip_window) continue;
					c = sentence_position - max_skip_window + a;
					if (sentence_length <= c || c < 0) continue;
					last_word = sen[c];
					if (last_word == -1) continue;
					d = last_word * vector_size;
					for (c=0; c<vector_size; c++) neu1[c] += syn0[c + d];
					cw++;
				}
				
				if (cw > 0)
				{
					for (c=0; c<vector_size; c++) neu1[c] /= cw;
					
					if (hs) for (d=0; d < vocab.get(word).codeLength(); d++)
					{
						f = 0;
						l2 = vocab.get(word).point[d] * vector_size;
						// Propagate hidden -> output
						for (c = 0; c < vector_size; c++) f += neu1[c] * syn1[c + l2];
						if (f <= -MAX_EXP) continue;
						else if (f >= MAX_EXP) continue;
						else f = sigmoid[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
						// 'g' is the gradient multiplied by the learning rate
						g = (1 - vocab.get(word).code[d] - f) * alpha;
						// Propagate errors output -> hidden
						for (c = 0; c < vector_size; c++) neu1e[c] += g * syn1[c + l2];
						// Learn weights hidden -> output
						for (c = 0; c < vector_size; c++) syn1[c + l2] += g * neu1[c];
					}
					
					// NEGATIVE SAMPLING
					if (num_negative > 0) for (d = 0; d < num_negative + 1; d++)
					{
						if (d == 0)
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
						f = 0;
						for (c = 0; c < vector_size; c++) f += neu1[c] * syn1neg[c + l2];
						if (f > MAX_EXP) g = (label - 1) * alpha;
						else if (f < -MAX_EXP) g = (label - 0) * alpha;
						else g = (label - sigmoid[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
						for (c = 0; c < vector_size; c++) neu1e[c] += g * syn1neg[c + l2];
						for (c = 0; c < vector_size; c++) syn1neg[c + l2] += g * neu1[c];
					}
					
					// hidden -> in
					for (a = b; a < max_skip_window * 2 + 1 - b; a++) if (a != max_skip_window)
					{
						c = sentence_position - max_skip_window + a;
						if (c < 0) continue;
						if (c >= sentence_length) continue;
						last_word = sen[c];
						if (last_word == -1) continue;
						for (c = 0; c < vector_size; c++) syn0[c + last_word * vector_size] += neu1e[c];
					}
				}
			}
			else	//train skip-gram
			{
				for (a = b; a < max_skip_window * 2 + 1 - b; a++) if (a != max_skip_window)
				{
					c = sentence_position - max_skip_window + a;
					if (c < 0) continue;
					if (c >= sentence_length) continue;
					last_word = sen[c];
					if (last_word == -1) continue;
					l1 = last_word * vector_size;
					Arrays.fill(neu1e, 0);
					
					// HIERARCHICAL SOFTMAX
					if (hs) for (d = 0; d < vocab.get(word).codeLength(); d++)
					{
						f = 0;
						l2 = vocab.get(word).point[d] * vector_size;
						// Propagate hidden -> output
						for (c = 0; c < vector_size; c++) f += syn0[c + l1] * syn1[c + l2];
						if (f <= -MAX_EXP) continue;
						else if (f >= MAX_EXP) continue;
						else f = sigmoid[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
						// 'g' is the gradient multiplied by the learning rate
						g = (1 - vocab.get(word).code[d] - f) * alpha;
						// Propagate errors output -> hidden
						for (c = 0; c < vector_size; c++) neu1e[c] += g * syn1[c + l2];
						// Learn weights hidden -> output
						for (c = 0; c < vector_size; c++) syn1[c + l2] += g * syn0[c + l1];
					}
					
					// NEGATIVE SAMPLING
					if (num_negative > 0) for (d = 0; d < num_negative + 1; d++)
					{
						if (d == 0)
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
						f = 0;
						for (c = 0; c < vector_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
						if (f > MAX_EXP) g = (label - 1) * alpha;
						else if (f < -MAX_EXP) g = (label - 0) * alpha;
						else g = (label - sigmoid[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
						for (c = 0; c < vector_size; c++) neu1e[c] += g * syn1neg[c + l2];
						for (c = 0; c < vector_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
					}
					
					// Learn weights input -> hidden
					for (c = 0; c < vector_size; c++) syn0[c + l1] += neu1e[c];
				}
			}
			
			sentence_position++;
			
			if (sentence_position >= sentence_length)
			{
				sentence_length = 0;
				continue;
			}
		}
		
		fi.close();
	}
	
//	=================================== Helper Methods ===================================
	
	void learnVocabulary(List<String> filenames) throws IOException
	{
		InputStream in;
		
		for (String filename : filenames)
		{
			in = new BufferedInputStream(new FileInputStream(filename));
			vocab.addAll(in, VOCAB_REDUCE_SIZE);
			in.close(); BinUtils.LOG.debug(".");
		}
		
		train_words = vocab.sort(min_count);
		vocab.generateHuffmanCodes(MAX_CODE_LENGTH);
	}
	
	/** @return the index of the next word in the reader ; if the word is not in the vocabulary, {@link Const#OOV}; if no word, {@link Const#EOF}. */
	int ReadWordIndex(InputStream fin) throws IOException
	{
		String word = vocab.read(fin);
		return (word != null) ? vocab.indexOf(word) : Const.EOF;
	}
	
	void saveModel() throws IOException
	{
		PrintStream out = new PrintStream(new BufferedOutputStream(new FileOutputStream(output_file)));
		network.save(out, binary);
		out.close();
	}
	
//	=================================== Neural Network ===================================
	
	
	
	class NegativeSampling
	{
		final int TABLE_SIZE = (int)1e8;
		final double DIST_POWER = 0.75;
		int[] dist_table;
		
		public NegativeSampling()
		{
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
	}
}
