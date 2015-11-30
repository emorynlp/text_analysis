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
package edu.emory.mathcs.nlp.text_analysis.word2vec;

import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.kohsuke.args4j.Option;

import edu.emory.mathcs.nlp.common.random.XORShiftRandom;
import edu.emory.mathcs.nlp.common.util.BinUtils;
import edu.emory.mathcs.nlp.common.util.FileUtils;
import edu.emory.mathcs.nlp.common.util.IOUtils;
import edu.emory.mathcs.nlp.common.util.MathUtils;
import edu.emory.mathcs.nlp.common.util.Sigmoid;
import edu.emory.mathcs.nlp.text_analysis.word2vec.optimizer.HierarchicalSoftmax;
import edu.emory.mathcs.nlp.text_analysis.word2vec.optimizer.NegativeSampling;
import edu.emory.mathcs.nlp.text_analysis.word2vec.optimizer.Optimizer;
import edu.emory.mathcs.nlp.text_analysis.word2vec.reader.SentenceReader;
import edu.emory.mathcs.nlp.text_analysis.word2vec.util.Vocabulary;

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
	float subsample_threshold = 0.001f;
	@Option(name="-negative", usage="number of negative examples (default: 5; common values are 3 - 10). If negative = 0, use Hierarchical Softmax instead of Negative Sampling.", required=false, metaVar="<int>")
	int negative_size = 5;
	@Option(name="-threads", usage="number of threads (default: 12).", required=false, metaVar="<int>")
	int thread_size = 12;
	@Option(name="-iter", usage="number of training iterations (default: 5).", required=false, metaVar="<int>")
	int train_iteration = 5;
	@Option(name="-min-count", usage="min-count of words (default: 5). This will discard words that appear less than <int> times.", required=false, metaVar="<int>")
	int min_count = 5;
	@Option(name="-alpha", usage="initial learning rate (default: 0.025 for skip-gram; use 0.05 for CBOW).", required=false, metaVar="<float>")
	float alpha_init = 0.025f;
	@Option(name="-binary", usage="If set, save the resulting vectors in binary moded.", required=false, metaVar="<boolean>")
	boolean binary = false;
	@Option(name="-cbow", usage="If set, use the continuous bag-of-words model instead of the skip-gram model.", required=false, metaVar="<boolean>")
	boolean cbow = false;
	@Option(name="-normalize", usage="If set, normalize each vector.", required=false, metaVar="<boolean>")
	boolean normalize = true;
	@Option(name="-evaluate", usage="If set, reserve section of input to evaluate vectors.", required=false, metaVar="<boolean>")
	boolean evaluate = false;
	
	final float ALPHA_MIN_RATE  = 0.0001f;      
	final int   MAX_CODE_LENGTH = 40;
	
	Sigmoid sigmoid;
	Vocabulary vocab;
	long word_count_train;
	float subsample_size;
	Optimizer optimizer;
	
	volatile long word_count_global;	// word count dynamically updated by all threads
	volatile float alpha_global;		// learning rate dynamically updated by all threads
	volatile public float[] W;			// weights between the input and the hidden layers
	volatile public float[] V;			// weights between the hidden and the output layers
	
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
	
	public void train(List<String> filenames) throws Exception
	{
		List<File> files = new List<File>();
		for(String filename : filenames)
			files.add(new File(filename));
		Reader<?> reader = new SentenceReader(files);
		Reader<?>[] r = evaluate ? reader.trainingAndTest() : null;
		
		Reader<?> training_reader = evaluate ? r[0] : reader;
		Reader<?> test_reader = evaluate ? r[1] : null;
		
		
		BinUtils.LOG.info("Reading vocabulary:\n");
		vocab = new Vocabulary();
		vocab.learn(training_reader, min_count);
		word_count_train = vocab.totalWords();
		BinUtils.LOG.info(String.format("- types = %d, tokens = %d\n", vocab.size(), word_count_train));
		
		BinUtils.LOG.info("Initializing neural network.\n");
		initNeuralNetwork();
		
		BinUtils.LOG.info("Initializing optimizer.\n");
		optimizer = isNegativeSampling() ? new NegativeSampling(vocab, sigmoid, vector_size, negative_size) : new HierarchicalSoftmax(vocab, sigmoid, vector_size);

		BinUtils.LOG.info("Training vectors:");
		word_count_global = 0;
		alpha_global      = alpha_init;
		subsample_size    = subsample_threshold * word_count_train;
		
		startThreads(training_reader, false);
		
		if(evaluate){
			startThreads(test_reader, true);
			System.out.println("Evaluated Error: " + optimizer.getError());
		}
		
		BinUtils.LOG.info("Saving word vectors.\n");
		save();
	}
	
	void startThreads(Reader<?> reader, boolean evaluate) throws IOException
	{
		Reader<?>[] readers = reader.split(threads);
		reader.close();
		ExecutorService executor = Executors.newFixedThreadPool(threads);
		for (int i = 0; i < threads; i++)
			executor.execute(new TrainTask(readers[i], i, evaluate));
			
		executor.shutdown();			
		try {
			executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
			
		for (int i = 0; i < threads; i++)
			readers[i].close();
	}
	
	class TrainTask implements Runnable
	{
		private Reader<?> reader;
		private final int id;
		private boolean evaluate;
		
		public TrainTask(Reader<?> reader, int id, boolean evaluate)
		{
			this.reader = reader;
			this.random = random;
			this.id = id;
			this.evaluate = evaluate;
		}
		
		@Override
		public void run()
		{
			Random  rand  = new XORShiftRandom(filename.hashCode());
			float[] neu1  = cbow ? new float[vector_size] : null;
			float[] neu1e = new float[vector_size];
			int     iter  = 0;
			int     index, window;
			int[]   words;
			
			while (true)
			{
				words = next(reader, rand);
				
				if (words == null)
				{
					BinUtils.LOG.info(String.format("Thread %d: %d\n", id, iter));
					if (++iter == train_iteration) break;
					adjustLearningRate();
					reader.startOver();
					continue;
				}
				
				for (index=0; index<words.length; index++)
				{
					window = 1 + rand.nextInt() % max_skip_window;	// dynamic window size
					if (cbow) Arrays.fill(neu1, 0);
					Arrays.fill(neu1e, 0);
					
					if (cbow) bagOfWords(words, index, window, rand, neu1e, neu1, evaluate);
					else      skipGram  (words, index, window, rand, neu1e, evaluate);
				}
			}
		}
	}
	
	void adjustLearningRate()
	{
		float rate = Math.max(ALPHA_MIN_RATE, 1 - (float)MathUtils.divide(word_count_global, train_iteration * word_count_train + 1));
		alpha_global = alpha_init * rate;
	}
	
	void bagOfWords(int[] words, int index, int window, Random rand, float[] neu1e, float[] neu1, boolean evaluate)
	{
		int i, j, k, l, wc = 0, word = words[index];

		// input -> hidden
		for (i=-window,j=index+i; i<=window; i++,j++)
		{
			if (i == 0 || words.length <= j || j < 0) continue;
			l = words[j] * vector_size;
			for (k=0; k<vector_size; k++) neu1[k] += W[k+l];
			wc++;
		}
		
		if (wc == 0) return;
		for (k=0; k<vector_size; k++) neu1[k] /= wc;
		
		if(evaluate){
			optimizer.testBagOfWords(rand, word, V, neu1, neu1e, alpha_global);
			return;
		}
		
		optimizer.learnBagOfWords(rand, word, V, neu1, neu1e, alpha_global);
		
		// hidden -> input
		for (i=-window,j=index+i; i<=window; i++,j++)
		{
			if (i == 0 || words.length <= j || j < 0) continue;
			l = words[j] * vector_size;
			for (k=0; k<vector_size; k++) W[k+l] += neu1e[k];
		}
	}
	
	void skipGram(int[] words, int index, int window, Random rand, float[] neu1e, boolean evaluate)
	{
		int i, j, k, l1, word = words[index];
		
		for (i=-window,j=index+i; i<=window; i++,j++)
		{
			if (i == 0 || words.length <= j || j < 0) continue;
			l1 = words[j] * vector_size;
			Arrays.fill(neu1e, 0);
			
			
			if(evaluate){
				optimizer.learnSkipGram(rand, word, W, V, neu1e, alpha_global, l1);
				continue;
			}
			
			optimizer.learnSkipGram(rand, word, W, V, neu1e, alpha_global, l1);
			
			// hidden -> input
			for (k=0; k<vector_size; k++) W[l1+k] += neu1e[k];
		}
	}
	
//	=================================== Helper Methods ===================================

	boolean isNegativeSampling()
	{
		return negative_size > 0;
	}
	
	/** Initializes weights between the input layer to the hidden layer using random numbers between [-0.5, 0.5]. */
	void initNeuralNetwork()
	{
		int size = vocab.size() * vector_size;
		Random rand = new XORShiftRandom(1);

		W = new float[size];
		V = new float[size];
		
		for (int i=0; i<size; i++)
			W[i] = (float)((rand.nextDouble() - 0.5) / vector_size);
	}
	
	int[] next(Reader<?> reader, Random rand)
	{
		Object[] words = reader.next();
		if (words == null) return null;
		int[] next = new int[words.length];
		int i, j, index, count = 0;
		double d;
		
		for (i=0,j=0; i<words.length; i++)
		{
			index = vocab.indexOf(words[i].toString());
			if (index < 0) continue;
			count++;
			
			// sub-sampling: randomly discards frequent words
			if (subsample_threshold > 0)
			{
				d = (Math.sqrt(MathUtils.divide(vocab.get(index).count, subsample_size)) + 1) * (subsample_size / vocab.get(index).count);
				if (d < rand.nextDouble()) continue;
			}
			
			next[j++] = index;
		}
		
		word_count_global += count;
		return (j == 0) ? next(reader, rand) : (j == words.length) ? next : Arrays.copyOf(next, j);
	}
	
	void save() throws IOException
	{
		save(IOUtils.createFileOutputStream(output_file));
	}
	
	public Map<String,float[]> toMap(boolean normalize)
	{
		Map<String,float[]> map = new HashMap<>();
		float[] vector;
		String key;
		int i, l;
		
		for (i=0; i<vocab.size(); i++)
		{
			l = i * vector_size;
			key = vocab.get(i).form;
			vector = Arrays.copyOfRange(W, l, l+vector_size);
			if (normalize) normalize(vector);
			map.put(key, vector);
		}
		
		return map;
	}
	
	public void normalize(float[] vector)
	{
		float z = 0;
		
		for (int i=0; i<vector.length; i++)
			z += MathUtils.sq(vector[i]);
		
		z = (float)Math.sqrt(z);
		
		for (int i=0; i<vector.length; i++)
			vector[i] /= z;
	}
	
	public void save(OutputStream out) throws IOException
	{
		ObjectOutputStream oout = IOUtils.createObjectXZBufferedOutputStream(out);
		oout.writeObject(toMap(normalize));
		oout.close();
	}
	
	static public void main(String[] args)
	{
		new Word2Vec(args);
	}
}
