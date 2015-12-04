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

import java.io.*;
import java.util.*;
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
import edu.emory.mathcs.nlp.text_analysis.word2vec.reader.Reader;
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
	@Option(name="-evaluate", usage="path to file or directory to evaluate vectors for squared error.", required=false, metaVar="<filename>")
	String eval_path = null;
	@Option(name="-triad-file", usage="triad word similarity file for (external) vector evaluation.", required=false, metaVar="<filename>")
	String triad_file = null;
	@Option(name="-ext", usage="extension of the training files (default: \"*\").", required=false, metaVar="<string>")
	String train_ext = "*";
	@Option(name="-size", usage="size of word vectors (default: 100).", required=false, metaVar="<int>")
	int vector_size = 100;
	@Option(name="-window", usage="max-window of contextual words (default: 5).", required=false, metaVar="<int>")
	int max_skip_window = 5;
	@Option(name="-sample", usage="threshold for occurrence of words (default: 1e-3). Those that appear with higher frequency in the training data will be randomly down-sampled.", required=false, metaVar="<float>")
	float subsample_threshold = 0.001f;
	@Option(name="-negative", usage="number of negative examples (common values are 3 - 10). If negative = 0, use Hierarchical Softmax instead of Negative Sampling.", required=false, metaVar="<int>")
	int negative_size = 5;
	@Option(name="-threads", usage="number of threads (default: 12).", required=false, metaVar="<int>")
	int thread_size = 12;
	@Option(name="-iter", usage="number of training iterations (default: 5).", required=false, metaVar="<int>")
	int train_iteration = 5;
	@Option(name="-min-count", usage="min-count of words (default: 5). This will discard words that appear less than <int> times.", required=false, metaVar="<int>")
	int min_count = 5;
	@Option(name="-alpha", usage="initial learning rate (default: 0.025 for skip-gram; use 0.05 for CBOW).", required=false, metaVar="<float>")
	float alpha_init = 0.025f;
	@Option(name="-cbow", usage="If set, use the continuous bag-of-words model instead of the skip-gram model.", required=false, metaVar="<boolean>")
	boolean cbow = false;
	@Option(name="-normalize", usage="If set, normalize each vector.", required=false, metaVar="<boolean>")
	boolean normalize = false;
	@Option(name="-lowercase", usage="If set, all words will be set to lowercase.", required=false, metaVar="<boolean>")
	boolean lowercase = false;
	@Option(name="-sentence-border", usage="If set, use symbols <s> and </s> for start and end of sentence.", required=false, metaVar="<boolean>")
	boolean sentence_border = false;
	
	final float ALPHA_MIN_RATE  = 0.0001f;
	
	Sigmoid sigmoid;
	public Vocabulary vocab;
	long word_count_train;
	float subsample_size;
	Optimizer optimizer;
	
	volatile long word_count_global;	// word count dynamically updated by all threads
	volatile float alpha_global;		// learning rate dynamically updated by all threads
	volatile public float[] W;			// weights between the input and the hidden layers
	volatile public float[] V;			// weights between the hidden and the output layers

	long start_time;

	public Word2Vec(String[] args)
	{
		BinUtils.initArgs(args, this);
		sigmoid = new Sigmoid();

		try
		{
			train(FileUtils.getFileList(train_path, train_ext, false));
		}
		catch (Exception e) {e.printStackTrace();}
	}
	
//	=================================== Training ===================================
	
	public void train(List<String> filenames) throws Exception
	{
		List<File> files = new ArrayList<File>();
		for(String filename : filenames)
			files.add(new File(filename));
		Reader<?> training_reader = new SentenceReader(files, lowercase, sentence_border);

		System.out.println("Word2Vec");
		System.out.println((cbow ? "Continuous Bag of Words" : "Skipgrams") + ", " + (isNegativeSampling() ? "Hierarchical Softmax" : "Negative Sampling"));
		System.out.println("Reading vocabulary:");

		BinUtils.LOG.info("Reading vocabulary:\n");
		vocab = new Vocabulary();
		vocab.learn(training_reader, min_count);
		word_count_train = vocab.totalWords();
		BinUtils.LOG.info(String.format("- types = %d, tokens = %d\n", vocab.size(), word_count_train));

		System.out.println("Vocab size "+vocab.size()+", Total Word Count "+word_count_train+"\n");
		System.out.println("Starting training: "+train_path);
		System.out.println("Files "+files.size()+", threads "+thread_size+", iterations "+train_iteration);

		BinUtils.LOG.info("Initializing neural network.\n");
		initNeuralNetwork();
		
		BinUtils.LOG.info("Initializing optimizer.\n");
		optimizer = isNegativeSampling() ? new NegativeSampling(vocab, sigmoid, vector_size, negative_size) : new HierarchicalSoftmax(vocab, sigmoid, vector_size);

		BinUtils.LOG.info("Training vectors:");
		word_count_global = 0;
		alpha_global      = alpha_init;
		subsample_size    = subsample_threshold * word_count_train;

		startThreads(training_reader, false);
		outputProgress(System.currentTimeMillis());
		
		if(eval_path != null){
			System.out.println("Starting Evaluation:");
			List<File> test_files = new ArrayList<File>();
			for(String f : FileUtils.getFileList(eval_path, train_ext, false))
				test_files.add(new File(f));

			Reader<?> test_reader = new SentenceReader(test_files, lowercase, sentence_border);
			startThreads(test_reader, true);
			System.out.println("Evaluated Error: " + optimizer.getError());
		}
		if(triad_file != null) {
			System.out.println("Triad Evaluation:");
			evaluateVectors(new File(triad_file));
		}

		BinUtils.LOG.info("Saving word vectors.\n");
		save();
	}
	
	void startThreads(Reader<?> reader, boolean evaluate) throws IOException
	{
		Reader<?>[] readers = reader.split(thread_size);
		reader.close();
		ExecutorService executor = Executors.newFixedThreadPool(thread_size);
		start_time = System.currentTimeMillis();

		for (int i = 0; i < thread_size; i++)
			executor.execute(new TrainTask(readers[i], i, evaluate));
			
		executor.shutdown();			
		try {
			executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
			
		for (int i = 0; i < thread_size; i++)
			readers[i].close();
	}
	
	class TrainTask implements Runnable
	{
		private Reader<?> reader;
		private final int id;
		private boolean evaluate;

		private long last_time = 0;
		
		public TrainTask(Reader<?> reader, int id, boolean evaluate)
		{
			this.reader = reader;
			this.id = id;
			this.evaluate = evaluate;
		}
		
		@Override
		public void run()
		{
			Random  rand  = new XORShiftRandom(reader.hashCode());
			float[] neu1  = cbow ? new float[vector_size] : null;
			float[] neu1e = new float[vector_size];
			int     iter  = 0;
			int     index, window;
			int[]   words = null;
			
			while (true)
			{
				try {
					words = next(reader, rand);
				} catch (IOException e) {
					e.printStackTrace();
				}

				if (words == null)
				{
					BinUtils.LOG.info(String.format("Thread %d: %d\n", id, iter));
					if (++iter == train_iteration) break;
					adjustLearningRate();
					try {
						reader.startOver();
					} catch (IOException e) {
						e.printStackTrace();
					}
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

				// output progress every 15 minutes
				if(id == 0){
					long now = System.currentTimeMillis();
					if(now-last_time > 15*1000*60){
						last_time = now;
						outputProgress(now);
					}
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
	
	int[] next(Reader<?> reader, Random rand) throws IOException
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

	void outputProgress(long now)
	{
		long time_seconds = (now - start_time)/1000;
		float progress = word_count_global / (float)(train_iteration * word_count_train);

		int time_left_hours = (int) ((1-progress)*(now - start_time)/(1000*60*60));
		int time_left_remainder =  ((int) ((1-progress)*(now - start_time)/(1000*60)))% 60;

		System.out.print("Alpha: "+ String.format("%1$,.6f",alpha_global)+" ");
		System.out.print("Progress: "+ String.format("%1$,.2f", progress * 100) + "% ");
		System.out.print("Words/thread/sec: " + String.format("%1$,.4f", word_count_global / ((double) thread_size * time_seconds))+" ");
		System.out.print("Estimated Time Left: " +time_left_hours +":"+time_left_remainder +"\n");
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


	void evaluateVectors(File triad_file) throws IOException
	{
		double unweighted_eval = 0.0;
		int unweighted_count = 0;

		double weighted_eval = 0.0;
		int weighted_count = 0;


		BufferedReader br = new BufferedReader(new FileReader(triad_file));

		String line;
		while((line = br.readLine()) != null){
			String[] triad = line.split(",");
			if(triad.length != 5)
				throw new IOException("Could not read triad file. Incorrect format.");
			if(!(vocab.contains(triad[0]) && vocab.contains(triad[1]) && vocab.contains(triad[2])))
				continue;

			int word_count1 = Integer.parseInt(triad[3]);
			int word_count2 = Integer.parseInt(triad[4]);

			if((word_count1 > word_count2) == (similarity(triad[1],triad[0]) > similarity(triad[2],triad[0]))) {
				unweighted_eval++;
			}
			unweighted_count++;
			for(int i=0; i<Math.abs(word_count1-word_count2); i++){
				weighted_eval++;
				weighted_count++;
			}

		}
		br.close();

		if(unweighted_count>0)
			unweighted_eval /= unweighted_count;
		if(weighted_count>0)
			weighted_eval /= weighted_count;

		System.out.println("Weighted Triad Evaluation: "+weighted_eval);
		System.out.println("Unweighted Triad Evaluation: "+unweighted_eval);
	}

	double similarity(String word1, String word2)
	{
		int l1 = vocab.indexOf(word1)*vector_size;
		int l2 = vocab.indexOf(word2)*vector_size;

		double norm1 = 0.0, norm2 = 0.0;

		double dot_product = 0.0;
		for(int c=0; c<vector_size; c++){
			dot_product += W[l1+c]*W[l2+c];
			norm1 += W[l1+c]*W[l1+c];
			norm2 += W[l2+c]*W[l2+c];
		}
		norm1 = Math.sqrt(norm1);
		norm2 = Math.sqrt(norm2);

		return dot_product/(norm1*norm2);
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
