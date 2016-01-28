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
package edu.emory.mathcs.nlp.vsm;

import edu.emory.mathcs.nlp.common.random.XORShiftRandom;
import edu.emory.mathcs.nlp.common.util.BinUtils;
import edu.emory.mathcs.nlp.common.util.FileUtils;
import edu.emory.mathcs.nlp.common.util.MathUtils;
import edu.emory.mathcs.nlp.common.util.Sigmoid;
import edu.emory.mathcs.nlp.vsm.optimizer.HierarchicalSoftmax;
import edu.emory.mathcs.nlp.vsm.optimizer.NegativeSampling;
import edu.emory.mathcs.nlp.vsm.optimizer.Optimizer;
import edu.emory.mathcs.nlp.vsm.reader.Reader;
import edu.emory.mathcs.nlp.vsm.reader.SentenceReader;
import edu.emory.mathcs.nlp.vsm.util.Vocabulary;
import org.kohsuke.args4j.Option;

import java.io.*;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

/**
 * @author Austin Blodgett
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
	@Option(name="-cbow", usage="If set, use the continuous bag-of-words model instead of the skip-gram model.", required=false, metaVar="<boolean>")
	boolean cbow = false;
	@Option(name="-normalize", usage="If set, normalize each vector.", required=false, metaVar="<boolean>")
	boolean normalize = true;

	/* TODO Austin
	 * Add cmd line options
	 * tokenize, lowercase, border, evaluate
	 */

	final float ALPHA_MIN_RATE  = 0.0001f;

	/* Note that in regular word2vec, the input and output layers
	 * are the same. In cases where we want to allow asymmetry between
	 * these layers (like in syntactic word2vec), we have to distinguish
	 * between input and output vocabularies.
	 */
	public Vocabulary in_vocab;
	public Vocabulary out_vocab;

	Sigmoid sigmoid;
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

	// ----- Austin's Code ----------------------------------------------------

	Reader<String> getReader(List<File> files)
	{
		return new SentenceReader(files);
	}

	// ---------------------------------------------------------------
	
	public void train(List<String> filenames) throws Exception
	{
		BinUtils.LOG.info("Reading vocabulary:\n");

		// ------- Austin's code -------------------------------------
		in_vocab = (out_vocab = new Vocabulary());

		List<Reader<String>> readers = getReader(filenames.stream().map(File::new).collect(Collectors.toList()))
												.splitParallel(thread_size);
		in_vocab.learnParallel(readers, min_count);
		word_count_train = in_vocab.totalCount();
		// -----------------------------------------------------------

		BinUtils.LOG.info(String.format("- types = %d, tokens = %d\n", in_vocab.size(), word_count_train));
		
		BinUtils.LOG.info("Initializing neural network.\n");
		initNeuralNetwork();
		
		BinUtils.LOG.info("Initializing optimizer.\n");
		optimizer = isNegativeSampling() ? new NegativeSampling(in_vocab, sigmoid, vector_size, negative_size) : new HierarchicalSoftmax(in_vocab, sigmoid, vector_size);

		BinUtils.LOG.info("Training vectors:");
		word_count_global = 0;
		alpha_global      = alpha_init;
		subsample_size    = subsample_threshold * word_count_train;
		ExecutorService executor = Executors.newFixedThreadPool(thread_size);

		// ------- Austin's code -------------------------------------
		start_time = System.currentTimeMillis();

		int id = 0;
		for (Reader<String> r: readers)
		{
			executor.execute(new TrainTask(r,id));
			id++;
		}
		// -----------------------------------------------------------

		executor.shutdown();
		
		try { executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS); }
		catch (InterruptedException e) {e.printStackTrace();}


		BinUtils.LOG.info("Saving word vectors.\n");

		save(new File(output_file));
	}
	
	class TrainTask implements Runnable
	{
		// ------- Austin ----------------------
		private Reader<String> reader;
		private int id;
		private long last_time = System.currentTimeMillis() - 14*60*100; // set back 14 minutes (first output after 60 seconds)

		/* Tasks are each parameterized by a reader which is dedicated to a section of the corpus
		 * (not necesarily one file). The corpus is split to divide it evenly between Tasks without breaking up sentences. */
		public TrainTask(Reader<String> reader, int id)
		{
			this.reader = reader;
			this.id = id;
		}
		// -------------------------------------

		@Override
		public void run()
		{
			Random  rand  = new XORShiftRandom(reader.hashCode());

			float[] neu1  = cbow ? new float[vector_size] : null;
			float[] neu1e = new float[vector_size];
			int     iter  = 0;
			int     index, window;
			int[]   words;
			
			while (true)
			{
				words = next(reader, rand, true);

				if (words == null)
				{
					if (++iter == train_iteration) break;
					adjustLearningRate();
					// readers have a built in restart button - Austin
					try { reader.restart(); } catch (IOException e) { e.printStackTrace(); }
					continue;
				}
				
				for (index=0; index<words.length; index++)
				{
					window = 1 + rand.nextInt() % max_skip_window;	// dynamic window size
					if (cbow) Arrays.fill(neu1, 0);
					Arrays.fill(neu1e, 0);
					
					if (cbow) bagOfWords(words, index, window, rand, neu1e, neu1);
					else      skipGram  (words, index, window, rand, neu1e);
				}

				// output progress every 15 minutes
				if(id == 0){
					long now = System.currentTimeMillis();
					if(now-last_time > 15*1000*60){
						outputProgress(now);
						last_time = now;
					}
				}

			}
		}
	}

	// -------------- Austin's code ------------------------------------------------------

	void outputProgress(long now)
	{
		float time_seconds = (now - start_time)/1000f;
		float progress = word_count_global / (float)(train_iteration * word_count_train + 1);

		int time_left_hours = (int) (((1-progress)/progress)*time_seconds/(60*60));
		int time_left_remainder =  (int) (((1-progress)/progress)*time_seconds/60) % 60;

		Runtime runtime = Runtime.getRuntime();
		long memory_usage = runtime.totalMemory()-runtime.freeMemory();

		System.out.println("Alpha: "+ String.format("%1$,.4f",alpha_global)+" "+
				"Progress: "+ String.format("%1$,.2f", progress * 100) + "% "+
				"Words/thread/sec: " + (int)(word_count_global / thread_size / time_seconds) +" "+
				"Estimated Time Left: " +time_left_hours +":"+String.format("%02d",time_left_remainder) +" "+
				"Memory Usage: " + (int)(memory_usage/(1024*1024)) +"M");
	}

	// -----------------------------------------------------------------------------------
	
	void adjustLearningRate()
	{
		float rate = Math.max(ALPHA_MIN_RATE, 1 - (float)MathUtils.divide(word_count_global, train_iteration * word_count_train + 1));
		alpha_global = alpha_init * rate;
	}
	
	void bagOfWords(int[] words, int index, int window, Random rand, float[] neu1e, float[] neu1)
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
		optimizer.learnBagOfWords(rand, word, V, neu1, neu1e, alpha_global);
		
		// hidden -> input
		for (i=-window,j=index+i; i<=window; i++,j++)
		{
			if (i == 0 || words.length <= j || j < 0) continue;
			l = words[j] * vector_size;
			for (k=0; k<vector_size; k++) W[k+l] += neu1e[k];
		}
	}
	
	void skipGram(int[] words, int index, int window, Random rand, float[] neu1e)
	{
		int i, j, k, l1, word = words[index];
		
		for (i=-window,j=index+i; i<=window; i++,j++)
		{
			if (i == 0 || words.length <= j || j < 0) continue;
			l1 = words[j] * vector_size;
			Arrays.fill(neu1e, 0);
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
		int size1 = in_vocab.size() * vector_size;
		int size2 = out_vocab.size() * vector_size;
		Random rand = new XORShiftRandom(1);

		W = new float[size1];
		V = new float[size2];
		
		for (int i=0; i<size1; i++)
			W[i] = (float)((rand.nextDouble() - 0.5) / vector_size);
	}

	/* If input layer and output layer are asymmetrical, param in_layer
	 * determines if you want to return input layer indices or output
	 * layer indices.
     */
	int[] next(Reader<String> reader, Random rand, boolean in_layer)
	{
		// minor changes in this method - Austin
		Vocabulary vocab = in_layer ? in_vocab : out_vocab;

		List<String> words = null;
		try { words = reader.next(); }
		catch (IOException e)
		{
			System.err.println("Reader failure: progress "+reader.progress());
			e.printStackTrace();
			System.exit(1);
		}

		if (words == null) return null;
		int[] next = new int[words.size()];
		int i, j, index, count = 0;
		double d;
		
		for (i=0,j=0; i<words.size(); i++)
		{
			index = vocab.indexOf(words.get(i));
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
		return (j == 0) ? next(reader, rand, in_layer) : (j == words.size()) ? next : Arrays.copyOf(next, j);
	}

	public Map<String,float[]> toMap(boolean normalize)
	{
		Map<String,float[]> map = new HashMap<>();
		float[] vector;
		String key;
		int i, l;
		
		for (i=0; i<in_vocab.size(); i++)
		{
			l = i * vector_size;
			key = in_vocab.get(i).form;
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

	// ------ Austin's code --------------------------------

	public void save(File save_file) throws IOException
	{
		Map<String,float[]> map = toMap(normalize);
		BufferedWriter out = new BufferedWriter(new FileWriter(save_file));

		for (String word : map.keySet())
		{
			out.write(word+"\t");
			for (float f : map.get(word))
				out.write(f+"\t");
			out.write("\n");
		}
		out.close();
	}
	// -------------------------------------------------------
	
	static public void main(String[] args)
	{
		new Word2Vec(args);
	}
}
