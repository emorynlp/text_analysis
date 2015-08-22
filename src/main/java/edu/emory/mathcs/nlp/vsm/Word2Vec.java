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
package edu.emory.mathcs.nlp.deeplearning;

import it.unimi.dsi.fastutil.objects.Object2IntMap;
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.magicwerk.brownies.collections.GapList;

import edu.emory.mathcs.nlp.common.BinUtils;
import edu.emory.mathcs.nlp.common.MathUtils;
import edu.emory.mathcs.nlp.common.collection.tuple.BooleanIntPair;
import edu.emory.mathcs.nlp.common.collection.tuple.ObjectBooleanPair;
import edu.emory.mathcs.nlp.common.collection.tuple.Pair;

/**
 * @author Jinho D. Choi ({@code jinho.choi@emory.edu})
 * http://arxiv.org/pdf/1301.3781.pdf
 * http://www-personal.umich.edu/~ronxin/pdf/w2vexp.pdf
 */
public class Word2Vec
{
	final int VOCAB_INIT_SIZE = 100000;
	final double NEG_POWER = 0.75;
	final String EOLN = "</s>";
	final int table_size = 100000000;	// TODO: UNIGRAM_TABLE_SIZE
	final int OOV = -2;
	final int EOF = -1;
	final int EOL =  0;
	
	int vocab_reduce_size = 21000000;
	
	final int MAX_STRING = 100;
	final int EXP_TABLE_SIZE = 1000;
	final int MAX_EXP = 6;
	final int MAX_SENTENCE_LENGTH = 1000;
	final int MAX_CODE_LENGTH = 40;
	final int vocab_hash_size = 30000000;
	
	class vocab_word implements Comparable<vocab_word>
	{
		long cn;	// count
		int[] point;	// pointer
		String word;
		int codelen;	// char
		byte[] code;
		int index;
		
		public vocab_word(String word)
		{
			this.word = word;
			this.cn = 0;
		}
		
		public vocab_word(String word, int count)
		{
			this.word = word;
			this.cn = count;
		}
		
		public void incrementCount()
		{
			cn++;
		}
		
		@Override
		public int compareTo(vocab_word o)
		{
			return MathUtils.signum(cn - o.cn);
		}
	};
	
	Object2IntMap<String> vocab_hash;
	List<vocab_word> vocab;

	String train_file, output_file;
	String save_vocab_file, read_vocab_file;
	int debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;
	int vocab_max_size = 1000, layer1_size = 100; // long
	long train_words = 0, iter = 5, file_size = 0, classes = 0;
	volatile long word_count_actual = 0; // TODO: word_count_global
	double alpha = 0.025, sample = 1e-3;
	double starting_alpha;	// TODO: alpha_init
	double[] syn0, syn1, syn1neg, expTable;	// pointers
	boolean cbow = true;

	boolean hs = false, binary = false;
	int negative = 5;
	int[] table;	// unigram table

	public Word2Vec()
	{
		vocab_hash = new Object2IntOpenHashMap<>();
		vocab = new GapList<>(VOCAB_INIT_SIZE);
		AddWordToVocab(EOLN, 0);
		expTable = new double[EXP_TABLE_SIZE + 1];
		
		for (int i=0; i<EXP_TABLE_SIZE; i++)
		{
			expTable[i] = Math.exp((MathUtils.divide(i, EXP_TABLE_SIZE) * 2 - 1) * MAX_EXP); // Precompute the exp() table
			expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
		}
		
		TrainModel(null);
	}
	
//	=================================== Training ===================================
	
	public void TrainModel(List<String> trainFiles)
	{
		BinUtils.LOG.info("Reading vocabulary:");
		train_words = 0;
		trainFiles.forEach(trainFile -> LearnVocabFromTrainFile(trainFile));
		BinUtils.LOG.info(String.format("\n- types = %d, tokens = %d\n", vocab.size(), train_words));
		
		BinUtils.LOG.info("Initializing neural network.\n");
		InitNet();
		
		if (negative > 0)
		{
			BinUtils.LOG.info("Initializing negative sampling.\n");
			InitUnigramTable();
		}

		BinUtils.LOG.info("Training word vectors:");
		ExecutorService executor = Executors.newFixedThreadPool(num_threads);
		starting_alpha = alpha;
//		for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
//		for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);

		BinUtils.LOG.info("Saving word vectors.\n");
		PrintStream fo = new PrintStream(new BufferedOutputStream(new FileOutputStream(output_file)));
		double d;
		
		fo.printf("%d %d\n", vocab.size(), layer1_size);
		
		for (int a=0; a<vocab.size(); a++)
		{
			fo.print(vocab.get(a).word);
			
			for (int b=0; b<layer1_size; b++)
			{
				d = syn0[a * layer1_size + b];
				if (binary)	fo.print(" "+Long.toBinaryString(Double.doubleToRawLongBits(d)));
				else		fo.print(" "+d);
			}
			
			fo.println();
	    }
	}
	
	void TrainModelThread(String trainFile, int id)
	{
		long cw;
		long word_count = 0, last_word_count = 0;
		int[] sen = new int[MAX_SENTENCE_LENGTH];
		long l1, l2, target, label, local_iter = iter;
		long next_random = (long)id;
		double f, g, ran;
		double[] neu1  = new double[layer1_size];
		double[] neu1e = new double[layer1_size];
		
		InputStream fi = new BufferedInputStream(new FileInputStream(trainFile));
		int word, last_word, sentence_length = 0, sentence_position = 0, c, a, b, d;
		
		while (true)
		{
			if (word_count - last_word_count > 10000)
			{
				word_count_actual += word_count - last_word_count;
				last_word_count = word_count;
				alpha = starting_alpha * (1 - MathUtils.divide(word_count_actual, train_words * iter + 1));
				if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
		    }
			
			if (sentence_length == 0)
			{
				while (true)
				{
					word = ReadWordIndex(fi);
					if (word == EOF) break;		// end-of-file
					if (word == OOV) continue;	// out-of-vocab
					word_count++;
					if (word == EOL) break;		// end-of-line
					
					// sub-sampling randomly discards frequent words while keeping the ranking same
					if (sample > 0)
					{
						ran = (Math.sqrt((double)vocab.get(word).cn / (sample * train_words)) + 1) * (sample * train_words) / vocab.get(word).cn;
						next_random = nextRandom(next_random);
						if (ran < MathUtils.divide(next_random & 0xFFFF, 65536)) continue;
					}
					
					sen[sentence_length++] = word;
					if (sentence_length >= MAX_SENTENCE_LENGTH) break;
				}
				
				sentence_position = 0;
			}
			
			if (word == EOF)
			{
				word_count_actual += word_count - last_word_count;
				if (--local_iter == 0) break;
				word_count = 0;
				last_word_count = 0;
				sentence_length = 0;
				fi.close();
				fi = new BufferedInputStream(new FileInputStream(trainFile));
				continue;
			}
		
			word = sen[sentence_position];
			if (word == -1) continue;
			next_random = nextRandom(next_random);
			b = (int)(next_random % window);
			
			// continuos bag-of-words architecture
			if (cbow)
			{
				// in -> hidden
				cw = 0;
				
				for (a=b; a < window*2+1-b; a++)
				{
					if (a == window) continue;
					c = sentence_position - window + a;
					if (sentence_length <= c || c < 0) continue;
					last_word = sen[c];
					if (last_word == -1) continue;
					d = last_word * layer1_size;
					for (c=0; c<layer1_size; c++) neu1[c] += syn0[c + d];
					cw++;
				}
				
				if (cw > 0)
				{
					for (c=0; c<layer1_size; c++) neu1[c] /= cw;
					
					if (hs) for (d = 0; d < vocab[word].codelen; d++)
					{
						f = 0;
						l2 = vocab[word].point[d] * layer1_size;
						// Propagate hidden -> output
						for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];
						if (f <= -MAX_EXP) continue;
						else if (f >= MAX_EXP) continue;
						else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
						// 'g' is the gradient multiplied by the learning rate
						g = (1 - vocab[word].code[d] - f) * alpha;
						// Propagate errors output -> hidden
						for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
						// Learn weights hidden -> output
						for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
					}
					
					// NEGATIVE SAMPLING
					if (negative > 0) for (d = 0; d < negative + 1; d++)
					{
						if (d == 0)
						{
							target = word;
							label = 1;
						}
						else
						{
							next_random = next_random * (unsigned long long)25214903917 + 11;
							target = table[(next_random >> 16) % table_size];
							if (target == 0) target = next_random % (vocab_size - 1) + 1;
							if (target == word) continue;
							label = 0;
						}
						
						l2 = target * layer1_size;
						f = 0;
						for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
						if (f > MAX_EXP) g = (label - 1) * alpha;
						else if (f < -MAX_EXP) g = (label - 0) * alpha;
						else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
						for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
						for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
					}
					
					// hidden -> in
					for (a = b; a < window * 2 + 1 - b; a++) if (a != window)
					{
						c = sentence_position - window + a;
						if (c < 0) continue;
						if (c >= sentence_length) continue;
						last_word = sen[c];
						if (last_word == -1) continue;
						for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
					}
				}
			}
			else	//train skip-gram
			{
				for (a = b; a < window * 2 + 1 - b; a++) if (a != window)
				{
					c = sentence_position - window + a;
					if (c < 0) continue;
					if (c >= sentence_length) continue;
					last_word = sen[c];
					if (last_word == -1) continue;
					l1 = last_word * layer1_size;
					for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
					
					// HIERARCHICAL SOFTMAX
					if (hs) for (d = 0; d < vocab[word].codelen; d++)
					{
						f = 0;
						l2 = vocab[word].point[d] * layer1_size;
						// Propagate hidden -> output
						for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];
						if (f <= -MAX_EXP) continue;
						else if (f >= MAX_EXP) continue;
						else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
						// 'g' is the gradient multiplied by the learning rate
						g = (1 - vocab[word].code[d] - f) * alpha;
						// Propagate errors output -> hidden
						for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
						// Learn weights hidden -> output
						for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1];
					}
					
					// NEGATIVE SAMPLING
					if (negative > 0) for (d = 0; d < negative + 1; d++)
					{
						if (d == 0)
						{
							target = word;
							label = 1;
						}
						else
						{
							next_random = next_random * (unsigned long long)25214903917 + 11;
							target = table[(next_random >> 16) % table_size];
							if (target == 0) target = next_random % (vocab_size - 1) + 1;
							if (target == word) continue;
							label = 0;
						}
						
						l2 = target * layer1_size;
						f = 0;
						for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
						if (f > MAX_EXP) g = (label - 1) * alpha;
						else if (f < -MAX_EXP) g = (label - 0) * alpha;
						else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
						for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
						for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
					}
					
					// Learn weights input -> hidden
					for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
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
	
//	=================================== Vocabulary ===================================

	/** @return the index of the next word in the reader ; if the word is not in the vocabulary, {@link #OOV}; if no word, {@link #EOF}. */
	int ReadWordIndex(InputStream fin)
	{
		String word = ReadWord(fin);
		return (word != null) ? vocab_hash.getOrDefault(word, OOV) : EOF;
	}
	
	/**
	 * @return the pair next word in the reader if exists; otherwise, null.
	 * Uses ' ', '\t', and '\n' as delimiters; '\n' is returned as {@link #EOLN}.
	 */
	String ReadWord(InputStream fin) throws IOException
	{
		StringBuilder build = new StringBuilder();
		int ch;
		
		while ((ch = fin.read()) >= 0)
		{
			if (ch == 13) continue;	// carriage return
			
			if (ch == ' ' || ch == '\t' || ch == '\n')
			{
				if (build.length() > 0)
				{
					if (ch == '\n') fin.reset();
					break;
				}
				else if (ch == '\n')
					return EOLN;
				else
					continue;
			}
			
			build.append((char)ch);
			fin.mark(1);
		}
		
		return build.length() > 0 ? build.toString() : null;
	}
	
	/** Adds a word to the vocabulary and increments its count. */
	void AddWordToVocab(String word, int initialCount)
	{
		int index = vocab_hash.computeIfAbsent(word, k -> vocab.size());
		
		if (index < vocab.size())
			vocab.get(index).incrementCount();
		else
			vocab.add(new vocab_word(word, initialCount));
	}
	
	/**
	 * Sorts {@link #vocab} by word counts but keeps {@link #EOLN} as the first element.  
	 * PRE: {@link #vocab}.get(0).word = {@link #EOLN}.
	 */
	void SortVocab()
	{
		GapList<vocab_word> sortedVocab = new GapList<>(vocab.size());
		vocab_word v, eol = vocab.get(EOL);
		train_words = eol.cn;
		
		for (int a=1; a<vocab.size(); a++)
		{
			v = vocab.get(a);
			
			if (v.cn < min_count)
				vocab_hash.remove(v.word);
			else
			{
				sortedVocab.add(v);
				train_words += v.cn;
			}
		}
		
		Collections.sort(sortedVocab, Collections.reverseOrder());
		sortedVocab.addFirst(eol);
		sortedVocab.trimToSize();
		vocab = sortedVocab;
	}
	
	/** Reduces the vocabulary by removing infrequent word. */
	void ReduceVocab()
	{
		Iterator<vocab_word> it = vocab.iterator();
		vocab_word v;
		
		while (it.hasNext())
		{
			v = it.next();
			
			if (v.cn <= min_reduce)
			{
				it.remove();
				vocab_hash.remove(v.word);
			}
		}
		
		min_reduce++;
	}
	
	/** Populates the vocabulary using the train file. */
	void LearnVocabFromTrainFile(String trainFile) throws IOException
	{
		InputStream fin = new BufferedInputStream(new FileInputStream(trainFile));
		String next;
		BinUtils.LOG.debug(".");
		
		while ((next = ReadWord(fin)) != null)
		{
			train_words++;
			AddWordToVocab(next, 1);
			if (vocab.size() > vocab_reduce_size) ReduceVocab();
		}

		fin.close();
		SortVocab();
	}
	
//	=================================== Neural Network ===================================
	
	long nextRandom(long prev)
	{
		return prev * 25214903917L + 11;
	}
	
	double nextDistribution(int index, double Z)
	{
		return Math.pow(vocab.get(index).cn, NEG_POWER) / Z;
		
	}
	
	/** Initializes the neural network. */
	void InitNet()
	{
		int size = vocab.size() * layer1_size;
		long random = 1;	// TODO: replace with new Random()?
		
		syn0 = new double[size];
		if (hs) syn1 = new double[size];
		if (negative > 0) syn1neg = new double[size];
		
		for (int a=0; a<size; a++)
		{
			random = nextRandom(random);
			syn0[a] = (MathUtils.divide(random & 0xFFFF, 65536) - 0.5) / layer1_size;
		}
		
		CreateBinaryTree();
	}

	
	/**
	 * Creates a binary Huffman tree using word counts, and assigns the binary code to each word in {@link #vocab}.
	 * PRE: {@link #vocab} is already sorted by word counts in descending order.
	 */
	void CreateBinaryTree()
	{
		int a, b, len, pos1, pos2, min1, min2;
		int vocabSize = vocab.size();
		int treeSize  = vocabSize * 2 - 1;
		long[] count  = new long[treeSize];
		byte[] binary = new byte[treeSize];
		int [] parent = new int [treeSize];
		
		for (a=0; a<vocabSize; a++)	count[a] = vocab.get(a).cn;
		for (a=vocabSize; a<treeSize; a++)	count[a] = Long.MAX_VALUE;
		pos1 = vocabSize - 1;
		pos2 = vocabSize;
		
		// create binary Huffman tree
		for (a=vocabSize; a<treeSize; a++)
		{
			min1 = (pos1 < 0) ? pos2++ : (count[pos1] < count[pos2]) ? pos1-- : pos2++;
			min2 = (pos1 < 0) ? pos2++ : (count[pos1] < count[pos2]) ? pos1-- : pos2++;
			
			count[a] = count[min1] + count[min2];
			parent[min1] = a;
			parent[min2] = a;
			binary[min2] = 1;
		}
		
		byte[] code  = new byte[MAX_CODE_LENGTH];
		int [] point = new int [MAX_CODE_LENGTH];
		vocab_word v;
		
		// assign binary code to each word
		for (a=0; a<vocabSize; a++)
		{
			len = 0;
			b   = a;
			
			do
			{
				code [len] = binary[b];
				point[len] = b;
				b = parent[b];
				len++;
			}
			while (b != treeSize - 1);
			
			v = vocab.get(a);
			v.codelen = len;
			v.code    = new byte[len];
			v.point   = new int [len+1];
			
			// TODO: what does "point" do?
			v.point[0] = vocabSize - 2;
			
			for (b=0; b<len; b++)
			{
				v.code [len-b-1] = code [b];
				v.point[len-b]   = point[b] - vocabSize;
			}
		}
	}
	
	/** Noise distribution = U(w)^.75 / Z for negative sampling. */
	void InitUnigramTable()
	{
		double d1, Z = vocab.stream().mapToDouble(v -> Math.pow(v.cn, NEG_POWER)).sum();
		int a, i = 0;
		
		table = new int[table_size];
		d1 = nextDistribution(i, Z);
		
		for (a=0; a<table_size; a++)
		{
			table[a] = i;
			
			if (MathUtils.divide(a, table_size) > d1)
				d1 += nextDistribution(++i, Z);
			
			if (i >= vocab.size())
			{
				Arrays.fill(table, a+1, table_size, vocab.size()-1);
				break;
			}
		}
	}
}
