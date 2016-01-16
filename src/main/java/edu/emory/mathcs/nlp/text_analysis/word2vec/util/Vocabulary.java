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
package edu.emory.mathcs.nlp.text_analysis.word2vec.util;

import edu.emory.mathcs.nlp.text_analysis.word2vec.reader.Reader;
import it.unimi.dsi.fastutil.objects.Object2IntMap;
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.*;

import edu.emory.mathcs.nlp.common.util.Joiner;

/**
 * @author Jinho D. Choi ({@code jinho.choi@emory.edu})
 */
public class Vocabulary implements Serializable
{
	private static final long serialVersionUID = 5406441768049538210L;
	public static final int MAX_CODE_LENGTH = 40;
	
	private Object2IntMap<String> index_map;
	private List<Word>            word_list;
	private int                   min_reduce;
	private long 				  total_count;

	public Vocabulary()
	{
		index_map  = new Object2IntOpenHashMap<>();
		word_list  = new ArrayList<>();
		min_reduce = 1;
	}

	//-------- Austin's Code ----------------------------------------------------------------------------

	/**
	 * @return total number of word tokens in vocabulary.
	 */
	public long totalCount() {return total_count;}


	/**
<<<<<<< HEAD
=======
	 * Add every word in reader to vocabulary,
	 * then sort vocabulary and restart reader.
	 *
	 * @param reader - source of words of type nlp.reader.Reader
	 */
	public void learn(Reader<String> reader) throws IOException { learn(reader, 0); }

	/**
	 * Add every word in reader to vocabulary,
	 * then sort vocabulary and restart reader.
	 * Remove words with count less than min_word_count
	 *
	 * @param reader - source of words of type nlp.reader.Reader
	 * @param min_word_count - words with counts less than this will be removed
	 */
	public void learn(Reader<String> reader, int min_word_count) throws IOException {
		int word_counter = 0;

		List<String> words;
		while ((words = reader.next()) != null) {
			for (String word : words)
				add(word);
			word_counter += words.size();
			if (word_counter > 1000000) {
				System.out.print(String.format("%.1f", reader.progress())+"%\r");
				word_counter %= 1000000;
			}
		}
		System.out.println(total_count + " total words");
		reader.restart();
		sort(min_word_count);
	}

	/**
	 * Add every word in readers to vocabulary in parallel,
	 * then sort vocabulary and restart reader.
	 *
	 * @param readers - source of words of type nlp.reader.Reader
	 */
	public void learnParallel(List<Reader<String>> readers)
	{
		learnParallel(readers, 0);
	}

	/**
	 * Add every word in reader to vocabulary in parallel,
	 * then sort vocabulary and restart reader.
	 * Remove words with count less than min_word_count
	 *
	 * @param readers - list of input sources of type nlp.reader.Reader to be run in parallel
	 * @param min_word_count - words with counts less than this will be removed
	 */
	public void learnParallel(List<Reader<String>> readers, int min_word_count)
	{
		List<LearnTask> task_list = new ArrayList<>();
		int i=0;
		for (Reader<String> r : readers) {
			task_list.add(new LearnTask(new Vocabulary(), r, i));
			i++;
		}

		ExecutorService ex = Executors.newFixedThreadPool(readers.size());
		try
		{
			List<Future<Vocabulary>> futures = ex.invokeAll(task_list);
			for(Future<Vocabulary> f : futures)
				this.addAll(f.get());
			ex.shutdown();
		}
		catch (InterruptedException | ExecutionException e)
		{
			e.printStackTrace();
		}
		sort(min_word_count);

		System.out.println(total_count + " total words");
	}

	private static class LearnTask implements Callable<Vocabulary>
	{
		Vocabulary vocab;
		Reader<String> reader;

		int id;

		public LearnTask(Vocabulary vocab, Reader<String> reader, int id)
		{
			this.vocab = vocab;
			this.reader = reader;
			this.id = id;
		}

		@Override
		public Vocabulary call() throws Exception {
			int word_counter = 0;

			List<String> words;
			while ((words = reader.next()) != null) {
				for (String word : words)
					vocab.add(word);
				if (id == 0) {
					word_counter += words.size();
					if (word_counter > 100000) {
						System.out.print(String.format("%.1f", reader.progress()) + "%\r");
						word_counter %= 100000;
					}
				}
			}
			if (id == 0) System.out.print(String.format("%.1f", reader.progress()) + "%\r");
			reader.restart();
			return vocab;
		}
	}

	/**
>>>>>>> refs/remotes/origin/ablodge-branch
	 * Adds the word to the vocabulary if absent, and increments its count by 1. 
	 * @return the word object either already existing or newly introduced.
	 */
	public Word add(String word)
	{
		int index = index_map.computeIfAbsent(word, k -> size());
		Word w;
		
		if (index < size())
		{
			w = get(index);
			w.increment(1);
		}
		else
		{
			w = new Word(word, 1);
			word_list.add(w);
		}
		total_count++;  // I only added this line - Austin
		return w;
	}

	/**
	 * Add new word to vocabulary and increment by count.
	 * @param word - word to add to vocabulary
	 */
	public Word add(Word word)
	{
		if (index_map.containsKey(word.form))
		{
			word_list.get(index_map.get(word.form)).count += word.count;
		}
		else
		{
			index_map.put(word.form, word_list.size());
			word_list.add(word);
		}
		total_count += word.count;
		return word;
	}

	/**
	 * Add all words in vocab and increment word counts.
	 * @param vocab - vocabulary of words to add to this vocabulary
	 */
	public void addAll(Vocabulary vocab)
	{
		vocab.list().stream().forEach(this::add);
	}

	/**
	 * Read serialized vocabulary from file.
	 *
	 * @param read_vocab_file - file containing vocab
	 * @throws IOException - if this method can't read file
	 */
	public void readVocab(File read_vocab_file) throws IOException
	{
		ObjectInputStream oin = new ObjectInputStream(new FileInputStream(read_vocab_file));
		try { addAll((Vocabulary) oin.readObject()); }
		catch (ClassNotFoundException e) { e.printStackTrace(); }
		oin.close();
	}

	/**
	 * Write vocabulary to serialized file.
	 *
	 * @param write_vocab_file - file to write vocab vocab to
	 * @throws IOException - if this method can't write to file
	 */
	public void writeVocab(File write_vocab_file) throws IOException
	{
		ObjectOutputStream oout = new ObjectOutputStream(new FileOutputStream(write_vocab_file));
		oout.writeObject(this);
		oout.close();
	}

	// ----- end of Austin's code -----------------------------------------------------------------------------
	
	public Word get(int index)
	{
		return word_list.get(index);
	}
	
	/** @return index of the word if exists; otherwise, -1. */
	public int indexOf(String word)
	{
		return index_map.getOrDefault(word, -1);
	}
	
	public int size()
	{
		return word_list.size();
	}
	
	public List<Word> list()
	{
		return word_list;
	}
	
	/**
	 * Sorts {@link #word_list} by count in descending order.  
	 * @param minCount words whose counts are less than the minimum count will be discarded.
	 * @return total number of word counts after sorting.
	 */
	public long sort(int minCount)
	{
		return reduce(minCount, true);
	}
	
	/**
	 * Reduces the vocabulary by removing infrequent words.
	 * @return total number of word counts after reducing.
	 */
	public long reduce()
	{
		return reduce(++min_reduce, false);
	}
	
	long reduce(int minCount, boolean sort)
	{
		ArrayList<Word> list = new ArrayList<>(size());
		long count = 0;
		
		for (Word w : word_list)
		{
			if (w.count >= minCount)
			{
				count += w.count;
				list.add(w);
			}
		}
		
		if (sort) Collections.sort(list, Collections.reverseOrder());
		list.trimToSize(); word_list = list;
		
		index_map = new Object2IntOpenHashMap<>(size());
		for (int i=0; i<size(); i++) index_map.put(get(i).form, i);
		return count;
	}
	
	/**
	 * Assigns the Huffman code to each word using its count.
	 * PRE: {@link #word_list} is already sorted by count in descending order.
	 */
	public void generateHuffmanCodes()
	{
		int i, j, len, pos1, pos2, min1, min2;
		final int treeSize  = size() * 2 - 1;
		long[] count  = new long[treeSize];
		byte[] binary = new byte[treeSize];
		int [] parent = new int [treeSize];
		
		for (i=0     ; i<size()  ; i++) count[i] = get(i).count;
		for (i=size(); i<treeSize; i++)	count[i] = Long.MAX_VALUE;
		pos1 = size() - 1;
		pos2 = pos1 + 1;
		
		// create binary Huffman tree
		for (i=size(); i<treeSize; i++)
		{
			min1 = (pos1 < 0) ? pos2++ : (count[pos1] < count[pos2]) ? pos1-- : pos2++;
			min2 = (pos1 < 0) ? pos2++ : (count[pos1] < count[pos2]) ? pos1-- : pos2++;
			
			count[i] = count[min1] + count[min2];
			parent[min1] = i;
			parent[min2] = i;
			binary[min2] = 1;
		}
		
		byte[] code  = new byte[MAX_CODE_LENGTH];
		int [] point = new int [MAX_CODE_LENGTH];
		Word w;
		
		// assign binary code to each word
		for (i=0; i<size(); i++)
		{
			len = 0;
			j = i;
			
			do
			{
				code [len] = binary[j];
				point[len] = j;
				j = parent[j];
				len++;
			}
			while (j != treeSize - 1);
			
			w = get(i);
			w.code  = new byte[len];
			w.point = new int [len];
			w.point[0] = size() - 2; // = treeSize - size() - 1
			
			for (j=0; j<len; j++)
			{
				w.code[len-j-1] = code[j];
				if (j > 0) w.point[len-j] = point[j] - size();
			}
		}
	}
	
	@Override
	public String toString()
	{
		return Joiner.join(word_list, " ");
	}
}