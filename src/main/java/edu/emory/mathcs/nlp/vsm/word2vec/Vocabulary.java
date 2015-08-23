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

import it.unimi.dsi.fastutil.objects.Object2IntMap;
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;

import java.io.IOException;
import java.io.InputStream;
import java.io.Serializable;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import org.magicwerk.brownies.collections.GapList;

import edu.emory.mathcs.nlp.common.constant.StringConst;

/**
 * @author Jinho D. Choi ({@code jinho.choi@emory.edu})
 */
public class Vocabulary implements Serializable
{
	private static final long serialVersionUID = 5406441768049538210L;
	private Object2IntMap<String> index_map;
	private List<Word> word_list;
	private int min_reduce;
	
	public Vocabulary()
	{
		index_map  = new Object2IntOpenHashMap<>();
		word_list  = new GapList<>();
		min_reduce = 1;
		add(StringConst.NEW_LINE).count = 0;
	}
	
	/**
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
		
		return w;
	}
	
	/**
	 * Adds all words in the input stream to the vocabulary.
	 * @param reduceSize calls {@link #reduce()} when the number of word types reach this. 
	 */
	public void addAll(InputStream in, int reduceSize) throws IOException
	{
		String next;
		
		while ((next = read(in)) != null)
		{
			add(next);
			if (size() > reduceSize) reduce();
		}

		in.close();
	}
	
	/**
	 * @return the next word in the reader if exists; otherwise, null.
	 * Uses ' ', '\t', and '\n' as delimiters; '\n' is returned as {@link StringConst#NEW_LINE}.
	 */
	public String read(InputStream fin) throws IOException
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
					return StringConst.NEW_LINE;
				else
					continue;
			}
			
			build.append((char)ch);
			fin.mark(1);
		}
		
		return build.length() > 0 ? build.toString() : null;
	}
	
	public Word get(int index)
	{
		return word_list.get(index);
	}
	
	/** @return index of the word if exists; otherwise, {@link Const#OOV}. */
	public int indexOf(String word)
	{
		return index_map.getOrDefault(word, Const.OOV);
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
	 * Sorts {@link #word_list} by count in descending order; except, keeps {@link StringConst#NEW_LINE} as the first element.  
	 * @param minCount words whose counts are less than the minimum count will be discarded.
	 * @return total number of word counts after sorting.
	 */
	public long sort(int minCount)
	{
		GapList<Word> list = new GapList<>(size());
		Word w, eol = get(Const.EOL);
		long count = eol.count;
		
		for (int a=1; a<size(); a++)
		{
			w = get(a);
			
			if (w.count < minCount)
				index_map.remove(w.word);
			else
			{
				count += w.count;
				list.add(w);
			}
		}
		
		Collections.sort(list, Collections.reverseOrder());
		list.addFirst(eol);
		list.trimToSize();
		word_list = list;
		return count;
	}
	
	/** Reduces the vocabulary by removing infrequent words. */
	public void reduce()
	{
		Iterator<Word> it = word_list.iterator();
		Word v;
		
		while (it.hasNext())
		{
			v = it.next();
			
			if (v.count <= min_reduce)
			{
				it.remove();
				index_map.remove(v.word);
			}
		}
		
		min_reduce++;
	}
	
	/**
	 * Assigns the Huffman code to each word using its count.
	 * PRE: {@link #word_list} is already sorted by count in descending order.
	 * @param maxDepth maximum depth of the binary Huffman tree.
	 */
	public void generateHuffmanCodes(int maxDepth)
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
		
		byte[] code  = new byte[maxDepth];
		int [] point = new int [maxDepth];
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
}