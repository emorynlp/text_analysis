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
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;

import edu.emory.mathcs.nlp.common.constant.StringConst;

/**
 * @author Jinho D. Choi ({@code jinho.choi@emory.edu})
 */
public class WordReader
{
	private InputStream in;
	private boolean new_line;
	
	public WordReader()
	{
		init(null);
	}
	
	public WordReader(InputStream in)
	{
		init(in);
	}
	
	public void init(InputStream in)
	{
		this.in  = in;
		new_line = false;
	}
	
	public void close() throws IOException
	{
		if (in != null) in.close();
	}

	/**
	 * @return the next word in the reader if exists (including {@link StringConst#NEW_LINE}); otherwise, null.
	 * Words are delimited by ' ' and sentences are delimited by '\n'.
	 */
	public String read() throws IOException
	{
		if (new_line)
		{
			new_line = false;
			return StringConst.NEW_LINE;
		}
		
		StringBuilder build = new StringBuilder();
		int ch;
		
		while ((ch = in.read()) >= 0)
		{
			if (ch == 13) continue;	// carriage return
			
			if (ch == ' ' || ch == '\n')
			{
				if (build.length() > 0)
				{
					new_line = (ch == '\n');
					break;
				}
				else
					continue;
			}
			
			build.append((char)ch);
		}
		
		return build.length() > 0 ? build.toString() : null;
	}
	
	/**
	 * All words in the training files are first added then sorted by their counts in descending order.
	 * @param minCount words whose counts are less than this are discarded. 
	 * @param reduceSize if the vocabulary becomes larger than this, it gets reduced.
	 * @return the total number of word tokens learned.
	 */
	public long learn(List<String> filenames, Vocabulary vocab, int minCount, int reduceSize) throws IOException
	{
		String next;
		
		for (String filename : filenames)
		{
			init(new BufferedInputStream(new FileInputStream(filename)));
			
			while ((next = read()) != null)
			{
				if (!StringConst.NEW_LINE.equals(next))
					vocab.add(next);
			}

			close();
			if (vocab.size() >= reduceSize) vocab.reduce();
		}
		
		long count = vocab.sort(minCount);
		vocab.generateHuffmanCodes();
		return count;
	}
}
