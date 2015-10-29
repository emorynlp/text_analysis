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
package edu.emory.mathcs.nlp.text_analysis.dbpedia;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.util.HashSet;
import java.util.Set;

import edu.emory.mathcs.nlp.common.collection.tree.PrefixNode;
import edu.emory.mathcs.nlp.common.collection.tree.PrefixTree;
import edu.emory.mathcs.nlp.common.util.DSUtils;
import edu.emory.mathcs.nlp.common.util.IOUtils;
import edu.emory.mathcs.nlp.tokenization.EnglishTokenizer;
import edu.emory.mathcs.nlp.tokenization.Tokenizer;

/**
 * @author Jinho D. Choi ({@code jinho.choi@emory.edu})
 */
public class PrefixTreeExtender
{
	private PrefixTree<String,Set<String>> prefix_tree;
	private Tokenizer tokenizer;
	
	@SuppressWarnings("unchecked")
	public PrefixTreeExtender(InputStream in) throws Exception
	{
		ObjectInputStream oin = IOUtils.createObjectXZBufferedInputStream(in);
		tokenizer = new EnglishTokenizer();
		System.out.println("Loading");
		prefix_tree = (PrefixTree<String,Set<String>>)oin.readObject();
		oin.close();
	}

	public void extend(InputStream in, String type) throws Exception
	{
		BufferedReader reader = IOUtils.createBufferedReader(in);
		PrefixNode<String,Set<String>> node;
		Set<String> set;
		String[] array;
		String line;
		
		System.out.println("Extending");
		
		while ((line = reader.readLine()) != null)
		{
			line = line.trim();
			if (line.isEmpty()) continue;
			array = DSUtils.toArray(tokenizer.tokenize(line));
			node  = prefix_tree.add(array, 0, array.length, String::toString);
			set   = node.getValue();
			
			if (set == null)
			{
				set = new HashSet<>();
				node.setValue(set);
			}
			
			set.add(type);
		}
		
		reader.close();
	}
	
	public void print(OutputStream out) throws Exception
	{
		ObjectOutputStream fout = IOUtils.createObjectXZBufferedOutputStream(out);
		System.out.println("Printing");
		fout.writeObject(prefix_tree);
		fout.close();
	}
	
	static public void main(String[] args) throws Exception
	{
		final String prefixFile = args[0];
		final String inputFile  = args[1];
		final String type       = args[2];
		final String outputFile = args[3];
		
		try
		{
			PrefixTreeExtender ex = new PrefixTreeExtender(IOUtils.createFileInputStream(prefixFile));
			ex.extend(IOUtils.createFileInputStream(inputFile), type);
			ex.print(IOUtils.createFileOutputStream(outputFile));
		}
		catch (Exception e) {e.printStackTrace();}
	}
}
