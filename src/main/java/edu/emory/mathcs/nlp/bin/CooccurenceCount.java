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
package edu.emory.mathcs.nlp.bin;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.stream.Collectors;

import org.kohsuke.args4j.Option;

import edu.emory.mathcs.nlp.common.util.BinUtils;
import edu.emory.mathcs.nlp.common.util.DSUtils;
import edu.emory.mathcs.nlp.common.util.IOUtils;
import edu.emory.mathcs.nlp.component.template.node.NLPNode;
import edu.emory.mathcs.nlp.tokenization.EnglishTokenizer;
import edu.emory.mathcs.nlp.tokenization.Tokenizer;

/**
 * @author Jinho D. Choi ({@code jinho.choi@emory.edu})
 */
public class CooccurenceCount
{
	@Option(name="-i", usage="input file (required)", required=true, metaVar="<filename>")
	private String input_file;
	@Option(name="-o", usage="output file (required)", required=true, metaVar="<filename>")
	private String output_file;
	@Option(name="-w", usage="window size (default: 5)", required=false, metaVar="<integer>")
	protected int window = 5;
	@Option(name="-p", usage="include punctuation (default: false)", required=false, metaVar="<boolean>")
	protected boolean include_punctuation = false;
	@Option(name="-c", usage="uncapitalize (default: false)", required=false, metaVar="<boolean>")
	protected boolean uncapitalize = false;
	
	public final String DELIM = "\t";
	
	public CooccurenceCount(String[] args) throws Exception
	{
		BinUtils.initArgs(args, this);
		
		Map<String,int[]> map = wordCount(IOUtils.createFileInputStream(input_file), window);
		print(IOUtils.createFileOutputStream(output_file), map);
	}
	
	public Map<String,int[]> wordCount(InputStream in, int window) throws Exception
	{
		BufferedReader reader = IOUtils.createBufferedReader(in);
		Tokenizer tokenizer = new EnglishTokenizer();
		Map<String,int[]> map = new HashMap<>();
		Set<String> set = new HashSet<>();
		List<NLPNode[]> sentences;
		String line; int dc;

		while ((line = reader.readLine()) != null)
		{
			sentences = tokenizer.segmentize(line);
			set.clear();
			
			//fixed
			for (NLPNode[] sentence : sentences){
				List<String> tempHolder = new ArrayList<String>();	
				for(NLPNode node : sentence) {
					tempHolder.add(node.getWordForm());
				}
				set.addAll(count(map, tempHolder, window));
			}
			dc = window * 2 + 1;
			for (String s : set) map.get(s)[dc]++;
		}
		
		in.close();
		return map;
	}
	
	public void print(OutputStream out, Map<String,int[]> map)
	{
		List<Entry<String,int[]>> entries = new ArrayList<>(map.entrySet());
		Collections.sort(entries, Entry.comparingByKey());
		
		PrintStream fout = IOUtils.createBufferedPrintStream(out);
		String s;
		
		StringBuilder build = new StringBuilder();
		build.append("central\tcontext\t");
		int j, size = entries.get(0).getValue().length, window = (size-2)/2;
		for (j=-window; j<0; j++) build.append(j+"\t");
		for (j=1; j<=window; j++) build.append(j+"\t");
		build.append("sentence\tdocument");
		fout.println(build.toString());
		
		for (Entry<String,int[]> e : entries)
		{
			s = Arrays.stream(e.getValue()).mapToObj(i -> Integer.toString(i)).collect(Collectors.joining(DELIM));
			fout.println(e.getKey() + DELIM + s);
		}
		
		fout.close();
	}

	private Set<String> count(Map<String,int[]> map, List<String> sentence, int window)
	{
		Set<String> set = new HashSet<>();
		int i, j, sc;
		String key;
		
		for (i=0; i<sentence.size(); i++)
		{
			for (j=-window; j<0; j++)
			{
				key = count(map, sentence, i, j, window);
				if (key != null) set.add(key);
			}
			
			for (j=1; j<=window; j++)
			{
				key = count(map, sentence, i, j, window);
				if (key != null) set.add(key);
			}
		}
		
		sc = window * 2;
		for (String s : set) map.get(s)[sc]++;
		return set;
	}
	
	private String count(Map<String,int[]> map, List<String> sentence, int i, int j, int window)
	{
		int k = i + j;
		
		if (DSUtils.isRange(sentence, k))
		{
			String current = sentence.get(i);
			String context = sentence.get(k);
			String key = current + DELIM + context;
			int[] counts = map.get(key);
			
			if (counts == null)
			{
				counts = new int[window*2+2];
				map.put(key, counts);
			}
			
			int index = j + window;
			if (j > 0) index--;
			counts[index]++;
			return key;
		}
		
		return null;
	}
	
	static public void main(String[] args)
	{
		try
		{
			new CooccurenceCount(args);
		}
		catch (Exception e) {e.printStackTrace();}
	}
}
