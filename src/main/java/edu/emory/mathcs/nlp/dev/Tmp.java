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
package edu.emory.mathcs.nlp.dev;

import java.io.BufferedReader;
import java.io.ObjectInputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.PriorityQueue;
import java.util.StringJoiner;

import edu.emory.mathcs.nlp.common.collection.tuple.ObjectDoublePair;
import edu.emory.mathcs.nlp.common.util.IOUtils;
import edu.emory.mathcs.nlp.common.util.Joiner;
import edu.emory.mathcs.nlp.common.util.MathUtils;
import edu.emory.mathcs.nlp.component.template.node.NLPNode;
import edu.emory.mathcs.nlp.tokenization.EnglishTokenizer;
import edu.emory.mathcs.nlp.tokenization.Tokenizer;

/**
 * @author Jinho D. Choi ({@code jinho.choi@emory.edu})
 */
public class Tmp
{
	@SuppressWarnings("unchecked")
	public Tmp(String[] args) throws Exception
	{
		final String INPUT_FILE  = args[0];
		final String OUTPUT_FILE = args[1];
		
		ObjectInputStream in = IOUtils.createObjectXZBufferedInputStream(INPUT_FILE);
		Map<String,float[]> map = (Map<String,float[]>)in.readObject();
		in.close();
	
		PrintStream fout = IOUtils.createBufferedPrintStream(OUTPUT_FILE);
		List<Entry<String,float[]>> list = new ArrayList<>(map.entrySet());
		Collections.sort(list, Entry.comparingByKey());
		PriorityQueue<ObjectDoublePair<String>> q;
		Entry<String,float[]> ei, ej;
		StringJoiner join;
		
		for (int i=0; i<list.size(); i++)
		{
			ei = list.get(i);
			q  = new PriorityQueue<>(Collections.reverseOrder());
			
			for (int j=0; j<list.size(); j++)
			{
				if (i == j) continue;
				ej = list.get(j);
				q.add(new ObjectDoublePair<String>(ej.getKey(), MathUtils.cosineSimilarity(ei.getValue(), ej.getValue())));
			}
			
			join = new StringJoiner("\t");
			join.add(ei.getKey());
			
			for (int j=0; j<20; j++)
				join.add(q.poll().o);

			fout.println(join.toString());
		}

		fout.close();
	}
	
	public void tokenize(String[] args) throws Exception
	{
		final String INPUT_FILE  = args[0];
		final String OUTPUT_FILE = args[1];
		
		Tokenizer tok = new EnglishTokenizer();
		BufferedReader reader = IOUtils.createBufferedReader(INPUT_FILE);
		PrintStream out = IOUtils.createBufferedPrintStream(OUTPUT_FILE);
		String line;
		
		while ((line = reader.readLine()) != null)
		{
			line = line.trim();
			if (line.isEmpty() || line.startsWith("<Windows Document")) continue;
			
			//fixed
			for (NLPNode[] sen : tok.segmentize(line)) {
				List<String> tempHolder = new ArrayList<String>();	
				for(NLPNode node : sen) {
					tempHolder.add(node.getWordForm());
				}
				out.println(Joiner.join(tempHolder, " "));
			}
		}
		
		reader.close();
		out.close();
	}
	
	static public void main(String[] args) throws Exception
	{
		new Tmp(args);
	}
}
