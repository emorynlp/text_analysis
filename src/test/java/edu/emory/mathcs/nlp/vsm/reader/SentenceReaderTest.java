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
package edu.emory.mathcs.nlp.vsm.reader;

import static org.junit.Assert.assertEquals;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import edu.emory.mathcs.nlp.common.util.FileUtils;
import org.junit.Test;

import edu.emory.mathcs.nlp.common.util.DSUtils;
import edu.emory.mathcs.nlp.text_analysis.word2vec.reader.SentenceReader;
import edu.emory.mathcs.nlp.text_analysis.word2vec.util.Vocabulary;

/**
 * @author Jinho D. Choi ({@code jinho.choi@emory.edu})
 */
public class SentenceReaderTest
{
	@Test
	public void testLearn() throws Exception
	{
		List<String> filenames = DSUtils.toList("src/test/resources/dat/word2vec.txt");
		List<File> files = new ArrayList<File>();
		for(String filename : filenames)
			files.add(new File(filename));

		Vocabulary vocab = new Vocabulary();
		SentenceReader in = new SentenceReader(files, false, false);
		vocab.learn(in, 0);
		
		long count = vocab.totalWords();
		assertEquals("D:4 E:4 F:4 C:3 G:3 B:2 H:2 A:1 I:1", vocab.toString());
		assertEquals(24, count);

		vocab.learn(in, 0);

		count = vocab.totalWords();
		assertEquals("D:8 E:8 F:8 C:6 G:6 B:4 H:4 A:2 I:2", vocab.toString());
		assertEquals(48, count);
	}

	@Test
	public void testRead() throws Exception {

		List<String> filenames = FileUtils.getFileList("src/test/resources/dat/test_files","*");
		List<File> files = new ArrayList<File>();
		for(String f : filenames)
			files.add(new File(f));

		SentenceReader sr = new SentenceReader(files,false,true);

		SentenceReader[] readers = sr.split(2);

		String[] words;
		for(SentenceReader r : readers){
			while((words = r.next())!=null){
				for(String word : words)
					System.out.print(word+" ");
				System.out.println();
			}
			System.out.println();
		}

		System.out.println("Finished");
	}
}
