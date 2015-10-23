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

import java.util.List;

import org.junit.Test;

import edu.emory.mathcs.nlp.common.util.DSUtils;
import edu.emory.mathcs.nlp.text_analysis.vsm.reader.SentenceReader;
import edu.emory.mathcs.nlp.text_analysis.vsm.util.Vocabulary;

/**
 * @author Jinho D. Choi ({@code jinho.choi@emory.edu})
 */
public class SentenceReaderTest
{
	@Test
	public void test() throws Exception
	{
		List<String> filenames = DSUtils.toList("src/test/resources/dat/word2vec.txt");
		Vocabulary vocab = new Vocabulary();
		SentenceReader in = new SentenceReader();
		
		long count = in.learn(filenames, vocab, 0);
		assertEquals("D:4 E:4 F:4 C:3 G:3 B:2 H:2 A:1 I:1", vocab.toString());
		assertEquals(24, count);
		
		count = in.learn(filenames, vocab, 0);
		assertEquals("D:8 E:8 F:8 C:6 G:6 B:4 H:4 A:2 I:2", vocab.toString());
		assertEquals(48, count);
	}
}
