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
package edu.emory.mathcs.nlp.text_analysis.word2vec.optimizer;

import java.util.Random;

import edu.emory.mathcs.nlp.common.util.Sigmoid;
import edu.emory.mathcs.nlp.text_analysis.word2vec.util.Vocabulary;

/**
 * @author Jinho D. Choi ({@code jinho.choi@emory.edu})
 */
public class HierarchicalSoftmax extends Optimizer
{
	public HierarchicalSoftmax(Vocabulary vocab, Sigmoid sigmoid, int vectorSize)
	{
		super(vocab, sigmoid, vectorSize);
		vocab.generateHuffmanCodes();
	}
	
	@Override
	public void learnBagOfWords(Random rand, int word, float[][] syn1, float[] neu1, float[] neu1e, float alpha)
	{
		byte[] code  = vocab.get(word).code;
		int [] point = vocab.get(word).point;
		
		for (int i=0; i<code.length; i++)
			learnBagOfWords(code[i], point[i], syn1, neu1, neu1e, alpha);
	}
	
	@Override
	public void learnSkipGram(Random rand, int word, float[][] syn0, float[][] syn1, float[] neu1e, float alpha, int context)
	{
		byte[] code  = vocab.get(word).code;
		int[]  point = vocab.get(word).point;
		
		for (int i=0; i<code.length; i++)
			learnSkipGram(code[i], point[i], syn0, syn1, neu1e, alpha, context);
	}


	@Override
	public void testBagOfWords(Random rand, int word, float[][] syn1, float[] neu1)
	{
		byte[] code  = vocab.get(word).code;
		int [] point = vocab.get(word).point;
		
		for (int i=0; i<code.length; i++)
			testBagOfWords(code[i], point[i], syn1, neu1);
	}
	
	@Override
	public void testSkipGram(Random rand, int word, float[][] syn0, float[][] syn1, int context)
	{
		byte[] code  = vocab.get(word).code;
		int[]  point = vocab.get(word).point;
		
		for (int i=0; i<code.length; i++)
			testSkipGram(code[i], point[i], syn0, syn1, context);
	}
}
