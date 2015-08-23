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

import java.io.PrintStream;

import edu.emory.mathcs.nlp.common.MathUtils;

/**
 * @author Jinho D. Choi ({@code jinho.choi@emory.edu})
 */
public class NeuralNetwork
{
	private final int layer_size;
	private float[] syn0, syn1;
	private Vocabulary vocab;
	
	public NeuralNetwork(Vocabulary vocab, int layerSize)
	{
		int size = vocab.size() * layerSize;
		long random = 1;
		
		this.vocab = vocab;
		layer_size = layerSize;
		syn0 = new float[size];
		syn1 = new float[size];
		
		for (int i=0; i<size; i++)
		{
			random = nextRandom(random);
			syn0[i] = (float)((MathUtils.divide(random & 0xFFFF, 65536) - 0.5) / layerSize);
		}
	}
	
	public void save(PrintStream out, boolean binary)
	{
		out.printf("%d %d\n", vocab.size(), layer_size);
		float d;
		
		for (int i=0; i<vocab.size(); i++)
		{
			out.print(vocab.get(i).word);
			
			for (int j=0; j<layer_size; j++)
			{
				d = syn0[i * layer_size + j];
				if (binary)	out.print(" "+Long.toBinaryString(Double.doubleToRawLongBits(d)));
				else		out.print(" "+d);
			}
			
			out.println();
	    }
	}
	
	static public long nextRandom(long prev)
	{
		return prev * 25214903917L + 11;
	}
}