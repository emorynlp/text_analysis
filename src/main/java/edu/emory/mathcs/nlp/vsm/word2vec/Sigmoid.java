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

import edu.emory.mathcs.nlp.common.MathUtils;

/**
 * @author Jinho D. Choi ({@code jinho.choi@emory.edu})
 */
public class Sigmoid
{
	private final int TABLE_SIZE = 1000;
	private final int MAX_EXP    = 6;
	private final double NORM    = TABLE_SIZE / MAX_EXP / 2; 

	private float[] exp_table;
	
	public Sigmoid()
	{
		exp_table = new float[TABLE_SIZE];
		
		for (int i=0; i<TABLE_SIZE; i++)
		{
			exp_table[i] = (float)Math.exp((MathUtils.divide(i, TABLE_SIZE) * 2 - 1) * MAX_EXP);
			exp_table[i] /=  (exp_table[i] + 1);
		}
	}

	public double getGradientNegativeSampling(int label, double score)
	{
		if (score >  MAX_EXP) return label - 1;	
		if (score < -MAX_EXP) return label;
		return label - exp_table[(int)((score + MAX_EXP) * NORM)];
	}
	
//	public double getGradientHierarchicalSoftmax(int label, double score)
//	{
//		if (score <= -MAX_EXP) continue;
//		else if (d >= MAX_EXP) continue;
//		else d = sigmoid[(int)((d + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
//	}
}
