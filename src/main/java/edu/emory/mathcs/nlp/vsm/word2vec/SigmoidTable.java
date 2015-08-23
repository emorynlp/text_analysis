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
public class SigmoidTable
{
	public final int TABLE_SIZE = 1000;
	public final int MAX_EXP    = 6;

	private float[] exp_table;
	
	public SigmoidTable()
	{
		exp_table = new float[TABLE_SIZE];
		
		for (int i=0; i<TABLE_SIZE; i++)
		{
			exp_table[i] = (float)Math.exp((MathUtils.divide(i, TABLE_SIZE) * 2 - 1) * MAX_EXP);
			exp_table[i] /=  (exp_table[i] + 1);
		}
	}
}
