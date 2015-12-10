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
public abstract class Optimizer
{
	protected Sigmoid sigmoid;
	protected Vocabulary vocab;
	protected int vector_size;
	
	volatile double error = 0.0;
	volatile long normalizer = 0;
	
	
	public Optimizer(Vocabulary vocab, Sigmoid sigmoid, int vectorSize)
	{
		this.vocab   = vocab;
		this.sigmoid = sigmoid;
		vector_size  = vectorSize;
	}
	
	public abstract void learnBagOfWords(Random rand, int word, float[][] syn1, float[] neu1, float[] neu1e, float alpha);
	public abstract void learnSkipGram  (Random rand, int word, float[][] syn0, float[][] syn1, float[] neu1e, float alpha, int context);

	public abstract void testBagOfWords(Random rand, int word, float[][] syn1, float[] neu1);
	public abstract void testSkipGram  (Random rand, int word, float[][] syn0, float[][] syn1, int context);


	public double getError(){
		return this.error/this.normalizer;
	}

	protected void learnBagOfWords(int label, int word, float[][] syn1, float[] neu1, float[] neu1e, float alpha)
	{
		int k;
		float score = 0, gradient;

		// hidden -> output
		for (k=0; k<vector_size; k++) score += neu1[k] * syn1[word][k];
		gradient = (label - sigmoid.get(score)) * alpha;

		if (gradient != 0)
		{
			// output -> hidden
			for (k=0; k<vector_size; k++) neu1e[k] += syn1[word][k] * gradient;
			// hidden -> output
			for (k=0; k<vector_size; k++) syn1[word][k] += neu1[k] * gradient;
		}
	}

	protected void learnSkipGram(int label, int word, float[][] syn0, float[][] syn1, float[] neu1e, float alpha, int context)
	{
		int k;
		float score = 0, gradient;

		// input -> output
		for (k=0; k<vector_size; k++) score += syn0[context][k] * syn1[word][k];
		gradient = (label - sigmoid.get(score)) * alpha;

		if (gradient != 0)
		{
			// output -> hidden
			for (k=0; k<vector_size; k++) neu1e[k] += syn1[word][k] * gradient;
			// input -> output
			for (k=0; k<vector_size; k++) syn1[word][k] += syn0[context][k] * gradient;
		}
	}

	protected void testBagOfWords(int label, int word, float[][] syn1, float[] neu1)
	{
		int k;
		float score = 0;
		
		// hidden -> output
		for (k=0; k<vector_size; k++) score += neu1[k] * syn1[word][k];
		
		double squared_error = (label - sigmoid.get(score));
	    squared_error = squared_error * squared_error;
	
	    this.error += squared_error;
	    this.normalizer++;
	}

	protected void testSkipGram(int label, int word, float[][] syn0, float[][] syn1, int context)
	{
		int k;
		float score = 0;
		
		// input -> output
		for (k=0; k<vector_size; k++) score += syn0[context][k] * syn1[word][k];
		
	    double squared_error = (label - sigmoid.get(score));
	    squared_error = squared_error * squared_error;
	
	    this.error += squared_error;
	    this.normalizer++;
	}

	public void polysemousBagOfWords(boolean probabilistic, Random rand, int word, float[][][] syn1, float[][] neu1, float[][] neu1e, float alpha, int senses)
	{
		int max = 0, s, k;
		float score = 0, sum = 0;
		float[] expectations = new float[senses];

		for (s = 0; s < senses; s++) {
			// hidden -> output
			for (k = 0; k < vector_size; k++) score += neu1[s][k] * syn1[s][word][k];
			expectations[s] = (1 - sigmoid.get(score));
			expectations[s] = 1 - expectations[s] * expectations[s]; // 1 - squared error
			if (expectations[s] > expectations[max])
				max = s;
			sum += expectations[s];
		}

		if (sum == 0) return;

		for (s = 0; s < senses; s++)
			expectations[s] /= sum;

		if (probabilistic) {
			// expectation maximization
			for (s = 0; s < senses; s++)
				learnBagOfWords(rand, word, syn1[s], neu1[s], neu1e[s], expectations[s] * alpha);
		} else {
			// greedy
			learnBagOfWords(rand, word, syn1[max], neu1[max], neu1e[max], alpha);
		}
	}

	public void polysemousSkipGram(boolean probabilistic, Random rand, int word, float[][][] syn0, float[][][] syn1, float[][] neu1e, float alpha, int context, int senses)
	{
		int max = 0, s, k;
		float score = 0, sum = 0;
		float[] expectations = new float[senses];

		for (s = 0; s < senses; s++) {
			// hidden -> output
			for (k = 0; k < vector_size; k++) score += syn0[s][context][k] * syn1[s][word][k];
			expectations[s] = (1 - sigmoid.get(score));
			expectations[s] = 1 - expectations[s] * expectations[s];
			if (expectations[s] > expectations[max])
				max = s;
			sum += expectations[s];
		}

		if (sum == 0) return;

		for (s = 0; s < senses; s++)
			expectations[s] /= sum;

		if (probabilistic) {
			// expectation maximization
			for (s = 0; s < senses; s++)
				learnSkipGram(rand, word, syn0[s], syn1[s], neu1e[s], expectations[s]*alpha, context);
		} else {
			// greedy
			learnSkipGram(rand, word, syn0[max], syn1[max], neu1e[max], alpha, context);
		}
	}

	public void testPolysemousBagOfWords(Random rand, int word, float[][][] syn1, float[][] neu1, int senses)
	{
		int max = 0, s, k;
		float score = 0;
		float[] expectations = new float[senses];

		for (s = 0; s < senses; s++) {
			// hidden -> output
			for (k = 0; k < vector_size; k++) score += neu1[s][k] * syn1[s][word][k];
			expectations[s] = (1 - sigmoid.get(score));
			expectations[s] = 1 - expectations[s] * expectations[s];
			if (expectations[s] > expectations[max])
				max = s;
		}

		testBagOfWords(rand, word, syn1[max], neu1[max]);

	}

	public void testPolysemousSkipGram(Random rand, int word, float[][][] syn0, float[][][] syn1, int context, int senses)
	{
		int max = 0, s, k;
		float score = 0;
		float[] expectations = new float[senses];

		for (s = 0; s < senses; s++) {
			// hidden -> output
			for (k = 0; k < vector_size; k++) score += syn0[s][context][k] * syn1[s][word][k];
			expectations[s] = (1 - sigmoid.get(score));
			expectations[s] = 1 - expectations[s] * expectations[s];
			if (expectations[s] > expectations[max])
				max = s;
		}

		testSkipGram(rand, word, syn0[max], syn1[max], context);
	}

}
