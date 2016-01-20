package edu.emory.mathcs.nlp.text_analysis.word2vec;

import edu.emory.mathcs.nlp.common.random.XORShiftRandom;
import org.kohsuke.args4j.Option;

import java.util.Arrays;
import java.util.Random;

/**
 * @author Austin Blodgett
 */
public class PolysemousWord2Vec extends Word2Vec
{
    @Option(name="-senses", usage="number of senses for each vector.", required=true, metaVar="<integer>")
    int senses = 5;

    volatile float[] sense_dist;
    volatile float[] sense_norm;

    public PolysemousWord2Vec(String[] args) { super(args); }

    /** Initializes weights between the input layer to the hidden layer using random numbers between [-0.5, 0.5]. */
    void initNeuralNetwork()
    {
        int size1 = senses * in_vocab.size() * vector_size;
        int size2 = out_vocab.size() * vector_size;
        Random rand = new XORShiftRandom(1);

        W = new float[size1]; // W[vocab_size*word_sense + vector_size*word_index + component]
        V = new float[size2]; // V[vector_size*word_index + component]

        for (int i=0; i<size1; i++)
            W[i] = (float)((rand.nextDouble() - 0.5) / vector_size);
        // these keep track of proportionality of use for each sense
        sense_dist = new float[size1];
        sense_norm = new float[size2];
    }

    void bagOfWords(int[] words, int index, int window, Random rand, float[] neu1e, float[] neu1)
    {
        int i, j, k, l, wc = 0, word = words[index];

        // input -> hidden
        for (i=-window,j=index+i; i<=window; i++,j++)
        {
            if (i == 0 || words.length <= j || j < 0) continue;
            l = words[j] * vector_size;
            // for each sense
            for (int s=0; s<senses; s++) for (k=0; k<vector_size; k++) neu1[s*in_vocab.size()+k] += W[s*in_vocab.size()+l+k];
            wc++;
        }

        if (wc == 0) return;
        for (k=0; k<vector_size; k++) neu1[k] /= wc;
        polysemousBagOfWords(rand, word, neu1e, neu1);

        // hidden -> input
        for (i=-window,j=index+i; i<=window; i++,j++)
        {
            if (i == 0 || words.length <= j || j < 0) continue;
            l = words[j] * vector_size;
            for (k=0; k<vector_size; k++) W[k+l] += neu1e[k];
        }
    }

    public void polysemousBagOfWords(Random rand, int word, float[] neu1e, float[] neu1)
    {
        int max = 0, s, k;
        float score = 0, sum = 0;
        float[] E = new float[senses];

        for (s = 0; s < senses; s++) {
            // hidden -> output
            for (k = 0; k < vector_size; k++) score += neu1[s*in_vocab.size() + k] * V[word*vector_size + k];
            E[s] = (1 - sigmoid.get(score));
            E[s] = 1 - E[s] * E[s]; // 1 - squared error
            if (E[s] > E[max])
                max = s;
            sum += E[s];
        }
        if (sum == 0) return;
        for (s = 0; s < senses; s++)
            E[s] /= sum;

        // expectation maximization
        for (s = 0; s < senses; s++) {
            optimizer.learnBagOfWords(rand, word, V, neu1, neu1e, alpha_global);
            sense_dist[s*in_vocab.size()+word] += E[s];
        }
        sense_norm[word]++;
    }

    void skipGram(int[] words, int index, int window, Random rand, float[] neu1e)
    {
        int i, j, k, l1, word = words[index];

        for (i=-window,j=index+i; i<=window; i++,j++)
        {
            if (i == 0 || words.length <= j || j < 0) continue;
            l1 = words[j] * vector_size;
            Arrays.fill(neu1e, 0);
            polysemousSkipGram(rand, word, neu1e, l1);

            // hidden -> input
            for (k=0; k<vector_size; k++) W[l1+k] += neu1e[k];
        }
    }

    public void polysemousSkipGram(Random rand, int word, float[] neu1e, int l1)
    {
        int max = 0, s, k;
        float score = 0, sum = 0;
        float[] E = new float[senses];

        for (s = 0; s < senses; s++) {
            // hidden -> output
            for (k = 0; k < vector_size; k++) score += W[s*in_vocab.size() + l1] * V[word*vector_size + k];
            E[s] = (1 - sigmoid.get(score));
            E[s] = 1 - E[s] * E[s];
            if (E[s] > E[max])
                max = s;
            sum += E[s];
        }

        if (sum == 0) return;

        for (s = 0; s < senses; s++)
            E[s] /= sum;

        // expectation maximization
        for (s = 0; s < senses; s++) {
            optimizer.learnSkipGram(rand, word, W, V, neu1e, E[s]*alpha_global, l1);
            sense_dist[s*in_vocab.size() + word] += E[s];
        }
        sense_norm[word]++;
    }

    static public void main(String[] args) { new PolysemousWord2Vec(args); }
}
