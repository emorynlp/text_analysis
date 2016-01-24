package edu.emory.mathcs.nlp.text_analysis.word2vec;

import edu.emory.mathcs.nlp.common.random.XORShiftRandom;
import edu.emory.mathcs.nlp.common.util.BinUtils;
import edu.emory.mathcs.nlp.text_analysis.word2vec.optimizer.HierarchicalSoftmax;
import edu.emory.mathcs.nlp.text_analysis.word2vec.optimizer.NegativeSampling;
import edu.emory.mathcs.nlp.text_analysis.word2vec.reader.Reader;
import edu.emory.mathcs.nlp.text_analysis.word2vec.util.Vocabulary;
import org.kohsuke.args4j.Option;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

/**
 * @author Austin Blodgett
 */
public class PolysemousWord2Vec extends Word2Vec
{
    @Option(name="-senses", usage="number of senses for each vector.", required=true, metaVar="<integer>")
    int senses = 0;

    volatile float[][] sense_dist;
    volatile float[] sense_norm;

    volatile public float[][] S; /* This object replaces W! */

    public PolysemousWord2Vec(String[] args) { super(args); }

    /** Initializes weights between the input layer to the hidden layer using random numbers between [-0.5, 0.5]. */
    void initNeuralNetwork()
    {
        int size1 = in_vocab.size() * vector_size;
        Random rand = new XORShiftRandom(1);

        S = new float[senses][size1]; // S[word_sense][vector_size*word_index + component]
        V = new float[size1]; // V[vector_size*word_index + component]

        for (int s=0; s<senses; s++) for (int i=0; i<size1; i++)
            S[s][i] = (float)((rand.nextDouble() - 0.5) / vector_size);
        // these keep track of proportionality of use for each sense
        sense_dist = new float[senses][in_vocab.size()]; // sense_dist[word_sense][word_index]
        sense_norm = new float[out_vocab.size()];         // sense_norm[word_index]
    }

    public void train(List<String> filenames) throws Exception
    {
        BinUtils.LOG.info("Reading vocabulary:\n");

        // ------- Austin's code -------------------------------------
        in_vocab = (out_vocab = new Vocabulary());

        List<Reader<String>> readers = getReader(filenames.stream().map(File::new).collect(Collectors.toList()))
                .splitParallel(thread_size);
        in_vocab.learnParallel(readers, min_count);
        word_count_train = in_vocab.totalCount();
        // -----------------------------------------------------------

        BinUtils.LOG.info(String.format("- types = %d, tokens = %d\n", in_vocab.size(), word_count_train));

        BinUtils.LOG.info("Initializing neural network.\n");
        initNeuralNetwork();

        BinUtils.LOG.info("Initializing optimizer.\n");
        optimizer = isNegativeSampling() ? new NegativeSampling(in_vocab, sigmoid, vector_size, negative_size) : new HierarchicalSoftmax(in_vocab, sigmoid, vector_size);

        BinUtils.LOG.info("Training vectors:");
        word_count_global = 0;
        alpha_global      = alpha_init;
        subsample_size    = subsample_threshold * word_count_train;
        ExecutorService executor = Executors.newFixedThreadPool(thread_size);

        // ------- Austin's code -------------------------------------
        start_time = System.currentTimeMillis();

        int id = 0;
        for (Reader<String> r: readers)
        {
            executor.execute(new TrainTask(r,id));
            id++;
        }
        // -----------------------------------------------------------

        executor.shutdown();

        try { executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS); }
        catch (InterruptedException e) {e.printStackTrace();}


        BinUtils.LOG.info("Saving word vectors.\n");

        save(new File(output_file));
    }

    class TrainTask implements Runnable
    {
        // ------- Austin ----------------------
        private Reader<String> reader;
        private int id;
        private long last_time = System.currentTimeMillis() - 14*60*100; // set back 14 minutes (first output after 60 seconds)

        /* Tasks are each parameterized by a reader which is dedicated to a section of the corpus
         * (not necesarily one file). The corpus is split to divide it evenly between Tasks without breaking up sentences. */
        public TrainTask(Reader<String> reader, int id)
        {
            this.reader = reader;
            this.id = id;
        }
        // -------------------------------------

        @Override
        public void run()
        {
            Random  rand  = new XORShiftRandom(reader.hashCode());

            float[][] neu1s  = cbow ? new float[senses][vector_size] : null;
            float[] neu1e = new float[vector_size];
            int     iter  = 0;
            int     index, window;
            int[]   words;

            while (true)
            {
                words = next(reader, rand, true);

                if (words == null)
                {
                    if (++iter == train_iteration) break;
                    adjustLearningRate();
                    // readers have a built in restart button - Austin
                    try { reader.restart(); } catch (IOException e) { e.printStackTrace(); }
                    continue;
                }

                for (index=0; index<words.length; index++)
                {
                    window = 1 + rand.nextInt() % max_skip_window;	// dynamic window size
                    if (cbow) Arrays.fill(neu1s, 0);
                    Arrays.fill(neu1e, 0);

                    if (cbow) bagOfWords(words, index, window, rand, neu1e, neu1s);
                    else      skipGram  (words, index, window, rand, neu1e);
                }

                // output progress every 15 minutes
                if(id == 0){
                    long now = System.currentTimeMillis();
                    if(now-last_time > 15*1000*60){
                        outputProgress(now);
                        last_time = now;
                    }
                }

            }
        }
    }

    void bagOfWords(int[] words, int index, int window, Random rand, float[] neu1e, float[][] neu1s)
    {
        int i, j, k, l, wc = 0, word = words[index];

        // input -> hidden
        for (i=-window,j=index+i; i<=window; i++,j++)
        {
            if (i == 0 || words.length <= j || j < 0) continue;
            l = words[j] * vector_size;
            for (int s=0; s<senses; s++) for (k = 0; k < vector_size; k++) neu1s[s][k] += S[s][l + k];
            wc++;
        }

        if (wc == 0) return;
        for (int s=0; s<senses; s++) for (k=0; k<vector_size; k++) neu1s[s][vector_size+k] /= wc;
        polysemousBagOfWords(rand, word, neu1e, neu1s);

        // hidden -> input
        for (i=-window,j=index+i; i<=window; i++,j++)
        {
            l = words[j] * vector_size;
            if (i == 0 || words.length <= j || j < 0) continue;
            for (int s=0; s<senses; s++) for (k = 0; k < vector_size; k++) S[s][k+l] += neu1e[k];
        }
    }

    public void polysemousBagOfWords(Random rand, int word, float[] neu1e, float[][] neu1s)
    {
        int max = 0, s, k, l;
        float score = 0, sum = 0;
        float[] E = new float[senses];

        l = word * vector_size;

        for (s = 0; s < senses; s++) {
            // hidden -> output
            for (k = 0; k < vector_size; k++) score += neu1s[s][k] * V[l+k];
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
            optimizer.learnBagOfWords(rand, word, V, neu1s[s], neu1e, alpha_global);
            sense_dist[s][word] += E[s];
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
            polysemousSkipGram(rand, word, neu1e, words[j]);

            // hidden -> input
            for (int s=0; s<senses; s++) for (k = 0; k < vector_size; k++) S[s][l1+k] += neu1e[k];
        }
    }

    public void polysemousSkipGram(Random rand, int word, float[] neu1e, int context)
    {
        int max = 0, s, k, l1, l2;
        float score = 0, sum = 0;
        float[] E = new float[senses];

        for (s = 0; s < senses; s++) {
            l1 = context * vector_size;
            l2 = word * vector_size;
            // hidden -> output
            for (k = 0; k < vector_size; k++) score += S[s][l1+k] * V[l2+k];
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
            l1 = context * vector_size;
            optimizer.learnSkipGram(rand, word, S[s], V, neu1e, E[s]*alpha_global, l1);
            sense_dist[s][word] += E[s];
        }
        sense_norm[word]++;
    }


    String senseToString(int sense, int word_index){
        return in_vocab.get(word_index).form
                +"."+String.format("%02d",sense)
                +"("+(int)(100*sense_dist[sense][word_index]/sense_norm[word_index])+"%)";
    }

    @Override
    public Map<String,float[]> toMap(boolean normalize)
    {
        Map<String,float[]> map = new HashMap<>();
        float[] vector;
        String key;
        int i, l;

        for (int s=0; s<senses; s++)
        {
            for (i = 0; i < in_vocab.size(); i++) {
                l = i * vector_size;
                key = senseToString(s, i);
                vector = Arrays.copyOfRange(S[s], l, l + vector_size);
                if (normalize) normalize(vector);
                map.put(key, vector);
            }
        }

        return map;
    }

    static public void main(String[] args) { new PolysemousWord2Vec(args); }
}
