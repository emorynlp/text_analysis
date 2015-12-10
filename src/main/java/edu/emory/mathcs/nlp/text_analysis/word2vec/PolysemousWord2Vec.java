package edu.emory.mathcs.nlp.text_analysis.word2vec;

import edu.emory.mathcs.nlp.common.random.XORShiftRandom;
import edu.emory.mathcs.nlp.common.util.BinUtils;
import edu.emory.mathcs.nlp.text_analysis.word2vec.optimizer.HierarchicalSoftmax;
import edu.emory.mathcs.nlp.text_analysis.word2vec.optimizer.NegativeSampling;
import edu.emory.mathcs.nlp.text_analysis.word2vec.reader.Reader;
import edu.emory.mathcs.nlp.text_analysis.word2vec.util.Vocabulary;
import org.kohsuke.args4j.Option;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

/**
 * Created by austin on 12/9/2015.
 */
public class PolysemousWord2Vec extends Word2Vec {


    @Option(name="-senses", usage="number of senses for each word (default: 5).", required=true, metaVar="<int>")
    int senses = 5;
    @Option(name="-probabilistic", usage="If set, use probabilistic (instead of greedy) sense training.", required=false, metaVar="<boolean>")
    boolean probabilistic = false;

    public float[][][] W; // weights between input and hidden layers (W[sense][word][component])
    public float[][][] V; // weights between input and hidden layers (V[sense][word][component])

    static final int MAX_SENSES = 100;

    public PolysemousWord2Vec(String[] params) { super(params); }


    void train(List<String> filenames) throws Exception
    {
        List<File> files = filenames.stream().map(File::new).collect(Collectors.toList());
        Reader<?> training_reader = initReader(files);

        if(senses > MAX_SENSES) senses = MAX_SENSES;

        BinUtils.LOG.info("\nPolysemous Word2Vec");
        BinUtils.LOG.info("Sense: "+senses+"\n");
        BinUtils.LOG.info("Reading vocabulary:");

        vocab = new Vocabulary();
        vocab.learn(training_reader, min_count);
        word_count_train = vocab.totalWords();
        BinUtils.LOG.info("Vocab size "+vocab.size()+", Total Word Count "+word_count_train+"\n");

        initNeuralNetwork();

        optimizer = isNegativeSampling() ? new NegativeSampling(vocab, sigmoid, vector_size, negative_size) : new HierarchicalSoftmax(vocab, sigmoid, vector_size);

        BinUtils.LOG.info("Training vectors "+train_path);
        BinUtils.LOG.info((cbow ? "Continuous Bag of Words" : "Skipgrams") + ", " + (isNegativeSampling() ? "Negative Sampling" : "Hierarchical Softmax"));
        BinUtils.LOG.info("Files "+files.size()+", threads "+thread_size+", iterations "+train_iteration+"\n");

        word_count_global = 0;
        alpha_global      = alpha_init;
        subsample_size    = subsample_threshold * word_count_train;

        start_time = System.currentTimeMillis();

        startThreads(training_reader);

        end_time = System.currentTimeMillis();

        outputProgress(end_time);
        BinUtils.LOG.info("\nTotal time: "+((end_time - start_time)/1000/60/60f)+" hours");

        BinUtils.LOG.info("Saving word vectors.");
        saveVectors();

        if(model_file != null){
            BinUtils.LOG.info("Saving Word2Vec model.");
            saveModel();
        }
        BinUtils.LOG.info("");
    }

    void startThreads(Reader<?> reader) throws IOException
    {
        Reader<?>[] readers = reader.split(thread_size);
        reader.close();
        ExecutorService executor = Executors.newFixedThreadPool(thread_size);

        for (int i = 0; i < thread_size; i++)
            executor.execute(new TrainTask(readers[i], i));

        executor.shutdown();
        try {
            executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        for (int i = 0; i < thread_size; i++)
            readers[i].close();
    }

    class TrainTask implements Runnable
    {
        private Reader<?> reader;
        private final int id;

        private long last_time;

        TrainTask(Reader<?> reader, int id)
        {
            this.reader = reader;
            this.id = id;

            // output after 10 seconds
            last_time = start_time + 10000;
        }

        @Override
        public void run()
        {
            Random  rand  = new XORShiftRandom(reader.hashCode());
            float[][] neu1  = cbow ? new float[senses][vector_size] : null;
            float[][] neu1e = new float[senses][vector_size];
            int     iter  = 0;
            int     index, window;
            int[]   words = null;

            while (true)
            {
                try {
                    words = next(reader, rand);
                } catch (IOException e) {
                    e.printStackTrace();
                }

                if (words == null)
                {
                    if (++iter == train_iteration) break;
                    adjustLearningRate();
                    try {
                        reader.startOver();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    continue;
                }

                for (index=0; index<words.length; index++)
                {
                    window = 1 + rand.nextInt() % max_skip_window;	// dynamic window size
                    if (cbow) for(int s=0; s<senses; s++) Arrays.fill(neu1[s], 0);
                    for(int s=0; s<senses; s++) Arrays.fill(neu1e[s], 0);

                    if (cbow) bagOfWords(words, index, window, rand, neu1e, neu1);
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

    /** Initializes weights between the input layer to the hidden layer using random numbers between [-0.5, 0.5]. */
    void initNeuralNetwork()
    {
        Random rand = new XORShiftRandom(1);

        W = new float[senses][vocab.size()][vector_size];
        V = new float[senses][vocab.size()][vector_size];

        for (int s=0; s<senses; s++)
            for (int w=0; w<vocab.size(); w++)
                for (int k=0; k<vector_size; k++)
                    W[s][w][k] = (float)((rand.nextDouble() - 0.5) / vector_size);
    }

    void bagOfWords(int[] words, int index, int window, Random rand, float[][] neu1e, float[][] neu1)
    {
        int i, j, k, s, wc = 0, context, word = words[index];

        // input -> hidden
        for (i=-window,j=index+i; i<=window; i++,j++)
        {
            if (i == 0 || words.length <= j || j < 0) continue;
            context = words[j];
            for (k=0; k<vector_size; k++)
                for (s=0; s<senses; s++)
                    neu1[s][k] += W[s][context][k];
            wc++;
        }

        if (wc == 0) return;
        for (k=0; k<vector_size; k++)
            for (s=0; s<senses; s++)
                neu1[s][k] /= wc;

        optimizer.polysemousBagOfWords(probabilistic, rand, word, V, neu1, neu1e, alpha_global, senses);

        // hidden -> input
        for (i=-window,j=index+i; i<=window; i++,j++)
        {
            if (i == 0 || words.length <= j || j < 0) continue;
            context = words[j];
            for (k=0; k<vector_size; k++)
                for (s=0; s<senses; s++)
                    W[s][context][k] += neu1e[s][k];
        }
    }

    void skipGram(int[] words, int index, int window, Random rand, float[][] neu1e)
    {
        int i, j, k, s, context, word = words[index];

        for (i=-window,j=index+i; i<=window; i++,j++)
        {
            if (i == 0 || words.length <= j || j < 0) continue;
            context = words[j];
            for(s=0; s<senses; s++) Arrays.fill(neu1e[s], 0);

            optimizer.polysemousSkipGram(probabilistic, rand, word, W, V, neu1e, alpha_global, context, senses);

            // hidden -> input
            for (k=0; k<vector_size; k++)
                for (s=0; s<senses; s++)
                    W[s][context][k] += neu1e[s][k];
        }
    }

    Map<String,float[]> toMap(boolean normalize)
    {
        Map<String,float[]> map = new HashMap<>();
        float[] vector;
        String key;

        for (int s=0; s<senses; s++)
        {
            for (int i=0; i<vocab.size(); i++)
            {
                key = vocab.get(i).form+"."+String.format("%02d",s);
                vector = W[s][i];
                if (normalize) normalize(vector);
                map.put(key, vector);
            }
        }

        return map;
    }

    public static void main(String[] args) {
        new PolysemousWord2Vec(args);
    }
}
