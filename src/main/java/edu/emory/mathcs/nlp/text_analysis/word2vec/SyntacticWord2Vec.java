package edu.emory.mathcs.nlp.text_analysis.word2vec;

import edu.emory.mathcs.nlp.common.random.XORShiftRandom;
import edu.emory.mathcs.nlp.common.util.BinUtils;
import edu.emory.mathcs.nlp.component.template.node.NLPNode;
import edu.emory.mathcs.nlp.text_analysis.word2vec.reader.DependencyReader;
import edu.emory.mathcs.nlp.text_analysis.word2vec.optimizer.HierarchicalSoftmax;
import edu.emory.mathcs.nlp.text_analysis.word2vec.optimizer.NegativeSampling;
import edu.emory.mathcs.nlp.text_analysis.word2vec.reader.Reader;
import edu.emory.mathcs.nlp.text_analysis.word2vec.util.Vocabulary;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

/**
 * This is an extension of classical word2vec to include features of dependency syntax.
 *
 * @author Austin Blodgett
 */
public class SyntacticWord2Vec extends Word2Vec
{
    public SyntacticWord2Vec(String[] args) {
        super(args);
    }

    public void train(List<String> filenames) throws Exception
    {
        BinUtils.LOG.info("Reading vocabulary:\n");

        // ------- Austin's code -------------------------------------
        Vocabulary lemma_vocab  = new Vocabulary();
        Vocabulary depend_vocab  = new Vocabulary();

        DependencyReader reader = new DependencyReader(filenames.stream().map(File::new).collect(Collectors.toList()));

        // each word is a lemma, e.g., "go"
        lemma_vocab.learnParallel(reader.addFeature(NLPNode::getLemma).splitParallel(thread_size), min_count);
        // each word is a lemma with dependency, e.g., "root_go"
        depend_vocab.learnParallel(reader.addFeature(this::wordFeatures).splitParallel(thread_size), min_count);
        word_count_train = lemma_vocab.totalCount();

        in_vocab = lemma_vocab;
        out_vocab = depend_vocab;
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
        for (Reader<NLPNode> r: reader.splitParallel(thread_size))
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
        private Reader<NLPNode> reader;
        private int id;
        private long last_time = System.currentTimeMillis() - 14*60*100; // set back 14 minutes (first output after 60 seconds)

        /* Tasks are each parameterized by a reader which is dedicated to a section of the corpus
         * (not necesarily one file). The corpus is split to divide it evenly between Tasks without breaking up sentences. */
        public TrainTask(Reader<NLPNode> reader, int id)
        {
            this.reader = reader;
            this.id = id;
        }
        // -------------------------------------

        @Override
        public void run()
        {
            Random rand  = new XORShiftRandom(reader.hashCode());
            float[] neu1  = cbow ? new float[vector_size] : null;
            float[] neu1e = new float[vector_size];
            int     iter  = 0;
            int     index;
            List<NLPNode> words = null;

            while (true)
            {
                try {
                    words = reader.next();
                    if (words != null)
                    {
                        System.out.println("id "+id+" ");
                        for(NLPNode n : words)
                            System.out.print(n.getLemma()+" ");
                        System.out.println();
                    }
                } catch (IOException e) {
                    System.err.println("Reader failure: progress "+reader.progress());
                    e.printStackTrace();
                    System.exit(1);
                }

                if (words == null)
                {
                    if (++iter == train_iteration) break;
                    adjustLearningRate();
                    // readers have a built in restart button - Austin
                    try { reader.restart(); } catch (IOException e) { e.printStackTrace(); }
                    continue;
                }

                for (index=0; index<words.size(); index++)
                {
                    if (cbow) Arrays.fill(neu1, 0);
                    Arrays.fill(neu1e, 0);

                    if (cbow) bagOfWords(words, index, rand, neu1e, neu1);
                    else      skipGram  (words, index, rand, neu1e);
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


    void bagOfWords(List<NLPNode> words, int index, Random rand, float[] neu1e, float[] neu1)
    {
        int k, l, wc = 0;
        NLPNode word = words.get(index);
        int word_index = out_vocab.indexOf(wordFeatures(word));

        List<NLPNode> context_words = word.getDependentList(); // include dependents

        // input -> hidden
        for (NLPNode context : context_words)
        {
            int context_index = in_vocab.indexOf(context.getLemma());
            l = context_index * vector_size;
            for (k=0; k<vector_size; k++) neu1[k] += W[k+l];
            wc++;
        }

        if (wc == 0) return;
        for (k=0; k<vector_size; k++) neu1[k] /= wc;
        optimizer.learnBagOfWords(rand, word_index, V, neu1, neu1e, alpha_global);

        // hidden -> input
        for (NLPNode context : context_words)
        {
            int context_index = in_vocab.indexOf(context.getLemma());
            l = context_index * vector_size;

            for (k=0; k<vector_size; k++) W[k+l] += neu1e[k];
        }
    }

    void skipGram(List<NLPNode> words, int index, Random rand, float[] neu1e)
    {
        int k, l1;
        NLPNode word = words.get(index);
        int word_index = out_vocab.indexOf(wordFeatures(word));

        List<NLPNode> context_words = word.getDependentList(); // include dependents

        for (NLPNode context : context_words)
        {
            int context_index = in_vocab.indexOf(context.getLemma());
            l1 = context_index * vector_size;
            Arrays.fill(neu1e, 0);
            optimizer.learnSkipGram(rand, word_index, W, V, neu1e, alpha_global, l1);

            // hidden -> input
            for (k=0; k<vector_size; k++) W[l1+k] += neu1e[k];
        }
    }

    public String wordFeatures(NLPNode word)
    {
        return word.getDependencyLabel()+"_"+word.getLemma();
    }



    static public void main(String[] args)
    {
        new Word2Vec(args);
    }
}
