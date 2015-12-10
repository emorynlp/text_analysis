package edu.emory.mathcs.nlp.text_analysis.word2vec;

import edu.emory.mathcs.nlp.common.random.XORShiftRandom;
import edu.emory.mathcs.nlp.common.util.BinUtils;
import edu.emory.mathcs.nlp.common.util.MathUtils;
import edu.emory.mathcs.nlp.text_analysis.word2vec.optimizer.HierarchicalSoftmax;
import edu.emory.mathcs.nlp.text_analysis.word2vec.optimizer.NegativeSampling;
import edu.emory.mathcs.nlp.text_analysis.word2vec.reader.DependencyReader;
import edu.emory.mathcs.nlp.text_analysis.word2vec.reader.Reader;
import edu.emory.mathcs.nlp.text_analysis.word2vec.util.Vocabulary;
import org.kohsuke.args4j.Option;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by austin on 12/1/2015.
 */
public class SyntacticWord2Vec extends Word2Vec {

    @Option(name="-tree-window", usage="If set, use tree distance instead of word-order distance in window.", required=false, metaVar="<boolean>")
    boolean tree_window = false;

    Vocabulary depend_vocab;
    HashMap<Integer, Integer> dependToLemma;

    public SyntacticWord2Vec(String[] args) {super(args);}

//	=================================== Training ===================================

    DependencyReader initReader(List<File> files, int mode){
        return new DependencyReader(files, mode);
    }

    @Override
    public void train(List<String> filenames) throws Exception
    {
        List<File> files = filenames.stream().map(File::new).collect(Collectors.toList());

        DependencyReader lemma_reader = initReader(files, DependencyReader.LEMMA_MODE);
        DependencyReader depend_reader = initReader(files, DependencyReader.DEPEND_MODE);

        BinUtils.LOG.info("\nSyntactic Word2Vec\n");
        BinUtils.LOG.info("Reading vocabulary:");

        vocab = new Vocabulary();
        depend_vocab = new Vocabulary();
        vocab.learn(lemma_reader, min_count);
        depend_vocab.learn(depend_reader, min_count);
        word_count_train = vocab.totalWords();
        dependToLemma = new HashMap<>();

        // each string in depend_vocab is of the form [dependency]_[lemma]
        for(int i=0; i<depend_vocab.size(); i++)
            dependToLemma.put(i, vocab.indexOf(depend_vocab.get(i).form.split("_")[1]));

        BinUtils.LOG.info("Vocab size "+vocab.size()+", Total Word Count "+word_count_train+"\n");
        
        initNeuralNetwork();

        optimizer = isNegativeSampling() ? new NegativeSampling(vocab, sigmoid, vector_size, negative_size) : new HierarchicalSoftmax(vocab, sigmoid, vector_size);
        
        BinUtils.LOG.info("Training vectors "+train_path);
        BinUtils.LOG.info((cbow ? "Continuous Bag of Words" : "Skipgrams") + ", " + (isNegativeSampling() ? "Negative Sampling" : "Hierarchical Softmax"));
        BinUtils.LOG.info("Files "+files.size()+", threads "+thread_size+", iterations "+train_iteration+"\n");

        word_count_global = 0;
        alpha_global      = alpha_init;
        subsample_size    = subsample_threshold * word_count_train;

        startThreads(depend_reader);

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

    /** Initializes weights between the input layer to the hidden layer using random numbers between [-0.5, 0.5]. */
    @Override
    void initNeuralNetwork()
    {
        Random rand = new XORShiftRandom(1);

        W = new float[vocab.size()][vector_size];
        V = new float[depend_vocab.size()][vector_size];

        for (int i=0; i<vocab.size(); i++)
            for (int j=0; j<vector_size; j++)
                W[i][j] = (float)((rand.nextDouble() - 0.5) / vector_size);
    }

    // next returns indices in depend_vocab which can be mapped to lemmas later
    @Override
    int[] next(Reader<?> reader, Random rand) throws IOException
    {
        Object[] words = reader.next();
        if (words == null) return null;
        int[] next = new int[words.length];
        int i, j, index, count = 0;
        double d;

        for (i=0,j=0; i<words.length; i++)
        {
            index = depend_vocab.indexOf(words[i].toString());
            if (index < 0) continue;
            count++;

            // sub-sampling: randomly discards frequent words
            if (subsample_threshold > 0)
            {
                d = (Math.sqrt(MathUtils.divide(depend_vocab.get(index).count, subsample_size)) + 1) * (subsample_size / depend_vocab.get(index).count);
                if (d < rand.nextDouble()) continue;
            }

            next[j++] = index;
        }

        word_count_global += count;
        return (j == 0) ? next(reader, rand) : (j == words.length) ? next : Arrays.copyOf(next, j);
    }

    // bagOfWords should map words[j] to lemma
    @Override
    void bagOfWords(int[] words, int index, int window, Random rand, float[] neu1e, float[] neu1)
    {
        int i, j, k, wc = 0, context, word = words[index];

        // input -> hidden
        for (i=-window,j=index+i; i<=window; i++,j++)
        {
            if (i == 0 || words.length <= j || j < 0) continue;
            context = dependToLemma.get(words[j]);
            if (context < 0) continue;

            for (k=0; k<vector_size; k++) neu1[k] += W[context][k];
            wc++;
        }

        if (wc == 0) return;
        for (k=0; k<vector_size; k++) neu1[k] /= wc;

        optimizer.learnBagOfWords(rand, word, V, neu1, neu1e, alpha_global);

        // hidden -> input
        for (i=-window,j=index+i; i<=window; i++,j++)
        {
            if (i == 0 || words.length <= j || j < 0) continue;
            context = dependToLemma.get(words[j]);
            if (context < 0) continue;

            for (k=0; k<vector_size; k++) W[context][k] += neu1e[k];
        }
    }


    // skipGram should map words[j] to lemma
    @Override
    void skipGram(int[] words, int index, int window, Random rand, float[] neu1e)
    {
        int i, j, k, context, word = words[index];

        for (i=-window,j=index+i; i<=window; i++,j++)
        {
            if (i == 0 || words.length <= j || j < 0) continue;
            context = dependToLemma.get(words[j]);
            if (context < 0) continue;
            Arrays.fill(neu1e, 0);

            optimizer.learnSkipGram(rand, word, W, V, neu1e, alpha_global, context);

            // hidden -> input
            for (k=0; k<vector_size; k++) W[context][k] += neu1e[k];
        }
    }

    static public void main(String[] args)
    {
        new SyntacticWord2Vec(args);
    }
}
