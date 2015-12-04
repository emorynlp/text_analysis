package edu.emory.mathcs.nlp.text_analysis.word2vec;

import edu.emory.mathcs.nlp.common.random.XORShiftRandom;
import edu.emory.mathcs.nlp.common.util.BinUtils;
import edu.emory.mathcs.nlp.common.util.FileUtils;
import edu.emory.mathcs.nlp.common.util.MathUtils;
import edu.emory.mathcs.nlp.text_analysis.word2vec.optimizer.HierarchicalSoftmax;
import edu.emory.mathcs.nlp.text_analysis.word2vec.optimizer.NegativeSampling;
import edu.emory.mathcs.nlp.text_analysis.word2vec.reader.DependencyReader;
import edu.emory.mathcs.nlp.text_analysis.word2vec.reader.Reader;
import edu.emory.mathcs.nlp.text_analysis.word2vec.reader.SentenceReader;
import edu.emory.mathcs.nlp.text_analysis.word2vec.util.Vocabulary;
import edu.emory.mathcs.nlp.text_analysis.word2vec.util.Word;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * Created by austin on 12/1/2015.
 */
public class SyntacticWord2Vec extends Word2Vec {

    Vocabulary depend_vocab;
    HashMap<Integer, Integer> dependToLemma;

    public SyntacticWord2Vec(String[] args) {super(args);}

//	=================================== Training ===================================

    public void train(List<String> filenames) throws Exception
    {
        List<File> files = new ArrayList<>();
        for(String filename : filenames)
            files.add(new File(filename));
        Reader<?> lemma_reader = new DependencyReader(files, DependencyReader.LEMMA_MODE);
        Reader<?> depend_reader = new DependencyReader(files, DependencyReader.DEPEND_MODE);

        System.out.println("Syntactic Word2Vec");
        System.out.println((cbow ? "Continuous Bag of Words" : "Skipgrams") + ", " + (isNegativeSampling() ? "Hierarchical Softmax" : "Negative Sampling"));
        System.out.println("Reading vocabulary:");

        BinUtils.LOG.info("Reading vocabulary:\n");
        vocab = new Vocabulary();
        depend_vocab = new Vocabulary();
        vocab.learn(lemma_reader, min_count);
        depend_vocab.learn(depend_reader, min_count);
        word_count_train = vocab.totalWords();
        dependToLemma = new HashMap<Integer, Integer>();

        // each string in depend_vocab is of the form [dependency]_[lemma]
        for(int i=0; i<depend_vocab.size(); i++)
            dependToLemma.put(i, vocab.indexOf(depend_vocab.get(i).form.split("_")[1]));

        BinUtils.LOG.info(String.format("- types = %d, tokens = %d\n", vocab.size(), word_count_train));


        System.out.println("Vocab size "+vocab.size()+", Total Word Count "+word_count_train+"\n");
        System.out.println("Starting training: "+train_path);
        System.out.println("Files "+files.size()+", threads "+thread_size+", iterations "+train_iteration);

        BinUtils.LOG.info("Initializing neural network.\n");
        initNeuralNetwork();

        BinUtils.LOG.info("Initializing optimizer.\n");
        optimizer = isNegativeSampling() ? new NegativeSampling(vocab, sigmoid, vector_size, negative_size) : new HierarchicalSoftmax(vocab, sigmoid, vector_size);

        BinUtils.LOG.info("Training vectors:");
        word_count_global = 0;
        alpha_global      = alpha_init;
        subsample_size    = subsample_threshold * word_count_train;

        startThreads(depend_reader, false);

        if(eval_path != null){
            System.out.println("Starting Evaluation:");
            List<File> test_files = new ArrayList<File>();
            for(String f : FileUtils.getFileList(eval_path, train_ext, false))
                test_files.add(new File(f));

            Reader<?> depend_test_reader = new SentenceReader(test_files, lowercase, sentence_border);
            startThreads(depend_test_reader, true);
            System.out.println("Evaluated Error: " + optimizer.getError());
        }

        if(triad_file != null) {
            System.out.println("Triad Evaluation:");
            evaluateVectors(new File(triad_file));
        }

        BinUtils.LOG.info("Saving word vectors.\n");
        save();
    }


    /** Initializes weights between the input layer to the hidden layer using random numbers between [-0.5, 0.5]. */
    void initNeuralNetwork()
    {
        int lemma_size = vocab.size() * vector_size;
        int depend_size = depend_vocab.size() * vector_size;
        Random rand = new XORShiftRandom(1);

        W = new float[lemma_size];
        V = new float[depend_size];

        for (int i=0; i<lemma_size; i++)
            W[i] = (float)((rand.nextDouble() - 0.5) / vector_size);
    }

    // next returns indices in depend_vocab which can be mapped to lemmas later
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
    void bagOfWords(int[] words, int index, int window, Random rand, float[] neu1e, float[] neu1, boolean evaluate)
    {
        int i, j, k, l, wc = 0, word = words[index];
        int lemma;
        // input -> hidden
        for (i=-window,j=index+i; i<=window; i++,j++)
        {
            if (i == 0 || words.length <= j || j < 0) continue;
            lemma = dependToLemma.get(words[j]);
            if (lemma < 0) continue;
            l = lemma * vector_size;
            for (k=0; k<vector_size; k++) neu1[k] += W[k+l];
            wc++;
        }

        if (wc == 0) return;
        for (k=0; k<vector_size; k++) neu1[k] /= wc;

        if(evaluate){
            optimizer.testBagOfWords(rand, word, V, neu1, neu1e, alpha_global);
            return;
        }

        optimizer.learnBagOfWords(rand, word, V, neu1, neu1e, alpha_global);

        // hidden -> input
        for (i=-window,j=index+i; i<=window; i++,j++)
        {
            if (i == 0 || words.length <= j || j < 0) continue;
            lemma = dependToLemma.get(words[j]);
            if (lemma < 0) continue;
            l = lemma * vector_size;
            for (k=0; k<vector_size; k++) W[k+l] += neu1e[k];
        }
    }


    // skipGram should map words[j] to lemma
    void skipGram(int[] words, int index, int window, Random rand, float[] neu1e, boolean evaluate)
    {
        int i, j, k, l1, word = words[index];
        int lemma;

        for (i=-window,j=index+i; i<=window; i++,j++)
        {
            if (i == 0 || words.length <= j || j < 0) continue;
            lemma = dependToLemma.get(words[j]);
            if (lemma < 0) continue;
            l1 = lemma * vector_size;
            Arrays.fill(neu1e, 0);


            if(evaluate){
                optimizer.learnSkipGram(rand, word, W, V, neu1e, alpha_global, l1);
                continue;
            }

            optimizer.learnSkipGram(rand, word, W, V, neu1e, alpha_global, l1);

            // hidden -> input
            for (k=0; k<vector_size; k++) W[l1+k] += neu1e[k];
        }
    }


    static public void main(String[] args)
    {
        new DepWord2Vec(args);
    }
}
