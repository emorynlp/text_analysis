package edu.emory.mathcs.nlp.text_analysis.word2vec;

import edu.emory.mathcs.nlp.common.util.BinUtils;
import edu.emory.mathcs.nlp.common.util.FileUtils;
import edu.emory.mathcs.nlp.text_analysis.word2vec.optimizer.HierarchicalSoftmax;
import edu.emory.mathcs.nlp.text_analysis.word2vec.optimizer.NegativeSampling;
import edu.emory.mathcs.nlp.text_analysis.word2vec.reader.DependencyReader;
import edu.emory.mathcs.nlp.text_analysis.word2vec.reader.Reader;
import edu.emory.mathcs.nlp.text_analysis.word2vec.reader.SentenceReader;
import edu.emory.mathcs.nlp.text_analysis.word2vec.util.Vocabulary;
import org.kohsuke.args4j.Option;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by austin on 12/1/2015.
 */
public class DepWord2Vec extends Word2Vec {

    @Option(name="-mode", usage="mode 0: lemma; mode 1: dependency; mode 2: part of speech (default: 0).", required=false, metaVar="<int>")
    int mode = DependencyReader.LEMMA_MODE;

    public DepWord2Vec(String[] args) {super(args);}

//	=================================== Training ===================================


    public void train(List<String> filenames) throws Exception
    {
        List<File> files = new ArrayList<File>();
        for(String f : filenames)
            files.add(new File(f));
        Reader<?> training_reader = new DependencyReader(files, mode);

        System.out.println("Word2Vec");
        System.out.println((cbow ? "Continuous Bag of Words" : "Skipgrams") + ", " + (isNegativeSampling() ? "Hierarchical Softmax" : "Negative Sampling"));
        System.out.println("Reading vocabulary:");

        BinUtils.LOG.info("Reading vocabulary:\n");
        vocab = new Vocabulary();
        vocab.learn(training_reader, min_count);
        word_count_train = vocab.totalWords();
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

        startThreads(training_reader, false);
        outputProgress(System.currentTimeMillis());

        if(eval_path != null){
            System.out.println("Starting Evaluation:");
            List<File> test_files = new ArrayList<File>();
            for(String f : FileUtils.getFileList(eval_path, train_ext, false))
                test_files.add(new File(f));

            Reader<?> test_reader = new DependencyReader(test_files, mode);
            startThreads(test_reader, true);
            System.out.println("Evaluated Error: " + optimizer.getError());
        }
        if(triad_file != null) {
            System.out.println("Triad Evaluation:");
            evaluateVectors(new File(triad_file));
        }

        BinUtils.LOG.info("Saving word vectors.\n");
        save();
    }


    static public void main(String[] args)
    {
        new DepWord2Vec(args);
    }
}
