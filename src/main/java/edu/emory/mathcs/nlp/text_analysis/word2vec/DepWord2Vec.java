package edu.emory.mathcs.nlp.text_analysis.word2vec;

import edu.emory.mathcs.nlp.common.util.BinUtils;
import edu.emory.mathcs.nlp.text_analysis.word2vec.optimizer.HierarchicalSoftmax;
import edu.emory.mathcs.nlp.text_analysis.word2vec.optimizer.NegativeSampling;
import edu.emory.mathcs.nlp.text_analysis.word2vec.reader.DependencyReader;
import edu.emory.mathcs.nlp.text_analysis.word2vec.reader.Reader;
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
        List<File> files = new ArrayList<>();
        for(String filename : filenames)
            files.add(new File(filename));
        Reader<?> reader = new DependencyReader(files, mode);
        Reader<?>[] r = evaluate ? reader.trainingAndTest() : null;

        Reader<?> training_reader = evaluate ? r[0] : reader;
        Reader<?> test_reader = evaluate ? r[1] : null;


        BinUtils.LOG.info("Reading vocabulary:\n");
        vocab = new Vocabulary();
        vocab.learn(training_reader, min_count);
        word_count_train = vocab.totalWords();
        BinUtils.LOG.info(String.format("- types = %d, tokens = %d\n", vocab.size(), word_count_train));

        BinUtils.LOG.info("Initializing neural network.\n");
        initNeuralNetwork();

        BinUtils.LOG.info("Initializing optimizer.\n");
        optimizer = isNegativeSampling() ? new NegativeSampling(vocab, sigmoid, vector_size, negative_size) : new HierarchicalSoftmax(vocab, sigmoid, vector_size);

        BinUtils.LOG.info("Training vectors:");
        word_count_global = 0;
        alpha_global      = alpha_init;
        subsample_size    = subsample_threshold * word_count_train;

        startThreads(training_reader, false);

        if(evaluate){
            startThreads(test_reader, true);
            System.out.println("Evaluated Error: " + optimizer.getError());
        }
        if(triad_file != null)
            evaluateVectors(new File(triad_file));

        BinUtils.LOG.info("Saving word vectors.\n");
        save();
    }


    static public void main(String[] args)
    {
        new DepWord2Vec(args);
    }
}
