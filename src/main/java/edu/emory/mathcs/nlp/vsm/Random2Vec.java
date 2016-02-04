package edu.emory.mathcs.nlp.vsm;

import edu.emory.mathcs.nlp.common.util.BinUtils;
import edu.emory.mathcs.nlp.vsm.reader.Reader;
import edu.emory.mathcs.nlp.vsm.util.Vocabulary;

import java.io.File;
import java.util.List;
import java.util.stream.Collectors;

/**
 * This class is strictly to be used as a control for evaluating Word2Vec against.
 * It just initializes random word vectors and then outputs them, with no training.
 * @author Austin Blodgett
 */
public class Random2Vec extends Word2Vec
{
    public Random2Vec(String[] args) {
        super(args);
    }

    @Override
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

        BinUtils.LOG.info("Saving word vectors.\n");

        save(new File(output_file));
    }


    static public void main(String[] args) { new Random2Vec(args); }
}
