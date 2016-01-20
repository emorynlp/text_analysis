package edu.emory.mathcs.nlp.text_analysis.word2vec;

import edu.emory.mathcs.nlp.common.util.BinUtils;
import edu.emory.mathcs.nlp.component.template.node.NLPNode;
import edu.emory.mathcs.nlp.text_analysis.word2vec.optimizer.HierarchicalSoftmax;
import edu.emory.mathcs.nlp.text_analysis.word2vec.optimizer.NegativeSampling;
import edu.emory.mathcs.nlp.text_analysis.word2vec.reader.DependencyReader;
import edu.emory.mathcs.nlp.text_analysis.word2vec.reader.Reader;
import edu.emory.mathcs.nlp.text_analysis.word2vec.util.Vocabulary;

import java.io.File;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

/**
 * @author Austin Blodgett
 */
public class POSWord2Vec extends Word2Vec {

    public POSWord2Vec(String[] args) {
        super(args);
    }

    String getWordLabel(NLPNode word)
    {
        return word.getPartOfSpeechTag()+"_"+word.getLemma();
    }

    @Override
    Reader<String> getReader(List<File> files)
    {
        return new DependencyReader(files).addFeature(this::getWordLabel);
    }

    static public void main(String[] args) { new POSWord2Vec(args); }
}
