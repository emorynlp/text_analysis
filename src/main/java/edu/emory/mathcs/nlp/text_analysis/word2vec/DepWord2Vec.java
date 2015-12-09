package edu.emory.mathcs.nlp.text_analysis.word2vec;

import edu.emory.mathcs.nlp.text_analysis.word2vec.reader.DependencyReader;
import edu.emory.mathcs.nlp.text_analysis.word2vec.reader.Reader;
import org.kohsuke.args4j.Option;

import java.io.File;
import java.util.List;

/**
 * Created by austin on 12/1/2015.
 */
public class DepWord2Vec extends Word2Vec {

    @Option(name="-mode", usage="mode 0: lemma; mode 1: dependency; mode 2: part of speech (default: 0).", required=false, metaVar="<int>")
    int mode = DependencyReader.LEMMA_MODE;

    public DepWord2Vec(String[] args) {super(args);}

    @Override
    Reader<?> initReader(List<File> files){
        return new DependencyReader(files, mode);
    }

    static public void main(String[] args)
    {
        new DepWord2Vec(args);
    }
}
