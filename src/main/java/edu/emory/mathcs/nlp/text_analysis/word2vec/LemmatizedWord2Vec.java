package edu.emory.mathcs.nlp.text_analysis.word2vec;

import edu.emory.mathcs.nlp.component.template.node.NLPNode;
import edu.emory.mathcs.nlp.text_analysis.word2vec.reader.DependencyReader;
import edu.emory.mathcs.nlp.text_analysis.word2vec.reader.Reader;

import java.io.File;
import java.util.List;

/**
 * @author Austin Blodgett
 */
public class LemmatizedWord2Vec extends Word2Vec
{

    public LemmatizedWord2Vec(String[] args) {
        super(args);
    }

    @Override
    Reader<String> getReader(List<File> files)
    {
        return new DependencyReader(files).addFeature(NLPNode::getLemma);
    }

    static public void main(String[] args) { new LemmatizedWord2Vec(args); }
}
