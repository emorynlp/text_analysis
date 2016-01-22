package edu.emory.mathcs.nlp.vsm;

import java.io.File;
import java.util.List;

import edu.emory.mathcs.nlp.component.template.node.NLPNode;
import edu.emory.mathcs.nlp.vsm.reader.DEPTreeReader;
import edu.emory.mathcs.nlp.vsm.reader.Reader;

/**
 * @author Austin Blodgett
 */
public class POSLemma2Vec extends Word2Vec {

    public POSLemma2Vec(String[] args) {
        super(args);
    }

    String getWordLabel(NLPNode word)
    {
        return word.getPartOfSpeechTag()+"_"+word.getLemma();
    }

	@Override
	@SuppressWarnings("resource")
    Reader<String> getReader(List<File> files)
    {
        return new DEPTreeReader(files).addFeature(this::getWordLabel);
    }

    static public void main(String[] args) { new POSLemma2Vec(args); }
}
