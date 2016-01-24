package edu.emory.mathcs.nlp.vsm;

import java.io.File;
import java.util.List;

import edu.emory.mathcs.nlp.component.template.node.NLPNode;
import edu.emory.mathcs.nlp.vsm.reader.DEPTreeReader;
import edu.emory.mathcs.nlp.vsm.reader.Reader;

/**
 * @author Austin Blodgett
 */
public class Lemma2Vec extends Word2Vec
{

    public Lemma2Vec(String[] args) {
        super(args);
    }

	@Override
	@SuppressWarnings("resource")
    Reader<String> getReader(List<File> files)
    {
        return new DEPTreeReader(files).addFeature(NLPNode::getLemma);
    }

    static public void main(String[] args) { new Lemma2Vec(args); }
}
