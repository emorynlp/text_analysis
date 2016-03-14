package edu.emory.mathcs.nlp.vsm;

import java.io.Serializable;
import java.util.List;

import edu.emory.mathcs.nlp.component.template.node.NLPNode;
import edu.emory.mathcs.nlp.vsm.reader.Reader;
import edu.emory.mathcs.nlp.vsm.util.Vocabulary;

public class VSMModel implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	float[] W;
	float[] V;
	Vocabulary in_vocab;
	Vocabulary out_vocab;


	public float[] getW() {
		return W;
	}

	public float[] getV() {
		return V;
	}

	public Vocabulary getIn_vocab() {
		return in_vocab;
	}

	public Vocabulary getOut_vocab() {
		return out_vocab;
	}

	
	public VSMModel(float[] w, float[] v, Vocabulary in_vocab, Vocabulary out_vocab) {
		W = w;
		V = v;
		this.in_vocab = in_vocab;
		this.out_vocab = out_vocab;
	}

}

