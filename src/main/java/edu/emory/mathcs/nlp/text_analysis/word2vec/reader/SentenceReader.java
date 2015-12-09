package edu.emory.mathcs.nlp.text_analysis.word2vec.reader;

import edu.emory.mathcs.nlp.tokenization.Tokenizer;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class SentenceReader extends Reader<String> {
	
	/* An instance of SentenceReader is constructed from a FileInputStream and
	 * on next() returns the next word in the input stream one at a time. 
	 * Tokenization includes basic separation of punctuation and keeps abbreviations
	 * together. Sentences also include "<s>" and "</s>" to mark start and end of 
	 * sentence respectively. Sentences are assumed to be separated by "\n" but multiple
	 * new lines will be ignored.
	 * 
	 */
	
	private static final int MAX_STRING = 100;
	private static final int MAX_SENTENCE_LENGTH = 1000;
	
	boolean lowercase = false;
	boolean mark_sentence_border = false;

	Tokenizer tokenizer;

	public SentenceReader(File file, Tokenizer tokenizer) {
		super(file);
		this.tokenizer = tokenizer;
	}

	public SentenceReader(List<File> files, Tokenizer tokenizer) {
		super(files);
		this.tokenizer = tokenizer;
	}

	public SentenceReader(File file, Tokenizer tokenizer, boolean lowercase, boolean mark_sentence_border) {
		super(file);
		this.tokenizer = tokenizer;
		this.lowercase = lowercase;
		this.mark_sentence_border = mark_sentence_border;
	}
	
	public SentenceReader(List<File> files, Tokenizer tokenizer, boolean lowercase, boolean mark_sentence_border) {
		super(files);
		this.tokenizer = tokenizer;
		this.lowercase = lowercase;
		this.mark_sentence_border = mark_sentence_border;
	}
	
	public SentenceReader(SentenceReader r, long start_index, long end_index) {
		super(r, start_index, end_index);
		this.tokenizer = r.tokenizer;
		this.lowercase = r.lowercase;
		this.mark_sentence_border = r.mark_sentence_border;
	}

	public String[] next() throws IOException{
		/* This function reads one sentence (assuming one sentence per line) 
		 * and returns it as an array. */

		String line = readLine();

		if (line == null) return null;
		if (line.isEmpty()) return next();

		List<String> words;
		if (tokenizer == null) {
			words = new ArrayList();
			for(String word : line.split("\\s+"))
			words.add(word);
		}
		else
			words = tokenizer.tokenize(line);


		String[] sentence = new String[mark_sentence_border ? words.size()+2 : words.size()];

		if(mark_sentence_border) {
			sentence[0] = "<s>";
			sentence[1] = "</s>";
			for (int i=0; i<words.size(); i++)
				sentence[i+1] = lowercase? words.get(i).toLowerCase() : words.get(i);
		}
		else {
			for (int i=0; i<words.size(); i++)
				sentence[i] = lowercase? words.get(i).toLowerCase() : words.get(i);
		}

		return sentence;
	}
	
	@Override
	public SentenceReader[] split(int count){
		if(!finished) generateFileSizes();
		SentenceReader[] split = new SentenceReader[count];
		
		long size = (end_index - start_index)/count;
		long start = start_index;
		for(int i=0; i<count; i++){
			split[i] = new SentenceReader(this, start, start+size);
			start += size;
		}
		split[split.length-1].end_index = end_index;
		return split;
	}
}
