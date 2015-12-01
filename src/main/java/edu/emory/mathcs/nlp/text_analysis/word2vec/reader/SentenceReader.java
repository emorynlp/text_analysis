package edu.emory.mathcs.nlp.text_analysis.word2vec.reader;

import java.io.File;
import java.io.IOException;
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

	public SentenceReader(File file) {
		super(file);
	}

	public SentenceReader(List<File> files) {
		super(files);
	}

	public SentenceReader(File file, boolean lowercase, boolean mark_sentence_border) {
		super(file);
		this.lowercase = lowercase;
		this.mark_sentence_border = mark_sentence_border;
	}
	
	public SentenceReader(List<File> files, boolean lowercase, boolean mark_sentence_border) {
		super(files);
		this.lowercase = lowercase;
		this.mark_sentence_border = mark_sentence_border;
	}
	
	public SentenceReader(List<File> files, int start_index, int end_index, boolean lowercase, boolean mark_sentence_border) {
		super(files, start_index, end_index);
		this.lowercase = lowercase;
		this.mark_sentence_border = mark_sentence_border;
	}
	
	private boolean new_line = true;
	private int punctuation = -1;
	private String nextWord() throws IOException {
		String word = "";
		int c;
		
		// This records new lines
		if(new_line){
			new_line = false;
			return "\n";
		}
		
		// This catches punctuation at the ends of words
		if(punctuation != -1){
			word += (char) punctuation;
			punctuation = -1;
			return word;
		}
		
		// for each character
		while((c = read()) != -1){
			// whitespace
			if (c == '\r') continue;
		    if ((c == ' ') || (c == '\t')){
		    	if (word.length() > 0) break;
		    	else continue;
		    }
		    if(c == '\n'){
		    	if (word.length() > 0){
		    		new_line = true;
		    		break;
		    	}
		    	else return "\n";
		    }
		    // punctuation
		    if ((c == '.') || (c == ',') || (c == '?') || (c == '!') || (c == '"') || (c == ':') || (c == ';')) {
		    	if (word.length() > 0) {
		    		// if not abbreviation and not decimal number
		    		if( !(word.charAt(word.length()-1) >= 'A' && word.charAt(word.length()-1) <= 'Z') 
		    		 && !(word.charAt(word.length()-1) >= '0' && word.charAt(word.length()-1) <= '9')){
		    			punctuation = c;
		    			return word;
		    		}
		    	}
		    	else  return "" + (char) c;
		    }
		    // all other characters
		    word += (char) c;
		    if (word.length() == MAX_STRING)
		    	return word;   // Truncate too long words
		}
		
		if(word.isEmpty())
			return null;
		return word;
	}
	
	private String[] sentence = new String[MAX_SENTENCE_LENGTH];
	public String[] next() throws IOException{
		/* This function reads one sentence (assuming one sentence per line) 
		 * and returns it as an array. */
		int sentence_length = 0;
		
		// add "<s>" (start of sentence)
		if(mark_sentence_border)
			sentence[sentence_length++] = "<s>";
		
		String word;
		while ((word = nextWord()) != null){
			// beginning of new sentence
			if(word.equals("\n")){
				if(sentence_length > 1) break;
				else continue;
			}	
			
			if(lowercase) sentence[sentence_length++] = word.toLowerCase();
			else 		  sentence[sentence_length++] = word;
			
			if (sentence_length == MAX_SENTENCE_LENGTH)
				break;
		}
		
		// end of file
		if (sentence_length <= 1)
			return null;
		
		// add "</s>" (end of sentence)
		if(mark_sentence_border)
			sentence[sentence_length++] = "</s>";
				
		// to save space, save sentence in smaller array
		String[] small_sentence = new String[sentence_length];
		for (int i=0; i<sentence_length; i++)
			small_sentence[i] = sentence[i];
		
		return small_sentence;
	}
	
	@Override
	public SentenceReader[] split(int count){
		SentenceReader[] split = new SentenceReader[count];
		
		int size = (end_index - start_index)/count;
		int start = start_index;
		for(int i=0; i<count; i++){
			split[i] = new SentenceReader(Arrays.asList(files), start, start+size, lowercase, mark_sentence_border);
			start += size;
		}
		split[split.length-1].end_index = end_index;
		return split;
	}

	@Override
	public SentenceReader[] trainingAndTest() {
		SentenceReader[] split = new SentenceReader[2];
		
		int size = (int) ((end_index - start_index)*TRAINING_PORTION);
		split[0] = new SentenceReader(Arrays.asList(files), start_index, start_index+size, lowercase, mark_sentence_border);
		split[1] = new SentenceReader(Arrays.asList(files), start_index+size, end_index, lowercase, mark_sentence_border);
		return split;
	}
}
