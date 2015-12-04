package edu.emory.mathcs.nlp.text_analysis.word2vec.reader;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import edu.emory.mathcs.nlp.text_analysis.word2vec.reader.DependencyReader.DependencyWord;

public class DependencyReader extends Reader<DependencyWord> {
	
	public static final int LEMMA_MODE = 0;
	public static final int DEPEND_MODE = 1;
	public static final int POS_MODE = 2;

	private int mode;

	public DependencyReader(File file, int mode) {
		super(file);
		this.mode = mode;
	}

	public DependencyReader(List<File> files, int mode) {
		super(files);
		this.mode = mode;
	}
	
	public DependencyReader(DependencyReader r, long start_index, long end_index, int mode) {
		super(r, start_index, end_index);
		this.mode = mode;
	}
	
	public DependencyWord[] next() throws IOException {
		List<DependencyWord> words = new ArrayList<DependencyWord>();
		
		String line;
		while((line = readLine()) != null){
			if(line.trim().isEmpty()){
				if(words.size() > 0) break;
				else continue;
			}
			
			String[] word = line.split("\t");
			
			if(word.length < 7)
				throw new IOException("Incorrect File Type! Must be Dependecy Tree file.");
			
			words.add(new DependencyWord(word));
		}
		
		if(words.isEmpty()) return null;
		
		return words.toArray(new DependencyWord[words.size()]);
	}
	
	// each sentence break is an empty line
	private boolean last_line_break = false;
	@Override
	protected boolean isLineBreak(int c){
		boolean sentence_break = c == '\n' && last_line_break;
		
		if(last_line_break){
			if(c != ' ' && c != '\t' && c != '\r')
				last_line_break = false;
		}
		else if(c == '\n')
			last_line_break = true;			
		
		return sentence_break;
	}
	
	@Override
	public DependencyReader[] split(int count){
		if(!finished) generateFileSizes();
		DependencyReader[] split = new DependencyReader[count];
		
		long size = (end_index - start_index)/count;
		long start = start_index;
		for(int i=0; i<count; i++){
			split[i] = new DependencyReader(this, start, start+size, mode);
			start += size;
		}
		split[split.length-1].end_index = end_index;
		return split;
	}
	
	public class DependencyWord {
		
		/* Constants for positions in .srl data format */
		final static int INDEX = 0;
		final static int STRING = 1;
		final static int LEMMA = 2;
		final static int POS = 3;
		final static int FEAT = 4;
		final static int DEPEND_INDEX = 5;
		final static int DEPEND = 6;
		final static int SEM_ROLE = 7;
		
		public int index;
		public String string;
		public String lemma;
		public String pos;
		public String depend;
		public int depend_index;
		
		public DependencyWord(String[] word) {
			
			this.index = Integer.parseInt(word[INDEX]);
			this.string = word[STRING];
			this.lemma = word[LEMMA];
			this.pos = word[POS];
			this.depend = word[DEPEND];
			this.depend_index = Integer.parseInt(word[DEPEND_INDEX]);
			
		}
		
		@Override
		public String toString(){
			switch(mode){
				case LEMMA_MODE: return lemma;
				case DEPEND_MODE: return depend+"_"+lemma;
				case POS_MODE: return pos+"_"+lemma;
				default: return depend+"_"+lemma;
			}
		}
	}
}
