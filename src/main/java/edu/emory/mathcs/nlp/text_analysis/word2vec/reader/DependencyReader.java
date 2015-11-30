package io;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import io.DependencyReader.DependencyWord;

public class DependencyReader extends Reader<DependencyWord> {
	
	
	public DependencyReader(File file) {
		super(file);
	}
	
	public DependencyReader(List<File> files) {
		super(files);
	}
	
	public DependencyReader(List<File> files, int start_index, int end_index) {
		super(files, start_index, end_index);
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
		DependencyReader[] split = new DependencyReader[count];
		
		int size = (end_index - start_index)/count;
		int start = start_index;
		for(int i=0; i<count; i++){
			split[i] = new DependencyReader(Arrays.asList(files), start, start+size);
			start += size;
		}
		split[split.length-1].end_index = end_index;
		return split;
	}
	
	@Override
	public DependencyReader[] trainingAndTest() {
		DependencyReader[] split = new DependencyReader[2];
		
		int size = (int) ((end_index - start_index)*TRAINING_PORTION);
		split[0] = new DependencyReader(Arrays.asList(files), start_index, start_index+size);
		split[1] = new DependencyReader(Arrays.asList(files), start_index+size, end_index);
		return split;
	}
	
	public static class DependencyWord {
		
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
			return depend+"_"+lemma;
		}
	}
}
