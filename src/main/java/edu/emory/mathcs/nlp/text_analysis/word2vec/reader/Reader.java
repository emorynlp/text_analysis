package edu.emory.mathcs.nlp.text_analysis.word2vec.reader;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;

public abstract class Reader<T> {
	
	protected final static float TRAINING_PORTION = 0.9f;
	
	protected File[] files;
	protected int[] file_sizes; // cumulative number of lines
	protected int file_index = 0;
	protected int line_index = 0;
	protected int size = 0;
	
	// start_index inclusive, end_index exclusive
	protected int start_index, end_index;
	
	private InputStream in;
	
	public Reader(File file){
		files = new File[] {file};
		size = numLines(file);
		file_sizes = new int[] {size};
		
		try {
			in = new FileInputStream(files[file_index]);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		start_index = 0; 
		end_index = size;
	}
	
	public Reader(List<File> files){
		this.files = files.toArray(new File[files.size()]);
		file_sizes = new int[files.size()];
		
		for(int i=0; i<this.files.length; i++){
			size += numLines(this.files[i]);
			file_sizes[i] = size;
		}
		
		try {
			in = new FileInputStream(this.files[file_index]);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		start_index = 0; 
		end_index = size;
	}
	
	protected Reader(List<File> files, int start_index, int end_index){
		this(files);
		this.start_index = start_index;
		this.end_index = end_index;
		
		if(start_index > 0)
			try {
				gotoLine(start_index);
			} catch (IOException e) {e.printStackTrace();}
	}
	
	abstract public T[] next() throws IOException;
	
	abstract public Reader<T>[] split(int count);
	
	abstract public Reader<T>[] trainingAndTest();	
	
	public void close() throws IOException{
		in.close();
	}
	
	public void startOver() throws IOException{
		gotoLine(start_index);
	}
	
	protected int read() throws IOException {
		if(line_index >= end_index)
			return -1;
		
		int c = in.read();
		
		if(c == -1){
			if(file_index+1 < files.length) {
				line_index++;
				gotoFile(++file_index);				
				return '\n';
			}
		}
		if(isLineBreak(c)) line_index++;
		
		return c;
	}
	
	protected String readLine() throws IOException {
		String line = null;
		
		int c;
		while((c = read()) != -1){
			if(line == null) line = "";
			if(c == '\n') break;
			
			line += (char) c;
		}

		return line;
	}
	
	protected boolean isLineBreak(int c){
		return c == '\n';
	}
	
	private void gotoFile(int index) throws IOException{
		assert(index>=0 && index<files.length);			
		
		file_index = index;
		
		if(file_index > 0)
			line_index = file_sizes[file_index-1];
		
		in.close();
		in = new FileInputStream(this.files[file_index]);
		
	}
	
	private void gotoLine(int index) throws IOException{
		file_index = 0;
		line_index = 0;
		
		while(index >= file_sizes[file_index])
			file_index++;
		
		gotoFile(file_index);
		
		if(index > (file_index > 0 ? file_sizes[file_index-1] : 0)){
			int c = read();
			while(index > line_index && c != -1)
				c = read();
		}
	}	
	
	private int numLines(File input){
		int index = 0;
		int c;
		
		try {
			FileInputStream in = new FileInputStream(input);		
			while((c = in.read()) != -1)
				if(isLineBreak(c)) index++;
			in.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		index++; // treat eof as new line
		return index;
	}
	
}
