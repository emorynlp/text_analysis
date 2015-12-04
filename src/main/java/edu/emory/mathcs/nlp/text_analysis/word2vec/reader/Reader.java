package edu.emory.mathcs.nlp.text_analysis.word2vec.reader;

import edu.emory.mathcs.nlp.text_analysis.word2vec.util.Vocabulary;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.List;

public abstract class Reader<T> {
	
	protected File[] files;
	protected long[] file_sizes; // cumulative number of lines
	protected int file_index = 0;
	protected long line_index = 0;
	protected long size;
	
	// start_index inclusive, end_index exclusive
	protected long start_index, end_index;
	protected boolean finished = false; // some functions like split() require a first pass
	
	private InputStream in;

	public Reader(File file){
		files = new File[] {file};
		file_sizes = new long[files.length];
		Arrays.fill(file_sizes, -1);

		try {
			in = new FileInputStream(this.files[file_index]);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		start_index = 0;
		end_index = Long.MAX_VALUE;
	}

	public Reader(List<File> files){
		this.files = files.toArray(new File[files.size()]);
		file_sizes = new long[this.files.length];
		Arrays.fill(file_sizes, -1);

		try {
			in = new FileInputStream(this.files[file_index]);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		start_index = 0;
		end_index = Long.MAX_VALUE;
	}

	protected Reader(Reader<T> r, long start_index, long end_index){
		this.files = r.files;
		this.file_sizes = r.file_sizes;
		this.size = r.size;

		finished = true;
		this.start_index = start_index;
		this.end_index = end_index;

		try {
			in = new FileInputStream(this.files[file_index]);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		if(start_index > 0)
			try {
				gotoLine(start_index);
			} catch (IOException e) {e.printStackTrace();}
	}

	abstract public T[] next() throws IOException;
	
	abstract public Reader<T>[] split(int count);
	
	public void close() throws IOException{
		in.close();
	}
	
	public void startOver() throws IOException{
		gotoLine(start_index);
	}
	
	protected int read() throws IOException {
		if(line_index >= end_index) {
			finished = true;
			return -1;
		}
		
		int c = in.read();
		
		if(c == -1){
			line_index++;
			size = line_index;
			file_sizes[file_index] = size;

			if(++file_index < files.length) {
				in.close();
				in = new FileInputStream(files[file_index]);
				return '\n';
			}
			else
				end_index = size;
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

	/* Requires one pass through all files.
	 * Only use this function if you have not
	 * already done first pass.
	 */
	protected void generateFileSizes(){

		for(int i=0; i<files.length; i++){
			if(file_sizes[i] == -1) {
				size += numLines(files[i]);
				file_sizes[i] = size;
			}
		}
		finished = true;
		if(end_index == Long.MAX_VALUE) end_index = size;
	}

	private void gotoLine(long index) throws IOException{
		for(int i=0; i<files.length; i++){
			if(file_sizes[i] == -1) {
				size += numLines(files[i]);
				file_sizes[i] = size;
			}
			if(size > index)
				break;
		}

		file_index = 0;
		line_index = 0;

		while(index >= file_sizes[file_index])
			file_index++;

		if(file_index > 0)
			line_index = file_sizes[file_index-1];

		in.close();
		in = new FileInputStream(this.files[file_index]);
		
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
