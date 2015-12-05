package edu.emory.mathcs.nlp.text_analysis.word2vec.reader;

import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * Created by austin on 12/5/2015.
 */
public class SimpleReader extends Reader<String> {

	/* Designed to mimic the bahavior of wordzvec.c
	 */

    public SimpleReader(File file) {
        super(file);
    }

    public SimpleReader(List<File> files) {
        super(files);
    }

    public SimpleReader(SimpleReader r, long start_index, long end_index) {
        super(r, start_index, end_index);
    }

    public String[] next() throws IOException{
		/* This function reads one sentence (assuming one sentence per line)
		 * and returns it as an array. */
        int sentence_length = 0;

        String line = "";
        while(line.isEmpty())
            line = readLine();

        if (line == null) return null;

        String[] words = line.split("\\s+");
        return words;
    }

    @Override
    public SimpleReader[] split(int count){
        if(!finished) generateFileSizes();
        SimpleReader[] split = new SimpleReader[count];

        long size = (end_index - start_index)/count;
        long start = start_index;
        for(int i=0; i<count; i++){
            split[i] = new SimpleReader(this, start, start+size);
            start += size;
        }
        split[split.length-1].end_index = end_index;
        return split;
    }
}
