package edu.emory.mathcs.nlp.text_analysis.word2vec.reader;

import edu.emory.mathcs.nlp.tokenization.Tokenizer;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import static java.util.stream.Collectors.toList;

/**
 * TODO
 * @author Austin Blodgett
 */
public class SentenceReader extends Reader<String> {

    private Tokenizer tokenizer;

    private static Pattern spaces = Pattern.compile("\\s+");

    public SentenceReader(List<File> files)
    {
        super(files);
        this.tokenizer = null;
    }

    public SentenceReader(List<File> files, Tokenizer tokenizer)
    {
        super(files);
        this.tokenizer = tokenizer;
    }

    protected SentenceReader(SentenceReader r, long start, long end)
    {
        super(r, start, end);
        this.tokenizer = r.tokenizer;
    }

    public List<String> next() throws IOException {
		/* This function reads one sentence (assuming one sentence per line)
		 * and returns it as an array. */

        String line = readLine();

        if (line == null) return null;
        if (line.isEmpty()) return next();

        List<String> words;
        if (tokenizer == null)
            words = Arrays.stream(spaces.split(line)).collect(Collectors.toList());
        else
            words = tokenizer.tokenize(line);

        return words;
    }

    @Override
    protected SentenceReader subReader(long start, long end)
    {
        return new SentenceReader(this,start,end);
    }

}
