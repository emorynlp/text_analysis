package edu.emory.mathcs.nlp.text_analysis.word2vec.reader;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Much of the technical code necesary to implement Reader is included in this
 * abstract class which can be extended by other Readers.
 *
 * This example creates Reader that converts to lowercase,
 * only considers alphabetical tokens, lemmatize words, and returns instances
 * of type Word:
 *  Reader<Word> wordReader = new SentenceReader(files, tokenizer)
 *                                              .lowercase(true)
 *                                              .addFilter(s -> s.matches("[A-z]+"))
 *                                              .addMap(new Lemmatizer())
 *                                              .addFeature(s -> new Word(s,1));
 *
 * @author Austin Blodgett
 */
public abstract class Reader<T> extends InputStream
{
    List<File> files;

    private int file_index = 0;
    private long index = 0;
    private final long start, end;

    private boolean finished = false;

    private RandomAccessFile in;

    /* When a Reader is split into multiple Readers,
     * the new Readers start after a sentence break
     * and end on a sentence break. To do this, at the
     * start and end of a split Reader, the Reader reads
     * ahead until the next sentence break. */
    private Pattern sentence_break = Pattern.compile("\\n");
    private StringBuilder end_of_sentence = new StringBuilder();


    public Reader(List<File> files)
    {
        this.files = new ArrayList<>(files);
        end = files.stream().mapToLong(File::length).sum();
        start = 0;

        try { restart(); } catch (IOException e) { e.printStackTrace(); }
    }

    protected Reader(List<File> files, Pattern sentence_break)
    {
        this.files = new ArrayList<>(files);
        this.sentence_break = sentence_break;
        end = files.stream().mapToLong(File::length).sum();
        start = 0;

        try { restart(); } catch (IOException e) { e.printStackTrace(); }
    }

    protected Reader(Reader<T> r, long start, long end)
    {
        this.files = new ArrayList<>(r.files);
        this.sentence_break = r.sentence_break;
        this.start = start;
        this.end = end;

        try { restart(); } catch (IOException e) { e.printStackTrace(); }
    }

    protected <S> Reader(Reader<S> r)
    {
        this.files = new ArrayList<>(r.files);
        this.sentence_break = r.sentence_break;
        this.start = r.start;
        this.end = r.end;

        try { restart(); } catch (IOException e) { e.printStackTrace(); }
    }


    abstract public List<T> next() throws IOException;

    abstract protected Reader<T> subReader(long start, long end);

    
    public long length() { return end - start; }

    /**
     * Get the percentage of this reader that has already been read.
     * @return - float between 0% and 100%
     */
     public float progress() { return (index > end) ? 100.0f : 100*(float)(index-start)/(end-start); }

     /**
     * Reset this reader to it's initial state. Using this function
     * you can make multiple passes though the same reader.
     * @throws IOException
     */
    
    public void restart() throws IOException
    {
        finished = false;
        file_index = 0;
        index = 0;

        if (files.size() == 0){ finished = true; return; }

        // find start position
        while (index + files.get(file_index).length() < start)
        {
            index += files.get(file_index++).length();
            if (file_index >= files.size()){ finished = true; return; }
        }
        // open in
        openFile(file_index);
        in.seek(start - index);
        index = start;

        // make sure that reader starts at the beginning of a sentence
        if (index > 0)
        {
            if (end_of_sentence.length() > 0) end_of_sentence = new StringBuilder();
            while (!sentence_break.matcher(end_of_sentence).find())
                end_of_sentence.append((char) read());
        }
        end_of_sentence = new StringBuilder();
    }

    public Reader<T> addFilter(Predicate<T> filter)
    {
        return new Wrapper<>(this, l -> {l.removeIf(filter.negate()); return l;});
    }

    public <S> Reader<S> addFeature(Function<T, S> feature)
    {
        return new Wrapper<>(this, l -> l.stream().map(feature).collect(Collectors.toList()));
    }

    public <S> Reader<S> addMap(Function<List<T>, List<S>> map)
    {
        return new Wrapper<>(this, map);
    }

    /**
     * This function splits this reader evenly into a list of readers
     * that can then be parallelized.
     * @param count - the number of readers to be returned
     * @return - list of readers
     */
    
    public List<Reader<T>> splitParallel(int count)
    {
        List<Reader<T>> readers = new ArrayList<>(count);
        long size = (end - start)/count;
        for (int i=0; i<count-1; i++)
            readers.add(subReader(start+i*size, start+(i+1)*size));
        readers.add(subReader(start+(count-1)*size, end));
        return readers;
    }

    /**
     * This function splits this reader into two readers,
     * 90% for training and 10% for testing.
     * @throws IllegalArgumentException - if training_portion is not between 0 and 1
     * @return - list of readers
     */
    
    public List<Reader<T>> splitTrainAndTest(float training_portion)
    {
        if (training_portion <= 0 || training_portion >= 1)
            throw new IllegalArgumentException("Training portion must be strictly between 0 and 1.");

        List<Reader<T>> readers = new ArrayList<>(2);
        long size = (long) (training_portion * (end - start));
        readers.add(subReader(start, start+size));
        readers.add(subReader(start+size, end));
        return readers;
    }

    public int read() throws IOException
    {
        if (finished){
            if (in != null) close();
            return -1;
        }

        int ch = in.read();
        if (ch == -1)
        {
            if (++file_index < files.size()) { openFile(file_index); return read(); }
            else { finished = true; return -1; }
        }
        if (index > end)
        {
            end_of_sentence.append((char) ch);
            if (sentence_break.matcher(end_of_sentence).find())
                finished = true;
        }

        index++;

        return ch;
    }

    public String readLine() throws IOException
    {
        StringBuilder sb = new StringBuilder();

        int c;
        while ((c = read()) != -1)
        {
            if (c == '\n') break;
            sb.append((char) c);
        }

        return (sb.length()>0 || c!=-1) ? sb.toString() : null;
    }

    private void openFile(int file_index) throws IOException
    {
        if (in != null)
            in.close();
        in = null;
        in = new RandomAccessFile(files.get(file_index), "r");
    }

    public void close() throws IOException
    {
        if (in != null)
            in.close();
        in = null;
    }
}
