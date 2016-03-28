package edu.emory.mathcs.nlp.vsm.reader;

import java.io.Closeable;
import java.io.IOException;
import java.util.List;
import java.util.function.Function;
import java.util.function.Predicate;

/**
 * @author Austin Blodgett
 */
public interface Reader<T> extends Closeable
{
    void open() throws IOException;

    void close() throws IOException;

    List<T> next() throws IOException;

    long length();

    /**
     * Get the percentage of this reader that has already been read.
     * @return - float between 0% and 100%
     */
    float progress();

    /**
     * Reset this reader to it's initial state. Using this function
     * you can make multiple passes though the same reader.
     * @throws IOException
     */
    void restart() throws IOException;

    Reader<T> addFilter(Predicate<T> filter);

    <S> Reader<S> addFeature(Function<T, S> feature);

    <S> Reader<S> addMap(Function<List<T>, List<S>> map);

    /**
     * This function splits this reader evenly into a list of readers
     * that can then be parallelized.
     * @param count - the number of readers to be returned
     * @return - list of readers
     */
    List<Reader<T>> splitParallel(int count);

    List<Reader<T>> splitTrainAndTest(float training_portion);

    List<Reader<T>> splitByFile();

    int read() throws IOException;
}
