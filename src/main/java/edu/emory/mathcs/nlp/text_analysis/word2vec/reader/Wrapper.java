package edu.emory.mathcs.nlp.text_analysis.word2vec.reader;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;

/**
 * This class extends Reader and is designed for adding functional features to other readers.
 * When adding a map, filter, or feature to some Reader, you actually create an instance of
 * this class which contains the Reader and converts its output by some function. You will never
 * call this class directly and you should never cast a Reader to type Wrapper.
 *
 * @author Austin Blodgett
 */
class Wrapper<S,T> extends Reader<T>
{
    Reader<S> reader;
    Function<List<S>, List<T>> convert;

    protected Wrapper(Reader<S> reader, Function<List<S>, List<T>> convert)
    {
        super(reader);

        this.reader = reader;
        this.convert = convert;
    }

    @Override
    public List<T> next() throws IOException
    {
        List<S> next = reader.next();
        return next==null ? null : convert.apply(next);
    }

    @Override
    protected Reader<T> subReader(long start, long end)
    {
        return new Wrapper<>(reader.subReader(start, end), convert);
    }

    @Override
    public float progress() { return reader.progress(); }
}
