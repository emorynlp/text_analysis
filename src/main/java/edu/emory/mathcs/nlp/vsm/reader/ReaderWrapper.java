/**
 * Copyright 2015, Emory University
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package edu.emory.mathcs.nlp.vsm.reader;

import java.io.IOException;
import java.util.List;
import java.util.function.Function;

/**
 * This class extends Reader and is designed for adding functional features to other readers.
 * When adding a map, filter, or feature to some Reader, you actually create an instance of
 * this class which contains the Reader and converts its output by some function. You will never
 * call this class directly and you should never cast a Reader to type Wrapper.
 *
 * @author Austin Blodgett
 */
class ReaderWrapper<S,T> extends Reader<T>
{
    Reader<S> reader;
    Function<List<S>, List<T>> convert;

    protected ReaderWrapper(Reader<S> reader, Function<List<S>, List<T>> convert)
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
        return new ReaderWrapper<>(reader.subReader(start, end), convert);
    }

    @Override
    public float progress() { return reader.progress(); }
}
