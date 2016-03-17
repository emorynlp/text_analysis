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

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import edu.emory.mathcs.nlp.component.template.node.NLPNode;
import edu.emory.mathcs.nlp.tokenization.Tokenizer;

/**
 * TODO
 * @author Jinho D. Choi ({@code jinho.choi@emory.edu}), Austin Blodgett
 */
public class SentenceReader extends AbstractReader<String> {

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
        if (tokenizer == null){
            words = Arrays.stream(spaces.split(line)).collect(Collectors.toList());
        }else {  //words = tokenizer.tokenize(line);
        	List<NLPNode> tempHolder = tokenizer.tokenize(line);
        	words = new ArrayList<String>();
        	for(NLPNode node : tempHolder){
        		words.add(node.getWordForm());
        	}
        }
        return words;
    }

    @Override
    protected SentenceReader subReader(long start, long end)
    {
        return new SentenceReader(this,start,end);
    }

}
