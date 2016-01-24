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
import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import edu.emory.mathcs.nlp.component.template.node.NLPNode;
import edu.emory.mathcs.nlp.component.template.util.TSVReader;

/**
 * @author Austin Blodgett
 */
public class DEPTreeReader extends Reader<NLPNode>
{
    private static final Pattern sentence_break = Pattern.compile("\\n\\s*\\n");
    private TSVReader tree_reader;

    public DEPTreeReader(List<File> files)
    {
        super(files, sentence_break);
    }

    protected DEPTreeReader(DEPTreeReader reader, long start, long end)
    {
        super(reader, start, end);
    }

    @Override
    public List<NLPNode> next() throws IOException
    {
        NLPNode[] words = tree_reader.next();

        return words==null ? null : Arrays.stream(words).collect(Collectors.toList());
    }

    @Override
    protected DEPTreeReader subReader(long start, long end)
    {
        return new DEPTreeReader(this, start, end);
    }

    @Override
    public void restart() throws IOException {
        super.restart();
        tree_reader = new TSVReader();
        tree_reader.open(this);

        tree_reader.form   = 1;
        tree_reader.lemma  = 2;
        tree_reader.pos    = 3;
        tree_reader.dhead  = 5;
        tree_reader.deprel = 6;
    }
}
