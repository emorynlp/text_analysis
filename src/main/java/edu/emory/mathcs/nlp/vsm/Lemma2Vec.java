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
package edu.emory.mathcs.nlp.vsm;

import java.io.File;
import java.util.List;

import edu.emory.mathcs.nlp.component.template.node.NLPNode;
import edu.emory.mathcs.nlp.vsm.reader.Reader;
import edu.emory.mathcs.nlp.vsm.reader.DEPTreeReader;

/**
 * @author Austin Blodgett
 */
public class Lemma2Vec extends Word2Vec
{

    public Lemma2Vec(String[] args) {
        super(args);
    }

	@Override
	@SuppressWarnings("resource")
    Reader<String> getReader(List<File> files)
    {
        return new DEPTreeReader(files).addFeature(NLPNode::getLemma);
    }

    static public void main(String[] args) { new Lemma2Vec(args); }
}
