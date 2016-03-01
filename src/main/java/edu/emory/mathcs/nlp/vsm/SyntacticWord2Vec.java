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

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import edu.emory.mathcs.nlp.common.util.BinUtils;
import edu.emory.mathcs.nlp.component.template.node.NLPNode;
import edu.emory.mathcs.nlp.vsm.reader.DEPTreeReader;
import edu.emory.mathcs.nlp.vsm.reader.Reader;
import edu.emory.mathcs.nlp.vsm.util.Vocabulary;

/**
 * This is an extension of classical word2vec to include features of dependency syntax.
 *
 * @author Austin Blodgett
 */
public class SyntacticWord2Vec extends Word2Vec
{
    private static final long serialVersionUID = -5597377581114506257L;
	Map<String, Integer> verbs; 
	Map<String, Integer> nouns; 
	Map<String, Integer> adjs; 
	Map<String, Integer> adverbs; 


    public SyntacticWord2Vec(String[] args) {
        super(args);
    }

    String getWordLabel(NLPNode word)
    {
        return word.getLemma();
    }

    @Override
    public void train(List<String> filenames) throws Exception
    {
    	verbs = new HashMap<String, Integer>();
    	nouns = new HashMap<String, Integer>();
    	adjs = new HashMap<String, Integer>();
    	adverbs = new HashMap<String, Integer>();

        BinUtils.LOG.info("Reading vocabulary:\n");

        // ------- Austin's code -------------------------------------
        in_vocab = new Vocabulary();
        out_vocab = new Vocabulary();

        List<Reader<NLPNode>> readers = new DEPTreeReader(filenames.stream().map(File::new).collect(Collectors.toList()))
                .splitParallel(thread_size);
        List<Reader<NLPNode>> train_readers = evaluate ? readers.subList(0,thread_size-1) : readers;
        Reader<NLPNode>       test_reader   = evaluate ? readers.get(thread_size-1)       : null;

        if (read_vocab_file == null) in_vocab.learnParallel(train_readers.stream()
                                                            .map(r -> r.addFeature(this::getWordLabel))
                                                            .collect(Collectors.toList()), min_count);
        else 						 in_vocab.readVocab(new File(read_vocab_file), min_count);
        out_vocab.learnParallel(train_readers.stream()
                .map(r -> r.addFeature(this::getWordLabel))
                .collect(Collectors.toList()), min_count);
        
        word_count_train = in_vocab.totalCount();
        // -----------------------------------------------------------

        BinUtils.LOG.info(String.format("- types = %d, tokens = %d\n", in_vocab.size(), word_count_train));

        BinUtils.LOG.info("Going through vocab vectors:");
        ExecutorService executor = Executors.newFixedThreadPool(1);
        start_time = System.currentTimeMillis();

        int id = 0;
        for (Reader<NLPNode> r: train_readers)
        {
            executor.execute(new SynTrainTask(r,id));
            id++;
        }

        executor.shutdown();

        try { executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS); }
        catch (InterruptedException e) {e.printStackTrace();}

        BinUtils.LOG.info("Finding top\n");

        
        BinUtils.LOG.info("Finding verbs\n");
        findTop(verbs, 5000, "/home/azureuser/poslists/verbList.txt");
        BinUtils.LOG.info("Finding nouns\n");
		findTop(nouns, 5000, "/home/azureuser/poslists/nounList.txt");
        BinUtils.LOG.info("Finding adjs\n");
        findTop(adjs,5000, "/home/azureuser/poslists/adjectiveList.txt");
        BinUtils.LOG.info("Finding adverbs\n");
        findTop(adverbs, 5000, "/home/azureuser/poslists/adverbList.txt");
        
       BinUtils.LOG.info("Saved\n");
    }

    class SynTrainTask implements Runnable
    {
        protected Reader<NLPNode> reader;
        protected int id;
        protected float last_progress = 0;
        protected long num_sentences = 0;

        /* Tasks are each parameterized by a reader which is dedicated to a section of the corpus
         * (not necesarily one file). The corpus is split to divide it evenly between Tasks without breaking up sentences. */
        public SynTrainTask(Reader<NLPNode> reader, int id)
        {
            this.reader = reader;
            this.id = id;
        }

        @Override
        public void run()
        {
            BinUtils.LOG.info("Running and reading\n");


            int     iter  = 0;
            int     index;
            List<NLPNode> words = null;
            
            try {
                words = reader.next();
                word_count_global += words == null ? 0 : words.size();
                num_sentences++;
            } catch (IOException e) {
                System.err.println("Reader failure: progress "+reader.progress());
                e.printStackTrace();
                System.exit(1);
            }

            while (words != null)
            {

                try {
                    words = reader.next();
                    word_count_global += words == null ? 0 : words.size();
                    num_sentences++;
                } catch (IOException e) {
                    System.err.println("Reader failure: progress "+reader.progress());
                    e.printStackTrace();
                    System.exit(1);
                }

                for (index=0; index<words.size(); index++)
                {

                    NLPNode word = words.get(index);
                    String pos = word.getPartOfSpeechTag();
                    BinUtils.LOG.info(pos);
                    int count = 0;
                    switch (pos) {
                    	case "VB": //verb
                    		putWord(verbs, word.getLemma());
                    		break;
                    	case "NN": //noun
                    		putWord(nouns, word.getLemma());
                    		break;
                    	case "JJ": //adjective
                    		putWord(adjs, word.getLemma());
                    		break;
                    	case "RB": //adverb
                    		putWord(adverbs, word.getLemma());
                    		break;
                    }
                }
            }
        }
    }
    void findTop(Map<String,Integer> map, int k, String filePath) throws IOException{
    	File outFile = new File(filePath);
        BufferedWriter out;
    	if (!outFile.isFile()) outFile.createNewFile();
		out = new BufferedWriter(new FileWriter(outFile));
		
        HashMap<String, Integer> maph = sortByValues(map); 
        for (Map.Entry<String, Integer> entry : maph.entrySet()) {
			out.write(entry.getKey() + " " + entry.getValue());
            if (k == 0) break;
            k--;
        }
       out.write("k: " + k);
       
       out.flush();
       out.close();
    }
    
    @SuppressWarnings({ "unchecked", "rawtypes" })
	private static HashMap<String, Integer> sortByValues(Map<String, Integer> map) { 
    	if(map == null) BinUtils.LOG.info("Map is null");
    	if(map.entrySet() == null) BinUtils.LOG.info("map entry set is null");

        List list = new LinkedList(map.entrySet());
        Collections.sort(list, new Comparator() {
             public int compare(Object o1, Object o2) {
                return ((Comparable) ((Map.Entry) (o1)).getValue())
                   .compareTo(((Map.Entry) (o2)).getValue());
             }
        });

        HashMap<String, Integer> sortedHashMap = new LinkedHashMap<String, Integer>();
        for (Iterator it = list.iterator(); it.hasNext();) {
               Map.Entry<String, Integer> entry = (Entry<String, Integer>) it.next();
               sortedHashMap.put(entry.getKey(), entry.getValue());
        } 
        return sortedHashMap;
   }
    
    void putWord(Map<String, Integer> map, String lemma) {
    	int count = 0;
		if(map.containsKey(lemma)){
			count = map.get(lemma);
		}
		count++;
		map.put(lemma, count);
    }
    
    static public void main(String[] args) { new SyntacticWord2Vec(args); }
}
