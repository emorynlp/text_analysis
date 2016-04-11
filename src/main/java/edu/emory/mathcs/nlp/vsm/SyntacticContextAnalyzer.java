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
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import edu.emory.mathcs.nlp.common.util.BinUtils;
import edu.emory.mathcs.nlp.component.dep.DEPArc;
import edu.emory.mathcs.nlp.component.template.node.NLPNode;
import edu.emory.mathcs.nlp.vsm.reader.Reader;
import edu.emory.mathcs.nlp.vsm.reader.DEPTreeReader;
import edu.emory.mathcs.nlp.vsm.util.Vocabulary;

/**
 * This is an extension of classical word2vec to include features of dependency syntax.
 *
 * @author Austin Blodgett
 */
public class SyntacticContextAnalyzer extends Word2Vec
{
    private static final long serialVersionUID = -5597377581114506257L;
    volatile Map<String, Long> extCount;
    volatile Map<String, Long> counts;
    int window = 5;
    public String[] structs = {"sib1", "sib2", "sib1dep1", "sib1dep2", "sib2dep1", "sib2dep2", "dep1", "dep1h", "dep2", "dep2h", "srl1", "allSiblings"};
    public String[] headed = {"dep1h", "dep2h"};
    public String[] sib1s = {"sib1", "sib1dep1", "sib1dep2", "sib2", "sib2dep1", "sib2dep2"};
    public String[] sib2s = {"sib2", "sib2dep1", "sib2dep2"};
    public String[] dep1s = {"dep1", "dep1h", "dep2", "dep2h", "sib1dep1", "sib2dep1", "sib1dep2", "sib2dep2"};
    public String[] dep2s = {"dep2", "dep2h", "sib1dep2", "sib2dep2"};
    public SyntacticContextAnalyzer(String[] args) {
        super(args);
        window = max_skip_window;
    }


    String getWordLabel(NLPNode word)
    {
        return word.getLemma();
    }

    @SuppressWarnings("resource")
    @Override
    public void train(List<String> filenames) throws Exception
    {

        String[] structs = {"sib1", "sib2", "sib1dep1", "sib1dep2", "sib2dep1", "sib2dep2", "dep1", "dep1h", "dep2", "dep2h", "srl1", "allSiblings"};
        extCount = new HashMap<String, Long>();
        counts = new HashMap<String, Long>();
        for(String struct : structs){
            extCount.put(struct, (long)0);
            counts.put(struct, (long)0);
        }

        //run
        List<Reader<NLPNode>> train_readers;
        List<Reader<NLPNode>> readers;

            BinUtils.LOG.info("Reading vocabulary:\n");

            in_vocab = (out_vocab = new Vocabulary());

            readers = new DEPTreeReader(filenames.stream().map(File::new).collect(Collectors.toList()))
                    .splitParallel(thread_size);
            train_readers = evaluate ? readers.subList(0,thread_size-1) : readers;

            if (read_vocab_file == null) in_vocab.learnParallel(train_readers.stream()
                                                                .map(r -> r.addFeature(this::getWordLabel))
                                                                .collect(Collectors.toList()), min_count);
            else                          in_vocab.readVocab(new File(read_vocab_file), min_count);
            out_vocab.learnParallel(train_readers.stream()
                    .map(r -> r.addFeature(this::getWordLabel))
                    .collect(Collectors.toList()), min_count);
            word_count_train = in_vocab.totalCount();
            // -----------------------------------------------------------

            BinUtils.LOG.info(String.format("- types = %d, tokens = %d\n", in_vocab.size(), word_count_train));

        BinUtils.LOG.info("Training vectors:");
        word_count_global = 0;
        ExecutorService executor = Executors.newFixedThreadPool(thread_size);

        start_time = System.currentTimeMillis();
        BinUtils.LOG.info("time" + start_time   + "\n");


        int id = 0;
        for (Reader<NLPNode> r: train_readers)
        {
            r.open();
            executor.execute(new ContextAnalyzeTask(r,id));
            id++;
        }


        executor.shutdown();

        try { executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS); }
        catch (InterruptedException e) {e.printStackTrace();}

        for (Reader<NLPNode> r: train_readers)
            r.close();

        BinUtils.LOG.info("Writing output of things.\n");

        BufferedWriter bw = new BufferedWriter(new FileWriter(new File(output_file + ".txt")));


        for (int i = 0; i < structs.length; i++) {
            String str = structs[i];
            bw.write("Struct: " + str + " " + counts.get(str) + " " + extCount.get(str) + "\n");
        }
        bw.flush();
        bw.close();


    }

    class ContextAnalyzeTask implements Runnable
    {
        protected Reader<NLPNode> reader;
        protected int id;
        protected float last_progress = 0;
        protected long num_sentences = 0;

        /* Tasks are each parameterized by a reader which is dedicated to a section of the corpus
         * (not necesarily one file). The corpus is split to divide it evenly between Tasks without breaking up sentences. */
        public ContextAnalyzeTask(Reader<NLPNode> reader, int id)
        {
            this.reader = reader;
            this.id = id;
        }

        @Override
        public void run()
        {
            int     index;
            List<NLPNode> words = null;
            Map<NLPNode,Set<NLPNode>> sargs;


            BinUtils.LOG.info("Entering while Loop");

            while (true)
            {
                try {
                    words = reader.next();
                    word_count_global += words == null ? 0 : words.size();
                } catch (IOException e) {
                    System.err.println("Reader failure: progress "+reader.progress());
                    e.printStackTrace();
                    System.exit(1);
                }

                if (words == null)
                {
                    break;
                }

                for (index=0; index<words.size(); index++){
                    //context here
                    measureContext(words, index);
                }
            }
        }
    }

   boolean equalsAny(String s1, String[] sArr)
   {
        for(String s : sArr)
            if(s.equals(s1))
                return true;
        return false;
   }

   long inWindow(List<NLPNode> words, int index, int id, int window)
   {
        if(id > words.get(index).getID() - window && id < words.get(index).getID() + 5)
            return 1;
        else
            return 0;
   }

    void measureContext(List<NLPNode> words, int index){
        NLPNode word = words.get(index);
        int word_index = out_vocab.indexOf(getWordLabel(word));
        if (word_index < 0) return;

        long contextSize;
        long extSize;
        int id;
        String[] structs = {"sib1", "sib2", "sib1dep1", "sib1dep2", "sib2dep1", "sib2dep2", "dep1", "dep1h", "dep2", "dep2h", "srl1", "allSiblings"};
        String[] headed = {"dep1h", "dep2h"};
        String[] sib1s = {"sib1", "sib1dep1", "sib1dep2", "sib2", "sib2dep1", "sib2dep2"};
        String[] sib2s = {"sib2", "sib2dep1", "sib2dep2"};
        String[] dep1s = {"dep1", "dep1h", "dep2", "dep2h", "sib1dep1", "sib2dep1", "sib1dep2", "sib2dep2"};
        String[] dep2s = {"dep2", "dep2h", "sib1dep2", "sib2dep2"};
        for(String struct : structs)
        {
            contextSize = 0;
            extSize     = 0;
           //if head
           if(equalsAny(struct, headed)){
                if(word.getDependencyHead() != null)
                {
                    contextSize++;
                    extSize += inWindow(words, index, word.getDependencyHead().getID(), 5);
                }
           }
           //if sib1
           if(equalsAny(struct, sib1s))
            {
                if(word.getRightNearestSibling() != null)
                {
                    contextSize++;
                    extSize += inWindow(words, index, word.getRightNearestSibling().getID(), 5);
                }
                if(word.getLeftNearestSibling() != null)
                {
                    contextSize++;
                    extSize += inWindow(words, index, word.getLeftNearestSibling().getID(), 5);
                }
            }
           //if sib2
            if(equalsAny(struct, sib2s))
            {
                if(word.getRightNearestSibling() != null)
                    if(word.getRightNearestSibling().getRightNearestSibling() != null)
                    {
                        contextSize++;
                        extSize += inWindow(words, index, word.getRightNearestSibling().getRightNearestSibling().getID(), 5);
                    }
                if(word.getLeftNearestSibling() != null)
                    if(word.getLeftNearestSibling().getLeftNearestSibling() != null)
                    {
                        contextSize++;
                        extSize += inWindow(words, index, word.getLeftNearestSibling().getLeftNearestSibling().getID(), 5);
                    }

            }
            //if dep1
            if(equalsAny(struct, dep1s))
            {
                for(NLPNode node : word.getDependentList())
                {
                    contextSize++;
                    extSize += inWindow(words, index, node.getID(), 5);
                }
            }
            //if dep2
            if(equalsAny(struct, dep2s))
            {
                for(NLPNode node : word.getGrandDependentList())
                {
                    contextSize++;
                    extSize += inWindow(words, index, node.getID(), 5);
                }
            }
            //if all sibs
            if(struct.equals("allSiblings"))
            {
                for(NLPNode node : getAllSiblings(word))
                {
                    contextSize++;
                    extSize += inWindow(words, index, node.getID(), 5);
                }

            }
            //if srl
            if(struct.equals("srl1")){
                for(DEPArc arc : word.getSemanticHeadList())
                {
                    contextSize++;
                    extSize += inWindow(words, index, arc.getNode().getID(), 5);
                }
            }
            synchronized(extCount)
            {
                extCount.put(struct, extCount.get(struct) + extSize);
            }

            synchronized(counts)
            {
                counts.put(struct, counts.get(struct) + contextSize);
            }


        }
    }

    Set<NLPNode> getAllSiblings(NLPNode node){
        Set<NLPNode> siblings = new HashSet<NLPNode>();
        int id = node.getID();
        if(node.getDependencyHead() == null) return siblings;

        for(NLPNode sib : node.getDependencyHead().getDependentList()) {
            if(id != sib.getID())
                siblings.add(sib);
        }
        return siblings;
    }

    static public void main(String[] args) { new SyntacticContextAnalyzer(args); }
}
