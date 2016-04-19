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

import edu.emory.mathcs.nlp.common.random.XORShiftRandom;
import edu.emory.mathcs.nlp.common.util.BinUtils;
import edu.emory.mathcs.nlp.component.template.node.NLPNode;
import edu.emory.mathcs.nlp.vsm.optimizer.HierarchicalSoftmax;
import edu.emory.mathcs.nlp.vsm.optimizer.NegativeSampling;
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
    	if(ensemble) {
    		BinUtils.LOG.info("Running Ensemble Method:\n");
    	}
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
        
        if(structure.equals("")) BinUtils.LOG.info("Running context structure dep1");
        else BinUtils.LOG.info("Running text structure " + structure);


        BinUtils.LOG.info("Initializing neural network.\n");
        initNeuralNetwork();

        BinUtils.LOG.info("Initializing optimizer.\n");
        optimizer = isNegativeSampling() ? new NegativeSampling(in_vocab, sigmoid, vector_size, negative_size) : new HierarchicalSoftmax(in_vocab, sigmoid, vector_size);

        BinUtils.LOG.info("Training vectors:");
        word_count_global = 0;
        alpha_global      = alpha_init;
        subsample_size    = subsample_threshold * word_count_train;
        ExecutorService executor = Executors.newFixedThreadPool(thread_size);

        // ------- Austin's code -------------------------------------
        start_time = System.currentTimeMillis();

        int id = 0;
        for (Reader<NLPNode> r: train_readers)
        {
            executor.execute(new SynTrainTask(r,id));
            id++;
        }
        if (evaluate) executor.execute(new SynTestTask(test_reader,id));
        // -----------------------------------------------------------

        executor.shutdown();

        try { executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS); }
        catch (InterruptedException e) {e.printStackTrace();}


        BinUtils.LOG.info("Saving word vectors.\n");

        save(new File(output_file));
        save2(new File(output_file+".2"));
        if (write_vocab_file != null)
        {
            File f = new File(write_vocab_file);
            if (!f.isFile()) f.createNewFile();
            in_vocab.writeVocab(f);
        }
        if (feature_file != null) saveFeatures(new File(feature_file));
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
            Random rand  = new XORShiftRandom(reader.hashCode());
            float[] neu1  = cbow ? new float[vector_size] : null;
            float[] neu1e = new float[vector_size];
            int     iter  = 0;
            int     index;
            List<NLPNode> words = null;
            Map<NLPNode,Set<NLPNode>> sargs;

            while (true)
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

                if (words == null)
                {
                    System.out.println("thread "+id+" "+iter+" "+num_sentences);
                    if (++iter == train_iteration) break;
                    adjustLearningRate();
                    // readers have a built in restart button - Austin
                    try { reader.restart(); } catch (IOException e) { e.printStackTrace(); }
                    continue;
                }
                
                sargs = getSemanticArgumentMap(words);

                for (index=0; index<words.size(); index++)
                {
                    if (cbow) Arrays.fill(neu1, 0);
                    Arrays.fill(neu1e, 0);

                    if (cbow) bagOfWords(words, index, rand, neu1e, neu1, sargs);
                    else      skipGram  (words, index, rand, neu1e, sargs);
                }

                // output progress
                if(id == 0)
                {
                    float progress = (iter + reader.progress()/100)/train_iteration;
                    if(progress-last_progress > 0.025f)
                    {
                        outputProgress(System.currentTimeMillis(), progress);
                        last_progress += 0.1f;
                    }
                }

            }
        }
    }


    class SynTestTask extends SynTrainTask
    {
        public SynTestTask(Reader<NLPNode> reader, int id)
        {
            super(reader,id);
        }

        @Override
        public void run()
        {
            Random rand  = new XORShiftRandom(reader.hashCode());
            float[] neu1  = cbow ? new float[vector_size] : null;
            float[] neu1e = new float[vector_size];
            int     iter  = 0;
            int     index;
            List<NLPNode> words = null;

            while (true)
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

                if (words == null)
                {
                    System.out.println("error "+optimizer.getError()+" "+num_sentences);
                    optimizer.resetError();
                    if (++iter == train_iteration) break;
                    adjustLearningRate();
                    // readers have a built in restart button - Austin
                    try { reader.restart(); } catch (IOException e) { e.printStackTrace(); }
                    continue;
                }

                for (index=0; index<words.size(); index++)
                {
                    if (cbow) Arrays.fill(neu1, 0);
                    Arrays.fill(neu1e, 0);

                    if (cbow) testBagOfWords(words, index, rand, neu1e, neu1);
                    else      testSkipGram  (words, index, rand, neu1e);
                }
            }
        }
    }

    void bagOfWords(List<NLPNode> words, int index, Random rand, float[] neu1e, float[] neu1, Map<NLPNode,Set<NLPNode>> sargs) {
        int k, l, wc = 0;
        NLPNode word = words.get(index);
        int word_index = out_vocab.indexOf(getWordLabel(word));
        if (word_index < 0) return;

        Set<NLPNode> context_words = new HashSet<NLPNode>();
        context_words.addAll(word.getDependentList());

        // input -> hidden
        for (NLPNode context : context_words)
        {
            int context_index = in_vocab.indexOf(getWordLabel(context));
            if (context_index < 0) continue;
            l = context_index * vector_size;
            for (k=0; k<vector_size; k++) neu1[k] += W[k+l];
            wc++;
        }

        if (wc == 0) return;
        for (k=0; k<vector_size; k++) neu1[k] /= wc;
        optimizer.learnBagOfWords(rand, word_index, V, neu1, neu1e, alpha_global);

        // hidden -> input
        for (NLPNode context : context_words)
        {
            int context_index = in_vocab.indexOf(getWordLabel(context));
            l = context_index * vector_size;

            for (k=0; k<vector_size; k++) W[k+l] += neu1e[k];
        }
    }

    void skipGram(List<NLPNode> words, int index, Random rand, float[] neu1e, Map<NLPNode,Set<NLPNode>> sargs) {
        int k, l1;
        NLPNode word = words.get(index);
        int word_index = out_vocab.indexOf(getWordLabel(word));
        if (word_index < 0) return;
        	
        //do ensemble method if set
        if (ensemble){
        	String s = getEnsembleStructure(word);
        	if(s!=null) {
        		structure = s;
        	}
        }
        	
        Set<NLPNode> context_words = new HashSet<NLPNode>();
        context_words.addAll(word.getDependentList());
        
        //add other types of context structures
        if(structure.equals("deph")) {
        	if(word.getDependencyHead() != null) context_words.add(word.getDependencyHead());
        }
        if(structure.equals("dep2")) context_words.addAll(word.getGrandDependentList());
        if(structure.equals("dep2h")) {
        	if(word.getDependencyHead() != null) context_words.add(word.getDependencyHead());
        	context_words.addAll(word.getGrandDependentList());
        }
        if(structure.equals("srlarguments")) addSRLNodes(word, context_words, sargs);
        if(structure.equals("closestSiblings")){
        	if(word.getRightNearestSibling()!= null) context_words.add(word.getRightNearestSibling());
        	if(word.getLeftNearestSibling()!= null) context_words.add(word.getLeftNearestSibling());
        }
        if(structure.equals("allSibilings")) context_words.addAll(getAllSiblings(word));
        
        //train
        for (NLPNode context : context_words)
        {
            int context_index = in_vocab.indexOf(getWordLabel(context));
            if (context_index < 0) continue;

            l1 = context_index * vector_size;
            Arrays.fill(neu1e, 0);
            optimizer.learnSkipGram(rand, word_index, W, V, neu1e, alpha_global, l1);

            // hidden -> input
            for (k=0; k<vector_size; k++) W[l1+k] += neu1e[k];
        }
    }
    
    String getEnsembleStructure(NLPNode word){
        String partOfSpeech = word.getPartOfSpeechTag();
        	
        if(partOfSpeech.length() > 1) {
            	if(partOfSpeech.length() > 2) {
            		partOfSpeech = partOfSpeech.substring(0, 2);
            	}
                switch (partOfSpeech) {
                	case "NN": //noun
                		return "allSibilings";
                	case "JJ": //adjective
                		return "allSibilings";
                	case "RB": //adverb
                		return "deph";
                	case "VB": //verb
                		return "dep1";
                }
        }  
        return null;
    }
    
    
    
    
    void addSRLNodes(NLPNode word, Set<NLPNode> context_words, Map<NLPNode,Set<NLPNode>> sargs) {
        context_words.addAll(word.getDependentList());
        Set<NLPNode> set = sargs.get(word);
        if (set != null)
        {
        	for (NLPNode s: set){
        		context_words.add(s);
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
    
    Map<NLPNode,Set<NLPNode>> getSemanticArgumentMap(List<NLPNode> nodes)
    {
    	Map<NLPNode,Set<NLPNode>> map = new HashMap<>();
    	Set<NLPNode> args;
    	NLPNode node;
    	
    	for (int i=1; i<nodes.size(); i++)
    	{
    		node = nodes.get(i);
    		args = node.getSemanticHeadList().stream().map(a -> a.getNode()).collect(Collectors.toSet());
    		map.put(node, args);
    	}
    	
    	return map;
    }


    void testBagOfWords(List<NLPNode> words, int index, Random rand, float[] neu1e, float[] neu1) {
        int k, l, wc = 0;
        NLPNode word = words.get(index);
        int word_index = out_vocab.indexOf(getWordLabel(word));
        if (word_index < 0) return;

        Set<NLPNode> context_words = new HashSet<NLPNode>();
        context_words.addAll(word.getDependentList());
        
        // input -> hidden
        for (NLPNode context : context_words)
        {
            int context_index = in_vocab.indexOf(getWordLabel(context));
            if (context_index < 0) continue;
            l = context_index * vector_size;
            for (k=0; k<vector_size; k++) neu1[k] += W[k+l];
            wc++;
        }

        if (wc == 0) return;
        for (k=0; k<vector_size; k++) neu1[k] /= wc;
        optimizer.testBagOfWords(rand, word_index, V, neu1, neu1e, alpha_global);
    }

    void testSkipGram(List<NLPNode> words, int index, Random rand, float[] neu1e)
    {
        int l1;
        NLPNode word = words.get(index);
        int word_index = out_vocab.indexOf(getWordLabel(word));
        if (word_index < 0) return;

        Set<NLPNode> context_words = new HashSet<NLPNode>();
        context_words.addAll(word.getDependentList());
        
        for (NLPNode context : context_words)
        {
            int context_index = in_vocab.indexOf(getWordLabel(context));
            if (context_index < 0) continue;

            l1 = context_index * vector_size;
            Arrays.fill(neu1e, 0);
            optimizer.testSkipGram(rand, word_index, W, V, neu1e, alpha_global, l1);
        }
    }

    public Map<String,float[]> toMap2(boolean normalize)
    {
        Map<String,float[]> map = new HashMap<>();
        float[] vector;
        String key;
        int i, l;

        for (i=0; i<out_vocab.size(); i++)
        {
            l = i * vector_size;
            key = out_vocab.get(i).form;
            vector = Arrays.copyOfRange(V, l, l+vector_size);
            if (normalize) normalize(vector);
            map.put(key, vector);
        }

        return map;
    }

    public void save2(File save_file) throws IOException
    {
        if (!save_file.isFile()) save_file.createNewFile();

        Map<String,float[]> map = toMap2(normalize);
        BufferedWriter out = new BufferedWriter(new FileWriter(save_file));

        for (String word : map.keySet())
        {
            out.write(word+ "\t");
            for (float f : map.get(word))
                out.write(f + "\t");
            out.write("\n");
        }
        out.close();
    }

    static public void main(String[] args) { new SyntacticWord2Vec(args); }
}
