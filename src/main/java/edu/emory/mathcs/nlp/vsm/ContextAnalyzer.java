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
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
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
import edu.emory.mathcs.nlp.vsm.reader.Reader;
import edu.emory.mathcs.nlp.vsm.reader.DEPTreeReader;
import edu.emory.mathcs.nlp.vsm.util.Vocabulary;

/**
 * This is an extension of classical word2vec to include features of dependency syntax.
 *
 * @author Austin Blodgett
 */
public class ContextAnalyzer extends Word2Vec
{
    private static final long serialVersionUID = -5597377581114506257L;
    Map<String, Map<String, Integer>> sums;
    Map<String, Map<String, Integer>> counts;
	String[] strucs = {"dep", "deph", "dep2", "dep2h", "srlarguments", "closestSiblings", "allSibilings"};
	String[] pos = {"adjective", "adverb", "allPos", "noun", "verb"};

	
	
    int window;
    public ContextAnalyzer(String[] args) {
        super(args);
        window = max_skip_window;
    }


    String getWordLabel(NLPNode word)
    {
        return word.getLemma();
    }

    @Override
    public void train(List<String> filenames) throws Exception
    {

    	//stuff to save
    
    	int allContextSum;
    	int contextsRun;
    	
    	sums = new HashMap<String, Map<String, Integer>>();
    	sums.put("adjective", new HashMap<String, Integer>());
    	sums.put("adverb", new HashMap<String, Integer>());
    	sums.put("allPos", new HashMap<String, Integer>());
    	sums.put("noun", new HashMap<String, Integer>());
    	sums.put("previousVerb", new HashMap<String, Integer>());
    	sums.put("verb", new HashMap<String, Integer>());
    	for (Map.Entry<String, Map<String, Integer>> entry : sums.entrySet()) {
    	    Map<String, Integer> map = entry.getValue();
    	    for(String str: strucs) {
        	    map.put(str, 0);
    	    }
    	}    	
    	
    	counts = new HashMap<String, Map<String, Integer>>();
    	counts.put("adjective", new HashMap<String, Integer>());
    	counts.put("adverb", new HashMap<String, Integer>());
    	counts.put("allPos", new HashMap<String, Integer>());
    	counts.put("noun", new HashMap<String, Integer>());
    	counts.put("previousVerb", new HashMap<String, Integer>());
    	counts.put("verb", new HashMap<String, Integer>());
    	for (Map.Entry<String, Map<String, Integer>> entry : counts.entrySet()) {
    	    Map<String, Integer> map = entry.getValue();
    	    for(String str: strucs) {
        	    map.put(str, 0);
    	    }
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


        BinUtils.LOG.info("Initializing optimizer.\n");
        //optimizer = isNegativeSampling() ? new NegativeSampling(in_vocab, sigmoid, vector_size, negative_size) : new HierarchicalSoftmax(in_vocab, sigmoid, vector_size);

        BinUtils.LOG.info("Training vectors:");
        word_count_global = 0;
        alpha_global      = alpha_init;
        subsample_size    = subsample_threshold * word_count_train;
        ExecutorService executor = Executors.newFixedThreadPool(1);

        start_time = System.currentTimeMillis();

        int id = 0;
        for (Reader<NLPNode> r: train_readers)
        {
            r.open();
            executor.execute(new SynTrainTask(r,id));
            id++;
        }


        executor.shutdown();

        try { executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS); }
        catch (InterruptedException e) {e.printStackTrace();}

        for (Reader<NLPNode> r: train_readers)
            r.close();

        BinUtils.LOG.info("Writing output of things.\n");
        
        BufferedWriter bw = new BufferedWriter(new FileWriter(new File(output_file + ".txt")));
        
        
        for (String str: pos) {
        	bw.write("POS: " + str + "\n");
        	Map<String, Integer> cnt = counts.get(str);
        	Map<String, Integer> sms = sums.get(str);
        	for (Map.Entry<String, Integer> entry : cnt.entrySet()) {
        		String struct = entry.getKey();
            	bw.write("\tStruct: " + struct + "  ");
            	double avg =  (double) sms.get(struct) / (double) entry.getValue();
            	bw.write(avg + "\n");
        	}
        }
        bw.flush();
        bw.close();

        
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
            //float[] neu1  = cbow ? new float[vector_size] : null;
            //float[] neu1e = new float[vector_size];
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
                    break;
                }

                sargs = getSemanticArgumentMap(words);

                for (index=0; index<words.size(); index++)
                {

                	//context here
                	
                	measureContext(words, index, sargs);
                    //skipGram  (words, index, rand, neu1e, sargs);
                }


            }
        }
    }
    
    
    void measureContext(List<NLPNode> words, int index, Map<NLPNode,Set<NLPNode>> sargs){
        int k, l1;
        NLPNode word = words.get(index);
        int word_index = out_vocab.indexOf(getWordLabel(word));
        if (word_index < 0) return;        
        analyzePOS(word, sargs);
     
    }
    
    void analyzePOS(NLPNode word, Map<NLPNode, Set<NLPNode>> sargs) {
        String pos = word.getPartOfSpeechTag();
        if(pos.length() > 1) {
        	if(pos.length() > 2) {
        		pos = pos.substring(0, 2);
        	}
            int count = 0;
            switch (pos) {
            	case "VB": //verb
            		countContextPOS("verb", word, sargs);
            		countContextPOS("allPos", word, sargs);
            		break;
            	case "NN": //noun
            		countContextPOS("noun", word, sargs);
            		countContextPOS("allPos", word, sargs);
            		break;
            	case "JJ": //adjective
            		countContextPOS("adjective", word, sargs);
            		countContextPOS("allPos", word, sargs);
            		break;
            	case "RB": //adverb
            		countContextPOS("adverb", word, sargs);
            		countContextPOS("allPos", word, sargs);
            		break;
            }
        }
    }
		
	


	void countContextPOS(String pos, NLPNode word, Map<NLPNode,Set<NLPNode>> sargs){
		Map<String, Integer> mapSum = sums.get(pos);
		int sum;
		int dep1Size = word.getDependentList().size();
		Map<String, Integer> mapCount = counts.get(pos);
		int num;
		
		
        //if(structure.equals("dep")) {
			sum = mapSum.get("dep");
            mapSum.put("dep", sum + dep1Size);
            
			num = mapCount.get("dep");
			mapCount.put("dep", num + 1);
			
        //if(structure.equals("deph")) {
			sum = mapSum.get("deph");
        	if(word.getDependencyHead() != null) {
                mapSum.put("deph", sum + dep1Size + 1);
        	}else{
                mapSum.put("deph", sum + dep1Size);
        	}
        	
			num = mapCount.get("deph");
			mapCount.put("deph", num + 1);
			
        //if(structure.equals("dep2")) {
			sum = mapSum.get("dep2");
            mapSum.put("dep2", sum + dep1Size + word.getGrandDependentList().size());  
            
			num = mapCount.get("dep2");
			mapCount.put("dep2", num + 1);
			
        //if(structure.equals("dep2h")) {
			sum = mapSum.get("dep2h");
			if(word.getDependencyHead() != null) {
				mapSum.put("dep2h", sum + dep1Size + 1 + word.getGrandDependentList().size());
		    }else{
				mapSum.put("dep2h", sum + dep1Size + word.getGrandDependentList().size());
		    }        
			
			num = mapCount.get("dep2h");
			mapCount.put("dep2h", num + 1);
			
        //if(structure.equals("srlarguments")) {
			sum = mapSum.get("srlarguments");
			int count = 0;
	        Set<NLPNode> set = sargs.get(word);
	        if (set != null) {
	            for (NLPNode s: set) count++;
	         }				
			mapSum.put("srlarguments", sum + dep1Size + count);
			
			num = mapCount.get("srlarguments");
			mapCount.put("srlarguments", num + 1);
			
        //if(structure.equals("closestSiblings")){
			sum = mapSum.get("closestSiblings");
			count = 0;
        	if(word.getRightNearestSibling()!= null) count += 1;
        	if(word.getLeftNearestSibling()!= null) count += 1;
			mapSum.put("closestSiblings", sum + dep1Size + count);
			
			num = mapCount.get("closestSiblings");
			mapCount.put("closestSiblings", num + 1);
			
        //if(structure.equals("allSibilings")) {
			sum = mapSum.get("allSibilings");
			if(getAllSiblings(word)!= null) mapSum.put("allSibilings", sum + dep1Size + getAllSiblings(word).size());
			else mapSum.put("allSibilings", sum + dep1Size);        
			
			num = mapCount.get("allSibilings");
			mapCount.put("allSibilings", num + 1);
    }

    void skipGram(List<NLPNode> words, int index, Random rand, float[] neu1e, Map<NLPNode,Set<NLPNode>> sargs) {
        int k, l1;
        NLPNode word = words.get(index);
        int word_index = out_vocab.indexOf(getWordLabel(word));
        if (word_index < 0) return;

        Set<NLPNode> context_words = new HashSet<NLPNode>();
        
        //add other types of context structures
        if(structure.equals("dep")) {
            context_words.addAll(word.getDependentList());
        }
        if(structure.equals("deph")) {
            context_words.addAll(word.getDependentList());
        	if(word.getDependencyHead() != null) context_words.add(word.getDependencyHead());
        }
        if(structure.equals("dep2")) {
            context_words.addAll(word.getDependentList());
        	context_words.addAll(word.getGrandDependentList());
        }
        if(structure.equals("dep2h")) {
            context_words.addAll(word.getDependentList());
        	context_words.add(word.getDependencyHead());
        	context_words.addAll(word.getGrandDependentList());
        }
        if(structure.equals("srlarguments")) {
            context_words.addAll(word.getDependentList());
        	addSRLNodes(word, context_words, sargs);
        }
        if(structure.equals("closestSiblings")){
            context_words.addAll(word.getDependentList());
        	if(word.getRightNearestSibling()!= null) context_words.add(word.getRightNearestSibling());
        	if(word.getLeftNearestSibling()!= null) context_words.add(word.getLeftNearestSibling());
        }
        if(structure.equals("allSibilings")) {
            context_words.addAll(word.getDependentList());
        	context_words.addAll(getAllSiblings(word));
        }
        if(structure.equals("w2vdep")) {
            context_words.addAll(word.getDependentList());
            int i, j;

            for (i=-window,j=index+i; i<=window; i++,j++)
            {
                if (i == 0 || words.size() <= j || j < 0) continue;
                
                l1 = out_vocab.indexOf(getWordLabel(words.get(j))) * vector_size;
                Arrays.fill(neu1e, 0);
                optimizer.learnSkipGram(rand, word_index, W, V, neu1e, alpha_global, l1);

                // hidden -> input
                for (k=0; k<vector_size; k++) W[l1+k] += neu1e[k];
            }        
        }
        
        
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

    void addSRLNodes(NLPNode word, Set<NLPNode> context_words, Map<NLPNode,Set<NLPNode>> sargs) {
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
            out.write(word+"\t");
            for (float f : map.get(word))
                out.write(f+"\t");
            out.write("\n");
        }
        out.close();
    }

    static public void main(String[] args) { new ContextAnalyzer(args); }
}
