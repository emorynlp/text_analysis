/**
 * Copyright 2015, Emory University
 *
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
import java.util.ArrayList;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import org.kohsuke.args4j.Option;

import edu.emory.mathcs.nlp.common.random.XORShiftRandom;
import edu.emory.mathcs.nlp.common.util.BinUtils;
import edu.emory.mathcs.nlp.common.util.FileUtils;
import edu.emory.mathcs.nlp.component.template.node.NLPNode;
import edu.emory.mathcs.nlp.vsm.reader.DEPTreeReader;
import edu.emory.mathcs.nlp.vsm.reader.Reader;
import edu.emory.mathcs.nlp.vsm.util.Vocabulary;

/**
 * Gathers data about the context sizes of models trained with SyntacticWord2Vec
 *
 * @author Austin Blodgett, Reid Kilgore
 */
public class ContextSummarizer
{
    private static final long serialVersionUID = -5597377581114506257L;
    @Option(name="-train", usage="path to the context file or the directory containing the context files.", required=true, metaVar="<filepath>")
    String train_path = null;
    @Option(name="-train2", usage="path to a second context file or the directory containing the context files.", required=false, metaVar="<filepath>")
    String train_path2 = null;
    @Option(name="-output", usage="output files.", required=true, metaVar="<filename>")
    String output_file = null;
    @Option(name="-window", usage="window of contextual words (default: 5).", required=false, metaVar="<int>")
    int window = 5;
    @Option(name="-threads", usage="number of threads (default: 12).", required=false, metaVar="<int>")
    int thread_size = 12;
    @Option(name="-min-count", usage="min-count of words (default: 5). This will discard words that appear less than <int> times.", required=false, metaVar="<int>")
    int min_count = 5;
    @Option(name="-context-types", usage="types of context to gather data for, delimited by a comma", required=true, metaVar="<String>")
    String context_types = null;

    volatile Map<String,int[]> stats;
    volatile long word_count_global;
    long word_count_extract;
    long start_time;
    public Vocabulary in_vocab;
    public Vocabulary out_vocab;
    String[] contexts;

    public ContextSummarizer(String[] args) {
        BinUtils.initArgs(args, this);
        contexts = context_types.split(",");

        stats = new HashMap<String,int[]>();
        for ( String context : contexts )
            stats.put(context, new int [] {0, 0, 0});

        List<String> filenames;
        try
        {
            filenames = FileUtils.getFileList(train_path, ".cnlp", false);
            if(train_path2 != null)
                filenames.addAll(FileUtils.getFileList(train_path2, ".cnlp", false));
            extract(filenames);

        }
        catch (Exception e) {e.printStackTrace();}
    }

    String getWordLabel(NLPNode word)
    {
        return word.getLemma();
    }

    public void extract(List<String> filenames) throws Exception
    {
        BinUtils.LOG.info("Reading vocabulary:\n");

        // ------- Austin's code -------------------------------------
        in_vocab = new Vocabulary();
        out_vocab = new Vocabulary();

        List<Reader<NLPNode>> readers = new DEPTreeReader(filenames.stream().map(File::new).collect(Collectors.toList()))
                .splitParallel(thread_size);
        List<Reader<NLPNode>> extract_readers = readers;

        in_vocab.learnParallel(extract_readers.stream()
                            .map(r -> r.addFeature(this::getWordLabel))
                            .collect(Collectors.toList()), min_count);
        word_count_extract = in_vocab.totalCount();
        // -----------------------------------------------------------

        BinUtils.LOG.info(String.format("- types = %d, tokens = %d\n", in_vocab.size(), word_count_extract));
        ExecutorService executor = Executors.newFixedThreadPool(thread_size);
        start_time = System.currentTimeMillis();

        int id = 0;
        for (Reader<NLPNode> r: extract_readers)
        {
            executor.execute(new ExtractionTask(r,id));
            id++;
        }
        // -----------------------------------------------------------

        executor.shutdown();

        try { executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS); }
        catch (InterruptedException e) {e.printStackTrace();}

        save(new File(output_file));
    }


    class ExtractionTask implements Runnable
    {
        protected Reader<NLPNode> reader;
        protected int id;
        private float last_progress = 0;
        protected long num_sentences = 0;
        
        public ExtractionTask(Reader<NLPNode> nlpReader, int i)
        {
            reader = nlpReader;
            id = i;
            /* Inside Outside Count */
        }

        @Override
        public void run()
        {
            int index;
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
                    // readers have a built in restart button - Austin
                    try { reader.restart(); } catch (IOException e) { e.printStackTrace(); }
                    continue;
                }

                for (index=0; index<words.size(); index++)
                {
                    extractContext(words, index);
                }
                for (String ctype : contexts)
                    stats.get(ctype)[2] += words.size();
            }
        }

        public void extractContext(List<NLPNode> words, int index){
            List<NLPNode> context;
            for (String context_type : contexts)
            {
                context  = new ArrayList<NLPNode>();
                switch(context_type){
                    case "dep1h":
                        context.add(words.get(index).getDependencyHead());
                    case "dep1":
                        context.addAll(words.get(index).getDependentList());
                        break;
                    case "dep2h":
                        context.add(words.get(index).getDependencyHead());
                    case "dep2":
                        context.addAll(words.get(index).getDependentList());
                        for(NLPNode dep : context)
                            context.addAll(dep.getDependentList());
                        break;
                    case "sib1dep1h":
                        context.add(words.get(index).getDependencyHead());
                    case "sib1dep1":
                        if(words.get(index).getRightNearestDependent() != null)
                            context.add(words.get(index).getRightNearestDependent());
                        if(words.get(index).getLeftNearestDependent() != null)
                            context.add(words.get(index).getLeftNearestDependent());
                        context.addAll(words.get(index).getDependentList());
                        break;
                }
                for(NLPNode word : context){
                    if(word.getID() < words.get(index).getID() - 5) stats.get(context_type)[1]++;
                    else if (word.getID() > words.get(index).getID() + 5) stats.get(context_type)[1]++;
                    else stats.get(context_type)[0]++;
                }
            }
        }
    }
 
    public void save(File save_file) throws IOException
    {
        if (!save_file.isFile())
            save_file.createNewFile();
        
        BufferedWriter out = new BufferedWriter(new FileWriter(save_file));
        String output = "Inside\tOutside\tCount\n";
        out.write(output);
        for (String ctype : contexts)
            output = ctype + " " + stats.get(ctype)[0] + "\t" + stats.get(ctype)[1] + "\t" + stats.get(ctype)[2] + "\n";
        out.write(output);
        out.close();
    }

    static public void main(String[] args) { new ContextSummarizer(args); }
}
