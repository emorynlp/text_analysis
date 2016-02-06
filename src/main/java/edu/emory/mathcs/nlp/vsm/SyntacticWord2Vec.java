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
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
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
    public SyntacticWord2Vec(String[] args) {
        super(args);
    }

    @Override
    @SuppressWarnings("resource")
    Reader<String> getReader(List<File> files)
    {
        return new DEPTreeReader(files).addFeature(NLPNode::getLemma);
    }

    @Override
    public void train(List<String> filenames) throws Exception
    {
        BinUtils.LOG.info("Reading vocabulary:\n");

        // ------- Austin's code -------------------------------------
        in_vocab = (out_vocab = new Vocabulary());
        List<Reader<NLPNode>> readers = new DEPTreeReader(filenames.stream().map(File::new).collect(Collectors.toList()))
                .splitParallel(thread_size);
        List<Reader<NLPNode>> train_readers = evaluate ? readers.subList(0,thread_size-1) : readers;
        Reader<NLPNode>       test_reader   = evaluate ? readers.get(thread_size-1)       : null;

        if (read_vocab_file == null) in_vocab.learnParallel(train_readers.stream().map(r->r.addFeature(NLPNode::getLemma)).collect(Collectors.toList()), min_count);
        else 						 in_vocab.readVocab(new File(read_vocab_file));
        word_count_train = in_vocab.totalCount();
        // -----------------------------------------------------------

        BinUtils.LOG.info(String.format("- types = %d, tokens = %d\n", in_vocab.size(), word_count_train));

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
        if (write_vocab_file != null)
        {
            File f = new File(write_vocab_file);
            if (!f.isFile()) f.createNewFile();
            in_vocab.writeVocab(f);
        }
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

                for (index=0; index<words.size(); index++)
                {
                    if (cbow) Arrays.fill(neu1, 0);
                    Arrays.fill(neu1e, 0);

                    if (cbow) bagOfWords(words, index, rand, neu1e, neu1);
                    else      skipGram  (words, index, rand, neu1e);
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

    void bagOfWords(List<NLPNode> words, int index, Random rand, float[] neu1e, float[] neu1)
    {
        int k, l, wc = 0;
        NLPNode word = words.get(index);
        int word_index = out_vocab.indexOf(word.getLemma());
        if (word_index < 0) return;

        List<NLPNode> context_words = word.getDependentList();

        // input -> hidden
        for (NLPNode context : context_words)
        {
            int context_index = in_vocab.indexOf(context.getLemma());
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
            int context_index = in_vocab.indexOf(context.getLemma());
            l = context_index * vector_size;

            for (k=0; k<vector_size; k++) W[k+l] += neu1e[k];
        }
    }

    void skipGram(List<NLPNode> words, int index, Random rand, float[] neu1e)
    {
        int k, l1;
        NLPNode word = words.get(index);
        int word_index = out_vocab.indexOf(word.getLemma());
        if (word_index < 0) return;

        List<NLPNode> context_words = word.getDependentList();

        for (NLPNode context : context_words)
        {
            int context_index = in_vocab.indexOf(context.getLemma());
            if (context_index < 0) continue;

            l1 = context_index * vector_size;
            Arrays.fill(neu1e, 0);
            optimizer.learnSkipGram(rand, word_index, W, V, neu1e, alpha_global, l1);

            // hidden -> input
            for (k=0; k<vector_size; k++) W[l1+k] += neu1e[k];
        }
    }


    void testBagOfWords(List<NLPNode> words, int index, Random rand, float[] neu1e, float[] neu1)
    {
        int k, l, wc = 0;
        NLPNode word = words.get(index);
        int word_index = out_vocab.indexOf(word.getLemma());
        if (word_index < 0) return;

        List<NLPNode> context_words = word.getDependentList();

        // input -> hidden
        for (NLPNode context : context_words)
        {
            int context_index = in_vocab.indexOf(context.getLemma());
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
        int word_index = out_vocab.indexOf(word.getLemma());
        if (word_index < 0) return;

        List<NLPNode> context_words = word.getDependentList();

        for (NLPNode context : context_words)
        {
            int context_index = in_vocab.indexOf(context.getLemma());
            if (context_index < 0) continue;

            l1 = context_index * vector_size;
            Arrays.fill(neu1e, 0);
            optimizer.testSkipGram(rand, word_index, W, V, neu1e, alpha_global, l1);
        }
    }

    static public void main(String[] args) { new SyntacticWord2Vec(args); }
}
