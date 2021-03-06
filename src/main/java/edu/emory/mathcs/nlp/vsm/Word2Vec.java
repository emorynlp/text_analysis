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

import edu.emory.mathcs.nlp.common.random.XORShiftRandom;
import edu.emory.mathcs.nlp.common.util.BinUtils;
import edu.emory.mathcs.nlp.common.util.FileUtils;
import edu.emory.mathcs.nlp.common.util.MathUtils;
import edu.emory.mathcs.nlp.common.util.Sigmoid;
import edu.emory.mathcs.nlp.vsm.evaluate.TopNQueue;
import edu.emory.mathcs.nlp.vsm.optimizer.HierarchicalSoftmax;
import edu.emory.mathcs.nlp.vsm.optimizer.NegativeSampling;
import edu.emory.mathcs.nlp.vsm.optimizer.Optimizer;
import edu.emory.mathcs.nlp.vsm.reader.Reader;
import edu.emory.mathcs.nlp.vsm.reader.SentenceReader;
import edu.emory.mathcs.nlp.vsm.util.Vocabulary;
import org.kohsuke.args4j.Option;

import java.io.*;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

/**
 * @author Austin Blodgett
 */
public class Word2Vec implements Serializable
{
    private static final long serialVersionUID = 7372230197727987245L;
    /* Options for child classes */
    @Option(name="-output-features", usage="output file to save words associated with each hidden layer component.", required=false, metaVar="<filename>")
	String feature_file = null;
    @Option(name="-structure", usage="If set, use the context structure specificed.", required=false, metaVar="<string>")
	String structure = null;
    @Option(name="-structureIsList", usage="If set, treat the structure variable as a list of structures", required=false, metaVar="<boolean>")
    boolean structureIsList = false;
    /* End child options */

    /* Files */
    @Option(name="-train", usage="path to the context file or the directory containing the context files.", required=true, metaVar="<String>")
    String train_path = null;
    @Option(name="-output", usage="output files.", required=true, metaVar="<filename>")
    String output_file = null;
    @Option(name="-load-model", usage="If set, a preexisting model and vocab are loaded from the path specified.", required = false, metaVar="<filename>")
    String model_file = null;
    @Option(name="-ext", usage="extension of the training files (default: \"*\").", required=false, metaVar="<string>")
    String train_ext = "*";
    @Option(name="-write-vocab", usage="file to save serialized vocabulary.", required=false, metaVar="<filename>")
  	String write_vocab_file = null;
  	@Option(name="-read-vocab", usage="file with serialized vocabulary to read.", required=false, metaVar="<filename>")
  	String read_vocab_file = null;
    @Option(name="-isfilelist", usage="If set, treat train file as list of files using given variable as split.", required=false, metaVar="<boolean>")
    String isFileList = null;
    /* End Files */

    /* Hyperparameters */
    @Option(name="-size", usage="size of word vectors (default: 100).", required=false, metaVar="<int>")
    public int vector_size = 100;
    @Option(name="-window", usage="max-window of contextual words (default: 5).", required=false, metaVar="<int>")
    int max_skip_window = 5;
    @Option(name="-sample", usage="threshold for occurrence of words (default: 1e-3). Those that appear with higher frequency in the training data will be randomly down-sampled.", required=false, metaVar="<float>")
    float subsample_threshold = 0.001f;
    @Option(name="-negative", usage="number of negative examples (default: 5; common values are 3 - 10). If negative = 0, use Hierarchical Softmax instead of Negative Sampling.", required=false, metaVar="<int>")
    int negative_size = 5;
    @Option(name="-w", usage="number of training iterations (default: 5).", required=false, metaVar="<int>")
    int train_iteration = 5;
    @Option(name="-min-count", usage="min-count of words (default: 5). This will discard words that appear less than <int> times.", required=false, metaVar="<int>")
    int min_count = 5;
    @Option(name="-alpha", usage="initial learning rate (default: 0.025 for skip-gram; use 0.05 for CBOW).", required=false, metaVar="<float>")
    float alpha_init = 0.025f;
    /* End Hyperparameters */

    /* Training Options */
    @Option(name="-cbow", usage="If set, use the continuous bag-of-words model instead of the skip-gram model.", required=false, metaVar="<boolean>")
    boolean cbow = false;
    @Option(name="-normalize", usage="If set, normalize each vector.", required=false, metaVar="<boolean>")
    boolean normalize = false;
    @Option(name="-save-iter", usage="If set, save the model at each iteration.", required=false, metaVar="<boolean>")
    boolean saveIter = false;
    /* End Training Options */

    /* Debugging */
    @Option(name="-evaluate", usage="If set, reserve portion of training corpus for evaluating.", required=false, metaVar="<boolean>")
    boolean evaluate = false;
    @Option(name="-debug", usage="If set, output more to command line.", required=false, metaVar="<boolean>")
    boolean debug = false;
    /* End Debugging */

    @Option(name="-threads", usage="number of threads (default: 12).", required=false, metaVar="<int>")
    int thread_size = 12;


    final float ALPHA_MIN_RATE  = 0.0001f;

    /* Note that in regular word2vec, the input and output layers
     * are the same. In cases where we want to allow asymmetry between
     * these layers (like in syntactic word2vec), we have to distinguish
     * between input and output vocabularies.
     */
    public Vocabulary in_vocab;
    public Vocabulary out_vocab;

    Sigmoid sigmoid;
    long word_count_train;
    float subsample_size;
    Optimizer optimizer;

    volatile long word_count_global;    // word count dynamically updated by all threads
    volatile float alpha_global;        // learning rate dynamically updated by all threads
    volatile public float[] W;            // weights between the input and the hidden layers
    volatile public float[] V;            // weights between the hidden and the output layers

    long start_time;

    public Word2Vec(String[] args)
    {
        BinUtils.initArgs(args, this);
        sigmoid = new Sigmoid();

        try
        {
            List<String> filenames = new ArrayList<String>();
            if(isFileList != null)
            {
                System.out.println("Splitting on " + isFileList + " . Result: " + Arrays.toString(train_path.split(isFileList)));
                for(String single_path : train_path.split(isFileList))
                {
                    System.out.println("Adding files from " + single_path);
                    filenames.addAll(FileUtils.getFileList(single_path, train_ext, false));
                }
            }
            else
                filenames = FileUtils.getFileList(train_path, train_ext, false);
            train(filenames);
        }
        catch (Exception e) {e.printStackTrace();}
    }

//    =================================== Training ===================================

    Reader<String> getReader(List<File> files)
    {
        return new SentenceReader(files);
    }

    public void train(List<String> filenames) throws Exception
    {
        List<Reader<String>> readers;
        List<Reader<String>> train_readers;
        Reader<String> test_reader;

        if(model_file == null) {
            BinUtils.LOG.info("Reading vocabulary:\n");

            in_vocab = (out_vocab = new Vocabulary());

            readers = getReader(filenames.stream().map(File::new).collect(Collectors.toList()))
                    .splitParallel(thread_size);
            train_readers = evaluate ? readers.subList(0,thread_size-1) : readers;
            test_reader   = evaluate ? readers.get(thread_size-1)          : null;

            if (read_vocab_file == null) in_vocab.learnParallel(train_readers, min_count);
            else                          in_vocab.readVocab(new File(read_vocab_file), min_count);
            // -----------------------------------------------------------

            BinUtils.LOG.info(String.format("- types = %d, tokens = %d\n", in_vocab.size(), word_count_train));

            BinUtils.LOG.info("Initializing neural network.\n");
            initNeuralNetwork();
        } else {
            BinUtils.LOG.info("Loading Model\n");
            ObjectInputStream objin = new ObjectInputStream(new FileInputStream(model_file));
            VSMModel model = (VSMModel) objin.readObject();
            objin.close();
            in_vocab = model.getIn_vocab();
            out_vocab = model.getOut_vocab();
            W = model.getW();
            V = model.getV();
            readers = getReader(filenames.stream().map(File::new).collect(Collectors.toList()))
                    .splitParallel(thread_size);
            train_readers = evaluate ? readers.subList(0,thread_size-1) : readers;
            test_reader   = evaluate ? readers.get(thread_size-1)          : null;
        }

        word_count_train = in_vocab.totalCount();

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
        for (Reader<String> r: train_readers)
        {
            r.open();
            executor.execute(new TrainTask(r,id));
            id++;
        }
        if (evaluate)
        {
            test_reader.open();
            executor.execute(new TestTask(test_reader,id));
        }

        executor.shutdown();

        try { executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS); }
        catch (InterruptedException e) {e.printStackTrace();}

        // -----------------------------------------------------------

        for (Reader<String> r: train_readers) r.close();
        if (evaluate) test_reader.close();


        //Full Model
        saveModel();
        // BinUtils.LOG.info("Saving model.\n");
        // VSMModel model = new VSMModel(W, V, in_vocab, out_vocab);
        // FileOutputStream out = new FileOutputStream(output_file + ".model");
        // ObjectOutputStream object = new ObjectOutputStream(out);
        // object.writeObject(model);
        // object.close();
        // out.close();
    }

    void saveModel()
    {
        try{
            BinUtils.LOG.info("Saving word vectors.\n");
            save(new File(output_file));
            if (write_vocab_file != null)
            {
                File f = new File(write_vocab_file);
                if (!f.isFile()) f.createNewFile();
                in_vocab.writeVocab(f);
            }

            if (feature_file != null) saveFeatures(new File(feature_file));
            BinUtils.LOG.info("Saving model.\n");
            VSMModel model = new VSMModel(W, V, in_vocab, out_vocab);
            FileOutputStream out = new FileOutputStream(output_file + ".model");
            ObjectOutputStream object = new ObjectOutputStream(out);
            object.writeObject(model);
            object.close();
            out.close();
        } catch (Exception e) {e.printStackTrace();}
    }

    void saveModel(int id)
    {
        try{
            BinUtils.LOG.info("Saving word vectors.\n");
            save(new File(output_file + "." + id));
            if (write_vocab_file != null)
            {
                File f = new File(write_vocab_file + "." + id);
                if (!f.isFile()) f.createNewFile();
                in_vocab.writeVocab(f);
            }

            BinUtils.LOG.info("Saving model " + id + ".\n");
            VSMModel model = new VSMModel(W, V, in_vocab, out_vocab);
            FileOutputStream out = new FileOutputStream(output_file + ".model." + id);
            ObjectOutputStream object = new ObjectOutputStream(out);
            object.writeObject(model);
            object.close();
            out.close();
        } catch (Exception e) {e.printStackTrace();}
    }

    class TrainTask implements Runnable
    {
        // ------- Austin ----------------------
        protected Reader<String> reader;
        private int id;
        private float last_progress = 0;
        protected long num_sentences = 0;

        /* Tasks are each parameterized by a reader which is dedicated to a section of the corpus
         * (not necesarily one file). The corpus is split to divide it evenly between Tasks without breaking up sentences. */
        public TrainTask(Reader<String> reader, int id)
        {
            this.reader = reader;
            this.id = id;
        }
        // -------------------------------------

        @Override
        public void run()
        {
            Random  rand  = new XORShiftRandom(reader.hashCode());

            float[] neu1  = cbow ? new float[vector_size] : null;
            float[] neu1e = new float[vector_size];
            int     iter  = 0;
            int     index, window;
            int[]   words;

            while (true)
            {
                words = next(reader, rand, true);
                num_sentences++;

                if (words == null)
                {
                    if (debug) System.out.println("thread "+id+" "+iter+" "+num_sentences);
                    if (++iter == train_iteration) break;
                    adjustLearningRate();
                    // readers have a built in restart button - Austin
                    try { reader.restart(); } catch (IOException e) { e.printStackTrace(); }
                    continue;
                }

                for (index=0; index<words.length; index++)
                {
                    window = 1 + rand.nextInt() % max_skip_window;    // dynamic window size
                    if (cbow) Arrays.fill(neu1, 0);
                    Arrays.fill(neu1e, 0);

                    if (cbow) bagOfWords(words, index, window, rand, neu1e, neu1);
                    else      skipGram  (words, index, window, rand, neu1e);
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

    // -------------- Austin's code ------------------------------------------------------

    class TestTask extends TrainTask
    {
        public TestTask(Reader<String> reader, int id) { super(reader, id); }

        @Override
        public void run()
        {
            Random  rand  = new XORShiftRandom(reader.hashCode());

            float[] neu1  = cbow ? new float[vector_size] : null;
            float[] neu1e = new float[vector_size];
            int     iter  = 0;
            int     index, window;
            int[]   words;

            while (true)
            {
                words = next(reader, rand, true);
                num_sentences++;

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

                for (index=0; index<words.length; index++)
                {
                    window = 1 + rand.nextInt() % max_skip_window;    // dynamic window size
                    if (cbow) Arrays.fill(neu1, 0);
                    Arrays.fill(neu1e, 0);

                    if (cbow) testBagOfWords(words, index, window, rand, neu1e, neu1);
                    else      testSkipGram  (words, index, window, rand, neu1e);
                }
            }
        }
    }

    void outputProgress(long now, float progress)
    {
        float time_seconds = (now - start_time)/1000f;
        int time_left_hours = (int) (((1-progress)/progress)*time_seconds/(60*60));
        int time_left_remainder =  (int) (((1-progress)/progress)*time_seconds/60) % 60;

        Runtime runtime = Runtime.getRuntime();
        long memory_usage = runtime.totalMemory()-runtime.freeMemory();

        System.out.println("Alpha: "+ String.format("%1$,.4f",alpha_global)+" "+
                "Progress: "+ String.format("%1$,.1f", progress * 100) + "% "+
                "Words/thread/sec: " + (int)(word_count_global / thread_size / time_seconds) +" "+
                "Estimated Time Left: " +time_left_hours +":"+String.format("%02d",time_left_remainder) +" "+
                "Memory Usage: " + (int)(memory_usage/(1024*1024)) +"M");
    }

    // -----------------------------------------------------------------------------------

    void adjustLearningRate()
    {
        float rate = Math.max(ALPHA_MIN_RATE, 1 - (float)MathUtils.divide(word_count_global, train_iteration * word_count_train + 1));
        alpha_global = alpha_init * rate;
    }

    void bagOfWords(int[] words, int index, int window, Random rand, float[] neu1e, float[] neu1)
    {
        int i, j, k, l, wc = 0, word = words[index];

        // input -> hidden
        for (i=-window,j=index+i; i<=window; i++,j++)
        {
            if (i == 0 || words.length <= j || j < 0) continue;
            l = words[j] * vector_size;
            for (k=0; k<vector_size; k++) neu1[k] += W[k+l];
            wc++;
        }

        if (wc == 0) return;
        for (k=0; k<vector_size; k++) neu1[k] /= wc;
        optimizer.learnBagOfWords(rand, word, V, neu1, neu1e, alpha_global);

        // hidden -> input
        for (i=-window,j=index+i; i<=window; i++,j++)
        {
            if (i == 0 || words.length <= j || j < 0) continue;
            l = words[j] * vector_size;
            for (k=0; k<vector_size; k++) W[k+l] += neu1e[k];
        }
    }

    void skipGram(int[] words, int index, int window, Random rand, float[] neu1e)
    {
        int i, j, k, l1, word = words[index];

        for (i=-window,j=index+i; i<=window; i++,j++)
        {
            if (i == 0 || words.length <= j || j < 0) continue;
            l1 = words[j] * vector_size;
            Arrays.fill(neu1e, 0);
            optimizer.learnSkipGram(rand, word, W, V, neu1e, alpha_global, l1);

            // hidden -> input
            for (k=0; k<vector_size; k++) W[l1+k] += neu1e[k];
        }
    }

//    =================================== Helper Methods ===================================

    boolean isNegativeSampling()
    {
        return negative_size > 0;
    }

    /** Initializes weights between the input layer to the hidden layer using random numbers between [-0.5, 0.5]. */
    void initNeuralNetwork()
    {
        int size1 = in_vocab.size() * vector_size;
        int size2 = out_vocab.size() * vector_size;
        Random rand = new XORShiftRandom(1);

        W = new float[size1];
        V = new float[size2];

        for (int i=0; i<size1; i++)
            W[i] = (float)((rand.nextDouble() - 0.5) / vector_size);
    }

    /* If input layer and output layer are asymmetrical, param in_layer
     * determines if you want to return input layer indices or output
     * layer indices.
     */
    int[] next(Reader<String> reader, Random rand, boolean in_layer)
    {
        // minor changes in this method - Austin
        Vocabulary vocab = in_layer ? in_vocab : out_vocab;

        List<String> words = null;
        try { words = reader.next(); }
        catch (IOException e)
        {
            System.err.println("Reader failure: progress "+reader.progress());
            e.printStackTrace();
            System.exit(1);
        }

        if (words == null) return null;
        int[] next = new int[words.size()];
        int i, j, index, count = 0;
        double d;

        for (i=0,j=0; i<words.size(); i++)
        {
            index = vocab.indexOf(words.get(i));
            if (index < 0) continue;
            count++;

            // sub-sampling: randomly discards frequent words
            if (subsample_threshold > 0)
            {
                d = (Math.sqrt(MathUtils.divide(vocab.get(index).count, subsample_size)) + 1) * (subsample_size / vocab.get(index).count);
                if (d < rand.nextDouble()) continue;
            }

            next[j++] = index;
        }

        word_count_global += count;
        return (j == 0) ? next(reader, rand, in_layer) : (j == words.size()) ? next : Arrays.copyOf(next, j);
    }


    void testBagOfWords(int[] words, int index, int window, Random rand, float[] neu1e, float[] neu1)
    {
        int i, j, k, l, wc = 0, word = words[index];

        // input -> hidden
        for (i=-window,j=index+i; i<=window; i++,j++)
        {
            if (i == 0 || words.length <= j || j < 0) continue;
            l = words[j] * vector_size;
            for (k=0; k<vector_size; k++) neu1[k] += W[k+l];
            wc++;
        }

        if (wc == 0) return;
        for (k=0; k<vector_size; k++) neu1[k] /= wc;
        optimizer.testBagOfWords(rand, word, V, neu1, neu1e, alpha_global);
    }

    void testSkipGram(int[] words, int index, int window, Random rand, float[] neu1e)
    {
        int i, j, l1, word = words[index];

        for (i=-window,j=index+i; i<=window; i++,j++)
        {
            if (i == 0 || words.length <= j || j < 0) continue;
            l1 = words[j] * vector_size;
            Arrays.fill(neu1e, 0);
            optimizer.testSkipGram(rand, word, W, V, neu1e, alpha_global, l1);
        }
    }



    public Map<String,float[]> toMap(boolean normalize)
    {
        Map<String,float[]> map = new HashMap<>();
        float[] vector;
        String key;
        int i, l;

        for (i=0; i<in_vocab.size(); i++)
        {
            l = i * vector_size;
            key = in_vocab.get(i).form;
            vector = Arrays.copyOfRange(W, l, l+vector_size);
            if (normalize) normalize(vector);
            map.put(key, vector);
        }

        return map;
    }

    public void normalize(float[] vector)
    {
        float z = 0;

        for (int i=0; i<vector.length; i++)
            z += MathUtils.sq(vector[i]);

        z = (float)Math.sqrt(z);

        for (int i=0; i<vector.length; i++)
            vector[i] /= z;
    }

    // ------ Austin's code --------------------------------

    public void save(File save_file) throws IOException
    {
        if (!save_file.isFile()) save_file.createNewFile();

        Map<String,float[]> map = toMap(normalize);
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

    void saveFeatures(File feature_file) throws IOException
    {
        if (!feature_file.isFile())
            feature_file.createNewFile();

        int N = 100;
        float value;
        BufferedWriter out = new BufferedWriter(new FileWriter(feature_file));

        for (int k=0; k<vector_size; k++)
        {
            TopNQueue top = new TopNQueue(N);
            for (int v=0; v<out_vocab.size(); v++)
            {
                value = V[v*vector_size + k] * (float) Math.pow(out_vocab.get(v).count, 0.75);
                top.add(out_vocab.get(v).form, value);
            }
            out.write(k+"\t");
            for (String s : top.list())
                out.write(s+"\t");
            out.write("\n");
        }

        out.close();
    }

    // -------------------------------------------------------

    static public void main(String[] args)
    {
        new Word2Vec(args);
    }
}
