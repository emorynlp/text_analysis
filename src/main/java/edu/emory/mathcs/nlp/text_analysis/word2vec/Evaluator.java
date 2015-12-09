package edu.emory.mathcs.nlp.text_analysis.word2vec;

import edu.emory.mathcs.nlp.common.random.XORShiftRandom;
import edu.emory.mathcs.nlp.common.util.BinUtils;
import edu.emory.mathcs.nlp.common.util.FileUtils;
import edu.emory.mathcs.nlp.common.util.IOUtils;
import edu.emory.mathcs.nlp.text_analysis.word2vec.reader.Reader;
import org.kohsuke.args4j.Option;

import java.io.*;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

/**
 * Created by austin on 12/6/2015.
 */
public class Evaluator {

    @Option(name="-evaluate", usage="path to file or directory to evaluate vectors.", required=true, metaVar="<filename>")
    String eval_path = null;
    @Option(name="-word2vec-model", usage="triad word similarity file for (external) vector evaluation.", required=true, metaVar="<filename>")
    String model_path = null;
    @Option(name="-output", usage="path to file to write evaluation data.", required=true, metaVar="<filename>")
    String output_path = null;
    @Option(name="-ext", usage="extension of the training files (default: \"*\").", required=false, metaVar="<string>")
    String train_ext = "*";
    @Option(name="-triad-file", usage="triad word similarity file for (external) vector evaluation.", required=false, metaVar="<filename>")
    String triad_path = default_triad_file;

    Word2Vec word2vec;

    static final String default_triad_file = "src/main/resources/Triads_1202.csv";

    public Evaluator(Word2Vec word2vec, String eval_path, String output_filename)
    {
        this.word2vec = word2vec;
        this.eval_path = eval_path;
        this.output_path = output_filename;

        try
        {
            evaluate(FileUtils.getFileList(eval_path, train_ext, false));
        }
        catch (Exception e) {e.printStackTrace();}
    }

    public Evaluator(String[] args)
    {
        BinUtils.initArgs(args, this);

        try {
            word2vec = readModel(model_path);
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }

        try
        {
            evaluate(FileUtils.getFileList(eval_path, train_ext, false));
        }
        catch (Exception e) {e.printStackTrace();}
    }

    Word2Vec readModel(String filename) throws IOException, ClassNotFoundException {
        return (Word2Vec) IOUtils.createObjectXZBufferedInputStream(filename).readObject();
    }

    public void evaluate(List<String> filenames) throws IOException
    {
        BinUtils.LOG.info("Evaluating");
        List<File> test_files = filenames.stream().map(File::new).collect(Collectors.toList());

        Reader<?> test_reader = word2vec.initReader(test_files);
        startThreads(test_reader);

        float[] triad_eval = triadEvaluation(new File(triad_path));
        float time_seconds = (word2vec.end_time - word2vec.start_time) / 1000f;


        System.out.println("Evaluated Error: " + String.format("%1$,.6f",word2vec.optimizer.getError()));
        System.out.println("Weighted Triad Evaluation: " + triad_eval[0]);
        System.out.println("Unweighted Triad Evaluation: " + triad_eval[1]);
        System.out.println("Words/thread/sec: " + (int)(word2vec.word_count_global / (word2vec.thread_size * time_seconds)));
        System.out.println("Total time: " + String.format("%1$,.2f", time_seconds / 60 / 60.0));
        System.out.println("Max Memory Usage: " + word2vec.max_memory+"\n");

        if (output_path != null) {
            File output_file = new File(output_path);
            if (!output_file.exists())
                output_file.createNewFile();

            FileWriter out = new FileWriter(output_file, true);
            out.write(word2vec.getClass() + " " + (word2vec.cbow ? "CBOW" : "Skip-grams") +", " + ( word2vec.isNegativeSampling() ? "Negative" : "Hierarchical")+"\n");
            if(word2vec.tokenize)
                out.write("Tokenized ");
            if(word2vec.lowercase)
                out.write("Lowercase ");
            if (word2vec.sentence_border)
                out.write("Sentence Border");
            out.write("\nVector Size: " + word2vec.vector_size+"\n");
            out.write("Iterations: " + word2vec.train_iteration+"\n");
            out.write("Evaluated Error: " + word2vec.optimizer.getError() + "\n");
            out.write("Weighted Triad Evaluation: " + triad_eval[0] + "\n");
            out.write("Unweighted Triad Evaluation: " + triad_eval[1] + "\n");
            out.write("Total time: " + String.format("%1$,.2f", time_seconds / 60 / 60.0) + "\n");
            out.write("Words/thread/sec: " + (int)(word2vec.word_count_global / (word2vec.thread_size * time_seconds)) + "\n");
            out.write("Max Memory Usage: " + word2vec.max_memory+"\n\n");

            out.close();
        }
    }

    void startThreads(Reader<?> reader) throws IOException
    {
        Reader<?>[] readers = reader.split(word2vec.thread_size);
        reader.close();
        ExecutorService executor = Executors.newFixedThreadPool(word2vec.thread_size);

        for (int i = 0; i < word2vec.thread_size; i++)
            executor.execute(new TrainTask(readers[i], i));

        executor.shutdown();
        try {
            executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        for (int i = 0; i < word2vec.thread_size; i++)
            readers[i].close();
    }

    class TrainTask implements Runnable
    {
        private Reader<?> reader;
        private final int id;

        public TrainTask(Reader<?> reader, int id)
        {
            this.reader = reader;
            this.id = id;
        }

        @Override
        public void run()
        {
            Random  rand  = new XORShiftRandom(reader.hashCode());
            float[] neu1  = word2vec.cbow ? new float[word2vec.vector_size] : null;
            int     iter  = 0;
            int     index, window;
            int[]   words = null;

            while (true)
            {
                try {
                    words = word2vec.next(reader, rand);
                } catch (IOException e) {
                    e.printStackTrace();
                }

                if (words == null)
                {
                    if (++iter == word2vec.train_iteration) break;
                    try {
                        reader.startOver();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    continue;
                }

                for (index=0; index<words.length; index++)
                {
                    window = 1 + rand.nextInt() % word2vec.max_skip_window;	// dynamic window size
                    if (word2vec.cbow) Arrays.fill(neu1, 0);

                    if (word2vec.cbow) evalBagOfWords(words, index, window, rand, neu1);
                    else      evalSkipGram  (words, index, window, rand);
                }
            }
        }
    }

    void evalBagOfWords(int[] words, int index, int window, Random rand, float[] neu1)
    {
        int i, j, k, l, wc = 0, word = words[index];

        // input -> hidden
        for (i=-window,j=index+i; i<=window; i++,j++)
        {
            if (i == 0 || words.length <= j || j < 0) continue;
            l = words[j] * word2vec.vector_size;
            for (k=0; k<word2vec.vector_size; k++) neu1[k] += word2vec.W[k+l];
            wc++;
        }

        if (wc == 0) return;
        for (k=0; k<word2vec.vector_size; k++) neu1[k] /= wc;

        word2vec.optimizer.testBagOfWords(rand, word, word2vec.V, neu1);
    }

    void evalSkipGram(int[] words, int index, int window, Random rand)
    {
        int i, j, l1, word = words[index];

        for (i=-window,j=index+i; i<=window; i++,j++)
        {
            if (i == 0 || words.length <= j || j < 0) continue;
            l1 = words[j] * word2vec.vector_size;

            word2vec.optimizer.testSkipGram(rand, word, word2vec.W, word2vec.V, l1);
        }
    }

    public float[] triadEvaluation(File triad_file) throws IOException
    {
        float unweighted_eval = 0;
        int unweighted_count = 0;

        float weighted_eval = 0;
        int weighted_count = 0;


        BufferedReader br = new BufferedReader(new FileReader(triad_file));

        String line;
        while((line = br.readLine()) != null){
            String[] triad = line.split(",");
            if(triad.length != 5)
                throw new IOException("Could not read triad file. Incorrect format.");
            if(!(word2vec.vocab.contains(triad[0]) && word2vec.vocab.contains(triad[1]) && word2vec.vocab.contains(triad[2])))
                continue;

            int word_count1 = Integer.parseInt(triad[3]);
            int word_count2 = Integer.parseInt(triad[4]);

            if((word_count1 > word_count2) == (similarity(triad[1],triad[0]) > similarity(triad[2],triad[0]))) {
                for(int i=0; i<Math.abs(word_count1-word_count2); i++)
                    weighted_eval++;
                unweighted_eval++;
            }

            for(int i=0; i<Math.abs(word_count1-word_count2); i++)
                weighted_count++;
            unweighted_count++;
        }
        br.close();

        if(unweighted_count != 0)
            unweighted_eval /= unweighted_count;
        if(weighted_count != 0)
            weighted_eval /= weighted_count;

        return new float[] {weighted_eval, unweighted_eval};
    }

    double similarity(String word1, String word2)
    {
        int l1 = word2vec.vocab.indexOf(word1)*word2vec.vector_size;
        int l2 = word2vec.vocab.indexOf(word2)*word2vec.vector_size;

        double norm1 = 0.0, norm2 = 0.0;

        double dot_product = 0.0;
        for(int c=0; c<word2vec.vector_size; c++){
            dot_product += word2vec.W[l1+c]*word2vec.W[l2+c];
            norm1 += word2vec.W[l1+c]*word2vec.W[l1+c];
            norm2 += word2vec.W[l2+c]*word2vec.W[l2+c];
        }
        norm1 = Math.sqrt(norm1);
        norm2 = Math.sqrt(norm2);

        return dot_product/(norm1*norm2);
    }

    public static void main(String[] args) throws IOException {
        new Evaluator(args);
    }
}
