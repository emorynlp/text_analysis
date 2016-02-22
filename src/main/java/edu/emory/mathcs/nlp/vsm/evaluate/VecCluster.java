package edu.emory.mathcs.nlp.vsm.evaluate;

import edu.emory.mathcs.nlp.common.util.BinUtils;
import org.kohsuke.args4j.Option;

import java.io.*;
import java.util.*;

/**
 * Created by austin on 2/10/2016.
 */
public class VecCluster
{

    @Option(name="-input", usage="file of word vectors to cluster.", required=true, metaVar="<filename>")
    String vector_file = null;
    @Option(name="-output", usage="output file to save clusters.", required=true, metaVar="<filename>")
    String output_file = null;
    @Option(name="-word-file", usage="file with list of words to consider (for faster processing).", required=false, metaVar="<filename>")
    String word_file = null;
    @Option(name="-clusters", usage="number of clusters to find.", required=false, metaVar="<integer>")
    int num_clusters = 10;
    @Option(name="-iter", usage="number of iterations to run.", required=false, metaVar="<integer>")
    int num_iterations = 10;

    int vector_size;
    int num_vectors;

    float[][] vectors;  // num_vectors  x vector_size
    float[][] means;	// num_clusters x vector_size
    double[][] clusters; // num_clusters x num_vectors

    List<String> word_list;
    final float std_dev = 100;

    public VecCluster(String[] args)
    {
        BinUtils.initArgs(args, this);
        if (word_file != null) {
            try { word_list = getWordList(new File(word_file));}
            catch (IOException e) { System.err.println("Could not read word file."); e.printStackTrace(); System.exit(1);}
        }

        try { vectors = getVectors(new File(vector_file), word_list); }
        catch (IOException e) { System.err.println("Could not load Word2Vec vectors."); e.printStackTrace(); System.exit(1); }

        this.num_vectors = vectors.length;
        this.vector_size = vectors[0].length;

        means = new float[num_clusters][vector_size];
        clusters = new double[num_clusters][num_vectors];

        cluster();
        try { write(new File(output_file)); } catch (IOException e) { e.printStackTrace(); }
    }

    private float[][] getVectors(File vector_file, List<String> word_list) throws IOException
    {
        Map<String,float[]> map = new HashMap<>();

        BufferedReader in = new BufferedReader(new FileReader(vector_file));
        String line;
        while((line = in.readLine()) != null)
        {
            String[] split = line.split("\t");
            String word = split[0];
            if (!word_list.contains(word)) continue;

            float[] vector = new float[split.length - 1];
            for (int i=1; i<split.length; i++)
                vector[i-1] = Float.parseFloat(split[i]);
            map.put(word, vector);
        }
        in.close();

        float[][] vectors = new float[word_list.size()][];
        for (int i=0; i<word_list.size(); i++)
            vectors[i] = map.get(word_list.get(i));

        return vectors;
    }

    private List<String> getWordList(File word_file) throws IOException
    {
        List<String> word_list = new ArrayList<>();
        BufferedReader in = new BufferedReader(new FileReader(word_file));

        String word;
        while((word = in.readLine()) != null)
            word_list.add(word);

        in.close();
        return word_list;
    }

    public void cluster(){

        // normalize vectors
        for(int i=0; i<vectors.length; i++)
            normalize(vectors[i]);

		/* Randomly choose K vectors to act as the means for each cluster */
        Set<Integer> random_vectors = new HashSet<>(num_clusters);
        for(int i=0;i<num_clusters;i++)
        {
            int vector;

            do {
                vector = (int) (vectors.length*Math.random());
            } while(random_vectors.contains(vector));

            random_vectors.add(vector);
            means[i] = Arrays.copyOf(vectors[vector],vectors[vector].length);
        }

        int iter = 0;
        while(iter < num_iterations)
        {
            expectation();
            maximization();
            System.out.println("expectation-maximization "+iter);
            iter++;
        }
    }

    void expectation()
    {
        double normalizer = 0.0;

        for(int v=0; v<num_vectors; v++)
        {
            for(int c=0; c<num_clusters; c++)
            {
                double p = gaussianProb(euclideanDist(vectors[v], means[c]));
                clusters[c][v] = p;
                normalizer += p;
            }

            if (normalizer == 0.0) continue;
            for(int c=0; c<num_clusters; c++)
                clusters[c][v] /= normalizer;
        }
    }

    void maximization()
    {
        for(int c=0; c<num_clusters; c++)
        {
            double normalizer = 0.0;

            for (int v = 0; v < num_vectors; v++)
            {
                for (int i = 0; i < vector_size; i++)
                    means[c][i] += clusters[c][v] * vectors[v][i];

                normalizer += clusters[c][v];
            }

            if (normalizer == 0.0) continue;
            for (int i = 0; i < vector_size; i++)
                means[c][i] /= normalizer;
        }
    }

    void normalize(float[] v)
    {
        double norm = 0.0f;

        for(int i=0; i<vector_size; i++)
            norm += v[i]*v[i];
        norm = Math.sqrt(norm);

        if (norm == 0.0) return;
        for(int i=0; i<vector_size; i++)
            v[i] /= norm;
    }

    double gaussianProb(double distance)
    {
        // Note that any constant factor will be normalized out in expectation() phase
        double p = Math.exp(- distance*distance/(2*std_dev*std_dev));
        return p!=0 ? p : Double.MIN_VALUE;
    }

    double euclideanDist(float[] v, float[] w)
    {
        double distance = 0.0f;

        for(int i=0; i<vector_size; i++)
            distance += (v[i]-w[i])*(v[i]-w[i]);

        distance = Math.sqrt(distance);

        return distance;
    }

    public void write(File output_file) throws IOException {
        Writer out = new BufferedWriter(new FileWriter(output_file));

        for(int v=0; v<num_vectors; v++)
        {
            int max = 0;
            for(int c=0; c<num_clusters; c++)
                if (clusters[c][v]>clusters[max][v])
                    max = c;

            out.write(word_list.get(v)+"\t");
            out.write(max+"\t");
            for(int c=0; c<num_clusters; c++)
                out.write(clusters[c][v]+"\t");
            out.write("\n");
        }
        out.close();
    }

    public static void main(String[] args)
    {
        System.out.println("loading...");
        new VecCluster(args);
    }
}
