package edu.emory.mathcs.nlp.vsm.evaluate;

import edu.emory.mathcs.nlp.common.util.BinUtils;
import edu.emory.mathcs.nlp.vsm.Word2Vec;
import org.kohsuke.args4j.Option;

import java.io.*;

/**
 * Created by austin on 2/12/2016.
 */
public class VecFeatures
{
    @Option(name="-input", usage="file of word2vec model to evaluate.", required=true, metaVar="<filename>")
    String model_file = null;
    @Option(name="-output", usage="output file to save evaluation.", required=false, metaVar="<filename>")
    String output_file = null;
    @Option(name="--N", usage="number of closest word vectors to find.", required=false, metaVar="<integer>")
    int N = 20;

    Word2Vec word2vec;

    public VecFeatures(String[] args)
    {
        BinUtils.initArgs(args, this);
        try { readModel(new File(model_file)); } catch (IOException e) { e.printStackTrace(); }

        try { saveFeatures(new File(output_file)); } catch (IOException e) { e.printStackTrace(); }
    }


    public void readModel(File read_model_file) throws IOException
    {
        ObjectInputStream oin = new ObjectInputStream(new FileInputStream(read_model_file));
        try
        {
            word2vec = (Word2Vec) oin.readObject();
        }
        catch (ClassNotFoundException e) { e.printStackTrace(); }
        oin.close();
    }

    void saveFeatures(File feature_file) throws IOException
    {
        if (!feature_file.isFile())
            feature_file.createNewFile();

        float value;
        BufferedWriter out = new BufferedWriter(new FileWriter(feature_file));

        for (int k=0; k<word2vec.vector_size; k++)
        {
            TopNQueue top = new TopNQueue(N);
            for (int v=0; v<word2vec.out_vocab.size(); v++)
            {
                value = word2vec.V[v*word2vec.vector_size + k] * (float) Math.pow(word2vec.out_vocab.get(v).count, 0.25);
                top.add(word2vec.out_vocab.get(v).form, value);
            }
            out.write(k+"\t");
            for (String s : top.list())
                out.write(s+"\t");
            out.write("\n");
        }

        out.close();
    }

    public static void main(String[] args)
    {
        new VecFeatures(args);
    }
}
