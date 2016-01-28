/**
 * Copyright 2016, Emory University
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
package edu.emory.mathcs.nlp.vsm.evaluate;

import edu.emory.mathcs.nlp.common.util.BinUtils;
import org.kohsuke.args4j.Option;

import java.io.*;
import java.util.HashMap;
import java.util.Map;

/**
 * @author Austin Blodgett
 */
public class TriadEvaluator
{
    @Option(name="-input", usage="file of word vectors to evaluate.", required=true, metaVar="<filename>")
    String vector_file = null;
    @Option(name="-triads", usage="file with data from participant triad task.", required=false, metaVar="<filename>")
    String triad_file = null;
    @Option(name="-output", usage="output file to save evaluation.", required=false, metaVar="<filename>")
    String output_file = null;

    Map<String,float[]> map;

    public TriadEvaluator(String[] args)
    {
        BinUtils.initArgs(args, this);
        try { map = getVectors(new File(vector_file)); }
        catch (IOException e) { System.err.println("Could not load Word2Vec vectors."); e.printStackTrace(); System.exit(1); }

        try { evaluate(new File(triad_file)); }
        catch (IOException e) { System.err.println("Could not evaluate triads."); e.printStackTrace(); System.exit(1); }
    }

    private Map<String,float[]> getVectors(File vector_file) throws IOException
    {
        Map<String,float[]> map = new HashMap<>();

        BufferedReader in = new BufferedReader(new FileReader(vector_file));
        String line;
        while((line = in.readLine()) != null)
        {
            String[] split = line.split("\t");
            String word = split[0];
            float[] vector = new float[split.length - 1];
            for (int i=1; i<split.length; i++)
                vector[i-1] = Float.parseFloat(split[i]);
            map.put(word, vector);
        }
        in.close();

        return map;
    }

    public void evaluate(File triad_file) throws IOException
    {
        BufferedReader in = new BufferedReader(new FileReader(triad_file));

        float weighted_eval = 0;
        int weighted_count = 0;


        String line;
        while((line = in.readLine()) != null){
            String[] triad = line.split(",");
            if(triad.length != 5) throw new IOException("Could not read triad file. Incorrect format.");

            if(!(map.containsKey(triad[0]) && map.containsKey(triad[1]) && map.containsKey(triad[2])))
                    continue;

            int word_count1 = Integer.parseInt(triad[3]);
            int word_count2 = Integer.parseInt(triad[4]);

            if((word_count1 > word_count2) == (similarity(triad[1],triad[0]) > similarity(triad[2],triad[0]))) {
                for(int i=0; i<Math.abs(word_count1-word_count2); i++)
                    weighted_eval++;
            }

            for(int i=0; i<Math.abs(word_count1-word_count2); i++)
                weighted_count++;
        }

        in.close();

        if(weighted_count != 0)
            weighted_eval /= weighted_count;

        System.out.println(vector_file+" Weighted Traid Evaluation: " + weighted_eval);

        if (output_file != null)
        {
            BufferedWriter out = new BufferedWriter(new FileWriter(output_file));
            out.write(vector_file + " Weighted Traid Evaluation: " + weighted_eval + "\n");
            out.close();
        }
    }

    double similarity(String word1, String word2)
    {
        float[] w1 = map.get(word1);
        float[] w2 = map.get(word2);

        double norm1 = 0.0, norm2 = 0.0;

        double dot_product = 0.0;
        for(int k=0; k<w1.length; k++){
            dot_product += w1[k]*w2[k];
            norm1       += w1[k]*w1[k];
            norm2       += w2[k]*w2[k];
        }
        norm1 = Math.sqrt(norm1);
        norm2 = Math.sqrt(norm2);

        return dot_product/(norm1*norm2);
    }

    public static void main(String[] args) { new TriadEvaluator(args); }
}
