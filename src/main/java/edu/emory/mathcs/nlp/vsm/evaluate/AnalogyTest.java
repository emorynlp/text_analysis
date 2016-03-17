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
import java.util.*;
import java.util.regex.Pattern;

/**
 * @author Reid Kilgore, Austin Blodgett
 */
public class AnalogyTest
{

    @Option(name="-input", usage="file of word vectors to evaluate.", required=true, metaVar="<filename>")
    String vector_file = null;
    @Option(name="-output", usage="output file to save evaluation.", required=false, metaVar="<filename>")
    String output_file = null;
    @Option(name="-test-file", usage="file of tests.", required=false, metaVar="<filename>")
    String test_file = null;

    Map<String,float[]> map = null;
    List<String[]> testList = null;
    Map<String,Map<String,Float>> matrix = new HashMap<>();

    public AnalogyTest(String[] args)
    {
        BinUtils.initArgs(args, this);
        try { map = getVectors(new File(vector_file)); }
        catch (IOException e) { System.err.println("Could not load Word2Vec vectors."); e.printStackTrace(); System.exit(1); }

        try { testList = getTestList(new File(test_file));}
        catch (IOException e) { System.err.println("Could not read word file."); e.printStackTrace(); System.exit(1);}
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

    private List<String[]> getTestList(File test_file) throws IOException
    {
        List<String[]> testList = new ArrayList<String[]>();
        BufferedReader in = new BufferedReader(new FileReader(test_file));

        String word;
        String line[];
        while((word = in.readLine()) != null){
            testList.add(word.split(" "));
        }

        in.close();
        return testList;
    }

    float[] getTestVector(float[] v1, float[] v2, float[] v3)
    {
        float[] a = subtractVectors(v1, v2);
        a = addVectors(a, v3);
        return a;
    }

    float[] getTestVector(String w1, String w2, String w3)
    {
        if(!map.containsKey(w1) || !map.containsKey(w2) || !map.containsKey(w3))
            return null;
        return getTestVector(map.get(w1), map.get(w2), map.get(w3));
    }


    private void runTests() throws IOException
    {
        BufferedWriter out = null;
        //File output = new File(output_file);
        //if(!output.isFile()) output.createNewFile();
        //out = new BufferedWriter(new FileWriter(output));
        out = new BufferedWriter(new FileWriter(output_file));


        out.write("+W1 -W2 +W3 =Truth Prediction Correct?\n");
        float [] tVector = null;
        String prediction = null;
        int total = 0;
        int stotal = 0;
        int correct = 0;
        int scorrect = 0;
        String pos1;
        String neg1;
        String pos2;
        String gold;
        String lastSection = null;
        Map<String,float[]> memo = new HashMap<>();

        for(String[] list : testList){
            if(list.length != 4){
                if(lastSection != null)
                {

                    // New Section
                    out.write(list[1] + "\n");
                    out.write(lastSection + " " + stotal + " " + scorrect + " " + ( (float)scorrect / (float)stotal ) + "\n" );
                    System.out.println(lastSection + " " + stotal + " " + scorrect + " " + ( (float)scorrect / (float)stotal ) + "\n" );
                    stotal = 0;
                    scorrect = 0;
                }

                lastSection = list[1];

                continue;
            }

            // Get test vector, w1 - w2 + w4 should = w3.
            // Dynamic
            pos1 = list[2].toLowerCase();
            neg1 = list[0].toLowerCase();
            pos2 = list[1].toLowerCase();
            gold = list[3].toLowerCase();

            tVector = getTestVector(pos1, neg1, pos2);
            prediction = getNearestWord(tVector);

            out.write(pos1 + " " + neg1 + " " + pos2 + " " + gold + " " + prediction);

            if(prediction.toLowerCase().equals(gold)){
                out.write(" Y\n");
                correct++;
                scorrect++;
            }
            else
                out.write(" N\n");

            stotal++;
            total++;

            //Logging
            if(total%100 == 0)
            {
                out.flush();
                System.out.println("Running total: " + total + " Correct: " + correct + " Score: " + ( (float)correct / (float)total ) );
            }
        }
        out.write(lastSection + " " + stotal + " " + scorrect + " " + ( (float)scorrect / (float)stotal ) + "\n" );
        System.out.println(lastSection + " " + stotal + " " + scorrect + " " + ( (float)scorrect / (float)stotal ) + "\n" );
        out.write("Total: " + total + " correct: " + correct + " score: " + ( (float)correct / (float)total ));
        System.out.println("Total: " + total + " correct: " + correct + " score: " + ( (float)correct / (float)total ));
        out.close();
    }

    private float[] addVectors(float[] a, float[] b)
    {
        float[] c = new float[a.length];
        for(int i = 0; i < c.length; i++)
           c[i] = a[i] + b[i];
       return c;
    }

    private float[] subtractVectors(float[] a, float[] b)
    {
        float[] c = new float[a.length];
        for(int i = 0; i < c.length; i++)
           c[i] = a[i] - b[i];
       return c;
    }

    private String getNearestWord(String word){
        String nearest = null;
        float maxSimilarity = -Float.MAX_VALUE;
        float cos;

        for (String word2 : map.keySet())
        {
            cos = cosine(word, word2);
            if ( cos > maxSimilarity)
            {
                nearest = word2;
                maxSimilarity = cos;
            }
        }
        return nearest;
    }

    private String getNearestWord(float[] vector){
        String nearest = null;
        float maxSimilarity = -Float.MAX_VALUE;
        float cos;

        for (String word2 : map.keySet())
        {
            cos = cosine(vector, map.get(word2));
            if ( cos > maxSimilarity)
            {
                nearest = word2;
                maxSimilarity = cos;
            }
        }
        return nearest;
    }

    float cosine(String w1, String w2)
    {
        return cosine(map.get(w1), map.get(w2));
    }

    float cosine(float[] w1, float[] w2)
    {
        double norm1 = 0.0f, norm2 = 0.0f;

        double dot_product = 0.0f;
        for(int k=0; k<w1.length; k++){
            dot_product += w1[k]*w2[k];
            norm1       += w1[k]*w1[k];
            norm2       += w2[k]*w2[k];
        }
        norm1 = Math.sqrt(norm1);
        norm2 = Math.sqrt(norm2);

        return (float)(dot_product/(norm1*norm2));
    }


    void buildCosineMatrix()
    {
        float similarity = 0;
        for(String key : map.keySet())
        {
            for(String key2 : map.keySet())
            {
                if(!matrix.containsKey(key))
                    matrix.put(key, new HashMap<>());
                if(!matrix.containsKey(key2))
                    matrix.put(key2, new HashMap<>());
                if(matrix.get(key).containsKey(key2))
                    continue;

                matrix.get(key).put(key2, cosine(key, key2));
                if(! key.equals(key2) )
                    matrix.get(key2).put(key, matrix.get(key).get(key2));
            }
        }
    }

    public static void main(String[] args) throws IOException
    {
        System.out.println("loading...");
        AnalogyTest test = new AnalogyTest(args);
        System.out.println("Finished building analogy tests.");
        System.out.println("Running tests.");
        //buildCosineMatrix();
        test.runTests();


    }

}
