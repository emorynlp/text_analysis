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

import java.lang.Math;

import java.util.Arrays;
import java.util.Map;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * @author Reid Kilgore, Austin Blodgett
 */
public class AnalogyTestMulti
{

    @Option(name="-input", usage="file of word vectors to evaluate.", required=true, metaVar="<filename>")
    String vector_file = null;
    @Option(name="-output", usage="output file to save evaluation.", required=false, metaVar="<filename>")
    String output_file = null;
    @Option(name="-test-file", usage="file of tests.", required=false, metaVar="<filename>")
    String test_file = null;
    @Option(name="-threads", usage="number of threads to use.", required=false, metaVar="<int>")
    int threads = 4;


    Map<String,float[]> map = null;
    int totalTests;

    //Not necessary for current implementation yet
    Map<String,Map<String,Float>> matrix = new HashMap<>();

    volatile Map<String,float[]> memo = new HashMap<>();
    volatile Map<String,int[]> results = new HashMap<>();
    volatile Map<String,List<String[]>> analogyMap = new HashMap<String,List<String[]>>();


    public AnalogyTestMulti(String[] args)
    {
        System.out.println("Loading...");
        BinUtils.initArgs(args, this);
        try { map = getVectors(new File(vector_file)); }
        catch (IOException e) { System.err.println("Could not load Word2Vec vectors."); e.printStackTrace(); System.exit(1); }

        try { buildTestMap(new File(test_file));}
        catch (IOException e) { System.err.println("Could not read word file."); e.printStackTrace(); System.exit(1);}

        int splitTasks = totalTests/threads;


        System.out.println("Finished building analogy tests.");
        System.out.println("Running tests.");

        ExecutorService executor = Executors.newFixedThreadPool(threads);
        int id = 0;
        int splitter, subTasks;
        System.out.println("Starting " + analogyMap.keySet().size() + " categories");

        /*  TODO (Reid) - this is too messy, won't result in even task lists,
        *   not worth the easier updates.
        */
        for(String category : analogyMap.keySet())
        {
            subTasks = analogyMap.get(category).size();
            System.out.println(category + " size is " + subTasks);
            splitter = 0;
            while( splitter + splitTasks < subTasks )
            {
                executor.execute(new AnalogyTask(category, analogyMap.get(category).subList(splitter, splitter + splitTasks), id++));
                splitter += splitTasks;
            }

            executor.execute(new AnalogyTask(category, analogyMap.get(category).subList(splitter, subTasks), id++));
        }

        executor.shutdown();
        try { executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS); }
        catch (InterruptedException e) {e.printStackTrace();}


        for(String category : results.keySet())
            System.out.println("I think " + category + " had " + results.get(category)[1] + " correct");
        try{
            BufferedWriter out = new BufferedWriter(new FileWriter(output_file));
            String nl = "\n";
            String sp = " ";

            for(String category : analogyMap.keySet())
            {
                out.write(category + nl);
                for(String[] analogyResults : analogyMap.get(category))
                {
                    for(String analogyResult : analogyResults)
                        out.write(analogyResult + sp);
                    out.write(nl);
                }
                out.write(nl);
            }

            int total = 0;
            int correct = 0;
            int stotal;
            int scorrect;

            for(String category : results.keySet()){
                total   +=  (stotal = results.get(category)[0]);
                correct +=  (scorrect = results.get(category)[1]);
                out.write(category + sp + total + sp + correct + nl);
            }
            out.write("Total: " + total + " Correct: " + correct + " Score: " + (float)correct/(float)total);
            out.close();
        } catch (Exception e) {e.printStackTrace();}

    }

    class AnalogyTask implements Runnable
    {
        public List<String[]> analogies;
        int id;
        String category;
        public int total = 0;
        public int correct = 0;

        public AnalogyTask(String category, List<String[]> analogies, int id)
        {
            this.analogies = analogies;
            this.id = id;
            this.category = category;
        }

        @Override
        public void run()
        {
            System.out.println("Thread " + id + " running task");
            int l = analogies.size();
            int res;
            for(String[] analogy : analogies){
                res = runTest(analogy);
                if(res != -1)
                    correct += res;
                total++;
                if(total % 100 == 0)
                    System.out.println("Task " + id + " total: " + total + " Progress: " + (float)total/(float)l);
            }
            synchronized(results){
                results.get(category)[0] += total;
                results.get(category)[1] += correct;
            }
            System.out.println(category + " correct is now " + results.get(category)[1] + " because of " + correct + " write");
            System.out.println("Thread " + id + " finished task");
        }

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

    private void buildTestMap(File test_file) throws IOException
    {
        //TODO (Reid) think about changing the object designs here
        totalTests = 0;
        BufferedReader in = new BufferedReader(new FileReader(test_file));

        String word;
        String category;
        String line[];
        String question[];
        List<String[]> testList = new ArrayList<String[]>();

        while((word = in.readLine()) != null){
            line = word.split(" ");
            if(line[0].charAt(0) == ':')    //New Category
            {
                category = Arrays.toString(Arrays.copyOfRange(line, 1, line.length));
                testList = new ArrayList<String[]>();
                // expects format:
                    //: category-name
                System.out.println("Building for " + category + " with size " + testList.size());
                analogyMap.put(category, testList);
                results.put(category, new int[] {0, 0});
            }
            else if (line.length == 4)
            {
                totalTests += 1;
                question = new String[6];
                for(int i = 0; i < 4; i++)  question[i] = line[i];
                testList.add(question);
            }
            else System.out.println("Unparseable: " + Arrays.toString(line));
        }

        in.close();
    }

    float[] getTestVector(float[] v1, float[] v2, float[] v3)
    {
        float[] a = subtractVectors(v1, v2);
        a = addVectors(a, v3);

        //This is in the original implmentation
        float len = 0;
        for(float f : a)                    len += f * f;
        len = (float)Math.sqrt((double)len);
        for(int i = 0; i < a.length; i++)   a[i] /= len;


        return a;
    }

    float[] getTestVector(String w1, String w2, String w3)
    {
        if( !(map.containsKey(w1) && map.containsKey(w2) && map.containsKey(w3)) )
            return null;
        return getTestVector(map.get(w1), map.get(w2), map.get(w3));
    }

    private int runTest(String[] analogy)
    {
        float[] answerVector;
        String pos1 = analogy[1].toLowerCase();
        String neg1 = analogy[0].toLowerCase();
        String pos2 = analogy[2].toLowerCase();

        String gold = analogy[3].toLowerCase();
        answerVector = getTestVector(pos1, neg1, pos2);
        if(answerVector == null){
            analogy[4] = "OOV";
            analogy[5] = "OOV";
            return -1;
        }

        String prediction = getNearestWord(answerVector);
        analogy[4] = prediction;
        if(prediction.equals(gold)){
            analogy[5] = "correct";
            return 1;
        }
        else{
            analogy[5] = "incorrect";
            return 0;
        }

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
        new AnalogyTestMulti(args);
    }

}
