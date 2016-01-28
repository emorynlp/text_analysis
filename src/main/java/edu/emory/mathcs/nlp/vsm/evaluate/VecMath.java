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

/**
 * @author Austin Blodgett
 */
public class VecMath
{


    @Option(name="-input", usage="file of word vectors to evaluate.", required=true, metaVar="<filename>")
    String vector_file = null;
    @Option(name="-output", usage="output file to save evaluation.", required=false, metaVar="<filename>")
    String output_file = null;
    @Option(name="-word-file", usage="file with list of words to consider (for faster processing).", required=false, metaVar="<filename>")
    String word_file = null;

    Map<String,float[]> map = null;
    Set<String> word_list = null;

    public VecMath(String[] args)
    {
        BinUtils.initArgs(args, this);
        try { map = getVectors(new File(vector_file)); }
        catch (IOException e) { System.err.println("Could not load Word2Vec vectors."); e.printStackTrace(); System.exit(1); }

        if (word_file != null) {
            try { word_list = getWordList(new File(word_file));}
            catch (IOException e) { System.err.println("Could not read word file."); e.printStackTrace(); System.exit(1);}
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

    private Set<String> getWordList(File word_file) throws IOException
    {
        Set<String> word_list = new HashSet<>();
        BufferedReader in = new BufferedReader(new FileReader(word_file));

        String word;
        while((word = in.readLine()) != null)
            word_list.add(word);

        in.close();
        return word_list;
    }

    void add(float[] a, float[] vector)
    {
        for (int i=0; i<a.length; i++)
            vector[i] += a[i];
    }

    void subtract(float[] a, float[] vector)
    {
        for (int i=0; i<a.length; i++)
            vector[i] -= a[i];
    }

    void parseCommand(String line) throws IOException
    {
        float[] vector;
        line = line.replaceAll("\\s+","");

        List<String> parse = new ArrayList<>();

        int i=0;
        while (i<line.length())
        {
            int next_plus  = line.indexOf('+',i);
            int next_minus = line.indexOf('-',i);
            if (next_plus < 0)  next_plus = line.length();
            if (next_minus < 0) next_minus = line.length();

            String word = line.substring(i, Math.min(next_plus, next_minus));

            if (!map.containsKey(word))
            {
                System.out.println("Cannot find word vector "+word+".");
                return;
            }
            if (word.isEmpty())
            {
                System.out.println("Cannot parse empty string as word vector.");
                return;
            }
            parse.add(word);
            if (next_plus < next_minus)      parse.add("+");
            else if (next_minus < next_plus) parse.add("-");

            i = Math.min(next_plus, next_minus) + 1;
        }
        if (parse.isEmpty())
        {
            System.out.println("Please input a command for vector arithmetic or type q to quit.");
            return;
        }

        vector = map.get(parse.get(0));

        for (i=1; i<parse.size(); i+=2)
        {
            if      (parse.get(i).equals("+")) add(map.get(parse.get(i)),vector);
            else if (parse.get(i).equals("-")) subtract(map.get(parse.get(i)),vector);
            else
            {
                System.out.println("Parsing failure. Please try again.");
                return;
            }
        }

        StringBuilder sb = new StringBuilder();

        sb.append(vector_file).append(" top matches").append("\n");

        Map<String, Float> top = getTopTen(vector);
        for (String word2 : top.keySet())
            sb.append(word2).append(" ").append(top.get(word2)).append("\n");


        System.out.println(sb.toString());
        if (output_file != null)
        {
            BufferedWriter out = new BufferedWriter(new FileWriter(output_file));
            out.write(line+"\n");
            out.write(sb.toString());
            out.close();
        }
    }

    Map<String,Float> getTopTen(float[] vector)
    {
        TopNQueue top_ten = new TopNQueue(10);

        for (String word2 : map.keySet())
        {
            if (word_list != null && !word_list.contains(word2))
                continue;

            top_ten.add(word2, cosine(vector, map.get(word2)));
        }

        return top_ten.toMap();
    }

    float cosine(float[] w1, float[] w2)
    {
        float norm1 = 0.0f, norm2 = 0.0f;

        float dot_product = 0.0f;
        for(int k=0; k<w1.length; k++){
            dot_product += w1[k]*w2[k];
            norm1       += w1[k]*w1[k];
            norm2       += w2[k]*w2[k];
        }
        norm1 = (float) Math.sqrt(norm1);
        norm2 = (float) Math.sqrt(norm2);

        return dot_product/(norm1*norm2);
    }

    class TopNQueue
    {
        int size;
        // top 10 values from samllest to largest
        List<String> words  = new ArrayList<>(size);
        List<Float>  values = new ArrayList<>(size);

        TopNQueue(int N)
        {
            size = N;
            for (int i=0; i<size; i++)
            {
                words.add(null);
                values.add(Float.MIN_VALUE);
            }
        }

        void add(String word, float value)
        {
            if (value > values.get(0))
            {
                words.set(0, word);
                values.set(0, value);
            }
            int i = 1;
            while (i < size && value > values.get(i))
            {
                words.set(i-1, words.get(i));
                values.set(i-1, values.get(i));
                words.set(i, word);
                values.set(i, value);
                i++;
            }
        }

        Map<String,Float> toMap()
        {
            Map<String, Float> map = new HashMap<>(size);

            for (int i=0; i<size; i++)
            {
                if (words.get(i) != null)
                    map.put(words.get(i),values.get(i));
            }

            return map;
        }
    }

    public static void main(String[] args)
    {
        System.out.println("loading...");
        VecMath vec_math = new VecMath(args);

        try{
            BufferedReader br = new BufferedReader(new InputStreamReader(System.in));

            String input;
            System.out.println("Please input a command for vector arithmetic or type q to quit.");
            System.out.println("Example: king - man + woman");
            while((input=br.readLine())!=null){
                if (input.equals("q")) break;
                vec_math.parseCommand(input);
            }

        }catch(IOException io){
            io.printStackTrace();
        }

    }

}
