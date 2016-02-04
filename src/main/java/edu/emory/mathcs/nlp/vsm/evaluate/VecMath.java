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
    @Option(name="--N", usage="number of closest word vectors to find.", required=false, metaVar="<integer>")
    int N = 10;

    Map<String,float[]> map = null;
    Set<String> word_list = null;

    static Pattern vec_arithmetic = Pattern.compile("(\\w+)\\s*(\\s*[+-]\\s*\\w+\\s*)*");

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

    void parseCommand(String line) throws IOException
    {
        line = line.replaceAll("\\s+","");

        // records words in arithmetic expression as keys,
        // with value false if needs to be subtracted, true otherwise.
        Map<String, Boolean> word_is_plus = new HashMap<>();
        boolean isPlus = true;
        int i=0;
        while (i<line.length())
        {
            if (line.charAt(i) == '-') { isPlus = false; i++; }
            else if (line.charAt(i) == '+') { isPlus = true; i++; }

            int next_plus  = line.indexOf('+',i+1);
            int next_minus = line.indexOf('-',i+1);
            if (next_plus < 0)  next_plus = line.length();
            if (next_minus < 0) next_minus = line.length();

            String word = line.substring(i, Math.min(next_plus, next_minus));

            if (!map.containsKey(word))
            {
                System.out.println("Cannot find word vector "+word+".");
                return;
            }
            word_is_plus.put(word, isPlus);

            i = Math.min(next_plus, next_minus);
        }

        float[] vector = null;
        for (String w : word_is_plus.keySet())
        {
            float[] a = map.get(w);
            if (vector == null) { vector = new float[a.length]; Arrays.fill(vector, 0f); }

            if (word_is_plus.get(w))
                for (i=0; i<a.length; i++) vector[i] += a[i];
            else
                for (i=0; i<a.length; i++) vector[i] -= a[i];

        }

        StringBuilder sb = new StringBuilder();

        sb.append(vector_file).append(" top matches").append("\n");

        Map<String, Float> top = getTopN(vector, N);
        // sort in descending order
        List<String> list = new ArrayList<>(top.keySet());
        Collections.sort(list, (o1, o2) -> top.get(o1)<top.get(o2) ? 1 : -1);


        for (String word2 : list)
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

    Map<String,Float> getTopN(float[] vector, int N)
    {
        TopNQueue top_ten = new TopNQueue(N);

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

    class TopNQueue
    {
        int size;
        // top N values from smallest to largest
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
                if (!vec_arithmetic.matcher(input).matches())
                {
                    System.err.println("Cannot parse vector arithmetic.");
                    System.out.println("Example: king - man + woman");
                    System.out.println("Type q to quit.");
                    continue;
                }
                vec_math.parseCommand(input);
            }

        }catch(IOException io){
            io.printStackTrace();
        }

    }

}
