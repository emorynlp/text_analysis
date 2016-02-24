package edu.emory.mathcs.nlp.vsm.evaluate;

import java.util.*;

/**
 * Created by austin on 2/9/2016.
 */
public class TopNQueue
{
    int size;
    // top N values from smallest to largest
    List<String> words  = new ArrayList<>(size);
    List<Float>  values = new ArrayList<>(size);

    public TopNQueue(int N)
    {
        size = N;
        for (int i=0; i<size; i++)
        {
            words.add(null);
            values.add(Float.MIN_VALUE);
        }
    }

    public void add(String word, float value)
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

    public Map<String,Float> toMap()
    {
        Map<String, Float> map = new HashMap<>(size);

        for (int i=0; i<size; i++)
        {
            if (words.get(i) != null)
                map.put(words.get(i),values.get(i));
        }

        return map;
    }

    public List<String> list()
    {
        List<String> list = new ArrayList<>(words);
        Collections.reverse(list);
        return list;
    }
}
