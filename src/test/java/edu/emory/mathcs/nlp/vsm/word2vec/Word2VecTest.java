package edu.emory.mathcs.nlp.vsm.word2vec;

import edu.emory.mathcs.nlp.text_analysis.word2vec.Word2Vec;
import org.junit.Test;

import java.io.IOException;

/**
 * Created by austin on 11/30/2015.
 */
public class Word2VecTest {

    static int vector_size = 20;

    @Test
    public void test() throws Exception {
        System.out.println("Skipgrams:");
        test_skipgrams();
        System.out.println("CBOW:");
        test_cbow();
    }


    public static void test_skipgrams() throws IOException {
        String[] params = {	"-train","src/test/resources/dat/test_files",
                "-output","src/test/resources/dat/skip_vectors",
                "-size",""+vector_size,
                "-threads",  "2",
                "-min-count","1"};
        test(params);
    }

    public static void test_cbow() throws IOException {
        String[] params = {	"-train","src/test/resources/dat/test_files",
                "-output","src/test/resources/dat/cbow_vectors",
                "-size",""+vector_size,
                "-threads",  "2",
                "-min-count","1",
                "-cbow"};
        test(params);
    }

    public static void test(String[] params) throws IOException {
        Word2Vec word2vec = new Word2Vec(params);

        for(int i=0; i<word2vec.in_vocab.size(); i++){
            System.out.print(word2vec.in_vocab.get(i).form+" ");
            for(int j=0; j<vector_size; j++)
                System.out.print(String.format("%1$,.6f",word2vec.W[i*vector_size+j])+" ");
            System.out.println();
        }
    }

}
