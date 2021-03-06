package edu.emory.mathcs.nlp.vsm.word2vec;

import edu.emory.mathcs.nlp.vsm.Lemma2Vec;
import org.junit.Test;

import edu.emory.mathcs.nlp.vsm.SyntacticWord2Vec;
import edu.emory.mathcs.nlp.vsm.Word2Vec;

import java.io.IOException;

/**
 * Created by austin on 12/1/2015.
 */
public class SyntacticWord2VecTest {

    static int vector_size = 20;

    @Test
    public void test() throws Exception {
        System.out.println("Skipgrams:");
        test_skipgrams();
        System.out.println("CBOW:");
        test_cbow();
    }

    public static void test_skipgrams() throws IOException {
        String[] params = {	"-train","resources/dat/dep_test_files",
                "-output","resources/dat/skip_vectors",
                "-size",""+vector_size,
                "-threads",  "2",
                "-min-count","1",
                "-evaluate"};
        test(params);
    }

    public static void test_cbow() throws IOException {
        String[] params = {	"-train","resources/dat/dep_test_files",
                "-output","resources/dat/cbow_vectors",
                "-size",""+vector_size,
                "-threads",  "3",
                "-min-count","1",
                "-cbow",
                "-evaluate"};
        test(params);
    }

    public static void test(String[] params) throws IOException {
        Word2Vec word2vec = new Lemma2Vec(params);

        for(int i=0; i<word2vec.in_vocab.size(); i++){
            System.out.print(word2vec.in_vocab.get(i).form+" ");
            for(int j=0; j<vector_size; j++)
                System.out.print(String.format("%1$,.6f",word2vec.W[i*vector_size+j])+" ");
            System.out.println();
        }
    }

}
