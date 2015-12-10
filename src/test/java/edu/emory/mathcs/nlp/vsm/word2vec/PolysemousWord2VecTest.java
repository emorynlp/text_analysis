package edu.emory.mathcs.nlp.vsm.word2vec;

import edu.emory.mathcs.nlp.text_analysis.word2vec.PolysemousWord2Vec;
import edu.emory.mathcs.nlp.text_analysis.word2vec.Word2Vec;
import org.junit.Test;

import java.io.IOException;

/**
 * Created by austin on 12/9/2015.
 */
public class PolysemousWord2VecTest {

    static int vector_size = 20;
    static int senses = 5;


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
                "-min-count","1",
                "-senses","5",
                "-tokenize",
                "-probabilistic",
                "-evaluate", "src/test/resources/dat/test_files/test1.txt"};
        test(params);
    }

    public static void test_cbow() throws IOException {
        String[] params = {	"-train","src/test/resources/dat/test_files",
                "-output","src/test/resources/dat/cbow_vectors",
                "-size",""+vector_size,
                "-threads",  "2",
                "-min-count","1",
                "-senses","5",
                "-cbow",
                "-tokenize",
                "-probabilistic",
                "-evaluate", "src/test/resources/dat/test_files/test1.txt"};
        test(params);
    }

    public static void test(String[] params) throws IOException {
        PolysemousWord2Vec word2vec = new PolysemousWord2Vec(params);

        for(int i=0; i<word2vec.vocab.size(); i++) {
            for (int s = 0; s < senses; s++) {
                System.out.print(word2vec.vocab.get(i).form+"."+s+" ");
                for (int j = 0; j < vector_size; j++)
                    System.out.print(String.format("%1$,.6f", word2vec.W[s][i][j]) + " ");
                System.out.println();
            }
        }
    }

}
