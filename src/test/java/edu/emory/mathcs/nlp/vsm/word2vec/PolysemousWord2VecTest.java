package edu.emory.mathcs.nlp.vsm.word2vec;

import org.junit.Test;

import edu.emory.mathcs.nlp.vsm.PolysemousWord2Vec;

import java.io.IOException;

/**
 * Created by austin on 1/20/2016.
 */
public class PolysemousWord2VecTest {

    static int vector_size = 20;
    static int senses = 10;

        @Test
        public void test() throws Exception {
            System.out.println("Skipgrams:");
            test_skipgrams();
            System.out.println("CBOW:");
            test_cbow();
        }


        public static void test_skipgrams() throws IOException {
            String[] params = {	"-train","resources/dat/test_files",
                    "-output","resources/dat/skip_vectors",
                    "-size",""+vector_size,
                    "-threads",  "2",
                    "-min-count","1",
                    "-senses", ""+senses};
            test(params);
        }

        public static void test_cbow() throws IOException {
            String[] params = {"-train","resources/dat/test_files",
                    "-output","resources/dat/cbow_vectors",
                    "-size",""+vector_size,
                    "-threads","2",
                    "-min-count","1",
                    "-cbow",
                    "-senses",""+senses};
            test(params);
        }

        public static void test(String[] params) throws IOException {
            PolysemousWord2Vec word2vec = new PolysemousWord2Vec(params);

            for(int i=0; i<word2vec.in_vocab.size(); i++) for (int s=0; s<senses; s++)
            {
                System.out.print(word2vec.in_vocab.get(i).form+s+" ");
                for(int j=0; j<vector_size; j++)
                    System.out.print(String.format("%1$,.6f",word2vec.S[s][i*vector_size+j])+" ");
                System.out.println();
            }
        }

    }
