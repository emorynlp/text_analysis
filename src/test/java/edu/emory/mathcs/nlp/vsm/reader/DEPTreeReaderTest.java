package edu.emory.mathcs.nlp.vsm.reader;

import java.io.File;
import java.io.FileInputStream;
import java.util.List;
import java.util.stream.Collectors;

import org.junit.Test;

import edu.emory.mathcs.nlp.common.util.FileUtils;
import edu.emory.mathcs.nlp.component.template.node.NLPNode;
import edu.emory.mathcs.nlp.component.template.util.TSVReader;
import edu.emory.mathcs.nlp.vsm.reader.DEPTreeReader;
import edu.emory.mathcs.nlp.vsm.reader.Reader;

/**
 * Created by austin on 11/30/2015.
 */
public class DEPTreeReaderTest {

    @Test
    public void testNext() throws Exception {

        List<String> filenames = FileUtils.getFileList("resources/dat/dep_test_files", "*");
        List<File> files = filenames.stream().map(File::new).collect(Collectors.toList());

        DEPTreeReader dr = new DEPTreeReader(files);

        List<Reader<NLPNode>> readers = dr.splitParallel(4);

        List<NLPNode> words;
        int i=0;
        for (Reader<NLPNode> r : readers) {
            System.out.println(i);
            while ((words = r.next()) != null) {
                for (NLPNode word : words)
                    System.out.print(word.getLemma() + " ");
                System.out.println();
            }
            i++;
        }

        dr.close();
        System.out.println("Finished");
    }

    @Test
    public void testTSVReader() throws Exception {

        List<String> filenames = FileUtils.getFileList("resources/dat/dep_test_files", "*");

        TSVReader tree_reader = new TSVReader();

        tree_reader.form = 1;
        tree_reader.lemma = 2;
        tree_reader.pos = 3;
        tree_reader.dhead = 5;
        tree_reader.deprel = 6;

        for (String f : filenames) {
            tree_reader.open(new FileInputStream(f));
            NLPNode[] words;
            while ((words = tree_reader.next()) != null) {
                for (NLPNode word : words)
                    System.out.print(word.getLemma() + " ");
                System.out.println();
            }
            tree_reader.close();
        }
        System.out.println("Finished");
    }

    @Test
    public void testRestart() throws Exception {

        List<String> filenames = FileUtils.getFileList("resources/dat/dep_test_files", "*");
        List<File> files = filenames.stream().map(File::new).collect(Collectors.toList());

        DEPTreeReader dr = new DEPTreeReader(files);

        List<Reader<NLPNode>> readers = dr.splitParallel(4);

        List<NLPNode> words;
        int i=0;
        for (Reader<NLPNode> r : readers) {
            System.out.println(i);
            while ((words = r.next()) != null) {
                for (NLPNode word : words)
                    System.out.print(word.getLemma() + " ");
                System.out.println();
            }
            r.restart();
            System.out.println(i);
            while ((words = r.next()) != null) {
                for (NLPNode word : words)
                    System.out.print(word.getLemma() + " ");
                System.out.println();
            }
            i++;
        }

        dr.close();
        System.out.println("Finished");
    }

    @Test
    public void testProgress() throws Exception
    {
        List<String> filenames = FileUtils.getFileList("resources/dat/dep_test_files", "*");
        List<File> files = filenames.stream().map(File::new).collect(Collectors.toList());

        Reader<String> reader = new DEPTreeReader(files).addFeature(NLPNode::getLemma);

        while (reader.next() != null)
        {
            System.out.print(String.format("%.1f", reader.progress()) + "%\n");
        }
        System.out.print(String.format("%.1f", reader.progress()) + "%\n");
        System.out.println("finished");
    }

}