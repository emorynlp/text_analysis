package edu.emory.mathcs.nlp.vsm.reader;

import edu.emory.mathcs.nlp.common.util.FileUtils;
import edu.emory.mathcs.nlp.text_analysis.word2vec.reader.Reader;
import edu.emory.mathcs.nlp.text_analysis.word2vec.reader.SentenceReader;
import org.junit.Test;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by austin on 12/24/2015.
 */
public class SentenceReaderTest {


    @Test
    public void testRead() throws Exception
    {
        List<String> filenames = FileUtils.getFileList("src/test/resources/dat/test_files", "*");
        List<File> files = filenames.stream().map(File::new).collect(Collectors.toList());

        List<Reader<String>> readers = new SentenceReader(files)
                .splitParallel(4);

        StringBuilder sb1 = new StringBuilder();
        int c;
        for (Reader<String> r : readers) {
            while ((c = r.read()) != -1)
                sb1.append((char) c);
        }

        InputStream in;

        StringBuilder sb2 = new StringBuilder();
        for (String f : filenames)
        {
            in = new FileInputStream(f);
            while((c = in.read()) != -1)
                sb2.append((char)c);
            in.close();
        }

        assert(sb1.toString().equals(sb2.toString()));
    }

    @Test
    public void testNext() throws Exception
    {
        Reader<String> reader = new SentenceReader(FileUtils.getFileList("src/test/resources/dat/test_files","*")
                                                    .stream().map(File::new).collect(Collectors.toList()));

        StringBuilder sb = new StringBuilder();

        List<String> words;
        while((words = reader.next()) != null)
        {
            for (String w : words)
                sb.append(w).append(" ");
            sb.append("\n");
        }
        System.out.println(sb);
    }

    @Test
    public void testParallel() throws Exception
    {
        Reader<String> reader = new SentenceReader(FileUtils.getFileList("src/test/resources/dat/test_files","*")
                                                 .stream().map(File::new).collect(Collectors.toList()));

        List<String> words;
        for (Reader<String> r : reader.splitParallel(2))
        {
            StringBuilder sb = new StringBuilder();
            while ((words = r.next()) != null)
            {
                for (String w : words)
                    sb.append(w).append(" ");
                sb.append("\n");
            }
            System.out.println(sb);
        }
    }

    @Test
    public void testTrainAndTest() throws Exception
    {
        Reader<String> reader = new SentenceReader(FileUtils.getFileList("src/test/resources/dat/test_files","*")
                .stream().map(File::new).collect(Collectors.toList()));

        List<String> words;
        for (Reader<String> r : reader.splitTrainAndTest(0.8f))
        {
            StringBuilder sb = new StringBuilder();
            while ((words = r.next()) != null)
            {
                for (String w : words)
                    sb.append(w).append(" ");
                sb.append("\n");
            }
            System.out.println(sb);
        }
    }

    @Test
    public void testProgress() throws Exception
    {
        Reader<String> reader = new SentenceReader(FileUtils.getFileList("src/test/resources/dat/test_files","*")
                .stream().map(File::new).collect(Collectors.toList()));

        List<String> words;
        for (Reader<String> r : reader.splitParallel(2))
        {
            while ((words = r.next()) != null)
            {
                for (String w : words)
                    System.out.print(w+" ");
                System.out.println();
                System.out.println(r.progress());
            }
            System.out.println();
        }
    }

    @Test
    public void testRestart() throws Exception
    {
        Reader<String> reader = new SentenceReader(FileUtils.getFileList("src/test/resources/dat/test_files","*")
                .stream().map(File::new).collect(Collectors.toList()));

        while(reader.next() != null);

        // restart
        reader.restart();

        StringBuilder sb = new StringBuilder();
        List<String> words;
        while((words = reader.next()) != null)
        {
            for (String w : words)
                sb.append(w).append(" ");
            sb.append("\n");
        }
        System.out.println(sb);
    }

    @Test
    public void testAddFilter() throws Exception
    {
        Reader<String> reader = new SentenceReader(FileUtils.getFileList("src/test/resources/dat/test_files","*")
                                            .stream().map(File::new).collect(Collectors.toList()))
                                            .addFilter(w -> w.contains("o"));

        StringBuilder sb = new StringBuilder();
        List<String> words;
        while((words = reader.next()) != null)
        {
            for (String w : words)
                sb.append(w).append(" ");
            sb.append("\n");
        }
        System.out.println(sb);
    }

    @Test
    public void testAddFeature() throws Exception
    {
        Reader<String> reader = new SentenceReader(FileUtils.getFileList("src/test/resources/dat/test_files","*")
                                        .stream().map(File::new).collect(Collectors.toList()))
                                        .addFeature(String::toUpperCase)
                                        .addFeature(String::toLowerCase);

        StringBuilder sb = new StringBuilder();
        List<String> words;
        while((words = reader.next()) != null)
        {
            for (String w : words)
                sb.append(w).append(" ");
            sb.append("\n");
        }
        System.out.println(sb);
    }

    @Test
    public void testAddFeature1() throws Exception
    {
        Reader<Integer> reader = new SentenceReader(FileUtils.getFileList("src/test/resources/dat/test_files","*")
                                            .stream().map(File::new).collect(Collectors.toList()))
                                            .addFeature(String::hashCode);

        StringBuilder sb = new StringBuilder();
        List<Integer> words;
        while((words = reader.next()) != null)
        {
            for (Integer w : words)
                sb.append(w).append(" ");
            sb.append("\n");
        }
        System.out.println(sb);
    }

    @Test
    public void testAddMap() throws Exception
    {
        Reader<String> reader = new SentenceReader(FileUtils.getFileList("src/test/resources/dat/test_files","*")
                                            .stream().map(File::new).collect(Collectors.toList()))
                                            .addMap(l -> l.subList(0, l.size()/2));

        StringBuilder sb = new StringBuilder();

        List<String> words;
        while((words = reader.next()) != null)
        {
            for (String w : words)
                sb.append(w).append(" ");
            sb.append("\n");
        }
        System.out.println(sb);
    }
}