package edu.emory.mathcs.nlp.vsm.reader;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.List;
import java.util.stream.Collectors;

import org.junit.Test;

import edu.emory.mathcs.nlp.common.util.FileUtils;

/**
 * Created by austin on 12/24/2015.
 */
public class SentenceReaderTest {

	@Test
    public void testRead() throws Exception
    {
        List<String> filenames = FileUtils.getFileList("resources/dat/test_files", "*");
        List<File> files = filenames.stream().map(File::new).collect(Collectors.toList());

        @SuppressWarnings("resource")
		List<Reader<String>> readers = new SentenceReader(files).splitParallel(4);

        StringBuilder sb1 = new StringBuilder();
        int c;
        for (Reader<String> r : readers) {
            r.open();
            while ((c = r.read()) != -1)
                sb1.append((char) c);
            r.close();
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

        // test restart
        StringBuilder sb3 = new StringBuilder();
        for (Reader<String> r : readers) {
            r.open();
            while ((c = r.read()) != -1)
                sb3.append((char) c);
            r.close();
        }

        System.out.println(sb1.toString());
        System.out.println(sb2.toString());
        System.out.println(sb3.toString());


        assert(sb3.toString().equals(sb2.toString()));
    }

    @Test
    public void testNext() throws Exception
    {
        Reader<String> reader = new SentenceReader(FileUtils.getFileList("resources/dat/test_files","*")
                                                    .stream().map(File::new).collect(Collectors.toList()));

        StringBuilder sb = new StringBuilder();

        reader.open();

        List<String> words;
        while((words = reader.next()) != null)
        {
            for (String w : words)
                sb.append(w).append(" ");
            sb.append("\n");
        }
        System.out.println(sb);

        reader.close();
    }

    @Test
    public void testParallel() throws Exception
    {
        Reader<String> reader = new SentenceReader(FileUtils.getFileList("resources/dat/test_files","*")
                                                 .stream().map(File::new).collect(Collectors.toList()));

        List<String> words;
        for (Reader<String> r : reader.splitParallel(2))
        {
            r.open();
            StringBuilder sb = new StringBuilder();
            while ((words = r.next()) != null)
            {
                for (String w : words)
                    sb.append(w).append(" ");
                sb.append("\n");
            }
            System.out.println(sb);
            r.close();
        }
    }

    @Test
    public void testTrainAndTest() throws Exception
    {
        Reader<String> reader = new SentenceReader(FileUtils.getFileList("resources/dat/test_files","*")
                .stream().map(File::new).collect(Collectors.toList()));

        List<String> words;
        for (Reader<String> r : reader.splitTrainAndTest(0.8f))
        {
            r.open();
            StringBuilder sb = new StringBuilder();
            while ((words = r.next()) != null)
            {
                for (String w : words)
                    sb.append(w).append(" ");
                sb.append("\n");
            }
            System.out.println(sb);
            r.close();
        }
    }

    @Test
    public void testProgress() throws Exception
    {
        Reader<String> reader = new SentenceReader(FileUtils.getFileList("resources/dat/test_files","*")
                .stream().map(File::new).collect(Collectors.toList()));

        List<String> words;
        for (Reader<String> r : reader.splitParallel(2))
        {
            r.open();
            while ((words = r.next()) != null)
            {
                for (String w : words)
                    System.out.print(w+" ");
                System.out.println();
                System.out.println(r.progress());
            }
            System.out.println();
            r.close();
        }
    }

    @Test
    public void testRestart() throws Exception
    {
        Reader<String> reader = new SentenceReader(FileUtils.getFileList("resources/dat/test_files","*")
                .stream().map(File::new).collect(Collectors.toList()));

        reader.open();
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
        reader.close();
    }

    @Test
    public void testAddFilter() throws Exception
    {
        @SuppressWarnings("resource")
        Reader<String> reader = new SentenceReader(FileUtils.getFileList("resources/dat/test_files","*")
                                            .stream().map(File::new).collect(Collectors.toList()))
                                            .addFilter(w -> w.contains("o"));

        reader.open();

        StringBuilder sb = new StringBuilder();
        List<String> words;
        while((words = reader.next()) != null)
        {
            for (String w : words)
                sb.append(w).append(" ");
            sb.append("\n");
        }
        System.out.println(sb);

        reader.close();
    }

    @Test
    public void testAddFeature() throws Exception
    {
        @SuppressWarnings("resource")
        Reader<String> reader = new SentenceReader(FileUtils.getFileList("resources/dat/test_files","*")
                                        .stream().map(File::new).collect(Collectors.toList()))
                                        .addFeature(String::toUpperCase)
                                        .addFeature(String::toLowerCase);

        reader.open();

        StringBuilder sb = new StringBuilder();
        List<String> words;
        while((words = reader.next()) != null)
        {
            for (String w : words)
                sb.append(w).append(" ");
            sb.append("\n");
        }
        System.out.println(sb);

        reader.close();
    }

    @Test
    public void testAddFeature1() throws Exception
    {
    	@SuppressWarnings("resource")
        Reader<Integer> reader = new SentenceReader(FileUtils.getFileList("resources/dat/test_files","*")
                                            .stream().map(File::new).collect(Collectors.toList()))
                                            .addFeature(String::hashCode);

        reader.open();

        StringBuilder sb = new StringBuilder();
        List<Integer> words;
        while((words = reader.next()) != null)
        {
            for (Integer w : words)
                sb.append(w).append(" ");
            sb.append("\n");
        }
        System.out.println(sb);

        reader.close();
    }

    @Test
    public void testAddMap() throws Exception
    {
    	@SuppressWarnings("resource")
        Reader<String> reader = new SentenceReader(FileUtils.getFileList("resources/dat/test_files","*")
                                            .stream().map(File::new).collect(Collectors.toList()))
                                            .addMap(l -> l.subList(0, l.size()/2));

        reader.open();

        StringBuilder sb = new StringBuilder();

        List<String> words;
        while((words = reader.next()) != null)
        {
            for (String w : words)
                sb.append(w).append(" ");
            sb.append("\n");
        }
        System.out.println(sb);
        reader.close();
    }
}