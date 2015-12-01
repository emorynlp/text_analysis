package edu.emory.mathcs.nlp.vsm.reader;

import edu.emory.mathcs.nlp.common.util.FileUtils;
import edu.emory.mathcs.nlp.text_analysis.word2vec.reader.DependencyReader;
import org.junit.Test;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by austin on 11/30/2015.
 */
public class DependencyReaderTest {

    @Test
    public void testRead() throws Exception {

        List<String> filenames = FileUtils.getFileList("src/test/resources/dat/dep_test_files","*");
        List<File> files = new ArrayList<File>();
        for(String f : filenames)
            files.add(new File(f));

        DependencyReader dr = new DependencyReader(files, DependencyReader.LEMMA_MODE);

        DependencyReader[] readers = dr.split(2);

        DependencyReader.DependencyWord[] words;
        for(DependencyReader r : readers){
            while((words = r.next())!=null){
                for(int i=0; i<words.length; i++)
                    System.out.print(words[i].toString()+" ");
                System.out.println();
            }
            System.out.println();
        }

        System.out.println("Finished");
    }

}
