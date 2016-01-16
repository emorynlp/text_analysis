package edu.emory.mathcs.nlp.vsm.types;

import edu.emory.mathcs.nlp.common.constant.StringConst;
import edu.emory.mathcs.nlp.common.util.FileUtils;
import edu.emory.mathcs.nlp.text_analysis.word2vec.reader.Reader;
import edu.emory.mathcs.nlp.text_analysis.word2vec.reader.SentenceReader;
import edu.emory.mathcs.nlp.text_analysis.word2vec.util.Vocabulary;
import edu.emory.mathcs.nlp.text_analysis.word2vec.util.Word;
import org.junit.Test;

import java.io.File;
import java.util.Arrays;
import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author Austin Blodgett
 */
public class VocabularyTest {

    @Test
    public void testVocabulary()
    {
        Vocabulary vocab = new Vocabulary();
        Word w, w0, w1, w2;

        w0 = vocab.add("A");
        w1 = vocab.add("B"); vocab.add("B");
        w2 = vocab.add("C");

        assertEquals(w0, vocab.get(0));
        assertEquals(w1, vocab.get(1));
        assertEquals(w2, vocab.get(2));
        assertEquals(0, vocab.indexOf("A"));
        assertEquals(1, vocab.indexOf("B"));
        assertEquals(2, vocab.indexOf("C"));
        assertEquals("A:1 B:2 C:1", vocab.toString());

        vocab.sort(0);
        assertEquals("B:2 A:1 C:1", vocab.toString());

        vocab.add("C"); vocab.add("D");
        assertEquals(1, vocab.indexOf("A"));
        assertEquals(0, vocab.indexOf("B"));
        assertEquals(2, vocab.indexOf("C"));
        assertEquals(3, vocab.indexOf("D"));
        assertEquals("B:2 A:1 C:2 D:1", vocab.toString());

        vocab.reduce();
        assertTrue(vocab.indexOf("A") < 0);
        assertEquals(0, vocab.indexOf("B"));
        assertEquals(1, vocab.indexOf("C"));
        assertEquals("B:2 C:2", vocab.toString());

        vocab.add("A"); vocab.add("D");
        assertEquals("B:2 C:2 A:1 D:1", vocab.toString());

//		      6
//		     / \
//		    0   5
//		       / \
//		      4   1
//		     / \
//		    3   2
        vocab.generateHuffmanCodes();
        w = vocab.get(0);	assertEquals("[0]"      , Arrays.toString(w.code));	assertEquals("[2]"      , Arrays.toString(w.point));
        w = vocab.get(1);	assertEquals("[1, 1]"   , Arrays.toString(w.code));	assertEquals("[2, 1]"   , Arrays.toString(w.point));
        w = vocab.get(2);	assertEquals("[1, 0, 1]", Arrays.toString(w.code));	assertEquals("[2, 1, 0]", Arrays.toString(w.point));
        w = vocab.get(3);	assertEquals("[1, 0, 0]", Arrays.toString(w.code));	assertEquals("[2, 1, 0]", Arrays.toString(w.point));

        vocab.add("C"); vocab.add("D"); vocab.add("E"); vocab.add("D");
        assertEquals("B:2 C:3 A:1 D:3 E:1", vocab.toString());

        vocab.sort(3);
        assertEquals("C:3 D:3", vocab.toString());

        vocab.reduce();
        assertEquals("C:3 D:3", vocab.toString());

        vocab.reduce();
        assertEquals(StringConst.EMPTY, vocab.toString());

//		assertEquals(0, vocab.indexOf("B"));
//		assertEquals(1, vocab.indexOf("A"));
//		assertEquals(Vocabulary.OOV, vocab.indexOf("C"));
//		assertEquals("B:2 C:3 A:1 D:3 E:1", vocab.toString());
//
//		vocab.add("A"); vocab.add("C");
//		assertEquals(1, vocab.indexOf("A"));
//		assertEquals(0, vocab.indexOf("B"));
//		assertEquals(2, vocab.indexOf("C"));
//		assertEquals("B:2 A:3 C:1", vocab.toString());
//
//		vocab.reduce();
//		assertEquals(1, vocab.indexOf("A"));
//		assertEquals(0, vocab.indexOf("B"));
//		assertEquals(Vocabulary.OOV, vocab.indexOf("C"));
//		assertEquals("B:2 A:3", vocab.toString());
//
//		vocab.reduce();
//		assertEquals("A:3", vocab.toString());
    }

    @Test
    public void testLearn() throws Exception
    {
        Reader<String> reader = new SentenceReader(FileUtils.getFileList("src/test/resources/dat/test_files","*")
                .stream().map(File::new).collect(Collectors.toList()));
        Vocabulary vocab = new Vocabulary();

        vocab.learn(reader, 2);

        System.out.println(vocab.toString());
    }

    @Test
    public void testLearnParallel() throws Exception
    {
        Reader<String> reader = new SentenceReader(FileUtils.getFileList("src/test/resources/dat/test_files","*")
                                        .stream().map(File::new).collect(Collectors.toList()));
        Vocabulary vocab = new Vocabulary();

        vocab.learnParallel(reader.splitParallel(3), 2);

        System.out.println(vocab.toString());
    }

    @Test
    public void testAddAll() throws Exception
    {
        Reader<String> reader = new SentenceReader(FileUtils.getFileList("src/test/resources/dat/test_files","*")
                                        .stream().map(File::new).collect(Collectors.toList()));
        Vocabulary vocab1 = new Vocabulary();

        vocab1.learn(reader, 2);

        Vocabulary vocab2 = new Vocabulary();
        vocab2.addAll(vocab1);

        System.out.println(vocab2.toString());
    }

    @Test
    public void testWriteVocab() throws Exception
    {
        Reader<String> reader = new SentenceReader(FileUtils.getFileList("src/test/resources/dat/test_files","*")
                .stream().map(File::new).collect(Collectors.toList()));
        Vocabulary vocab = new Vocabulary();

        vocab.learn(reader);
        vocab.writeVocab(new File("src/test/resources/dat/vocab"));
    }

    @Test
    public void testReadVocab() throws Exception
    {
        Vocabulary vocab = new Vocabulary();
        vocab.readVocab(new File("src/test/resources/dat/vocab"));

        System.out.println(vocab);
    }

}