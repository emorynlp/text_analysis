package edu.emory.mathcs.nlp.text_analysis.word2vec.reader;

import edu.emory.mathcs.nlp.component.template.node.NLPNode;
import edu.emory.mathcs.nlp.component.template.reader.TSVReader;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * @author Austin Blodgett
 */
public class DependencyReader extends Reader<NLPNode>
{

    TSVReader tree_reader;

    private static final Pattern sentence_break = Pattern.compile("\\n\\s*\\n");

    public DependencyReader(List<File> files)
    {
        super(files, sentence_break);
    }

    protected DependencyReader(DependencyReader reader, long start, long end)
    {
        super(reader, start, end);
    }

    @Override
    public List<NLPNode> next() throws IOException
    {
        NLPNode[] words = tree_reader.next();

        return words==null ? null : Arrays.stream(words).collect(Collectors.toList());
    }

    @Override
    protected DependencyReader subReader(long start, long end)
    {
        return new DependencyReader(this, start, end);
    }

    @Override
    public void restart() throws IOException {
        super.restart();
        tree_reader = new TSVReader();
        tree_reader.open(this);

        tree_reader.form = 1;
        tree_reader.lemma = 2;
        tree_reader.pos = 3;
        tree_reader.dhead = 5;
        tree_reader.deprel = 6;
    }
}
