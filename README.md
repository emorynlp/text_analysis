# edu.emory.mathcs.nlp.vsm

Vector space models.

## Word2Vec

* Replication of the original c version.
* Preserves long strings.
* Words are deliminated by `' '` and `'\n'`. 
* Multiple training files 

* Preprocess data where each word is delimited by ' ' and sentence is delimited by '\n'.

#Word2Vec.java Usage
#####Enviornment Details  
* Make sure not to run with the 32bit JDK, it limits heap size.
* Other Java implementations of word2vec reccomend around 10g heap size. This implementation needs more than that (details to come).  
To set these options from maven, use:  
```
set MAVENOPTS=-Xmx<NUM_GIGS>g -d64
```
where NUM_GIGS is the number of gigabytes you want to allocate for the heap.

#####Command Line parameters, (each prepended with a tag '-'), most take additional arguments
* train: "path to the training file or the directory containig the training files.  
* output: "output file to save the resulting word vectors.  
* ext: "extension of the training files (default: \"*\").  
* size: "size of word vectors (default: 100).  
* window: "max-window of contextual words (default: 5).  
* sample: "threshold for occurrence of words (default: 1e-3). Those that appear with higher frequency in the training data will be randomly down-sampled.  
* negative: "number of negative examples (default: 5; common values are 3 - 10). If negative = 0, use Hierarchical Softmax instead of Negative Sampling.  
* threads: "number of threads (default: 12).  
* iter: "number of training iterations (default: 5).  
* min-count: "min-count of words (default: 5). This will discard words that appear less than <int> times.  
* alpha: "initial learning rate (default: 0.025 for skip-gram; use 0.05 for CBOW).  
* binary: "If set, save the resulting vectors in binary moded.  
* cbow: "If set, use the continuous bag-of-words model instead of the skip-gram model.
