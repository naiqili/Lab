import os, codecs, zipfile, config, cPickle
from os import path
import numpy as np
from nltk.tokenize import sexpr
from six.moves import urllib

def download_and_unzip(data_dir, url_base, zip_name, *file_names):
    zip_path = os.path.join(data_dir, zip_name)
    url = url_base + zip_name
    out_paths = []
    if not os.path.exists(zip_path):
        print('downloading %s to %s' % (url, zip_path))
        urllib.request.urlretrieve(url, zip_path)
    else:
        print('already downloaded: %s' % zip_path)
    with zipfile.ZipFile(zip_path, 'r') as f:
        for file_name in file_names:
            if not os.path.exists(os.path.join(data_dir, file_name)):
                print('extracting %s' % file_name)
                out_paths.append(f.extract(file_name, path=data_dir))      
            else:
                out_paths.append(path.join(data_dir, file_name))
                print('already extracted: %s' % path.join(data_dir, file_name))
    return out_paths

def load_embeddings():
    """Loads embedings, returns weight matrix and dict from words to indices."""
    if path.exists(config.EMBEDDING_MAT_PATH) and path.exists(config.WORD_IDX_PATH):
        weight_vectors = cPickle.load(open(config.EMBEDDING_MAT_PATH))
        word_idx = cPickle.load(open(config.WORD_IDX_PATH))
        return weight_vectors, word_idx
    embedding_path = config.FILTER_GLOVE_PATH
    print('loading word embeddings from %s' % embedding_path)
    weight_vectors = []
    word_idx = {}
    with codecs.open(embedding_path, encoding='utf-8') as f:
        for line in f:
            word, vec = line.split(u' ', 1)
            word_idx[word] = len(weight_vectors)
            weight_vectors.append(np.array(vec.split(), dtype=np.float32))
    # Annoying implementation detail; '(' and ')' are replaced by '-LRB-' and
    # '-RRB-' respectively in the parse-trees.
    word_idx[u'-LRB-'] = word_idx.pop(u'(')
    word_idx[u'-RRB-'] = word_idx.pop(u')')
    # Random embedding vector for unknown words.
    weight_vectors.append(np.random.uniform(
        -0.05, 0.05, weight_vectors[0].shape).astype(np.float32))
    cPickle.dump(np.stack(weight_vectors), open(config.EMBEDDING_MAT_PATH, 'w'))
    cPickle.dump(word_idx, open(config.WORD_IDX_PATH, 'w'))                 
    return np.stack(weight_vectors), word_idx

def load_trees(filename):
    with codecs.open(filename, encoding='utf-8') as f:
        # Drop the trailing newline and strip \s.
        trees = [line.strip().replace('\\', '') for line in f]
        print('loaded %s trees from %s' % (len(trees), filename))
    return trees

def tokenize(s):
    label, phrase = s[1:-1].split(None, 1)
    return label, sexpr.sexpr_tokenize(phrase)

def filter_glove():
    vocab = set()
    filter_glove_path = config.FILTER_GLOVE_PATH
    # Download the full set of unlabeled sentences separated by '|'.
    sentence_path, = download_and_unzip(
        config.DATA_DIR, 'http://nlp.stanford.edu/~socherr/', 'stanfordSentimentTreebank.zip', 
        'stanfordSentimentTreebank/SOStr.txt')
    with codecs.open(sentence_path, encoding='utf-8') as f:
        for line in f:
            # Drop the trailing newline and strip backslashes. Split into words.
            vocab.update(line.strip().replace('\\', '').split('|'))
    nread = 0
    nwrote = 0
    with codecs.open(config.GLOVE_PATH, encoding='utf-8') as f:
        with codecs.open(config.FILTER_GLOVE_PATH, 'w', encoding='utf-8') as out:
            for line in f:
                nread += 1
                line = line.strip()
                if not line: continue
                if line.split(u' ', 1)[0] in vocab:
                    out.write(line + '\n')
                    nwrote += 1
    print('read %s lines, wrote %s' % (nread, nwrote))

def data_prepare():
    data_dir = config.DATA_DIR
    full_glove_path, = download_and_unzip(
        data_dir,
        'http://nlp.stanford.edu/data/', 'glove.840B.300d.zip',
        'glove.840B.300d.txt')
    train_path, dev_path, test_path = download_and_unzip(
        data_dir,
        'http://nlp.stanford.edu/sentiment/', 'trainDevTestTrees_PTB.zip', 
        config.TRAIN_FILENAME, config.VALID_FILENAME, config.TEST_FILENAME)
    filter_glove()
    load_embeddings()
    
if __name__=='__main__':
    data_prepare()