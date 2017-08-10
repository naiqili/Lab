from nltk import Tree
import cPickle, codecs
from zipfile import ZipFile
import numpy as np
from collections import defaultdict

train_tree_path = '../assignment3/trees/train.txt'
valid_tree_path = '../assignment3/trees/dev.txt'
test_tree_path = '../assignment3/trees/test.txt'

#wv_path = '/mnt/hgfs/share/data/LM/glove.42B.300d.zip'
#wv_path = '../../data/glove.42B.300d.zip'
#wv_filename = 'glove.42B.300d.txt'
wv_path = '../../data/glove.840B.300d.zip'
wv_filename = 'glove.840B.300d.txt'
wv_size = 300

output_path = 'tmp/words.txt'
dict_path = '_data/dict.pkl'
embed_path = 'tmp/embeddings.pkl'
word_info_path = 'tmp/word_info.pkl'

def dfs_get_words(t):
    if isinstance(t, Tree) and len(t) >= 2:
        return set.union(dfs_get_words(t[0]), dfs_get_words(t[1]))
    else:
        return set([t[0].lower()])

def get_words(path):
    res = set()
    with codecs.open(path, encoding='utf-8') as f:
        for line in f:
            t = Tree.fromstring(line)
            words = dfs_get_words(t)
            res.update(words)
    return res

all_words = set()
train_words = get_words(train_tree_path)
valid_words = get_words(valid_tree_path)
test_words = get_words(test_tree_path)
all_words.update(train_words)
all_words.update(valid_words)
all_words.update(test_words)

found_words = set()
word2embed = {}
cnt = 0
with ZipFile(wv_path) as zf:
    with zf.open(wv_filename) as f_wv:
        for line in f_wv:
            line = codecs.decode(line, 'utf-8')
            cnt += 1
            if cnt % 5000 == 0:
                print cnt
            line = line.strip().split()
            w = line[0]
            w = w.lower()
            if w in all_words and w not in found_words:
                found_words.add(w)
                vec = map(float, line[1:])
                word2embed[w] = vec

embed_mat = np.random.rand(len(found_words)+1, wv_size)
word2ind = {}
ind2word = {}
for (ind, word) in enumerate(found_words):
    word2ind[word] = ind+1
    ind2word[ind+1] = word
    embed_mat[ind+1] = word2embed[word]

cPickle.dump((word2ind, ind2word), open(dict_path, 'w'))
cPickle.dump(embed_mat, open(embed_path, 'w'))

unfound_words = all_words.difference(found_words)

print "Found word size:", len(found_words)
print "Unfound word size:", len(unfound_words)

uf_train, uf_valid, uf_test = defaultdict(int), defaultdict(int), defaultdict(int)

for w_set, uf_dict in [(train_words, uf_train), (valid_words, uf_valid), (test_words, uf_test)]:
    for w in w_set:
        if w in unfound_words:
            uf_dict[w] += 1

for w in unfound_words:
    print w, uf_train[w], uf_valid[w], uf_test[w]

cPickle.dump((found_words, unfound_words), open(word_info_path, 'w'))
