from nltk import Tree
import cPickle
from zipfile import ZipFile
import numpy as np

train_tree_path = '../assignment3/trees/train.txt'
valid_tree_path = '../assignment3/trees/dev.txt'
test_tree_path = '../assignment3/trees/test.txt'

wv_path = '/mnt/hgfs/share/data/LM/glove.42B.300d.zip'
wv_filename = 'glove.42B.300d.txt'
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
    with open(path) as f:
        for line in f:
            t = Tree.fromstring(line)
            words = dfs_get_words(t)
            res.update(words)
    return res

all_words = set()
all_words.update(get_words(train_tree_path))
all_words.update(get_words(valid_tree_path))
all_words.update(get_words(test_tree_path))

found_words = set()
word2embed = {}
cnt = 0
with ZipFile(wv_path) as zf:
    with zf.open(wv_filename) as f_wv:
        for line in f_wv:
            cnt += 1
            if cnt % 5000 == 0:
                print cnt
            line = line.strip().split()
            w = line[0]
            if w in all_words:
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

for w in unfound_words:
    print w,

cPickle.dump((found_words, unfound_words), open(word_info_path, 'w'))
