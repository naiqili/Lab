import cPickle
from zipfile import ZipFile
import numpy as np

wv_path = '../../data/glove.840B.300d.zip'
wv_filename = 'glove.840B.300d.txt'
dict_path = '_data/dict.pkl'

output_path = 'tmp/embeddings.pkl'

(word2ind, ind2word) = cPickle.load(open(dict_path))

word_cnt = len(word2ind)
embed_mat = np.random.rand(word_cnt+1, 300)

all_words = set(word2ind.keys())

with ZipFile(wv_path) as zf:
    with zf.open(wv_filename) as f_wv:
        for line in f_wv:
            line = line.strip().split()
            w = line[0]
            if w in all_words:
                all_words.discard(w)
                vec = map(float, line[1:])
                embed_mat[word2ind[w]] = vec
                if len(all_words) == 0:
                    print "ALL FOUND!"
                    break

print "Unfound words:"
for w in all_words:
    print w

print embed_mat

cPickle.dump(embed_mat, open(output_path, 'w'))
