import numpy as np
from scipy import spatial
from pprint import pprint

word2vec_file = '/mnt/hgfs/share/data/LM/glove.42B.300d.txt'

wv = {}
w_cnt = 20000

with open(word2vec_file) as f:
    for i in range(w_cnt):
        line = f.readline().strip().split()
        wv[line[0]] = map(float, line[1:])

def similar(word, wv, top=10):
    res = []
    for (w, vec) in wv.items():
        sim = 1-spatial.distance.cosine(wv[word], vec)
        res.append((w, sim))
    res = sorted(res, key=lambda d: d[1], reverse=True)
    return res[:top]

test_words = ['monday', 'happy', 'king', 'swim']

for word in test_words:
    t_res = similar(word, wv)
    pprint(t_res)
    print
