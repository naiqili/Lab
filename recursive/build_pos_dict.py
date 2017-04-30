import cPickle

tag_dict_path = 'tmp/pos_dict.pkl'
tag_path = '_data/tags.txt'

word2ind = {}
ind2word = {}

all_words = []
with open(tag_path) as f:
    f.readline()
    for line in f:
        all_words.append(line.strip())
    for (i, w) in enumerate(all_words):
        word2ind[w] = i
        ind2word[i] = w

cPickle.dump((word2ind, ind2word), open(tag_dict_path, 'w'))
