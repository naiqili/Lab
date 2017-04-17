import collections, nltk
import logging
import cPickle
from zipfile import ZipFile
import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('log/preprocess.log'))
logging.basicConfig(level = logging.DEBUG, format = "%(asctime)s: %(name)s: %(levelname)s: %(message)s")

#wordvec_file = '/mnt/hgfs/share/data/LM/glove.42B.300d.zip'
wordvec_file = '/home/naiqi/data/LM/glove.42B.300d.zip'
unk_file = './_data/unk_file'
vocab_size = 50000

datafile = './_data/7000/nlu_data.pkl'
dictfile = './tmp/dict.pkl'
wordlist_file = './tmp/wordlist'
emb_file = './tmp/embedding.pkl'

all_data = cPickle.load(open(datafile))

maxlen = 0

for data in all_data:
    maxlen = max(maxlen, len(data['tok_text']))

word2ind = {}
ind2word = {}
wordlist = []
embedding = np.zeros((vocab_size+2, 300), dtype=np.float32)
with ZipFile(wordvec_file) as zf:
    with zf.open('glove.42B.300d.txt') as f:
        for idx in range(vocab_size):
            line = f.readline().strip()
            line = line.split()
            w = line[0]
            vec = map(float, line[1:])
            wordlist.append(w)
            word2ind[w] = idx+1
            ind2word[idx+1] = w
            embedding[idx+1] = vec

with open(wordlist_file, 'w') as f:
    for w in wordlist:
        f.write("%s\n" % w)
        
with open(unk_file) as f:
    line = f.readline().strip()
    line = line.split()
    vec = map(float, line[1:])
    word2ind['<unk>'] = vocab_size+1
    ind2word[vocab_size+1] = '<unk>'
    embedding[vocab_size+1] = vec

cPickle.dump((word2ind, ind2word), open(dictfile, 'w'))
cPickle.dump(embedding, open(emb_file, 'w'))

logger.debug('maxlen: %d' % maxlen)
logger.debug('vocab size: %d + 1' % len(word2ind))
for d in word2ind.items()[:10]:
    logger.debug(str(d))
for d in ind2word.items()[:10]:
    logger.debug(str(d))

print 'Done'
