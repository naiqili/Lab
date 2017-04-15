import collections, nltk
import logging
import cPickle

logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('log/preprocess.log'))
logging.basicConfig(level = logging.DEBUG, format = "%(asctime)s: %(name)s: %(levelname)s: %(message)s")

datafile = './_data/7000/nlu_data.pkl'
dictfile = './tmp/dict7000.pkl'

all_data = cPickle.load(open(datafile))

maxlen = 0
all_words = set()
for data in all_data:
    maxlen = max(maxlen, len(data['tok_text']))
    all_words = all_words.union(set(data['tok_text']))

word2ind = {}
ind2word = {}
for idx, w in enumerate(all_words):
    word2ind[w] = idx+1
    ind2word[idx+1] = w

logger.debug('maxlen: %d' % maxlen)
logger.debug('vocab size: %d + 1' % len(word2ind))
for d in word2ind.items()[:10]:
    logger.debug(str(d))
for d in ind2word.items()[:10]:
    logger.debug(str(d))

cPickle.dump((word2ind, ind2word), open(dictfile, 'w'))
print 'Done'
