import collections, nltk
import logging
import cPickle
from ner_clean import ner_clean

logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('log/preprocess.log'))
logging.basicConfig(level = logging.DEBUG, format = "%(asctime)s: %(name)s: %(levelname)s: %(message)s")

meta_path = './_data/movie_titles_metadata.txt'
movie_lines_path = './_data/movie_lines.txt'
convers_path = './_data/movie_conversations.txt'

cnt_movie = 614
cnt_train = 3
cnt_dev = 4
cnt_test = 5

line_data = {}
with open(movie_lines_path) as f:
    for line in f:
        line = line.strip()
        try:
            line_id, _, _, _, text = line.split(' +++$+++ ')
            #line_data[line_id] = ner_clean(text)
            line_data[line_id] = text
        except:
            pass

convers_data = []
with open(convers_path) as f:
    for line in f:
        line = line.strip()
        _, _, movie_id, lst = line.split(' +++$+++ ')
        convers_data.append((movie_id, eval(lst)))


'''        
for _, lst in convers_data[:20]:
    print
    for line_id in lst:
        print line_data[line_id]
'''

train_triples = []
dev_triples = []
test_triples = []
for movie_id, lst in convers_data:
    cur_triples = None
    if len(lst) < 2:
        continue
    if int(movie_id[1:]) <= cnt_train:
        cur_triples = train_triples
    elif  int(movie_id[1:]) <= cnt_dev:
        cur_triples = dev_triples
    elif int(movie_id[1:]) <= cnt_test:
        cur_triples = test_triples
    for k in range(len(lst)-1):
        try:
            cur_triples.append((line_data[lst[k]], line_data[lst[k+1]]))
        except:
            pass

#print len(train_triples), len(dev_triples), len(test_triples)
logger.debug("train size: %d, dev size: %d, test size: %d" % (len(train_triples), len(dev_triples), len(test_triples)))

max_sent1_len = 0
max_sent2_len = 0
max_sent_len = 0

cnt = collections.Counter()
for sent1, sent2 in train_triples + dev_triples + test_triples:
    try:
        sent_lst1 = nltk.word_tokenize(sent1.lower()) 
        cnt.update(sent_lst1)
        sent_lst2 = nltk.word_tokenize(sent2.lower()) 
        cnt.update(sent_lst2)
        max_sent1_len = max(max_sent1_len, len(sent_lst1))
        max_sent2_len = max(max_sent2_len, len(sent_lst2))
        max_sent_len = max(max_sent_len, len(sent_lst1)+len(sent_lst2))
    except:
        pass

logger.debug("max_sent1_len: %d, max_sent2_len: %d, max_sent_len: %d" % (max_sent1_len, max_sent2_len, max_sent_len))

cnt.update(['<START>', '<END>'])

words = list(cnt.keys())
word2ind = dict(zip(words, range(1,len(words)+1)))
ind2word = dict([(ind, w) for (w, ind) in word2ind.items()])

logger.debug("dict size: %d" % len(word2ind))

'''
print len(word2ind), len(ind2word)
print word2ind.items()[:20]
print ind2word.items()[:20]
'''

traindata = []
devdata = []
testdata = []

for sent1, sent2 in train_triples:
    try:
        sent1_ind = [word2ind[w] for w in nltk.word_tokenize(sent1.lower())]
        sent2_ind = [word2ind[w] for w in nltk.word_tokenize(sent2.lower())]
        traindata.append(([word2ind['<START>']] + sent1_ind + [word2ind['<END>']], \
                          [word2ind['<START>']] + sent2_ind + [word2ind['<END>']]))
    except:
        pass

for sent1, sent2 in dev_triples:
    try:
        sent1_ind = [word2ind[w] for w in nltk.word_tokenize(sent1.lower())]
        sent2_ind = [word2ind[w] for w in nltk.word_tokenize(sent2.lower())]
        devdata.append(([word2ind['<START>']] + sent1_ind + [word2ind['<END>']], \
                        [word2ind['<START>']] + sent2_ind + [word2ind['<END>']]))
    except:
        pass

for sent1, sent2 in test_triples:
    try:
        sent1_ind = [word2ind[w] for w in nltk.word_tokenize(sent1.lower())]
        sent2_ind = [word2ind[w] for w in nltk.word_tokenize(sent2.lower())]
        testdata.append(([word2ind['<START>']] + sent1_ind + [word2ind['<END>']], \
                         [word2ind['<START>']] + sent2_ind + [word2ind['<END>']]))
    except:
        pass

#cPickle.dump((word2ind, ind2word), open('./tmp/dict.pkl', 'w'))
cPickle.dump(traindata, open('./tmp/toy_traindata.pkl', 'w'))
cPickle.dump(devdata, open('./tmp/toy_devdata.pkl', 'w'))
cPickle.dump(testdata, open('./tmp/toy_testdata.pkl', 'w'))
