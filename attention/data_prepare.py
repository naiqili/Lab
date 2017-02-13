import cPickle
from pprint import pprint
import logging

logger = logging.getLogger('data_prepare')
logger.addHandler(logging.FileHandler('./log/data_prepare.log'))
logging.basicConfig(level = logging.INFO,
                    format = "%(asctime)s: %(name)s: %(levelname)s: %(message)s")

dic_output = 'tmp/dic.pkl'
log_output = 'tmp/data_prepare.log'

word2ind = {}
ind2word = {}

acttype = cPickle.load(open('tmp/acttype_cnt.pkl'))
train_data = cPickle.load(open('tmp/train_data.pkl'))
dev_data = cPickle.load(open('tmp/dev_data.pkl'))

all_words = acttype.keys()

train_data_full = []
dev_data_full = []

train_data_coded = []
dev_data_coded = []


acttype_len = 0
for act in acttype:
    acttype_len = acttype_len + max(1, acttype[act])
logger.info("acttype full len: %d" % acttype_len)

for (data, trans) in train_data:
    abstract_list = []
    natural_list = []
    for act in acttype:
        act_len = max(acttype[act], 1)
        s = data[act].split()
        all_words = all_words + s
        while len(s) < act_len:
            s.append('<NULL>')
        abstract_list = abstract_list + s
    natural_list = ['<START>'] + trans.split() + ['<END>']
    all_words = all_words + trans.split()
    all_words = list(set(all_words))
    train_data_full.append((abstract_list, natural_list))

logger.info("Train data size: %d", len(train_data))

for (data, trans) in dev_data:
    abstract_list = []
    natural_list = []
    for act in acttype:
        act_len = max(acttype[act], 1)
        s = data[act].split()
        all_words = all_words + s
        while len(s) < act_len:
            s.append('<NULL>')
        abstract_list = abstract_list + s
    natural_list = ['<START>'] + trans.split() + ['<END>']
    all_words = all_words + trans.split()
    all_words = list(set(all_words))
    dev_data_full.append((abstract_list, natural_list))

logger.info("Dev data size: %d", len(dev_data))

all_words = ['<END>', '<START>', '<YES>', '<NO>', '<NULL>'] + list(set(all_words))
for (ind, w) in enumerate(all_words):
    word2ind[w] = ind
    ind2word[ind] = w

for (data, trans) in train_data_full:
    (data_coded, trans_coded) = ([word2ind[w] for w in data],
                                 [word2ind[w] for w in trans])
    train_data_coded.append((data_coded, trans_coded))

for (data, trans) in dev_data_full:
    (data_coded, trans_coded) = ([word2ind[w] for w in data],
                                 [word2ind[w] for w in trans])
    dev_data_coded.append((data_coded, trans_coded))

cPickle.dump((word2ind, ind2word), open('tmp/dict.pkl', 'w'))
cPickle.dump(train_data_full, open('tmp/train_data_full.pkl', 'w'))
cPickle.dump(dev_data_full, open('tmp/dev_data_full.pkl', 'w'))
cPickle.dump(train_data_coded, open('tmp/train_data_coded.pkl', 'w'))
cPickle.dump(dev_data_coded, open('tmp/dev_data_coded.pkl', 'w'))

logger.info('dict examples')
for (k,v) in list(word2ind.items())[:10] + list(ind2word.items())[:10]:
    logger.info((k,v))
logger.info('train_data_full examples')
for (data, trans) in train_data_full[:5]:
    logger.info((data, trans))
logger.info('dev_data_full examples')
for (data, trans) in dev_data_full[:5]:
    logger.info((data, trans))
logger.info('train_data_coded examples')
for (data, trans) in train_data_coded[:5]:
    logger.info((data, trans))
logger.info('dev_data_coded examples')
for (data, trans) in dev_data_coded[:5]:
    logger.info((data, trans))

logger.info('FINISH.')
