import cPickle
from pprint import pprint
import logging

logger = logging.getLogger('data_prepare')
logger.addHandler(logging.FileHandler('./log/simple_data_prepare.log'))
logging.basicConfig(level = logging.INFO,
                    format = "%(asctime)s: %(name)s: %(levelname)s: %(message)s")

dic_output = 'tmp/dic.pkl'

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


simple_acttype_len = len(acttype)
logger.info("acttype simple len: %d" % simple_acttype_len)

for (data, trans) in train_data:
    coded_abstract_list = []
    natural_list = []
    for act in acttype:
        act_len = max(acttype[act], 1)
        s = data[act].split()
        if s[0] == '<NO>':
            coded_abstract_list.append(0)
        else:
            coded_abstract_list.append(1)
    natural_list = ['<START>'] + trans.split() + ['<END>']
    all_words = all_words + trans.split()
    all_words = list(set(all_words))
    train_data_full.append((coded_abstract_list, natural_list))

logger.info("Train data size: %d", len(train_data))

for (data, trans) in dev_data:
    coded_abstract_list = []
    natural_list = []
    for act in acttype:
        act_len = max(acttype[act], 1)
        s = data[act].split()
        if s[0] == '<NO>':
            coded_abstract_list.append(0)
        else:
            coded_abstract_list.append(1)
    natural_list = ['<START>'] + trans.split() + ['<END>']
    all_words = all_words + trans.split()
    all_words = list(set(all_words))
    dev_data_full.append((coded_abstract_list, natural_list))

logger.info("Dev data size: %d", len(dev_data))

(word2ind, ind2word) = cPickle.load(open('tmp/dict.pkl'))

for (data, trans) in train_data_full:
    (data_coded, trans_coded) = (data,
                                 [word2ind[w] for w in trans])
    train_data_coded.append((data_coded, trans_coded))

for (data, trans) in dev_data_full:
    (data_coded, trans_coded) = (data,
                                 [word2ind[w] for w in trans])
    dev_data_coded.append((data_coded, trans_coded))

cPickle.dump(train_data_coded, open('tmp/simple_train_data_coded.pkl', 'w'))
cPickle.dump(dev_data_coded, open('tmp/simple_dev_data_coded.pkl', 'w'))

logger.info('dict examples')
for (k,v) in list(word2ind.items())[:10] + list(ind2word.items())[:10]:
    logger.info((k,v))
logger.info('simple_train_data_full examples')
for (data, trans) in train_data_full[:5]:
    logger.info((data, trans))
logger.info('simple_dev_data_full examples')
for (data, trans) in dev_data_full[:5]:
    logger.info((data, trans))
logger.info('simple_train_data_coded examples')
for (data, trans) in train_data_coded[:5]:
    logger.info((data, trans))
logger.info('simple_dev_data_coded examples')
for (data, trans) in dev_data_coded[:5]:
    logger.info((data, trans))

logger.info('FINISH.')
