import cPickle
import logging
from utils import *

logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('log/prepare_informfood.log'))
logging.basicConfig(level = logging.DEBUG,
                    format = "%(asctime)s: %(name)s: %(levelname)s: %(message)s")

(word2ind, ind2word) = cPickle.load(open('tmp/dict.pkl'))
train_coded = cPickle.load(open('tmp/train_data_coded.pkl'))
dev_coded = cPickle.load(open('tmp/dev_data_coded.pkl'))

informfood_train = []
informfood_dev = []

for (abstract, natural) in train_coded:
    abs_tmp = abstract[16:18]
    _abs = [1]
    for ind in abs_tmp:
        if ind2word[ind] != '<NULL>':
            _abs.append(ind)
    _abs.append(0)
    if ind2word[_abs[1]] != '<NO>':
        print [ind2word[w] for w in _abs]
        print [ind2word[w] for w in natural]
    informfood_train.append((_abs, natural))

for (abstract, natural) in dev_coded:
    abs_tmp = abstract[16:18]
    _abs = [1]
    for ind in abs_tmp:
        if ind2word[ind] != '<NULL>':
            _abs.append(ind)
    _abs.append(0)
    informfood_dev.append((_abs, natural))

logger.debug('informfood_train')
for (abstract, natural) in informfood_train[:50]:
    logger.debug(str(abstract))
    logger.debug(str(natural))
    logger.debug(str([ind2word[ind] for ind in abstract]))
    logger.debug(str([ind2word[ind] for ind in natural]))

logger.debug('informfood_dev')
for (abstract, natural) in informfood_dev[:50]:
    logger.debug(str(abstract))
    logger.debug(str(natural))
    logger.debug(str([ind2word[ind] for ind in abstract]))
    logger.debug(str([ind2word[ind] for ind in natural]))
    
cPickle.dump(informfood_train, open('tmp/informfood_train.pkl', 'w'))
cPickle.dump(informfood_dev, open('tmp/informfood_dev.pkl', 'w'))
