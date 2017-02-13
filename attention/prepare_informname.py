import cPickle
import logging
from utils import *

logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('log/prepare_informname.log'))
logging.basicConfig(level = logging.DEBUG,
                    format = "%(asctime)s: %(name)s: %(levelname)s: %(message)s")

(word2ind, ind2word) = cPickle.load(open('tmp/dict.pkl'))
train_coded = cPickle.load(open('tmp/train_data_coded.pkl'))
dev_coded = cPickle.load(open('tmp/dev_data_coded.pkl'))

informname_train = []
informname_dev = []

for (abstract, natural) in train_coded:
    abs_tmp = abstract[6:10]
    _abs = [1]
    for ind in abs_tmp:
        if ind2word[ind] != '<NULL>':
            _abs.append(ind)
    _abs.append(0)
    if len(_abs) > 3:
        print [ind2word[w] for w in _abs]
        print [ind2word[w] for w in natural]
    informname_train.append((_abs, natural))

for (abstract, natural) in dev_coded:
    abs_tmp = abstract[6:10]
    _abs = [1]
    for ind in abs_tmp:
        if ind2word[ind] != '<NULL>':
            _abs.append(ind)
    _abs.append(0)
    informname_dev.append((_abs, natural))

logger.debug('informname_train')
for (abstract, natural) in informname_train[:5]:
    logger.debug(str(abstract))
    logger.debug(str(natural))
    logger.debug(str([ind2word[ind] for ind in abstract]))
    logger.debug(str([ind2word[ind] for ind in natural]))

logger.debug('informname_dev')
for (abstract, natural) in informname_dev[:5]:
    logger.debug(str(abstract))
    logger.debug(str(natural))
    logger.debug(str([ind2word[ind] for ind in abstract]))
    logger.debug(str([ind2word[ind] for ind in natural]))
    
cPickle.dump(informname_train, open('tmp/informname_train.pkl', 'w'))
cPickle.dump(informname_dev, open('tmp/informname_dev.pkl', 'w'))
