import theano
import theano.tensor as T
import numpy as np
import cPickle
import logging
import collections

from theano import scan
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.nnet.conv3d2d import *
from collections import OrderedDict

from model import *
from utils import *
from state import *

from abstract_encoder import AbstractEncoder
from natural_encoder import NaturalEncoder
from emb_model import EmbModel

import operator

# Theano speed-up
theano.config.scan.allow_gc = False
#

logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('log/get_embedding.log'))
logging.basicConfig(level = logging.DEBUG,
                        format = "%(asctime)s: %(name)s: %(levelname)s: %(message)s")

model_name = 'model/trainemb_emb100_h100'
nat_path = model_name + '_natural_model.npz'
abs_path = model_name + '_abstract_model.npz'
emb_path = 'model/word_emb.npz'

state = simple_state()
embModel = EmbModel(state)
nat_model = embModel.natural_encoder
abs_model = embModel.abstract_encoder

nat_model.load(nat_path)
abs_model.load(abs_path)
W_emb = embModel.W_emb
W_emb.set_value(numpy.load(emb_path)[W_emb.name])

logger.debug('Models loaded')

train_path = 'tmp/simple_train_data_coded.pkl'
dev_path = 'tmp/simple_dev_data_coded.pkl'

train_data = cPickle.load(open(train_path))
dev_data = cPickle.load(open(dev_path))

nat_output_fn = theano.function([embModel.natural_input],
                                embModel.nat_output)
abs_output_fn = theano.function([embModel.abstract_input],
                                embModel.abs_output)

emb_res = []
for (abs_coded, nat_coded) in train_data:
    m = state['seqlen']
    nat_coded_mat = numpy.zeros((m, 1), dtype='int32')
    sent_len = len(nat_coded)
    nat_coded_mat[:sent_len, 0] = nat_coded
    abs_coded_mat = [abs_coded]
    nat_emb = nat_output_fn(nat_coded_mat)[0]
    #print nat_emb
    #print len(nat_emb[0])
    abs_emb = abs_output_fn(abs_coded_mat)[0]
    #print abs_emb
    #print len(abs_emb[0])
    emb_res.append([abs_coded, nat_coded, nat_emb, abs_emb])

cPickle.dump(emb_res, open('tmp/emb_train.pkl', 'w'))
print 'DONE'
