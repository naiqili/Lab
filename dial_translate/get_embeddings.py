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

from abstract_encoder import AbstractEncoder
from natural_encoder import NaturalEncoder

import operator

# Theano speed-up
theano.config.scan.allow_gc = False
#

logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('log/get_embedding.log'))
logging.basicConfig(level = logging.DEBUG,
                        format = "%(asctime)s: %(name)s: %(levelname)s: %(message)s")

model_name = ''
nat_path = model_name + '_natural_model.npz'
abs_path = model_name + '_abstract_model.npz'
emb_path = model_name + '_word_emb.npz'

state = simple_state()
embModel = EmbModel(state)
nat_model = embModel.natural_encoder
abs_model = embModel.abstract_encoder

nat_model.load(nat_path)
abs_model.load(abs_path)
W_emb = embModel.W_emb
W_emb.set_value(numpy.load(emb_path)[W_emb.name])

logger.debug('Models loaded')

