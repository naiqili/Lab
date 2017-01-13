# -*- coding: utf-8 -*-

import theano
import theano.tensor as T
import numpy as np
import cPickle
import logging
import collections
logger = logging.getLogger(__name__)

from theano import scan
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.nnet.conv3d2d import *
from collections import OrderedDict

from model import *
from utils import *

import operator

# Theano speed-up
theano.config.scan.allow_gc = False
#

def add_to_params(params, new_param):
    params.append(new_param)
    return new_param
    
class AbstractEncoder():
    def __init__(self, state, rng, emb):
        Model.__init__(self)
        self.rng = rng
        self.state = state
        self.__dict__.update(state)
        self.name = 'AbstractEncoder'
        self.activation = eval(self.activation)
        self.params = []
        self.init_params(emb)
        
    def init_params(self, emb):
        #self.W_abs_emb = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.word_dim, self.abs_emb_dim), name='W_abs_emb'+self.name))
        self.W_emb = emb
        self.W = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.abs_emb_dim * self.acttype_cnt, self.h_dim), name='W'+self.name))
        self.b = add_to_params(self.params, theano.shared(value=np.zeros((self.h_dim,), dtype='float32'), name='b'+self.name))
        
    def approx_embedder(self, x):
        return self.W_emb[x]

    def build_output(self, x):
        embx = self.approx_embedder(x)
        embx_flatten = embx.flatten(2)
        self.ot = self.activation(T.dot(embx_flatten, self.W) + self.b)
        return self.ot


