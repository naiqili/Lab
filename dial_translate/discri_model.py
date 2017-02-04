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

from abstract_encoder import AbstractEncoder
from natural_encoder import NaturalEncoder

import operator

# Theano speed-up
theano.config.scan.allow_gc = False
#

def add_to_params(params, new_param):
    params.append(new_param)
    return new_param
    
class DiscriModel(Model):
    def __init__(self, state):
        Model.__init__(self)
        self.rng = numpy.random.RandomState(state['seed'])
        self.state = state
        self.__dict__.update(state)
        self.name = 'DiscriModel'
        self.activation = eval(self.activation)
        self.params = []
        self.init_params()
        self.natural_encoder = NaturalEncoder(state, self.rng, self.W_emb)
        self.params = self.params + self.natural_encoder.params

        self.natural_input = T.imatrix('natural_input')
        self.abstract_output = T.imatrix('abstract_output')

        self.output = self.build_output(self.natural_input)
        (self.cost, self.acc) = self.build_cost(self.output,
                                    self.abstract_output)
        self.updates = self.compute_updates(self.cost, self.params)
        
    def init_params(self):
        self.W_emb = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.word_dim, self.emb_dim), name='W_emb'+self.name))
        self.W_abs = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.acttype_cnt, self.emb_dim, self.word_dim), name='W_abs'+self.name))
        self.b_abs = add_to_params(self.params, theano.shared(value=np.zeros((self.acttype_cnt, self.word_dim,), dtype='float32'), name='b_abs'+self.name))

    def build_output(self, natural_input):
        # nat_emb: bs x emb_dim
        # y:       bs x acttype_cnt
        # W_abs:   acttype_cnt x emb_dim x word_dim
        # ot = nat_emb x W_abs:
        #          bs x acttype_cnt x word_dimn
        # (nat_emb x W_abs).flatten(2) & dimshuffle:
        #          (bs x acttype_cnt) x word_dim
        nat_emb = self.natural_encoder.build_output(natural_input)
        o_t = T.dot(nat_emb, self.W_abs) + self.b_abs
        o_t = SoftMax(o_t)
        return o_t
        
    def build_cost(self, ot, y):
        # nat_emb: bs x emb_dim
        # y:       bs x acttype_cnt
        # W_abs:   acttype_cnt x emb_dim x word_dim
        # ot = nat_emb x W_abs:
        #          bs x acttype_cnt x word_dimn
        # (nat_emb x W_abs).flatten(2):
        #          bs x (acttype_cnt x word_dim)
        nat_flatten = ot.flatten(2)
        y_flatten = y.flatten()
        cost = nat_flatten[T.arange(y_flatten.shape[0]), \
                           y_flatten]
        neg_log = T.sum(-T.log(cost))
        cost = neg_log
        self.pred = T.argmax(nat_flatten, axis=1)
        acc = T.sum(T.eq(self.pred, y_flatten))
        acc = acc / (self.bs * self.acttype_cnt)
        return (cost, acc)

    def build_train_function(self):
        if not hasattr(self, 'train_fn'):
            self.train_fn = \
                            theano.function(inputs=[self.abstract_output,
                                                    self.natural_input],
                                            outputs=[self.cost, self.acc],
                                            updates=self.updates,
                                            name="train_fn")
        return self.train_fn

    def build_eval_function(self):
        if not hasattr(self, 'eval_fn'):
            self.eval_fn = \
                           theano.function(inputs=[self.abstract_output,
                                                    self.natural_input],
                                           outputs=[self.cost, self.acc],
                                           name="eval_fn")
            return self.eval_fn

    def compute_updates(self, training_cost, params):
        updates = []
         
        grads = T.grad(training_cost, params)
        grads = OrderedDict(zip(params, grads))

        # Clip stuff
        c = numpy.float32(self.cutoff)
        clip_grads = []
        
        norm_gs = T.sqrt(sum(T.sum(g ** 2) for p, g in grads.items()))
        normalization = T.switch(T.ge(norm_gs, c), c / norm_gs, np.float32(1.))
        notfinite = T.or_(T.isnan(norm_gs), T.isinf(norm_gs))
         
        for p, g in grads.items():
            clip_grads.append((p, T.switch(notfinite, numpy.float32(.1) * p, g * normalization)))
        
        grads = OrderedDict(clip_grads)

        if self.updater == 'adagrad':
            updates = Adagrad(grads, self.lr)  
        elif self.updater == 'sgd':
            raise Exception("Sgd not implemented!")
        elif self.updater == 'adadelta':
            updates = Adadelta(grads)
        elif self.updater == 'rmsprop':
            updates = RMSProp(grads, self.lr)
        elif self.updater == 'adam':
            updates = Adam(grads)
        else:
            raise Exception("Updater not understood!") 

        return updates
