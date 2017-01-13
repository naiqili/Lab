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
    
class EmbModel(Model):
    def __init__(self, state):
        Model.__init__(self)
        self.rng = numpy.random.RandomState(state['seed'])
        self.state = state
        self.__dict__.update(state)
        self.name = 'AbstractEncoder'
        self.activation = eval(self.activation)
        self.params = []
        self.init_params()
        self.abstract_encoder = AbstractEncoder(state, self.rng, self.W_emb)
        self.natural_encoder = NaturalEncoder(state, self.rng, self.W_emb)
        self.params = self.params + self.abstract_encoder.params + self.natural_encoder.params

        self.noise_vars = []
        for k in range(self.noise_cnt):
            self.noise_vars.append(T.imatrix('noise_' + str(k)))
        self.natural_input = T.imatrix('natural_input')
        self.abstract_input = T.imatrix('abstract_input')

        self.cost = self.build_cost(self.natural_input,
                                    self.abstract_input,
                                    self.noise_vars)
        
    def init_params(self):
        self.W_emb = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.word_dim, self.emb_dim), name='W_emb'+self.name))
        
    def build_cost(self, natural, abstract, noise_vars):
        nat_emb = self.natural_encoder.build_output(natural)
        abs_emb = self.abstract_encoder.build_output(abstract)
        noise_embs = []
        for var in noise_vars:
            noise_emb = self.abstract_encoder.build_output(var)
            noise_embs.append(noise_emb)
        cost_nat_abs = T.sum( (nat_emb-abs_emb) ** 2 )
        cost_nat_noise = T.sum( [(nat_emb-noise_emb) ** 2 for noise_emb in noise_embs] )
        cost = T.max(

    def build_train_function(self):

    def build_eval_function(self):

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
