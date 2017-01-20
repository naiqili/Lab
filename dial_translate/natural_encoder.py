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
    
class NaturalEncoder():
    def __init__(self, state, rng, emb):
        self.rng = rng
        self.state = state
        self.__dict__.update(state)
        self.name = 'NaturalEncoder'
        self.activation = eval(self.activation)
        self.params = []
        self.init_params(emb)

    def init_params(self, emb):
        self.W_emb = emb

        self.U_i = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.emb_dim, self.h_dim), name='U_i'+self.name))
        self.W_i = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.emb_dim, self.h_dim), name='W_i'+self.name))
        self.b_i = add_to_params(self.params, theano.shared(value=np.zeros((self.h_dim,), dtype='float32'), name='b_i'+self.name))

        self.U_f = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.emb_dim, self.h_dim), name='U_f'+self.name))
        self.W_f = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.emb_dim, self.h_dim), name='W_f'+self.name))
        self.b_f = add_to_params(self.params, theano.shared(value=np.zeros((self.h_dim,), dtype='float32'), name='b_f'+self.name))

        self.U_o = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.emb_dim, self.h_dim), name='U_o'+self.name))
        self.W_o = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.emb_dim, self.h_dim), name='W_o'+self.name))
        self.b_o = add_to_params(self.params, theano.shared(value=np.zeros((self.h_dim,), dtype='float32'), name='b_o'+self.name))

        self.U_g = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.emb_dim, self.h_dim), name='U_g'+self.name))
        self.W_g = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.emb_dim, self.h_dim), name='W_g'+self.name))
        self.b_g = add_to_params(self.params, theano.shared(value=np.zeros((self.h_dim,), dtype='float32'), name='b_g'+self.name))
        
    def approx_embedder(self, x):
        return self.W_emb[x]

    def gated_step(self, x_t, m_t, h_tm1):
        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')

        i_t = T.nnet.sigmoid(T.dot(x_t, self.U_i) + T.dot(h_tm1, self.W_i) + self.b_i)
        f_t = T.nnet.sigmoid(T.dot(x_t, self.U_f) + T.dot(h_tm1, self.W_f) + self.b_f)
        o_t = T.nnet.sigmoid(T.dot(x_t, self.U_o) + T.dot(h_tm1, self.W_o) + self.b_o)
        g_t = self.activation(T.dot(x_t, self.U_g) + T.dot(h_tm1, self. W_g) + self.b_g)
        c_t = c_tm1 * f_t + g_t * i_t
        h_t_tmp = self.activation(c_t) * o_t
        h_t = m_t * h_t_tmp + (np.float32(1.0) - m_t) * h_tm1

        return [h_t, c_t]

    def build_output(self, x):
        batch_size = x.shape[1]
        h_0 = T.alloc(np.float32(0), batch_size, self.h_dim)
        c_0 = T.alloc(np.float32(0), batch_size, self.h_dim)
        xe = self.approx_embedder(x)
        xmask = T.neq(x, self.eos_sym)
        f_enc = self.gated_step
        [h_t, c_t], _ = theano.scan(f_enc, \
                                    sequences=[xe, xmask], \
                                    outputs_info=[h_0, c_0])
        return h_t[-1]
