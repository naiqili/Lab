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
    
class NaturalEncoder2(Model):
    def __init__(self, state, rng, emb):
        Model.__init__(self)
        self.rng = rng
        self.state = state
        self.__dict__.update(state)
        self.name = '_NaturalEncoder'
        self.activation = eval(self.activation)
        self.params = []
        self.init_params(emb)

    def init_params(self, emb):
        self.W_emb = emb

        self.W_out = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rnnh_dim, self.h_dim), name='W_out'+self.name))
        self.b_out = add_to_params(self.params, theano.shared(value=np.zeros((self.h_dim,), dtype='float32'), name='b_out'+self.name))

        self.U1_i = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.emb_dim, self.rnnh_dim), name='U1_i'+self.name))
        self.W1_i = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rnnh_dim, self.rnnh_dim), name='W1_i'+self.name))
        self.b1_i = add_to_params(self.params, theano.shared(value=np.zeros((self.rnnh_dim,), dtype='float32'), name='b1_i'+self.name))

        self.U1_f = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.emb_dim, self.rnnh_dim), name='U1_f'+self.name))
        self.W1_f = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rnnh_dim, self.rnnh_dim), name='W1_f'+self.name))
        self.b1_f = add_to_params(self.params, theano.shared(value=np.zeros((self.rnnh_dim,), dtype='float32'), name='b1_f'+self.name))

        self.U1_o = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.emb_dim, self.rnnh_dim), name='U1_o'+self.name))
        self.W1_o = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rnnh_dim, self.rnnh_dim), name='W1_o'+self.name))
        self.b1_o = add_to_params(self.params, theano.shared(value=np.zeros((self.rnnh_dim,), dtype='float32'), name='b1_o'+self.name))

        self.U1_g = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.emb_dim, self.rnnh_dim), name='U1_g'+self.name))
        self.W1_g = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rnnh_dim, self.rnnh_dim), name='W1_g'+self.name))
        self.b1_g = add_to_params(self.params, theano.shared(value=np.zeros((self.rnnh_dim,), dtype='float32'), name='b1_g'+self.name))

        self.U2_i = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rnnh_dim, self.rnnh_dim), name='U2_i'+self.name))
        self.W2_i = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rnnh_dim, self.rnnh_dim), name='W2_i'+self.name))
        self.b2_i = add_to_params(self.params, theano.shared(value=np.zeros((self.rnnh_dim,), dtype='float32'), name='b2_i'+self.name))

        self.U2_f = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rnnh_dim, self.rnnh_dim), name='U2_f'+self.name))
        self.W2_f = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rnnh_dim, self.rnnh_dim), name='W2_f'+self.name))
        self.b2_f = add_to_params(self.params, theano.shared(value=np.zeros((self.rnnh_dim,), dtype='float32'), name='b2_f'+self.name))

        self.U2_o = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rnnh_dim, self.rnnh_dim), name='U2_o'+self.name))
        self.W2_o = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rnnh_dim, self.rnnh_dim), name='W2_o'+self.name))
        self.b2_o = add_to_params(self.params, theano.shared(value=np.zeros((self.rnnh_dim,), dtype='float32'), name='b2_o'+self.name))

        self.U2_g = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rnnh_dim, self.rnnh_dim), name='U2_g'+self.name))
        self.W2_g = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rnnh_dim, self.rnnh_dim), name='W2_g'+self.name))
        self.b2_g = add_to_params(self.params, theano.shared(value=np.zeros((self.rnnh_dim,), dtype='float32'), name='b2_g'+self.name))
        
    def approx_embedder(self, x):
        return self.W_emb[x]

    def gated_step(self, x_t, m_t, h1_tm1, c1_tm1, h2_tm1, c2_tm1, *args):
        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')

        i1_t = T.nnet.sigmoid(T.dot(x_t, self.U1_i) + T.dot(h1_tm1, self.W1_i) + self.b1_i)
        f1_t = T.nnet.sigmoid(T.dot(x_t, self.U1_f) + T.dot(h1_tm1, self.W1_f) + self.b1_f)
        o1_t = T.nnet.sigmoid(T.dot(x_t, self.U1_o) + T.dot(h1_tm1, self.W1_o) + self.b1_o)
        g1_t = self.activation(T.dot(x_t, self.U1_g) + T.dot(h1_tm1, self.W1_g) + self.b1_g)
        c1_t = c1_tm1 * f1_t + g1_t * i1_t
        h1_t_tmp = self.activation(c1_t) * o1_t
        h1_t = m_t * h1_t_tmp + (np.float32(1.0) - m_t) * h1_tm1

        i2_t = T.nnet.sigmoid(T.dot(h1_t, self.U2_i) + T.dot(h2_tm1, self.W2_i) + self.b2_i)
        f2_t = T.nnet.sigmoid(T.dot(h1_t, self.U2_f) + T.dot(h2_tm1, self.W2_f) + self.b2_f)
        o2_t = T.nnet.sigmoid(T.dot(h1_t, self.U2_o) + T.dot(h2_tm1, self.W2_o) + self.b2_o)
        g2_t = self.activation(T.dot(h1_t, self.U2_g) + T.dot(h2_tm1, self.W2_g) + self.b2_g)
        c2_t = c2_tm1 * f2_t + g2_t * i2_t
        h2_t_tmp = self.activation(c2_t) * o2_t

        h2_t = m_t * h2_t_tmp + (np.float32(1.0) - m_t) * h2_tm1

        o_t = T.dot(h2_t, self.W_out) + self.b_out

        return [h1_t, c1_t, h2_t, c2_t, o_t]

    def build_output(self, x):
        batch_size = x.shape[1]
        h1_0 = T.alloc(np.float32(0), batch_size, self.h_dim)
        c1_0 = T.alloc(np.float32(0), batch_size, self.h_dim)
        h2_0 = T.alloc(np.float32(0), batch_size, self.h_dim)
        c2_0 = T.alloc(np.float32(0), batch_size, self.h_dim)
        xe = self.approx_embedder(x)
        xmask = T.neq(x, self.eos_sym)
        f_enc = self.gated_step
        [_, _, _, _, o_t], _ = theano.scan(f_enc, \
                                    sequences=[xe, xmask], \
                                    outputs_info=[h1_0, c1_0, h2_0, c2_0, None])
        return o_t[-1]
