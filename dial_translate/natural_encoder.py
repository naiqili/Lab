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
        Model.__init__(self)
        self.rng = rng
        self.state = state
        self.__dict__.update(state)
        self.name = 'NaturalEncoder'
        self.activation = eval(self.activation)
        self.params = []
        self.init_params(emb)

    def init_params(self, emb):
        self.W_emb = emb
        self.W_in = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.embdim, self.hdim), name='W_in'+self.name))
        self.W_hh = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.hdim, self.hdim), name='W_hh'+self.name))
        self.b_hh = add_to_params(self.params, theano.shared(value=np.zeros((self.hdim,), dtype='float32'), name='b_hh'+self.name))
        self.W_out = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.hdim, self.worddim), name='W_out'+self.name))
        self.b_out = add_to_params(self.params, theano.shared(value=np.zeros((self.worddim,), dtype='float32'), name='b_out'+self.name))
        
        self.W_in_r = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.embdim, self.hdim), name='W_in_r'+self.name))
        self.W_in_z = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.embdim, self.hdim), name='W_in_z'+self.name))
        self.W_hh_r = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.hdim, self.hdim), name='W_hh_r'+self.name))
        self.W_hh_z = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.hdim, self.hdim), name='W_hh_z'+self.name))
        self.b_z = add_to_params(self.params, theano.shared(value=np.zeros((self.hdim,), dtype='float32'), name='b_z'+self.name))
        self.b_r = add_to_params(self.params, theano.shared(value=np.zeros((self.hdim,), dtype='float32'), name='b_r'+self.name))
        
    def approx_embedder(self, x):
        return self.W_emb[x]

    def gated_step(self, x_t, m_t, h_tm1):
        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')
        
        r_t = T.nnet.sigmoid(T.dot(x_t, self.W_in_r) + T.dot(h_tm1, self.W_hh_r) + self.b_r)
        z_t = T.nnet.sigmoid(T.dot(x_t, self.W_in_z) + T.dot(h_tm1, self.W_hh_z) + self.b_z)
        h_tilde = self.activation(T.dot(x_t, self.W_in) + T.dot(r_t * h_tm1, self.W_hh) + self.b_hh)
        h_t_tmp = (np.float32(1.0) - z_t) * h_tm1 + z_t * h_tilde
        h_t = m_t * h_t_tmp + (np.float32(1.0) - m_t) * h_tm1
        # return both reset state and non-reset state
        return h_t

    def bulid_hidden_states(self, x):
        batch_size = x.shape[1]
        h_0 = T.alloc(np.float32(0), batch_size, self.hdim)
        o_enc_info = [h_0]
        xe = self.approx_embedder(x)
        xmask = T.neq(x, self.eos_sym)
        f_enc = self.gated_step
        _res, _ = theano.scan(f_enc,
                              sequences=[xe, xmask],\
                              outputs_info=o_enc_info)
        return _res
        
    def simple_step(self, h):
        o_t = self.activation(T.dot(h, self.W_out) + self.b_out)
        o_t = SoftMax(o_t)
        return o_t
        
    def build_output(self, x):
        hs = self.build_hidden_states(x)
        last_h = hs[-1]
        self.ot = self.simple_step(last_h)
        return self.ot
        
