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
    
class GRU():
    def __init__(self, state, rng, in_dim, h_dim):
        self.rng = rng
        self.state = state
        self.__dict__.update(state)
        self.name = 'GRU'
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.activation = eval(self.activation)
        self.params = []
        self.init_params(in_dim, h_dim)
        
    def init_params(self, in_dim, h_dim):
        self.W_in = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, in_dim, h_dim), name='W_in'+self.name))
        self.W_hh = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, h_dim, h_dim), name='W_hh'+self.name))
        self.b_hh = add_to_params(self.params, theano.shared(value=np.zeros((h_dim,), dtype='float32'), name='b_hh'+self.name))
        
        self.W_in_r = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, in_dim, h_dim), name='W_in_r'+self.name))
        self.W_in_z = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, in_dim, h_dim), name='W_in_z'+self.name))
        self.W_hh_r = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, h_dim, h_dim), name='W_hh_r'+self.name))
        self.W_hh_z = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, h_dim, h_dim), name='W_hh_z'+self.name))
        self.b_z = add_to_params(self.params, theano.shared(value=np.zeros((h_dim,), dtype='float32'), name='b_z'+self.name))
        self.b_r = add_to_params(self.params, theano.shared(value=np.zeros((h_dim,), dtype='float32'), name='b_r'+self.name))
        
    def gated_step(self, x_t, m_t, h_tm1, *args):
        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')
        
        r_t = T.nnet.hard_sigmoid(T.dot(x_t, self.W_in_r) + T.dot(h_tm1, self.W_hh_r) + self.b_r)
        z_t = T.nnet.hard_sigmoid(T.dot(x_t, self.W_in_z) + T.dot(h_tm1, self.W_hh_z) + self.b_z)
        h_tilde = self.activation(T.dot(x_t, self.W_in) + T.dot(r_t * h_tm1, self.W_hh) + self.b_hh)
        h_t_tmp = (np.float32(1.0) - z_t) * h_tm1 + z_t * h_tilde
        h_t = m_t * h_t_tmp + (np.float32(1.0) - m_t) * h_tm1

        return h_t

    def build_output(self, batch_size, xe, xmask):
        h_0 = T.alloc(np.float32(0), batch_size, self.h_dim)
        o_enc_info = [h_0]
        f_enc = self.gated_step
        h_t, _ = theano.scan(f_enc, \
                                    sequences=[xe, xmask], \
                                    outputs_info=[h_0])
        return h_t
