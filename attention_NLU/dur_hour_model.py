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
    
class DurHourModel(Model):
    def __init__(self, state, test_mode=False):
        Model.__init__(self)
        self.rng = numpy.random.RandomState(state['seed'])
        self.state = state
        self.__dict__.update(state)
        self.test_mode = test_mode
        self.name = 'DurHourModel'
        self.active = eval(self.active)
        self.params = []
        self.init_params()

        self.x_data = T.imatrix('x_data')
        self.y_data = T.ivector('y_data')

        self.xmask = T.matrix('x_mask')

        self.h_enc_basic = self.encode(self.x_data, self.xmask)
        self.h_enc_emb = self.approx_embedder(self.x_data)
        self.h_enc = T.concatenate([self.h_enc_basic, self.h_enc_emb], axis=2)
        [self.pt, self.ot, self.alpha] = self.decode()
        
        self.cost = self.build_cost(self.pt,
                                    self.y_data)
        self.updates = self.compute_updates(self.cost, self.params)
        
    def init_params(self):
        self.W_emb = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.word_dim, self.emb_dim), name='W_emb'+self.name))
        self.H_enc = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.h_dim, self.h_dim), name='H_enc'+self.name))
        self.P_enc = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.emb_dim, self.h_dim), name='P_enc'+self.name))
        self.U = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, (self.h_dim + self.emb_dim), self.h_dim), name='U_dec'+self.name))
        self.O_z = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, (self.h_dim + self.emb_dim), self.h_dim), name='O_z_dec'+self.name))
        self.out_emb = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.h_dim, self.out_dim), name='out_emb'+self.name))
        self.b = add_to_params(self.params, theano.shared(value=np.zeros((self.h_dim,), dtype='float32'), name='b'+self.name))
        self.b = self.b.dimshuffle('x', 'x', 0)
        self.encode_b = add_to_params(self.params, theano.shared(value=np.zeros((self.h_dim,), dtype='float32'), name='encode_b'+self.name))

    def approx_embedder(self, x):
        return self.W_emb[x]

    def encode(self, x_data, mask):
        if self.test_mode:
            batch_size = 2
        else:
            batch_size = self.bs
        emb_x = self.approx_embedder(x_data)
        def encode_step(x_t, h_tm1):
            h_t = self.active(T.dot(h_tm1, self.H_enc) + \
                              T.dot(x_t, self.P_enc) + \
                              self.encode_b)
            return h_t
        h_0 = T.alloc(np.float32(0), batch_size, self.h_dim)
        h_enc, _ = theano.scan(encode_step, \
                               sequences=[emb_x], \
                               outputs_info=[h_0])
        return h_enc

    def decode_step(self, h_enc, xmask, b):
        tmp = T.dot(h_enc, self.U)
        beta_t = T.sum(b * tmp, axis=2)
        alpha_t = T.exp(beta_t) * xmask / T.sum(T.exp(beta_t) * xmask, axis=0)
        z_tmp = h_enc * (alpha_t).dimshuffle(0, 1, 'x')
        z_t = T.sum(z_tmp, axis=0)
        g_t = T.dot(T.dot(z_t, self.O_z), self.out_emb)
        p_t = SoftMax(g_t)
        o_t = p_t.argmax(axis=1)
        return [p_t, o_t, alpha_t]
        
    def decode(self):
        batch_size = self.bs
        h_enc = self.h_enc
        xmask = self.xmask
        
        [p_t, o_t, alpha] = self.decode_step(h_enc, xmask, self.b)
        return [p_t, o_t, alpha]
        
    def build_cost(self, ot, abs_out):
        x_flatten = ot
        y_flatten = abs_out

        cost = x_flatten[T.arange(y_flatten.shape[0]), \
                         y_flatten]
        neg_log_cost_sum = T.sum(-T.log(cost))
        cost_res = neg_log_cost_sum / self.bs

        self.pred = x_flatten.argmax(axis=1)
        self.acc = 1.0 * T.sum(T.eq(self.pred, y_flatten)) / self.bs
        return cost_res

    def build_train_function(self):
        if not hasattr(self, 'train_fn'):
            self.train_fn = \
                            theano.function(inputs=[self.x_data,
                                                    self.xmask,
                                                    self.y_data],
                                            outputs=[self.cost, \
                                                     self.acc],
                                            updates=self.updates,
                                            name="train_fn")
        return self.train_fn

    def build_eval_function(self):
        if not hasattr(self, 'eval_fn'):
            self.eval_fn = \
                           theano.function(inputs=[self.x_data,
                                                   self.xmask,
                                                   self.y_data],
                                           outputs=[self.cost, \
                                                    self.acc],
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
