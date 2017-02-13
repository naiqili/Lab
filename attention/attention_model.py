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
    
class AttentionModel(Model):
    def __init__(self, state):
        Model.__init__(self)
        self.rng = numpy.random.RandomState(state['seed'])
        self.state = state
        self.__dict__.update(state)
        self.name = 'AttentionModel'
        self.active = eval(self.active)
        self.params = []
        self.init_params()

        self.x_data = T.imatrix('x_data')
        self.y_data = T.imatrix('y_data')

        self.xmask = T.matrix('x_mask')
        self.ymask = T.matrix('y_mask')

        self.h_enc = self.encode(self.x_data, self.xmask)
        [self.pt, self.alpha] = self.decode(self.h_enc)

        self.cost = self.build_cost(self.pt,
                                    self.y_data,
                                    self.ymask)
        self.updates = self.compute_updates(self.cost, self.params)
        
    def init_params(self):
        self.W_emb = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.word_dim, self.emb_dim), name='W_emb'+self.name))
        self.H_enc = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.h_dim, self.h_dim), name='H_enc'+self.name))
        self.P_enc = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.emb_dim, self.h_dim), name='P_enc'+self.name))
        self.H_dec = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.h_dim, self.h_dim), name='H_enc'+self.name))
        self.P_dec = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.emb_dim, self.h_dim), name='P_enc'+self.name))
        self.W = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.emb_dim, self.h_dim), name='W_dec'+self.name))
        self.U = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.h_dim, self.h_dim), name='U_dec'+self.name))
        self.O_h = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.h_dim, self.h_dim), name='O_h_dec'+self.name))
        self.O_z = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.h_dim, self.h_dim), name='O_z_dec'+self.name))
        self.out_emb = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.h_dim, self.word_dim), name='out_emb'+self.name))
        self.b = add_to_params(self.params, theano.shared(value=np.zeros((self.h_dim,), dtype='float32'), name='b'+self.name))

    def approx_embedder(self, x):
        return self.W_emb[x]

    def encode(self, x_data, mask):
        batch_size = self.bs
        emb_x = self.approx_embedder(x_data)
        def encode_step(x_t, m_t, h_tm1):
            m_t = m_t.dimshuffle(0, 'x')
            h_t = self.active(T.dot(h_tm1, self.H_enc) + \
                                  T.dot(x_t, self.P_enc))
            h_t = m_t * h_t + (1 - m_t) * h_tm1
            return h_t
        h_0 = T.alloc(np.float32(0), batch_size, self.h_dim)
        h_enc, _ = theano.scan(encode_step, \
                               sequences=[emb_x, mask], \
                               outputs_info=[h_0])
        return h_enc

    def decode(self, h_enc):
        batch_size = self.bs
        self.b = self.b.dimshuffle('x', 'x', 0)
        def decode_step(x_tm1, h_tm1, h_enc, b):
            h_t = self.active(T.dot(h_tm1, self.H_dec) + \
                                T.dot(x_tm1, self.P_dec))
            tmp = T.dot(h_tm1, self.W).dimshuffle('x', 0, 1) + \
                  T.dot(h_enc, self.U)
            beta_t = T.sum(b * tmp, axis=2)
            alpha_t = T.exp(beta_t) / T.sum(T.exp(beta_t), axis=0)
            z_tmp = h_enc * alpha_t.dimshuffle(0, 1, 'x')
            z_t = T.sum(z_tmp, axis=0)
            g_t = T.dot(T.dot(h_t, self.O_h) + T.dot(z_t, self.O_z), \
                        self.out_emb)
            p_t = SoftMax(g_t)
            o_t = p_t.argmax(axis=1)
            x_t = self.approx_embedder(o_t)
            return [x_t, p_t, h_t, alpha_t]
        x_0 = theano.shared(np.zeros((batch_size, self.emb_dim), \
                                     dtype='float32'), \
                            name='decode_x0')
        h_0 = theano.shared(np.zeros((batch_size, self.h_dim), \
                                     dtype='float32'), \
                            name='decode_h0')
        alpha_0 = theano.shared(np.zeros((self.seq_len_out, batch_size), \
                                     dtype='float32'), \
                            name='decode_alpha0')

        [x_t, p_t, hs, alpha], _ = theano.scan(decode_step, \
                                               outputs_info=[x_0, None, h_0, None], \
                                         non_sequences=[h_enc, self.b], \
                                         n_steps=self.seq_len_out)
        return [p_t, alpha]
        
    def build_cost(self, ot, y_data, ymask):
        x_flatten = ot.dimshuffle(2,0,1)
        x_flatten = x_flatten.flatten(2).dimshuffle(1, 0)
        y_flatten = y_data.flatten()

        cost = x_flatten[T.arange(y_flatten.shape[0]), \
                         y_flatten]
        neg_log_cost_sum = T.sum(-T.log(cost) * ymask.flatten())
        cost_res = neg_log_cost_sum

        self.pred = x_flatten.argmax()
        self.acc = 1.0 * T.sum(T.eq(self.pred, y_flatten)) / T.sum(ymask)
        return cost_res

    def build_train_function(self):
        if not hasattr(self, 'train_fn'):
            self.train_fn = \
                            theano.function(inputs=[self.x_data,
                                                    self.xmask,
                                                    self.y_data,
                                                    self.ymask],
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
                                                   self.y_data,
                                                   self.ymask],
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
