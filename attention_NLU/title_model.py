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
    
class TitleModel(Model):
    def __init__(self, state, test_mode=False):
        Model.__init__(self)
        self.rng = numpy.random.RandomState(state['seed'])
        self.state = state
        self.__dict__.update(state)
        self.test_mode = test_mode
        self.name = 'TitleModel'
        self.active = eval(self.active)
        self.params = []
        self.init_params()

        self.x_data = T.imatrix('x_data')
        self.abs_in = T.imatrix('abs_in')
        self.abs_out = T.imatrix('abs_out')

        self.xmask = T.matrix('x_mask')
        self.ymask = T.matrix('y_mask')

        self.h_enc_basic = self.encode(self.x_data, self.xmask)
        self.h_enc_emb = self.approx_embedder(self.x_data)
        self.h_enc = T.concatenate([self.h_enc_basic, self.h_enc_emb], axis=2)
        [self.pt, self.ot, self.h_t, self.alpha] = self.decode()
        
        self.cost = self.build_cost(self.pt,
                                    self.abs_out,
                                    self.ymask)
        self.updates = self.compute_updates(self.cost, self.params)

        self.gen_h = theano.shared(value=np.zeros((2, self.h_dim), dtype='float64'), name='gen_h')
        self.gen_x = T.ivector('gen_x')
        [self.gen_pred, self.gen_ot, self.gen_alpha, self.gen_updates] = self.build_gen()
        
    def init_params(self):
        self.W_emb = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.word_dim, self.emb_dim), name='W_emb'+self.name))
        self.H_enc = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.h_dim, self.h_dim), name='H_enc'+self.name))
        self.P_enc = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.emb_dim, self.h_dim), name='P_enc'+self.name))
        self.H_dec = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.h_dim, self.h_dim), name='H_dec'+self.name))
        self.P_dec = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.emb_dim, self.h_dim), name='P_dec'+self.name))
        self.W = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.h_dim, self.h_dim), name='W_dec'+self.name))
        self.U = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, (self.h_dim + self.emb_dim), self.h_dim), name='U_dec'+self.name))
        self.O_h = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.h_dim, self.h_dim), name='O_h_dec'+self.name))
        self.O_z = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, (self.h_dim + self.emb_dim), self.h_dim), name='O_z_dec'+self.name))
        self.out_emb = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.h_dim, self.word_dim), name='out_emb'+self.name))
        self.b = add_to_params(self.params, theano.shared(value=np.zeros((self.h_dim,), dtype='float64'), name='b'+self.name))
        self.b = self.b.dimshuffle('x', 'x', 0)
        self.encode_b = add_to_params(self.params, theano.shared(value=np.zeros((self.h_dim,), dtype='float64'), name='encode_b'+self.name))
        self.decode_b = add_to_params(self.params, theano.shared(value=np.zeros((self.h_dim,), dtype='float64'), name='decode_b'+self.name))

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
        h_0 = T.alloc(np.float64(0), batch_size, self.h_dim)
        h_enc, _ = theano.scan(encode_step, \
                               sequences=[emb_x], \
                               outputs_info=[h_0])
        return h_enc

    def decode_step(self, abs_in_t, h_tm1, h_enc, xmask, b):
        x_t = self.approx_embedder(abs_in_t)
        h_t = self.active(T.dot(h_tm1, self.H_dec) + \
                          T.dot(x_t, self.P_dec) + \
                          self.decode_b)
        tmp = T.dot(h_tm1, self.W).dimshuffle('x', 0, 1) + \
              T.dot(h_enc, self.U)
        beta_t = T.sum(b * tmp, axis=2)
        alpha_t = T.exp(beta_t) * xmask / T.sum(T.exp(beta_t) * xmask, axis=0)
        z_tmp = h_enc * (alpha_t).dimshuffle(0, 1, 'x')
        z_t = T.sum(z_tmp, axis=0)
        g_t = T.dot(T.dot(h_t, self.O_h) + T.dot(z_t, self.O_z), \
                    self.out_emb)
        p_t = SoftMax(g_t)
        o_t = p_t.argmax(axis=1)
        return [p_t, o_t, h_t, alpha_t]

    def build_gen(self):
        x_t = self.approx_embedder(self.gen_x)
        h_tm1 = self.gen_h
        h_enc = self.h_enc
        xmask = self.xmask
        b = self.b
        h_t = self.active(T.dot(h_tm1, self.H_dec) + \
                          T.dot(x_t, self.P_dec) + \
                          self.decode_b)
        tmp = T.dot(h_tm1, self.W).dimshuffle('x', 0, 1) + \
              T.dot(h_enc, self.U)
        beta_t = T.sum(b * tmp, axis=2)
        beta_t2 = beta_t - T.max(beta_t)
        alpha_t = T.exp(beta_t2) * xmask / T.sum(T.exp(beta_t2) * xmask, axis=0)
        z_tmp = h_enc * (alpha_t).dimshuffle(0, 1, 'x')
        z_t = T.sum(z_tmp, axis=0)
        g_t = T.dot(T.dot(h_t, self.O_h) + T.dot(z_t, self.O_z), \
                    self.out_emb)
        p_t = SoftMax(g_t)
        o_t = p_t.argmax(axis=1)
        updates = [(self.gen_h, h_t)]
        return [p_t, o_t, alpha_t, updates]

    def gen_reset(self):
        self.gen_h.set_value(np.zeros((2, self.h_dim), dtype='float64'))

    def gen_next(self, abs_in, h_enc, xmask, b):
        abs_in_emb = self.approx_embedder([abs_in, 0])
        gen_fn = self.build_gen_function()
        p_t = gen_fn(self.x_data, self.x_mask, self.gen_x)
        return p_t
        
    def decode(self):
        batch_size = self.bs
        h_enc = self.h_enc
        xmask = self.xmask

        h_0 = theano.shared(np.zeros((batch_size, self.h_dim), \
                                     dtype='float64'), \
                            name='decode_h0')
            
        [p_t, o_t, h_t, alpha], _ = theano.scan(self.decode_step, \
                                          outputs_info=[None, None, h_0, None], \
                                          non_sequences=[h_enc, xmask, self.b], \
                                          sequences=[self.abs_in])
        return [p_t, o_t, h_t, alpha]
        
    def build_cost(self, ot, abs_out, ymask):
        x_flatten = ot.dimshuffle(2,0,1)
        x_flatten = x_flatten.flatten(2).dimshuffle(1, 0)
        y_flatten = abs_out.flatten()

        cost = x_flatten[T.arange(y_flatten.shape[0]), \
                         y_flatten]
        neg_log_cost_sum = T.sum(-T.log(cost) * ymask.flatten())
        cost_res = neg_log_cost_sum

        self.pred = x_flatten.argmax(axis=1)
        self.acc = 1.0 * T.sum(T.eq(self.pred, y_flatten) * ymask.flatten()) / T.sum(ymask)
        return cost_res

    def build_train_function(self):
        if not hasattr(self, 'train_fn'):
            self.train_fn = \
                            theano.function(inputs=[self.x_data,
                                                    self.xmask,
                                                    self.abs_in,
                                                    self.abs_out,
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
                                                   self.abs_in,
                                                   self.abs_out,
                                                   self.ymask],
                                           outputs=[self.cost, \
                                                    self.acc],
                                           name="eval_fn")
        return self.eval_fn

    def build_gen_function(self):
        if not hasattr(self, 'gen_fn'):
            self.gen_fn = \
                           theano.function(inputs=[self.x_data,
                                                   self.xmask,
                                                   self.gen_x],
                                           outputs=[self.gen_pred, self.gen_ot, self.gen_alpha],
                                           updates=self.gen_updates,
                                           name="gen_fn")
        return self.gen_fn

    def compute_updates(self, training_cost, params):
        updates = []
         
        grads = T.grad(training_cost, params)
        grads = OrderedDict(zip(params, grads))

        # Clip stuff
        c = numpy.float64(self.cutoff)
        clip_grads = []
        
        norm_gs = T.sqrt(sum(T.sum(g ** 2) for p, g in grads.items()))
        normalization = T.switch(T.ge(norm_gs, c), c / norm_gs, np.float64(1.))
        notfinite = T.or_(T.isnan(norm_gs), T.isinf(norm_gs))
         
        for p, g in grads.items():
            clip_grads.append((p, T.switch(notfinite, numpy.float64(.1) * p, g * normalization)))
        
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
