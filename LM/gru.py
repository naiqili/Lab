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
    
class GRU(Model):
    def __init__(self, state):
        Model.__init__(self)
        self.rng = numpy.random.RandomState(state['seed'])
        self.state = state
        self.__dict__.update(state)
        self.name = 'GRU'
        self.activation = eval(self.activation)
        self.params = []
        self.global_params = []
        self.init_params()
        
        self.x_data = T.imatrix('x_data')
        self.y_data = T.imatrix('y_data')
        
        [self.hs, self.ot] = self.build_output(self.x_data)
        
        training_mask = T.neq(self.x_data, self.eos_sym)
        self.training_cost = self.build_cost(self.ot, self.y_data, training_mask)
        self.updates = self.compute_updates(self.training_cost, self.params)
        self.y_pred = self.ot.argmax(axis=2) # See lab/argmax_test.py
        
        self.genReset()
        (self.gen_pred, self.gen_updates) = self.build_gen_pred()
        
    def build_gen_pred(self):
        updates = collections.OrderedDict()
        x_t = self.genx
        x_t = self.approx_embedder(x_t)
        h_tm1 = self.genh
        
        r_t = T.nnet.hard_sigmoid(T.dot(x_t, self.W_in_r) + T.dot(h_tm1, self.W_hh_r) + self.b_r)
        z_t = T.nnet.hard_sigmoid(T.dot(x_t, self.W_in_z) + T.dot(h_tm1, self.W_hh_z) + self.b_z)
        h_tilde = self.activation(T.dot(x_t, self.W_in) + T.dot(r_t * h_tm1, self.W_hh) + self.b_hh)
        h_t = (T.ones_like(z_t) - z_t) * h_tm1 + z_t * h_tilde

        o_t = T.dot(h_t, self.W_out) + self.b_out
        self.gen_ot = SoftMax(o_t)
        res = self.gen_ot.argmax()
        updates[self.genx] = res
        updates[self.genh] = h_t
        return (self.gen_ot, updates)

    def init_params(self):
        self.W_emb = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.worddim, self.embdim), name='W_emb'+self.name))
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

    def gated_step(self, x_t, m_t, h_tm1, *args):
        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')
        
        r_t = T.nnet.hard_sigmoid(T.dot(x_t, self.W_in_r) + T.dot(h_tm1, self.W_hh_r) + self.b_r)
        z_t = T.nnet.hard_sigmoid(T.dot(x_t, self.W_in_z) + T.dot(h_tm1, self.W_hh_z) + self.b_z)
        h_tilde = self.activation(T.dot(x_t, self.W_in) + T.dot(r_t * h_tm1, self.W_hh) + self.b_hh)
        h_t = (T.ones_like(z_t) - z_t) * h_tm1 + z_t * h_tilde

        o_t = T.dot(h_t, self.W_out) + self.b_out
        o_t = SoftMax(o_t)
        return [h_t, o_t]

    def build_output(self, x):
        batch_size = x.shape[1]
        h_0 = T.alloc(np.float32(0), batch_size, self.hdim)
        o_enc_info = [h_0]
        xe = self.approx_embedder(x)
        xmask = T.neq(x, self.eos_sym)
        f_enc = self.gated_step
        [h_t, o_t], _ = theano.scan(f_enc, \
                                    sequences=[xe, xmask], \
                                    outputs_info=[h_0, None])
        return [h_t, o_t]        

    def build_cost(self, ot, y, mask):
        x_flatten = ot.dimshuffle(2,0,1)
        x_flatten = x_flatten.flatten(2).dimshuffle(1, 0)
        y_flatten = y.flatten()
        
        cost = x_flatten[T.arange(y_flatten.shape[0]), \
                         y_flatten]
        neg_log_cost = -T.log(cost) * mask.flatten()
        neg_log_cost_s = neg_log_cost.reshape(y.shape)
        sum_neg_log_cost = T.sum(neg_log_cost_s, axis=0)
        s_row = T.sum(mask, axis=0)
        norm_cost = sum_neg_log_cost / s_row
        cost_res = T.mean(norm_cost)
        return cost_res

    def build_train_function(self):
        if not hasattr(self, 'train_fn'):
            self.train_fn = \
                            theano.function(inputs=[self.x_data,
                                                    self.y_data],
                                            outputs=self.training_cost,
                                            updates=self.updates,
                                            name="train_fn")
            return self.train_fn

    def build_eval_function(self):
        if not hasattr(self, 'eval_fn'):
            self.eval_fn = \
                           theano.function(inputs=[self.x_data,
                                                   self.y_data],
                                           outputs=[self.training_cost],
                                           name="eval_fn")
            return self.eval_fn
            
    def build_gen_function(self):
        if not hasattr(self, 'gen_fn'):
            self.gen_fn = \
                           theano.function(inputs=[],
                                           outputs=[self.gen_pred],
                                           updates=self.gen_updates,
                                           name="gen_fn")
        return self.gen_fn

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

    def genReset(self):
        self.genh =  theano.shared(np.zeros((self.hdim,), dtype='float32'), name='h_gen')
        self.genx = theano.shared(np.asarray(1, dtype='int64'), name='x_gen')
        
    def genNext(self):
        gen_fn = self.build_gen_function()
        res = gen_fn()
        return res

    def genExample(self, max_len=30):
        example_sent = []
        self.genReset()
        for k in range(max_len):
            nw = self.genNext()[0]
            t_sel_ind = np.random.multinomial(1, nw)
            sel_ind = np.argmax(samples)
            self.genx.set_value(np.asarray(sel_ind, dtype='int64'))
            example_sent.append(sel_ind)
            if sel_ind == 0:
                break
        return example_sent
