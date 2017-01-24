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
    
class Layer2GRU(Model):
    def __init__(self, state):
        Model.__init__(self)
        self.rng = numpy.random.RandomState(state['seed'])
        self.state = state
        self.__dict__.update(state)
        self.name = 'Layer2GRU'
        self.activation = eval(self.activation)
        self.params = []
        self.init_params()

        self.x_data = T.imatrix('x_data')
        self.y_data = T.imatrix('y_data')

        self.ot = self.build_output(self.x_data)
        xmask = T.neq(self.x_data, self.eos_sym)
        
        self.training_cost = self.build_cost(self.ot, self.y_data, xmask)
        self.updates = self.compute_updates(self.training_cost, self.params)
        self.y_pred = self.ot.argmax(axis=2) # See lab/argmax_test.py
        
        self.genReset()
        (self.gen_pred, self.gen_updates) = self.build_gen_pred()
        
    def build_gen_pred(self):
        updates = collections.OrderedDict()
        x_t = self.genx
        x_t = self.approx_embedder(x_t)
        h1_tm1 = self.genh1
        h2_tm1 = self.genh2
        
        r1_t = T.nnet.hard_sigmoid(T.dot(x_t, self._1W_in_r) + T.dot(h1_tm1, self._1W_hh_r) + self._1b_r)
        z1_t = T.nnet.hard_sigmoid(T.dot(x_t, self._1W_in_z) + T.dot(h1_tm1, self._1W_hh_z) + self._1b_z)
        h1_tilde = self.activation(T.dot(x_t, self._1W_in) + T.dot(r1_t * h1_tm1, self._1W_hh) + self._1b_hh)
        h1_t = (T.ones_like(z1_t) - z1_t) * h1_tm1 + z1_t * h1_tilde

        r2_t = T.nnet.hard_sigmoid(T.dot(h1_t, self._2W_in_r) + T.dot(h2_tm1, self._2W_hh_r) + self._2b_r)
        z2_t = T.nnet.hard_sigmoid(T.dot(h1_t, self._2W_in_z) + T.dot(h2_tm1, self._2W_hh_z) + self._2b_z)
        h2_tilde = self.activation(T.dot(h1_t, self._2W_in) + T.dot(r2_t * h2_tm1, self._2W_hh) + self._2b_hh)
        h2_t = (T.ones_like(z2_t) - z2_t) * h2_tm1 + z2_t * h2_tilde

        self.gen_ot = SoftMax(T.dot(h2_t, self.W_out) + self.b_out)
        
        updates[self.genh1] = h1_t
        updates[self.genh2] = h2_t
        return (self.gen_ot, updates)

    def init_params(self):
        self.W_emb = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.word_dim, self.emb_dim), name='W_emb'+self.name))
        self.W_out = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.h2_dim, self.word_dim), name='W_out'+self.name))
        self.b_out = add_to_params(self.params, theano.shared(value=np.zeros((self.word_dim,), dtype='float32'), name='b_out'+self.name))
        
        in_dim = self.emb_dim
        h_dim = self.h1_dim
        self._1W_in = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, in_dim, h_dim), name='1W_in'+self.name))
        self._1W_hh = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, h_dim, h_dim), name='1W_hh'+self.name))
        self._1b_hh = add_to_params(self.params, theano.shared(value=np.zeros((h_dim,), dtype='float32'), name='1b_hh'+self.name))
        self._1W_in_r = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, in_dim, h_dim), name='1W_in_r'+self.name))
        self._1W_in_z = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, in_dim, h_dim), name='1W_in_z'+self.name))
        self._1W_hh_r = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, h_dim, h_dim), name='1W_hh_r'+self.name))
        self._1W_hh_z = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, h_dim, h_dim), name='1W_hh_z'+self.name))
        self._1b_z = add_to_params(self.params, theano.shared(value=np.zeros((h_dim,), dtype='float32'), name='1b_z'+self.name))
        self._1b_r = add_to_params(self.params, theano.shared(value=np.zeros((h_dim,), dtype='float32'), name='1b_r'+self.name))
        
        in_dim = self.h1_dim
        h_dim = self.h2_dim
        self._2W_in = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, in_dim, h_dim), name='2W_in'+self.name))
        self._2W_hh = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, h_dim, h_dim), name='2W_hh'+self.name))
        self._2b_hh = add_to_params(self.params, theano.shared(value=np.zeros((h_dim,), dtype='float32'), name='2b_hh'+self.name))
        self._2W_in_r = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, in_dim, h_dim), name='2W_in_r'+self.name))
        self._2W_in_z = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, in_dim, h_dim), name='2W_in_z'+self.name))
        self._2W_hh_r = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, h_dim, h_dim), name='2W_hh_r'+self.name))
        self._2W_hh_z = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, h_dim, h_dim), name='2W_hh_z'+self.name))
        self._2b_z = add_to_params(self.params, theano.shared(value=np.zeros((h_dim,), dtype='float32'), name='2b_z'+self.name))
        self._2b_r = add_to_params(self.params, theano.shared(value=np.zeros((h_dim,), dtype='float32'), name='2b_r'+self.name))
        
    def approx_embedder(self, x):
        return self.W_emb[x]

    def ff_step(self, x_t, h1_tm1, h2_tm1, *args):
        r1_t = T.nnet.hard_sigmoid(T.dot(x_t, self._1W_in_r) + T.dot(h1_tm1, self._1W_hh_r) + self._1b_r)
        z1_t = T.nnet.hard_sigmoid(T.dot(x_t, self._1W_in_z) + T.dot(h1_tm1, self._1W_hh_z) + self._1b_z)
        h1_tilde = self.activation(T.dot(x_t, self._1W_in) + T.dot(r1_t * h1_tm1, self._1W_hh) + self._1b_hh)
        h1_t = (T.ones_like(z1_t) - z1_t) * h1_tm1 + z1_t * h1_tilde

        r2_t = T.nnet.hard_sigmoid(T.dot(h1_t, self._2W_in_r) + T.dot(h2_tm1, self._2W_hh_r) + self._2b_r)
        z2_t = T.nnet.hard_sigmoid(T.dot(h1_t, self._2W_in_z) + T.dot(h2_tm1, self._2W_hh_z) + self._2b_z)
        h2_tilde = self.activation(T.dot(h1_t, self._2W_in) + T.dot(r2_t * h2_tm1, self._2W_hh) + self._2b_hh)
        h2_t = (T.ones_like(z2_t) - z2_t) * h2_tm1 + z2_t * h2_tilde

        o_t = SoftMax(T.dot(h2_t, self.W_out) + self.b_out)
        return [h1_t, h2_t, o_t]

    def build_output(self, x):
        batch_size = x.shape[1]

        xe = self.approx_embedder(x)
        
        h1_0 = T.alloc(np.float32(0), batch_size, self.h1_dim)
        h2_0 = T.alloc(np.float32(0), batch_size, self.h2_dim)

        #o_t = SoftMax(T.dot(self.hs2, self.W_out) + self.b_out)
        [_, _, o_t], _ = theano.scan(self.ff_step, \
                                     sequences=[xe], \
                             outputs_info=[h1_0, h2_0, None])

        return o_t

    def build_cost(self, ot, y, mask):
        x_flatten = ot.dimshuffle(2,0,1)
        x_flatten = x_flatten.flatten(2).dimshuffle(1, 0)
        y_flatten = y.flatten()
        
        cost = x_flatten[T.arange(y_flatten.shape[0]), \
                         y_flatten]
        neg_log_cost_sum = T.sum(-T.log(cost) * mask.flatten())
        self.mask_sum = T.sum(mask.flatten())
        cost_res = neg_log_cost_sum # / self.mask_sum
        
        return cost_res

    def build_train_function(self):
        if not hasattr(self, 'train_fn'):
            self.train_fn = \
                            theano.function(inputs=[self.x_data,
                                                    self.y_data],
                                            outputs=[self.training_cost, self.mask_sum],
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
        self.genh1 =  theano.shared(np.zeros((self.h1_dim,), dtype='float32'), name='h1_gen')
        self.genh2 =  theano.shared(np.zeros((self.h2_dim,), dtype='float32'), name='h2_gen')
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
            rnd_list = []
            for ind in range(len(nw)):
                rnd_list.append((ind, nw[ind]))
            sel_ind = random_select(rnd_list)
            self.genx.set_value(np.asarray(sel_ind, dtype='int64'))
            example_sent.append(sel_ind)
            if sel_ind == 0:
                break
        return example_sent
