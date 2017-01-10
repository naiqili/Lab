"""
Dialog hierarchical encoder-decoder code.
The code is inspired from nmt encdec code in groundhog
but we do not rely on groundhog infrastructure.
"""
__docformat__ = 'restructedtext en'
__authors__ = ("Alessandro Sordoni, Iulian Vlad Serban")
__contact__ = "Alessandro Sordoni <sordonia@iro.umontreal>"

import theano
import theano.tensor as T
import numpy as np
import cPickle
import logging
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

class EncoderDecoderBase():
    def __init__(self, state, rng, parent):
        self.rng = rng
        self.parent = parent
        
        self.state = state
        self.__dict__.update(state)
        
        self.triple_rec_activation = eval(self.triple_rec_activation)
        self.sent_rec_activation = eval(self.sent_rec_activation)
         
        self.params = []

class UtteranceEncoder(EncoderDecoderBase):
    def init_params(self, word_embedding_param):
        # Initialzie W_emb to given word embeddings
        assert(word_embedding_param != None)
        self.W_emb = word_embedding_param

        """ sent weights """
        self.W_in = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim), name='W_in'+self.name))
        self.W_hh = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.qdim, self.qdim), name='W_hh'+self.name))
        self.b_hh = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim,), dtype='float32'), name='b_hh'+self.name))
        
        if self.sent_step_type == "gated":
            self.W_in_r = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim), name='W_in_r'+self.name))
            self.W_in_z = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim), name='W_in_z'+self.name))
            self.W_hh_r = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.qdim, self.qdim), name='W_hh_r'+self.name))
            self.W_hh_z = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.qdim, self.qdim), name='W_hh_z'+self.name))
            self.b_z = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim,), dtype='float32'), name='b_z'+self.name))
            self.b_r = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim,), dtype='float32'), name='b_r'+self.name))

    def plain_sent_step(self, x_t, m_t, *args):
        args = iter(args)
        h_tm1 = next(args)

        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')
         
        hr_tm1 = m_t * h_tm1
        h_t = self.sent_rec_activation(T.dot(x_t, self.W_in) + T.dot(hr_tm1, self.W_hh) + self.b_hh)

        return [h_t]

    def gated_sent_step(self, x_t, m_t, *args):
        args = iter(args)
        h_tm1 = next(args)

        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x') 
         
        hr_tm1 = m_t * h_tm1

        r_t = T.nnet.sigmoid(T.dot(x_t, self.W_in_r) + T.dot(hr_tm1, self.W_hh_r) + self.b_r)
        z_t = T.nnet.sigmoid(T.dot(x_t, self.W_in_z) + T.dot(hr_tm1, self.W_hh_z) + self.b_z)
        h_tilde = self.sent_rec_activation(T.dot(x_t, self.W_in) + T.dot(r_t * hr_tm1, self.W_hh) + self.b_hh)
        h_t = (np.float32(1.0) - z_t) * hr_tm1 + z_t * h_tilde
        
        return [h_t, r_t, z_t, h_tilde]

    def approx_embedder(self, x):
        return self.W_emb[x]

    def build_encoder(self, x, xmask=None, return_L2_pooling = False, **kwargs):
        one_step = False
        if len(kwargs):
            one_step = True
         
        # if x.ndim == 2 then 
        # x = (n_steps, batch_size)
        if x.ndim == 2:
            batch_size = x.shape[1]
        # else x = (word_1, word_2, word_3, ...)
        # or x = (last_word_1, last_word_2, last_word_3, ..)
        # in this case batch_size is 
        else:
            batch_size = 1

        # if it is not one_step then we initialize everything to 0  
        if not one_step:
            h_0 = T.alloc(np.float32(0), batch_size, self.qdim)
        # in sampling mode (i.e. one step) we require 
        else:
            # in this case x.ndim != 2
            assert x.ndim != 2
            assert 'prev_h' in kwargs 
            h_0 = kwargs['prev_h']

        xe = self.approx_embedder(x)
        if xmask == None:
            xmask = T.neq(x, self.eos_sym)
        
        # Here we roll the mask so we avoid the need for separate
        # hr and h. The trick is simple: if the original mask is
        # 0 1 1 0 1 1 1 0 0 0 0 0 -- batch is filled with eos_sym
        # the rolled mask will be
        # 0 0 1 1 0 1 1 1 0 0 0 0 -- roll to the right
        # ^ ^
        # two resets </s> <s>
        # the first reset will reset h_init = 0
        # the second will reset </s> and update given x_t = <s>
        if xmask.ndim == 2:
            rolled_xmask = T.roll(xmask, 1, axis=0)
        else:
            rolled_xmask = T.roll(xmask, 1) 

        # Gated Encoder
        if self.sent_step_type == "gated":
            f_enc = self.gated_sent_step
            if return_L2_pooling:
                o_enc_info = [h_0, h_0_pooled, pool_len_0, None, None, None]
            else:
                o_enc_info = [h_0, None, None, None]

        else:
            f_enc = self.plain_sent_step
            o_enc_info = [h_0]


        # Run through all the sentence (encode everything)
        if not one_step: 
            _res, _ = theano.scan(f_enc,
                              sequences=[xe, rolled_xmask],\
                              outputs_info=o_enc_info)
        else: # Make just one step further
            _res = f_enc(xe, rolled_xmask, [h_0])[0]

        # Get the hidden state sequence
        h = _res[0]
        return h

    def __init__(self, state, rng, word_embedding_param, parent, name):
        EncoderDecoderBase.__init__(self, state, rng, parent)
        self.name = name
        self.init_params(word_embedding_param)

class DialogEncoder(EncoderDecoderBase):
    def init_params(self):
        """ Context weights """

        input_dim = self.qdim
        
        self.Ws_in = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, input_dim, self.sdim), name='Ws_in'))
        self.Ws_hh = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.sdim, self.sdim), name='Ws_hh'))
        self.bs_hh = add_to_params(self.params, theano.shared(value=np.zeros((self.sdim,), dtype='float32'), name='bs_hh')) 
         
        if self.triple_step_type == "gated":
            self.Ws_in_r = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, input_dim, self.sdim), name='Ws_in_r'))
            self.Ws_in_z = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, input_dim, self.sdim), name='Ws_in_z'))
            self.Ws_hh_r = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.sdim, self.sdim), name='Ws_hh_r'))
            self.Ws_hh_z = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.sdim, self.sdim), name='Ws_hh_z'))
            self.bs_z = add_to_params(self.params, theano.shared(value=np.zeros((self.sdim,), dtype='float32'), name='bs_z'))
            self.bs_r = add_to_params(self.params, theano.shared(value=np.zeros((self.sdim,), dtype='float32'), name='bs_r'))
    
    def plain_triple_step(self, h_t, m_t, hs_tm1):
        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')

        hs_tilde = self.triple_rec_activation(T.dot(h_t, self.Ws_in) + T.dot(hs_tm1, self.Ws_hh) + self.bs_hh) 
        hs_t = (m_t) * hs_tm1 + (1 - m_t) * hs_tilde 
        return hs_t

    def gated_triple_step(self, h_t, m_t, hs_tm1):
        rs_t = T.nnet.sigmoid(T.dot(h_t, self.Ws_in_r) + T.dot(hs_tm1, self.Ws_hh_r) + self.bs_r)
        zs_t = T.nnet.sigmoid(T.dot(h_t, self.Ws_in_z) + T.dot(hs_tm1, self.Ws_hh_z) + self.bs_z)
        hs_tilde = self.triple_rec_activation(T.dot(h_t, self.Ws_in) + T.dot(rs_t * hs_tm1, self.Ws_hh) + self.bs_hh)
        hs_update = (np.float32(1.) - zs_t) * hs_tm1 + zs_t * hs_tilde
         
        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')
         
        hs_t = (m_t) * hs_tm1 + (1 - m_t) * hs_update
        return hs_t, hs_tilde, rs_t, zs_t

    def build_encoder(self, h, x, xmask=None, **kwargs):
        one_step = False
        if len(kwargs):
            one_step = True
         
        # if x.ndim == 2 then 
        # x = (n_steps, batch_size)
        if x.ndim == 2:
            batch_size = x.shape[1]
        # else x = (word_1, word_2, word_3, ...)
        # or x = (last_word_1, last_word_2, last_word_3, ..)
        # in this case batch_size is 
        else:
            batch_size = 1
        
        # if it is not one_step then we initialize everything to 0  
        if not one_step:
            hs_0 = T.alloc(np.float32(0), batch_size, self.sdim) 
        # in sampling mode (i.e. one step) we require 
        else:
            # in this case x.ndim != 2
            assert x.ndim != 2
            assert 'prev_hs' in kwargs
            hs_0 = kwargs['prev_hs']

        if xmask == None:
            xmask = T.neq(x, self.eos_sym)
        
        # Here we roll the mask so we avoid the need for separate
        # hr and h. The trick is simple: if the original mask is
        # 0 1 1 0 1 1 1 0 0 0 0 0 -- batch is filled with eos_sym
        # the rolled mask will be
        # 0 0 1 1 0 1 1 1 0 0 0 0 -- roll to the right
        # ^ ^
        # two resets </s> <s>
        # the first reset will reset h_init = 0
        # the second will reset </s> and update given x_t = <s>
        if xmask.ndim == 2:
            rolled_xmask = T.roll(xmask, 1, axis=0)
        else:
            rolled_xmask = T.roll(xmask, 1)

        if self.triple_step_type == "gated":
            f_hier = self.gated_triple_step
            o_hier_info = [hs_0, None, None, None]
        else:
            f_hier = self.plain_triple_step
            o_hier_info = [hs_0]
        
        # All hierarchical sentence
        # The hs sequence is based on the original mask
        if not one_step:
            _res,  _ = theano.scan(f_hier,\
                               sequences=[h, xmask],\
                               outputs_info=o_hier_info)
        # Just one step further
        else:
            _res = f_hier(h, xmask, hs_0)

        if isinstance(_res, list) or isinstance(_res, tuple):
            hs = _res[0]
        else:
            hs = _res

        return hs 

    def __init__(self, state, rng, parent):
        EncoderDecoderBase.__init__(self, state, rng, parent)
        self.init_params()

class SimpleDecoder(EncoderDecoderBase):
    def __init__(self, state, rng, parent):
        EncoderDecoderBase.__init__(self, state, rng, parent)
        self.trng = MRG_RandomStreams(self.seed)
        self.init_params()

    def init_params(self):
        self.input_dim = self.sdim
        self.Wdec = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.input_dim, self.output_dim), name='Wdec'))
        self.bdec = add_to_params(self.params, theano.shared(value=np.zeros((self.output_dim, ), dtype=theano.config.floatX), name='bdec'))

    def simple_step(self, hs):
        o_t = self.triple_rec_activation(T.dot(hs, self.Wdec) + self.bdec)
        o_t = SoftMax(o_t)
        return o_t

    def build_output_layer(self, hs, xmask):
        one_step = False
        if not one_step:
            _res, _ = theano.scan(self.simple_step,
                                  sequences=[hs])
        else:
            _res = simple_step(hs)

        return _res


class EncoderDecoder(Model):
    def __init__(self, state):
        Model.__init__(self)
        self.state = state
        self.global_params = []
        self.__dict__.update(state)
        self.rng = numpy.random.RandomState(state['seed'])

        self.W_emb = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.vocab_size, self.rankdim), name='W_emb'))
        
        self.x_data = T.imatrix('x_data')
        self.y_data = T.imatrix('y_data')

        training_mask = T.neq(self.x_data, self.eot_sym)

        self.utterance_encoder = UtteranceEncoder(self.state, self.rng, self.W_emb, self, 'fwd')
        self.h = self.utterance_encoder.build_encoder(self.x_data, xmask=training_mask)

        self.dialog_encoder = DialogEncoder(self.state, self.rng, self)
        self.hs = self.dialog_encoder.build_encoder(self.h, self.x_data, xmask=training_mask)

        self.simple_decoder = SimpleDecoder(self.state, self.rng, self)
        self.ot = self.simple_decoder.build_output_layer(self.hs, training_mask)

        self.params = self.global_params + self.utterance_encoder.params + self.dialog_encoder.params + self.simple_decoder.params
        
        self.training_cost = self.build_cost(self.ot, self.y_data, training_mask)
        self.updates = self.compute_updates(self.training_cost, self.params)
        self.y_pred = self.ot.argmax(axis=2) # See lab/argmax_test.py
        self.accuracy = T.mean(T.sum(T.eq(self.y_pred, self.y_data) * (1 - training_mask), axis=0) / \
                               T.sum(1 - training_mask, axis=0))

    def build_cost(self, ot, y, mask):
        # See lab/dimshuffle_test.py for why I do this
        nmask = 1 - mask
        x_flatten = ot.dimshuffle(2,0,1).flatten(2).dimshuffle(1, 0)
        y_flatten = y.flatten()
        
        cost = x_flatten[T.arange(y_flatten.shape[0]), \
                         y_flatten]
        neg_log_cost = -T.log(cost) * nmask.flatten()
        cost = cost.reshape(y.shape)
        sum_neg_log_cost = T.sum(neg_log_cost, axis=0)
        s_row = T.sum(nmask, axis=0)
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
                                           outputs=[self.training_cost, self.accuracy],
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
