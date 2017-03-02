import numpy
import theano
import theano.tensor as T
import sys, getopt
import logging

from state import *
# from utils import *
from SS_dataset import *

import itertools
import sys
import pickle
import cPickle
import random
import datetime

logger = logging.getLogger(__name__)

def create_padded_batch(state, data_x_y):
    (ind2word, word2ind, ontology_bool, ontology_time, ontology_str) = cPickle.load(open('tmp/dict.pkl'))
    seq_len_in = state['seq_len_in']
    seq_len_out = state['title_seq_len_out']
    n = state['bs']
    
    X = numpy.zeros((seq_len_in, n), dtype='int32')
    Xmask = numpy.zeros((seq_len_in, n), dtype='float32')
    
    title_in = numpy.zeros((title_seq_len_out, n), dtype='int32')
    title_out = numpy.zeros((title_seq_len_out, n), dtype='int32')
    titlemask = numpy.zeros((title_seq_len_out, n), dtype='float32')
    
    who_in = numpy.zeros((who_seq_len_out, n), dtype='int32')
    who_out = numpy.zeros((who_seq_len_out, n), dtype='int32')
    whomask = numpy.zeros((who_seq_len_out, n), dtype='float32')
    
    loc_in = numpy.zeros((loc_seq_len_out, n), dtype='int32')
    loc_out = numpy.zeros((loc_seq_len_out, n), dtype='int32')
    locmask = numpy.zeros((loc_seq_len_out, n), dtype='float32')
    
    whenst_hour = numpy.zeros((n, 24), dtype='int32')
    whenst_min = numpy.zeros((n, 4), dtype='int32')
    whened_hour = numpy.zeros((n, 24), dtype='int32')
    whened_min = numpy.zeros((n, 4), dtype='int32')
    dur_hour = numpy.zeros((n, 24), dtype='int32')
    dur_min = numpy.zeros((n, 4), dtype='int32')
    
    bool_Y = numpy.zeros((n, len(ontology_bool), 2), dtype='int32')
    
    for ind in range(len(data_x_y)):
        data = data_x_y[ind]
        
        cur_data = [1] + [word2ind[w] for w in data['text']] + [0]
        X[:len(cur_data), ind] = cur_data[:len(cur_data)]
        Xmask[:len(cur_data), ind] = 1
        
        cur_data = [1] + [word2ind[w] for w in data['user_inform_title']] + [0]
        title_in[:len(cur_data)-1, ind] = cur_data[:len(cur_data)-1]
        title_out[:len(cur_data)-1, ind] = cur_data[1:len(cur_data)]
        titlemask[:len(cur_data)-1, ind] = 1
        
        cur_data = [1] + [word2ind[w] for w in data['user_inform_who']] + [0]
        who_in[:len(cur_data)-1, ind] = cur_data[:len(cur_data)-1]
        who_out[:len(cur_data)-1, ind] = cur_data[1:len(cur_data)]
        whomask[:len(cur_data)-1, ind] = 1
        
        cur_data = [1] + [word2ind[w] for w in data['user_inform_where']] + [0]
        where_in[:len(cur_data)-1, ind] = cur_data[:len(cur_data)-1]
        where_out[:len(cur_data)-1, ind] = cur_data[1:len(cur_data)]
        wheremask[:len(cur_data)-1, ind] = 1
        
        whenst_hour[ind, data['user_inform_whenstart_hour']] = 1
        whenst_min[ind, data['user_inform_whenstart_min']] = 1
        
        whened_hour[ind, data['user_inform_whened_hour']] = 1
        whened_min[ind, data['user_inform_whened_min']] = 1
    
        dur_hour[ind, data['user_inform_duration_hour']] = 1
        dur_min[ind, data['user_inform_duration_min']] = 1
        
        for (ont_i, ont) in enumerate(ontology_bool):
            if data[ont] == '<YES>':
                tmp_k = 1
            else:
                tmp_k = 0
            bool_Y[ind, ont_i, tmp_k] = 1
    
    return {'X': X, 
            'Xmask': Xmask,
            'title_in': title_in,
            'title_out': title_out,
            'titlemask': titlemask,
            'who_in': who_in,
            'who_out': who_out,
            'whomask': whomask,
            'where_in': where_in,
            'where_out': where_out,
            'wheremask': wheremask,
            'whenst_hour': whenst_hour,
            'whenst_min': whenst_min,
            'whened_hour': whened_hour,
            'whened_min': whened_min,
            'dur_hour': dur_hour,
            'dur_min': dur_min,
            'bool_Y': bool_Y            
           }

class Iterator(SSIterator):
    def __init__(self, data_file, batch_size, **kwargs):
        SSIterator.__init__(self, data_file, batch_size,                   \
                            use_infinite_loop=kwargs.pop('use_infinite_loop', False))
        self.k_batches = kwargs.pop('sort_k_batches', 1)
        self.state = kwargs.pop('state', None)
        # ---------------- 
        self.batch_iter = None

    def get_homogenous_batch_iter(self, batch_size = -1):
        while True:
            batch = SSIterator.next(self)
            if not batch:
                return
            batch = create_padded_batch(self.state, batch)
            if batch:
                yield batch
    
    def start(self):
        SSIterator.start(self)
        self.batch_iter = None

    def next(self, batch_size = -1):
        """ 
        We can specify a batch size,
        independent of the object initialization. 
        """
        if not self.batch_iter:
            self.batch_iter = self.get_homogenous_batch_iter(batch_size)
        try:
            batch = next(self.batch_iter)
        except StopIteration:
            return None
        return batch

def get_train_iterator(state):
    train_data = Iterator(
        state['train_file'],
        int(state['bs']),
        state=state,
        seed=state['seed'],
        use_infinite_loop=True) 
     
    valid_data = Iterator(
        state['valid_file'],
        int(state['bs']),
        state=state,
        seed=state['seed'],
        use_infinite_loop=False)
    return train_data, valid_data 

def get_test_iterator(state):
    assert 'test_triples' in state
    test_path = state.get('test_triples')
    semantic_test_path = state.get('test_semantic', None)

    test_data = Iterator(
        test_path,
        int(state['bs']), 
        state=state,
        seed=state['seed'],
        semantic_file=semantic_test_path,
        use_infinite_loop=False)
    return test_data


# Test
if __name__=='__main__':
#    numpy.set_printoptions(threshold='nan')
#    state = {}
#    state = {'bs': 5, 'seed': 1234, 'acttype_cnt': 35, 'seqlen': 30, 'output_dim': 5, 'noise_cnt': 5}
#    state['train_file'] = './tmp/train_data_coded.pkl'
#    state['valid_file'] = './tmp/dev_data_coded.pkl'
    state = prototype_state()
    train_data, valid_data = get_train_iterator(state)
    train_data.start()

    for i in xrange(2):
        batch = train_data.next()
        print batch['ABS']
        print
        print batch['NAT']
        print
