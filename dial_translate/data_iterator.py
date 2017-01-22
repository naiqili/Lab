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
import random
import datetime

logger = logging.getLogger(__name__)

def create_padded_batch(state, x):
    mx = state['seqlen']
    n = state['bs']
    
    Nat = numpy.zeros((mx, n), dtype='int32')
    Abs = numpy.zeros((n, state['acttype_cnt']), dtype='int32')
    NoiseRows = []

    (dataxy, _noises) = x
    
    for idx in xrange(len(dataxy)):
        (_abs, _nat) = dataxy[idx]
        
        sent_length = len(_nat)

        Nat[:sent_length, idx] = _nat[:sent_length]
        Abs[idx, :len(_abs)] = _abs

    for noise in _noises:
        noiserow = numpy.asarray([noise], dtype='int32')
        NoiseRows.append(noiserow)
    
    return {'Nat': Nat,                                                 \
            'Abs': Abs,
            'NoiseRows': NoiseRows
           }

class Iterator(SSIterator):
    def __init__(self, data_file, batch_size, **kwargs):
        SSIterator.__init__(self, data_file, batch_size,                   \
                            max_len=kwargs.pop('max_len', -1),               \
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
        use_infinite_loop=True,
        max_len=state['seqlen']) 
     
    valid_data = Iterator(
        state['valid_file'],
        int(state['bs']),
        state=state,
        seed=state['seed'],
        use_infinite_loop=False,
        max_len=state['seqlen'])
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
        use_infinite_loop=False,
        max_len=state['seqlen'])
    return test_data


# Test
if __name__=='__main__':
    numpy.set_printoptions(threshold='nan')
#    state = {}
#    state = {'bs': 5, 'seed': 1234, 'acttype_cnt': 35, 'seqlen': 30, 'output_dim': 5, 'noise_cnt': 5}
#    state['train_file'] = './tmp/train_data_coded.pkl'
#    state['valid_file'] = './tmp/dev_data_coded.pkl'
    state = prototype_state()
    train_data, valid_data = get_train_iterator(state)
    train_data.start()

    for i in xrange(2):
        batch = train_data.next()
        print batch['Abs']
        print
        print batch['Nat']
        print
        print len(batch['NoiseRows'])
        print batch['NoiseRows']

