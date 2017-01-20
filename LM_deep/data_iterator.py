import numpy
import theano
import theano.tensor as T
import sys, getopt
import logging

# from state import *
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
    
    X = numpy.zeros((mx, n), dtype='int32')
    Y = numpy.zeros((mx, n), dtype='int32')
    
    for idx in xrange(len(x[0])):
        # Insert sequence idx in a column of matrix X
        sent_length = len(x[0][idx]) 

        # Fiddle-it if it is too long ..
        if mx < sent_length: 
            continue

        X[:sent_length-1, idx] = x[0][idx][:sent_length-1]
        Y[:sent_length-1, idx] = x[0][idx][1:sent_length]
    
    return {'x': X,                                                 \
            'y': Y
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
            batch_size = self.batch_size if (batch_size == -1) else batch_size 
           
            data = []
            for k in range(self.k_batches):
                batch = SSIterator.next(self)
                if batch:
                    data.append(batch)
            if not len(data):
                return
            
            number_of_batches = len(data)
            # data is a set of batches
            # After chain.from_iterable, is a batch
            data = list(itertools.chain.from_iterable(data))
            
            data_x = []
            for i in range(len(data)):
                data_x.append(data[i])

            x = numpy.asarray(list(itertools.chain(data_x)))

            lens = numpy.asarray([map(len, x)])
            order = numpy.argsort(lens.max(axis=0))
                 
            for k in range(number_of_batches):
                indices = order[k * batch_size:(k + 1) * batch_size]
                batch = create_padded_batch(self.state, [x[indices]])

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
    state = {}
    state = {'bs': 7, 'seed': 1234, 'seqlen': 30, 'output_dim': 5}
    state['train_file'] = './tmp/train_data.txt'
    state['dev_file'] = './tmp/dev_data.txt'
    train_data = Iterator(state['train_file'],
                          int(state['bs']),
                          state=state,
                          use_infinite_loop=True)
    train_data.start()

    for i in xrange(1):
        batch = train_data.next()
        print(batch)
