# -*- coding: utf-8 -*-
#!/usr/bin/env python

from data_iterator import *
from state import *
from gru import *
from utils import *

import time
import traceback
import os.path
import sys
import argparse
import cPickle
import logging
import pprint
import numpy
import collections
import signal
import math


import matplotlib
matplotlib.use('Agg')
import pylab


class Unbuffered:
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)
logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('./log/' + __name__))

### Unique RUN_ID for this execution
RUN_ID = str(time.time())

### Additional measures can be set here
measures = ["train_cost", "valid_cost"]

def init_timings():
    timings = {}
    for m in measures:
        timings[m] = []
    return timings

def save(model, timings):
    print("Saving the model...")

    # ignore keyboard interrupt while saving
    start = time.time()
    s = signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    model.save(model.state['save_dir'] + '/' + model.state['run_id'] + "_" + model.state['prefix'] + 'model.npz')
    cPickle.dump(model.state, open(model.state['save_dir'] + '/' +  model.state['run_id'] + "_" + model.state['prefix'] + 'state.pkl', 'w'))
    cPickle.dump(timings, open(model.state['save_dir'] + '/' +  model.state['run_id'] + "_" + model.state['prefix'] + 'timings.pkl', 'w'))
    signal.signal(signal.SIGINT, s)
    
    print("Model saved, took {}".format(time.time() - start))

def load(model, filename):
    print("Loading the model...")

    # ignore keyboard interrupt while saving
    start = time.time()
    s = signal.signal(signal.SIGINT, signal.SIG_IGN)
    model.load(filename)
    signal.signal(signal.SIGINT, s)

    print("Model loaded, took {}".format(time.time() - start))

def main(args):     
    logging.basicConfig(level = logging.DEBUG,
                        format = "%(asctime)s: %(name)s: %(levelname)s: %(message)s")
     
    state = eval(args.prototype)() 
    timings = init_timings() 
    
    
    if args.resume != "":
        logger.debug("Resuming %s" % args.resume)
        
        state_file = args.resume + '_state.pkl'
        timings_file = args.resume + '_timings.pkl'
        
        if os.path.isfile(state_file) and os.path.isfile(timings_file):
            logger.debug("Loading previous state")
            
            state = cPickle.load(open(state_file, 'r'))
            timings = cPickle.load(open(timings_file, 'r'))
            
        else:
            raise Exception("Cannot resume, cannot find files!")

    logger.debug("State:\n{}".format(pprint.pformat(state)))
    logger.debug("Timings:\n{}".format(pprint.pformat(timings)))
 
    model = GRU(state)
    rng = model.rng 

    if args.resume != "":
        filename = args.resume + '_model.npz'
        if os.path.isfile(filename):
            logger.debug("Loading previous model")
            load(model, filename)
        else:
            raise Exception("Cannot resume, cannot find model file!")
        
        if 'run_id' not in model.state:
            raise Exception('Backward compatibility not ensured! (need run_id in state)')           

    else:
        # assign new run_id key
        model.state['run_id'] = 'main'

    logger.debug("Compile trainer")
     
    train_cost = 0
    
    (word2ind, ind2word) = cPickle.load(open('tmp/dic.pkl'))
     

    print("Generating example...")
    example_sent = [[1]]
    model.genReset()
    for k in range(30):
        nw = model.genNext()
        example_sent.append(nw)
        if nw == 0:
            break
    gen_str = ' '.join(map(str, [ind2word[np.int(idx[0])] for idx in example_sent]))
    print()
    print(gen_str)
    print()
      
            
    logger.debug("All done, exiting...")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default="", help="Resume training from that state")
    parser.add_argument("--prototype", type=str, help="Use the prototype", default='prototype_state')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Models only run with float32
    assert(theano.config.floatX == 'float32')

    args = parse_args()
    args.resume = 'model/GRU_model'
    main(args)
