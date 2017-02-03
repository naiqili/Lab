# -*- coding: utf-8 -*-
#!/usr/bin/env python

from data_iterator import *
from state import *
from discri_model import *
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



### Unique RUN_ID for this execution
RUN_ID = str(time.time())

### Additional measures can be set here
measures = ["train_cost", "valid_cost", "train_acc", "valid_acc"]

def init_timings():
    timings = {}
    for m in measures:
        timings[m] = []
    return timings

def save(model, timings, iters=''):
    print("Saving the model...")

    # ignore keyboard interrupt while saving
    start = time.time()
    s = signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    model.abstract_encoder.save(model.state['save_dir'] + '/' + model.state['run_id'] + '_abstract_model.npz')
    model.natural_encoder.save(model.state['save_dir'] + '/' + model.state['run_id'] + '_natural_model.npz')
    vals = dict([(model.W_emb.name, model.W_emb.get_value())])
    numpy.savez(model.state['save_dir'] + '/' + model.state['run_id'] + '_word_emb', **vals)
    cPickle.dump(model.state, open(model.state['save_dir'] + '/' +  model.state['run_id'] + '_state.pkl', 'w'))
    cPickle.dump(timings, open(model.state['save_dir'] + '/' +  model.state['run_id'] + '_timings.pkl', 'w'))
    signal.signal(signal.SIGINT, s)
    
    print("Model saved, took {}".format(time.time() - start))

def main(args):     
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.FileHandler(args.run_id))
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
 
    model = DiscriModel(state)
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


    model.state['run_id'] = args.run_id

    logger.debug("Compile trainer")
    logger.debug("Training with exact log-likelihood")
    train_batch = model.build_train_function()

    eval_batch = model.build_eval_function()
    # eval_misclass_batch = model.build_eval_misclassification_function()

    logger.debug("Load data")
    train_data, \
    valid_data, = get_train_iterator(state)
    train_data.start()
    
    # Start looping through the dataset
    step = -1
    patience = state['patience'] 
    start_time = time.time()
     
    train_cost = 0
    
    (word2ind, ind2word) = cPickle.load(open('tmp/dict.pkl'))
     
    while (step < state['loop_iters'] and
            (time.time() - start_time)/60. < state['time_stop'] and
            patience >= 0):

        step = step + 1
        # Sample stuff
        if step % 200 == 0:
            for param in model.params:
                logger.debug("%s = %.4f" % (param.name, numpy.sum(param.get_value() ** 2) ** 0.5))

        # Training phase
        batch = train_data.next() 

        # Train finished
        if not batch:
            # Restart training
            logger.debug("Got None...")
            break
        
        # logger.debug("[TRAIN] - Got batch %d" % (batch['x'].shape[1]))
        
        _abs = batch['Abs']
        _nat = batch['Nat']

        (c, acc) = train_batch(_abs, _nat)

        if numpy.isinf(c) or numpy.isnan(c):
            logger.warn("Got NaN cost .. skipping")
            continue

        train_cost = c
        timings["train_cost"].append(train_cost)
        timings["train_acc"].append(acc)
        
        this_time = time.time()
        
        if True: # step % state['train_freq'] == 0:
            elapsed = this_time - start_time
            h, m, s = ConvertTimedelta(this_time - start_time)
            logger.debug(".. %.2d:%.2d:%.2d %4d mb # %d bs %d cost = %.4f acc = %.4f" % (h, m, s,\
                                                                 state['time_stop'] - (time.time() - start_time)/60., \
                                                                 step, batch['Nat'].shape[1], float(c), acc))
        
        if valid_data is not None and step % state['valid_freq'] == 0 and step > 1:
            valid_data.start()

            logger.debug("[VALIDATION START]")
            vcost_list = []
            vacc_list = []
            
            while True:
                batch = valid_data.next()
                # Train finished
                if not batch:
                    break
                logger.debug("[VALID] - Got batch %d" % (batch['Nat'].shape[1]))

                _abs = batch['Abs']
                _nat = batch['Nat']
                
                (c, acc) = eval_batch([_abs, _nat])

                if numpy.isinf(c) or numpy.isnan(c):
                    continue
                        
                vcost_list.append(c)
                vacc_list.append(acc)
                
            valid_cost = numpy.mean(vcost_list)
            valid_acc = numpy.mean(vacc_list)

            logger.debug("[VALIDATION STEP]: %d" % step)
            logger.debug("[VALIDATION COST]: %.4f" % (valid_cost))
            logger.debug("[VALIDATION ACC]: %.4f" % (valid_acc))
            
            logger.debug("[VALIDATION END]")
            
            if len(timings["valid_cost"]) == 0 or valid_cost < numpy.min(timings["valid_cost"]):
                patience = state['patience']
                save(model, timings)
                
            elif valid_cost >= timings["valid_cost"][-1] * state['cost_threshold']:
                patience -= 1

            timings["valid_cost"].append(valid_cost)
            timings["valid_acc"].append(valid_acc)

            # Reset train cost, train misclass and train done
            train_cost = 0

            logger.debug("[VALIDATION COST]: %f" % valid_cost)

            # Plot histogram over validation costs
            try:
                pylab.figure()
                pylab.subplot(2,1,1)
                pylab.title("Training Cost")
                pylab.plot(timings["train_cost"])
                pylab.subplot(2,1,2)
                pylab.title("Validation Cost")
                pylab.plot(timings["valid_cost"])
                pylab.savefig('log/' + args.run_id + '_cost.png')
                pylab.close()
                pylab.figure()
                pylab.subplot(2,1,1)
                pylab.title("Training Accuracy")
                pylab.plot(timings["train_acc"])
                pylab.subplot(2,1,2)
                pylab.title("Validation Accuracy")
                pylab.plot(timings["valid_acc"])
                pylab.savefig('log/' + args.run_id + '_acc.png')
                pylab.close()
            except:
                pass
            
    logger.debug("All done, exiting...")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default="", help="Resume training from that state")
    parser.add_argument("--prototype", type=str, help="Use the prototype", default='prototype_state')
    parser.add_argument("--run_id", type=str, help="RUN_ID", default='main')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Models only run with float32
    assert(theano.config.floatX == 'float32')

    args = parse_args()
    args.run_id = 'trainemb_emb100_h100'
    args.prototype = 'simple_state'
    #args.resume = 'model/GRU_overtrain_model'
    main(args)
