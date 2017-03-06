# -*- coding: utf-8 -*-
#!/usr/bin/env python

from data_iterator import *
from state import *
from whenst_min_model import *
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

def save(model, timings, postfix=''):
    print("Saving the model...")

    # ignore keyboard interrupt while saving
    start = time.time()
    s = signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    model.save(model.state['save_dir'] + '/' + model.state['run_id'] + '_model'+ postfix + '.npz')
    cPickle.dump(model.state, open(model.state['save_dir'] + '/' +  model.state['run_id'] + '_state' + postfix +'.pkl', 'w'))
    cPickle.dump(timings, open(model.state['save_dir'] + '/' +  model.state['run_id'] + '_timings' + postfix + '.pkl', 'w'))
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
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.FileHandler('log/' + args.run_id))
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
            
            #state = cPickle.load(open(state_file, 'r'))
            #timings = cPickle.load(open(timings_file, 'r'))
            
        else:
            raise Exception("Cannot resume, cannot find files!")

    logger.debug("State:\n{}".format(pprint.pformat(state)))
    logger.debug("Timings:\n{}".format(pprint.pformat(timings)))
 
    model = WhenstMinModel(state)
    rng = model.rng 

    if args.resume != "":
        filename = args.resume + '_model.npz'
        if os.path.isfile(filename):
            logger.debug("Loading previous model")
            load(model, filename)
        else:
            raise Exception("Cannot resume, cannot find model file!")
        
        if 'run_id' not in model.state:
            print('Backward compatibility not ensured! (need run_id in state)')           


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
    
    # (word2ind, ind2word) = cPickle.load(open('tmp/dict.pkl'))
     
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
        
        X = batch['X']
        Xmask = batch['Xmask']
        whenst_min = batch['whenst_min']

        (c, acc) = train_batch(X, Xmask, whenst_min)
        #print 'Pred:', pred
        #print 'y_flatten:', y_flatten

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
                                                                                         step, batch['X'].shape[1], \
                                                                                         float(c), float(acc)))
        
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
                logger.debug("[VALID] - Got batch %d" % (batch['X'].shape[1]))

                X = batch['X']
                Xmask = batch['Xmask']
                whenst_min = batch['whenst_min']
                
                (c, acc) = eval_batch(X, Xmask, whenst_min)
                

                if numpy.isinf(c) or numpy.isnan(c):
                    continue
                        
                vcost_list.append(c)
                vacc_list.append(acc)
                
            valid_cost = numpy.mean(vcost_list)
            valid_cost = 1.0 * valid_cost
            valid_acc = numpy.mean(vacc_list)

            logger.debug("[VALIDATION STEP]: %d" % step)
            logger.debug("[VALIDATION COST]: %.4f" % (valid_cost))
            logger.debug("[VALIDATION ACC]: %.4f" % (valid_acc))
            
            logger.debug("[VALIDATION END]")
            
            if len(timings["valid_cost"]) == 0 or valid_cost < numpy.min(timings["valid_cost"]):
                patience = state['patience']
                logger.debug('Better model saved.')
                save(model, timings)
                
            elif valid_cost >= timings["valid_cost"][-1] * state['cost_threshold']:
                save(model, timings, postfix='_overfit')
                logger.debug('Overfit model saved.')
                patience -= 1

            timings["valid_cost"].append(valid_cost)
            timings["valid_acc"].append(valid_acc)

            # Reset train cost, train misclass and train done
            train_cost = 0

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
    args.run_id = 'whenst_min_emb256_h256_f32'
    args.prototype = 'whenst_min_state'
    args.resume = 'model/whenst_hour_emb256_h256_f32'
    main(args)
