import numpy 
import os, gc
import cPickle
import copy
import logging

import threading
import Queue

import collections

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.FileHandler('./log/' + __name__))

class SSFetcher(threading.Thread):
    def __init__(self, parent):
        threading.Thread.__init__(self)
        self.parent = parent
        self.rng = numpy.random.RandomState(self.parent.seed)
        self.indexes = numpy.arange(parent.data_len)

    def run(self):
        diter = self.parent
        self.rng.shuffle(self.indexes)

        offset = 0 
        while not diter.exit_flag:
            last_batch = False
            data_x_y = []

            while len(data_x_y) < diter.batch_size:
                if offset == diter.data_len:
                    if not diter.use_infinite_loop:
                        last_batch = True
                        break
                    else:
                        # Infinite loop here, we reshuffle the indexes
                        # and reset the offset
                        self.rng.shuffle(self.indexes)
                        offset = 0

                index = self.indexes[offset]
                (_abs, _nat) = diter.data[index]

                
                offset += 1

                data_x_y.append((_abs, _nat))

            if len(data_x_y):
                diter.queue.put(data_x_y)

            if last_batch:
                diter.queue.put(None)
                return

class SSIterator(object):
    def __init__(self,
                 data_file,
                 batch_size,
                 seed=1234,
                 max_len=-1,
                 use_infinite_loop=True,
                 dtype="int32"):

        self.data_file = data_file
        self.batch_size = batch_size

        args = locals()
        args.pop("self")
        self.__dict__.update(args)
        self.load_files()
        self.exit_flag = False

    def load_files(self):
        self.data = cPickle.load(open(self.data_file))
        self.data_len = len(self.data)
        logger.debug('Data len is %d' % self.data_len)

    def start(self):
        self.exit_flag = False
        self.queue = Queue.Queue(maxsize=1000)
        self.gather = SSFetcher(self)
        self.gather.daemon = True
        self.gather.start()

    def __del__(self):
        if hasattr(self, 'gather'):
            self.gather.exitFlag = True
            self.gather.join()

    def __iter__(self):
        return self

    def next(self):
        if self.exit_flag:
            return None
        
        batch = self.queue.get()
        if not batch:
            self.exit_flag = True
        return batch
