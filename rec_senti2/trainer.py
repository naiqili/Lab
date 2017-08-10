import matplotlib

import tensorflow as tf
import numpy as np
import tensorflow_fold as td
import logging
import cPickle
import os, shutil, random
from os import path

import config, utils
from utils import *
from lstm_model import LSTMModel

matplotlib.use('Agg')

import pylab
import pprint

class Trainer():
    def __init__(self, FLAGS, model_proto):
        """Required fields in FLAGS:
              model_name
              train_path
              valid_path
              test_path
              
              patience
              max_epochs              
              batch_size
              
              bestmodel_dir
              log_dir
              
              load_model
              save_model
              """
        self.FLAGS, self.model = FLAGS, model
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level = logging.DEBUG,
                            format = "%(asctime)s: %(name)s: %(levelname)s: %(message)s")
        self.logger.addHandler(logging.FileHandler(FLAGS.log_dir+'log.txt'))
        self.logger.debug(pprint.pformat(FLAGS.__dict__['__flags']))
        
        self.train_trees = load_trees(FLAGS.train_path)
        self.valid_trees = load_trees(FLAGS.valid_path)
        self.test_trees = load_trees(FLAGS.test_path)

        self.model = model_proto(FLAGS.__dict__['__flags'])
        self.model.build_model()
        compiler = self.model.compiler
        
        self.train_set = compiler.build_loom_inputs(self.train_trees)
        self.dev_feed_dict = compiler.build_feed_dict(self.valid_trees)
        
        self._patience = FLAGS.patience        
        self.best_valid_acc = 0.0
        
    def report_figure(self, valid_history, train_history):
        pass
        
    def train_step(self, batch):
        compiler = self.model.compiler
        train_feed_dict= {compiler.loom_input_tensor: batch, self.model.keep_prob_ph: self.FLAGS.keep_prob}
        _, batch_loss = self.sess.run([self.model.train_op, self.model.loss], train_feed_dict)
        return batch_loss
        
    def train_epoch(self, train_set):
        return sum([self.train_step(batch) for batch in td.group_by_batches(train_set, self.FLAGS.batch_size)])    
        
    def validate(self, epoch):
        logger = self.logger
        logger.debug("Start validation")
        
        valid_metrics = self.sess.run(self.model.metrics, self.dev_feed_dict)
        valid_loss = valid_metrics['all_loss']
        valid_acc = ['%s: %.4f' % (k, v * 100) for k, v in
                 sorted(valid_metrics.items()) if k.endswith('hits')]
        
        logger.debug('Validation: finish')

        print('epoch:%4d, dev_loss_avg: %f, dev_accuracy:\n  [%s]'
              % (epoch, valid_loss, ' '.join(valid_acc)))
        return valid_metrics['root_binary_hits']

    def run(self):
        logger = self.logger
        with tf.Session() as sess:
            self.sess = sess
            if self.FLAGS.load_model:
                tf.train.Saver().restore(sess, self.FLAGS.bestmodel_dir)
            else:
                tf.global_variables_initializer().run()

            for epoch, shuffled in enumerate(td.epochs(self.train_set, self.FLAGS.max_epochs), 1):
                if self._patience = 0:
                    return self.best_valid_acc
                train_loss = self.train_epoch(shuffled)
                print('epoch:%4d, train_loss_avg: %f' % (epoch, train_loss))
                acc = self.validate(epoch)
                if acc > self.best_valid_acc:
                    self._patience = self.FLAGS.patience
                    self.best_valid_acc = acc
                    if self.FLAGS.save_model:
                        saver = tf.train.Saver()
                        saver.save(sess, self.FLAGS.bestmodel_dir)
                    logger.debug('Better model')
                else:                    
                    self._patience -= 1
                    logger.debug('Not improved. Patience: %d' % self._patience)

            
if __name__=='__main__':
    flags = tf.flags

    flags.DEFINE_string("model_name", None, "word vec embedding path")
    flags.DEFINE_string("wv_embed_path", config.EMBEDDING_MAT_PATH, "word vec embedding path")
    flags.DEFINE_string("word_idx_path", config.WORD_IDX_PATH, "word dict path")
    flags.DEFINE_integer("lstm_num_units", None, "word dict path")
    flags.DEFINE_integer("num_classes", 5, "word dict path")
    flags.DEFINE_float("lr", None, "word dict path")
    flags.DEFINE_float("embed_lr_factor", 0.1, "word dict path")
    flags.DEFINE_float("keep_prob", None, "word dict path")
    
    flags.DEFINE_string("train_path", config.TRAIN_PATH, "word dict path")
    flags.DEFINE_string("valid_path", config.VALID_PATH, "word dict path")
    flags.DEFINE_string("test_path", config.TEST_PATH, "word dict path")
    flags.DEFINE_integer("patience", None, "word dict path")
    flags.DEFINE_integer("max_epochs", None, "word dict path")
    flags.DEFINE_integer("batch_size", None, "word dict path")
    flags.DEFINE_string("bestmodel_dir", None, "word dict path")
    flags.DEFINE_string("log_dir", None, "word dict path")
    
    flags.DEFINE_boolean("load_model", None, "word dict path")
    flags.DEFINE_boolean("save_model", None, "word dict path")

    FLAGS = flags.FLAGS
    
    model = eval(FLAGS.model_name)
    trainer = Trainer(FLAGS, model)
    best_valid_acc = trainer.run()