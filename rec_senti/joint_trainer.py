import matplotlib

import tensorflow as tf
import numpy as np
import logging
import cPickle
import os, shutil, random
from os import path

from tfrecord_reader import get_data

from lstm_model import LSTMModel
from lstm_attn_model import LSTMAttnModel
from rnn_attn_model import RNNAttnModel
from rnn_attn2_model import RNNAttn2Model

matplotlib.use('Agg')

import pylab
import pprint

class Trainer():
    def __init__(self, FLAGS, model, config_dict):
        self.FLAGS, self.model, self.config_dict = FLAGS, model, config_dict
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level = logging.DEBUG,
                            format = "%(asctime)s: %(name)s: %(levelname)s: %(message)s")
        self.logger.addHandler(logging.FileHandler(FLAGS.logdir+'log.txt'))
        self.logger.debug(pprint.pformat(config_dict))
        
        train_md = model(config_dict)
        train_md.is_training = True
        train_md.add_variables(reuse=False)
        valid_md = model(config_dict)
        valid_md.is_training = False
        valid_md.add_variables(reuse=True)

        l_tts, wv_tts, left_tts, right_tts, target_tts, is_leaf_tts, mask_tts = get_data(filename=FLAGS.train_record, shuffle=True, mask_type=FLAGS.mask_type)
        train_md.build_model(left_tts, right_tts, wv_tts, target_tts, is_leaf_tts, l_tts, mask_tts)

        l_vts, wv_vts, left_vts, right_vts, target_vts, is_leaf_vts, mask_vts = get_data(filename=FLAGS.valid_record, mask_type=FLAGS.mask_type)
        valid_md.build_model(left_vts, right_vts, wv_vts, target_vts, is_leaf_vts, l_vts, mask_vts)
        
        self.train_md = train_md
        self.valid_md = valid_md

        self._patience = FLAGS.patience
        self.batch_train_history = []
        
        if FLAGS.load_model:
            self.init_step = cPickle.load(open(path.join(self.FLAGS.bestmodel_dir, 'stat.pkl')))
        else:  
            self.init_step = -1
            
        self.best_valid_loss = 10000000.0
        self.best_valid_overall_acc = 0.0
        self.best_valid_root_acc = 0.0
        self.best_valid_overall_binary_acc = 0.0
        self.best_valid_root_binary_acc = 0.0
        self.valid_history = []
        self.train_history = []  
                
    def validate(self, sess, _step):
        logger = self.logger
        logger.debug("Start validation")
        valid_losses = []
        overall_metrics = np.zeros((self.FLAGS.class_size, self.FLAGS.class_size), dtype=np.int32)
        root_metrics = np.zeros((self.FLAGS.class_size, self.FLAGS.class_size), dtype=np.int32)
        overall_binary_metrics = np.zeros((2, 2), dtype=np.int32)
        root_binary_metrics = np.zeros((2, 2), dtype=np.int32)

        for i in xrange(self.FLAGS.valid_size):
            valid_loss, valid_pred, valid_binary_pred, target_v = \
                sess.run([self.valid_md.sum_loss, self.valid_md.pred, self.valid_md.binary_pred, self.valid_md.ground_truth])
            valid_losses.append(valid_loss)
            root_pred = valid_pred[-1]
            root_target = target_v[-1]
            if target_v[-1] != 2:
                root_binary_target = (target_v[-1] > 2).astype(np.int32)
                root_binary_pred = valid_binary_pred[-1]
                root_binary_metrics[root_binary_pred, root_binary_target] += 1
            root_metrics[root_pred, root_target] += 1
            for k in range(len(valid_pred)):
                overall_metrics[valid_pred[k], target_v[k]] += 1
                target_temp = (target_v[k] > 2).astype(np.int32)
                if not target_v[k] == 2:
                    overall_binary_metrics[valid_binary_pred[k], target_temp] += 1
            #logger.debug("Validation loss %f" % valid_loss)

        sum_loss = sum(valid_losses)
        mean_loss = sum_loss / self.FLAGS.valid_size
        logger.debug('Validation: finish')

        logger.debug('Root Metrics:\n %s' % str(root_metrics))
        logger.debug('Overall Metrics:\n %s' % str(overall_metrics))
        logger.debug('Root Binary Metrics:\n %s' % str(root_binary_metrics))
        logger.debug('Overall Binary Metrics:\n %s' % str(overall_binary_metrics))
        root_acc = 1.0 * np.trace(root_metrics) / np.sum(root_metrics)
        overall_acc = 1.0 * np.trace(overall_metrics) / np.sum(overall_metrics)
        root_binary_acc = 1.0 * np.trace(root_binary_metrics) / np.sum(root_binary_metrics)
        overall_binary_acc = 1.0 * np.trace(overall_binary_metrics) / np.sum(overall_binary_metrics)
        logger.debug('Step: %d Validation: mean loss: %f, root_acc: %f, overall_acc: %f, root_binary_acc: %f, overall_binary_acc: %f' % (_step, mean_loss, root_acc, overall_acc, root_binary_acc, overall_binary_acc))

        self.valid_history.append([[_step, mean_loss], [_step, overall_acc]])
        
        return mean_loss, root_acc, overall_acc, root_binary_acc, overall_binary_acc

    def run(self):
        logger = self.logger        
        self.train_writer = train_writer = tf.summary.FileWriter(path.join(self.FLAGS.logdir, 'train'))
        self.valid_writer = valid_writer = tf.summary.FileWriter(path.join(self.FLAGS.logdir, 'valid'))
        with tf.Session() as sess:
            if self.FLAGS.load_model:
                tf.train.Saver().restore(sess, self.FLAGS.bestmodel_dir)
            else:
                tf.global_variables_initializer().run()

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            for _step in xrange(self.init_step+1, self.FLAGS.max_step):
                if self._patience == 0:
                    break
                loss_ts = self.train_md.sum_loss
                train_op = self.train_md.train_op
                train_loss, _ = sess.run([loss_ts, train_op])
                self.batch_train_history.append(train_loss)
                summary=tf.Summary()
                summary.value.add(tag='loss', simple_value = train_loss)
                train_writer.add_summary(summary, _step)
                train_writer.flush()

                if _step % self.FLAGS.train_freq == 0:
                    batch_train_loss = np.mean(self.batch_train_history)
                    logger.debug('Current training loss: %f' % (batch_train_loss))

                if _step != 0 and _step % self.FLAGS.valid_freq == 0:
                    batch_train_loss = sum(self.batch_train_history) / self.FLAGS.valid_freq
                    logger.debug('Step: %d Training batch loss: %f' % (_step, batch_train_loss))
                    self.batch_train_history = []
                    self.train_history.append([_step, batch_train_loss])
                    
                    mean_loss, root_acc, overall_acc, root_binary_acc, overall_binary_acc = self.validate(sess, _step)
                    logger.debug('Current loss: %f, root_acc: %f, overall_acc: %f, root_binary_acc: %f, overall_binary_acc: %f' % (mean_loss, root_acc, overall_acc, root_binary_acc, overall_binary_acc))
                    summary=tf.Summary()
                    summary.value.add(tag='loss', simple_value = mean_loss)
                    summary.value.add(tag='root_acc', simple_value = root_acc)
                    summary.value.add(tag='overall_acc', simple_value = overall_acc)
                    summary.value.add(tag='root_binary_acc', simple_value = root_binary_acc)
                    summary.value.add(tag='overall_binary_acc', simple_value = overall_binary_acc)
                    valid_writer.add_summary(summary, _step)
                    valid_writer.flush()
                    
                    if mean_loss < self.best_valid_loss or root_acc > self.best_valid_root_acc or overall_acc > self.best_valid_overall_acc or \
                        root_binary_acc > self.best_valid_root_binary_acc or overall_binary_acc > self.best_valid_overall_binary_acc:
                        self.best_valid_loss = min(mean_loss, self.best_valid_loss)
                        self.best_valid_root_acc = max(root_acc, self.best_valid_root_acc)
                        self.best_valid_overall_acc = max(overall_acc, self.best_valid_overall_acc)
                        self.best_valid_root_binary_acc = max(root_binary_acc, self.best_valid_root_binary_acc)
                        self.best_valid_overall_binary_acc = max(overall_binary_acc, self.best_valid_overall_binary_acc)
                        self._patience = self.FLAGS.patience
                        if self.FLAGS.save_model:
                            saver = tf.train.Saver()
                            saver.save(sess, self.FLAGS.bestmodel_dir)
                            cPickle.dump(_step, open(path.join(self.FLAGS.bestmodel_dir, 'stat.pkl'), 'w'))
                        logger.debug('Better model')
                    else:                    
                        self._patience -= 1
                        logger.debug('Not improved. Patience: %d' % self._patience)
                    logger.debug('Best loss: %f, root_acc: %f, overall_acc: %f, root_binary_acc: %f, overall_binary_acc: %f' % (self.best_valid_loss, self.best_valid_root_acc, self.best_valid_overall_acc, self.best_valid_root_binary_acc, self.best_valid_overall_binary_acc))
            coord.request_stop()
            coord.join(threads)
            
if __name__=='__main__':
    flags = tf.flags

    flags.DEFINE_integer("wv_vocab_size", 50002, "Vocab size")
    flags.DEFINE_integer("embed_size", 300, "Embedding size")
    flags.DEFINE_integer("class_size", 3, "3 for binary; 5 for fine grained")
    flags.DEFINE_integer("fw_cell_size", 50, "Composition cell size")
    flags.DEFINE_integer("bw_cell_size", 50, "Decomposition cell size")
    flags.DEFINE_integer("attn_size", 50, "Attention size")
    flags.DEFINE_integer("train_freq", 10, "Training report frequency")
    flags.DEFINE_integer("valid_freq", 100, "Validation report frequency")
    flags.DEFINE_integer("valid_size", 1000, "Size of validation data")
    flags.DEFINE_integer("max_step", 10000, "Max training step")
    flags.DEFINE_integer("patience", 5, "Patience")
    flags.DEFINE_string("wv_dict", '../tf_nlu/tmp/dict.pkl', "word vec dict file")
    flags.DEFINE_string("train_record", '_data/train.record', "training record file")
    flags.DEFINE_string("valid_record", '_data/valid.record', "valid record file")
    flags.DEFINE_string("bestmodel_dir", './model/best/', "Path to save the best model")
    flags.DEFINE_string("logdir", './log/', "Path to save the log")
    flags.DEFINE_string("wv_emb_file", '../tf_nlu/tmp/embedding.pkl', "word vec embedding file")
    flags.DEFINE_string("model_name", 'LSTMModel', "Model name")
    flags.DEFINE_string("mask_type", 'false_mask', "Model name")
    flags.DEFINE_float("lr", 0.01, "Learning rate")
    flags.DEFINE_float("L2_lambda", 0.0001, "Lambda of L2 loss")
    flags.DEFINE_float("embed_keep_prob", 0.95, "Keep prob of embedding dropout")
    flags.DEFINE_float("weight_keep_prob", 0.5, "Keep prob of weight dropout")
    flags.DEFINE_float("fw_hs_keep_prob", 0.5, "Keep prob of fw_hs dropout")
    flags.DEFINE_float("fw_cs_keep_prob", 0.5, "Keep prob of fw_cs dropout")
    flags.DEFINE_float("bw_hs_keep_prob", 0.5, "Keep prob of bw_hs dropout")
    flags.DEFINE_float("bw_cs_keep_prob", 0.5, "Keep prob of bw_cs dropout")
    flags.DEFINE_float("output_keep_prob", 0.5, "Keep prob of bw_cs dropout")
    flags.DEFINE_float("rec_keep_prob", 1.0, "Keep prob of bw_cs dropout")
    flags.DEFINE_boolean("load_model", False, "Whether load the best model")
    flags.DEFINE_boolean("save_model", False, "Whether save the best model")
    flags.DEFINE_boolean("drop_embed", False, "Whether drop embeddings")
    flags.DEFINE_boolean("drop_weight", False, "Whether drop weights")
    flags.DEFINE_boolean("drop_fw_hs", False, "Whether drop forward h states")
    flags.DEFINE_boolean("drop_fw_cs", False, "Whether drop forward c states")
    flags.DEFINE_boolean("drop_bw_hs", False, "Whether drop backward h states")
    flags.DEFINE_boolean("drop_bw_cs", False, "Whether drop backward c states")

    FLAGS = flags.FLAGS
    
    config_dict = {'wv_vocab_size': FLAGS.wv_vocab_size, \
                   'embed_size': FLAGS.embed_size, \
                   'fw_cell_size': FLAGS.fw_cell_size, \
                   'bw_cell_size': FLAGS.bw_cell_size, \
                   'attn_size': FLAGS.attn_size, \
                   'lr': FLAGS.lr, \
                   'L2_lambda': FLAGS.L2_lambda, \
                   'wv_emb_file': FLAGS.wv_emb_file, \
                   'class_size': FLAGS.class_size, \
                   'drop_embed': FLAGS.drop_embed, \
                   'embed_keep_prob': FLAGS.embed_keep_prob, \
                   'drop_weight': FLAGS.drop_weight, \
                   'weight_keep_prob': FLAGS.weight_keep_prob, \
                   'drop_fw_hs': FLAGS.drop_fw_hs, \
                   'fw_hs_keep_prob': FLAGS.fw_hs_keep_prob, \
                   'drop_fw_cs': FLAGS.drop_fw_cs, \
                   'fw_cs_keep_prob': FLAGS.fw_cs_keep_prob, \
                   'drop_bw_hs': FLAGS.drop_bw_hs, \
                   'bw_hs_keep_prob': FLAGS.bw_hs_keep_prob, \
                   'drop_bw_cs': FLAGS.drop_bw_cs, \
                   'bw_cs_keep_prob': FLAGS.bw_cs_keep_prob, \
                   'output_keep_prob': FLAGS.output_keep_prob, \
                   'rec_keep_prob': FLAGS.rec_keep_prob
                   }
    
    model = eval(FLAGS.model_name)
    trainer = Trainer(FLAGS, model, config_dict)
    trainer.run()