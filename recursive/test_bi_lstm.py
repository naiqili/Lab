import matplotlib

import tensorflow as tf
import numpy as np
import logging
import cPickle
import os, shutil, random

from bi_lstm_model import BiLSTMModel
from tfrecord_reader import get_data

matplotlib.use('Agg')

import pylab

flags = tf.flags

flags.DEFINE_integer("wv_vocab_size", 50002, "Vocab size")
flags.DEFINE_integer("pos_vocab_size", 57, "Vocab size")
flags.DEFINE_integer("embed_size", 300, "Embedding size")
flags.DEFINE_integer("fw_cell_size", 50, "Composition cell size")
flags.DEFINE_integer("bw_cell_size", 50, "Decomposition cell size")
flags.DEFINE_integer("train_freq", 10, "Training report frequency")
flags.DEFINE_integer("test_freq", 100, "testation report frequency")
flags.DEFINE_integer("test_size", 1000, "Size of testation data")
flags.DEFINE_integer("max_step", 10000, "Max training step")
flags.DEFINE_integer("patience", 5, "Patience")
flags.DEFINE_string("wv_dict", '../tf_nlu/tmp/dict.pkl', "word vec dict file")
flags.DEFINE_string("pos_dict", 'tmp/pos_dict.pkl', "pos dict file")
flags.DEFINE_string("target", 'title', "What to train (title/location/data/whenst/whened/invitee)")
flags.DEFINE_string("train_record", '_data/train.record', "training record file")
flags.DEFINE_string("test_record", '_data/test.record', "test record file")
flags.DEFINE_string("fig_path", './log/fig/', "Path to save the figure")
flags.DEFINE_string("bestmodel_dir", './model/best/', "Path to save the best model")
flags.DEFINE_string("logdir", './log/', "Path to save the log")
flags.DEFINE_string("summary_dir", './summary/', "Path to save the summary")
flags.DEFINE_string("wv_emb_file", '../tf_nlu/tmp/embedding.pkl', "word vec embedding file")
flags.DEFINE_float("lr", 0.01, "Learning rate")
flags.DEFINE_float("fw_dropout_rate", 0.3, "Forward dropout rate")
flags.DEFINE_float("bw_dropout_rate", 0.3, "Backward dropout rate")
flags.DEFINE_boolean("load_model", False, "Whether load the best model")
flags.DEFINE_boolean("use_dropout", False, "Whether use dropout")

FLAGS = flags.FLAGS

logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler(FLAGS.logdir+FLAGS.target))
logging.basicConfig(level = logging.DEBUG,
                    format = "%(asctime)s: %(name)s: %(levelname)s: %(message)s")

def prop_op(pred, target, is_leaf, prop, left, right):
    i = len(pred)-1
    res = pred.copy()

    while i >= 0:
        if not is_leaf[i]:
            if prop == 'AND':
                res[left[i]] = int(bool(res[left[i]]) and bool(res[i]))
                res[right[i]] = int(bool(res[right[i]]) and bool(res[i]))
            elif prop == 'OR':
                res[left[i]] = int(bool(res[left[i]]) or bool(res[i]))
                res[right[i]] = int(bool(res[right[i]]) or bool(res[i]))
        i -= 1
    return res

def get_metrics(pred, target, is_leaf, only_leaf=True, prop=None, left=None, right=None):
    
    if prop == None:
        _pred = pred
    else:
        _pred = prop_op(pred, target, is_leaf, prop, left, right)
    v00, v01, v10, v11 = 0, 0, 0, 0

    for i in range(len(is_leaf)):
        if only_leaf and is_leaf[i] == 0:
            continue
        if _pred[i] == 0 and target[i] == 0:
            v00 += 1
        if _pred[i] == 0 and target[i] == 1:
            v01 += 1
        if _pred[i] == 1 and target[i] == 0:
            v10 += 1
        if _pred[i] == 1 and target[i] == 1:
            v11 += 1
    return v00, v01, v10, v11

def collect_metrics(_00, _01, _10, _11):
    acc = 1.0 * (_11 + _00) / (_00 + _01 + _10 + _11)
    if (_11 + _10) > 0:
        precision = 1.0 * _11 / (_11 + _10)
    else:
        precision = 0
    if (_11 + _01) > 0:
        recall = 1.0 * _11 / (_11 + _01)
    else:
        recall = 0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    return acc, precision, recall, f1
                

def test():
    (wv_word2ind, wv_ind2word) = cPickle.load(open(FLAGS.wv_dict))
    (pos_word2ind, pos_ind2word) = cPickle.load(open(FLAGS.pos_dict))

    config_dict = {'wv_vocab_size': FLAGS.wv_vocab_size, \
                   'pos_vocab_size': FLAGS.pos_vocab_size, \
                   'embed_size': FLAGS.embed_size, \
                   'fw_cell_size': FLAGS.fw_cell_size, \
                   'bw_cell_size': FLAGS.bw_cell_size, \
                   'lr': FLAGS.lr, \
                   'wv_emb_file': FLAGS.wv_emb_file, \
                   'use_dropout': FLAGS.use_dropout, \
                   'fw_dropout_rate': FLAGS.fw_dropout_rate, \
                   'bw_dropout_rate': FLAGS.bw_dropout_rate}

    test_md = BiLSTMModel(config_dict)
    test_md.is_training = False
    test_md.add_variables()

    l_tts, wv_tts, left_tts, right_tts, target_tts, is_leaf_tts = get_data(FLAGS.target, filename=FLAGS.test_record)
    test_md.build_model(left_tts, right_tts, wv_tts, target_tts, is_leaf_tts, l_tts)


    with tf.Session() as sess:
        tf.train.Saver().restore(sess, FLAGS.bestmodel_dir)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        logger.debug("Start test")
        test_losses = []
        metrics = 0, 0, 0, 0
        #a_metrics = 0, 0, 0, 0
        o_metrics = 0, 0, 0, 0
        for i in xrange(FLAGS.test_size):
            test_loss, test_summary, test_pred, target_v, is_leaf_v, left_v, right_v = \
                sess.run([test_md.mean_loss, test_md.loss_summary, test_md.pred, target_tts, is_leaf_tts, left_tts, right_tts])
            test_losses.append(test_loss)
            #logger.debug("testation loss %f" % test_loss)
            new_metrics = get_metrics(test_pred, target_v, is_leaf_v)
            metrics = map(lambda (x,y): x+y, zip(metrics, new_metrics))
            #a_new_metrics = get_metrics(test_pred, target_v, is_leaf_v, prop='AND', left=left_v, right=right_v)
            #a_metrics = map(lambda (x,y): x+y, zip(a_metrics, a_new_metrics))
            o_new_metrics = get_metrics(test_pred, target_v, is_leaf_v, prop='OR', left=left_v, right=right_v)
            o_metrics = map(lambda (x,y): x+y, zip(o_metrics, o_new_metrics))
            
        _00, _01, _10, _11 = metrics
        #a_00, a_01, a_10, a_11 = a_metrics
        o_00, o_01, o_10, o_11 = o_metrics

        acc, precision, recall, f1 = collect_metrics(_00, _01, _10, _11)
        #a_acc, a_precision, a_recall, a_f1 = collect_metrics(a_00, a_01, a_10, a_11)
        o_acc, o_precision, o_recall, o_f1 = collect_metrics(o_00, o_01, o_10, o_11)
        mean_loss = np.mean(test_losses)
        logger.debug('test: finish')
        logger.debug('Step: mean loss: %f' % (mean_loss))
        logger.debug('test: prop:None  acc: %f, precision: %f, recall: %f, f1: %f' % (acc, precision, recall, f1))
        #logger.debug('testation: prop:AND  acc: %f, precision: %f, recall: %f, f1: %f' % (a_acc, a_precision, a_recall, a_f1))
        logger.debug('test: prop:OR  acc: %f, precision: %f, recall: %f, f1: %f' % (o_acc, o_precision, o_recall, o_f1))

        coord.request_stop()
        coord.join(threads)
    
if __name__=='__main__':
    test()
