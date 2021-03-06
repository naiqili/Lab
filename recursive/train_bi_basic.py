import matplotlib

import tensorflow as tf
import numpy as np
import logging
import cPickle
import os, shutil, random

from bi_basic_model import BiBasicModel
from tfrecord_reader import get_data

matplotlib.use('Agg')

import pylab

flags = tf.flags

flags.DEFINE_integer("wv_vocab_size", 50002, "Vocab size")
flags.DEFINE_integer("pos_vocab_size", 57, "Vocab size")
flags.DEFINE_integer("embed_size", 300, "Embedding size")
flags.DEFINE_integer("comp_cell_size", 50, "Composition cell size")
flags.DEFINE_integer("de_cell_size", 50, "Decomposition cell size")
flags.DEFINE_integer("train_freq", 10, "Training report frequency")
flags.DEFINE_integer("valid_freq", 100, "Validation report frequency")
flags.DEFINE_integer("valid_size", 1000, "Size of validation data")
flags.DEFINE_integer("max_step", 10000, "Max training step")
flags.DEFINE_integer("patience", 5, "Patience")
flags.DEFINE_string("wv_dict", '../tf_nlu/tmp/dict.pkl', "word vec dict file")
flags.DEFINE_string("pos_dict", 'tmp/pos_dict.pkl', "pos dict file")
flags.DEFINE_string("target", 'title', "What to train (title/location/data/whenst/whened/invitee)")
flags.DEFINE_string("train_record", '_data/train.record', "training record file")
flags.DEFINE_string("valid_record", '_data/valid.record', "valid record file")
flags.DEFINE_string("fig_path", './log/fig/', "Path to save the figure")
flags.DEFINE_string("bestmodel_dir", './model/best/', "Path to save the best model")
flags.DEFINE_string("logdir", './log/', "Path to save the log")
flags.DEFINE_string("summary_dir", './summary/', "Path to save the summary")
flags.DEFINE_string("wv_emb_file", '../tf_nlu/tmp/embedding.pkl', "word vec embedding file")
flags.DEFINE_float("lr", 0.01, "Learning rate")
flags.DEFINE_boolean("load_model", False, "Whether load the best model")

FLAGS = flags.FLAGS

logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler(FLAGS.logdir+FLAGS.target))
logging.basicConfig(level = logging.DEBUG,
                    format = "%(asctime)s: %(name)s: %(levelname)s: %(message)s")

def report_figure(valid_history, train_history):
    (loss, acc, precision, recall, f1) = zip(*valid_history)
    try:
        pylab.figure()
        pylab.title("Loss")
        pylab.plot(train_history)
        pylab.savefig(FLAGS.fig_path + 'train_loss.png')
        pylab.close()
        pylab.figure()
        pylab.title("Loss")
        pylab.plot(loss)
        pylab.savefig(FLAGS.fig_path + 'loss.png')
        pylab.close()
        pylab.figure()
        pylab.title("Accuracy")
        pylab.plot(acc)
        pylab.savefig(FLAGS.fig_path + 'acc.png')
        pylab.close()
        pylab.figure()
        pylab.title("Recall")
        pylab.plot(recall)
        pylab.savefig(FLAGS.fig_path + 'recall.png')
        pylab.close()
        pylab.figure()
        pylab.title("Precision")
        pylab.plot(precision)
        pylab.savefig(FLAGS.fig_path + 'precision.png')
        pylab.close()
        pylab.figure()
        pylab.title("F1")
        pylab.plot(f1)
        pylab.savefig(FLAGS.fig_path + 'F1.png')
        pylab.close()
    except:
        pass

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
                

def train():
    (wv_word2ind, wv_ind2word) = cPickle.load(open(FLAGS.wv_dict))
    (pos_word2ind, pos_ind2word) = cPickle.load(open(FLAGS.pos_dict))

    config_dict = {'wv_vocab_size': FLAGS.wv_vocab_size, \
                   'pos_vocab_size': FLAGS.pos_vocab_size, \
                   'embed_size': FLAGS.embed_size, \
                   'comp_cell_size': FLAGS.comp_cell_size, \
                   'de_cell_size': FLAGS.de_cell_size, \
                   'lr': FLAGS.lr, \
                   'wv_emb_file': FLAGS.wv_emb_file}

    train_md = BiBasicModel(config_dict)
    train_md.is_training = True
    train_md.add_variables()
    valid_md = BiBasicModel(config_dict)
    valid_md.is_training = False

    l_tts, wv_tts, left_tts, right_tts, target_tts, is_leaf_tts = get_data(FLAGS.target, filename=FLAGS.train_record)
    train_md.build_model(left_tts, right_tts, wv_tts, target_tts, is_leaf_tts, l_tts)

    l_vts, wv_vts, left_vts, right_vts, target_vts, is_leaf_vts = get_data(FLAGS.target, filename=FLAGS.valid_record)
    valid_md.build_model(left_vts, right_vts, wv_vts, target_vts, is_leaf_vts, l_vts)
    
    _patience = FLAGS.patience
    best_valid_loss = 10000000
    valid_history = []
    train_history = []

    with tf.Session() as sess:
        if FLAGS.load_model:
            tf.train.Saver().restore(sess, FLAGS.bestmodel_dir)
        else:
            tf.global_variables_initializer().run()

        train_writer = tf.summary.FileWriter(FLAGS.summary_dir + 'train')
        valid_writer = tf.summary.FileWriter(FLAGS.summary_dir + 'valid')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for _step in xrange(FLAGS.max_step):
            if _patience == 0:
                break
            loss_ts = train_md.mean_loss
            train_summary = train_md.loss_summary
            train_op = train_md.train_op
            train_loss, train_summary, _ = sess.run([loss_ts, train_summary, train_op])
            train_history.append(train_loss)
            if _step % FLAGS.train_freq == 0:
                logger.debug("Step: %d Training: Loss: %f" % (_step, train_loss))
                train_writer.add_summary(train_summary, _step)

            if _step != 0 and _step % FLAGS.valid_freq == 0:
                logger.debug("Start validation")
                valid_losses = []
                metrics = 0, 0, 0, 0
                #a_metrics = 0, 0, 0, 0
                o_metrics = 0, 0, 0, 0
                for i in xrange(FLAGS.valid_size):
                    valid_loss, valid_summary, valid_pred, target_v, is_leaf_v, left_v, right_v = \
                        sess.run([valid_md.mean_loss, valid_md.loss_summary, valid_md.pred, target_vts, is_leaf_vts, left_vts, right_vts])
                    valid_losses.append(valid_loss)
                    #logger.debug("Validation loss %f" % valid_loss)
                    valid_writer.add_summary(valid_summary, _step)
                    new_metrics = get_metrics(valid_pred, target_v, is_leaf_v)
                    metrics = map(lambda (x,y): x+y, zip(metrics, new_metrics))
                    #a_new_metrics = get_metrics(valid_pred, target_v, is_leaf_v, prop='AND', left=left_v, right=right_v)
                    #a_metrics = map(lambda (x,y): x+y, zip(a_metrics, a_new_metrics))
                    o_new_metrics = get_metrics(valid_pred, target_v, is_leaf_v, prop='OR', left=left_v, right=right_v)
                    o_metrics = map(lambda (x,y): x+y, zip(o_metrics, o_new_metrics))
                    
                _00, _01, _10, _11 = metrics
                #a_00, a_01, a_10, a_11 = a_metrics
                o_00, o_01, o_10, o_11 = o_metrics

                acc, precision, recall, f1 = collect_metrics(_00, _01, _10, _11)
                #a_acc, a_precision, a_recall, a_f1 = collect_metrics(a_00, a_01, a_10, a_11)
                o_acc, o_precision, o_recall, o_f1 = collect_metrics(o_00, o_01, o_10, o_11)
                mean_loss = np.mean(valid_losses)
                logger.debug('Validation: finish')
                logger.debug('Step: %d Validation: mean loss: %f' % (_step, mean_loss))
                logger.debug('Validation: prop:None  acc: %f, precision: %f, recall: %f, f1: %f' % (acc, precision, recall, f1))
                #logger.debug('Validation: prop:AND  acc: %f, precision: %f, recall: %f, f1: %f' % (a_acc, a_precision, a_recall, a_f1))
                logger.debug('Validation: prop:OR  acc: %f, precision: %f, recall: %f, f1: %f' % (o_acc, o_precision, o_recall, o_f1))
                valid_history.append((mean_loss, acc, precision, recall, f1))
                report_figure(valid_history, train_history)
                logger.debug('Figure saved')

                if mean_loss < best_valid_loss:
                    best_valid_loss = mean_loss
                    _patience = FLAGS.patience
                    saver = tf.train.Saver()
                    saver.save(sess, FLAGS.bestmodel_dir)
                    logger.debug('Better model saved')
                else:                    
                    _patience -= 1
                    logger.debug('Not improved. Patience: %d' % _patience)
        coord.request_stop()
        coord.join(threads)
    
if __name__=='__main__':
    train()
