import tensorflow as tf
import numpy as np
import logging
import cPickle
import os, shutil, random, pylab

from basic_model import BasicModel
from tfrecord_reader import get_data

flags = tf.flags

flags.DEFINE_integer("wv_vocab_size", 50002, "Vocab size")
flags.DEFINE_integer("pos_vocab_size", 57, "Vocab size")
flags.DEFINE_integer("embed_size", 300, "Embedding size")
flags.DEFINE_integer("cell_size", 100, "Cell size")
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
flags.DEFINE_string("wv_emb_file", '../tf_nlu/tmp/embedding.pkl', "word vec embedding file")
flags.DEFINE_float("lr", 0.01, "Learning rate")

FLAGS = flags.FLAGS

logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler(FLAGS.logdir+FLAGS.target))
logging.basicConfig(level = logging.DEBUG,
                    format = "%(asctime)s: %(name)s: %(levelname)s: %(message)s")

def report_figure(history):
    (loss, acc, precision, recall, f1) = zip(*history)
    try:
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

def get_metrics(pred, target, is_leaf, only_leaf=True):
    v00, v01, v10, v11 = 0, 0, 0, 0
    for i in range(len(is_leaf)):
        if only_leaf and is_leaf[i] == 0:
            continue
        if pred[i] == 0 and target[i] == 0:
            v00 += 1
        if pred[i] == 0 and target[i] == 1:
            v01 += 1
        if pred[i] == 1 and target[i] == 0:
            v10 += 1
        if pred[i] == 0 and target[i] == 0:
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
                   'cell_size': FLAGS.cell_size, \
                   'lr': FLAGS.lr, \
                   'wv_emb_file': FLAGS.wv_emb_file}

    train_md = BasicModel(config_dict)
    train_md.is_training = True
    train_md.add_variables()
    valid_md = BasicModel(config_dict)
    valid_md.is_training = False

    l_tts, wv_tts, left_tts, right_tts, target_tts, is_leaf_tts = get_data(FLAGS.target, filename=FLAGS.train_record)
    train_md.build_model(left_tts, right_tts, wv_tts, target_tts, is_leaf_tts, l_tts)

    l_vts, wv_vts, left_vts, right_vts, target_vts, is_leaf_vts = get_data(FLAGS.target, filename=FLAGS.valid_record)
    valid_md.build_model(left_vts, right_vts, wv_vts, target_vts, is_leaf_vts, l_vts)
    
    _patience = FLAGS.patience
    best_valid_loss = 10000000
    valid_history = []

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for _step in xrange(FLAGS.max_step):
            if _patience == 0:
                break
            loss_ts = train_md.loss
            train_op = train_md.train_op
            train_loss, _ = sess.run([loss_ts, train_op])
            train_loss = np.mean(train_loss)
            if _step % FLAGS.train_freq == 0:
                logger.debug("Training loss: %f" % train_loss)

            if _step % FLAGS.valid_freq == 0:
                print _step
                valid_losses = []
                metrics = 0, 0, 0, 0
                for i in xrange(FLAGS.valid_size):
                    valid_loss, valid_pred, target_v, is_leaf_v = \
                        sess.run([valid_md.loss, valid_md.pred, target_vts, is_leaf_vts])
                    valid_loss = np.mean(valid_loss)
                    valid_losses.append(valid_loss)
                    logger.debug("Validation loss %f" % valid_loss)
                    print valid_pred
                    print target_v
                    print is_leaf_v
                    new_metrics = get_metrics(valid_pred, target_v, is_leaf_v)
                    metrics = map(lambda (x,y): x+y, zip(metrics, new_metrics))
                _00, _01, _10, _11 = metrics
                acc, precision, recall, f1 = collect_metrics(_00, _01, _10, 11)
                mean_loss = np.mean(valid_losses)
                logger.debug('Validation finish')
                logger.debug('  mean loss: %f' % mean_loss)
                logger.debug('  acc: %f, precision: %f, recall: %f, f1: %f' % (acc, precision, recall, f1))
                valid_history.append((mean_loss, acc, precision, recall, f1))
                report_figure(valid_history)
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
