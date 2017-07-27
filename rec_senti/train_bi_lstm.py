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
flags.DEFINE_integer("embed_size", 300, "Embedding size")
flags.DEFINE_integer("class_size", 3, "3 for binary; 5 for fine grained")
flags.DEFINE_integer("fw_cell_size", 50, "Composition cell size")
flags.DEFINE_integer("bw_cell_size", 50, "Decomposition cell size")
flags.DEFINE_integer("train_freq", 10, "Training report frequency")
flags.DEFINE_integer("valid_freq", 100, "Validation report frequency")
flags.DEFINE_integer("valid_size", 1000, "Size of validation data")
flags.DEFINE_integer("max_step", 10000, "Max training step")
flags.DEFINE_integer("patience", 5, "Patience")
flags.DEFINE_string("wv_dict", '../tf_nlu/tmp/dict.pkl', "word vec dict file")
flags.DEFINE_string("train_record", '_data/train.record', "training record file")
flags.DEFINE_string("valid_record", '_data/valid.record', "valid record file")
flags.DEFINE_string("fig_path", './log/fig/', "Path to save the figure")
flags.DEFINE_string("bestmodel_dir", './model/best/', "Path to save the best model")
flags.DEFINE_string("logdir", './log/', "Path to save the log")
flags.DEFINE_string("summary_dir", './summary/', "Path to save the summary")
flags.DEFINE_string("wv_emb_file", '../tf_nlu/tmp/embedding.pkl', "word vec embedding file")
flags.DEFINE_float("lr", 0.01, "Learning rate")
flags.DEFINE_float("L2_lambda", 0.0001, "Lambda of L2 loss")
flags.DEFINE_float("embed_keep_prob", 0.95, "Keep prob of embedding dropout")
flags.DEFINE_float("weight_keep_prob", 0.5, "Keep prob of weight dropout")
flags.DEFINE_float("fw_hs_keep_prob", 0.5, "Keep prob of fw_hs dropout")
flags.DEFINE_float("fw_cs_keep_prob", 0.5, "Keep prob of fw_cs dropout")
flags.DEFINE_float("bw_hs_keep_prob", 0.5, "Keep prob of bw_hs dropout")
flags.DEFINE_float("bw_cs_keep_prob", 0.5, "Keep prob of bw_cs dropout")
flags.DEFINE_boolean("load_model", False, "Whether load the best model")
flags.DEFINE_boolean("drop_embed", False, "Whether drop embeddings")
flags.DEFINE_boolean("drop_weight", False, "Whether drop weights")
flags.DEFINE_boolean("drop_fw_hs", False, "Whether drop forward h states")
flags.DEFINE_boolean("drop_fw_cs", False, "Whether drop forward c states")
flags.DEFINE_boolean("drop_bw_hs", False, "Whether drop backward h states")
flags.DEFINE_boolean("drop_bw_cs", False, "Whether drop backward c states")

FLAGS = flags.FLAGS

logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.DEBUG,
                    format = "%(asctime)s: %(name)s: %(levelname)s: %(message)s")
logger.addHandler(logging.FileHandler(FLAGS.logdir+'log.txt'))

def report_figure(valid_history, train_history):
    (loss, acc) = zip(*valid_history)
    train_x, train_y = zip(*train_history)
    valid_x, valid_y = zip(*loss)
    valid_acc_x, valid_acc_y = zip(*acc)
    try:
        pylab.figure()
        pylab.title("Loss & Accuracy")
        pylab.plot(train_x, train_y, 'r', label='train loss')
        pylab.plot(valid_x, valid_y, 'b', label='valid loss')
        pylab.plot(valid_acc_x, valid_acc_y, 'g', label='valid accuracy')
        #pylab.legend()
        pylab.savefig(FLAGS.fig_path + 'figure.png')
        pylab.close()
    except:
        pass

def train():
    (wv_word2ind, wv_ind2word) = cPickle.load(open(FLAGS.wv_dict))

    config_dict = {'wv_vocab_size': FLAGS.wv_vocab_size, \
                   'embed_size': FLAGS.embed_size, \
                   'fw_cell_size': FLAGS.fw_cell_size, \
                   'bw_cell_size': FLAGS.bw_cell_size, \
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
                   'bw_cs_keep_prob': FLAGS.bw_cs_keep_prob
                   }

    train_md = BiLSTMModel(config_dict)
    train_md.is_training = True
    train_md.add_variables(reuse=False)
    valid_md = BiLSTMModel(config_dict)
    valid_md.is_training = False
    valid_md.add_variables(reuse=True)

    l_tts, wv_tts, left_tts, right_tts, target_tts, is_leaf_tts = get_data(filename=FLAGS.train_record, shuffle=True)
    train_md.build_model(left_tts, right_tts, wv_tts, target_tts, is_leaf_tts, l_tts)

    l_vts, wv_vts, left_vts, right_vts, target_vts, is_leaf_vts = get_data(filename=FLAGS.valid_record)
    valid_md.build_model(left_vts, right_vts, wv_vts, target_vts, is_leaf_vts, l_vts)
    
    _patience = FLAGS.patience
    best_valid_loss = 10000000.0
    best_valid_acc = 0.0
    valid_history = []
    train_history = []

    with tf.Session() as sess:
        if FLAGS.load_model:
            tf.train.Saver().restore(sess, FLAGS.bestmodel_dir)
        else:
            tf.global_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        batch_train_history = []
        for _step in xrange(FLAGS.max_step):
            if _patience == 0:
                break
            loss_ts = train_md.mean_loss
            train_op = train_md.train_op
            train_loss, _ = sess.run([loss_ts, train_op])
            batch_train_history.append(train_loss)
            
            if _step % FLAGS.train_freq == 0:
                logger.debug("Step: %d Training: Loss: %f" % (_step, train_loss))

            if _step != 0 and _step % FLAGS.valid_freq == 0:
                logger.debug("Start validation")
                valid_losses = []
                metrics = np.zeros((FLAGS.class_size, FLAGS.class_size))

                for i in xrange(FLAGS.valid_size):
                    valid_loss, valid_pred, target_v = \
                        sess.run([valid_md.mean_loss, valid_md.pred, target_vts])
                    valid_losses.append(valid_loss)
                    root_pred = valid_pred[-1]
                    root_target = target_v[-1]
                    metrics[root_pred, root_target] += 1
                    #logger.debug("Validation loss %f" % valid_loss)

                mean_loss = np.mean(valid_losses)
                logger.debug('Validation: finish')
                
                logger.debug('Metrics:\n %s' % str(metrics))
                acc = 1.0 * np.trace(metrics) / np.sum(metrics)
                logger.debug('Step: %d Validation: mean loss: %f, accuracy: %f' % (_step, mean_loss, acc))

                valid_history.append([[_step, mean_loss], [_step, acc]])
                batch_train_loss = np.mean(batch_train_history)
                logger.debug('Step: %d Training batch loss: %f' % (_step, batch_train_loss))
                batch_train_history = []
                train_history.append([_step, batch_train_loss])
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
                if acc > best_valid_acc:
                    best_valid_acc = acc
                logger.debug('Best loss: %f, Best accuracy: %f' % (best_valid_loss, best_valid_acc))
        coord.request_stop()
        coord.join(threads)
    
if __name__=='__main__':
    train()
