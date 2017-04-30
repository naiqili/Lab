import tensorflow as tf
import numpy as np
import logging
from model import build_model, get_metrics, init_variables
import cPickle
import os, shutil, random, pylab

flags = tf.flags

flags.DEFINE_integer("vocab_size", 50000, "Vocab size")
flags.DEFINE_integer("emb_size", 300, "Embedding size")
flags.DEFINE_integer("cell_size", 100, "Cell size")
flags.DEFINE_integer("pos_size", 56, "Pos size")
flags.DEFINE_integer("train_freq", 10, "Training report frequency")
flags.DEFINE_integer("valid_freq", 1, "Validation report frequency")
flags.DEFINE_integer("max_epochs", 1000, "Max training epochs")
flags.DEFINE_integer("reset_after", 10, "Reset after")
flags.DEFINE_integer("patience", 5, "Patience")
flags.DEFINE_string("wv_dict", '../tf_nlu/tmp/dict.pkl', "word vec dict file")
flags.DEFINE_string("pos_dict", 'tmp/pos_dict.pkl', "pos dict file")
flags.DEFINE_string("target", 'title', "What to train (title/location/data/whenst/whened/invitee)")
flags.DEFINE_string("training_tree_path", '_data/train_tree.txt', "training tree file")
flags.DEFINE_string("valid_tree_path", '_data/valid_tree.txt', "valid tree file")
flags.DEFINE_string("model_dir", './model/train/', "Path to save the model")
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

def get_train_op(loss, lr):
    return tf.train.GradientDescentOptimizer(lr).minimize(loss)

def run_epoch(tree_data, new_model, is_training, wv_word2ind, pos_word2ind, _get_metrics=False):
    step = 0
    loss_history = []
    _00, _01, _10, _11 = 0, 0, 0, 0
    random.shuffle(tree_data)
    while step < len(tree_data):
        with tf.Graph().as_default(), tf.Session() as sess:
            init_variables(FLAGS.wv_emb_file, \
                           FLAGS.vocab_size, \
                           FLAGS.pos_size, \
                           FLAGS.emb_size, \
                           FLAGS.cell_size)
            if new_model:
                tf.global_variables_initializer().run()
            else:
                saver = tf.train.Saver()
                saver.restore(sess, FLAGS.model_dir+'train.temp')
            for _ in xrange(FLAGS.reset_after):
                if step >= len(tree_data):
                    break
                md = build_model(FLAGS.target,
                                 tree_data[step],
                                 wv_word2ind,
                                 pos_word2ind,
                                 is_training)
                loss_value = sess.run([md['avg_loss']])[0]
                if is_training:
                    train_op = get_train_op(md['avg_loss'], \
                                            FLAGS.lr)
                    sess.run([train_op])
                loss_history.append(loss_value)
                logger.debug("step: %d, loss: %f" % (step, loss_value))
                if _get_metrics:
                    v00, v01, v10, v11 = get_metrics(sess, md['tree_node'])
                    #logger.debug("%d %d %d %d" % (v00, v01, v10, v11))
                    _00 += v00
                    _01 += v01
                    _10 += v10
                    _11 += v11
                step += 1
            if is_training:
                saver = tf.train.Saver()
                saver.save(sess, FLAGS.model_dir+'train.temp')
    return np.mean(loss_history), (_00, _01, _10, _11)

def load_tree_data(path):
    res = []
    with open(path) as f:
        for line in f:
            res.append(line.strip())
    return res

def report_figure(history):
    (loss, acc, recall, precision) = zip(*history)
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
    except:
        pass

def train():
    (wv_word2ind, wv_ind2word) = cPickle.load(open(FLAGS.wv_dict))
    (pos_word2ind, pos_ind2word) = cPickle.load(open(FLAGS.pos_dict))

    train_trees = load_tree_data(FLAGS.training_tree_path)
    valid_trees = load_tree_data(FLAGS.valid_tree_path)

    _patience = FLAGS.patience
    best_valid_loss = 10000000
    valid_history = []
    
    for epoch in xrange(FLAGS.max_epochs):
        logger.debug('epoch %d' % epoch)
        if epoch == 0:
            train_loss, _ = run_epoch(train_trees, True, \
                                   True,
                                      wv_word2ind, pos_word2ind)
        else:
            train_loss, _ = run_epoch(train_trees, False, \
                                   True,
                                   wv_word2ind, pos_word2ind)
        logger.debug('training loss: %f' % train_loss)
        if epoch % FLAGS.valid_freq == 0:
            valid_loss, (v00, v01, v10, v11) = \
                         run_epoch(valid_trees, False, \
                                   False,
                                   wv_word2ind, pos_word2ind,
                                   True)
            logger.debug('validation loss: %f' % valid_loss)
            logger.debug('metrics: %d %d %d %d' % (v00, v01, v10, v11))
            _acc = 1.0 * (v00 + v11) / (v00 + v01 + v10 + v11)
            if (v11 + v01) > 0:
                _recall = 1.0 * v11 / (v11 + v01)
            else:
                _recall = 0
            if (v10 + v11) > 0:
                _precision = 1.0 * v11 / (v10 + v11)
            else:
                _precision = 0
            logger.debug("acc: %f, recall: %f, precision: %f" %
                         (_acc, _recall, _precision))

            valid_history.append((valid_loss, _acc, _recall, _precision))
            report_figure(valid_history)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if os.path.exists(FLAGS.bestmodel_dir):
                    shutil.rmtree(FLAGS.bestmodel_dir)
                shutil.copytree(FLAGS.model_dir, FLAGS.bestmodel_dir)
                logger.debug('better model saved')
                _patience = FLAGS.patience
            else:
                _patience = _patience-1
                if _patience == 0:
                    break

if __name__=='__main__':
    train()
