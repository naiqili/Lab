import tensorflow as tf
import numpy as np
from reader import *
import logging
from model import build_graph

flags = tf.flags

flags.DEFINE_integer("vocab_size", 50000, "Vocab size")
flags.DEFINE_integer("emb_size", 300, "Embedding size")
flags.DEFINE_integer("cell_size", 100, "Cell size")
flags.DEFINE_integer("batch_size", 100, "Batch size")
flags.DEFINE_integer("train_freq", 10, "Training report frequency")
flags.DEFINE_integer("dev_freq", 20, "Developing report frequency")
flags.DEFINE_integer("max_step", 5000, "Max training step")
flags.DEFINE_integer("patience", 5, "Patience")
flags.DEFINE_string("cell_type", 'GRU', "Cell type")
flags.DEFINE_string("rnn_type", 'bi_dynamic', "Cell type")
flags.DEFINE_string("optimizer", 'Adam', "Optimizer")
flags.DEFINE_string("datafile", './_data/7000/nlu_data.pkl', "Data file (.pkl)")
flags.DEFINE_string("dictfile", './tmp/dict.pkl', "Dict file (.pkl)")
flags.DEFINE_string("train_target", 'title', "What to train (title/location/data/whenst/whened/invitee)")
flags.DEFINE_string("modeldir", './model/', "Path to save the model")
flags.DEFINE_string("logdir", './log/', "Path to save the log")
flags.DEFINE_string("model_name", 'test', "Model name")
flags.DEFINE_string("emb_file", './tmp/embedding.pkl', "Model name")
flags.DEFINE_float("lr_rate", 0.01, "Learning rate")
flags.DEFINE_float("input_keep_prob", 0.5, "Dropout probability of input")
flags.DEFINE_float("output_keep_prob", 0.5, "Dropout probability of output")

FLAGS = flags.FLAGS

logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler(FLAGS.logdir+FLAGS.model_name))
logging.basicConfig(level = logging.DEBUG,
                    format = "%(asctime)s: %(name)s: %(levelname)s: %(message)s")


def train_network(g, dev_g, dev_batch_num, max_step=5000, model_path='./model/', log_path='./log/'):
    valid_mean = tf.placeholder(tf.float32, [])
    valid_summary_op = tf.summary.scalar("total_valid_loss", valid_mean)

    best_valid_loss = 10000000
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(log_path+'train/', sess.graph)
        valid_writer = tf.summary.FileWriter(log_path+'valid/', sess.graph)
        tf.global_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        _patience = FLAGS.patience
        for _step in range(max_step):
            summary, cur_loss, cur_acc, cur_state, _ = sess.run([g['summary'], g['total_loss'], g['acc'], g['final_state'], g['train_step']])
            if (_step+1) % FLAGS.train_freq == 0:
                logger.debug('Training loss: %f, accuracy: %f' % (cur_loss, cur_acc))
                train_writer.add_summary(summary, _step)
                g['saver'].save(sess, model_path+'train', global_step=_step)

            dev_losses = []
            dev_accs = []
            if (_step+1) % FLAGS.dev_freq == 0:
                for _batch in range(dev_batch_num):
                    cur_loss, cur_acc = sess.run([dev_g['total_loss'], dev_g['acc']])
                    dev_losses.append(cur_loss)
                    dev_accs.append(cur_acc)
                m_loss = np.mean(dev_losses)
                m_acc = np.mean(dev_accs)
                logger.debug('valid loss: %f, accuracy: %f' % (m_loss, m_acc))
                valid_sum = sess.run(valid_summary_op, feed_dict={valid_mean: m_loss})
                valid_writer.add_summary(valid_sum, _step)
                if m_loss < best_valid_loss:
                    best_valid_loss = m_loss
                    g['saver'].save(sess, model_path+'best')
                    logger.debug('better model saved')
                    _patience = FLAGS.patience
                else:
                    _patience = _patience-1
                    if _patience == 0:
                        break
                    

if __name__=='__main__':
    data = get_raw_data(FLAGS.datafile, FLAGS.dictfile)
    train_x, train_y, train_len, train_mask = get_producer(data['train_text'], data['train_'+FLAGS.train_target], data['train_text_len'], FLAGS.batch_size)
    valid_x, valid_y, valid_len, valid_mask = get_producer(data['valid_text'], data['valid_'+FLAGS.train_target], data['valid_text_len'], FLAGS.batch_size)
    with tf.variable_scope("Model", reuse=None) as scope:
        g = build_graph(train_x, train_y, train_len, train_mask, \
                        vocab_size=FLAGS.vocab_size, \
                        emb_size=FLAGS.emb_size, \
                        emb_file=FLAGS.emb_file, \
                        cell_size=FLAGS.cell_size, \
                        lr_rate=FLAGS.lr_rate, \
                        cell_type=FLAGS.cell_type, \
                        rnn_type=FLAGS.rnn_type, \
                        optimizer=FLAGS.optimizer, \
                        batch_size=FLAGS.batch_size, \
                        input_keep_prob=FLAGS.input_keep_prob, \
                        output_keep_prob=FLAGS.output_keep_prob, \
                        is_training=True)

    with tf.variable_scope("Model", reuse=True) as scope:
        dev_g = build_graph(valid_x, valid_y, valid_len, valid_mask, \
                        vocab_size=FLAGS.vocab_size, \
                        emb_size=FLAGS.emb_size, \
                        emb_file=FLAGS.emb_file, \
                        cell_size=FLAGS.cell_size, \
                        lr_rate=FLAGS.lr_rate, \
                        cell_type=FLAGS.cell_type, \
                        rnn_type=FLAGS.rnn_type, \
                        optimizer=FLAGS.optimizer, \
                        batch_size=FLAGS.batch_size, \
                        input_keep_prob=FLAGS.input_keep_prob, \
                        output_keep_prob=FLAGS.output_keep_prob, \
                        is_training=False)

    dev_batch_num = int(data['valid_size'] / FLAGS.batch_size)
    train_network(g, dev_g, dev_batch_num, \
                  max_step=FLAGS.max_step, \
                  model_path=FLAGS.modeldir, \
                  log_path=FLAGS.logdir)
                  
    logger.debug("ALL DONE")
