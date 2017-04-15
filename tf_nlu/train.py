import tensorflow as tf
import numpy as np
from reader import *

flags = tf.flags

flags.DEFINE_integer("vocab_size", 1917, "Vocab size (real: 52716, toy: 3669)")
flags.DEFINE_integer("emb_size", 300, "Embedding size")
flags.DEFINE_integer("cell_size", 100, "Cell size")
flags.DEFINE_integer("batch_size", 100, "Batch size")
flags.DEFINE_integer("train_freq", 10, "Training report frequency")
flags.DEFINE_integer("dev_freq", 20, "Developing report frequency")
flags.DEFINE_integer("max_step", 5000, "Max training step")
flags.DEFINE_string("cell_type", 'GRU', "Cell type")
flags.DEFINE_string("rnn_type", 'bi_dynamic', "Cell type")
flags.DEFINE_string("optimizer", 'Adam', "Optimizer")
flags.DEFINE_string("datafile", './_data/7000/nlu_data.pkl', "Data file (.pkl)")
flags.DEFINE_string("dictfile", './tmp/dict.pkl', "Dict file (.pkl)")
flags.DEFINE_string("train_target", 'title', "What to train (title/location/data/whenst/whened/invitee)")
flags.DEFINE_string("modeldir", './model/', "Path to save the model")
flags.DEFINE_string("logdir", './log/', "Path to save the log")
flags.DEFINE_float("lr_rate", 0.01, "Learning rate")

FLAGS = flags.FLAGS

def build_graph(x, y, mask, vocab_size=, emb_size=300, cell_size=200, lr_rate=0.01, cell_type='GRU', rnn_type='dynamic', optimizer='Adam', batch_size=50, is_training=True):
    embeddings = tf.get_variable('embedding_mat', [vocab_size, emb_size])
    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

    if cell_type == 'GRU':
        cell_fw = tf.contrib.rnn.GRUCell(cell_size)
        cell_bw = tf.contrib.rnn.GRUCell(cell_size)

    if rnn_type == 'bi_dynamic':
        rnn_outputs, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, rnn_inputs, sequence_length=data_len)

    with tf.variable_scope('logits'):
        W = tf.get_variable('w', [cell_size*2, 2])
        b = tf.get_variable('b', [2], initializer=tf.constant_initializer(0.0))

    rnn_outputs = tf.concat(rnn_outputs, 2)
    rnn_outputs = tf.reshape(rnn_outputs, [-1, cell_size*2])
    y_reshaped = tf.reshape(y, [-1])
    mask_reshaped = tf.reshape(mask, [-1])

    logits = tf.matmul(rnn_outputs, W) + b

    predictions = tf.nn.softmax(logits)

    total_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped) * mask_reshaped) / \
                 tf.reduce_sum(mask_reshaped)
    train_step = tf.train.AdamOptimizer(lr_rate).minimize(total_loss)
    train_loss_log = tf.summary.scalar('total_loss', total_loss)
    summary = tf.summary.merge([train_loss_log])

    return {
        'final_state': final_state,
        'total_loss': total_loss,
        'train_step': train_step,
        'preds': predictions,
        'saver': tf.train.Saver(),
        'summary': summary
        }

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
        
        for _step in range(max_step):
            summary, cur_loss, cur_state, _ = sess.run([g['summary'], g['total_loss'], g['final_state'], g['train_step']])
            if (_step+1) % FLAGS.train_freq == 0:
                print 'Training loss:', cur_loss
                train_writer.add_summary(summary, _step)
                g['saver'].save(sess, model_path+'train', global_step=_step)

            dev_losses = []
            if (_step+1) % FLAGS.dev_freq == 0:
                for _batch in range(dev_batch_num):
                    cur_loss = sess.run([dev_g['total_loss']])
                    dev_losses.append(cur_loss)
                m_loss = np.mean(dev_losses)
                print 'valid mean loss:', m_loss
                valid_sum = sess.run(valid_summary_op, feed_dict={valid_mean: m_loss})
                valid_writer.add_summary(valid_sum, _step)
                if m_loss < best_valid_loss:
                    best_valid_loss = m_loss
                    g['saver'].save(sess, model_path+'best')
                    

if __name__=='__main__':
    data = get_raw_data(FLAGS.datafile, FLAGS.dictfile)
    train_x, train_y, train_len, train_mask = get_producer(data['train_text'], data['train_'+FLAGS.train_target], data['train_text_len'], FLAGS.batch_size)
    valid_x, valid_y, valid_len, valid_mask = get_producer(data['valid_text'], data['valid_'+FLAGS.train_target], data['valid_text_len'], FLAGS.batch_size)
    with tf.variable_scope("Model", reuse=None) as scope:
        g = build_graph(train_x, train_y, train_len, train_mask, \
                        vocab_size=FLAGS.vocab_size, \
                        emb_size=FLAGS.emb_size, \
                        cell_size=FLAGS.cell_size, \
                        lr_rate=FLAGS.lr_rate, \
                        cell_type=FLAGS.cell_type, \
                        rnn_type=FLAGS.rnn_type, \
                        optimizer=FLAGS.optimizer, \
                        batch_size=FLAGS.batch_size, \
                        is_training=True)

    with tf.variable_scope("Model", reuse=True) as scope:
        dev_g = build_graph(valid_x, valid_y, valid_len, valid_mask, \
                        vocab_size=FLAGS.vocab_size, \
                        emb_size=FLAGS.emb_size, \
                        cell_size=FLAGS.cell_size, \
                        lr_rate=FLAGS.lr_rate, \
                        cell_type=FLAGS.cell_type, \
                        rnn_type=FLAGS.rnn_type, \
                        optimizer=FLAGS.optimizer, \
                        batch_size=FLAGS.batch_size, \
                        is_training=False)

    dev_batch_num = int(data['valid_size'] / FLAGS.batch_size)
    train_network(g, dev_g, dev_batch_num, \
                  max_step=FLAGS.max_step, \
                  model_path=FLAGS.model_path, \
                  log_path=FLAGS.log_path)
                  
