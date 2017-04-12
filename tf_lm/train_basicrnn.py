import tensorflow as tf
from reader import *

flags = tf.flags

flags.DEFINE_integer("vocab_size", 3669, "Vocab size (real: 52716, toy: 3669)")
flags.DEFINE_integer("emb_size", 100, "Embedding size")
flags.DEFINE_integer("cell_size", 100, "Cell size")
flags.DEFINE_integer("batch_size", 100, "Batch size")
flags.DEFINE_integer("train_freq", 10, "Training report frequency")
flags.DEFINE_integer("dev_freq", 20, "Developing report frequency")
flags.DEFINE_integer("max_step", 10000, "Max training step")
flags.DEFINE_string("cell_type", 'GRU', "Cell type")
flags.DEFINE_string("optimizer", 'Adam', "Optimizer")
flags.DEFINE_string("data", 'toy', "Data to be used (toy/real)")
flags.DEFINE_string("model_path", './model/test.ckpt', "Path to save the model")
flags.DEFINE_string("model_path", './log/test.log', "Path to save the log")
flags.DEFINE_float("lr_rate", 0.01, "Learning rate")

FLAGS = flags.FLAGS

def build_graph(x, y, mask, name, vocab_size=3670, emb_size=200, cell_size=200, lr_rate=0.01, cell_type='GRU', optimizer='Adam', batch_size=50):
#    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_holder')
#    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='output_holder')

    embeddings = tf.get_variable('embedding_mat', [vocab_size, emb_size])
    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

    if cell_type == 'GRU':
        cell = tf.contrib.rnn.GRUCell(cell_size)

    init_state = cell.zero_state(batch_size, tf.float32)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state = init_state)

    with tf.variable_scope('logits'):
        W = tf.get_variable('w', [cell_size, vocab_size])
        b = tf.get_variable('b', [vocab_size], initializer=tf.constant_initializer(0.0))

    rnn_outputs = tf.reshape(rnn_outputs, [-1, cell_size])
    y_reshaped = tf.reshape(y, [-1])
    mask_reshaped = tf.reshape(mask, [-1])

    logits = tf.matmul(rnn_outputs, W) + b

    predictions = tf.nn.softmax(logits)

    total_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped) * mask_reshaped) / \
                 tf.reduce_sum(mask_reshaped)
    perp = tf.exp(total_loss)
    train_step = tf.train.AdamOptimizer(lr_rate).minimize(total_loss)
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('perplexity', perp)
    summary = tf.summary.merge_all()

    return {
        'x': x,
        'y': y,
        'mask': mask,
        'init_state': init_state,
        'final_state': final_state,
        'total_loss': total_loss,
        'train_step': train_step,
        'preds': predictions,
        'saver': tf.train.Saver(),
        'summary': summary
        }

def train_network(g, dev_g, max_step=5000, model_path='./model/test.ckpt', log_path='./log/test.log'):
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(log_path)
        tf.global_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        for _step in range(max_step):
            summary, cur_loss, cur_state, _ = sess.run([g['summary'], g['total_loss'], g['final_state'], g['train_step']])
            if (step+1) % FLAGS.train_freq == 0:
                print 'Training loss:', cur_loss
                train_writer.add_summary(summary, _step)
                g['saver'].save(sess, model_path, global_step=_step)
            if (_step+1) % FLAGS.dev_freq == 0:
                summary, cur_loss, cur_state, _ = sess.run([dev_g['summary'], dev_g['total_loss'], dev_g['final_state'], dev_g['train_step']])
                print 'Dev loss:', cur_loss
                train_writer.add_summary(summary, _step)

if __name__=='__main__':
    raw_data = get_raw_data(FLAGS.data)
    x, y, mask = get_producer(raw_data['train_data'], FLAGS.batch_size)
    dev_x, dev_y, dev_mask = get_producer(raw_data['dev_data'], FLAGS.batch_size)
    g = build_graph(x, y, mask, \
                        vocab_size=FLAGS.vocab_size, \
                        emb_size=FLAGS.emb_size, \
                        cell_size=FLAGS.cell_size, \
                        lr_rate=FLAGS.lr_rate, \
                        cell_type=FLAGS.cell_type, \
                        optimizer=FLAGS.optimizer, \
                        batch_size=FLAGS.batch_size)
    tf.get_variable_scope().reuse_variables()
    dev_g = build_graph(dev_x, dev_y, dev_mask, \
                        vocab_size=FLAGS.vocab_size, \
                        emb_size=FLAGS.emb_size, \
                        cell_size=FLAGS.cell_size, \
                        lr_rate=FLAGS.lr_rate, \
                        cell_type=FLAGS.cell_type, \
                        optimizer=FLAGS.optimizer, \
                        batch_size=FLAGS.batch_size)
    train_network(g, dev_g, \
                  max_step=FLAGS.max_step, \
                  model_path=FLAGS.model_path, \
                  log_path=FLAGS.log_path)
                  