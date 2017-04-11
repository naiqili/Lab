import tensorflow as tf
from reader import *

def build_graph(x, y, mask, vocab_size=3669, emb_size=200, cell_size=200, lr_rate=0.01, cell_type='GRU', batch_size=50):
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

    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped) * mask_reshaped)
    train_step = tf.train.AdamOptimizer(lr_rate).minimize(total_loss)

    return {
        'x': x,
        'y': y,
        'mask': mask,
        'init_state': init_state,
        'final_state': final_state,
        'total_loss': total_loss,
        'train_step': train_step,
        'preds': predictions,
        'saver': tf.train.Saver()
        }

def train_network(g, epoch_num=5000, save_path='./model/basicrnn.mdl'):
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        for _epoch in range(epoch_num):
            cur_loss, cur_state, _ = sess.run([g['total_loss'], g['final_state'], g['train_step']])
            print cur_loss

        g['saver'].save(sess, save_path)

if __name__=='__main__':
    raw_data = get_raw_data()
    x, y, mask = get_producer(raw_data['train_data'], 50)
    g = build_graph(x, y, mask)
    train_network(g)
