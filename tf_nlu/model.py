import tensorflow as tf
import cPickle
import numpy as np

def build_graph(x, y, data_len, mask, vocab_size=1917, emb_size=300, emb_file='./tmp/embedding.pkl', cell_size=200, lr_rate=0.01, cell_type='GRU', rnn_type='dynamic', input_keep_prob=0.5, output_keep_prob=0.5, optimizer='Adam', batch_size=50, is_training=True):
    emb_mat = cPickle.load(open(emb_file))
    embeddings = tf.get_variable('embedding_mat', [vocab_size+2, emb_size], trainable=False, initializer=tf.constant_initializer(emb_mat))
    
    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

    if cell_type == 'GRU':
        cell_fw = tf.contrib.rnn.GRUCell(cell_size)
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, \
            input_keep_prob=input_keep_prob, \
            output_keep_prob=output_keep_prob)
        cell_bw = tf.contrib.rnn.GRUCell(cell_size)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, \
            input_keep_prob=input_keep_prob, \
            output_keep_prob=output_keep_prob)

    if rnn_type == 'bi_dynamic':
        rnn_outputs, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, rnn_inputs, sequence_length=data_len, dtype=tf.float32)

    with tf.variable_scope('logits'):
        W = tf.get_variable('w', [cell_size*2, 2])
        b = tf.get_variable('b', [2], initializer=tf.constant_initializer(0.0))

    rnn_outputs = tf.concat(rnn_outputs, 2)
    rnn_outputs = tf.reshape(rnn_outputs, [-1, cell_size*2])
    y_reshaped = tf.reshape(y, [-1])
    mask_reshaped = tf.reshape(mask, [-1])

    logits = tf.matmul(rnn_outputs, W) + b

    predictions = tf.nn.softmax(logits)
    predictions = tf.to_int32(tf.argmax(predictions, axis=1))

    total_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped) * mask_reshaped) / \
                 tf.reduce_sum(mask_reshaped)
    acc = tf.reduce_sum(tf.to_float(tf.equal(predictions, y_reshaped)) * mask_reshaped) / \
          tf.reduce_sum(mask_reshaped)
    train_step = tf.train.AdamOptimizer(lr_rate).minimize(total_loss)
    train_loss_log = tf.summary.scalar('total_loss', total_loss)
    train_acc_log = tf.summary.scalar('accuracy', acc)
    summary = tf.summary.merge([train_loss_log, train_acc_log])

    return {
        'final_state': final_state,
        'total_loss': total_loss,
        'train_step': train_step,
        'preds': predictions,
        'saver': tf.train.Saver(),
        'summary': summary,
        'acc': acc
        }
