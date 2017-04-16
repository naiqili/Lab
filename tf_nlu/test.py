import tensorflow as tf
import numpy as np
from model import build_graph
import cPickle
from reader import get_raw_data
import random

flags = tf.flags
flags.DEFINE_integer("vocab_size", 1917, "Vocab size")
flags.DEFINE_integer("emb_size", 300, "Embedding size")
flags.DEFINE_integer("cell_size", 100, "Cell size")
flags.DEFINE_integer("N", 5, "Number of test cases")
flags.DEFINE_string("cell_type", 'GRU', "Cell type")
flags.DEFINE_string("rnn_type", 'bi_dynamic', "Cell type")
flags.DEFINE_string("datafile", './_data/7000/nlu_data.pkl', "Data file (.pkl)")
flags.DEFINE_string("dictfile", './tmp/dict7000.pkl', "Dict file (.pkl)")
flags.DEFINE_string("modelfile", 'None', "model file")
flags.DEFINE_string("target", 'title', "What to test (title/location/data/whenst/whened/invitee)")
FLAGS = flags.FLAGS

all_data = cPickle.load(open(FLAGS.datafile))
(word2ind, ind2word) = cPickle.load(open(FLAGS.dictfile))

with tf.variable_scope("Model"):
    tf_x = tf.placeholder(tf.int32, shape=(1, 60), name='x')
    tf_y = tf.placeholder(tf.int32, shape=(1, 60), name='y')
    tf_len = tf.placeholder(tf.int32, shape=(1), name='len')
    tf_mask = tf.to_float(tf.not_equal(tf_x, 0))
    g = build_graph(tf_x, tf_y, tf_len, tf_mask, \
                        vocab_size=FLAGS.vocab_size, \
                        emb_size=FLAGS.emb_size, \
                        cell_size=FLAGS.cell_size, \
                        cell_type=FLAGS.cell_type, \
                        rnn_type=FLAGS.rnn_type, \
                        batch_size=1, \
                        is_training=False)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    if FLAGS.modelfile != 'None':
        g['saver'].restore(sess, FLAGS.modelfile)
    for _n in range(FLAGS.N):
        print 
        print "test case:", _n
        data = random.choice(all_data)
        tok_text = data['tok_text']
        _len = len(tok_text)
        tok_y = data['tok_'+FLAGS.target]

        x_mat = np.zeros((1, 60), dtype=np.int32)
        y_mat = np.zeros((1, 60), dtype=np.int32)
        len_mat = np.zeros((1,), dtype=np.int32)

        for i, w in enumerate(tok_text):
            x_mat[0][i] = word2ind[w]
        y_mat[0][:_len] = tok_y[:]
        len_mat[0] = _len

        loss, acc, preds = sess.run([g['total_loss'], g['acc'], g['preds']], feed_dict={tf_x: x_mat, tf_y: y_mat, tf_len: len_mat})
        print "sent:", ' '.join(tok_text)
        print "loss:", loss
        print "accuracy:", acc
        print "prediction:", preds
        pred_text = []
        for i, b in enumerate(preds):
            if i == len(tok_text):
                break
            if b == 1:
                pred_text.append(tok_text[i])
        print "pred text:", ' '.join(pred_text)
