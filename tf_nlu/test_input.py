import tensorflow as tf
import numpy as np
from model import build_graph
import cPickle
from reader import get_raw_data
import random
from reader import _word2ind
import os
import nltk

os.environ["CUDA_VISIBLE_DEVICES"]=""

vocab_size = 50000
emb_size = 300
cell_size = 100
N =5
cell_type = 'GRU'
rnn_type = 'bi_dynamic'
datafile = './_data/7000/nlu_data.pkl'
dictfile = './tmp/dict.pkl'
title_modelfile = './model/title/best'
location_modelfile = './model/location/best'
invitee_modelfile = './model/invitee/best'
day_modelfile = './model/day/best'
whenst_modelfile = './model/whenst/best'
whened_modelfile = './model/whened/best'

all_data = cPickle.load(open(datafile))
(word2ind, ind2word) = cPickle.load(open(dictfile))

print "OK"

tf_x = tf.placeholder(tf.int32, shape=(1, 60), name='x')
tf_y = tf.placeholder(tf.int32, shape=(1, 60), name='y')
tf_len = tf.placeholder(tf.int32, shape=(1), name='len')
tf_mask = tf.to_float(tf.not_equal(tf_x, 0))
with tf.device("/cpu:0"):
    with tf.variable_scope("Model_title"):    
        g_title = build_graph(tf_x, tf_y, tf_len, tf_mask, \
                            vocab_size=vocab_size, \
                            emb_size=emb_size, \
                            cell_size=cell_size, \
                            cell_type=cell_type, \
                            rnn_type=rnn_type, \
                            batch_size=1, \
                            is_training=False)
    
    with tf.variable_scope("Model_location"):    
        g_location = build_graph(tf_x, tf_y, tf_len, tf_mask, \
                            vocab_size=vocab_size, \
                            emb_size=emb_size, \
                            cell_size=cell_size, \
                            cell_type=cell_type, \
                            rnn_type=rnn_type, \
                            batch_size=1, \
                            is_training=False)
    '''
    with tf.variable_scope("Model_invitee"):    
        g_invitee = build_graph(tf_x, tf_y, tf_len, tf_mask, \
                            vocab_size=vocab_size, \
                            emb_size=emb_size, \
                            cell_size=cell_size, \
                            cell_type=cell_type, \
                            rnn_type=rnn_type, \
                            batch_size=1, \
                            is_training=False)
    with tf.variable_scope("Model_day"):    
        g_day = build_graph(tf_x, tf_y, tf_len, tf_mask, \
                            vocab_size=vocab_size, \
                            emb_size=emb_size, \
                            cell_size=cell_size, \
                            cell_type=cell_type, \
                            rnn_type=rnn_type, \
                            batch_size=1, \
                            is_training=False)
    with tf.variable_scope("Model_whenst"):    
        g_whenst = build_graph(tf_x, tf_y, tf_len, tf_mask, \
                            vocab_size=vocab_size, \
                            emb_size=emb_size, \
                            cell_size=cell_size, \
                            cell_type=cell_type, \
                            rnn_type=rnn_type, \
                            batch_size=1, \
                            is_training=False)
    with tf.variable_scope("Model_whened"):    
        g_whened = build_graph(tf_x, tf_y, tf_len, tf_mask, \
                            vocab_size=vocab_size, \
                            emb_size=emb_size, \
                            cell_size=cell_size, \
                            cell_type=cell_type, \
                            rnn_type=rnn_type, \
                            batch_size=1, \
                            is_training=False)
    '''
print "...OK"

test_sents = ["i would like to go to the library at three p.m. tomorrow afternoon",
              "go swimming with alice and bob tomorrow"]

with tf.device("/cpu:0"):
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    g_title['saver'].restore(sess, title_modelfile)
    
    g_location['saver'].restore(sess, location_modelfile)
    '''
    g_invitee['saver'].restore(sess, invitee_modelfile)
    g_day['saver'].restore(sess, day_modelfile)
    g_whenst['saver'].restore(sess, whenst_modelfile)
    g_whened['saver'].restore(sess, whened_modelfile)
    '''
    for _n, sent in enumerate(test_sents):
        print 
        print "test case:", _n
        print sent

        def process(g, text):
            tok_text = nltk.word_tokenize(text)
            _len = len(tok_text)

            x_mat = np.zeros((1, 60), dtype=np.int32)
            len_mat = np.zeros((1,), dtype=np.int32)

            for i, w in enumerate(tok_text):
                x_mat[0][i] = _word2ind(w, word2ind)
            len_mat[0] = _len

            preds = sess.run([g['preds']], feed_dict={tf_x: x_mat, tf_len: len_mat})
            #print "sent:", ' '.join(tok_text)
            #print "prediction:", preds
            pred_text = []
            for i, b in enumerate(preds[0]):
                if i == len(tok_text):
                    break
                #print b
                if b == 1:
                    pred_text.append(tok_text[i])
            return ' '.join(pred_text)

        pred_title = process(g_title, sent)
        
        pred_location = process(g_location, sent)
        '''
        pred_invitee = process(g_invitee, sent)
        pred_day = process(g_day, sent)
        pred_whenst = process(g_whenst, sent)
        pred_whened = process(g_whened, sent)
        '''
        print "title:", pred_title
        
        print "location:", pred_location
        '''
        print "invitee:", pred_invitee
        print "day:", pred_day
        print "whenst:", pred_whenst
        print "whened:", pred_whened
        '''
    sess.close()
