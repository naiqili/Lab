import tensorflow as tf
from nltk.tree import Tree
from tftree import TFTree, TFNode
import numpy as np
import cPickle

def fn_wv_word2ind(w, word2ind):
    if w in word2ind:
        return word2ind[w]
    else:
        return word2ind['<unk>']

def get_embedding(emb, w):
    return tf.expand_dims(
                  tf.nn.embedding_lookup(emb, w), 0)

def build_upcomp(node, wv_word2ind, pos_word2ind):
    with tf.variable_scope('Embedding', reuse=True):
        wv_emb = tf.get_variable('wv_emb')
        pos_emb = tf.get_variable('pos_embed')
    
    if node.isLeaf:
        with tf.variable_scope('LeafComposition', reuse=True):
            W_l = tf.get_variable('W_left')
            W_r = tf.get_variable('W_right')
            b = tf.get_variable('b')
        (w, pos_w) = node.words.split('+')
        _w = get_embedding(wv_emb, fn_wv_word2ind(w, wv_word2ind))
        _pos = get_embedding(pos_emb, pos_word2ind[pos_w])
        node.upcomp = tf.nn.relu(tf.matmul(_w, W_l) + tf.matmul(_pos, W_r) + b)
    else:
        build_upcomp(node.left_node, wv_word2ind, pos_word2ind)
        build_upcomp(node.right_node, wv_word2ind, pos_word2ind)
        with tf.variable_scope('InterComposition', reuse=True):
            W_l = tf.get_variable('W_left')
            W_r = tf.get_variable('W_right')
            W_m = tf.get_variable('W_mid')
            b = tf.get_variable('b')
        left = node.left_node.upcomp
        right = node.right_node.upcomp
        pos_w = node.words
        _pos = get_embedding(pos_emb, pos_word2ind[pos_w])
        node.upcomp = tf.nn.relu(tf.matmul(left, W_l) + tf.matmul(right, W_r) + tf.matmul(_pos, W_m) + b)

def build_projandloss(node):
    all_loss = []
    with tf.variable_scope('Projection', reuse=True):
        proj_U = tf.get_variable('U')
        proj_b = tf.get_variable('b')
    node.proj = tf.matmul(node.upcomp, proj_U) + proj_b
    node.pred = tf.argmax(node.proj, 1)
    node.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=node.proj, labels=tf.expand_dims(tf.constant(node.label_y), 0))
    all_loss.append(node.loss)
    if not node.isLeaf:
        all_loss.extend(build_projandloss(node.left_node))
        all_loss.extend(build_projandloss(node.right_node))
    return all_loss

def get_metrics(sess, node, onlyLeaf=True):
    _00, _01, _10, _11 = 0, 0, 0, 0
    if not node.isLeaf:
        l00, l01, l10, l11 = get_metrics(sess, node.left_node)
        r00, r01, r10, r11 = get_metrics(sess, node.right_node)
        _00 = l00 + r00
        _01 = l01 + r01
        _10 = l10 + r10
        _11 = l11 + r11
    if node.isLeaf or (not onlyLeaf):
        node._cur_pred = sess.run([node.pred])[0][0]
        if node._cur_pred == 0 and node.label_y == 0:
            _00 = _00 + 1
        if node._cur_pred == 0 and node.label_y == 1:
            _01 = _01 + 1
        if node._cur_pred == 1 and node.label_y == 0:
            _10 = _10 + 1
        if node._cur_pred == 1 and node.label_y == 1:
            _11 = _11 + 1
    return _00, _01, _10, _11

def init_variables(wv_emb, vocab_size, pos_size, emb_size, cell_size):
    # Initialize variables
    emb_mat = cPickle.load(open(wv_emb))
    with tf.variable_scope('Embedding'):
        wv_emb = tf.get_variable('wv_emb', [vocab_size+2, emb_size], trainable=False, initializer=tf.constant_initializer(emb_mat)) # word vec embeddings
        pos_emb  = tf.get_variable('pos_embed', [pos_size, emb_size]) # POS embeddings
    with tf.variable_scope('InterComposition'):
        tf.get_variable('W_left', [cell_size, cell_size])
        tf.get_variable('W_right', [cell_size, cell_size])
        tf.get_variable('W_mid', [emb_size, cell_size])
        tf.get_variable('b', [1, cell_size])
    with tf.variable_scope('LeafComposition'):
        tf.get_variable('W_left', [emb_size, cell_size])
        tf.get_variable('W_right', [emb_size, cell_size])
        tf.get_variable('b', [1, cell_size])
    with tf.variable_scope('Projection'):
        tf.get_variable('U', [cell_size, 2])
        tf.get_variable('b', [1, 2])
    
def build_model(target, tree_str, wv_word2ind, pos_word2ind, is_training):
    nltk_t = Tree.fromstring(tree_str)
    tf_t = TFTree(nltk_t)
    tf_t.root.set_label(target)
    build_upcomp(tf_t.root, wv_word2ind, pos_word2ind)
    total_loss = build_projandloss(tf_t.root)
    #print total_loss[0].shapem
    avg_loss = tf.reduce_mean(tf.concat(total_loss, 0))
    res = {'avg_loss': avg_loss, \
           'tree_node': tf_t.root
    }

    return res
