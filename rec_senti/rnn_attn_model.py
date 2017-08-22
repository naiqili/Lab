import tensorflow as tf
import cPickle
from utils import *

class RNNAttnModel():
    def __init__(self, config_dict):        
        self.__dict__.update(config_dict)
        self.emb_mat = cPickle.load(open(self.wv_emb_file))

    def add_variables(self, reuse=False):
        with tf.variable_scope('Embeddings', reuse=reuse):
            self.wv_embed = tf.get_variable('wv_embed',
                                     [self.wv_vocab_size, self.embed_size], \
                                     initializer=tf.constant_initializer(self.emb_mat))
            self.unk_embed = tf.get_variable('unk_embed',
                [1, self.embed_size])
            if self.drop_embed:
                self.wv_embed = tf.contrib.layers.dropout(self.wv_embed, self.embed_keep_prob, is_training=self.is_training)
                self.unk_embed = tf.contrib.layers.dropout(self.unk_embed, self.embed_keep_prob, is_training=self.is_training)
        with tf.variable_scope('Forward', reuse=reuse):
            self._rnn_W = self.rnn_W = tf.get_variable('rnn_W',
                [self.embed_size + 2*self.fw_cell_size, self.fw_cell_size])
            self._rnn_b = self.rnn_b = tf.get_variable('rnn_b',
                [1, self.fw_cell_size], \
                initializer=tf.constant_initializer(0.0))
            self._rnn_trans = self.rnn_trans = tf.get_variable('rnn_trans',
                [self.fw_cell_size, self.fw_cell_size])
            
            if self.drop_weight:
                self.rnn_W = tf.contrib.layers.dropout(self.rnn_W, self.weight_keep_prob, is_training=self.is_training)
                
        with tf.variable_scope('Attn', reuse=reuse):
            self._attn_W1 = self.attn_W1 = tf.get_variable('attn_W1',
                [self.fw_cell_size, self.attn_size])
            self._attn_W2 = self.attn_W2 = tf.get_variable('attn_W2',
                [self.fw_cell_size, self.attn_size])
            self._attn_b = self.attn_b = tf.get_variable('attn_b',
                [self.attn_size])

        with tf.variable_scope('Projection', reuse=reuse):
            self._proj_W = self.proj_W = tf.get_variable('W',
                            [(self.fw_cell_size), self.class_size])
            self._proj_attn_W = self.proj_attn_W = tf.get_variable('attn_W',
                            [(self.fw_cell_size), self.fw_cell_size])
            self._proj_b = self.proj_b = tf.get_variable('b',
                            [1, self.class_size])
            if self.drop_weight:
                self.proj_W = tf.contrib.layers.dropout(self.proj_W, self.weight_keep_prob, is_training=self.is_training)
                self.proj_attn_W = tf.contrib.layers.dropout(self.proj_attn_W, self.weight_keep_prob, is_training=self.is_training)

    def embed_word(self, word_index):
        embeddings = tf.concat([self.wv_embed, self.unk_embed], 0)
        with tf.device('/cpu:0'):
            return tf.expand_dims(tf.gather(embeddings, word_index), 0)

    def combine_children(self, x_mid, left_h, right_h):
        new_h = tf.matmul(tf.concat([x_mid, left_h, right_h], 1), self.rnn_W) + self.rnn_b
        new_h = activation(new_h)
        new_h = tf.contrib.layers.dropout(new_h, self.rec_keep_prob, is_training=self.is_training)
        return new_h

    def build_model(self, left_ts, right_ts, wv_ts, target_ts, is_leaf_ts, len_ts, mask):
        self.ground_truth = target_ts
        self.build_forward(left_ts, right_ts, wv_ts, is_leaf_ts, len_ts)
        self.build_attn(len_ts, mask)
        self.build_cost(self.fw_hs, self.cont, target_ts)
                    
    def build_forward(self, left_ts, right_ts, wv_ts, is_leaf_ts, len_ts):
        fw_hs = tf.TensorArray(tf.float32, size=0, dynamic_size=True, \
            clear_after_read=False, infer_shape=True)
        
        def forward_loop_body(fw_hs, i):
            node_is_leaf = tf.gather(is_leaf_ts, i)
            node_word_index = tf.gather(wv_ts, i)
            x_mid = tf.cond(
                tf.equal(node_is_leaf, 1),
                lambda: self.embed_word(node_word_index),
                lambda: tf.zeros([1, self.embed_size])
            )
            left_id, right_id = (tf.gather(left_ts, i), tf.gather(right_ts, i))
            left_h, right_h = tf.cond(
                tf.equal(node_is_leaf, 1),
                lambda: (tf.zeros([1, self.fw_cell_size]), tf.zeros([1, self.fw_cell_size])),
                lambda: (fw_hs.read(left_id), fw_hs.read(right_id)))
            new_h = self.combine_children(x_mid, left_h, right_h)
    
            fw_hs = fw_hs.write(i, new_h)
            i = tf.add(i, 1)
            return fw_hs, i

        forward_loop_cond = lambda fw_hs, i: \
                    tf.less(i, tf.squeeze(len_ts))

        fw_hs, _ = tf.while_loop(forward_loop_cond, forward_loop_body, [fw_hs, 0], parallel_iterations=1)
        self.fw_hs = fw_hs.concat()
        return fw_hs
    
    def build_attn(self, len_ts, mask):
        len_ts = tf.squeeze(len_ts)
        mask = tf.reshape(mask, [len_ts, len_ts])
        # len x h
        em = self.fw_hs
        em = tf.reshape(em, [len_ts, -1])
        # len x attn
        em_W1, em_W2 = tf.matmul(em, self.attn_W1), tf.matmul(em, self.attn_W2)
        # len x len x attn
        em_ex_W1 = tf.tile(tf.expand_dims(em_W1, 1), [1, len_ts, 1])
        em_ex_W2 = tf.tile(tf.expand_dims(em_W2, 0), [len_ts, 1, 1])
        # len x len x attn
        tmp = em_ex_W1 + em_ex_W2
        tmp = activation(tmp)
        # len x len
        _as = tf.einsum('ijk,k->ij', tmp, self.attn_b)
        _as = tf.nn.softmax(_as)
        _as = _as * mask
        # len x len x h
        cont_h = tf.einsum('ij,jk->ijk', _as, em)
        # len x h
        cont = tf.reduce_sum(cont_h, axis=1)
        self.attn_vecs = _as
        self.cont = cont

    def build_cost(self, fw_hs, cont, label_ts):
        fw_hs = tf.contrib.layers.dropout(fw_hs, self.output_keep_prob, is_training=self.is_training)
        logits = tf.matmul(tf.matmul(fw_hs, self.rnn_trans) + tf.matmul(cont, self.proj_attn_W), self.proj_W) + self.proj_b
        softmax = tf.nn.softmax(logits)
        self.pred = tf.squeeze(tf.argmax(logits, 1))
        self.binary_pred = tf.to_int32((softmax[:, 3] + softmax[:, 4]) > (softmax[:, 0] + softmax[:, 1]))
        _loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logits, labels=label_ts)
        all_weights = [self.rnn_W, self._proj_W, self._rnn_trans, self._proj_attn_W]
        self.loss = _loss + tf.reduce_sum([tf.nn.l2_loss(w) for w in all_weights]) * self.L2_lambda
        self.sum_loss = tf.reduce_sum(self.loss)
        if self.is_training:
            opt = tf.train.AdamOptimizer(self.lr)
            grads_and_vars = opt.compute_gradients(self.sum_loss)
            for i, (grad, var) in enumerate(grads_and_vars):
                if var == self.wv_embed or var == self.unk_embed:
                    grad = tf.scalar_mul(0.1, grad)
                    grads_and_vars[i] = (grad, var)
            self.train_op = opt.apply_gradients(grads_and_vars)