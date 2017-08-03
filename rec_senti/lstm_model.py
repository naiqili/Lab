import tensorflow as tf
import cPickle
from utils import *
import numpy as np

class LSTMModel():
    def __init__(self, config_dict):        
        self.__dict__.update(config_dict)
        self.emb_mat = cPickle.load(open(self.wv_emb_file))

    def add_variables(self, reuse=False):
        with tf.variable_scope('Embeddings', reuse=reuse):
            self.wv_embed = tf.get_variable('wv_embed',
                                     [self.wv_vocab_size, self.embed_size], \
                                     initializer=tf.constant_initializer(self.emb_mat))
            self.unk_embed = tf.get_variable('unk_embed',
                [1, self.embed_size], \
                initializer=tf.constant_initializer(np.random.uniform(-0.05, 0.05, [1, self.embed_size]).astype(np.float32)))
            if self.drop_embed:
                self.wv_embed = tf.contrib.layers.dropout(self.wv_embed, self.embed_keep_prob, is_training=self.is_training)
                self.unk_embed = tf.contrib.layers.dropout(self.unk_embed, self.embed_keep_prob, is_training=self.is_training)
                
        embed_fw_abs_val = -4 * np.sqrt(6. / (self.fw_cell_size + self.embed_size))
        fw_fw_abs_val = -4 * np.sqrt(6. / (self.embed_size + self.embed_size))
        with tf.variable_scope('Forward', reuse=reuse):
            self._W_i_mid = self.W_i_mid = tf.get_variable('W_i_mid',
                [self.embed_size, self.fw_cell_size], \
                initializer=tf.random_uniform_initializer(minval=-embed_fw_abs_val, maxval=embed_fw_abs_val))
            self._W_i = self.W_i = tf.get_variable('W_i',
                [self.fw_cell_size, self.fw_cell_size], \
                initializer=tf.random_uniform_initializer(minval=-fw_fw_abs_val, maxval=fw_fw_abs_val))
            self._b_i = self.b_i = tf.get_variable('b_i',
                [1, self.fw_cell_size], \
                initializer=tf.constant_initializer(0.0))

            self._W_f_mid = self.W_f_mid = tf.get_variable('W_f_mid',
                [self.embed_size, self.fw_cell_size], \
                initializer=tf.random_uniform_initializer(minval=-embed_fw_abs_val, maxval=embed_fw_abs_val))
            self._W_f_left = self.W_f_left = tf.get_variable('W_f_left',
                [self.fw_cell_size, self.fw_cell_size], \
                initializer=tf.random_uniform_initializer(minval=-fw_fw_abs_val, maxval=fw_fw_abs_val))
            self._W_f_right = self.W_f_right = tf.get_variable('W_f_right',
                [self.fw_cell_size, self.fw_cell_size], \
                initializer=tf.random_uniform_initializer(minval=-fw_fw_abs_val, maxval=fw_fw_abs_val))
            self._b_f = self.b_f = tf.get_variable('b_f',
                [1, self.fw_cell_size], \
                initializer=tf.constant_initializer(0.0))

            self._W_o_mid = self.W_o_mid = tf.get_variable('W_o_mid',
                [self.embed_size, self.fw_cell_size], \
                initializer=tf.random_uniform_initializer(minval=-embed_fw_abs_val, maxval=embed_fw_abs_val))
            self._W_o = self.W_o = tf.get_variable('W_o',
                [self.fw_cell_size, self.fw_cell_size], \
                initializer=tf.random_uniform_initializer(minval=-fw_fw_abs_val, maxval=fw_fw_abs_val))
            self._b_o = self.b_o = tf.get_variable('b_o',
                [1, self.fw_cell_size], \
                initializer=tf.constant_initializer(0.0))

            self._W_u_mid = self.W_u_mid = tf.get_variable('W_u_mid',
                [self.embed_size, self.fw_cell_size], \
                initializer=tf.random_uniform_initializer(minval=-embed_fw_abs_val, maxval=embed_fw_abs_val))
            self._W_u = self.W_u = tf.get_variable('W_u',
                [self.fw_cell_size, self.fw_cell_size], \
                initializer=tf.random_uniform_initializer(minval=-fw_fw_abs_val, maxval=fw_fw_abs_val))
            self._b_u = self.b_u = tf.get_variable('b_u',
                [1, self.fw_cell_size], \
                initializer=tf.constant_initializer(0.0))
            
            if self.drop_weight:
                self.W_i_mid = tf.contrib.layers.dropout(self.W_i_mid, self.weight_keep_prob, is_training=self.is_training)
                self.W_i = tf.contrib.layers.dropout(self.W_i, self.weight_keep_prob, is_training=self.is_training)
                
                self.W_f_mid = tf.contrib.layers.dropout(self.W_f_mid, self.weight_keep_prob, is_training=self.is_training)
                self.W_f_left = tf.contrib.layers.dropout(self.W_f_left, self.weight_keep_prob, is_training=self.is_training)
                self.W_f_right = tf.contrib.layers.dropout(self.W_f_right, self.weight_keep_prob, is_training=self.is_training)
                
                self.W_o_mid = tf.contrib.layers.dropout(self.W_o_mid, self.weight_keep_prob, is_training=self.is_training)
                self.W_o = tf.contrib.layers.dropout(self.W_o, self.weight_keep_prob, is_training=self.is_training)
                
                self.W_u_mid = tf.contrib.layers.dropout(self.W_u_mid, self.weight_keep_prob, is_training=self.is_training)
                self.W_u = tf.contrib.layers.dropout(self.W_u, self.weight_keep_prob, is_training=self.is_training)

        with tf.variable_scope('Projection', reuse=reuse):
            self._proj_W = self.proj_W = tf.get_variable('W',
                            [(self.fw_cell_size), self.class_size])
            self._proj_b = self.proj_b = tf.get_variable('b',
                            [1, self.class_size])
            if self.drop_weight:
                self.proj_W = tf.contrib.layers.dropout(self.proj_W, self.weight_keep_prob, is_training=self.is_training)

    def embed_word(self, word_index):
        embeddings = tf.concat([self.wv_embed, self.unk_embed], 0)
        with tf.device('/cpu:0'):
            return tf.expand_dims(tf.gather(embeddings, word_index), 0)

    def combine_children(self, x_mid, left_c, left_h, right_c, right_h):
        _i = sigmoid(tf.matmul(x_mid, self.W_i_mid) + \
                        tf.matmul(left_h, self.W_i) + \
                        tf.matmul(right_h, self.W_i) + \
                        self.b_i)
        _f_left = sigmoid(tf.matmul(x_mid, self.W_f_mid) + \
                             tf.matmul(left_h, self.W_f_left) + \
                             tf.matmul(right_h, self.W_f_left) + \
                             self.b_f)
        _f_right = sigmoid(tf.matmul(x_mid, self.W_f_mid) + \
                             tf.matmul(left_h, self.W_f_right) + \
                             tf.matmul(right_h, self.W_f_right) + \
                             self.b_f)
        _o = sigmoid(tf.matmul(x_mid, self.W_o_mid) + \
                        tf.matmul(left_h, self.W_o) + \
                        tf.matmul(right_h, self.W_o) + \
                        self.b_o)
        _u = activation(tf.matmul(x_mid, self.W_u_mid) + \
                     tf.matmul(left_h, self.W_u) + \
                     tf.matmul(right_h, self.W_u) + \
                     self.b_u)
        _c = _i * _u + _f_left * left_c + _f_right * right_c
        _h = _o * activation(_c)
        return _c, _h

    def build_model(self, left_ts, right_ts, wv_ts, target_ts, is_leaf_ts, len_ts):
        self.ground_truth = target_ts
        self.build_forward(left_ts, right_ts, wv_ts, is_leaf_ts, len_ts)
        self.build_cost(self.fw_hs, target_ts)
                    
    def build_forward(self, left_ts, right_ts, wv_ts, is_leaf_ts, len_ts):
        fw_cs = tf.TensorArray(tf.float32, size=0, dynamic_size=True, \
            clear_after_read=False, infer_shape=False)
        fw_hs = tf.TensorArray(tf.float32, size=0, dynamic_size=True, \
            clear_after_read=False, infer_shape=False)
        
        def forward_loop_body(fw_cs, fw_hs, i):
            node_is_leaf = tf.gather(is_leaf_ts, i)
            node_word_index = tf.gather(wv_ts, i)
            x_mid = tf.cond(
                tf.equal(node_is_leaf, 1),
                lambda: self.embed_word(node_word_index),
                lambda: tf.zeros([1, self.embed_size])
            )
            left_id, right_id = (tf.gather(left_ts, i), tf.gather(right_ts, i))
            left_c, left_h, right_c, right_h = tf.cond(
                tf.equal(node_is_leaf, 1),
                lambda: (tf.zeros([1, self.fw_cell_size]), tf.zeros([1, self.fw_cell_size]), tf.zeros([1, self.fw_cell_size]), tf.zeros([1, self.fw_cell_size])),
                lambda: (fw_cs.read(left_id), fw_hs.read(left_id), fw_cs.read(right_id), fw_hs.read(right_id)))
            new_c, new_h = self.combine_children(x_mid, left_c, left_h, right_c, right_h)
            if self.drop_fw_hs:
                new_h = tf.contrib.layers.dropout(new_h, self.fw_hs_keep_prob, is_training=self.is_training)
            if self.drop_fw_cs:
                new_c = tf.contrib.layers.dropout(new_c, self.fw_cs_keep_prob, is_training=self.is_training)
    
            fw_cs = fw_cs.write(i, new_c)
            fw_hs = fw_hs.write(i, new_h)
            i = tf.add(i, 1)
            return fw_cs, fw_hs, i

        forward_loop_cond = lambda fw_cs, fw_hs, i: \
                    tf.less(i, tf.squeeze(len_ts))

        fw_cs, fw_hs, _ = tf.while_loop(forward_loop_cond, forward_loop_body, [fw_cs, fw_hs, 0], parallel_iterations=1)
        self.fw_cs = fw_cs.concat()
        self.fw_hs = fw_hs.concat()
        return (fw_cs, fw_hs)

    def build_cost(self, fw_hs, label_ts):
        logits = tf.matmul(fw_hs, self.proj_W) + self.proj_b
        self.pred = tf.squeeze(tf.argmax(logits, 1))
        _loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logits, labels=label_ts)
        all_weights = [self._W_i_mid, self._W_i, self._W_f_mid, self._W_f_left, self._W_f_right, self._W_o_mid, self._W_o, self._W_u_mid, self._W_u, self._proj_W]
        self.loss = _loss + tf.reduce_sum([tf.nn.l2_loss(w) for w in all_weights]) * 0.5 * self.L2_lambda
        self.mean_loss = tf.reduce_mean(self.loss)
        self.loss_summary = tf.summary.scalar('Loss', self.mean_loss)
        if self.is_training:
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.mean_loss)
