import tensorflow as tf
import cPickle
from utils import *

class BiLSTMModel():
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
            self._fw_W_i_mid = self.fw_W_i_mid = tf.get_variable('W_i_mid',
                [self.embed_size, self.fw_cell_size])
            self._fw_W_i_left = self.fw_W_i_left = tf.get_variable('W_i_left',
                [self.fw_cell_size, self.fw_cell_size])
            self._fw_W_i_right = self.fw_W_i_right = tf.get_variable('W_i_right',
                [self.fw_cell_size, self.fw_cell_size])
            self._fw_b_i = self.fw_b_i = tf.get_variable('b_i',
                [1, self.fw_cell_size])

            self._fw_W_f_mid = self.fw_W_f_mid = tf.get_variable('W_f_mid',
                [self.embed_size, self.fw_cell_size])
            self._fw_W_f_left_left = self.fw_W_f_left_left = tf.get_variable('W_f_left_left',
                [self.fw_cell_size, self.fw_cell_size])
            self._fw_W_f_left_right = self.fw_W_f_left_right = tf.get_variable('W_f_left_right',
                [self.fw_cell_size, self.fw_cell_size])
            self._fw_W_f_right_left = self.fw_W_f_right_left = tf.get_variable('W_f_right_left',
                [self.fw_cell_size, self.fw_cell_size])
            self._fw_W_f_right_right = self.fw_W_f_right_right = tf.get_variable('W_f_right_right',
                [self.fw_cell_size, self.fw_cell_size])
            self._fw_b_f = self.fw_b_f = tf.get_variable('b_f',
                [1, self.fw_cell_size])

            self._fw_W_o_mid = self.fw_W_o_mid = tf.get_variable('W_o_mid',
                [self.embed_size, self.fw_cell_size])
            self._fw_W_o_left = self.fw_W_o_left = tf.get_variable('W_o_left',
                [self.fw_cell_size, self.fw_cell_size])
            self._fw_W_o_right = self.fw_W_o_right = tf.get_variable('W_o_right',
                [self.fw_cell_size, self.fw_cell_size])
            self._fw_b_o = self.fw_b_o = tf.get_variable('b_o',
                [1, self.fw_cell_size])

            self._fw_W_u_mid = self.fw_W_u_mid = tf.get_variable('W_u_mid',
                [self.embed_size, self.fw_cell_size])
            self._fw_W_u_left = self.fw_W_u_left = tf.get_variable('W_u_left',
                [self.fw_cell_size, self.fw_cell_size])
            self._fw_W_u_right = self.fw_W_u_right = tf.get_variable('W_u_right',
                [self.fw_cell_size, self.fw_cell_size])
            self._fw_b_u = self.fw_b_u = tf.get_variable('b_u',
                [1, self.fw_cell_size])
            
            if self.drop_weight:
                self.fw_W_i_mid = tf.contrib.layers.dropout(self.fw_W_i_mid, self.weight_keep_prob, is_training=self.is_training)
                self.fw_W_i_left = tf.contrib.layers.dropout(self.fw_W_i_left, self.weight_keep_prob, is_training=self.is_training)
                self.fw_W_i_right = tf.contrib.layers.dropout(self.fw_W_i_right, self.weight_keep_prob, is_training=self.is_training)
                
                self.fw_W_f_mid = tf.contrib.layers.dropout(self.fw_W_f_mid, self.weight_keep_prob, is_training=self.is_training)
                self.fw_W_f_left_left = tf.contrib.layers.dropout(self.fw_W_f_left_left, self.weight_keep_prob, is_training=self.is_training)
                self.fw_W_f_left_right = tf.contrib.layers.dropout(self.fw_W_f_left_right, self.weight_keep_prob, is_training=self.is_training)
                self.fw_W_f_right_left = tf.contrib.layers.dropout(self.fw_W_f_right_left, self.weight_keep_prob, is_training=self.is_training)
                self.fw_W_f_right_right = tf.contrib.layers.dropout(self.fw_W_f_right_right, self.weight_keep_prob, is_training=self.is_training)
                
                self.fw_W_o_mid = tf.contrib.layers.dropout(self.fw_W_o_mid, self.weight_keep_prob, is_training=self.is_training)
                self.fw_W_o_left = tf.contrib.layers.dropout(self.fw_W_o_left, self.weight_keep_prob, is_training=self.is_training)
                self.fw_W_o_right = tf.contrib.layers.dropout(self.fw_W_o_right, self.weight_keep_prob, is_training=self.is_training)
                
                self.fw_W_u_mid = tf.contrib.layers.dropout(self.fw_W_u_mid, self.weight_keep_prob, is_training=self.is_training)
                self.fw_W_u_left = tf.contrib.layers.dropout(self.fw_W_u_left, self.weight_keep_prob, is_training=self.is_training)
                self.fw_W_u_right = tf.contrib.layers.dropout(self.fw_W_u_right, self.weight_keep_prob, is_training=self.is_training)

        with tf.variable_scope('Backward', reuse=reuse):
            self._bw_W_i_p = self.bw_W_i_p = tf.get_variable('W_i_p',
                [self.fw_cell_size+self.bw_cell_size, self.bw_cell_size])
            self._bw_W_i_left = self.bw_W_i_left = tf.get_variable('W_i_left',
                [self.fw_cell_size, self.bw_cell_size])
            self._bw_W_i_right = self.bw_W_i_right = tf.get_variable('W_i_right',
                [self.fw_cell_size, self.bw_cell_size])
            self._bw_b_i_left = self.bw_b_i_left = tf.get_variable('b_i_left',
                [1, self.bw_cell_size])
            self._bw_b_i_right = self.bw_b_i_right = tf.get_variable('b_i_right',
                [1, self.bw_cell_size])

            self._bw_W_f_p = self.bw_W_f_p = tf.get_variable('W_f_p',
                [self.fw_cell_size+self.bw_cell_size, self.bw_cell_size])
            self._bw_W_f_left = self.bw_W_f_left = tf.get_variable('W_f_left',
                [self.fw_cell_size, self.bw_cell_size])
            self._bw_W_f_right = self.bw_W_f_right = tf.get_variable('W_f_right',
                [self.fw_cell_size, self.bw_cell_size])
            self._bw_b_f_left = self.bw_b_f_left = tf.get_variable('b_f_left',
                [1, self.bw_cell_size])
            self._bw_b_f_right = self.bw_b_f_right = tf.get_variable('b_f_right',
                [1, self.bw_cell_size])

            self._bw_W_o_p = self.bw_W_o_p = tf.get_variable('W_o_p',
                [self.fw_cell_size+self.bw_cell_size, self.bw_cell_size])
            self._bw_W_o_left = self.bw_W_o_left = tf.get_variable('W_o_left',
                [self.fw_cell_size, self.bw_cell_size])
            self._bw_W_o_right = self.bw_W_o_right = tf.get_variable('W_o_right',
                [self.fw_cell_size, self.bw_cell_size])
            self._bw_b_o_left = self.bw_b_o_left = tf.get_variable('b_o_left',
                [1, self.bw_cell_size])
            self._bw_b_o_right = self.bw_b_o_right = tf.get_variable('b_o_right',
                [1, self.bw_cell_size])

            self._bw_W_u_p = self.bw_W_u_p = tf.get_variable('W_u_p',
                [self.fw_cell_size+self.bw_cell_size, self.bw_cell_size])
            self._bw_W_u_left = self.bw_W_u_left = tf.get_variable('W_u_left',
                [self.fw_cell_size, self.bw_cell_size])
            self._bw_W_u_right = self.bw_W_u_right = tf.get_variable('W_u_right',
                [self.fw_cell_size, self.bw_cell_size])
            self._bw_b_u_left = self.bw_b_u_left = tf.get_variable('b_u_left',
                [1, self.bw_cell_size])
            self._bw_b_u_right = self.bw_b_u_right = tf.get_variable('b_u_right',
                [1, self.bw_cell_size])
            
            if self.drop_weight:
                self.bw_W_i_p = tf.contrib.layers.dropout(self.bw_W_i_p, self.weight_keep_prob, is_training=self.is_training)
                self.bw_W_i_left = tf.contrib.layers.dropout(self.bw_W_i_left, self.weight_keep_prob, is_training=self.is_training)
                self.bw_W_i_right = tf.contrib.layers.dropout(self.bw_W_i_right, self.weight_keep_prob, is_training=self.is_training)
                
                self.bw_W_f_p = tf.contrib.layers.dropout(self.bw_W_f_p, self.weight_keep_prob, is_training=self.is_training)
                self.bw_W_f_left = tf.contrib.layers.dropout(self.bw_W_f_left, self.weight_keep_prob, is_training=self.is_training)
                self.bw_W_f_right = tf.contrib.layers.dropout(self.bw_W_f_right, self.weight_keep_prob, is_training=self.is_training)
                
                self.bw_W_o_p = tf.contrib.layers.dropout(self.bw_W_o_p, self.weight_keep_prob, is_training=self.is_training)
                self.bw_W_o_left = tf.contrib.layers.dropout(self.bw_W_o_left, self.weight_keep_prob, is_training=self.is_training)
                self.bw_W_o_right = tf.contrib.layers.dropout(self.bw_W_o_right, self.weight_keep_prob, is_training=self.is_training)
                
                self.bw_W_u_p = tf.contrib.layers.dropout(self.bw_W_u_p, self.weight_keep_prob, is_training=self.is_training)
                self.bw_W_u_left = tf.contrib.layers.dropout(self.bw_W_u_left, self.weight_keep_prob, is_training=self.is_training)
                self.bw_W_u_right = tf.contrib.layers.dropout(self.bw_W_u_right, self.weight_keep_prob, is_training=self.is_training)
                
        with tf.variable_scope('Projection', reuse=reuse):
            self._proj_W = self.proj_W = tf.get_variable('W',
                            [(self.fw_cell_size+self.bw_cell_size), self.class_size])
            self._proj_b = self.proj_b = tf.get_variable('b',
                            [1, self.class_size])
            if self.drop_weight:
                self.proj_W = tf.contrib.layers.dropout(self.proj_W, self.weight_keep_prob, is_training=self.is_training)

    def embed_word(self, word_index):
        embeddings = tf.concat([self.wv_embed, self.unk_embed], 0)
        with tf.device('/cpu:0'):
            return tf.expand_dims(tf.gather(embeddings, word_index), 0)

    def combine_children(self, x_mid, left_c, left_h, right_c, right_h):
        _i = sigmoid(tf.matmul(x_mid, self.fw_W_i_mid) + \
                        tf.matmul(left_h, self.fw_W_i_left) + \
                        tf.matmul(right_h, self.fw_W_i_right) + \
                        self.fw_b_i)
        _f_left = sigmoid(tf.matmul(x_mid, self.fw_W_f_mid) + \
                             tf.matmul(left_h, self.fw_W_f_left_left) + \
                             tf.matmul(right_h, self.fw_W_f_left_right) + \
                             self.fw_b_f)
        _f_right = sigmoid(tf.matmul(x_mid, self.fw_W_f_mid) + \
                              tf.matmul(left_h, self.fw_W_f_right_left) + \
                              tf.matmul(right_h, self.fw_W_f_right_right) + \
                              self.fw_b_f)
        _o = sigmoid(tf.matmul(x_mid, self.fw_W_o_mid) + \
                        tf.matmul(left_h, self.fw_W_o_left) + \
                        tf.matmul(right_h, self.fw_W_o_right) + \
                        self.fw_b_o)
        _u = activation(tf.matmul(x_mid, self.fw_W_u_mid) + \
                     tf.matmul(left_h, self.fw_W_u_left) + \
                     tf.matmul(right_h, self.fw_W_u_right) + \
                     self.fw_b_u)
        _c = _i * _u + _f_left * left_c + _f_right * right_c
        _h = _o * activation(_c)
        return _c, _h
        
    def decompose_children(self, left_fw_h, right_fw_h, parent_join_h, parent_c):
        _i_left = sigmoid(tf.matmul(left_fw_h, self.bw_W_i_left) + \
                             tf.matmul(parent_join_h, self.bw_W_i_p) + \
                             self.bw_b_i_left)
        _i_right = sigmoid(tf.matmul(right_fw_h, self.bw_W_i_right) + \
                              tf.matmul(parent_join_h, self.bw_W_i_p) + \
                              self.bw_b_i_right)
        _f_left = sigmoid(tf.matmul(left_fw_h, self.bw_W_f_left) + \
                             tf.matmul(parent_join_h, self.bw_W_f_p) + \
                             self.bw_b_f_left)
        _f_right = sigmoid(tf.matmul(right_fw_h, self.bw_W_f_right) + \
                              tf.matmul(parent_join_h, self.bw_W_f_p) + \
                              self.bw_b_f_right)
        _o_left = sigmoid(tf.matmul(left_fw_h, self.bw_W_o_left) + \
                             tf.matmul(parent_join_h, self.bw_W_o_p) + \
                             self.bw_b_o_left)
        _o_right = sigmoid(tf.matmul(right_fw_h, self.bw_W_o_right) + \
                              tf.matmul(parent_join_h, self.bw_W_o_p) + \
                              self.bw_b_o_right)
        _u_left = activation(tf.matmul(left_fw_h, self.bw_W_u_left) + \
                          tf.matmul(parent_join_h, self.bw_W_u_p) + \
                          self.bw_b_u_left)
        _u_right = activation(tf.matmul(right_fw_h, self.bw_W_u_right) + \
                           tf.matmul(parent_join_h, self.bw_W_u_p) + \
                           self.bw_b_u_right)

        _c_left = _i_left * _u_left + _f_left * parent_c
        _c_right = _i_right * _u_right + _f_right * parent_c

        _h_left = _o_left * tf.tanh(_c_left)
        _h_right = _o_right * tf.tanh(_c_right)

        return _c_left, _h_left, _c_right, _h_right

    def build_model(self, left_ts, right_ts, wv_ts, target_ts, is_leaf_ts, len_ts):
        fw_cs, fw_hs = self.build_forward(left_ts, right_ts, wv_ts, is_leaf_ts, len_ts)
        self.build_backward(left_ts, right_ts, wv_ts, is_leaf_ts, len_ts, fw_hs)
        self.fw_bw_hs = tf.concat([self.fw_hs, self.bw_hs], 1)
        self.build_cost(self.fw_bw_hs, target_ts)
                    
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
    
    def build_backward(self, left_ts, right_ts, wv_ts, is_leaf_ts, len_ts, fw_hs):
        bw_cs = tf.TensorArray(tf.float32, size=0, dynamic_size=True, \
            clear_after_read=False, infer_shape=False)
        bw_hs = tf.TensorArray(tf.float32, size=0, dynamic_size=True, \
            clear_after_read=False, infer_shape=False)
        input_len = tf.squeeze(len_ts)
        bw_cs = bw_cs.write(input_len-1, tf.zeros([1, self.bw_cell_size]))
        bw_hs = bw_hs.write(input_len-1, tf.zeros([1, self.bw_cell_size]))
        
        def backward_loop_body(bw_cs, bw_hs, i):
            def fn1(bw_cs, bw_hs, i):
                parent_bw_h = bw_hs.read(i)
                parent_fw_h = fw_hs.read(i)
                parent_join_h = tf.concat([parent_bw_h, parent_fw_h], 1)
                parent_c = bw_cs.read(i)

                left_id = tf.gather(left_ts, i)
                right_id = tf.gather(right_ts, i)
                left_fw_h = fw_hs.read(left_id)
                right_fw_h = fw_hs.read(right_id)

                new_left_c, new_left_h, new_right_c, new_right_h = \
                    self.decompose_children(left_fw_h, right_fw_h, parent_join_h, parent_c)
                if self.drop_bw_hs:
                    new_left_h = tf.contrib.layers.dropout(new_left_h, self.bw_hs_keep_prob, is_training=self.is_training)
                    new_right_h = tf.contrib.layers.dropout(new_right_h, self.bw_hs_keep_prob, is_training=self.is_training)
                if self.drop_bw_cs:
                    new_left_c = tf.contrib.layers.dropout(new_left_c, self.bw_hs_keep_prob, is_training=self.is_training)
                    new_right_c = tf.contrib.layers.dropout(new_right_c, self.bw_hs_keep_prob, is_training=self.is_training)
            
                bw_cs = bw_cs.write(left_id, new_left_c)
                bw_cs = bw_cs.write(right_id, new_right_c)
                bw_hs = bw_hs.write(left_id, new_left_h)
                bw_hs = bw_hs.write(right_id, new_right_h)
                return bw_cs, bw_hs, i

            bw_cs, bw_hs, i = tf.cond(tf.equal(tf.gather(is_leaf_ts, i), False), \
                lambda: fn1(bw_cs, bw_hs, i),
                lambda: (bw_cs, bw_hs, i))
            
            i = i - 1
            return bw_cs, bw_hs, i

        backward_loop_cond  = lambda bw_cs, bw_hs, i: \
                              tf.greater_equal(i, 0)

        bw_cs, bw_hs, _ = tf.while_loop(backward_loop_cond, backward_loop_body, [bw_cs, bw_hs, input_len-1], \
                                             parallel_iterations=1)
        self.bw_cs = bw_cs.concat()
        self.bw_hs = bw_hs.concat()

    def build_cost(self, fw_bw_hs, label_ts):
        logits = tf.matmul(fw_bw_hs, self.proj_W) + self.proj_b
        self.pred = tf.squeeze(tf.argmax(logits, 1))
        _loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logits, labels=label_ts)
        all_weights = [self._fw_W_i_mid, self._fw_W_i_left, self._fw_W_i_right, self._fw_W_f_mid, self._fw_W_f_left_left, self._fw_W_f_left_right, self._fw_W_f_right_left, self._fw_W_f_right_right, self._fw_W_o_mid, self._fw_W_o_left, self._fw_W_o_right, self._fw_W_u_mid, self._fw_W_u_left, self._fw_W_u_right, \
            self._bw_W_i_p, self._bw_W_i_left, self._bw_W_i_right, self._bw_W_f_p, self._bw_W_f_left, self._bw_W_f_right, self._bw_W_o_p, self._bw_W_o_left, self._bw_W_o_right, self._bw_W_u_p, self._bw_W_u_left, self._bw_W_u_right, \
            self._proj_W]
        self.loss = _loss + tf.reduce_sum([tf.nn.l2_loss(w) for w in all_weights]) * 0.5 * self.L2_lambda
        self.mean_loss = tf.reduce_mean(self.loss)
        self.loss_summary = tf.summary.scalar('Loss', self.mean_loss)
        if self.is_training:
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.mean_loss)
