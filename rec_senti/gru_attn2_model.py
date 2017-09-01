import tensorflow as tf
import cPickle
from utils import *

class GRUAttn2Model():
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
            self._gru_W1 = self.gru_W1 = tf.get_variable('GRU_W1',
                [self.embed_size + 2*self.fw_cell_size, 4*self.fw_cell_size])
            self._gru_b1 = self.gru_b1 = tf.get_variable('GRU_b1',
                [1, 4*self.fw_cell_size], \
                initializer=tf.constant_initializer(0.0))
            self._gru_W1_h= self.gru_W1_h = tf.get_variable('GRU_W1_h',
                [self.embed_size + 2*self.fw_cell_size, self.fw_cell_size])
            self._gru_b1_h = self.gru_b1_h = tf.get_variable('GRU_b1_h',
                [1, self.fw_cell_size], \
                initializer=tf.constant_initializer(0.0))
            self._gru_trans1 = self.gru_trans1 = tf.get_variable('GRU_trans1',
                [2*self.fw_cell_size, self.fw_cell_size])
            
            self._gru_W2 = self.gru_W2 = tf.get_variable('GRU_W2',
                [3*self.fw_cell_size, 4*self.fw_cell_size])
            self._gru_b2 = self.gru_b2 = tf.get_variable('GRU_b2',
                [1, 4*self.fw_cell_size], \
                initializer=tf.constant_initializer(0.0))
            self._gru_W2_h= self.gru_W2_h = tf.get_variable('GRU_W2_h',
                [3*self.fw_cell_size, self.fw_cell_size])
            self._gru_b2_h = self.gru_b2_h = tf.get_variable('GRU_b2_h',
                [1, self.fw_cell_size], \
                initializer=tf.constant_initializer(0.0))
            self._gru_trans2 = self.gru_trans2 = tf.get_variable('GRU_trans2',
                [2*self.fw_cell_size, self.fw_cell_size])
                            
        with tf.variable_scope('Attn', reuse=reuse):
            self._attn1_W1 = self.attn1_W1 = tf.get_variable('attn1_W1',
                [self.fw_cell_size, self.attn_size])
            self._attn1_W2 = self.attn1_W2 = tf.get_variable('attn1_W2',
                [self.fw_cell_size, self.attn_size])
            self._attn1_b = self.attn1_b = tf.get_variable('attn1_b',
                [self.attn_size])

            self._attn2_W1 = self.attn2_W1 = tf.get_variable('attn2_W1',
                [self.fw_cell_size, self.attn_size])
            self._attn2_W2 = self.attn2_W2 = tf.get_variable('attn2_W2',
                [self.fw_cell_size, self.attn_size])
            self._attn2_b = self.attn2_b = tf.get_variable('attn2_b',
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

    def combine_children1(self, x_mid, left_s, right_s):
        _concat = tf.matmul(tf.concat([x_mid, left_s, right_s], 1), self.gru_W1) + self.gru_b1
        _concat = sigmoid(_concat)

        # r = reset_gate, z = update_gate
        r0, r1, z0, z1 = tf.split(value=_concat, num_or_size_splits=4, axis=1)
        
        zs = tf.concat([tf.zeros([1, self.fw_cell_size]), z0, z1], axis=0)
        zs = tf.nn.softmax(zs, dim=0)
        
        h = activation(tf.matmul(tf.concat([x_mid, left_s*r0, right_s*r1], axis=1), self.gru_W1_h) + self.gru_b1_h)
        h = tf.contrib.layers.dropout(h, self.rec_keep_prob, is_training=self.is_training)
        
        _z, _z0, _z1 = tf.gather(zs, 0), tf.gather(zs, 1), tf.gather(zs, 2)
        new_s = _z * h + _z0 * left_s + _z1 * right_s

        return new_s, h

    def combine_children2(self, x_mid, left_s, right_s):
        _concat = tf.matmul(tf.concat([[x_mid], left_s, right_s], 1), self.gru_W2) + self.gru_b2
        _concat = sigmoid(_concat)

        # r = reset_gate, z = update_gate
        r0, r1, z0, z1 = tf.split(value=_concat, num_or_size_splits=4, axis=1)
        
        zs = tf.concat([tf.zeros([1, self.fw_cell_size]), z0, z1], axis=0)
        zs = tf.nn.softmax(zs, dim=0)
        
        h = activation(tf.matmul(tf.concat([[x_mid], left_s*r0, right_s*r1], axis=1), self.gru_W2_h) + self.gru_b2_h)
        h = tf.contrib.layers.dropout(h, self.rec_keep_prob, is_training=self.is_training)
        
        _z, _z0, _z1 = tf.gather(zs, 0), tf.gather(zs, 1), tf.gather(zs, 2)
        new_s = _z * h + _z0 * left_s + _z1 * right_s

        return new_s, h

    def build_model(self, left_ts, right_ts, wv_ts, target_ts, is_leaf_ts, len_ts, mask):
        self.ground_truth = target_ts
        # len x h
        hs1 = self.build_forward1(left_ts, right_ts, wv_ts, is_leaf_ts, len_ts)
        # len x h
        cont1 = self.build_attn1(hs1, len_ts, mask)
        # len x h
        _hs1 = tf.matmul(tf.concat([hs1, cont1], 1), self.gru_trans1)
        
        # len x h
        hs2 = self.build_forward2(left_ts, right_ts, hs1, is_leaf_ts, len_ts)
        # len x h
        cont2 = self.build_attn2(hs2, len_ts, mask)
        # len x h
        _hs2 = tf.matmul(tf.concat([hs2, cont2], 1), self.gru_trans2)
        
        self.build_cost(_hs2, target_ts)
        self.cont1 = cont1
        self.cont2 = cont2
                    
    def build_forward1(self, left_ts, right_ts, wv_ts, is_leaf_ts, len_ts):
        fw_ss = tf.TensorArray(tf.float32, size=0, dynamic_size=True, \
            clear_after_read=False, infer_shape=True)
        fw_hs = tf.TensorArray(tf.float32, size=0, dynamic_size=True, \
            clear_after_read=False, infer_shape=True)
        
        def forward_loop_body(fw_ss, fw_hs, i):
            node_is_leaf = tf.gather(is_leaf_ts, i)
            node_word_index = tf.gather(wv_ts, i)
            x_mid = tf.cond(
                tf.equal(node_is_leaf, 1),
                lambda: self.embed_word(node_word_index),
                lambda: tf.zeros([1, self.embed_size])
            )
            left_id, right_id = (tf.gather(left_ts, i), tf.gather(right_ts, i))
            left_s, right_s = tf.cond(
                tf.equal(node_is_leaf, 1),
                lambda: (tf.zeros([1, self.fw_cell_size]), tf.zeros([1, self.fw_cell_size])),
                lambda: (fw_ss.read(left_id), fw_ss.read(right_id)))
            new_s, new_h = self.combine_children1(x_mid, left_s, right_s)
    
            fw_ss = fw_ss.write(i, new_s)
            fw_hs = fw_hs.write(i, new_h)
            i = tf.add(i, 1)
            return fw_ss, fw_hs, i

        forward_loop_cond = lambda fw_ss, fw_hs, i: \
                    tf.less(i, tf.squeeze(len_ts))

        fw_ss, fw_hs, _ = tf.while_loop(forward_loop_cond, forward_loop_body, [fw_ss, fw_hs, 0], parallel_iterations=1)
        self.fw_ss = fw_ss.concat()
        self.fw_hs = fw_hs.concat()
        return fw_hs.concat()
                    
    def build_forward2(self, left_ts, right_ts, hs, is_leaf_ts, len_ts):
        fw_ss = tf.TensorArray(tf.float32, size=0, dynamic_size=True, \
            clear_after_read=False, infer_shape=True)
        fw_hs = tf.TensorArray(tf.float32, size=0, dynamic_size=True, \
            clear_after_read=False, infer_shape=True)
        
        def forward_loop_body(fw_ss, fw_hs, i):
            node_is_leaf = tf.gather(is_leaf_ts, i)
            x_mid = tf.gather(hs, i)
            left_id, right_id = (tf.gather(left_ts, i), tf.gather(right_ts, i))
            left_s, right_s = tf.cond(
                tf.equal(node_is_leaf, 1),
                lambda: (tf.zeros([1, self.fw_cell_size]), tf.zeros([1, self.fw_cell_size])),
                lambda: (fw_ss.read(left_id), fw_ss.read(right_id)))
            new_s, new_h = self.combine_children2(x_mid, left_s, right_s)
    
            fw_ss = fw_ss.write(i, new_s)
            fw_hs = fw_hs.write(i, new_h)
            i = tf.add(i, 1)
            return fw_ss, fw_hs, i

        forward_loop_cond = lambda fw_ss, fw_hs, i: \
                    tf.less(i, tf.squeeze(len_ts))

        fw_ss, fw_hs, _ = tf.while_loop(forward_loop_cond, forward_loop_body, [fw_ss, fw_hs, 0], parallel_iterations=1)
        self.fw_ss = fw_ss.concat()
        self.fw_hs = fw_hs.concat()
        return fw_hs.concat()
    
    def build_attn1(self, hs, len_ts, mask):
        len_ts = tf.squeeze(len_ts)
        mask = tf.reshape(mask, [len_ts, len_ts])
        # len x h
        em = hs
        em = tf.reshape(em, [len_ts, -1])
        # len x attn
        em_W1, em_W2 = tf.matmul(em, self.attn1_W1), tf.matmul(em, self.attn1_W2)
        # len x len x attn
        em_ex_W1 = tf.tile(tf.expand_dims(em_W1, 1), [1, len_ts, 1])
        em_ex_W2 = tf.tile(tf.expand_dims(em_W2, 0), [len_ts, 1, 1])
        # len x len x attn
        tmp = em_ex_W1 + em_ex_W2
        tmp = activation(tmp)
        # len x len
        _as = tf.einsum('ijk,k->ij', tmp, self.attn1_b)
        _as = tf.nn.softmax(_as)
        _as = _as * mask
        # len x len x h
        cont_h = tf.einsum('ij,jk->ijk', _as, em)
        # len x h
        cont = tf.reduce_sum(cont_h, axis=1)
        self.attn_vecs1 = _as
        return cont
    
    def build_attn2(self, hs, len_ts, mask):
        len_ts = tf.squeeze(len_ts)
        mask = tf.reshape(mask, [len_ts, len_ts])
        # len x h
        em = hs
        em = tf.reshape(em, [len_ts, -1])
        # len x attn
        em_W1, em_W2 = tf.matmul(em, self.attn2_W1), tf.matmul(em, self.attn2_W2)
        # len x len x attn
        em_ex_W1 = tf.tile(tf.expand_dims(em_W1, 1), [1, len_ts, 1])
        em_ex_W2 = tf.tile(tf.expand_dims(em_W2, 0), [len_ts, 1, 1])
        # len x len x attn
        tmp = em_ex_W1 + em_ex_W2
        tmp = activation(tmp)
        # len x len
        _as = tf.einsum('ijk,k->ij', tmp, self.attn2_b)
        _as = tf.nn.softmax(_as)
        _as = _as * mask
        # len x len x h
        cont_h = tf.einsum('ij,jk->ijk', _as, em)
        # len x h
        cont = tf.reduce_sum(cont_h, axis=1)
        self.attn_vecs2 = _as
        return cont

    def build_cost(self, hs, label_ts):
        hs = tf.contrib.layers.dropout(hs, self.output_keep_prob, is_training=self.is_training)
        logits = tf.matmul(hs, self.proj_W) + self.proj_b
        softmax = tf.nn.softmax(logits)
        self.pred = tf.squeeze(tf.argmax(logits, 1))
        self.binary_pred = tf.to_int32((softmax[:, 3] + softmax[:, 4]) > (softmax[:, 0] + softmax[:, 1]))
        _loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logits, labels=label_ts)
        all_weights = [self._gru_W1, self._gru_W2, self._proj_W]
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
