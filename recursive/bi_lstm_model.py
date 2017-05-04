import tensorflow as tf
import cPickle

class BiLSTMModel():
    def __init__(self, config_dict):        
        self.__dict__.update(config_dict)
        self.emb_mat = cPickle.load(open(self.wv_emb_file))

    def add_variables(self):
        with tf.variable_scope('Embeddings'):
            wv_emb = tf.get_variable('wv_embed',
                                     [self.wv_vocab_size, self.embed_size], \
                                     trainable=False)
            pos_emb = tf.get_variable('pos_embed',
                [self.pos_vocab_size, self.embed_size])
        with tf.variable_scope('Forward'):
            tf.get_variable('W_i_mid',
                [self.embed_size, self.fw_cell_size])
            tf.get_variable('W_i_left',
                [self.fw_cell_size, self.fw_cell_size])
            tf.get_variable('W_i_right',
                [self.fw_cell_size, self.fw_cell_size])
            tf.get_variable('b_i',
                [1, self.fw_cell_size])

            tf.get_variable('W_f_mid',
                [self.embed_size, self.fw_cell_size])
            tf.get_variable('W_f_left_left',
                [self.fw_cell_size, self.fw_cell_size])
            tf.get_variable('W_f_left_right',
                [self.fw_cell_size, self.fw_cell_size])
            tf.get_variable('W_f_right_left',
                [self.fw_cell_size, self.fw_cell_size])
            tf.get_variable('W_f_right_right',
                [self.fw_cell_size, self.fw_cell_size])
            tf.get_variable('b_f',
                [1, self.fw_cell_size])

            tf.get_variable('W_o_mid',
                [self.embed_size, self.fw_cell_size])
            tf.get_variable('W_o_left',
                [self.fw_cell_size, self.fw_cell_size])
            tf.get_variable('W_o_right',
                [self.fw_cell_size, self.fw_cell_size])
            tf.get_variable('b_o',
                [1, self.fw_cell_size])

            tf.get_variable('W_u_mid',
                [self.embed_size, self.fw_cell_size])
            tf.get_variable('W_u_left',
                [self.fw_cell_size, self.fw_cell_size])
            tf.get_variable('W_u_right',
                [self.fw_cell_size, self.fw_cell_size])
            tf.get_variable('b_u',
                [1, self.fw_cell_size])
        with tf.variable_scope('Backward'):
            tf.get_variable('W_i_left',
                [self.fw_cell_size, self.bw_cell_size])
            tf.get_variable('W_i_right',
                [self.fw_cell_size, self.bw_cell_size])
            tf.get_variable('W_i_p',
                [self.fw_cell_size+self.bw_cell_size, self.bw_cell_size])
            tf.get_variable('b_i',
                [1, self.bw_cell_size])

            tf.get_variable('W_f_left',
                [self.fw_cell_size, self.bw_cell_size])
            tf.get_variable('W_f_right',
                [self.fw_cell_size, self.bw_cell_size])
            tf.get_variable('W_f_p',
                [self.fw_cell_size+self.bw_cell_size, self.bw_cell_size])
            tf.get_variable('b_f',
                [1, self.bw_cell_size])

            tf.get_variable('W_o_left',
                [self.fw_cell_size, self.bw_cell_size])
            tf.get_variable('W_o_right',
                [self.fw_cell_size, self.bw_cell_size])
            tf.get_variable('W_o_p',
                [self.fw_cell_size+self.bw_cell_size, self.bw_cell_size])
            tf.get_variable('b_o',
                [1, self.bw_cell_size])

            tf.get_variable('W_u_left',
                [self.fw_cell_size, self.bw_cell_size])
            tf.get_variable('W_u_right',
                [self.fw_cell_size, self.bw_cell_size])
            tf.get_variable('W_u_p',
                [self.fw_cell_size+self.bw_cell_size, self.bw_cell_size])
            tf.get_variable('b_u',
                [1, self.bw_cell_size])

        with tf.variable_scope('Projection'):
            tf.get_variable('W',
                            [(self.fw_cell_size+self.bw_cell_size), 2])
            tf.get_variable('b',
                            [1, 2])

    def embed_word(self, word_index):
        with tf.variable_scope('Embeddings', reuse=True):
            wv_embed = tf.get_variable('wv_embed')
            pos_embed = tf.get_variable('pos_embed')
            embeddings = tf.concat([wv_embed, pos_embed], 0)
        with tf.device('/cpu:0'):
            return tf.expand_dims(tf.gather(embeddings, word_index), 0)

    def combine_children(self, x_mid, left_c, left_h, right_c, right_h):
        with tf.variable_scope('Forward', reuse=True):
            W_i_mid = tf.get_variable('W_i_mid')
            W_i_left = tf.get_variable('W_i_left')
            W_i_right = tf.get_variable('W_i_right')
            b_i = tf.get_variable('b_i')

            W_f_mid = tf.get_variable('W_f_mid')
            W_f_left_left = tf.get_variable('W_f_left_left')
            W_f_left_right = tf.get_variable('W_f_left_right')
            W_f_right_left = tf.get_variable('W_f_right_left')
            W_f_right_right = tf.get_variable('W_f_right_right')
            b_f = tf.get_variable('b_f')

            W_o_mid = tf.get_variable('W_o_mid')
            W_o_left = tf.get_variable('W_o_left')
            W_o_right = tf.get_variable('W_o_right')
            b_o = tf.get_variable('b_o')

            W_u_mid = tf.get_variable('W_u_mid')
            W_u_left = tf.get_variable('W_u_left')
            W_u_right = tf.get_variable('W_u_right')
            b_u = tf.get_variable('b_u')
        _i = tf.sigmoid(tf.matmul(x_mid, W_i_mid) + \
                        tf.matmul(left_h, W_i_left) + \
                        tf.matmul(right_h, W_i_right) + \
                        b_i)
        _f_left = tf.sigmoid(tf.matmul(x_mid, W_f_mid) + \
                             tf.matmul(left_h, W_f_left_left) + \
                             tf.matmul(right_h, W_f_left_right) + \
                             b_f)
        _f_right = tf.sigmoid(tf.matmul(x_mid, W_f_mid) + \
                              tf.matmul(right_h, W_f_right_left) + \
                              tf.matmul(right_h, W_f_right_right) + \
                              b_f)
        _o = tf.sigmoid(tf.matmul(x_mid, W_o_mid) + \
                        tf.matmul(left_h, W_o_left) + \
                        tf.matmul(right_h, W_o_right) + \
                        b_o)
        _u = tf.tanh(tf.matmul(x_mid, W_u_mid) + \
                     tf.matmul(left_h, W_u_left) + \
                     tf.matmul(right_h, W_u_right) + \
                     b_u)
        _c = _i * _u + _f_left * left_c + _f_right * right_c
        _h = _o * tf.tanh(_c)
        return _c, _h

    def decompose_children(self, emb_left, emb_right, parent_join, parent_c):
        with tf.variable_scope('Backward', reuse=True):
            W_i_left = tf.get_variable('W_i_left')
            W_i_right = tf.get_variable('W_i_right')
            W_i_p = tf.get_variable('W_i_p')
            b_i = tf.get_variable('b_i')

            W_f_left = tf.get_variable('W_f_left')
            W_f_right = tf.get_variable('W_f_right')
            W_f_p = tf.get_variable('W_f_p')
            b_f = tf.get_variable('b_f')

            W_o_left = tf.get_variable('W_o_left')
            W_o_right = tf.get_variable('W_o_right')
            W_o_p = tf.get_variable('W_o_p')
            b_o = tf.get_variable('b_o')

            W_u_left = tf.get_variable('W_u_left')
            W_u_right = tf.get_variable('W_u_right')
            W_u_p = tf.get_variable('W_u_p')
            b_u = tf.get_variable('b_u')
        _i_left = tf.sigmoid(tf.matmul(emb_left, W_i_left) + \
                             tf.matmul(parent_join, W_i_p) + \
                             b_i)
        _i_right = tf.sigmoid(tf.matmul(emb_right, W_i_right) + \
                              tf.matmul(parent_join, W_i_p) + \
                              b_i)
        _f_left = tf.sigmoid(tf.matmul(emb_left, W_f_left) + \
                             tf.matmul(parent_join, W_f_p) + \
                             b_f)
        _f_right = tf.sigmoid(tf.matmul(emb_right, W_f_right) + \
                              tf.matmul(parent_join, W_f_p) + \
                              b_f)
        _o_left = tf.sigmoid(tf.matmul(emb_left, W_o_left) + \
                             tf.matmul(parent_join, W_o_p) + \
                             b_o)
        _o_right = tf.sigmoid(tf.matmul(emb_right, W_o_right) + \
                              tf.matmul(parent_join, W_o_p) + \
                              b_o)
        _u_left = tf.tanh(tf.matmul(emb_left, W_u_left) + \
                          tf.matmul(parent_join, W_u_p) + \
                          b_u)
        _u_right = tf.tanh(tf.matmul(emb_right, W_u_right) + \
                           tf.matmul(parent_join, W_u_p) + \
                           b_u)

        _c_left = _i_left * _u_left + _f_left * parent_c
        _c_right = _i_right * _u_right + _f_right * parent_c

        _h_left = _o_left * tf.tanh(_c_left)
        _h_right = _o_right * tf.tanh(_c_right)

        return _c_left, _h_left, _c_right, _h_right

    def build_model(self, left_ts, right_ts, wv_ts, target_ts, is_leaf_ts, len_ts):
        fw_cs, fw_hs = self.build_forward(left_ts, right_ts, wv_ts, is_leaf_ts, len_ts)
        input_len = tf.squeeze(len_ts)
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
            x_mid = self.embed_word(node_word_index)
            left_id, right_id = (tf.gather(left_ts, i), tf.gather(right_ts, i))
            left_c, left_h, right_c, right_h = tf.cond(
                tf.equal(node_is_leaf, 1),
                lambda: (tf.zeros([1, self.fw_cell_size]), tf.zeros([1, self.fw_cell_size]), tf.zeros([1, self.fw_cell_size]), tf.zeros([1, self.fw_cell_size])),
                lambda: (fw_cs.read(left_id), fw_hs.read(left_id), fw_cs.read(right_id), fw_hs.read(right_id)))
            new_c, new_h = self.combine_children(x_mid, left_c, left_h, right_c, right_h)
    
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
        
        def backward_loop_body(bw_cs, bw_hs, fw_hs, i):
            def fn1(bw_cs, bw_hs, fw_hs, i):
                parent_h = bw_hs.read(i)
                parent_fw = fw_hs.read(i)
                parent_join = tf.concat([parent_h, parent_fw], 1)
                parent_c = bw_cs.read(i)

                left_id = tf.gather(left_ts, i)
                right_id = tf.gather(right_ts, i)
                emb_left = fw_hs.read(left_id)
                emb_right = fw_hs.read(right_id)

                new_left_c, new_left_h, new_right_c, new_right_h = \
                    self.decompose_children(emb_left, emb_right, parent_join, parent_c)
            
                bw_cs = bw_cs.write(left_id, new_left_c)
                bw_cs = bw_cs.write(right_id, new_right_c)
                bw_hs = bw_hs.write(left_id, new_left_h)
                bw_hs = bw_hs.write(right_id, new_right_h)
                
                return bw_cs, bw_hs, fw_hs, i

            bw_cs, bw_hs, fw_hs, i = tf.cond(tf.equal(tf.gather(is_leaf_ts, i), False), \
                lambda: fn1(bw_cs, bw_hs, fw_hs, i),
                lambda: (bw_cs, bw_hs, fw_hs, i))
            i = i - 1
            return bw_cs, bw_hs, fw_hs, i

        backward_loop_cond  = lambda bw_cs, bw_hs, fw_hs, i: \
                              tf.greater_equal(i, 0)

        bw_cs, bw_hs, _, _ = tf.while_loop(backward_loop_cond, backward_loop_body, [bw_cs, bw_hs, fw_hs, input_len-1], \
                                             parallel_iterations=1)
        self.bw_cs = bw_cs.concat()
        self.bw_hs = bw_hs.concat()

    def build_cost(self, fw_bw_hs, label_ts):
        with tf.variable_scope('Projection', reuse=True):
            W = tf.get_variable('W')
            b = tf.get_variable('b')
        logits = tf.matmul(fw_bw_hs, W) + b
        self.pred = tf.squeeze(tf.argmax(logits, 1))
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logits, labels=label_ts)
        self.mean_loss = tf.reduce_mean(self.loss)
        self.loss_summary = tf.summary.scalar('Loss', self.mean_loss)
        if self.is_training:
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.mean_loss)
