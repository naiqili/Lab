import tensorflow as tf
import cPickle

class LSTMModel():
    def __init__(self, config_dict):        
        self.__dict__.update(config_dict)
        self.emb_mat = cPickle.load(open(self.wv_emb_file))

    def add_variables(self):
        with tf.variable_scope('Embeddings'):
            wv_emb = tf.get_variable('wv_embed',
                                     [self.wv_vocab_size, self.embed_size], \
                                     trainable=False)
            pos_emb = tf.get_variable('unk_embed',
                [1, self.embed_size])
        with tf.variable_scope('Forward'):
            tf.get_variable('W_i_left',
                [self.fw_cell_size, self.fw_cell_size])
            tf.get_variable('W_i_right',
                [self.fw_cell_size, self.fw_cell_size])
            tf.get_variable('b_i',
                [1, self.fw_cell_size])

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

            tf.get_variable('W_o_left',
                [self.fw_cell_size, self.fw_cell_size])
            tf.get_variable('W_o_right',
                [self.fw_cell_size, self.fw_cell_size])
            tf.get_variable('b_o',
                [1, self.fw_cell_size])

            tf.get_variable('W_u_left',
                [self.fw_cell_size, self.fw_cell_size])
            tf.get_variable('W_u_right',
                [self.fw_cell_size, self.fw_cell_size])
            tf.get_variable('b_u',
                [1, self.fw_cell_size])

        with tf.variable_scope('Projection'):
            tf.get_variable('W',
                            [(self.fw_cell_size), self.class_size])
            tf.get_variable('b',
                            [1, self.class_size])

    def embed_word(self, word_index):
        with tf.variable_scope('Embeddings', reuse=True):
            wv_embed = tf.get_variable('wv_embed')
            unk_embed = tf.get_variable('unk_embed')
            embeddings = tf.concat([wv_embed, unk_embed], 0)
        with tf.device('/cpu:0'):
            return tf.expand_dims(tf.gather(embeddings, word_index), 0)

    def combine_children(self, left_c, left_h, right_c, right_h):
        with tf.variable_scope('Forward', reuse=True):
            W_i_left = tf.get_variable('W_i_left')
            W_i_right = tf.get_variable('W_i_right')
            b_i = tf.get_variable('b_i')

            W_f_left_left = tf.get_variable('W_f_left_left')
            W_f_left_right = tf.get_variable('W_f_left_right')
            W_f_right_left = tf.get_variable('W_f_right_left')
            W_f_right_right = tf.get_variable('W_f_right_right')
            b_f = tf.get_variable('b_f')

            W_o_left = tf.get_variable('W_o_left')
            W_o_right = tf.get_variable('W_o_right')
            b_o = tf.get_variable('b_o')

            W_u_left = tf.get_variable('W_u_left')
            W_u_right = tf.get_variable('W_u_right')
            b_u = tf.get_variable('b_u')
        _i = tf.sigmoid(tf.matmul(left_h, W_i_left) + \
                        tf.matmul(right_h, W_i_right) + \
                        b_i)
        _f_left = tf.sigmoid(tf.matmul(left_h, W_f_left_left) + \
                             tf.matmul(right_h, W_f_left_right) + \
                             b_f)
        _f_right = tf.sigmoid(tf.matmul(right_h, W_f_right_left) + \
                              tf.matmul(right_h, W_f_right_right) + \
                              b_f)
        _o = tf.sigmoid(tf.matmul(left_h, W_o_left) + \
                        tf.matmul(right_h, W_o_right) + \
                        b_o)
        _u = tf.tanh(tf.matmul(left_h, W_u_left) + \
                     tf.matmul(right_h, W_u_right) + \
                     b_u)
        _c = _i * _u + _f_left * left_c + _f_right * right_c
        _h = _o * tf.tanh(_c)
        return _c, _h

    def build_model(self, left_ts, right_ts, wv_ts, target_ts, is_leaf_ts, len_ts):
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
            left_id, right_id = (tf.gather(left_ts, i), tf.gather(right_ts, i))
            left_c, left_h, right_c, right_h = tf.cond(
                tf.equal(node_is_leaf, 1),
                lambda: (tf.zeros([1, self.fw_cell_size]), tf.zeros([1, self.fw_cell_size]), tf.zeros([1, self.fw_cell_size]), tf.zeros([1, self.fw_cell_size])),
                lambda: (fw_cs.read(left_id), fw_hs.read(left_id), fw_cs.read(right_id), fw_hs.read(right_id)))
            new_c, new_h = self.combine_children(left_c, left_h, right_c, right_h)
    
            fw_cs = fw_cs.write(i, new_c)
            fw_hs = fw_hs.write(i, new_h)
            i = tf.add(i, 1)
            return fw_cs, fw_hs, i

        forward_loop_cond = lambda fw_cs, fw_hs, i: \
                    tf.less(i, tf.squeeze(len_ts))

        fw_cs, fw_hs, _ = tf.while_loop(forward_loop_cond, forward_loop_body, [fw_cs, fw_hs, 0], parallel_iterations=1)
        self.fw_cs = fw_cs.concat()
        self.fw_hs = fw_hs.concat()
        self.root_fw_hs = fw_hs.read(tf.squeeze(len_ts)-1)
        return (fw_cs, fw_hs)

    def build_cost(self, fw_hs, label_ts):
        with tf.variable_scope('Projection', reuse=True):
            W = tf.get_variable('W')
            b = tf.get_variable('b')
        logits = tf.matmul(fw_hs, W) + b
        self.pred = tf.squeeze(tf.argmax(logits, 1))
        _loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logits, labels=label_ts)
        all_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Forward')
        self.loss = _loss + tf.reduce_sum([tf.nn.l2_loss(w) for w in all_weights]) * 0.5 * self.L2_lambda
        self.mean_loss = tf.reduce_mean(self.loss)
        self.loss_summary = tf.summary.scalar('Loss', self.mean_loss)
        if self.is_training:
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.mean_loss)
