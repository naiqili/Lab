import tensorflow as tf
import cPickle

class BasicModel():
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
            tf.get_variable('CellProj',
                [self.embed_size, self.cell_size])
        with tf.variable_scope('Composition'):
            tf.get_variable('W_left',
                [self.cell_size, self.cell_size])
            tf.get_variable('W_right',
                [self.cell_size, self.cell_size])
            tf.get_variable('W_mid',
                [self.cell_size, self.cell_size])
            tf.get_variable('b',
                [1, self.cell_size])
        with tf.variable_scope('Projection'):
            tf.get_variable('W',
                            [self.cell_size, 2])
            tf.get_variable('b',
                            [1, 2])

    def embed_word(self, word_index):
        with tf.variable_scope('Embeddings', reuse=True):
            wv_embed = tf.get_variable('wv_embed')
            pos_embed = tf.get_variable('pos_embed')
            embeddings = tf.concat([wv_embed, pos_embed], 0)
            cell_proj = tf.get_variable('CellProj')
        with tf.device('/cpu:0'):
            return tf.matmul(tf.expand_dims(tf.gather(embeddings, word_index), 0), \
                             cell_proj)

    def combine_children(self, left_tensor, right_tensor, wv_ind):
        with tf.variable_scope('Composition', reuse=True):
            W_left = tf.get_variable('W_left')
            W_right = tf.get_variable('W_right')
            W_mid = tf.get_variable('W_mid')
            b = tf.get_variable('b')
        wv_emb = self.embed_word(wv_ind)
        return tf.nn.relu(tf.matmul(left_tensor, W_left) + \
                          tf.matmul(right_tensor, W_right) + \
                          tf.matmul(wv_emb, W_mid) + \
                          b)

    def build_model(self, left_ts, right_ts, wv_ts, target_ts, is_leaf_ts, len_ts):
        forward_tensors = self.build_forward(left_ts, right_ts, wv_ts, is_leaf_ts, len_ts)
        self.build_cost(forward_tensors, target_ts)
                    
    def build_forward(self, left_ts, right_ts, wv_ts, is_leaf_ts, len_ts):
        forward_tensors = tf.TensorArray(tf.float32, size=0, dynamic_size=True, \
            clear_after_read=False, infer_shape=False)
        
        def forward_loop_body(node_tensors, i):
            node_is_leaf = tf.gather(is_leaf_ts, i)
            node_word_index = tf.gather(wv_ts, i)
            left_child = tf.gather(left_ts, i)
            right_child = tf.gather(right_ts, i)
            node_tensor = tf.cond(
                tf.equal(node_is_leaf, 1),
                lambda: self.embed_word(node_word_index),
                lambda: self.combine_children(node_tensors.read(left_child),
                                         node_tensors.read(right_child),
                                         node_word_index))
            node_tensors = node_tensors.write(i, node_tensor)
            i = tf.add(i, 1)
            return node_tensors, i

        forward_loop_cond = lambda node_tensors, i: \
                    tf.less(i, tf.squeeze(len_ts))

        forward_tensors, _ = tf.while_loop(forward_loop_cond, forward_loop_body, [forward_tensors, 0], parallel_iterations=1)
        forward_tensors = forward_tensors.concat()
        self.forward_tensors = forward_tensors
        return forward_tensors

    def build_cost(self, forward_tensors, label_ts):
        with tf.variable_scope('Projection', reuse=True):
            W = tf.get_variable('W')
            b = tf.get_variable('b')
        logits = tf.matmul(forward_tensors, W) + b
        self.pred = tf.squeeze(tf.argmax(logits, 1))
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logits, labels=label_ts)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
