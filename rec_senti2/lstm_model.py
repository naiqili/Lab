import tensorflow as tf
import tensorflow_fold as td
import cPickle
from utils import *
import config
import numpy as np

class BinaryTreeLSTMCell(tf.contrib.rnn.BasicLSTMCell):
    def __init__(self, num_units, keep_prob=1.0):
        super(BinaryTreeLSTMCell, self).__init__(num_units)
        self._keep_prob = keep_prob

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            lhs, rhs = state
            c0, h0 = lhs
            c1, h1 = rhs
            concat = tf.contrib.layers.linear(
                tf.concat([inputs, h0, h1], 1), 5 * self._num_units)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f0, f1, o = tf.split(value=concat, num_or_size_splits=5, axis=1)

        j = self._activation(j)
        if not isinstance(self._keep_prob, float) or self._keep_prob < 1:
            j = tf.nn.dropout(j, self._keep_prob)

        new_c = (c0 * tf.sigmoid(f0 + self._forget_bias) + \
                 c1 * tf.sigmoid(f1 + self._forget_bias) + \
                 tf.sigmoid(i) * j)
        new_h = self._activation(new_c) * tf.sigmoid(o)

        new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

        return new_h, new_state
    
class LSTMModel():
    def __init__(self, config_dict):        
        '''Required fileds:
          wv_embed_path
          word_idx_path
          
          lstm_num_units
          num_classes
          
          lr
          embed_lr_factor
          '''
        self.__dict__.update(config_dict)
        self.weight_matrix = cPickle.load(open(self.wv_embed_path))
        self.word_idx = cPickle.load(open(self.word_idx_path))
        self.keep_prob_ph = tf.placeholder_with_default(1.0, [])
        
    def logits_and_state(self):
        """Creates a block that goes from tokens to (logits, state) tuples."""
        unknown_idx = len(self.word_idx)
        lookup_word = lambda word: self.word_idx.get(word, unknown_idx)

        word2vec = (td.GetItem(0) >> td.InputTransform(lookup_word) >>
                    td.Scalar('int32') >> self.word_embedding)

        pair2vec = (self.embed_subtree(), self.embed_subtree())

        # Trees are binary, so the tree layer takes two states as its input_state.
        zero_state = td.Zeros((self.tree_lstm.state_size,) * 2)
        # Input is a word vector.
        zero_inp = td.Zeros(self.word_embedding.output_type.shape[0])

        word_case = td.AllOf(word2vec, zero_state)
        pair_case = td.AllOf(zero_inp, pair2vec)

        tree2vec = td.OneOf(len, [(1, word_case), (2, pair_case)])

        return tree2vec >> self.tree_lstm >> (self.output_layer, td.Identity())


    def tf_node_loss(self, logits, labels):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

    def tf_fine_grained_hits(self, logits, labels):
        predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
        return tf.cast(tf.equal(predictions, labels), tf.float64)


    def tf_binary_hits(self, logits, labels):
        softmax = tf.nn.softmax(logits)
        binary_predictions = (softmax[:, 3] + softmax[:, 4]) > (softmax[:, 0] + softmax[:, 1])
        binary_labels = labels > 2
        return tf.cast(tf.equal(binary_predictions, binary_labels), tf.float64)

    def add_metrics(self, is_root, is_neutral):
        """A block that adds metrics for loss and hits; output is the LSTM state."""
        c = td.Composition(
            name='predict(is_root=%s, is_neutral=%s)' % (is_root, is_neutral))
        with c.scope():
            # destructure the input; (labels, (logits, state))
            labels = c.input[0]
            logits = td.GetItem(0).reads(c.input[1])
            state = td.GetItem(1).reads(c.input[1])

            # calculate loss
            loss = td.Function(self.tf_node_loss)
            td.Metric('all_loss').reads(loss.reads(logits, labels))
            if is_root: td.Metric('root_loss').reads(loss)

            # calculate fine-grained hits
            hits = td.Function(self.tf_fine_grained_hits)
            td.Metric('all_hits').reads(hits.reads(logits, labels))
            if is_root: td.Metric('root_hits').reads(hits)

            # calculate binary hits, if the label is not neutral
            if not is_neutral:
                binary_hits = td.Function(self.tf_binary_hits).reads(logits, labels)
                td.Metric('all_binary_hits').reads(binary_hits)
                if is_root: td.Metric('root_binary_hits').reads(binary_hits)

            # output the state, which will be read by our by parent's LSTM cell
            c.output.reads(state)
        return c
        
    def build_model(self):
        self.tree_lstm = td.ScopedLayer(
          tf.contrib.rnn.DropoutWrapper(
              BinaryTreeLSTMCell(self.lstm_num_units, keep_prob=self.keep_prob_ph),
              input_keep_prob=self.keep_prob_ph, output_keep_prob=self.keep_prob_ph),
          name_or_scope='tree_lstm')
        self.output_layer = td.FC(self.num_classes, activation=None, name='output_layer')
        self.word_embedding = word_embedding = td.Embedding(
            *self.weight_matrix.shape, initializer=self.weight_matrix, name='word_embedding')        
        self.embed_subtree = embed_subtree = td.ForwardDeclaration(name='embed_subtree')
        
        def embed_tree(logits_and_state, is_root):
            """Creates a block that embeds trees; output is tree LSTM state."""
            return td.InputTransform(tokenize) >> td.OneOf(
                  key_fn=lambda pair: pair[0] == '2',  # label 2 means neutral
                  case_blocks=(self.add_metrics(is_root, is_neutral=False),
                            self.add_metrics(is_root, is_neutral=True)),
                  pre_block=(td.Scalar('int32'), logits_and_state))
        model = embed_tree(self.logits_and_state(), is_root=True)
        embed_subtree.resolve_to(embed_tree(self.logits_and_state(), is_root=False))
        compiler = td.Compiler.create(model)
        metrics = {k: tf.reduce_mean(v) for k, v in compiler.metric_tensors.items()}
        loss = tf.reduce_sum(compiler.metric_tensors['all_loss'])
        
        opt = tf.train.AdagradOptimizer(self.lr)
        
        grads_and_vars = opt.compute_gradients(loss)
        found = 0
        for i, (grad, var) in enumerate(grads_and_vars):
            if var == word_embedding.weights:
                found += 1
                grad = tf.scalar_mul(self.embed_lr_factor, grad)
                grads_and_vars[i] = (grad, var)
        assert found == 1  # internal consistency check
        train_op = opt.apply_gradients(grads_and_vars)

        self.compiler = compiler
        self.metrics = metrics
        self.loss = loss
        self.train_op = train_op
        
        