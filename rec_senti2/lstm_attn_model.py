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
    
class LSTMAttnModel():
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
        """Creates a block that goes from (node_id, tree) to (logits, state) tuples."""
        unknown_idx = len(self.word_idx)
        lookup_word = lambda word: self.word_idx.get(word, unknown_idx)

        word2vec = (td.GetItem(1) >> 
                    td.InputTransform(lambda x: lookup_word(tokenize(x)[1][0])) >>
                    td.Scalar('int32') >> self.word_embedding)

        pair2vec = td.Composition()
        with pair2vec.scope():
            node_id = pair2vec.input[0]
            left_node_id = td.Function(lambda x: 2*x).reads(node_id)
            right_node_id = td.Function(lambda x: 2*x+1).reads(node_id)
            tree_s = pair2vec.input[1]
            tok_res = td.InputTransform(tokenize).reads(tree_s)
            left_tree_s = (td.GetItem(1) >> td.GetItem(0)).reads(tok_res)
            right_tree_s = (td.GetItem(1) >> td.GetItem(1)).reads(tok_res)
            left_ret = embed_subtree().reads(left_node_id, left_tree_s)
            right_ret = embed_subtree().reads(right_node_id, right_tree_s)
            pair2vec.output.reads(left_ret, right_ret)

        # Trees are binary, so the tree layer takes two states as its input_state.
        zero_state = td.Zeros((self.tree_lstm.state_size,) * 2)
        # Input is a word vector.
        zero_inp = td.Zeros(self.word_embedding.output_type.shape[0])

        word_case = td.AllOf(word2vec, zero_state)
        pair_case = td.AllOf(zero_inp, pair2vec)

        tree2vec = td.OneOf(td.GetItem(1) >> td.InputTransform(lambda x: len(tokenize(x)[1])), [(1, word_case), (2, pair_case)])

        res = td.AllOf(td.GetItem(0), # node_id
                  td.GetItem(1) >> td.InputTransform(lambda x: tokenize(x)[0]) >> td.Scalar('int32'),
                  tree2vec >> tree_lstm >> (output_layer, td.Identity())
                 )
        return res

    def add_metrics(self):
        """A block that adds metrics for loss and hits; output is the LSTM state."""
        c = td.Composition()
        with c.scope():
            # destructure the input; (id, label, (logits, state))
            node_id = c.input[0]
            label = c.input[1]

            logits = td.GetItem(0).reads(c.input[2])
            state = td.GetItem(1).reads(c.input[2])

            td.Metric('node_ids').reads(node_id)
            td.Metric('logits').reads(logits)
            td.Metric('labels').reads(label)

            # output the state, which will be read by our by parent's LSTM cell
            c.output.reads(state)
        return c
        
    def build_model(self):
        self.build_recursive()
        self.build_attn()
        self.build_loss()
        self.build_trainer()
        
    def build_recursive(self):
        self.tree_lstm = td.ScopedLayer(
          tf.contrib.rnn.DropoutWrapper(
              BinaryTreeLSTMCell(self.lstm_num_units, keep_prob=self.keep_prob_ph),
              input_keep_prob=self.keep_prob_ph, output_keep_prob=self.keep_prob_ph),
          name_or_scope='tree_lstm')
        self.output_layer = td.FC(self.num_classes, activation=None, name='output_layer')
        self.word_embedding = word_embedding = td.Embedding(
            *self.weight_matrix.shape, initializer=self.weight_matrix, name='word_embedding')        
        self.embed_subtree = embed_subtree = td.ForwardDeclaration(name='embed_subtree')
        
        def embed_tree(logits_and_state):
            """Creates a block that embeds (node_id, trees); output is tree LSTM state."""
            return logits_and_state >> add_metrics()
        model = (td.Scalar('int32'), td.Identity()) >> embed_tree(self.logits_and_state())
        embed_subtree.resolve_to(embed_tree(self.logits_and_state()))
        compiler = td.Compiler.create(model)
        
        self.node_ids = compiler.metric_tensors['node_ids']
        self.logits = compiler.metric_tensors['logits']
        self.labels = compiler.metric_tensors['labels']