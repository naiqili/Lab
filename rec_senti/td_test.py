import codecs
import functools
import os
import tempfile
import zipfile

from nltk.tokenize import sexpr
import numpy as np
from six.moves import urllib

os.environ['CUDA_VISIBLE_DEVICES']='1'
import tensorflow as tf
import tensorflow_fold as td
sess = tf.InteractiveSession()

data_dir = '../../data'
print('saving files to %s' % data_dir)

def download_and_unzip(url_base, zip_name, *file_names):
    zip_path = os.path.join(data_dir, zip_name)
    url = url_base + zip_name
    out_paths = []
    if not os.path.exists(zip_path):
        print('downloading %s to %s' % (url, zip_path))
        urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as f:
        for file_name in file_names:
            print('extracting %s' % file_name)
            if not os.path.exists(os.path.join(zip_path, file_name)):
                out_paths.append(f.extract(file_name, path=data_dir))      
            else:
                out_paths.append(file_name)
    return out_paths

full_glove_path, = download_and_unzip('http://nlp.stanford.edu/data/', 'glove.840B.300d.zip', 'glove.840B.300d.txt')
filtered_glove_path = os.path.join(data_dir, 'filtered_glove.txt')

train_path, dev_path, test_path = download_and_unzip(
  'http://nlp.stanford.edu/sentiment/', 'trainDevTestTrees_PTB.zip', 
  'trees/train.txt', 'trees/dev.txt', 'trees/test.txt')

def filter_glove():
    vocab = set()
    # Download the full set of unlabeled sentences separated by '|'.
    sentence_path, = download_and_unzip(
        'http://nlp.stanford.edu/~socherr/', 'stanfordSentimentTreebank.zip', 
        'stanfordSentimentTreebank/SOStr.txt')
    with codecs.open(sentence_path, encoding='utf-8') as f:
        for line in f:
            # Drop the trailing newline and strip backslashes. Split into words.
            vocab.update(line.strip().replace('\\', '').split('|'))
    print 'total word count:', len(vocab)
    nread = 0
    nwrote = 0
    with codecs.open(full_glove_path, encoding='utf-8') as f:
        with codecs.open(filtered_glove_path, 'w', encoding='utf-8') as out:
            for line in f:
                nread += 1
                line = line.strip()
                if not line: continue
                if line.split(u' ', 1)[0] in vocab:
                    out.write(line + '\n')
                    nwrote += 1
    print('read %s lines, wrote %s' % (nread, nwrote))
    print 'oov word count:', len(vocab)-nwrote

filter_glove()

def load_embeddings(embedding_path):
    """Loads embedings, returns weight matrix and dict from words to indices."""
    print('loading word embeddings from %s' % embedding_path)
    weight_vectors = []
    word_idx = {}
    with codecs.open(embedding_path, encoding='utf-8') as f:
        for line in f:
            word, vec = line.split(u' ', 1)
            word_idx[word] = len(weight_vectors)
            weight_vectors.append(np.array(vec.split(), dtype=np.float32))
    # Annoying implementation detail; '(' and ')' are replaced by '-LRB-' and
    # '-RRB-' respectively in the parse-trees.
    word_idx[u'-LRB-'] = word_idx.pop(u'(')
    word_idx[u'-RRB-'] = word_idx.pop(u')')
    # Random embedding vector for unknown words.
    weight_vectors.append(np.random.uniform(
        -0.05, 0.05, weight_vectors[0].shape).astype(np.float32))
    return np.stack(weight_vectors), word_idx

weight_matrix, word_idx = load_embeddings(filtered_glove_path)

print len(word_idx)
print word_idx.items()[:20]

def load_trees(filename):
    with codecs.open(filename, encoding='utf-8') as f:
        # Drop the trailing newline and strip \s.
        trees = [line.strip().replace('\\', '') for line in f]
        print('loaded %s trees from %s' % (len(trees), filename))
    return trees

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

keep_prob_ph = tf.placeholder_with_default(1.0, [])


lstm_num_units = 300  # Tai et al. used 150, but our regularization strategy is more effective
tree_lstm = td.ScopedLayer(
      tf.contrib.rnn.DropoutWrapper(
          BinaryTreeLSTMCell(lstm_num_units, keep_prob=keep_prob_ph),
          input_keep_prob=keep_prob_ph, output_keep_prob=keep_prob_ph),
      name_or_scope='tree_lstm')

NUM_CLASSES = 5  # number of distinct sentiment labels
output_layer = td.FC(NUM_CLASSES, activation=None, name='output_layer')

word_embedding = td.Embedding(
    *weight_matrix.shape, initializer=weight_matrix, name='word_embedding')

def load_trees(filename):
    with codecs.open(filename, encoding='utf-8') as f:
        # Drop the trailing newline and strip \s.
        trees = [line.strip().replace('\\', '') for line in f]
        print('loaded %s trees from %s' % (len(trees), filename))
        return trees

train_trees = load_trees(train_path)
dev_trees = load_trees(dev_path)
test_trees = load_trees(test_path)

embed_subtree = td.ForwardDeclaration(name='embed_subtree')

def logits_and_state():
    """Creates a block that goes from tokens to (logits, state) tuples."""
    unknown_idx = len(word_idx)
    lookup_word = lambda word: word_idx.get(word, unknown_idx)
  
    word2vec = (td.GetItem(0) >> td.InputTransform(lookup_word) >>
                td.Scalar('int32') >> word_embedding)

    pair2vec = (embed_subtree(), embed_subtree())

    # Trees are binary, so the tree layer takes two states as its input_state.
    zero_state = td.Zeros((tree_lstm.state_size,) * 2)
    # Input is a word vector.
    zero_inp = td.Zeros(word_embedding.output_type.shape[0])

    word_case = td.AllOf(word2vec, zero_state)
    pair_case = td.AllOf(zero_inp, pair2vec)

    tree2vec = td.OneOf(len, [(1, word_case), (2, pair_case)])

    return tree2vec >> tree_lstm >> (output_layer, td.Identity())


def tf_node_loss(logits, labels):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

def tf_fine_grained_hits(logits, labels):
    predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
    return tf.cast(tf.equal(predictions, labels), tf.float64)


def tf_binary_hits(logits, labels):
    softmax = tf.nn.softmax(logits)
    binary_predictions = (softmax[:, 3] + softmax[:, 4]) > (softmax[:, 0] + softmax[:, 1])
    binary_labels = labels > 2
    return tf.cast(tf.equal(binary_predictions, binary_labels), tf.float64)

def add_metrics(is_root, is_neutral):
    """A block that adds metrics for loss and hits; output is the LSTM state."""
    c = td.Composition(
        name='predict(is_root=%s, is_neutral=%s)' % (is_root, is_neutral))
    with c.scope():
        # destructure the input; (labels, (logits, state))
        labels = c.input[0]
        logits = td.GetItem(0).reads(c.input[1])
        state = td.GetItem(1).reads(c.input[1])

        # calculate loss
        loss = td.Function(tf_node_loss)
        td.Metric('all_loss').reads(loss.reads(logits, labels))
        if is_root: td.Metric('root_loss').reads(loss)

        # calculate fine-grained hits
        hits = td.Function(tf_fine_grained_hits)
        td.Metric('all_hits').reads(hits.reads(logits, labels))
        if is_root: td.Metric('root_hits').reads(hits)

        # calculate binary hits, if the label is not neutral
        if not is_neutral:
            binary_hits = td.Function(tf_binary_hits).reads(logits, labels)
            td.Metric('all_binary_hits').reads(binary_hits)
            if is_root: td.Metric('root_binary_hits').reads(binary_hits)

        # output the state, which will be read by our by parent's LSTM cell
        c.output.reads(state)
    return c

def tokenize(s):
    label, phrase = s[1:-1].split(None, 1)
    return label, sexpr.sexpr_tokenize(phrase)

tokenize('(X Y Z)')

def embed_tree(logits_and_state, is_root):
    """Creates a block that embeds trees; output is tree LSTM state."""
    return td.InputTransform(tokenize) >> td.OneOf(
          key_fn=lambda pair: pair[0] == '2',  # label 2 means neutral
          case_blocks=(add_metrics(is_root, is_neutral=False),
                       add_metrics(is_root, is_neutral=True)),
          pre_block=(td.Scalar('int32'), logits_and_state))

model = embed_tree(logits_and_state(), is_root=True)

embed_subtree.resolve_to(embed_tree(logits_and_state(), is_root=False))


compiler = td.Compiler.create(model)
print('input type: %s' % model.input_type)
print('output type: %s' % model.output_type)

metrics = {k: tf.reduce_mean(v) for k, v in compiler.metric_tensors.items()}


LEARNING_RATE = 0.05
KEEP_PROB = 0.75
BATCH_SIZE = 1
EPOCHS = 20
EMBEDDING_LEARNING_RATE_FACTOR = 0.1

train_feed_dict = {keep_prob_ph: KEEP_PROB}
loss = tf.reduce_sum(compiler.metric_tensors['all_loss'])
opt = tf.train.AdagradOptimizer(LEARNING_RATE)

grads_and_vars = opt.compute_gradients(loss)
found = 0
for i, (grad, var) in enumerate(grads_and_vars):
    if var == word_embedding.weights:
        found += 1
        grad = tf.scalar_mul(EMBEDDING_LEARNING_RATE_FACTOR, grad)
        grads_and_vars[i] = (grad, var)
assert found == 1  # internal consistency check
train = opt.apply_gradients(grads_and_vars)
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

def train_step(batch):
    train_feed_dict[compiler.loom_input_tensor] = batch
    _, batch_loss = sess.run([train, loss], train_feed_dict)
    return batch_loss

def train_epoch(train_set):
    return sum(train_step(batch) for batch in td.group_by_batches(train_set, BATCH_SIZE))


train_set = compiler.build_loom_inputs(train_trees)
dev_feed_dict = compiler.build_feed_dict(dev_trees)

def dev_eval(epoch, train_loss):
    dev_metrics = sess.run(metrics, dev_feed_dict)
    dev_loss = dev_metrics['all_loss']
    dev_accuracy = ['%s: %.2f' % (k, v * 100) for k, v in \
                  sorted(dev_metrics.items()) if k.endswith('hits')]
    print('epoch:%4d, train_loss: %.3e, dev_loss_avg: %.3e, dev_accuracy:\n  [%s]' \
        % (epoch, train_loss, dev_loss, ' '.join(dev_accuracy)))
    return dev_metrics['root_hits']

best_accuracy = 0.0
save_path = os.path.join(data_dir, 'sentiment_model')
for epoch, shuffled in enumerate(td.epochs(train_set, EPOCHS), 1):
    train_loss = train_epoch(shuffled)
    accuracy = dev_eval(epoch, train_loss)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        checkpoint_path = saver.save(sess, save_path, global_step=epoch)
        print('model saved in file: %s' % checkpoint_path)














