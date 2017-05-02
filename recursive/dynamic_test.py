import tensorflow as tf

vocab = {'the': 0, 'old': 1, 'cat': 2}
node_words = ['the', 'old', 'cat', '', '']
is_leaf = [True, True, True, False, False]
left_children = [-1, -1, -1, 1, 0]   # indices of left children nodes in this list
right_children = [-1, -1, -1, 2, 3]  # indices of right children nodes in this list

embed_size=7
label_size=2

with tf.variable_scope('Embeddings'):
  embeddings = \
    tf.get_variable('embeddings', [len(vocab), embed_size])
with tf.variable_scope('Composition'):
  W1 = tf.get_variable('W1',
                  [2 * embed_size, embed_size])
  b1 = tf.get_variable('b1', [1, embed_size])
with tf.variable_scope('Decompose'):
  dW1 = tf.get_variable('W1', [embed_size, embed_size])
  db1 = tf.get_variable('b1', [1, embed_size])
  dW2 = tf.get_variable('W2', [embed_size, embed_size])
  db2 = tf.get_variable('b2', [1, embed_size])
with tf.variable_scope('Projection'):
  U = tf.get_variable('U', [embed_size, label_size])
  bs = tf.get_variable('bs', [1, label_size])

node_word_indices = [vocab[word] if word else -1 for word in node_words]

forward_tensors = tf.TensorArray(tf.float32, size=0, dynamic_size=True,
                              clear_after_read=False, infer_shape=False)

backward_tensors = tf.TensorArray(tf.float32, size=0, dynamic_size=True,
                              clear_after_read=False, infer_shape=False)
                              
def embed_word(word_index):
  with tf.device('/cpu:0'):
    return tf.expand_dims(tf.gather(embeddings, word_index), 0)

def combine_children(left_tensor, right_tensor):
  return tf.nn.relu(tf.matmul(tf.concat([left_tensor, right_tensor], 1), W1) + b1)
  
def decompose_children(parent_tensor):
  left_tensor = tf.nn.relu(tf.matmul(parent_tensor, dW1) + db1)
  right_tensor = tf.nn.relu(tf.matmul(parent_tensor, dW2) + db2)
  return left_tensor, right_tensor

def loop_body(node_tensors, i):
  node_is_leaf = tf.gather(is_leaf, i)
  node_word_index = tf.gather(node_word_indices, i)
  left_child = tf.gather(left_children, i)
  right_child = tf.gather(right_children, i)
  node_tensor = tf.cond(
      node_is_leaf,
      lambda: embed_word(node_word_index),
      lambda: combine_children(node_tensors.read(left_child),
                               node_tensors.read(right_child)))
  node_tensors = node_tensors.write(i, node_tensor)
  i = tf.add(i, 1)
  return node_tensors, i
  
def backward_loop_body(node_tensors, i, input_len, last_forward):
  def fn1(node_tensors):
    parent_tensor = tf.cond(tf.equal(i, input_len),
                          lambda: last_forward,
                          lambda: node_tensors.read(i))
    left_child = tf.gather(left_children, i)
    right_child = tf.gather(right_children, i)
    left_tensor, right_tensor = decompose_children(parent_tensor)
    node_tensors = node_tensors.write(left_child, left_tensor)
    node_tensors = node_tensors.write(right_child, right_tensor)
    return node_tensors

  node_tensors = tf.cond(tf.equal(tf.gather(is_leaf, i), False),
          lambda: fn1(node_tensors), \
          lambda: node_tensors)
  i = i - 1
  return node_tensors, i, input_len, last_forward

input_len = tf.squeeze(tf.shape(is_leaf))
loop_cond = lambda node_tensors, i: \
        tf.less(i, input_len)
backward_loop_cond  = lambda node_tensors, i, len_1, last_forward: \
        tf.greater_equal(i, 0)

forward_tensors, _ = tf.while_loop(loop_cond, loop_body, [forward_tensors, 0],
                                     parallel_iterations=1)
                                     
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
                       
print sess.run(forward_tensors.concat())
              
last_forward = forward_tensors.read(input_len-1)
print sess.run(last_forward)
backward_tensors, _, _, _ = tf.while_loop(backward_loop_cond, backward_loop_body, [backward_tensors, input_len-1, input_len-1, last_forward],
                                     parallel_iterations=1)
                                     

print sess.run(backward_tensors.concat())
