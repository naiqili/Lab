import tensorflow as tf

def get_data(filename='_data/tree.record', shuffle=False):
    # first construct a queue containing a list of filenames.
    # this lets a user split up there dataset in multiple files to keep
    # size down
    filename_queue = tf.train.string_input_producer([filename],
                                                    num_epochs=None)
    # Unlike the TFRecordWriter, the TFRecordReader is symbolic
    reader = tf.TFRecordReader()
    # One can read a single serialized example from a filename
    # serialized_example is a Tensor of type string.
    _, serialized_example = reader.read(filename_queue)
    # The serialized example is converted back to actual values.
    # One needs to describe the format of the objects to be returned
    features = tf.parse_single_example(
        serialized_example,
        features={
            # We know the length of both fields. If not the
            # tf.VarLenFeature could be used
            'len': tf.VarLenFeature(tf.int64),
            'wv': tf.VarLenFeature(tf.int64),
            'left': tf.VarLenFeature(tf.int64),
            'right': tf.VarLenFeature(tf.int64),
            'label': tf.VarLenFeature(tf.int64),
            'is_leaf': tf.VarLenFeature(tf.int64)
        })
    # now return the converted data
    if shuffle:
        _l, _wv, _left, _right, _label, _is_leaf = tf.train.shuffle_batch([features['len'], features['wv'], features['left'], features['right'], features['label'], features['is_leaf']], \
          batch_size=1, num_threads=4, capacity=5000, min_after_dequeue=1000)
        _l = tf.to_int32(tf.sparse_tensor_to_dense(_l))
        _wv = tf.to_int32(tf.sparse_tensor_to_dense(_wv))
        _left = tf.to_int32(tf.sparse_tensor_to_dense(_left))
        _right = tf.to_int32(tf.sparse_tensor_to_dense(_right))
        _label = tf.to_int32(tf.sparse_tensor_to_dense(_label))
        _is_leaf = tf.to_int32(tf.sparse_tensor_to_dense(_is_leaf))    
        return _l[0], _wv[0], _left[0], _right[0], _label[0], _is_leaf[0]
    else:
        _l = tf.to_int32(tf.sparse_tensor_to_dense(features['len']))
        _wv = tf.to_int32(tf.sparse_tensor_to_dense(features['wv']))
        _left = tf.to_int32(tf.sparse_tensor_to_dense(features['left']))
        _right = tf.to_int32(tf.sparse_tensor_to_dense(features['right']))
        _label = tf.to_int32(tf.sparse_tensor_to_dense(features['label']))
        _is_leaf = tf.to_int32(tf.sparse_tensor_to_dense(features['is_leaf']))
        return tf.train.limit_epochs([_l, _wv, _left, _right, _label, _is_leaf])

if __name__=='__main__':
    l, wv, left, right, label, is_leaf = get_data('_data/binary_train.record')
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for _step in range(5):
            #print _step
            _l, _wv, _left, _right, _label, _is_leaf = sess.run([l, wv, left, right, label, is_leaf])
            #print _left
            #print _is_leaf
            print _left
            print _label
            print _l
            print

        coord.request_stop()
        coord.join(threads)
