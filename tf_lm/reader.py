import cPickle
import numpy as np
import tensorflow as tf

def get_raw_data(num_steps=700):
    train_data = cPickle.load(open('./tmp/traindata.pkl'))[:2000]
    dev_data = cPickle.load(open('./tmp/devdata.pkl'))
    test_data = cPickle.load(open('./tmp/testdata.pkl'))
    (word2ind, ind2word) = cPickle.load(open('./tmp/dict.pkl'))
    train_data = [sent1 + sent2 for (sent1, sent2) in train_data]
    dev_data = [sent1 + sent2 for (sent1, sent2) in dev_data]
    test_data = [sent1 + sent2 for (sent1, sent2) in test_data]
    train_len = len(train_data)
    dev_len = len(dev_data)
    test_len = len(test_data)

    train_mat = np.zeros([train_len, num_steps], dtype=np.int32)
    dev_mat = np.zeros([dev_len, num_steps], dtype=np.int32)
    test_mat = np.zeros([test_len, num_steps], dtype=np.int32)

    for i in range(train_len):
        train_mat[i, 0:len(train_data[i])] = train_data[i][0:len(train_data[i])]
    for i in range(dev_len):
        dev_mat[i, 0:len(dev_data[i])] = dev_data[i][0:len(dev_data[i])]
    for i in range(test_len):
        test_mat[i, 0:len(test_data[i])] = test_data[i][0:len(test_data[i])]

    return {'train_data': train_mat,
            'dev_data': dev_mat,
            'test_data': test_mat,
            'word2ind': word2ind,
            'ind2word': ind2word}

def get_producer(raw_data, batch_size, num_steps=700-1):
    with tf.name_scope("Producer"):
        data_len = len(raw_data)
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
        raw_x = tf.slice(raw_data, [0,0], [-1,num_steps])
        raw_y = tf.slice(raw_data, [0,1], [-1,num_steps])
        slice_x, slice_y = tf.train.slice_input_producer(
            [raw_x, raw_y],
            shuffle=True)
        batch_x, batch_y = tf.train.batch(
            [slice_x, slice_y],
            batch_size=batch_size)
        mask = tf.to_float(tf.not_equal(batch_y, 0))
        return batch_x, batch_y, mask

if __name__ == '__main__':
    data = get_raw_data()
    train_data = data['train_data']
    batch_x, batch_y, mask = get_producer(train_data, 5)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(3):
            print 'batch %d' % i
            print sess.run([batch_x, batch_y, mask])

        coord.request_stop()
        coord.join(threads)
        sess.close()
