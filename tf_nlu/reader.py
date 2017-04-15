import cPickle
import numpy as np
import tensorflow as tf

def get_raw_data(datafile, dictfile, train_size=5000, valid_size=1000, test_size=1000, num_steps=60):
    all_data = cPickle.load(open(datafile))
    train_data = all_data[:train_size]
    valid_data = all_data[train_size:(train_size+valid_size)]
    test_data = all_data[(train_size+valid_size):(train_size+valid_size+test_size)]
    (word2ind, ind2word) = cPickle.load(open(dictfile))

    train_text = np.zeros([train_size, num_steps], dtype=np.int32)
    train_text_len = np.zeros(train_size)
    train_title = np.zeros([train_size, num_steps], dtype=np.int32)
    train_location = np.zeros([train_size, num_steps], dtype=np.int32)
    train_day = np.zeros([train_size, num_steps], dtype=np.int32)
    train_whenst = np.zeros([train_size, num_steps], dtype=np.int32)
    train_whened = np.zeros([train_size, num_steps], dtype=np.int32)
    train_invitee = np.zeros([train_size, num_steps], dtype=np.int32)

    valid_text = np.zeros([valid_size, num_steps], dtype=np.int32)
    valid_text_len = np.zeros(valid_size)
    valid_title = np.zeros([valid_size, num_steps], dtype=np.int32)
    valid_location = np.zeros([valid_size, num_steps], dtype=np.int32)
    valid_day = np.zeros([valid_size, num_steps], dtype=np.int32)
    valid_whenst = np.zeros([valid_size, num_steps], dtype=np.int32)
    valid_whened = np.zeros([valid_size, num_steps], dtype=np.int32)
    valid_invitee = np.zeros([valid_size, num_steps], dtype=np.int32)

    for i in range(train_size):
        text = train_data[i]['tok_text']
        text_len = len(text)
        train_text_len[i] = text_len
        train_text[i][:text_len] = [word2ind[w] for w in text]
        train_title[i][:text_len] = train_data[i]['tok_title']
        train_location[i][:text_len] = train_data[i]['tok_location']
        train_day[i][:text_len] = train_data[i]['tok_day']
        train_whenst[i][:text_len] = train_data[i]['tok_whenst']
        train_whened[i][:text_len] = train_data[i]['tok_whened']
        train_invitee[i][:text_len] = train_data[i]['tok_invitee']
    for i in range(valid_size):
        text = valid_data[i]['tok_text']
        text_len = len(text)
        valid_text_len[i] = text_len
        valid_text[i][:text_len] = [word2ind[w] for w in text]
        valid_title[i][:text_len] = valid_data[i]['tok_title']
        valid_location[i][:text_len] = valid_data[i]['tok_location']
        valid_day[i][:text_len] = valid_data[i]['tok_day']
        valid_whenst[i][:text_len] = valid_data[i]['tok_whenst']
        valid_whened[i][:text_len] = valid_data[i]['tok_whened']
        valid_invitee[i][:text_len] = valid_data[i]['tok_invitee']

    return {'train_text': train_text,
            'train_text_len': train_text_len,
            'train_title': train_title,
            'train_location': train_location,
            'train_day': train_day,
            'train_whenst': train_whenst,
            'train_whened': train_whened,
            'train_invitee': train_invitee,
            'valid_text': valid_text,
            'valid_text_len': valid_text_len,
            'valid_title': valid_title,
            'valid_location': valid_location,
            'valid_day': valid_day,
            'valid_whenst': valid_whenst,
            'valid_whened': valid_whened,
            'valid_invitee': valid_invitee,
            'train_size': train_size,
            'valid_size': valid_size,
            'test_size': test_size
        }

def get_producer(data_x, data_y, data_len, batch_size, num_steps=60):
    with tf.name_scope("Producer"):
        raw_x = tf.convert_to_tensor(data_x, name="raw_x", dtype=tf.int32)
        raw_y = tf.convert_to_tensor(data_y, name="raw_y", dtype=tf.int32)
        raw_len = tf.convert_to_tensor(data_len, name="raw_len", dtype=tf.int32)

        slice_x, slice_y, slice_len = tf.train.slice_input_producer(
            [raw_x, raw_y, raw_len],
            shuffle=True)
            #num_epochs=num_epochs)
        batch_x, batch_y, batch_len = tf.train.batch(
            [slice_x, slice_y, slice_len],
            batch_size=batch_size)
        mask = tf.to_float(tf.not_equal(batch_x, 0))
        return batch_x, batch_y, batch_len, mask

if __name__ == '__main__':
    data = get_raw_data('./_data/7000/nlu_data.pkl', './tmp/dict7000.pkl')
    batch_x, batch_y, batch_len, mask = get_producer(data['train_text'], data['train_title'], data['train_text_len'], 5)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(3):
            print 'batch %d' % i
            _x, _y, _len, _mask = sess.run([batch_x, batch_y, batch_len, mask])
            print _x
            print _y
            print _len
            print _mask

        coord.request_stop()
        coord.join(threads)
        sess.close()
