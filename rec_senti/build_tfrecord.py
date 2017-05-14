from tftree import TFTree, TFNode
from nltk.tree import Tree
import cPickle
import tensorflow as tf

dict_path = "_data/dict.pkl"

train_tree_path = "../assignment3/trees/train.txt"
valid_tree_path = "../assignment3/trees/dev.txt"
test_tree_path = "../assignment3/trees/test.txt"

train_record_path = "_data/finegrained_train.record"
valid_record_path = "_data/finegrained_valid.record"
test_record_path = "_data/finegrained_test.record"

(wv_word2ind, wv_ind2word) = cPickle.load(open(dict_path))

def build_record(input_path, output_path):
    f = open(input_path)
    writer = tf.python_io.TFRecordWriter(output_path)
    cnt = 0
    for line in f:
        nltk_t = Tree.fromstring(line)
        tf_t = TFTree(nltk_t)
        tf_t.to_code_tree(wv_word2ind)
        tf_t.add_id()
        #print tf_t
        all_lst = tf_t.to_list(binary=False)
        if all_lst == None: # Neutral
            continue
        id_lst, wv_lst, left_lst, right_lst, label_lst, is_leaf_lst = all_lst

        example = tf.train.Example(
            features = tf.train.Features(
                feature = {
                    'len': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[len(id_lst)])),
                    'wv': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=wv_lst)),
                    'left': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=left_lst)),
                    'right': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=right_lst)),
                    'label': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=label_lst)),
                    'is_leaf': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=is_leaf_lst))
                    }
                )
            )
        serialized = example.SerializeToString()
        writer.write(serialized)
        cnt += 1
    f.close()
    writer.close()
    return cnt

train_size = build_record(train_tree_path, train_record_path)
valid_size = build_record(valid_tree_path, valid_record_path)
test_size = build_record(test_tree_path, test_record_path)

print "Train size:", train_size
print "Valid size:", valid_size
print "Test size:", test_size
