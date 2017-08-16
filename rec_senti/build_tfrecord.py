from tftree import TFTree, TFNode
from nltk.tree import Tree
import cPickle, codecs
import tensorflow as tf

bin_flag = False

dict_path = "_data/dict.pkl"

train_tree_path = "../assignment3/trees/train.txt"
valid_tree_path = "../assignment3/trees/dev.txt"
test_tree_path = "../assignment3/trees/test.txt"

if bin_flag:
    train_record_path = "_data/binary_train.record"
    valid_record_path = "_data/binary_valid.record"
    test_record_path = "_data/binary_test.record"
else:
    train_record_path = "_data/finegrained_train.record"
    valid_record_path = "_data/finegrained_valid.record"
    test_record_path = "_data/finegrained_test.record"

(wv_word2ind, wv_ind2word) = cPickle.load(open(dict_path))

def build_record(input_path, output_path):
    writer = tf.python_io.TFRecordWriter(output_path)
    cnt = 0
    with codecs.open(input_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip().replace('\\', '')
            nltk_t = Tree.fromstring(line)
            tf_t = TFTree(nltk_t)
            tf_t.to_code_tree(wv_word2ind)
            tf_t.add_id()
            #print tf_t
            all_lst = tf_t.to_list(binary=bin_flag)
            if all_lst == None: # Neutral
                continue
            tf_t.build_mask()
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
                            int64_list=tf.train.Int64List(value=is_leaf_lst)),
                        'false_mask': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=tf_t.false_mask.reshape([-1]))),
                        'subtree_mask': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=tf_t.subtree_mask.reshape([-1])))
                        }
                    )
                )
            serialized = example.SerializeToString()
            writer.write(serialized)
            cnt += 1
    writer.close()
    return cnt

train_size = build_record(train_tree_path, train_record_path)
valid_size = build_record(valid_tree_path, valid_record_path)
test_size = build_record(test_tree_path, test_record_path)

print "Train size:", train_size
print "Valid size:", valid_size
print "Test size:", test_size
