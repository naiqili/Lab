from tftree import TFTree, TFNode
from nltk.tree import Tree
import cPickle
import tensorflow as tf

train_size = 5000
valid_size = 1000

tree_input_path = '_data/compact_tree.txt'

train_path = 'tmp/train.record'
valid_path = 'tmp/valid.record'
test_path = 'tmp/test.record'

wv_dict_path = '../tf_nlu/tmp/dict.pkl'
pos_dict_path = 'tmp/pos_dict.pkl'

(wv_word2ind, wv_ind2word) = cPickle.load(open(wv_dict_path))
(pos_word2ind, pos_ind2word) = cPickle.load(open(pos_dict_path))

def get_tree_data(node):
    id_lst = []
    wv_lst = []
    left_lst = []
    right_lst = []
    is_leaf_lst = []

    title_lst = []
    location_lst = []
    invitee_lst = []
    day_lst = []
    whenst_lst = []
    whened_lst = []

    all_lst = [id_lst, wv_lst, left_lst, right_lst, title_lst, \
               location_lst, invitee_lst, day_lst, whenst_lst, \
               whened_lst, is_leaf_lst]

    if not node.is_leaf:
        left_all = get_tree_data(node.left_node)
        right_all = get_tree_data(node.right_node)
        for (lst, _lst) in zip(all_lst, left_all):
            lst.extend(_lst)
        for (lst, _lst) in zip(all_lst, right_all):
            lst.extend(_lst)
    id_lst.append(node.id)
    wv_lst.append(node.wv)
    if node.left_node != None:
        left_lst.append(node.left_node.id)
    else:
        left_lst.append(-1)
    if node.right_node != None:
        right_lst.append(node.right_node.id)
    else:
        right_lst.append(-1)
    title_lst.append(node.title_y)
    location_lst.append(node.location_y)
    invitee_lst.append(node.invitee_y)
    day_lst.append(node.day_y)
    whenst_lst.append(node.whenst_y)
    whened_lst.append(node.whened_y)
    is_leaf_lst.append(int(node.is_leaf))
    return all_lst

cnt = 0
with open(tree_input_path) as f:
    for line in f:
        if cnt == 0:
            writer = tf.python_io.TFRecordWriter(train_path)
        elif cnt == train_size:
            writer = tf.python_io.TFRecordWriter(valid_path)
        elif cnt == train_size + valid_size:
            writer = tf.python_io.TFRecordWriter(test_path)
        nltk_t = Tree.fromstring(line)
        tf_t = TFTree(nltk_t)
        #tf_t.set_label('title')
        tf_t.to_code_tree(wv_word2ind, pos_word2ind)
        tf_t.add_id()
        #print tf_t
        [id_lst, wv_lst, left_lst, right_lst, title_lst, \
         location_lst, invitee_lst, day_lst, whenst_lst, \
         whened_lst, is_leaf_lst] = \
            get_tree_data(tf_t.root)
        #print id_lst
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
                    'title': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=title_lst)),
                    'location': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=location_lst)),
                    'invitee': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=invitee_lst)),
                    'day': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=day_lst)),
                    'whenst': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=whenst_lst)),
                    'whened': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=whened_lst)),
                    'is_leaf': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=is_leaf_lst))
                    }
                )
            )
        serialized = example.SerializeToString()
        writer.write(serialized)
        cnt += 1
