from tftree import *
from nltk.tree import Tree

tree_input_path = '../assignment3/trees/train.txt'

with open(tree_input_path) as f:
    l = f.readline()
    nltk_t = Tree.fromstring(l)
    print nltk_t
    tf_t = TFTree(nltk_t)
    print str(tf_t)
