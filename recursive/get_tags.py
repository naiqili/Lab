import nltk
from nltk.tree import Tree

tree_input_path = '_data/compact_tree.txt'
tag_output_path = '_data/tags.txt'

def dfs_tag(tree):
    res = set()
    if isinstance(tree, Tree):
        for st in tree:
            res.update(dfs_tag(st))
        lb = tree.label()
        tag = lb.split('_')[-2].split('+')[-1]
        res.add(tag)
        return res
    else:
        tag = tree.split('_')[-2].split('+')[-1]
        res.add(tag)
        return res

with open(tree_input_path) as f_in, \
     open(tag_output_path, 'w') as f_out:
    all_tags = set()
    for line in f_in:
        t = Tree.fromstring(line)
        n_tags = dfs_tag(t)
        all_tags.update(n_tags)
    f_out.write("%d\n" % (len(all_tags)+1))
    f_out.write("LEAF\n")
    for tag in all_tags:
        f_out.write("%s\n" % tag)
