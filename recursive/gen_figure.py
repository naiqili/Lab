from nltk.tree import Tree
from nltk.draw.tree import TreeView

cnt = 1
with open('_data/tree.txt') as f:
    for line in f:
        t = Tree.fromstring(line)
        TreeView(t)._cframe.print_to_file('tmp/figs/%d.ps' % cnt)
        cnt = cnt + 1
