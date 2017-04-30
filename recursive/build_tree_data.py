import nltk
import cPickle
from nltk.tree import Tree
from nltk.draw.tree import TreeView

tree_input_path = '_data/tree.txt'
data_path = '_data/rec_data.pkl'

tree_output_path = '_data/compact_tree.txt'

def clean_tree(tree, data, i, j):
    if tree.label()[0] == "@":
        tree.set_label(tree.label()[1:])
    for (idx, st) in enumerate(tree):
        if isinstance(st, Tree):
            (i, j, n_tree) = clean_tree(st, data, i, j)
            tree[idx] = n_tree
        else:
            prefix = '_'.join(map(str, [data['tok_title'][i], data['tok_location'][i], data['tok_invitee'][i], data['tok_day'][i], data['tok_whenst'][i], data['tok_whened'][i]]))
            tree[idx] = prefix + '_' + tree[idx] + '+' + tree.label() + '_' + str(j)
            i = i + 1
            j = j + 1
    if len(tree) == 1:
        tree = tree[0]
        return (i, j, tree)
    else:
        left_tree = tree[0]
        right_tree = tree[1]
        if isinstance(left_tree, Tree):
            lst1 = left_tree.label().split('_')
        else:
            lst1 = left_tree.split('_')
        if isinstance(right_tree, Tree):
            lst2 = right_tree.label().split('_')
        else:
            lst2 = right_tree.split('_')
        lst = '_'.join([str(int(a) * int(b)) for (a,b) in zip(lst1[:-2], lst2[:-2])])
        tree.set_label(lst + '_' + tree.label() + '_' + str(j))
        j = j + 1
        return (i, j, tree)

def tree2txt(tree, res):
    if not isinstance(tree, Tree):
        tok_lst = tree.split('_')
        res['left_id'].append(-1)
        res['right_id'].append(-1)
        res['is_leaf'].append(1)
    else:
        left_id = tree2txt(tree[0], res)
        right_id = tree2txt(tree[1], res)
        tok_lst = tree.label().split('_')
        res['left_id'].append(left_id)
        res['right_id'].append(right_id)
        res['is_leaf'].append(0)
    res['title_y'].append(int(tok_lst[0]))
    res['location_y'].append(int(tok_lst[1]))
    res['invitee_y'].append(int(tok_lst[2]))
    res['day_y'].append(int(tok_lst[3]))
    res['whenst_y'].append(int(tok_lst[4]))
    res['whened_y'].append(int(tok_lst[5]))
    res['tok_text'].append(tok_lst[6])
    res['my_id'].append(int(tok_lst[7]))
    return res['my_id'][-1]
    
all_data = cPickle.load(open(data_path))
cnt = 0

all_res = []
with open(tree_input_path) as f, \
     open(tree_output_path, 'w') as f_out:
    for line in f:
        data = all_data[cnt]
        cnt = cnt + 1
        t = Tree.fromstring(line)
        #TreeView(t)._cframe.print_to_file('before.ps')
        _, _, t = clean_tree(t, data, 0, 0)
        t_str = ' '.join(str(t).split())
        f_out.write('%s\n' % t_str)
        '''
        TreeView(t)._cframe.print_to_file('_data/tree_example/%d.ps' % cnt)
        res = {'tok_text': [], \
               'left_id': [],
               'right_id': [],
               'is_leaf': [],
               'label': [],
               'title_y': [],
               'location_y': [],
               'invitee_y': [],
               'day_y': [],
               'whenst_y': [],
               'whened_y': [],
               'my_id': []
        }
        tree2txt(t, res)
        all_res.append(res)
        #f_out.write(str(res) + '\n')
    cPickle.dump(all_data, f_out)
        '''
