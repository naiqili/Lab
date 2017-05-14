from nltk.tree import Tree

class TFNode:
    def __init__(self, label, is_leaf=True, left_node=None, right_node=None):
        if is_leaf:
            self.is_leaf = True
            self.left_node = None
            self.right_node = None
            self.label = int(label)
        else:
            self.is_leaf = False
            self.label = int(label)
            self.left_node = left_node
            self.right_node = right_node
        self.word = None

    def __str__(self):
        cur_s = str(self.label)
        if self.left_node != None and self.right_node != None:
            left_s = str(self.left_node)
            right_s = str(self.right_node)
            return "(%s (%s) (%s))" % (cur_s, left_s, right_s)
        else:
            return "(%s %s)" % (cur_s, self.word)

class TFTree:
    def __init__(self, nltk_t):
        self.root = self._dfs_build(nltk_t)
        

    def _dfs_build(self, nltk_t):
        if isinstance(nltk_t, Tree) and len(nltk_t) >= 2:
            s = nltk_t.label()
            left_node = self._dfs_build(nltk_t[0])
            right_node = self._dfs_build(nltk_t[1])
            cur_node = TFNode(s, False, left_node, right_node)
        else:
            s = nltk_t.label()
            cur_node = TFNode(s, True)
            cur_node.word = nltk_t[0]
        return cur_node

    def add_id(self, node='', i=0):
        if node == '':
            node = self.root
        if node == None:
            return i        
        i = self.add_id(node.left_node, i)
        i = self.add_id(node.right_node, i)
        node.id = i
        i = i+1
        return i

    def to_code_tree(self, wv_word2ind, node=None):
        if node == None:
            node = self.root
        if node.is_leaf:
            word = node.word
            if word in wv_word2ind:
                node.word_ind = wv_word2ind[word]
            else:
                node.word_ind = len(wv_word2ind) + 1
        else:
            node.word_ind = -1
            self.to_code_tree(wv_word2ind, node.left_node)
            self.to_code_tree(wv_word2ind, node.right_node)

    def to_list(self, binary=True, node=None):
        if node == None and binary and self.root.label == 2:
            return None # Neutral
        if node == None:
            node = self.root
        if binary:
            if node.label < 2:
                label = 0
            elif node.label > 2:
                label = 1
            else:
                label = node.label
        else:
            label = node.label

        if node.is_leaf:
            return [node.id], [node.word_ind], [-1], [-1], \
                [label], [1]
        else:
            left_lsts = self.to_list(binary, node.left_node)
            right_lsts = self.to_list(binary, node.right_node)
            id_lst, wv_lst, left_lst, right_lst, label_lst, is_leaf_lst = map(lambda (x,y): x+y, zip(list(left_lsts), list(right_lsts)))
            id_lst.append(node.id)
            wv_lst.append(node.word_ind)
            left_lst.append(node.left_node.id)
            right_lst.append(node.right_node.id)
            label_lst.append(label)
            is_leaf_lst.append(0)
            return id_lst, wv_lst, left_lst, right_lst, label_lst, is_leaf_lst
                                            
    def __str__(self):
        return str(self.root)
