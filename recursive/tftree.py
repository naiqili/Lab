from nltk.tree import Tree

class TFNode:
    def __init__(self, ind):
        self.is_leaf = True
        self.left_node = None
        self.right_node = None
        self.wv = ind
        self._s = str(ind)
        
    def __init__(self, s='', isLeaf=True, left_node=None, right_node=None, ind=None):
        if ind != None:
            self.is_leaf = True
            self.left_node = None
            self.right_node = None
            self.wv = ind
            self._s = str(ind)
        else:
            self.is_leaf = isLeaf
            lst = s.split('_')
            self.title_y = int(lst[0])
            self.location_y = int(lst[1])
            self.invitee_y = int(lst[2])
            self.day_y = int(lst[3])
            self.whenst_y = int(lst[4])
            self.whened_y = int(lst[5])
            self.words = lst[6]
            self.left_node = left_node
            self.right_node = right_node
            self._s = s

    def set_label(self, target):
        self.label_y = self.__dict__[target+'_y']
        if self.left_node != None and self.right_node != None:
            self.left_node.set_label(target)
            self.right_node.set_label(target)

    def __str__(self):
        cur_s = self.words
        if self.left_node != None and self.right_node != None:
            left_s = str(self.left_node)
            right_s = str(self.right_node)
            return "(%s:%d (%s) (%s))" % (cur_s, self.wv, left_s, right_s)
        else:
            return "(%s:%d)" % (cur_s, self.wv)

class TFTree:
    def __init__(self, nltk_t):
        self.root = self._dfs_build(nltk_t)
        

    def _dfs_build(self, nltk_t):
        if isinstance(nltk_t, Tree):
            s = nltk_t.label()
            left_node = self._dfs_build(nltk_t[0])
            right_node = self._dfs_build(nltk_t[1])
            cur_node = TFNode(s, False, left_node, right_node)
        else:
            s = nltk_t
            cur_node = TFNode(s, True)
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

    def to_code_tree(self, wv_word2ind, pos_word2ind, node=None):
        if node == None:
            node = self.root
        if node.is_leaf:
            word, pos = node.words.split('+')
            if word in wv_word2ind:
                _word = wv_word2ind[word]
            else:
                _word = wv_word2ind['<unk>']
            _pos = pos_word2ind[pos]
            left_node = TFNode(ind=_word)
            right_node = TFNode(ind=_pos)
            node.wv = pos_word2ind['LEAF']
            node.is_leaf = False
            node.left_node = left_node
            node.right_node = right_node
            node.left_node.words = word
            node.right_node.words = pos
            node.right_node.title_y = node.left_node.title_y = node.title_y
            node.right_node.location_y = node.left_node.location_y = node.location_y
            node.right_node.invitee_y = node.left_node.invitee_y = node.invitee_y
            node.right_node.day_y = node.left_node.day_y = node.day_y
            node.right_node.whenst_y = node.left_node.whenst_y = node.whenst_y
            node.right_node.whened_y = node.left_node.whened_y = node.whened_y
        else:
            node.wv = pos_word2ind[node.words]
            self.to_code_tree(wv_word2ind, pos_word2ind, node.left_node)
            self.to_code_tree(wv_word2ind, pos_word2ind, node.right_node)

    def __str__(self):
        return str(self.root)
