from nltk.tree import Tree

class TFNode:
    def __init__(self, s, isLeaf, left_node=None, right_node=None):
        self.isLeaf = isLeaf
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
        cur_s = self._s
        if self.left_node != None and self.right_node != None:
            left_s = str(self.left_node)
            right_s = str(self.right_node)
            return "(%s:%d (%s) (%s))" % (cur_s, self.id, left_s, right_s)
        else:
            return "(%s:%d)" % (cur_s, self.id)

class TFTree:
    def __init__(self, nltk_t):
        self.root = self._dfs_build(nltk_t)
        self.add_id(self.root, 0)

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

    def add_id(self, node, i):
        if node == None:
            return i
        node.id = i
        i = i+1
        i = self.add_id(node.left_node, i)
        i = self.add_id(node.right_node, i)
        return i

    def __str__(self):
        return str(self.root)
