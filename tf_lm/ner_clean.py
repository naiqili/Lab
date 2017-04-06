import nltk
from nltk.tree import Tree

def dfs_clean(t):
    for idx, child in enumerate(t):
        if isinstance(child, Tree):
            if child.label() in ['PERSON', 'ORGANIZATION', 'TIME', 'LOCATION']:
                t[idx] = ('<' + child.label() + '>', child.label())
            else:
                dfs_clean(child)
        elif child[1] == 'CD':
            t[idx] = ('<NUMBER>', 'CD')
    
def ner_clean(sent):
    tok_sent = nltk.word_tokenize(sent)
    tag_sent = nltk.pos_tag(tok_sent)
    chunk_sent = nltk.ne_chunk(tag_sent)
    print chunk_sent
    print
    dfs_clean(chunk_sent)
    print [leaf[0] for leaf in chunk_sent.leaves()]

if __name__ == '__main__':
    ner_clean('I am going to KFC with Xiaojie and Amy at 3 p.m.')
