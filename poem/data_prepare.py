# coding=utf-8

import random
import codecs
import cPickle

data = []
author = ''
title = ''
text = ''

path = 'data/full_poem.txt'

with open(path) as f:
    for l in f:        
        l = l.decode('utf-8')
        if l[0] != u'　':
            continue
        l = l.strip(u'　\r\n')
        if l.find(u'【') != -1:
            if len(data) % 500 == 0:
                print len(data)
            if text != '' and len(text) < 288:
                data.append([author, title, text])
            st = l.find(u'【')
            ed = l.find(u'】')
            title = l[st+1:ed]
            author = l[ed+1:]
            text = ''
            continue
        text = text + l
    data.append([author, title, text])

word2ind = {}
ind2word = {}
all_words = []
maxlen = 0

with codecs.open('tmp/data_poem.txt', 'w', 'utf-8') as f:
    for row in data:
        f.write(' '.join(row) + '\n')
        all_words = all_words + list(''.join(row))
        all_words = list(set(all_words))
        maxlen = max(maxlen, len(row[2]))

all_words = ["<END>", "<START>"] + all_words
for (ind, w) in enumerate(all_words):
    word2ind[w] = ind
    ind2word[ind] = w

cPickle.dump((word2ind, ind2word), codecs.open('tmp/dic.pkl', 'w'))

random.shuffle(data)
sp = int(len(data)*0.8)

print 'Train data size:', sp
print 'Dev data size:', len(data)-sp
print 'Dict size:', len(word2ind)
print 'Maxlen:', maxlen+2

with codecs.open('tmp/train_poem.txt', 'w', 'utf-8') as f, \
     open('tmp/train_coded.txt', 'w') as f_coded:
    for row in data[:sp]:
        f.write(' '.join(row) + '\n')
        text = ['<START>'] + list(row[2]) + ['<END>']
        f_coded.write(' '.join([str(word2ind[w]) for w in text]) + '\n')

with codecs.open('tmp/dev_poem.txt', 'w', 'utf-8') as f, \
     open('tmp/dev_coded.txt', 'w') as f_coded:
    for row in data[sp:]:
        f.write(' '.join(row) + '\n')
        text = ['<START>'] + list(row[2]) + ['<END>']
        f_coded.write(' '.join([str(word2ind[w]) for w in text]) + '\n')