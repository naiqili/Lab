import cPickle

train_file = 'tmp/train.txt'
dev_file = 'tmp/dev.txt'
train_output = 'tmp/train_data.txt'
dev_output = 'tmp/dev_data.txt'
dic_output = 'tmp/dic.pkl'
log_output = 'tmp/data_prepare.log'

word2ind = {}
ind2word = {}

all_words = [] 

train_data = []
dev_data = []

train_data_cnt = 0
train_total_len = 0
train_max_len = 0

dev_data_cnt = 0
dev_total_len = 0
dev_max_len = 0

with open(train_file) as f:
    for line in f:
        words = line.split()
        train_data.append(['<START>'] + words + ['<END>'])
        all_words = all_words + words
        train_data_cnt = train_data_cnt + 1
        train_total_len = train_total_len + len(words)
        train_max_len = max(train_max_len, len(words))

with open(dev_file) as f:
    for line in f:
        words = line.split()
        dev_data.append(['<START>'] + words + ['<END>'])
        all_words = all_words + words
        dev_data_cnt = dev_data_cnt + 1
        dev_total_len = dev_total_len + len(words)
        dev_max_len = max(dev_max_len, len(words))
        
#print(train_data[0])

all_words = ['<END>', '<START>'] + list(set(all_words))
for (ind, w) in enumerate(all_words):
    word2ind[w] = ind
    ind2word[ind] = w
        
with open(log_output, 'w') as f:
    f.write("Training data info\n")
    f.write("Data count: %d\n" % train_data_cnt)
    f.write("Total len: %d\n" % train_total_len)
    f.write("Average len: %.4f\n" % (1.0 * train_total_len / train_data_cnt))
    f.write("Max len: %d\n" % train_max_len)
    f.write("\n\n")
    f.write("Dev data info\n")
    f.write("Data count: %d\n" % dev_data_cnt)
    f.write("Total len: %d\n" % dev_total_len)
    f.write("Average len: %.4f\n" % (1.0 * dev_total_len / dev_data_cnt))
    f.write("Max len: %d\n" % dev_max_len)
    f.write("\n\n")
    f.write("Vocab size: %d" % len(word2ind))

train_coded = [[word2ind[w] for w in sent] for sent in train_data]
dev_coded = [[word2ind[w] for w in sent] for sent in dev_data]

with open(train_output, 'w') as f:
    for sent in train_coded:
        f.write(' '.join(map(str, sent)))
        f.write('\n')
        
with open(dev_output, 'w') as f:
    for sent in dev_coded:
        f.write(' '.join(map(str, sent)))
        f.write('\n')

cPickle.dump((word2ind, ind2word), open(dic_output, 'w'))

print("Vocab size: %d" % len(word2ind))
print("Finish.")