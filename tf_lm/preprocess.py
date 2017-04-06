meta_path = './_data/movie_titles_metadata.txt'
movie_lines_path = './_data/movie_lines.txt'
convers_path = './_data/movie_conversations.txt'

cnt_movie = 614
cnt_train = 484
cnt_dev = 65 + 484
cnt_test = 65 + 65 + 484

line_data = {}
with open(movie_lines_path) as f:
    for line in f:
        line = line.strip()
        try:
            line_id, _, _, _, text = line.split(' +++$+++ ')
            line_data[line_id] = text
        except:
            pass

convers_data = []
with open(convers_path) as f:
    for line in f:
        line = line.strip()
        _, _, movie_id, lst = line.split(' +++$+++ ')
        convers_data.append((movie_id, eval(lst)))

'''        
for _, lst in convers_data[:20]:
    print
    for line_id in lst:
        print line_data[line_id]
'''

train_triples = []
dev_triples = []
test_triples = []
for movie_id, lst in convers_data:
    if len(lst) < 3:
        continue
    if int(movie_id[1:]) <= cnt_train:
        cur_triples = train_triples
    elif  int(movie_id[1:]) <= cnt_dev:
        cur_triples = dev_triples
    elif int(movie_id[1:]) <= cnt_test:
        cur_triples = test_triples
    for k in range(len(lst)-2):
        cur_triples.append([lst[k], lst[k+1], lst[k+2]])

print len(train_triples), len(dev_triples), len(test_triples)
        
