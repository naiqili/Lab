import nltk, random, json, itertools, cPickle
from nltk.parse.generate import Nonterminal
from grammar.generate import generate
from utils import *
from build_data1 import dur2str, time2str
from pprint import pprint

ontology_file = './ontology_itime3.json'
output_file = './tmp/nlu_data.pkl'

all_data = []

# Prepare ontology
ontology = json.load(open(ontology_file))
locations = list(itertools.chain.from_iterable([loc['short_name'] for loc in ontology['location']]))

def randomDay():
    flag = random_select([(True, 0.5), (False, 0.5)])
    if flag:
        lst1 = ["this", "next"]
        lst2 = ["Monday", "Tuesday", "Wednesday", "Thursday", \
            "Friday", "Saturday", "Sunday"]
    else:
        lst1 = ['january', 'february', 'march', 'april', 'may', 'june', 'july', \
                   'agaust', 'september', 'october', 'november', 'december']
        lst2 = list(map(str, range(1, 30)))
    res = [random.choice(l) for l in [lst1, lst2]]
    return res

# Prepare grammar
lst = ['WHAT', 'WHENST', 'WHENED', 'DAY', 'WHO', 'WHERE']

all_lst = subsets_of(lst)
all_lst.remove([])

str_lst = ['USER_INFORM_' + '_'.join(l) for l in all_lst]

grammar = nltk.data.load('file:grammar/grammar.cfg')

# Start generation
N = 7000
for n in range(N):
    grammar_start = random.choice(str_lst)
    sent = next(generate(grammar, start=Nonterminal(grammar_start), \
                         n=1))
    #print grammar_start, sent
    # Random title
    event_title = random.choice(ontology['title'])
    # Random names
    name_lst = []
    nxt_prob = 0.6
    for _ in range(4):
        name_lst.append(random.choice(ontology['invitee']).lower())
        if random.random() > nxt_prob:
            break
    # Random start time
    st_h = random.choice(range(24))
    st_m = random.choice(range(4))
    # Random duration
    dur_h = random.choice(range(9))
    dur_m = random.choice(range(4))
    # Random end time
    ed_h = random.choice(range(24))
    ed_m = random.choice(range(4))
    # Start-end consistency
    if st_h*60 + st_m*15 > ed_h*60 + ed_m*15:
        tmp = st_h
        st_h = ed_h
        ed_h = tmp
        tmp = st_m
        st_m = ed_m
        ed_m = tmp
    # Time string
    st_str = time2str(st_h, st_m*15)
    ed_str = time2str(st_h, st_m*15)
    dur_str = dur2str(dur_h, dur_m*15)
    # Random location
    location = random.choice(locations)
    # Random day
    day = randomDay()
    # print event_title, name_lst, st_h, st_m, dur_h, dur_m, ed_h, ed_m, location, day, time2str(st_h, st_m*15)
    text = ' '.join(sent).lower()
    words = text.split()
    res_text = []
    
    res_title = []
    res_invitee = []
    res_location = []
    res_day = []
    res_whenst = []
    res_whened = []
    res_dur = []

    plain_title = None
    plain_invitee = None
    plain_location = None
    plain_day = None
    plain_whenst = None
    plain_whened = None
    plain_dur = None
    
    for w in words:
        if w == '<what>':
            plain_title = event_title.lower()
            for ww in event_title.lower().split():
                res_text.append(ww)
                for lst in [res_title, res_invitee, res_location, res_day, res_whenst, res_whened, res_dur]:
                    lst.append(0)
                res_title[-1] = 1
        elif w == '<where>':
            plain_location = location.lower()
            for ww in location.lower().split():
                res_text.append(ww)
                for lst in [res_title, res_invitee, res_location, res_day, res_whenst, res_whened, res_dur]:
                    lst.append(0)
                res_location[-1] = 1
        elif w == '<day>':
            plain_day = ' '.join(day)
            for ww in day:
                res_text.append(ww)
                for lst in [res_title, res_invitee, res_location, res_day, res_whenst, res_whened, res_dur]:
                    lst.append(0)
                res_day[-1] = 1
        elif w == '<who>':
            plain_invitee = ' '.join(name_lst)
            for ww in name_lst:
                res_text.append(ww)
                for lst in [res_title, res_invitee, res_location, res_day, res_whenst, res_whened, res_dur]:
                    lst.append(0)
                res_invitee[-1] = 1
        elif w == '<st_time>':
            plain_whenst = st_str
            for ww in st_str.split():
                res_text.append(ww)
                for lst in [res_title, res_invitee, res_location, res_day, res_whenst, res_whened, res_dur]:
                    lst.append(0)
                res_whenst[-1] = 1
        elif w == '<ed_time>':
            plain_whened = ed_str
            for ww in ed_str.split():
                res_text.append(ww)
                for lst in [res_title, res_invitee, res_location, res_day, res_whenst, res_whened, res_dur]:
                    lst.append(0)
                res_whened[-1] = 1
        elif w == '<dur_time>':
            plain_dur = dur_str
            for ww in dur_str.split():
                res_text.append(ww)
                for lst in [res_title, res_invitee, res_location, res_day, res_whenst, res_whened, res_dur]:
                    lst.append(0)
                res_dur[-1] = 1
        else:
            res_text.append(w)
            for lst in [res_title, res_invitee, res_location, res_day, res_whenst, res_whened, res_dur]:
                lst.append(0)
    data = {
        'tok_text': res_text,
        'tok_title': res_title,
        'tok_location': res_location,
        'tok_invitee': res_invitee,
        'tok_whenst': res_whenst,
        'tok_whened': res_whened,
        'tok_dur': res_dur,
        'tok_day': res_day,
        'plain_text': text,
        'plain_title': plain_title,
        'plain_location': plain_location,
        'plain_invitee': plain_invitee,
        'plain_whenst': plain_whenst,
        'plain_whened': plain_whened,
        'plain_dur': plain_dur,
        'plain_day': plain_day
        }
    all_data.append(data)

#for d in all_data:
#    print(d)

# Write data    
cPickle.dump(all_data, open(output_file, 'w'))

print 'Done'
