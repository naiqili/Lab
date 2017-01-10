# -*- coding: utf-8 -*-

from scripts.dataset_walker import *
from pprint import pprint, pformat

dataroot_path = "d:/dataset/dstc2/data"
dataset_train = dataset_walker("dstc2_train", dataroot=dataroot_path, \
                               labels=True)
dataset_dev = dataset_walker("dstc2_dev", dataroot=dataroot_path, \
                               labels=True)

user_train = []
user_dev = []

def dialog_acts_to_str(acts):
    res = ""
    for act in acts:
        res = res + act["act"] + "("
        if len(act["slots"]) > 0:
            res = res + act["slots"][0][0] + ", " + act["slots"][0][1]
            for slot in act["slots"][1:]:
                res = res + ", " + slot[0] + slot[1]
        res = res + ")  "
    return res

def acts_to_set(acts):
    res = []
    for act in acts:
        s = act["act"]
        if s == "inform" or s == "deny" or s == "confirm":
            s = s + act["slots"][0][0]
            res.append([s, act["slots"][0][1]])
        elif s == "request":
            s = s + act["slots"][0][1]
            res.append([s])
        else:
            res.append([s])
    return res

print("Loading data ...")

for call in dataset_train :
    for _turn, _label in call :
        act_abstract = acts_to_set(_label["semantics"]["json"])
        trans = _label["transcription"]
        if act_abstract != [['hello']]:
            user_train.append((act_abstract, trans))

with open('tmp/train.txt', 'w') as f:
    for (abstract, trans) in user_train:
        f.write(trans + '\n')
        
        
for call in dataset_dev :
    for _turn, _label in call :
        act_abstract = acts_to_set(_label["semantics"]["json"])
        trans = _label["transcription"]
        if act_abstract != [['hello']]:
            user_dev.append((act_abstract, trans))

with open('tmp/dev.txt', 'w') as f:
    for (abstract, trans) in user_dev:
        f.write(trans + '\n')
        
uni_act_set = []
uni_act_type = []
for (act, _) in user_train + user_dev:
    act_s = set([k[0] for k in act])
    if act_s not in uni_act_set:
        uni_act_set.append(act_s)
        for a in act:
            if a[0] not in uni_act_type:
                uni_act_type.append(a[0])
                
with open('tmp/data_viewer.log', 'w') as f:
    f.write("Cnt of unique action set: %d\n" % len(uni_act_set))
    f.write(pformat(uni_act_set))
    f.write('\n=======================================\n')
    f.write("Cnt of unique action type: %d\n" % len(uni_act_type))
    f.write(pformat(uni_act_type))