import nltk
from nltk.parse.generate import Nonterminal
from generate import generate
from utils import subsets_of

grammar = nltk.data.load('file:grammar2.cfg')
k=30
for steps in range(k):
    sent = next(generate(grammar, start=Nonterminal('S'),\
                         n=1))
    print(' '.join(sent))

