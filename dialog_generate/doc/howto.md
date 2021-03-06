This document describes how to use the scripts to generate data for training the NLU component.

# Quick Example

Please click [here](../quick_example.ipynb).

# Data Format

- The bool ontology is a string, '<YES\>' or '<NO\>'. (e.g. 'user_affirm': '<NO\>')
- The string ontology is a string. (e.g. 'user_inform_what': u'watch a movie')
- The time ontology is a number. The following description is IMPORTANT:
    - The hour is in the range [0, 24]. Number 24 is INCLUDED, meaning that the sentence does not mention the hour information (start hour, finish hour).
    - The minute is in the range [0, 4]. 0 means xx:00, 1 means xx:15, 2 means xx:30, 3 means xx:45, 4 means the sentence does not mention the minute information (start minute, finish minute).
- The ontology 'user_inform_who' is a list. (e.g. 'user_inform_who': [u'kaja', u'allsun', u'rebecca', u'tam']).

# How To

* In event_generator.py, you can change the ontology file to be used in the construction function:
```
    ontology = json.load(open('./ontology_itime3.json'))
```
        By defining the ontology file, we can use different event titles, different people name, etc. There are currently 3 ontology files: ontology_itime1-3.json.

* Run build_data1.py. It writes the data to './tmp/all_data3.pkl', and prints the first 50 data in console. The following is an example:
```
     {'text': u"alarm me next monday from seven forty five at night till ten fifteen pm . set the location as old geology , and i'm going with jena . ",
      'user_ack': '<NO>',
      'user_affirm': '<NO>',
      'user_dont_want_report': '<NO>',
      'user_finish': '<NO>',
      'user_inform_day': 'next monday',
      'user_inform_duration_hour': 24,
      'user_inform_duration_min': 4,
      'user_inform_what': '<NO>',
      'user_inform_whened_hour': 22,
      'user_inform_whened_min': 1,
      'user_inform_whenstart_hour': 19,
      'user_inform_whenstart_min': 3,
      'user_inform_where': 'old geology',
      'user_inform_who': [u'jena'],
      'user_report': '<NO>',
      'user_restart': '<NO>',
      'user_start': '<NO>'}
```
        Basically, that's all you need to do to generate the data.

* (Optional) Run build_dict.py. It generates the dictionary files. It considers the vocabulary of the training data is given, while unseen words in the dev data will be replaced with special tokens.

* (Optional) There is another script build_traindata.py, which is not very important. It does two things:
    * Add noise to the data, by randomly replace one word with another.
    * It splits the data into training data and dev data. Furthermore, it considers the vocabulary of the training data is given, while unseen words in the dev data will be replaced with special tokens.
    
# How It Works

Most of the jobs are handled by the process() function in build_data1.py. The basic idea is to generate dialogs as usual. But since we are only interested in NLU, we will ignore unrelevent parts in the dialogs, such as the system's response. In process() the action list of the whole dialog is obtained by
```
a = list(act_list[k])
```
(You can print it to see what it looks like.)
Then we only process interested sentences:
```
for act in a:
    if act == 'user_affirm' or act == 'user_ack' or act == 'user_finish' \
       or act == 'user_start' or act == 'user_restart' \
       or act == 'user_report' or act == 'user_dont_want_report':
        res[act] = '<YES>'
        isUser = True
    elif isinstance(act, tuple) and act[0] == 'user_inform':
        if act[1] == 'what':
            res['user_inform_what'] = act[2].lower()
    ....
```
This means that it picks out the actions of 'user_affirm', 'user_ack', ..., 'user_inform'. Actually in the NLU task, only 'user_inform' is of interest. It is not difficult to change the script to only generate 'user_inform' sentences.

# Generating Raw Sentences

## Raw Sentences and Patterns

The followings are some examples of raw sentences (patterns):
```
USER_INFORM_WHAT_WHENST_WHENED_DAY
from <st_time> to <ed_time> i wish to <what> . it's <day>
i want to <what> from <st_time> to <ed_time> on <day>
<what> from <st_time> until <ed_time> , and the day is <day>
<what> . set the day to be <day> at <st_time> until <ed_time>
```

## Generate Patterns with Grammar

All the patterns are generated by grammars, essentially defined in the file grammar/grammar.cfg. Let's starts with a simple example:
```
USER_INFORM_WHENST -> IT_STARTS AT_FROM ST_TIME | WANT_TO GO AT ST_TIME | REMIND 'me' AT ST_TIME | 'set the start time' TOBE_AS ST_TIME
IT_STARTS -> 'it starts' | 'it will start' | 'it begins' | 'it will begin'
REMIND -> 'remind' | 'alarm' | 'call'
GO -> 'go' | 'leave' | 'set off'
TOBE_AS -> 'to be' | 'as'
...
```
The grammar USER_INFORM_WHENST means the user informs the starting time of the event. In the first line, it defines that a user can inform when start in 5 different way.

The first way is IT_STARTS AT_FROM ST_TIME. The following lines are some of the recursive definition. For instance, IT_STARTS can be 'it starts', 'it will start', etc. As a result, sentences like 'it starts from 7 am' or 'it will begin from 8 am' can all be generated.

Now let's move to a more complex grammar example:
```
USER_INFORM_WHAT_WHENST_WHENED -> USER_INFORM_WHAT FROM WHENST_BODY TO_TILL WHENED_BODY | FROM WHENST_BODY TO_TILL WHENED_BODY USER_INFORM_WHAT | USER_INFORM_WHAT_WHENST AND USER_INFORM_WHENED
```
This grammar means that, if a user want to inform the event title (what), the start time and the end time in a single sentence, he can say it in 3 major ways. The first way is like 'I want to go swimming from 5 pm till 8 pm'. The second way is like 'From 5 pm to 8 pm I will go swimming'...

The advantage of grammar is that, it could generate much more sentences than traditional patterns. Let cnt(GRAMMAR) be the number of patterns that a grammar can generate. In the above example:
```
cnt(USER_INFORM_WHAT_WHENST_WHENED) = cnt(USER_INFORM_WHAT) * cnt(FROM) * cnt(WHENST_BODY) * cnt(TO_TILL) * cnt(WHENED_BODY) +
    cnt(FROM) * cnt(WHENST_BODY) * ...
```
You can see that the number can go quite large.

## How to Generate Raw Sentences

You can start with the script grammar/grammar_test.py. The key function is
```
        sent = next(generate(grammar, start=Nonterminal(grammar_start),\
                         n=1))
```
Here 'grammar' is the structure loaded with NLTK, grammar_start is a string (e.g. 'USER_INFORM_WHAT_WHENST_WHENED'), 'n' is the number of generated sentences.

## TODO

The genrated sentences are 'raw'. For example: i want to <what\> , and <who\> will be with me from <st_time\> till <ed_time\> on <day\> at <where\>.

To generate a valid sentence, we need to replace each token with specific value.

One problem is that there are many ways to say about time. For example, 3: 30 am can be say as 'three thirty am', 'half past three am', 'half past three in the morning', ... In build_data1.py, the function time2str handles <st_time\>, <ed_time\>. Similarly, a user may say the duration in several different way. The function dur2str handles <dur_time\>.

Replacing other slots, like <what\>, <where\>, <who\> should be easy, but maybe you need to write your own scripts to handle them.