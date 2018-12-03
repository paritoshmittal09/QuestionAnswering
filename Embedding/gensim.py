import string
import re
import argparse
import json
import sys
import gensim, logging
import numpy as np
from gensim.models import Word2Vec

def normalize(s):
    
    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


sentences=[]
f = open("train.context",'r')
content = f.readlines()
content = [(x.strip()) for x in content]
#print content
vocal = [ ]

for i in content:
	temp = normalize(normalize(i))
	#print temp
	w = temp.split()
	sentences.append(w)

f = open("train.question",'r')
content = f.readlines()
content = [(x.strip()) for x in content]
#print content
vocal = [ ]

for i in content:
	temp = normalize(normalize(i))
	#print temp
	w = temp.split()
	sentences.append(w)


model = Word2Vec(sentences, min_count=1,size=100,sg=1,window=6,workers = 4,min_count=5)



model.wv.save_word2vec_format('model_SKIP.txt', binary=False)
