from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys
import numpy as np

def normalize_answer(s):
    
    def remove_articles(text):
	return re.sub(r'\b(a|an|the)\b', ' ', text)
        return text

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
    #return s




f = open("total.answer",'r')
content3 = f.readlines()
content_answer = [(x.strip()) for x in content3]


f = open("total.context",'r')
content4 = f.readlines()
content_content = [(x.strip()) for x in content4]

f = open("total.question",'r')
content4 = f.readlines()
content_question = [(x.strip()) for x in content4]


print(len(content_answer))
count = 0
r=[]
start=[]
end=[]
for i in range(0,len(content_answer)):
	z= normalize_answer(content_answer[i])
	z1= normalize_answer(content_content[i])
	a = z.split(" ")
	b = z1.split(" ")
	flag=0
	for j in range(0,len(b)-len(a)+1): 
		if b[j:j+len(a)]==a:
			count +=1
			flag=1
			r.append(i)
			start.append(j)
			end.append(j+len(a))
			break


with open("total_updated.answer", "w") as f1:
	for i in range(0,len(r)):
		f1.writelines(content_answer[r[i]]+"\n")


with open("total_updated.question", "w") as f1:#
	for i in range(0,len(r)):
		f1.writelines(content_question[r[i]]+"\n")


with open("total_updated.context", "w") as f1:#
	for i in range(0,len(r)):
		f1.writelines(content_content[r[i]]+"\n")

print(len(start),len(r),len(end))
with open("total_updated.span", "w") as f1:
	for i in range(0,len(r)):
		f1.writelines(str(start[i])+" "+str(end[i])+"\n")