import numpy as np

def helper(s,content,start,end):
	count = 0 
	#print content
	with open(s, "w") as f1:
		for j in content:
			if start <= count < end:
				f1.writelines(j+"\n")
			count +=1
	print count


f = open("total_updated.span",'r')
content = f.readlines()

content_start = [(x.strip()) for x in content]
total = len(content)

f = open("total_updated.question",'r')
content2 = f.readlines()
content_question = [(x.strip()) for x in content2]

f = open("total_updated.answer",'r')
content3 = f.readlines()
content_answer = [(x.strip()) for x in content3]


f = open("total_updated.context",'r')
content4 = f.readlines()
content_content = [(x.strip()) for x in content4]

from random import shuffle
import random
x = [[i] for i in range(total)]
random.Random(4).shuffle(x)
print x
print x[3]
	

helper("train.content",content_content,0,65001)
helper("val.content",content_content,65001,75001)
helper("test.content",content_content,75001,total)
