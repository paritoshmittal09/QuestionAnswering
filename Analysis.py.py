from __future__ import division
import os
import glob
import numpy as np


train_context = 'train.context'

context_len = []

with open(train_context) as fp:  
    lines = fp.readlines()
    print(len(lines))
    for line in lines:
        words = len(line.split())
        context_len.append(words)



import matplotlib.pyplot as plt

import seaborn as sns
sns.set(color_codes=True)
ax = sns.distplot(context_len)
ax.set_title('Distribution of Context Length')
ax.set_ylabel('Context Length')
plt.savefig('context_length.png')
plt.show()

context_array = np.array(context_len)
print("Min:   ", np.min(context_array))
print("Max:   ", np.max(context_array))
print("Mean:   ", np.mean(context_array))
print("25th percentile:   ", np.percentile(context_array, 25))
print("Median:            ", np.median(context_array))
print("75th percentile:   ", np.percentile(context_array, 75))
print("95th percentile:   ", np.percentile(context_array, 95))
print("99th percentile:   ", np.percentile(context_array, 99))



train_ques = 'train.question'

ques_len = []

with open(train_ques) as fp:  
    lines = fp.readlines()
    print(len(lines))
    for line in lines:
        words = len(line.split())
        ques_len.append(words)

plt.plot(ques_len)
plt.ylabel('Question Length')
plt.show()


import matplotlib.pyplot as plt
#matplotlib inline
import seaborn as sns
sns.set(color_codes=True)
ax = sns.distplot(ques_len)
ax.set_title('Distribution of Question Length')
ax.set_ylabel('Question Length')
plt.savefig('question_length.png')
plt.show()


ques_array = np.array(ques_len)

print("Min:   ", np.min(ques_array))
print("Max:   ", np.max(ques_array))
print("Mean:   ", np.mean(ques_array))
print("25th percentile:   ", np.percentile(ques_array, 25))
print("Median:            ", np.median(ques_array))
print("75th percentile:   ", np.percentile(ques_array, 75))
print("95th percentile:   ", np.percentile(ques_array, 95))
print("99th percentile:   ", np.percentile(ques_array, 99))




train_ans = 'train.answer'

ans_len = []

with open(train_ans) as fp:  
    lines = fp.readlines()
    print(len(lines))
    for line in lines:
        words = len(line.split())
        ans_len.append(words)


ans_array = np.array(ans_len)

print("Min:   ", np.min(ans_array))
print("Max:   ", np.max(ans_array))
print("Mean:   ", np.mean(ans_array))
print("25th percentile:   ", np.percentile(ans_array, 25))
print("Median:            ", np.median(ans_array))
print("75th percentile:   ", np.percentile(ans_array, 75))
print("95th percentile:   ", np.percentile(ans_array, 95))
print("99th percentile:   ", np.percentile(ans_array, 99))



import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
sns.set(color_codes=True)
ax = sns.distplot(ans_len)
ax.set_title('Distribution of Answer Length')
ax.set_ylabel('Answer Length')
plt.savefig('answer_length.png')
plt.show()


train_ans = 'train.span'

ans_start = []
ans_end = []

with open(train_ans) as fp:  
    lines = fp.readlines()
    print(len(lines))
    for line in lines:
        words = line.split()
        ans_start.append(int(words[0]))
        ans_end.append(int(words[1]))
        
        
plt.plot(ans_start)
plt.ylabel('Answer Start')
plt.show()


ans_start_array = np.array(ans_start)

print("Min:   ", np.min(ans_start_array))
print("Max:   ", np.max(ans_start_array))
print("Mean:   ", np.mean(ans_start_array))
print("25th percentile:   ", np.percentile(ans_start_array, 25))
print("Median:            ", np.median(ans_start_array))
print("75th percentile:   ", np.percentile(ans_start_array, 75))
print("95th percentile:   ", np.percentile(ans_start_array, 95))
print("99th percentile:   ", np.percentile(ans_start_array, 99))


ans_start_relative = np.true_divide(ans_start_array, context_array)


print("Min:   ", np.min(ans_start_relative))
print("Max:   ", np.max(ans_start_relative))
print("Mean:   ", np.mean(ans_start_relative))
print("25th percentile:   ", np.percentile(ans_start_relative, 25))
print("Median:            ", np.median(ans_start_relative))
print("75th percentile:   ", np.percentile(ans_start_relative, 75))
print("95th percentile:   ", np.percentile(ans_start_relative, 95))
print("99th percentile:   ", np.percentile(ans_start_relative, 99))

