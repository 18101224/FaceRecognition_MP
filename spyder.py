import matplotlib.pyplot as plt 
import numpy as np 

def str_sum(lst):
    s =''
    for i in lst:
        s+=i
    return s[:-1] + 'finetuned'


remain_accs = [90.6,81.258, 80.867, 73.5 ]


finetuned = [ str_sum([f'block {i} ,' for i in range(1,4-t)]) for t in range(3) ] + ['linear_probing']

deleted_accs = [ 90.6, 68.54, 34.9 ]

deleted = [ 'no-removing', 'removed block 3', 'removed block 2 block 3' ]


plt.title('partial finetuning')
plt.xlabel('number of frozen blocks')
plt.ylim(0,100)
for idx, (acc, label) in enumerate(zip(remain_accs, finetuned)):
    plt.scatter(idx, acc, label=label)
plt.legend()
plt.savefig('remain_accs.png')
plt.close()


plt.title('removing blocks')
plt.xlabel('number of removed blocks')
plt.ylim(0,100)
for idx, (acc, label) in enumerate(zip(deleted_accs, deleted)):
    plt.scatter(idx, acc, label=label)
plt.legend()
plt.savefig('deleted_accs.png')
plt.close()
