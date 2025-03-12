from torch.nn.functional import softmax
import torch
from collections import Counter

class ClassAdaLoss:
    def __init__(self, labels, alpha):
        count = Counter(labels)
        array = []
        for key, value in count.items():
            array.append((int(key),value))
        array = torch.tensor(array).float()
        self.array = self.get_adapt(array,alpha)

    def get_adapt(self,array,alpha):
        x = (-array[:,1]/sum(array[:,1]))+alpha
        x = softmax(x,dim=0)
        return torch.concat((array[:,0],x),dim=-1).reshape(2,-1).transpose(-1,-2)

    def map_ada_logit(self,label):
        return self.array[label,1]

def class_quality_weight(class_qualities, labels,alpha=1):
    return (1-class_qualities[labels]*alpha)

