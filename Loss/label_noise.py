from torch import nn
import torch

def get_label_noise(label,pred):
    n_classes = pred.shape[-1]
    pred = nn.functional.softmax(pred,dim=-1)
    _, y_hat = torch.max(pred, dim=1)
    confidence = pred[torch.arange(pred.shape[0]),label.int()]
    one_hot = nn.functional.one_hot(label,num_classes=pred.shape[-1]).to(pred.device) # bs, num_classes
    j = (n_classes+1/n_classes)*confidence - (1/n_classes)*torch.sum(pred,dim=-1,keepdim=False).reshape(pred.shape[0])
    j = j.reshape(pred.shape[0],1)
    sign = 2*((label == y_hat).reshape(pred.shape[0],1).int() - 1/2)
    return sign*(one_hot * j)

def including_margin(cos,j,m):
    #return cos + j*m
    return cos - (j*m)