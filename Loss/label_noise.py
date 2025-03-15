from torch import nn
import torch

def get_label_noise(label,pred):
    pred = nn.functional.softmax(pred,dim=-1)
    _, y_hat = torch.max(pred, dim=1)
    confidence = pred[torch.arange(pred.shape[0]),label]
    one_hot = nn.functional.one_hot(label,num_classes=pred.shape[-1]).to(pred.device) # bs, num_classes
    j = (8/7)*confidence - (1/7)*torch.sum(pred,dim=-1,keepdim=False).reshape(pred.shape[0])
    j = j.reshape(pred.shape[0],1)
    sign = (label == y_hat).reshape(pred.shape[0],1)
    return sign*(one_hot * j)

def including_margin(cos,j,m):
    return cos + j*m