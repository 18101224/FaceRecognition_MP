from torch.nn import functional 
import torch 

def compute_adv_loss(anchor_features, neg_features): 
    bs = anchor_features.shape[0]
    positives = ( anchor_features * neg_features ).sum(dim=-1).reshape(-1)
    similarities = neg_features@neg_features.T # bs, bs 
    fr_label = torch.arange(bs,device=torch.device('cuda'))
    similarities[fr_label, fr_label] = positives
    fr_loss = functional.cross_entropy(similarities, fr_label)
    return fr_loss 