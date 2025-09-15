from torch.nn import functional 
import torch 
from .analysis import analyze_and_update_gradients

__all__ = ['compute_adv_loss', 'analyze_and_update_gradients']
def compute_adv_loss(anchor_features, neg_features): 
    bs = anchor_features.shape[0]
    positives = ( anchor_features * neg_features ).sum(dim=-1).reshape(-1)
    similarities = neg_features@neg_features.T # bs, bs 
    fr_label = torch.arange(bs,device=torch.device('cuda'))
    similarities[fr_label, fr_label] = positives
    fr_loss = functional.cross_entropy(similarities, fr_label)
    return fr_loss 