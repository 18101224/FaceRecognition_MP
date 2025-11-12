import torch 
from torch import nn 

class SCL:
    def __init__(self, temperature=0.1, contrast_mode='all', base_temperature=0.07):
        self.temperature = temperature 
        self.contrast_mode = contrast_mode 
        self.base_temperature = base_temperature
        
    def __call__(self,logits, features, y, **kwargs ):
        ce_loss = torch.nn.functional.cross_entropy(logits, y)
        label_mask = y.unsqueeze(1) == y.unsqueeze(0)
        label_mask.fill_diagonal_(False)
        sims = features @ features.T / self.temperature 
        negative_mask = torch.eye(y.shape[0],device=features.device,dtype=torch.bool)
        num = torch.logsumexp(sims.masked_fill(label_mask,float('-inf')),dim=1)
        den = torch.log(torch.sum(torch.exp(sims.masked_fill(negative_mask,float('-inf'))),dim=1))
        cl_loss = -(1/label_mask.sum(dim=-1,keepdim=False) * (num-den)).mean()
        return ce_loss, cl_loss, None

