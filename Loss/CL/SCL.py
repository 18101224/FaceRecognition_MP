import torch 
from torch import nn 

class SCL:
    def __init__(self, temperature=0.1, contrast_mode='all', base_temperature=0.07):
        self.temperature = temperature 
        self.contrast_mode = contrast_mode 
        self.base_temperature = base_temperature
        
def __call__(self, logits, features, y, **kwargs):
    ce_loss = torch.nn.functional.cross_entropy(logits, y)
    # masks
    pos_mask = (y.unsqueeze(1) == y.unsqueeze(0))
    pos_mask.fill_diagonal_(False)
    self_mask = torch.eye(y.shape[0], device=features.device, dtype=torch.bool)
    # similarities
    sims = features @ features.T / self.temperature
    # numerical stability
    sims = sims - sims.max(dim=1, keepdim=True)[0]
    # log-sum-exp denominator (exclude self)
    sims_den = sims.masked_fill(self_mask, float('-inf'))
    log_den = torch.logsumexp(sims_den, dim=1)
    # numerator: positives only
    sims_pos = sims.masked_fill(~pos_mask, float('-inf'))
    log_num = torch.logsumexp(sims_pos, dim=1)
    # per-anchor loss
    pos_count = pos_mask.sum(dim=1).clamp_min(1)
    cl_loss = -((log_num - log_den) / pos_count).mean()
    return ce_loss, cl_loss, None

    