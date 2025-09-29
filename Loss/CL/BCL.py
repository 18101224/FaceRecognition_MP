import torch 



import torch
import torch.nn as nn


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LogitAdjust:

    def __init__(self, cls_num_list, tau=1, weight=None):

        if cls_num_list is not None:
            cls_num_list = torch.cuda.FloatTensor(cls_num_list)
            cls_p_list = cls_num_list / cls_num_list.sum()
            m_list = tau * torch.log(cls_p_list)
            self.m_list = m_list.view(1, -1)
        self.weight = weight

    def __call__(self, x, target, use_logit_adjust=False):
        if use_logit_adjust:
            x_m = x + self.m_list
        else:
            x_m = x
        return F.cross_entropy(x_m, target, weight=self.weight)

class BalSCL:
    def __init__(self, cls_num_list=None, temperature=0.1):
        self.temperature = temperature
        self.cls_num_list = cls_num_list

    def __call__(self, centers1, features, targets, ):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        batch_size = features.shape[0]
        targets = targets.contiguous().view(-1, 1)
        num_classes = centers1.shape[0]
        targets_centers = torch.arange(num_classes, device=device).view(-1, 1)
        targets = torch.cat([targets.repeat(2, 1), targets_centers], dim=0)
        batch_cls_count = torch.eye(num_classes, device=device)[targets].sum(dim=0).squeeze()

        mask = torch.eq(targets[:2 * batch_size], targets.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # class-complement
        features = torch.cat(torch.unbind(features, dim=1), dim=0) # 2*bs, dim
        features = torch.cat([features, centers1], dim=0) # 2*bs+C, dim
        logits = features[:2 * batch_size].mm(features.T) # 2*bs, 2*bs+C
        logits = torch.div(logits, self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True) 
        logits = logits - logits_max.detach()


        # class-averaging
        exp_logits = torch.exp(logits) * logits_mask # 2*bs, 2*bs+C
        per_ins_weight = torch.tensor([batch_cls_count[i] for i in targets], device=device).view(1, -1).expand(
            2 * batch_size, 2 * batch_size + num_classes) - mask
        exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)
        
        log_prob = logits - torch.log(exp_logits_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos
        loss = loss.view(2, batch_size).mean()
        return loss


class BCL: 
    def __init__(self, cls_num_list=None, temperature=0.1):
        self.criterion_ce = LogitAdjust(cls_num_list) 
        self.criterion_scl = BalSCL(cls_num_list, temperature)
    
    def __call__(self,logits, features, y, weight=None, centers=None, aligner=None, positive_pair=None, requires_grad=False, **kwargs):
        '''
        features : 2*bs, dim ( concat original and view features)
        centers : C, dim 
        logits : bs, C 
        targets : bs ( will be exptended in this function )
        and you should process weighted sum for the losses. 
        '''

        bs = y.shape[0]
        _, f1, f2 = torch.split(features, [bs,bs,bs], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        ce_loss = self.criterion_ce(x=logits[:bs], target=y, use_logit_adjust=False)
        cl_loss = self.criterion_scl(centers1=weight, features=features, targets=y)

        return ce_loss, cl_loss