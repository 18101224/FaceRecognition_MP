"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn.functional as F

__all__ = ['BCLLoss']


class BalSCL:
    def __init__(self, cls_num_list=None, temperature=0.1):
        self.temperature = temperature
        self.cls_num_list = cls_num_list

    def __call__(self,  targets, features, centers1=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        batch_size = features.shape[0]
        targets = targets.contiguous().view(-1, 1).to(device)
        targets_centers = torch.arange(len(self.cls_num_list), device=device).view(-1, 1)
        if centers1 is not None:
            targets = torch.cat([targets.repeat(2, 1), targets_centers], dim=0) # 2*batch_size + C , 1 
        else:
            targets = targets.repeat(2, 1)
        batch_cls_count = torch.eye(len(self.cls_num_list), device=device)[targets].sum(dim=0).squeeze() 


        mask = torch.eq(targets[:2 * batch_size], targets.T).float().to(device)
        logits_mask = torch.scatter(torch.ones_like(mask),1,torch.arange(batch_size * 2).view(-1, 1).to(device),0)
        mask = mask * logits_mask

        # class-complement
        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        if centers1 is not None:
            features = torch.cat([features, centers1], dim=0) #here 

        logits = features[:2 * batch_size].mm(features.T)
        logits = torch.div(logits, self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # class-averaging
        to_add = len(self.cls_num_list) if centers1 is not None else 0
        exp_logits = torch.exp(logits) * logits_mask
        per_ins_weight = torch.tensor([batch_cls_count[i] for i in targets], device=device).view(1, -1).expand(
            2 * batch_size, 2 * batch_size + to_add ) - mask
        exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)

        log_prob = logits - torch.log(exp_logits_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos
        loss = loss.view(2, batch_size).mean()
        return loss


class LogitAdjust:
    def __init__(self, cls_num_list, cosine_scaling=1., tau=1. , weight=None):
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight
        self.scale = cosine_scaling
    def __call__(self, x, target):
        x_m =  (self.scale*x + self.m_list)
        return F.cross_entropy(x_m, target, weight=self.weight)


class BCLLoss:
    def __init__(self, cls_num_list, args, temperature=1.):
        self.criterion_ce = LogitAdjust(cls_num_list, cosine_scaling=args.cosine_scaling)
        self.criterion_scl = BalSCL(cls_num_list, temperature)


    def __call__(self, centers, logits, features, targets, processed_features=None):
        '''
        centers : C, dim 
        logits : batch_size, C 
        features : batch_size, dim 
        targets : batch_size 
        processed_features : batch_size, dim 
        '''
        ce_loss = self.criterion_ce(logits, targets)

        if processed_features is not None:
            cls_loss = self.criterion_scl(targets=targets, features=processed_features, centers1=centers)
            rep_loss = self.criterion_scl(targets=targets, features=features)
            return ce_loss ,  (cls_loss + rep_loss)
        else:

            return ce_loss, self.criterion_scl(targets=targets, features=features, centers1=centers)
