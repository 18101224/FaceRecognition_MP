import torch
from torch import nn 
import torch.nn.functional as F
import torch.distributed as dist
import math
from .KCL import KCL
from .ETF import compute_etf_loss
from .EKCL import EKCL
from .BCL import BCL
import sys 
from .Moco import Moco
sys.path.append('../..')
from dataset import ClassBatchSampler, get_fer_transforms


__all__ =  ['Moco', 'KCL', 'compute_etf_loss', 'EKCL', 'BCL', 'get_cl_loss']


def include(loss, losses):
    to_check = loss.split('_')
    for loss_name in losses : 
        for loss in to_check : 
            if loss_name == loss : 
                return True 
    return False

def get_cl_loss(args, model=None, init_queue=None):
    if include(args.loss, ['CE']) : 
        return CE()
    elif include(args.loss, ['EKCL']) :
        return EKCL(args, fetcher =  ClassBatchSampler(args, transform=get_fer_transforms(train=True, model_type=args.model_type),idx=False,num_workers=args.num_workers)
        , temperature=args.temperature)
    elif include(args.loss, ['KBCL']) :
        return KCL(args, model, temperature=args.temperature, init_queue=init_queue)
    elif include(args.loss, ['BCL']) :
        return BCL(cls_num_list=None, temperature=args.temperature)
    else:
        raise ValueError(f'{args.loss} is not supported')

class CE:
    def __init__(self):
        self.fn = nn.CrossEntropyLoss().to(torch.device('cuda'))
    def __call__(self, logits, y, **kwargs): 
        return self.fn(logits,y), torch.tensor(0.0, device=logits.device, requires_grad=True), None # ce, cl, K 
        
