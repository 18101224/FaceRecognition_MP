import torch
from torch import nn 
import torch.nn.functional as F
import torch.distributed as dist
import math
from .KCL import KCL
from .ETF import compute_etf_loss
from .EKCL import EKCL
from .BCL import BCL
from .SCL import SCL
from .EAC import EAC
from .SCN import SCN
import sys 
import os 
# Temporarily add project root to import from `dataset`, then remove it
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_added_tmp_path = False
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
    _added_tmp_path = True
try:
    from dataset import ClassBatchSampler, get_fer_transforms
finally:
    if _added_tmp_path and _PROJECT_ROOT in sys.path:
        sys.path.remove(_PROJECT_ROOT)



__all__ =  ['KCL', 'compute_etf_loss', 'EKCL', 'BCL', 'get_cl_loss']


def include(loss, losses):
    to_check = loss.split('_')
    for loss_name in losses : 
        for loss in to_check : 
            if loss_name == loss : 
                return True 
    return False

def get_cl_loss(args, model=None, init_queue=None, class_counts=None):
    if include(args.loss, ['CE']) : 
        return CE()
    elif include(args.loss, ['EKCL']) :
        return EKCL(args, fetcher =  ClassBatchSampler(args, transform=get_fer_transforms(train=True, model_type=args.model_type),idx=False,num_workers=args.num_workers)
        , temperature=args.temperature, class_counts=class_counts)
    elif include(args.loss, ['KBCL']) :
        return KCL(args, model, dim=args.dim, temperature=args.temperature, init_queue=init_queue)
    elif include(args.loss, ['BCL']) :
        return BCL(cls_num_list=None, temperature=args.temperature)
    elif include(args.loss, ['SCL']) :
        return SCL(temperature=args.temperature,)
    elif include(args.loss, ['EAC', 'BEAC']) :
        return EAC(args, class_counts=class_counts)
    elif include(args.loss, ['SCN']) :
        return SCN()
    else:
        raise ValueError(f'{args.loss} is not supported')

class CE:
    def __init__(self):
        self.fn = nn.CrossEntropyLoss().to(torch.device('cuda'))
    def __call__(self, logits, y, **kwargs): 
        return self.fn(logits,y), torch.tensor(0.0, device=logits.device, requires_grad=True), None # ce, cl, K 
        
