import torch
from torch import nn 
from .KCL import KCL
from .BCL import BCL
from .SCL import SCL
from .EAC import EAC
from .SCN import SCN
from .NCL import NCL


__all__ =  ['get_cl_loss']


def include(loss, losses):
    to_check = loss.split('_')
    for loss_name in losses : 
        for loss in to_check : 
            if loss_name == loss : 
                return True 
    return False

def get_cl_loss(args, model=None, init_queue=None, class_counts=None):
    if include(args.loss, ['CE','BSCE']) : 
        return CE(args, class_counts)
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
    elif include(args.loss, ['NCL']) :
        return NCL(args, model, num_classes=args.num_classes, class_counts=class_counts, dim=args.dim, temperature=args.temperature)
    else:
        raise ValueError(f'{args.loss} is not supported')

class CE:
    def __init__(self,args, class_counts):
        self.bsce = True if args.loss != 'CE' else False 
        self.cc = torch.log(torch.as_tensor(class_counts, dtype=torch.float32))
        self.fn = nn.CrossEntropyLoss().to(torch.device('cuda'))
    def __call__(self, logits, y, **kwargs): 
        if self.bsce :
            logits = logits + self.cc.to(logits.device)
        loss = self.fn(logits, y)
        return loss, torch.tensor(0.0, device=logits.device, requires_grad=True), None # ce, cl, K 
        
