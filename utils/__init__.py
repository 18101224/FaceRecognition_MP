import torch
from tqdm import tqdm
import pandas as pd
import os
import shutil
import psutil
import pickle
from .plot import *
from .balancing import *
from .rl_logger import *
from datetime import datetime
from .confusion import *
from .eval_quality import *
from .pushover import send_message
from .epx_log import get_exp_id
from .interpolation import calculate_class_centers, slerp_ema
from .DDP import sync_scalar, concat_all_gather, sync_defaultdict
from .visualize import crop_to_square_grid
from facenet_pytorch import MTCNN
from .grad import measure_grad

__all__ = ['get_acc', 'get_macro_acc', 'get_norm', 'get_pd', 'save_pd', 'save_pkl', 
'get_mem', 'get_dict', 'save_dict', 'sync', 'sync_tensor', 'torchload', 'get_grads',
 'calculate_class_centers', 'slerp_ema', 'sync_scalar', 'concat_all_gather', 'get_ldmk',
  'crop_to_square_grid', 'get_exp_id', 'measure_grad', 'sync_defaultdict']

def get_grads(obj):
    """
    Compute L2 norm of gradients for either a torch.nn.Module or a torch.nn.Parameter.

    Returns a 0-dim torch.Tensor on the correct device, or None if no grads.
    """
    # Handle None/bool sentinels
    if obj is None or isinstance(obj, bool):
        return None

    # Single parameter
    if isinstance(obj, torch.nn.Parameter):
        if obj.grad is None:
            return None
        return obj.grad.norm()

    # Module: iterate parameters
    if isinstance(obj, torch.nn.Module):
        grads = [p.grad for p in obj.parameters() if p.grad is not None]
        if not grads:
            return None
        device = grads[0].device
        dtype = grads[0].dtype
        total_sq = torch.tensor(0.0, device=device, dtype=dtype)
        for g in grads:
            total_sq = total_sq + g.norm().pow(2)
        return total_sq.sqrt()

    # Iterable of parameters (fallback)
    if isinstance(obj, (list, tuple)):
        grads = [p.grad for p in obj if isinstance(p, torch.nn.Parameter) and p.grad is not None]
        if not grads:
            return None
        device = grads[0].device
        dtype = grads[0].dtype
        total_sq = torch.tensor(0.0, device=device, dtype=dtype)
        for g in grads:
            total_sq = total_sq + g.norm().pow(2)
        return total_sq.sqrt()

    # Unknown type
    return None

@torch.no_grad()
def get_acc(x,y):
    '''
    :param x: bs, num_classes // gpu 
    :param y: bs // gpu 
    :return: scalar that is averaged
    '''
    _,pred = torch.max(x,dim=-1)
    bs = x.shape[0]
    result = pred == y
    result = result
    result = sum(result)
    return (result/bs).detach().item()

@torch.no_grad()
def get_macro_acc(x,y):
    '''
    :param x: bs, num_classes // gpu 
    :param y: bs // gpu 
    :return: returns a num_classes shaped tensor, counting the number of correct predictions per class
    '''
    _, pred = torch.max(x,dim=-1)
    binary = pred == y
    n_classes = x.shape[-1]
    result = torch.zeros((n_classes)).to(x.device)
    for c in range(n_classes):
        a = sum(binary[y==c])
        result[c] = a
    return result.float().reshape(-1).detach()

def get_norm(model,loader,device):
    with torch.no_grad():
        result = 0
        n = 0
        for img, _ in tqdm(loader) :
            img = img.to(device)
            z, _ = model(img)
            z = torch.nn.functional.avg_pool2d(z,kernel_size=14).reshape(-1,256)
            norm = torch.norm(z,dim=0,keepdim=False).reshape(-1)
            bs = img.shape[0]
            mean = torch.mean(norm,dim=0,keepdim=False).reshape(-1).item()
            n+=bs
            result = result*((n-bs)/n) + mean*(bs/n)
    return result

def get_pd(path):
    if not os.path.exists(path):
        df = {'best':[], 'bs':[], 'lr':[], 'save_path':[]}
        df = pd.DataFrame(df)
        return df
    else:
        data = pd.read_excel(path)
        return pd.DataFrame(data)

def save_pd(path, data):
    data.to_excel(path,index=False)

def save_pkl(index):
    path = os.path.join(str(index),'log.pkl')
    if not os.path.exists('logs'):
        os.mkdir('logs')
    shutil.copy(path,f'logs/{index}.pkl')
    return path

def get_mem():
    pid = os.getpid()
    current_process = psutil.Process(pid)
    memory_info = current_process.memory_info()
    rss_memory = memory_info.rss
    rss_memory_mb = rss_memory / ( 1024 ** 3 )
    avail = psutil.virtual_memory().available / ( 1024 ** 3 )
    return avail, rss_memory_mb


def get_dict(path):
    if not os.path.exists(path):
        return {'gamma':[],'gp':[],'k':[],'save path':[],'best_reward':[],'best_acc':[]}
    else:
        with open(path,'rb') as f :
            log = pickle.load(f)
        return log

def save_dict(dict,path):
    with open(path,'wb') as f :
        pickle.dump(dict,f)

def sync(metric,device):
    metric = torch.tensor(metric,device=device,dtype=torch.float64)
    torch.distributed.all_reduce(metric,op=torch.distributed.ReduceOp.SUM)
    return metric.item()

def sync_tensor(tensor):
    torch.distributed.all_reduce(tensor,op=torch.distributed.ReduceOp.SUM)
    return tensor

def torchload(path,weights_only,map_location=None):
    try:
        return torch.load(path,weights_only=weights_only,map_location=map_location)
    except:
        return torch.load(path,map_location=map_location)


@torch.no_grad()
def get_ldmk(img, aligner):
    _,_,ldmk,_,_,_ = aligner(img)
    return ldmk 