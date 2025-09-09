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

__all__ = ['get_acc', 'get_macro_acc', 'get_norm', 'get_pd', 'save_pd', 'save_pkl', 'get_mem', 'get_dict', 'save_dict', 'sync', 'sync_tensor', 'torchload']

def get_acc(x,y):
    '''
    :param x:
    :param y:
    :return: scalar
    '''
    _,pred = torch.max(x,dim=-1)
    bs = x.shape[0]
    result = pred == y
    result = result.detach().cpu()
    result = sum(result)
    return (result/bs).item()

def get_macro_acc(x,y):
    _, pred = torch.max(x,dim=-1)
    binary = pred == y
    n_classes = x.shape[-1]
    result = torch.zeros((n_classes)).to(x.device)
    for c in range(n_classes):
        a = sum(binary[y==c])
        result[c] = a
    return result.reshape(-1)

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
