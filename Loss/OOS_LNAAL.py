import torch 
import sys 
sys.path.append('../')
from dataset import get_kfolds, get_loaders
from tqdm import tqdm 

__all__ = ['get_confidence_db', 'get_instant_margin', 'apply_margin']

@torch.no_grad()
def get_confidence_db(models, aligner, loaders, datasets, device):
    confidence_db = []
    for fold_idx, (model, loader, dataset) in enumerate(zip(models, loaders, datasets)):
        original_indices = dataset.val_idx
        for img, label, idx in tqdm(loader, desc=f"Computing confidence db for fold {fold_idx}"):
            img = img.to(device)
            label = label.to(device)
            original_indice = torch.tensor(original_indices[idx]).to(device).reshape(-1,1)
            if aligner is not None:
                _, _, keypoint, _, _, _ = aligner(img)
                pred = model(img,keypoint)
            else:
                pred = model(img)
            result = torch.cat((original_indice, pred), dim=-1)
            confidence_db.append(result)
    confidence_db = torch.cat(confidence_db, dim=0)
    sorted_indices = torch.argsort(confidence_db[:, 0], dim=0)
    confidence_db = confidence_db[sorted_indices]
    return confidence_db[:,1:]


@torch.no_grad()
def get_instant_margin(pred, label):
    label = label.long()
    '''
    pred: prediction from g network
    label: original label 
    '''
    n_c = pred.shape[-1]
    y_hat = torch.argmax(pred, dim=-1)
    sign = ((y_hat==label).float()-0.5)*2
    pred = torch.softmax(pred, dim=-1)
    y_hat = pred[torch.arange(pred.shape[0],device=pred.device), label]
    return  sign*((n_c-2)/(n_c-1))*y_hat-(n_c/(n_c-1))*torch.mean(pred, dim=-1)


def apply_margin(pred, label, j, m,as_bias):
    '''
    pred : cosine similarities between target centre and image
    '''
    label = label.long()
    if as_bias:
        pred[torch.arange(pred.shape[0],device=pred.device),label] -= j*m
    else:
        angles = torch.arccos(pred)
        angles[torch.arange(pred.shape[0],device=pred.device),label] -= j*m
        pred = torch.cos(angles)
    return pred 


