from .vit_kprpe import load_model
from omegaconf import OmegaConf
import os
import sys
sys.path.extend('..')
from kp import get_kprpe
import torch




def get_kprpe_pretrained(cfg_path,token_path=False,force_download=False):
    if not token_path :
        cfg = OmegaConf.load(os.path.join(cfg_path,'model.yaml'))
        model = load_model(cfg)
        model.load_state_dict_from_path(os.path.join(cfg_path,'model.pt'))
    else:
        model = get_kprpe(token_path,force_download)
    return model

def load_kprpe_finetuned(model,ckpt_path):
    cls_path = os.path.join(ckpt_path,'classifier.pt')
    vit_path = os.path.join(ckpt_path,'model.pt')
    model.classifier.load_state_dict(torch.load(cls_path))
    model.kp_rpe.load_state_dict(torch.load(vit_path))
    return model

def load_ckpt_kprep(model,ckpt_path):
    classifier_path = os.path.join(ckpt_path,'classifier.pt')
    kp_rpe_path = os.path.join(ckpt_path,'model.pt')
    model.classifier.load_state_dict(torch.load(classifier_path))
    model.kp_rpe.load_state_dict(torch.load(kp_rpe_path))
    return model