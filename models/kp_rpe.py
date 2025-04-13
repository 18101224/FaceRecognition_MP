from .vit_kprpe import load_model
from omegaconf import OmegaConf
import os
from torch import nn
import sys
sys.path.extend('..')
from kp import get_kprpe
import torch
from .resmodule import Residual


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.mlps = Residual(in_features=512,num_layers=15)
        for i in range(4): # 원래 4였음
            self.blocks.append(
                nn.Sequential(
                nn.Linear(512,512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(),
                nn.Dropout(0.2))

            )
            self.norms.append(
                nn.BatchNorm1d(512)
            )
            self.activations.append(
                nn.SiLU()
            )

        self.classifier = nn.Linear(512,7)


    def forward(self,x):
        for block, norm, act in zip(self.blocks,self.norms,self.activations):
            res = x.clone()
            x = block(x)
            x = x+res
            x = norm(x)
            x = act(x)
        return self.classifier(x)


class CosClassifier(Classifier):
    def __init__(self):
        super().__init__()
        del self.classifier
        weight = torch.randn((7, 512))
        weight = nn.functional.normalize(weight, p=2, dim=1)
        self.kernel = nn.Parameter(
            weight.transpose(-1, -2), requires_grad=True
        ) # dim, num_class
    def get_angles(self):
        kernel = nn.functional.normalize(self.kernel,dim=0,p=2)
        sims = kernel.transpose(-1, -2) @ kernel
        sims.fill_diagonal_(0) # shaped as num_classes X num_classses ( cos similarities )
        sims = torch.triu(sims)
        return sims # shaped as 2D matrix


    def get_margin(self, sim):
        _, indices = torch.max(sim,dim=-1)
        margins = torch.arccos(sim[torch.arange(sim.shape[0]),indices])
        return margins


    def forward(self,x):
        for block, norm, act in zip(self.blocks,self.norms,self.activations):
            res = x.clone()
            x = block(x)
            x = x+res
            x = norm(x)
            x = act(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        kernel = nn.functional.normalize(self.kernel, dim=0, p=2) # dim, num_class
        cos = x @ kernel  # bs, num_classes
        angles = self.get_angles()
        return angles, cos



def get_kprpe_pretrained(cfg_path,token_path,force_download):
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