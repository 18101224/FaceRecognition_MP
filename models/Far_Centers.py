import torch
from torch import nn
import numpy as np
from tqdm import tqdm

class Far_Weight(nn.Module):
    def __init__(self,dims, n_classes, target=(2/3)*torch.pi):
        super().__init__()
        self.n_classes = n_classes
        self.dims = dims
        self.target = target
        self.weight = torch.randn((n_classes,dims),requires_grad=True)
        self.norm_weight()
        print(self.check_opt())
        self.initialize()
        self.device = torch.device('cuda')
        self.weight = self.weight.to(self.device)


    def initialize(self,):
        opt = torch.optim.SGD([self.weight], lr=0.001)

        for i in tqdm(range(400000)):
            opt.zero_grad()
            norm = self.weight.norm(p=2,dim=1)
            weight = self.weight/norm.reshape(self.n_classes,1)
            sims =  weight@weight.T
            loss = torch.sum(sims,dim=[0,1],keepdim=False)
            loss.backward(retain_graph=True)
            opt.step()

        self.norm_weight()


    def norm_weight(self):
        self.weight = self.weight / self.weight.norm(p=2, dim=1).reshape(self.n_classes, 1)


    def check_opt(self):
        weight = self.weight.data
        sims = weight@weight.T
        return sims