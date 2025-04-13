import torch


def get_angle_loss(sims):
    return torch.sum(sims,dim=[0,1],keepdim=False)