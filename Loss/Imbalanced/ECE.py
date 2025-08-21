import torch 
import numpy as np

__all__ = ['get_angle_loss', 'weight_scheduling']

def get_angle_loss(kernel):
    sims = kernel.T @ kernel 
    i, j = torch.triu_indices(kernel.shape[1], kernel.shape[1], offset=1)
    up = sims[i,j].reshape(-1)
    mean, std = torch.mean(up), torch.std(up)
    return mean,0.7*std


def weight_scheduling(method, beta, epoch, n_epochs):
    if method == 'linear':
        return beta * min(1.0, epoch / n_epochs)
    elif method == 'cosine':
        # epoch/n_epochs: 0~1 구간
        if epoch >= n_epochs:
            return beta
        return beta * (1 - np.cos(np.pi * epoch / (2 * n_epochs)))
    elif method == 'sigmoid':
        if epoch >= n_epochs:
            return beta
        phase = 1.0 - epoch / n_epochs
        return beta * np.exp(-5 * phase * phase)
    elif method == 'piecewise':
        # 예시: 0~30%는 0.1, 30~60%는 0.5, 이후는 1.0 (직접 수정해라)
        ratio = epoch / n_epochs
        if ratio < 0.3:
            return beta * 0.1
        elif ratio < 0.6:
            return beta * 0.5
        else:
            return beta * 1.0
    else:
        raise ValueError("Unknown method: {}".format(method))