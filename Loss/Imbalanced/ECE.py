import torch 
import numpy as np
from torch_kmeans import SoftKMeans
from torch_kmeans.utils.distances import CosineSimilarity
from torch.nn.parallel import DistributedDataParallel as DDP
__all__ = ['get_angle_loss', 'weight_scheduling', 'ECELoss']

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
    
class ECELoss:
    def __init__(self, args, k, hard_weight, soft_weight, num_classes , surrogate:bool, temp=5, max_iter=50, std_weight=1.0, mean_weight=1.0):
        self.args = args 
        kmeans = SoftKMeans(n_clusters=k,
                                    distance=CosineSimilarity,
                                    normalize='unit',
                                    temperature=temp,
                                    max_iter=max_iter)
        self.kmeans = kmeans.cuda()
        self.hard_weight = hard_weight
        self.soft_weight = soft_weight
        self.surrogate = surrogate
        if not surrogate :
            self.rho = -1/(num_classes-1) ## 이거 원래 -1 아니었음 !! 
        self.std_weight = std_weight
        self.mean_weight = mean_weight
    def __call__(self, weight):
        '''
        weight : num_classes, dim. normalized weight 
        '''
        if self.hard_weight != self.soft_weight : 
            result = self.kmeans(weight.unsqueeze(0))
            labels = result.labels.squeeze(0).reshape(-1,1) # num_classes,1
            centers = result.centers # num_classes, K, dim 
            mask = (labels.T==labels)
            weight_matrix = torch.where(mask, self.soft_weight, self.hard_weight) # if same 
        else:
            weight_matrix = torch.ones([weight.shape[0]]*2).to(weight.device)
        sims = weight@weight.T 
        triu_indices = torch.triu_indices(sims.shape[0], sims.shape[1], offset=1)
        if self.surrogate :
            std = sims[triu_indices].reshape(-1).std()
            mean = (sims[triu_indices]*weight_matrix[triu_indices]).reshape(-1).mean()
            loss = self.mean_weight*mean*(int(bool(self.args.use_mean)))+self.std_weight*std
        else: 
            loss = (((sims[triu_indices]-self.rho)**2)*weight_matrix[triu_indices]).reshape(-1).mean()
        return loss*self.args.ece_weight


