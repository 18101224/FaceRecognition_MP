import torch 

__all__ = ['compute_etf_loss']

def compute_etf_loss(weight, etf_weight, statistics=False, std_weight=0.7):
    '''
    weight : num_classes, dim, normalized weight
    '''
    sims = weight @ weight.T 
    i, j = torch.triu_indices(weight.shape[0], weight.shape[0], offset=1)

    if not statistics:
        rho = -1/(sims.shape[0]-1)
        loss = ((sims[i,j].reshape(-1)-rho)**2).mean()
    else:
        loss = -sims[i,j].reshape(-1).mean() + sims[i,j].reshape(-1).std() * std_weight
    return loss*etf_weight