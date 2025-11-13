import torch 
import numpy as np 

class EAC: 
    def __init__(self, args, class_counts=None):
        self.args = args 
        self.balanced = False
        for loss in args.loss.split('_'):
            if loss == 'BEAC' :
                self.balanced = True 
        beta = 0.9999
        self.weight = ((1-beta) / (1-(beta)**class_counts)).reshape(-1)
        
    def __call__(self, logits, features, y, weight, **kwargs):
        '''
        features : 2bs, L, dim 
        weight : n_c, dim, 
        y : bs 
        logits : 2bs, n_c 
        '''
        ce_loss = torch.nn.functional.cross_entropy(logits, y.repeat(2))
        bs = y.shape[0]
        # bs,n_c, L 
        sims = (features.unsqueeze(1) @ weight.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
        if self.balanced : 
            sims = sims * self.weight.view(1,-1,1)
        sims_o, sims_f = torch.split(sims, [bs,bs], dim=0)
        h,w = int(np.sqrt(sims_o.shape[-1])), int(np.sqrt(sims_o.shape[-1]))
        sims_o = sims_o.reshape(*sims_o.shape[:-1],h,w)
        sims_f = sims_f.reshape(*sims_f.shape[:-1],h,w)
        sims_ff = torch.flip(sims_f, dims=[-1])
        
        cl_loss = ((sims_o - sims_ff)**2).sum() / (h*w*logits.shape[-1]*bs)

        return ce_loss, cl_loss, None