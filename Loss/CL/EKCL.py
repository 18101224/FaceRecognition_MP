import torch 
from torch.nn import functional as F
from torch import dist 
import sys;sys.path.append('../..')
from utils import gather_tensor


__all__ = ['EKCL']

def ddp_ready():
    return dist.is_available() and dist.is_initialized()

class EKCL : 
    def __init__(self, args, fetcher, temperature=0.1,):
        '''
        args should have num_workers
        '''
        self.tau = temperature
        self.fetcher = fetcher
        self.args = args
        if self.args.k_meeting is not None : 
            self.qk, self.kk = list(map(int,list(self.args.k_meeting.split('_'))))
            
    @torch.no_grad()
    def fetch_positive(self,y,k):
        '''
        returns bs, k, 3, h, w
        '''

        return self.fetcher.sample_pairs(y,k)


    def process_positives(self, k_pairs, model, aligner=None):
        '''
        k_pairs: bs, k, 3, h, w
        returns bs, k, d
        '''
        bs,k,_,h,w = k_pairs.shape
        imgs = k_pairs.reshape(-1,3,h,w)
        with torch.inference_mode():
            if aligner is not None:
                _,_,kp,_,_,_ = aligner(imgs)
            else:
                kp = None
        logit, feature, centers = model(imgs, keypoint=kp, features=True)
        return feature.reshape(bs,k,-1)


    def compute_sims(self, q, k, y, ):
        '''
        q : bs, dim
        k : bs+C+C, dim 
        y : bs+C+C
        '''
        bs = q.shape[0]

        counts = y.bincount().to(q.device) # (C)
        k_sims = q@k.T/self.tau # (bs, bs+C+C) 
        positive_mask = y[:bs].unsqueeze(1) == y.unsqueeze(0) # (bs, bs+C+C)

        negative_mask = ~positive_mask # (bs, bs+C+C)
        # k_sims -> bs, bs+C+C, counts[y].reshape(1,-1) -> (1, bs+C+C), 
        norm = counts[y].reshape(1,-1) if self.args.balanced_cl else 1 
        den = torch.log(torch.sum(torch.exp(k_sims.masked_fill(positive_mask, float('-inf')))/ norm, dim=-1) ) # bs only negatives
        num = torch.logsumexp(k_sims.masked_fill(negative_mask,float('-inf')),dim=1) # bs ]
        
        norm = counts[y[:bs]].reshape(bs) if self.args.balanced_cl else 1 
        loss = -(num-den) / norm 

        return loss.mean()
        
    def __call__(self, features, y, weight, centers ,model, aligner=None, requires_grad=False, k_imgs=None):
        if k_imgs is None :
            k_imgs, _ = self.fetch_positive(y, self.args.kcl_k)
        func = torch.autograd.enable_grad if requires_grad else torch.no_grad
        with func():
            k_features = self.process_positives(k_imgs, model, aligner) # bs, k, dim 
        q_ = gather_tensor(features) if self.args.world_size>1 else features # bs, dim
        y_ = gather_tensor(y) if self.args.world_size>1 else y # bs
        k_ = gather_tensor(spherical_frechet_mean(k_features)) if self.args.world_size>1 else spherical_frechet_mean(k_features) # bs+C+C, dim 
        # bs, dim 

        y_ = torch.concat([y_, torch.arange(weight.shape[0]).repeat(int(bool(centers is not None))+int(bool(weight is not None))).to(q.device)]) if centers is not None and weight is not None else y_ # bs+C+C 
        k_ = torch.concat([k_, weight], dim=0 ) if weight is not None else k_ # bs+C+C, dim 
        k_ = torch.concat([k_, centers], dim=0) if centers is not None else k_ # bs+C+C, dim 
        loss = self.compute_sims(q_, k_, y_)
        return loss, k_imgs



def spherical_frechet_mean(X, max_iters=20, tol=1e-6, step=1.0, eps=1e-7):
    """
    X: (bs, K, dim) unit vectors (or arbitrary -> will be normalized)
    Returns: mu (bs, dim) unit vectors as spherical Fréchet means
    """
    bs, K, d = X.shape
    # Project to sphere
    Xn = F.normalize(X, dim=-1)

    # Initialization: normalized Euclidean mean
    mu = F.normalize(Xn.mean(dim=1), dim=-1)  # (bs, d)

    def safe_acos(x):
        return torch.acos(torch.clamp(x, -1.0 + 1e-7, 1.0 - 1e-7))

    for _ in range(max_iters):
        # Compute angles theta_i = arccos(<mu, x_i>)
        dots = (mu.unsqueeze(1) * Xn).sum(-1)               # (bs, K)
        theta = safe_acos(dots)                              # (bs, K)

        # Handle near-zero angle to avoid division by zero
        sin_theta = torch.sin(theta).clamp_min(eps)          # (bs, K)

        # Log map at mu: log_mu(x_i) = (theta / sin(theta)) * (x_i - cos(theta)*mu)
        # Parallel transport-like difference
        # v_i direction in tangent space
        diff = Xn - dots.unsqueeze(-1) * mu.unsqueeze(1)     # (bs, K, d)
        coeff = (theta / sin_theta).unsqueeze(-1)            # (bs, K, 1)
        log_vecs = coeff * diff                              # (bs, K, d)

        # Mean tangent vector
        v = log_vecs.mean(dim=1)                             # (bs, d)

        # Convergence check: norm of update
        v_norm = v.norm(dim=-1, keepdim=True).clamp_min(eps) # (bs, 1)
        max_change = v_norm.max().item()
        if max_change < tol:
            break

        # Retraction / Exp map: exp_mu(v) = cos(|v|) mu + sin(|v|) * v/|v|
        # Use a step size if desired
        v_step = step * v
        v_step_norm = v_step.norm(dim=-1, keepdim=True).clamp_min(eps)
        mu = (torch.cos(v_step_norm) * mu
              + torch.sin(v_step_norm) * (v_step / v_step_norm))
        mu = F.normalize(mu, dim=-1)

    return mu