import torch 
from torch.nn import functional as F

__all__ = ['EKCL']

class EKCL : 
    def __init__(self, args, fetcher, temperature=0.1,):
        '''
        args should have num_workers
        '''
        self.tau = temperature
        self.fetcher = fetcher
        self.args = args

    @torch.no_grad()
    def fetch_positive(self,y,k):
        '''
        returns bs, k, 3, h, w
        '''

        return self.fetcher.sample_pairs(y,k,num_workers=self.args.num_workers)


    def process_positives(self, k_pairs, model, aligner=None):
        '''
        k_pairs: bs, k, 3, h, w
        returns bs, k, d
        '''
        bs,k,_,h,w = k_pairs.shape
        imgs = k_pairs.reshape(-1,3,h,w)
        if aligner is not None:
            _,_,kp,_,_,_ = aligner(k_pairs)
        else:
            kp = None
        logit, feature, centers = model(imgs, keypoint=kp, features=True)
        return feature.reshape(bs,k,-1)


    def compute_sims(self, q, k, y, ):

        counts = y.bincount().to(q.device) # (C)
        k_sims = q@k.T/self.tau # (bs+C+C, bs+C+C) 
        mask = y.unsqueeze(1) == y.unsqueeze(0) # (bs+C+C, bs+C+C)
        num = torch.logsumexp(k_sims * mask.masked_fill(mask==0, float('-inf')),dim=1)
        den = torch.log(torch.sum(torch.exp(k_sims * mask.masked_fill(mask==1, float('-inf')))/counts[y].reshape(1,q.shape[0]), dim=-1))

        loss = -(num - den)/counts[y].reshape(1,q.shape[0])

        return loss.mean()
        
    def __call__(self, q, y, weight, centers ,model, aligner=None, requires_grad=False, k=None, ):
        y = torch.concat([y, torch.arange(weight.shape[0]).repeat(2).to(q.device)])
        q = torch.concat([q,weight,centers],dim=0)
        if k is None :
            k_pairs, _ = self.fetch_positive(y, self.args.kcl_k)
        func = torch.autograd.enable_grad if requires_grad else torch.inference_mode
        with func():
            k_features = self.process_positives(k_pairs, model, aligner)
        k_centers = spherical_frechet_mean(k_features)
        loss = self.compute_sims(q, k=k_centers, y=y)
        return loss, k


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