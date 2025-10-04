import torch 
from torch.nn import functional as F


__all__ = ['EKCL']

class EKCL(torch.nn.Module) : 
    def __init__(self, args, fetcher, temperature=0.1,):
        super().__init__()
        '''
        args should have num_workers
        '''
        self.tau = temperature
        self.fetcher = fetcher
        self.args = args
        if self.args.k_meeting is not None : 
            self.qk, self.kk = list(map(int,list(self.args.k_meeting.split('_'))))
            

    def fetch_positive(self,y,k):
        '''
        returns bs, k, 3, h, w
        '''

        return self.fetcher.sample_pairs(y,k) if self.args.world_size==1 else self.fetcher.sample_pairs_dist_gathered(y,k)


    def process_positives(self, k_pairs, model, aligner=None):
        '''
        k_pairs: bs, k, 3, h, w
        returns bs, k, d
        '''
        bs,k,_,h,w = k_pairs.shape
        imgs = k_pairs.reshape(-1,3,h,w)
        with torch.no_grad():
            if aligner is not None:
                _,_,kp,_,_,_ = aligner(imgs)
            else:
                kp = None
        _, feature, _ = model(imgs, keypoint=kp, features=True)
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

    def compute_sims_k_meeting(self,q,k,y):
        '''
        q : bs+C+C, dim
        k : bs, dim 
        y : bs+C+C
        '''

        bs = k.shape[0]

        counts = y.bincount().to(q.device) # (C)

        k_sims = q@k.T/self.tau # (bs+C+C, bs) 


        positive_mask = y.unsqueeze(1) == y[:bs].unsqueeze(0) # (bs+C+C, bs)

        negative_mask = ~positive_mask # (bs+C+C, bs)
        # k_sims -> bs, bs+C+C, counts[y].reshape(1,-1) -> (1, bs+C+C), 
        norm = counts[y].reshape(1,-1) if self.args.balanced_cl else 1 

        den = torch.log(torch.sum(torch.exp(k_sims.masked_fill(positive_mask, float('-inf')))/ norm, dim=-1) ) # bs only negatives
        num = torch.logsumexp(k_sims.masked_fill(negative_mask,float('-inf')),dim=1) # bs ]
        
        norm = counts[y[:bs]].reshape(bs) if self.args.balanced_cl else 1 
        loss = -(num-den) / norm 

        return loss.mean()
        
    def forward(self, logits, features, y, weight, centers ,model, aligner=None, requires_grad=False, positive_pair=None, **kwargs):

        ce_loss =torch.nn.functional.cross_entropy(logits, y)

        if positive_pair is None :
            positive_pair = self.fetch_positive(y, self.args.kcl_k) # bs, k, 3, h, w
        
        if self.args.k_meeting is not None : 
            query = positive_pair[:,:self.qk-1,:,:,:]
            if query.ndim ==4 :
                query = query.unsqueeze(1)
            key = positive_pair[:,self.qk-1:,:,:,:]
            if key.ndim ==4 :
                key = key.unsqueeze(1)
            features, k_features = self.process_k_meeting(features=features, y=y, weight=weight, query=query, key=key, centers=centers, model=model, aligner=aligner, requires_grad=requires_grad)
        else:
            func = torch.autograd.enable_grad if requires_grad else torch.no_grad
            model = model.module if self.args.world_size > 1 else model
            with func() : 
                k_features = self.process_positives(positive_pair, model, aligner) # bs, k, dim 
            k_features = spherical_frechet_mean(k_features)

        # bs, dim 
        if centers is not None or weight is not None :
            y = torch.concat([y, torch.arange(weight.shape[0]).repeat(int(bool(centers is not None))+int(bool(weight is not None))).to(y.device)])
            to_concat = [features if self.args.k_meeting is not None else k_features]
            if weight is not None : 
                to_concat.append(weight)
            if centers is not None : 
                to_concat.append(centers)
            if self.args.k_meeting is not None :
                features = torch.concat(to_concat, dim=0)
            else:
                k_features = torch.concat(to_concat, dim=0)

        cl_loss = self.compute_sims(features, k_features, y) if self.args.k_meeting is None else self.compute_sims_k_meeting(features, k_features, y)

        return ce_loss, cl_loss,  positive_pair

    def process_k_meeting(self, features, y, weight, query, key, centers, model, aligner=None, requires_grad=False):
        query_features = self.process_positives(query, model, aligner) # bs, qk-1, dim 
        func = torch.autograd.enable_grad if requires_grad else torch.no_grad 
        with func() : 
            key_features = self.process_positives(key, model, aligner) # bs, kk, dim 
        query_features = spherical_frechet_mean(torch.cat([features.unsqueeze(1),query_features], dim=1))
        key_features = spherical_frechet_mean(key_features)
        return query_features, key_features


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