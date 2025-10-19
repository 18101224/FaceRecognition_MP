from tkinter import Y
import torch 
from torch.nn import functional as F
import math
import itertools
from typing import Optional, Tuple


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
        norm = counts[y[:bs]].reshape(1,-1) if self.args.balanced_cl else 1 

        den = k_sims.masked_fill(positive_mask, float('-inf')) 
        den = torch.log(torch.sum(torch.exp(den)/ norm, dim=-1) ) # bs only negatives
        #den = torch.log(torch.sum(torch.exp(k_sims.masked_fill(positive_mask, float('-inf')))/ norm, dim=-1) ) # bs only negatives
        num = torch.logsumexp(k_sims.masked_fill(negative_mask,float('-inf')),dim=1) # bs ]
        
        norm = counts[y].reshape(-1) if self.args.balanced_cl else 1 
        loss = -(num-den) / norm 

        return loss.mean()

    def gen_clusters(self,X,y,n:list,k:list):
        clusters = []
        labels = []
        for i in range(len(self.args.num_clusters)):
            indices, valid_mask, repeated_mask = build_unique_cluster_indices(y, n=n[i],k=k[i], num_classes=self.args.num_classes)
            indices = indices[valid_mask] # n_c, k, n ( only valid classes ) 
            cluster_embeddings = X[indices] # shaped as (num_classes, k, n, dim)
            cluster_means = spherical_frechet_mean_groups(cluster_embeddings, max_iters=20, tol=1e-6, step=1.0, eps=1e-7) # shaped as (num_classes, k, dim)
            temp_y = torch.arange(self.args.num_classes,device=X.device).unsqueeze(1).expand(-1,k[i])[valid_mask].reshape(-1) # n_c*k
            clusters.append(cluster_means.reshape(-1,cluster_means.shape[-1]))
            labels.append(temp_y)
        clusters = torch.cat(clusters, dim=0) if len(clusters) > 1 else clusters[0]
        labels = torch.cat(labels, dim=0) if len(labels) > 1 else labels[0]
        return clusters, labels

    def compute_self_sims(self,features, labels):
        '''
        features : bs, dim 
        labels : bs 
        '''
        bs = features.shape[0]
    
        sims = features @ features.T/self.tau # bs, bs 

        positive_mask = labels.unsqueeze(1) == labels.unsqueeze(0) # bs, bs including self
        negative_mask = ~positive_mask # bs,bs excluding self 
        positive_mask = positive_mask& ~(torch.eye(bs, device=features.device, dtype=torch.bool)) # bs,bs excluding self 

        norm_factor = labels.bincount().to(features.device)[labels].reshape(bs) if self.args.balanced_cl else torch.ones(bs, device=features.device) 
        num = torch.logsumexp(sims.masked_fill(negative_mask, float('-inf')),dim=1)
        den_logits = torch.exp(sims.masked_fill(positive_mask, float('-inf'))) / norm_factor.reshape(1,-1)
        den = torch.log(torch.sum(den_logits,dim=-1))
        loss = -(num-den) / norm_factor.reshape(-1,1)
        return loss.mean()


    def forward(self, logits, features, y, weight, centers ,model, aligner=None, requires_grad=False, positive_pair=None, **kwargs):

        ce_loss =torch.nn.functional.cross_entropy(logits, y)

        if self.args.batch_pairs_only : 
            clusters, labels = self.gen_clusters(features, y, self.args.sizes_clusters, self.args.num_clusters)
            if weight is not None : 
                y = torch.cat([labels, torch.arange(weight.shape[0], device=y.device)],dim=0)
                features = torch.cat([clusters, weight], dim=0)
                cl_loss = self.compute_self_sims(features, y)
            else:
                cl_loss = self.compute_self_sims(clusters, labels)
            return ce_loss, cl_loss, None 
            
        else:
            if positive_pair is None :
                positive_pair = self.fetch_positive(y, self.args.kcl_k) # bs, k, 3, h, w
            
            if self.args.k_meeting is not None : 
                query = positive_pair[:,:self.qk-1,:,:,:]
                if query.ndim ==4 :
                    query = query.unsqueeze(1)
                key = positive_pair[:,self.qk-1:,:,:,:]
                if key.ndim ==4 :
                    key = key.unsqueeze(1)
                features, k_features, query_dist, key_dist = self.process_k_meeting(features=features, y=y, weight=weight, query=query, key=key, centers=centers, model=model, aligner=aligner, requires_grad=requires_grad)
            else:
                func = torch.autograd.enable_grad if requires_grad else torch.no_grad
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
            if self.args.k_meeting is not None and self.args.k_meeting_dist : 
                cl_loss += self.args.k_meeting_dist*(query_dist+key_dist)
            return ce_loss, cl_loss,  positive_pair

    def process_k_meeting(self, features, y, weight, query, key, centers, model, aligner=None, requires_grad=False):
        query_features = self.process_positives(query, model, aligner) # bs, qk-1, dim 
        func = torch.autograd.enable_grad if requires_grad else torch.no_grad 
        with func() : 
            key_features = self.process_positives(key, model, aligner) # bs, kk, dim 
            key_means = spherical_frechet_mean(key_features)
            key_dist = calc_cluster_distance(key_features, key_means) * int(bool(requires_grad))
        query_means = spherical_frechet_mean(torch.cat([features.unsqueeze(1),query_features], dim=1))
        query_dist = calc_cluster_distance(query_features, query_means)
        return query_means, key_means, query_dist, key_dist


def calc_cluster_distance(features, means):
    '''
    features : bs, k, dim 
    means : bs, dim
    '''
    sims = (features @ means.unsqueeze(1).transpose(-1,-2)).squeeze(-1) # bs, k
    sims = torch.clamp(sims, min=-1+1e-7, max=1-1e-7)
    return torch.sum(torch.arccos(sims)**2, dim=-1, keepdim=True).reshape(-1).mean()



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


def spherical_frechet_mean_groups(
    X_groups: torch.Tensor,   # (NC, k, n, dim)
    max_iters: int = 20,
    tol: float = 1e-6,
    step: float = 1.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Compute spherical Fréchet means for each (NC, k, n, dim) group.
    Returns:
      mu: (NC, k, dim)
    """
    assert X_groups.dim() == 4, "Expected (NC, k, n, dim)"
    NC, K, N, D = X_groups.shape
    X = X_groups.reshape(NC * K, N, D)  # (B, N, D), B = NC*K

    # Project to sphere
    Xn = F.normalize(X, dim=-1)  # (B, N, D)

    # Initialization: normalized Euclidean mean
    mu = F.normalize(Xn.mean(dim=1), dim=-1)  # (B, D)

    def safe_acos(x: torch.Tensor) -> torch.Tensor:
        return torch.acos(torch.clamp(x, -1.0 + 1e-7, 1.0 - 1e-7))

    for _ in range(max_iters):
        # angles theta_i = arccos(<mu, x_i>)
        dots = torch.sum(mu.unsqueeze(1) * Xn, dim=-1)             # (B, N)
        theta = safe_acos(dots)                                     # (B, N)

        # avoid division by zero
        sin_theta = torch.sin(theta).clamp_min(eps)                 # (B, N)

        # log_mu(x_i) = (theta/sin theta) * (x_i - cos(theta)*mu)
        diff = Xn - dots.unsqueeze(-1) * mu.unsqueeze(1)            # (B, N, D)
        coeff = (theta / sin_theta).unsqueeze(-1)                   # (B, N, 1)
        log_vecs = coeff * diff                                     # (B, N, D)

        # mean tangent vector
        v = log_vecs.mean(dim=1)                                    # (B, D)

        # convergence check
        v_norm = v.norm(dim=-1, keepdim=True).clamp_min(eps)        # (B, 1)
        if torch.max(v_norm).item() < tol:
            break

        # exp map step on sphere
        v_step = step * v
        v_step_norm = v_step.norm(dim=-1, keepdim=True).clamp_min(eps)  # (B, 1)
        mu = (torch.cos(v_step_norm) * mu
              + torch.sin(v_step_norm) * (v_step / v_step_norm))        # (B, D)
        mu = F.normalize(mu, dim=-1)

    mu = mu.reshape(NC, K, D)  # (NC, k, dim)
    return mu


@torch.no_grad()
def build_unique_cluster_indices(
    y: torch.Tensor,     # (bs,)
    n: int,              # subset size
    k: int,              # subsets per class
    num_classes: Optional[int] = None,
    *,
    max_exact_enum: int = 20000,     # 전수조합 상한
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Output:
      idx_tensor: (NC, k, n) long, 각 클래스별 k개의 길이 n 조합(배치 인덱스)
      valid_mask: (NC,) bool, 적어도 1개 조합 생성 성공 여부
      repeated_mask: (NC,) bool, 고유 조합 부족으로 반복 채움 발생 여부

    예외 정책:
      - L == 0: invalid (모두 placeholder), valid_mask=False
      - 0 < L < n: invalid (조합 불가), valid_mask=False
      - L >= n, C(L,n) < k: 가능한 모든 고유 조합 사용 후 반복 채움, repeated_mask=True
      - L >= n, C(L,n) >= k: 중복 없이 k개 조합
    """
    assert y.dim() == 1, "y must be (bs,)"
    bs = y.numel()
    if device is None:
        device = y.device
    y = y.to(device)

    if num_classes is None:
        # NC 추정: y 내 최대 라벨 + 1
        NC = int(y.max().item()) + 1 if bs > 0 else 0
    else:
        NC = int(num_classes)

    idx_by_class = [[] for _ in range(NC)]
    for i in range(bs):
        label = int(y[i].item())
        if 0 <= label < NC:
            idx_by_class[label].append(i)

    idx_tensor = torch.zeros((NC, k, n), dtype=torch.long, device=device)
    valid_mask = torch.zeros((NC,), dtype=torch.bool, device=device)
    repeated_mask = torch.zeros((NC,), dtype=torch.bool, device=device)

    rng = generator if generator is not None else torch.Generator(device='cpu')

    for c in range(NC):
        indices = idx_by_class[c]
        L = len(indices)
        if L == 0:
            # invalid 그대로 유지
            continue

        if L < n:
            # 조합 불가
            continue

        base = torch.tensor(indices, dtype=torch.long, device=device)
        comb_count = math.comb(L, n)

        if comb_count <= max_exact_enum:
            # 전수조합 후 무작위 선택
            # itertools.combinations는 오름차순 인덱스 튜플을 반환하여 조합 유니크 보장
            all_combos = list(itertools.combinations(range(L), n))
            m = len(all_combos)
            if m >= k:
                # k개 무작위 선택(중복 없음)
                perm = torch.randperm(m, generator=rng)[:k].tolist()
                chosen = [all_combos[p] for p in perm]
                sel = torch.tensor(chosen, dtype=torch.long, device=device)
                idx_tensor[c] = base[sel]
                valid_mask[c] = True
            else:
                # 가능한 모든 고유 조합을 넣고, 나머지는 반복 채움
                sel = torch.tensor(all_combos, dtype=torch.long, device=device)
                if sel.numel() > 0:
                    cnt = sel.shape[0]
                    idx_tensor[c, :cnt] = base[sel]
                    need = k - cnt
                    if need > 0:
                        rep_idx = torch.randint(low=0, high=cnt, size=(need,),
                                                generator=rng, device='cpu').tolist()
                        rep_sel = sel[torch.tensor(rep_idx, dtype=torch.long, device=device)]
                        idx_tensor[c, cnt:] = base[rep_sel]
                        repeated_mask[c] = True
                    valid_mask[c] = True
                else:
                    # n==0 같은 특수 케이스가 아니라면 여기로 오지 않음
                    # 그래도 안전하게 invalid 유지
                    pass
        else:
            # 확률적 유니크 샘플링: 유니크 조합 set 수집
            # 각 조합은 '오름차순 정렬된 길이 n 인덱스'로 표준화
            chosen_set = set()
            chosen_list = []
            attempts = 0
            max_attempts = max(200, k * 50)  # 재시도 한도

            while len(chosen_list) < k and attempts < max_attempts:
                attempts += 1
                # L개 중 n개 무작위 샘플링(서로 다른 원소): randperm 사용
                perm = torch.randperm(L, generator=rng)[:n].tolist()
                perm.sort()
                key = tuple(perm)
                if key not in chosen_set:
                    chosen_set.add(key)
                    chosen_list.append(key)

            if len(chosen_list) == 0:
                # 드물지만 실패 시 invalid 유지
                continue

            if len(chosen_list) < k:
                # 고유 조합 부족 → 반복 채움
                m = len(chosen_list)
                need = k - m
                extra = torch.randint(low=0, high=m, size=(need,),
                                      generator=rng, device='cpu').tolist()
                for e in extra:
                    chosen_list.append(chosen_list[e])
                repeated_mask[c] = True

            sel = torch.tensor(chosen_list, dtype=torch.long, device=device)
            idx_tensor[c] = base[sel]
            valid_mask[c] = True

    return idx_tensor, valid_mask, repeated_mask
