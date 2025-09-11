import torch
import torch.nn.functional as F
from collections import defaultdict

__all__ = ['calculate_class_centers', 'slerp_ema']

@torch.no_grad()
def calculate_class_centers(train_loader, model, aligner,  num_classes, device=None, iters=5, step=1.0,):
    """
    각 클래스별 spherical Karcher mean을 계산해 [C, D] 텐서로 반환합니다.
    - train_loader: (images, labels) 배치를 반환
    - model: 특징 추출 모델 (eval 모드 권장)
    - num_classes: 클래스 개수
    - device: 장치 (None이면 모델 파라미터 장치 사용)
    - iters, step: Riemannian GD 반복 횟수와 스텝
    - feature_fn: 사용자 정의 특징 추출 함수. None이면 model(images)를 사용
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    # 클래스별 특징 누적
    buckets = defaultdict(list)
    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 특징 추출
        if aligner is not None:
            _,_,keypoint,_,_,_ = aligner(images)
        else:
            keypoint = None

        feats,_,_ = model(images, keypoint=keypoint, features=True)
        

        # 단위구로 정규화
        feats = F.normalize(feats, dim=1)

        # 클래스별로 분배
        for k in labels.unique().tolist():
            mask = (labels == k)
            if mask.any():
                buckets[k].append(feats[mask].detach())  # 로깅/초기화: 그래프 필요 없으므로 분리

    # 클래스별 카처 평균 계산
    centers = []
    dim = None
    for k in range(num_classes):
        if len(buckets[k]) == 0:
            # 비어 있으면 임시로 표준 기준축을 반환(혹은 직전 값/무작위 초기화로 대체)
            if dim is None:
                # 다른 버킷에서 차원을 유추
                for v in buckets.values():
                    if len(v) > 0:
                        dim = v.shape[6]
                        break
            if dim is None:
                raise RuntimeError("특징 차원을 유추할 수 없습니다. 로더/모델을 확인하세요.")
            centers.append(F.normalize(torch.randn(dim, device=device), dim=0))
            continue

        Xk = torch.cat(buckets[k], dim=0)  # [N_k, D]
        mu_k = spherical_karcher_mean(Xk, iters=iters, step=step)  # [D]
        centers.append(mu_k)

    centers = torch.stack(centers, dim=0)  # [C, D]
    return centers


def slerp(u: torch.Tensor, v: torch.Tensor, t: torch.Tensor, eps: float = 1e-6):
    # u, v: [..., D], assumed near unit; t: [..., 1] in [0,1]
    u = F.normalize(u, dim=-1)
    v = F.normalize(v, dim=-1)
    dot = (u * v).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    # shortest arc: flip v if dot < 0
    v = torch.where(dot < 0, -v, v)
    dot = torch.where(dot < 0, -dot, dot)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    small = sin_theta.abs() < eps
    w1 = torch.sin((1 - t) * theta) / (sin_theta + eps)
    w2 = torch.sin(t * theta) / (sin_theta + eps)
    s = w1 * u + w2 * v
    lerp = F.normalize((1 - t) * u + t * v, dim=-1)
    out = torch.where(small, lerp, s)
    return F.normalize(out, dim=-1)

def log_map_sphere(mu: torch.Tensor, x: torch.Tensor, eps: float = 1e-6):
    # mu, x: [..., D] unit vectors, returns tangent vector at mu
    mu = F.normalize(mu, dim=-1)
    x  = F.normalize(x,  dim=-1)
    dot = (mu * x).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    # component orthogonal to mu
    v = x - dot * mu
    v = v / (sin_theta + eps) * theta
    # when theta ~ 0, use first-order approximation
    near = theta.abs() < 1e-4
    v_approx = x - mu
    return torch.where(near, v_approx, v)

def exp_map_sphere(mu: torch.Tensor, v: torch.Tensor, eps: float = 1e-6):
    # mu: [..., D] unit, v: tangent vector at mu
    mu = F.normalize(mu, dim=-1)
    v_norm = v.norm(dim=-1, keepdim=True).clamp_min(eps)
    out = torch.cos(v_norm) * mu + torch.sin(v_norm) * (v / v_norm)
    # if v ~ 0, stay at mu
    small = v_norm < 1e-4
    out = torch.where(small, mu, out)
    return F.normalize(out, dim=-1)

def spherical_karcher_mean(X: torch.Tensor, w: torch.Tensor = None, iters: int = 5, step: float = 1.0):
    """
    X: [N, D] unit vectors (will be normalized); w: [N] nonneg, sum to 1 (optional)
    Returns mu: [D] unit Karcher mean via Riemannian gradient descent.
    """
    X = F.normalize(X, dim=-1)
    N, D = X.shape
    if w is None:
        w = torch.full((N,), 1.0 / N, device=X.device, dtype=X.dtype)
    w = w / (w.sum() + 1e-12)

    # init: normalize Euclidean mean
    mu = F.normalize((w[:, None] * X).sum(dim=0, keepdim=False), dim=-1)

    for _ in range(iters):
        # gradient in tangent: sum_i w_i * Log_mu(x_i)
        V = log_map_sphere(mu[None, :].expand_as(X), X)          # [N, D]
        g = (w[:, None] * V).sum(dim=0, keepdim=False)           # [D]
        # update on sphere
        mu = exp_map_sphere(mu, step * g)
    return mu  # [D], unit

def slerp_ema(center: torch.Tensor, batch_mean: torch.Tensor, momentum: float):
    """
    center, batch_mean: [D] unit vectors; momentum in [0,1)
    """
    t = torch.tensor([1.0 - momentum], device=center.device, dtype=center.dtype).view(1, 1)
    return slerp(center[None, :], batch_mean[None, :], t)

