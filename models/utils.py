import torch
from torch import distributed as dist

__all__ = ['slerp', 'compute_class_spherical_means', 'calc_class_mean']

# ---------- 기본 유틸 ----------
def normalize(x, dim=-1, eps=1e-8):
    return x / torch.clamp(x.norm(dim=dim, keepdim=True), min=eps)

def geodesic_distance(p, q, eps=1e-7):
    p = normalize(p, dim=-1, eps=eps)
    q = normalize(q, dim=-1, eps=eps)
    dot = (p * q).sum(dim=-1).clamp(min=-1 + eps, max=1 - eps)
    return torch.acos(dot)

# ---------- SLERP ----------
def slerp(p0, p1, t, eps=1e-7):
    """
    p0, p1: (..., D), unit vectors
    t: (..., 1) or broadcastable scalar in [0,1]
    returns: (..., D), unit vector
    """
    p0 = normalize(p0, dim=-1)
    p1 = normalize(p1, dim=-1)

    dot = (p0 * p1).sum(dim=-1, keepdim=True).clamp(-1 + eps, 1 - eps)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)

    # Coefficients for slerp
    coeff0 = torch.sin((1 - t) * omega) / sin_omega
    coeff1 = torch.sin(t * omega) / sin_omega
    out = coeff0 * p0 + coeff1 * p1

    # Small-angle fallback: normalized lerp
    small = (omega < 1e-4)
    out_lerp = normalize((1 - t) * p0 + t * p1, dim=-1)
    return torch.where(small, out_lerp, out)

# ---------- Sphere log / exp ----------
def _any_tangent_unit(p, eps=1e-8):
    """
    Construct a deterministic unit vector orthogonal to p for antipodal fallback.
    """
    # pick index of smallest abs component to avoid near-colinearity
    idx = p.abs().argmin(dim=-1, keepdim=True)  # (...,1)
    e = torch.zeros_like(p)
    e = e.scatter(-1, idx, 1.0)
    u = e - (e * p).sum(dim=-1, keepdim=True) * p
    return normalize(u, dim=-1, eps=eps)

def log_map_sphere(p, q, eps=1e-7):
    """
    p, q: (..., D) unit vectors
    returns v in T_p S^n: (..., D)
    """
    p = normalize(p, dim=-1)
    q = normalize(q, dim=-1)

    dot = (p * q).sum(dim=-1, keepdim=True).clamp(-1 + eps, 1 - eps)
    theta = torch.acos(dot)  # (...,1)

    # direction in tangent space
    # u = (q - dot*p) / sin(theta), with sin^2(theta) = 1 - dot^2
    sin_theta = torch.sqrt(torch.clamp(1 - dot * dot, min=eps))
    u = (q - dot * p) / sin_theta

    # antipodal fallback (direction undefined when theta ~ pi)
    antipodal = (dot < (-1 + 1e-4))
    u_fallback = _any_tangent_unit(p)
    u = torch.where(antipodal, u_fallback, u)

    v = theta * u  # magnitude = theta
    return v

def exp_map_sphere(p, v, eps=1e-7):
    """
    p: (..., D) unit vector, v: (..., D) tangent at p
    returns point on sphere: (..., D)
    """
    p = normalize(p, dim=-1)
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=eps)
    v_dir = v / v_norm
    x = torch.cos(v_norm) * p + torch.sin(v_norm) * v_dir
    return normalize(x, dim=-1)

# ---------- 카처(프레셰) 평균 ----------
def spherical_frechet_mean(X, w=None, max_iter=50, step_size=1.0, tol=1e-6, eps=1e-7):
    """
    X: (N, D) unit vectors
    w: (N,) nonnegative weights (optional, will be normalized)
    returns: (D,) unit vector (Riemannian mean on S^n)
    """
    N, D = X.shape
    Xn = normalize(X, dim=-1)

    if w is None:
        w = torch.ones(N, device=X.device, dtype=X.dtype) / N
    else:
        w = w / (w.sum() + 1e-12)

    # init: normalized Euclidean weighted mean
    m = normalize((Xn * w[:, None]).sum(dim=0, keepdim=False), dim=-1)

    for _ in range(max_iter):
        # compute tangent updates
        m_expand = m.unsqueeze(0).expand_as(Xn)
        v_i = log_map_sphere(m_expand, Xn, eps=eps)  # (N, D)
        v = (w[:, None] * v_i).sum(dim=0)  # (D,)

        v_norm = v.norm()
        if v_norm.item() < tol:
            break

        m = exp_map_sphere(m, step_size * v, eps=eps)

    return normalize(m, dim=-1)

def calc_class_mean(X,y, num_classes):
    '''
    X : (N, D) unit vectors
    y : (N,) labels
    return : (num_classes, D) unit vectors
    '''
    mean = torch.zeros((num_classes, X.shape[1]), device=X.device, dtype=X.dtype)
    for c in range(num_classes):
        mask = (y == c)
        X_c = X[mask]
        mean_c = spherical_frechet_mean(X_c)
        mean[c] = mean_c
    mask = torch.zeros((num_classes), dtype=torch.bool, device=X.device)
    mask[y.unique()] = True
    return mean, mask 

def _extract_batch_from_loader_item(loader_item):
    """
    Support (images, labels) or ([images,...], labels) tuples from datasets using multi-transform.
    Returns images, labels.
    """
    if isinstance(loader_item, (list, tuple)):
        images = loader_item[0]
        labels = loader_item[1]
        # If images is a list of augmented views, pick the first for feature aggregation
        if isinstance(images, (list, tuple)):
            images = images[0]
        return images, labels
    raise ValueError("Loader item must be a (images, labels) tuple.")


@torch.no_grad()
def compute_class_spherical_means(
    loader,
    model,
    aligner=None,
    device=None,
    num_classes=None,
    max_iter=50,
    step_size=1.0,
    tol=1e-6,
    eps=1e-7,
    subset=False
):
    """
    Compute per-class spherical Fréchet means from features produced by model over a DataLoader.

    Args:
        loader: PyTorch DataLoader yielding (images, labels) or ([images,...], labels)
        model: model that supports `model(images, features=True)` returning (features, ...),
               or returns logits; in that case we use the normalized penultimate tensor-like output.
        device: torch.device; if None, infer from model parameters; data moved to this device
        num_classes: optional int; if None, inferred from labels observed in loader
        max_iter, step_size, tol, eps: parameters for spherical_frechet_mean

    Returns:
        Tensor of shape (num_classes, D) with per-class unit feature means on the sphere.
    """
    model_was_training = model.training
    model.eval()

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class_to_features = {}
    feature_dim = None

    for batch in loader:
        images, labels = _extract_batch_from_loader_item(batch)
        images = images.to(device)
        labels = labels.to(device)
        if aligner is not None:
            _,_,ldmk,_,_,_ = aligner(images)
        else:
            ldmk = None
        # Try preferred interface: features=True
        try:
            outputs = model(images, features=True, keypoint=ldmk)
            if isinstance(outputs, (list, tuple)):
                features = outputs[1]
            else:
                features = outputs
        except TypeError:
            # Fallback: use model(images) and pick a reasonable tensor output
            outputs = model(images)
            if isinstance(outputs, (list, tuple)):
                # pick the first tensor-like output with shape (B, D)
                picked = None
                for out in outputs:
                    if isinstance(out, torch.Tensor) and out.dim() == 2 and out.shape[0] == images.shape[0]:
                        picked = out
                        break
                if picked is None:
                    # last resort: take first tensor
                    for out in outputs:
                        if isinstance(out, torch.Tensor):
                            picked = out
                            break
                features = picked
            else:
                features = outputs

        if not isinstance(features, torch.Tensor):
            raise RuntimeError("Model did not return a tensor features/logits; cannot compute means.")

        # Ensure 2D shape (B, D)
        if features.dim() > 2:
            features = features.flatten(1)

        # Normalize to lie on the unit sphere
        features = normalize(features, dim=-1)

        # If running under DDP, gather features and labels across ranks
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            # Gather batch sizes to handle variable last batches
            local_bs = torch.tensor([features.shape[0]], device=features.device, dtype=torch.int64)
            bs_list = [torch.zeros_like(local_bs) for _ in range(world_size)]
            dist.all_gather(bs_list, local_bs)
            max_bs = int(torch.stack(bs_list).max().item())

            # Pad to max_bs
            pad = max_bs - features.shape[0]
            if pad > 0:
                pad_feats = torch.zeros((pad, features.shape[1]), device=features.device, dtype=features.dtype)
                pad_labels = torch.full((pad,), -1, device=labels.device, dtype=labels.dtype)
                features_p = torch.cat([features, pad_feats], dim=0)
                labels_p = torch.cat([labels, pad_labels], dim=0) # padded tensor 를 만드신다. 
            else:
                features_p = features
                labels_p = labels

            gathered_feats = [torch.zeros_like(features_p) for _ in range(world_size)]
            gathered_labels = [torch.zeros_like(labels_p) for _ in range(world_size)]
            dist.all_gather(gathered_feats, features_p)
            dist.all_gather(gathered_labels, labels_p)

            features = torch.cat(gathered_feats, dim=0)
            labels = torch.cat(gathered_labels, dim=0)
            valid = labels != -1
            features = features[valid]
            labels = labels[valid] # remove padded labels 

        if feature_dim is None:
            feature_dim = features.shape[1]

        # Accumulate per class
        for c in labels.unique().tolist():
            mask = (labels == c)
            feats_c = features[mask]
            if feats_c.numel() == 0:
                continue
            if c not in class_to_features:
                class_to_features[c] = [feats_c.detach().cpu()]
            else:
                class_to_features[c].append(feats_c.detach().cpu())
    
    # Determine num_classes if not provided
    if num_classes is None:
        num_classes = (max(class_to_features.keys()) + 1) if class_to_features else 0

    # Compute spherical Fréchet mean per class
    means = []
    for c in range(num_classes):
        if c in class_to_features:
            if subset is not False: 
                temp = []
                for i in range(subset):
                    perm = torch.randperm(len(class_to_features[c]))
                    selected_indices = perm[:64].tolist()  # Convert tensor to list
                    X = torch.cat([class_to_features[c][i] for i in selected_indices], dim=0).to(device)
                    mean_c = spherical_frechet_mean(X, w=None, max_iter=max_iter, step_size=step_size, tol=tol, eps=eps)
                    temp.append(mean_c.detach())
                means.append(torch.stack(temp, dim=0))
            else:
                X = torch.cat(class_to_features[c], dim=0).to(device)
                mean_c = spherical_frechet_mean(X, w=None, max_iter=max_iter, step_size=step_size, tol=tol, eps=eps)
                means.append(mean_c.detach())
        else:
            # If class not observed, use zero vector placeholder with correct dim
            means.append(torch.zeros(feature_dim if feature_dim is not None else 0))


    means = torch.stack(means, dim=0) if len(means) > 0 else torch.empty(0)
        
    if model_was_training:
        model.train()

    return means

def compute_class_mean(features, labels, legacy, alpha):
    '''
    features : (bs, dim)
    labels : (bs,)
    return : (num_classes, dim)
    '''
    if not isinstance(features, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise ValueError('features and labels must be torch.Tensor')
    if legacy is None:
        raise ValueError('legacy class centers must be provided')

    # Ensure devices/dtypes align
    device = legacy.device
    dtype = legacy.dtype
    features = features.to(device=device, dtype=dtype)
    labels = labels.to(device=device)
    alpha = float(alpha)

    # Normalize features onto the unit sphere
    features = normalize(features, dim=-1)

    num_classes, dim = legacy.shape

    # Accumulate sums per class
    sums = torch.zeros((num_classes, dim), device=device, dtype=dtype)
    # scatter_add along class index
    index = labels.view(-1, 1).expand(-1, dim)
    sums = sums.scatter_add(0, index, features)

    # Counts per class
    counts = torch.bincount(labels, minlength=num_classes).to(device)

    # Compute batch means where available and normalize to unit sphere
    counts_safe = counts.clamp_min(1).unsqueeze(-1)
    batch_means = sums / counts_safe
    batch_means = normalize(batch_means, dim=-1)

    # Mask for classes present in this batch
    present = counts > 0

    # Momentum update on present classes; keep legacy for absent ones
    updated = legacy.clone()
    if present.any():
        # Spherical interpolation (differentiable) between legacy and batch means
        updated[present] = slerp(legacy[present], batch_means[present], t=alpha)
    
    return updated

# ---------- 예시 검증 ----------
if __name__ == "__main__":
    torch.manual_seed(0)
    N, D = 8, 5
    X = normalize(torch.randn(N, D), dim=-1)
    w = torch.softmax(torch.randn(N), dim=0)

    # mean computation
    m = spherical_frechet_mean(X, w=w, max_iter=60, step_size=1.0)

