import torch

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

