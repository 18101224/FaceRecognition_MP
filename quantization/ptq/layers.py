"""
Quantization primitives:
  - round_ste          : rounding with Straight-Through Estimator
  - AdaRoundQuantizer  : per-channel weight quantizer (ICML 2020)
  - LSQActivationQuantizer : signed/unsigned activation quantizer (ICLR 2020)
  - QuantConv2d        : Conv2d with AdaRound weight quantization
  - QuantLinear        : Linear with AdaRound weight quantization

PReLU note:
  PReLU output can be negative, so activation quantizers default to *signed* mode
  (range [−2^(b−1), 2^(b−1)−1]) rather than the unsigned mode assumed in BRECQ's
  original ReLU-based code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

_MAX_PERCENTILE_SAMPLES = 1_000_000


# ---------------------------------------------------------------------------
# Straight-Through Estimator helpers
# ---------------------------------------------------------------------------

def round_ste(x: torch.Tensor) -> torch.Tensor:
    """Round with Straight-Through Estimator: gradient passes unchanged."""
    return (x.round() - x).detach() + x


def clamp_ste(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Clamp with STE: gradient is 1 everywhere (pass-through)."""
    return (x.clamp(lo, hi) - x).detach() + x


# ---------------------------------------------------------------------------
# AdaRound weight quantizer
# ---------------------------------------------------------------------------

class AdaRoundQuantizer(nn.Module):
    """Per-output-channel AdaRound weight quantizer.

    After calling freeze(), v is committed to {0,1} and the module uses
    hard rounding (no longer differentiable, but deterministic).

    Args:
        weight  : the Conv/Linear weight tensor (used only for initialising v and scale)
        n_bits  : quantisation bit-width (default 4)
        sym     : symmetric quantisation (default True)
    """

    def __init__(self, weight: torch.Tensor, n_bits: int = 4, sym: bool = True):
        super().__init__()
        self.n_bits = n_bits
        self.sym = sym
        self.Qn = -(2 ** (n_bits - 1))
        self.Qp =  (2 ** (n_bits - 1)) - 1

        # ---- fixed per-channel scale ----
        scale = self._init_scale(weight)
        self.register_buffer("scale", scale)   # not a learnable parameter

        # ---- soft rounding variable v ----
        # We want sigmoid(v) ≈ (w/s − floor(w/s)), i.e. the fractional part.
        with torch.no_grad():
            w_norm = weight.detach() / self.scale
            frac   = w_norm - w_norm.floor()            # in [0, 1)
            # Initialise v so that sigmoid(v) = frac (clamped away from 0/1).
            frac_c = frac.clamp(1e-3, 1 - 1e-3)
            v_init = torch.log(frac_c / (1.0 - frac_c))
        self.v = nn.Parameter(v_init)

        self._frozen = False

    # ------------------------------------------------------------------
    def _init_scale(self, weight: torch.Tensor) -> torch.Tensor:
        """Symmetric per-output-channel min-max scale."""
        w = weight.detach()
        is_conv = (w.dim() == 4)
        w_flat  = w.view(w.size(0), -1)
        max_val = w_flat.abs().max(dim=1)[0]          # [out_ch]
        scale   = (max_val / self.Qp).clamp(min=1e-8)
        if is_conv:
            return scale.view(-1, 1, 1, 1)
        else:
            return scale.view(-1, 1)

    # ------------------------------------------------------------------
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        """Return quantised weight (soft during optimisation, hard after freeze)."""
        if self._frozen:
            # v has been committed to {0, 1}
            w_int = (weight / self.scale).floor() + self.v.detach()
        else:
            # Soft: differentiable via sigmoid
            w_int = (weight / self.scale).floor() + torch.sigmoid(self.v)

        w_q = w_int.clamp(self.Qn, self.Qp) * self.scale
        return w_q

    def freeze(self) -> None:
        """Commit v to hard {0, 1}.  Call once after optimisation."""
        with torch.no_grad():
            # Loading a saved frozen checkpoint restores v as exact {0,1}.
            # In that case, re-applying sigmoid-thresholding would map 0 -> 1
            # because sigmoid(0) == 0.5. Preserve already-frozen states.
            is_binary = torch.logical_or(self.v.data == 0, self.v.data == 1).all()
            if not is_binary:
                self.v.data = (torch.sigmoid(self.v) >= 0.5).float()
        self._frozen = True

    def regularization(self, beta: float) -> torch.Tensor:
        """AdaRound regularisation: pushes sigmoid(v) toward 0 or 1.

        R = Σ (1 − |2·σ(v) − 1|^β)

        beta anneals 20 → 2 over the optimisation; the caller is responsible
        for computing beta at each step.
        """
        h = torch.sigmoid(self.v)
        return (1.0 - (2.0 * h - 1.0).abs().pow(beta)).sum()


# ---------------------------------------------------------------------------
# LSQ activation quantizer
# ---------------------------------------------------------------------------

class LSQActivationQuantizer(nn.Module):
    """Learned Step-Size Quantization (LSQ) for activations.

    Supports *signed* quantisation to handle PReLU's negative outputs.

    Args:
        n_bits : bit-width (default 4)
        signed : if True, range is [−2^(b−1), 2^(b−1)−1]  ← use for PReLU
                 if False, range is [0, 2^b−1]             ← use for ReLU
    """

    def __init__(self, n_bits: int = 4, signed: bool = True,
                 init_mode: str = "lsq", init_percentile: float = 0.999):
        super().__init__()
        self.n_bits  = n_bits
        self.signed  = signed
        self.Qn = -(2 ** (n_bits - 1)) if signed else 0
        self.Qp =  (2 ** (n_bits - 1)) - 1 if signed else (2 ** n_bits) - 1
        self.init_mode = init_mode
        self.init_percentile = init_percentile

        self.scale = nn.Parameter(torch.ones(1))
        self._initialized = False

    def _init_scale_value(self, x: torch.Tensor) -> torch.Tensor:
        x_abs = x.detach().abs()

        if self.init_mode == "lsq":
            return (2.0 * x_abs.mean() / sqrt(self.Qp)).clamp(min=1e-8)

        if self.init_mode == "max":
            clip_val = x_abs.max()
        elif self.init_mode == "percentile":
            pct = float(min(max(self.init_percentile, 0.0), 1.0))
            flat = x_abs.reshape(-1)
            if flat.numel() > _MAX_PERCENTILE_SAMPLES:
                step = max(1, flat.numel() // _MAX_PERCENTILE_SAMPLES)
                flat = flat[::step]
            clip_val = flat.max() if pct >= 1.0 else torch.quantile(flat, pct)
        else:
            raise ValueError(f"Unsupported activation init mode: {self.init_mode}")

        return (clip_val / self.Qp).clamp(min=1e-8)

    def init_from_data(self, x: torch.Tensor) -> None:
        """Initialise scale from calibration statistics (call once before optimising)."""
        with torch.no_grad():
            scale = self._init_scale_value(x)
            self.scale.data.fill_(scale.item())
        self._initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._initialized:
            self.init_from_data(x.detach())

        s = self.scale.abs().clamp(min=1e-8)
        x_q = round_ste((x / s).clamp(self.Qn, self.Qp)) * s
        return x_q


# ---------------------------------------------------------------------------
# Quantised Conv2d / Linear wrappers
# ---------------------------------------------------------------------------

class QuantConv2d(nn.Module):
    """Conv2d with AdaRound weight quantisation.

    The original weight is stored as a fixed buffer (not updated during BRECQ).
    Only the AdaRound variable v (inside weight_quantizer) is learnable.
    """

    def __init__(self, conv: nn.Conv2d, w_bits: int = 4):
        super().__init__()
        self.stride       = conv.stride
        self.padding      = conv.padding
        self.dilation     = conv.dilation
        self.groups       = conv.groups
        self.padding_mode = conv.padding_mode

        # Fixed weight & bias (no gradients needed on the original values)
        self.register_buffer("weight", conv.weight.data.clone())
        if conv.bias is not None:
            self.register_buffer("bias", conv.bias.data.clone())
        else:
            self.bias = None

        self.weight_quantizer = AdaRoundQuantizer(conv.weight.data, n_bits=w_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = self.weight_quantizer(self.weight)
        return F.conv2d(x, w_q, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


class QuantLinear(nn.Module):
    """Linear with AdaRound weight quantisation."""

    def __init__(self, linear: nn.Linear, w_bits: int = 4):
        super().__init__()
        self.register_buffer("weight", linear.weight.data.clone())
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.data.clone())
        else:
            self.bias = None

        self.weight_quantizer = AdaRoundQuantizer(linear.weight.data, n_bits=w_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = self.weight_quantizer(self.weight)
        return F.linear(x, w_q, self.bias)
