"""
IR50FPModel — simplified full-precision model for the specific config:

    img_size=224, keypoint=None, features=False,
    input_dist=False, scn_weight=None, decomposition=None, cos=True

Forward path (matches ImbalancedModel for the config above):
    x  ->  interpolate(224x224)
       ->  backbone(x)  [IR50 internally interpolates to 112x112]
       ->  z = embedding[0]        # [B, 256]
       ->  z = L2_norm(z, dim=-1)
       ->  W = L2_norm(weight, dim=0)   # [256, num_classes]
       ->  logit = z @ W                # [B, num_classes]

load(path) / load_from_state_dict(path):
    Reads a checkpoint saved by ImbalancedModel training
    (format: {'model_state_dict': {...}}).
    Loads ONLY  backbone.*  and  weight  keys — ignores head, head_fc, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone import Backbone


class IR50FPModel(nn.Module):
    """Full-precision IR50 + cosine head.

    Backward-compatible with ImbalancedModel constructor interface:
    accepts model_type, feature_branch, cos, etc. via **kwargs
    (only num_classes and img_size are used).
    """

    def __init__(
        self,
        num_classes: int = 7,
        ir50_pretrain: str | None = None,
        img_size: int = 224,
        model_type: str = "ir50",
        **kwargs,
    ):
        super().__init__()
        self.img_size = img_size
        self.backbone = Backbone(ir50_pretrain)
        self.weight = nn.Parameter(
            torch.randn(256, num_classes)
            .uniform_(-1, 1)
            .renorm(2, 1, 1e-5)
            .mul_(1e5)
        )

    # ------------------------------------------------------------------
    def forward(self, x, features=False, keypoint=None,
                wo_branch=False, featuremap=False):
        """Return logits [B, num_classes].

        Extra kwargs (features, keypoint, wo_branch, featuremap) are accepted
        for backward compatibility with ImbalancedModel but ignored.
        """
        if self.img_size == 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        z, _ = self.backbone(x)
        z = F.normalize(z, dim=-1, eps=1e-6)
        W = F.normalize(self.weight, dim=0)
        return z @ W

    @torch.no_grad()
    def get_embedding(self, x):
        """Return L2-normalised 256-dim embedding (no classification head)."""
        if self.img_size == 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        z, _ = self.backbone(x)
        return F.normalize(z, dim=-1, eps=1e-6)

    # ------------------------------------------------------------------
    def load(self, path, strict_backbone=False):
        """Load backbone + classifier from an ImbalancedModel checkpoint.

        Expected format: {'model_state_dict': {'backbone.*': ..., 'weight': ...}}
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        state = {
            (k[len("module."): ] if k.startswith("module.") else k): v
            for k, v in state.items()
        }

        # backbone
        backbone_sd = {
            k[len("backbone."):]: v
            for k, v in state.items()
            if k.startswith("backbone.")
        }
        if backbone_sd:
            missing, unexpected = self.backbone.load_state_dict(
                backbone_sd, strict=strict_backbone
            )
            print(f"[IR50FPModel] backbone: loaded {len(backbone_sd)} tensors", end="")
            if missing:
                print(f"  |  missing: {len(missing)}", end="")
            if unexpected:
                print(f"  |  unexpected: {len(unexpected)}", end="")
            print()
        else:
            print("[IR50FPModel] WARNING: no backbone.* keys in checkpoint")

        # classifier weight
        weight_key = "weight" if "weight" in state else None
        if weight_key is not None:
            w = state[weight_key]
            if w.shape != self.weight.shape:
                raise ValueError(
                    f"weight shape mismatch: ckpt {tuple(w.shape)} vs model {tuple(self.weight.shape)}"
                )
            self.weight.data.copy_(w)
            print(f"[IR50FPModel] classifier weight: loaded {tuple(w.shape)}")
        else:
            print("[IR50FPModel] WARNING: no 'weight' key — keeping random init")

    def load_from_state_dict(self, ckpt_path, clear_weight=True):
        """Backward-compatible with ImbalancedModel.load_from_state_dict."""
        self.load(ckpt_path)
        if clear_weight:
            self.weight.data = (
                torch.randn_like(self.weight)
                .uniform_(-1, 1)
                .renorm(2, 1, 1e-5)
                .mul_(1e5)
            )
