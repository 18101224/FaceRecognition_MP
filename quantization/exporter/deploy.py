"""
Deployment helpers for exporting and benchmarking IR50 classifiers.

This module builds inference-only wrappers for:
  - FP32 IR50 + cosine classifier
  - BRECQ fake-quant IR50 + cosine classifier

The BRECQ path exports the current fake-quant graph exactly as implemented in
this repo. It is useful for measuring deployment/runtime overhead, but it does
not automatically become a true INT4 TensorRT engine. Real INT4 kernels still
require a TensorRT/plugin path that understands Q/DQ or custom low-bit ops.
"""

from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ptq.optimizer import BRECQOptimizer
from ..calibration.fp_model import IR50FPModel
from .qdq import QDQIR50Classifier, build_qdq_export_report


class QuantizedIR50Classifier(nn.Module):
    """Inference-only IR50 classifier backed by frozen BRECQ quant blocks."""

    def __init__(
        self,
        quant_blocks,
        classifier_weight: torch.Tensor,
        img_size: int = 224,
    ):
        super().__init__()
        self.img_size = img_size
        self.blocks = nn.ModuleList([deepcopy(block).eval() for block in quant_blocks])
        self.register_buffer("weight", classifier_weight.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Match IR50FPModel -> Backbone path exactly.
        if self.img_size == 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        x = F.interpolate(x, size=112)
        for block in self.blocks:
            x = block(x)

        emb = F.adaptive_avg_pool2d(x, 1).reshape(x.size(0), -1)
        emb = F.normalize(emb, dim=-1, eps=1e-6)
        weight = F.normalize(self.weight, dim=0)
        return emb @ weight


def _load_checkpoint_payload(checkpoint_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    state = {
        key[len("module."):] if key.startswith("module.") else key: value
        for key, value in state.items()
    }
    params = dict(ckpt.get("model_params", {}))
    return state, params


def load_ir50_model_params(checkpoint_path: str | Path) -> dict[str, Any]:
    state, params = _load_checkpoint_payload(checkpoint_path)
    if "num_classes" not in params:
        if "weight" not in state:
            raise ValueError("Checkpoint is missing 'model_params.num_classes' and classifier 'weight'")
        params["num_classes"] = state["weight"].shape[1]
    params.setdefault("img_size", 224)
    params.setdefault("model_type", "ir50")
    return params


def load_fp32_ir50_classifier(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> IR50FPModel:
    params = load_ir50_model_params(checkpoint_path)
    model = IR50FPModel(**params)
    model.load(checkpoint_path)
    return model.to(device).eval()


def load_brecq_ir50_classifier(
    checkpoint_path: str | Path,
    quant_state_path: str | Path,
    device: str | torch.device = "cpu",
) -> QuantizedIR50Classifier:
    fp_model = load_fp32_ir50_classifier(checkpoint_path, device=device)
    quant_state = torch.load(quant_state_path, map_location="cpu", weights_only=False)

    optimizer = BRECQOptimizer(
        fp_model.backbone,
        w_bits=quant_state["w_bits"],
        a_bits=quant_state["a_bits"],
        first_last_bits=quant_state.get("first_last_bits", 8),
        n_iters=1,
        lam=0.0,
        batch_size=1,
        use_fisher=False,
        precompute=False,
        opt_target=quant_state.get("opt_target", "both"),
        reg_reduction=quant_state.get("reg_reduction", "sum"),
        act_init_mode=quant_state.get("act_init_mode", "lsq"),
        act_init_percentile=quant_state.get("act_init_percentile", 0.999),
        act_init_samples=quant_state.get("act_init_samples", 64),
        device=str(device),
        verbose=False,
    )
    optimizer.load(str(quant_state_path))

    model = QuantizedIR50Classifier(
        optimizer._quant_blocks,
        classifier_weight=fp_model.weight,
        img_size=fp_model.img_size,
    )
    return model.to(device).eval()


def load_brecq_qdq_ir50_classifier(
    checkpoint_path: str | Path,
    quant_state_path: str | Path,
    device: str | torch.device = "cpu",
) -> QDQIR50Classifier:
    model, _ = load_brecq_qdq_ir50_export_bundle(
        checkpoint_path,
        quant_state_path,
        device=device,
    )
    return model


def load_brecq_qdq_ir50_export_bundle(
    checkpoint_path: str | Path,
    quant_state_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[QDQIR50Classifier, dict[str, Any]]:
    fp_model = load_fp32_ir50_classifier(checkpoint_path, device=device)
    quant_state = torch.load(quant_state_path, map_location="cpu", weights_only=False)

    optimizer = BRECQOptimizer(
        fp_model.backbone,
        w_bits=quant_state["w_bits"],
        a_bits=quant_state["a_bits"],
        first_last_bits=quant_state.get("first_last_bits", 8),
        n_iters=1,
        lam=0.0,
        batch_size=1,
        use_fisher=False,
        precompute=False,
        opt_target=quant_state.get("opt_target", "both"),
        reg_reduction=quant_state.get("reg_reduction", "sum"),
        act_init_mode=quant_state.get("act_init_mode", "lsq"),
        act_init_percentile=quant_state.get("act_init_percentile", 0.999),
        act_init_samples=quant_state.get("act_init_samples", 64),
        device=str(device),
        verbose=False,
    )
    optimizer.load(str(quant_state_path))

    model = QDQIR50Classifier(
        optimizer._quant_blocks,
        classifier_weight=fp_model.weight,
        img_size=fp_model.img_size,
    )
    report = build_qdq_export_report(
        optimizer._quant_blocks,
        block_names=optimizer._block_names,
    )
    return model.to(device).eval(), report


def export_ir50_onnx(
    model: nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, int, int, int],
    dynamic_batch: bool = False,
    opset: int = 17,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(*input_shape, device=next(model.parameters(), torch.empty(0, device="cpu")).device)
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}}

    with torch.no_grad():
        export_kwargs = dict(
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            export_params=True,
            do_constant_folding=not getattr(model, "preserve_qdq_nodes", False),
            opset_version=opset,
        )
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            export_kwargs["dynamo"] = False

        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            **export_kwargs,
        )
