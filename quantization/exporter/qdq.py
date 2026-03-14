"""
Q/DQ export helpers for explicit TensorRT quantization.

This path converts the repo's BRECQ fake-quant modules into ONNX-friendly Q/DQ
graphs. The exported graph is intended for explicit INT8 TensorRT deployment:

  - activation tensors use per-tensor Q/DQ
  - conv weights use per-output-channel Q/DQ
  - 4-bit layers keep their original narrow clipping range, but the ONNX tensor
    type is INT8 so TensorRT can consume the graph as explicit INT8 quantization

This preserves the learned AdaRound rounding decisions by exporting already
quantized float weights on the learned grid rather than the original FP weights.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _reshape_channel_vector(x: torch.Tensor, axis: int, ndim: int) -> torch.Tensor:
    if axis < 0 or x.ndim != 1:
        return x
    shape = [1] * ndim
    shape[axis] = x.numel()
    return x.reshape(shape)


def _preview_tensor(x: torch.Tensor, limit: int = 8) -> list[float]:
    flat = x.detach().reshape(-1).float().cpu()
    return [float(v) for v in flat[:limit]]


def _needs_explicit_clip(qmin: int, qmax: int) -> bool:
    full_int8 = (qmin, qmax) == (-128, 127)
    full_uint8 = (qmin, qmax) == (0, 255)
    return not (full_int8 or full_uint8)


def _quant_module_report(
    path: str,
    *,
    kind: str,
    n_bits: int,
    qmin: int,
    qmax: int,
    scale: torch.Tensor,
    granularity: str,
    axis: int | None,
    signed: bool,
    runtime_dtype: str = "int8",
) -> dict:
    report = {
        "path": path,
        "kind": kind,
        "runtime_dtype": runtime_dtype,
        "trained_bits": int(n_bits),
        "signed": bool(signed),
        "qmin": int(qmin),
        "qmax": int(qmax),
        "granularity": granularity,
        "scale_shape": list(scale.shape),
        "scale_preview": _preview_tensor(scale),
    }
    if axis is not None:
        report["axis"] = int(axis)
    return report


def build_qdq_export_report(quant_blocks, block_names=None) -> dict:
    if block_names is None:
        block_names = [f"block_{idx}" for idx in range(len(quant_blocks))]
    if len(block_names) != len(quant_blocks):
        raise ValueError("block_names length must match quant_blocks length")

    entries: list[dict] = []

    for block_name, block in zip(block_names, quant_blocks):
        if hasattr(block, "conv"):
            q = block.conv.weight_quantizer
            entries.append(_quant_module_report(
                f"{block_name}.conv.weight",
                kind="weight",
                n_bits=q.n_bits,
                qmin=q.Qn,
                qmax=q.Qp,
                scale=q.scale.detach().view(-1).float(),
                granularity="per_channel",
                axis=0,
                signed=True,
            ))
            aq = block.act_quant
            entries.append(_quant_module_report(
                f"{block_name}.activation",
                kind="activation",
                n_bits=aq.n_bits,
                qmin=aq.Qn,
                qmax=aq.Qp,
                scale=aq.scale.detach().reshape(1).float(),
                granularity="per_tensor",
                axis=None,
                signed=aq.signed,
            ))
            continue

        q_modules = [
            ("conv1.weight", block.conv1.weight_quantizer),
            ("conv2.weight", block.conv2.weight_quantizer),
        ]
        if not isinstance(block.shortcut, nn.MaxPool2d):
            q_modules.append(("shortcut.weight", block.shortcut.weight_quantizer))

        for suffix, q in q_modules:
            entries.append(_quant_module_report(
                f"{block_name}.{suffix}",
                kind="weight",
                n_bits=q.n_bits,
                qmin=q.Qn,
                qmax=q.Qp,
                scale=q.scale.detach().view(-1).float(),
                granularity="per_channel",
                axis=0,
                signed=True,
            ))

        aq = block.act_quant
        entries.append(_quant_module_report(
            f"{block_name}.activation",
            kind="activation",
            n_bits=aq.n_bits,
            qmin=aq.Qn,
            qmax=aq.Qp,
            scale=aq.scale.detach().reshape(1).float(),
            granularity="per_tensor",
            axis=None,
            signed=aq.signed,
        ))

    return {
        "runtime_format": "onnx_qdq_int8",
        "notes": [
            "runtime_dtype=int8 indicates the exported ONNX Q/DQ tensor type",
            "trained_bits/qmin/qmax capture the original learned BRECQ range",
            "classifier head remains FP32 in the current deployment wrapper",
        ],
        "num_blocks": len(quant_blocks),
        "entries": entries,
    }


def save_qdq_export_report(path: str | Path, quant_blocks, block_names=None) -> dict:
    report = build_qdq_export_report(quant_blocks, block_names=block_names)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report


class _QDQFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        clip_min: torch.Tensor,
        clip_max: torch.Tensor,
        axis: int,
        use_clip: int,
    ) -> torch.Tensor:
        del ctx, zero_point

        scale = scale.to(device=x.device, dtype=x.dtype)
        scale_b = _reshape_channel_vector(scale, axis, x.ndim)

        if use_clip:
            clip_min = clip_min.to(device=x.device, dtype=x.dtype)
            clip_max = clip_max.to(device=x.device, dtype=x.dtype)
            clip_min_b = clip_min if clip_min.ndim == x.ndim else _reshape_channel_vector(clip_min, axis, x.ndim)
            clip_max_b = clip_max if clip_max.ndim == x.ndim else _reshape_channel_vector(clip_max, axis, x.ndim)
            x = torch.clamp(x, min=clip_min_b, max=clip_max_b)
        x = torch.round(x / scale_b) * scale_b
        return x

    @staticmethod
    def symbolic(g, x, scale, zero_point, clip_min, clip_max, axis, use_clip):
        quant_input = x
        if use_clip:
            quant_input = g.op("Clip", x, clip_min, clip_max)
        if axis >= 0:
            q = g.op("QuantizeLinear", quant_input, scale, zero_point, axis_i=axis)
            return g.op("DequantizeLinear", q, scale, zero_point, axis_i=axis)

        q = g.op("QuantizeLinear", quant_input, scale, zero_point)
        return g.op("DequantizeLinear", q, scale, zero_point)


class ONNXExplicitQuantizer(nn.Module):
    """Q/DQ wrapper for a tensor or a per-channel weight."""

    def __init__(
        self,
        scale: torch.Tensor,
        qmin: int,
        qmax: int,
        axis: int = -1,
        clip_ndim: int | None = None,
        use_clip: bool = True,
    ):
        super().__init__()
        scale = scale.detach().clone().float()
        if scale.ndim == 0:
            scale = scale.reshape(1)
        self.axis = axis
        self.use_clip = int(use_clip)

        zero_point = torch.zeros_like(scale, dtype=torch.int8)
        clip_scale = scale
        if clip_ndim is not None and scale.ndim == 1:
            clip_scale = _reshape_channel_vector(scale, axis, clip_ndim)

        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point)
        self.register_buffer("clip_min", clip_scale * float(qmin))
        self.register_buffer("clip_max", clip_scale * float(qmax))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _QDQFunction.apply(
            x,
            self.scale,
            self.zero_point,
            self.clip_min,
            self.clip_max,
            self.axis,
            self.use_clip,
        )


class QDQConv2d(nn.Module):
    """Conv2d with explicit ONNX Q/DQ weight export."""

    def __init__(self, quant_conv):
        super().__init__()
        q = quant_conv.weight_quantizer
        quant_weight = q(quant_conv.weight).detach().clone()

        self.stride = quant_conv.stride
        self.padding = quant_conv.padding
        self.dilation = quant_conv.dilation
        self.groups = quant_conv.groups

        self.register_buffer("weight", quant_weight)
        if quant_conv.bias is not None:
            self.register_buffer("bias", quant_conv.bias.detach().clone())
        else:
            self.bias = None

        self.weight_qdq = ONNXExplicitQuantizer(
            scale=q.scale.detach().view(-1).float(),
            qmin=int(q.Qn),
            qmax=int(q.Qp),
            axis=0,
            clip_ndim=quant_weight.ndim,
            use_clip=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight_qdq(self.weight)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class QDQInputLayer(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.conv = QDQConv2d(block.conv)
        self.prelu = deepcopy(block.prelu)
        self.act_qdq = ONNXExplicitQuantizer(
            scale=block.act_quant.scale.detach().reshape(1).float(),
            qmin=int(block.act_quant.Qn),
            qmax=int(block.act_quant.Qp),
            use_clip=_needs_explicit_clip(int(block.act_quant.Qn), int(block.act_quant.Qp)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.prelu(x)
        x = self.act_qdq(x)
        return x


class QDQBottleneckIR(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.conv1 = QDQConv2d(block.conv1)
        self.prelu = deepcopy(block.prelu)
        self.act_qdq = ONNXExplicitQuantizer(
            scale=block.act_quant.scale.detach().reshape(1).float(),
            qmin=int(block.act_quant.Qn),
            qmax=int(block.act_quant.Qp),
            use_clip=_needs_explicit_clip(int(block.act_quant.Qn), int(block.act_quant.Qp)),
        )
        self.conv2 = QDQConv2d(block.conv2)

        if isinstance(block.shortcut, nn.MaxPool2d):
            self.shortcut = deepcopy(block.shortcut)
        else:
            self.shortcut = QDQConv2d(block.shortcut)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.shortcut(x)
        r = self.conv1(x)
        r = self.prelu(r)
        r = self.act_qdq(r)
        r = self.conv2(r)
        return r + s


class QDQIR50Classifier(nn.Module):
    """IR50 cosine classifier with explicit INT8 Q/DQ backbone export."""

    preserve_qdq_nodes = True

    def __init__(self, quant_blocks, classifier_weight: torch.Tensor, img_size: int = 224):
        super().__init__()
        self.img_size = img_size
        blocks = []
        for idx, block in enumerate(quant_blocks):
            if idx == 0:
                blocks.append(QDQInputLayer(block))
            else:
                blocks.append(QDQBottleneckIR(block))
        self.blocks = nn.ModuleList(blocks)
        self.register_buffer("weight", classifier_weight.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.img_size == 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        x = F.interpolate(x, size=112)
        for block in self.blocks:
            x = block(x)

        emb = F.adaptive_avg_pool2d(x, 1).reshape(x.size(0), -1)
        emb = F.normalize(emb, dim=-1, eps=1e-6)
        weight = F.normalize(self.weight, dim=0)
        return emb @ weight
