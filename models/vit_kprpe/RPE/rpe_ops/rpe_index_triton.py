from __future__ import annotations

from typing import Tuple

import torch


try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - runtime availability depends on environment.
    triton = None
    tl = None
    _HAS_TRITON = False


def _validate_input(input: torch.Tensor, index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if input.ndim != 4:
        raise ValueError(f"`input` must be 4D (B,H,L_query,num_buckets). Got: {tuple(input.shape)}")
    if index.ndim != 2:
        raise ValueError(f"`index` must be 2D (L_query,L_key). Got: {tuple(index.shape)}")
    if input.shape[2] != index.shape[0]:
        raise ValueError(
            f"Shape mismatch: input L_query={input.shape[2]} but index L_query={index.shape[0]}"
        )
    if index.dtype != torch.int32:
        index = index.to(torch.int32)
    return input, index.contiguous()


def _forward_torch(input: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    B, H, L_query, num_buckets = input.shape
    L_key = index.shape[1]
    flat = input.reshape(B, H, L_query * num_buckets)
    offset = (
        torch.arange(0, L_query, device=index.device, dtype=index.dtype).view(-1, 1) * int(num_buckets)
    )
    gather_index = (index + offset).reshape(-1).to(torch.long)
    output = flat.index_select(dim=2, index=gather_index).view(B, H, L_query, L_key)
    return output


def _backward_torch(grad_output: torch.Tensor, index: torch.Tensor, input_shape: Tuple[int, ...]) -> torch.Tensor:
    _, _, L_query, _ = input_shape
    expanded_index = index.to(torch.long).view(1, 1, L_query, -1).expand_as(grad_output)
    grad_input = grad_output.new_zeros(input_shape)
    grad_input.scatter_add_(dim=3, index=expanded_index, src=grad_output)
    return grad_input


if _HAS_TRITON:

    @triton.jit
    def _rpe_index_forward_kernel(
        input_ptr,
        index_ptr,
        output_ptr,
        numel,
        H,
        L_query,
        L_key,
        num_buckets,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < numel

        tmp = offsets
        j = tmp % L_key
        tmp = tmp // L_key
        i = tmp % L_query
        tmp = tmp // L_query
        h = tmp % H
        b = tmp // H

        index_offset = i * L_key + j
        bucket = tl.load(index_ptr + index_offset, mask=mask, other=0).to(tl.int32)
        input_offset = (((b * H + h) * L_query + i) * num_buckets + bucket).to(tl.int64)
        values = tl.load(input_ptr + input_offset, mask=mask, other=0)
        tl.store(output_ptr + offsets, values, mask=mask)


    @triton.jit
    def _rpe_index_backward_kernel(
        grad_input_ptr,
        index_ptr,
        grad_output_ptr,
        numel,
        H,
        L_query,
        L_key,
        num_buckets,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < numel

        tmp = offsets
        j = tmp % L_key
        tmp = tmp // L_key
        i = tmp % L_query
        tmp = tmp // L_query
        h = tmp % H
        b = tmp // H

        index_offset = i * L_key + j
        bucket = tl.load(index_ptr + index_offset, mask=mask, other=0).to(tl.int32)

        grad_input_offset = (((b * H + h) * L_query + i) * num_buckets + bucket).to(tl.int64)
        grad_values = tl.load(grad_output_ptr + offsets, mask=mask, other=0)
        tl.atomic_add(grad_input_ptr + grad_input_offset, grad_values, mask=mask)


def _forward_triton(input: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    if not _HAS_TRITON:
        raise RuntimeError("rpe_impl=triton requested but Triton is not available in this environment.")
    if input.device.type != "cuda":
        return _forward_torch(input, index)

    x = input.contiguous()
    idx = index.contiguous()
    B, H, L_query, num_buckets = x.shape
    L_key = idx.shape[1]
    output = torch.empty((B, H, L_query, L_key), device=x.device, dtype=x.dtype)
    numel = output.numel()
    block = 256
    grid = (triton.cdiv(numel, block),)
    _rpe_index_forward_kernel[grid](x, idx, output, numel, H, L_query, L_key, num_buckets, BLOCK=block)
    return output


def _backward_triton(grad_output: torch.Tensor, index: torch.Tensor, input_shape: Tuple[int, ...]) -> torch.Tensor:
    if not _HAS_TRITON:
        raise RuntimeError("rpe_impl=triton requested but Triton is not available in this environment.")
    if grad_output.device.type != "cuda":
        return _backward_torch(grad_output, index, input_shape)

    grad_output = grad_output.contiguous()
    grad_input = grad_output.new_zeros(input_shape)
    _, H, L_query, num_buckets = input_shape
    L_key = index.shape[1]
    numel = grad_output.numel()
    block = 256
    grid = (triton.cdiv(numel, block),)
    _rpe_index_backward_kernel[grid](
        grad_input,
        index.contiguous(),
        grad_output,
        numel,
        H,
        L_query,
        L_key,
        num_buckets,
        BLOCK=block,
    )
    return grad_input


class RPEIndexFunctionTriton(torch.autograd.Function):
    """Y[b, h, i, j] = input[b, h, i, index[i, j]]"""

    @staticmethod
    def forward(ctx, input: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        input, index = _validate_input(input=input, index=index)
        ctx.input_shape = tuple(int(v) for v in input.shape)
        ctx.save_for_backward(index)
        return _forward_triton(input=input, index=index)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        index = ctx.saved_tensors[0]
        if not ctx.needs_input_grad[0]:
            return None, None
        grad_input = _backward_triton(
            grad_output=grad_output,
            index=index,
            input_shape=ctx.input_shape,
        )
        return grad_input, None


_TORCH_LIBRARY_REGISTERED = False


def _register_torch_library_op() -> None:
    global _TORCH_LIBRARY_REGISTERED
    if _TORCH_LIBRARY_REGISTERED:
        return
    if not hasattr(torch, "library"):
        return

    try:
        lib = torch.library.Library("kprpe_ops", "DEF")
        lib.define("rpe_index(Tensor input, Tensor index) -> Tensor")

        lib_impl = torch.library.Library("kprpe_ops", "IMPL")
        lib_impl.impl(
            "rpe_index",
            lambda input, index: RPEIndexFunctionTriton.apply(input, index),
            "CompositeExplicitAutograd",
        )
        _TORCH_LIBRARY_REGISTERED = True
    except Exception:
        # Keep runtime robust even if registration is unavailable on this torch version.
        _TORCH_LIBRARY_REGISTERED = False


_register_torch_library_op()


def triton_rpe_index(input: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    if _TORCH_LIBRARY_REGISTERED and hasattr(torch.ops, "kprpe_ops") and hasattr(torch.ops.kprpe_ops, "rpe_index"):
        return torch.ops.kprpe_ops.rpe_index(input, index)
    return RPEIndexFunctionTriton.apply(input, index)


class RPEIndexFunctionTorchLibrary:
    @staticmethod
    def apply(input: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return triton_rpe_index(input, index)
