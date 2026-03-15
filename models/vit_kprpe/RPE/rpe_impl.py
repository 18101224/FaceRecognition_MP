from __future__ import annotations

from typing import Optional


_ALLOWED_IMPLS = {"extension", "triton"}
_CURRENT_IMPL = "extension"


def normalize_rpe_impl(value: Optional[str]) -> str:
    if value is None:
        return "extension"
    normalized = str(value).strip().lower()
    if normalized not in _ALLOWED_IMPLS:
        raise ValueError(f"Unsupported rpe_impl: {value}. Supported: {sorted(_ALLOWED_IMPLS)}")
    return normalized


def configure_rpe_impl(value: Optional[str]) -> str:
    global _CURRENT_IMPL
    _CURRENT_IMPL = normalize_rpe_impl(value)
    return _CURRENT_IMPL


def get_rpe_impl() -> str:
    return _CURRENT_IMPL


def get_rpe_index_function():
    impl = get_rpe_impl()
    if impl == "extension":
        from .rpe_ops.rpe_index import RPEIndexFunction

        return RPEIndexFunction
    if impl == "triton":
        from .rpe_ops.rpe_index_triton import RPEIndexFunctionTorchLibrary

        return RPEIndexFunctionTorchLibrary
    raise RuntimeError(f"Unexpected rpe_impl: {impl}")
