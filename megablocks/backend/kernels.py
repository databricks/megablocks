# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional

import torch


_TRITON: Optional[object]
try:
    # Triton kernels (CUDA fast-path).
    from megablocks.backend import triton_kernels as _TRITON  # type: ignore
except Exception:
    _TRITON = None


def _fallback():
    # Pure-PyTorch implementation used on Ascend NPU (and when Triton isn't available).
    from megablocks.backend import npu_kernels as _NPU  # type: ignore

    return _NPU


def _should_fallback(x: torch.Tensor) -> bool:
    # NPU runtime doesn't support Triton kernels.
    return x.device.type == "npu" or _TRITON is None


def padded_gather(x, indices, bin_ids, weights, bins, padded_bins, top_k):
    if _should_fallback(x):
        return _fallback().padded_gather(x, indices, bin_ids, weights, bins, padded_bins, top_k)
    return _TRITON.padded_gather(x, indices, bin_ids, weights, bins, padded_bins, top_k)  # type: ignore[union-attr]


def gather(x, indices, bin_ids, weights, bins, top_k):
    if _should_fallback(x):
        return _fallback().gather(x, indices, bin_ids, weights, bins, top_k)
    return _TRITON.gather(x, indices, bin_ids, weights, bins, top_k)  # type: ignore[union-attr]


def padded_scatter(x, indices, bin_ids, weights, bins, padded_bins, top_k):
    if _should_fallback(x):
        return _fallback().padded_scatter(x, indices, bin_ids, weights, bins, padded_bins, top_k)
    return _TRITON.padded_scatter(x, indices, bin_ids, weights, bins, padded_bins, top_k)  # type: ignore[union-attr]


def scatter(x, indices, bin_ids, weights, bins, top_k):
    if _should_fallback(x):
        return _fallback().scatter(x, indices, bin_ids, weights, bins, top_k)
    return _TRITON.scatter(x, indices, bin_ids, weights, bins, top_k)  # type: ignore[union-attr]


def padded_scatter_wgrad(x, grad, indices, bin_ids, bins, padded_bins, top_k):
    if _should_fallback(x):
        return _fallback().padded_scatter_wgrad(x, grad, indices, bin_ids, bins, padded_bins, top_k)
    return _TRITON.padded_scatter_wgrad(x, grad, indices, bin_ids, bins, padded_bins, top_k)  # type: ignore[union-attr]


def scatter_wgrad(x, grad, indices, bin_ids, bins, top_k):
    if _should_fallback(x):
        return _fallback().scatter_wgrad(x, grad, indices, bin_ids, bins, top_k)
    return _TRITON.scatter_wgrad(x, grad, indices, bin_ids, bins, top_k)  # type: ignore[union-attr]


def binned_gather(x, indices, weights, bins, expert_capacity, top_k):
    if _should_fallback(x):
        return _fallback().binned_gather(x, indices, weights, bins, expert_capacity, top_k)
    return _TRITON.binned_gather(x, indices, weights, bins, expert_capacity, top_k)  # type: ignore[union-attr]


def binned_scatter(x, indices, weights, bins, top_k):
    if _should_fallback(x):
        return _fallback().binned_scatter(x, indices, weights, bins, top_k)
    return _TRITON.binned_scatter(x, indices, weights, bins, top_k)  # type: ignore[union-attr]


def binned_scatter_wgrad(x, grad, indices, bins, top_k):
    if _should_fallback(x):
        return _fallback().binned_scatter_wgrad(x, grad, indices, bins, top_k)
    return _TRITON.binned_scatter_wgrad(x, grad, indices, bins, top_k)  # type: ignore[union-attr]

