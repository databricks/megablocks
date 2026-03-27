# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

from typing import Any

# NOTE: Torch needs to be imported before the custom
# extensions. Otherwise libc10.so cannot be found.
import torch

try:
    import megablocks_ops as _ops  # type: ignore
except ModuleNotFoundError:
    _ops = None


def _histogram_fallback(x: torch.Tensor, max_val: float) -> torch.Tensor:
    # `max_val` is used as number of bins throughout Megablocks.
    num_bins = int(max_val)

    # The C++ kernel supports both 1D and 2D inputs.
    if x.ndim == 1:
        return torch.bincount(x.to(torch.int64), minlength=num_bins).to(torch.int32)

    if x.ndim != 2:
        raise ValueError(f'Expected 1D or 2D tensor but got {x.ndim}D.')

    batch_size = x.shape[0]
    offsets = torch.arange(batch_size, device=x.device, dtype=torch.int64) * num_bins
    x_shifted = x.to(torch.int64) + offsets.unsqueeze(1)
    counts = torch.bincount(x_shifted.flatten(), minlength=batch_size * num_bins)
    return counts.view(batch_size, num_bins).to(torch.int32)


# Autograd wrapper for histogram kernel.
# NOTE: Does not support gradients.
class HistogramOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, max_val: float):
        # Ascend NPU does not support the CUDA extension; fall back to PyTorch.
        if x.device.type == 'npu':
            return _histogram_fallback(x, max_val)
        if _ops is None:
            raise ModuleNotFoundError("No module named 'megablocks_ops'.")
        return _ops.histogram(x, max_val)


histogram = HistogramOp.apply
