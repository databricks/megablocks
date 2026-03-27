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


def _bin_counts_from_inclusive_bins(bins: torch.Tensor) -> torch.Tensor:
    zeros = bins.new_zeros((1,))
    counts = torch.diff(bins.to(torch.long), prepend=zeros)
    return counts


# Autograd wrapper for replicate kernel.
class ReplicateOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, bins: torch.Tensor, num_outputs: int):
        ctx.save_for_backward(bins)
        if x.device.type == 'npu':
            counts = _bin_counts_from_inclusive_bins(bins)
            out = torch.repeat_interleave(x, counts, dim=1)
            # Defensive: match the requested output width.
            if out.shape[1] < num_outputs:
                out = torch.nn.functional.pad(out, (0, num_outputs - out.shape[1]))
            elif out.shape[1] > num_outputs:
                out = out[:, :num_outputs]
            return out
        if _ops is None:
            raise ModuleNotFoundError("No module named 'megablocks_ops'.")
        out = torch.empty((x.shape[0], num_outputs), dtype=x.dtype, device=x.device)
        _ops.replicate_forward(x, bins, out)
        return out

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor):
        bins, = ctx.saved_tensors
        if grad.device.type == 'npu':
            counts = _bin_counts_from_inclusive_bins(bins)
            num_bins = bins.shape[0]
            bin_ids = torch.repeat_interleave(
                torch.arange(num_bins, device=grad.device, dtype=torch.long),
                counts,
            )
            expanded = bin_ids.unsqueeze(0).expand(grad.shape[0], -1)
            out = torch.zeros((grad.shape[0], num_bins), dtype=grad.dtype, device=grad.device)
            out.scatter_add_(dim=1, index=expanded, src=grad)
            return out, None, None
        if _ops is None:
            raise ModuleNotFoundError("No module named 'megablocks_ops'.")
        out = torch.empty((grad.shape[0], bins.shape[0]), dtype=grad.dtype, device=grad.device)
        _ops.replicate_backward(grad, bins, out)
        return out, None, None


replicate = ReplicateOp.apply
