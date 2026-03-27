# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

import torch


def replicate_forward(x: torch.Tensor, bins: torch.Tensor, out: torch.Tensor) -> None:
    """Pure PyTorch replicate_forward matching `megablocks_ops.replicate_forward`."""
    zero = torch.tensor([0], device=bins.device, dtype=bins.dtype)
    counts = torch.diff(bins, prepend=zero).to(torch.long)
    res = torch.repeat_interleave(x, counts, dim=1)
    out.copy_(res)


def replicate_backward(grad: torch.Tensor, bins: torch.Tensor, out: torch.Tensor) -> None:
    """Pure PyTorch replicate_backward matching `megablocks_ops.replicate_backward`."""
    zero = torch.tensor([0], device=bins.device, dtype=bins.dtype)
    counts = torch.diff(bins, prepend=zero).to(torch.long)

    num_bins = bins.size(0)
    bin_indices = torch.repeat_interleave(
        torch.arange(num_bins, device=grad.device, dtype=torch.long),
        counts,
    )
    batch_size = grad.size(0)
    expanded_indices = bin_indices.unsqueeze(0).expand(batch_size, -1)

    out.zero_()
    out.scatter_add_(dim=1, index=expanded_indices, src=grad)

