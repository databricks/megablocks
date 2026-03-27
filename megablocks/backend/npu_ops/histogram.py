# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

import torch


def histogram(x: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Pure PyTorch histogram matching `megablocks_ops.histogram`.

    Supports both 1D and 2D inputs. Returns int32 counts.
    """
    if x.ndim not in (1, 2):
        raise ValueError(f"Expected 1D or 2D tensor but got {x.ndim}D.")

    original_ndim = x.ndim
    if original_ndim == 1:
        x = x.view(1, -1)

    batch_size = x.shape[0]
    if x.numel() == 0:
        out = torch.zeros(batch_size, num_bins, device=x.device, dtype=torch.int32)
        return out.flatten() if original_ndim == 1 else out

    # Batched bincount by shifting each row into a disjoint bin range.
    offsets = torch.arange(batch_size, device=x.device, dtype=torch.int64) * int(num_bins)
    x_flat_shifted = (x.to(torch.int64) + offsets.unsqueeze(1)).reshape(-1)
    counts = torch.bincount(x_flat_shifted, minlength=batch_size * int(num_bins))
    out = counts.view(batch_size, int(num_bins)).to(torch.int32)
    return out.flatten() if original_ndim == 1 else out

