# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

import torch


def sort(
    x: torch.Tensor,
    end_bit: int,
    x_out: torch.Tensor,
    iota_out: torch.Tensor,
) -> None:
    """Pure PyTorch sort matching `megablocks_ops.sort` out-params."""
    del end_bit
    if x.ndim != 1:
        raise ValueError("Expected a 1D tensor.")

    sorted_values, sorted_indices = torch.sort(x, stable=True)
    x_out.copy_(sorted_values)
    iota_out.copy_(sorted_indices.to(iota_out.dtype))

