# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

import torch


def indices(
    padded_bins: torch.Tensor,
    block_size: int,
    output_block_rows: int,
    output_block_columns: int,
    out: torch.Tensor,
) -> None:
    """Pure PyTorch indices matching `megablocks_ops.indices` out-params."""
    zeros = torch.zeros(1, device=padded_bins.device, dtype=padded_bins.dtype)
    starts_tokens = torch.cat([zeros, padded_bins[:-1]])
    ends_tokens = padded_bins

    starts_blocks = starts_tokens.div(block_size, rounding_mode="floor")
    ends_blocks = ends_tokens.div(block_size, rounding_mode="floor")
    rows_per_bin = ends_blocks - starts_blocks

    num_bins = padded_bins.numel()
    bin_ids = torch.arange(num_bins, device=padded_bins.device)
    bin_base_values = bin_ids * output_block_columns
    row_base_vals = torch.repeat_interleave(bin_base_values, rows_per_bin)

    if row_base_vals.numel() == 0:
        return

    col_offsets = torch.arange(output_block_columns, device=padded_bins.device).unsqueeze(0)
    result_matrix = row_base_vals.unsqueeze(1) + col_offsets
    result_flat = result_matrix.flatten().to(dtype=torch.int16)

    num_elements = min(out.numel(), result_flat.numel())
    out[:num_elements] = result_flat[:num_elements]

