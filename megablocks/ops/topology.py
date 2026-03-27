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


def _indices_fallback(
    padded_bins: torch.Tensor,
    block_size: int,
    output_block_rows: int,
    output_block_columns: int,
) -> torch.Tensor:
    # Mirror `csrc/indices.h` on any device type.
    num_bins = padded_bins.numel()
    if num_bins == 0 or output_block_rows * output_block_columns == 0:
        return torch.empty(
            output_block_rows * output_block_columns,
            dtype=torch.int16,
            device=padded_bins.device,
        )

    starts_tokens = torch.zeros_like(padded_bins)
    starts_tokens[1:] = padded_bins[:-1]
    starts_blocks = torch.div(starts_tokens, block_size, rounding_mode='floor')
    ends_blocks = torch.div(padded_bins, block_size, rounding_mode='floor')

    out = torch.empty(
        output_block_rows * output_block_columns,
        dtype=torch.int16,
        device=padded_bins.device,
    )
    out.zero_()
    col_offsets = torch.arange(
        output_block_columns,
        device=padded_bins.device,
        dtype=torch.int16,
    )

    # Fill each block-row belonging to each bin.
    for bin_id in range(num_bins):
        start = int(starts_blocks[bin_id].item())
        end = int(ends_blocks[bin_id].item())
        value = torch.as_tensor(bin_id * output_block_columns, device=out.device, dtype=torch.int16) + col_offsets
        for row in range(start, min(end, output_block_rows)):
            base = row * output_block_columns
            out[base: base + output_block_columns] = value

    return out


# Autograd wrapper for topology kernel.
# NOTE: Does not support gradients.
class TopologyOp(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: Any,
        padded_bins: torch.Tensor,
        block_size: int,
        output_block_rows: int,
        output_block_columns: int,
    ):
        if padded_bins.device.type == 'npu':
            return _indices_fallback(
                padded_bins,
                block_size,
                output_block_rows,
                output_block_columns,
            )
        if _ops is None:
            raise ModuleNotFoundError("No module named 'megablocks_ops'.")
        out = torch.empty(
            output_block_rows * output_block_columns,
            dtype=torch.int16,
            device=padded_bins.device,
        )
        _ops.indices(
            padded_bins,
            block_size,
            output_block_rows,
            output_block_columns,
            out,
        )
        return out


topology = TopologyOp.apply
