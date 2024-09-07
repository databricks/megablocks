# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

from typing import Any

# NOTE: Torch needs to be imported before the custom
# extensions. Otherwise libc10.so cannot be found.
import torch

# Wrap this in a try-block with better error message and
# instructions for building the c++ operations.
try:
    import megablocks_ops as ops  # type: ignore
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("No module named 'megablocks_ops'.") from e


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
        out = torch.empty(
            output_block_rows * output_block_columns,
            dtype=torch.int16,
            device=padded_bins.device,
        )
        ops.indices(
            padded_bins,
            block_size,
            output_block_rows,
            output_block_columns,
            out,
        )
        return out


topology = TopologyOp.apply
