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


# Autograd wrapper for replicate kernel.
class ReplicateOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, bins: torch.Tensor, num_outputs: int):
        ctx.save_for_backward(bins)
        out = torch.empty((x.shape[0], num_outputs), dtype=x.dtype, device=x.device)
        ops.replicate_forward(x, bins, out)
        return out

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor):
        bins, = ctx.saved_tensors
        out = torch.empty((grad.shape[0], bins.shape[0]), dtype=grad.dtype, device=grad.device)
        ops.replicate_backward(grad, bins, out)
        return out, None, None


replicate = ReplicateOp.apply
