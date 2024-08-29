# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0
from typing import Any

import torch
from stk.backend.autocast import custom_bwd, custom_fwd

from megablocks.backend import kernels


# Autograd wrapper for binned_scatter kernel.
class BinnedScatterOp(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(
        ctx: Any,
        x: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        bins: torch.Tensor,
        top_k: int,
    ):
        assert len(x.size()) == 3
        ctx.bin_size = x.size(1)
        ctx.top_k = top_k

        # TODO(tgale): Don't save 'x' for backwards if we don't need to
        # calculate the gradient w.r.t. 'weights'.
        ctx.save_for_backward(x, indices, weights, bins)
        return kernels.binned_scatter(x, indices, weights, bins, top_k)

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, grad: torch.Tensor):
        grad = grad.contiguous()
        x, indices, weights, bins = ctx.saved_tensors
        out = kernels.binned_gather(
            grad,
            indices,
            weights,
            bins,
            ctx.bin_size,
            ctx.top_k,
        )

        wgrad = None
        if ctx.needs_input_grad[2]:
            wgrad = kernels.binned_scatter_wgrad(
                x,
                grad,
                indices,
                bins,
                ctx.top_k,
            )
        return out, None, wgrad, None, None


binned_scatter = BinnedScatterOp.apply
