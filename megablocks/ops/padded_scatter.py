# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0
from typing import Any

import torch
from stk.backend.autocast import custom_bwd, custom_fwd

from megablocks.backend import kernels


# Autograd wrapper for padded_scatter kernel.
class PaddedScatterOp(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(
        ctx: Any,
        x: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        weights: torch.Tensor,
        bins: torch.Tensor,
        padded_bins: torch.Tensor,
        top_k: int,
    ):
        maybe_x = [x] if ctx.needs_input_grad[3] else []
        ctx.save_for_backward(
            indices,
            bin_ids,
            weights,
            bins,
            padded_bins,
            *maybe_x,
        )
        ctx.top_k = top_k
        ctx.x_shape = x.shape
        return kernels.padded_scatter(
            x,
            indices,
            bin_ids,
            weights,
            bins,
            padded_bins,
            top_k,
        )

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, grad: torch.Tensor):
        grad = grad.contiguous()
        saved_tensors = ctx.saved_tensors

        indices, bin_ids, weights, bins, padded_bins = saved_tensors[:5]
        dgrad = None
        if ctx.needs_input_grad[0]:
            dgrad = kernels.padded_gather(
                grad,
                indices,
                bin_ids,
                weights,
                bins,
                padded_bins,
                ctx.top_k,
            )

        wgrad = None
        if ctx.needs_input_grad[3]:  # need wgrad
            x = saved_tensors[-1]
            wgrad = kernels.padded_scatter_wgrad(
                x,
                grad,
                indices,
                bin_ids,
                bins,
                padded_bins,
                ctx.top_k,
            )
        return dgrad, None, None, wgrad, None, None, None, None


def padded_scatter(
    x: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    weights: torch.Tensor,
    bins: torch.Tensor,
    padded_bins: torch.Tensor,
    top_k: int,
):
    return PaddedScatterOp.apply(
        x,
        indices,
        bin_ids,
        weights,
        bins,
        padded_bins,
        top_k,
    )
