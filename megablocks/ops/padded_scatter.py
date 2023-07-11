import torch
from megablocks.backend import kernels
from stk.backend.autocast import custom_fwd, custom_bwd


# Autograd wrapper for padded_scatter kernel.
class PaddedScatterOp(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, x, indices, bin_ids, bins, padded_bins):
        ctx.save_for_backward(indices, bin_ids, bins, padded_bins)
        return kernels.padded_scatter(x, indices, bin_ids, bins, padded_bins)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        grad = grad.contiguous()

        indices, bin_ids, bins, padded_bins = ctx.saved_tensors
        out = kernels.padded_gather(grad, indices, bin_ids, bins, padded_bins)
        return out, None, None, None, None
padded_scatter = PaddedScatterOp.apply
