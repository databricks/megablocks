import torch
from megablocks.backend import kernels
from stk.backend.autocast import custom_fwd, custom_bwd

# Autograd wrapper for binned_gather kernel.
class BinnedGatherOp(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, x, indices, bins, bin_size):
        ctx.save_for_backward(indices, bins)
        return kernels.binned_gather(x, indices, bins, bin_size)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        indices, bins = ctx.saved_tensors
        return kernels.binned_scatter(grad, indices, bins), None, None, None
binned_gather = BinnedGatherOp.apply
