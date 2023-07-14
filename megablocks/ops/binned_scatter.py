import torch
from megablocks.backend import kernels
from stk.backend.autocast import custom_fwd, custom_bwd

# Autograd wrapper for binned_scatter kernel.
class BinnedScatterOp(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, x, indices, bins):
        assert len(x.size()) == 3
        ctx.bin_size = x.size(1)
        ctx.save_for_backward(indices, bins)
        return kernels.binned_scatter(x, indices, bins)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        indices, bins = ctx.saved_tensors
        return kernels.binned_gather(grad, indices, bins, ctx.bin_size), None, None
binned_scatter = BinnedScatterOp.apply
