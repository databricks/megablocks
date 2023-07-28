import torch
from megablocks.backend import kernels
from stk.backend.autocast import custom_fwd, custom_bwd

# Autograd wrapper for binned_gather kernel.
class BinnedGatherOp(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, x, indices, bins, bin_size, top_k):
        ctx.save_for_backward(indices, bins)
        ctx.top_k = top_k
        return kernels.binned_gather(x, indices, None, bins, bin_size, top_k)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        grad = grad.contiguous()
        indices, bins = ctx.saved_tensors
        out = kernels.binned_scatter(grad, indices, None, bins, ctx.top_k)
        return out, None, None, None, None
binned_gather = BinnedGatherOp.apply
