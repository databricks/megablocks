import torch
from megablocks.backend import kernels
from stk.backend.autocast import custom_fwd, custom_bwd


# Autograd wrapper for gather kernel.
class GatherOp(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, x, indices, bin_ids, bins, top_k):
        ctx.save_for_backward(indices, bin_ids, bins)
        ctx.top_k = top_k
        return kernels.gather(
            x, indices, bin_ids, None, bins, top_k)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        grad = grad.contiguous()

        indices, bin_ids, bins = ctx.saved_tensors
        out = kernels.scatter(
            grad, indices, bin_ids, None, bins, ctx.top_k)
        return out, None, None, None, None, None
gather = GatherOp.apply
