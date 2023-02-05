# NOTE: Torch needs to be imported before the custom
# extensions. Otherwise libc10.so cannot be found.
import torch

from stk.backend.autocast import custom_fwd, custom_bwd

# TODO(tgale): Wrap this in a try-block with better
# error message and instructions for building the
# c++ operations.
import megablocks_ops as ops


# Autograd wrapper for padded_gather kernel.
class PaddedGatherOp(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, x, indices, bin_ids, bins, padded_bins):
        ctx.save_for_backward(indices, bin_ids, bins, padded_bins)
        return ops.padded_gather(x, indices, bin_ids, bins, padded_bins)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        grad = grad.contiguous()

        indices, bin_ids, bins, padded_bins = ctx.saved_tensors
        out = ops.padded_scatter(grad, indices, bin_ids, bins, padded_bins)
        return out, None, None, None, None
padded_gather = PaddedGatherOp.apply
