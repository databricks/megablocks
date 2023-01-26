# NOTE: Torch needs to be imported before the custom
# extensions. Otherwise libc10.so cannot be found.
import torch

# TODO(tgale): Wrap this in a try-block with better
# error message and instructions for building the
# c++ operations.
import megablocks_ops as ops

# Autograd wrapper for binned_scatter kernel.
class BinnedScatterOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, indices, bins):
        assert len(x.size()) == 3
        ctx.bin_size = x.size(1)
        ctx.save_for_backward(indices, bins)
        return ops.binned_scatter(x, indices, bins)

    @staticmethod
    def backward(ctx, grad):
        indices, bins = ctx.saved_tensors
        return ops.binned_gather(grad, indices, bins, ctx.bin_size), None, None
binned_scatter = BinnedScatterOp.apply
