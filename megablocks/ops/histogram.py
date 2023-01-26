# NOTE: Torch needs to be imported before the custom
# extensions. Otherwise libc10.so cannot be found.
import torch

# TODO(tgale): Wrap this in a try-block with better
# error message and instructions for building the
# c++ operations.
import megablocks_ops as ops

# Autograd wrapper for histogram kernel.
#
# NOTE: Does not support gradients.
class HistogramOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, max_val):
        return ops.histogram(x, max_val)
histogram = HistogramOp.apply
