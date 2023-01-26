# NOTE: Torch needs to be imported before the custom
# extensions. Otherwise libc10.so cannot be found.
import torch

# TODO(tgale): Wrap this in a try-block with better
# error message and instructions for building the
# c++ operations.
import megablocks_ops as ops

# Autograd wrapper for replicate kernel.
class ReplicateOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, bins, num_outputs):
        ctx.save_for_backward(bins)
        out = torch.empty(
            (x.shape[0], num_outputs),
            dtype=x.dtype,
            device=x.device)
        ops.replicate_forward(x, bins, out)
        return out

    @staticmethod
    def backward(ctx, grad):
        bins, = ctx.saved_tensors
        out = torch.empty(
            (grad.shape[0], bins.shape[0]),
            dtype=grad.dtype,
            device=grad.device)
        ops.replicate_backward(grad, bins, out)
        return out, None, None
replicate = ReplicateOp.apply
