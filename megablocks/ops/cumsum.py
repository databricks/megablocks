# NOTE: Torch needs to be imported before the custom
# extensions. Otherwise libc10.so cannot be found.
import torch

# TODO(tgale): Wrap this in a try-block with better
# error message and instructions for building the
# c++ operations.
import megablocks_ops as ops

# Autograd wrappers for cumsum kernels.
#
# NOTE: Does not support gradients.
class ExclusiveCumsumOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, dim):
        if len(x.size()) == 1:
            x = x.view([1, -1])
            out = torch.empty_like(x)
            ops.exclusive_cumsum(x, 1, out)
            return out.squeeze()
        out = torch.empty_like(x)
        ops.exclusive_cumsum(x, dim, out)
        return out
exclusive_cumsum = ExclusiveCumsumOp.apply

class InclusiveCumsumOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, dim):
        if len(x.size()) == 1:
            x = x.view([1, -1])
            out = torch.empty_like(x)
            ops.inclusive_cumsum(x, 1, out)
            return out.squeeze()
        out = torch.empty_like(x)
        ops.inclusive_cumsum(x, dim, out)
        return out
inclusive_cumsum = InclusiveCumsumOp.apply
