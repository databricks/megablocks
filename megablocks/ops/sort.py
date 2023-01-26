# NOTE: Torch needs to be imported before the custom
# extensions. Otherwise libc10.so cannot be found.
import torch

# TODO(tgale): Wrap this in a try-block with better
# error message and instructions for building the
# c++ operations.
import megablocks_ops as ops


_BITS_FOR_DTYPE = {
    torch.int16: 16,
    torch.int32: 32,
    torch.int64: 64,
}

# Autograd wrapper for sort kernel.
#
# NOTE: Does not support gradients.
class SortOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, end_bit=None):
        if end_bit is None:
            end_bit = _BITS_FOR_DTYPE[x.dtype]
        x_out = torch.empty_like(x)
        iota_out = torch.empty_like(x)
        ops.sort(x, end_bit, x_out, iota_out)
        return (x_out, iota_out)
sort = SortOp.apply
