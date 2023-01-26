# NOTE: Torch needs to be imported before the custom
# extensions. Otherwise libc10.so cannot be found.
import torch

# TODO(tgale): Wrap this in a try-block with better
# error message and instructions for building the
# c++ operations.
import megablocks_ops as ops

# Autograd wrapper for topology kernel.
#
# NOTE: Does not support gradients.
class TopologyOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                padded_bins,
                block_size,
                output_block_rows,
                output_block_columns):
        out = torch.empty(output_block_rows * output_block_columns,
                          dtype=torch.int16,
                          device=padded_bins.device)
        ops.indices(padded_bins,
                    block_size,
                    output_block_rows,
                    output_block_columns,
                    out)
        return out
topology = TopologyOp.apply
