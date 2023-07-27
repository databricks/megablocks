# NOTE: Torch needs to be imported before the custom
# extensions. Otherwise libc10.so cannot be found.
import torch

# TODO(tgale): Wrap this in a try-block with better
# error message and instructions for building the
# c++ operations.
import megablocks_ops as ops


def _get_default_group():
    return torch.distributed.distributed_c10d._get_default_group()


class AllToAllOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, output_split_sizes, input_split_sizes, group, async_op):
        group = _get_default_group() if group is None else group
        ctx.input_shape = x.shape
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.group = group

        out = ops.all_to_all(
            x,
            output_split_sizes,
            input_split_sizes,
            torch.distributed.get_world_size(group),
            torch.distributed.get_rank(group))
        assert not async_op
        print("going to block")
        x = ops.block_current_stream(x)
        print("done blocking")
        return out



def all_to_all(x, output_split_sizes, input_split_sizes, group, async_op=False):
    return AllToAllOp.apply(
        x, output_split_sizes, input_split_sizes, group, async_op)
