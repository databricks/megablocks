# NOTE: Torch needs to be imported before the custom
# extensions. Otherwise libc10.so cannot be found.
import torch

# TODO(tgale): Wrap this in a try-block with better
# error message and instructions for building the
# c++ operations.
import megablocks_ops as ops


_COMM_INITIALIZED = False


def _is_comm_initialized():
    global _COMM_INITIALIZED
    return _COMM_INITIALIZED


def _initialize_comm(group):
    # NOTE: To create a NCCL communicator we need to broadcast a unique id to
    # all of the ranks. The process group that we receive here is a NCLL group,
    # so we copy the id to the GPU before broadcasting and then back to the CPU
    # to create the (internal) NCCL communicator.
    unique_id = ops.nccl_get_unique_id().cuda()
    torch.distributed.broadcast(unique_id, 0, group=group)
    ops.create_nccl_comm(
        unique_id.cpu(),
        torch.distributed.get_world_size(group),
        torch.distributed.get_rank(group))

    global _COMM_INITIALIZED
    _COMM_INITIALIZED = True


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

        if not _is_comm_initialized():
            _initialize_comm(group)

        out = ops.all_to_all(
            x,
            output_split_sizes,
            input_split_sizes)
        assert not async_op
        x = ops.block_current_stream(x)
        return out



def all_to_all(x, output_split_sizes, input_split_sizes, group, async_op=False):
    return AllToAllOp.apply(
        x, output_split_sizes, input_split_sizes, group, async_op)
