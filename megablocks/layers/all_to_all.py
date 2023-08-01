import torch

class AllToAllOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, output_split_sizes, input_split_sizes, group, async_op):
        out = torch.empty(
            (sum(output_split_sizes),) + x.shape[1:],
            device=x.device, dtype=x.dtype)

        ctx.input_shape = x.shape
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.group = group
        handle = torch.distributed.all_to_all_single(
            out, x,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=async_op)
        return out, handle

    @staticmethod
    def backward(ctx, grad, _):
        if ctx.needs_input_grad[0]:
            out = torch.empty(
                ctx.input_shape,
                device=grad.device,
                dtype=grad.dtype)
            torch.distributed.all_to_all_single(
                out, grad,
                output_split_sizes=ctx.input_split_sizes,
                input_split_sizes=ctx.output_split_sizes,
                group=ctx.group)
            return out, None, None, None, None
        return None, None, None, None, None

def all_to_all(x, output_split_sizes, input_split_sizes, group, async_op=False):
    return AllToAllOp.apply(
        x, output_split_sizes, input_split_sizes, group, async_op)
