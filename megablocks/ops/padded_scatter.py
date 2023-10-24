import torch
from megablocks.backend import kernels
from megablocks import turbo_util as turbo
from stk.backend.autocast import custom_fwd, custom_bwd


# Autograd wrapper for padded_scatter kernel.
class PaddedScatterOp(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, x, indices, bin_ids, weights, bins, padded_bins, top_k,
                num_bits):
        saved_x = x if ctx.needs_input_grad[3] else None
        if saved_x is None:
            save_inputs = []
        elif num_bits == -1:
            save_inputs = (saved_x,)
        else:
            x_q, x_scales = turbo.quantize_signed(saved_x, num_bits=num_bits)
            save_inputs = (x_q, x_scales)

        ctx.save_for_backward(
            indices, bin_ids, weights, bins, padded_bins, *save_inputs)
        ctx.top_k = top_k
        ctx.x_shape = x.shape
        ctx.num_bits = num_bits
        return kernels.padded_scatter(
            x, indices, bin_ids, weights, bins, padded_bins, top_k)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        grad = grad.contiguous()
        saved_tensors = ctx.saved_tensors

        indices, bin_ids, weights, bins, padded_bins = saved_tensors[:5]
        dgrad = None
        if ctx.needs_input_grad[0]:
            dgrad = kernels.padded_gather(
                grad,
                indices,
                bin_ids,
                weights,
                bins,
                padded_bins,
                ctx.top_k)

        wgrad = None
        if ctx.needs_input_grad[3]:  # need wgrad
            if ctx.num_bits == -1:  # input saved without quantization
                x = saved_tensors[-1]
            else:  # dequantize input
                x_q, x_scales = saved_tensors[-2:]
                x = turbo.dequantize_signed(
                    x_q, x_scales, num_bits=ctx.num_bits, out_shape=ctx.x_shape)

            wgrad = kernels.padded_scatter_wgrad(
                x,
                grad,
                indices,
                bin_ids,
                bins,
                padded_bins,
                ctx.top_k)
        return dgrad, None, None, wgrad, None, None, None, None


# wrap apply so that num_bits is optional and defaults to no quantization
def padded_scatter(x: torch.Tensor,
                   indices: torch.Tensor,
                   bin_ids: torch.Tensor,
                   weights: torch.Tensor,
                   bins: torch.Tensor,
                   padded_bins: torch.Tensor,
                   top_k: int,
                   num_bits: int = -1):
    return PaddedScatterOp.apply(x, indices, bin_ids, weights, bins,
                                 padded_bins, top_k, num_bits)
