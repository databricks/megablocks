import torch
from megablocks.backend import kernels
from stk.backend.autocast import custom_fwd, custom_bwd
import turbo

# Autograd wrapper for padded_scatter kernel.
class PaddedScatterOp(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, x, indices, bin_ids, weights, bins, padded_bins, top_k, num_bits):
        saved_x = None if weights is None else x
        if num_bits == -1:
            save_inputs = (saved_x,)
        else:
            save_inputs = turbo.quantize_signed(saved_x, num_bits=num_bits)

        ctx.save_for_backward(*save_inputs, indices, bin_ids, weights, bins, padded_bins)
        ctx.top_k = top_k
        ctx.num_bits = num_bits
        return kernels.padded_scatter(
            x, indices, bin_ids, weights, bins, padded_bins, top_k)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        grad = grad.contiguous()

        if ctx.num_bits == -1:
            x = ctx.saved_tensors[0]
        else:
            x_q, x_scales = ctx.saved_tensors[:2]
            x = turbo.dequantize_signed(x_q, x_scales, num_bits=num_bits)

        indices, bin_ids, weights, bins, padded_bins = ctx.saved_tensors[-5:]
        out = kernels.padded_gather(
            grad,
            indices,
            bin_ids,
            weights,
            bins,
            padded_bins,
            ctx.top_k)

        wgrad = None
        if ctx.needs_input_grad[3]:
            wgrad = kernels.padded_scatter_wgrad(
                x,
                grad,
                indices,
                bin_ids,
                bins,
                padded_bins,
                ctx.top_k)
        return out, None, None, wgrad, None, None, None, None


# wrap apply so that num_bits is optional and defaults to no quantization
def padded_scatter(*args, num_bits: int = -1):
    return PaddedScatterOp.apply(*args, num_bits)
