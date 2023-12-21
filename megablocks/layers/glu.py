from megablocks.layers import common
from megablocks.layers.activation_fn import act_fn
from megablocks.layers.mlp import SparseMLP, create_dmoe_expert_weights
from megablocks.layers import mpu
from megablocks.layers.arguments import Arguments, DEFAULT_ACTIVATION_FN
from megablocks import turbo_util as turbo
from megablocks import grouped_gemm_util as gg
import stk
import torch


class SparseGLU(SparseMLP):

    def __init__(self, args : Arguments):
        super().__init__(args)
        self.v1 = torch.nn.Parameter(torch.empty(
            self._num_rows_per_rank,
            args.hidden_size,
            device=args.device,
            dtype=common.dtype(args)))
        with torch.no_grad():
            self.v1.copy_(create_dmoe_expert_weights(
                args, args.moe_num_experts, args.ffn_hidden_size,
                args.hidden_size, args.init_method))

        mpu.set_expert_model_parallel_attributes(
            self.v1, self._should_set_parallelism_attribute)

        if self.args.moe_weight_parallelism:
            raise NotImplementedError("Weight parallelism not yet supported with GLU.")

    def forward(self, x, topo):
        if self.args.memory_optimized_mlp:
            raise NotImplementedError("Memory optimized implementation not yet supported with GLU with sparse kernels.")

        w1, v1, w2 = (self.scale_grad(self.w1), self.scale_grad(self.v1), self.scale_grad(self.w2))

        # Compute the GLU.
        x1 = stk.ops.sdd(x, w1.t(), topo)
        x2 = stk.ops.sdd(x, v1.t(), topo)

        activation_fn_out = act_fn(x1, self.args.activation_fn)
        x1 = stk.ops.mul(activation_fn_out, x2)

        return stk.ops.dsd(x1, w2)

class MemoryOptimizedGroupedGLU(torch.autograd.Function):
    """GroupedMLP with manually scheduled memory reuse."""

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x, w1, v1, w2, batch_sizes, num_input_bits, num_remat_bits, activation_fn):
        # x: [m, k], w1: [n, k], v1: [n, k], w2: [n, k]
        if (not x.is_contiguous() or not w1.is_contiguous() or
            not v1.is_contiguous() or not w2.is_contiguous()):
            raise ValueError("Expected contiguous 'x', 'w1', 'v1' and 'w2'.")

        # Layer 0: x @ w1.t().
        sdd_out = gg.backend.gmm(x, w1, batch_sizes, trans_b=True)
        v1_out = gg.backend.gmm(x, v1, batch_sizes, trans_b=True)

        # Save input tensor, quantizing if needed
        input_save_args = (x,)
        if num_input_bits != -1:
            x_q, x_scales = turbo.quantize_signed(x, num_bits=num_input_bits)
            input_save_args = (x_q, x_scales)

        # GeLU.
        if num_remat_bits == -1:
            activation_fn_out = activation_fn(sdd_out) * v1_out
            input_save_args += (sdd_out, v1_out,)
        else:
            if activation_fn is not DEFAULT_ACTIVATION_FN:
                raise NotImplementedError(f'`num_remat_bits` != -1 not implemented for custom {activation_fn=} ({num_remat_bits=}).')
            # Fused GELU into sdd_out buffer while quantizing input
            hidden_q_sdd, hidden_scales_sdd, _ = turbo.quantize_signed(
                sdd_out, num_bits=num_remat_bits,
                op=turbo.ElemwiseOps.GELU_FORWARD, x_forward=sdd_out)
            activation_fn_out = sdd_out * v1_out
            hidden_q_v1, hidden_scales_v1, _ = turbo.quantize_signed(
                v1_out, num_bits=num_remat_bits)
            input_save_args += (hidden_q_sdd, hidden_scales_sdd, hidden_q_v1, hidden_scales_v1)
            raise NotImplementedError(f'Activation compression of hidden state not implemented. Set `num_remat_bits = -1`.')

        # Layer 1: x @ w2.
        dsd_out = gg.backend.gmm(activation_fn_out, w2, batch_sizes)

        # NOTE: Save the input to the layer and the activation_fn input for
        # gradient computation. We'll re-compute the activation_fn forward
        # pass in the backward pass to avoid materializing another
        # intermediate.
        ctx.num_input_bits = num_input_bits
        ctx.num_remat_bits = num_remat_bits
        ctx.x_shape = x.shape
        ctx.sdd_out_shape = sdd_out.shape
        ctx.dtype = x.dtype
        ctx.activation_fn = activation_fn
        ctx.save_for_backward(w1, v1, w2, batch_sizes, *input_save_args)
        return dsd_out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, ddsd_out):
        if (not ctx.needs_input_grad[0] or
            not ctx.needs_input_grad[1] or
            not ctx.needs_input_grad[2]):
            raise ValueError("Expected all MLP inputs to need grad.")

        # Unpack saved tensors; ugly because quantizing changes tensor count
        dtype = ctx.dtype
        saved_tensors = ctx.saved_tensors
        w1, v1, w2 = saved_tensors[:3]
        batch_sizes = saved_tensors[3]

        # Either 1 or 2 tensors for MLP input after the always-present tensors
        if ctx.num_input_bits == -1:
            x = saved_tensors[4]
        else:
            x_q, x_scales = saved_tensors[4:6]

        # Either 1 or 4 tensors at the end for saved GELU input / sdd output
        if ctx.num_remat_bits == -1:
            sdd_out, v1_out = saved_tensors[-2:]
        else:
            hidden_q_sdd, hidden_scales_sdd, hidden_q_v1, hidden_scales_v1 = saved_tensors[-4:]

        # Rematerialize activation_fn output.
        activation_fn = ctx.activation_fn
        activation_grad_fn = None
        if ctx.num_remat_bits == -1:
            with torch.set_grad_enabled(True):
                sdd_out.requires_grad = True
                v1_out.requires_grad = True
                activation_fn_out = activation_fn(sdd_out) * v1_out
                activation_grad_fn = activation_fn_out.backward
        else:
            if activation_fn is not DEFAULT_ACTIVATION_FN:
                raise NotImplementedError(f'`num_remat_bits` != -1 not implemented for custom {activation_fn=} ({num_remat_bits=}).')
            sdd_out = turbo.dequantize_signed(
                hidden_q_sdd, hidden_scales_sdd, num_bits=ctx.num_remat_bits,
                op=turbo.ElemwiseOps.GELU_FORWARD,
                out_shape=ctx.sdd_out_shape, out_dtype=dtype)
            v1_out = turbo.dequantize_signed(
                hidden_q_v1, hidden_scales_v1, num_bits=ctx.num_remat_bits,
                out_shape=ctx.sdd_out_shape, out_dtype=dtype)
            activation_fn_out = sdd_out * v1_out

        # Compute dw2 with recomputed activation_fn output.
        dw2 = gg.backend.gmm(
            activation_fn_out, ddsd_out, batch_sizes, trans_a=True)

        # Compute dactivation_fn_out.
        #
        # NOTE: We reuse the activation_fn_out allocation.
        dactivation_fn_out = activation_fn_out
        gg.backend.gmm(
            ddsd_out, w2, batch_sizes, trans_b=True, c=dactivation_fn_out)

        # Compute dsdd_out.
        #
        # NOTE: This reuses the dactivation_fn_out allocation.
        if ctx.num_remat_bits == -1:
            assert activation_grad_fn is not None
            activation_grad_fn(dactivation_fn_out)
            dsdd_out = sdd_out.grad
            dv1_out = v1_out.grad
        else:
            # confusingly, x_out is interpreted as the gradient to overwrite
            # in-place when the elemwise op is a backwards op
            dsdd_out = turbo.dequantize_signed(
                hidden_q_sdd, hidden_scales_sdd, num_bits=ctx.num_remat_bits,
                op=turbo.ElemwiseOps.GELU_BACKWARD, x_out=dactivation_fn_out.dat * v1_out)
            dv1_out = turbo.dequantize_signed(
                hidden_q_v1, hidden_scales_v1, num_bits=ctx.num_remat_bits,
                op=turbo.ElemwiseOps.IDENTITY, x_out=dactivation_fn_out.dat * sdd_out)

        # rematerialize MLP input now that we need it
        if ctx.num_input_bits != -1:
            x = turbo.dequantize_signed(
                x_q, x_scales, num_bits=ctx.num_input_bits,
                out_dtype=dtype, out_shape=ctx.x_shape)

        # Compute dw1.
        dw1 = gg.backend.gmm(dsdd_out, x, batch_sizes, trans_a=True)

        # Compute dv1.
        dv1 = gg.backend.gmm(dv1_out, x, batch_sizes, trans_a=True)

        # Compute dx.
        #
        # NOTE: This reuses the ddsd_out allocation.
        dx = ddsd_out
        gg.backend.gmm(dsdd_out, w1, batch_sizes, c=dx)
        dx += gg.backend.gmm(dv1_out, v1, batch_sizes)
        return dx, dw1, dv1, dw2, None, None, None, None

memory_optimized_grouped_glu = MemoryOptimizedGroupedGLU.apply


class GroupedGLU(SparseGLU):
    def forward(self, x, tokens_per_expert):
        batch_sizes = tokens_per_expert.cpu().to(torch.long)
        w1, v1, w2 = (self.scale_grad(self.w1), self.scale_grad(self.v1), self.scale_grad(self.w2))

        # Re-shape the weights for the grouped GEMMs.
        ne = mpu.experts_per_rank(self.args)
        w1 = w1.view(ne, -1, self.args.hidden_size)
        v1 = v1.view(ne, -1, self.args.hidden_size)
        w2 = w2.view(ne, -1, self.args.hidden_size)

        if self.args.memory_optimized_mlp:
            return memory_optimized_grouped_glu(
                x, w1, v1, w2, batch_sizes,
                self.args.quantize_inputs_num_bits,
                self.args.quantize_rematerialize_num_bits,
                self.args.activation_fn)

        # Compute the MLP.
        x1 = gg.ops.gmm(x, w1, batch_sizes, trans_b=True)
        x2 = gg.ops.gmm(x, v1, batch_sizes, trans_b=True)
        x1 = self.args.activation_fn(x1) * x2
        return gg.ops.gmm(x1, w2, batch_sizes)
