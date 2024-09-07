# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

import stk.ops
import torch

from megablocks import grouped_gemm_util as gg
from megablocks.layers import common, mpu
from megablocks.layers.activation_fn import act_fn
from megablocks.layers.arguments import Arguments
from megablocks.layers.mlp import (
    SharedMLP,
    SparseMLP,
    create_dmoe_expert_weights,
    resolve_dtensor,
)


class SparseGLU(SparseMLP):

    def __init__(self, args: Arguments):
        super().__init__(args)
        self.v1 = torch.nn.Parameter(
            torch.empty(
                self._num_rows_per_rank,
                args.hidden_size,
                device=args.device,
                dtype=common.dtype(args),
            ),
        )
        with torch.no_grad():
            self.v1.copy_(
                create_dmoe_expert_weights(
                    args,
                    args.moe_num_experts,
                    args.ffn_hidden_size,
                    args.hidden_size,
                    args.init_method,
                ),
            )

        mpu.set_expert_model_parallel_attributes(
            self.v1,
            self._should_set_parallelism_attribute,
        )

    def forward(self, x, topo):
        if self.args.memory_optimized_mlp:
            raise NotImplementedError(
                'Memory optimized implementation not yet supported with GLU with sparse kernels.',
            )

        w1, v1, w2 = self.scale_grad(self.w1), self.scale_grad(self.v1,), self.scale_grad(self.w2)
        w1, v1, w2 = resolve_dtensor(w1), resolve_dtensor(v1,), resolve_dtensor(w2)

        # Compute the GLU.
        x1 = stk.ops.sdd(x, w1.t(), topo)
        x2 = stk.ops.sdd(x, v1.t(), topo)

        activation_fn_out = act_fn(x1, self.args.activation_fn)
        x1 = stk.ops.mul(activation_fn_out, x2)

        return stk.ops.dsd(x1, w2)


class MemoryOptimizedGroupedGLU(torch.autograd.Function):
    """GroupedMLP with manually scheduled memory reuse."""

    @staticmethod
    @torch.amp.autocast_mode.custom_fwd(device_type='cuda')
    def forward(ctx, x, w1, v1, w2, batch_sizes, activation_fn):
        # Cast inputs using ctx dtype from AMP
        if ctx._fwd_used_autocast:
            x = x.to(ctx._dtype)
            w1 = w1.to(ctx._dtype)
            v1 = v1.to(ctx._dtype)
            w2 = w2.to(ctx._dtype)
        # x: [m, k], w1: [n, k], v1: [n, k], w2: [n, k]
        if (not x.is_contiguous() or not w1.is_contiguous() or not v1.is_contiguous() or not w2.is_contiguous()):
            raise ValueError("Expected contiguous 'x', 'w1', 'v1' and 'w2'.")

        # Layer 0: x @ w1.t().
        assert gg.backend is not None
        sdd_out = gg.backend.gmm(x, w1, batch_sizes, trans_b=True)
        v1_out = gg.backend.gmm(x, v1, batch_sizes, trans_b=True)

        # GeLU.
        activation_fn_out = activation_fn(sdd_out) * v1_out

        # Layer 1: x @ w2.
        dsd_out = gg.backend.gmm(activation_fn_out, w2, batch_sizes)

        # NOTE: Save the input to the layer and the activation_fn input for
        # gradient computation. We'll re-compute the activation_fn forward
        # pass in the backward pass to avoid materializing another
        # intermediate.
        ctx.x_shape = x.shape
        ctx.sdd_out_shape = sdd_out.shape
        ctx.dtype = x.dtype
        ctx.activation_fn = activation_fn
        ctx.save_for_backward(w1, v1, w2, batch_sizes, x, sdd_out, v1_out)
        return dsd_out

    @staticmethod
    @torch.amp.autocast_mode.custom_bwd(device_type='cuda')
    def backward(ctx, ddsd_out):
        if (not ctx.needs_input_grad[0] or not ctx.needs_input_grad[1] or not ctx.needs_input_grad[2]):
            raise ValueError('Expected all MLP inputs to need grad.')

        # Unpack saved tensors
        # dtype = ctx.dtype
        saved_tensors = ctx.saved_tensors
        w1, v1, w2 = saved_tensors[:3]
        batch_sizes = saved_tensors[3]
        x = saved_tensors[4]
        sdd_out, v1_out = saved_tensors[5:7]

        # Rematerialize activation_fn output.
        activation_fn = ctx.activation_fn
        with torch.set_grad_enabled(True):
            sdd_out.requires_grad = True
            v1_out.requires_grad = True
            activation_fn_out = activation_fn(sdd_out) * v1_out
            activation_grad_fn = activation_fn_out.backward

        # Compute dw2 with recomputed activation_fn output.
        assert gg.backend is not None
        dw2 = gg.backend.gmm(
            activation_fn_out,
            ddsd_out,
            batch_sizes,
            trans_a=True,
        )

        # Compute dactivation_fn_out.
        #
        # NOTE: We reuse the activation_fn_out allocation.
        dactivation_fn_out = activation_fn_out
        gg.backend.gmm(
            ddsd_out,
            w2,
            batch_sizes,
            trans_b=True,
            c=dactivation_fn_out,
        )

        # Compute dsdd_out.
        #
        # NOTE: This reuses the dactivation_fn_out allocation.
        assert activation_grad_fn is not None
        activation_grad_fn(dactivation_fn_out)
        dsdd_out = sdd_out.grad
        dv1_out = v1_out.grad

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
        return dx, dw1, dv1, dw2, None, None


memory_optimized_grouped_glu = MemoryOptimizedGroupedGLU.apply


class GroupedGLU(SparseGLU):

    def forward(self, x, tokens_per_expert):
        batch_sizes = tokens_per_expert.cpu().to(torch.long)
        w1, v1, w2 = (
            self.scale_grad(self.w1),
            self.scale_grad(self.v1),
            self.scale_grad(self.w2),
        )
        w1, v1, w2 = resolve_dtensor(w1), resolve_dtensor(v1,), resolve_dtensor(w2)

        # Re-shape the weights for the grouped GEMMs.
        ne = mpu.experts_per_rank(self.args)
        w1 = w1.view(ne, -1, self.args.hidden_size)
        v1 = v1.view(ne, -1, self.args.hidden_size)
        w2 = w2.view(ne, -1, self.args.hidden_size)

        if self.args.memory_optimized_mlp:
            return memory_optimized_grouped_glu(
                x,
                w1,
                v1,
                w2,
                batch_sizes,
                self.args.activation_fn,
            )

        # Compute the MLP.
        assert gg.ops is not None
        x1 = gg.ops.gmm(x, w1, batch_sizes, trans_b=True)
        x2 = gg.ops.gmm(x, v1, batch_sizes, trans_b=True)
        x1 = self.args.activation_fn(x1) * x2
        return gg.ops.gmm(x1, w2, batch_sizes)


class SharedGLU(SharedMLP):
    """GPU for shared expert.

    Note: this is a copy -> pasta -> modify of the LLM-Foundry MPTGLU class
    """

    def __init__(self, args: Arguments):
        super().__init__(args)
        self.gate_proj = args.fc_cls(
            args.hidden_size,
            self.args.shared_expert_hidden_size,
            **self.fc_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
