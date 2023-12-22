from megablocks.layers import common
from megablocks.layers import gelu
from megablocks.layers.activation_fn import act_fn
from megablocks.layers import mpu
from megablocks.layers import weight_parallel as wp
from megablocks.layers.arguments import Arguments, InitFn, DEFAULT_ACTIVATION_FN
from megablocks import turbo_util as turbo
from megablocks import grouped_gemm_util as gg
import stk
import torch
import torch.nn.functional as F
from packaging import version


class ScaleGradient(torch.autograd.Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad):
        return grad * ctx.scale, None
scale_gradient = ScaleGradient.apply


def resolve_dtensor(weight):
    if version.parse(torch.__version__) >= version.parse('2.0.0'):
        from torch.distributed._tensor import DTensor
        if isinstance(weight, DTensor):
            return weight.to_local()
    return weight


def create_moe_expert_weights(args : Arguments,
                              num_experts : int,
                              ffn_hidden_size : int,
                              hidden_size : int,
                              init_method : InitFn):
    # Create the entire weight matrix such that the sampled weights will
    # not vary between data parallelism and expert model parallelism for
    # the same random seed.
    master_weights = torch.empty(
        num_experts, ffn_hidden_size, hidden_size,
        device=args.device,
        dtype=common.dtype(args))
    init_method(master_weights)

    if not args.moe_expert_model_parallelism:
        return master_weights

    # Calculate the amount of sharding in each dimension.
    expert_sharding_degree = mpu.expert_sharding_degree(args)
    hidden_sharding_degree = mpu.hidden_sharding_degree(args)

    # Calculate the experts per rank.
    #
    # NOTE: We assign ranks to be expert parallel before going
    # tensor parallel.
    rank = mpu.get_expert_parallel_rank(args)
    expert_rank = rank % expert_sharding_degree
    num_experts_per_rank = num_experts // expert_sharding_degree
    start_expert = expert_rank * num_experts_per_rank
    end_expert = (expert_rank + 1) * num_experts_per_rank

    # Calculate the rows per rank.
    row_rank = rank // expert_sharding_degree
    num_rows_per_rank = ffn_hidden_size // hidden_sharding_degree
    start_row = row_rank * num_rows_per_rank
    end_row = (row_rank + 1) * num_rows_per_rank

    # Slice the weight matrix to get the chunk for this rank.
    with torch.no_grad():
        weights = master_weights[
            start_expert:end_expert, start_row:end_row]
    return weights


class MLP(torch.nn.Module):

    def __init__(self, args : Arguments):
        super().__init__()
        self.args = args
        expert_parallel_world_size = mpu.get_expert_parallel_world_size(args)
        experts_per_rank = mpu.experts_per_rank(args)

        self.w1 = torch.nn.Parameter(torch.empty(
            experts_per_rank,
            args.hidden_size,
            mpu.features_per_rank(args),
            device=args.device,
            dtype=common.dtype(args)))
        self.w2 = torch.nn.Parameter(torch.empty(
            experts_per_rank,
            mpu.features_per_rank(args),
            args.hidden_size,
            device=args.device,
            dtype=common.dtype(args)))
        mpu.set_expert_model_parallel_attributes(
            self.w1, args.moe_expert_model_parallelism)
        mpu.set_expert_model_parallel_attributes(
            self.w2, args.moe_expert_model_parallelism)

        # Initialize the parameters for the MLP.
        #
        # NOTE: It is important that we create the weight tensors prior
        # to creating the master weights and slicing our the piece for
        # this rank. If the master weights are created first the PyTorch
        # caching allocator appears to use the same memory block for these
        # and the slice which causes large increases in our peak memory
        # usage.
        with torch.no_grad():
            w1 = create_moe_expert_weights(
                args, args.moe_num_experts, args.ffn_hidden_size,
                args.hidden_size, args.init_method)
            self.w1.copy_(w1.transpose(1, 2).contiguous())
            self.w2.copy_(create_moe_expert_weights(
                args, args.moe_num_experts, args.ffn_hidden_size,
                args.hidden_size, args.output_layer_init_method))

        self.gradient_scale = None
        if self.args.moe_expert_model_parallelism:
            self.gradient_scale = 1 / mpu.get_expert_parallel_world_size(self.args)

    def scale_grad(self, w):
        if self.gradient_scale is None:
            return w
        return scale_gradient(w, self.gradient_scale)

    def forward(self, x):
        w1, w2 = self.scale_grad(self.w1), self.scale_grad(self.w2)
        w1, w2 = resolve_dtensor(w1), resolve_dtensor(w2)
        x = torch.bmm(x, w1)
        x = self.args.activation_fn(x)
        return torch.bmm(x, w2)


def create_dmoe_expert_weights(args : Arguments,
                               num_experts : int,
                               rows : int,
                               columns : int,
                               init_method : InitFn):
    weights = create_moe_expert_weights(
        args, num_experts, rows, columns, init_method)
    weights = weights.view([-1, columns])
    rows, columns = weights.shape

    if not args.moe_weight_parallelism:
        return weights

    # Caclculate the number of rows on this weight parallel partition.
    # 'rows' must be divisible by weight parallel world size.
    weight_parallel_world_size = mpu.get_weight_parallel_world_size(args)
    assert (rows % weight_parallel_world_size) == 0
    num_rows_per_rank = rows // weight_parallel_world_size
    rank = mpu.get_weight_parallel_rank(args)
    start_row = rank * num_rows_per_rank
    end_row = (rank + 1) * num_rows_per_rank
    return weights[start_row:end_row]


class MemoryOptimizedMLP(torch.autograd.Function):
    """Sparse MLP with manually scheduled memory reuse."""

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x, w1, w2, topo, num_input_bits, num_remat_bits, activation_fn):
        # x: [m, k], w1: [n, k], w2: [n, k]
        if (not x.is_contiguous() or not w1.is_contiguous() or
            not w2.is_contiguous()):
            raise ValueError("Expected contiguous 'x', 'w1' and 'w2'.")

        topo_tensors = (topo.row_indices,
                        topo.column_indices,
                        topo.offsets,
                        topo.column_indices_t,
                        topo.offsets_t,
                        topo.block_offsets_t)

        # Layer 0: x @ w1.t().
        sdd_out = stk.ops.sdd(x, w1.t(), topo)

        # save input tensor, quantizing if needed
        input_save_args = (x,)
        if num_input_bits != -1:
            x_q, x_scales = turbo.quantize_signed(x, num_bits=num_input_bits)
            input_save_args = (x_q, x_scales)

        # Activation function.
        if num_remat_bits == -1:
            activation_fn_out = act_fn(sdd_out, activation_fn)
            input_save_args += (sdd_out.data,)
        else:
            if activation_fn is not DEFAULT_ACTIVATION_FN:
                raise NotImplementedError(f'`num_remat_bits` != -1 not implemented for custom {activation_fn=} ({num_remat_bits=}).')
            # fused GELU into sdd_out buffer while quantizing input
            hidden_q, hidden_scales, activation_fn_out_data = turbo.quantize_signed(
                sdd_out.data, num_bits=num_remat_bits,
                op=turbo.ElemwiseOps.GELU_FORWARD, x_forward=sdd_out.data)
            activation_fn_out = sdd_out
            input_save_args += (hidden_q, hidden_scales)

        # Layer 1: x @ w2.
        dsd_out = stk.ops.dsd(activation_fn_out, w2)

        # NOTE: Save the input to the layer and the activation_fn input for
        # gradient computation. We'll re-compute the activation_fn forward
        # pass in the backward pass to avoid materializing another
        # intermediate.
        ctx.shape = topo.shape
        ctx.num_input_bits = num_input_bits
        ctx.num_remat_bits = num_remat_bits
        ctx.x_shape = x.shape
        ctx.sdd_out_shape = sdd_out.data.shape
        ctx.dtype = x.dtype
        ctx.activation_fn = activation_fn
        ctx.save_for_backward(w1, w2, *topo_tensors, *input_save_args)
        return dsd_out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, ddsd_out):
        if (not ctx.needs_input_grad[0] or
            not ctx.needs_input_grad[1] or
            not ctx.needs_input_grad[2]):
            raise ValueError("Expected all MLP inputs to need grad.")

        # unpack saved tensors; ugly because quantizing changes tensor count
        #
        dtype = ctx.dtype
        saved_tensors = ctx.saved_tensors
        w1, w2 = saved_tensors[:2]
        topo_tensors = saved_tensors[2:8]
        # either 1 or 2 tensors for MLP input after the always-present tensors
        if ctx.num_input_bits == -1:
            x = saved_tensors[8]
        else:
            x_q, x_scales = saved_tensors[8:10]
        # either 1 or 2 tensors at the end for saved GELU input / sdd output
        if ctx.num_remat_bits == -1:
            sdd_out_data = saved_tensors[-1]
        else:
            hidden_q, hidden_scales = saved_tensors[-2:]

        # rematerialize activation function output
        activation_fn = ctx.activation_fn
        activation_grad_fn = None
        if ctx.num_remat_bits == -1:
            sdd_out = stk.Matrix(ctx.shape, sdd_out_data, *topo_tensors)
            activation_fn_out, activation_grad_fn = act_fn(sdd_out, activation_fn, return_grad_fn=True)
        else:
            if activation_fn is not DEFAULT_ACTIVATION_FN:
                raise NotImplementedError(f'`num_remat_bits` != -1 not implemented for custom {activation_fn=} ({ctx.num_remat_bits=}).')
            activation_fn_out_tensor = turbo.dequantize_signed(
                hidden_q, hidden_scales, num_bits=ctx.num_remat_bits,
                op=turbo.ElemwiseOps.GELU_FORWARD,
                out_shape=ctx.sdd_out_shape, out_dtype=dtype)
            activation_fn_out = stk.Matrix(ctx.shape, activation_fn_out_tensor, *topo_tensors)

        # Compute dw2 with recomputed activation_fn output.
        dw2 = stk.ops.dsd(activation_fn_out.t(), ddsd_out)

        # Compute dactivation_fn_out.
        #
        # NOTE: We reuse the activation_fn_out allocation.
        dactivation_fn_out = activation_fn_out
        stk.backend.triton_kernels.sdd(
            ddsd_out, w2.t(),
            dactivation_fn_out.shape,
            dactivation_fn_out.data,
            dactivation_fn_out.offsets,
            dactivation_fn_out.row_indices,
            dactivation_fn_out.column_indices)

        # Compute dsdd_out.
        #
        # NOTE: This reuses the dactivation_fn_out allocation.
        if ctx.num_remat_bits == -1:
            if activation_fn is DEFAULT_ACTIVATION_FN:
                dsdd_out = gelu.gelu_backward_(dactivation_fn_out, sdd_out)
            else:
                assert activation_grad_fn is not None
                activation_grad_fn(dactivation_fn_out.data)
                dsdd_out = stk.Matrix(ctx.shape, sdd_out.data.grad, *topo_tensors)
        else:
            # confusingly, x_out is interpreted as the gradient to overwrite
            # in-place when the elemwise op is a backwards op
            ddsd_out_tensor = turbo.dequantize_signed(
                hidden_q, hidden_scales, num_bits=ctx.num_remat_bits,
                op=turbo.ElemwiseOps.GELU_BACKWARD, x_out=dactivation_fn_out.data)
            dsdd_out = stk.Matrix(ctx.shape, ddsd_out_tensor, *topo_tensors)

        # rematerialize MLP input now that we need it
        if ctx.num_input_bits != -1:
            x = turbo.dequantize_signed(
                x_q, x_scales, num_bits=ctx.num_input_bits,
                out_dtype=dtype, out_shape=ctx.x_shape)

        # Compute dw1.
        dw1 = stk.ops.dsd(dsdd_out.t(), x)

        # Compute dx.
        #
        # NOTE: This reuses the ddsd_out allocation.
        stk.backend.triton_kernels.dsd(
            dsdd_out.shape,
            dsdd_out.data,
            dsdd_out.offsets,
            dsdd_out.row_indices,
            dsdd_out.column_indices,
            dsdd_out.offsets_t,
            dsdd_out.column_indices_t,
            dsdd_out.block_offsets_t,
            False,
            w1,
            ddsd_out)
        dx = ddsd_out
        return dx, dw1, dw2, None, None, None, None

memory_optimized_mlp = MemoryOptimizedMLP.apply


class SparseMLP(torch.nn.Module):

    def __init__(self, args : Arguments):
        super().__init__()
        self.args = args
        self._num_rows_per_rank = (
            (mpu.experts_per_rank(args) * mpu.features_per_rank(args)) //
            mpu.get_weight_parallel_world_size(args)
        )

        self.w1 = torch.nn.Parameter(torch.empty(
            self._num_rows_per_rank,
            args.hidden_size,
            device=args.device,
            dtype=common.dtype(args)))
        self.w2 = torch.nn.Parameter(torch.empty(
            self._num_rows_per_rank,
            args.hidden_size,
            device=args.device,
            dtype=common.dtype(args)))

        # Initialize the parameters for the MLP.
        #
        # NOTE: It is important that we create the weight tensors prior
        # to creating the master weights and slicing our the piece for
        # this rank. If the master weights are created first the PyTorch
        # caching allocator appears to use the same memory block for these
        # and the slice which causes large increases in our peak memory
        # usage.
        with torch.no_grad():
            self.w1.copy_(create_dmoe_expert_weights(
                args, args.moe_num_experts, args.ffn_hidden_size,
                args.hidden_size, args.init_method))
            self.w2.copy_(create_dmoe_expert_weights(
                args, args.moe_num_experts, args.ffn_hidden_size,
                args.hidden_size, args.output_layer_init_method))

        self._should_set_parallelism_attribute = (
            args.moe_expert_model_parallelism or args.moe_weight_parallelism)
        mpu.set_expert_model_parallel_attributes(
            self.w1, self._should_set_parallelism_attribute)
        mpu.set_expert_model_parallel_attributes(
            self.w2, self._should_set_parallelism_attribute)

        self.gradient_scale = None
        if self.args.moe_expert_model_parallelism:
            self.gradient_scale = 1 / mpu.get_expert_parallel_world_size(self.args)

    def scale_grad(self, w):
        if self.gradient_scale is None:
            return w
        return scale_gradient(w, self.gradient_scale)

    def parallel_forward(self, x, topo):
        group = self.args.weight_parallel_group
        w1, w2 = (self.scale_grad(self.w1), self.scale_grad(self.w2))
        if self.args.memory_optimized_mlp:
            if self.args.activation_fn is not DEFAULT_ACTIVATION_FN:
                raise NotImplementedError(f'memory_optimized_weight_parallel_mlp not implemented for custom {activation_fn=}.')
            return wp.memory_optimized_weight_parallel_mlp(
                x, w1, w2, topo, group)

        # Compute the MLP.
        x = wp.sdd_nt(x, w1, topo, group)
        activation_fn_out = act_fn(x, self.args.activation_fn)
        return wp.dsd_nn(activation_fn_out, w2, group)

    def forward(self, x, topo):
        w1, w2 = self.scale_grad(self.w1), self.scale_grad(self.w2)
        w1, w2 = resolve_dtensor(w1), resolve_dtensor(w2)
        if self.args.moe_weight_parallelism:
            return self.parallel_forward(x, topo)
        elif self.args.memory_optimized_mlp:
            return memory_optimized_mlp(
                x, w1, w2, topo, self.args.quantize_inputs_num_bits,
                self.args.quantize_rematerialize_num_bits, self.args.activation_fn)

        # Compute the MLP.
        x = stk.ops.sdd(x, w1.t(), topo)
        activation_fn_out = act_fn(x, self.args.activation_fn)
        return stk.ops.dsd(activation_fn_out, w2)


class MemoryOptimizedGroupedMLP(torch.autograd.Function):
    """GroupedMLP with manually scheduled memory reuse."""

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x, w1, w2, batch_sizes, num_input_bits, num_remat_bits, activation_fn):
        # x: [m, k], w1: [n, k], w2: [n, k]
        if (not x.is_contiguous() or not w1.is_contiguous() or
            not w2.is_contiguous()):
            raise ValueError("Expected contiguous 'x', 'w1' and 'w2'.")

        # Layer 0: x @ w1.t().
        sdd_out = gg.backend.gmm(x, w1, batch_sizes, trans_b=True)

        # Save input tensor, quantizing if needed
        input_save_args = (x,)
        if num_input_bits != -1:
            x_q, x_scales = turbo.quantize_signed(x, num_bits=num_input_bits)
            input_save_args = (x_q, x_scales)

        # GeLU.
        if num_remat_bits == -1:
            activation_fn_out = activation_fn(sdd_out)
            input_save_args += (sdd_out,)
        else:
            if activation_fn is not DEFAULT_ACTIVATION_FN:
                raise NotImplementedError(f'`num_remat_bits` != -1 not implemented for custom {activation_fn=} ({num_remat_bits=}).')
            # Fused GELU into sdd_out buffer while quantizing input
            hidden_q, hidden_scales, activation_fn_out_data = turbo.quantize_signed(
                sdd_out, num_bits=num_remat_bits,
                op=turbo.ElemwiseOps.GELU_FORWARD, x_forward=sdd_out)
            activation_fn_out = sdd_out
            input_save_args += (hidden_q, hidden_scales)

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
        ctx.save_for_backward(w1, w2, batch_sizes, *input_save_args)
        return dsd_out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, ddsd_out):
        if (not ctx.needs_input_grad[0] or
            not ctx.needs_input_grad[1] or
            not ctx.needs_input_grad[2]):
            raise ValueError("Expected all MLP inputs to need grad.")

        # Unpack saved tensors; ugly because quantizing changes tensor count
        #
        dtype = ctx.dtype
        saved_tensors = ctx.saved_tensors
        w1, w2 = saved_tensors[:2]
        batch_sizes = saved_tensors[2]

        # Either 1 or 2 tensors for MLP input after the always-present tensors
        if ctx.num_input_bits == -1:
            x = saved_tensors[3]
        else:
            x_q, x_scales = saved_tensors[3:5]

        # Either 1 or 2 tensors at the end for saved GELU input / sdd output
        if ctx.num_remat_bits == -1:
            sdd_out = saved_tensors[-1]
        else:
            hidden_q, hidden_scales = saved_tensors[-2:]

        # Rematerialize activation_fn output.
        activation_fn = ctx.activation_fn
        activation_grad_fn = None
        if ctx.num_remat_bits == -1:
            with torch.set_grad_enabled(True):
                sdd_out.requires_grad = True
                activation_fn_out = activation_fn(sdd_out)
                activation_grad_fn = activation_fn_out.backward
        else:
            if activation_fn is not DEFAULT_ACTIVATION_FN:
                raise NotImplementedError(f'`num_remat_bits` != -1 not implemented for custom {activation_fn=} ({ctx.num_remat_bits=}).')
            activation_fn_out = turbo.dequantize_signed(
                hidden_q, hidden_scales, num_bits=ctx.num_remat_bits,
                op=turbo.ElemwiseOps.GELU_FORWARD,
                out_shape=ctx.sdd_out_shape, out_dtype=dtype)

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
            if activation_fn is DEFAULT_ACTIVATION_FN:
                dsdd_out = gelu.gelu_backward_(dactivation_fn_out, sdd_out)
            else:
                assert activation_grad_fn is not None
                activation_grad_fn(dactivation_fn_out)
                dsdd_out = sdd_out.grad
        else:
            # confusingly, x_out is interpreted as the gradient to overwrite
            # in-place when the elemwise op is a backwards op
            dsdd_out = turbo.dequantize_signed(
                hidden_q, hidden_scales, num_bits=ctx.num_remat_bits,
                op=turbo.ElemwiseOps.GELU_BACKWARD, x_out=dactivation_fn_out.data)

        # rematerialize MLP input now that we need it
        if ctx.num_input_bits != -1:
            x = turbo.dequantize_signed(
                x_q, x_scales, num_bits=ctx.num_input_bits,
                out_dtype=dtype, out_shape=ctx.x_shape)

        # Compute dw1.
        dw1 = gg.backend.gmm(dsdd_out, x, batch_sizes, trans_a=True)

        # Compute dx.
        #
        # NOTE: This reuses the ddsd_out allocation.
        gg.backend.gmm(dsdd_out, w1, batch_sizes, c=ddsd_out)
        dx = ddsd_out
        return dx, dw1, dw2, None, None, None, None

memory_optimized_grouped_mlp = MemoryOptimizedGroupedMLP.apply


class GroupedMLP(SparseMLP):

    def forward(self, x, tokens_per_expert):
        batch_sizes = tokens_per_expert.cpu().to(torch.long)
        w1, w2 = (self.scale_grad(self.w1), self.scale_grad(self.w2))

        # Re-shape the weights for the grouped GEMMs.
        ne = mpu.experts_per_rank(self.args)
        w1 = resolve_dtensor(w1).view(ne, -1, self.args.hidden_size)
        w2 = resolve_dtensor(w2).view(ne, -1, self.args.hidden_size)

        if self.args.moe_weight_parallelism:
            raise NotImplementedError(
                "Weight parallelism not yet supported with GroupedMLP.")

        if self.args.memory_optimized_mlp:
            return memory_optimized_grouped_mlp(
                x, w1, w2, batch_sizes,
                self.args.quantize_inputs_num_bits,
                self.args.quantize_rematerialize_num_bits,
                self.args.activation_fn)

        # Compute the MLP.
        x = gg.ops.gmm(x, w1, batch_sizes, trans_b=True)
        x = self.args.activation_fn(x)
        return gg.ops.gmm(x, w2, batch_sizes)
