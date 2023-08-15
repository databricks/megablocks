from megablocks.layers import common
from megablocks.layers import gelu
from megablocks.layers import mpu
from megablocks.layers import weight_parallel as wp
from megablocks.layers.arguments import Arguments, InitFn
import stk
import torch
import torch.nn.functional as F


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

    def forward(self, x):
        return torch.bmm(F.gelu(
            torch.bmm(x, self.w1), approximate="tanh"), self.w2)


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
    def forward(ctx, x, w1, w2, topo):
        # x: [m, k], w1: [n, k], w2: [n, k]
        if (not x.is_contiguous() or not w1.is_contiguous() or
            not w2.is_contiguous()):
            raise ValueError("Expected contiguous 'x', 'w1' and 'w2'.")

        # Layer 0: x @ w1.t().
        sdd_out = stk.ops.sdd(x, w1.t(), topo)

        # GeLU.
        gelu_out = gelu.gelu_forward(sdd_out)

        # Layer 1: x @ w2.
        dsd_out = stk.ops.dsd(gelu_out, w2)

        # NOTE: Save the input to the layer and the gelu input for
        # gradient computation. We'll re-compute the gelu forward
        # pass in the backward pass to avoid materializing another
        # intermediate.
        ctx.shape = topo.shape
        ctx.save_for_backward(
            x, w1, w2, sdd_out.data,
            topo.row_indices,
            topo.column_indices,
            topo.offsets,
            topo.column_indices_t,
            topo.offsets_t,
            topo.block_offsets_t)
        return dsd_out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, ddsd_out):
        x, w1, w2 = ctx.saved_tensors[:3]
        sdd_out = stk.Matrix(ctx.shape, *ctx.saved_tensors[3:])

        if (not ctx.needs_input_grad[0] or
            not ctx.needs_input_grad[1] or
            not ctx.needs_input_grad[2]):
            raise ValueError("Expected all MLP inputs to need grad.")

        # Compute dw2 with recomputed gelu output.
        gelu_out = gelu.gelu_forward(sdd_out)
        dw2 = stk.ops.dsd(gelu_out.t(), ddsd_out)

        # Compute dgelu_out.
        #
        # NOTE: We reuse the gelu_out allocation.
        stk.backend.triton_kernels.sdd(
            ddsd_out, w2.t(),
            sdd_out.shape,
            gelu_out.data,
            sdd_out.offsets,
            sdd_out.row_indices,
            sdd_out.column_indices)
        dgelu_out = gelu_out

        # Compute dsdd_out.
        #
        # NOTE: This reuses the dgelu_out allocation.
        dsdd_out = gelu.gelu_backward_inplace(dgelu_out, sdd_out)

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
        return dx, dw1, dw2, None

memory_optimized_mlp = MemoryOptimizedMLP.apply


class SparseMLP(torch.nn.Module):

    def __init__(self, args : Arguments):
        super().__init__()
        self.args = args
        num_rows_per_rank = (
            (mpu.experts_per_rank(args) * mpu.features_per_rank(args)) //
            mpu.get_weight_parallel_world_size(args)
        )

        self.w1 = torch.nn.Parameter(torch.empty(
            num_rows_per_rank,
            args.hidden_size,
            device=args.device,
            dtype=common.dtype(args)))
        self.w2 = torch.nn.Parameter(torch.empty(
            num_rows_per_rank,
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

        should_set_attribute = (
            args.moe_expert_model_parallelism or args.moe_weight_parallelism)
        mpu.set_expert_model_parallel_attributes(
            self.w1, should_set_attribute)
        mpu.set_expert_model_parallel_attributes(
            self.w2, should_set_attribute)

    def parallel_forward(self, x, topo):
        x = wp.sdd_nt(x, self.w1, topo, self.args.weight_parallel_group)
        return wp.dsd_nn(gelu.gelu(x), self.w2, self.args.weight_parallel_group)

    def forward(self, x, topo):
        if self.args.moe_weight_parallelism:
            return self.parallel_forward(x, topo)
        return memory_optimized_mlp(x, self.w1, self.w2, topo)
