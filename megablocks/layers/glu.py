from megablocks.layers import common
from megablocks.layers import gelu
from megablocks.layers.mlp import create_dmoe_expert_weights, scale_gradient
from megablocks.layers import mpu
from megablocks.layers import weight_parallel as wp
from megablocks.layers.arguments import Arguments, InitFn
from megablocks import turbo_util as turbo
from megablocks import grouped_gemm_util as gg
import stk
import torch
import torch.nn.functional as F


class GLU(torch.nn.Module):

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

        self.v1 = torch.nn.Parameter(torch.empty(
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
            self.v1.copy_(create_dmoe_expert_weights(
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
            self.v1, should_set_attribute)
        mpu.set_expert_model_parallel_attributes(
            self.w2, should_set_attribute)

        self.gradient_scale = None
        if self.args.moe_expert_model_parallelism:
            self.gradient_scale = 1 / mpu.get_expert_parallel_world_size(self.args)

    def scale_grad(self, w):
        if self.gradient_scale is None:
            return w
        return scale_gradient(w, self.gradient_scale)

    def forward(self, x, topo):
        w1, v1, w2 = (self.scale_grad(self.w1), self.scale_grad(self.v1), self.scale_grad(self.w2))
        if self.args.moe_weight_parallelism:
            raise NotImplementedError("Currently not supported.")
        elif self.args.memory_optimized_mlp:
            raise NotImplementedError("Currently not supported.")

        # Compute the MLP.
        x1 = stk.ops.sdd(x, w1.t(), topo)
        x2 = stk.ops.sdd(x, v1.t(), topo)

        element_wise_res = stk.ops.to_dense(gelu.gelu(x1)) * stk.ops.to_dense(x2)
        sparse_res = stk.ops.to_sparse(element_wise_res, 128)

        return stk.ops.dsd(sparse_res, w2)