from megablocks.layers import common
from megablocks.layers import gelu
from megablocks.layers.mlp import SparseMLP, create_dmoe_expert_weights
from megablocks.layers import mpu
from megablocks.layers.arguments import Arguments, InitFn
from megablocks import grouped_gemm_util as gg
import stk
import torch
import torch.nn.functional as F


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

    def forward(self, x, topo):
        w1, v1, w2 = (self.scale_grad(self.w1), self.scale_grad(self.v1), self.scale_grad(self.w2))
        if self.args.moe_weight_parallelism:
            raise NotImplementedError("Weight parallelism not yet supported with GLU.")
        elif self.args.memory_optimized_mlp:
            raise NotImplementedError("Memory optimized implementation not yet supported with GLU.")

        # Compute the GLU.
        x1 = stk.ops.sdd(x, w1.t(), topo)
        x2 = stk.ops.sdd(x, v1.t(), topo)

        x1._data = gelu.gelu(x1)._data * x2._data

        return stk.ops.dsd(x1, w2)
    
class GroupedGLU(SparseGLU):
    def forward(self, x, tokens_per_expert):
        batch_sizes = tokens_per_expert.cpu().to(torch.long)
        w1, v1, w2 = (self.scale_grad(self.w1), self.scale_grad(self.v1), self.scale_grad(self.w2))

        # Re-shape the weights for the grouped GEMMs.
        ne = mpu.experts_per_rank(self.args)
        w1 = w1.view(ne, -1, self.args.hidden_size)
        v1 = v1.view(ne, -1, self.args.hidden_size)
        w2 = w2.view(ne, -1, self.args.hidden_size)

        if self.args.moe_weight_parallelism:
            raise ValueError(
                "Weight parallelism not yet supported with GroupedMLP.")

        if self.args.memory_optimized_mlp:
            raise ValueError(
                "Memory optimized implementation not yet supported with GroupedGLU.")

        # Compute the MLP.
        x1 = gg.ops.gmm(x, w1, batch_sizes, trans_b=True)
        x2 = gg.ops.gmm(x, v1, batch_sizes, trans_b=True)
        x1 = F.gelu(x1, approximate="tanh") * x2
        return gg.ops.gmm(x1, w2, batch_sizes)
