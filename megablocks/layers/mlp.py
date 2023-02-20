from megablocks.layers import mpu
from megablocks.layers.arguments import Arguments, InitFn
from megablocks.layers.gelu import gelu
import stk
import torch
import torch.nn.functional as F


def create_expert_weights(args : Arguments,
                          num_experts : int,
                          rows : int,
                          columns : int,
                          init_method : InitFn):
    # Create the entire weight matrix such that the sampled weights will
    # not vary between data parallelism and expert model parallelism for
    # the same random seed.
    master_weights = torch.empty(
        num_experts, rows, columns,
        device=args.device,
        dtype=torch.float16 if args.fp16 else torch.float32)
    init_method(master_weights)

    if not args.moe_expert_model_parallelism:
        return master_weights

    # Calculate the number of experts on this tensor model parallel
    # partition. Note that 'num_experts' must be divisible by expert
    # parallel world size.
    world_size = mpu.get_expert_parallel_world_size(args)
    assert (num_experts % world_size) == 0
    num_experts_per_rank = num_experts // world_size
    rank = mpu.get_expert_parallel_rank(args)
    start_expert = rank * num_experts_per_rank
    end_expert = (rank + 1) * num_experts_per_rank

    # Slice the weight matrix to get the chunk for this rank.
    with torch.no_grad():
        weights = master_weights[start_expert:end_expert]
    return weights


class MLP(torch.nn.Module):

    def __init__(self, args : Arguments):
        super().__init__()
        world_size = mpu.get_expert_parallel_world_size(args)
        num_experts_per_rank = args.moe_num_experts // world_size
        
        self.w1 = torch.nn.Parameter(torch.empty(
            num_experts_per_rank,
            args.hidden_size,
            args.ffn_hidden_size,
            device=args.device,
            dtype=torch.float16 if args.fp16 else torch.float32))
        self.w2 = torch.nn.Parameter(torch.empty(
            num_experts_per_rank,
            args.ffn_hidden_size,
            args.hidden_size,
            device=args.device,
            dtype=torch.float16 if args.fp16 else torch.float32))
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
            self.w1.copy_(create_expert_weights(
                args, args.moe_num_experts, args.hidden_size,
                args.ffn_hidden_size, args.init_method))
            self.w2.copy_(create_expert_weights(
                args, args.moe_num_experts, args.ffn_hidden_size,
                args.hidden_size, args.output_layer_init_method))

    def forward(self, x):
        return torch.bmm(F.gelu(
            torch.bmm(x, self.w1), approximate="tanh"), self.w2)

    
class SparseMLP(MLP):

    def __init__(self, args : Arguments):
        super().__init__(args)

        # Re-shape the weight matrices to be how we want them for
        # the block-sparse matrix multiplication operations.
        with torch.no_grad():
            num_experts, hidden_size, ffn_hidden_size = self.w1.size()
            w1 = torch.transpose(self.w1, 1, 2).contiguous()
            self.w1 = torch.nn.Parameter(w1.view([-1, hidden_size]))
            self.w2 = torch.nn.Parameter(self.w2.view([-1, hidden_size]))
            mpu.set_expert_model_parallel_attributes(
                self.w1, args.moe_expert_model_parallelism)
            mpu.set_expert_model_parallel_attributes(
                self.w2, args.moe_expert_model_parallelism)

    def forward(self, x, topo):
        return stk.ops.dsd(gelu(
            stk.ops.sdd(x, self.w1.t(), topo)), self.w2)
        
