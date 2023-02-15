from megablocks.layers import mpu
from megablocks.layers import router
from megablocks.layers.all_to_all import all_to_all
from megablocks.layers.arguments import Arguments, InitFn
import megablocks.ops as ops
import numpy as np
import torch
import torch.nn.functional as F


_LOAD_BALANCING_LOSS = []


def save_load_balancing_loss(loss):
    global _LOAD_BALANCING_LOSS
    _LOAD_BALANCING_LOSS.append(loss)


def get_load_balancing_loss():
    global _LOAD_BALANCING_LOSS
    return _LOAD_BALANCING_LOSS


def clear_load_balancing_loss():
    global _LOAD_BALANCING_LOSS
    _LOAD_BALANCING_LOSS.clear()


def batched_load_balancing_loss(args : Arguments):
    # tokens_per_expert[i].shape = (num_experts)
    # expert_scores[i].shape = (tokens, num_experts)
    tokens_per_expert, expert_scores = zip(*get_load_balancing_loss())
    num_layers_per_pipeline_stage = (
        args.num_layers // args.pipeline_model_parallel_size)
    if args.num_layers_per_virtual_pipeline_stage is not None:
        num_layers_per_pipeline_stage = args.num_layers_per_virtual_pipeline_stage

    if len(tokens_per_expert) != num_layers_per_pipeline_stage:
        raise ValueError(
            f"Expected {num_layers_per_pipeline_stage} token_per_experts "
            f"but found {len(tokens_per_expert)}.\nnum_layers = "
            f"{args.num_layers}\npipeline_model_parallel_size = "
            f"{args.pipeline_model_parallel_size}\n"
            "num_layers_per_virtual_pipeline_stage"
            f" = {args.num_layers_per_virtual_pipeline_stage}")
    if len(expert_scores) != num_layers_per_pipeline_stage:
        raise ValueError(
            f"Expected {num_layers_per_pipeline_stage} expert_scores "
            f"but found {len(tokens_per_expert)}.\nnum_layers = "
            f"{args.num_layers}\npipeline_model_parallel_size = "
            f"{args.pipeline_model_parallel_size}\n"
            "num_layers_per_virtual_pipeline_stage"
            f" = {args.num_layers_per_virtual_pipeline_stage}")

    # Verify the shape of the tokens_per_expert and expert_scores tensors.
    assert all([
        x.ndim == 1 and x.numel() == args.moe_num_experts
        for x in tokens_per_expert
    ])

    tokens = expert_scores[0].shape[0]
    assert all([
        (x.ndim == 2 and x.shape[1] == args.moe_num_experts and
         x.shape[0] == tokens) for x in expert_scores
    ])


    # Concatenate the contributions of each layer and convert to
    # the correct types and formats for the dot product.
    if args.moe_lbl_in_fp32:
        tokens_per_expert = torch.cat(tokens_per_expert).float()
        expert_scores = torch.cat(expert_scores, dim=1).float().mean(dim=0)
    else:
        tokens_per_expert = torch.cat(tokens_per_expert).half()
        expert_scores = torch.cat(expert_scores, dim=1).mean(dim=0)

    expected_values = num_layers_per_pipeline_stage * args.moe_num_experts
    assert tokens_per_expert.numel() == expected_values
    assert expert_scores.numel() == expected_values

    # Calculate the total scale across all factors.
    #
    # loss_weight * num_experts / (num_layers * tokens * top_k)
    scale_numerator = (
        args.moe_num_experts *
        args.moe_loss_weight
    )
    scale_denominator = (
        args.num_layers *
        tokens *
        args.moe_top_k
    )
    scale = scale_numerator / scale_denominator
    return scale * torch.dot(tokens_per_expert, expert_scores)


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

    if not args.expert_model_parallelism:
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


class MoE(torch.nn.Module):

    def __init__(self, args : Arguments):
        super(MoE, self).__init__()
        self.args = args

        # Calculate the number of experts in total and the number of experts
        # owned by this rank.
        world_size = mpu.get_expert_parallel_world_size(args)
        self.num_experts = args.moe_num_experts
        self.num_experts_per_rank = self.num_experts // world_size

        # Calculate the number of bits needed to represent the expert indices
        # so that we can pass it to radix sort.
        self.sort_end_bit = max(int(np.ceil(np.log2(self.num_experts))), 1)

        # Token router.
        self.router = router.LearnedRouter(args)

        # Learned MLP parameters.
        self.w1 = torch.nn.Parameter(torch.empty(
            self.num_experts_per_rank,
            args.hidden_size,
            args.ffn_hidden_size,
            device=args.device,
            dtype=torch.float16 if args.fp16 else torch.float32))
        self.w2 = torch.nn.Parameter(torch.empty(
            self.num_experts_per_rank,
            args.ffn_hidden_size,
            args.hidden_size,
            device=args.device,
            dtype=torch.float16 if args.fp16 else torch.float32))
        mpu.set_expert_model_parallel_attributes(
            self.w1, args.expert_model_parallelism)
        mpu.set_expert_model_parallel_attributes(
            self.w2, args.expert_model_parallelism)

        # Note that the output bias is not parallelized with expert
        # model parallelism.
        self.bias = torch.nn.Parameter(torch.empty(
            1, 1, args.hidden_size,
            device=args.device,
            dtype=torch.float16 if args.fp16 else torch.float32))

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
                args, self.num_experts, args.hidden_size,
                args.ffn_hidden_size, args.init_method))
            self.w2.copy_(create_expert_weights(
                args, self.num_experts, args.ffn_hidden_size,
                args.hidden_size, args.output_layer_init_method))
        torch.nn.init.zeros_(self.bias)

        # Select the forward function for the operating mode.
        self.forward_fn = (
            self.parallel_forward_once if
            args.expert_model_parallelism else
            self.forward_once)

    def expert_capacity(self, tokens):
        tokens_per_expert = tokens / self.num_experts_per_rank
        return int(self.args.moe_capacity_factor * tokens_per_expert)

    def load_balancing_loss(self, tokens_per_expert, expert_scores):
        """Calculate the load balancing loss contribution."""
        assert len(expert_scores.size()) == 2
        tokens, num_experts = expert_scores.size()
        assert num_experts == self.num_experts
        assert len(tokens_per_expert.size()) == 1
        num_experts, = tokens_per_expert.size()
        assert num_experts == self.num_experts
        scale = self.num_experts / (tokens * self.args.moe_top_k)
        return scale * torch.dot(
            tokens_per_expert.half(),
            expert_scores.mean(dim=0))

    def indices_and_bins(self, top_expert):
        # Sort the expert ids to produce the scatter/gather
        # indices for the permutation.
        #
        # TODO(tgale): Is it worth doing this conversion to 32-bit
        # prior? Could we place the `torch.max` operation to return
        # 32-bit expert indices?
        top_expert = top_expert.int()
        bin_ids, indices = ops.sort(top_expert, self.sort_end_bit)

        # Histogram the expert ids to identify the number of
        # tokens routed to each expert.
        #
        # TODO(tgale): Does the sorted data produce a more favorable
        # data distribution for histogram? Or is the op parallelism
        # worth more?
        tokens_per_expert = ops.histogram(top_expert, self.num_experts)

        # Calculate the bin bounds for the sorted tokens.
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)
        bins = bins.view(1) if not len(bins.size()) else bins
        return indices, bin_ids, bins, tokens_per_expert

    def compute(self, x):
        return torch.bmm(F.gelu(torch.bmm(x, self.w1), approximate=True), self.w2)

    def permute_and_compute(
            self,
            x,
            tokens_per_expert, # unused
            indices,
            bin_ids, # unused
            bins,
            expert_capacity):
        # Route the tokens for MoE computation.
        x = x.view(-1, x.shape[-1])
        x = ops.binned_gather(x, indices, bins, expert_capacity)

        # Perform the expert computation. Note that we don't
        # use biases for these linear operations.
        x = self.compute(x)

        # Un-route the data for the MoE output.
        return ops.binned_scatter(x, indices, bins)

    def forward_once(self, x, top_expert):
        with torch.no_grad():
            indices, bin_ids, bins, tokens_per_expert = (
                self.indices_and_bins(top_expert))

            # If expert_capacity is set to zero, set the number of tokens
            # per expert to the maximum we need to avoid dropping tokens.
            sl, bs, hs = x.size()
            expert_capacity = self.expert_capacity(sl * bs)
            if expert_capacity == 0:
                expert_capacity = torch.max(tokens_per_expert)
        x = self.permute_and_compute(
            x, tokens_per_expert, indices, bin_ids, bins, expert_capacity)
        return x, tokens_per_expert

    def parallel_forward_once(self, x, top_expert):
        # NOTE: This function implements the same computation as forward_once
        # but with expert model parallelism.
        #
        # 1. Permute the tokens locally so that they are grouped by their
        # expert assignments. This allows us to transfer all of the tokens
        # for a remote device in one communication primitive.
        #
        # 2. Permute the tokens across the expert parallel devices. After
        # this is completed each device has all of the tokens assigned to
        # its set of experts in its local HBM.
        #
        # 3. Permute the tokens locally so that they are grouped by their
        # expert assignement. After the distributed permutation the tokens
        # are grouped by which device they came from. We re-order them
        # locally to allow for efficient computation.
        #
        # After this series of permutations we compute the linear layers
        # and then repeat these three steps in reverse to produce the final
        # output.
        #
        # Compute the mapping of local tokens to experts.
        with torch.no_grad():
            indices, bin_ids, bins, tokens_per_expert = (
                self.indices_and_bins(top_expert))

            # Pass token count information to the device on which the
            # target expert resides.
            parallel_tokens_per_expert = torch.empty_like(
                tokens_per_expert)
            tpe_handle = torch.distributed.all_to_all_single(
                parallel_tokens_per_expert,
                tokens_per_expert,
                group=self.args.expert_parallel_group,
                async_op=True)

        # Permute locally and without any padding so that tokens for each
        # parallel device are stored contiguously.
        #
        # TODO(tgale): We can tune these kernels for this special case by
        # skipping the memset if tokens == padded_tokens and also taking
        # in an optional padded_tokens rather than copying it from the
        # device.
        #
        # This view updates the shape of the tensor from [sl, bs, hs] to
        # [sl * bs, hs] prior to the permutation.
        x = x.view(-1, x.shape[-1])
        x = ops.padded_gather(x, indices, bin_ids, bins, bins)

        # Compute the number of tokens that will be received from each
        # device and permute the input data across the devices.
        with torch.no_grad():
            tpe_handle.wait()
            world_size = mpu.get_expert_parallel_world_size(self.args)

            # Reshape to [world_size, num_experts_per_rank].
            tokens_per_expert = tokens_per_expert.view(world_size, -1)
            parallel_tokens_per_expert = (
                parallel_tokens_per_expert.view(world_size, -1))

            # TODO(tgale): It might be faster to do this on the GPU and
            # then communicate the results back to the host.
            send_counts = tokens_per_expert.cpu().sum(dim=-1)
            recv_counts = parallel_tokens_per_expert.cpu().sum(dim=-1)

            # Convert the send/recv counts to lists.
            send_counts = send_counts.tolist()
            recv_counts = recv_counts.tolist()
            tokens_received = sum(recv_counts)

            # After we do the cross-device permutation we have the tokens on the
            # correct device but not yet grouped by expert because we received
            # tokens from each device as contiguous chunks. To group the tokens
            # for expert computation we'll do one more local permutation. The
            # rest of this torch.no_grad() scope sets up the indices and bins
            # for this permutation.
            replicate_bins = ops.inclusive_cumsum(
                parallel_tokens_per_expert.flatten(), 0)
            replicate_bins = (
                replicate_bins.view(1)
                if not len(replicate_bins.size())
                else replicate_bins
            )

            # Construct the expert indices for the permuted tokens.
            parallel_top_expert = torch.remainder(
                torch.arange(
                    self.num_experts, dtype=torch.int32, device=indices.device),
                self.num_experts_per_rank,
            )
            parallel_top_expert = ops.replicate(
                parallel_top_expert.unsqueeze(dim=0),
                replicate_bins, tokens_received).flatten()

            # TODO(tgale): The sort_end_bit here can be reduced.
            parallel_bin_ids, parallel_indices = ops.sort(
                parallel_top_expert, self.sort_end_bit)

            # Calculate the bins boundaries from the token counts.
            parallel_tokens_per_expert = parallel_tokens_per_expert.sum(
                dim=0, dtype=torch.int)
            parallel_bins = ops.inclusive_cumsum(
                parallel_tokens_per_expert, 0)
            parallel_bins = (
                parallel_bins.view(1)
                if not len(parallel_bins.size())
                else parallel_bins
            )

            # If expert_capacity is set to zero, set the number of tokens
            # per expert to the maximum we need to avoid dropping tokens.
            tokens, hs = x.size()
            expert_capacity = self.expert_capacity(tokens)
            if expert_capacity == 0:
                expert_capacity = torch.max(parallel_tokens_per_expert)

        # Permute the tokens across the devices.
        parallel_x = all_to_all(
            x, recv_counts, send_counts,
            self.args.expert_parallel_group)

        # Locally permute the tokens and perform the expert computation.
        parallel_x = self.permute_and_compute(
            parallel_x,
            parallel_tokens_per_expert,
            parallel_indices,
            parallel_bin_ids,
            parallel_bins,
            expert_capacity)

        # Un-permute the tokens across the devices.
        x = all_to_all(
            parallel_x, send_counts, recv_counts,
            self.args.expert_parallel_group)

        # Un-permute locally to setup for the next series of operations.
        x = ops.padded_scatter(x, indices, bin_ids, bins, bins)
        return x, tokens_per_expert.flatten()

    def forward(self, x):
        sl, bs, hs = x.size()

        # Compute the top-1 expert routing.
        scores, expert_weights, top_experts = self.router(x)

        # Simplified code-path for the common case of top_k == 1.
        if self.args.moe_top_k == 1:
            x, tokens_per_expert = self.forward_fn(x, top_experts)
            x = x * expert_weights.view(-1, 1)
            save_load_balancing_loss((tokens_per_expert, scores))
            return x.view(sl, bs, hs), self.bias

        # Chunk the routing/weight data for each 'k'.
        top_experts = top_experts.chunk(self.args.moe_top_k, dim=-1)
        expert_weights = expert_weights.chunk(self.args.moe_top_k, dim=-1)

        # Compute the FFN layers for each 'k'.
        x, tokens_per_expert = zip(*[
            self.forward_fn(x, routing.squeeze())
            for routing in top_experts
        ])

        # Weight and combine the expert outputs.
        #
        # TODO(tgale): We should fused this to save memory and
        # bandwidth. We can likely fuse the scale + add for top-k
        # routing.
        #
        # Sum the token counts for each expert.
        x = sum([
            out * weight.view(-1, 1)
            for (out, weight) in zip(x, expert_weights)
        ])
        tokens_per_expert = sum(tokens_per_expert)

        # Save the matrices needed for load balancing loss computation.
        save_load_balancing_loss((tokens_per_expert, scores))
        return x.view(sl, bs, hs), self.bias
