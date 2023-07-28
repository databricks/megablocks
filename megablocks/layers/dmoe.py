from megablocks import benchmark_util
from megablocks.layers import common
from megablocks.layers import mlp
from megablocks.layers import moe
from megablocks.layers.arguments import Arguments
import megablocks.ops as ops
import numpy as np
import stk
import torch


def promote_scalar(x):
    return x.view(1) if not len(x.size()) else x


class dMoE(moe.MoE):

    def __init__(self, args : Arguments):
        super(dMoE, self).__init__(args)
        self.hidden_size = args.hidden_size
        self.ffn_hidden_size = args.ffn_hidden_size
        self.blocking = 128

        # Sparse expert MLP.
        self.mlp = mlp.SparseMLP(args)

        # Calculate the number of bits needed to represent the column indices
        # in the intermediate sparse matrix.
        max_column_index = (
            (self.ffn_hidden_size * self.num_experts) // self.blocking)
        self.transpose_sort_end_bit = max(
            int(np.ceil(np.log2(max_column_index))), 1)

    def sparse_transpose(self, size, data, row_indices, column_indices, offsets):
        block_columns = size[1] // data.shape[1]

        # Sort row indices by column indices to get the transposed matrix's
        # column indices.
        #
        # NOTE: Our sort operation uses the same width indices as the input values.
        # To avoid overflow when we have large activation matrices we cast to
        # 32-bit before sorting.
        _, gather_indices = ops.sort(
            column_indices.int(), self.transpose_sort_end_bit)

        # There are a constant number of blocks in every row of the sparse matrix.
        # A blocks offset is:
        #
        # row_index * blocks_per_row + column_index % blocks_per_row
        #
        # Once we have the block offsets ordered for transposition we can divide
        # by blocks_per_row to get the transposed column indices.
        column_indices_t = row_indices.gather(0, gather_indices.long())
        block_offsets_t = gather_indices.int()

        zero = torch.zeros((1,), dtype=torch.int32, device=data.device)
        nnz_per_column = ops.histogram(column_indices, block_columns)
        nnz_per_column = ops.inclusive_cumsum(nnz_per_column, 0)
        offsets_t = torch.cat([zero, nnz_per_column])
        return column_indices_t, offsets_t, block_offsets_t

    def topology(self, x, padded_bins):
        padded_tokens, _ = x.size()
        assert padded_tokens % self.blocking == 0
        assert self.ffn_hidden_size % self.blocking == 0

        # Offsets for the sparse matrix. All rows have the
        # same number of nonzero blocks dictated by the
        # dimensionality of a single expert.
        block_rows = padded_tokens // self.blocking
        blocks_per_row = self.ffn_hidden_size // self.blocking
        offsets = torch.arange(
            0,
            block_rows * blocks_per_row + 1,
            blocks_per_row,
            dtype=torch.int32,
            device=x.device)

        # Indices for the sparse matrix. The indices for
        # the intermediate matrix are dynamic depending
        # on the mapping of tokens to experts.
        column_indices = ops.topology(padded_bins,
                                      self.blocking,
                                      block_rows,
                                      blocks_per_row)

        # TODO(tgale): This is unused. Remove the need for this in stk.
        data = torch.empty(
            column_indices.numel(),
            self.blocking,
            self.blocking,
            dtype=common.dtype(self.args),
            device=x.device)
        shape = (padded_tokens, self.ffn_hidden_size * self.num_experts_per_rank)
        row_indices = stk.ops.row_indices(
            shape, data, offsets, column_indices)
        column_indices_t, offsets_t, block_offsets_t = self.sparse_transpose(
            shape, data, row_indices, column_indices, offsets)
        return stk.Matrix(shape, data, row_indices, column_indices, offsets,
                          column_indices_t, offsets_t, block_offsets_t)

    def indices_and_padded_bins(self, top_experts):
        # Sort the expert ids to produce the scatter/gather
        # indices for the permutation.
        top_experts = top_experts.int()
        bin_ids, indices = ops.sort(top_experts, self.sort_end_bit)

        # Histogram the expert ids to identify the number of
        # tokens routed to each expert.
        tokens_per_expert = ops.histogram(top_experts, self.num_experts)

        # Round the token counts up to the block size used in
        # the matrix muliplications. Caculate the starting
        # position of each bin.
        padded_tokens_per_expert = ops.round_up(
            tokens_per_expert, self.blocking)
        padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0)
        padded_bins = promote_scalar(padded_bins)

        # Calculate the bin bounds for the sorted tokens.
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)
        bins = promote_scalar(bins)
        return indices, bin_ids, bins, padded_bins, tokens_per_expert

    def forward_once(self, x, expert_weights, top_experts):
        # x: [sl, bs, hs]
        # expert_weights: [sl * bs, top-k]
        # top_experts: [sl * bs, top-k]
        expert_weights = expert_weights.flatten()
        top_experts = top_experts.flatten()
        with torch.no_grad():
            indices, bin_ids, bins, padded_bins, tokens_per_expert = (
                self.indices_and_padded_bins(top_experts))
        sl, bs, hs = x.size()

        # Route the tokens for MoE computation.
        x = x.view(sl * bs, hs)
        x = ops.padded_gather(
            x,
            indices,
            bin_ids,
            bins,
            padded_bins,
            self.top_k)

        # Create the sparse matrix topology.
        with torch.no_grad():
            topo = self.topology(x, padded_bins)

        # Perform the expert computation.
        x = self.mlp(x, topo)

        # Un-route the data for the MoE output.
        x = ops.padded_scatter(
            x,
            indices,
            bin_ids,
            expert_weights,
            bins,
            padded_bins,
            self.top_k)
        return x, tokens_per_expert

    # For use in the base-class parallel_forward_once.
    def permute_and_compute(
            self,
            x,
            tokens_per_expert,
            indices,
            bin_ids,
            expert_weights,
            bins,
            expert_capactiy,  # unused
            top_k):

        # Round the token counts up to the block size used in the matrix
        # multiplication. Calculate the starting position of each bin.
        padded_tokens_per_expert = ops.round_up(
            tokens_per_expert, self.blocking)
        padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0)
        padded_bins = promote_scalar(padded_bins)

        # Route the tokens for MoE computation.
        x = x.view(-1, x.shape[-1])
        x = ops.padded_gather(
            x,
            indices,
            bin_ids,
            bins,
            padded_bins,
            top_k)

        # Create the sparse matrix topology.
        t = benchmark_util.Timer("topology")            
        x = t.start(x)                    
        with torch.no_grad():
            topo = self.topology(x, padded_bins)
        x = t.end(x)

        # Perform the expert computation.
        t = benchmark_util.Timer("compute")
        x = t.start(x)
        x = self.mlp(x, topo)
        x = t.end(x)

        # Un-route the data for the MoE output.
        return ops.padded_scatter(
            x,
            indices,
            bin_ids,
            expert_weights,
            bins,
            padded_bins,
            top_k)
