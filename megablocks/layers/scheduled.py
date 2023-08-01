from megablocks.layers.all_to_all import all_to_all
from megablocks.layers import dmoe
import megablocks.ops as ops
import torch


def _compute_padded_bins(self, tokens_per_expert):
    padded_tokens_per_expert = ops.round_up(
        tokens_per_expert, self.blocking)
    padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0)
    return dmoe.promote_scalar(padded_bins)


def _permute_and_compute(
        self,
        x,
        indices,
        bin_ids,
        bins,
        padded_bins,
        topology):
    x = ops.padded_gather(
        x,
        indices,
        bin_ids,
        bins,
        padded_bins,
        top_k=1)
    x = self.mlp(x, topology)
    return ops.padded_scatter(
        x,
        indices,
        bin_ids,
        None,  # expert_weights
        bins,
        padded_bins,
        top_k=1)


def dmoe_parallel_forward_once(self, x, expert_weights, top_experts):
    class ScheduledForwardBackward(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, expert_weights, top_experts):
            # NOTE: This function implements the same algorithm as MoE
            # `parallel_forward_once`, but with hand-scheduled forward
            # and backward passes to hide communication.
            expert_weights = expert_weights.flatten()
            top_experts = top_experts.flatten()
            indices, bin_ids, bins, tokens_per_expert = (
                self.indices_and_bins(top_experts))

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
            # This view updates the shape of the tensor from [sl, bs, hs] to
            # [sl * bs, hs] prior to the permutation.
            x = x.view(-1, x.shape[-1])
            x = ops.padded_gather(
                x,
                indices,
                bin_ids,
                bins,
                bins,
                self.top_k)

            # Compute the number of tokens that will be received from each
            # device and permute the input data across the devices.
            tpe_handle.wait()
            send_counts, recv_counts, tokens_received = (
                self._compute_send_recv_counts(
                    tokens_per_expert,
                    parallel_tokens_per_expert)
            )

            # Start the cross-device permutation asynchronously so we can
            # overlap communication with computation.
            parallel_x, parallel_x_handle = all_to_all(
                x, recv_counts, send_counts,
                self.args.expert_parallel_group,
                async_op=True)

            # After we do the cross-device permutation we have the tokens on the
            # correct device but not yet grouped by expert because we received
            # tokens from each device as contiguous chunks. To group the tokens
            # for expert computation we'll do one more local permutation. The
            # rest of this torch.no_grad() scope sets up the indices and bins
            # for this permutation.
            (parallel_tokens_per_expert,
             parallel_indices,
             parallel_bin_ids,
             parallel_bins) = (
                 self._compute_parallel_metadata(
                     parallel_tokens_per_expert,
                     tokens_received)
             )

            # Round the token counts up to the block size used in the matrix
            # multiplication. Calculate the starting position of each bin.
            parallel_padded_bins = _compute_padded_bins(self, tokens_per_expert)

            # Create the sparse matrix topology.
            topology = self.topology(x, parallel_padded_bins)

            # Locally permute the tokens and perform the expert computation.
            # Block to make sure that the cross-device permutation is complete.
            parallel_x_handle.wait()
            parallel_x = _permute_and_compute(
                self
                parallel_x,
                parallel_tokens_per_expert,
                parallel_indices,
                parallel_bin_ids,
                parallel_bins,
                parallel_padded_bins,
                topology)

            # Un-permute the tokens across the devices.
            x, _ = all_to_all(
                parallel_x, send_counts, recv_counts,
                self.args.expert_parallel_group)

            # Un-permute locally to setup for the next series of operations.
            x = ops.padded_scatter(
                x,
                indices,
                bin_ids,
                expert_weights,
                bins,
                bins,
                self.top_k)
        return x, tokens_per_expert.flatten()
