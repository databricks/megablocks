import unittest

from absl.testing import parameterized
from megablocks import ops
from megablocks import benchmark_util
import torch


_PADDED_SCATTER_BENCHMARK = (
    # dMoE-Medium, 8-way EMP.
    (1024 * 16, 1024, 8, 4),
    # dMoE-Medium, post-all-to-all.
    (1024 * 16 * 4, 1024, 8, 1),
)


class PaddedScatterTest(parameterized.TestCase):

    @parameterized.parameters(*_PADDED_SCATTER_BENCHMARK)
    def testPaddedScatter(self, sl, hs, ne, top_k):
        # Create the data and indices.
        x = torch.randn((sl, hs)).cuda().half()

        # Randomly assign tokens to experts.
        top_expert = torch.randint(0, ne, (sl * top_k,)).cuda().int()
        bin_ids, indices = ops.sort(top_expert)
        tokens_per_expert = ops.histogram(top_expert, ne)
        padded_tokens_per_expert = ops.round_up(tokens_per_expert, 128)
        padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0)
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)

        # Sample weights for the scatter reduce.
        weights = torch.rand((sl * top_k,)).cuda().half()

        # Gather the data to prepare for backwards.
        x = ops.padded_gather(x, indices, bin_ids, bins, padded_bins, top_k)

        fn = lambda: ops.padded_scatter(
            x, indices, bin_ids, weights, bins, padded_bins, top_k)

        time, std = benchmark_util.benchmark_function(fn)
        benchmark_util.log_benchmark(
            "Padded Scatter",
            {"sequence_length": sl,
             "hidden_size": hs,
             "num_experts": ne,
             "top_k": top_k},
            time,
            std)


if __name__ == '__main__':
    unittest.main()
