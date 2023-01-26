import unittest

from absl.testing import parameterized
from megablocks import benchmark_util
from megablocks import ops
import numpy as np
import stk
import torch


_PERMUTE_TESTS = (
    (16384, 768, 2),
    (16384, 768, 4),
    (16384, 768, 8),
    (16384, 768, 16),
    (16384, 768, 32),
    (16384, 768, 64),
    (16384, 768, 128),
    (16384 * 8, 768, 2),
    (16384 * 8, 768, 4),
    (16384 * 8, 768, 8),
    (16384 * 8, 768, 16),
    (16384 * 8, 768, 32),
    (16384 * 8, 768, 64),
    (16384 * 8, 768, 128)
)


class PermuteBenchmark(parameterized.TestCase):

    @parameterized.parameters(*_PERMUTE_TESTS)
    def testBinnedGather(self, sl, hs, ne):
        # NOTE: Capacity factor == 1.
        ec = sl // ne

        # Create the data and indices.
        x = torch.randn((sl, hs)).cuda().half()
        top_expert = torch.randint(0, ne, (sl,)).cuda().int()
        bin_ids, indices = ops.sort(top_expert)
        tokens_per_expert = ops.histogram(indices, ne)
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)

        benchmark = lambda: ops.binned_gather(x, indices, bins, ec)
        mean_t, std_t = benchmark_util.benchmark_function(benchmark)
        arguments = {
            "sequence_length": sl,
            "hidden_size": hs,
            "num_experts": ne
        }
        benchmark_util.log_benchmark("BinnedGather", arguments, mean_t, std_t)

    @parameterized.parameters(*_PERMUTE_TESTS)
    def testBinnedScatter(self, sl, hs, ne):
        # NOTE: Capacity factor == 1.
        ec = sl // ne

        # Create the data and indices.
        x = torch.randn((sl, hs)).cuda().half()
        top_expert = torch.randint(0, ne, (sl,)).cuda().int()
        bin_ids, indices = ops.sort(top_expert)
        tokens_per_expert = ops.histogram(indices, ne)
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)
        x = ops.binned_gather(x, indices, bins, ec)

        benchmark = lambda: ops.binned_scatter(x, indices, bins)
        mean_t, std_t = benchmark_util.benchmark_function(benchmark)
        arguments = {
            "sequence_length": sl,
            "hidden_size": hs,
            "num_experts": ne
        }
        benchmark_util.log_benchmark("BinnedScatter", arguments, mean_t, std_t)

    @parameterized.parameters(*_PERMUTE_TESTS)
    def testPaddedGather(self, sl, hs, ne):
        # Create the data and indices.
        x = torch.randn((sl, hs)).cuda().half()

        # Randomly assign tokens to experts.
        top_expert = torch.randint(0, ne, (sl,)).cuda().int()
        bin_ids, indices = ops.sort(top_expert)
        tokens_per_expert = ops.histogram(top_expert, ne)
        padded_tokens_per_expert = ops.round_up(tokens_per_expert, 128)
        padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0)
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)

        benchmark = lambda: ops.padded_gather(x, indices, bin_ids, bins, padded_bins)
        mean_t, std_t = benchmark_util.benchmark_function(benchmark)
        arguments = {
            "sequence_length": sl,
            "hidden_size": hs,
            "num_experts": ne
        }
        benchmark_util.log_benchmark("PaddedGather", arguments, mean_t, std_t)

    @parameterized.parameters(*_PERMUTE_TESTS)
    def testPaddedScatter(self, sl, hs, ne):
        # Create the data and indices.
        x = torch.randn((sl, hs)).cuda().half()

        # Randomly assign tokens to experts.
        top_expert = torch.randint(0, ne, (sl,)).cuda().int()
        bin_ids, indices = ops.sort(top_expert)
        tokens_per_expert = ops.histogram(top_expert, ne)
        padded_tokens_per_expert = ops.round_up(tokens_per_expert, 128)
        padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0)
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)
        x = ops.padded_gather(x, indices, bin_ids, bins, padded_bins)

        benchmark = lambda: ops.padded_scatter(x, indices, bin_ids, bins, padded_bins)
        mean_t, std_t = benchmark_util.benchmark_function(benchmark)
        arguments = {
            "sequence_length": sl,
            "hidden_size": hs,
            "num_experts": ne
        }
        benchmark_util.log_benchmark("PaddedScatter", arguments, mean_t, std_t)

    @parameterized.parameters(*_PERMUTE_TESTS)
    def testCopy(self, sl, hs, ne):
        # NOTE: Capacity factor == 1.
        ec = sl // ne

        # Create the data and indices.
        x = torch.randn((sl, hs)).cuda().half()
        y = x.clone()

        benchmark = lambda: y.copy_(x)
        mean_t, std_t = benchmark_util.benchmark_function(benchmark)
        arguments = {
            "sequence_length": sl,
            "hidden_size": hs,
            "num_experts": ne
        }
        benchmark_util.log_benchmark("Copy", arguments, mean_t, std_t)


if __name__ == '__main__':
    unittest.main()
