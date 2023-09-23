import unittest

from absl.testing import parameterized
from megablocks import benchmark_util
from megablocks import ops
import stk
import torch


# Calling tensor.t() calls tensor.transpose(0, 1) which calls
# torch.as_strided(...). Circumvent this chain to avoid an overhead
# this adds.
def transpose_view(x):
    return torch.as_strided(
        x, (x.shape[1], x.shape[0]), (x.stride()[1], x.stride()[0]))


_MATMUL_TESTS = (
    (64 * 1024, 512, 2048, 64),
    (32 * 1024, 768, 3072, 64),
    (8 * 1024, 1024, 4096, 64),
    (4 * 2048, 4096, 4 * 4096, 4),
)


def log_benchmark(name, arguments, time, std, flops):
    benchmark_util.log_benchmark(name, arguments, time, std)
    print("flops = {:.2f}B".format(flops / 1e9))
    print("throughput = {:.2f}T".format(flops / 1e9 / time))
    print("="*60)


class MatmulBenchmark(parameterized.TestCase):

    def build_sparse_matrix(self, x, padded_bins, fhs, ne):
        blocking = 128
        padded_tokens, _ = x.size()
        assert padded_tokens % blocking == 0
        assert fhs % blocking == 0

        # Offsets for the sparse matrix. All rows have the
        # same number of nonzero blocks dictated by the
        # dimensionality of a single expert.
        block_rows = padded_tokens // blocking
        blocks_per_row = fhs // blocking
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
                                      blocking,
                                      block_rows,
                                      blocks_per_row)
        data = torch.empty(
            column_indices.numel(),
            blocking,
            blocking,
            dtype=torch.float16,
            device=x.device)
        shape = (padded_tokens, fhs * ne)
        row_indices = stk.ops.row_indices(
            shape, data, offsets, column_indices)
        return stk.Matrix(shape,
                          data,
                          row_indices,
                          column_indices,
                          offsets)

    def build_input_matrix(self, sl, hs, ne):
        x = torch.randn((sl, hs)).cuda().half()

        # Assign tokens to experts uniformly.
        top_expert = torch.arange(0, sl).cuda().int() % ne

        bin_ids, indices = ops.sort(top_expert)
        tokens_per_expert = ops.histogram(top_expert, ne)
        padded_tokens_per_expert = ops.round_up(tokens_per_expert, 128)
        padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0)
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)
        out = ops.padded_gather(x, indices, bin_ids, bins, padded_bins, 1)
        return out, padded_bins

    def build_weight_matrix(self, ne, hs, fhs):
        return torch.randn((hs, ne * fhs)).cuda().half()

    @parameterized.parameters(*_MATMUL_TESTS)
    def testFFN_Linear0_Fwd_SDD_NT(self, sl, hs, fhs, ne):
        x, padded_bins = self.build_input_matrix(sl, hs, ne)
        w = self.build_weight_matrix(ne, hs, fhs).t().contiguous()
        topo = self.build_sparse_matrix(x, padded_bins, fhs, ne)
        w = transpose_view(w)

        benchmark = lambda: stk.ops.sdd(x, w, topo)
        mean_t, std_t = benchmark_util.benchmark_function(benchmark)
        arguments = {
            "sequence_length": sl,
            "hidden_size": hs,
            "ffn_hidden_size": fhs,
            "num_experts": ne
        }
        log_benchmark("0::Fwd::SDD::NT", arguments, mean_t, std_t,
                      x.numel() * fhs * 2)

    @parameterized.parameters(*_MATMUL_TESTS)
    def testFFN_Linear0_GradX_DSD_NN(self, sl, hs, fhs, ne):
        x, padded_bins = self.build_input_matrix(sl, hs, ne)
        w = self.build_weight_matrix(ne, hs, fhs).t().contiguous()
        topo = self.build_sparse_matrix(x, padded_bins, fhs, ne)

        benchmark = lambda: stk.ops.dsd(topo, w)
        mean_t, std_t = benchmark_util.benchmark_function(benchmark)
        arguments = {
            "sequence_length": sl,
            "hidden_size": hs,
            "ffn_hidden_size": fhs,
            "num_experts": ne
        }
        log_benchmark("0::GradX::DSD::NN", arguments, mean_t, std_t,
                      x.numel() * fhs * 2)

    @parameterized.parameters(*_MATMUL_TESTS)
    def testFFN_Linear0_GradW_DSD_TN(self, sl, hs, fhs, ne):
        x, padded_bins = self.build_input_matrix(sl, hs, ne)
        topo = self.build_sparse_matrix(x, padded_bins, fhs, ne)
        topo = topo.t()

        benchmark = lambda: stk.ops.dsd(topo, x)
        mean_t, std_t = benchmark_util.benchmark_function(benchmark)
        arguments = {
            "sequence_length": sl,
            "hidden_size": hs,
            "ffn_hidden_size": fhs,
            "num_experts": ne
        }
        log_benchmark("0::GradW::DSD::TN", arguments, mean_t, std_t,
                      x.numel() * fhs * 2)

    @parameterized.parameters(*_MATMUL_TESTS)
    def testFFN_Linear1_Fwd_DSD_NN(self, sl, hs, fhs, ne):
        x, padded_bins = self.build_input_matrix(sl, hs, ne)
        w = self.build_weight_matrix(ne, hs, fhs).t().contiguous()
        x = self.build_sparse_matrix(x, padded_bins, fhs, ne)

        benchmark = lambda: stk.ops.dsd(x, w)
        mean_t, std_t = benchmark_util.benchmark_function(benchmark)
        arguments = {
            "sequence_length": sl,
            "hidden_size": hs,
            "ffn_hidden_size": fhs,
            "num_experts": ne
        }
        log_benchmark("1::Fwd::DSD::NN", arguments, mean_t, std_t,
                      x.nnz * hs * 2)

    @parameterized.parameters(*_MATMUL_TESTS)
    def testFFN_Linear1_GradX_SDD_NT(self, sl, hs, fhs, ne):
        x, padded_bins = self.build_input_matrix(sl, hs, ne)
        w = self.build_weight_matrix(ne, hs, fhs).t().contiguous()
        x = self.build_sparse_matrix(x, padded_bins, fhs, ne)
        out = stk.ops.dsd(x, w)
        w = transpose_view(w)

        benchmark = lambda: stk.ops.sdd(out, w, x)
        mean_t, std_t = benchmark_util.benchmark_function(benchmark)
        arguments = {
            "sequence_length": sl,
            "hidden_size": hs,
            "ffn_hidden_size": fhs,
            "num_experts": ne
        }
        log_benchmark("1::GradX::SDD::NT", arguments, mean_t, std_t,
                      x.nnz * hs * 2)

    @parameterized.parameters(*_MATMUL_TESTS)
    def testFFN_Linear1_GradW_DSD_TN(self, sl, hs, fhs, ne):
        x, padded_bins = self.build_input_matrix(sl, hs, ne)
        w = self.build_weight_matrix(ne, hs, fhs).t().contiguous()
        x = self.build_sparse_matrix(x, padded_bins, fhs, ne)
        out = stk.ops.dsd(x, w)
        x = x.t()

        benchmark = lambda: stk.ops.dsd(x, out)
        mean_t, std_t = benchmark_util.benchmark_function(benchmark)
        arguments = {
            "sequence_length": sl,
            "hidden_size": hs,
            "ffn_hidden_size": fhs,
            "num_experts": ne
        }
        log_benchmark("1::GradW::DSD::TN", arguments, mean_t, std_t,
                      x.nnz * hs * 2)

    @parameterized.parameters(*_MATMUL_TESTS)
    def testFFN_Linear0_Fwd_DDD_NT(self, sl, hs, fhs, ne):
        assert (sl % ne) == 0
        x = torch.randn((ne, sl // ne, hs)).cuda().half()
        w = torch.randn((ne, hs, fhs)).cuda().half()

        w = w.transpose(1, 2).contiguous()
        w = w.transpose(1, 2)

        benchmark = lambda: torch.bmm(x, w)
        mean_t, std_t = benchmark_util.benchmark_function(benchmark)
        arguments = {
            "sequence_length": sl,
            "hidden_size": hs,
            "ffn_hidden_size": fhs,
            "num_experts": ne
        }
        log_benchmark("0::Fwd:DDD::NT", arguments, mean_t, std_t,
                      x.numel() * fhs * 2)

    @parameterized.parameters(*_MATMUL_TESTS)
    def testFFN_Linear0_GradX_DDD_NN(self, sl, hs, fhs, ne):
        assert (sl % ne) == 0
        x = torch.randn((ne, sl // ne, hs)).cuda().half()
        w = torch.randn((ne, hs, fhs)).cuda().half()
        out = torch.bmm(x, w)
        w = w.transpose(1, 2).contiguous()

        benchmark = lambda: torch.bmm(out, w)
        mean_t, std_t = benchmark_util.benchmark_function(benchmark)
        arguments = {
            "sequence_length": sl,
            "hidden_size": hs,
            "ffn_hidden_size": fhs,
            "num_experts": ne
        }
        log_benchmark("0:GradX:DDD::NN", arguments, mean_t, std_t,
                      x.numel() * fhs * 2)

    @parameterized.parameters(*_MATMUL_TESTS)
    def testFFN_Linear0_GradW_DDD_TN(self, sl, hs, fhs, ne):
        assert (sl % ne) == 0
        x = torch.randn((ne, sl // ne, hs)).cuda().half()
        w = torch.randn((ne, hs, fhs)).cuda().half()
        out = torch.bmm(x, w)
        out = out.transpose(1, 2)

        benchmark = lambda: torch.bmm(out, x)
        mean_t, std_t = benchmark_util.benchmark_function(benchmark)
        arguments = {
            "sequence_length": sl,
            "hidden_size": hs,
            "ffn_hidden_size": fhs,
            "num_experts": ne
        }
        log_benchmark("0:GradW:DDD::TN", arguments, mean_t, std_t,
                      x.numel() * fhs * 2)

    @parameterized.parameters(*_MATMUL_TESTS)
    def testFFN_Linear1_Fwd_DDD_NN(self, sl, hs, fhs, ne):
        assert (sl % ne) == 0
        x = torch.randn((ne, sl // ne, fhs)).cuda().half()
        w = torch.randn((ne, fhs, hs)).cuda().half()

        benchmark = lambda: torch.bmm(x, w)
        mean_t, std_t = benchmark_util.benchmark_function(benchmark)
        arguments = {
            "sequence_length": sl,
            "hidden_size": hs,
            "ffn_hidden_size": fhs,
            "num_experts": ne
        }
        log_benchmark("1::Fwd::DDD::NN", arguments, mean_t, std_t,
                      x.numel() * hs * 2)

    @parameterized.parameters(*_MATMUL_TESTS)
    def testFFN_Linear1_GradX_DDD_NT(self, sl, hs, fhs, ne):
        assert (sl % ne) == 0
        x = torch.randn((ne, sl // ne, fhs)).cuda().half()
        w = torch.randn((ne, fhs, hs)).cuda().half()
        out = torch.bmm(x, w)
        w = torch.transpose(w, 1, 2)

        benchmark = lambda: torch.bmm(out, w)
        mean_t, std_t = benchmark_util.benchmark_function(benchmark)
        arguments = {
            "sequence_length": sl,
            "hidden_size": hs,
            "ffn_hidden_size": fhs,
            "num_experts": ne
        }
        log_benchmark("1::GradX::DDD::NT", arguments, mean_t, std_t,
                      x.numel() * hs * 2)

    @parameterized.parameters(*_MATMUL_TESTS)
    def testFFN_Linear1_GradW_DDD_TN(self, sl, hs, fhs, ne):
        assert (sl % ne) == 0
        x = torch.randn((ne, sl // ne, fhs)).cuda().half()
        w = torch.randn((ne, fhs, hs)).cuda().half()
        out = torch.bmm(x, w)
        x = torch.transpose(x, 1, 2)

        benchmark = lambda: torch.bmm(x, out)
        mean_t, std_t = benchmark_util.benchmark_function(benchmark)
        arguments = {
            "sequence_length": sl,
            "hidden_size": hs,
            "ffn_hidden_size": fhs,
            "num_experts": ne
        }
        log_benchmark("1::GradW::DDD::TN", arguments, mean_t, std_t,
                      x.numel() * hs * 2)


if __name__ == '__main__':
    unittest.main()
