import unittest

from absl.testing import parameterized
from megablocks import benchmark_util
from megablocks.layers.arguments import Arguments
from megablocks import ops
from megablocks.layers.mlp import MLP, SparseMLP, GroupedMLP, TransformerEngineFp8MLP, TorchMLP
import stk

import torch

def log_benchmark(name, arguments, time, std, flops):
    benchmark_util.log_benchmark(name, arguments, time, std)
    print("="*60)

_MATMUL_TESTS = (
    # (64 * 1024, 512, 2048, 64),
    # (32 * 1024, 768, 3072, 64),
    # (8 * 1024, 1024, 4096, 64),
    # (4 * 2048, 4096, 4 * 4096, 4),
    (2048, 8192, 8192, 4),
)

class MLPBenchmark(parameterized.TestCase):
    def build_input_matrix(self, sl, hs, ne):
        x = torch.randn((sl, hs), device="cuda", dtype=torch.bfloat16)

        # Assign tokens to experts uniformly.
        top_expert = torch.arange(0, sl).cuda().int() % ne

        bin_ids, indices = ops.sort(top_expert)
        tokens_per_expert = ops.histogram(top_expert, ne)
        padded_tokens_per_expert = ops.round_up(tokens_per_expert, 128)
        padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0)
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)
        out = ops.padded_gather(x, indices, bin_ids, bins, padded_bins, 1)
        return out, padded_bins
    
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


    def forwardBackward(self, model, x, kwargs=None):
        y = model(x, **kwargs) if kwargs else model(x)
        y_grad = y.sum().backward()
    
    def transformerEngineForwardBackward(self, model, x, kwargs=None):
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import Format, DelayedScaling

        fp8_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
        fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            y = model(x, **kwargs) if kwargs else model(x)
        y_grad = y.mean().backward()
    
    def get_uniform_tokens_per_expert(self, sl, ne):
        return torch.Tensor([sl // ne for _ in range(ne)]).cuda()
        
    @parameterized.parameters(*_MATMUL_TESTS)
    def testBMMMLPBenchmark(self, sl, hs, fhs, ne):
        args = Arguments(
            hidden_size=hs,
            ffn_hidden_size=fhs,
            moe_num_experts=ne,
        )
        model = MLP(args).to(torch.bfloat16)
        x = torch.randn(ne, sl // ne, hs, device="cuda", dtype=torch.bfloat16)

        benchmark = lambda: self.forwardBackward(model, x)
        mean_t, std_t = benchmark_util.benchmark_function(benchmark)
        arguments = {
            "sequence_length": sl,
            "hidden_size": hs,
            "ffn_hidden_size": fhs,
            "num_experts": ne
        }
        benchmark_util.log_benchmark("1::testBMMMLPBenchmark", arguments, mean_t, std_t)
        
    @parameterized.parameters(*_MATMUL_TESTS)
    def testTorchMLPBenchmark(self, sl, hs, fhs, ne):
        args = Arguments(
            hidden_size=hs,
            ffn_hidden_size=fhs,
            moe_num_experts=ne,
            grouped_mlp=True,
        )
        model = TorchMLP(args).to(torch.bfloat16)
        assert model.w1[0].dtype == torch.bfloat16
        x = torch.randn(ne, sl // ne, hs, device="cuda", dtype=torch.bfloat16)
        tokens_per_expert = self.get_uniform_tokens_per_expert(sl, ne)

        benchmark = lambda: self.forwardBackward(model, x, kwargs={"tokens_per_expert": tokens_per_expert})
        mean_t, std_t = benchmark_util.benchmark_function(benchmark)
        arguments = {
            "sequence_length": sl,
            "hidden_size": hs,
            "ffn_hidden_size": fhs,
            "num_experts": ne
        }
        benchmark_util.log_benchmark("2::testTorchMLPBenchmark", arguments, mean_t, std_t)
        
    @parameterized.parameters(*_MATMUL_TESTS)
    def testTransformerEngineMLPBenchmark(self, sl, hs, fhs, ne):
        args = Arguments(
            hidden_size=hs,
            ffn_hidden_size=fhs,
            moe_num_experts=ne,
            grouped_mlp=True,
            fp8=True,
        )
        model = TransformerEngineFp8MLP(args)
        x = torch.randn(sl, hs, device="cuda", dtype=torch.bfloat16)
        tokens_per_expert = self.get_uniform_tokens_per_expert(sl, ne)

        benchmark = lambda: self.transformerEngineForwardBackward(model, x, kwargs={"tokens_per_expert": tokens_per_expert})
        mean_t, std_t = benchmark_util.benchmark_function(benchmark)
        arguments = {
            "sequence_length": sl,
            "hidden_size": hs,
            "ffn_hidden_size": fhs,
            "num_experts": ne
        }
        benchmark_util.log_benchmark("3::testTransformerEngineMLPBenchmark", arguments, mean_t, std_t)

    @parameterized.parameters(*_MATMUL_TESTS)
    def testGroupedMLPBenchmark(self, sl, hs, fhs, ne):
        args = Arguments(
            hidden_size=hs,
            ffn_hidden_size=fhs,
            moe_num_experts=ne,
            grouped_mlp=True,
        )
        model = GroupedMLP(args).to(torch.bfloat16)
        x = torch.randn(sl, hs, device="cuda", dtype=torch.bfloat16)
        tokens_per_expert = self.get_uniform_tokens_per_expert(sl, ne)

        benchmark = lambda: self.forwardBackward(model, x, kwargs={"tokens_per_expert": tokens_per_expert})
        mean_t, std_t = benchmark_util.benchmark_function(benchmark)
        arguments = {
            "sequence_length": sl,
            "hidden_size": hs,
            "ffn_hidden_size": fhs,
            "num_experts": ne
        }
        benchmark_util.log_benchmark("4::testGroupedMLPBenchmark", arguments, mean_t, std_t)

    @parameterized.parameters(*_MATMUL_TESTS)
    def testSparseMLPBenchmark(self, sl, hs, fhs, ne):
        args = Arguments(
            hidden_size=hs,
            ffn_hidden_size=fhs,
            moe_num_experts=ne,
            grouped_mlp=False,
        )
        model = SparseMLP(args).to(torch.bfloat16)
        x, padded_bins = self.build_input_matrix(sl, hs, ne)
        topo = self.build_sparse_matrix(x, padded_bins, fhs, ne)

        benchmark = lambda: self.forwardBackward(model, x, kwargs={"topo": topo})
        mean_t, std_t = benchmark_util.benchmark_function(benchmark)
        arguments = {
            "sequence_length": sl,
            "hidden_size": hs,
            "ffn_hidden_size": fhs,
            "num_experts": ne
        }
        benchmark_util.log_benchmark("5::testSparseMLPBenchmark", arguments, mean_t, std_t)