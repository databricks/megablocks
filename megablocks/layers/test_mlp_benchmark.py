import unittest

from absl.testing import parameterized
from megablocks import benchmark_util
from megablocks.layers.arguments import Arguments
from megablocks import ops
from megablocks.layers import mpu
from megablocks.layers.mlp import MLP, SparseMLP, GroupedMLP, TransformerEngineFp8MLP, TorchMLP
import stk

import torch

def log_benchmark(name, arguments, time, std, flops):
    benchmark_util.log_benchmark(name, arguments, time, std)
    print("="*60)

_MATMUL_TESTS = (
    (4, 2048, 8192, 8192, 32, 8, 4),
)

class MLPBenchmark(parameterized.TestCase):
    def forwardBackward(self, model, x, forward_only=False, kwargs=None):
        y = model(x, **kwargs) if kwargs else model(x)
        if forward_only: return
        y_grad = y.sum().backward()
    
    def transformerEngineForwardBackward(self, model, x, forward_only=False, kwargs=None):
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import Format, DelayedScaling

        fp8_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
        fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            y = model(x, **kwargs) if kwargs else model(x)
        if forward_only: return
        y_grad = y.sum().backward()
    
    def get_uniform_tokens_per_expert(self, num_tokens_per_expert, expert_count):
        return torch.Tensor([num_tokens_per_expert for _ in range(expert_count)]).cuda()
    
    @parameterized.parameters(*_MATMUL_TESTS)
    def testTorchMLPBenchmark(self, bs, sl, hs, fhs, ne, ws, top_k):
        args = Arguments(
            hidden_size=hs,
            ffn_hidden_size=fhs,
            moe_num_experts=ne,
            moe_top_k=top_k,
            grouped_mlp=True,
            moe_expert_model_parallelism=True,
        )
        arguments = {
            "batch_size": bs,
            "sequence_length": sl,
            "hidden_size": hs,
            "ffn_hidden_size": fhs,
            "num_experts": ne,
            "world_size": ws,
            "top_k": top_k
        }
        # mock the expert parallelism functions
        mpu.get_expert_parallel_rank = lambda args: 0
        mpu.get_expert_parallel_world_size = lambda args: ws
        epr = mpu.experts_per_rank(args)
        assert epr == ne // ws

        # Generate input and uniform tokens_per_expert
        num_tokens_per_expert = (bs * sl * top_k * ws) // ne
        model = TorchMLP(args).to(torch.bfloat16)
        x = torch.randn(epr * num_tokens_per_expert, hs, device="cuda", dtype=torch.bfloat16)
        tokens_per_expert = self.get_uniform_tokens_per_expert(num_tokens_per_expert, epr)

        # run benchmark forward backward
        benchmark = lambda: self.forwardBackward(model, x, kwargs={"tokens_per_expert": tokens_per_expert}, forward_only=False)
        mean_t, std_t = benchmark_util.benchmark_function(benchmark)
        benchmark_util.log_benchmark("1::testTorchForwardBackwardBenchmark", arguments, mean_t, std_t)

        # run benchmark forward
        benchmark = lambda: self.forwardBackward(model, x, kwargs={"tokens_per_expert": tokens_per_expert}, forward_only=True)
        mean_t, std_t = benchmark_util.benchmark_function(benchmark)
        benchmark_util.log_benchmark("1::testTorchForwardBenchmark", arguments, mean_t, std_t)

    @parameterized.parameters(*_MATMUL_TESTS)
    def testTransformerEngineMLPBenchmark(self, bs, sl, hs, fhs, ne, ws, top_k):
        args = Arguments(
            hidden_size=hs,
            ffn_hidden_size=fhs,
            moe_num_experts=ne,
            moe_top_k=top_k,
            grouped_mlp=True,
            fp8=True,
            moe_expert_model_parallelism=True,
        )
        arguments = {
            "batch_size": bs,
            "sequence_length": sl,
            "hidden_size": hs,
            "ffn_hidden_size": fhs,
            "num_experts": ne,
            "world_size": ws,
            "top_k": top_k,
            "fp8": True,
        }
        # mock the expert parallelism functions
        mpu.get_expert_parallel_rank = lambda args: 0
        mpu.get_expert_parallel_world_size = lambda args: ws
        epr = mpu.experts_per_rank(args)
        assert epr == ne // ws

        # Generate input and uniform tokens_per_expert
        num_tokens_per_expert = (bs * sl * top_k * ws) // ne
        model = TransformerEngineFp8MLP(args)
        x = torch.randn(epr * num_tokens_per_expert, hs, device="cuda", dtype=torch.bfloat16)
        tokens_per_expert = self.get_uniform_tokens_per_expert(num_tokens_per_expert, epr)

        # run benchmark forward backward
        benchmark = lambda: self.transformerEngineForwardBackward(model, x, kwargs={"tokens_per_expert": tokens_per_expert}, forward_only=False)
        mean_t, std_t = benchmark_util.benchmark_function(benchmark)
        benchmark_util.log_benchmark("2::testTransformerEngineForwardBackwardBenchmark", arguments, mean_t, std_t)

        # run benchmark forward
        benchmark = lambda: self.transformerEngineForwardBackward(model, x, kwargs={"tokens_per_expert": tokens_per_expert}, forward_only=True)
        mean_t, std_t = benchmark_util.benchmark_function(benchmark)
        benchmark_util.log_benchmark("2::testTransformerEngineForwardBenchmark", arguments, mean_t, std_t)

    @parameterized.parameters(*_MATMUL_TESTS)
    def testGroupedMLPBenchmark(self, bs, sl, hs, fhs, ne, ws, top_k):
        args = Arguments(
            hidden_size=hs,
            ffn_hidden_size=fhs,
            moe_num_experts=ne,
            moe_top_k=top_k,
            grouped_mlp=True,
            moe_expert_model_parallelism=True,
        )
        arguments = {
            "batch_size": bs,
            "sequence_length": sl,
            "hidden_size": hs,
            "ffn_hidden_size": fhs,
            "num_experts": ne,
            "world_size": ws,
            "top_k": top_k,
        }
        # mock the expert parallelism functions
        mpu.get_expert_parallel_rank = lambda args: 0
        mpu.get_expert_parallel_world_size = lambda args: ws
        epr = mpu.experts_per_rank(args)
        assert epr == ne // ws

        # Generate input and uniform tokens_per_expert
        num_tokens_per_expert = (bs * sl * top_k * ws) // ne
        model = GroupedMLP(args).to(torch.bfloat16)
        x = torch.randn(epr * num_tokens_per_expert, hs, device="cuda", dtype=torch.bfloat16)
        tokens_per_expert = self.get_uniform_tokens_per_expert(num_tokens_per_expert, epr)

        # run benchmark forward backward
        benchmark = lambda: self.forwardBackward(model, x, kwargs={"tokens_per_expert": tokens_per_expert}, forward_only=False)
        mean_t, std_t = benchmark_util.benchmark_function(benchmark)
        benchmark_util.log_benchmark("3::testGroupedForwardBackwardBenchmark", arguments, mean_t, std_t)

        # run benchmark forward
        benchmark = lambda: self.forwardBackward(model, x, kwargs={"tokens_per_expert": tokens_per_expert}, forward_only=True)
        mean_t, std_t = benchmark_util.benchmark_function(benchmark)
        benchmark_util.log_benchmark("3::testGroupedForwardBenchmark", arguments, mean_t, std_t)