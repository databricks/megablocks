import unittest

from absl.testing import parameterized
from megablocks import benchmark_util
from megablocks.layers import mpu
from megablocks.layers.arguments import Arguments
from megablocks.layers import dmlp_registry
import torch

_MLP_TESTS = (
    ('grouped', 4, 2048, 8192, 8192, 32, 8, 4),
    ('torch', 4, 2048, 8192, 8192, 32, 8, 4),
    ('te', 4, 2048, 8192, 8192, 32, 8, 4),
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
    
    @parameterized.parameters(*_MLP_TESTS)
    def testMLPRuntimeBenchmarks(self, mlp_impl, bs, sl, hs, fhs, ne, ws, top_k):
        """ Benchmark the runtimes of the MLP forward and backward pass with expert parallelism. """

        args = Arguments(
            hidden_size=hs,
            ffn_hidden_size=fhs,
            moe_num_experts=ne,
            moe_top_k=top_k,
            moe_expert_model_parallelism=True,
            mlp_impl=mlp_impl,
        )
        arguments = {
            "mlp_impl": mlp_impl,
            "batch_size": bs,
            "sequence_length": sl,
            "hidden_size": hs,
            "ffn_hidden_size": fhs,
            "num_experts": ne,
            "world_size": ws,
            "top_k": top_k,
        }
        # Mock the expert parallelism functions
        mpu.get_expert_parallel_rank = lambda args: 0
        mpu.get_expert_parallel_world_size = lambda args: ws
        experts_per_rank = mpu.experts_per_rank(args)
        assert experts_per_rank == ne // ws

        # Generate input data and tokens_per_expert
        num_tokens_per_expert = (bs * sl * top_k * ws) // ne
        total_tokens = experts_per_rank * num_tokens_per_expert
        x = torch.randn(total_tokens, hs, device="cuda", dtype=torch.bfloat16)
        tokens_per_expert = self.get_uniform_tokens_per_expert(num_tokens_per_expert, experts_per_rank)
        assert total_tokens == bs * sl * top_k
        
        # Get model and forward/backward function
        model = dmlp_registry.get(args)
        if mlp_impl == 'te':
            forward_func = self.transformerEngineForwardBackward
        else:
            model.to(torch.bfloat16)
            forward_func = self.forwardBackward

        # run forward backward benchmark
        benchmark = lambda: forward_func(model, x, kwargs={"tokens_per_expert": tokens_per_expert}, forward_only=False)
        mean_t, std_t = benchmark_util.benchmark_function(benchmark)
        benchmark_util.log_benchmark(f"test_{mlp_impl}_forward_backward_benchmark", arguments, mean_t, std_t)

        # run forward benchmark
        benchmark = lambda: forward_func(model, x, kwargs={"tokens_per_expert": tokens_per_expert}, forward_only=True)
        mean_t, std_t = benchmark_util.benchmark_function(benchmark)
        benchmark_util.log_benchmark(f"test_{mlp_impl}_forward_benchmark", arguments, mean_t, std_t)