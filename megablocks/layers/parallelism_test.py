import functools

from megablocks.layers import dmoe, arguments, mpu
from megablocks import benchmark_util
import numpy as np
import torch

_TESTS = (
    (64, 1024, 512, 2048, 64, 1, False),
    (64, 1024, 512, 2048, 64, 1, True),
    # Test with fewer experts than ranks to verify tensor
    # sharding in tandem with expert sharding.
    (4, 1, 512, 2048, 4, 1, False),
    (4, 1, 512, 2048, 4, 1, True),
)

def test_expert_parallel_versus_weight_parallel(
        group,
        batch_size,
        sequence_length,
        hidden_size,
        ffn_hidden_size,
        num_experts,
        top_k,
        memory_optimized):
    init_fn = functools.partial(torch.nn.init.normal_, mean=0.0, std=0.1)
    ep_args = arguments.Arguments(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        moe_num_experts=num_experts,
        moe_top_k=top_k,
        moe_expert_model_parallelism=True,
        expert_parallel_group=group,
        fp16=False,
        bf16=False,
        device=torch.cuda.current_device(),
        init_method=init_fn,
        memory_optimized_mlp=memory_optimized)
    wp_args = arguments.Arguments(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        moe_num_experts=num_experts,
        moe_top_k=top_k,
        moe_weight_parallelism=True,
        weight_parallel_group=group,
        fp16=False,
        bf16=False,
        device=torch.cuda.current_device(),
        init_method=init_fn,
        memory_optimized_mlp=memory_optimized)

    # NOTE: Reset the seed so that the models get identical weights.
    torch.manual_seed(1234)
    ep = dmoe.dMoE(ep_args)
    torch.manual_seed(1234)
    wp = dmoe.dMoE(wp_args)

    # NOTE: Include the rank in the seed so we get different data per rank.
    rank = torch.distributed.get_rank(group)
    torch.manual_seed(1234 * rank)
    x = torch.randn(
        (batch_size, sequence_length, hidden_size),
        device=torch.cuda.current_device(),
        dtype=torch.float32).requires_grad_(True)

    # Test forward.
    out, _ = wp(x)
    expected_out, _ = ep(x)

    # Check the forward outputs.
    for i in range(torch.distributed.get_world_size(group)):
        torch.distributed.barrier(group)
        if i == rank:
            np.testing.assert_allclose(
                out.detach().float().cpu(),
                expected_out.detach().float().cpu(),
                rtol=1e-4, atol=1e-4)

    # Test backward.
    out.mean().backward()
    expected_out.mean().backward()

    # NOTE: If tensor parallelism is used different weights can be on
    # different ranks. Gather the full grads to rank 0 to compare.
    def gather(x):
        m, n = x.shape
        world_size = torch.distributed.get_world_size(group)
        out = torch.empty(
            m * world_size, n, device=x.device, dtype=x.dtype)
        torch.distributed.all_gather_into_tensor(out, x, group=group)
        return out

    def permute(x):
        esd = mpu.expert_sharding_degree(ep_args)
        hsd = mpu.hidden_sharding_degree(ep_args)
        out = x.view(hsd, esd, -1).transpose(1, 0).contiguous()
        return out.view(num_experts * ffn_hidden_size, hidden_size)

    wp_w2_grad = gather(wp.experts.mlp.w2.grad)
    ep_w2_grad = permute(gather(ep.experts.mlp.w2.grad))
    if rank == 0:
        np.testing.assert_allclose(
            wp_w2_grad.float().cpu(),
            ep_w2_grad.float().cpu(),
            rtol=1e-5, atol=1e-5)

    wp_w1_grad = gather(wp.experts.mlp.w1.grad)
    ep_w1_grad = permute(gather(ep.experts.mlp.w1.grad))
    if rank == 0:
        np.testing.assert_allclose(
            wp_w1_grad.float().cpu(),
            ep_w1_grad.float().cpu(),
            rtol=1e-5, atol=1e-5)

    # Verify the router weight gradient, which is not sharded.
    for i in range(torch.distributed.get_world_size(group)):
        torch.distributed.barrier(group)
        if i == rank:
            np.testing.assert_allclose(
                wp.router.layer.weight.grad.float().cpu(),
                ep.router.layer.weight.grad.float().cpu(),
                rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    assert torch.distributed.is_available()
    group = torch.distributed.init_process_group(backend='nccl')
    local_rank = torch.distributed.get_rank(group)
    torch.cuda.set_device(local_rank)

    for args in _TESTS:
        if local_rank == 0:
            print(f"TEST: {args}")
        test_expert_parallel_versus_weight_parallel(group, *args)
