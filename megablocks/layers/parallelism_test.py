from megablocks.layers import dmoe, arguments
from megablocks import benchmark_util
import numpy as np
import torch

_TESTS = (
    (64, 1024, 512, 2048, 64, 1),
)

def test_expert_parallel_versus_weight_parallel(
        group, batch_size, sequence_length, hidden_size, ffn_hidden_size, num_experts, top_k):
    torch.manual_seed(1234)

    ep_args = arguments.Arguments(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        moe_num_experts=num_experts,
        moe_top_k=top_k,
        moe_expert_model_parallelism=True,
        expert_parallel_group=group,
        fp16=False,
        bf16=True,
        device=torch.cuda.current_device())
    wp_args = arguments.Arguments(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        moe_num_experts=num_experts,
        moe_top_k=top_k,
        moe_weight_parallelism=True,
        weight_parallel_group=group,
        fp16=False,
        bf16=True,
        device=torch.cuda.current_device())
    ep = dmoe.dMoE(ep_args)
    wp = dmoe.dMoE(wp_args)

    # Copy the weights from ep to wp.
    with torch.no_grad():
        wp.router.layer.weight.copy_(ep.router.layer.weight)
        wp.mlp.w1.copy_(ep.mlp.w1)
        wp.mlp.w2.copy_(ep.mlp.w2)

    x = torch.randn(
        (batch_size, sequence_length, hidden_size),
        device=torch.cuda.current_device(),
        dtype=torch.bfloat16).requires_grad_(True)

    # Test forward.
    out, _ = wp(x)
    expected_out, _ = ep(x)
    assert torch.allclose(out, expected_out)

    # Test backward.
    out.mean().backward()
    expected_out.mean().backward()

    rank = torch.distributed.get_rank(group)
    for i in range(torch.distributed.get_world_size(group)):
        torch.distributed.barrier(group)
        if i == rank:
            np.testing.assert_allclose(
                wp.mlp.w2.grad.float().cpu(),
                ep.mlp.w2.grad.float().cpu(),
                rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(
                wp.mlp.w1.grad.float().cpu(),
                ep.mlp.w1.grad.float().cpu(),
                rtol=1e-5, atol=1e-5)
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
        test_expert_parallel_versus_weight_parallel(group, *args)
