import functools
import gc

from megablocks.layers import dmoe, arguments, mpu
from megablocks import benchmark_util
import numpy as np
import torch

_TESTS = (
    (8, 2048, 4096, 4096, 32, 4),

)


def get_tensors():
    ptrs = set([])
    out = []
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            if not obj.is_contiguous() or obj.data_ptr() in ptrs:
                continue
            out.append(obj)
            ptrs.add(obj.data_ptr())
    return out


def test_memory(
        group,
        batch_size,
        sequence_length,
        hidden_size,
        ffn_hidden_size,
        num_experts,
        top_k):
    args = arguments.Arguments(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        moe_num_experts=num_experts,
        moe_top_k=top_k,
        moe_expert_model_parallelism=True,
        expert_parallel_group=group,
        fp16=False,
        bf16=True,
        device=torch.cuda.current_device())
    layer = dmoe.dMoE(args).cuda()

    x = torch.randn(
        (batch_size, sequence_length, hidden_size),
        device=torch.cuda.current_device(),
        dtype=torch.bfloat16).requires_grad_(True)
    torch.cuda.empty_cache()

    # Run forward + backward.
    # with torch.autograd.detect_anomaly():
    out, _ = layer(x)
    out.mean().backward()

    # Report peak memory.
    mem = torch.cuda.max_memory_allocated()
    print("Max Memory Allocated = {:0.0f}MiB".format(
        mem / 1e6))
    print("Max Memory Reserved = {:0.0f}MiB".format(
        torch.cuda.max_memory_reserved() / 1e6))

    # Calculate weight and gradient memory usage.
    weight_memory = 2 * (
        layer.router.layer.weight.numel() +
        layer.experts.mlp.w1.numel() +
        layer.experts.mlp.w2.numel())

    def grad_numel(x):
        if x.grad is not None:
            return x.grad.numel()
        return 0

    grad_memory = 2 * (
        grad_numel(layer.router.layer.weight) +
        grad_numel(layer.experts.mlp.w1) +
        grad_numel(layer.experts.mlp.w2))
    weight_memory += grad_memory

    print("Weight Memory Allocated = {:0.0f}MiB".format(
        weight_memory / 1e6))
    print("Activation Memory Allocated = {:0.0f}MiB".format(
        (mem - weight_memory) / 1e6))

    # Manually calculate GPU memory usage from the garbage
    # collector.
    gc.collect()
    total = 0
    tensors = get_tensors()
    tensors = sorted(tensors, key=lambda x: -x.numel())
    for i, t in enumerate(tensors):
        total += t.numel()
        print(f"{i}: {t.shape}, {t.numel() * 2}")
    del tensors

    print("Total Bytes Found = {:0.0f}MiB".format(
        total * 2 / 1e6))


if __name__ == '__main__':
    assert torch.distributed.is_available()
    group = torch.distributed.init_process_group(backend='nccl')
    local_rank = torch.distributed.get_rank(group)
    torch.cuda.set_device(local_rank)

    for args in _TESTS:
        test_memory(group, *args)
