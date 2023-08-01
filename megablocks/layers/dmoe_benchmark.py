from megablocks import benchmark_util
from megablocks.layers.arguments import Arguments
from megablocks.layers import dmoe
import torch


_DMOE_BENCHMARK = (
    # dMoE-356M
    (16, 1024, 1024, 1024, 32, 4),
    # MPT-1B
    (2, 2048, 2048, 2048, 32, 4),
    # MPT-3B
    (2, 2048, 2560, 2560, 32, 4),
    # MPT-7B
    (2, 2048, 4096, 4096, 32, 4),
)


def benchmark_dmoe(
        expert_parallel_group,
        batch_size,
        sequence_length,
        hidden_size,
        ffn_hidden_size,
        num_experts,
        top_k):
    details = {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "hidden_size": hidden_size,
        "ffn_hidden_size": ffn_hidden_size,
        "num_experts": num_experts,
        "top_k": top_k,
    }
    args = Arguments(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        moe_num_experts=num_experts,
        moe_top_k=top_k,
        moe_expert_model_parallelism=True,
        bf16=True,
        fp16=False)
    layer = dmoe.dMoE(args).cuda().to(torch.bfloat16)

    x = torch.randn((batch_size, sequence_length, hidden_size)).cuda().to(torch.bfloat16)
    fn = lambda: layer(x)
    time, std = benchmark_util.benchmark_function(fn)

    if torch.distributed.get_rank(group) == 0:
        benchmark_util.log_benchmark("dMoE (Fwd)", details, time, std)


if __name__ == '__main__':
    assert torch.distributed.is_available()
    group = torch.distributed.init_process_group(backend='nccl')
    local_rank = torch.distributed.get_rank(group)
    torch.cuda.set_device(local_rank)

    for args in _DMOE_BENCHMARK:
        benchmark_dmoe(group, *args)
