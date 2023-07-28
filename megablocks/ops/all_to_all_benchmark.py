from megablocks import benchmark_util
from megablocks import ops
import torch


_ALL_TO_ALL_BENCHMARK = (
    # dMoE-Medium e32, top-1
    (16 * 1024, 1, 1024),
    (16 * 1024 // 32, 32, 1024),
    # dMoE-Medium e64, top-1
    (16 * 1024, 1, 1024),
    (16 * 1024 // 64, 64, 1024),
    # dMoE-Medium e32, top-4
    (4 * 16 * 1024, 1, 1024),
    (4 * 16 * 1024 // 32, 32, 1024),
    # dMoE-Medium e64, top-4
    (4 * 16 * 1024, 1, 1024),
    (4 * 16 * 1024 // 64, 64, 1024),
    # hs=4k, e32, top-4, bs=2, sl=2k
    (4 * 2 * 2048, 1, 4096),
    (4 * 2 * 2048 // 32, 32, 4096),
    # hs=8k, e32, top4, bs=1, sl=2k
    (4 * 2048, 1, 8192),
    (4 * 2048 // 32, 32, 8192),
    # hs=16k, e32, top4, bs=1, sl=2k
    (4 * 2048, 1, 16384),
    (4 * 2048 // 32, 32, 16384)
)


def benchmark_all_to_all(group, sl, ne, hs):
        world_size = torch.distributed.get_world_size(group)
        assert (sl % world_size) == 0
        send_recv_sizes = [sl // world_size] * world_size * ne

        x = torch.randn((sl * ne, hs)).cuda().half()

        details = {
            "world_size": world_size,
            "sequence_length": sl,
            "num_experts": ne,
            "hidden_size": hs,
            "total_size (B)": sum(send_recv_sizes) * hs * 2,  # 2B elements.
            "message_size (B)": send_recv_sizes[0] * hs * 2,  # 2B elements.
        }

        fn = lambda: ops.all_to_all(x, send_recv_sizes, send_recv_sizes, group)
        time, std = benchmark_util.benchmark_function(fn)

        if torch.distributed.get_rank(group) == 0:
            benchmark_util.log_benchmark("All-To-All", details, time, std)


if __name__ == '__main__':
    assert torch.distributed.is_available()
    group = torch.distributed.init_process_group(backend='nccl')
    local_rank = torch.distributed.get_rank(group)
    torch.cuda.set_device(local_rank)

    for args in _ALL_TO_ALL_BENCHMARK:
        benchmark_all_to_all(group, *args)
