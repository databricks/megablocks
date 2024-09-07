# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.distributed as dist

from megablocks import benchmark_util
from megablocks.layers.all_to_all import all_to_all

_ALL_TO_ALL_BENCHMARK = (
    (8, 1024),
    (16, 1024),
    (32, 1024),
    (64, 1024),
    (128, 1024),
    (256, 1024),
    (512, 1024),
    (1024, 1024),
    (2 * 1024, 1024),
    (4 * 1024, 1024),
    (8 * 1024, 1024),
    (16 * 1024, 1024),
    (32 * 1024, 1024),
    (64 * 1024, 1024),
    (128 * 1024, 1024),
    (256 * 1024, 1024),
    (512 * 1024, 1024),
    (1024 * 1024, 1024),
)


def benchmark_all_to_all(group, sl, hs):
    world_size = dist.get_world_size(group)
    assert (sl % world_size) == 0
    send_recv_sizes = [sl // world_size] * world_size

    x = torch.randn((sl, hs)).cuda().half()

    details = {
        'world_size': world_size,
        'message_size (B)': send_recv_sizes[0] * hs * 2,  # 2B elements.
    }

    def benchmark():
        return all_to_all(x, send_recv_sizes, send_recv_sizes, group)

    time, std = benchmark_util.benchmark_function(benchmark)

    if dist.get_rank(group) == 0:
        benchmark_util.log_benchmark('All-To-All', details, time, std)


if __name__ == '__main__':
    assert dist.is_available()
    group = dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank(group)
    torch.cuda.set_device(local_rank)

    for args in _ALL_TO_ALL_BENCHMARK:
        benchmark_all_to_all(group, *args)
