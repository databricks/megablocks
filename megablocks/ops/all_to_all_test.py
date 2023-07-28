from megablocks import ops
from megablocks.layers.all_to_all import all_to_all
import torch

_ALL_TO_ALL_TEST = (
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


def test_all_to_all(group, sl, hs):
        world_size = torch.distributed.get_world_size(group)
        assert (sl % world_size) == 0
        send_recv_sizes = [sl // world_size] * world_size

        x = torch.randn((sl, hs)).cuda().half()

        out = ops.all_to_all(x, send_recv_sizes, send_recv_sizes, group)
        # expected_out, _ = all_to_all(x, send_recv_sizes, send_recv_sizes, group)
        # if not torch.equal(out, expected_out):

        #     rank = torch.distributed.get_rank(group)
        #     for i in range(world_size):
        #         torch.distributed.barrier(group)
        #         if i == rank:
        #             print(f"FAILED: sl = {sl}, hs = {hs}")
        #             print(f"rank = {rank}, out = {out}")
        #             print(f"rank = {rank}, expected_out = {expected_out}")

if __name__ == '__main__':
    assert torch.distributed.is_available()
    group = torch.distributed.init_process_group(backend='nccl')
    local_rank = torch.distributed.get_rank(group)
    torch.cuda.set_device(local_rank)

    for args in _ALL_TO_ALL_TEST:
        if local_rank == 0:
            print(f"Testing {args}:")
        test_all_to_all(group, *args)
