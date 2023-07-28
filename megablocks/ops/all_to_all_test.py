from megablocks import ops
from megablocks.layers.all_to_all import all_to_all
import torch


_ALL_TO_ALL_TEST = (
    (8, 1, 1024),
    (16, 1, 1024),
    (32, 1, 1024),
    (64, 1, 1024),
    (128, 1, 1024),
    (256, 1, 1024),
    (512, 1, 1024),
    (1024, 1, 1024),
    (8, 2, 1024),
    (16, 2, 1024),
    (32, 2, 1024),
    (64, 2, 1024),
    (128, 2, 1024),
    (256, 2, 1024),
    (512, 2, 1024),
    (1024, 2, 1024),
    (8, 32, 1024),
    (8, 32, 1024),
    (16, 32, 1024),
    (32, 32, 1024),
    (64, 32, 1024),
    (128, 32, 1024),
    (256, 32, 1024),
    (512, 32, 1024),
    (1024, 32, 1024),
)


def test_all_to_all(group, sl, ne, hs):
        rank = torch.distributed.get_rank(group)
        world_size = torch.distributed.get_world_size(group)
        assert (sl % world_size) == 0
        send_recv_sizes = [sl // world_size] * world_size * ne

        x = torch.randn((sl * ne, hs)).cuda().half()

        out = ops.all_to_all(x, send_recv_sizes, send_recv_sizes, group)

        # NOTE: This assumes the tokens are evently distributed.
        send_recv_sizes = [(sl * ne) // world_size] * world_size
        expected_out, _ = all_to_all(x, send_recv_sizes, send_recv_sizes, group)
        expected_out = expected_out.view(world_size, ne, sl // world_size, hs)
        expected_out = expected_out.transpose(0, 1).contiguous()
        expected_out = expected_out.view(sl * ne, hs)

        if not torch.equal(out, expected_out):
            for i in range(world_size):
                torch.distributed.barrier(group)
                if i == rank:
                    print(f"FAILED: sl = {sl}, ne = {ne}, hs = {hs}")
                    print(f"rank = {rank}, out = {out}")
                    print(f"rank = {rank}, expected_out = {expected_out}")
        else:
            if rank == 0:
                print("PASSED")


if __name__ == '__main__':
    assert torch.distributed.is_available()
    group = torch.distributed.init_process_group(backend='nccl')
    local_rank = torch.distributed.get_rank(group)
    torch.cuda.set_device(local_rank)

    for args in _ALL_TO_ALL_TEST:
        if local_rank == 0:
            print(f"Testing {args}:")
        test_all_to_all(group, *args)
