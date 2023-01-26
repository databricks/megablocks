import megatron
from megatron import mpu
import torch


def init_weight_gpu(weight, init_method):
    if megatron.mpu.model_parallel_is_initialized():
        with megatron.mpu.get_cuda_rng_tracker().fork():
            init_method(weight)
    else:
        init_method(weight)


def synchronized_print(*x):
    rank = mpu.get_data_parallel_rank()
    for i in range(mpu.get_data_parallel_world_size()):
        torch.distributed.barrier(group=mpu.get_data_parallel_group())
        if i == rank:
            print(f"rank = {rank}", *x)
