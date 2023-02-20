from megablocks.layers.arguments import Arguments
import torch


def is_moe_param(tensor : torch.Tensor) -> bool:
    return hasattr(tensor, 'expert_model_parallel')


def get_expert_parallel_world_size(args : Arguments) -> int:
    return (
        torch.distributed.get_world_size(args.expert_parallel_group)
        if args.moe_expert_model_parallelism else 1
    )


def get_expert_parallel_rank(args : Arguments) -> int:
    return (
        torch.distributed.get_rank(args.expert_parallel_group)
        if args.moe_expert_model_parallelism else 0
    )


def set_expert_model_parallel_attributes(tensor : torch.Tensor,
                                         is_parallel : bool):
    assert not hasattr(tensor, 'expert_model_parallel')
    setattr(tensor, 'expert_model_parallel', is_parallel)


def param_is_expert_model_parallel(param : torch.Tensor) -> bool:
    return (hasattr(param, 'expert_model_parallel') and
            param.expert_model_parallel)


def copy_expert_model_parallel_attributes(destination_tensor : torch.Tensor,
                                          source_tensor : torch.Tensor):
    if hasattr(source_tensor, 'expert_model_parallel'):
        setattr(destination_tensor, 'expert_model_parallel',
                getattr(source_tensor,'expert_model_parallel'))


def synchronized_print(args : Arguments, *x):
    rank = get_expert_parallel_rank(args)
    for i in range(get_expert_parallel_world_size(args)):
        torch.distributed.barrier(args.expert_parallel_group)
        if i == rank:
            print(f"rank = {rank}", *x)
