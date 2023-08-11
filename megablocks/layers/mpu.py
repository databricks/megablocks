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


def get_weight_parallel_world_size(args : Arguments) -> int:
    return (
        torch.distributed.get_world_size(args.weight_parallel_group)
        if args.moe_weight_parallelism else 1
    )


def get_weight_parallel_rank(args : Arguments) -> int:
    return (
        torch.distributed.get_rank(args.weight_parallel_group)
        if args.moe_weight_parallelism else 0
    )


def synchronized_print(group, *x):
    world_size = torch.distributed.get_world_size(group)
    rank = torch.distributed.get_rank(group)
    for i in range(world_size):
        torch.distributed.barrier(group)
        if i == rank:
            print(f"rank = {rank}", *x)


# Helpers for expert/tensor sharding.
def expert_sharding_degree(args : Arguments) -> int:
    world_size = get_expert_parallel_world_size(args)
    esd = min(world_size, args.moe_num_experts)

    if (args.moe_num_experts % esd) != 0:
        raise ValueError(
            f"Cannot shard {args.moe_num_experts} experts {esd} ways.")
    return esd


def hidden_sharding_degree(args : Arguments) -> int:
    world_size = get_expert_parallel_world_size(args)
    esd = expert_sharding_degree(args)
    hsd = world_size // esd

    if (args.ffn_hidden_size % hsd) != 0:
        raise ValueError(
            f"Cannot shard {args.ffn_hidden_size} features {hsd} ways.")
    if (esd * hsd) != world_size:
        raise ValueError(
            f"Invalid sharding. 'expert_sharding_degree' "
            f"({esd}) * hidden_sharding_degree "
            f"({hsd}) != world_size ({world_size}).")
    return hsd


def experts_per_rank(args : Arguments) -> int:
    return args.moe_num_experts // expert_sharding_degree(args)


def features_per_rank(args : Arguments) -> int:
    return args.ffn_hidden_size // hidden_sharding_degree(args)
