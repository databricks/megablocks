import dataclasses
from functools import partial
import torch
from typing import Callable, Optional

# Type annotation for in-place Tensor initialization function.
InitFn = Callable[[torch.Tensor], None]


@dataclasses.dataclass
class Arguments:
    # Model arguments.
    hidden_size : int = 1024
    ffn_hidden_size : int = 4096
    num_layers : int = 1

    # MoE arguments.
    moe_num_experts : int = 1
    moe_top_k : int = 1
    moe_capacity_factor : int = 1
    moe_loss_weight : float = 0.1
    moe_jitter_eps : Optional[float] = None
    moe_lbl_in_fp32 : bool = False

    # Parallelism arguments.
    moe_expert_model_parallelism : bool = False
    expert_parallel_group : Optional[torch.distributed.ProcessGroup] = None
    pipeline_model_parallel_size : int = 1
    num_layers_per_virtual_pipeline_stage : Optional[int] = None

    # Initialization arguments.
    fp16 : bool = True
    bf16: bool = False
    device : torch.device = torch.cuda.current_device()
    init_method : InitFn =  partial(torch.nn.init.normal_, mean=0.0, std=0.02)
    output_layer_init_method : InitFn = init_method


def from_megatron(megatron_args):
    args = Arguments()
    for field in dataclasses.fields(args):
        if hasattr(megatron_args, field.name):
            setattr(args, field.name, getattr(megatron_args, field.name))
    return args
