import dataclasses
from functools import partial
import megablocks.turbo_util as turbo
import megablocks.grouped_gemm_util as grouped_gemm
import torch
from typing import Callable, Optional, Union

# Type annotation for in-place Tensor initialization function.
InitFn = Callable[[torch.Tensor], None]

_ALLOWED_BITWIDTHS = (-1, 4, 8)


@dataclasses.dataclass
class Arguments:
    # Model arguments.
    hidden_size : int = 1024
    ffn_hidden_size : int = 4096
    num_layers : int = 1
    bias : bool = True
    return_bias : bool = True

    # MoE arguments.
    moe_num_experts : int = 1
    moe_top_k : int = 1
    moe_capacity_factor : int = 1
    moe_normalize_expert_weights: Optional[Union[int, float]] = None
    moe_loss_weight : float = 0.1
    moe_jitter_eps : Optional[float] = None
    moe_lbl_in_fp32 : bool = False

    # Parallelism arguments.
    moe_expert_model_parallelism : bool = False
    expert_parallel_group : Optional[torch.distributed.ProcessGroup] = None
    moe_weight_parallelism : bool = False
    weight_parallel_group : Optional[torch.distributed.ProcessGroup] = None
    pipeline_model_parallel_size : int = 1
    num_layers_per_virtual_pipeline_stage : Optional[int] = None

    # Compute arguments.
    memory_optimized_mlp : bool = False
    mlp_type : str = 'mlp'
    grouped_mlp : bool = False
    quantize_inputs_num_bits: int = -1  # -1 = no quantization
    quantize_rematerialize_num_bits: int = -1
    quantize_scatter_num_bits: int = -1

    # Initialization arguments.
    fp16 : bool = True
    bf16: bool = False
    device : torch.device = torch.cuda.current_device()
    init_method : InitFn =  partial(torch.nn.init.normal_, mean=0.0, std=0.02)
    output_layer_init_method : InitFn = init_method

    # Benchmarking arguments.
    uniform_expert_assignment : bool = False

    def __post_init__(self):
        for attr in ('quantize_inputs_num_bits',
                     'quantize_rematerialize_num_bits',
                     'quantize_scatter_num_bits'):
            nbits = self.__getattribute__(attr)
            if nbits not in _ALLOWED_BITWIDTHS:
                raise ValueError(f'{attr} must be one of ' +
                                 f'{_ALLOWED_BITWIDTHS}; got {nbits}')

            if nbits != -1:
                turbo.assert_turbo_is_available()

        if self.__getattribute__('grouped_mlp'):
            grouped_gemm.assert_grouped_gemm_is_available()


def from_megatron(megatron_args):
    args = Arguments()
    for field in dataclasses.fields(args):
        if hasattr(megatron_args, field.name):
            setattr(args, field.name, getattr(megatron_args, field.name))
    return args
