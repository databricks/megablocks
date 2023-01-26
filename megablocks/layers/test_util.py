from dataclasses import dataclass
import os

import megatron
from megatron import initialize
import torch


_IS_INITIALIZED = False


def initialize_megatron():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6000"
    initialize.set_jit_fusion_options()
    initialize._initialize_distributed()
    initialize._set_random_seed(1234)
    initialize._compile_dependencies()


@dataclass
class Arguments:
    fp16 : bool = True
    bf16 : bool = False
    apply_query_key_layer_scaling : bool = True
    attention_softmax_in_fp32 : bool = False
    masked_softmax_fusion : bool = True
    attention_dropout : float = 0.0
    kv_channels : int = 128
    num_attention_heads : int = 8
    hidden_size : int = 1024
    rank : int = 0
    local_rank : int = 0
    distributed_backend : str = "nccl"
    world_size : int = 1
    tensor_model_parallel_size : int = 1
    pipeline_model_parallel_size : int = 1
    virtual_pipeline_model_parallel_size = None
    pipeline_model_parallel_split_rank = None
    no_async_tensor_model_parallel_allreduce : bool = True
    seq_length : int = 4
    micro_batch_size : int = 2
    use_cpu_initialization : bool = False
    params_dtype = torch.float16
    ffn_hidden_size : int = 4096
    num_layers : int = 1
    bias_gelu_fusion : bool = True
    openai_gelu : bool = False
    onnx_safe = None
    apply_residual_connection_post_layernorm : bool = False
    fp32_residual_connection : bool = False
    bias_dropout_fusion : bool = True
    layernorm_epsilon : float = 1e-5
    hidden_dropout : float = 0.0
    fp16_lm_cross_entropy : bool = False
    init_method_std : float = 0.02
    padded_vocab_size : int = 51200
    max_position_embeddings : int = 1024
    activations_checkpoint_method = None
    activations_checkpoint_num_layers : int = 1
    distribute_checkpointed_activations : bool = False
    DDP_impl : str = "local"
    accumulate_allreduce_grads_in_fp32 : bool = False
    use_contiguous_buffers_in_local_ddp : bool = True
    optimizer : str = "adam"
    lr : float = 0.00015
    weight_decay : float = 0.01
    adam_beta1 : float = 0.9
    adam_beta2 : float = 0.999
    adam_eps : float = 1e-08
    loss_scale = None
    initial_loss_scale : int = 4294967296
    min_loss_scale : float = 1.0
    loss_scale_window : int = 1000
    hysteresis : int = 2
    clip_grad : float = 1.0
    log_num_zeros_in_grad : bool = False
    train_iters : int = 20000
    lr_decay_iters : int = 20000
    lr_decay_style : str = "cosine"
    global_batch_size : int = 512
    lr_warmup_fraction : float = 0.01
    min_lr : float = 1e-05
    use_checkpoint_lr_scheduler : bool = False
    override_lr_scheduler : bool = False
    load = None
    mob_scale_factor : int = 1
    mob_router_type : str = "dense"
    mob_type : str = "2d"
    moe_scale_factor : int = 1
    moe_expert_capacity : int = 1
    moe_jitter_eps : float = None
    moe_experts_per_token : int = 1
    moe_log_tokens_per_expert : bool = False
    save : str = None
    moe_num_centers : int = 1
    moe_normalize_weights : bool = False
    moe_debug_logging : bool = False
    moe_learned_routing : bool = True
    moe_routing_mode : str = "centers_to_experts"
    moe_softmax_block_weights : bool = True
    sequence_parallel : bool = False
    gradient_accumulation_fusion : bool = False
    async_tensor_model_parallel_allreduce : bool = False
    moe_graphed_functions : bool = False
    moe_loss_weight : float = 1.0
    perform_initialization : bool = True
    expert_model_parallelism : bool = False
    moe_lbl_in_fp32 : bool = False
    num_layers_per_virtual_pipeline_stage : int = None
    moe_expert_sparsity: float = 0.0
    num_experts : int = 0


class Timers:

    class Nop:

        def start(self):
            pass

        def stop(self):
            pass

    def __call__(self, _):
        return self.Nop()


def set_megatron_arguments(**kwargs):
    out = Arguments(**kwargs)
    megatron.global_vars._GLOBAL_ARGS = out
    megatron.global_vars._GLOBAL_TIMERS = Timers()
    global _IS_INITIALIZED
    if not _IS_INITIALIZED:
        initialize_megatron()
        _IS_INITIALIZED = True
