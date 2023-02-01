import megatron
from megatron import mpu
from megatron.model.module import MegatronModule
from megablocks import layers
import torch


class MoE(MegatronModule):

    def __init__(self, init_method, output_layer_init_method):
        super().__init__()
        args = layers.arguments.from_megatron(megatron.get_args())
        args.device = torch.cuda.current_device()
        args.init_method = init_method
        args.output_layer_init_method = output_layer_init_method
        if args.expert_model_parallelism:
            args.expert_parallel_group = mpu.get_data_parallel_group()
        self.moe = layers.moe.MoE(args)

    def forward(self, x):
        return self.moe.forward(x)
