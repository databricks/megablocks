from megablocks.layers import arguments
from megablocks.layers import dmoe
from megablocks.layers import moe
import megatron
from megatron.core import parallel_state
from megatron.model.module import MegatronModule
import torch


class MegatronHelper(MegatronModule):

    def __init__(self, layer_cls, init_method, output_layer_init_method):
        super().__init__()
        args = arguments.from_megatron(megatron.get_args())
        args.device = torch.cuda.current_device()
        args.init_method = init_method
        args.output_layer_init_method = output_layer_init_method
        if args.expert_model_parallelism:
            args.expert_parallel_group = parallel_state.get_data_parallel_group()
        self.moe = layer_cls(args)

    def forward(self, x):
        return self.moe.forward(x)


class MoE(MegatronHelper):

    def __init__(self, init_method, output_layer_init_method):
        super().__init__(moe.MoE, init_method, output_layer_init_method)


class dMoE(MegatronHelper):

    def __init__(self, init_method, output_layer_init_method):
        super().__init__(dmoe.dMoE, init_method, output_layer_init_method)
