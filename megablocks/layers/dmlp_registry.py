from typing import Union
from megablocks.layers import mlp
from megablocks.layers import glu
from megablocks.layers.arguments import Arguments

MlpType = Union[mlp.SparseMLP, glu.SparseGLU]

class DMlpRegistry:
    """
    Abstraction for creating different underlying MLPs.
    Currently only supports MLPs that are used by dMoE. 
    """
    REGISTRY = {
        'mlp': {'grouped': mlp.GroupedMLP, 'sparse' : mlp.SparseMLP},
        'glu': {'grouped': glu.GroupedGLU, 'sparse': glu.SparseGLU},
    }

    @staticmethod
    def get(args: Arguments) -> MlpType:

        if args.mlp_type not in DMlpRegistry.REGISTRY: 
            raise ValueError(f'Unsupported mlp type: {args.mlp_type}')

        mlp_impl = 'grouped' if args.use_grouped_gemm else 'sparse'

        if mlp_impl not in DMlpRegistry.REGISTRY[args.mlp_type]:
            raise ValueError(f'{args.mlp_type} does not support {mlp_impl} backend.')

        return DMlpRegistry.REGISTRY[args.mlp_type][mlp_impl](args)