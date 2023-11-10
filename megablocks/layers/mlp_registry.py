from typing import Union
from megablocks.layers import mlp
from megablocks.layers import glu
from megablocks.layers.arguments import Arguments

MlpType = Union[mlp.SparseMLP, glu.SparseGLU]

class MlpRegistry:
    """
    Abstraction for creating different underlying MLPs.
    """
    REGISTRY = {
        'mlp': {'grouped': mlp.GroupedMLP, 'sparse' : mlp.SparseMLP},
        'glu': {'grouped': glu.GroupedGLU, 'sparse': glu.SparseGLU},
    }

    @staticmethod
    def get(args: Arguments) -> MlpType:

        if args.mlp_type not in MlpRegistry.REGISTRY: 
            raise ValueError(f'Unsupported mlp type: {args.mlp_type}')

        mlp_impl = 'grouped' if args.use_grouped_gemm else 'sparse'

        if mlp_impl not in MlpRegistry.REGISTRY[args.mlp_type]:
            raise ValueError(f'{args.mlp_type} does not support {mlp_impl} backend.')

        return MlpRegistry.REGISTRY[args.mlp_type][mlp_impl](args)