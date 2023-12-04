from typing import Union
from megablocks.layers import mlp
from megablocks.layers import glu
from megablocks.layers.arguments import Arguments

MlpType = Union[mlp.SparseMLP, glu.SparseGLU]

_REGISTRY = {
    'mlp': {'grouped': mlp.GroupedMLP, 'sparse' : mlp.SparseMLP, 'torch' : mlp.TorchMLP},
    'glu': {'grouped': glu.GroupedGLU, 'sparse': glu.SparseGLU},
}

def get(args: Arguments) -> MlpType:
    """Returns an MLP for use in a dMoE instance.

    Uses the provided arguments to instantiate the appropriate
    MLP instance. This only contains MLPs for use in dMoEs 
    (ie. only for the dropless versions of MoEs).

    Args:
        args: propagated Arguments dataclass.

    Returns:
        An instantiated MLP constructed using the input args.

    """
    if args.mlp_type not in _REGISTRY: 
        raise ValueError(f'Unsupported mlp type: {args.mlp_type}')
    if args.torch_mlp:
        mlp_impl = 'torch'
    elif args.grouped_mlp:
        mlp_impl = 'grouped'
    else: 
        mlp_impl = 'sparse'

    if mlp_impl not in _REGISTRY[args.mlp_type]:
        raise ValueError(f'{args.mlp_type} does not support {mlp_impl} backend.')

    return _REGISTRY[args.mlp_type][mlp_impl](args)