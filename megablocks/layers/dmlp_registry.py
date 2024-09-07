# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

from typing import Union

from megablocks.layers import glu, mlp
from megablocks.layers.arguments import Arguments

MlpType = Union[mlp.SparseMLP, glu.SparseGLU]

_REGISTRY = {
    'mlp': {
        'grouped': mlp.GroupedMLP,
        'sparse': mlp.SparseMLP,
    },
    'glu': {
        'grouped': glu.GroupedGLU,
        'sparse': glu.SparseGLU,
    },
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

    if args.mlp_impl not in _REGISTRY[args.mlp_type]:
        raise ValueError(f'{args.mlp_type} does not support {args.mlp_impl} backend.',)

    return _REGISTRY[args.mlp_type][args.mlp_impl](args)
