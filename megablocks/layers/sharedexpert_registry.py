# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

from typing import Union

from megablocks.layers import glu, mlp
from megablocks.layers.arguments import Arguments

_REGISTRY = {
    'mlp': mlp.SharedMLP,
    'glu': glu.SharedGLU,
}


def get(args: Arguments) -> Union[mlp.SharedMLP, glu.SharedGLU]:
    """Returns an SharedMLP for use in a dMoE instance.

    Uses the provided arguments to instantiate the appropriate
    SharedMLP instance.

    Args:
        args: propagated Arguments dataclass.

    Returns:
        An instantiated SharedMLP constructed using the input args.
    """
    if args.mlp_type not in _REGISTRY:
        raise ValueError(f'Unsupported mlp type: {args.mlp_type}')

    return _REGISTRY[args.mlp_type](args)
