# Copyright 2024 MosaicML MegaBlocks authors
# SPDX-License-Identifier: Apache-2.0

from megablocks._version import __version__
from megablocks.layers import dmoe, moe

"""Some key classes are available directly in the ``MegaBlocks`` namespace."""

__all__ = [
    'dmoe',
    'moe',
]
