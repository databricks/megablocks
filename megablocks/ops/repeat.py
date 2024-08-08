# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

# Copyright 2024 MosaicML MegaBlocks authors
# SPDX-License-Identifier: Apache-2.0


def repeat(x, tiling):
    if all((t == 1 for t in tiling)):
        return x
    return x.repeat(*tiling)
