# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

import torch


def repeat(x: torch.Tensor, tiling: torch.Size):
    if all((t == 1 for t in tiling)):
        return x
    return x.repeat(*tiling)
