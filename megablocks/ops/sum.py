# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0
import torch


def sum(x: torch.Tensor, dim: int = 0):
    if x.shape[dim] == 1:
        return x.squeeze(dim=dim)
    return x.sum(dim=dim)
