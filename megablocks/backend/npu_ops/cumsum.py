# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

import torch


def inclusive_cumsum(x: torch.Tensor, dim: int, out: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2
    assert dim == 1
    return torch.cumsum(x, dim=dim, out=out)


def exclusive_cumsum(x: torch.Tensor, dim: int, out: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2
    assert dim == 1

    if out is not None:
        torch.cumsum(x, dim=dim, out=out)
        out.sub_(x)
        return out
    return torch.cumsum(x, dim=dim) - x

