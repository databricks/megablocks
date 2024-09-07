# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Union

import torch
from stk import Matrix


def act_fn(
    x: Matrix,
    function: Callable,
    return_grad_fn: bool = False,
    **kwargs,
) -> Union[tuple[Matrix, Any] | Matrix]:
    assert isinstance(x, Matrix)
    with torch.set_grad_enabled(torch.is_grad_enabled() or return_grad_fn):
        if return_grad_fn:
            x.data.requires_grad = True
        out = function(x.data, **kwargs)
        y = Matrix(
            x.size(),
            out,
            x.row_indices,
            x.column_indices,
            x.offsets,
            x.column_indices_t,
            x.offsets_t,
            x.block_offsets_t,
        )
        if return_grad_fn:
            return y, out.backward
        return y
