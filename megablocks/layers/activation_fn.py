# Copyright 2024 MosaicML MegaBlocks authors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import stk
import torch


def act_fn(
    x: stk.Matrix,
    function: Callable,
    return_grad_fn: bool = False,
    **kwargs,
):
    assert isinstance(x, stk.Matrix)
    with torch.set_grad_enabled(torch.is_grad_enabled() or return_grad_fn):
        if return_grad_fn:
            x.data.requires_grad = True
        out = function(x.data, **kwargs)
        y = stk.Matrix(
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
