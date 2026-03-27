# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Tuple

# NOTE: Torch needs to be imported before the custom
# extensions. Otherwise libc10.so cannot be found.
import torch

try:
    import megablocks_ops as _ops  # type: ignore
except ModuleNotFoundError:
    _ops = None

_BITS_FOR_DTYPE = {
    torch.int16: 16,
    torch.int32: 32,
    torch.int64: 64,
}


# Autograd wrapper for sort kernel.
# NOTE: Does not support gradients.
class SortOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, end_bit: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if end_bit is None:
            end_bit = _BITS_FOR_DTYPE[x.dtype]
        if x.device.type == 'npu':
            # `end_bit` is ignored by the PyTorch fallback.
            x_out, idx = torch.sort(x)
            return (x_out, idx.to(dtype=x.dtype))
        if _ops is None:
            raise ModuleNotFoundError("No module named 'megablocks_ops'.")
        x_out = torch.empty_like(x)
        iota_out = torch.empty_like(x)
        _ops.sort(x, end_bit, x_out, iota_out)
        return (x_out, iota_out)


sort = SortOp.apply
