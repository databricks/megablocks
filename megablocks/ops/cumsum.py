# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

from typing import Any

# NOTE: Torch needs to be imported before the custom
# extensions. Otherwise libc10.so cannot be found.
import torch

try:
    import megablocks_ops as _ops  # type: ignore
except ModuleNotFoundError:
    _ops = None


def _inclusive_cumsum_fallback(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.cumsum(x, dim=dim)


def _exclusive_cumsum_fallback(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.cumsum(x, dim=dim) - x


# Autograd wrappers for cumsum kernels.
# NOTE: Does not support gradients.
class ExclusiveCumsumOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, dim: int):
        if len(x.size()) == 1:
            x = x.view([1, -1])
            if x.device.type == 'npu':
                return _exclusive_cumsum_fallback(x, 1).squeeze()
            if _ops is None:
                raise ModuleNotFoundError("No module named 'megablocks_ops'.")
            out = torch.empty_like(x)
            _ops.exclusive_cumsum(x, 1, out)
            return out.squeeze()
        if x.device.type == 'npu':
            return _exclusive_cumsum_fallback(x, dim)
        if _ops is None:
            raise ModuleNotFoundError("No module named 'megablocks_ops'.")
        out = torch.empty_like(x)
        _ops.exclusive_cumsum(x, dim, out)
        return out


exclusive_cumsum = ExclusiveCumsumOp.apply


class InclusiveCumsumOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, dim: int) -> torch.Tensor:
        if len(x.size()) == 1:
            x = x.view([1, -1])
            if x.device.type == 'npu':
                return _inclusive_cumsum_fallback(x, 1).squeeze()
            if _ops is None:
                raise ModuleNotFoundError("No module named 'megablocks_ops'.")
            out = torch.empty_like(x)
            _ops.inclusive_cumsum(x, 1, out)
            return out.squeeze()
        if x.device.type == 'npu':
            return _inclusive_cumsum_fallback(x, dim)
        if _ops is None:
            raise ModuleNotFoundError("No module named 'megablocks_ops'.")
        out = torch.empty_like(x)
        _ops.inclusive_cumsum(x, dim, out)
        return out


inclusive_cumsum = InclusiveCumsumOp.apply
