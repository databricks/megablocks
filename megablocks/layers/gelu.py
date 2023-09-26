import stk
import torch
import torch.nn.functional as F


@torch.jit.script
def _gelu_backward_inplace(g, x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = (
        0.5 * x * (
            (1 - tanh_out * tanh_out) *
            (0.79788456 + 0.1070322243 * x * x)
        ) + 0.5 * (1 + tanh_out)
    )
    return g.mul_(ff)


def gelu_backward_(grad: stk.Matrix, x: stk.Matrix):
    # NOTE: The two sparse matrices must have the same topology.
    if isinstance(grad, stk.Matrix) and isinstance(x, stk.Matrix):
        return stk.Matrix(
            x.size(),
            _gelu_backward_inplace(grad.data, x.data),
            x.row_indices,
            x.column_indices,
            x.offsets,
            x.column_indices_t,
            x.offsets_t,
            x.block_offsets_t)
    return _gelu_backward_inplace(grad, x)


def gelu(x: stk.Matrix):
    assert isinstance(x, stk.Matrix)
    return stk.Matrix(
        x.size(),
        F.gelu(x.data, approximate="tanh"),
        x.row_indices,
        x.column_indices,
        x.offsets,
        x.column_indices_t,
        x.offsets_t,
        x.block_offsets_t)
