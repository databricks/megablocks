import stk
import torch.nn.functional as F

def gelu(x):
    assert isinstance(x, stk.Matrix)
    return stk.Matrix(
        x.size(),
        F.gelu(x.data, approximate=True),
        x.row_indices,
        x.column_indices,
        x.offsets,
        x.column_indices_t,
        x.offsets_t,
        x.block_offsets_t)
