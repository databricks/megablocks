import unittest
from functools import partial

from absl.testing import parameterized
from megablocks.layers.arguments import Arguments
from megablocks.layers.glu import SparseGLU, GroupedGLU
from megablocks.layers import testing

from megablocks import ops
import torch
import stk
import numpy as np

def _sparse_transpose(size, row_indices, column_indices, offsets, blocking, transpose_sort_end_bit):
    block_columns = size[1] // blocking

    # Sort row indices by column indices to get the transposed matrix's
    # column indices.
    #
    # NOTE: Our sort operation uses the same width indices as the input values.
    # To avoid overflow when we have large activation matrices we cast to
    # 32-bit before sorting.
    _, gather_indices = ops.sort(
        column_indices.int(), transpose_sort_end_bit)

    # There are a constant number of blocks in every row of the sparse matrix.
    # A blocks offset is:
    #
    # row_index * blocks_per_row + column_index % blocks_per_row
    #
    # Once we have the block offsets ordered for transposition we can divide
    # by blocks_per_row to get the transposed column indices.
    column_indices_t = row_indices.gather(0, gather_indices.long())
    block_offsets_t = gather_indices.int()

    zero = torch.zeros((1,), dtype=torch.int32, device=row_indices.device)
    nnz_per_column = ops.histogram(column_indices, block_columns)
    nnz_per_column = ops.inclusive_cumsum(nnz_per_column, 0)
    offsets_t = torch.cat([zero, nnz_per_column])
    return column_indices_t, offsets_t, block_offsets_t

def _setup_topology(bs, sl, ffn_hidden_size, blocking=128, dtype=torch.bfloat16):
    padded_tokens = bs * sl
    padded_bins = torch.tensor([bs * sl]).type(torch.int32).cuda()
    block_rows = padded_tokens // blocking
    blocks_per_row = ffn_hidden_size // blocking

    # equivaelent to blocks_per_row in the 1 expert case
    max_column_index = blocks_per_row

    transpose_sort_end_bit = max(int(np.ceil(np.log2(max_column_index))), 1)

    offsets = torch.arange(
        0,
        block_rows * blocks_per_row + 1,
        blocks_per_row,
        dtype=torch.int32,
        device='cuda')

    column_indices = ops.topology(
            padded_bins, 
            blocking,
            block_rows,
            blocks_per_row
        )
    data = torch.empty(
        column_indices.numel(),
        blocking,
        blocking,
        dtype=dtype,
        device='cuda') 

    padded_tokens = bs * sl

    shape = (
        padded_tokens,
        ffn_hidden_size
    )

    row_indices = stk.ops.row_indices(
        shape, data, offsets, column_indices)
    
    column_indices_t, offsets_t, block_offsets_t = _sparse_transpose(
        shape, row_indices, column_indices, offsets, blocking, transpose_sort_end_bit)
    
    return stk.Matrix(shape, data, row_indices, column_indices, offsets,
                        column_indices_t, offsets_t, block_offsets_t)


def test_modules(
        hidden_size,
        ffn_hidden_size,
        grouped_mlp=False):
    init_method = partial(torch.nn.init.normal_, mean=0.0, std=0.1)
    args = Arguments(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        moe_num_experts=1,
        moe_top_k=1,
        init_method=init_method,
        memory_optimized_mlp=False,
        mlp_type='glu',
        grouped_mlp=grouped_mlp,
        fp16=False,
        bf16=True)

    glu = testing.GLU(args)
    dmoe_glu = GroupedGLU(args) if grouped_mlp else SparseGLU(args)

    dmoe_glu.cuda(torch.cuda.current_device()).to(torch.bfloat16)
    glu.cuda(torch.cuda.current_device()).to(torch.bfloat16)

    with torch.no_grad():
        glu.w1.copy_(dmoe_glu.w1.T)
        glu.v1.copy_(dmoe_glu.v1.T)
        glu.w2.copy_(dmoe_glu.w2)

    return args, glu, dmoe_glu

_DENSE_TESTS = (
    (16, 1024, 512),
    (8, 2048, 512),
)

class GLUTest(parameterized.TestCase):

    @parameterized.parameters(*_DENSE_TESTS)
    def testGLU_forward_grouped_mlp(self, bs, sl, hs):

        x = torch.randn(sl, bs, hs).to(torch.bfloat16).cuda()

        _, glu, dmoe_glu = test_modules(
            hidden_size=hs,
            ffn_hidden_size=hs * 2,
            grouped_mlp=True)

        expected_out = glu(x)
        tokens_per_expert = torch.tensor([bs * sl]).cuda()
        out = dmoe_glu(x.view(bs * sl, hs), tokens_per_expert)
        out = out.view(sl, bs, hs)

        self.assertSequenceEqual(out.shape, x.shape)
        self.assertSequenceEqual(expected_out.shape, x.shape)
        self.assertTrue(testing.allclose(out, expected_out))

    @parameterized.parameters(*_DENSE_TESTS)
    def testGLU_forward_sparse(self, bs, sl, hs):
        x = torch.randn(sl, bs, hs).to(torch.bfloat16).cuda()

        _, glu, dmoe_glu = test_modules(
            hidden_size=hs,
            ffn_hidden_size=hs * 2,
            grouped_mlp=False)

        expected_out = glu(x)

        with torch.no_grad():
            topo = _setup_topology(bs, sl, hs * 2)
        out = dmoe_glu(x.view(bs * sl, hs), topo)

        out = out.view(sl, bs, hs)

        self.assertSequenceEqual(out.shape, x.shape)
        self.assertSequenceEqual(expected_out.shape, x.shape)
        self.assertTrue(testing.allclose(out, expected_out))

if __name__ == '__main__':
    unittest.main()
