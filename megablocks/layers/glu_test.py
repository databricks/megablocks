import unittest
from functools import partial

from absl.testing import parameterized
from megablocks.layers.arguments import Arguments
from megablocks.layers.glu import SparseGLU, GroupedGLU
from megablocks.layers import dmlp_registry
from megablocks.layers import testing

import torch
import stk
import numpy as np

def test_modules(
        hidden_size,
        ffn_hidden_size,
        mlp_impl='sparse',
        memory_optimized_mlp=False):
    init_method = partial(torch.nn.init.normal_, mean=0.0, std=0.1)
    args = Arguments(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        moe_num_experts=1,
        moe_top_k=1,
        init_method=init_method,
        memory_optimized_mlp=memory_optimized_mlp,
        mlp_type='glu',
        mlp_impl=mlp_impl,
        fp16=False,
        bf16=True)

    glu = testing.GLU(args)
    dmoe_glu = dmlp_registry.get(args)

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
    def testGLU_ForwardGroupedMLP(self, bs, sl, hs):
        x = torch.randn(sl, bs, hs).to(torch.bfloat16).cuda()

        _, glu, dmoe_glu = test_modules(
            hidden_size=hs,
            ffn_hidden_size=hs * 2,
            mlp_impl='grouped')

        expected_out = glu(x)
        tokens_per_expert = torch.tensor([bs * sl]).cuda()
        out = dmoe_glu(x.view(bs * sl, hs), tokens_per_expert)
        out = out.view(sl, bs, hs)

        self.assertSequenceEqual(out.shape, x.shape)
        self.assertSequenceEqual(expected_out.shape, x.shape)
        self.assertTrue(testing.allclose(out, expected_out))

    @parameterized.parameters(*_DENSE_TESTS)
    def testGLU_ForwardGroupedMLP_MemOpt(self, bs, sl, hs):
        x = torch.randn(sl, bs, hs).to(torch.bfloat16).cuda()

        _, glu, dmoe_glu = test_modules(
            hidden_size=hs,
            ffn_hidden_size=hs * 2,
            mlp_impl='grouped',
            memory_optimized_mlp=True)

        expected_out = glu(x)
        tokens_per_expert = torch.tensor([bs * sl]).cuda()
        out = dmoe_glu(x.view(bs * sl, hs), tokens_per_expert)
        out = out.view(sl, bs, hs)

        self.assertSequenceEqual(out.shape, x.shape)
        self.assertSequenceEqual(expected_out.shape, x.shape)
        self.assertTrue(testing.allclose(out, expected_out))

    @parameterized.parameters(*_DENSE_TESTS)
    def testGLU_ForwardSparseMLP(self, bs, sl, hs):
        x = torch.randn(sl, bs, hs).to(torch.bfloat16).cuda()

        _, glu, dmoe_glu = test_modules(
            hidden_size=hs,
            ffn_hidden_size=hs * 2,
            mlp_impl='sparse')

        expected_out = glu(x)
        with torch.no_grad():
            topo = stk.random.mask(bs * sl, hs * 2, 0, blocking=128).cuda()
        out = dmoe_glu(x.view(bs * sl, hs), topo)
        out = out.view(sl, bs, hs)

        self.assertSequenceEqual(out.shape, x.shape)
        self.assertSequenceEqual(expected_out.shape, x.shape)
        self.assertTrue(testing.allclose(out, expected_out))

if __name__ == '__main__':
    unittest.main()
