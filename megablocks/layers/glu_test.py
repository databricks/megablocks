import unittest
from functools import partial

from absl.testing import parameterized
from megablocks.layers.arguments import Arguments
from megablocks.layers import dmoe
from megablocks.layers import testing
import torch


def test_modules(
        hidden_size,
        ffn_hidden_size,
        moe_num_experts=1,
        moe_capacity_factor=1,
        moe_top_k=1,
        use_grouped_gemm=False):
    init_method = partial(torch.nn.init.normal_, mean=0.0, std=0.1)
    args = Arguments(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        moe_num_experts=moe_num_experts,
        moe_capacity_factor=moe_capacity_factor,
        moe_top_k=moe_top_k,
        init_method=init_method,
        memory_optimized_mlp=False,
        mlp_type='glu',
        use_grouped_gemm=use_grouped_gemm,
        fp16=False,
        bf16=True)

    glu = testing.GLU(args)
    dmoe_glu = dmoe.dMoE(args)

    glu.cuda(torch.cuda.current_device()).to(torch.bfloat16)
    dmoe_glu.cuda(torch.cuda.current_device()).to(torch.bfloat16)

    with torch.no_grad():
        if moe_num_experts == 1:
            glu.w1.copy_(dmoe_glu.experts.mlp.w1.T)
            glu.v1.copy_(dmoe_glu.experts.mlp.v1.T)
            glu.w2.copy_(dmoe_glu.experts.mlp.w2)
    return args, glu, dmoe_glu

_DENSE_TESTS = (
    (16, 1024, 512),
    (8, 2048, 512),
)

_DENSE_TESTS_GROUPED = tuple([
    p + (True, ) for p in _DENSE_TESTS
])
_ALL_DENSE_TESTS = (_DENSE_TESTS + _DENSE_TESTS_GROUPED)


class GLUTest(parameterized.TestCase):

    @parameterized.parameters(*_ALL_DENSE_TESTS)
    def testdMoE_ForwardVersusBaseline(self, bs, sl, hs, use_grouped_gemm=False):
        x = torch.randn(sl, bs, hs).to(torch.bfloat16).cuda()

        _, glu, dmoe_glu = test_modules(
            hidden_size=hs,
            ffn_hidden_size=hs * 2,
            use_grouped_gemm=use_grouped_gemm)

        expected_out = glu(x)
        out, _ = dmoe_glu(x)
        self.assertSequenceEqual(out.shape, x.shape)
        self.assertSequenceEqual(expected_out.shape, x.shape)
        self.assertTrue(testing.allclose(out, expected_out))

if __name__ == '__main__':
    unittest.main()
