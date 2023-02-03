import unittest
from functools import partial

from absl.testing import parameterized
from megablocks.layers.arguments import Arguments
from megablocks.layers import dmoe
from megablocks.layers import moe
from megablocks.layers import testing
import torch


def test_modules(
        hidden_size,
        ffn_hidden_size,
        moe_num_experts=1,
        moe_capacity_factor=1,
        moe_top_k=1):
    init_method = partial(torch.nn.init.normal_, mean=0.0, std=0.1)
    args = Arguments(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        moe_num_experts=moe_num_experts,
        moe_capacity_factor=moe_capacity_factor,
        moe_top_k=moe_top_k,
        init_method=init_method)

    mlp = testing.FFN(args)
    moe_mlp = moe.MoE(args)
    dmoe_mlp = dmoe.dMoE(args)

    mlp.cuda(torch.cuda.current_device()).half()
    moe_mlp.cuda(torch.cuda.current_device()).half()
    dmoe_mlp.cuda(torch.cuda.current_device()).half()

    # Set the baseline parameters to match exactly.
    with torch.no_grad():
        ne, hs, fhs = moe_mlp.w1.size()
        w1 = dmoe_mlp.w1.view([ne, fhs, hs])
        moe_mlp.w1.copy_(torch.transpose(w1, 1, 2).contiguous())
        moe_mlp.w2.copy_(dmoe_mlp.w2.view([ne, fhs, hs]))
        moe_mlp.router_weight.copy_(dmoe_mlp.router_weight)
        if moe_num_experts == 1:
            mlp.w1.copy_(moe_mlp.w1.squeeze())
            mlp.w2.copy_(moe_mlp.w2.squeeze())
    return args, mlp, moe_mlp, dmoe_mlp

# min size: (1, 2, 128, 2, 1)
_FORWARD_TESTS = (
    (16, 1024, 512, 1, 1),
    (16, 1024, 512, 2, 1),
    (16, 1024, 512, 4, 1),
    (16, 1024, 512, 8, 1),
    (8, 2048, 512, 1, 1),
    (8, 2048, 512, 2, 1),
    (8, 2048, 512, 4, 1),
    (16, 1024, 512, 2, 2),
    (16, 1024, 512, 4, 2),
    (16, 1024, 512, 4, 4),
    (16, 1024, 512, 8, 2),
    (16, 1024, 512, 8, 4),
    (16, 1024, 512, 8, 8),
)


_DENSE_TESTS = (
    (16, 1024, 512),
    (8, 2048, 512),
)


class dMoETest(parameterized.TestCase):

    @staticmethod
    def tearDown():
        moe.clear_load_balancing_loss()

    @parameterized.parameters(*_FORWARD_TESTS)
    def testdMoE_Forward(
            self, bs, sl, hs, num_experts, top_k):
        x = torch.randn(sl, bs, hs).half().cuda()

        _, _, _, layer = test_modules(
            hidden_size=hs,
            ffn_hidden_size=hs * 2,
            moe_num_experts=num_experts,
            moe_top_k=top_k)

        out, _ = layer(x)
        self.assertSequenceEqual(out.shape, x.shape)

    @parameterized.parameters(*_FORWARD_TESTS)
    def testdMoE_ForwardBackward(
            self, bs, sl, hs, num_experts, top_k):
        x = torch.randn(sl, bs, hs).half().cuda()
        x.requires_grad_(True)

        args, _, _, layer = test_modules(
            hidden_size=hs,
            ffn_hidden_size=hs * 2,
            moe_num_experts=num_experts,
            moe_top_k=top_k)

        out, _ = layer(x)
        self.assertSequenceEqual(out.shape, x.shape)
        loss = out.sum() + moe.batched_load_balancing_loss(args)
        loss.backward()
        layer.zero_grad(set_to_none=True)
        x.grad = None
        moe.clear_load_balancing_loss()

    @parameterized.parameters(*_DENSE_TESTS)
    def testdMoE_ForwardVersusBaseline(self, bs, sl, hs):
        x = torch.randn(sl, bs, hs).half().cuda()

        _, mlp, _, dmoe_mlp = test_modules(
            hidden_size=hs,
            ffn_hidden_size=hs * 2)

        expected_out = mlp(x)
        out, _ = dmoe_mlp(x)
        self.assertSequenceEqual(out.shape, x.shape)
        self.assertSequenceEqual(expected_out.shape, x.shape)
        self.assertTrue(testing.allclose(out, expected_out))

    @parameterized.parameters(*_FORWARD_TESTS)
    def testdMoE_ForwardVersusMoE(
            self, bs, sl, hs, num_experts, top_k):
        x = torch.randn(sl, bs, hs).half().cuda()

        _, _, moe_mlp, dmoe_mlp = test_modules(
            hidden_size=hs,
            ffn_hidden_size=hs,
            moe_num_experts=num_experts,
            moe_capacity_factor=0)

        expected_out, _= moe_mlp(x)
        out, _ = dmoe_mlp(x)
        self.assertSequenceEqual(out.shape, x.shape)
        self.assertSequenceEqual(expected_out.shape, x.shape)
        self.assertTrue(testing.allclose(out, expected_out))


if __name__ == '__main__':
    unittest.main()
