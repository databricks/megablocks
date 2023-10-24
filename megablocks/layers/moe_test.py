import unittest
from functools import partial

from absl.testing import parameterized
from megablocks.layers.arguments import Arguments
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

    mlp.cuda(torch.cuda.current_device()).half()
    moe_mlp.cuda(torch.cuda.current_device()).half()

    # Set the baseline parameters to match exactly.
    if moe_num_experts == 1:
        with torch.no_grad():
            mlp.w1.copy_(moe_mlp.experts.mlp.w1.squeeze())
            mlp.w2.copy_(moe_mlp.experts.mlp.w2.squeeze())
    return args, mlp, moe_mlp


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


class MoETest(parameterized.TestCase):

    @staticmethod
    def tearDown():
        moe.clear_load_balancing_loss()

    @parameterized.parameters(*_FORWARD_TESTS)
    def testMoE_Forward(
            self, bs, sl, hs, num_experts, top_k):
        x = torch.randn(sl, bs, hs).half().cuda()

        _, _, layer = test_modules(
            hidden_size=hs,
            ffn_hidden_size=hs * 2,
            moe_num_experts=num_experts,
            moe_top_k=top_k)

        out, _ = layer(x)
        self.assertSequenceEqual(out.shape, x.shape)

    @parameterized.parameters(*_FORWARD_TESTS)
    def testMoE_ForwardBackward(
            self, bs, sl, hs, num_experts, top_k):
        x = torch.randn(sl, bs, hs).half().cuda()
        x.requires_grad_(True)

        args, _, layer = test_modules(
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
    def testMoE_ForwardVersusDense(self, bs, sl, hs):
        x = torch.randn(sl, bs, hs).half().cuda()

        _, mlp, moe_mlp = test_modules(
            hidden_size=hs,
            ffn_hidden_size=hs * 2)

        expected_out = mlp(x)
        out, _ = moe_mlp(x)
        self.assertSequenceEqual(out.shape, x.shape)
        self.assertSequenceEqual(expected_out.shape, x.shape)
        self.assertTrue(testing.allclose(out, expected_out))

    @parameterized.parameters(*_DENSE_TESTS)
    def testMoE_ForwardBackwardVersusDense(self, bs, sl, hs):
        x = torch.randn(sl, bs, hs).half().cuda()
        x.requires_grad_(True)

        _, mlp, moe_mlp = test_modules(
            hidden_size=hs,
            ffn_hidden_size=hs * 2)

        out, _ = moe_mlp(x)
        loss = out.sum()
        loss.backward()
        w1_grad = moe_mlp.experts.mlp.w1.grad.detach().squeeze()
        w2_grad = moe_mlp.experts.mlp.w2.grad.detach().squeeze()
        moe_mlp.zero_grad(set_to_none=True)
        x.grad = None
        moe.clear_load_balancing_loss()

        expected_out = mlp(x)
        expected_loss = expected_out.sum()
        expected_loss.backward()
        expected_w1_grad = mlp.w1.grad.detach()
        expected_w2_grad = mlp.w2.grad.detach()
        mlp.zero_grad(set_to_none=True)
        x.grad = None

        # Verify the gradients match.
        self.assertSequenceEqual(w1_grad.shape, expected_w1_grad.shape)
        self.assertTrue(testing.allclose(w1_grad, expected_w1_grad))
        self.assertSequenceEqual(w2_grad.shape, expected_w2_grad.shape)
        self.assertTrue(testing.allclose(w2_grad, expected_w2_grad))


if __name__ == '__main__':
    unittest.main()
