import unittest
from functools import partial

from absl.testing import parameterized
from megablocks import grouped_gemm_util as gg
from megablocks import turbo_util as turbo
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
        moe_top_k=1,
        num_input_bits=-1,
        num_remat_bits=-1,
        mlp_impl='sparse'):
    init_method = partial(torch.nn.init.normal_, mean=0.0, std=0.1)
    args = Arguments(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        moe_num_experts=moe_num_experts,
        moe_capacity_factor=moe_capacity_factor,
        moe_top_k=moe_top_k,
        init_method=init_method,
        memory_optimized_mlp=True,
        quantize_inputs_num_bits=num_input_bits,
        quantize_rematerialize_num_bits=num_remat_bits,
        mlp_type='mlp',
        mlp_impl=mlp_impl,
        fp16=False,
        bf16=True)

    mlp = testing.FFN(args)
    moe_mlp = moe.MoE(args)
    dmoe_mlp = dmoe.dMoE(args)

    mlp.cuda(torch.cuda.current_device()).to(torch.bfloat16)
    moe_mlp.cuda(torch.cuda.current_device()).to(torch.bfloat16)
    dmoe_mlp.cuda(torch.cuda.current_device()).to(torch.bfloat16)

    # Set the baseline parameters to match exactly.
    with torch.no_grad():
        ne, hs, fhs = moe_mlp.experts.mlp.w1.size()
        w1 = dmoe_mlp.experts.mlp.w1.view([ne, fhs, hs])
        moe_mlp.experts.mlp.w1.copy_(torch.transpose(w1, 1, 2).contiguous())
        moe_mlp.experts.mlp.w2.copy_(dmoe_mlp.experts.mlp.w2.view([ne, fhs, hs]))
        moe_mlp.router.layer.weight.copy_(dmoe_mlp.router.layer.weight)
        if moe_num_experts == 1:
            mlp.w1.copy_(moe_mlp.experts.mlp.w1.squeeze())
            mlp.w2.copy_(moe_mlp.experts.mlp.w2.squeeze())
    return args, mlp, moe_mlp, dmoe_mlp

# min size: (1, 2, 128, 2, 1)
_FORWARD_TESTS_NO_QUANTIZE = (
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

_FORWARD_TESTS_GROUPED_MLP = tuple([
    p + (-1, -1, 'grouped') for p in _FORWARD_TESTS_NO_QUANTIZE
]) if gg.grouped_gemm_is_available() else ()

# quantization tests; assorted small sizes, systematic bitwidths
_FORWARD_TESTS_QUANTIZE_HIDDEN = (
    (1, 2, 128, 2, 2, -1, -1),
    (1, 8, 128, 2, 2, -1, 4),
    (2, 8, 128, 2, 1, -1, 8),
) if turbo.turbo_is_available() else ()

_FORWARD_TESTS_QUANTIZE_INPUT = (
    (1, 2, 128, 2, 1, 4, -1),
    (2, 8, 128, 4, 1, 8, -1),
) if turbo.turbo_is_available() else ()

_FORWARD_TESTS_QUANTIZE_BOTH = (
    (2, 2, 128, 2, 2, 4, 4),
    (1, 8, 128, 4, 2, 4, 8),
    (1, 2, 128, 4, 2, 8, 4),
    (2, 2, 128, 4, 2, 8, 8),
) if turbo.turbo_is_available() else ()

_FORWARD_TESTS = (_FORWARD_TESTS_NO_QUANTIZE +
                  _FORWARD_TESTS_QUANTIZE_HIDDEN +
                  _FORWARD_TESTS_QUANTIZE_INPUT +
                  _FORWARD_TESTS_QUANTIZE_BOTH +
                  _FORWARD_TESTS_GROUPED_MLP)

_FORWARD_TESTS_WITH_HIDDEN_QUANTIZE = (
    _FORWARD_TESTS_NO_QUANTIZE +
    _FORWARD_TESTS_QUANTIZE_HIDDEN +
    _FORWARD_TESTS_GROUPED_MLP)


_DENSE_TESTS = (
    (16, 1024, 512),
    (8, 2048, 512),
)


class dMoETest(parameterized.TestCase):

    @staticmethod
    def tearDown():
        moe.clear_load_balancing_loss()

    @parameterized.parameters(*_FORWARD_TESTS)
    def testdMoE_Forward(self, bs, sl, hs, num_experts, top_k,
                         num_input_bits=-1, num_remat_bits=-1,
                         mlp_impl='sparse'):
        x = torch.randn(sl, bs, hs).to(torch.bfloat16).cuda()

        _, _, _, layer = test_modules(
            hidden_size=hs,
            ffn_hidden_size=hs * 2,
            moe_num_experts=num_experts,
            moe_top_k=top_k,
            num_input_bits=num_input_bits,
            num_remat_bits=num_remat_bits,
            mlp_impl=mlp_impl)

        out, _ = layer(x)
        self.assertSequenceEqual(out.shape, x.shape)

    @parameterized.parameters(*_FORWARD_TESTS)
    def testdMoE_ForwardBackward(
            self, bs, sl, hs, num_experts, top_k,
            num_input_bits=-1, num_remat_bits=-1,
            mlp_impl='sparse'):
        x = torch.randn(sl, bs, hs).to(torch.bfloat16).cuda()
        x.requires_grad_(True)

        args, _, _, layer = test_modules(
            hidden_size=hs,
            ffn_hidden_size=hs * 2,
            moe_num_experts=num_experts,
            moe_top_k=top_k,
            num_input_bits=num_input_bits,
            num_remat_bits=num_remat_bits,
            mlp_impl=mlp_impl)

        out, _ = layer(x)
        self.assertSequenceEqual(out.shape, x.shape)
        loss = out.sum() + moe.batched_load_balancing_loss(args)
        loss.backward()
        self.assertTrue(x.grad is not None)
        layer.zero_grad(set_to_none=True)
        x.grad = None
        moe.clear_load_balancing_loss()

    @parameterized.parameters(*_DENSE_TESTS)
    def testdMoE_ForwardVersusBaseline(self, bs, sl, hs):
        x = torch.randn(sl, bs, hs).to(torch.bfloat16).cuda()

        _, mlp, _, dmoe_mlp = test_modules(
            hidden_size=hs,
            ffn_hidden_size=hs * 2)

        expected_out = mlp(x)
        out, _ = dmoe_mlp(x)
        self.assertSequenceEqual(out.shape, x.shape)
        self.assertSequenceEqual(expected_out.shape, x.shape)
        self.assertTrue(testing.allclose(out, expected_out))

    # we don't run the input quantization cases just to avoid redundancy,
    # since input quantization doesn't affect any of these asserts
    @parameterized.parameters(*_FORWARD_TESTS_WITH_HIDDEN_QUANTIZE)
    def testdMoE_ForwardVersusMoE(
            self, bs, sl, hs, num_experts, top_k,
            num_input_bits=-1, num_remat_bits=-1,
            mlp_impl='sparse'):
        torch.manual_seed(42)

        x = torch.randn(sl, bs, hs).to(torch.bfloat16).cuda()

        _, _, moe_mlp, dmoe_mlp = test_modules(
            hidden_size=hs,
            ffn_hidden_size=hs,
            moe_num_experts=num_experts,
            moe_capacity_factor=0,
            mlp_impl=mlp_impl)

        expected_out, _= moe_mlp(x)
        out, _ = dmoe_mlp(x)
        self.assertSequenceEqual(out.shape, x.shape)
        self.assertSequenceEqual(expected_out.shape, x.shape)
        self.assertTrue(testing.allclose(out, expected_out))


if __name__ == '__main__':
    unittest.main()
