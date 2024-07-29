# Copyright 2024 MosaicML MegaBlocks authors
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import pytest
import torch

from megablocks import grouped_gemm_util as gg
from megablocks.layers import dmoe, moe, testing
from megablocks.layers.arguments import Arguments

# min size: (1, 2, 128, 2, 1)
_FORWARD_TESTS_DEFAULT = (
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
    (16, 1024, 128, 1, 1),
)

_FORWARD_TESTS_GROUPED_MLP = tuple([
    p + ('grouped',) for p in _FORWARD_TESTS_DEFAULT
]) if gg.grouped_gemm_is_available() else ()

_FORWARD_TESTS_SPARSE_MLP = tuple([
    p + ('sparse',) for p in _FORWARD_TESTS_DEFAULT
])

_FORWARD_TESTS = (_FORWARD_TESTS_SPARSE_MLP + _FORWARD_TESTS_GROUPED_MLP)

_DENSE_TESTS = (
    (16, 1024, 512),
    (8, 2048, 512),
)


def construct_moes(hidden_size: int,
                   ffn_hidden_size: int,
                   moe_num_experts: int = 1,
                   moe_capacity_factor: int = 1,
                   moe_top_k: int = 1,
                   mlp_impl: str = 'sparse'):
    init_method = partial(torch.nn.init.normal_, mean=0.0, std=0.1)
    args = Arguments(hidden_size=hidden_size,
                     ffn_hidden_size=ffn_hidden_size,
                     moe_num_experts=moe_num_experts,
                     moe_capacity_factor=moe_capacity_factor,
                     moe_top_k=moe_top_k,
                     init_method=init_method,
                     memory_optimized_mlp=True,
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
        moe_mlp.experts.mlp.w2.copy_(dmoe_mlp.experts.mlp.w2.view([ne, fhs,
                                                                   hs]))
        moe_mlp.router.layer.weight.copy_(dmoe_mlp.router.layer.weight)
        if moe_num_experts == 1:
            mlp.w1.copy_(moe_mlp.experts.mlp.w1.squeeze())
            mlp.w2.copy_(moe_mlp.experts.mlp.w2.squeeze())
    return args, mlp, moe_mlp, dmoe_mlp


@pytest.mark.gpu
@pytest.mark.parametrize(('bs', 'sl', 'hs', 'num_experts', 'top_k', 'mlp_impl'),
                         _FORWARD_TESTS)
def test_dmoe_forward(bs: int,
                      sl: int,
                      hs: int,
                      num_experts: int,
                      top_k: int,
                      mlp_impl: str):
    x = torch.randn(sl, bs, hs).to(torch.bfloat16).cuda()
    _, _, _, layer = construct_moes(hidden_size=hs,
                                    ffn_hidden_size=hs * 2,
                                    moe_num_experts=num_experts,
                                    moe_top_k=top_k,
                                    mlp_impl=mlp_impl)

    out, _ = layer(x)
    assert out.shape == x.shape
    moe.clear_load_balancing_loss()


@pytest.mark.gpu
@pytest.mark.parametrize(('bs', 'sl', 'hs', 'num_experts', 'top_k', 'mlp_impl'),
                         _FORWARD_TESTS)
def test_dmoe_forward_backward(bs: int,
                               sl: int,
                               hs: int,
                               num_experts: int,
                               top_k: int,
                               mlp_impl: str):
    x = torch.randn(sl, bs, hs).to(torch.bfloat16).cuda()
    x.requires_grad_(True)

    args, _, _, layer = construct_moes(hidden_size=hs,
                                       ffn_hidden_size=hs * 2,
                                       moe_num_experts=num_experts,
                                       moe_top_k=top_k,
                                       mlp_impl=mlp_impl)

    out, _ = layer(x)
    assert out.shape == x.shape
    loss = out.sum() + moe.batched_load_balancing_loss(args)
    loss.backward()
    assert x.grad is not None
    layer.zero_grad(set_to_none=True)
    x.grad = None
    moe.clear_load_balancing_loss()


@pytest.mark.gpu
@pytest.mark.parametrize(('bs', 'sl', 'hs'), _DENSE_TESTS)
def test_dmoe_forward_vs_baseline(bs: int,
                                  sl: int,
                                  hs: int,
                                  mlp_impl: str = 'sparse'):
    x = torch.randn(sl, bs, hs).to(torch.bfloat16).cuda()

    _, mlp, _, dmoe_mlp = construct_moes(hidden_size=hs,
                                         ffn_hidden_size=hs * 2,
                                         moe_num_experts=1,
                                         moe_capacity_factor=1,
                                         moe_top_k=1,
                                         mlp_impl=mlp_impl)

    expected_out = mlp(x)
    out, _ = dmoe_mlp(x)
    assert out.shape == x.shape == expected_out.shape
    assert torch.allclose(out, expected_out)


@pytest.mark.gpu
@pytest.mark.parametrize(('bs', 'sl', 'hs', 'num_experts', 'top_k', 'mlp_impl'),
                         _FORWARD_TESTS)
def test_dmoe_forward_vs_moe(bs: int,
                             sl: int,
                             hs: int,
                             num_experts: int,
                             top_k: int,
                             mlp_impl: str):
    torch.manual_seed(42)

    x = torch.randn(sl, bs, hs).to(torch.bfloat16).cuda()

    _, _, moe_mlp, dmoe_mlp = construct_moes(hidden_size=hs,
                                             ffn_hidden_size=hs,
                                             moe_num_experts=num_experts,
                                             moe_capacity_factor=0,
                                             mlp_impl=mlp_impl)

    expected_out, _ = moe_mlp(x)
    out, _ = dmoe_mlp(x)
    assert out.shape == x.shape == expected_out.shape
    assert torch.allclose(out, expected_out)
