# Copyright 2024 MosaicML MegaBlocks authors
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import pytest
import stk
import torch

from megablocks.layers import dmlp_registry, testing
from megablocks.layers.arguments import Arguments

_DENSE_TESTS = (
    (16, 1024, 512),
    (8, 2048, 512),
)


def construct_dmoe_glu(
    hidden_size: int,
    ffn_hidden_size: int,
    mlp_impl: str = 'sparse',
    memory_optimized_mlp: bool = False,
):
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
        bf16=True,
    )

    glu = testing.GLU(args)
    dmoe_glu = dmlp_registry.get(args)

    dmoe_glu.cuda(torch.cuda.current_device()).to(torch.bfloat16)
    glu.cuda(torch.cuda.current_device()).to(torch.bfloat16)

    with torch.no_grad():
        glu.w1.copy_(dmoe_glu.w1.T)
        glu.v1.copy_(dmoe_glu.v1.T)
        glu.w2.copy_(dmoe_glu.w2)

    return args, glu, dmoe_glu


@pytest.mark.gpu
@pytest.mark.parametrize(('bs', 'sl', 'hs'), _DENSE_TESTS)
def test_glu_forward_grouped_mlp(bs: int, sl: int, hs: int):
    x = torch.randn(sl, bs, hs).to(torch.bfloat16).cuda()

    _, glu, dmoe_glu = construct_dmoe_glu(
        hidden_size=hs,
        ffn_hidden_size=hs * 2,
        mlp_impl='grouped',
    )

    expected_out = glu(x)
    tokens_per_expert = torch.tensor([bs * sl]).cuda()
    out = dmoe_glu(x.view(bs * sl, hs), tokens_per_expert)
    out = out.view(sl, bs, hs)

    assert out.shape == x.shape == expected_out.shape
    assert torch.allclose(out, expected_out)


@pytest.mark.gpu
@pytest.mark.parametrize(('bs', 'sl', 'hs'), _DENSE_TESTS)
def test_glu_forward_grouped_mlp_mem_opt(bs: int, sl: int, hs: int):
    x = torch.randn(sl, bs, hs).to(torch.bfloat16).cuda()

    _, glu, dmoe_glu = construct_dmoe_glu(
        hidden_size=hs,
        ffn_hidden_size=hs * 2,
        mlp_impl='grouped',
        memory_optimized_mlp=True,
    )

    expected_out = glu(x)
    tokens_per_expert = torch.tensor([bs * sl]).cuda()
    out = dmoe_glu(x.view(bs * sl, hs), tokens_per_expert)
    out = out.view(sl, bs, hs)

    assert out.shape == x.shape == expected_out.shape
    assert torch.allclose(out, expected_out)


@pytest.mark.gpu
@pytest.mark.parametrize(('bs', 'sl', 'hs'), _DENSE_TESTS)
def test_glu_forward_sparse_mlp(bs: int, sl: int, hs: int):
    x = torch.randn(sl, bs, hs).to(torch.bfloat16).cuda()

    _, glu, dmoe_glu = construct_dmoe_glu(
        hidden_size=hs,
        ffn_hidden_size=hs * 2,
        mlp_impl='sparse',
    )

    expected_out = glu(x)
    with torch.no_grad():
        topo = stk.random.mask(bs * sl, hs * 2, 0, blocking=128).cuda()
    out = dmoe_glu(x.view(bs * sl, hs), topo)
    out = out.view(sl, bs, hs)

    assert out.shape == x.shape == expected_out.shape
    assert torch.allclose(out, expected_out)
