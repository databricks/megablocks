# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import pytest
import torch

from megablocks.layers.arguments import Arguments
from megablocks.layers.moe import MoE, batched_load_balancing_loss, clear_load_balancing_loss
from megablocks.layers.router import batched_router_zloss, clear_router_zloss
from tests.layers.architectures import FFN

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


def construct_moe(
    hidden_size: int,
    ffn_hidden_size: int,
    moe_num_experts: int = 1,
    moe_capacity_factor: int = 1,
    moe_top_k: int = 1,
    moe_zloss_weight: float = 0,
):
    # All tests are skipped if triton >=3.2.0 is installed since sparse is not supported
    # TODO: Remove this once sparse is supported with triton >=3.2.0
    try:
        import triton
        if triton.__version__ >= '3.2.0':
            pytest.skip('Sparse MLP is not supported with triton >=3.2.0')
    except ImportError:
        pass

    init_method = partial(torch.nn.init.normal_, mean=0.0, std=0.1)
    args = Arguments(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        moe_num_experts=moe_num_experts,
        moe_capacity_factor=moe_capacity_factor,
        moe_top_k=moe_top_k,
        init_method=init_method,
        moe_zloss_weight=moe_zloss_weight,
    )

    mlp = FFN(args)
    moe_mlp = MoE(args)

    mlp.cuda(torch.cuda.current_device()).half()
    moe_mlp.cuda(torch.cuda.current_device()).half()

    # Set the baseline parameters to match exactly.
    if moe_num_experts == 1:
        with torch.no_grad():
            mlp.w1.copy_(moe_mlp.experts.mlp.w1.squeeze())
            mlp.w2.copy_(moe_mlp.experts.mlp.w2.squeeze())
    return args, mlp, moe_mlp


@pytest.mark.gpu
@pytest.mark.parametrize(('bs', 'sl', 'hs', 'num_experts', 'top_k'), _FORWARD_TESTS)
def test_moe_forward(bs: int, sl: int, hs: int, num_experts: int, top_k: int):
    x = torch.randn(sl, bs, hs).half().cuda()

    _, _, layer = construct_moe(
        hidden_size=hs,
        ffn_hidden_size=hs * 2,
        moe_num_experts=num_experts,
        moe_top_k=top_k,
    )

    out, _ = layer(x)
    assert out.shape == x.shape
    clear_load_balancing_loss()


@pytest.mark.gpu
@pytest.mark.parametrize(('bs', 'sl', 'hs', 'num_experts', 'top_k'), _FORWARD_TESTS)
def test_moe_forward_backward(
    bs: int,
    sl: int,
    hs: int,
    num_experts: int,
    top_k: int,
):
    x = torch.randn(sl, bs, hs).half().cuda()
    x.requires_grad_(True)

    args, _, layer = construct_moe(
        hidden_size=hs,
        ffn_hidden_size=hs * 2,
        moe_num_experts=num_experts,
        moe_top_k=top_k,
    )

    out, _ = layer(x)
    assert out.shape == x.shape

    loss = out.sum() + batched_load_balancing_loss(args)
    loss.backward()
    layer.zero_grad(set_to_none=True)
    x.grad = None
    clear_load_balancing_loss()


@pytest.mark.gpu
@pytest.mark.parametrize(('bs', 'sl', 'hs', 'num_experts', 'top_k'), _FORWARD_TESTS)
def test_moe_forward_backward_with_zloss(
    bs: int,
    sl: int,
    hs: int,
    num_experts: int,
    top_k: int,
):
    x = torch.randn(sl, bs, hs).half().cuda()
    x.requires_grad_(True)

    args, _, layer = construct_moe(
        hidden_size=hs,
        ffn_hidden_size=hs * 2,
        moe_num_experts=num_experts,
        moe_top_k=top_k,
        moe_zloss_weight=1e-3,
    )

    out, _ = layer(x)
    assert out.shape == x.shape

    loss = out.sum() + batched_load_balancing_loss(args)
    loss.backward()
    layer.zero_grad(set_to_none=True)
    x.grad = None
    clear_load_balancing_loss()
    clear_router_zloss()


@pytest.mark.gpu
@pytest.mark.parametrize(('bs', 'sl', 'hs'), _DENSE_TESTS)
def test_moe_forward_vs_dense(bs: int, sl: int, hs: int):
    x = torch.randn(sl, bs, hs).half().cuda()

    _, mlp, moe_mlp = construct_moe(hidden_size=hs, ffn_hidden_size=hs * 2)

    expected_out = mlp(x)
    out, _ = moe_mlp(x)
    assert out.shape == x.shape == expected_out.shape
    assert torch.allclose(out, expected_out)
    clear_load_balancing_loss()


@pytest.mark.gpu
@pytest.mark.parametrize(('bs', 'sl', 'hs'), _DENSE_TESTS)
def test_moe_forward_backward_vs_dense(bs: int, sl: int, hs: int):
    x = torch.randn(sl, bs, hs).half().cuda()
    x.requires_grad_(True)

    _, mlp, moe_mlp = construct_moe(hidden_size=hs, ffn_hidden_size=hs * 2)

    out, _ = moe_mlp(x)
    loss = out.sum()
    loss.backward()
    w1_grad = moe_mlp.experts.mlp.w1.grad.detach().squeeze()
    w2_grad = moe_mlp.experts.mlp.w2.grad.detach().squeeze()
    moe_mlp.zero_grad(set_to_none=True)
    x.grad = None
    clear_load_balancing_loss()

    expected_out = mlp(x)
    expected_loss = expected_out.sum()
    expected_loss.backward()
    expected_w1_grad = mlp.w1.grad.detach()
    expected_w2_grad = mlp.w2.grad.detach()
    mlp.zero_grad(set_to_none=True)
    x.grad = None

    # Verify the gradients match.
    assert w1_grad.shape == expected_w1_grad.shape
    assert w2_grad.shape == expected_w2_grad.shape
    assert torch.allclose(w1_grad, expected_w1_grad)
    assert torch.allclose(w2_grad, expected_w2_grad)
    clear_load_balancing_loss()
