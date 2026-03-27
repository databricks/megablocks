# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

import torch

from megablocks.backend import npu_kernels


def _reference_gather(x: torch.Tensor, top_k: int) -> torch.Tensor:
    # indices = arange(tokens * top_k), single bin
    tokens = x.shape[0]
    idx = torch.arange(tokens * top_k, device=x.device)
    return x[idx // top_k]


def test_npu_kernels_gather_simple_repeats_tokens():
    tokens, hidden, top_k = 8, 16, 2
    x = torch.randn(tokens, hidden)
    indices = torch.arange(tokens * top_k, dtype=torch.int64)
    bin_ids = torch.zeros(tokens * top_k, dtype=torch.int64)
    bins = torch.tensor([tokens * top_k], dtype=torch.int64)
    out = npu_kernels.gather(x, indices, bin_ids, None, bins, top_k)
    assert torch.allclose(out, _reference_gather(x, top_k))


def test_npu_kernels_scatter_inverts_gather_for_unit_weights():
    tokens, hidden, top_k = 8, 16, 2
    x = torch.randn(tokens, hidden)
    gathered = _reference_gather(x, top_k)

    indices = torch.arange(tokens * top_k, dtype=torch.int64)
    bin_ids = torch.zeros(tokens * top_k, dtype=torch.int64)
    bins = torch.tensor([tokens * top_k], dtype=torch.int64)
    scattered = npu_kernels.scatter(gathered, indices, bin_ids, None, bins, top_k)

    # Each token appears `top_k` times.
    assert torch.allclose(scattered, x * top_k)


def test_npu_kernels_padded_gather_respects_padding():
    tokens, hidden, top_k = 6, 8, 1
    x = torch.randn(tokens, hidden)

    # All entries belong to expert 0; expert 1 is empty.
    indices = torch.arange(tokens * top_k, dtype=torch.int64)
    bin_ids = torch.zeros(tokens * top_k, dtype=torch.int64)
    bins = torch.tensor([tokens, tokens], dtype=torch.int64)  # inclusive
    padded_bins = torch.tensor([tokens + 4, tokens + 4], dtype=torch.int64)  # pad expert 0 by 4 rows

    out = npu_kernels.padded_gather(x, indices, bin_ids, None, bins, padded_bins, top_k)
    assert out.shape == (tokens + 4, hidden)
    assert torch.allclose(out[:tokens], x)
    assert torch.allclose(out[tokens:], torch.zeros_like(out[tokens:]))

