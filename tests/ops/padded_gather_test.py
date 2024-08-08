# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from megablocks import ops

PADDED_GATHER_TESTS = (
    (4, 2, 2, 1),
    (4, 2, 2, 2),
    (1024, 1, 4, 1),
    (1024, 1, 4, 2),
    (1024, 1, 4, 4),
    (1024, 1, 64, 1),
    (1024, 1, 64, 2),
    (1024, 1, 64, 4),
    (1024, 1, 128, 1),
    (1024, 1, 128, 2),
    (1024, 1, 128, 4),
    (1024, 1536, 4, 1),
    (1024, 1536, 4, 2),
    (1024, 1536, 4, 4),
    (1024, 1536, 64, 1),
    (1024, 1536, 64, 2),
    (1024, 1536, 64, 4),
    (1024, 1536, 128, 1),
    (1024, 1536, 128, 2),
    (1024, 1536, 128, 4),
    (16384, 768, 4, 1),
    (16384, 768, 4, 2),
    (16384, 768, 4, 4),
    (16384, 768, 64, 1),
    (16384, 768, 64, 2),
    (16384, 768, 64, 4),
    (16384, 768, 128, 1),
    (16384, 768, 128, 2),
    (16384, 768, 128, 4),
    (16384, 1, 4, 1),
    (16384, 1, 4, 2),
    (16384, 1, 4, 4),
    (16384, 1, 64, 1),
    (16384, 1, 64, 2),
    (16384, 1, 64, 4),
    (16384, 1, 128, 1),
    (16384, 1, 128, 2),
    (16384, 1, 128, 4),
)


@pytest.mark.gpu
@pytest.mark.parametrize(('sl', 'hs', 'ne', 'top_k'), PADDED_GATHER_TESTS)
def testPaddedGather(sl: int, hs: int, ne: int, top_k: int):
    # Create the data and indices.
    x = torch.randn((sl, hs)).cuda().half()

    # Randomly assign tokens to experts.
    top_expert = torch.randint(0, ne, (sl * top_k,)).cuda().int()
    bin_ids, indices = ops.sort(top_expert)
    tokens_per_expert = ops.histogram(top_expert, ne)
    padded_tokens_per_expert = ops.round_up(tokens_per_expert, 128)
    padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0)
    bins = ops.inclusive_cumsum(tokens_per_expert, 0)

    def padded_gather(
        x: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
        padded_bins: torch.Tensor,
        top_k: int,
    ):
        x = x.cpu().numpy()
        indices = indices.cpu().numpy()
        bin_ids = bin_ids.cpu().numpy()
        bins = bins.cpu().numpy()
        padded_bins = padded_bins.cpu().numpy()

        out = np.zeros((padded_bins[-1], hs))
        in_idx = 0
        for i, end in enumerate(bins):
            out_idx = 0 if i == 0 else padded_bins[i - 1]
            end = bins[i]
            while in_idx < end:
                load_idx = indices[in_idx] // top_k
                out[out_idx, :] = x[load_idx, :]
                in_idx += 1
                out_idx += 1
        return torch.from_numpy(out).cuda().half()

    out = ops.padded_gather(x, indices, bin_ids, bins, padded_bins, top_k)
    expected_out = padded_gather(x, indices, bin_ids, bins, padded_bins, top_k)
    assert torch.all(torch.eq(out, expected_out))
