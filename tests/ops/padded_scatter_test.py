# Copyright 2024 MosaicML MegaBlocks authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from megablocks import ops

PADDED_SCATTER_TESTS = [
    (4, 2, 2, 2),
    (4, 2, 2, 1),
    (4, 2, 2, 1),
    (4, 2, 2, 1),
    (4, 2, 2, 2),
    (4, 2, 2, 2),
    (1024, 1, 4, 1),
    (1024, 1, 4, 2),
    (1024, 1, 4, 4),
    (1024, 1, 4, 1),
    (1024, 1, 4, 2),
    (1024, 1, 4, 4),
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
    (1024, 1536, 4, 4),
    (1024, 1536, 4, 4),
    (1024, 1536, 64, 1),
    (1024, 1536, 64, 2),
    (1024, 1536, 64, 4),
    (1024, 1536, 128, 1),
    (1024, 1536, 128, 2),
    (1024, 1536, 128, 4),
    (1024, 1536, 128, 1),
    (1024, 1536, 128, 1),
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
    (16384, 1, 128, 2),
    (16384, 1, 128, 2),
]


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


@pytest.mark.gpu
@pytest.mark.parametrize((
    'sl',
    'hs',
    'ne',
    'top_k',
), PADDED_SCATTER_TESTS)
def testPaddedScatter(sl: int, hs: int, ne: int, top_k: int):
    # Create the data and indices.
    x = torch.randn((sl, hs), requires_grad=True).cuda().half()

    # Randomly assign tokens to experts.
    top_expert = torch.randint(0, ne, (sl * top_k,)).cuda().int()
    bin_ids, indices = ops.sort(top_expert)
    tokens_per_expert = ops.histogram(top_expert, ne)
    padded_tokens_per_expert = ops.round_up(tokens_per_expert, 128)
    padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0)
    bins = ops.inclusive_cumsum(tokens_per_expert, 0)

    # Sample weights for the scatter reduce.
    weights = torch.rand((sl * top_k,), requires_grad=True).cuda().half()

    # Gather the data to prepare for backwards.
    x = ops.padded_gather(x, indices, bin_ids, bins, padded_bins, top_k)

    def padded_scatter(
        x: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        weights: torch.Tensor,
        bins: torch.Tensor,
        padded_bins: torch.Tensor,
        top_k: int,
    ):
        x = x.detach().cpu().numpy()
        indices: np.ndarray = _to_numpy(indices)
        bin_ids: np.ndarray = _to_numpy(bin_ids)
        weights: np.ndarray = _to_numpy(weights)
        bins: np.ndarray = _to_numpy(bins)
        padded_bins: np.ndarray = _to_numpy(padded_bins)

        out = np.zeros((indices.shape[0] // top_k, hs))
        out_idx = 0
        for i in range(len(bins)):
            in_idx = 0 if i == 0 else padded_bins[i - 1]
            end = bins[i]
            while out_idx < end:
                store_idx = indices[out_idx]
                scale = weights[store_idx]
                store_idx //= top_k

                out[store_idx, :] += scale * x[in_idx, :]
                out_idx += 1
                in_idx += 1
        return torch.from_numpy(out).cuda().half()

    out = ops.padded_scatter(
        x,
        indices,
        bin_ids,
        weights,
        bins,
        padded_bins,
        top_k,
    )
    expected_out = padded_scatter(
        x,
        indices,
        bin_ids,
        weights,
        bins,
        padded_bins,
        top_k,
    )

    out.backward(torch.randn_like(out))  # sanity check backward pass

    # NOTE: We need to check approximate equality because the scatter reduce uses atomics.
    # np.testing.assert_allclose returns `None` if no error and raises an AssertionError if an error exists
    assert np.testing.assert_allclose(
        _to_numpy(out),
        _to_numpy(expected_out),
        rtol=5e-3,
    ) is None
