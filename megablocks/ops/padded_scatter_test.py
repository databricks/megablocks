import unittest

from absl.testing import parameterized
from megablocks import ops
import numpy as np
import torch


_PADDED_SCATTER_TESTS = (
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


class PaddedScatterTest(parameterized.TestCase):

    @parameterized.parameters(*_PADDED_SCATTER_TESTS)
    def testPaddedScatter(self, sl, hs, ne, top_k):
        # Create the data and indices.
        x = torch.randn((sl, hs)).cuda().half()

        # Randomly assign tokens to experts.
        top_expert = torch.randint(0, ne, (sl * top_k,)).cuda().int()
        bin_ids, indices = ops.sort(top_expert)
        tokens_per_expert = ops.histogram(top_expert, ne)
        padded_tokens_per_expert = ops.round_up(tokens_per_expert, 128)
        padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0)
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)

        # Sample weights for the scatter reduce.
        weights = torch.rand((sl * top_k,)).cuda().half()

        # Gather the data to prepare for backwards.
        x = ops.padded_gather(x, indices, bin_ids, bins, padded_bins, top_k)

        def padded_scatter(x, indices, bin_ids, weights, bins, padded_bins, top_k):
            x = x.cpu().numpy()
            indices = indices.cpu().numpy()
            bin_ids = bin_ids.cpu().numpy()
            weights = weights.cpu().numpy()
            bins = bins.cpu().numpy()
            padded_bins = padded_bins.cpu().numpy()

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
            x, indices, bin_ids, weights, bins, padded_bins, top_k)
        expected_out = padded_scatter(
            x, indices, bin_ids, weights, bins, padded_bins, top_k)

        # NOTE: We need to check approximate equality because the
        # scatter reduce uses atomics.
        np.testing.assert_allclose(
            out.cpu(), expected_out.cpu(), rtol=5e-3)


if __name__ == '__main__':
    unittest.main()
