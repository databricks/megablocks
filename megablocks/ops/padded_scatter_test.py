import unittest

from absl.testing import parameterized
from megablocks import ops
import numpy as np
import torch


_PADDED_SCATTER_TESTS = (
    (4, 2, 2),
    (1024, 1, 2),
    (1024, 1, 4),
    (1024, 1, 8),
    (1024, 1, 16),
    (1024, 1, 32),
    (1024, 1, 64),
    (1024, 1, 128),
    (1024, 1, 256),
    (1024, 1, 512),
    (1024, 1536, 2),
    (1024, 1536, 4),
    (1024, 1536, 8),
    (1024, 1536, 16),
    (1024, 1536, 32),
    (1024, 1536, 64),
    (1024, 1536, 128),
    (1024, 1536, 256),
    (1024, 1536, 512),
    (16384, 768, 2),
    (16384, 768, 4),
    (16384, 768, 8),
    (16384, 768, 16),
    (16384, 768, 32),
    (16384, 768, 64),
    (16384, 768, 128),
    (16384, 768, 256),
    (16384, 768, 512),
    (16384, 768, 1024),
    (16384, 1, 2),
    (16384, 1, 4),
    (16384, 1, 8),
    (16384, 1, 16),
    (16384, 1, 32),
    (16384, 1, 64),
    (16384, 1, 128),
    (16384, 1, 256),
    (16384, 1, 512),
)


class PaddedScatterTest(parameterized.TestCase):

    @parameterized.parameters(*_PADDED_SCATTER_TESTS)
    def testPaddedScatter(self, sl, hs, ne):
        # Create the data and indices.
        x = torch.randn((sl, hs)).cuda().half()

        # Randomly assign tokens to experts.
        top_expert = torch.randint(0, ne, (sl,)).cuda().int()
        bin_ids, indices = ops.sort(top_expert)
        tokens_per_expert = ops.histogram(top_expert, ne)
        padded_tokens_per_expert = ops.round_up(tokens_per_expert, 128)
        padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0)
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)

        # Gather the data to prepare for backwards.
        x = ops.padded_gather(x, indices, bin_ids, bins, padded_bins)

        def padded_scatter(x, indices, bin_ids, bins, padded_bins):
            x = x.cpu().numpy()
            indices = indices.cpu().numpy()
            bin_ids = bin_ids.cpu().numpy()
            bins = bins.cpu().numpy()
            padded_bins = padded_bins.cpu().numpy()

            out = np.zeros((indices.shape[0], hs))
            out_idx = 0
            for i in range(len(bins)):
                in_idx = 0 if i == 0 else padded_bins[i - 1]
                end = bins[i]
                while out_idx < end:
                    store_idx = indices[out_idx]
                    out[store_idx, :] = x[in_idx, :]
                    out_idx += 1
                    in_idx += 1
            return torch.from_numpy(out).cuda().half()

        out = ops.padded_scatter(x, indices, bin_ids, bins, padded_bins)
        expected_out = padded_scatter(x, indices, bin_ids, bins, padded_bins)
        self.assertTrue(torch.all(torch.eq(out, expected_out)))


if __name__ == '__main__':
    unittest.main()
