import unittest

from absl.testing import parameterized
from megablocks import ops
import numpy as np
import torch


_BINNED_SCATTER_TESTS = (
    (4, 2, 2),
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
)


class BinnedScatterTest(parameterized.TestCase):

    @parameterized.parameters(*_BINNED_SCATTER_TESTS)
    def testBinnedScatter(self, sl, hs, ne):
        # NOTE: Capacity factor == 1.
        ec = sl // ne

        # Create the data and indices.
        x = torch.randn((sl, hs)).cuda().half()

        # Randomly assign tokens to experts.
        top_expert = torch.randint(0, ne, (sl,)).cuda().int()
        _, indices = ops.sort(top_expert)
        bins = ops.inclusive_cumsum(ops.histogram(top_expert, ne), 0)
        x = ops.binned_gather(x, indices, bins, ec)

        def binned_scatter(x, indices, bins):
            x = x.cpu().numpy()
            indices = indices.cpu().numpy()
            bins = bins.cpu().numpy()
            start = 0
            out = np.zeros((sl, hs))
            for i in range(ne):
                end = bins[i]
                for j in range(min(ec, end - start)):
                    index = indices[start + j]
                    out[index, :] = x[i, j, :]
                start = end
            return torch.from_numpy(out).cuda().half()

        out = ops.binned_scatter(x, indices, bins)
        expected_out = binned_scatter(x, indices, bins)
        self.assertTrue(torch.all(torch.eq(out, expected_out)))


if __name__ == '__main__':
    unittest.main()
