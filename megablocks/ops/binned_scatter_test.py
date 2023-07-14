import unittest

from absl.testing import parameterized
from megablocks import ops
import numpy as np
import torch


_BINNED_SCATTER_TESTS = (
    (4, 2, 2, 1),
    (4, 2, 2, 2),
    (4, 2, 2, 4),
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
)


class BinnedScatterTest(parameterized.TestCase):

    @parameterized.parameters(*_BINNED_SCATTER_TESTS)
    def testBinnedScatter(self, sl, hs, ne, top_k):
        # NOTE: Capacity factor == 1.
        ec = (sl * top_k) // ne

        # Create the data and indices.
        x = torch.randn((sl, hs)).cuda().half()

        # Randomly assign tokens to experts.
        top_expert = torch.randint(0, ne, (sl * top_k,)).cuda().int()
        _, indices = ops.sort(top_expert)
        bins = ops.inclusive_cumsum(ops.histogram(top_expert, ne), 0)

        # Sample weights for the scatter reduce.
        weights = torch.rand((sl * top_k,)).cuda().half()

        x = ops.binned_gather(x, indices, bins, ec, top_k)

        def binned_scatter(x, indices, weights, bins, top_k):
            x = x.cpu().numpy()
            indices = indices.cpu().numpy()
            weights = weights.cpu().numpy()
            bins = bins.cpu().numpy()
            start = 0
            out = np.zeros((sl, hs))
            for i in range(ne):
                end = bins[i]
                for j in range(min(ec, end - start)):
                    index = indices[start + j]
                    scale = weights[index]
                    index //= top_k

                    out[index, :] += scale * x[i, j, :]
                start = end
            return torch.from_numpy(out).cuda().half()
        out = ops.binned_scatter(x, indices, weights, bins, top_k)
        expected_out = binned_scatter(x, indices, weights, bins, top_k)

        # NOTE: We need to check approximate equality because the
        # scatter reduce uses atomics.
        np.testing.assert_allclose(
            out.cpu(), expected_out.cpu(), rtol=5e-3)


if __name__ == '__main__':
    unittest.main()
