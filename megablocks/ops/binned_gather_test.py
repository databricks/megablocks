import unittest

from absl.testing import parameterized
from megablocks import ops
import numpy as np
import torch


_BINNED_GATHER_TESTS = (
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


class BinnedGatherTest(parameterized.TestCase):

    @parameterized.parameters(*_BINNED_GATHER_TESTS)
    def testBinnedGather(self, sl, hs, ne, top_k):
        # NOTE: Capacity factor == 1.
        ec = (sl * top_k) // ne

        # Create the data and indices.
        x = torch.randn((sl, hs)).cuda().half()

        # Randomly assign tokens to experts.
        top_expert = torch.randint(0, ne, (sl * top_k,)).cuda().int()
        _, indices = ops.sort(top_expert)
        bins = ops.inclusive_cumsum(ops.histogram(top_expert, ne), 0)

        def binned_gather(x, indices, bins, ec, top_k):
            x = x.cpu().numpy()
            indices = indices.cpu().numpy()
            bins = bins.cpu().numpy()
            start = 0
            out = np.zeros((ne, ec, hs))
            for i in range(ne):
                end = bins[i]
                for j in range(min(ec, end - start)):
                    index = indices[start + j] // top_k
                    out[i, j, :] = x[index, :]
                start = end
            return torch.from_numpy(out).cuda().half()

        out = ops.binned_gather(x, indices, bins, ec, top_k)
        expected_out = binned_gather(x, indices, bins, ec, top_k)
        self.assertTrue(torch.all(torch.eq(out, expected_out)))


if __name__ == '__main__':
    unittest.main()
