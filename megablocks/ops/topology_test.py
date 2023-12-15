import unittest

from absl.testing import parameterized
from megablocks import ops
import math
import numpy as np
import torch


_TOPOLOGY_TESTS = (
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
    (8, 14336, 8),
)


class TopologyTest(parameterized.TestCase):

    @parameterized.parameters(*_TOPOLOGY_TESTS)
    def testTopology(self, sl, hs, ne):
        # Create the data and indices.
        blocking = 128
        assert hs % blocking == 0

        # Randomly assign tokens to experts.
        top_expert = torch.randint(0, ne, (sl,)).cuda().int()
        tokens_per_expert = ops.histogram(top_expert, ne)
        padded_tokens_per_expert = ops.round_up(tokens_per_expert, blocking)
        padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0)

        # Dimensions for the output indices.
        output_block_rows = int(padded_bins[-1]) // blocking
        output_block_columns = hs // blocking

        def topology(padded_bins, blocking, rows, columns):
            padded_bins = padded_bins.cpu().numpy()

            out = np.zeros([rows * columns])
            start = 0
            for i in range(padded_bins.shape[0]):
                end = padded_bins[i] // blocking
                while start < end:
                    for j in range(columns):
                        out[start * columns + j] = j + i * columns
                    start += 1
            return torch.from_numpy(out).cuda().short()

        out = ops.topology(padded_bins,
                           blocking,
                           output_block_rows,
                           output_block_columns)
        expected_out = topology(padded_bins,
                                blocking,
                                output_block_rows,
                                output_block_columns)
        self.assertTrue(torch.all(torch.eq(out, expected_out)))


if __name__ == '__main__':
    unittest.main()
