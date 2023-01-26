import unittest

from absl.testing import parameterized
from megablocks import ops
import megablocks_ops as backend
import numpy as np
import torch


_REPLICATE_TESTS = (
    (8, 1, 1),
    (8, 2, 1),
    (8, 4, 1),
    (8, 8, 1),
    (8, 2, 2),
    (8, 4, 2),
    (8, 8, 2),
    (8, 2, 4),
    (8, 4, 4),
    (8, 8, 4),
    (8, 2, 8),
    (8, 4, 8),
    (8, 8, 8),
    (16384, 2, 1),
    (16384, 4, 1),
    (16384, 8, 1),
    (16384, 16, 1),
    (16384, 32, 1),
    (16384, 64, 1),
    (16384, 128, 1),
    (16384, 2, 2),
    (16384, 4, 2),
    (16384, 8, 2),
    (16384, 16, 2),
    (16384, 32, 2),
    (16384, 64, 2),
    (16384, 128, 2),
    (16384, 2, 4),
    (16384, 4, 4),
    (16384, 8, 4),
    (16384, 16, 4),
    (16384, 32, 4),
    (16384, 64, 4),
    (16384, 128, 4),
    (16384, 2, 8),
    (16384, 4, 8),
    (16384, 8, 8),
    (16384, 16, 8),
    (16384, 32, 8),
    (16384, 64, 8),
    (16384, 128, 8),
)


def promote_scalar(x):
    return x.view(1) if not len(x.size()) else x


class ReplicateTest(parameterized.TestCase):

    @parameterized.parameters(*_REPLICATE_TESTS)
    def testReplicate(self, tokens, num_centers, top_k):
        tokens_to_centers = torch.randint(0, num_centers, (tokens,)).cuda().int()
        tokens_per_center = ops.histogram(tokens_to_centers, num_centers)
        bins = ops.inclusive_cumsum(tokens_per_center, 0)
        bins = promote_scalar(bins)
        center_weights = torch.randn(top_k, num_centers).cuda().half()

        def replicate(x, bins, num_outputs):
            x = x.cpu().numpy()
            bins = bins.cpu().numpy()
            out = np.zeros((x.shape[0], num_outputs))
            for batch_idx in range(x.shape[0]):
                start = 0
                for i in range(len(bins)):
                    end = bins[i]
                    value = x[batch_idx, i]
                    while start < end:
                        out[batch_idx, start] = value
                        start += 1
            return torch.from_numpy(out).cuda().half()

        out = ops.replicate(center_weights, bins, tokens)
        expected_out = replicate(center_weights, bins, tokens)
        self.assertTrue(torch.all(torch.eq(out, expected_out)))

    @parameterized.parameters(*_REPLICATE_TESTS)
    def testReplicate_Backward(self, tokens, num_centers, top_k):
        tokens_to_centers = torch.randint(0, num_centers, (tokens,)).cuda().int()
        tokens_per_center = ops.histogram(tokens_to_centers, num_centers)
        bins = ops.inclusive_cumsum(tokens_per_center, 0)
        bins = promote_scalar(bins)
        center_weights = torch.randn(top_k, num_centers).cuda().half()

        grad = ops.replicate(center_weights, bins, tokens)

        out = torch.empty_like(center_weights)
        backend.replicate_backward(grad, bins, out)
        expected_out = center_weights * tokens_per_center.view([1, num_centers])

        # NOTE: This floating-point reduction could be a problem for
        # training stability and accuracy.
        self.assertTrue(torch.allclose(out, expected_out, rtol=1e-2))


if __name__ == '__main__':
    unittest.main()
