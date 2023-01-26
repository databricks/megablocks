import unittest

from absl.testing import parameterized
from megablocks import ops
import torch


_CUMSUM_TESTS = (
    (1, 32),
    (2, 32),
    (2, 1024),
    (4, 1024),
    (8, 1024),
    (16, 1024),
    (32, 1024),
    (64, 1024),
    (128, 1024),
    (2, 16384),
    (4, 16384),
    (8, 16384),
    (16, 16384),
    (32, 16384),
    (64, 16384),
    (128, 16384),
)


class CumsumTest(parameterized.TestCase):

    @parameterized.parameters(*_CUMSUM_TESTS)
    def testExclusiveCumsum(self, n, m):
        x = torch.randint(0, 2, (n, m)).long().cuda()
        out = ops.exclusive_cumsum(x, 1) * x
        expected_out = (torch.cumsum(x, dim=1) - 1) * x
        self.assertTrue(torch.all(torch.eq(out, expected_out)))

    @parameterized.parameters(*_CUMSUM_TESTS)
    def testInclusiveCumsum(self, n, m):
        x = torch.randint(0, 2, (n, m)).long().cuda()
        out = ops.inclusive_cumsum(x, 1)
        expected_out = torch.cumsum(x, dim=1)
        self.assertTrue(torch.all(torch.eq(out, expected_out)))


if __name__ == '__main__':
    unittest.main()
