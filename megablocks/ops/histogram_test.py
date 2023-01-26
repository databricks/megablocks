import unittest

from absl.testing import parameterized
from megablocks import ops
import torch


_HISTOGRAM_TESTS = (
    (1, 32, torch.int16, 128),
    (1, 1024, torch.int16, 128),
    (1, 16384, torch.int16, 128),
    (1, 32, torch.int32, 128),
    (1, 1024, torch.int32, 128),
    (1, 16384, torch.int32, 128),
    (1, 32, torch.int64, 128),
    (1, 1024, torch.int64, 128),
    (1, 16384, torch.int64, 128),
    (1, 32, torch.int16, 1024),
    (1, 1024, torch.int16, 1024),
    (1, 16384, torch.int16, 1024),
    (1, 32, torch.int32, 1024),
    (1, 1024, torch.int32, 1024),
    (1, 16384, torch.int32, 1024),
    (1, 32, torch.int64, 1024),
    (1, 1024, torch.int64, 1024),
    (1, 16384, torch.int64, 1024),
    (2, 32, torch.int16, 128),
    (2, 1024, torch.int16, 128),
    (2, 16384, torch.int16, 128),
    (2, 32, torch.int32, 128),
    (2, 1024, torch.int32, 128),
    (2, 16384, torch.int32, 128),
    (2, 32, torch.int64, 128),
    (2, 1024, torch.int64, 128),
    (2, 16384, torch.int64, 128),
    (2, 32, torch.int16, 1024),
    (2, 1024, torch.int16, 1024),
    (2, 16384, torch.int16, 1024),
    (2, 32, torch.int32, 1024),
    (2, 1024, torch.int32, 1024),
    (2, 16384, torch.int32, 1024),
    (2, 32, torch.int64, 1024),
    (2, 1024, torch.int64, 1024),
    (2, 16384, torch.int64, 1024),
    (8, 32, torch.int16, 128),
    (8, 1024, torch.int16, 128),
    (8, 16384, torch.int16, 128),
    (8, 32, torch.int32, 128),
    (8, 1024, torch.int32, 128),
    (8, 16384, torch.int32, 128),
    (8, 32, torch.int64, 128),
    (8, 1024, torch.int64, 128),
    (8, 16384, torch.int64, 128),
    (8, 32, torch.int16, 1024),
    (8, 1024, torch.int16, 1024),
    (8, 16384, torch.int16, 1024),
    (8, 32, torch.int32, 1024),
    (8, 1024, torch.int32, 1024),
    (8, 16384, torch.int32, 1024),
    (8, 32, torch.int64, 1024),
    (8, 1024, torch.int64, 1024),
    (8, 16384, torch.int64, 1024),
)


class HistogramTest(parameterized.TestCase):

    @parameterized.parameters(*_HISTOGRAM_TESTS)
    def testHistogram(self, m, n, dtype, max_val):
        x = torch.randint(0, max_val, (m, n)).cuda().to(dtype)

        out = ops.histogram(x, max_val)
        expected_out = torch.stack(
            [torch.histc(y, max_val, 0, max_val - 1)
             for y in torch.split(x, 1)]
        )
        self.assertTrue(torch.all(torch.eq(out, expected_out)))


if __name__ == '__main__':
    unittest.main()
