import unittest

from absl.testing import parameterized
from megablocks import ops
import numpy as np
import torch


_SORT_TESTS = (
    (32, torch.int16, None),
    (1024, torch.int16, None),
    (16384, torch.int16, None),
    (32, torch.int32, None),
    (1024, torch.int32, None),
    (16384, torch.int32, None),
    (32, torch.int64, None),
    (1024, torch.int64, None),
    (16384, torch.int64, None),
    (32, torch.int16, 128),
    (1024, torch.int16, 128),
    (16384, torch.int16, 128),
    (32, torch.int32, 128),
    (1024, torch.int32, 128),
    (16384, torch.int32, 128),
    (32, torch.int64, 128),
    (1024, torch.int64, 128),
    (16384, torch.int64, 128),
)


def numpy_dtype(dtype):
    types = {
        torch.int16: np.int16,
        torch.int32: np.int32,
        torch.int64: np.int64
    }
    return types[dtype]


class SortTest(parameterized.TestCase):

    @parameterized.parameters(*_SORT_TESTS)
    def testSort(self, n, dtype, max_val):
        if max_val is None:
            max_val = np.iinfo(numpy_dtype(dtype)).max
        end_bit = int(np.ceil(np.log2(max_val)))
        x = torch.randint(0, max_val, (n,)).cuda().to(dtype)

        out, indices = ops.sort(x, end_bit)
        expected_out, expected_indices = torch.sort(x)
        self.assertTrue(torch.all(torch.eq(out, expected_out)))

        # NOTE: The indices can be in different order depending
        # on sort stability if multiple values in the array are
        # equal.
        data = torch.empty_like(x)
        data.scatter_(0, indices.long(), out)
        expected_data = torch.empty_like(x)
        expected_data.scatter_(0, expected_indices, expected_out)
        self.assertTrue(torch.all(torch.eq(data, expected_data)))


if __name__ == '__main__':
    unittest.main()
