import unittest

from absl.testing import parameterized
from megablocks import ops
import numpy as np
import torch


_SORT_TESTS = (
    (16384, torch.int32, None),
    (16384, torch.int32, 2),
    (16384, torch.int32, 128),
)

_BASELINE_SORT_TESTS = (
    (16384,),
)


def numpy_dtype(dtype):
    types = {
        torch.int16: np.int16,
        torch.int32: np.int32,
        torch.int64: np.int64
    }
    return types[dtype]


def benchmark_function(fn, iterations=10):
    # Run once to get rid of startup overhead.
    fn()
    times = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times = np.array(times)
    return times.mean(), times.std(), times.max(), times.min()


def log_benchmark(arguments, mean_t, std_t):
    print("="*60)
    print("Benchmark Parameters:")
    for (key, value) in arguments.items():
        print(f"{key} = {value}")
    print("Results:")
    print("mean / std = {:.2f}ms / {:.2f}ms".format(mean_t, std_t))
    print("="*60)


class SortBenchmark(parameterized.TestCase):

    @parameterized.parameters(*_SORT_TESTS)
    def testSort(self, n, dtype, max_val):
        if max_val is None:
            max_val = np.iinfo(numpy_dtype(dtype)).max
        end_bit = int(np.ceil(np.log2(max_val)))
        x = torch.randint(0, max_val, (n,)).cuda().to(dtype)

        mean_t, std_t, max_t, min_t = benchmark_function(
            lambda: ops.sort(x, end_bit))
        arguments = {
            "n": n,
            "dtype": dtype,
            "max_val": max_val
        }
        log_benchmark(arguments, mean_t, std_t)

    @parameterized.parameters(*_BASELINE_SORT_TESTS)
    def testTorchSort(self, n):
        x = torch.randint(0, 128, (n,)).cuda().to(torch.int32)

        mean_t, std_t, max_t, min_t = benchmark_function(
            lambda: torch.sort(x))
        arguments = {"n": n,}
        log_benchmark(arguments, mean_t, std_t)


if __name__ == '__main__':
    unittest.main()
