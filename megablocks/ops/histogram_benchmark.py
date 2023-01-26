import unittest

from absl.testing import parameterized
from megablocks import ops
import numpy as np
import torch


_HISTOGRAM_TESTS = (
    (16384, torch.int32, 2),
    (16384, torch.int32, 4),
    (16384, torch.int32, 8),
    (16384, torch.int32, 16),
    (16384, torch.int32, 32),
    (16384, torch.int32, 64),
    (16384, torch.int32, 128),
    (16384, torch.int32, 256),
)

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


class HistogramBenchmark(parameterized.TestCase):

    @parameterized.parameters(*_HISTOGRAM_TESTS)
    def testHistogram(self, n, dtype, max_val):
        x = torch.randint(0, max_val, (n,)).cuda().to(dtype)

        mean_t, std_t, max_t, min_t = benchmark_function(
            lambda: ops.histogram(x, max_val))
        arguments = {
            "n": n,
            "dtype": dtype,
            "max_val": max_val
        }
        log_benchmark(arguments, mean_t, std_t)

    @parameterized.parameters(*_HISTOGRAM_TESTS)
    def testTorchHistogram(self, n, dtype, max_val):
        x = torch.randint(0, 128, (n,)).cuda().to(dtype)

        mean_t, std_t, max_t, min_t = benchmark_function(
            lambda: torch.histc(x, max_val, 0, max_val-1))
        arguments = {
            "n": n,
            "dtype": dtype,
            "max_val": max_val
        }
        log_benchmark(arguments, mean_t, std_t)


if __name__ == '__main__':
    unittest.main()
