import numpy as np
from megablocks.layers.arguments import Arguments
from megablocks.layers import mpu
import torch


def log_benchmark(name, arguments, time, std):
    print("="*60)
    print(f"{name} Benchmark")
    print("Benchmark Parameters:")
    for (key, value) in arguments.items():
        print(f"{key} = {value}")
    print("Results:")
    print("mean time = {:.3f}ms, std time = {:.3f}ms".format(time, std))
    print("="*60)


def benchmark_function(fn, iterations=100, warmup=10):
    # Warmup iterations.
    for _ in range(warmup):
        fn()

    times = []
    for i in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        fn()
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return np.mean(times), np.std(times)


_FWD_TIMES = {}
_BWD_TIMES = {}


def report_times(args : Arguments):
    global _FWD_TIMES
    global _BWD_TIMES
    
    for key, times in _FWD_TIMES.items():
        if len(times) != args.num_layers:
            return
        time = np.mean(times)
        std = np.std(times)
        mpu.synchronized_print(
            args, "(fwd) {}: {:0.3f}ms, {:0.3f}ms".format(key, time, std))
        
    for key, times in _BWD_TIMES.items():
        if len(times) != args.num_layers:
            return
        time = np.mean(times)
        std = np.std(times)
        mpu.synchronized_print(
            args, "(bwd) {}: {:0.3f}ms, {:0.3f}ms".format(key, time, std))

    _FWD_TIMES = {}
    _BWD_TIMES = {}


class Timer:

    def __init__(self, tag):
        self.tag = tag
        self.fwd_start = torch.cuda.Event(enable_timing=True)
        self.fwd_end = torch.cuda.Event(enable_timing=True)
        self.bwd_start = torch.cuda.Event(enable_timing=True)
        self.bwd_end = torch.cuda.Event(enable_timing=True)

    def _record_fwd_time(self, time):
        global _FWD_TIMES
        if self.tag not in _FWD_TIMES:
            _FWD_TIMES[self.tag] = []
        _FWD_TIMES[self.tag].append(time)

    def _record_bwd_time(self, time):
        global _BWD_TIMES
        if self.tag not in _BWD_TIMES:
            _BWD_TIMES[self.tag] = []
        _BWD_TIMES[self.tag].append(time)

    def start(self, x):
        class Start(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x):
                self.fwd_start.record()
                return x

            @staticmethod
            def backward(ctx, grad):
                self.bwd_end.record()
                torch.cuda.synchronize()
                self._record_bwd_time(self.bwd_start.elapsed_time(self.bwd_end))
                return grad
        return Start.apply(x)

    def end(self, x):
        class End(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x):
                self.fwd_end.record()
                torch.cuda.synchronize()
                self._record_fwd_time(self.fwd_start.elapsed_time(self.fwd_end))
                return x

            @staticmethod
            def backward(ctx, grad):
                self.bwd_start.record()
                return grad
        return End.apply(x)
