from megablocks import benchmark_util
import torch
import torch.nn.functional as F


_FFN_BENCHMARK = (
    # MPT-1B
    (2, 2048, 2048, 2048 * 4),
    # MPT-3B
    (2, 2048, 2560, 2560 * 4),
    # MPT-7B
    (2, 2048, 4096, 4096 * 4),
    # MPT-13B
    (2, 2048, 5120, 5120 * 4),
    # MPT-30B
    (2, 2048, 7168, 7168 * 4),
)


class FFN(torch.nn.Module):

    def __init__(self, hidden_size, ffn_hidden_size):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.empty(
            hidden_size,
            ffn_hidden_size,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16))
        self.w2 = torch.nn.Parameter(torch.empty(
            ffn_hidden_size,
            hidden_size,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16))

    def forward(self, x):
        return torch.matmul(F.gelu(torch.matmul(x, self.w1), approximate='tanh'), self.w2)


def benchmark_ffn(
        batch_size,
        sequence_length,
        hidden_size,
        ffn_hidden_size):
    details = {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "hidden_size": hidden_size,
        "ffn_hidden_size": ffn_hidden_size,
    }
    layer = FFN(hidden_size, ffn_hidden_size).cuda().to(torch.bfloat16)

    x = torch.randn((batch_size, sequence_length, hidden_size)).cuda().to(torch.bfloat16)
    fn = lambda: layer(x)
    time, std = benchmark_util.benchmark_function(fn)
    benchmark_util.log_benchmark("FFN (Fwd)", details, time, std)


if __name__ == '__main__':
    for args in _FFN_BENCHMARK:
        benchmark_ffn(*args)
