from megablocks.layers.arguments import Arguments
import torch
import torch.nn.functional as F


def allclose(x, y, pct=0.5):
    mask = torch.isclose(x, y, rtol=5e-2)
    pct_diff = (mask.numel() - mask.sum()) / mask.numel() * 100
    if pct_diff > pct:
        print("{:.2f}% of values not close.".format(pct_diff))
        return False
    return True


class FFN(torch.nn.Module):

    def __init__(self, args : Arguments):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.empty(
            args.hidden_size,
            args.ffn_hidden_size,
            device=args.device,
            dtype=torch.float16 if args.fp16 else torch.float32))
        self.w2 = torch.nn.Parameter(torch.empty(
            args.ffn_hidden_size,
            args.hidden_size,
            device=args.device,
            dtype=torch.float16 if args.fp16 else torch.float32))

    def forward(self, x):
        return torch.matmul(F.gelu(
            torch.matmul(x, self.w1), approximate="tanh"), self.w2)

class GLU(FFN):

    def __init__(self, args : Arguments):
        super().__init__(args)
        self.v1 = torch.nn.Parameter(torch.empty(
            args.hidden_size,
            args.ffn_hidden_size,
            device=args.device,
            dtype=torch.float16 if args.fp16 else torch.float32))

    def forward(self, x):
        x1 = F.gelu(torch.matmul(x, self.w1), approximate="tanh") * torch.matmul(x, self.v1)
        return torch.matmul(x1, self.w2)
