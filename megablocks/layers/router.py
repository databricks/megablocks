from megablocks.layers.arguments import Arguments
import torch

class LearnedRouter(torch.nn.Module):

    def __init__(self, args : Arguments):
        super().__init__()
        self.args = args

        # Learned router parameters.
        #
        # NOTE: This weight matrix is not parallelized with expert model
        # parallelism. Each device needs the entire router weight matrix
        # so that it can route its batch of data correctly.
        self.layer = torch.nn.Linear(
            args.hidden_size,
            args.moe_num_experts,
            bias=False,
            dtype=torch.float16 if args.fp16 else torch.float32,
            device=args.device)
        args.init_method(self.layer.weight)

    def jitter(self, x):
        low = 1.0 - self.args.moe_jitter_eps
        high = 1.0 + self.args.moe_jitter_eps
        noise = torch.rand(x.size(), dtype=x.dtype, device=x.device)
        return low + noise * (high - low)

    def forward(self, x):
        if self.training and self.args.moe_jitter_eps is not None:
            x = x * self.jitter(x)

        sl, bs, hs = x.size()
        scores = self.layer(x.view(-1, hs)).softmax(dim=-1)
        if self.args.moe_top_k == 1:
            return scores, *scores.max(dim=-1)
        return scores, *torch.topk(scores, self.args.moe_top_k, dim=-1)
