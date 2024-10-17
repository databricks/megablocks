# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0
from typing import Any

import torch

from megablocks.layers import common
from megablocks.layers.arguments import Arguments

_ROUTER_LOGITS = []


def _save_router_logits(logits: torch.Tensor, args: Arguments):
    if args.moe_zloss_weight == 0:
        return
    global _ROUTER_LOGITS
    _ROUTER_LOGITS.append(logits)


def clear_router_zloss():
    global _ROUTER_LOGITS
    _ROUTER_LOGITS.clear()


def batched_router_zloss(args: Arguments):
    global _ROUTER_LOGITS

    if args.moe_zloss_weight == 0:
        import warnings
        warnings.warn('Call to batched_router_zloss, but moe_zloss_weight=0')
        return 0

    logits_per_router = _ROUTER_LOGITS

    if args.moe_zloss_in_fp32:
        logits_per_router = [logits.float() for logits in logits_per_router]

    unscaled_zloss_per_router = torch.stack([
        torch.logsumexp(logits, dim=1).square().mean() for logits in logits_per_router
    ])

    return args.moe_zloss_weight * unscaled_zloss_per_router


# NOTE: To enable end-to-end benchmarking without convergence we
# support a flag to force the router to assign tokens uniformly
# across the experts. We do this with a custom autograd operation
# so that PyTorch still executes the full set of router operation.
class _UniformExpertAssignment(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, num_experts: int):
        out = torch.arange(x.numel(), dtype=x.dtype, device=x.device)
        out = torch.remainder(out, num_experts)
        return out.view(x.shape)


_uniform_expert_assignment = _UniformExpertAssignment.apply


class LearnedRouter(torch.nn.Module):

    def __init__(self, args: Arguments):
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
            dtype=common.dtype(args),
            device=args.device,
        )
        args.init_method(self.layer.weight)

    def jitter(self, x: torch.Tensor):
        low: float = 1.0 - self.args.moe_jitter_eps
        high: float = 1.0 + self.args.moe_jitter_eps
        noise = torch.rand(x.size(), dtype=x.dtype, device=x.device)
        return low + noise * (high - low)

    def _top_k(self, scores: torch.Tensor):
        if self.args.moe_top_k == 1:
            return scores.max(dim=-1, keepdim=True)
        return torch.topk(scores, self.args.moe_top_k, dim=-1)

    def forward(self, x: torch.Tensor):
        if self.training and self.args.moe_jitter_eps is not None:
            x = x * self.jitter(x)

        logits = self.layer(x.view(-1, x.shape[-1]))
        _save_router_logits(logits, self.args)
        scores = logits.softmax(dim=-1)
        expert_weights, expert_indices = self._top_k(scores)
        if self.args.moe_normalize_expert_weights:
            expert_weights = expert_weights / torch.norm(
                expert_weights,
                p=self.args.moe_normalize_expert_weights,
                dim=-1,
                keepdim=True,
            )

        expert_indices = (
            _uniform_expert_assignment(
                expert_indices,
                self.args.moe_num_experts,
            ) if self.args.uniform_expert_assignment else expert_indices
        )
        return scores, expert_weights, expert_indices
