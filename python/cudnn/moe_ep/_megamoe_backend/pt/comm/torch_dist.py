# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""c10d (torch.distributed) TokenComm backend.

The only custom autograd op in the whole EP layer lives here: raw c10d
collectives are not differentiable, so ``_AllToAllSingle`` provides the
adjoint explicitly — backward is the same ``all_to_all_single`` with input
and output split sizes swapped. Everything around it (permutations, top-k
replication, weighted combine) is autograd-native, which makes the full
dispatch/combine backward readable directly from the forward code.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.distributed as dist

from .base import TokenComm, register_comm


def _all_to_all_single(
    x: torch.Tensor,
    output_splits: Sequence[int],
    input_splits: Sequence[int],
    group: Optional[dist.ProcessGroup],
) -> torch.Tensor:
    out = x.new_empty((sum(output_splits),) + tuple(x.shape[1:]))
    dist.all_to_all_single(
        out,
        x.contiguous(),
        output_split_sizes=list(output_splits),
        input_split_sizes=list(input_splits),
        group=group,
    )
    return out


class _AllToAllSingle(torch.autograd.Function):
    """Differentiable uneven all-to-all; backward = adjoint exchange."""

    @staticmethod
    def forward(ctx, x, output_splits, input_splits, group):
        ctx.output_splits = tuple(output_splits)
        ctx.input_splits = tuple(input_splits)
        ctx.group = group
        return _all_to_all_single(x, output_splits, input_splits, group)

    @staticmethod
    def backward(ctx, grad_out):
        # Rows this rank sent to rank r in forward came back as grad rows
        # from rank r here: swap the split-size lists.
        grad_in = _all_to_all_single(
            grad_out, ctx.input_splits, ctx.output_splits, ctx.group
        )
        return grad_in, None, None, None


@register_comm("torch_dist")
class TorchDistComm(TokenComm):
    def __init__(self, group: Optional[dist.ProcessGroup] = None):
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "torch_dist comm backend requires torch.distributed to be "
                "initialized (init_process_group) before layer construction"
            )
        self._group = group

    def all_to_all(self, x, output_splits, input_splits):
        return _AllToAllSingle.apply(x, output_splits, input_splits, self._group)

    def all_to_all_no_grad(self, x, output_splits, input_splits):
        with torch.no_grad():
            return _all_to_all_single(x, output_splits, input_splits, self._group)

    def exchange_counts(self, send_counts):
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts.contiguous(), group=self._group)
        return recv_counts
