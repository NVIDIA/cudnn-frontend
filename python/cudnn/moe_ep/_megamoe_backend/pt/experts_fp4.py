# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Simulated-NVFP4 grouped SiLU-GLU expert FFN — quantized twin of experts.py.

Same weight layout and grouping contract as :mod:`pt.experts`; both GEMMs
per expert go through :class:`pt.quant.QuantGemmT` (fake-quantized operands,
fp32 accumulate). With ``qcfg.turboquant`` the fc1 hidden dim is rotated by
a randomized-Hadamard block Q, applied IN-GRAPH to both the token slab and
w13 — mathematically a no-op pre-quantization, and autograd supplies the
exact adjoint so master weights stay unrotated. fc2 is untouched (its K is
the intermediate dim), mirroring megamoe/turboquant.py.

Note: here the rotation/quantization happens after dispatch (per-token ops
commute with the permutation), so the pt path's all-to-all wire stays bf16;
the megamoe kernel's 4-bit wire has the same numerics as fc1's quantized
A-operand, which IS modeled.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from .quant import QuantConfig, QuantGemmT, rotate_trailing


def expert_ffn_fp4(
    x: torch.Tensor, w13_e: torch.Tensor, w2_e: torch.Tensor, qcfg: QuantConfig
) -> torch.Tensor:
    """One expert's FFN on its ``[n, hidden]`` token slab (n may be 0)."""
    fc1 = QuantGemmT.apply(
        x, w13_e, qcfg.fprop_fmt, qcfg.bprop_fmt,
        qcfg.quant_bprop, qcfg.stochastic_rounding_grads,
    )
    linear, gate = fc1.chunk(2, dim=-1)
    return QuantGemmT.apply(
        F.silu(gate) * linear, w2_e, qcfg.fprop_fmt, qcfg.bprop_fmt,
        qcfg.quant_bprop, qcfg.stochastic_rounding_grads,
    )


def grouped_expert_ffn_fp4(
    x_grouped: torch.Tensor,
    tokens_per_expert: Sequence[int],
    w13: torch.Tensor,
    w2: torch.Tensor,
    qcfg: QuantConfig,
    q_rot: torch.Tensor | None = None,
) -> torch.Tensor:
    """Quantized FFN over a token slab pre-grouped by expert (same contract
    as :func:`pt.experts.grouped_expert_ffn`)."""
    if len(tokens_per_expert) != w13.shape[0]:
        raise ValueError(
            f"tokens_per_expert has {len(tokens_per_expert)} entries but "
            f"w13 has {w13.shape[0]} experts"
        )
    if sum(tokens_per_expert) != x_grouped.shape[0]:
        raise ValueError(
            f"tokens_per_expert sums to {sum(tokens_per_expert)} but "
            f"x_grouped has {x_grouped.shape[0]} rows"
        )
    if qcfg.turboquant:
        if q_rot is None:
            raise ValueError("qcfg.turboquant requires q_rot")
        x_grouped = rotate_trailing(x_grouped, q_rot)
        w13 = rotate_trailing(w13, q_rot)
    outs = []
    start = 0
    for e, n in enumerate(tokens_per_expert):
        outs.append(expert_ffn_fp4(x_grouped[start : start + n], w13[e], w2[e], qcfg))
        start += n
    if not outs:
        return x_grouped.new_zeros((0, w2.shape[1]))
    return torch.cat(outs, dim=0)
