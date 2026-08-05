# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""bf16 expert weights -> the backward kernel's MXFP8 B operands.

Mirrors megamoe/weights.py but for the adjoint GEMMs (BWD_DESIGN.md D2/D4):

  gemm1 (dA = doutg @ W2^T):    B = W2^T, logical (E, hidden, I) with hidden
                                stride-1 (K=H), quantized along H.
  gemm2 (dxg = DFC1 @ W13^T):   B = W13^T over the INTERLEAVED gate/up axis,
                                logical (E, 2I, hidden) with 2I stride-1
                                (K=2I), quantized along 2I.

Column convention: the backward epilogue writes DFC1 pool columns as
interleaved 32-blocks in (gate, up) slot order — [dgate_b0 | dup_b0 | ...] —
so gemm2's K axis must use the same `interleave_gate_up` ordering as the
forward fc1 N axis.  SFs are per-expert 32x4x4 atom-swizzled (`to_blocked`),
plain copies kept for host reference dequant.
"""

from dataclasses import dataclass

import torch

import megamoe.repo_path  # noqa: F401

from common.host_utils import mxfp8_quantize_per_block_32
from common.megamoe_constants import Mxfp8BlockSize
from moe_nvfp4_swapab.runner_common import (
    to_blocked,
    _stack_byte_reinterpretable_tensors,
)

from megamoe.weights import _KIND_TO_TORCH_DTYPE, interleave_gate_up


@dataclass
class QuantizedBwdWeights:
    """Kernel-ready backward B operands (+ plain SFs for host reference)."""

    gemm1_weight: torch.Tensor         # (E, H, I) fp8 view, H stride-1
    gemm1_weight_sf: torch.Tensor      # (E, flat) E8M0, atom-swizzled
    gemm2_weight: torch.Tensor         # (E, 2I, H) fp8 view, 2I stride-1
    gemm2_weight_sf: torch.Tensor      # (E, flat) E8M0, atom-swizzled
    gemm1_weight_sf_plain: torch.Tensor  # (E, I, H//32) E8M0
    gemm2_weight_sf_plain: torch.Tensor  # (E, H, 2I//32) E8M0


def quantize_moe_weights_mxfp8_bwd(
    w13: torch.Tensor,     # [E, 2I, H] bf16, pt layout ([:I]=linear/up, [I:]=gate)
    w2: torch.Tensor,      # [E, H, I] bf16
    kind: str = "mxfp8_e4m3",
) -> QuantizedBwdWeights:
    """Quantize bf16 masters into the backward kernel's MXFP8 layout."""
    data_dtype = _KIND_TO_TORCH_DTYPE[kind]
    E, two_i, H = w13.shape
    I = two_i // 2
    if w2.shape != (E, H, I):
        raise ValueError(f"w2 must be (E={E}, H={H}, I={I}), got {tuple(w2.shape)}.")
    if H % Mxfp8BlockSize or I % Mxfp8BlockSize:
        raise ValueError(
            f"hidden ({H}) and intermediate ({I}) must be multiples of "
            f"{Mxfp8BlockSize}."
        )

    # gemm1: W2^T (E, I, H) contiguous, quantize along K=hidden.
    g1_f32 = w2.float().cuda().transpose(1, 2).contiguous()
    g1_q, g1_sf = mxfp8_quantize_per_block_32(g1_f32.reshape(E * I, H), data_dtype)
    g1_q = g1_q.reshape(E, I, H)
    g1_sf_plain = g1_sf.reshape(E, I, H // Mxfp8BlockSize)
    gemm1_weight = g1_q.permute(0, 2, 1)  # logical (E, H, I), H stride-1

    # gemm2: W13^T over the interleaved gate/up axis, (E, H, 2I) contiguous,
    # quantize along K=2I.
    g2_f32 = interleave_gate_up(w13).float().cuda().transpose(1, 2).contiguous()
    g2_q, g2_sf = mxfp8_quantize_per_block_32(g2_f32.reshape(E * H, two_i), data_dtype)
    g2_q = g2_q.reshape(E, H, two_i)
    g2_sf_plain = g2_sf.reshape(E, H, two_i // Mxfp8BlockSize)
    gemm2_weight = g2_q.permute(0, 2, 1)  # logical (E, 2I, H), 2I stride-1

    g1_sf_sw = _stack_byte_reinterpretable_tensors(
        [to_blocked(g1_sf_plain[e]) for e in range(E)], dim=0
    )
    g2_sf_sw = _stack_byte_reinterpretable_tensors(
        [to_blocked(g2_sf_plain[e]) for e in range(E)], dim=0
    )

    return QuantizedBwdWeights(
        gemm1_weight=gemm1_weight,
        gemm1_weight_sf=g1_sf_sw,
        gemm2_weight=gemm2_weight,
        gemm2_weight_sf=g2_sf_sw,
        gemm1_weight_sf_plain=g1_sf_plain,
        gemm2_weight_sf_plain=g2_sf_plain,
    )
