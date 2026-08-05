# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""bf16 expert weights -> the MXFP8 layout the MegaMoE kernel consumes.

Input convention (matches `pt/` and flashinfer.moe_ep):

  w13  [E, 2*I, H]  rows [:I] = linear/up projection, rows [I:] = gate;
                    activation is silu(gate) * linear.
  w2   [E, H, I]    down projection.

Kernel convention (see moe_mxfp8_glu/mega_runner.py + mega_reference_mxfp8.py):

  fc1_weight     logical (E, hidden, 2*I) with hidden stride-1 (K-major);
                 the N=2*I axis interleaves gate/up in 32-wide blocks:
                 [gate_block0 | up_block0 | gate_block1 | up_block1 | ...]
                 ("PostSwigluHalf" interleave, Mxfp8BlockSize granularity).
  fc2_weight     logical (E, I, hidden) with I stride-1 (K-major).
  *_weight_sf    per-expert 32x4x4 atom-swizzled flat E8M0 tensors
                 (`to_blocked`); the plain per-32-block SFs are kept too so
                 host references can dequantize.
"""

from dataclasses import dataclass

import torch

import megamoe.repo_path  # noqa: F401  (sys.path side effect)

from common.host_utils import mxfp8_quantize_per_block_32
from common.megamoe_constants import Mxfp8BlockSize
from moe_nvfp4_swapab.runner_common import (
    to_blocked,
    _stack_byte_reinterpretable_tensors,
)

_KIND_TO_TORCH_DTYPE = {
    "mxfp8_e4m3": torch.float8_e4m3fn,
    "mxfp8_e5m2": torch.float8_e5m2,
}


@dataclass
class QuantizedExpertWeights:
    """Kernel-ready weights (+ plain SFs for host reference dequant)."""

    fc1_weight: torch.Tensor        # (E, H, 2I) fp8 view, H stride-1
    fc1_weight_sf: torch.Tensor     # (E, flat) E8M0, atom-swizzled
    fc2_weight: torch.Tensor        # (E, I, H) fp8 view, I stride-1
    fc2_weight_sf: torch.Tensor     # (E, flat) E8M0, atom-swizzled
    fc1_weight_sf_plain: torch.Tensor  # (E, 2I, H//32) E8M0
    fc2_weight_sf_plain: torch.Tensor  # (E, H, I//32) E8M0


def interleave_gate_up(w13: torch.Tensor) -> torch.Tensor:
    """[E, 2I, H] ([:I]=up/linear, [I:]=gate) -> kernel N-interleaved [E, 2I, H]."""
    E, two_i, H = w13.shape
    I = two_i // 2
    if I % Mxfp8BlockSize != 0:
        raise ValueError(f"intermediate ({I}) must be a multiple of {Mxfp8BlockSize}.")
    up = w13[:, :I].reshape(E, I // Mxfp8BlockSize, Mxfp8BlockSize, H)
    gate = w13[:, I:].reshape(E, I // Mxfp8BlockSize, Mxfp8BlockSize, H)
    # pair order (gate, up): mega_reference_mxfp8 splits _reshaped[:, :, 0]=gate,
    # [:, :, 1]=up.
    inter = torch.stack((gate, up), dim=2)  # (E, I/32, 2, 32, H)
    return inter.reshape(E, two_i, H)


def quantize_moe_weights_mxfp8(
    w13: torch.Tensor,
    w2: torch.Tensor,
    kind: str = "mxfp8_e4m3",
) -> QuantizedExpertWeights:
    """Quantize bf16/fp32 expert weights into the kernel's MXFP8 layout."""
    data_dtype = _KIND_TO_TORCH_DTYPE[kind]
    E, two_i, H = w13.shape
    I = two_i // 2
    if w2.shape != (E, H, I):
        raise ValueError(f"w2 must be (E={E}, H={H}, I={I}), got {tuple(w2.shape)}.")
    if H % Mxfp8BlockSize != 0 or I % Mxfp8BlockSize != 0:
        raise ValueError(
            f"hidden ({H}) and intermediate ({I}) must be multiples of "
            f"{Mxfp8BlockSize}."
        )

    # fc1: interleave gate/up, quantize along K=hidden in 32-blocks.
    fc1_bf16 = interleave_gate_up(w13).float().cuda()
    fc1_q, fc1_sf = mxfp8_quantize_per_block_32(fc1_bf16.reshape(E * two_i, H), data_dtype)
    fc1_q = fc1_q.reshape(E, two_i, H)
    fc1_sf_plain = fc1_sf.reshape(E, two_i, H // Mxfp8BlockSize)
    # (E, 2I, H) contiguous -> logical (E, H, 2I), hidden stride-1.  Do NOT
    # .contiguous() -- the K-as-stride-1 invariant is what the kernel expects.
    fc1_weight = fc1_q.permute(0, 2, 1)

    # fc2: quantize along K=intermediate in 32-blocks.
    fc2_f32 = w2.float().cuda()
    fc2_q, fc2_sf = mxfp8_quantize_per_block_32(fc2_f32.reshape(E * H, I), data_dtype)
    fc2_q = fc2_q.reshape(E, H, I)
    fc2_sf_plain = fc2_sf.reshape(E, H, I // Mxfp8BlockSize)
    fc2_weight = fc2_q.permute(0, 2, 1)  # logical (E, I, H), I stride-1

    # 32x4x4 atom swizzle per expert (what the TMA SFA descriptor reads).
    fc1_sf_sw = _stack_byte_reinterpretable_tensors(
        [to_blocked(fc1_sf_plain[e]) for e in range(E)], dim=0
    )
    fc2_sf_sw = _stack_byte_reinterpretable_tensors(
        [to_blocked(fc2_sf_plain[e]) for e in range(E)], dim=0
    )

    return QuantizedExpertWeights(
        fc1_weight=fc1_weight,
        fc1_weight_sf=fc1_sf_sw,
        fc2_weight=fc2_weight,
        fc2_weight_sf=fc2_sf_sw,
        fc1_weight_sf_plain=fc1_sf_plain,
        fc2_weight_sf_plain=fc2_sf_plain,
    )
