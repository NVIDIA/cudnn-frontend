# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Saved-state scalar moments and deterministic final weight reduction."""

import triton
import triton.language as tl


@triton.jit
def _signed_root(dot):
    root = tl.sqrt(tl.maximum(tl.abs(dot), 1.0e-6))
    bits = root.to(tl.uint32, bitcast=True) | (dot.to(tl.uint32, bitcast=True) & 0x80000000)
    return root, bits.to(tl.float32, bitcast=True)


@triton.jit
def cudnn_engram_gate_saved_moments(V, G, SAVED, COEF, VS: tl.constexpr, D: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    token = row // 4
    col = tl.arange(0, B)
    gate = tl.load(SAVED + row * 4)
    dot = tl.load(SAVED + row * 4 + 1)
    rx = tl.load(SAVED + row * 4 + 2)
    rk = tl.load(SAVED + row * 4 + 3)
    scale = rx * rk * D**-0.5
    root, signed = _signed_root(dot)
    g = tl.load(G + row * D + col, col < D, 0).to(tl.float32)
    v = tl.load(V + token * VS + col, col < D, 0).to(tl.float32)
    gv = tl.sum(g * v, 0)
    ddot = tl.where(tl.abs(dot) >= 1.0e-6, gv * gate * (1.0 - gate) * (0.5 / root), 0.0)
    coefficient = ddot * scale
    cx = ddot * dot * rx * rx / D
    ck = ddot * dot * rk * rk / D
    tl.store(COEF + row * 4, gate)
    tl.store(COEF + row * 4 + 1, coefficient)
    tl.store(COEF + row * 4 + 2, cx)
    tl.store(COEF + row * 4 + 3, ck)


@triton.jit
def cudnn_engram_gate_weight_reduce(PART, DW, SPLITS: tl.constexpr, D: tl.constexpr, BS: tl.constexpr, BD: tl.constexpr):
    col = tl.program_id(0) * BD + tl.arange(0, BD)
    split = tl.arange(0, BS)
    values = tl.load(PART + split[:, None] * (4 * D) + col[None, :], (split[:, None] < SPLITS) & (col[None, :] < 4 * D), 0)
    tl.store(DW + col, tl.sum(values, 0), col < 4 * D)
