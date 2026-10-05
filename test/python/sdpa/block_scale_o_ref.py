# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torch reference for block-scaled SDPA outputs (NVFP4 and MXFP8 O + SF_O).

Mirrors the kernel epilogue exactly:

* NVFP4 (block 16 along d): ``sf = e4m3(amax_16 / 6)``; data = ``e2m1(o * (1 / decode(sf)))``
  round-to-nearest-even, two nibbles per byte (element ``2j`` in the low nibble).
* MXFP8 (block 32 along d): ``e = ue8m0_ceil(amax_32 / 448)`` (TE ``float_to_e8m0``
  semantics, rounded UP); data = ``e4m3(o * 2^(127 - e))``.

``o`` is the fp32 output AFTER ``scale_o`` (the kernel folds ``scale_o`` into the
row normalization), so dequantization is ``code * decode(sf) / scale_o``.

SF_O is laid out in the cuDNN / TE ``F8_128x4`` atom order over a matrix of
``[rows, cols]`` scale factors — either one plane per (b, h) (``rows = S`` padded
to 128, ``cols = d / block`` padded to 4) or one token-major matrix
(``rows = B * S``, ``cols = H * d / block``) that a downstream GEMM consumes
directly.  ``sfo_geometry`` returns the offsets the kernel needs for either.
"""

from __future__ import annotations

import torch

from sdpa.mxfp8_quant import _swizzle_128x4, ceil_div, e8m0_ceil

E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E4M3_MAX = 448.0
E2M1_MAX = 6.0


def round_up(a: int, b: int) -> int:
    return ceil_div(a, b) * b


def quantize_e2m1(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> E2M1 code (uint8 0..15), round-to-nearest-even, saturating at 6."""
    sign = (x < 0).to(torch.uint8)
    mag = x.abs().clamp(max=E2M1_MAX).float()
    values = E2M1_VALUES.to(x.device)
    # nearest with ties-to-even on the CODE index (matches cvt.rn.satfinite.e2m1x2)
    idx = torch.bucketize(mag, values, right=False)  # values[idx-1] <= mag < values[idx]
    idx = idx.clamp(1, len(values) - 1)
    lo, hi = values[idx - 1], values[idx]
    pick_hi = (mag - lo) > (hi - mag)
    tie = (mag - lo) == (hi - mag)
    pick_hi = pick_hi | (tie & ((idx % 2) == 0))  # even code wins the tie
    code = torch.where(pick_hi, idx, idx - 1).to(torch.uint8)
    return code | (sign << 3)


def dequantize_e2m1(code: torch.Tensor) -> torch.Tensor:
    values = E2M1_VALUES.to(code.device)
    mag = values[(code & 0x7).long()]
    return torch.where((code & 0x8) != 0, -mag, mag)


def pack_e2m1(code: torch.Tensor) -> torch.Tensor:
    """[..., d] codes -> [..., d // 2] bytes, element 2j in the low nibble."""
    assert code.shape[-1] % 2 == 0
    lo = code[..., 0::2]
    hi = code[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8)


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    return torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def quantize_o_nvfp4(o_scaled: torch.Tensor):
    """``o_scaled`` [..., d] fp32 -> (packed uint8 [..., d//2], sf e4m3 [..., d//16], dequant fp32)."""
    d = o_scaled.shape[-1]
    assert d % 16 == 0
    blocks = o_scaled.float().reshape(*o_scaled.shape[:-1], d // 16, 16)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    sf = (amax / E2M1_MAX).to(torch.float8_e4m3fn)
    sf_dec = sf.float()
    inv = torch.where(sf_dec > 0, 1.0 / sf_dec, torch.zeros_like(sf_dec))
    code = quantize_e2m1(blocks * inv).reshape(*o_scaled.shape[:-1], d)
    deq = (dequantize_e2m1(code).reshape_as(blocks) * sf_dec).reshape(*o_scaled.shape[:-1], d)
    return pack_e2m1(code), sf.reshape(*o_scaled.shape[:-1], d // 16), deq


def quantize_o_mxfp8(o_scaled: torch.Tensor):
    """``o_scaled`` [..., d] fp32 -> (e4m3 data [..., d], sf ue8m0 uint8 [..., d//32], dequant fp32)."""
    d = o_scaled.shape[-1]
    assert d % 32 == 0
    blocks = o_scaled.float().reshape(*o_scaled.shape[:-1], d // 32, 32)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    e = e8m0_ceil(amax / E4M3_MAX)  # biased exponent byte, rounded up
    # kernel: inv = 2^(127-e) for 0 < e < 254; e == 0 -> 0 (all-zero block)
    inv = torch.where(e > 0, torch.pow(2.0, 127.0 - e.float()), torch.zeros_like(amax))
    data = (blocks * inv).to(torch.float8_e4m3fn)
    scale = torch.pow(2.0, e.float() - 127.0)
    deq = (data.float() * scale).reshape(*o_scaled.shape[:-1], d)
    return data.reshape(*o_scaled.shape[:-1], d), e.to(torch.uint8).reshape(*o_scaled.shape[:-1], d // 32), deq


def sfo_matrix_planes(sf_bshc: torch.Tensor) -> torch.Tensor:
    """Per-(b,h) planes: ``sf`` [B, S, H, C] -> swizzled uint8 [B, H, S_pad128, C_pad4]."""
    b, s, h, c = sf_bshc.shape
    s_pad, c_pad = round_up(s, 128), round_up(c, 4)
    m = torch.zeros(b, h, s_pad, c_pad, dtype=torch.uint8, device=sf_bshc.device)
    m[:, :, :s, :c] = sf_bshc.permute(0, 2, 1, 3).view(torch.uint8)
    return _swizzle_128x4(m)


def sfo_matrix_token_major(sf_bshc: torch.Tensor) -> torch.Tensor:
    """Token-major matrix a GEMM consumes: ``sf`` [B, S, H, C] -> swizzled uint8 [round_up(B*S,128), H*C]."""
    b, s, h, c = sf_bshc.shape
    rows = round_up(b * s, 128)
    m = torch.zeros(rows, h * c, dtype=torch.uint8, device=sf_bshc.device)
    m[: b * s] = sf_bshc.reshape(b * s, h * c).view(torch.uint8)
    assert (h * c) % 4 == 0, "token-major SF_O needs H*C % 4 == 0"
    return _swizzle_128x4(m)


def unswizzle_128x4(m: torch.Tensor) -> torch.Tensor:
    """Inverse of ``_swizzle_128x4`` for a [..., R, C] uint8 matrix."""
    *lead, R, C = m.shape
    assert R % 128 == 0 and C % 4 == 0
    # forward: (R//128, 4, 32, C//4, 4) permuted from logical (R//128, 32, 4, C//4, 4)
    # logical index (rb, r32, r4, cb, c4) with r = rb*128 + r4*32 + r32 ... derive from the
    # byte-offset formula off = rb*128*C + cb*512 + r32*16 + r4*4 + c4.
    flat = m.reshape(*lead, R // 128, C // 4, 32, 4, 4)  # (rb, cb, r32, r4, c4) in offset order
    out = flat.permute(*range(len(lead)), len(lead) + 0, len(lead) + 3, len(lead) + 2, len(lead) + 1, len(lead) + 4)
    # (rb, r4, r32, cb, c4) -> r = rb*128 + r4*32 + r32
    return out.reshape(*lead, R, C)


def dequant_block_scaled_o(o_bytes: torch.Tensor, sf_o_buf: torch.Tensor, blk: int, layout: str, b: int, h: int, s: int, d_v: int, c: int):
    """(O container, sf_o bytes) -> fp32 O [B, H, S, d_v] in ``scale_o`` units, plus whether
    every kernel-owned pad row of a per-(b,h) plane came back zero (token-major has none).

    ``o_bytes`` is the BSHD-physical O as the graph sees it ([B, H, S, d_v // pack] view):
    the packed E2M1 container for ``blk == 16``, E4M3 for ``blk == 32``. ``sf_o_buf`` is
    the swizzled uint8 SF_O in the declared ``layout`` ("planes": [B, H, R, C]; "token_major":
    [round_up(B*S, 128), H*C]); ``c`` is the padded scale-column count (``d_v // blk`` rounded to 4).
    """
    c_used = d_v // blk
    if layout == "planes":
        logical = unswizzle_128x4(sf_o_buf)  # (B, H, R, C)
        sf_log = logical[:, :, :s, :c_used]
        pad_ok = bool((logical[:, :, s:, :] == 0).all().item())
    else:
        logical = unswizzle_128x4(sf_o_buf)  # (B*S padded, H*C)
        sf_log = logical[: b * s].reshape(b, s, h, c)[..., :c_used].permute(0, 2, 1, 3)
        pad_ok = True
    if blk == 16:
        codes = unpack_e2m1(o_bytes.transpose(1, 2).contiguous().view(torch.uint8)).reshape(b, s, h, d_v).permute(0, 2, 1, 3)
        sf = sf_log.view(torch.float8_e4m3fn).float()
        vals = dequantize_e2m1(codes)
    else:
        sf = torch.pow(2.0, sf_log.float() - 127.0)
        vals = o_bytes.float()
    o = (vals.reshape(b, h, s, c_used, blk) * sf[..., None]).reshape(b, h, s, d_v)
    return o, pad_ok
