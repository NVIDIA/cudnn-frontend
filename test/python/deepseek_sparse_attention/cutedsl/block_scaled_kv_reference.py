# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torch reference for the block-scaled packed KV caches of the sparse attention forward.

- ``widen_kv(kv, kv_sf)`` is the BF16 pool that codes ``kv`` and scales
  ``kv_sf`` stand for, which a packed plan must read byte for byte: the FP32
  product of each code and its scale, rounded once to BF16.
- ``quantize_mxfp8`` / ``quantize_nvfp4`` write codes and scales the way the
  pinned DeepSeek writers do (``benchmark/rope_qdq/references_fp8.py`` and
  ``references_compkv.py`` hold the QDQ form of the same rules).
- ``pack_rows`` builds the rows of an ``mxfp8_528`` or ``nvfp4_288`` cache
  (the ``kv`` of a ``kv_format`` plan); ``row_views`` splits them back into
  zero-copy codes and scales.
- ``quantize_rows`` / ``widen_rows`` do both ends by format name: the rows of
  a cache that stores ``x``, and the BF16 pool a cache's rows stand for.

Everything here computes on the tensors' device with plain torch ops; FP4 codes
are handled as ``uint8`` because torch has no arithmetic on ``float4_e2m1fn_x2``.
"""

from typing import Optional, Tuple

import torch

MXFP8_BLOCK = 32
NVFP4_BLOCK = 16
E4M3_MAX = 448.0
E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)  # codes 0..7; code | 0x8 is the negative


def e2m1_values(codes: torch.Tensor) -> torch.Tensor:
    """4-bit E2M1 codes (``uint8`` 0..15, sign in bit 3) -> FP32 values (code 8 is ``-0.0``)."""
    lut = torch.tensor(E2M1_GRID + tuple(-v for v in E2M1_GRID), dtype=torch.float32, device=codes.device)
    return lut[codes.long()]


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """Bytes ``[..., n]`` -> codes ``[..., 2n]``; channel ``2i`` is the low nibble of byte ``i``."""
    packed = packed.view(torch.uint8)
    return torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2)


def pack_e2m1(codes: torch.Tensor) -> torch.Tensor:
    """Codes ``[..., 2n]`` -> ``float4_e2m1fn_x2 [..., n]`` with channel ``2i`` in the low nibble of byte ``i``."""
    codes = codes.to(torch.uint8)
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).view(torch.float4_e2m1fn_x2)


def widen_mxfp8(kv: torch.Tensor, kv_sf: torch.Tensor) -> torch.Tensor:
    """``float8_e4m3fn (S, D)`` codes with ``float8_e8m0fnu (S, D/32)`` scales -> BF16 ``(S, D)``."""
    scale = kv_sf.float().repeat_interleave(MXFP8_BLOCK, dim=-1)
    return (kv.float() * scale).to(torch.bfloat16)


def widen_nvfp4(kv: torch.Tensor, kv_sf: torch.Tensor) -> torch.Tensor:
    """``float4_e2m1fn_x2 (S, D/2)`` codes with ``float8_e4m3fn (S, D/16)`` scales -> BF16 ``(S, D)``."""
    scale = kv_sf.float().repeat_interleave(NVFP4_BLOCK, dim=-1)
    return (e2m1_values(unpack_e2m1(kv)) * scale).to(torch.bfloat16)


def widen_kv(kv: torch.Tensor, kv_sf: torch.Tensor) -> torch.Tensor:
    """The BF16 pool that codes ``kv`` with scales ``kv_sf`` stand for, in either block-scaled format."""
    if kv.dtype == torch.float8_e4m3fn and kv_sf.dtype == torch.float8_e8m0fnu:
        return widen_mxfp8(kv, kv_sf)
    if kv.dtype == torch.float4_e2m1fn_x2 and kv_sf.dtype == torch.float8_e4m3fn:
        return widen_nvfp4(kv, kv_sf)
    raise ValueError(f"no block-scaled KV format has {kv.dtype} codes with {kv_sf.dtype} scales")


def quantize_mxfp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(S, D)`` -> E4M3 codes ``(S, D)`` and UE8M0 scales ``(S, D/32)``.

    Per 32 channels: ``ratio = max(amax, 1e-4) * fp32(1/448)``, the scale is
    ``ratio`` rounded up to a power of two, and the codes are
    ``e4m3(clamp(x / scale, -448, 448))`` rounded to nearest even.
    """
    groups = x.float().reshape(*x.shape[:-1], x.shape[-1] // MXFP8_BLOCK, MXFP8_BLOCK)
    ratio = groups.abs().amax(dim=-1).clamp_min(1e-4) * (1.0 / E4M3_MAX)
    bits = ratio.view(torch.int32)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    scale = (exponent << 23).view(torch.float32)
    codes = (groups / scale[..., None]).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return codes.reshape(x.shape), exponent.to(torch.uint8).view(torch.float8_e8m0fnu)


def _e2m1_rne_codes(x: torch.Tensor) -> torch.Tensor:
    """FP32 in ``[-6, 6]`` -> E2M1 codes, rounding to nearest with ties to the even code (sign kept, ``-0`` is code 8)."""
    magnitude = x.abs()
    index = torch.zeros_like(magnitude, dtype=torch.uint8)
    for code, (above, tie_up) in enumerate(((0.25, False), (0.75, True), (1.25, False), (1.75, True), (2.5, False), (3.5, True), (5.0, False)), start=1):
        crossed = magnitude >= above if tie_up else magnitude > above
        index = torch.where(crossed, torch.tensor(code, dtype=torch.uint8, device=x.device), index)
    return index | (torch.signbit(x).to(torch.uint8) << 3)


def quantize_nvfp4(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(S, D)`` -> E2M1 codes ``float4_e2m1fn_x2 (S, D/2)`` and E4M3 scales ``(S, D/16)``, with no tensor scale.

    Per 16 channels: the scale is ``e4m3(min(max(amax, 6 * 2**-9) / 6, 448))``
    and the codes are ``e2m1(clamp(x / scale, -6, 6))``, dividing by the
    rounded scale.
    """
    groups = x.float().reshape(*x.shape[:-1], x.shape[-1] // NVFP4_BLOCK, NVFP4_BLOCK)
    ratio = groups.abs().amax(dim=-1).clamp_min(6.0 * 2**-9) / 6.0
    scale = ratio.clamp_max(E4M3_MAX).to(torch.float8_e4m3fn)
    normalized = (groups / scale.float()[..., None]).clamp(-6.0, 6.0)
    return pack_e2m1(_e2m1_rne_codes(normalized).reshape(x.shape)), scale


def pack_rows(kv: torch.Tensor, kv_sf: torch.Tensor, *, row_pitch: Optional[int] = None) -> torch.Tensor:
    """Codes and scales -> ``uint8 (S, row_pitch)`` rows ``[codes | scales | padding]``; padding bytes are ``0xFF``."""
    codes, scales = kv.view(torch.uint8), kv_sf.view(torch.uint8)
    record = codes.shape[-1] + scales.shape[-1]
    row_pitch = record if row_pitch is None else row_pitch
    if row_pitch < record:
        raise ValueError(f"row_pitch {row_pitch} is smaller than the {record}-byte record")
    rows = torch.full((codes.shape[0], row_pitch), 0xFF, dtype=torch.uint8, device=codes.device)
    rows[:, : codes.shape[-1]] = codes
    rows[:, codes.shape[-1] : record] = scales
    return rows


def row_views(rows: torch.Tensor, kv_dtype: torch.dtype, code_bytes: int, scale_bytes: int, scale_dtype: torch.dtype):
    """The codes and scales of rows written by ``pack_rows``, as zero-copy views."""
    return rows[:, :code_bytes].view(kv_dtype), rows[:, code_bytes : code_bytes + scale_bytes].view(scale_dtype)


# Each served format's record in ``row_views`` argument order: (code dtype, code bytes, scale bytes, scale dtype).
ROW_LAYOUTS = {
    "mxfp8_528": (torch.float8_e4m3fn, 512, 16, torch.float8_e8m0fnu),
    "nvfp4_288": (torch.float4_e2m1fn_x2, 256, 32, torch.float8_e4m3fn),
}


def quantize_rows(x: torch.Tensor, kv_format: str, *, row_pitch: Optional[int] = None) -> torch.Tensor:
    """``(S, 512)`` -> the ``uint8`` rows of a ``kv_format`` cache that stores ``x`` (``pack_rows`` of its codes and scales)."""
    quantize = {"mxfp8_528": quantize_mxfp8, "nvfp4_288": quantize_nvfp4}[kv_format]
    return pack_rows(*quantize(x), row_pitch=row_pitch)


def widen_rows(rows: torch.Tensor, kv_format: str) -> torch.Tensor:
    """The BF16 pool the rows of a ``kv_format`` cache stand for."""
    return widen_kv(*row_views(rows, *ROW_LAYOUTS[kv_format]))
