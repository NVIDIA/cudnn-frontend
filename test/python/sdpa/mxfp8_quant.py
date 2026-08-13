# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Pure-torch MXFP8 block quantization + cuDNN F8_128x4 scale-factor swizzle.

Drop-in replacement for the TransformerEngine primitives the SDPA MXFP8 tests
used to require (``MXFP8Quantizer`` / ``MXFP8Tensor`` /
``tex.swizzle_scales_for_gemm_``), so the mxfp8 test populations run without a
TransformerEngine install.

Semantics are replicated from the TE sources (verified against the checkout at
/home/scratch.vagarwalla_gpu/TransformerEngine):

* E8M0 scale computation — ``transformer_engine/common/util/ptx.cuh
  float_to_e8m0``: the per-32-element block scale is
  ``e8m0 = ceil_to_power_of_two(amax * (1.0f/max_norm))`` where the exponent is
  rounded **up** (``cvt.rp.satfinite.ue8m0x2.f32`` on Blackwell; the pre-Blackwell
  C fallback increments the biased fp32 exponent whenever the mantissa is
  non-zero).  Special cases (same file): ``amax == 0 -> 0x00`` (scale 2^-127),
  ``inf -> 0xFE`` (satfinite, 2^127), ``nan -> 0xFF``.
* ``max_norm`` — ``transformer_engine/common/utils.cuh Quantized_Limits``:
  448 for E4M3, 57344 for E5M2; ``max_norm_rcp`` is the fp32 rounding of
  ``1.0/max_norm`` and the ``amax * max_norm_rcp`` product is a single fp32
  multiply (both reproduced bit-exactly here).
* Data cast — ``transformer_engine/common/cast/mxfp8/quantize_mxfp8.cuh``:
  ``data = cast_fp8(x * exp2f_rcp(e8m0))`` where ``exp2f_rcp`` is the *exact*
  power-of-two reciprocal ``2^(127-e)`` (``ptx.cuh exp2f_rcp``, incl. the
  ``e == 254 -> 2^-127`` subnormal special case).  TE converts with satfinite;
  torch's fp8 cast is not guaranteed saturating, so we clamp to ±max_norm first
  (a no-op except for 1-ulp boundary cases, where it matches TE's satfinite).
* Storage layout — ``transformer_engine/pytorch/tensor/mxfp8_tensor.py``:
  for a 2-D input ``[M, K]``: rowwise (scales along K) scale_inv is
  ``[round_up(M,128), round_up(K/32,4)]``; columnwise (scales along M) data is
  *not* transposed and scale_inv is ``[round_up(M/32,4), round_up(K,128)]``.
  The SDPA tests pre-pad M and K to multiples of 128, making TE's padding a
  no-op; this module asserts that.
* F8_128x4 swizzle — ``transformer_engine/common/swizzle/swizzle.cu``
  (``swizzle_row_scaling_kernel`` / ``swizzle_col_scaling_kernel``, the target
  of ``tex.swizzle_scales_for_gemm_``), cross-checked against the consumer side
  in this repo (``python/cudnn/sdpa/fwd/api_dsl.py`` ``_reshape_sf`` +
  ``kernels/prefill_d128_mxfp8_sm100.py`` SF_SMEM_SIZE_*): the scale matrix
  ``[R, C]`` (R = rows of the GEMM operand as consumed, C = number of
  32-element blocks along the contraction dim) is tiled into 128x4 "atoms" of
  512 bytes, atoms ordered row-major, and inside an atom byte
  ``(r % 32) * 16 + (r // 32) * 4 + (c % 4)`` holds scale ``(r, c)``.
  For the columnwise (transposed) operand the same rule is applied to the
  transposed scale matrix ``[K, M/32]``.

Everything here is torch-only (CPU or CUDA) and used to generate *test inputs*;
nothing is needed by the cudnn runtime lowering.
"""

import torch

__all__ = [
    "FP8_MAX_NORM",
    "e8m0_ceil",
    "e8m0_to_float",
    "exp2_rcp",
    "quantize_blocks",
    "quantize_mxfp8_2d",
    "swizzle_sf_rowwise",
    "swizzle_sf_columnwise",
    "quantize_to_mxfp8",
]

# fmt: off

FP8_MAX_NORM = {
    torch.float8_e4m3fn: 448.0,
    torch.float8_e5m2: 57344.0,
}


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def e8m0_ceil(v: torch.Tensor) -> torch.Tensor:
    """fp32 tensor (>= 0) -> biased E8M0 exponent byte, TE ``float_to_e8m0`` semantics.

    Rounds the scale UP to the next power of two (unless the value already is
    one).  0 -> 0x00, inf -> 0xFE (satfinite), nan -> 0xFF.
    """
    v = v.float().contiguous()
    bits = v.view(torch.int32)
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    # TE ptx.cuh float_to_e8m0 (non-Blackwell fallback, == cvt.rp.satfinite for
    # finite inputs): round the exponent up whenever the mantissa is non-zero,
    # except deep subnormals (exp==0, mant<=0x400000 i.e. v <= 2^-127) which
    # stay at 0x00.
    round_up = (mant > 0) & (exp != 0xFE) & ~((exp == 0) & (mant <= 0x400000))
    e = exp + round_up.to(torch.int32)
    e = torch.where(torch.isnan(v), torch.full_like(e, 0xFF), e)
    e = torch.where(torch.isinf(v), torch.full_like(e, 0xFE), e)
    return e.to(torch.uint8)


def exp2_rcp(e: torch.Tensor) -> torch.Tensor:
    """Biased E8M0 byte -> exact fp32 ``2^(127-e)`` (TE ``exp2f_rcp``), incl. the
    ``e == 254 -> 2^-127`` (fp32 subnormal 0x00400000) special case."""
    e32 = e.to(torch.int32)
    rcp_bits = torch.where(e32 == 254, torch.full_like(e32, 0x00400000), (254 - e32) << 23)
    return rcp_bits.view(torch.float32)


def e8m0_to_float(e: torch.Tensor) -> torch.Tensor:
    """Biased E8M0 byte -> fp32 scale value ``2^(e-127)`` (dequant scale)."""
    return e.contiguous().view(torch.float8_e8m0fnu).float()


def quantize_blocks(x: torch.Tensor, fp8_dtype: torch.dtype):
    """Quantize ``x[..., 32]`` blocks (last dim = one 32-element block).

    Returns ``(data fp8 same shape, e8m0 uint8 x.shape[:-1])``.
    """
    max_norm = FP8_MAX_NORM[fp8_dtype]
    x = x.float()
    amax = x.abs().amax(dim=-1)
    # single fp32 multiply by the fp32 rounding of 1/max_norm, exactly as TE
    inv = torch.tensor(1.0 / max_norm, dtype=torch.float32, device=x.device)
    e = e8m0_ceil(amax * inv)
    rcp = exp2_rcp(e).unsqueeze(-1)
    data = (x * rcp).clamp_(-max_norm, max_norm).to(fp8_dtype)
    return data, e


def quantize_mxfp8_2d(t2d: torch.Tensor, fp8_dtype: torch.dtype):
    """TE ``MXFP8Quantizer(rowwise=True, columnwise=True)`` on a 2-D ``[M, K]``
    tensor with M, K multiples of 128 (SDPA tests pre-pad to this).

    Returns ``(rowwise_data, rowwise_scale_inv, columnwise_data,
    columnwise_scale_inv)`` with TE's shapes:
    rowwise_data fp8 ``[M, K]`` (scaled along K), rowwise_scale_inv uint8
    ``[M, K//32]``; columnwise_data fp8 ``[M, K]`` (scaled along M, NOT
    transposed), columnwise_scale_inv uint8 ``[M//32, K]``.
    """
    M, K = t2d.shape
    assert M % 128 == 0 and K % 128 == 0, f"pre-pad to 128 multiples, got {t2d.shape}"
    x = t2d.float()
    row_data, row_e = quantize_blocks(x.reshape(M, K // 32, 32), fp8_dtype)
    col_data_t, col_e_t = quantize_blocks(x.T.contiguous().reshape(K, M // 32, 32), fp8_dtype)
    return (
        row_data.reshape(M, K),
        row_e,                                        # [M, K//32]
        col_data_t.reshape(K, M).T.contiguous(),      # [M, K]
        col_e_t.transpose(0, 1).contiguous(),         # [M//32, K]
    )


def _swizzle_128x4(sf: torch.Tensor) -> torch.Tensor:
    """Core F8_128x4 reorder of a logical scale matrix ``[..., R, C]`` (uint8),
    R % 128 == 0, C % 4 == 0.  Atoms (128 rows x 4 cols = 512 B) are laid out
    row-major; inside an atom, scale (r, c) lives at byte
    ``(r % 32) * 16 + (r // 32) * 4 + c``.  Output is uint8 ``[..., R, C]``
    (same byte count; only the physical order matters)."""
    assert sf.dtype == torch.uint8
    *lead, R, C = sf.shape
    assert R % 128 == 0 and C % 4 == 0, f"F8_128x4 needs R%128==0, C%4==0, got {sf.shape}"
    v = sf.reshape(*lead, R // 128, 4, 32, C // 4, 4)  # (rt, rg, rr, ct, cc)
    n = len(lead)
    v = v.permute(*range(n), n + 0, n + 3, n + 2, n + 1, n + 4)  # (rt, ct, rr, rg, cc)
    return v.contiguous().reshape(*lead, R, C)


def swizzle_sf_rowwise(sf: torch.Tensor) -> torch.Tensor:
    """Swizzle a rowwise scale_inv ``[..., M, K//32]`` (scales along the
    contraction dim K) into cuDNN's F8_128x4 order (TE
    ``swizzle_row_scaling_kernel``)."""
    return _swizzle_128x4(sf)


def swizzle_sf_columnwise(sf: torch.Tensor) -> torch.Tensor:
    """Swizzle a columnwise scale_inv stored as ``[..., M//32, K]`` (TE storage;
    scales along M) into cuDNN's F8_128x4 order for the transposed operand
    ``[K, M]`` (TE ``swizzle_col_scaling_kernel``): the 128x4 rule is applied to
    the transposed scale matrix ``[K, M//32]``.  Output keeps the storage shape
    ``[..., M//32, K]`` (bytes are what the kernel consumes)."""
    out = _swizzle_128x4(sf.transpose(-1, -2).contiguous())
    return out.reshape(sf.shape)


def quantize_to_mxfp8(
    tensor,
    b,
    h,
    s,
    d,
    block_size=32,
    fp8_dtype=torch.float8_e4m3fn,
    with_ref=True,
):
    """Torch-only drop-in for the TE-based ``quantize_to_mxfp8`` in
    ``test/python/sdpa/mxfp8.py`` (same signature and return convention).

    Returns ``(fp8_data_d, sf_d_ref, sf_d_swizzle, fp8_data_s, sf_s_ref,
    sf_s_swizzle)`` where the ``_d`` triple is the rowwise quantization (scales
    along D, for Q/K/dO) and ``_s`` the columnwise one (scales along S, for
    V/transposed operands).  ``sf_*_ref`` are per-element fp32 dequant scales
    ``[b, h, s, d]`` (None when ``with_ref=False``); ``sf_*_swizzle`` are the
    F8_128x4-reordered uint8 scale tensors cuDNN/the FROST kernel consume.
    """
    assert block_size == 32, "MXFP8 uses 32-element blocks"
    l = b * h

    d_scale_padded = ceil_div(ceil_div(d, block_size), 4) * 4
    d_padded = d_scale_padded * block_size
    s_scale_padded = ceil_div(ceil_div(s, block_size), 4) * 4
    s_padded = s_scale_padded * block_size

    tensor_3d = tensor.float().reshape(l, s, d)
    pad_d = d_padded - d
    pad_s = s_padded - s
    if pad_s > 0 or pad_d > 0:
        tensor_3d = torch.nn.functional.pad(tensor_3d, (0, pad_d, 0, pad_s))
    tensor_2d = tensor_3d.reshape(l * s_padded, d_padded)

    row_data, row_e, col_data, col_e = quantize_mxfp8_2d(tensor_2d, fp8_dtype)

    # --- Rowwise results (quantized along D) ---
    fp8_data_d = row_data.reshape(l, s_padded, d_padded)[:, :s, :d].contiguous().reshape(b, h, s, d)
    sf_d_ref = None
    if with_ref:
        sf_d_f32 = e8m0_to_float(row_e)
        sf_d_ref = torch.repeat_interleave(
            sf_d_f32.reshape(l, s_padded, d_scale_padded), repeats=block_size, dim=2
        )[:, :s, :d].contiguous()
    sf_d_swizzle = swizzle_sf_rowwise(row_e)  # uint8 [l*s_padded, d_scale_padded]

    # --- Columnwise results (quantized along S) ---
    fp8_data_s = col_data.reshape(l, s_padded, d_padded)[:, :s, :d].contiguous().reshape(b, h, s, d)
    sf_s_ref = None
    if with_ref:
        sf_s_f32 = e8m0_to_float(col_e)
        sf_s_ref = torch.repeat_interleave(
            sf_s_f32.reshape(l, s_scale_padded, d_padded), repeats=block_size, dim=1
        )[:, :s, :d].contiguous()
    sf_s_swizzle = swizzle_sf_columnwise(col_e)  # uint8 [l*s_scale_padded, d_padded]

    return fp8_data_d, sf_d_ref, sf_d_swizzle, fp8_data_s, sf_s_ref, sf_s_swizzle
