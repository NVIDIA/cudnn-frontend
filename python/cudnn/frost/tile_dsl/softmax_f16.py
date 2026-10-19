# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Shared f16x2 exponent arms for the Blackwell-family SDPA forward softmax.

The quantized (per-tensor FP8 and MXFP8) forward kernels store P in the FP8 input format.  Their
default softmax chain runs ``exp2`` in f32 (one MUFU.EX2 per score) and casts to FP8 pairs; the arms
here run the exponent as MUFU EX2.F16x2 on packed pairs (half the MUFU issues) and cast straight from
f16x2 to the FP8 pair format.  ``softmax_precision=HALF`` selects them (TemplateParams.softmax_f16).

With the pre-folded scale (``attn_scale_prefolded`` / TemplateParams.softmax_scale_prefolded: the caller
multiplied Q by attn_scale * log2(e), so the raw QK^T already sits in the log2 domain) the per-score
shift and the f32 -> f16x2 convert fuse into ONE instruction per pair (``nvvm sub.packed f16x2 <-
f32x2 - f32x2``, SASS FHADD2 .FTZ.RZ): 64 fewer instructions per 128 scores.  The op exists in
round-toward-zero form only, so the exponent input is biased toward zero by at most one f16 ulp (P
high by <= 0.03 % near the row max, 0.5 % on the far tail -- below the FP8 cast noise).  A DSL
without the op falls back to the unfused f16 arm (same numerics as HALF alone); the gate also
requires the result-type-first builder form (res, src_a, src_b) the call is written against.

Two row-sum shapes exist among the kernels:

* ones-MMA kernels (the d128 / d192x128 families): O is normalized by the tensor-core row-sum of the
  stored P, so a stats-less graph needs no register sum (``f16_exp_chunk``); the Stats specialization
  keeps the EXACT f32 denominator for the published LSE (``f16_exp_chunk_sum``: a second f32 ``exp2``
  of the shifted scores -- honored, not faster).  The fused arm never materializes the shifted f32
  scores, so it is stats-less only.
* register-sum kernels (the d256 / d512 families): O's denominator is a register reduction, so the
  stats-less HALF arm sums the f16x2 P words it stores (``*_f16sum``: an HADD2 tree over pairs, at
  most eight P values per f16 partial, then f32) -- self-consistent with the P the MMA consumes, and
  below the FP8 P noise.  Stats keep the exact f32 sum (``f16_exp_chunk_sum``), as above.

Every helper takes the FP8 pair tag (``"e4m3"`` / ``"e5m2"``) as a compile-time argument so the
kernels share one copy.  Exp arguments are bounded (<= RESCALE_THRESHOLD, so P <= 2^RESCALE_THRESHOLD):
f16 range is exact where it matters; args below f16 range saturate to -inf -> exp2 -> 0, identical to
the f32 path's underflow.
"""

import inspect as _inspect

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir as _ir
from cutlass._mlir.dialects import llvm as _llvm
from cutlass._mlir.dialects import nvvm as _nvvm_ops
from cutlass._mlir.dialects import vector as _vector
from cutlass._mlir.extras import types as _T
from cutlass.cute.arch.nvvm_wrappers import inline_ptx

from .pointwise import ex2_f16x2, f16x2_to_f32, f16x2x2_to_fp8_word, fp32_to_fp16, row_reduction_pair

# The fused shift+convert needs the DSL's packed f16x2 <- f32x2 subtract in its result-type-first
# builder form.  The f32x2 op family's form differs and is not a proxy for this op.
FUSED_SHIFT_CVT_AVAILABLE = bool(
    hasattr(_nvvm_ops, "sub_packed_f16x2_f32x2_f32x2") and "res" in _inspect.signature(_nvvm_ops.sub_packed_f16x2_f32x2_f32x2).parameters
)

# Depth of the f16x2 pair tree in the register-sum arms: 3 levels => each f16 partial holds at most
# 2^3 = 8 P values.  With P <= 2^RESCALE_THRESHOLD (4 on the quantized kernels) a partial stays
# <= 128, where the f16 ulp is 2^-4 -- a 2^-11 relative rounding per add.
F16_SUM_TREE_LEVELS = 3


def fp8_pair_tag(dtype_qkv: int) -> str:
    """TemplateParams.dtype_qkv code -> the pair tag f16x2x2_to_fp8_word speaks (0 = E4M3, 1 = E5M2)."""
    if dtype_qkv == 0:
        return "e4m3"
    if dtype_qkv == 1:
        return "e5m2"
    raise ValueError(f"fp8_pair_tag: dtype_qkv={dtype_qkv} is not an FP8 input code (0 = E4M3, 1 = E5M2)")


@cute.jit
def add_f16x2(lhs: cutlass.Int32, rhs: cutlass.Int32) -> cutlass.Int32:
    """Packed f16x2 add (one HADD2) -- the pair-tree primitive of the register-sum arms."""
    return inline_ptx("add.f16x2 $0, $1, $2;", write_only_types=[cutlass.Int32], read_only_args=[lhs, rhs])


def f16_pairs_sum_pair(p_pairs):
    """Row-sum of the P values held in packed f16x2 words, as the (even, odd) f32 pair the kernels'
    ``total_sum`` accumulates (the layout :func:`pointwise.row_reduction_pair` produces).

    ``F16_SUM_TREE_LEVELS`` HADD2 levels over the words (each partial sums <= 8 P values, see the
    module docstring), then the surviving words unpack to f32 and finish in f32.  Plain-Python helper
    over DSL values, called from the jit bodies like the per-element helpers."""
    words = list(p_pairs)
    for _ in range(F16_SUM_TREE_LEVELS):
        if len(words) < 2:
            break
        nxt = [add_f16x2(words[2 * i], words[2 * i + 1]) for i in range(len(words) // 2)]
        if len(words) % 2:
            nxt.append(words[-1])
        words = nxt
    lo_sum = None
    hi_sum = None
    for w in words:
        lo, hi = f16x2_to_f32(w)
        lo_sum = lo if lo_sum is None else lo_sum + lo
        hi_sum = hi if hi_sum is None else hi_sum + hi
    return cutlass.Vector.from_elements((lo_sum, hi_sum), cutlass.Float32)


def _exp_pairs(elems, n):
    """``n`` shifted f32 scores -> ``n // 2`` f16x2 words of P (CVT pairs, MUFU EX2.F16x2)."""
    pairs = [fp32_to_fp16(elems[2 * i], elems[2 * i + 1]) for i in range(n // 2)]
    return [ex2_f16x2(w) for w in pairs]


def _fp8_words(p_pairs, fp8_tag, n):
    """``n // 2`` f16x2 P words -> ``n // 4`` packed FP8 words in :func:`pointwise.fp32_to_fp8_pack` byte order."""
    return [f16x2x2_to_fp8_word(p_pairs[2 * g], p_pairs[2 * g + 1], fp8_tag) for g in range(n // 4)]


def fused_shift_f16_pairs(elems, m, n):
    """Fused arm: ``n`` RAW f32 scores -> ``n // 2`` f16x2 words of (score - m), one fused
    sub+convert per pair (FHADD2 .FTZ.RZ).  Plain-Python helper over the MLIR builders; the caller
    guarantees :data:`FUSED_SHIFT_CVT_AVAILABLE`."""
    f32x2 = _ir.VectorType.get([2], _T.f32())
    f16x2 = _ir.VectorType.get([2], _T.f16())
    m_pair = _vector.broadcast(f32x2, m.ir_value())
    words = []
    for i in range(n // 2):
        pair = _vector.from_elements(f32x2, [elems[2 * i].ir_value(), elems[2 * i + 1].ir_value()])
        res = _nvvm_ops.sub_packed_f16x2_f32x2_f32x2(f16x2, pair, m_pair)
        words.append(cutlass.Int32(_llvm.bitcast(_T.i32(), res)))
    return words


@cute.jit
def f16_exp_chunk(chunk_S, fp8_tag: cutlass.Constexpr, n: cutlass.Constexpr[int] = 64):
    """HALF tail for one ``n``-elem chunk of shifted exp-args (f32): f32 pairs pack to f16x2 (CVT),
    the exponent runs as MUFU EX2.F16x2, and P casts straight from f16x2 to the FP8 pair format.
    Returns the ``n // 4`` packed FP8 words (ones-MMA kernels, stats-less)."""
    elems = [chunk_S[i] for i in range(n)]
    words = _fp8_words(_exp_pairs(elems, n), fp8_tag, n)
    return cutlass.Vector.from_elements(tuple(words), cutlass.Int32)


@cute.jit
def f16_exp_chunk_sum(chunk_S, fp8_tag: cutlass.Constexpr, n: cutlass.Constexpr[int] = 64):
    """:func:`f16_exp_chunk` plus the EXACT f32 row-sum pair of P (Stats specializations): the
    published LSE keeps the f32 denominator (summing the f16 P instead measured rms 3.6e-4 off it).
    HALF + Stats is honored, not faster than the f32 chain."""
    elems = [chunk_S[i] for i in range(n)]
    words = _fp8_words(_exp_pairs(elems, n), fp8_tag, n)
    p_sum = row_reduction_pair(cute.math.exp2(chunk_S, fastmath=True))
    return cutlass.Vector.from_elements(tuple(words), cutlass.Int32), p_sum


@cute.jit
def f16_exp_chunk_f16sum(chunk_S, fp8_tag: cutlass.Constexpr, n: cutlass.Constexpr[int] = 64):
    """:func:`f16_exp_chunk` plus the f16-tree row-sum pair of the SAME P words (register-sum kernels,
    stats-less): O's denominator stays self-consistent with the P the MMA consumes."""
    elems = [chunk_S[i] for i in range(n)]
    p_pairs = _exp_pairs(elems, n)
    words = _fp8_words(p_pairs, fp8_tag, n)
    return cutlass.Vector.from_elements(tuple(words), cutlass.Int32), f16_pairs_sum_pair(p_pairs)


@cute.jit
def fused_shift_f16_exp_chunk(chunk_S_raw, m, fp8_tag: cutlass.Constexpr, n: cutlass.Constexpr[int] = 64):
    """HALF + pre-folded tail on RAW scores: fused (score - m) -> f16x2, MUFU EX2.F16x2, f16x2x2 -> FP8
    word.  Same output contract as :func:`f16_exp_chunk` (ones-MMA kernels, stats-less)."""
    elems = [chunk_S_raw[i] for i in range(n)]
    p_pairs = [ex2_f16x2(w) for w in fused_shift_f16_pairs(elems, m, n)]
    words = _fp8_words(p_pairs, fp8_tag, n)
    return cutlass.Vector.from_elements(tuple(words), cutlass.Int32)


@cute.jit
def fused_shift_f16_exp_chunk_f16sum(chunk_S_raw, m, fp8_tag: cutlass.Constexpr, n: cutlass.Constexpr[int] = 64):
    """:func:`fused_shift_f16_exp_chunk` plus the f16-tree row-sum pair (register-sum kernels, stats-less)."""
    elems = [chunk_S_raw[i] for i in range(n)]
    p_pairs = [ex2_f16x2(w) for w in fused_shift_f16_pairs(elems, m, n)]
    words = _fp8_words(p_pairs, fp8_tag, n)
    return cutlass.Vector.from_elements(tuple(words), cutlass.Int32), f16_pairs_sum_pair(p_pairs)


def f16_exp_values(values, fp8_tag, n, *, fused_m=None):
    """Granularity-free form for kernels that pack P per 16-byte vector (the role-split d512 bodies):
    ``values`` is a list of ``n`` f32 scores (shifted, or RAW when ``fused_m`` is given); returns
    ``(fp8_words, p_pairs)`` as Python lists -- ``n // 4`` packed FP8 words in fp32_to_fp8_pack byte
    order and the ``n // 2`` f16x2 P words, so the caller can gather the pairs of a whole kv-step and
    call :func:`f16_pairs_sum_pair` once."""
    if fused_m is None:
        p_pairs = _exp_pairs(values, n)
    else:
        p_pairs = [ex2_f16x2(w) for w in fused_shift_f16_pairs(values, fused_m, n)]
    return _fp8_words(p_pairs, fp8_tag, n), p_pairs
