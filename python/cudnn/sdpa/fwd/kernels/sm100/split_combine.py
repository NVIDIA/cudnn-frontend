# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""KV-split combine (reduction) pass for the SM100 DSL prefill SDPA kernels.

Each split writes ``O_s`` (normalized by its own running sum) and ``lse_s`` into
a split-major workspace at batch coord ``b + s*B``; this pass reduces over ``s``.

A split whose range came out empty ends with total_sum == 0, which the epilogue
turns into ``O := 0 / lse := -inf`` — the identity here, so empty splits need no
special case.

One block per (q_row, head, batch); the block's threads stride over d_v, and
each thread walks the split axis in registers.
"""

from cudnn.frost.compiled_cache import compile_cached as _compile_cached, template_key as _template_key
from typing import Callable, Optional, Tuple

from functools import lru_cache
import hashlib
from pathlib import Path

from cudnn.sdpa.fwd.kernels._quantized import _unscale_amax_kernel

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm
import cuda.bindings.driver as _cuda_driver  # noqa: F401  (cute.compile pulls cuda)

# This helper is imported normally, outside the parameterized template loader.
# Supply its source identity so template_key can persist its pointer entry points;
# the cache environment manifest also covers all transitive Python sources.
FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]

# One block per (q_row, head, batch); 128 lanes stride over d_v, including
# the wider d256/d512 flavors.
THREADS = 128

NEG_INF = float("-inf")

# The FP8 entries are reachable only as the OUTPUT type, where this pass
# performs the single cast.  "f32" is a PARTIAL-only type: the split epilogue can
# write its fp32 accumulator straight to global (no SMEM staging), so the
# reduction reads exactly what the mainloop produced with no intermediate
# rounding.
_ELEM = {
    "f32": cutlass.Float32,
    "f16": cutlass.Float16,
    "bf16": cutlass.BFloat16,
    "e4m3": cutlass.Float8E4M3FN,
    "e5m2": cutlass.Float8E5M2,
}


@cute.kernel
def _combine_kernel(
    o_partial: cute.Tensor,  # [S*B, S_q, H, D] — split-major partial O
    lse_partial: cute.Tensor,  # [S*B, H, S_q]   — split-major partial LSE
    o_out: cute.Tensor,  # [B, S_q, H, D]
    lse_out: Optional[cute.Tensor],  # [B, H, S_q] or None (None-specialized)
    # 1-element fp32 amax of the RECOMBINED O.  The per-split epilogues cannot
    # compute this: each sees only its own partial, and O is a convex
    # combination of those, so a max over partials over-reports (~2.9x at 8
    # splits).  The FP8 kernels therefore skip their in-kernel amax write when
    # SPLIT_KV > 1 and leave it to this pass.  None-specialized off otherwise.
    amax_o: Optional[cute.Tensor],
    # 1-element fp32 output scale, present only when O is stored quantized.
    # The split kernels leave it out of their epilogue and write UNSCALED
    # partials; this pass applies it at the single cast, so the FP8 rounding
    # happens once, on the recombined value.
    scale_o: Optional[cute.Tensor],
    n_batch: cutlass.Int32,
    n_splits: cutlass.Int32,
    d_v: cutlass.Int32,
    stats_log2: cutlass.Constexpr[bool],  # write the FINAL LSE in base 2 (stats_use_log2)
    # Ragged final rows (the decode tile's RAGGED_Q leg): (B+1,) int32 / int64 ragged
    # offsets of Q, O and Stats in ELEMENTS and their elements-per-token
    # divisors.  The partials stay dense; only the final O / LSE rows move to
    # ``offset[b] / div + q_row`` on the packed [1, T, ...] outputs, and a row
    # at or past this sequence's Q length (offset[b+1] - offset[b]) / div --
    # every row of a zero-length sequence -- is not written at all.  None
    # (None-specialized) keeps the dense (batch, q_row) placement.
    ragged_q: Optional[cute.Tensor] = None,
    ragged_o: Optional[cute.Tensor] = None,
    ragged_lse: Optional[cute.Tensor] = None,
    ragged_q_div: cutlass.Int32 = 1,
    ragged_o_div: cutlass.Int32 = 1,
    ragged_lse_div: cutlass.Int32 = 1,
    # Packed token capacities of the final O and Stats buffers (ragged only): a
    # row whose ragged placement lands at or past them is not written, so a
    # short buffer or a bad offset can never store outside the caller's bytes.
    ragged_o_cap: cutlass.Int32 = 0,
    ragged_lse_cap: cutlass.Int32 = 0,
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    q_row = cute.arch.block_idx()[0]
    head = cute.arch.block_idx()[1]
    batch = cute.arch.block_idx()[2]

    op = cutlass.make_array_view(o_partial)
    lp = cutlass.make_array_view(lse_partial)
    oo = cutlass.make_array_view(o_out)

    # Where the recombined row lands: dense (batch, q_row), or the ragged
    # placement on the packed outputs (batch coord 0).
    o_batch = batch
    o_tok = q_row
    lse_batch = batch
    lse_tok = q_row
    row_live = cutlass.Int32(1) == cutlass.Int32(1)
    lse_live = row_live
    if cutlass.const_expr(ragged_q is not None):
        # Offsets are int32 or int64 elements; divide in 64 bits (a large packed
        # buffer's element offset can exceed 2^31) and keep the token index in 32.
        # Bounds are checked in 64 bits BEFORE narrowing: a negative or wrapped
        # offset must fail the capacity test, not alias a valid token after the cast.
        rq = cutlass.make_array_view(ragged_q)
        q_base = cutlass.Int64(rq[batch]) // cutlass.Int64(ragged_q_div)
        q_len64 = cutlass.Int64(rq[batch + cutlass.Int32(1)]) // cutlass.Int64(ragged_q_div) - q_base
        in_seq = cutlass.Int64(q_row) < q_len64
        o_batch = cutlass.Int32(0)
        o_tok64 = cutlass.Int64(cutlass.make_array_view(ragged_o)[batch]) // cutlass.Int64(ragged_o_div) + cutlass.Int64(q_row)
        row_live = in_seq & (o_tok64 >= cutlass.Int64(0)) & (o_tok64 < cutlass.Int64(ragged_o_cap))
        o_tok = cutlass.Int32(o_tok64)
        lse_live = row_live
        if cutlass.const_expr(ragged_lse is not None):
            lse_batch = cutlass.Int32(0)
            lse_tok64 = cutlass.Int64(cutlass.make_array_view(ragged_lse)[batch]) // cutlass.Int64(ragged_lse_div) + cutlass.Int64(q_row)
            lse_live = in_seq & (lse_tok64 >= cutlass.Int64(0)) & (lse_tok64 < cutlass.Int64(ragged_lse_cap))
            lse_tok = cutlass.Int32(lse_tok64)

    # --- pass 1: M = max_s lse_s, then den = sum_s exp(lse_s - M) ---
    # Every lane redundantly walks the (very short) split axis; the values are
    # block-uniform and hit L1, which is cheaper than staging them through SMEM.
    m = cutlass.Float32(NEG_INF)
    for s in cutlass.range(0, n_splits, 1, unroll=1):
        lse_row = lp[batch + s * n_batch, head, :]
        m = cute.math.max(m, cutlass.Float32(lse_row[q_row]))

    # All splits dead (every row fully masked): emit O := 0 / lse := -inf rather
    # than exp(-inf - -inf) == NaN.  m_safe only feeds the exponentials.
    all_dead = m == cutlass.Float32(NEG_INF)
    m_safe = cutlass.Float32(arith.select(all_dead.ir_value(), cutlass.Float32(0.0).ir_value(), m.ir_value()))

    # Same reasoning as pass 2: skip dead splits rather than trusting a fastmath
    # exp(-inf) to be exactly 0.
    den = cutlass.Float32(0.0)
    for s in cutlass.range(0, n_splits, 1, unroll=1):
        lse_row = lp[batch + s * n_batch, head, :]
        lse_s = cutlass.Float32(lse_row[q_row])
        if lse_s > cutlass.Float32(NEG_INF):
            den = den + cute.math.exp(lse_s - m_safe, fastmath=True)

    inv_den = cutlass.Float32(1.0) / cute.math.max(den, cutlass.Float32(1e-30))
    inv_den = cutlass.Float32(arith.select(all_dead.ir_value(), cutlass.Float32(0.0).ir_value(), inv_den.ir_value()))

    # --- pass 2: O = sum_s w_s O_s / den, accumulated in fp32 ---
    #
    # A dead split (empty KV range) carries lse_s = -inf, so its weight is
    # exp(-inf) == 0 and it should contribute nothing.  Relying on the ARITHMETIC
    # to erase it is not safe: 0 * x is NaN for a non-finite x, and under
    # fastmath the weight itself is only approximately zero.  Skip such splits
    # outright -- they are the identity element of this reduction by
    # construction, so branching is exact where multiplying is not.  (Observed:
    # d512 with 5 KV tiles over 8 splits produced NaN in the recombined O
    # without this guard, even though every partial slot held a clean
    # -inf / 0.)
    neg_inf = cutlass.Float32(NEG_INF)
    amax_local = cutlass.Float32(0.0)
    q_scale = cutlass.Float32(1.0)
    if cutlass.const_expr(scale_o is not None):
        q_scale = cutlass.Float32(cutlass.make_array_view(scale_o)[0])
    for d0 in cutlass.range(tidx, d_v, THREADS, unroll=1):
        acc = cutlass.Float32(0.0)
        for s in cutlass.range(0, n_splits, 1, unroll=1):
            lse_row = lp[batch + s * n_batch, head, :]
            lse_s = cutlass.Float32(lse_row[q_row])
            if lse_s > neg_inf:
                w = cute.math.exp(lse_s - m_safe, fastmath=True)
                o_row = op[batch + s * n_batch, q_row, head, :]
                acc = acc + w * cutlass.Float32(o_row[d0])
        o_val = acc * inv_den
        if cutlass.const_expr(amax_o is not None):
            # Measured before scale_o, so this is already the pre-quant amax and
            # needs no post-hoc divide (the single-pass epilogue's does).
            amax_local = cute.math.max(amax_local, cute.math.max(o_val, -o_val))
        if cutlass.const_expr(scale_o is not None):
            o_val = o_val * q_scale
        # Index all modes: ArrayView's row slice is a pointer and drops the
        # final mode's stride, so a subsequent [d0] would assume contiguous D.
        if row_live:
            oo[o_batch, o_tok, head, d0] = o_val.to(o_out.element_type)

    # One atomic per lane.  The value is non-negative, so its fp32 bit pattern
    # orders the same as the float and an integer atomicMax is exact -- the same
    # trick the kernels' own epilogues use.
    if cutlass.const_expr(amax_o is not None):
        _amax_ptr = Pointer(amax_o.iterator.raw_ptr(), dtype=cutlass.Int32)
        nvvm.atomicrmw(nvvm.AtomicOp.MAX, _amax_ptr, amax_local.bitcast(cutlass.Int32))

    # --- the recombined LSE (only when the caller asked for Stats) ---
    if cutlass.const_expr(lse_out is not None):
        if (tidx == cutlass.Int32(0)) & lse_live:
            lo = cutlass.make_array_view(lse_out)
            lse_val = m_safe + cute.math.log(cute.math.max(den, cutlass.Float32(1e-30)), fastmath=True)
            lse_val = cutlass.Float32(arith.select(all_dead.ir_value(), cutlass.Float32(NEG_INF).ir_value(), lse_val.ir_value()))
            # Base-2 Stats (stats_use_log2): the partials stay natural (the merge
            # above needs them); only the final value converts. -inf stays -inf.
            if cutlass.const_expr(stats_log2):
                lse_val = lse_val * cutlass.Float32(1.4426950408889634)
            lo[lse_batch, head, lse_tok] = lse_val


_combine_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _launch_combine(
    o_partial: cute.Tensor,
    lse_partial: cute.Tensor,
    o_out: cute.Tensor,
    lse_out: Optional[cute.Tensor],
    amax_o: Optional[cute.Tensor],
    scale_o: Optional[cute.Tensor],
    problem_size: Tuple[int, int, int, int],
    n_splits: cutlass.Int32,
    stats_log2: cutlass.Constexpr[bool],
    ragged_q: Optional[cute.Tensor],
    ragged_o: Optional[cute.Tensor],
    ragged_lse: Optional[cute.Tensor],
    ragged_divs: Tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
    ragged_caps: Tuple[cutlass.Int32, cutlass.Int32],
    stream: _cuda_driver.CUstream = None,
) -> None:
    """One launch for both ABIs: dense placement (ragged tensors None) or the
    ragged-Q leg's placement at the offsets, bounded by the (O, Stats) packed
    token capacities."""
    B, H, SQ, D = problem_size
    _combine_kernel(
        o_partial,
        lse_partial,
        o_out,
        lse_out,
        amax_o,
        scale_o,
        cutlass.Int32(B),
        n_splits,
        cutlass.Int32(D),
        stats_log2,
        ragged_q,
        ragged_o,
        ragged_lse,
        cutlass.Int32(ragged_divs[0]),
        cutlass.Int32(ragged_divs[1]),
        cutlass.Int32(ragged_divs[2]),
        cutlass.Int32(ragged_caps[0]),
        cutlass.Int32(ragged_caps[1]),
    ).launch(
        grid=(SQ, H, B),
        block=[THREADS, 1, 1],
        stream=stream,
    )


@cute.jit
def _host_ptr(
    o_partial_ptr: cute.Pointer,
    lse_partial_ptr: cute.Pointer,
    o_out_ptr: cute.Pointer,
    lse_out_ptr: Optional[cute.Pointer],
    problem_size: Tuple[int, int, int, int],
    n_splits: cutlass.Int32,
    o_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64, cutlass.Int64],
    lse_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    stats_log2: cutlass.Constexpr[bool],
    stream: _cuda_driver.CUstream = None,
) -> None:
    """Bind prepared f16/bf16 pointers to the same reduction as tensor callers.

    Partial workspaces are compact; final O and Stats use the actual caller
    strides. Widen before computing compact strides, including split batches.
    This is the dense placement's positional ABI (unchanged); the ragged-Q
    decode leg compiles :func:`_host_ptr_ragged` instead.
    """
    o_partial, lse_partial, o_out, lse_out = _ptr_operands(
        o_partial_ptr, lse_partial_ptr, o_out_ptr, lse_out_ptr, problem_size, n_splits, o_strides, lse_strides
    )
    _launch_combine(o_partial, lse_partial, o_out, lse_out, None, None, problem_size, n_splits, stats_log2, None, None, None, (1, 1, 1), (0, 0), stream)


@cute.jit
def _host_ptr_quantized(
    o_partial_ptr: cute.Pointer,
    lse_partial_ptr: cute.Pointer,
    o_out_ptr: cute.Pointer,
    lse_out_ptr: Optional[cute.Pointer],
    problem_size: Tuple[int, int, int, int],
    n_splits: cutlass.Int32,
    o_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64, cutlass.Int64],
    lse_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    amax_o_ptr: Optional[cute.Pointer],
    scale_o_ptr: Optional[cute.Pointer],
    has_scale_o: cutlass.Constexpr[bool],
    stats_log2: cutlass.Constexpr[bool],
    stream: _cuda_driver.CUstream = None,
) -> None:
    """Reduce FP8-input partials, then scale/cast and report the recombined Amax.

    Quantized outputs keep unscaled partials and apply scale_o here. Half outputs
    retain their existing scaled partials; normalize their requested Amax after
    reduction. The preceding split-attention kernel initializes Amax, ordered
    before this reduction on the same stream. MXFP8 has no scalar output
    scale: its None-specialized pointer also removes Amax unscaling.
    """
    o_partial, lse_partial, o_out, lse_out = _ptr_operands(
        o_partial_ptr, lse_partial_ptr, o_out_ptr, lse_out_ptr, problem_size, n_splits, o_strides, lse_strides
    )
    amax_o = None
    if cutlass.const_expr(amax_o_ptr is not None):
        amax_o = cute.make_tensor(amax_o_ptr, cute.make_layout((1,), stride=(1,)))
    scale_o = None
    if cutlass.const_expr(has_scale_o):
        scale_o = cute.make_tensor(scale_o_ptr, cute.make_layout((1,), stride=(1,)))
    _launch_combine(o_partial, lse_partial, o_out, lse_out, amax_o, scale_o, problem_size, n_splits, stats_log2, None, None, None, (1, 1, 1), (0, 0), stream)
    if cutlass.const_expr(amax_o_ptr is not None and not has_scale_o and scale_o_ptr is not None):
        _unscale_amax_kernel(amax_o_ptr, scale_o_ptr).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)


@cute.jit
def _host_ptr_ragged(
    o_partial_ptr: cute.Pointer,
    lse_partial_ptr: cute.Pointer,
    o_out_ptr: cute.Pointer,
    lse_out_ptr: Optional[cute.Pointer],
    problem_size: Tuple[int, int, int, int],
    n_splits: cutlass.Int32,
    o_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64, cutlass.Int64],
    lse_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    ragged_q_ptr: cute.Pointer,
    ragged_o_ptr: cute.Pointer,
    ragged_lse_ptr: Optional[cute.Pointer],
    ragged_divs: Tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
    ragged_caps: Tuple[cutlass.Int32, cutlass.Int32],
    stats_log2: cutlass.Constexpr[bool],
    stream: _cuda_driver.CUstream = None,
) -> None:
    """The ragged-Q decode leg's pointer ABI: :func:`_host_ptr` plus the (B+1,)
    ragged offsets of Q / O / Stats (int32 or int64 elements), their
    elements-per-token divisors and the (O, Stats) packed token capacities; the
    final O / Stats are the packed outputs addressed at ``offset[b] / div +
    q_row`` with batch coord 0 (``o_strides[0]`` / ``lse_strides[0]`` are never
    stepped), bounded by the capacities."""
    o_partial, lse_partial, o_out, lse_out = _ptr_operands(
        o_partial_ptr, lse_partial_ptr, o_out_ptr, lse_out_ptr, problem_size, n_splits, o_strides, lse_strides
    )
    B = problem_size[0]
    n_off = cutlass.Int32(B) + cutlass.Int32(1)
    ragged_q = cute.make_tensor(ragged_q_ptr, cute.make_layout((n_off,), stride=(1,)))
    ragged_o = cute.make_tensor(ragged_o_ptr, cute.make_layout((n_off,), stride=(1,)))
    ragged_lse = None
    if cutlass.const_expr(ragged_lse_ptr is not None):
        ragged_lse = cute.make_tensor(ragged_lse_ptr, cute.make_layout((n_off,), stride=(1,)))
    _launch_combine(
        o_partial, lse_partial, o_out, lse_out, None, None, problem_size, n_splits, stats_log2, ragged_q, ragged_o, ragged_lse, ragged_divs, ragged_caps, stream
    )


@cute.jit
def _ptr_operands(o_partial_ptr, lse_partial_ptr, o_out_ptr, lse_out_ptr, problem_size, n_splits, o_strides, lse_strides):
    """The compact split-major partial slabs and the strided final O / Stats views of a pointer entry."""
    B, H, SQ, D = problem_size
    b64, h64, sq64, d64 = cutlass.Int64(B), cutlass.Int64(H), cutlass.Int64(SQ), cutlass.Int64(D)
    split_batches = b64 * cutlass.Int64(n_splits)
    o_partial = cute.make_tensor(o_partial_ptr, cute.make_layout((split_batches, SQ, H, D), stride=(sq64 * h64 * d64, h64 * d64, d64, 1)))
    lse_partial = cute.make_tensor(lse_partial_ptr, cute.make_layout((split_batches, H, SQ), stride=(h64 * sq64, sq64, 1)))
    o_out = cute.make_tensor(o_out_ptr, cute.make_layout((B, SQ, H, D), stride=o_strides))
    lse_out = None
    if cutlass.const_expr(lse_out_ptr is not None):
        lse_out = cute.make_tensor(lse_out_ptr, cute.make_layout((B, H, SQ), stride=lse_strides))
    return o_partial, lse_partial, o_out, lse_out


@lru_cache(maxsize=None)
def compile_ptr(
    dtype_o: str = "f16",
    dtype_partial: str = "f32",
    has_lse: bool = False,
    stats_log2: bool = False,
    ragged: bool = False,
    ragged_i64: bool = False,
    quantized: bool = False,
    has_amax: bool = False,
    has_scale_o: bool = False,
    has_scale_o_input: bool = True,
) -> Callable:
    """Compile a shape-generic pointer entry for prepared split execution.

    The runtime arguments are partial-O/partial-LSE/O/optional-LSE pointers,
    ``(B, H, S_q, D_v)``, split count, O strides in BSHD order, LSE strides
    in BHS order, the three ragged-offset pointers (``ragged``: (B+1,) Q / O /
    Stats offsets, int32 or -- ``ragged_i64`` -- int64; None-specialized off
    otherwise) and their elements-per-token divisors. Every stride is Int64,
    including singleton dimensions. All pointer entries share the reduction
    and launch. The quantized dense entry appends Amax/scale
    pointers; half and ragged entries keep their existing positional ABI.
    ``has_scale_o_input=False`` removes the scalar input and Amax unscale
    launch for MXFP8. Per-tensor FP8 retains both by default.
    """
    if dtype_o not in ("f16", "bf16", "e4m3", "e5m2") or dtype_partial not in ("f16", "bf16", "f32"):
        raise ValueError("prepared split combine requires half/FP8 O and f16/bf16/f32 partials")
    if (has_amax or has_scale_o or dtype_o in ("e4m3", "e5m2")) and not quantized:
        raise ValueError("FP8 outputs and scalar operands require the quantized pointer entry")
    if has_scale_o and not has_scale_o_input:
        raise ValueError("scaled combine requires a scale_o input")
    if quantized and ragged:
        raise ValueError("the quantized pointer entry serves dense split launches")
    if ragged_i64 and not ragged:
        raise ValueError("ragged_i64 is a ragged specialization")
    _cache_key = _template_key(globals(), locals(), "compile_ptr")
    gmem = cute.AddressSpace.gmem

    def P(dtype):
        return cute.runtime.make_ptr(dtype, 16, gmem, assumed_align=dtype.width // 8)

    common = (
        P(_ELEM[dtype_partial]),
        P(cutlass.Float32),
        P(_ELEM[dtype_o]),
        P(cutlass.Float32) if has_lse else None,
        (0, 0, 0, 0),
        cutlass.Int32(0),
        (cutlass.Int64(0),) * 4,
        (cutlass.Int64(0),) * 3,
    )
    if quantized:
        entry, extra = _host_ptr_quantized, (P(cutlass.Float32) if has_amax else None, P(cutlass.Float32) if has_scale_o_input else None, bool(has_scale_o))
    elif ragged:
        # The ragged-Q leg's entry appends the offsets / divisors / capacities; the
        # dense entry's positional ABI stays exactly what its callers pass.
        off_t = cutlass.Int64 if ragged_i64 else cutlass.Int32
        entry, extra = _host_ptr_ragged, (P(off_t), P(off_t), P(off_t) if has_lse else None, (cutlass.Int32(1),) * 3, (cutlass.Int32(0),) * 2)
    else:
        entry, extra = _host_ptr, ()
    return _compile_cached(
        entry,
        *common,
        *extra,
        bool(stats_log2),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
        cache_key=_cache_key,
        symbol="frost_sdpa_fwd_combine_ptr",
    )
