# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pointer hosts for the SM107 (Rubin) d=256 backward chains -- the half row and the per-tensor FP8 row.

One ``@cute.jit`` host per row runs the WHOLE chain from device pointers and the caller's workspace: the padding
copies, the ``seq_kv`` fill and the dS zero-fill (kernels of this artifact, not torch ops), ``dot`` (delta), the
per-chunk main kernel + the two stage-3 GEMMs, and the GQA fold (half) / the fold + FP8 epilogue (fp8: dV always, dK
under GQA; with the e4m3 dS workspace the GEMMs' own epilogue descales -- and quantizes dQ / MHA dK -- so no Q / K
upcast and no dQ / dK fold pass runs; the bf16-dS twin keeps the upcasts and all three fold passes).  Every tensor
the chain touches is a view built here from a pointer + a plan-time geometry (``_view``) or from a workspace region
(``_scratch``); nothing is allocated, nothing synchronizes, so the compiled artifact rebinds per call, follows the
handle's stream and captures into a CUDA graph.  ``prepared_sm107.compile_plan`` builds the geometry / regions and
the ``BwdLaunchSpec`` that ``engines.lower_dsl_bwd*`` hands the graph plan.

The stage-3 operand orders follow the kv-major ``[B, H, S_kv, S_q]`` workspace (see ``api_dsl_sm107``): dK reads dS
as ``A[kv, q]`` (K-major) against ``Q[D, q]``, dQ reads ``dS^T[q, kv]`` (M-major) against ``K[D, kv]``.  Under GQA the K
head is shared by ``group`` Q heads: the shipped dQ rendering indexes its B by ``h // group`` itself
(``MatmulTemplateParams.b_head_group = group``), so ONE launch covers the whole chunk; the per-head rendering
(``b_head_group = 1``, the ``api_dsl_sm107.DQ_SINGLE_LAUNCH = False`` twin) runs once per group member over every
``group``-th Q head so each launch's operands line up with the KV heads (``_stage3``).
"""

from typing import Optional

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver

from cudnn.frost.compiled_cache import compile_cached
from cudnn.frost.tile_dsl.tma import st_global_v4
from cudnn.sdpa.bwd.kernels.bprop_chain_common import DOT_CHUNK_ELEMS, DOT_Q_TILE, dkv_reduce_host, dot_do_o_host, dot_do_o_scaled_host, fold_quant_host
from cudnn.sdpa.bwd.kernels.sm120.prepared_host import _scratch, _view

_THREADS = 256
_D = 256  # d_qk = d_v of both rows (the bodies hardcode the tile)

# Workspace region slots, in the FIXED order ``prepared_sm107.compile_plan`` emits them (None = not carved for this plan).
# Half row: 0..12; fp8 row: 0..8 then its own 9..14.  Names match ``api_dsl_sm107._scratch_plan``.
R_DELTA, R_DS, R_SEQ_KV, R_DESC, R_Q_PAD, R_DO_PAD, R_LSE_PAD, R_K_PAD, R_V_PAD = range(9)
R_DV_PART, R_DK_PART, R_DK_FOLD, R_DV_FOLD = 9, 10, 11, 12  # half row
R_FP8_DV_PART, R_FP8_DK_PART, R_DQ_WS, R_Q_BF16, R_K_BF16, R_AMAX_SCRATCH = 9, 10, 11, 12, 13, 14  # fp8 row
N_REGIONS_F16 = 13
N_REGIONS_FP8 = 15
# amax scratch slots (fp32 [8]): 0..3 = the four amax outputs the graph left virtual, 4 = the main kernel's own dV amax
# (recomputed by the fold pass over the folded value; the kernel's copy is never read).
AMAX_SLOT_DP, AMAX_SLOT_DV_KERNEL = 3, 4


# --- small chain kernels ------------------------------------------------------------------------------------------------


@cute.kernel
def _zero_bytes(t: cute.Tensor, n16: cutlass.Constexpr[int]):
    """Zero a 128-B-aligned region whose byte size is a multiple of 16 (one 128-bit store per 16 B)."""
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    blocks, _, _ = cute.arch.grid_dim()
    i = cutlass.Int64(bid) * _THREADS + tid
    addr = t.iterator.toint()
    while i < n16:
        zeros = [cutlass.Int32(0)] * 4
        st_global_v4(addr + i * 16, zeros, cutlass.Int32)
        i += cutlass.Int64(blocks) * _THREADS


_zero_bytes.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def _pad_rows(src: cute.Tensor, dst: cute.Tensor):
    """``dst[b, s, h, :] = src[b, s, h, :]`` for ``s < S_src``, zeros for the rows past it; ``dst``'s extent bounds the walk.

    Both are the ``_words`` views of COMPACT ``[B, S, H, D]`` tensors of one dtype (the rows' BSHD-physical claim): 32-bit
    words, the row ``D * itemsize / 4`` wide, so the copy is dtype-agnostic and moves 16 bytes per thread step.  Serves the
    zero-padded Q / dO / K / V staging (``S_dst > S_src``) AND the real-row copy-out of a padded partial (``S_dst < S_src``:
    every row copied).
    """
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    blocks, _, _ = cute.arch.grid_dim()
    B, S_dst, H, row_words = dst.shape
    S_src = src.shape[1]
    total = B * S_dst * H * row_words // 4  # 16-byte steps over dst
    src_ptr = src.iterator.raw_ptr()
    dst_ptr = dst.iterator.raw_ptr()
    zeros = cutlass.Vector.from_elements((cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0)), cutlass.Int32)
    i = cutlass.Int64(bid) * _THREADS + tid
    while i < total:
        pos = i * 4  # dst offset in 32-bit words
        col = pos % row_words
        row = pos // row_words  # (b * S_dst + s) * H + h
        h = row % H
        bs = row // H
        s = bs % S_dst
        b = bs // S_dst
        if s < S_src:
            v = (src_ptr + ((b * S_src + s) * H + h) * row_words + col).load(count=4)
            (dst_ptr + pos).store(v, alignment=16)
        else:
            (dst_ptr + pos).store(zeros, alignment=16)
        i += cutlass.Int64(blocks) * _THREADS


_pad_rows.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def _pad_lse(src: cute.Tensor, dst: cute.Tensor):
    """``dst[r, s] = src[r, s]`` for ``s < S``, ``+inf`` past it: a q row past the real length gets ``P = exp2(S - inf) = 0``.

    ``src`` is the contiguous ``[B, H, S]`` fp32 Stats, ``dst`` the ``[B, H, S_pad]`` staging copy."""
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    blocks, _, _ = cute.arch.grid_dim()
    rows = dst.shape[0] * dst.shape[1]
    S, S_pad = src.shape[2], dst.shape[2]
    src_ptr = src.iterator.raw_ptr()
    dst_ptr = dst.iterator.raw_ptr()
    inf = cutlass.Float32(float("inf"))
    i = cutlass.Int64(bid) * _THREADS + tid
    while i < rows * S_pad:
        r = i // S_pad
        s = i % S_pad
        if s < S:
            (dst_ptr + i).store((src_ptr + r * S + s).load())
        else:
            (dst_ptr + i).store(inf)
        i += cutlass.Int64(blocks) * _THREADS


_pad_lse.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def _fill_i32(dst: cute.Tensor, value: cutlass.Constexpr[int]):
    """``dst[:] = value`` over an int32 vector of ANY length (the per-batch kv lengths the padded mask arm reads): one block,
    every thread striding by ``_THREADS``.  A bare ``tid < size`` guard covers 256 entries only -- a ragged-S_kv graph with
    B > 256 then reads an UNWRITTEN ``seq_kv_lens[b]`` for every batch past the block (workspace residue; a 0 there is a dead
    batch whose dQ / dK / dV come back as exact zeros -- Codex review on #1212, B = 257 / S_kv = 129)."""
    tid, _, _ = cute.arch.thread_idx()
    i = tid
    while i < cute.size(dst):
        dst.iterator[i] = cutlass.Int32(value)
        i += _THREADS


_fill_i32.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def _zero_amax(
    amax_dq: Optional[cute.Tensor],
    amax_dk: Optional[cute.Tensor],
    amax_dv: Optional[cute.Tensor],
    amax_dp: cute.Tensor,
    amax_dv_kernel: cute.Tensor,
):
    """Reset the amax accumulators (the C++ node's semantics: an amax output is RESET on every execute, then atomicMax'd)."""
    tid, _, _ = cute.arch.thread_idx()
    if cutlass.const_expr(amax_dq is not None):
        if tid == 0:
            amax_dq.iterator[0] = cutlass.Float32(0)
    if cutlass.const_expr(amax_dk is not None):
        if tid == 1:
            amax_dk.iterator[0] = cutlass.Float32(0)
    if cutlass.const_expr(amax_dv is not None):
        if tid == 2:
            amax_dv.iterator[0] = cutlass.Float32(0)
    if tid == 3:
        amax_dp.iterator[0] = cutlass.Float32(0)
    if tid == 4:
        amax_dv_kernel.iterator[0] = cutlass.Float32(0)


_zero_amax.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def _cast_fp8_to_bf16(src: cute.Tensor, dst: cute.Tensor, n: cutlass.Constexpr[int]):
    """``dst = bf16(src)`` over ``n`` compact e4m3 elements, EXACT (3 mantissa bits into 7, the exponent range covered): the bf16
    B operand of the stage-3 GEMMs.  Eight elements per thread: an 8-byte fp8 load, a 16-byte bf16 store."""
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    blocks, _, _ = cute.arch.grid_dim()
    src_ptr = src.iterator.raw_ptr()
    dst_ptr = dst.iterator.raw_ptr()
    i = cutlass.Int64(bid) * _THREADS + tid
    while i < n // 8:
        pos = i * 8
        v = (src_ptr + pos).load(count=8)
        out = cutlass.Vector.from_elements(tuple(v[e].to(cutlass.Float32).to(cutlass.BFloat16) for e in range(8)), cutlass.BFloat16)
        (dst_ptr + pos).store(out, alignment=16)
        i += cutlass.Int64(blocks) * _THREADS


_cast_fp8_to_bf16.set_name_prefix("cudnn", remove_cutlass_symbol=True)


# --- view helpers ---------------------------------------------------------------------------------------------------------


@cute.jit
def _window(tensor: cute.Tensor, dim: cutlass.Constexpr, begin, count: cutlass.Constexpr, step: cutlass.Constexpr = 1):
    """``tensor`` narrowed along ``dim`` to ``count`` entries starting at ``begin``, every ``step``-th (a view: ``t[..., begin::step, ...]``)."""
    shape = tuple(count if i == dim else tensor.shape[i] for i in range(4))
    strides = tuple(tensor.stride[i] * step if i == dim else tensor.stride[i] for i in range(4))
    return cute.make_tensor(tensor.iterator + cutlass.Int64(begin) * tensor.stride[dim], cute.make_layout(shape, stride=strides))


@cute.jit
def _extent(tensor: cute.Tensor, shape: cutlass.Constexpr):
    """The leading ``shape`` of ``tensor`` under its own strides (a real-extent view of a padded buffer)."""
    return cute.make_tensor(tensor.iterator, cute.make_layout(shape, stride=tensor.stride))


@cute.jit
def _permuted(tensor: cute.Tensor, order: cutlass.Constexpr):
    return cute.make_tensor(tensor.iterator, cute.make_layout(tuple(tensor.shape[i] for i in order), stride=tuple(tensor.stride[i] for i in order)))


@cute.jit
def _matmul(entry: cutlass.Constexpr, a, b, output, heads: cutlass.Constexpr, batches: cutlass.Constexpr, meta, desc, stream, epi: cutlass.Constexpr = None):
    """One ``(batch, head)``-batched stage-3 GEMM: ``a`` / ``b`` / ``output`` already in the template's ``(M|N, K, H, B)`` / ``(M, N, H, B)``
    order.  The problem tuple is the retired ``matmul_bh``'s, widened to Int64 before the descriptor's byte products; the grid's M
    is A's M (dense: the operands' shared extent).  ``epi`` = the fp8 arm's ``(descale_0, descale_1, scale_out, amax)`` fp32 [1]
    tensors (``scale_out`` / ``amax`` None outside EPI_QUANT / for an unrequested amax); None = a rendering without an epilogue,
    called with the SM100 chain's positional seven arguments."""
    problem = tuple(
        cutlass.Int64(x)
        for x in (a.shape[0], b.shape[0], a.shape[1], heads, batches, *a.stride, *b.stride, *output.stride, b.shape[1], a.shape[0], output.shape[0])
    )
    if cutlass.const_expr(epi is None):
        entry(problem, a, b, output, meta, desc, stream)
    else:
        entry(problem, a, b, output, meta, desc, stream, epi[0], epi[1], epi[2], epi[3])


@cute.jit
def _words(tensor: cute.Tensor, itemsize: cutlass.Constexpr[int]):
    """A compact ``[B, S, H, D]`` tensor as 32-bit words: ``[B, S, H, D * itemsize / 4]``, the dtype-agnostic copy view."""
    b, s, h, d = tensor.shape
    w = d * itemsize // 4
    ptr = cute.make_ptr(cutlass.Int32, tensor.iterator.toint(), cute.AddressSpace.gmem, assumed_align=16)
    return cute.make_tensor(ptr, cute.make_layout((b, s, h, w), stride=(s * h * w, h * w, w, 1)))


@cute.jit
def _pad_copy(src: cute.Tensor, dst: cute.Tensor, itemsize: cutlass.Constexpr[int], stream):
    """Launch ``_pad_rows`` over the word views of two compact ``[B, S, H, D]`` tensors (rows past ``src``'s extent read as zero)."""
    d = _words(dst, itemsize)
    _pad_rows(_words(src, itemsize), d).launch(grid=(min((cute.size(d) // 4 + _THREADS - 1) // _THREADS, 4096), 1, 1), block=(_THREADS, 1, 1), stream=stream)


@cute.jit
def _slot(scratch: cute.Tensor, index: cutlass.Constexpr[int]):
    """One fp32 element of the amax scratch as a ``[1]`` tensor (the shape the kernels' amax operands take)."""
    return cute.make_tensor(scratch.iterator + index, cute.make_layout((1,), stride=(1,)))


@cute.jit
def _grid_16b(t: cute.Tensor, itemsize: cutlass.Constexpr[int]):
    return (min((cute.size(t) * itemsize // 16 + _THREADS - 1) // _THREADS, 4096), 1, 1)


@cute.jit
def _stage2_inputs(
    q, k, v, do, stats, workspace, regions: cutlass.Constexpr, config: cutlass.Constexpr, dtype: cutlass.Constexpr, ds_dtype: cutlass.Constexpr, stream
):
    """The kernel-facing Q / dO / K / V / LSE (zero- / +inf-padded staging copies when the graph's extents are not tile multiples),
    the per-batch kv lengths and the (zero-filled under a mask) dS workspace -- as launches of this artifact."""
    b, h, hk, d, sq, skv, sqp, skvp, bc, hc, zero_ws, itemsize, bpe_ds, dq_bhg = config
    q_k, do_k, lse_k, k_k, v_k = q, do, stats, k, v
    if cutlass.const_expr(regions[R_Q_PAD] is not None):
        q_k = _scratch(workspace, regions[R_Q_PAD], dtype)
        do_k = _scratch(workspace, regions[R_DO_PAD], dtype)
        lse_k = _scratch(workspace, regions[R_LSE_PAD], cutlass.Float32)
        _pad_copy(q, q_k, itemsize, stream)
        _pad_copy(do, do_k, itemsize, stream)
        _pad_lse(stats, lse_k).launch(grid=(min((b * h * sqp + _THREADS - 1) // _THREADS, 4096), 1, 1), block=(_THREADS, 1, 1), stream=stream)
    if cutlass.const_expr(regions[R_K_PAD] is not None):
        k_k = _scratch(workspace, regions[R_K_PAD], dtype)
        v_k = _scratch(workspace, regions[R_V_PAD], dtype)
        _pad_copy(k, k_k, itemsize, stream)
        _pad_copy(v, v_k, itemsize, stream)
    # Read by the padded mask arm only (the uniform real kv length, one entry per batch); carved and written regardless (fixed
    # kernel ABI).  ONE block: the fill strides over all B entries itself, so the launch needs no B-derived grid (B is at most a
    # few hundred here; a second block would cost more than the loop it saves).
    seq_kv = _scratch(workspace, regions[R_SEQ_KV], cutlass.Int32)
    _fill_i32(seq_kv, skv).launch(grid=(1, 1, 1), block=(_THREADS, 1, 1), stream=stream)
    # Zero ONCE, ahead of every chunk, and ONLY when the adapter says so (`api_dsl_sm107._stage3_needs_zero_fill`): with the
    # two-sided K-trim the stage-3 GEMMs read only the tiles the main kernel wrote (it rounds every kv block's q range
    # outward to the GEMMs' 256-row pair), so no mask needs it -- except the untrimmed twin (every tile read) and a top-left
    # window with S_q > roundup(S_kv + W, 256), where the q pairs past the last kv block's window are written by nobody.
    # The skipped set is the same for every chunk.
    ds_full = _scratch(workspace, regions[R_DS], ds_dtype)
    if cutlass.const_expr(zero_ws):
        n16 = bc * hc * skvp * sqp * bpe_ds // 16
        _zero_bytes(ds_full, n16).launch(grid=(min((n16 + _THREADS - 1) // _THREADS, 4096), 1, 1), block=(_THREADS, 1, 1), stream=stream)
    return q_k, do_k, lse_k, k_k, v_k, seq_kv, ds_full


@cute.jit
def _stage3(
    mm_dk: cutlass.Constexpr,
    mm_dq: cutlass.Constexpr,
    ds,
    q,
    k,
    dk_out,
    dq_out,
    bb,
    bc: cutlass.Constexpr,
    hb,
    hc: cutlass.Constexpr,
    group: cutlass.Constexpr,
    meta,
    desc,
    stream,
    dk_epi: cutlass.Constexpr = None,
    dq_epi: cutlass.Constexpr = None,
    dq_b_head_group: cutlass.Constexpr = 1,
):
    """dK = dS . Q into ``dk_out[bb:bb+bc, :, hb:hb+hc]`` and dQ = dS^T . K into ``dq_out[bb:bb+bc, :, hb:hb+hc]`` for one (batch, head)
    chunk; ``ds`` is the chunk's REAL-extent ``[bc, hc, S_kv, S_q]`` workspace view, ``q`` / ``k`` the full ``[B, S, H, D]`` operands,
    ``dk_out`` a ``[B, S_kv, H_q, D]`` real-row view.  ``dk_epi`` / ``dq_epi`` are the fp8 arm's epilogue operands (``_matmul``);
    ``dq_b_head_group`` is the dQ rendering's ``MatmulTemplateParams.b_head_group`` (how it indexes its B = K head: ``h // group``
    at the GQA group, per head at 1), which decides the launch count (``_dq_launches``).  Every operand is a view; nothing is copied."""
    q_c = _window(_window(q, 0, bb, bc), 2, hb, hc)  # [bc, S_q, hc, D]
    dk_c = _window(_window(dk_out, 0, bb, bc), 2, hb, hc)  # [bc, S_kv, hc, D]
    # dK = dS . Q: A = dS[kv, q] (M, K, H, B) K-major; B = Q (D, q, H, B); out (kv, D, H, B).
    _matmul(mm_dk, _permuted(ds, (2, 3, 1, 0)), _permuted(q_c, (3, 1, 2, 0)), _permuted(dk_c, (1, 3, 2, 0)), hc, bc, meta, desc, stream, dk_epi)
    # dQ = dS^T . K: A = dS^T[q, kv] (M, K, H, B) M-major; B = K (D, kv, H_kv, B); out (q, D, H, B).  Under GQA the K head is
    # shared by `group` Q heads.  The shipped rendering (`dq_b_head_group == group`) indexes B by `h // group` itself, so ONE
    # launch covers the chunk's `hc` Q heads (A = the whole dS chunk, out = the whole dQ chunk, B = the chunk's kv_n K heads:
    # kv_n x (S_q / 256) x group clusters instead of kv_n x (S_q / 256), `group` times).  The per-head rendering
    # (`dq_b_head_group == 1`, the `DQ_SINGLE_LAUNCH = False` twin) runs once per group MEMBER over every `group`-th Q head, so
    # each launch's A / out heads line up with its B heads.  Both walk the same k tiles per output tile: bitwise-equal dQ.
    kv_n = hc // group
    n_launch = _dq_launches(group, dq_b_head_group)
    heads = hc // n_launch  # Q heads per dQ launch: kv_n * dq_b_head_group
    k_c = _window(_window(k, 0, bb, bc), 2, hb // group, kv_n)  # [bc, S_kv, kv_n, D]
    for member in range(n_launch):
        a_g = _window(ds, 1, member, heads, n_launch)  # ds[:, member::n_launch] -> [bc, heads, S_kv, S_q] (ds itself at one launch)
        o_g = _window(_window(dq_out, 0, bb, bc), 2, hb + member, heads, n_launch)  # dq[bs, :, hb+member::n_launch] -> [bc, S_q, heads, D]
        _matmul(mm_dq, _permuted(a_g, (3, 2, 1, 0)), _permuted(k_c, (3, 1, 2, 0)), _permuted(o_g, (1, 3, 2, 0)), heads, bc, meta, desc, stream, dq_epi)


def _dq_launches(group: int, dq_b_head_group: int) -> int:
    """How many dQ GEMM launches one chunk takes: ``group // dq_b_head_group`` -- 1 when the dQ rendering groups its B head by
    the GQA group (``MatmulTemplateParams.b_head_group == group``), ``group`` when B is batched per head (the per-member loop).
    Plain Python at trace time, so a rendering / host mismatch -- which would silently pair a Q head with the wrong K head --
    raises instead of launching."""
    if dq_b_head_group not in (1, group):
        raise ValueError(
            f"sm107 bwd d256 stage 3: the dQ rendering's b_head_group ({dq_b_head_group}) must be 1 (one launch per GQA group member) or the group "
            f"({group}, one launch per chunk); nothing in between pairs every Q head with its K head"
        )
    return group // dq_b_head_group


# --- the half row --------------------------------------------------------------------------------------------------------


@cute.jit
def host_f16(
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    v_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    do_ptr: cute.Pointer,
    stats_ptr: cute.Pointer,
    dq_ptr: cute.Pointer,
    dk_ptr: cute.Pointer,
    dv_ptr: cute.Pointer,
    workspace: cute.Pointer,
    scale: cutlass.Float32,
    main: cutlass.Constexpr,
    mm_dk: cutlass.Constexpr,
    mm_dq: cutlass.Constexpr,
    config: cutlass.Constexpr,
    geometry: cutlass.Constexpr,
    regions: cutlass.Constexpr,
    dtype: cutlass.Constexpr,
    stream: driver.CUstream,
):
    b, h, hk, d, sq, skv, sqp, skvp, bc, hc, zero_ws, itemsize, bpe_ds, dq_bhg = config
    q = _view(q_ptr, geometry[0])
    k = _view(k_ptr, geometry[1])
    v = _view(v_ptr, geometry[2])
    o = _view(o_ptr, geometry[3])
    do = _view(do_ptr, geometry[4])
    stats = _view(stats_ptr, geometry[5])  # [B, H, S_q] fp32
    dq = _view(dq_ptr, geometry[6])
    dk = _view(dk_ptr, geometry[7])
    dv = _view(dv_ptr, geometry[8])
    delta = _scratch(workspace, regions[R_DELTA], cutlass.Float32)  # [B, H, S_q_pad]
    desc = _scratch(workspace, regions[R_DESC], cutlass.Int64)  # the GEMMs' dead THD slot (dense: never read)
    q_k, do_k, lse_k, k_k, v_k, seq_kv, ds_full = _stage2_inputs(q, k, v, do, stats, workspace, regions, config, dtype, dtype, stream)
    group = h // hk
    kv_padded = regions[R_K_PAD] is not None
    # stage 2's dV per Q head: the caller's dV only when MHA and no kv padding; stage 3's dK per Q head: the caller's dK when MHA.
    dv_k = _scratch(workspace, regions[R_DV_PART], dtype) if cutlass.const_expr(regions[R_DV_PART] is not None) else dv
    dk_tgt = _scratch(workspace, regions[R_DK_PART], dtype) if cutlass.const_expr(regions[R_DK_PART] is not None) else dk
    dk_real = _extent(dk_tgt, (b, skv, h, d))  # stage 3 writes the real rows only
    ds = _extent(ds_full, (bc, hc, skv, sq))  # what stage 3 reads: the real extents (padded rows / cols never reach a GEMM)

    # STAGE 1, hoisted out of the chunk loop: one streaming pass over O and dO.
    dot_do_o_host(o, do, delta, None, None, DOT_Q_TILE, d, d, DOT_CHUNK_ELEMS, False, False, stream)

    for bi in range(b // bc):
        bb = bi * bc
        for ci in range(h // hc):
            hb = ci * hc
            # STAGE 2: head_base / batch_base offset every full-tensor read; dS stays chunk-local.
            main(q_k, do_k, k_k, v_k, dv_k, ds_full, lse_k, delta, seq_kv, (b, h, hk, sqp, skvp, hc, bc, sq, skv), scale, hb, bb, stream)
            # STAGE 3: consume the chunk's workspace, write the outputs' (batch, head) slice.
            _stage3(mm_dk, mm_dq, ds, q, k, dk_real, dq, bb, bc, hb, hc, group, seq_kv, desc, stream, dq_b_head_group=dq_bhg)

    # STAGE 4: fold the per-Q-head partials onto the KV heads (fixed order, deterministic); copy real rows out of a padded staging.
    if cutlass.const_expr(group > 1):
        dk_out = _scratch(workspace, regions[R_DK_FOLD], dtype) if cutlass.const_expr(kv_padded) else dk
        dv_out = _scratch(workspace, regions[R_DV_FOLD], dtype) if cutlass.const_expr(kv_padded) else dv
        dkv_reduce_host(dk_tgt, dv_k, dk_out, dv_out, d, d, group, dtype, False, stream)
        if cutlass.const_expr(kv_padded):
            _pad_copy(dk_out, dk, itemsize, stream)
            _pad_copy(dv_out, dv, itemsize, stream)
    elif cutlass.const_expr(kv_padded):
        _pad_copy(dv_k, dv, itemsize, stream)


# --- the per-tensor FP8 row ---------------------------------------------------------------------------------------------


@cute.jit
def host_fp8(
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    v_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    do_ptr: cute.Pointer,
    stats_ptr: cute.Pointer,
    dq_ptr: cute.Pointer,
    dk_ptr: cute.Pointer,
    dv_ptr: cute.Pointer,
    descale_q_ptr: cute.Pointer,
    descale_k_ptr: cute.Pointer,
    descale_v_ptr: cute.Pointer,
    descale_s_ptr: cute.Pointer,
    scale_s_ptr: cute.Pointer,
    descale_o_ptr: cute.Pointer,
    descale_do_ptr: cute.Pointer,
    descale_dp_ptr: cute.Pointer,
    scale_dq_ptr: cute.Pointer,
    scale_dk_ptr: cute.Pointer,
    scale_dv_ptr: cute.Pointer,
    scale_dp_ptr: cute.Pointer,
    amax_dq_ptr: Optional[cute.Pointer],
    amax_dk_ptr: Optional[cute.Pointer],
    amax_dv_ptr: Optional[cute.Pointer],
    amax_dp_ptr: Optional[cute.Pointer],
    workspace: cute.Pointer,
    scale_log2: cutlass.Float32,
    scale: cutlass.Float32,
    main: cutlass.Constexpr,
    mm_dk: cutlass.Constexpr,
    mm_dq: cutlass.Constexpr,
    config: cutlass.Constexpr,
    geometry: cutlass.Constexpr,
    regions: cutlass.Constexpr,
    dtype: cutlass.Constexpr,
    grad_dtype: cutlass.Constexpr,
    stream: driver.CUstream,
):
    b, h, hk, d, sq, skv, sqp, skvp, bc, hc, zero_ws, itemsize, bpe_ds, dq_bhg = config
    q = _view(q_ptr, geometry[0])
    k = _view(k_ptr, geometry[1])
    v = _view(v_ptr, geometry[2])
    o = _view(o_ptr, geometry[3])
    do = _view(do_ptr, geometry[4])
    stats = _view(stats_ptr, geometry[5])
    dq = _view(dq_ptr, geometry[6])
    dk = _view(dk_ptr, geometry[7])
    dv = _view(dv_ptr, geometry[8])
    scalar = ((1,), (1,))
    descale_q, descale_k, descale_v = _view(descale_q_ptr, scalar), _view(descale_k_ptr, scalar), _view(descale_v_ptr, scalar)
    descale_s, scale_s = _view(descale_s_ptr, scalar), _view(scale_s_ptr, scalar)
    descale_o, descale_do = _view(descale_o_ptr, scalar), _view(descale_do_ptr, scalar)
    descale_dp, scale_dp = _view(descale_dp_ptr, scalar), _view(scale_dp_ptr, scalar)
    scale_dq, scale_dk, scale_dv = _view(scale_dq_ptr, scalar), _view(scale_dk_ptr, scalar), _view(scale_dv_ptr, scalar)
    # The dS workspace dtype decides the chain shape (api_dsl_sm107.FP8_DS_DTYPE): e4m3 = dS_q = e4m3(dS * scale_dP) into the fp8
    # K64 GEMM arm, whose epilogue applies descale_dP * descale_{q|k} (and, for dQ / MHA dK, amax + scale + the gradient cast);
    # bf16 = the pre-quantized twin (bf16 GEMMs over exact e4m3 -> bf16 upcasts; descale_dP / scale_dP bound and never applied).
    ds_fp8 = bpe_ds == 1
    ds_dtype = cutlass.Float8E4M3FN if cutlass.const_expr(ds_fp8) else cutlass.BFloat16
    delta = _scratch(workspace, regions[R_DELTA], cutlass.Float32)
    desc = _scratch(workspace, regions[R_DESC], cutlass.Int64)
    q_k, do_k, lse_k, k_k, v_k, seq_kv, ds_full = _stage2_inputs(q, k, v, do, stats, workspace, regions, config, dtype, ds_dtype, stream)
    group = h // hk
    dv_part = _scratch(workspace, regions[R_FP8_DV_PART], cutlass.BFloat16)  # [B, kv_rows, H, D] stage 2's per-Q-head dV_true
    amax_scratch = _scratch(workspace, regions[R_AMAX_SCRATCH], cutlass.Float32)  # [8]
    # amax targets: the graph's outputs where requested (Operand), scratch otherwise (None pointer) -- all reset here, then
    # atomicMax'd.  The main kernel's dV amax is over the per-Q-head partials, so the fold pass recomputes it over the folded
    # value; the kernel's copy lands in scratch.  An unrequested amax_dQ / dK / dV folds its atomics out of the pass that
    # owns it (the GEMM epilogue for dQ and MHA dK on the e4m3 chain, the fold pass otherwise).
    amax_dq = _view(amax_dq_ptr, scalar)
    amax_dk = _view(amax_dk_ptr, scalar)
    amax_dv = _view(amax_dv_ptr, scalar)
    amax_dp = _view(amax_dp_ptr, scalar) if cutlass.const_expr(amax_dp_ptr is not None) else _slot(amax_scratch, AMAX_SLOT_DP)
    amax_dv_kernel = _slot(amax_scratch, AMAX_SLOT_DV_KERNEL)
    _zero_amax(amax_dq, amax_dk, amax_dv, amax_dp, amax_dv_kernel).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    # STAGE 1: delta in TRUE units = rowsum(dO8 * O8) * descale_o * descale_dO.
    dot_do_o_scaled_host(o, do, delta, descale_o, descale_do, DOT_Q_TILE, d, DOT_CHUNK_ELEMS, stream)

    ds = _extent(ds_full, (b, hc, skv, sq))
    if cutlass.const_expr(ds_fp8):
        # The GEMMs' B operands are the caller's e4m3 payloads; dQ lands in the caller's dQ (EPI_QUANT); dK in the caller's dK
        # at MHA (EPI_QUANT, real rows) or in the bf16 TRUE-unit per-Q-head partials under GQA (EPI_DESCALE, folded below).
        dk_tgt = _scratch(workspace, regions[R_FP8_DK_PART], cutlass.BFloat16) if cutlass.const_expr(group > 1) else dk
        dk_real = _extent(dk_tgt, (b, skv, h, d))
        dk_epi = (descale_dp, descale_q, None, None) if cutlass.const_expr(group > 1) else (descale_dp, descale_q, scale_dk, amax_dk)
        dq_epi = (descale_dp, descale_k, scale_dq, amax_dq)
        for ci in range(h // hc):
            hb = ci * hc
            # STAGE 2 (whole batch in-grid; head_base walks the chunks): dS_q = e4m3(dS * scale_dP).  The kernel's seqlen_kv is the
            # REAL length: the padded arm's mask bound and the amax row gate.
            main(
                q_k,
                do_k,
                k_k,
                v_k,
                dv_part,
                ds_full,
                lse_k,
                delta,
                descale_q,
                descale_k,
                descale_v,
                descale_do,
                descale_s,
                scale_s,
                scale_dv,
                scale_dp,
                amax_dv_kernel,
                amax_dp,
                (b, h, hk, sqp, skvp, hc),
                scale,
                scale_log2,
                hb,
                skv,
                stream,
            )
            # STAGE 3: the fp8 K64 arm over the e4m3 dS and the e4m3 Q / K payloads; the epilogue undoes scale_dP and the payload's scale.
            _stage3(mm_dk, mm_dq, ds, q, k, dk_real, dq, 0, b, hb, hc, group, seq_kv, desc, stream, dk_epi, dq_epi, dq_b_head_group=dq_bhg)
        # STAGE 4: dV always folds + quantizes here (the kernel publishes bf16 per-Q-head dV_true); dK only under GQA, where the
        # bf16 true-unit partials are summed in fixed order BEFORE the amax fold, the scale and the cast (the backend's order).
        fold_quant_host(dv_part, dv, None, scale_dv, amax_dv, d, group, grad_dtype, stream)
        if cutlass.const_expr(group > 1):
            fold_quant_host(dk_tgt, dk, None, scale_dk, amax_dk, d, group, grad_dtype, stream)
    else:
        dk_part = _scratch(workspace, regions[R_FP8_DK_PART], cutlass.BFloat16)  # [B, kv_rows, H, D] stage 3's per-Q-head dS . Q8 (descale_q pending)
        dq_ws = _scratch(workspace, regions[R_DQ_WS], cutlass.BFloat16)  # [B, S_q, H, D] stage 3's dS^T . K8 (descale_k pending)
        q_bf16 = _scratch(workspace, regions[R_Q_BF16], cutlass.BFloat16)  # [B, S_q, H, D]
        k_bf16 = _scratch(workspace, regions[R_K_BF16], cutlass.BFloat16)  # [B, S_kv, H_kv, D]
        # The bf16 GEMM operands: e4m3 -> bf16 is exact; coalesced, one pass each.
        _cast_fp8_to_bf16(q, q_bf16, b * sq * h * d).launch(grid=_grid_16b(q_bf16, 2), block=(_THREADS, 1, 1), stream=stream)
        _cast_fp8_to_bf16(k, k_bf16, b * skv * hk * d).launch(grid=_grid_16b(k_bf16, 2), block=(_THREADS, 1, 1), stream=stream)
        dk_real = _extent(dk_part, (b, skv, h, d))
        for ci in range(h // hc):
            hb = ci * hc
            # STAGE 2 (whole batch in-grid; head_base walks the chunks); the bf16 dS: scale_dP is read and never applied.
            main(
                q_k,
                do_k,
                k_k,
                v_k,
                dv_part,
                ds_full,
                lse_k,
                delta,
                descale_q,
                descale_k,
                descale_v,
                descale_do,
                descale_s,
                scale_s,
                scale_dv,
                scale_dp,
                amax_dv_kernel,
                amax_dp,
                (b, h, hk, sqp, skvp, hc),
                scale,
                scale_log2,
                hb,
                skv,
                stream,
            )
            # STAGE 3 at bf16 over the exact upcasts; the partials still carry descale_q / descale_k.
            _stage3(mm_dk, mm_dq, ds, q_bf16, k_bf16, dk_real, dq_ws, 0, b, hb, hc, group, seq_kv, desc, stream, dq_b_head_group=dq_bhg)
        # STAGE 4: fold (GQA) + the per-tensor FP8 epilogue (descale, amax, scale, cast) into the caller's gradients.
        fold_quant_host(dv_part, dv, None, scale_dv, amax_dv, d, group, grad_dtype, stream)
        fold_quant_host(dk_part, dk, descale_q, scale_dk, amax_dk, d, group, grad_dtype, stream)
        fold_quant_host(dq_ws, dq, descale_k, scale_dq, amax_dq, d, 1, grad_dtype, stream)


# --- compilation -----------------------------------------------------------------------------------------------------------


def _check_target(sm: int) -> None:
    # The bodies are Rubin-line kernels (327 KiB SMEM carveout, 576 TMEM columns, the K=64 dense-FP8 MMA form).
    if not 107 <= sm <= 119:
        raise ValueError(f"SM107 SDPA bwd d256 has codegen targets for the Rubin line (SM107-SM119); got SM{sm}")


def _ptr(t, align=16):
    return cute.runtime.make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=align)


def compile_host_f16(main, mm_dk, mm_dq, config, geometry, regions, dtype, sm, cache_key):
    """The half row's artifact: ``dtype`` is the io / gradient DSL type (bf16 or fp16)."""
    _check_target(sm)
    args = [_ptr(dtype) for _ in range(5)] + [_ptr(cutlass.Float32, 4)] + [_ptr(dtype) for _ in range(3)]
    # The persistent artifact wrapper accepts primitive constexpr tuples; a dataclass argument prevents export.
    return compile_cached(
        host_f16,
        *args,
        _ptr(cutlass.Uint8),
        cutlass.Float32(1),
        main,
        mm_dk,
        mm_dq,
        tuple(config),
        geometry,
        regions,
        dtype,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options=f"--enable-tvm-ffi --gpu-arch sm_{sm}a",
        cache_key=cache_key,
        symbol="frost_sdpa_bwd_sm107_prepared",
    )


def compile_host_fp8(main, mm_dk, mm_dq, config, geometry, regions, grad_dtype, amax_requested, sm, cache_key):
    """The fp8 row's artifact: e4m3 payloads, fp32 scalars, gradients in ``grad_dtype`` (e4m3 / bf16 / fp16); ``amax_requested`` is
    the 4-tuple of bools (dQ, dK, dV, dP) selecting which amax pointers the artifact binds (None-specialized otherwise)."""
    _check_target(sm)
    fp8 = cutlass.Float8E4M3FN
    args = [_ptr(fp8) for _ in range(5)] + [_ptr(cutlass.Float32, 4)] + [_ptr(grad_dtype) for _ in range(3)]
    args += [_ptr(cutlass.Float32, 4) for _ in range(12)]
    args += [_ptr(cutlass.Float32, 4) if requested else None for requested in amax_requested]
    return compile_cached(
        host_fp8,
        *args,
        _ptr(cutlass.Uint8),
        cutlass.Float32(1),
        cutlass.Float32(1),
        main,
        mm_dk,
        mm_dq,
        tuple(config),
        geometry,
        regions,
        fp8,
        grad_dtype,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options=f"--enable-tvm-ffi --gpu-arch sm_{sm}a",
        cache_key=cache_key,
        symbol="frost_sdpa_bwd_sm107_fp8_prepared",
    )
