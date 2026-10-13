# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pointer hosts for the SM107 (Rubin) d=256 backward chains -- the half row, the per-tensor FP8 row and the MXFP8 row.

One ``@cute.jit`` host per row runs the WHOLE chain from device pointers and the caller's workspace: the padding
copies, the ``seq_kv`` fill (or, on the half row, the caller's per-batch kv lengths bound in its place -- the
``seq_kv_lens`` operand of a plan built with ``seq_kv_lens_present``) and the dS zero-fill (kernels of this artifact, not
torch ops), ``dot`` (delta), the
per-chunk main kernel + the two stage-3 GEMMs, and the GQA fold (half) / the fold + FP8 epilogue (fp8: dV always, dK
under GQA; with the e4m3 dS workspace the GEMMs' own epilogue descales -- and quantizes dQ / MHA dK -- so no Q / K
upcast and no dQ / dK fold pass runs; the bf16-dS twin keeps the upcasts and all three fold passes).  The MXFP8 row
(``host_mxfp8``) adds the zero-filled scale-factor pad staging (``_pad_sf_atoms``) ahead of the main kernel and, per its
dS policy, either the SF-aware ``e4m3 x 2^(e-127) -> bf16`` dequant of the columnwise q_T / k_T ahead of the bf16
stage-3 GEMMs (P-c, bf16 dS) or the block-scale GEMM arm over the kernel's two 1x32-scaled e4m3 dS payloads + E8M0
atoms and the columnwise q_T / k_T with their own scale factors (P-b, ``_stage3_block_scale``; no dequant pass).  Every dense
host takes two OPTIONAL appended pointers -- its role list's last two slots (half row 9 / 10, fp8 row 25 / 26, MXFP8 row
20 / 21) -- independent plan facts a plan may bind both, either or neither of: the caller's per-batch kv lengths above (the
padded mask arm reads ``seq_kv_lens[b]`` in place of the uniform fill; on the rows without batch chunking the dS zero-fill a
bottom-right band then needs runs ONCE ahead of the head loop), and the caller's ``delta`` (``external_delta=True``: a producer
that already reads O and dO -- the gated block's sigmoid-gate backward -- writes ``rowsum(dO * O)`` in ``dot_do_o``'s own
order); with it bound the ``dot`` launch and the workspace's ``delta`` region do not exist.  On the fp8 row the delta is read in
TRUE units, UNSCALED -- a caller's delta binds as is, nobody applies ``descale_o * descale_dO`` to it, so it is not bitwise the
row's own scaled pre-pass; on the MXFP8 row it IS bitwise the chain's own ``dot`` over the ``o_f16`` / ``dO_f16`` ports.  The
THD chains are SIBLING hosts (``host_f16_thd``, ``host_fp8_thd``, ``host_mxfp8_thd``) with their own ABI (two length operands at
slots 9 / 10, the same optional appended ``delta`` LAST -- the PACKED head-major ``[1, H, ceil128(T_q)]``, viewed from the plan's
token capacity --, ``lens_form``; the MXFP8 one also the two per-call packed scale-factor tile counts) and cache keys.  Every tensor
the chain touches is a view built here from a pointer + a plan-time geometry (``_view``) or from a workspace region
(``_scratch``); nothing is allocated, nothing synchronizes, so the compiled artifact rebinds per call, follows the
handle's stream and captures into a CUDA graph.  ``prepared_sm107.compile_plan`` builds the geometry / regions and
the ``BwdLaunchSpec`` that ``engines.lower_dsl_bwd*`` hands the graph plan.

The stage-3 operand orders follow the kv-major ``[B, H, S_kv, S_q]`` workspace (see ``api_dsl_sm107``): dK reads dS
as ``A[kv, q]`` (K-major) against ``Q[D, q]``, dQ reads ``dS^T[q, kv]`` (M-major) against ``K[D, kv]``.  Under GQA the K
head is shared by ``group`` Q heads: the shipped dQ rendering indexes its B by ``h // group`` itself
(``MatmulTemplateParams.b_head_group = group``), so ONE launch covers the whole chunk; the per-head rendering
(``b_head_group = 1``, the ``api_dsl_sm107.DQ_SINGLE_LAUNCH = False`` twin) runs once per group member over every
``group``-th Q head so each launch's operands line up with the KV heads (``_stage3``).  The block-scale hosts
(``_stage3_block_scale`` / ``_stage3_block_scale_thd``) take the same ``dq_b_head_group`` and the same loop: the template
indexes B AND its scale-factor descriptor by ``h // b_head_group`` (``_b_head``), so the single launch binds the whole dS /
atom / dQ chunk against the kv-head window of ``k_T`` and of its scale-factor planes view -- no per-member descriptor.
"""

from typing import Optional

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver

from cudnn.frost.compiled_cache import compile_cached
from cudnn.frost.tile_dsl.tma import st_global_v4
from cudnn.sdpa.bwd.config_sm100 import STAGE3_THD_SF_CU_K_OFF, STAGE3_THD_SF_CU_Q_OFF, stage3_thd_sfb_layout
from cudnn.sdpa.bwd.config_sm107 import DS_SF_P_B, DS_SF_P_C, DS_SF_POLICY_DEFAULT, MX_BLOCK, SF_ATOM_BYTES, SF_ATOM_ROWS
from cudnn.frost.tile_dsl.thd import THD_CU_K_TOTAL_OFF, THD_CU_Q_TOTAL_OFF
from cudnn.sdpa.bwd.kernels.bprop_chain_common import (
    DOT_CHUNK_ELEMS,
    DOT_Q_TILE,
    dkv_reduce_bounded_host,
    dkv_reduce_host,
    dot_do_o_host,
    dot_do_o_scaled_host,
    fold_quant_host,
    fold_quant_pair_host,
)
from cudnn.sdpa.bwd.kernels.sm120.prepared_host import _scratch, _view
from cudnn.sdpa.bwd.kernels import thd_helpers as _thd_helpers
from cudnn.sdpa.bwd.kernels.thd_helpers import thd_bwd_setup_host

_THREADS = 256
_D = 256  # d_qk = d_v of both rows (the bodies hardcode the tile)
# The half row's THD chain: the kv-blocked dS workspace's row granularity = the main kernel's 256-row kv block (its persistent
# unit; `config_sm107.CfgBwdD256.WS_BLOCK_ROWS`), which is also the setup launch's unit height.  Never re-literalled below.
_THD_KV_BLOCK = 256

# Workspace region slots, in the FIXED order ``prepared_sm107.compile_plan`` emits them (None = not carved for this plan).
# Half row: 0..12; fp8 row: 0..8 then its own 9..14.  Names match ``api_dsl_sm107._scratch_plan``.
R_DELTA, R_DS, R_SEQ_KV, R_DESC, R_Q_PAD, R_DO_PAD, R_LSE_PAD, R_K_PAD, R_V_PAD = range(9)
R_DV_PART, R_DK_PART, R_DK_FOLD, R_DV_FOLD = 9, 10, 11, 12  # half row
R_FP8_DV_PART, R_FP8_DK_PART, R_DQ_WS, R_Q_BF16, R_K_BF16, R_AMAX_SCRATCH = 9, 10, 11, 12, 13, 14  # fp8 row
# MXFP8 row (both dS chains): 0..8 shared, then the dO_T staging copy, the five zero-filled SF pad slabs (Q side: rows / groups past
# S_q_real; kv side: rows past S_kv_real up to the kernel's 256-row pad), the two dequantized bf16 stage-3 operands (the bf16-dS
# chain's; None under the block-scaled default), the partials.
R_DOT_PAD, R_SF_Q_PAD, R_SF_DO_PAD, R_SF_DOT_PAD, R_SF_K_PAD, R_SF_V_PAD, R_QT_BF16, R_KT_BF16 = 9, 10, 11, 12, 13, 14, 15, 16
R_MX_DV_PART, R_MX_DK_PART, R_MX_DK_FOLD, R_MX_DV_FOLD = 17, 18, 19, 20
# MXFP8 row, the block-scaled dS chain (P-b; appended): the second e4m3 dS payload (ds_dq, scaled per 32-kv block -- the first,
# ds_dk, is R_DS), the two F8_128x4 E8M0 atom tensors (sf_ds_dk [B, H_chunk, S_kv/128, S_q/128, 512], sf_ds_dq the transpose), and
# the columnwise q_T / k_T SF pad slabs the block-scale GEMMs' SFB reads when S_q / S_kv is ragged (None under P-c).
R_MX_DS_DQ, R_MX_SF_DS_DK, R_MX_SF_DS_DQ, R_MX_SF_QT_PAD, R_MX_SF_KT_PAD = 21, 22, 23, 24, 25
# MXFP8 row, the THD chain (appended): the per-sequence scale-factor TILE prefixes ``[cu_sf_q(B+1) | cu_sf_k(B+1)]``
# (``config_sm100.STAGE3_THD_SF_*``), a region of its own next to the shared THD metadata (None dense).  The THD chain's packed
# staging copies of the five SF tensors whose pad bytes are read reuse the ``sf_*_pad`` slots above at the plan's SF tile capacity.
R_MX_SF_META = 26
N_REGIONS_F16 = 13
N_REGIONS_FP8 = 15
N_REGIONS_MXFP8 = 27
# The F8_128x4 atom geometry is the config's (``config_sm107.SF_ATOM_BYTES`` = 512 B per 128-row x 4-group atom, ``SF_ATOM_ROWS``
# = 128, ``MX_BLOCK`` = 32), never re-literaled here.  Atoms per 128-row tile of one (b, h): the columnwise SF tensor has one atom per
# D-PLANE (``_D // SF_ATOM_ROWS``), the rowwise one per 4-group d-chunk (``(_D // MX_BLOCK) // 4``) -- the two coincide (both = D / 128)
# because an atom is 128 rows x 4 groups x 32 elements = 128 x 128 elements either way; ``_SF_ATOMS_PER_TILE`` is that count, derived.
# (No module-level assert -- engine-contract "no module-level asserts"; the equality (_D // MX_BLOCK) // 4 == _D // SF_ATOM_ROWS is
# pinned by test_sdpa_bwd_mxfp8_sm107.py::test_sf_pad_staging_geometry_derives_from_the_config.)
_SF_ATOMS_PER_TILE = _D // SF_ATOM_ROWS  # 2 at d = 256
# The MXFP8 row's zero-filled SF pad staging (``_pad_sf_atoms``).  True = what ships.  False = the RED half of the
# poisoned-SF-pad tests: the kernel reads the producer's undefined pad bytes as they are, so a
# 0xFF (E8M0 NaN) there lands NaN in dV / dS.  A module constant read at plan build (part of the artifact's cache key), never
# a knob: it must never differ per plan.
MXFP8_STAGE_SF_PADS: bool = True
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


@cute.jit
def _zero_ds(ds_full, config: cutlass.Constexpr, stream):
    """Zero ONE chunk's dS workspace (``[b_chunk, qh_chunk, S_kv_pad, S_q_pad]`` in the dS dtype) -- the fill the adapter asks
    for (``api_dsl_sm107._stage3_needs_zero_fill``): once per execute ahead of every chunk (``_stage2_inputs``), or ahead of
    EACH chunk under the caller's per-batch kv lengths (``host_f16``)."""
    b, h, hk, d, sq, skv, sqp, skvp, bc, hc, zero_ws, itemsize, bpe_ds, dq_bhg = config
    n16 = bc * hc * skvp * sqp * bpe_ds // 16
    _zero_bytes(ds_full, n16).launch(grid=(min((n16 + _THREADS - 1) // _THREADS, 4096), 1, 1), block=(_THREADS, 1, 1), stream=stream)


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
def _fp8_setup(
    seq_kv: Optional[cute.Tensor],
    skv: cutlass.Constexpr[int],
    amax_dq: Optional[cute.Tensor],
    amax_dk: Optional[cute.Tensor],
    amax_dv: Optional[cute.Tensor],
    amax_dp: cute.Tensor,
    amax_dv_kernel: cute.Tensor,
):
    """The fp8 row's ONE setup launch (one block of ``_THREADS``): the uniform per-batch kv length fill -- ``_fill_i32``'s loop over a
    ``seq_kv`` region of ANY length, None when the caller's lengths stand in for it or under THD (the metadata carries them) -- and
    the amax resets (the C++ node's semantics: an amax output is RESET on every execute, then atomicMax'd; the four graph outputs
    where requested, the kernel's own dV amax slot always).  Two kernels of the chain in one launch, disjoint words: bitwise."""
    tid, _, _ = cute.arch.thread_idx()
    if cutlass.const_expr(seq_kv is not None):
        i = tid
        while i < cute.size(seq_kv):
            seq_kv.iterator[i] = cutlass.Int32(skv)
            i += _THREADS
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


_fp8_setup.set_name_prefix("cudnn", remove_cutlass_symbol=True)


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


@cute.jit
def _e8m0_scale(e: cutlass.Int32) -> cutlass.Float32:
    """Biased E8M0 byte -> the EXACT fp32 dequant scale ``2^(e - 127)`` (the oracle's ``mxfp8_quant.e8m0_to_float``): the bit pattern
    ``e << 23`` for ``1 <= e <= 254``; ``e == 0`` is the fp32 SUBNORMAL 2^-127 (``0x00400000``, not +0.0: an all-zero block's payload is
    0 either way, a non-zero payload under byte 0 -- amax below 448 x 2^-127 -- dequantizes to its true tiny value as in the oracle);
    ``e == 255`` is NaN (E8M0 NaN; a poisoned pad byte must surface as non-finite, never as +inf x 0 = NaN by accident).

    ``e`` must be the UNSIGNED byte, 0..255 (``_sf_byte``): the DSL's ``raw_ptr`` carries Int8, so a bare ``.load().to(Int32)`` of a
    byte >= 128 (a block whose amax exceeds 448 -- legal MXFP8, E8M0 128..254) arrives as ``e - 256`` and ``(e - 256) << 23`` is
    ``0x80000000 | (e << 23)``: the RIGHT magnitude with a FLIPPED sign, finite and plausible, and the 255 sentinel unreachable (0xFF
    reads -1 -> -inf).  Found by review 2026-09-30 on the sm_107a PTX (``ld.global.s8`` straight into ``shl.b32 23``, no mask); the
    unit-normal test data (|x| < 6 -> e <= 127) had never produced such a byte.  Pinned by the all-256-bytes Rubin test and the
    zero-extending byte-load PTX pin (the mask folds into ``ld.global.b8``; an ``s8`` load needs an ``and.b32 255``) in
    ``test_sdpa_bwd_mxfp8_sm107.py``."""
    bits = e << 23
    if e == cutlass.Int32(0):
        bits = cutlass.Int32(0x00400000)
    if e == cutlass.Int32(255):
        bits = cutlass.Int32(0x7FC00000)
    return bits.bitcast(cutlass.Float32)


@cute.jit
def _sf_byte(ptr) -> cutlass.Int32:
    """The E8M0 byte at ``ptr`` (an Int8 ``raw_ptr``) as an UNSIGNED Int32 in 0..255 -- the mask undoes the Int8 sign-extension
    (see :func:`_e8m0_scale`).  Every SF byte that reaches arithmetic goes through here."""
    return ptr.load().to(cutlass.Int32) & cutlass.Int32(0xFF)


@cute.kernel
def _dequant_mxfp8_to_bf16(src: cute.Tensor, sf: cute.Tensor, dst: cute.Tensor, columnwise: cutlass.Constexpr[bool]):
    """``dst = bf16(e4m3(src) x 2^(e - 127))`` over a compact ``[B, S_pad, H, D]`` MXFP8 payload whose E8M0 scale factors ``sf`` are cuDNN
    F8_128x4 atoms -- the SF-aware twin of :func:`_cast_fp8_to_bf16` (which applies NO scale), the bf16 stage-3 operand of the MXFP8
    row's P-c chain (``dK = dS . Q_T``, ``dQ = dS^T . K_T`` over the dequantized COLUMNWISE q_T / k_T).  EXACT: an e4m3 value (3 mantissa
    bits) times a power of two is a bf16 value (7 bits), the exponent range covered.

    The atom rule (``test/python/sdpa/mxfp8_quant.py`` ``_swizzle_128x4``; the descriptors of ``sdpa/kernels/_mxfp8_sf.py``): a 512-B atom
    holds a 128 (rows) x 4 (32-element groups) block of the logical scale matrix at byte ``(r % 32) * 16 + (r // 32) * 4 + c``.
      columnwise (scales along S; ``sf`` = the ``[B, H, 8, S_pad]`` bytes of the kernel ABI): logical matrix ``[D, S/32]`` per (b, h) ->
        rows r = d % 128 in D-plane ``d // 128``, groups c = (s // 32) % 4 in tile ``s // 128``; atoms are D-PLANE-major over the WHOLE
        tensor: atom = ``(plane * B*H*T + (b*H + h) * T + s // 128) * 512`` (the plane stride grows with S -- mma-tma-matrix.md s7).
      rowwise (scales along D; ``sf`` = ``[B, H, S_pad, 8]``): logical ``[S, D/32]`` per (b, h) -> r = s % 128, c = (d // 32) % 4, atoms
        per (b, h, tile) contiguous, the two d-chunks (groups 0-3 / 4-7) at +0 / +512: atom = ``((b*H + h) * T + s // 128) * 1024 +
        ((d // 32) // 4) * 512``.
    ``S`` is the payload's row extent -- the padded ``S_pad`` (the bring-up driver) or the graph's REAL length (the prepared host:
    the caller's q_T / k_T are real-extent tensors) -- and the SF tensor holds ``T = ceil(S / 128)`` atoms per (b, h) per plane (the
    F8_128x4 rule pads rows to 128; the adapter's byte-count check pins it), so a ragged S walks its real rows against the same
    atoms.  Eight consecutive d per thread (an 8-byte fp8 load, a 16-byte bf16 store); the eight columnwise bytes of one
    thread sit 16 B apart inside ONE atom (8 divides 128), the rowwise eight share ONE byte.  A bring-up / host-side pass: the SF
    gathers hit L1 (one atom serves 128 x 128 elements); its perf is a follow-up (the shipped block-scaled chain launches no dequant pass).
    """
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    blocks, _, _ = cute.arch.grid_dim()
    B, S, H, D = dst.shape
    T = (S + SF_ATOM_ROWS - 1) // SF_ATOM_ROWS
    total = B * S * H * D // 8
    src_ptr = src.iterator.raw_ptr()
    sf_ptr = sf.iterator.raw_ptr()
    dst_ptr = dst.iterator.raw_ptr()
    i = cutlass.Int64(bid) * _THREADS + tid
    while i < total:
        pos = i * 8
        d0 = pos % D
        row = pos // D  # (b * S + s) * H + h
        h = row % H
        bs = row // H
        s = bs % S
        b = bs // S
        v = (src_ptr + pos).load(count=8)
        if cutlass.const_expr(columnwise):
            plane = d0 // SF_ATOM_ROWS
            atom = (plane * (B * H * T) + (b * H + h) * T + s // SF_ATOM_ROWS) * SF_ATOM_BYTES
            c = (s // MX_BLOCK) % 4
            r0 = d0 % SF_ATOM_ROWS
            # r = r0 + e for element e: the eight bytes sit 16 B apart inside one atom (r0 % 8 == 0, so (r // 32) is one value).
            # The (r % 32) * 16 + (r // 32) * 4 + c byte rule is the F8_128x4 atom's own (fixed literals, not geometry).
            base = atom + (r0 // 32) * 4 + c
            out = cutlass.Vector.from_elements(
                tuple((v[e].to(cutlass.Float32) * _e8m0_scale(_sf_byte(sf_ptr + (base + ((r0 % 32) + e) * 16)))).to(cutlass.BFloat16) for e in range(8)),
                cutlass.BFloat16,
            )
            (dst_ptr + pos).store(out, alignment=16)
        else:
            g = d0 // MX_BLOCK
            atom = ((b * H + h) * T + s // SF_ATOM_ROWS) * (_SF_ATOMS_PER_TILE * SF_ATOM_BYTES) + (g // 4) * SF_ATOM_BYTES
            r = s % SF_ATOM_ROWS
            sc = _e8m0_scale(_sf_byte(sf_ptr + (atom + (r % 32) * 16 + (r // 32) * 4 + (g % 4))))
            out = cutlass.Vector.from_elements(tuple((v[e].to(cutlass.Float32) * sc).to(cutlass.BFloat16) for e in range(8)), cutlass.BFloat16)
            (dst_ptr + pos).store(out, alignment=16)
        i += cutlass.Int64(blocks) * _THREADS


_dequant_mxfp8_to_bf16.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def dequant_mxfp8_to_bf16_host(src: cute.Tensor, sf: cute.Tensor, dst: cute.Tensor, columnwise: cutlass.Constexpr[bool], stream):
    """Launch :func:`_dequant_mxfp8_to_bf16` over the compact ``[B, S_pad, H, D]`` payload ``src`` (e4m3) into ``dst`` (bf16, same shape)
    with the F8_128x4 SF bytes ``sf`` (``columnwise`` selects the atom rule).  The host of the MXFP8 P-c chain calls it once per stage-3
    operand (q_T, k_T); the bring-up driver pins it bitwise against the torch dequant."""
    _dequant_mxfp8_to_bf16(src, sf, dst, columnwise).launch(grid=_grid_16b(dst, 2), block=(_THREADS, 1, 1), stream=stream)


@cute.kernel
def _dequant_mxfp8_to_bf16_thd(
    src: cute.Tensor,
    sf: cute.Tensor,
    dst: cute.Tensor,
    meta: cute.Tensor,
    sf_meta: cute.Tensor,
    n_batch: cutlass.Int32,
    n_tiles: cutlass.Int32,
    kv_side: cutlass.Constexpr[bool],
):
    """The THD twin of :func:`_dequant_mxfp8_to_bf16` for a COLUMNWISE packed operand: ``dst = bf16(e4m3(src) x 2^(e - 127))`` over the
    packed ``[1, T_cap, H, D]`` q_T / k_T (the P-c chain's bf16 stage-3 operands) -- read through the port's OWN token / head strides
    (a padded token stride >= H x D, what ``api_dsl_sm107._thd_packed_ok`` admits and the plan's geometry carries; the dense twin walks
    a compact slab) into the compact bf16 scratch -- with the scale factors in the PACKED per-sequence-
    tile-padded layout (``[H, T_sf, 2 x 512]``, both D planes of a (head, tile) contiguous).  Token ``t`` belongs to the sequence
    ``_thd_seq_of`` finds over the token prefixes; its scale sits in tile ``cu_sf[b] + (t - cu[b]) // 128``, group
    ``((t - cu[b]) // 32) % 4``, atom ``((h * T_sf + tile) * planes + d // 128)``, byte ``(r % 32) * 16 + (r // 32) * 4 + c`` with
    ``r = d % 128``.  Tokens at or past the live total ``cu[B]`` (the capacity tail) are written as exact zeros -- the GEMMs reach them
    only through the packed-total-clamped B descriptor, which reads them as TMA-OOB zeros anyway.  No pad byte is ever read: the
    packed token axis has no per-sequence pads, only the SF tiles do.  Eight consecutive d per thread, as the dense kernel."""
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    blocks, _, _ = cute.arch.grid_dim()
    B, S, H, D = dst.shape
    total = B * S * H * D // 8
    src_ptr = src.iterator.raw_ptr()
    sf_ptr = sf.iterator.raw_ptr()
    dst_ptr = dst.iterator.raw_ptr()
    meta_v = cutlass.make_array_view(meta)
    sf_v = cutlass.make_array_view(sf_meta)
    cu0, sf0 = _thd_prefix_bases(n_batch, kv_side)
    live_tokens = cutlass.Int32(meta_v[cu0 + n_batch])
    # The source's token / head strides are the PORT's (static ints of the host's layout): a packed port with a padded token stride
    # walked as a compact [T, H, D] slab reads the wrong tokens into every dK / dQ -- finite and wrong, dV exact.  Compact ports keep
    # the destination's linear offset (the common case, one index).
    _, src_ts, src_hs, _ = src.stride
    src_strided = (src_ts, src_hs) != (H * D, D)
    i = cutlass.Int64(bid) * _THREADS + tid
    while i < total:
        pos = i * 8
        d0 = pos % D
        row = pos // D  # t * H + h (one packed batch element)
        h = row % H
        t = cutlass.Int32(row // H)
        out = cutlass.Vector.from_elements(tuple(cutlass.BFloat16(0.0) for _ in range(8)), cutlass.BFloat16)
        if t < live_tokens:
            b = _thd_seq_of(meta_v, cu0, n_batch, t)
            local = t - cutlass.Int32(meta_v[cu0 + b])
            tile = cutlass.Int32(sf_v[sf0 + b]) + local // SF_ATOM_ROWS
            c = (local // MX_BLOCK) % 4
            plane = d0 // SF_ATOM_ROWS
            r0 = d0 % SF_ATOM_ROWS
            atom = (cutlass.Int64(h) * n_tiles + tile) * (_SF_ATOMS_PER_TILE * SF_ATOM_BYTES) + plane * SF_ATOM_BYTES
            base = atom + (r0 // 32) * 4 + c
            src_off = (cutlass.Int64(t) * src_ts + h * src_hs + d0) if cutlass.const_expr(src_strided) else pos
            v = (src_ptr + src_off).load(count=8)
            out = cutlass.Vector.from_elements(
                tuple((v[e].to(cutlass.Float32) * _e8m0_scale(_sf_byte(sf_ptr + (base + ((r0 % 32) + e) * 16)))).to(cutlass.BFloat16) for e in range(8)),
                cutlass.BFloat16,
            )
        (dst_ptr + pos).store(out, alignment=16)
        i += cutlass.Int64(blocks) * _THREADS


_dequant_mxfp8_to_bf16_thd.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def dequant_mxfp8_to_bf16_thd_host(
    src: cute.Tensor, sf: cute.Tensor, dst: cute.Tensor, meta: cute.Tensor, sf_meta: cute.Tensor, n_batch, n_tiles, kv_side: cutlass.Constexpr[bool], stream
):
    """Launch :func:`_dequant_mxfp8_to_bf16_thd` over the packed ``[1, T_cap, H, D]`` payload ``src`` (at the port's own token / head
    strides) into the compact ``dst`` with the packed columnwise SF ``sf`` (``n_tiles`` live tiles per head, the per-call count), the metadata's token prefixes and the SF tile prefixes of
    ``kv_side``.  The MXFP8 THD P-c chain calls it once per stage-3 operand (q_T over the q side, k_T over the kv side)."""
    _dequant_mxfp8_to_bf16_thd(src, sf, dst, meta, sf_meta, n_batch, n_tiles, kv_side).launch(grid=_grid_16b(dst, 2), block=(_THREADS, 1, 1), stream=stream)


# --- view helpers ---------------------------------------------------------------------------------------------------------


@cute.jit
def _window(tensor: cute.Tensor, dim: cutlass.Constexpr, begin, count: cutlass.Constexpr, step: cutlass.Constexpr = 1):
    """``tensor`` narrowed along ``dim`` to ``count`` entries starting at ``begin``, every ``step``-th (a view: ``t[..., begin::step, ...]``);
    any rank (the 4-D operands, the block-scale arm's 5-D scale-factor views)."""
    rank = len(tensor.shape)
    shape = tuple(count if i == dim else tensor.shape[i] for i in range(rank))
    strides = tuple(tensor.stride[i] * step if i == dim else tensor.stride[i] for i in range(rank))
    return cute.make_tensor(tensor.iterator + cutlass.Int64(begin) * tensor.stride[dim], cute.make_layout(shape, stride=strides))


@cute.jit
def _extent(tensor: cute.Tensor, shape: cutlass.Constexpr):
    """The leading ``shape`` of ``tensor`` under its own strides (a real-extent view of a padded buffer)."""
    return cute.make_tensor(tensor.iterator, cute.make_layout(shape, stride=tensor.stride))


@cute.jit
def _permuted(tensor: cute.Tensor, order: cutlass.Constexpr):
    return cute.make_tensor(tensor.iterator, cute.make_layout(tuple(tensor.shape[i] for i in order), stride=tuple(tensor.stride[i] for i in order)))


@cute.jit
def _matmul(
    entry: cutlass.Constexpr,
    a,
    b,
    output,
    heads: cutlass.Constexpr,
    batches: cutlass.Constexpr,
    meta,
    desc,
    stream,
    epi: cutlass.Constexpr = None,
    sf: cutlass.Constexpr = None,
    grid_m: cutlass.Constexpr = None,
    sf_meta: cutlass.Constexpr = None,
):
    """One ``(batch, head)``-batched stage-3 GEMM: ``a`` / ``b`` / ``output`` already in the template's ``(M|N, K, H, B)`` / ``(M, N, H, B)``
    order.  The problem tuple is the retired ``matmul_bh``'s, widened to Int64 before the descriptor's byte products; the grid's M
    is A's M (dense: the operands' shared extent) unless ``grid_m`` (appended) names it -- the THD chain's ENVELOPE M (every
    sequence's tiles cover the longest sequence; a shorter one's spare tiles read other rows and are clipped by its own C
    descriptor), where A's M is the whole blocked workspace.  ``epi`` = the fp8 arm's ``(descale_0, descale_1, scale_out, amax)`` fp32 [1]
    tensors (``scale_out`` / ``amax`` None outside EPI_QUANT / for an unrequested amax); None = a rendering without an epilogue,
    called with the SM100 chain's positional seven arguments.  ``sf`` (appended) = the block-scale arm's ``(sfa, sfb)`` -- the A
    operand's F8_128x4 atoms as ``(512 B, K tiles, M tiles, H, B)`` and the columnwise B scale factors as ``(512 B, D planes, K
    tiles, H, B)``, uint8 views with BYTE strides -- passed as the template's two trailing operands (no epilogue: the MMA dequantizes).
    ``sf_meta`` (appended) = the block-scale arm's THD leg's int32 ``[cu_sf_q(B+1) | cu_sf_k(B+1)]`` per-sequence SF tile prefixes
    (``config_sm100.STAGE3_THD_SF_*``), the template's last operand; None = the dense block-scale call shape."""
    m_grid = a.shape[0] if cutlass.const_expr(grid_m is None) else grid_m
    problem = tuple(
        cutlass.Int64(x)
        for x in (m_grid, b.shape[0], a.shape[1], heads, batches, *a.stride, *b.stride, *output.stride, b.shape[1], a.shape[0], output.shape[0])
    )
    if cutlass.const_expr(sf is not None and sf_meta is not None):
        entry(problem, a, b, output, meta, desc, stream, None, None, None, None, sf[0], sf[1], sf_meta)
    elif cutlass.const_expr(sf is not None):
        entry(problem, a, b, output, meta, desc, stream, None, None, None, None, sf[0], sf[1])
    elif cutlass.const_expr(epi is None):
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


@cute.kernel
def _pad_sf_atoms(
    src: cute.Tensor,
    dst: cute.Tensor,
    s_real: cutlass.Constexpr[int],
    groups: cutlass.Constexpr[int],
    t_src: cutlass.Constexpr[int],
    t_dst: cutlass.Constexpr[int],
    columnwise: cutlass.Constexpr[bool],
):
    """``dst`` = the F8_128x4 scale-factor atoms of ``src`` with every byte that scales a PAD position zeroed -- the MXFP8 row's
    zero-filled SF pad staging (sdpa-invariants s2).

    The producer's SF tensor covers ``ceil128(S_real)`` rows / groups and its pad bytes are producer-defined: a 0xFF there is an
    E8M0 NaN, and the kernel READS the pad positions (``S[kv, q_pad] = 0 x NaN`` on the Q side -- the +inf-LSE / SELECT-zero P no
    longer saves BMM2 from folding NaN x dO_T into dV on every kv row; on the kv side the descriptors span the tensor's padded
    tiles, so a poisoned pad tile lands NaN in the dead rows' dS).  MEASURED 2026-09-30 on Rubin: 131072 / 131072 dV NaN at S_q 160 with
    poisoned Q-side pads, 458752 dS NaN at S_kv 800 with poisoned kv pads; zero-filled pads -> every gate green (both RED-then-green).

    ``groups`` = B * H of the tensor; ``t_src`` / ``t_dst`` = its tiles per (b, h) in the source (``ceil128(S_real) / 128``) and in the
    kernel's extent (``S_pad / 128``: equal on the Q side, ``>=`` on the kv side where the kernel pads to 256 rows and the atoms to
    128 -- a whole zero tile is appended there).  Byte ``(r % 32) * 16 + (r // 32) * 4 + c`` of an atom scales row ``r`` (of 128),
    group ``c`` (of 4) (``test/python/sdpa/mxfp8_quant.py::_swizzle_128x4``):
      rowwise (scales along D; ``[B, H, S, 8]``): atoms ``((g * T + tile) * _SF_ATOMS_PER_TILE + chunk)`` (one atom per 4-group d-chunk,
        D / 128 of them), position ``s = tile * 128 + r`` -> pad iff ``s >= s_real``;
      columnwise (scales along S; ``[B, H, 8, S]``): atoms ``plane * (groups * T) + g * T + tile`` (D-plane-major over the whole tensor,
        the plane stride grows with S -- rules/mma-tma-matrix.md s7), group ``tile * 4 + c`` -> pad iff its first element
        ``(tile * 4 + c) * 32 >= s_real``.
    One byte per thread step: the tensors are 1/32 of a payload, and consecutive columnwise bytes belong to different groups.
    """
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    blocks, _, _ = cute.arch.grid_dim()
    src_ptr = src.iterator.raw_ptr()
    dst_ptr = dst.iterator.raw_ptr()
    total = groups * t_dst * _SF_ATOMS_PER_TILE * SF_ATOM_BYTES
    i = cutlass.Int64(bid) * _THREADS + tid
    while i < total:
        off = i % SF_ATOM_BYTES
        atom = i // SF_ATOM_BYTES
        r = (off // 16) + ((off % 16) // 4) * 32  # the F8_128x4 atom's byte rule, inverted (fixed literals, not geometry)
        c = off % 4
        if cutlass.const_expr(columnwise):
            plane = atom // (groups * t_dst)
            rest = atom % (groups * t_dst)
            g = rest // t_dst
            tile = rest % t_dst
            keep = ((tile * 4 + c) * MX_BLOCK < s_real) & (tile < t_src)
            src_atom = plane * (groups * t_src) + g * t_src + tile
        else:
            g = atom // (_SF_ATOMS_PER_TILE * t_dst)
            rest = atom % (_SF_ATOMS_PER_TILE * t_dst)
            tile = rest // _SF_ATOMS_PER_TILE
            chunk = rest % _SF_ATOMS_PER_TILE
            keep = (tile * SF_ATOM_ROWS + r < s_real) & (tile < t_src)
            src_atom = (g * t_src + tile) * _SF_ATOMS_PER_TILE + chunk
        v = cutlass.Int8(0)  # raw byte pointers load / store Int8 (the type the DSL's raw_ptr carries); a pure byte copy, sign-agnostic
        if keep:
            v = (src_ptr + (src_atom * SF_ATOM_BYTES + off)).load()
        (dst_ptr + i).store(v)
        i += cutlass.Int64(blocks) * _THREADS


_pad_sf_atoms.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _pad_sf(
    src: cute.Tensor,
    dst: cute.Tensor,
    s_real: cutlass.Constexpr[int],
    groups: cutlass.Constexpr[int],
    t_src: cutlass.Constexpr[int],
    t_dst: cutlass.Constexpr[int],
    columnwise: cutlass.Constexpr[bool],
    stream,
):
    """Launch :func:`_pad_sf_atoms`: ``src`` the caller's F8_128x4 SF bytes (any view; only the base address is read), ``dst`` the
    ``groups * t_dst * _SF_ATOMS_PER_TILE`` atoms of the kernel-facing slab."""
    total = groups * t_dst * _SF_ATOMS_PER_TILE * SF_ATOM_BYTES
    _pad_sf_atoms(src, dst, s_real, groups, t_src, t_dst, columnwise).launch(
        grid=(min((total + _THREADS - 1) // _THREADS, 4096), 1, 1), block=(_THREADS, 1, 1), stream=stream
    )


# The packed (THD) scale-factor layout, per SF tensor: ``[H, T_sf, _SF_TILE_BYTES]`` -- every head's tiles in cu_seqlens order,
# sequence b's tiles at ``cu_sf[b] .. cu_sf[b+1]`` (``cu_sf[b] = SUM_{i<b} ceil(s_i / 128)``, the ``sf_meta`` region), 1024 B per
# (head, 128-token tile) at d = 256: rowwise the two d-chunk atoms (groups 0-3 / 4-7), columnwise the two D-plane atoms -- BOTH planes
# of a (head, tile) contiguous (``config_sm100.stage3_thd_sfb_layout``; the dense tensor is D-plane-major over the whole tensor).
# The per-sequence pad zeroing of that layout is the chain's ``thd_helpers.pad_sf_atoms_thd_host`` (next to the setup launch that
# writes the prefixes it reads); this module keeps the P-c dequant's per-token twin of its sequence lookup below.
_SF_TILE_BYTES = _SF_ATOMS_PER_TILE * SF_ATOM_BYTES


def _thd_delta_geometry(h: int, t_q: int):
    """The THD plans' delta ``(shape, strides)`` -- PACKED head-major ``[1, H, ceil128(T_q)]`` fp32 contiguous over the plan's token
    capacity ``t_q`` (``config[4]``), padded to ``dot_do_o``'s 128-row q tile: EXACTLY the carved ``delta`` region's extent
    (``api_dsl_sm107._scratch_shapes``, the THD arm; ``external_delta_shape``; ``prepared_sm107._delta_geometry`` is the adapter-side
    twin), so a caller's tensor and the chain's own region are read by the main kernel through one and the same view.  Plain Python,
    NOT ``@cute.jit`` (``_dq_launches``'s form): int arithmetic on two ``config`` entries, evaluated once at trace time inside the THD
    hosts' ``const_expr`` ternary, whose tuple reaches ``_view`` exactly as a dense host's ``geometry[i]`` does."""
    t_pad = -(-t_q // DOT_Q_TILE) * DOT_Q_TILE
    return (1, h, t_pad), (h * t_pad, t_pad, 1)


@cute.jit
def _thd_prefix_bases(n_batch, kv_side: cutlass.Constexpr[bool]):
    """``(token prefix base in meta, tile prefix base in sf_meta)`` of one token side: the shared metadata's ``cu_q`` sits at word B
    and ``cu_k`` at 2B+1 (``tile_dsl.thd``), the SF tile prefixes at ``config_sm100.STAGE3_THD_SF_CU_{Q,K}_OFF``."""
    if cutlass.const_expr(kv_side):
        return cutlass.Int32(2) * n_batch + cutlass.Int32(1), cutlass.Int32(STAGE3_THD_SF_CU_K_OFF(n_batch))
    return cutlass.Int32(n_batch), cutlass.Int32(STAGE3_THD_SF_CU_Q_OFF)


@cute.jit
def _thd_seq_of(prefix, base, n_batch, pos):
    """The sequence a packed position belongs to: the LARGEST ``b`` in ``[0, B)`` with ``prefix[base + b] <= pos`` (binary search over a
    non-decreasing prefix; an empty sequence shares its neighbour's prefix and is skipped).  The caller checks ``pos < prefix[base + B]``."""
    lo = cutlass.Int32(0)
    hi = cutlass.Int32(n_batch)
    while hi - lo > cutlass.Int32(1):
        mid = (lo + hi) // cutlass.Int32(2)
        if cutlass.Int32(prefix[base + mid]) <= pos:
            lo = mid
        else:
            hi = mid
    return lo


@cute.jit
def _slot(scratch: cute.Tensor, index: cutlass.Constexpr[int]):
    """One fp32 element of the amax scratch as a ``[1]`` tensor (the shape the kernels' amax operands take)."""
    return cute.make_tensor(scratch.iterator + index, cute.make_layout((1,), stride=(1,)))


@cute.jit
def _grid_16b(t: cute.Tensor, itemsize: cutlass.Constexpr[int]):
    return (min((cute.size(t) * itemsize // 16 + _THREADS - 1) // _THREADS, 4096), 1, 1)


@cute.jit
def _stage2_inputs(
    q,
    k,
    v,
    do,
    stats,
    workspace,
    regions: cutlass.Constexpr,
    config: cutlass.Constexpr,
    dtype: cutlass.Constexpr,
    ds_dtype: cutlass.Constexpr,
    stream,
    seq_kv_lens=None,
    fill_seq_kv: cutlass.Constexpr[bool] = True,
):
    """The kernel-facing Q / dO / K / V / LSE (zero- / +inf-padded staging copies when the graph's extents are not tile multiples),
    the per-batch kv lengths and the (zero-filled under a mask) dS workspace -- as launches of this artifact.

    ``seq_kv_lens`` (appended; None = the uniform fill) is the caller's ``[B]`` int32 per-batch kv lengths when the plan was built
    with ``seq_kv_lens_present`` (the half row's standalone surface): the kernel's padded-mask arm then reads them in place of the
    ``seq_kv`` region, which stays carved (fixed workspace plan) and unwritten.  ``fill_seq_kv`` (appended, default True) = this
    helper launches the uniform fill; the fp8 row passes False and fills the region inside its own setup launch (``_fp8_setup``)."""
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
    # Read by the padded mask arm only (the uniform real kv length, one entry per batch); carved regardless (fixed kernel ABI) and
    # written unless the caller's per-batch lengths stand in for it.  ONE block: the fill strides over all B entries itself, so the
    # launch needs no B-derived grid (B is at most a few hundred here; a second block would cost more than the loop it saves).
    seq_kv = seq_kv_lens if cutlass.const_expr(seq_kv_lens is not None) else _scratch(workspace, regions[R_SEQ_KV], cutlass.Int32)
    if cutlass.const_expr(seq_kv_lens is None and fill_seq_kv):
        _fill_i32(seq_kv, skv).launch(grid=(1, 1, 1), block=(_THREADS, 1, 1), stream=stream)
    # Zero ONCE, ahead of every chunk, and ONLY when the adapter says so (`api_dsl_sm107._stage3_needs_zero_fill`): with the
    # two-sided K-trim the stage-3 GEMMs read only the tiles the main kernel wrote (it rounds every kv block's q range
    # outward to the GEMMs' 256-row pair), so no mask needs it -- except the untrimmed twin (every tile read) and a top-left
    # window with S_q > roundup(S_kv + W, 256), where the q pairs past the last kv block's window are written by nobody.
    # The skipped set is the same for every chunk -- under the UNIFORM length.  With the caller's per-batch lengths it is per
    # BATCH (the bottom-right band moves with seq_kv_lens[b]), so `host_f16` zeroes ahead of EACH chunk instead: a workspace
    # slot a later chunk's batch reuses would otherwise hold the earlier batch's dS in the tiles the later band does not
    # write (measured as a dead batch coming back with non-zero dQ / dK behind a live one).
    ds_full = _scratch(workspace, regions[R_DS], ds_dtype)
    if cutlass.const_expr(zero_ws and seq_kv_lens is None):
        _zero_ds(ds_full, config, stream)
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


@cute.jit
def _stage3_thd(
    mm_dk: cutlass.Constexpr,
    mm_dq: cutlass.Constexpr,
    ds,
    q,
    k,
    dk_out,
    dq_out,
    hb,
    hc: cutlass.Constexpr,
    group: cutlass.Constexpr,
    n_seq: cutlass.Constexpr,
    meta,
    desc,
    stream,
    grid_m_kv: cutlass.Constexpr,
    grid_m_q: cutlass.Constexpr,
    dq_b_head_group: cutlass.Constexpr = 1,
    dk_epi: cutlass.Constexpr = None,
    dq_epi: cutlass.Constexpr = None,
):
    """The THD twin of :func:`_stage3`: dK = dS . Q and dQ = dS^T . K over the kv-BLOCKED workspace ``ds`` (``[1, hc, R_kv_cap,
    S_q_pad]``, the chunk's view), the PACKED ``q`` / ``k`` (``[1, T, H, D]``) and the packed ``dk_out`` (``[1, T_kv, H_q, D]``, the
    per-Q-head partials under GQA) / ``dq_out`` (``[1, T_q, H_q, D]``).  Both renderings carry ``thd_varlen`` + ``thd_rows_kv``: each
    (head, sequence) group reads its own blocked rows at ``row_off[b]`` (A), its packed B rows at ``cu_*[b]`` through the
    packed-total-clamped slot, reduces over the sequence's REAL length -- trimmed to the sequence's own band under a mask
    (``_thd_causal_k_range``) -- and stores through the sequence's clipped C descriptor (the template's THD arm; the GEMM's own
    patch launch builds them from ``meta`` into ``desc``).  ``n_seq`` is the template's batch = the SEQUENCE count (the packed
    operands hold one batch element); ``grid_m_kv`` / ``grid_m_q`` are the ENVELOPE's M extents (``_matmul(grid_m=)``).
    ``dq_b_head_group`` (appended) is the dQ rendering's ``MatmulTemplateParams.b_head_group`` and decides the launch count
    exactly as in :func:`_stage3` (``_dq_launches``): the GQA group = ONE launch over the chunk's ``hc`` Q heads whose B = K is
    indexed by ``h // group`` (the packed B descriptor's head extent is ``kv_n``; its per-sequence clamp touches only the
    token extent), 1 = one launch per group MEMBER over every ``group``-th Q head -- the bitwise twin.  ``dk_epi`` / ``dq_epi``
    (appended, default None = a rendering without an epilogue) are the fp8 K64 arm's epilogue operands, exactly as in
    :func:`_stage3` -- a THD plan of the fp8 row that left them off would silently run EPI_NONE-shaped GEMMs."""
    q_c = _window(q, 2, hb, hc)  # [1, T_q, hc, D]
    dk_c = _window(dk_out, 2, hb, hc)  # [1, T_kv, hc, D]
    # dK = dS . Q: A = dS[kv rows, q cols] (M, K, H, 1) K-major; B = Q (D, T_q, H, 1) packed; out (T_kv, D, H, 1) packed.
    _matmul(
        mm_dk, _permuted(ds, (2, 3, 1, 0)), _permuted(q_c, (3, 1, 2, 0)), _permuted(dk_c, (1, 3, 2, 0)), hc, n_seq, meta, desc, stream, dk_epi, grid_m=grid_m_kv
    )
    kv_n = hc // group
    n_launch = _dq_launches(group, dq_b_head_group)
    heads = hc // n_launch  # Q heads per dQ launch: kv_n * dq_b_head_group
    k_c = _window(k, 2, hb // group, kv_n)  # [1, T_kv, kv_n, D]
    for member in range(n_launch):
        a_g = _window(ds, 1, member, heads, n_launch)  # ds[:, member::n_launch] -> [1, heads, R_kv_cap, S_q_pad] (ds itself at one launch)
        o_g = _window(dq_out, 2, hb + member, heads, n_launch)  # dq[:, :, hb+member::n_launch] -> [1, T_q, heads, D]
        # dQ = dS^T . K: A = dS^T[q cols, kv rows] (M, K, H, 1) M-major; B = K (D, T_kv, H_kv, 1) packed; out (T_q, D, H, 1) packed.
        _matmul(
            mm_dq,
            _permuted(a_g, (3, 2, 1, 0)),
            _permuted(k_c, (3, 1, 2, 0)),
            _permuted(o_g, (1, 3, 2, 0)),
            heads,
            n_seq,
            meta,
            desc,
            stream,
            dq_epi,
            grid_m=grid_m_q,
        )


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


@cute.jit
def _sf_planes_view(sf: cute.Tensor, planes: cutlass.Constexpr, tiles: cutlass.Constexpr, heads: cutlass.Constexpr, batches: cutlass.Constexpr):
    """The block-scale GEMM arm's SFB view of a COLUMNWISE F8_128x4 scale tensor (``sdpa.kernels._mxfp8_sf``: D-plane-major atoms, the
    plane stride = ``batches * heads * tiles`` atoms -- it GROWS with S): ``(512 B atom, D planes, S/128 tiles, H, B)`` over the
    tensor's base with BYTE strides ``(1, B * H * tiles * 512, 512, tiles * 512, H * tiles * 512)``.  ``tiles`` is the SF tensor's OWN
    ``ceil128(S) / 128`` (the producer's padded extent), never the kernel's 256-row pad."""
    shape = (SF_ATOM_BYTES, planes, tiles, heads, batches)
    strides = (1, batches * heads * tiles * SF_ATOM_BYTES, SF_ATOM_BYTES, tiles * SF_ATOM_BYTES, heads * tiles * SF_ATOM_BYTES)
    return cute.make_tensor(sf.iterator, cute.make_layout(shape, stride=strides))


@cute.jit
def _sf_planes_view_thd(sf: cute.Tensor, planes: cutlass.Constexpr, tiles, heads: cutlass.Constexpr):
    """The block-scale GEMM arm's SFB view of a PACKED (THD) columnwise F8_128x4 scale tensor -- ``(512 B atom, D planes, packed tiles,
    H, 1)`` with the byte strides ``config_sm100.stage3_thd_sfb_layout`` spells (plane stride ONE atom, tile stride the ``planes x 512``-byte
    slab: both planes of a (head, sequence-tile) sit together, the forward's per-sequence-tile-padded convention), ``tiles`` the bound
    buffer's PACKED tile count (the per-call Int32 the binder derives).  The THD twin of :func:`_sf_planes_view`; reading a packed tensor
    through the dense D-plane-major view would fetch plane 1 from the wrong place by an S-dependent offset (rules/mma-tma-matrix.md s7).
    The template's THD SFB coordinate ``(0, plane, cu_sf[b] + k_tile, h, 0)`` is written against THIS layout."""
    shape, strides = stage3_thd_sfb_layout(planes, tiles, heads, SF_ATOM_BYTES)
    return cute.make_tensor(sf.iterator, cute.make_layout(shape, stride=strides))


@cute.jit
def _stage3_block_scale(
    mm_dk: cutlass.Constexpr,
    mm_dq: cutlass.Constexpr,
    ds_dk,
    ds_dq,
    sf_ds_dk,
    sf_ds_dq,
    q_T,
    sf_q_T,
    k_T,
    sf_k_T,
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
    dq_b_head_group: cutlass.Constexpr = 1,
):
    """The block-scaled dS chain's stage 3 (``MatmulTemplateParams.block_scale``): dK = ds_dk . q_T and dQ = ds_dq^T . k_T for one
    (batch, head) chunk, every operand a view.  ``ds_dk`` / ``ds_dq`` are the chunk's REAL-extent ``[bc, hc, S_kv, S_q]`` e4m3 payloads
    (scaled per 32-q block / per 32-kv block), ``sf_ds_dk`` ``[bc, hc, S_kv/128, S_q/128, 512]`` and ``sf_ds_dq`` ``[bc, hc, S_q/128,
    S_kv/128, 512]`` their F8_128x4 atoms (the workspace contract; the padded tile grid -- the GEMM visits only the real K tiles),
    ``q_T`` / ``k_T`` the full columnwise-quantized ``[B, S, H, D]`` payloads and ``sf_q_T`` / ``sf_k_T`` their ``_sf_planes_view``.
    dK: A = ds_dk[kv, q] (M, K, H, B) K-major, SFA atoms (512, K = q tiles, M = kv tiles, H, B); B = q_T (D, q, H, B), SFB the q_T
    planes view windowed to the chunk's heads.  dQ: A = ds_dq^T[q, kv] (M, K, H, B) M-major, SFA (512, K = kv tiles, M = q tiles,
    H, B); B = k_T (D, kv, H_kv, B), SFB the k_T planes view windowed to the chunk's ``kv_n`` K heads.  ``dq_b_head_group`` (appended,
    default 1) is the dQ rendering's ``MatmulTemplateParams.b_head_group`` and decides the launch count exactly as in :func:`_stage3`
    (``_dq_launches``): the GQA group = ONE launch over the chunk's ``hc`` Q heads -- A = the whole ``ds_dq`` chunk, SFA = the whole
    ``sf_ds_dq`` chunk, out = the whole dQ chunk, B / SFB the kv-head windows; the template indexes B AND its scale-factor descriptor
    by ``h // b_head_group`` (``_b_head``), so no per-member descriptor exists -- 1 = one launch per group MEMBER over every
    ``group``-th Q head, the atoms windowed like the payload (the ``DQ_SINGLE_LAUNCH = False`` twin).  Both walk the same k tiles per
    output tile into the same fp32 accumulator: bitwise-equal dQ.  EPI_NONE: the MMA dequantizes both operands, the accumulator
    is the true-unit gradient."""
    q_c = _window(_window(q_T, 0, bb, bc), 2, hb, hc)  # [bc, S_q, hc, D]
    sfq_c = _window(_window(sf_q_T, 4, bb, bc), 3, hb, hc)  # (512, planes, q tiles, hc, bc)
    dk_c = _window(_window(dk_out, 0, bb, bc), 2, hb, hc)  # [bc, S_kv, hc, D]
    _matmul(
        mm_dk,
        _permuted(ds_dk, (2, 3, 1, 0)),
        _permuted(q_c, (3, 1, 2, 0)),
        _permuted(dk_c, (1, 3, 2, 0)),
        hc,
        bc,
        meta,
        desc,
        stream,
        None,
        (_permuted(sf_ds_dk, (4, 3, 2, 1, 0)), sfq_c),
    )
    kv_n = hc // group
    n_launch = _dq_launches(group, dq_b_head_group)
    heads = hc // n_launch  # Q heads per dQ launch: kv_n * dq_b_head_group
    k_c = _window(_window(k_T, 0, bb, bc), 2, hb // group, kv_n)  # [bc, S_kv, kv_n, D]
    sfk_c = _window(_window(sf_k_T, 4, bb, bc), 3, hb // group, kv_n)  # (512, planes, kv tiles, kv_n, bc)
    for member in range(n_launch):
        a_g = _window(ds_dq, 1, member, heads, n_launch)  # ds_dq[:, member::n_launch] -> [bc, heads, S_kv, S_q] (ds_dq itself at one launch)
        sfa_g = _window(sf_ds_dq, 1, member, heads, n_launch)  # [bc, heads, S_q/128, S_kv/128, 512] (sf_ds_dq itself at one launch)
        o_g = _window(_window(dq_out, 0, bb, bc), 2, hb + member, heads, n_launch)  # dq[bs, :, hb+member::n_launch] -> [bc, S_q, heads, D]
        _matmul(
            mm_dq,
            _permuted(a_g, (3, 2, 1, 0)),
            _permuted(k_c, (3, 1, 2, 0)),
            _permuted(o_g, (1, 3, 2, 0)),
            heads,
            bc,
            meta,
            desc,
            stream,
            None,
            (_permuted(sfa_g, (4, 3, 2, 1, 0)), sfk_c),
        )


@cute.jit
def _stage3_block_scale_thd(
    mm_dk: cutlass.Constexpr,
    mm_dq: cutlass.Constexpr,
    ds_dk,
    ds_dq,
    sf_ds_dk,
    sf_ds_dq,
    q_T,
    sf_q_T_planes,
    k_T,
    sf_k_T_planes,
    dk_out,
    dq_out,
    hb,
    hc: cutlass.Constexpr,
    group: cutlass.Constexpr,
    n_seq: cutlass.Constexpr,
    meta,
    desc,
    sf_meta,
    stream,
    grid_m_kv: cutlass.Constexpr,
    grid_m_q: cutlass.Constexpr,
    dq_b_head_group: cutlass.Constexpr = 1,
):
    """The THD twin of :func:`_stage3_block_scale` (the block-scale arm's THD leg): dK = ds_dk . q_T and dQ = ds_dq^T . k_T over the
    kv-BLOCKED e4m3 payloads ``ds_dk`` / ``ds_dq`` (``[1, hc, R_kv_cap, S_q_pad]``, the chunk's views) and their atoms ``sf_ds_dk``
    ``[1, hc, R_kv_cap/128, S_q_pad/128, 512]`` / ``sf_ds_dq`` (the transpose) -- A and SFA ride the blocked rows exactly like the plain
    THD arm's A (the template adds ``row_off[b] / 128`` to the atom coordinate) -- the PACKED columnwise ``q_T`` / ``k_T`` (``[1, T, H, D]``)
    with their scale factors as ``_sf_planes_view_thd`` views (B indexed through the per-sequence SF tile prefixes ``sf_meta`` the
    template reads, ``_thd_sf_tile_base``), and the packed outputs ``dk_out`` (``[1, T_kv, H_q, D]``: the per-Q-head partials under
    GQA) / ``dq_out`` (``[1, T_q, H_q, D]``) through per-sequence clipped descriptors.  ``n_seq`` is the template's batch = the
    SEQUENCE count; ``grid_m_kv`` / ``grid_m_q`` the ENVELOPE's M extents (``_matmul(grid_m=)``).  ``dq_b_head_group`` (appended,
    default 1) is the dQ rendering's ``MatmulTemplateParams.b_head_group`` and decides the launch count exactly as on the dense
    block-scale chain (``_dq_launches``): the GQA group = ONE launch over the chunk's ``hc`` Q heads whose B = k_T AND its packed
    scale-factor planes are indexed by ``h // group`` (the packed B descriptor's head extent is ``kv_n``; its per-sequence clamp and the
    SF tile prefix ``cu_sf[b]`` touch only the token side), 1 = one launch per group MEMBER over every ``group``-th Q head, the atoms
    windowed like the payload -- the bitwise twin.  EPI_NONE: the MMA dequantizes both operands, the accumulator is the true-unit
    gradient."""
    q_c = _window(q_T, 2, hb, hc)  # [1, T_q, hc, D]
    sfq_c = _window(sf_q_T_planes, 3, hb, hc)  # (512, planes, packed q tiles, hc, 1)
    dk_c = _window(dk_out, 2, hb, hc)  # [1, T_kv, hc, D]
    # dK = dS . Q: A = ds_dk[kv rows, q cols] (M, K, H, 1) K-major, SFA (512, K = q tiles, M = kv tiles, H, 1); B = q_T (D, T_q, H, 1)
    # packed, SFB the packed planes view; out (T_kv, D, H, 1) packed.
    _matmul(
        mm_dk,
        _permuted(ds_dk, (2, 3, 1, 0)),
        _permuted(q_c, (3, 1, 2, 0)),
        _permuted(dk_c, (1, 3, 2, 0)),
        hc,
        n_seq,
        meta,
        desc,
        stream,
        None,
        (_permuted(sf_ds_dk, (4, 3, 2, 1, 0)), sfq_c),
        grid_m=grid_m_kv,
        sf_meta=sf_meta,
    )
    kv_n = hc // group
    n_launch = _dq_launches(group, dq_b_head_group)
    heads = hc // n_launch  # Q heads per dQ launch: kv_n * dq_b_head_group
    k_c = _window(k_T, 2, hb // group, kv_n)  # [1, T_kv, kv_n, D]
    sfk_c = _window(sf_k_T_planes, 3, hb // group, kv_n)  # (512, planes, packed kv tiles, kv_n, 1)
    for member in range(n_launch):
        a_g = _window(ds_dq, 1, member, heads, n_launch)  # ds_dq[:, member::n_launch] -> [1, heads, R_kv_cap, S_q_pad] (ds_dq itself at one launch)
        sfa_g = _window(sf_ds_dq, 1, member, heads, n_launch)  # [1, heads, S_q_pad/128, R_kv_cap/128, 512] (sf_ds_dq itself at one launch)
        o_g = _window(dq_out, 2, hb + member, heads, n_launch)  # dq[:, :, hb+member::n_launch] -> [1, T_q, heads, D]
        # dQ = dS^T . K: A = ds_dq^T[q cols, kv rows] (M, K, H, 1) M-major, SFA (512, K = kv tiles, M = q tiles, H, 1); B = k_T
        # (D, T_kv, H_kv, 1) packed, SFB the packed planes view; out (T_q, D, H, 1) packed.
        _matmul(
            mm_dq,
            _permuted(a_g, (3, 2, 1, 0)),
            _permuted(k_c, (3, 1, 2, 0)),
            _permuted(o_g, (1, 3, 2, 0)),
            heads,
            n_seq,
            meta,
            desc,
            stream,
            None,
            (_permuted(sfa_g, (4, 3, 2, 1, 0)), sfk_c),
            grid_m=grid_m_q,
            sf_meta=sf_meta,
        )


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
    seq_kv_ptr: Optional[cute.Pointer],
    delta_ptr: Optional[cute.Pointer],
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
    # The two appended pointers are independent plan facts (a plan may bind both, either, or neither), each None-specialized out of
    # a plan built without its flag; geometry[i] is operand i's static layout for every slot, these two included.
    # The caller's per-batch kv lengths ([B] int32, geometry[9]): the padded mask arm reads them in place of the uniform ``seq_kv`` fill.
    seq_kv_lens = _view(seq_kv_ptr, geometry[9])
    # delta [B, H, S_q_pad] fp32: the caller's (external_delta -- the carve has no region, the chain launches no dot) or
    # the workspace region stage 1 fills below.  Same static layout either way (geometry[10] is the region's shape).
    external_delta = cutlass.const_expr(delta_ptr is not None)
    delta = _view(delta_ptr, geometry[10]) if cutlass.const_expr(external_delta) else _scratch(workspace, regions[R_DELTA], cutlass.Float32)
    desc = _scratch(workspace, regions[R_DESC], cutlass.Int64)  # the GEMMs' dead THD slot (dense: never read)
    q_k, do_k, lse_k, k_k, v_k, seq_kv, ds_full = _stage2_inputs(q, k, v, do, stats, workspace, regions, config, dtype, dtype, stream, seq_kv_lens)
    group = h // hk
    kv_padded = regions[R_K_PAD] is not None
    # stage 2's dV per Q head: the caller's dV only when MHA and no kv padding; stage 3's dK per Q head: the caller's dK when MHA.
    dv_k = _scratch(workspace, regions[R_DV_PART], dtype) if cutlass.const_expr(regions[R_DV_PART] is not None) else dv
    dk_tgt = _scratch(workspace, regions[R_DK_PART], dtype) if cutlass.const_expr(regions[R_DK_PART] is not None) else dk
    dk_real = _extent(dk_tgt, (b, skv, h, d))  # stage 3 writes the real rows only
    ds = _extent(ds_full, (bc, hc, skv, sq))  # what stage 3 reads: the real extents (padded rows / cols never reach a GEMM)

    # STAGE 1, hoisted out of the chunk loop: one streaming pass over O and dO -- unless the caller computed it
    # (external_delta: the producer of dO writes rowsum(dO * O) in this kernel's order, so the launch and the read go).
    if cutlass.const_expr(not external_delta):
        dot_do_o_host(o, do, delta, None, None, DOT_Q_TILE, d, d, DOT_CHUNK_ELEMS, False, False, stream)

    for bi in range(b // bc):
        bb = bi * bc
        for ci in range(h // hc):
            hb = ci * hc
            # Per-batch kv lengths: the dS zero-fill goes ahead of EACH chunk's kernel (`_stage2_inputs` skipped its once-per-execute
            # fill) -- under bottom-right the skipped set is per batch, and this slot may still hold the previous chunk's batch in
            # the tiles this batch's narrower band does not write.
            if cutlass.const_expr(zero_ws and seq_kv_ptr is not None):
                _zero_ds(ds_full, config, stream)
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


@cute.jit
def host_f16_thd(
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    v_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    do_ptr: cute.Pointer,
    stats_ptr: cute.Pointer,
    dq_ptr: cute.Pointer,
    dk_ptr: cute.Pointer,
    dv_ptr: cute.Pointer,
    seq_q_ptr: cute.Pointer,
    seq_kv_ptr: cute.Pointer,
    delta_ptr: Optional[cute.Pointer],
    workspace: cute.Pointer,
    scale: cutlass.Float32,
    lens_form: cutlass.Int32,
    main: cutlass.Constexpr,
    mm_dk: cutlass.Constexpr,
    mm_dq: cutlass.Constexpr,
    config: cutlass.Constexpr,
    geometry: cutlass.Constexpr,
    regions: cutlass.Constexpr,
    dtype: cutlass.Constexpr,
    stream: driver.CUstream,
):
    """The half row's THD / varlen chain (``SdpaBwdDslSm107(thd=True)``): PACKED ``[1, T, H, D]`` operands at the plan's token
    capacities, a kv-BLOCKED dS workspace, the per-sequence lengths from the caller's two length tensors, and -- APPENDED, optional,
    the dense host's rule -- the caller's ``delta`` (``external_delta=True``: the PACKED head-major ``[1, H, ceil128(T_q)]`` fp32
    ``rowsum(dO * O)``, the ``dot_do_o`` layout at ``B = 1, S = T_q``, viewed from the plan's token capacity; with it bound the
    ``dot`` launch and the workspace's ``delta`` region do not exist).  A SIBLING of :func:`host_f16` with its own ABI (two length
    operands + the delta + ``lens_form``), its own frame and its own cache key: the dense artifact is untouched.

        setup    thd_bwd_setup_host(kv_blocked=True): [seq_kv_lens | cu_q | cu_k | batch_remap | live | ctr | row_off] with the
                 row offsets over the KV lengths at the kernel's 256-row block (``prepared_sm107`` reserves the main kernel's
                 (5 + B) tensor maps after it, in the same region); ONCE per execute
        fill     the dS workspace zeroed ONCE per execute ONLY for the untrimmed / wide-tile twins
                 (``api_dsl_sm107._stage3_thd_needs_zero_fill``): the shipped stage 3 trims PER SEQUENCE and reads only tiles the
                 main kernel wrote (an empty band is an empty K range and a select-zero store).  Dense THD never needs it:
                 every q tile of every kv block of every sequence is written and the GEMMs read only ceil(len/64) tiles inside
                 each sequence's block
        delta    dot_do_o over the packed O / dO -> [1, H, ceil128(T_q)] -- unless the caller computed it (external_delta)
        per head chunk: the main kernel (its own setup launch clamps the five input descriptors, emits the per-sequence dV
                 descriptors and resets live / ctr for THIS launch's heads), then dK / dQ through the THD stage-3 arm
        fold     GQA: the per-Q-head dK / dV partials over the PACKED kv axis, rows below the live total cu_k[B] only (a
                 device word) -> the KV heads (fixed order); the caller's capacity tail past cu_k[B] is never written

    ``lens_form`` bit 0 / 1 = the Q / KV length tensor is a ``(B+1,)`` prefix (``bind()`` derives it from numel); the setup
    kernel branches on it before reading the prefix tail, so both tensors are viewed ``(B+1,)``.
    """
    b, h, hk, d, t_q, t_kv, sqp, rcap, hc, zero_ws, itemsize, units, sq_env, skv_env, dq_bhg = config
    q = _view(q_ptr, geometry[0])  # packed [1, T_q, H_q, D]
    k = _view(k_ptr, geometry[1])  # packed [1, T_kv, H_kv, D]
    v = _view(v_ptr, geometry[2])
    o = _view(o_ptr, geometry[3])
    do = _view(do_ptr, geometry[4])
    stats = _view(stats_ptr, geometry[5])  # (T_q, H_q) token-major or (1, H_q, head_stride) head-major, the forward's packing
    dq = _view(dq_ptr, geometry[6])
    dk = _view(dk_ptr, geometry[7])
    dv = _view(dv_ptr, geometry[8])
    q_lens = _view(seq_q_ptr, ((b + 1,), (1,)))
    kv_lens = _view(seq_kv_ptr, ((b + 1,), (1,)))
    # delta [1, H, ceil128(T_q)] fp32, PACKED head-major: the caller's (external_delta -- the carve has no region, the chain launches
    # no dot; the view is the plan's token capacity padded to the q tile, exactly the carved region's extent) or the region stage 1
    # fills below.  The main kernel reads it at the packed token index (cu_q[b] + pos) either way.
    external_delta = cutlass.const_expr(delta_ptr is not None)
    delta = _view(delta_ptr, _thd_delta_geometry(h, t_q)) if cutlass.const_expr(external_delta) else _scratch(workspace, regions[R_DELTA], cutlass.Float32)
    meta = _scratch(workspace, regions[R_SEQ_KV], cutlass.Int32)  # the metadata words + the main kernel's tensor maps
    desc3 = _scratch(workspace, regions[R_DESC], cutlass.Int64)  # stage 3's (B + 1) descriptors, patched per GEMM launch
    ds_full = _scratch(workspace, regions[R_DS], dtype)  # [1, hc, R_kv_cap, S_q_pad]
    thd_bwd_setup_host(meta, q_lens, kv_lens, lens_form, hc, b, _THD_KV_BLOCK, _THD_KV_BLOCK, units, stream, kv_blocked=True)
    if cutlass.const_expr(zero_ws):
        n16 = hc * rcap * sqp * itemsize // 16
        _zero_bytes(ds_full, n16).launch(grid=(min((n16 + _THREADS - 1) // _THREADS, 4096), 1, 1), block=(_THREADS, 1, 1), stream=stream)
    group = h // hk
    # stage 2's dV per Q head and stage 3's dK per Q head: the caller's packed dV / dK at MHA, the packed partials under GQA.
    dv_k = _scratch(workspace, regions[R_DV_PART], dtype) if cutlass.const_expr(regions[R_DV_PART] is not None) else dv
    dk_tgt = _scratch(workspace, regions[R_DK_PART], dtype) if cutlass.const_expr(regions[R_DK_PART] is not None) else dk
    ds = _extent(ds_full, (1, hc, rcap, sqp))

    # STAGE 1: one streaming pass over the packed O and dO -- unless the caller computed delta (external_delta: a producer that
    # already reads O and dO writes rowsum(dO * O) in this kernel's order over the packed tokens, so the launch and the read go).
    if cutlass.const_expr(not external_delta):
        dot_do_o_host(o, do, delta, None, None, DOT_Q_TILE, d, d, DOT_CHUNK_ELEMS, False, False, stream)

    grid_m_kv = -(-skv_env // _THD_KV_BLOCK) * _THD_KV_BLOCK  # the kv envelope's M tiles (dK); dQ's is the padded q envelope (sqp)
    for ci in range(h // hc):
        hb = ci * hc
        # STAGE 2: packed operands, the kv-blocked workspace, this launch's heads; batch_base 0 (one packed batch).
        main(q, do, k, v, dv_k, ds_full, stats, delta, meta, (b, h, hk, sqp, rcap, hc, 1, sq_env, skv_env, units), scale, hb, 0, stream)
        # STAGE 3: the THD arm over the chunk's workspace; the outputs' head slice, every sequence through its own descriptor.
        _stage3_thd(mm_dk, mm_dq, ds, q, k, dk_tgt, dq, hb, hc, group, b, meta, desc3, stream, grid_m_kv, sqp, dq_b_head_group=dq_bhg)

    # STAGE 4: fold the per-Q-head partials onto the KV heads over the packed kv axis (fixed order), rows [0, cu_k[B]) ONLY.
    # The partials past the live total were never written (the kernel's dV and the dK GEMM store through per-sequence clipped
    # descriptors), so an unbounded fold would copy the 0xFF-poisoned workspace (NaN) into the caller's dK / dV capacity tail;
    # the limit is the metadata's cu_k[B] word read on device, so a rebind with new lengths needs no host work.
    if cutlass.const_expr(group > 1):
        dkv_reduce_bounded_host(dk_tgt, dv_k, dk, dv, d, d, group, dtype, False, _window(meta, 0, THD_CU_K_TOTAL_OFF(b), 1), stream)


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
    seq_kv_ptr: Optional[cute.Pointer],
    delta_ptr: Optional[cute.Pointer],
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
    sm_count: cutlass.Constexpr,
    stream: driver.CUstream,
):
    """The per-tensor fp8 row's chain (``sdpa_fp8_backward`` at d = 256 on Rubin).  The two appended pointers (slots 25 / 26 of
    ``prepared_sm107.ROLES_FP8``) are independent plan facts, each None-specialized out of a plan built without its flag:
    ``seq_kv_ptr``, the caller's ``[B]`` int32 per-batch kv lengths (``geometry[25]``) the kernel's padded-mask arm reads per batch
    in place of the uniform ``seq_kv`` fill -- its amax row gate follows the same per-batch length, so a dead or shortened batch
    folds nothing into ``amax_dV`` / ``amax_dP``; ``delta_ptr``, the caller's fp32 ``[B, H, S_q_pad]`` delta (``geometry[26]``) in
    TRUE units, bound AS IS (the kernel reads delta unscaled; the row's own pre-pass is the scaled dot of the e4m3 payloads, so
    a caller's delta is not bitwise it) -- with it the scaled ``dot`` launch and the workspace's ``delta`` region do not exist."""
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
    # The caller's per-batch kv lengths ([B] int32, geometry[25]): the padded mask arm reads them in place of the uniform fill.
    seq_kv_lens = _view(seq_kv_ptr, geometry[25])
    # delta [B, H, S_q_pad] fp32: the caller's (external_delta; no region, no scaled dot) or the region stage 1 fills below.
    external_delta = cutlass.const_expr(delta_ptr is not None)
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
    delta = _view(delta_ptr, geometry[26]) if cutlass.const_expr(external_delta) else _scratch(workspace, regions[R_DELTA], cutlass.Float32)
    desc = _scratch(workspace, regions[R_DESC], cutlass.Int64)
    q_k, do_k, lse_k, k_k, v_k, seq_kv, ds_full = _stage2_inputs(
        q, k, v, do, stats, workspace, regions, config, dtype, ds_dtype, stream, seq_kv_lens, fill_seq_kv=False
    )
    group = h // hk
    # The per-Q-head partials' dtype (``api_dsl_sm107.SdpaBwdDslSm107Fp8._partial_dtype``, the regions were carved for it): fp32 under
    # GQA on the e4m3-dS chain -- the main kernel stores dV_true UNROUNDED (DTYPE_O = FP32) and the dK GEMM's DESCALE epilogue its
    # fp32 true-unit value, so the fold's fixed-order fp32 sum of the group rounds ONCE, like the reference; bf16 at MHA (the fold is
    # a copy + amax -- the pre-fp32 kernels and bits) and on the bf16-dS twin (no SMEM for the fp32 dV staging beside its 96 KiB ring).
    part_dtype = cutlass.Float32 if (ds_fp8 and group > 1) else cutlass.BFloat16
    dv_part = _scratch(workspace, regions[R_FP8_DV_PART], part_dtype)  # [B, kv_rows, H, D] stage 2's per-Q-head dV_true
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
    # ONE setup launch: the uniform kv-length fill (skipped when the caller's per-batch lengths are bound: never written) + the amax resets.
    _fp8_setup(seq_kv if cutlass.const_expr(seq_kv_ptr is None) else None, skv, amax_dq, amax_dk, amax_dv, amax_dp, amax_dv_kernel).launch(
        grid=(1, 1, 1), block=(_THREADS, 1, 1), stream=stream
    )

    # STAGE 1: delta in TRUE units = rowsum(dO8 * O8) * descale_o * descale_dO -- unless the caller computed it (external_delta:
    # the producer of dO writes rowsum(dO * O) in true units, so the launch and the read of O go).
    if cutlass.const_expr(not external_delta):
        dot_do_o_scaled_host(o, do, delta, descale_o, descale_do, DOT_Q_TILE, d, DOT_CHUNK_ELEMS, stream)
    # Per-batch kv lengths under a bottom-right band: the dS zero-fill `_stage2_inputs` skipped runs ONCE here, ahead of the head
    # loop -- this row walks the whole batch in-grid (no batch chunking), so every head chunk sees every batch and the tiles a
    # shorter batch's band does not write hold this fill's zeros, never another batch's dS (the half row, batch-chunked, fills
    # per chunk instead).
    if cutlass.const_expr(zero_ws and seq_kv_ptr is not None):
        _zero_ds(ds_full, config, stream)

    ds = _extent(ds_full, (b, hc, skv, sq))
    if cutlass.const_expr(ds_fp8):
        # The GEMMs' B operands are the caller's e4m3 payloads; dQ lands in the caller's dQ (EPI_QUANT); dK in the caller's dK
        # at MHA (EPI_QUANT, real rows) or in the fp32 TRUE-unit per-Q-head partials under GQA (EPI_DESCALE, folded below).
        dk_tgt = _scratch(workspace, regions[R_FP8_DK_PART], part_dtype) if cutlass.const_expr(group > 1) else dk
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
                # The REAL lengths as TYPED scalars (a nested jit call passes a bare Python int through as an int, and the kernel's
                # host reads them as DSL values): the kv length (the padded arm's bound and the amax row gate) and the q length (the
                # bottom-right diagonal S_kv - S_q and the q-tile trim: a ragged S_q is served exactly); then the per-batch kv lengths
                # (read under the padded arm only: the uniform fill, or the caller's).
                cutlass.Int32(skv),
                cutlass.Int32(sq),
                seq_kv,
                stream,
            )
            # STAGE 3: the fp8 K64 arm over the e4m3 dS and the e4m3 Q / K payloads; the epilogue undoes scale_dP and the payload's scale.
            _stage3(mm_dk, mm_dq, ds, q, k, dk_real, dq, 0, b, hb, hc, group, seq_kv, desc, stream, dk_epi, dq_epi, dq_b_head_group=dq_bhg)
        # STAGE 4: dV always folds + quantizes here (the kernel publishes per-Q-head dV_true); dK only under GQA, where the fp32
        # true-unit partials are summed in fixed order BEFORE the amax fold, the scale and the cast (the backend's order) -- the dV
        # and dK folds in ONE launch.
        if cutlass.const_expr(group > 1):
            fold_quant_pair_host(dv_part, dv, None, scale_dv, amax_dv, dk_tgt, dk, None, scale_dk, amax_dk, d, group, grad_dtype, sm_count, stream)
        else:
            fold_quant_host(dv_part, dv, None, scale_dv, amax_dv, d, group, grad_dtype, sm_count, stream)
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
                cutlass.Int32(skv),
                cutlass.Int32(sq),
                seq_kv,
                stream,
            )
            # STAGE 3 at bf16 over the exact upcasts; the partials still carry descale_q / descale_k.
            _stage3(mm_dk, mm_dq, ds, q_bf16, k_bf16, dk_real, dq_ws, 0, b, hb, hc, group, seq_kv, desc, stream, dq_b_head_group=dq_bhg)
        # STAGE 4: fold (GQA) + the per-tensor FP8 epilogue (descale, amax, scale, cast) into the caller's gradients: dV + dK in one
        # launch (bf16 partials on this twin: its dS ring leaves no SMEM for the fp32 dV staging), then dQ.
        fold_quant_pair_host(dv_part, dv, None, scale_dv, amax_dv, dk_part, dk, descale_q, scale_dk, amax_dk, d, group, grad_dtype, sm_count, stream)
        fold_quant_host(dq_ws, dq, descale_k, scale_dq, amax_dq, d, 1, grad_dtype, sm_count, stream)


@cute.jit
def host_fp8_thd(
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    v_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    do_ptr: cute.Pointer,
    stats_ptr: cute.Pointer,
    dq_ptr: cute.Pointer,
    dk_ptr: cute.Pointer,
    dv_ptr: cute.Pointer,
    seq_q_ptr: cute.Pointer,
    seq_kv_ptr: cute.Pointer,
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
    delta_ptr: Optional[cute.Pointer],
    workspace: cute.Pointer,
    scale_log2: cutlass.Float32,
    scale: cutlass.Float32,
    lens_form: cutlass.Int32,
    main: cutlass.Constexpr,
    mm_dk: cutlass.Constexpr,
    mm_dq: cutlass.Constexpr,
    config: cutlass.Constexpr,
    geometry: cutlass.Constexpr,
    regions: cutlass.Constexpr,
    dtype: cutlass.Constexpr,
    grad_dtype: cutlass.Constexpr,
    sm_count: cutlass.Constexpr,
    stream: driver.CUstream,
):
    """The fp8 row's THD / varlen chain (``SdpaBwdDslSm107Fp8(thd=True)``): PACKED ``[1, T, H, D]`` e4m3 operands at the plan's
    token capacities, a kv-BLOCKED dS workspace (e4m3, or bf16 on the twin), the per-sequence lengths from the caller's two length
    tensors, the twelve scalars and the requested amax, and -- APPENDED, optional -- the caller's ``delta`` (``external_delta=True``:
    the PACKED head-major ``[1, H, ceil128(T_q)]`` fp32 delta in TRUE units, read UNSCALED like the dense row's -- nobody applies
    ``descale_o * descale_dO`` to it; with it bound the scaled ``dot`` launch and the ``delta`` region do not exist).  A SIBLING of
    :func:`host_fp8` with the THD ABI of :func:`host_f16_thd` (the two length operands right after the nine tensors, the delta after
    the amax, ``lens_form`` after ``scale``), its own frame and its own cache key -- the dense artifact is untouched.

        setup    thd_bwd_setup_host(kv_blocked=True): [seq_kv_lens | cu_q | cu_k | batch_remap | live | ctr | row_off] with the
                 row offsets over the KV lengths at the kernel's 256-row block (the plan reserves the main kernel's (5 + B)
                 tensor maps after it, in the same region); ONCE per execute
        fill     the dS workspace zeroed ONCE per execute ONLY for the untrimmed / wide-tile twins
                 (``api_dsl_sm107._stage3_thd_needs_zero_fill``); the byte count is the dS element size's -- e4m3 = 1, not the io
                 itemsize the half row's chain uses
        amax     the accumulators reset (the C++ node's semantics; ``_fp8_setup`` without the dense chain's kv-length fill)
        delta    the scaled dot over the packed O / dO -> [1, H, ceil128(T_q)] in TRUE units (zeros past T_q) -- unless the caller
                 computed it (external_delta)
        per head chunk: the main kernel (its own setup launch clamps the five input descriptors to the live packed totals, emits
                 the per-sequence clipped dV descriptors and resets live / ctr for THIS launch's heads; amax_dP and the kernel's
                 dV amax fold the LIVE region only: kv rows below s_kv[b], q columns below s_q[b], live units), then dK / dQ
                 through the THD stage-3 arm with the fp8 K64 epilogue (``_stage3_thd(dk_epi, dq_epi)``: dQ quantized in place
                 with amax_dQ over the live tiles; dK likewise at MHA, fp32 true-unit partials under GQA)
        fold     fold_quant bounded ON DEVICE at the live kv total cu_k[B] (dV always; dK under GQA, the two in ONE launch over fp32
                 partials the fold rounds once) -- the partial rows past the live total were never written (per-sequence clipped
                 stores), so an unbounded fold would copy the 0xFF-poisoned capacity tail (NaN) into the caller's gradients AND into
                 amax_dV / amax_dK; the bf16-dS twin bounds its folds the same way (dV + dK, then dQ at the live q total cu_q[B])

    Every amax is the max over the packed LIVE region (dead units, pad rows, pad columns and the capacity tail excluded), one
    ``scale_dP`` per packed batch.  ``lens_form`` bit 0 / 1 = the Q / KV length tensor is a ``(B+1,)`` prefix (``bind()`` derives
    it from numel); the setup kernel branches on it before reading the prefix tail, so both tensors are viewed ``(B+1,)``.
    """
    b, h, hk, d, t_q, t_kv, sqp, rcap, hc, zero_ws, bpe_ds, units, sq_env, skv_env, dq_bhg = config
    q = _view(q_ptr, geometry[0])  # packed [1, T_q, H_q, D] e4m3
    k = _view(k_ptr, geometry[1])  # packed [1, T_kv, H_kv, D]
    v = _view(v_ptr, geometry[2])
    o = _view(o_ptr, geometry[3])
    do = _view(do_ptr, geometry[4])
    stats = _view(stats_ptr, geometry[5])  # (T_q, H_q) token-major or (1, H_q, head_stride) head-major, the forward's packing
    dq = _view(dq_ptr, geometry[6])  # packed [1, T_q, H_q, D] in the gradient dtype
    dk = _view(dk_ptr, geometry[7])  # packed [1, T_kv, H_kv, D]
    dv = _view(dv_ptr, geometry[8])
    q_lens = _view(seq_q_ptr, ((b + 1,), (1,)))
    kv_lens = _view(seq_kv_ptr, ((b + 1,), (1,)))
    scalar = ((1,), (1,))
    descale_q, descale_k, descale_v = _view(descale_q_ptr, scalar), _view(descale_k_ptr, scalar), _view(descale_v_ptr, scalar)
    descale_s, scale_s = _view(descale_s_ptr, scalar), _view(scale_s_ptr, scalar)
    descale_o, descale_do = _view(descale_o_ptr, scalar), _view(descale_do_ptr, scalar)
    descale_dp, scale_dp = _view(descale_dp_ptr, scalar), _view(scale_dp_ptr, scalar)
    scale_dq, scale_dk, scale_dv = _view(scale_dq_ptr, scalar), _view(scale_dk_ptr, scalar), _view(scale_dv_ptr, scalar)
    ds_fp8 = bpe_ds == 1
    ds_dtype = cutlass.Float8E4M3FN if cutlass.const_expr(ds_fp8) else cutlass.BFloat16
    # delta [1, H, ceil128(T_q)] fp32 in TRUE units, PACKED head-major: the caller's (external_delta; no region, no scaled dot) or
    # the region stage 1 fills below -- the dense host's rule over the packed token capacity.
    external_delta = cutlass.const_expr(delta_ptr is not None)
    delta = _view(delta_ptr, _thd_delta_geometry(h, t_q)) if cutlass.const_expr(external_delta) else _scratch(workspace, regions[R_DELTA], cutlass.Float32)
    meta = _scratch(workspace, regions[R_SEQ_KV], cutlass.Int32)  # the metadata words + the main kernel's tensor maps
    desc3 = _scratch(workspace, regions[R_DESC], cutlass.Int64)  # stage 3's (B + 1) descriptors, patched per GEMM launch
    ds_full = _scratch(workspace, regions[R_DS], ds_dtype)  # [1, hc, R_kv_cap, S_q_pad]
    thd_bwd_setup_host(meta, q_lens, kv_lens, lens_form, hc, b, _THD_KV_BLOCK, _THD_KV_BLOCK, units, stream, kv_blocked=True)
    if cutlass.const_expr(zero_ws):
        n16 = hc * rcap * sqp * bpe_ds // 16
        _zero_bytes(ds_full, n16).launch(grid=(min((n16 + _THREADS - 1) // _THREADS, 4096), 1, 1), block=(_THREADS, 1, 1), stream=stream)
    group = h // hk
    # fp32 partials under GQA on the e4m3 chain (the fold rounds once), bf16 at MHA / on the twin -- as on the dense host.
    part_dtype = cutlass.Float32 if (ds_fp8 and group > 1) else cutlass.BFloat16
    dv_part = _scratch(workspace, regions[R_FP8_DV_PART], part_dtype)  # [1, T_kv_cap, H, D] stage 2's per-Q-head dV_true
    amax_scratch = _scratch(workspace, regions[R_AMAX_SCRATCH], cutlass.Float32)  # [8]
    amax_dq = _view(amax_dq_ptr, scalar)
    amax_dk = _view(amax_dk_ptr, scalar)
    amax_dv = _view(amax_dv_ptr, scalar)
    amax_dp = _view(amax_dp_ptr, scalar) if cutlass.const_expr(amax_dp_ptr is not None) else _slot(amax_scratch, AMAX_SLOT_DP)
    amax_dv_kernel = _slot(amax_scratch, AMAX_SLOT_DV_KERNEL)
    # The amax resets alone (no kv-length fill under THD: the metadata buffer carries the per-sequence lengths).
    _fp8_setup(None, 0, amax_dq, amax_dk, amax_dv, amax_dp, amax_dv_kernel).launch(grid=(1, 1, 1), block=(_THREADS, 1, 1), stream=stream)
    # The live packed totals, as device words: the fold passes stop there (nothing past them is written into the caller's gradients).
    live_kv = _window(meta, 0, THD_CU_K_TOTAL_OFF(b), 1)
    live_q = _window(meta, 0, THD_CU_Q_TOTAL_OFF(b), 1)

    # STAGE 1: delta in TRUE units over the packed O / dO (one streaming pass; rows past the packed capacity read as 0) -- unless the
    # caller computed it (external_delta: bound AS IS, in TRUE units; the kernel multiplies by attn_scale * descale_s only).
    if cutlass.const_expr(not external_delta):
        dot_do_o_scaled_host(o, do, delta, descale_o, descale_do, DOT_Q_TILE, d, DOT_CHUNK_ELEMS, stream)

    ds = _extent(ds_full, (1, hc, rcap, sqp))
    grid_m_kv = -(-skv_env // _THD_KV_BLOCK) * _THD_KV_BLOCK  # the kv envelope's M tiles (dK); dQ's is the padded q envelope (sqp)
    problem = (b, h, hk, sqp, rcap, hc, sq_env, skv_env, units)
    if cutlass.const_expr(ds_fp8):
        # The GEMMs' B operands are the caller's packed e4m3 payloads (through the template's packed-total-clamped B descriptor);
        # dQ lands in the caller's packed dQ (EPI_QUANT); dK in the caller's packed dK at MHA (EPI_QUANT) or in the fp32 TRUE-unit
        # per-Q-head partials under GQA (EPI_DESCALE, folded below).
        dk_tgt = _scratch(workspace, regions[R_FP8_DK_PART], part_dtype) if cutlass.const_expr(group > 1) else dk
        dk_epi = (descale_dp, descale_q, None, None) if cutlass.const_expr(group > 1) else (descale_dp, descale_q, scale_dk, amax_dk)
        dq_epi = (descale_dp, descale_k, scale_dq, amax_dq)
        for ci in range(h // hc):
            hb = ci * hc
            # STAGE 2: packed operands, the kv-blocked workspace, this launch's heads; the metadata + maps buffer in the lengths slot.
            main(
                q,
                do,
                k,
                v,
                dv_part,
                ds_full,
                stats,
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
                problem,
                scale,
                scale_log2,
                hb,
                cutlass.Int32(skv_env),
                cutlass.Int32(sq_env),
                meta,
                stream,
            )
            # STAGE 3: the THD arm of the fp8 K64 renderings over the chunk's blocked workspace; every sequence through its own descriptor.
            _stage3_thd(mm_dk, mm_dq, ds, q, k, dk_tgt, dq, hb, hc, group, b, meta, desc3, stream, grid_m_kv, sqp, dq_bhg, dk_epi, dq_epi)
        # STAGE 4: dV always folds + quantizes here; dK under GQA (with dV in ONE launch) -- both bounded at the live kv total.
        if cutlass.const_expr(group > 1):
            fold_quant_pair_host(dv_part, dv, None, scale_dv, amax_dv, dk_tgt, dk, None, scale_dk, amax_dk, d, group, grad_dtype, sm_count, stream, live_kv)
        else:
            fold_quant_host(dv_part, dv, None, scale_dv, amax_dv, d, group, grad_dtype, sm_count, stream, live_kv)
    else:
        dk_part = _scratch(workspace, regions[R_FP8_DK_PART], cutlass.BFloat16)  # [1, T_kv_cap, H, D] stage 3's per-Q-head dS . Q8
        dq_ws = _scratch(workspace, regions[R_DQ_WS], cutlass.BFloat16)  # [1, T_q_cap, H, D] stage 3's dS^T . K8 (descale_k pending)
        q_bf16 = _scratch(workspace, regions[R_Q_BF16], cutlass.BFloat16)  # [1, T_q_cap, H, D]
        k_bf16 = _scratch(workspace, regions[R_K_BF16], cutlass.BFloat16)  # [1, T_kv_cap, H_kv, D]
        # The bf16 GEMM operands: e4m3 -> bf16 is exact, over the COMPACT packed rows (the adapter requires token stride == H * D
        # on this twin); a NaN capacity tail copies as NaN and stays out of reach of the GEMMs through the clamped B descriptor.
        _cast_fp8_to_bf16(q, q_bf16, t_q * h * d).launch(grid=_grid_16b(q_bf16, 2), block=(_THREADS, 1, 1), stream=stream)
        _cast_fp8_to_bf16(k, k_bf16, t_kv * hk * d).launch(grid=_grid_16b(k_bf16, 2), block=(_THREADS, 1, 1), stream=stream)
        for ci in range(h // hc):
            hb = ci * hc
            main(
                q,
                do,
                k,
                v,
                dv_part,
                ds_full,
                stats,
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
                problem,
                scale,
                scale_log2,
                hb,
                cutlass.Int32(skv_env),
                cutlass.Int32(sq_env),
                meta,
                stream,
            )
            # STAGE 3 at bf16 over the exact upcasts; the partials still carry descale_q / descale_k.
            _stage3_thd(mm_dk, mm_dq, ds, q_bf16, k_bf16, dk_part, dq_ws, hb, hc, group, b, meta, desc3, stream, grid_m_kv, sqp, dq_bhg)
        # STAGE 4: fold (GQA) + the per-tensor FP8 epilogue into the caller's packed gradients, each bounded at its live total: dV + dK
        # in one launch at the live kv total, dQ at the live q total.
        fold_quant_pair_host(dv_part, dv, None, scale_dv, amax_dv, dk_part, dk, descale_q, scale_dk, amax_dk, d, group, grad_dtype, sm_count, stream, live_kv)
        fold_quant_host(dq_ws, dq, descale_k, scale_dq, amax_dq, d, 1, grad_dtype, sm_count, stream, live_q)


# --- the MXFP8 row (the block-scaled P-b chain, the default, and the bf16-dS P-c twin) ----------------------------------------


@cute.jit
def host_mxfp8(
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    v_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    do_ptr: cute.Pointer,
    stats_ptr: cute.Pointer,
    dq_ptr: cute.Pointer,
    dk_ptr: cute.Pointer,
    dv_ptr: cute.Pointer,
    q_T_ptr: cute.Pointer,
    k_T_ptr: cute.Pointer,
    do_T_ptr: cute.Pointer,
    do_f16_ptr: cute.Pointer,
    sf_q_ptr: cute.Pointer,
    sf_q_T_ptr: cute.Pointer,
    sf_k_ptr: cute.Pointer,
    sf_k_T_ptr: cute.Pointer,
    sf_v_ptr: cute.Pointer,
    sf_do_ptr: cute.Pointer,
    sf_do_T_ptr: cute.Pointer,
    seq_kv_ptr: Optional[cute.Pointer],
    delta_ptr: Optional[cute.Pointer],
    workspace: cute.Pointer,
    scale_log2: cutlass.Float32,
    scale: cutlass.Float32,
    main: cutlass.Constexpr,
    mm_dk: cutlass.Constexpr,
    mm_dq: cutlass.Constexpr,
    config: cutlass.Constexpr,
    geometry: cutlass.Constexpr,
    regions: cutlass.Constexpr,
    stage_sf_pads: cutlass.Constexpr,
    stream: driver.CUstream,
    ds_sf_policy: cutlass.Constexpr = DS_SF_POLICY_DEFAULT,
):
    """The MXFP8 row's chain (``sdpa_mxfp8_backward`` at d = 256 on Rubin; ``sm107/bprop_d256_mxfp8.py``) under the dS policy
    ``ds_sf_policy`` (appended, default ``DS_SF_POLICY_DEFAULT``; ``DS_SF_P_B`` = the block-scaled chain that ships, ``DS_SF_P_C`` = the
    bf16-dS oracle twin):

        stage 0  zero-padded staging of Q / dO / dO_T / K / V (+inf LSE) as the other rows, PLUS the zero-filled SF pad slabs
                 (``_pad_sf_atoms``: the producer's pad bytes are undefined and the kernel reads them -- a NaN there is NaN dV)
        stage 1  delta = rowsum(dO_f16 * o_f16) in TRUE units (the half row's ``dot_do_o`` arm)
        stage 2  per head chunk: the main kernel -- S = K.Q^T and dP = V.dO^T with the block scales dequantizing IN the MMA,
                 P = exp2(S * scale * log2e - LSE * log2e) from the exact fp32 Stats, e4m3(P * 2^8) into the TMEM P ring, dV += P.dO_T
                 (bf16 per-Q-head partials, TRUE units), dS = scale * P (dP - delta) from the fp32 P -> P-c: the bf16 kv-major
                 workspace; P-b: two e4m3 payloads (ds_dk scaled per 32-q block, ds_dq per 32-kv block) + their E8M0 atoms
        stage 3  dK = dS . Q_T and dQ = dS^T . K_T: P-c the bf16 renderings over the EXACTLY dequantized bf16 q_T / k_T
                 (``dequant_mxfp8_to_bf16_host``, the columnwise payloads and their SF); P-b the block-scale GEMM arm over the e4m3
                 payloads + atoms and the columnwise q_T / k_T with their own scale factors (``_stage3_block_scale``: no dequant pass,
                 the MMA dequantizes; a ragged S_q / S_kv re-stages the q_T / k_T scale factors with their pad groups zeroed)
        stage 4  GQA fold of the per-Q-head dK / dV partials (``dkv_reduce``, fixed order; the dK partials fp32 on the block-scaled chain,
                 so dK is rounded ONCE, the dV partials bf16); real-row copy-out under kv padding

    No per-tensor scale, no amax (a graph requesting amax outputs is declined, typed).  Gradients are bf16 (the bf16 GEMM writes its io dtype;
    the block-scale dK GEMM stores its fp32 accumulator under GQA -- the partial the fold rounds -- and the caller's bf16 dK at MHA).

    The two appended pointers (slots 20 / 21 of ``prepared_sm107.ROLES_MXFP8``) are independent plan facts, each None-specialized
    out of a plan built without its flag: ``seq_kv_ptr``, the caller's ``[B]`` int32 per-batch kv lengths (``geometry[20]``) the
    kernel's padded-mask arm reads per batch in place of the uniform fill (a bottom-right band then zero-fills the dS payloads
    ONCE ahead of the head loop: this row walks the whole batch in-grid); ``delta_ptr``, the caller's fp32 ``[B, H, S_q_pad]``
    delta (``geometry[21]``) -- bitwise the row's own ``dot`` over the ``o_f16`` / ``dO_f16`` ports when the producer forms it in that
    order; with it bound the ``dot`` launch and the workspace's ``delta`` region do not exist.  The pad rows ``[S_q, S_q_pad)`` of a
    caller's delta must be finite zeros: under P-b a 32-element dS block straddling the q pad reads them (a NaN or non-zero
    there corrupts the block's REAL columns' scale), and no host check can see device data."""
    b, h, hk, d, sq, skv, sqp, skvp, bc, hc, zero_ws, itemsize, bpe_ds, dq_bhg = config
    fp8 = cutlass.Float8E4M3FN
    half = cutlass.BFloat16
    # The dS policy is a compile-time fact of the artifact (folded into its cache key): the dS workspace dtype, the operands the
    # kernel binds and the stage-3 arm all follow it; P-c traces exactly the chain it did before P-b existed.
    p_b = ds_sf_policy == DS_SF_P_B
    ds_dtype = fp8 if cutlass.const_expr(p_b) else half
    q = _view(q_ptr, geometry[0])
    k = _view(k_ptr, geometry[1])
    v = _view(v_ptr, geometry[2])
    o_f16 = _view(o_ptr, geometry[3])
    do = _view(do_ptr, geometry[4])  # the ROWWISE e4m3 dO (the dP operand)
    stats = _view(stats_ptr, geometry[5])
    dq = _view(dq_ptr, geometry[6])
    dk = _view(dk_ptr, geometry[7])
    dv = _view(dv_ptr, geometry[8])
    q_T = _view(q_T_ptr, geometry[9])
    k_T = _view(k_T_ptr, geometry[10])
    do_T = _view(do_T_ptr, geometry[11])  # the COLUMNWISE e4m3 dO (the dV operand)
    do_f16 = _view(do_f16_ptr, geometry[12])
    sf_q = _view(sf_q_ptr, geometry[13])
    sf_q_T = _view(sf_q_T_ptr, geometry[14])
    sf_k = _view(sf_k_ptr, geometry[15])
    sf_k_T = _view(sf_k_T_ptr, geometry[16])
    sf_v = _view(sf_v_ptr, geometry[17])
    sf_do = _view(sf_do_ptr, geometry[18])
    sf_do_T = _view(sf_do_T_ptr, geometry[19])
    seq_kv_lens = _view(seq_kv_ptr, geometry[20])  # the caller's per-batch kv lengths ([B] int32), or None
    external_delta = cutlass.const_expr(delta_ptr is not None)
    # delta [B, H, S_q_pad] fp32: the caller's (external_delta -- no region, no dot) or the region stage 1 fills below.
    delta = _view(delta_ptr, geometry[21]) if cutlass.const_expr(external_delta) else _scratch(workspace, regions[R_DELTA], cutlass.Float32)
    desc = _scratch(workspace, regions[R_DESC], cutlass.Int64)
    q_k, do_k, lse_k, k_k, v_k, seq_kv, ds_full = _stage2_inputs(q, k, v, do, stats, workspace, regions, config, fp8, ds_dtype, stream, seq_kv_lens)
    group = h // hk
    q_padded = regions[R_Q_PAD] is not None
    kv_padded = regions[R_K_PAD] is not None
    t_kv_sf = (skv + SF_ATOM_ROWS - 1) // SF_ATOM_ROWS  # the kv SF tensors' OWN tile count (ceil128(S_kv) / 128), the dQ GEMM's K tiles
    # STAGE 0 (MXFP8 half): dO_T pads like dO; the Q-side SF slabs are re-staged with the rows / groups past S_q zeroed (they sit at
    # the kernel's q pad already: S_q_pad == ceil128(S_q)); the kv-side slabs grow to the kernel's 256-row pad with rows past S_kv zeroed.
    do_T_k, sf_q_k, sf_do_k, sf_do_T_k = do_T, sf_q, sf_do, sf_do_T
    t_q = sqp // SF_ATOM_ROWS
    if cutlass.const_expr(q_padded):
        do_T_k = _scratch(workspace, regions[R_DOT_PAD], fp8)
        _pad_copy(do_T, do_T_k, itemsize, stream)
        if cutlass.const_expr(stage_sf_pads):
            sf_q_k = _scratch(workspace, regions[R_SF_Q_PAD], cutlass.Uint8)
            sf_do_k = _scratch(workspace, regions[R_SF_DO_PAD], cutlass.Uint8)
            sf_do_T_k = _scratch(workspace, regions[R_SF_DOT_PAD], cutlass.Uint8)
            _pad_sf(sf_q, sf_q_k, sq, b * h, t_q, t_q, False, stream)
            _pad_sf(sf_do, sf_do_k, sq, b * h, t_q, t_q, False, stream)
            _pad_sf(sf_do_T, sf_do_T_k, sq, b * h, t_q, t_q, True, stream)
    sf_k_k, sf_v_k = sf_k, sf_v
    if cutlass.const_expr(kv_padded):
        # The kv slab MUST grow to the kernel's 256-row extent even in the RED twin (the descriptors span it, and a tile past the
        # caller's tensor is out-of-bounds memory, not a numerics probe); the switch removes only the zeroing of the producer's own pad
        # rows -- the RED twin copies them verbatim by declaring every row inside the producer's ceil128 extent "real".
        t_kv_src = (skv + SF_ATOM_ROWS - 1) // SF_ATOM_ROWS
        t_kv_dst = skvp // SF_ATOM_ROWS
        sf_k_k = _scratch(workspace, regions[R_SF_K_PAD], cutlass.Uint8)
        sf_v_k = _scratch(workspace, regions[R_SF_V_PAD], cutlass.Uint8)
        kv_real = skv if cutlass.const_expr(stage_sf_pads) else t_kv_src * SF_ATOM_ROWS
        _pad_sf(sf_k, sf_k_k, kv_real, b * hk, t_kv_src, t_kv_dst, False, stream)
        _pad_sf(sf_v, sf_v_k, kv_real, b * hk, t_kv_src, t_kv_dst, False, stream)

    # STAGE 1: delta in TRUE units over the half-precision O / dO (zeros past S_q: the kernel's finite-delta-pad ABI) -- unless the
    # caller computed it (external_delta: the same layout, the same finite-pad obligation, now the caller's).
    if cutlass.const_expr(not external_delta):
        dot_do_o_host(o_f16, do_f16, delta, None, None, DOT_Q_TILE, d, d, DOT_CHUNK_ELEMS, False, False, stream)
    # Per-batch kv lengths under a bottom-right band: the dS zero-fill `_stage2_inputs` skipped runs ONCE here (no batch chunking on
    # this row: every head chunk sees every batch).  Under P-b the second payload and the two atom tensors are zeroed below under
    # the same `zero_ws`, lengths bound or not.
    if cutlass.const_expr(zero_ws and seq_kv_ptr is not None):
        _zero_ds(ds_full, config, stream)

    if cutlass.const_expr(p_b):
        # P-b.  The kernel's second payload and the two E8M0 atom tensors (the first payload, ds_dk, is ``ds_full`` = R_DS); under
        # a mask geometry that reads tiles the kernel never writes (``zero_ws``) they are zeroed like the first payload -- a zero
        # SF byte is the scale 2^-127 and 0 x 2^-127 is exactly 0, so an unwritten tile contributes nothing to dK / dQ.
        ds_dq_full = _scratch(workspace, regions[R_MX_DS_DQ], fp8)
        sf_ds_dk = _scratch(workspace, regions[R_MX_SF_DS_DK], cutlass.Uint8)
        sf_ds_dq = _scratch(workspace, regions[R_MX_SF_DS_DQ], cutlass.Uint8)
        if cutlass.const_expr(zero_ws):
            n16 = bc * hc * skvp * sqp * bpe_ds // 16
            _zero_bytes(ds_dq_full, n16).launch(grid=(min((n16 + _THREADS - 1) // _THREADS, 4096), 1, 1), block=(_THREADS, 1, 1), stream=stream)
            n16_sf = bc * hc * (skvp // SF_ATOM_ROWS) * (sqp // SF_ATOM_ROWS) * SF_ATOM_BYTES // 16
            _zero_bytes(sf_ds_dk, n16_sf).launch(grid=(min((n16_sf + _THREADS - 1) // _THREADS, 4096), 1, 1), block=(_THREADS, 1, 1), stream=stream)
            _zero_bytes(sf_ds_dq, n16_sf).launch(grid=(min((n16_sf + _THREADS - 1) // _THREADS, 4096), 1, 1), block=(_THREADS, 1, 1), stream=stream)
        # The stage-3 B operands are the COLUMNWISE e4m3 q_T / k_T themselves; their scale factors reach the block-scale MMA as
        # WHOLE F8_128x4 atoms (the dequant pass that read real extents only is gone), so where S_q / S_kv is ragged the producer's
        # undefined pad groups are re-staged zeroed (a 0xFF there is 0 x NaN in the MMA), at the SF tensor's own ceil128 tile count.
        sf_qT_k, sf_kT_k = sf_q_T, sf_k_T
        if cutlass.const_expr(q_padded and stage_sf_pads):
            sf_qT_k = _scratch(workspace, regions[R_MX_SF_QT_PAD], cutlass.Uint8)
            _pad_sf(sf_q_T, sf_qT_k, sq, b * h, t_q, t_q, True, stream)
        if cutlass.const_expr(kv_padded and stage_sf_pads):
            sf_kT_k = _scratch(workspace, regions[R_MX_SF_KT_PAD], cutlass.Uint8)
            _pad_sf(sf_k_T, sf_kT_k, skv, b * hk, t_kv_sf, t_kv_sf, True, stream)
        sf_qT_planes = _sf_planes_view(sf_qT_k, d // SF_ATOM_ROWS, t_q, h, b)
        sf_kT_planes = _sf_planes_view(sf_kT_k, d // SF_ATOM_ROWS, t_kv_sf, hk, b)
    else:
        # The stage-3 B operands: the COLUMNWISE q_T / k_T (blocks along the contraction axis) dequantized EXACTLY to bf16.
        q_T_bf16 = _scratch(workspace, regions[R_QT_BF16], half)  # [B, S_q, H, D]
        k_T_bf16 = _scratch(workspace, regions[R_KT_BF16], half)  # [B, S_kv, H_kv, D]
        dequant_mxfp8_to_bf16_host(q_T, sf_q_T, q_T_bf16, True, stream)
        dequant_mxfp8_to_bf16_host(k_T, sf_k_T, k_T_bf16, True, stream)

    # stage 2's dV per Q head (bf16, the kernel's epilogue dtype): the caller's dV only when MHA and no kv padding; stage 3's dK per Q
    # head: the caller's dK when MHA, else the partial in the dtype the adapter carved it (``SdpaBwdDslSm107Mxfp8._dk_part_fp32``) --
    # DERIVED here from the two facts the host already holds, the policy and the group, never a second flag: fp32 on the block-scaled
    # chain, where the dK GEMM's EPI_NONE store is the true-unit fp32 accumulator (the MMA dequantizes) and the fold's fixed-order fp32
    # sum of the group rounds ONCE, like the reference; bf16 on the bf16-dS twin (its renderings store the io dtype).  An fp32-carved
    # region viewed bf16 would read half the region -- a wrong dK, no crash -- which is why the view keys on the carve's own facts.
    dk_part_dtype = cutlass.Float32 if (p_b and group > 1) else half
    dv_k = _scratch(workspace, regions[R_MX_DV_PART], half) if cutlass.const_expr(regions[R_MX_DV_PART] is not None) else dv
    dk_tgt = _scratch(workspace, regions[R_MX_DK_PART], dk_part_dtype) if cutlass.const_expr(regions[R_MX_DK_PART] is not None) else dk
    dk_real = _extent(dk_tgt, (b, skv, h, d))
    ds = _extent(ds_full, (b, hc, skv, sq))
    for ci in range(h // hc):
        hb = ci * hc
        # STAGE 2 (whole batch in-grid; head_base walks the chunks).  seqlen_kv_real / seqlen_q_real are the REAL lengths: the
        # padded-kv mask arm's bound and the MASK_Q_PAD band (both fold out when the extents are tile multiples).  The dS operands
        # follow the policy (the kernel's Launch ABI, ONE positional shape): P-c binds the bf16 ds_ws and None for the four
        # appended operands; P-b binds None for ds_ws and the two e4m3 payloads + two atom tensors.  ``stream`` is the LAST
        # positional and the appended operands precede it -- the ABI appends, so a caller that forgets them hands the stream to
        # ``ds_dk`` and launches with none.
        if cutlass.const_expr(p_b):
            main(
                q_k,
                k_k,
                v_k,
                do_k,
                do_T_k,
                dv_k,
                None,
                lse_k,
                delta,
                sf_q_k,
                sf_k_k,
                sf_v_k,
                sf_do_k,
                sf_do_T_k,
                (b, h, hk, sqp, skvp, hc),
                scale,
                scale_log2,
                hb,
                cutlass.Int32(skv),  # the REAL lengths as typed scalars (a nested jit call passes a bare Python int through as an int)
                cutlass.Int32(sq),
                ds_full,
                ds_dq_full,
                sf_ds_dk,
                sf_ds_dq,
                seq_kv,  # the per-batch kv lengths (read under the padded arm only: the uniform fill, or the caller's)
                stream,
            )
            # STAGE 3: the block-scale arm over the e4m3 payloads + atoms and the columnwise q_T / k_T + their SF; TRUE-unit bf16 out.
            # dQ launches the way its rendering indexes B (`dq_bhg` = the record's b_head_group, copied off it at compile): ONE launch
            # per head chunk at the GQA group (B and its scale-factor descriptor indexed by `h // group`), one per group member at 1.
            _stage3_block_scale(
                mm_dk,
                mm_dq,
                ds,
                _extent(ds_dq_full, (b, hc, skv, sq)),
                sf_ds_dk,
                sf_ds_dq,
                q_T,
                sf_qT_planes,
                k_T,
                sf_kT_planes,
                dk_real,
                dq,
                0,
                b,
                hb,
                hc,
                group,
                seq_kv,
                desc,
                stream,
                dq_b_head_group=dq_bhg,
            )
        else:
            main(
                q_k,
                k_k,
                v_k,
                do_k,
                do_T_k,
                dv_k,
                ds_full,
                lse_k,
                delta,
                sf_q_k,
                sf_k_k,
                sf_v_k,
                sf_do_k,
                sf_do_T_k,
                (b, h, hk, sqp, skvp, hc),
                scale,
                scale_log2,
                hb,
                cutlass.Int32(skv),
                cutlass.Int32(sq),
                None,
                None,
                None,
                None,
                seq_kv,
                stream,
            )
            # STAGE 3 at bf16 over the dequantized columnwise operands; the outputs are TRUE-unit bf16.
            _stage3(mm_dk, mm_dq, ds, q_T_bf16, k_T_bf16, dk_real, dq, 0, b, hb, hc, group, seq_kv, desc, stream, dq_b_head_group=dq_bhg)

    # STAGE 4: fold the per-Q-head partials onto the KV heads (fixed order); copy real rows out of a padded staging.
    if cutlass.const_expr(group > 1):
        dk_out = _scratch(workspace, regions[R_MX_DK_FOLD], half) if cutlass.const_expr(kv_padded) else dk
        dv_out = _scratch(workspace, regions[R_MX_DV_FOLD], half) if cutlass.const_expr(kv_padded) else dv
        dkv_reduce_host(dk_tgt, dv_k, dk_out, dv_out, d, d, group, half, False, stream)
        if cutlass.const_expr(kv_padded):
            _pad_copy(dk_out, dk, 2, stream)
            _pad_copy(dv_out, dv, 2, stream)
    elif cutlass.const_expr(kv_padded):
        _pad_copy(dv_k, dv, 2, stream)


@cute.jit
def host_mxfp8_thd(
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    v_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    do_ptr: cute.Pointer,
    stats_ptr: cute.Pointer,
    dq_ptr: cute.Pointer,
    dk_ptr: cute.Pointer,
    dv_ptr: cute.Pointer,
    seq_q_ptr: cute.Pointer,
    seq_kv_ptr: cute.Pointer,
    q_T_ptr: cute.Pointer,
    k_T_ptr: cute.Pointer,
    do_T_ptr: cute.Pointer,
    do_f16_ptr: cute.Pointer,
    sf_q_ptr: cute.Pointer,
    sf_q_T_ptr: cute.Pointer,
    sf_k_ptr: cute.Pointer,
    sf_k_T_ptr: cute.Pointer,
    sf_v_ptr: cute.Pointer,
    sf_do_ptr: cute.Pointer,
    sf_do_T_ptr: cute.Pointer,
    delta_ptr: Optional[cute.Pointer],
    workspace: cute.Pointer,
    scale_log2: cutlass.Float32,
    scale: cutlass.Float32,
    lens_form: cutlass.Int32,
    sf_tiles_q: cutlass.Int32,
    sf_tiles_kv: cutlass.Int32,
    main: cutlass.Constexpr,
    mm_dk: cutlass.Constexpr,
    mm_dq: cutlass.Constexpr,
    config: cutlass.Constexpr,
    geometry: cutlass.Constexpr,
    regions: cutlass.Constexpr,
    stage_sf_pads: cutlass.Constexpr,
    stream: driver.CUstream,
    ds_sf_policy: cutlass.Constexpr = DS_SF_POLICY_DEFAULT,
):
    """The MXFP8 row's THD / varlen chain (``SdpaBwdDslSm107Mxfp8(thd=True)``) under the dS policy ``ds_sf_policy``: PACKED
    ``[1, T, H, D]`` e4m3 payloads (q / k / v / dO and the transposed-quantization q_T / k_T / dO_T) and bf16 ``o_f16`` / ``dO_f16`` at
    the plan's token capacities, the per-sequence lengths from the caller's two length tensors, and the seven scale-factor tensors
    PACKED per-sequence-TILE-padded (the forward's convention: per head, every sequence's ``ceil(s_b / 128)`` tiles in cu_seqlens
    order, 1024 B per (head, tile); the columnwise ones with both D planes of a (head, tile) contiguous) whose live tile counts
    ``sf_tiles_q`` / ``sf_tiles_kv`` the binder derived from the bound buffers' byte sizes for THIS call, and -- APPENDED, optional --
    the caller's ``delta`` (``external_delta=True``: the PACKED head-major ``[1, H, ceil128(T_q)]`` fp32 delta in TRUE units, bitwise
    the row's own ``dot`` over the packed ``o_f16`` / ``dO_f16`` ports when the producer reproduces its order; with it bound the ``dot``
    launch and the ``delta`` region do not exist).  A SIBLING of :func:`host_mxfp8` with the THD ABI of :func:`host_fp8_thd` (the delta
    after the seven scale-factor blobs) plus the two tile counts after ``lens_form``, its own frame and cache key.

        setup    thd_bwd_setup_host(kv_blocked=True, sf_meta_t=): [seq_kv_lens | cu_q | cu_k | batch_remap | live | ctr | row_off] with
                 the row offsets over the KV lengths at the kernel's 256-row block (the plan reserves the main kernel's (10 + B) tensor
                 maps after it: five clamped payload maps, five clamped scale-factor maps, B clipped dV maps), PLUS the per-sequence SF
                 TILE prefixes [cu_sf_q(B+1) | cu_sf_k(B+1)] into the ``sf_meta`` region (``config_sm100.STAGE3_THD_SF_*``); ONCE per execute
        fill     the dS payloads (and, under P-b, the atoms) zeroed ONCE per execute ONLY for the untrimmed / wide-tile twins
                 (``api_dsl_sm107._stage3_thd_needs_zero_fill``); the byte count is the dS element size's (e4m3 under P-b, bf16 under P-c)
        SF pads  the five scale-factor tensors whose PAD positions are read -- sf_v / sf_do / sf_do_T by the main kernel (dP / dV), and
                 under P-b sf_q_T / sf_k_T by the block-scale GEMMs (whole atoms) -- re-staged per execute into packed staging copies
                 with every byte scaling a position at or past its sequence's length zeroed (``thd_helpers.pad_sf_atoms_thd_host``, from
                 the device prefixes; a producer's 0xFF pad byte is an E8M0 NaN -> NaN dV / dS); sf_q / sf_k pads are harmless (S is
                 select-dead) and bind as they are.  ``stage_sf_pads`` False = the RED twin (the kernel reads the caller's bytes as they are)
        delta    dot_do_o over the packed o_f16 / dO_f16 -> [1, H, ceil128(T_q)] in TRUE units (zeros past T_q) -- unless the caller
                 computed it (external_delta)
        per head chunk: the main kernel (its own setup launch clamps the five payload maps to the live packed totals and the five SF maps
                 to the live SF tile totals, emits the per-sequence clipped dV maps and resets live / ctr for THIS launch's heads; its SF
                 descriptors are built at the per-call packed tile counts), then dK / dQ: P-c through the half row's THD stage-3 arm over
                 the packed q_T / k_T dequantized EXACTLY to bf16 per token (``_dequant_mxfp8_to_bf16_thd``: no pad byte is ever read);
                 P-b through the block-scale arm's THD leg (``_stage3_block_scale_thd``: the kv-blocked payloads + atoms, the packed
                 columnwise q_T / k_T with their scale factors through the SF tile prefixes, dQ once per head chunk under GQA --
                 the record's ``b_head_group`` = the group, as on the dense P-b chain)
        fold     GQA: the per-Q-head dK / dV partials over the PACKED kv axis (dK fp32 under P-b -- rounded ONCE --, dV bf16), rows below
                 the live total cu_k[B] only (a device word) -> the KV heads (fixed order); the caller's capacity tail past cu_k[B] is
                 never written

    No amax, no per-tensor scalars (the row's contract).  ``lens_form`` bit 0 / 1 = the Q / KV length tensor is a ``(B+1,)`` prefix
    (``bind()`` derives it from numel); both tensors are viewed ``(B+1,)``.
    """
    b, h, hk, d, t_q, t_kv, sqp, rcap, hc, zero_ws, bpe_ds, units, sq_env, skv_env, dq_bhg = config
    fp8 = cutlass.Float8E4M3FN
    half = cutlass.BFloat16
    p_b = ds_sf_policy == DS_SF_P_B
    ds_dtype = fp8 if cutlass.const_expr(p_b) else half
    q = _view(q_ptr, geometry[0])  # packed [1, T_q, H_q, D] e4m3 (rowwise)
    k = _view(k_ptr, geometry[1])  # packed [1, T_kv, H_kv, D]
    v = _view(v_ptr, geometry[2])
    o_f16 = _view(o_ptr, geometry[3])  # packed [1, T_q, H_q, D] bf16
    do = _view(do_ptr, geometry[4])  # the ROWWISE e4m3 dO (the dP operand)
    stats = _view(stats_ptr, geometry[5])  # (T_q, H_q) token-major or (1, H_q, head_stride) head-major, the forward's packing
    dq = _view(dq_ptr, geometry[6])  # packed [1, T_q, H_q, D] bf16
    dk = _view(dk_ptr, geometry[7])  # packed [1, T_kv, H_kv, D]
    dv = _view(dv_ptr, geometry[8])
    q_lens = _view(seq_q_ptr, ((b + 1,), (1,)))
    kv_lens = _view(seq_kv_ptr, ((b + 1,), (1,)))
    q_T = _view(q_T_ptr, geometry[11])  # packed [1, T_q, H_q, D] e4m3 (columnwise)
    k_T = _view(k_T_ptr, geometry[12])  # packed [1, T_kv, H_kv, D]
    do_T = _view(do_T_ptr, geometry[13])  # the COLUMNWISE e4m3 dO (the dV operand)
    do_f16 = _view(do_f16_ptr, geometry[14])
    sf_q = _view(sf_q_ptr, geometry[15])  # the packed SF blobs: base address + the F8_128x4 atom rule only
    sf_q_T = _view(sf_q_T_ptr, geometry[16])
    sf_k = _view(sf_k_ptr, geometry[17])
    sf_k_T = _view(sf_k_T_ptr, geometry[18])
    sf_v = _view(sf_v_ptr, geometry[19])
    sf_do = _view(sf_do_ptr, geometry[20])
    sf_do_T = _view(sf_do_T_ptr, geometry[21])
    # delta [1, H, ceil128(T_q)] fp32 in TRUE units, PACKED head-major: the caller's (external_delta -- no region, no dot) or the
    # region stage 1 fills below (the dense host's rule over the packed token capacity).
    external_delta = cutlass.const_expr(delta_ptr is not None)
    delta = _view(delta_ptr, _thd_delta_geometry(h, t_q)) if cutlass.const_expr(external_delta) else _scratch(workspace, regions[R_DELTA], cutlass.Float32)
    meta = _scratch(workspace, regions[R_SEQ_KV], cutlass.Int32)  # the metadata words + the main kernel's tensor maps
    sf_meta = _scratch(workspace, regions[R_MX_SF_META], cutlass.Int32)  # [cu_sf_q(B+1) | cu_sf_k(B+1)]
    desc3 = _scratch(workspace, regions[R_DESC], cutlass.Int64)  # stage 3's (B + 1) descriptors, patched per GEMM launch
    ds_full = _scratch(workspace, regions[R_DS], ds_dtype)  # [1, hc, R_kv_cap, S_q_pad]: P-c's bf16 dS, P-b's ds_dk payload
    thd_bwd_setup_host(meta, q_lens, kv_lens, lens_form, hc, b, _THD_KV_BLOCK, _THD_KV_BLOCK, units, stream, kv_blocked=True, sf_meta_t=sf_meta)
    if cutlass.const_expr(zero_ws):
        n16 = hc * rcap * sqp * bpe_ds // 16
        _zero_bytes(ds_full, n16).launch(grid=(min((n16 + _THREADS - 1) // _THREADS, 4096), 1, 1), block=(_THREADS, 1, 1), stream=stream)
    group = h // hk
    live_kv = _window(meta, 0, THD_CU_K_TOTAL_OFF(b), 1)  # the live packed kv total, a device word: the fold stops there

    # SF pads: the five hazard tensors re-staged PACKED with their per-sequence pad bytes zeroed (the capacity tail zeroed whole).
    sf_v_k, sf_do_k, sf_do_T_k = sf_v, sf_do, sf_do_T
    if cutlass.const_expr(stage_sf_pads):
        sf_v_k = _scratch(workspace, regions[R_SF_V_PAD], cutlass.Uint8)
        sf_do_k = _scratch(workspace, regions[R_SF_DO_PAD], cutlass.Uint8)
        sf_do_T_k = _scratch(workspace, regions[R_SF_DOT_PAD], cutlass.Uint8)
        _thd_helpers.pad_sf_atoms_thd_host(sf_v, sf_v_k, meta, sf_meta, b, hk, sf_tiles_kv, True, False, stream)
        _thd_helpers.pad_sf_atoms_thd_host(sf_do, sf_do_k, meta, sf_meta, b, h, sf_tiles_q, False, False, stream)
        _thd_helpers.pad_sf_atoms_thd_host(sf_do_T, sf_do_T_k, meta, sf_meta, b, h, sf_tiles_q, False, True, stream)

    # STAGE 1: delta in TRUE units over the packed half-precision O / dO (one streaming pass; zeros past the packed capacity) -- unless
    # the caller computed it (external_delta: the same layout, the same finite-pad obligation, now the caller's).
    if cutlass.const_expr(not external_delta):
        dot_do_o_host(o_f16, do_f16, delta, None, None, DOT_Q_TILE, d, d, DOT_CHUNK_ELEMS, False, False, stream)

    if cutlass.const_expr(p_b):
        # P-b: the second payload and the two atom tensors over the kv-BLOCKED rows; zeroed with the first under ``zero_ws``.
        ds_dq_full = _scratch(workspace, regions[R_MX_DS_DQ], fp8)
        sf_ds_dk = _scratch(workspace, regions[R_MX_SF_DS_DK], cutlass.Uint8)
        sf_ds_dq = _scratch(workspace, regions[R_MX_SF_DS_DQ], cutlass.Uint8)
        if cutlass.const_expr(zero_ws):
            n16 = hc * rcap * sqp * bpe_ds // 16
            _zero_bytes(ds_dq_full, n16).launch(grid=(min((n16 + _THREADS - 1) // _THREADS, 4096), 1, 1), block=(_THREADS, 1, 1), stream=stream)
            n16_sf = hc * (rcap // SF_ATOM_ROWS) * (sqp // SF_ATOM_ROWS) * SF_ATOM_BYTES // 16
            _zero_bytes(sf_ds_dk, n16_sf).launch(grid=(min((n16_sf + _THREADS - 1) // _THREADS, 4096), 1, 1), block=(_THREADS, 1, 1), stream=stream)
            _zero_bytes(sf_ds_dq, n16_sf).launch(grid=(min((n16_sf + _THREADS - 1) // _THREADS, 4096), 1, 1), block=(_THREADS, 1, 1), stream=stream)
        # The block-scale GEMMs' SFB operands read WHOLE atoms of the packed columnwise q_T / k_T scale factors: their per-sequence
        # pad groups are re-staged zeroed too (0 x 2^-127 of a zero payload is exactly 0, a producer's 0xFF would be 0 x NaN).
        sf_qT_k, sf_kT_k = sf_q_T, sf_k_T
        if cutlass.const_expr(stage_sf_pads):
            sf_qT_k = _scratch(workspace, regions[R_MX_SF_QT_PAD], cutlass.Uint8)
            sf_kT_k = _scratch(workspace, regions[R_MX_SF_KT_PAD], cutlass.Uint8)
            _thd_helpers.pad_sf_atoms_thd_host(sf_q_T, sf_qT_k, meta, sf_meta, b, h, sf_tiles_q, False, True, stream)
            _thd_helpers.pad_sf_atoms_thd_host(sf_k_T, sf_kT_k, meta, sf_meta, b, hk, sf_tiles_kv, True, True, stream)
        sf_qT_planes = _sf_planes_view_thd(sf_qT_k, d // SF_ATOM_ROWS, sf_tiles_q, h)
        sf_kT_planes = _sf_planes_view_thd(sf_kT_k, d // SF_ATOM_ROWS, sf_tiles_kv, hk)
    else:
        # P-c: the stage-3 B operands are the packed columnwise q_T / k_T dequantized EXACTLY to bf16, per token through the
        # per-sequence SF tile prefixes (tokens past the live total written as zeros; the clamped B descriptor never reads them).
        q_T_bf16 = _scratch(workspace, regions[R_QT_BF16], half)  # [1, T_q_cap, H, D]
        k_T_bf16 = _scratch(workspace, regions[R_KT_BF16], half)  # [1, T_kv_cap, H_kv, D]
        dequant_mxfp8_to_bf16_thd_host(q_T, sf_q_T, q_T_bf16, meta, sf_meta, b, sf_tiles_q, False, stream)
        dequant_mxfp8_to_bf16_thd_host(k_T, sf_k_T, k_T_bf16, meta, sf_meta, b, sf_tiles_kv, True, stream)

    # stage 2's dV per Q head and stage 3's dK per Q head: the caller's packed dV / dK at MHA, the packed partials under GQA -- the dK
    # partial in the dtype the adapter carved it, derived from the policy and the group as the dense arm does (fp32 under P-b: the
    # block-scale dK GEMM's true-unit fp32 accumulator, rounded ONCE by the bounded fold; bf16 on the P-c twin); dV bf16.
    dk_part_dtype = cutlass.Float32 if (p_b and group > 1) else half
    dv_k = _scratch(workspace, regions[R_MX_DV_PART], half) if cutlass.const_expr(regions[R_MX_DV_PART] is not None) else dv
    dk_tgt = _scratch(workspace, regions[R_MX_DK_PART], dk_part_dtype) if cutlass.const_expr(regions[R_MX_DK_PART] is not None) else dk
    ds = _extent(ds_full, (1, hc, rcap, sqp))
    grid_m_kv = -(-skv_env // _THD_KV_BLOCK) * _THD_KV_BLOCK  # the kv envelope's M tiles (dK); dQ's is the padded q envelope (sqp)
    problem = (b, h, hk, sqp, rcap, hc, sq_env, skv_env, units)
    for ci in range(h // hc):
        hb = ci * hc
        # STAGE 2: packed operands, the kv-blocked workspace, this launch's heads.  The dS operands follow the policy (the kernel's
        # ONE positional shape): P-c binds the bf16 ds_ws and None for the four P-b operands; P-b None for ds_ws and the two payloads
        # + two atom tensors.  The THD operands are APPENDED after them and passed by keyword: the metadata + maps buffer in the
        # lengths slot, the SF tile prefixes, and the per-call packed SF tile counts the SF descriptors are built at.
        if cutlass.const_expr(p_b):
            main(
                q,
                k,
                v,
                do,
                do_T,
                dv_k,
                None,
                stats,
                delta,
                sf_q,
                sf_k,
                sf_v_k,
                sf_do_k,
                sf_do_T_k,
                problem,
                scale,
                scale_log2,
                hb,
                cutlass.Int32(skv_env),
                cutlass.Int32(sq_env),
                ds_full,
                ds_dq_full,
                sf_ds_dk,
                sf_ds_dq,
                seq_kv_lens_tensor=meta,
                sf_meta_tensor=sf_meta,
                sf_tiles_q=sf_tiles_q,
                sf_tiles_kv=sf_tiles_kv,
                stream=stream,
            )
            # STAGE 3: the block-scale arm's THD leg over the chunk's blocked payloads + atoms; every sequence through its own descriptor;
            # dQ once per head chunk at the GQA group (`dq_bhg` = the record's b_head_group), once per group member at 1.
            _stage3_block_scale_thd(
                mm_dk,
                mm_dq,
                ds,
                _extent(ds_dq_full, (1, hc, rcap, sqp)),
                sf_ds_dk,
                sf_ds_dq,
                q_T,
                sf_qT_planes,
                k_T,
                sf_kT_planes,
                dk_tgt,
                dq,
                hb,
                hc,
                group,
                b,
                meta,
                desc3,
                sf_meta,
                stream,
                grid_m_kv,
                sqp,
                dq_b_head_group=dq_bhg,
            )
        else:
            main(
                q,
                k,
                v,
                do,
                do_T,
                dv_k,
                ds_full,
                stats,
                delta,
                sf_q,
                sf_k,
                sf_v_k,
                sf_do_k,
                sf_do_T_k,
                problem,
                scale,
                scale_log2,
                hb,
                cutlass.Int32(skv_env),
                cutlass.Int32(sq_env),
                None,
                None,
                None,
                None,
                seq_kv_lens_tensor=meta,
                sf_meta_tensor=sf_meta,
                sf_tiles_q=sf_tiles_q,
                sf_tiles_kv=sf_tiles_kv,
                stream=stream,
            )
            # STAGE 3 at bf16 over the exactly dequantized packed operands through the half row's THD arm; TRUE-unit bf16 out.
            _stage3_thd(mm_dk, mm_dq, ds, q_T_bf16, k_T_bf16, dk_tgt, dq, hb, hc, group, b, meta, desc3, stream, grid_m_kv, sqp, dq_b_head_group=dq_bhg)

    # STAGE 4: fold the per-Q-head partials onto the KV heads over the packed kv axis (fixed order), rows [0, cu_k[B]) ONLY -- the
    # partials past the live total were never written (per-sequence clipped stores), so an unbounded fold would copy the poisoned
    # workspace into the caller's dK / dV capacity tail.
    if cutlass.const_expr(group > 1):
        dkv_reduce_bounded_host(dk_tgt, dv_k, dk, dv, d, d, group, half, False, live_kv, stream)


# --- compilation -----------------------------------------------------------------------------------------------------------


def _check_target(sm: int, lo: int = 107) -> None:
    # The fp8 body is a Rubin-line kernel (327 KiB SMEM carveout, 576 TMEM columns, the K=64 dense-FP8 MMA form): lo = 107.
    # The half chain also runs the 2x2-datapath body (kernels/bprop_d256_2x2_f16.py: 512 TMEM columns, 210 KiB) on the
    # SM100 line (sdpa_bwd_sm100_d256), so compile_host_f16 admits SM100-SM119; the 4x1 f16 body never reaches sm < 107
    # because its engine row declares sm_lo = 107.
    if not lo <= sm <= 119:
        raise ValueError(f"SM107 SDPA bwd d256 has codegen targets for SM{lo}-SM119; got SM{sm}")


def _check_sm_count(sm_count) -> int:
    # The device's multiprocessor count -- a plan fact the caller reads from the device and folds into the cache key, never a
    # literal: the fold passes' persistent grid is ``sm_count x FOLD_QUANT_CTAS_PER_SM`` CTAs, so a placeholder compiles a CORRECT
    # artifact whose fold pass streams half a gigabyte of partials through a handful of CTAs.
    if not isinstance(sm_count, int) or sm_count < 1:
        raise ValueError(f"sm_count must be the device's multiprocessor count (a positive int, a plan fact of the fp8 row); got {sm_count!r}")
    return sm_count


def _ptr(t, align=16):
    return cute.runtime.make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=align)


def compile_host_f16(
    main,
    mm_dk,
    mm_dq,
    config,
    geometry,
    regions,
    dtype,
    sm,
    cache_key,
    seq_kv_present=False,
    external_delta=False,
    symbol: str = "frost_sdpa_bwd_sm107_prepared",
):
    """The half rows' artifact: ``dtype`` is the io / gradient DSL type (bf16 or fp16).  Two appended flags, each default False
    and independent of the other, decide the two appended pointer slots -- the slot stays in the positional ABI either way
    (``prepared.bind`` frames a None for an absent operand), so ``bind()`` refuses a buffer the plan did not ask for and requires
    the one it did; the caller folds both into ``cache_key``.  ``seq_kv_present`` binds the caller's ``[B]`` int32 per-batch kv
    lengths as the tenth operand; ``external_delta`` binds the caller's ``[B, H, S_q_pad]`` fp32 delta (16-B aligned like the
    region it replaces) as the eleventh.  ``geometry`` carries both slots' static layouts (``geometry[9]`` / ``geometry[10]``) whether
    or not they are bound.  ``symbol`` (appended) names the row's artifact (``frost_<engine>_prepared``): SM100-SM119, because the
    chain hosts the Rubin 4x1 body AND the 2x2 body of the SM100 d256 row."""
    _check_target(sm, lo=100)
    args = [_ptr(dtype) for _ in range(5)] + [_ptr(cutlass.Float32, 4)] + [_ptr(dtype) for _ in range(3)]
    args += [_ptr(cutlass.Int32, 4) if seq_kv_present else None]
    args += [_ptr(cutlass.Float32, 16) if external_delta else None]
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
        symbol=symbol,
    )


def compile_host_f16_thd(main, mm_dk, mm_dq, config, geometry, regions, dtype, sm, cache_key, external_delta=False):
    """The half row's THD artifact (:func:`host_f16_thd`): the nine packed tensor operands, the two ``[B]`` / ``[B+1]`` int32
    length operands, the appended delta slot (``external_delta``, default False: the caller's PACKED ``[1, H, ceil128(T_q)]`` fp32
    delta, 16-B aligned; None-specialized otherwise -- the slot stays in the positional ABI either way, as on the dense entry), the
    workspace, the scale and the host-derived ``lens_form``.  Its own entry and cache key (the caller folds the THD config and the
    delta fact into ``cache_key``): the dense ``host_f16`` artifact's ABI and key are untouched."""
    _check_target(sm)
    args = [_ptr(dtype) for _ in range(5)] + [_ptr(cutlass.Float32, 4)] + [_ptr(dtype) for _ in range(3)]
    args += [_ptr(cutlass.Int32, 4), _ptr(cutlass.Int32, 4)]
    args += [_ptr(cutlass.Float32, 16) if external_delta else None]
    return compile_cached(
        host_f16_thd,
        *args,
        _ptr(cutlass.Uint8),
        cutlass.Float32(1),
        cutlass.Int32(0),
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
        symbol="frost_sdpa_bwd_sm107_thd_prepared",
    )


def compile_host_fp8(
    main, mm_dk, mm_dq, config, geometry, regions, grad_dtype, amax_requested, sm, cache_key, seq_kv_present=False, external_delta=False, *, sm_count
):
    """The fp8 row's artifact: e4m3 payloads, fp32 scalars, gradients in ``grad_dtype`` (e4m3 / bf16 / fp16); ``amax_requested`` is
    the 4-tuple of bools (dQ, dK, dV, dP) selecting which amax pointers the artifact binds (None-specialized otherwise).  Two
    appended flags, each default False and independent of the other, decide the two appended pointer slots exactly as on the
    half row (``compile_host_f16``): ``seq_kv_present`` binds the caller's ``[B]`` int32 per-batch kv lengths (slot 25),
    ``external_delta`` the caller's ``[B, H, S_q_pad]`` fp32 delta (slot 26); the caller folds both into ``cache_key``.
    ``sm_count`` (keyword-only, NO default; the caller folds it into ``cache_key`` too) is the device's multiprocessor count: the
    fold passes' persistent grid is sized on it (``bprop_chain_common.fold_quant_ctas``), so it comes from the device, never a
    placeholder (``_check_sm_count``)."""
    _check_target(sm)
    sm_count = _check_sm_count(sm_count)
    fp8 = cutlass.Float8E4M3FN
    args = [_ptr(fp8) for _ in range(5)] + [_ptr(cutlass.Float32, 4)] + [_ptr(grad_dtype) for _ in range(3)]
    args += [_ptr(cutlass.Float32, 4) for _ in range(12)]
    args += [_ptr(cutlass.Float32, 4) if requested else None for requested in amax_requested]
    args += [_ptr(cutlass.Int32, 4) if seq_kv_present else None]
    args += [_ptr(cutlass.Float32, 16) if external_delta else None]
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
        sm_count,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options=f"--enable-tvm-ffi --gpu-arch sm_{sm}a",
        cache_key=cache_key,
        symbol="frost_sdpa_bwd_sm107_fp8_prepared",
    )


def compile_host_fp8_thd(main, mm_dk, mm_dq, config, geometry, regions, grad_dtype, amax_requested, sm, cache_key, *, sm_count, external_delta=False):
    """The fp8 row's THD artifact (:func:`host_fp8_thd`): the nine packed tensor operands, the two ``[B]`` / ``[B+1]`` int32 length
    operands, the twelve scalars, the requested amax, the appended delta slot (``external_delta``, default False: the caller's PACKED
    ``[1, H, ceil128(T_q)]`` fp32 TRUE-unit delta, None-specialized otherwise; the slot stays in the positional ABI either way), the
    workspace, the two scales and the host-derived ``lens_form``.  Its own entry and cache key (the caller folds the THD config and
    the delta fact into ``cache_key``): the dense ``host_fp8`` artifact's ABI and key are untouched.  ``sm_count`` (keyword-only, NO
    default; folded into ``cache_key`` by the caller) is the device's multiprocessor count the fold passes' persistent grid is sized
    on -- from the device, never a placeholder (``_check_sm_count``)."""
    _check_target(sm)
    sm_count = _check_sm_count(sm_count)
    fp8 = cutlass.Float8E4M3FN
    args = [_ptr(fp8) for _ in range(5)] + [_ptr(cutlass.Float32, 4)] + [_ptr(grad_dtype) for _ in range(3)]
    args += [_ptr(cutlass.Int32, 4), _ptr(cutlass.Int32, 4)]
    args += [_ptr(cutlass.Float32, 4) for _ in range(12)]
    args += [_ptr(cutlass.Float32, 4) if requested else None for requested in amax_requested]
    args += [_ptr(cutlass.Float32, 16) if external_delta else None]
    return compile_cached(
        host_fp8_thd,
        *args,
        _ptr(cutlass.Uint8),
        cutlass.Float32(1),
        cutlass.Float32(1),
        cutlass.Int32(0),
        main,
        mm_dk,
        mm_dq,
        tuple(config),
        geometry,
        regions,
        fp8,
        grad_dtype,
        sm_count,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options=f"--enable-tvm-ffi --gpu-arch sm_{sm}a",
        cache_key=cache_key,
        symbol="frost_sdpa_bwd_sm107_fp8_thd_prepared",
    )


def compile_host_mxfp8(
    main,
    mm_dk,
    mm_dq,
    config,
    geometry,
    regions,
    sm,
    cache_key,
    stage_sf_pads=True,
    ds_sf_policy=DS_SF_POLICY_DEFAULT,
    seq_kv_present=False,
    external_delta=False,
):
    """The MXFP8 row's artifact: e4m3 payloads (q, k, v, dO, q_T, k_T, dO_T), bf16 o_f16 / dO_f16 / dQ / dK / dV, fp32 Stats and
    seven uint8 F8_128x4 scale-factor blobs (``geometry`` carries each as a flat byte view: only the base address is read).
    ``stage_sf_pads`` = ``MXFP8_STAGE_SF_PADS`` as the plan read it (the caller folds it into ``cache_key``); ``ds_sf_policy``
    (appended, default ``DS_SF_POLICY_DEFAULT``) = the adapter's dS policy -- ``DS_SF_P_B`` the block-scaled chain that ships, ``DS_SF_P_C``
    the bf16-dS oracle twin -- likewise keyed.  ``seq_kv_present`` / ``external_delta`` (appended, default False, independent) decide
    the two appended pointer slots 20 / 21 exactly as on the half row: the caller's ``[B]`` int32 per-batch kv lengths and the
    caller's ``[B, H, S_q_pad]`` fp32 delta; both keyed by the caller."""
    _check_target(sm)
    if ds_sf_policy not in (DS_SF_P_C, DS_SF_P_B):
        raise ValueError(f"SM107 MXFP8 bwd: ds_sf_policy must be DS_SF_P_C ({DS_SF_P_C}) or DS_SF_P_B ({DS_SF_P_B}); got {ds_sf_policy}")
    fp8, half = cutlass.Float8E4M3FN, cutlass.BFloat16
    args = [_ptr(fp8), _ptr(fp8), _ptr(fp8), _ptr(half), _ptr(fp8), _ptr(cutlass.Float32, 4), _ptr(half), _ptr(half), _ptr(half)]
    args += [_ptr(fp8), _ptr(fp8), _ptr(fp8), _ptr(half)]
    args += [_ptr(cutlass.Uint8) for _ in range(7)]
    args += [_ptr(cutlass.Int32, 4) if seq_kv_present else None]
    args += [_ptr(cutlass.Float32, 16) if external_delta else None]
    return compile_cached(
        host_mxfp8,
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
        bool(stage_sf_pads),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        int(ds_sf_policy),
        options=f"--enable-tvm-ffi --gpu-arch sm_{sm}a",
        cache_key=cache_key,
        symbol="frost_sdpa_bwd_sm107_mxfp8_prepared",
    )


def compile_host_mxfp8_thd(
    main, mm_dk, mm_dq, config, geometry, regions, sm, cache_key, stage_sf_pads=True, ds_sf_policy=DS_SF_POLICY_DEFAULT, external_delta=False
):
    """The MXFP8 row's THD artifact (:func:`host_mxfp8_thd`): the nine packed tensor operands, the two ``[B]`` / ``[B+1]`` int32 length
    operands, the four packed payloads, the seven PACKED scale-factor blobs, the appended delta slot (``external_delta``, default
    False: the caller's PACKED ``[1, H, ceil128(T_q)]`` fp32 TRUE-unit delta, None-specialized otherwise; the slot stays in the
    positional ABI either way), the workspace, the two scales, the host-derived ``lens_form`` and the two per-call packed SF tile
    counts (``sf_tiles_q``, ``sf_tiles_kv``: ``prepared.bind`` frames them from the bound buffers' byte sizes).  Its own entry and cache
    key (the caller folds the THD config, ``stage_sf_pads``, ``ds_sf_policy`` and the delta fact into ``cache_key``): the dense
    ``host_mxfp8`` artifact's ABI and key are untouched."""
    _check_target(sm)
    if ds_sf_policy not in (DS_SF_P_C, DS_SF_P_B):
        raise ValueError(f"SM107 MXFP8 bwd THD: ds_sf_policy must be DS_SF_P_C ({DS_SF_P_C}) or DS_SF_P_B ({DS_SF_P_B}); got {ds_sf_policy}")
    fp8, half = cutlass.Float8E4M3FN, cutlass.BFloat16
    args = [_ptr(fp8), _ptr(fp8), _ptr(fp8), _ptr(half), _ptr(fp8), _ptr(cutlass.Float32, 4), _ptr(half), _ptr(half), _ptr(half)]
    args += [_ptr(cutlass.Int32, 4), _ptr(cutlass.Int32, 4)]
    args += [_ptr(fp8), _ptr(fp8), _ptr(fp8), _ptr(half)]
    args += [_ptr(cutlass.Uint8) for _ in range(7)]
    args += [_ptr(cutlass.Float32, 16) if external_delta else None]
    return compile_cached(
        host_mxfp8_thd,
        *args,
        _ptr(cutlass.Uint8),
        cutlass.Float32(1),
        cutlass.Float32(1),
        cutlass.Int32(0),
        cutlass.Int32(1),
        cutlass.Int32(1),
        main,
        mm_dk,
        mm_dq,
        tuple(config),
        geometry,
        regions,
        bool(stage_sf_pads),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        int(ds_sf_policy),
        options=f"--enable-tvm-ffi --gpu-arch sm_{sm}a",
        cache_key=cache_key,
        symbol="frost_sdpa_bwd_sm107_mxfp8_thd_prepared",
    )
