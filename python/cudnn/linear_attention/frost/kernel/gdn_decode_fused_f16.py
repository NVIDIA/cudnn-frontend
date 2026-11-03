# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Fused Gated DeltaNet DECODE step (SM100-class parts, Cutlass primitives) for the geometry of ``gdn_decode_fused_config.py`` (the
Qwen 3.5-architecture dense 27B GDN sub-layer at TP 1: 16 K heads, 48 V heads, head dim 128, conv width 4, hidden 5120): BA projection, causal conv +
SiLU, Q / K RMS scales, the gated delta-rule state update, gated RMSNorm, the static FP8 output quantization and the conv-history
shift in one launch, after the stock FP8 QKVZ projection of a decode step.

Per CTA (one V head ``h`` of one request row; K head ``kh = h // 3``), 256 threads = two warpgroups.  Lane ``l`` of warp ``w`` owns the
column slice ``tpr = l // RG`` (``C = 128 / TPR`` columns) of the ``R = TPR / 2`` consecutive state rows ``BOX_ROWS * rg + R * w + m``
(``rg = l % RG``, ``RG = 32 / TPR``, ``BOX_ROWS = 128 / RG``): the RG row groups of a warp sit in RG different state BOXES, so the eight
lanes of a 128-bit shared-memory phase never share a bank (the bank map is in the config's docstring and in the state buffer row of the
table below).  The row slice is REGISTER-RESIDENT for the two passes (``R x C = 64`` fp32 per thread at 256 threads) unless
``STATE_STREAM`` re-reads it per pass.

  entry      every thread: ``slot = indices[row]``, ``w[orow]``, ``scale``, ``dt_bias[h]``, ``A_log[h]`` (``orow`` = the row this thread
             outputs: ``m = tpr``); warpgroup 0: the two BA weight rows (10 x LDG.128, the reference 128-thread partition) and, off PDL,
             ``hidden[row]`` (5 x LDG.128) and ``z[orow]``; warpgroup 1: lane 0 of its first warp initialises the conv-operand
             mbarrier (+ fence), ``bar.sync 1, 128`` publishes it; warp 0 lane 0 initialises the state mbarrier (bulk path; under
             ``CONV_FIRST`` the conv role's lane initialises both, one fence).
  slot > 0   ldg128: every thread 16 x LDG.128 of its row slice straight into registers;
             bulk:   warp 0 lane 0 arms ``expect_tx(65536)`` and issues the RG ``cp.async.bulk`` boxes of the head's slab into the
                     padded shared-memory boxes;
             warpgroup 1's lane 0 arms ``expect_tx(6144)`` and issues the ``cp.async.bulk`` copies of this head's conv weights and
             history (12, or 6 when the taps are contiguous) -- under ``CONV_FIRST`` the SAME lane then arms and issues the slab boxes,
             so the 6 KB of conv operands never queue behind the 64 KB slab in the SM's bulk-copy path; then, under PDL, every thread's
             ``griddepcontrol.wait``; then the three ``x`` copies (the predecessor's output), ``z`` and ``hidden``.
  BA         warpgroup 0: fp32 dots -> two 5-step butterflies -> 4 warp partials (``bar.sync 2, 128``) -> 5-step totals -> bf16 ->
             ``av = a + dt``; ``soft = av <= 20 ? log(1 + exp(av)) : av``; ``decay = exp(-exp(A_log) * soft)``;
             ``beta = 1 / (1 + exp(-b))``.
  conv       warpgroup 1: mbarrier wait; the elected lane draws the reader ticket (``atom.relaxed.gpu.inc``); each thread's Q, K
             and V channel: ``acc += bf16(tap * weight)`` x 4, the stock SiLU, bf16; its V channel's history shifted in place; raw
             q / k to smem (``bar.sync 1, 128``) -> the reference 16-lane RMS partials and butterflies -> ``qscale`` / ``kscale``
             -> the scaled values into the padded per-slice smem arrays.
  bar.sync 0 {q, k, v, decay, beta, ticket} published.
  bulk       every thread waits the state mbarrier, 16 x LDS.128 of its row slice (conflict-free by the box pad).
  pass 1     ``h = S * decay``; per 8-column group an 8-step FFMA chain against k; the balanced tree over the thread's groups, then
             ``log2(TPR)`` xor shuffles with offsets ``RG << step`` (= the reference 16-leaf tree); ``inn = (v - dot) * beta``.
  pass 2     ``h += inn * k``; ``o = h . q`` (same tree); the updated row slice back: 16 x STG.128 (ldg128) or 16 x STS.128 into the
             same padded boxes (bulk; ``STATE_STREAM`` re-reads the row and recomputes ``h = S * decay`` first, bit-identical);
             ``value = f32(bf16(o))``.
  norm       every thread pre-sums its R rows' ``value^2`` in row order (explicit ``mul.rn.f32``; an aligned block of the 128-leaf tree);
             the ``tpr == 0`` lanes publish ``norm_part[rg * WARPS + w]``; bulk: every thread's ``fence.proxy.async.shared::cta``
             behind its STS (``STORE_PIPELINE``: then its warpgroup's named barrier -- warps 1-3 arrive, warp 0 syncs -- and warp 0's
             lane 0 issues the RG half-box stores of its warpgroup's rows at once); ``bar.sync 0`` #2; bulk without the pipeline: thread 0
             issues the RG bulk stores of the boxes (``bulk_group``, commit); every thread: the balanced tree over the 8 warps of each
             box, then over the boxes; ``inv = rsqrt(ss / 128 + eps)``.
  epilogue   lane ``tpr < R`` of each row group (row ``m = tpr``): ``bf16(((value * inv) * w) * silu(z))`` -> ``out``, ``e4m3(out /
             scale)`` -> ``qout``; the ``tpr == TPR - 1`` lanes of the CTA that drew the last ticket shift the K head's Q / K conv history
             (reads from smem); bulk: warp 0 lane 0 ``cp.async.bulk.wait_group.read 0`` before exit.
  slot <= 0  zero ``out`` / ``qout`` rows; no state, conv-state or ticket write; nothing in flight.

Barriers executed per thread: <= 4 ``BAR.SYNC`` (two CTA-wide, up to two warpgroup-local; warpgroup 0 executes 3 with the pin knob off),
one or two mbarrier waits, one relaxed atomic; no cluster, no ``membar``, no ``cp.async`` (LDGSTS).  Barrier table (init count = the
lanes that arrive, on every path):

  mbarrier ``mb_ops``     init ``CONV_MBAR_ARRIVERS`` = 1 by lane 0 of warp 4 (the conv role's first warp); one ``arrive.expect_tx(6144)``
                          by the same lane; the 15 (or 9) bulk copies complete the bytes; waited by the 128 conv-role threads (phase 0)
                          before the conv.
  mbarrier ``mb_state``   (bulk) init ``STATE_MBAR_ARRIVERS`` = 1 by warp 0 lane 0 at entry -- under ``CONV_FIRST`` by the conv role's
                          lane 0 together with ``mb_ops`` (one fence) -- (+ fence; published to the waiters by the first ``bar.sync 0``,
                          which every waiter passes after the initialising thread's own arrival there); one ``arrive.expect_tx(65536)`` by
                          the same lane (under ``CONV_FIRST`` right AFTER its conv-operand copies); the RG box copies complete the bytes;
                          waited by every thread (phase 0) after the first ``bar.sync 0``.
  ``barrier.sync 1, 128``  (non-aligned) the conv warpgroup, twice: after the mbarrier init (visibility), after the raw q / k smem writes.
  ``barrier.sync 2, 128``  (non-aligned) the BA warpgroup, once: the four warps' BA partials before the totals (twice with the ldg128 pin knob).
  ``barrier.sync 0, 256`` (non-aligned) twice: {q, k, v, decay, beta, ticket} published; the norm partials published (and, bulk, every thread's
                          STS + ``fence.proxy.async`` of the updated rows before the one lane's bulk stores read them).
  bulk store               RG ``cp.async.bulk.global.shared::cta.bulk_group`` + ``commit_group`` by thread 0 after ``bar.sync 0`` #2;
                          the same thread's ``wait_group.read 0`` is its last instruction (the smem is read out before the CTA
                          retires; the global writes complete with the grid).  ``STORE_PIPELINE``: each warpgroup's warp-0 lane 0
                          instead issues the RG half-box stores of its warpgroup's rows right after warp 0's ``barrier.sync {2 | 1}, 128``
                          (warps 1-3 ``barrier.arrive``, 96 + 32 = 128: the warpgroup's STS + proxy fences published to it; whole warps
                          participate, no intra-warp divergence at a barrier), and each of the two lanes drains its own group before exit.
  ticket                   one ``atom.relaxed.gpu.inc`` after the mbarrier wait (all history bytes read); consumed at exit.
  PDL (knob)               ``griddepcontrol.wait`` by every thread in both slot arms, after the predecessor-independent loads and
                          issues (slot, parameters, BA weights, the state slab, the conv weights and history) and before ``z``,
                          ``hidden``, the ``x`` copies and every global write; ``launch_dependents`` at entry under ``PDL_RELEASE``.
                          CALLER OBLIGATION: the kernel immediately preceding this launch in the stream must not write this layer's
                          ``state``, ``cs`` or ``indices`` -- the pre-wait loads read them before the wait, so a predecessor that writes
                          them (the same layer launched twice in a row) races the slab boxes / history copies against its stores and
                          its write-back: a wrong state, no error.  The serving chain's predecessor (the QKVZ projection) writes only
                          ``x`` / ``z``, which are read after the wait; the host entry cannot check the obligation.
  null slot                no arrive, nothing in flight at exit.

Numerics: the reference kernel's arithmetic order and PTX forms (the reference kernel: FlashInfer's built-in ``frost`` decode backend for
the Qwen 3.5 GDN block; every product that feeds an add is single-use where the reference
fuses and multi-use where it does not; the norm squares are explicit ``mul.rn.f32``), so the fp32 state, the conv state, the bf16
``out`` and the fp8 ``qout`` are bitwise with it under every knob: the thread mapping moves the tree's leaves between threads, never its
association (the in-thread groups are contiguous 8-column groups combined in order; the xor steps pair the same partials; the norm's
``RG x WARPS`` partials are aligned blocks of consecutive rows combined in row order).

Shared memory (bytes; no swizzle: no descriptor reads these buffers, every per-lane pitch that could hit one bank is padded): x pieces
768, conv weights 3072, conv history 2304, raw q / k 1024, q / k slices 2 x TPR x (C + 4) x 4 (the slices of a phase land in distinct
banks), v 512 (index-permuted so the RG lanes of a phase read consecutive words), scalars 16, ticket 16, BA partials 32, norm partials
RG x WARPS x 4, the two mbarriers 16 + 16, the state buffer RG boxes x (BOX_ROWS x 512 + 16) B (bulk; 16 B otherwise: box pitch
padded by one 16-B chunk -> bank = 4 rg + (128 / TPR) tpr + 4 c4 mod 32, eight distinct quads per 8-lane phase at every TPR), plus the
occupancy pad of ``Cfg.CTAS_PER_SM`` derived from the device's shared memory per SM.
"""

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm
from cutlass._mlir.dialects import arith

from cudnn.frost.tile_dsl.barrier import launch_dependent_grids, wait, wait_on_dependent_grids
from cudnn.frost.tile_dsl.tma import ld_global_v4, ld_shared_v4, st_global_v4, st_shared_v4

from .gdn_decode_fused_config import (
    BA_CHUNKS,
    CONV_BULK_BYTES,
    CONV_MBAR_ARRIVERS,
    CONV_WIDTH,
    HEAD_DIM,
    HIDDEN,
    K_CHANNEL0,
    NORM_MBAR_ARRIVERS,
    NORM_XFER_BYTES,
    QKV_CHANNELS,
    RMS_LANES,
    STATE_MBAR_ARRIVERS,
    TICKET_STRIDE,
    V_CHANNEL0,
    V_HEADS,
)

FP32 = cutlass.Float32
BF16 = cutlass.BFloat16
INT32 = cutlass.Int32
INT64 = cutlass.Int64
UINT8 = cutlass.Uint8

# the reference's q scale constant 128 ** -0.5 (its fp32 literal), the RMS epsilon and the softplus threshold
Q_SCALE = 0.08838834764831845
RMS_EPS = 1e-6
SOFTPLUS_LINEAR_ABOVE = 20.0
RMS_COLS = HEAD_DIM // RMS_LANES  # 8 raw values per RMS lane
ROW_BYTES = HEAD_DIM * 4  # one state row in the shared-memory boxes (the natural 512-B pitch inside a box)


# --------------------------------------------------------------------------------------------------------------------------
# PTX forms shared with the reference kernel (inline PTX through the DSL primitive; outputs number first, inputs after)
# --------------------------------------------------------------------------------------------------------------------------
def _unpack_bf16x2(word):
    """A bf16x2 word -> (lo, hi) as fp32: ``shl 16`` / ``and 0xffff0000`` (the reference helper's exact ``* 1.0`` dropped)."""
    return nvvm.inline_ptx(
        "{\n\t.reg .b32 t;\n\tshl.b32 t, $2, 16;\n\tmov.b32 $0, t;\n\tand.b32 t, $2, 0xffff0000;\n\tmov.b32 $1, t;\n\t}",
        write_only_types=[FP32, FP32],
        read_only_args=[word],
    )


def _stock_conv_silu(value):
    """The stock causal-conv SiLU: ``x / (1 + ex2(-x * log2 e))`` with ``ex2.approx`` and ``div.full``."""
    return nvvm.inline_ptx(
        "{\n\t.reg .f32 t, e, d;\n\tmul.f32 t, $1, 0fBFB8AA3B;\n\tex2.approx.f32 e, t;\n\tadd.f32 d, e, 0f3F800000;\n\tdiv.full.f32 $0, $1, d;\n\t}",
        write_only_types=[FP32],
        read_only_args=[value],
    )


def _quant_fp8_word(value, scale):
    """The stock per-tensor FP8 quantization: ``v * (1 / scale)`` with a ``div.rn`` reciprocal, ``cvt.rn.satfinite.e4m3x2``; the
    e4m3 byte is the low byte of the returned word."""
    return nvvm.inline_ptx(
        "{\n\t.reg .f32 inv, v;\n\t.reg .b16 p;\n\tdiv.rn.f32 inv, 0f3F800000, $2;\n\tmul.f32 v, $1, inv;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 p, 0f00000000, v;\n\tcvt.u32.u16 $0, p;\n\t}",
        write_only_types=[INT32],
        read_only_args=[value, scale],
    )


def _mul_rn_f32(a, b):
    """An fp32 product the compiler cannot fuse into a following add (the reference's norm squares are unfused ``FMUL``)."""
    return nvvm.inline_ptx("mul.rn.f32 $0, $1, $2;", write_only_types=[FP32], read_only_args=[a, b])


def _atom_inc_relaxed_gpu(addr, limit):
    """``atom.relaxed.gpu.global.inc.u32``: the reader ticket; returns the old value, wraps to 0 past ``limit``."""
    return nvvm.inline_ptx("atom.relaxed.gpu.global.inc.u32 $0, [$1], $2;", write_only_types=[INT32], read_only_args=[addr, limit])


def _smem_addr(arr):
    """32-bit shared-window address of a shared-memory ``cutlass.Array``."""
    return INT32(arr.data_ptr().toint())


def _mapa_u32(addr32, rank):
    """``mapa.shared::cluster``: this CTA's shared-window address translated to the same offset in cluster CTA ``rank``'s window."""
    return nvvm.inline_ptx("mapa.shared::cluster.u32 $0, $1, $2;", write_only_types=[INT32], read_only_args=[addr32, rank])


def _st_async_f32(peer_addr32, value, peer_mbar32):
    """One fp32 into a peer CTA's shared memory, completing 4 tx bytes on the peer's mbarrier (``st.async``; the DSMEM path, not the
    bulk-copy unit -- the write-back boxes queued there would delay it).  Both addresses are ``mapa``-translated."""
    nvvm.inline_ptx("st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.f32 [$0], $1, [$2];", read_only_args=[peer_addr32, value, peer_mbar32])


def _cta_barrier(threads):
    """The CTA-wide barrier (id 0, every thread) in its NON-aligned form: the two warpgroup roles reach it from two traced copies of
    the body (two PCs), which the ``.aligned`` ``bar.sync 0`` promise ("every thread executes the SAME barrier instruction") does not
    cover -- synccheck flags the aligned form as divergent; the hardware barrier itself counts arrivals per id."""
    nvvm.barrier_cta_sync(0, thread_count=threads)


def _wg_barrier(barrier_id):
    """A warpgroup-local named barrier (128 threads): the NON-aligned ``barrier.sync id, 128`` -- it is reached from a warp-role
    branch, so the ``.aligned`` form (``bar.sync``, which promises that every thread of the CTA executes it) would be undefined
    (synccheck flags it: "divergent thread(s) in block")."""
    nvvm.barrier_cta_sync(barrier_id, thread_count=128)


def _select_f32(cond, a, b):
    """``cond ? a : b`` on fp32 values as one select (no branch)."""
    return FP32(arith.select(cond.ir_value(), a.ir_value(), b.ir_value()))


def _balanced_sum(parts):
    """The balanced binary tree over ``parts`` in order (``((p0+p1)+(p2+p3))+...``); one part returns itself."""
    while len(parts) > 1:
        parts = [parts[i] + parts[i + 1] for i in range(0, len(parts), 2)]
    return parts[0]


# --------------------------------------------------------------------------------------------------------------------------
# straight-line helpers (plain Python: traced inline, no DSL control flow inside)
# --------------------------------------------------------------------------------------------------------------------------
def _ba_weight_loads(baweight, head, t):
    """The reference BA partition, weight half: thread ``t`` loads elements ``chunk * 1024 + t * 8 .. + 7`` of ``Wb[h]`` and ``Wa[h]``
    (parameters: never a predecessor's output, so they load ahead of a PDL wait)."""
    bw, aw = [], []
    for c in range(BA_CHUNKS):
        kk = c * 1024 + t * 8
        bw.append(ld_global_v4((baweight.iterator + (head * HIDDEN + kk)).toint(), INT32))
        aw.append(ld_global_v4((baweight.iterator + ((head + V_HEADS) * HIDDEN + kk)).toint(), INT32))
    return bw, aw


def _ba_hidden_loads(hidden, row, t):
    """The reference BA partition, activation half: elements ``chunk * 1024 + t * 8 .. + 7`` of ``hidden[row]``."""
    hx = []
    for c in range(BA_CHUNKS):
        kk = c * 1024 + t * 8
        hx.append(ld_global_v4((hidden.iterator + (row * HIDDEN + kk)).toint(), INT32))
    return hx


def _ba_dots(hx, bw, aw, zero):
    """The reference per-thread fp32 dot chains (40 FFMA each, in its order), accumulating from ``zero`` (+0.0: the constant, or the
    same +0.0 read from shared memory behind the state-load barrier -- ``fma(h, w, +0.0)`` is bit-identical either way)."""
    bdot = zero
    adot = zero
    for c in range(BA_CHUNKS):
        for j in range(4):
            hlo, hhi = _unpack_bf16x2(hx[c][j])
            blo, bhi = _unpack_bf16x2(bw[c][j])
            alo, ahi = _unpack_bf16x2(aw[c][j])
            bdot = bdot + hlo * blo
            bdot = bdot + hhi * bhi
            adot = adot + hlo * alo
            adot = adot + hhi * ahi
    return bdot, adot


def _gmem_ptr(tensor, elem_offset, dtype=BF16):
    """A 16-byte-aligned GLOBAL pointer to ``tensor[elem_offset]`` for the bulk copies (``cp.async.bulk`` wants an explicit gmem
    address space; a generic pointer -- what a host trace of CPU tensors carries -- is refused by the primitive)."""
    return cute.make_ptr(dtype, (tensor.iterator + elem_offset).toint(), cute.AddressSpace.gmem, assumed_align=16)


def _row_gaddr(state, slot, head, rowm, col0, SS):
    """The global byte address of columns ``col0 ..`` of state row ``rowm`` of head ``head`` in slot ``slot``."""
    return (state.iterator + (slot * SS + (head * HEAD_DIM + rowm) * HEAD_DIM + col0)).toint()


def _load_row_gmem(state, slot, head, rowm, col0, C, SS):
    """One row slice (``C`` fp32) straight into registers by 16-B vector loads (the ldg128 path)."""
    base = _row_gaddr(state, slot, head, rowm, col0, SS)
    vals = []
    for c4 in range(C // 4):
        vals.extend(ld_global_v4(base + 16 * c4, FP32))
    return vals


def _store_row_gmem(state, slot, head, rowm, col0, C, SS, vals):
    base = _row_gaddr(state, slot, head, rowm, col0, SS)
    for c4 in range(C // 4):
        st_global_v4(base + 16 * c4, vals[4 * c4 : 4 * c4 + 4], FP32)


def _load_row_smem(sbase, m, C):
    """One row slice from the thread's shared-memory box (``sbase`` = its first row's slice address; rows at the 512-B pitch)."""
    vals = []
    for c4 in range(C // 4):
        vals.extend(ld_shared_v4(sbase + m * ROW_BYTES + 16 * c4, FP32))
    return vals


def _store_row_smem(sbase, m, C, vals):
    for c4 in range(C // 4):
        st_shared_v4(sbase + m * ROW_BYTES + 16 * c4, vals[4 * c4 : 4 * c4 + 4], FP32)


def _slab_row0(cfg, head, r):
    """The first state row (within the slot) of the rows CTA ``r`` of head ``head`` owns: ``head x 128 + CTA_ROWS x r``."""
    return head * HEAD_DIM + cfg.CTA_ROWS * r


def _issue_state_boxes(cfg, state, state_s, mb_state, slot, head, r, SS):
    """Arm the state mbarrier with this CTA's slab bytes and issue the RG box copies (box ``b`` = the CTA's rows ``BOX_ROWS b ..``, one
    contiguous run in global memory, landing at box pitch ``STATE_BOX_PITCH_ELEMS`` in shared memory)."""
    nvvm.mbarrier_arrive_expect_tx(mb_state, cfg.STATE_SLAB_BYTES_CTA)
    for b in range(cfg.RG):
        nvvm.cp_async_bulk_shared_cluster_global(
            state_s.subview(b * cfg.STATE_BOX_PITCH_ELEMS),
            _gmem_ptr(state, slot * SS + (_slab_row0(cfg, head, r) + b * cfg.BOX_ROWS) * HEAD_DIM, FP32),
            mb_state,
            cfg.STATE_BOX_BYTES,
        )


def _issue_state_stores(cfg, state, state_s, slot, head, r, SS):
    """The RG box stores of the updated rows (``cp.async.bulk.global.shared::cta.bulk_group``) and their commit; the caller waits the
    group (``wait_group.read 0``) before it exits."""
    for b in range(cfg.RG):
        nvvm.cp_async_bulk_global_shared_cta(
            _gmem_ptr(state, slot * SS + (_slab_row0(cfg, head, r) + b * cfg.BOX_ROWS) * HEAD_DIM, FP32),
            state_s.subview(b * cfg.STATE_BOX_PITCH_ELEMS),
            cfg.STATE_BOX_BYTES,
        )
    nvvm.cp_async_bulk_commit_group()


def _issue_state_half_stores(cfg, state, state_s, slot, head, r, SS, wg):
    """STORE_PIPELINE: warpgroup ``wg``'s half of every box -- its warps hold rows ``BOX_ROWS b + (BOX_ROWS / 2) wg ..`` of box ``b``, one
    contiguous run of ``STATE_BOX_BYTES / 2`` in shared memory and in global memory -- as RG bulk stores + one commit, issued by that
    warpgroup's lane 0 as soon as its own pass 2 is published (the other warpgroup's half follows on its own clock)."""
    half_rows = cfg.BOX_ROWS // 2
    for b in range(cfg.RG):
        nvvm.cp_async_bulk_global_shared_cta(
            _gmem_ptr(state, slot * SS + (_slab_row0(cfg, head, r) + b * cfg.BOX_ROWS + wg * half_rows) * HEAD_DIM, FP32),
            state_s.subview(b * cfg.STATE_BOX_PITCH_ELEMS + wg * half_rows * HEAD_DIM),
            cfg.STATE_BOX_BYTES // 2,
        )
    nvvm.cp_async_bulk_commit_group()


def _conv_piece_channel0(piece, head, kh):
    if piece == 0:
        return kh * HEAD_DIM
    if piece == 1:
        return K_CHANNEL0 + kh * HEAD_DIM
    return V_CHANNEL0 + head * HEAD_DIM


def _issue_conv_param_copies(cw, cs, cw_s, hist_s, mb_arr, head, kh, slot, CSS, CSD, CST):
    """The ``cp.async.bulk`` copies of this head's conv WEIGHTS (3 x 1 KB) and conv HISTORY (per tap 3 x 256 B when the taps are 10240
    apart, or one 768 B run per piece when the taps are contiguous per channel): written by earlier steps, never by the predecessor
    grid, so they are issued ahead of a PDL wait."""
    for piece in range(3):
        ch0 = _conv_piece_channel0(piece, head, kh)
        nvvm.cp_async_bulk_shared_cluster_global(
            cw_s.subview(piece * HEAD_DIM * CONV_WIDTH), _gmem_ptr(cw, ch0 * CONV_WIDTH), mb_arr, HEAD_DIM * CONV_WIDTH * 2
        )
        if CSD == 1:
            for tap in range(CONV_WIDTH - 1):
                nvvm.cp_async_bulk_shared_cluster_global(
                    hist_s.subview(piece * (CONV_WIDTH - 1) * HEAD_DIM + tap * HEAD_DIM), _gmem_ptr(cs, slot * CSS + ch0 + tap * CST), mb_arr, HEAD_DIM * 2
                )
        else:
            nvvm.cp_async_bulk_shared_cluster_global(
                hist_s.subview(piece * (CONV_WIDTH - 1) * HEAD_DIM), _gmem_ptr(cs, slot * CSS + ch0 * (CONV_WIDTH - 1)), mb_arr, (CONV_WIDTH - 1) * HEAD_DIM * 2
            )


def _issue_conv_x_copies(x, x_s, mb_arr, row, head, kh, XS):
    """The three 256-B copies of this head's Q / K / V pieces of ``x`` (the predecessor's output: after the PDL wait)."""
    for piece in range(3):
        ch0 = _conv_piece_channel0(piece, head, kh)
        nvvm.cp_async_bulk_shared_cluster_global(x_s.subview(piece * HEAD_DIM), _gmem_ptr(x, row * XS + ch0), mb_arr, HEAD_DIM * 2)


def _hist(hist_s, piece, tap, c, CSD):
    """The bf16 conv-history tap ``tap`` of channel ``c`` of piece ``piece`` in shared memory (layout follows the copies above)."""
    if CSD == 1:
        return hist_s[piece * (CONV_WIDTH - 1) * HEAD_DIM + tap * HEAD_DIM + c]
    return hist_s[piece * (CONV_WIDTH - 1) * HEAD_DIM + c * (CONV_WIDTH - 1) + tap]


def _conv_channel(piece, u, x_s, cw_s, hist_s, CSD):
    """The reference ``conv_channel`` (PROUND): four bf16-rounded tap products summed in order, the stock SiLU, bf16.  Returns the
    fp32 value and the raw bf16 taps 1, 2 and the current input (the shifted history of this channel)."""
    ch = piece * HEAD_DIM + u
    taps = [_hist(hist_s, piece, tap, u, CSD) for tap in range(CONV_WIDTH - 1)]
    cur = x_s[ch]
    vals = [t.to(FP32) for t in taps] + [cur.to(FP32)]
    acc = FP32(0)
    for j in range(CONV_WIDTH):
        prod = vals[j] * cw_s[ch * CONV_WIDTH + j].to(FP32)
        prod = prod.to(BF16).to(FP32)
        acc = acc + prod
    value = _stock_conv_silu(acc).to(BF16).to(FP32)
    return value, taps[1], taps[2], cur


def _slice_values(base_addr, C):
    """The ``C`` fp32 values of one q / k slice from shared memory (16-B vector loads)."""
    vals = []
    for c4 in range(C // 4):
        vals.extend(ld_shared_v4(base_addr + 16 * c4, FP32))
    return vals


def _pick_row_value(values, tpr):
    """``values[tpr]`` for ``tpr < R`` by a select chain (every lane holds every row's value after the butterflies)."""
    val = values[0]
    for m in range(1, len(values)):
        val = _select_f32(tpr == m, values[m], val)
    return val


def _k_group(k, kbase, g, cfg):
    """The 8 k values of 8-column group ``g``: from the resident slice, or re-read from shared memory (``K_RESIDENT`` off)."""
    return k[8 * g : 8 * g + 8] if cfg.K_RESIDENT else _slice_values(kbase + 32 * g, 8)


def _load_rows_gmem(state, slot, head, rowbase, col0, R, C, SS):
    """The thread's R row slices straight into registers (the ldg128 path)."""
    return [_load_row_gmem(state, slot, head, rowbase + m, col0, C, SS) for m in range(R)]


def _load_rows_smem(sbase, R, C):
    """The thread's R row slices from its shared-memory box (the register-resident bulk form)."""
    return [_load_row_smem(sbase, m, C) for m in range(R)]


def _state_passes(cfg, S_rows, state, slot, head, rowbase, col0, sbase, k_s, q_s, v_s, vbase, tpr, decay, beta, SS):
    """Pass 1 (decay, ``dot = h . k``), the innovation, pass 2 (``h += inn * k``, ``o = h . q``, the row back).  Returns the per-row
    ``f32(bf16(o))`` values.  Every chain is the reference's: 8 sequential FFMA per 8-column group, the balanced tree over the thread's
    groups, ``log2(TPR)`` xor shuffles (offsets ``RG << step``: the partners differ in the tpr lane bits).  ``S_rows`` = the R row
    slices in registers (ldg128, or bulk after the LDS), or None on the streamed bulk form (each pass re-reads the row from ``sbase``
    and pass 2 recomputes ``h = S * decay``: one FMUL, bit-identical)."""
    R, C, G, TPR, RG = cfg.R, cfg.C, cfg.GROUPS, cfg.TPR, cfg.RG
    steps = TPR.bit_length() - 1
    k_base = _smem_addr(k_s) + tpr * (cfg.SLICE_PITCH * 4)
    q_base = _smem_addr(q_s) + tpr * (cfg.SLICE_PITCH * 4)
    k = _slice_values(k_base, C) if cfg.K_RESIDENT else None
    streamed = S_rows is None
    # pass 1
    h = []
    dots = []
    for m in range(R):
        S = _load_row_smem(sbase, m, C) if streamed else S_rows[m]
        hm = [S[j] * decay for j in range(C)]
        parts = []
        for g in range(G):
            kg = _k_group(k, k_base, g, cfg)
            p = FP32(0)
            for j in range(8):
                p = p + hm[8 * g + j] * kg[j]
            parts.append(p)
        s = _balanced_sum(parts)
        for step in range(steps):
            s = s + cute.arch.shuffle_sync_bfly(s, offset=RG << step)
        dots.append(s)
        h.append(None if streamed else hm)
    inn = [(v_s[vbase + m * cfg.V_PERM_STRIDE] - dots[m]) * beta for m in range(R)]
    # pass 2
    values = []
    for m in range(R):
        if streamed:
            # the recomputed decayed row is SINGLE-USE here (pass 1's copy fed the dot), so the compiler could fuse it into the update
            # as fma(S, decay, inn * k) -- the reference rounds the product first (its FMUL), then fma(inn, k, h): spell it unfusable
            S = _load_row_smem(sbase, m, C)
            hm = [_mul_rn_f32(S[j], decay) for j in range(C)]
        else:
            hm = h[m]
        parts = []
        for g in range(G):
            kg = _k_group(k, k_base, g, cfg)
            qg = _slice_values(q_base + 32 * g, 8)
            o = FP32(0)
            for j in range(8):
                idx = 8 * g + j
                hm[idx] = hm[idx] + inn[m] * kg[j]
                o = o + hm[idx] * qg[j]
            parts.append(o)
        s = _balanced_sum(parts)
        for step in range(steps):
            s = s + cute.arch.shuffle_sync_bfly(s, offset=RG << step)
        values.append(s.to(BF16).to(FP32))
        if cfg.BULK_STATE:
            _store_row_smem(sbase, m, C, hm)
        else:
            _store_row_gmem(state, slot, head, rowbase + m, col0, C, SS, hm)
    return values


def _row_square_sum(values):
    """The in-thread leaf of the norm tree: the balanced sum of the rows' squares (explicit ``mul.rn.f32``, never fused)."""
    return _balanced_sum([_mul_rn_f32(v, v) for v in values])


def _norm_total(norm_part, RG, WARPS):
    """The reference 128-leaf balanced tree from the aligned-block partials: over the WARPS warps of each box in order, then over the
    boxes in order."""
    return _balanced_sum([_balanced_sum([norm_part[g * WARPS + wi] for wi in range(WARPS)]) for g in range(RG)])


# --------------------------------------------------------------------------------------------------------------------------
# the thread geometry (plain Python: the same integers in the role bodies and in the tail)
# --------------------------------------------------------------------------------------------------------------------------
def _lane_geometry(cfg, tid, r, state_s):
    """``warp, lane, tpr, rg, rowbase, col0, orow, vbase, u, wl, sbase`` of thread ``tid`` of CTA ``r`` of the head's pair (0 on the
    one-CTA form) -- the mapping of the module docstring."""
    R, C, RG = cfg.R, cfg.C, cfg.RG
    warp = tid // 32
    lane = tid % 32
    tpr = lane // RG  # the column slice
    rg = lane % RG  # the row group = the state box
    rowbase = cfg.CTA_ROWS * r + cfg.BOX_ROWS * rg + R * warp  # the thread's rows: rowbase + m, m < R (R consecutive, R-aligned rows of box rg)
    col0 = tpr * C
    mo = cutlass.min(tpr, R - 1)  # the row this thread outputs (the tpr < R threads each take one of the row group's R rows)
    orow = rowbase + mo
    vbase = (R * warp) * cfg.V_PERM_STRIDE + r * RG + rg  # v_s index of row rowbase (+ m * V_PERM_STRIDE): the RG lanes of a phase read consecutive words
    u = tid % 128  # the BA partition index and the conv-role channel index
    wl = (tid % 128) // 32  # warp within the warpgroup
    sbase = INT32(0)
    if cfg.BULK_STATE:
        sbase = _smem_addr(state_s) + rg * (cfg.STATE_BOX_PITCH_ELEMS * 4) + (R * warp) * ROW_BYTES + col0 * 4
    return warp, lane, tpr, rg, rowbase, col0, orow, vbase, u, wl, sbase


# --------------------------------------------------------------------------------------------------------------------------
# the role-independent tail: the hand-over, (bulk) the slab wait + LDS, the passes, the norm, the outputs, the shift, the drain
# --------------------------------------------------------------------------------------------------------------------------
@cute.jit
def _tail_live(
    cfg: cutlass.Constexpr,
    tid,
    head,
    row,
    kh,
    r,
    slot,
    scv,
    wv,
    zv_in,
    S_rows_in,
    cs,
    state,
    z,
    out,
    qout,
    raw,
    x_s,
    hist_s,
    q_s,
    k_s,
    v_s,
    scal_s,
    ticket_s,
    norm_part,
    norm_peer_s,
    mb_state,
    mb_norm,
    state_s,
    ZS: cutlass.Constexpr,
    CSS: cutlass.Constexpr,
    CSD: cutlass.Constexpr,
    CST: cutlass.Constexpr,
    SS: cutlass.Constexpr,
    EPS: cutlass.Constexpr,
):
    """A live slot after the role work.  ``S_rows_in`` = the row slices the role body loaded into registers (ldg128) or an empty
    list (bulk: they are read from the shared-memory boxes here).  Traced once per role copy on the twice-traced body, ONCE per kernel
    under ``SINGLE_TAIL``; the slab stores and their drain belong to thread 0."""
    TPR = cfg.TPR
    R = cfg.R
    C = cfg.C
    RG = cfg.RG
    WARPS = cfg.WARPS
    BULK = cfg.BULK_STATE
    SHIFT_CHANNELS = 2 * HEAD_DIM // cfg.SHIFT_LANES
    warp, lane, tpr, rg, rowbase, col0, orow, vbase, u, wl, sbase = _lane_geometry(cfg, tid, r, state_s)
    zv = zv_in
    if cutlass.const_expr(cfg.PDL):
        # the predecessor's output: read after this thread's griddepcontrol.wait (in its role body); issued here, consumed at the epilogue
        zv = z[row * ZS + head * HEAD_DIM + orow].to(FP32)
    _cta_barrier(cfg.THREADS)
    S_rows = S_rows_in
    if cutlass.const_expr(BULK):
        wait(mb_state, 0)
        if cutlass.const_expr(not cfg.STATE_STREAM):
            S_rows = _load_rows_smem(sbase, R, C)
        else:
            S_rows = None
    values = _state_passes(cfg, S_rows, state, slot, head, rowbase, col0, sbase, k_s, q_s, v_s, vbase, tpr, scal_s[0], scal_s[1], SS)
    if cutlass.const_expr(BULK):
        cute.arch.fence_view_async_shared()  # this thread's STS of the updated rows -> visible to the bulk store (async proxy)
        if cutlass.const_expr(cfg.STORE_PIPELINE):
            # this warpgroup's half of every box is final: its warps 1-3 ARRIVE on the warpgroup's named barrier (96 threads, no wait), its
            # warp 0 SYNCS on it (32 threads: warp-uniform participation, no intra-warp divergence at a barrier) and that warp's lane 0
            # issues the RG half-box stores -- the write-back of each half starts at its own pass-2 end instead of at the CTA barrier
            # below.  Ids: 2 on warpgroup 0, 1 on warpgroup 1 (both free again: their earlier uses are complete).  SUM = 96 + 32 = 128
            wg = tid // 128
            store_bar = 2 - wg
            if wl == 0:
                nvvm.barrier_cta_sync(store_bar, thread_count=128)
                if lane == 0:
                    _issue_state_half_stores(cfg, state, state_s, slot, head, r, SS, wg)
            else:
                nvvm.barrier_cta_arrive(store_bar, 128)
    pre = _row_square_sum(values)  # the aligned block of this thread's R consecutive rows
    if tpr == 0:
        norm_part[rg * WARPS + warp] = pre
    _cta_barrier(cfg.THREADS)
    if cutlass.const_expr(BULK and not cfg.STORE_PIPELINE):
        if tid == 0:
            _issue_state_stores(cfg, state, state_s, slot, head, r, SS)
    ss = _norm_total(norm_part, RG, WARPS)  # the balanced tree over this CTA's rows (T128, or T64 of its half)
    if cutlass.const_expr(cfg.TWO_CTAS_PER_HEAD):
        # the peer's T64: thread 0 ships this CTA's T64 into the peer's slot (one st.async completing 4 tx bytes on the peer's mb_norm)
        # after the P4 cluster wait (the peer's mbarrier init is visible), arming its OWN mb_norm first; every thread waits the peer's
        # bytes.  T128 = T64(rows 0-63) + T64(rows 64-127): fp32 addition is commutative, so both ranks compute the same bits.
        nvvm.barrier_cluster_wait()
        if tid == 0:
            nvvm.mbarrier_arrive_expect_tx(mb_norm, NORM_XFER_BYTES)
            peer = INT32(cfg.CTAS_PER_HEAD - 1) - r
            _st_async_f32(_mapa_u32(_smem_addr(norm_peer_s), peer), ss, _mapa_u32(_smem_addr(mb_norm), peer))
        wait(mb_norm, 0)
        ss = ss + norm_peer_s[0]
    inv = cute.math.rsqrt(ss / FP32(HEAD_DIM) + FP32(EPS))
    if tpr < R:
        val = _pick_row_value(values, tpr)
        oi = row * (V_HEADS * HEAD_DIM) + head * HEAD_DIM + orow
        rounded = (((val * inv) * wv) * (zv / (FP32(1) + cute.math.exp(-zv)))).to(BF16)
        qout[oi] = _quant_fp8_word(rounded.to(FP32), scv).to(UINT8)
        out[oi] = rounded
        if cutlass.const_expr(cfg.DEBUG):
            raw[oi] = val.to(BF16)
    # ---- the last of the three readers shifts the K head's Q / K conv history (its taps are in smem) ----
    if ticket_s[0] == cfg.TICKET_LIMIT:
        if tpr == TPR - 1:
            sidx2 = warp * RG + rg
            for i in cutlass.range_constexpr(SHIFT_CHANNELS):
                ix = sidx2 + i * cfg.SHIFT_LANES
                piece = ix // HEAD_DIM
                c = ix % HEAD_DIM
                channel = kh * HEAD_DIM + c + piece * K_CHANNEL0
                off = slot * CSS + channel * CSD
                cs[off] = _hist(hist_s, piece, 1, c, CSD)
                cs[off + CST] = _hist(hist_s, piece, 2, c, CSD)
                cs[off + 2 * CST] = x_s[piece * HEAD_DIM + c]
    if cutlass.const_expr(BULK):
        if cutlass.const_expr(cfg.STORE_PIPELINE):
            if tid % 128 == 0:
                nvvm.cp_async_bulk_wait_group(0, read=True)  # each issuing lane: its half-boxes are read out of smem before the CTA retires
        else:
            if tid == 0:
                nvvm.cp_async_bulk_wait_group(0, read=True)  # the boxes are read out of smem before this CTA retires


@cute.jit
def _tail_null(cfg: cutlass.Constexpr, tid, head, row, r, out, qout, raw, state_s):
    """The null slot: zero outputs, nothing else touched (the PDL wait, when any, was taken in the role body; on the two-CTA form the
    cluster wait pairs with the entry arrive -- nothing is shipped)."""
    warp, lane, tpr, rg, rowbase, col0, orow, vbase, u, wl, sbase = _lane_geometry(cfg, tid, r, state_s)
    if cutlass.const_expr(cfg.TWO_CTAS_PER_HEAD):
        nvvm.barrier_cluster_wait()
    if tpr < cfg.R:
        oi = row * (V_HEADS * HEAD_DIM) + head * HEAD_DIM + orow
        qout[oi] = UINT8(0)
        out[oi] = BF16(0)
        if cutlass.const_expr(cfg.DEBUG):
            raw[oi] = BF16(0)


# --------------------------------------------------------------------------------------------------------------------------
# the warpgroup role body (ROLE 0 = BA, 1 = conv, 2 = both, 3 = neither); WITH_TAIL traces the tail inside it (the twice-traced body)
# --------------------------------------------------------------------------------------------------------------------------
@cute.jit
def _role_body(
    cfg: cutlass.Constexpr,
    ROLE: cutlass.Constexpr[int],
    WITH_TAIL: cutlass.Constexpr[bool],
    tid,
    head,
    row,
    kh,
    r,
    slot,
    scv,
    dtv,
    alv,
    wv,
    zv,
    x,
    cs,
    cw,
    state,
    z,
    counts,
    out,
    qout,
    hidden,
    baweight,
    raw,
    convout,
    baout,
    x_s,
    cw_s,
    hist_s,
    qk_raw,
    q_s,
    k_s,
    v_s,
    scal_s,
    ticket_s,
    ba_part,
    norm_part,
    norm_peer_s,
    mb_arr,
    mb_state,
    mb_norm,
    state_s,
    XS: cutlass.Constexpr,
    ZS: cutlass.Constexpr,
    CSS: cutlass.Constexpr,
    CSD: cutlass.Constexpr,
    CST: cutlass.Constexpr,
    SS: cutlass.Constexpr,
    EPS: cutlass.Constexpr,
):
    R = cfg.R
    C = cfg.C
    BULK = cfg.BULK_STATE
    TWO = cfg.TWO_CTAS_PER_HEAD
    DO_BA = ROLE in (0, 2)
    DO_CONV = ROLE in (1, 2)
    # the slab LOADS (and the state mbarrier) belong to the conv role's lane 0 under CONV_FIRST (issued right after its conv-operand
    # copies: one queue, the small copies first), else to the BA role's warp 0 lane 0; the slab STORES belong to thread 0 (the tail)
    ISSUE_LOADS = DO_CONV if cfg.CONV_FIRST else DO_BA
    warp, lane, tpr, rg, rowbase, col0, orow, vbase, u, wl, sbase = _lane_geometry(cfg, tid, r, state_s)

    # ---- entry loads of the role (parameters: never a predecessor's output) and the mbarrier inits ----
    if cutlass.const_expr(DO_BA):
        bw, aw = _ba_weight_loads(baweight, head, u)
        if cutlass.const_expr(not cfg.PDL):
            hx_entry = _ba_hidden_loads(hidden, row, u)
    if cutlass.const_expr(cfg.PIN_ENTRY_LOADS):
        nvvm.bar_warp_sync(0xFFFFFFFF)  # the entry loads stay above the slot branch (a warp-scope ordering point, no wait)
    if cutlass.const_expr(DO_CONV):
        if wl == 0:
            if lane == 0:
                nvvm.mbarrier_init(mb_arr, CONV_MBAR_ARRIVERS)
                if cutlass.const_expr(BULK and cfg.CONV_FIRST):
                    nvvm.mbarrier_init(mb_state, STATE_MBAR_ARRIVERS)  # the same lane arms it after its conv copies; one fence for both
                    if cutlass.const_expr(TWO):
                        nvvm.mbarrier_init(mb_norm, NORM_MBAR_ARRIVERS)
                nvvm.fence_mbarrier_init()
        _wg_barrier(1)
    if cutlass.const_expr(BULK and ISSUE_LOADS and not cfg.CONV_FIRST):
        if wl == 0:
            if lane == 0:
                nvvm.mbarrier_init(mb_state, STATE_MBAR_ARRIVERS)
                if cutlass.const_expr(TWO):
                    nvvm.mbarrier_init(mb_norm, NORM_MBAR_ARRIVERS)
                nvvm.fence_mbarrier_init()  # published to the waiters by the first CTA-wide barrier, which every waiter passes first
    if cutlass.const_expr(TWO):
        # P4: every thread of both CTAs arrives (relaxed; the init lane's fence.mbarrier_init orders its inits ahead) -- the matching
        # cluster wait sits in the tail, right before the peer's st.async may land on this CTA's mb_norm
        nvvm.barrier_cluster_arrive_relaxed()

    if slot > 0:
        # ---- the state slab: row slices -> registers (ldg128) or the RG boxes -> smem (bulk, async); the conv operands -> smem ----
        if cutlass.const_expr(cfg.PIN_ENTRY_LOADS and DO_BA):
            if u == 0:
                scal_s[3] = FP32(0)
        S_rows = []
        if cutlass.const_expr(not BULK):
            S_rows = _load_rows_gmem(state, slot, head, rowbase, col0, R, C, SS)
        zero = FP32(0)
        if cutlass.const_expr(cfg.PIN_ENTRY_LOADS and DO_BA):
            # The BA warpgroup's state rows are requested BEFORE its dot chain.  ptxas keeps memory ops above a bar.sync (which does
            # not wait for in-flight register loads) but hoists pure ALU work above it, so the dots start from a +0.0 that is READ
            # FROM SHARED MEMORY after the barrier: a shared load cannot cross the barrier, and the FFMA chains depend on it.
            _wg_barrier(2)
            zero = scal_s[3]
        if cutlass.const_expr(BULK and ISSUE_LOADS and not cfg.CONV_FIRST):
            if wl == 0:
                if lane == 0:
                    _issue_state_boxes(cfg, state, state_s, mb_state, slot, head, r, SS)
        if cutlass.const_expr(DO_CONV):
            if wl == 0:
                if lane == 0:
                    nvvm.mbarrier_arrive_expect_tx(mb_arr, CONV_BULK_BYTES)
                    _issue_conv_param_copies(cw, cs, cw_s, hist_s, mb_arr, head, kh, slot, CSS, CSD, CST)
                    if cutlass.const_expr(not cfg.PDL):
                        _issue_conv_x_copies(x, x_s, mb_arr, row, head, kh, XS)
                    if cutlass.const_expr(BULK and cfg.CONV_FIRST):
                        _issue_state_boxes(cfg, state, state_s, mb_state, slot, head, r, SS)  # behind the 6 KB of conv operands, never ahead
        if cutlass.const_expr(cfg.PDL):
            # everything above was written before this step's graph; the QKVZ projection (x, z) and the normed hidden it consumed
            # are read only past this point (z is read at the top of the tail)
            wait_on_dependent_grids()
            if cutlass.const_expr(DO_BA):
                hx_pdl = _ba_hidden_loads(hidden, row, u)
            if cutlass.const_expr(DO_CONV):
                if wl == 0:
                    if lane == 0:
                        _issue_conv_x_copies(x, x_s, mb_arr, row, head, kh, XS)

        # ---- BA role: dots -> decay / beta ----
        if cutlass.const_expr(DO_BA):
            hx = hx_pdl if cutlass.const_expr(cfg.PDL) else hx_entry
            bdot, adot = _ba_dots(hx, bw, aw, zero)
            for step in cutlass.range_constexpr(5):
                bdot = bdot + cute.arch.shuffle_sync_bfly(bdot, offset=1 << step)
                adot = adot + cute.arch.shuffle_sync_bfly(adot, offset=1 << step)
            if lane == 0:
                ba_part[wl] = bdot
                ba_part[4 + wl] = adot
            _wg_barrier(2)
            btotal = FP32(0)
            atotal = FP32(0)
            if lane < 4:
                btotal = ba_part[lane]
                atotal = ba_part[4 + lane]
            for step in cutlass.range_constexpr(5):
                btotal = btotal + cute.arch.shuffle_sync_bfly(btotal, offset=1 << step)
                atotal = atotal + cute.arch.shuffle_sync_bfly(atotal, offset=1 << step)
            bvalue = btotal.to(BF16).to(FP32)
            avalue = atotal.to(BF16).to(FP32)
            av = avalue + dtv
            soft = av
            if av <= FP32(SOFTPLUS_LINEAR_ABOVE):
                soft = cute.math.log(FP32(1) + cute.math.exp(av))
            decay = cute.math.exp(-cute.math.exp(alv) * soft)
            beta = FP32(1) / (FP32(1) + cute.math.exp(-bvalue))
            if u == 0:
                scal_s[0] = decay
                scal_s[1] = beta
                if cutlass.const_expr(cfg.DEBUG):
                    baout[row * 2 * V_HEADS + head] = bvalue.to(BF16)
                    baout[row * 2 * V_HEADS + head + V_HEADS] = avalue.to(BF16)

        # ---- conv role: operands landed -> ticket, conv + SiLU, V shift, RMS scales, scaled q / k ----
        if cutlass.const_expr(DO_CONV):
            wait(mb_arr, 0)
            ticket = INT32(0)
            if wl == 0:
                if lane == 0:  # lane 0 both here and at the ticket_s store below: the value is carried between the two guards
                    ticket = _atom_inc_relaxed_gpu((counts.iterator + (row * TICKET_STRIDE + kh * 32)).toint(), INT32(cfg.TICKET_LIMIT))
            qval, _, _, _ = _conv_channel(0, u, x_s, cw_s, hist_s, CSD)
            kval, _, _, _ = _conv_channel(1, u, x_s, cw_s, hist_s, CSD)
            vval, vc1, vc2, vcur = _conv_channel(2, u, x_s, cw_s, hist_s, CSD)
            vbase_cs = slot * CSS + (V_CHANNEL0 + head * HEAD_DIM + u) * CSD
            if cutlass.const_expr(TWO):
                if u // cfg.CTA_ROWS == r:  # each CTA of the pair shifts the V history of its own 64 channels (both compute all 128)
                    cs[vbase_cs] = vc1
                    cs[vbase_cs + CST] = vc2
                    cs[vbase_cs + 2 * CST] = vcur
            else:
                cs[vbase_cs] = vc1
                cs[vbase_cs + CST] = vc2
                cs[vbase_cs + 2 * CST] = vcur
            qk_raw[u] = qval
            qk_raw[HEAD_DIM + u] = kval
            v_s[(u % cfg.BOX_ROWS) * cfg.V_PERM_STRIDE + u // cfg.BOX_ROWS] = vval  # the permuted index of head row u (see vbase)
            if cutlass.const_expr(cfg.DEBUG):
                convout[row * QKV_CHANNELS + V_CHANNEL0 + head * HEAD_DIM + u] = vval.to(BF16)
                if head % 3 == 0:
                    convout[row * QKV_CHANNELS + kh * HEAD_DIM + u] = qval.to(BF16)
                    convout[row * QKV_CHANNELS + K_CHANNEL0 + kh * HEAD_DIM + u] = kval.to(BF16)
            _wg_barrier(1)
            kcol = (u % RMS_LANES) * RMS_COLS
            qs = FP32(0)
            ks = FP32(0)
            for j in cutlass.range_constexpr(RMS_COLS):
                qj = qk_raw[kcol + j]
                kj = qk_raw[HEAD_DIM + kcol + j]
                qs = qs + qj * qj
                ks = ks + kj * kj
            for step in cutlass.range_constexpr(RMS_LANES.bit_length() - 1):
                qs = qs + cute.arch.shuffle_sync_bfly(qs, offset=1 << step)
                ks = ks + cute.arch.shuffle_sync_bfly(ks, offset=1 << step)
            qscale = FP32(Q_SCALE) / cute.math.sqrt(qs + FP32(RMS_EPS))
            kscale = FP32(1) / cute.math.sqrt(ks + FP32(RMS_EPS))
            sidx = (u // C) * cfg.SLICE_PITCH + (u % C)
            q_s[sidx] = qval * qscale
            k_s[sidx] = kval * kscale
            if wl == 0:
                if lane == 0:
                    ticket_s[0] = ticket

        if cutlass.const_expr(WITH_TAIL):
            _tail_live(
                cfg, tid, head, row, kh, r, slot, scv, wv, zv, S_rows, cs, state, z, out, qout, raw, x_s, hist_s, q_s, k_s, v_s, scal_s, ticket_s,
                norm_part, norm_peer_s, mb_state, mb_norm, state_s, ZS, CSS, CSD, CST, SS, EPS,
            )  # fmt: skip
    else:
        if cutlass.const_expr(cfg.PDL):
            wait_on_dependent_grids()
        if cutlass.const_expr(WITH_TAIL):
            _tail_null(cfg, tid, head, row, r, out, qout, raw, state_s)


# --------------------------------------------------------------------------------------------------------------------------
# the kernel
# --------------------------------------------------------------------------------------------------------------------------
@cute.kernel
def frost_gdn_decode_fused(
    cfg: cutlass.Constexpr,
    x: cute.Tensor,
    cs: cute.Tensor,
    cw: cute.Tensor,
    al: cute.Tensor,
    dt: cute.Tensor,
    state: cute.Tensor,
    indices: cute.Tensor,
    z: cute.Tensor,
    w: cute.Tensor,
    scale: cute.Tensor,
    counts: cute.Tensor,
    out: cute.Tensor,
    qout: cute.Tensor,
    hidden: cute.Tensor,
    baweight: cute.Tensor,
    raw: Optional[cute.Tensor],
    convout: Optional[cute.Tensor],
    baout: Optional[cute.Tensor],
    XS: cutlass.Constexpr,
    ZS: cutlass.Constexpr,
    CSS: cutlass.Constexpr,
    CSD: cutlass.Constexpr,
    CST: cutlass.Constexpr,
    SS: cutlass.Constexpr,
    EPS: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    bx, row, _ = cute.arch.block_idx()
    # the two-CTA form: blocks (2 h, 2 h + 1) are one cluster along x; r = the CTA's rank in it (= %cluster_ctarank) and its row half
    head = bx // cfg.CTAS_PER_HEAD if cfg.TWO_CTAS_PER_HEAD else bx
    r = bx % cfg.CTAS_PER_HEAD if cfg.TWO_CTAS_PER_HEAD else INT32(0)
    kh = head // 3
    if cutlass.const_expr(cfg.PDL_RELEASE):
        launch_dependent_grids()

    SMEM = cutlass.AddressSpace.smem
    x_s = cutlass.Array(BF16, 3 * HEAD_DIM, space=SMEM, alignment=16)
    cw_s = cutlass.Array(BF16, 3 * HEAD_DIM * CONV_WIDTH, space=SMEM, alignment=16)
    hist_s = cutlass.Array(BF16, 3 * (CONV_WIDTH - 1) * HEAD_DIM, space=SMEM, alignment=16)
    qk_raw = cutlass.Array(FP32, 2 * HEAD_DIM, space=SMEM, alignment=16)
    q_s = cutlass.Array(FP32, cfg.TPR * cfg.SLICE_PITCH, space=SMEM, alignment=16)
    k_s = cutlass.Array(FP32, cfg.TPR * cfg.SLICE_PITCH, space=SMEM, alignment=16)
    v_s = cutlass.Array(FP32, HEAD_DIM, space=SMEM, alignment=16)
    scal_s = cutlass.Array(FP32, 4, space=SMEM, alignment=16)
    ticket_s = cutlass.Array(INT32, 4, space=SMEM, alignment=16)
    ba_part = cutlass.Array(FP32, 8, space=SMEM, alignment=16)
    norm_part = cutlass.Array(FP32, cfg.NORM_PARTS, space=SMEM, alignment=16)
    mb_arr = cutlass.Array(INT64, 1, space=SMEM, alignment=16)
    mb_state = cutlass.Array(INT64, 1, space=SMEM, alignment=16)
    # the two-CTA form: the peer's norm partial lands in norm_peer_s[0] and completes 4 tx bytes on mb_norm (one-CTA form: unused aliases)
    norm_peer_s = cutlass.Array(FP32, 4, space=SMEM, alignment=16) if cfg.TWO_CTAS_PER_HEAD else scal_s
    mb_norm = cutlass.Array(INT64, 1, space=SMEM, alignment=16) if cfg.TWO_CTAS_PER_HEAD else mb_state
    state_s = cutlass.Array(FP32, cfg.STATE_SMEM_ELEMS, space=SMEM, alignment=16)
    if cutlass.const_expr(cfg.SMEM_PAD_BYTES > 0):
        pad_s = cutlass.Array(INT32, cfg.SMEM_PAD_BYTES // 4, space=SMEM, alignment=16)
        if head == INT32(0x7FFFFFF):  # never true for a 48-wide grid: keeps the occupancy pad allocated
            pad_s[0] = tid

    # ---- the entry loads every thread makes (the slot and the parameters: never a predecessor's output) ----
    warp, lane, tpr, rg, rowbase, col0, orow, vbase, u, wl, sbase = _lane_geometry(cfg, tid, r, state_s)
    slot = indices[row].to(INT64)
    scv = scale[0]
    dtv = dt[head].to(FP32)
    alv = al[head]
    wv = w[orow].to(FP32)
    zv = FP32(0)
    if cutlass.const_expr(not cfg.PDL):
        zv = z[row * ZS + head * HEAD_DIM + orow].to(FP32)  # off PDL the predecessor's output loads with the parameters (under PDL: the tail, after the wait)

    role_args = (
        tid, head, row, kh, r, slot, scv, dtv, alv, wv, zv,
        x, cs, cw, state, z, counts, out, qout, hidden, baweight, raw, convout, baout,
        x_s, cw_s, hist_s, qk_raw, q_s, k_s, v_s, scal_s, ticket_s, ba_part, norm_part, norm_peer_s, mb_arr, mb_state, mb_norm, state_s,
        XS, ZS, CSS, CSD, CST, SS, EPS,
    )  # fmt: skip
    tail_args = (
        tid, head, row, kh, r, slot, scv, wv, zv, [], cs, state, z, out, qout, raw,
        x_s, hist_s, q_s, k_s, v_s, scal_s, ticket_s, norm_part, norm_peer_s, mb_state, mb_norm, state_s,
        ZS, CSS, CSD, CST, SS, EPS,
    )  # fmt: skip
    if cutlass.const_expr(cfg.THREADS == 128):
        _role_body(cfg, 2, True, *role_args)  # one warpgroup, both roles in sequence: one traced body already
    elif cutlass.const_expr(cfg.SINGLE_TAIL and cfg.BULK_STATE):
        # the role arms reconverge before the hand-over; the tail is traced ONCE (one PTX site per CTA-wide barrier); the register
        # state path keeps the twice-traced body (its row registers would have to cross the join)
        if cutlass.const_expr(cfg.WARPGROUP_SPLIT):
            if tid < 128:
                _role_body(cfg, 0, False, *role_args)
            else:
                _role_body(cfg, 1, False, *role_args)
        else:
            if tid < 128:
                _role_body(cfg, 2, False, *role_args)
            else:
                _role_body(cfg, 3, False, *role_args)
        if slot > 0:
            _tail_live(cfg, *tail_args)
        else:
            _tail_null(cfg, tid, head, row, r, out, qout, raw, state_s)
    elif cutlass.const_expr(cfg.WARPGROUP_SPLIT):
        if tid < 128:
            _role_body(cfg, 0, True, *role_args)
        else:
            _role_body(cfg, 1, True, *role_args)
    else:
        if tid < 128:
            _role_body(cfg, 2, True, *role_args)
        else:
            _role_body(cfg, 3, True, *role_args)


frost_gdn_decode_fused.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def launch(
    cfg: cutlass.Constexpr,
    x: cute.Tensor,
    cs: cute.Tensor,
    cw: cute.Tensor,
    al: cute.Tensor,
    dt: cute.Tensor,
    state: cute.Tensor,
    indices: cute.Tensor,
    z: cute.Tensor,
    w: cute.Tensor,
    scale: cute.Tensor,
    counts: cute.Tensor,
    out: cute.Tensor,
    qout: cute.Tensor,
    hidden: cute.Tensor,
    baweight: cute.Tensor,
    raw: Optional[cute.Tensor],
    convout: Optional[cute.Tensor],
    baout: Optional[cute.Tensor],
    BATCH: cutlass.Constexpr,
    XS: cutlass.Constexpr,
    ZS: cutlass.Constexpr,
    CSS: cutlass.Constexpr,
    CSD: cutlass.Constexpr,
    CST: cutlass.Constexpr,
    SS: cutlass.Constexpr,
    EPS: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    """One CTA per (V head, request row) -- a cluster of two per head under ``TWO_CTAS_PER_HEAD``; 256 (or 128) threads;
    ``MIN_CTAS_PER_SM`` sets ptxas's register cap (1 -> 255, 2 -> 128 at 256 threads) and ``CTAS_PER_SM`` the shared-memory cap.
    ``raw`` / ``convout`` / ``baout`` are the DEBUG dumps: ``None`` unless ``cfg.DEBUG`` -- the DSL specializes on ``None`` and the
    slots are compiled out (every use sits under the DEBUG constexpr), never bound to an allocation."""
    kernel = frost_gdn_decode_fused(
        cfg, x, cs, cw, al, dt, state, indices, z, w, scale, counts, out, qout, hidden, baweight, raw, convout, baout, XS, ZS, CSS, CSD, CST, SS, EPS
    )
    if cutlass.const_expr(cfg.TWO_CTAS_PER_HEAD):
        kernel.launch(
            grid=(V_HEADS * cfg.CTAS_PER_HEAD, BATCH, 1),
            block=(cfg.THREADS, 1, 1),
            cluster=(cfg.CTAS_PER_HEAD, 1, 1),
            stream=stream,
            min_blocks_per_mp=cfg.MIN_CTAS_PER_SM,
            use_pdl=cfg.PDL,
        )
    else:
        kernel.launch(
            grid=(V_HEADS, BATCH, 1),
            block=(cfg.THREADS, 1, 1),
            stream=stream,
            min_blocks_per_mp=cfg.MIN_CTAS_PER_SM,
            use_pdl=cfg.PDL,
        )
