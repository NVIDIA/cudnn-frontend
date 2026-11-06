# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shared-expert DECODE kernel pair (bf16 in / out, fp32 accumulation in a fixed order per output element): launch 1 ``gate_up``
(``act = bf16(silu(x W_gate^T) * (x W_up^T))`` plus the partials of the gate logit ``x W_g^T``), launch 2 ``down``
(``y = bf16((act W_d^T) * sigmoid(gate))``); config and knobs in ``shared_expert_decode_config.py``, the host entry in
``moe/frost/shared_expert_decode.py``.  Math path: ``mma.sync.m16n8k16`` bf16 -> fp32 in both launches, tokens on M (``T`` zero-padded
to ``16 x TOKEN_TILES`` rows), weight rows on N, K walked in k16 steps; operands reach the tensor core by ``ldmatrix`` off 16-B-padded
shared-memory rows.  Arch-portable PTX (``mma.sync``, ``cp.async``, ``cp.async.bulk``, ``st.async``, clusters, ``griddepcontrol``); the
arch axis is an admission fact of the host entry (cc 10.x), not a directory.

THE FIXED REDUCTION ORDER (bitwise replay, batch invariance).  Per output element, with ``mma_k16(A_row, B_col, C)`` the tensor core's own
fixed-order k16 contraction of ONE row of A against ONE column of B:

* ``G[t][j]`` (``j = 4c + i``) ``= sum_{ks} ( sum_{w = 0..7} ( chain over (chunk, step in 4w .. 4w+3) of mma_k16(x[t, k16], W_gu[j, k16]) ) )``
  -- the innermost chain is warp ``w``'s single accumulator in (chunk, step) order; the warp sum is the sequence ``((C_0 + C_1) + C_2) + ..``
  through ``red_s``; the slice sum ``((C^{ks=0} + C^{ks=1}) + ..)`` in rank order through ``mail_s``.  ``U[t][j]`` identically with the up
  row (the n column next to it).  ``act[t][j] = bf16(silu(G) * U)`` with ``silu(v) = v * sigmoid(v)``.
* ``gate[t] = sigmoid( butterfly_5( sequence over the lane's GATE_PARTS / 32 partials ) )`` with each partial ``p_q = butterfly_5( fma chain
  over the lane's column pairs of CTA q's GATE_COLS columns )`` -- a fixed GATE_PARTS-leaf tree per token (64 / 128 / 256 leaves by K_SPLIT).
* ``y[t][n] = bf16( gate[t] x chain_{s = 0 .. I/16 - 1} mma_k16(act[t, k16(s)], W_d[n, k16(s)]) )`` -- one warp, one chain.

Why T-independent and batch-invariant: (1) the grid, ``K_SPLIT``, the chunking, the warp -> k-step map and the n-tile map depend on the
geometry alone; (2) a token occupies a ROW of the A operand (m-tile ``t // 16``, row ``t % 16``): the mma computes row ``g`` of C from
row ``g`` of A alone, so the zero-filled rows past ``T`` and the other tokens cannot touch it; (3) ROW-POSITION INVARIANCE (a premise,
not a proof): the ``m16n8k16`` datapath's rounding of a row's k16 contraction does not depend on WHICH row of the fragment the token
sits in (row 0 of m-tile 0 alone at T = 1, row ``t % 16`` of m-tile ``t // 16`` inside T = 24) -- the batch-invariance cells of the
determinism suite (a token alone vs inside T 6 / 24, the permuted rows, the duplicated token) are its tripwire; if it ever fails, the
contingency is one m-tile per token at row 0 (``TILES = T``, every other row zero; 24 mma per warp per chunk instead of 1-2) or the
warp-FMA path -- NOT ``TOKEN_TILES = 2`` for every T, which keeps a token's row position tied to its batch index; (4) no atomics, no
T-dependent scheduling, every reduction a fixed tree.  Knobs ``TOKEN_TILES``, ``ROWS_PER_CTA``, ``X_STAGES``, ``W_PHASES``, ``PDL*`` and
``MIN_CTAS_PER_SM`` move no addition; ``K_SPLIT`` changes the slice order AND the gate tree's leaf count (a rounding-order arm, pinned per
geometry by the host entry's default).

BARRIER TABLE -- launch 1 (``gate_up``), 256 threads, cluster ``(K_SPLIT, 1, 1)``; ``SUM(issuing lanes) == init count`` on every path:

  mb_w[p], p < W_PHASES   tx mbarrier (TMA_LOAD class), init W_MBAR_ARRIVERS = 1.  Arrive: warp 0 lane 0 (``if warp == 0: if lane == 0``,
                          the warp gate AND the lane gate, P3 / P4): ``arrive.expect_tx(W_PHASE_BYTES = 8 x K_SLICE / W_PHASES x 2)``, then
                          eight ``cp.async.bulk.shared::cluster.global`` of W_PHASE_ROW_BYTES each (local destination: bytes stay local,
                          P9), issued at entry BEFORE ``griddepcontrol.wait`` (weights are never a predecessor's output).  Wait: all 256
                          threads, phase 0, right before the first ``ldmatrix`` of the phase's K range (after that chunk's ``bar.sync``).
                          SUM: 1 arrive = init 1; tx 8 x W_PHASE_ROW_BYTES = expect.  Bit-neutral in W_PHASES: each warp's (chunk, step)
                          chain order is unchanged.
  mb_kred (K_SPLIT > 1)   tx-only mbarrier on the LEADER (rank 0), init KRED_MBAR_ARRIVERS = 1 (every CTA initialises it with the same
                          constant, P8; the peers' copies are never waited).  Arrive: leader warp 0 lane 0 ``arrive.expect_tx(KRED_BYTES =
                          (K_SPLIT - 1) x TILES x 32 x 16)`` after the cluster wait; peers (rank >= 1) warp 0, ALL 32 lanes, TILES x
                          ``st.async ... complete_tx::bytes.v4.f32`` (16 B each) onto ``mapa(mail_s slot, 0)`` / ``mapa(mb_kred, 0)``
                          after the cluster wait (P4 step 3: no cross-CTA op before every CTA's init is published).  Wait: leader warp 0,
                          32 lanes, phase 0, with the CLUSTER-scope acquire (``wait_cluster_acquire``: the peers' complete_tx releases
                          at cluster scope; a ``.acquire.cta`` wait would not formally order the leader's ``ld.shared`` of mail_s after the
                          remote write -- the GDN ``mb_norm`` precedent runs the cta form bitwise in production, so this is a formal gap
                          closed, not an observed one).  SUM: 1 arrive = init 1; tx = (K_SPLIT - 1) x TILES x 32 lanes x 16 B = expect; a
                          complete_tx landing before the expect_tx leaves the count transiently negative and the phase open until the
                          arrive (the GDN ``mb_norm`` pattern).
  init                    warp 0 lane 0: ``mbarrier.init(mb_w[p], 1)`` for every p, ``mbarrier.init(mb_kred, 1)``, ``fence.mbarrier_init``;
                          then ``barrier.cluster.arrive.relaxed`` by EVERY thread (P4 step 3, ``const_expr(K_SPLIT > 1)``), and
                          ``barrier.cluster.wait`` by every thread right before the first cross-CTA op (after the intra-CTA fold's
                          ``bar.sync``); the CTA-local ``bar.sync`` of chunk 0 publishes the inits to the local mb_w waiters.
  x ring (P18)            ``cp.async`` groups + ``bar.sync 0, 256``, no mbarrier.  EVERY thread: the prologue commits X_PROLOGUE_GROUPS =
                          X_STAGES - 1 groups (chunks 0 .. X_STAGES - 2; X_COPIES copies each, a dead row = a zero-filled copy of a CLAMPED,
                          valid source address, never a skipped copy); iteration i: ``cp.async.wait_group(X_WAIT_PENDING = X_STAGES - 2)``
                          [ledger: committed = (X_STAGES - 1) + i, pending <= X_STAGES - 2 -> chunks <= i landed], ``bar.sync 0, 256`` [RAW:
                          everyone's copies visible; WAR: the stage refilled next, (i + X_STAGES - 1) % X_STAGES = (i - 1) % X_STAGES, was
                          last read in iteration i - 1, before THIS barrier], issue chunk i + X_STAGES - 1 if < X_CHUNKS and COMMIT (an
                          empty group on the tail path: every thread commits on every path), compute chunk i; after the loop
                          ``wait_group 0`` (nothing outstanding at exit, P18 rule 5).  SUM at every ``bar.sync``: 256 = the launch's threads;
                          X_CHUNKS barriers per CTA under ``range_constexpr`` (uniform control flow, P18 rule 4).
  red_s publish           ``bar.sync 0, 256`` (#X_CHUNKS + 1): every thread, after warps 1 .. 7 stored their accumulators.  SUM 256.
  exit                    leader: its mb_kred wait IS the drain (every shipped byte landed before its epilogue; the bulk copies completed
                          at the mb_w waits, the cp.async groups at ``wait_group 0``); peers: ``st.async`` has a register source and nobody
                          addresses a peer's SMEM, so no drain is owed (P15: the consumer-side wait is the drain; no ``_empty`` ring exists).

BARRIER TABLE -- launch 2 (``down``), THREADS2 = 4 x ROWS_PER_CTA threads, no cluster, no mbarrier:

  group W                 every thread: WD_COPIES ``cp.async`` of 16 B (the CTA's W_d rows into wd_s), ``commit_group`` -- BEFORE the wait.
  griddepcontrol.wait     every thread, after group W, before any read of ``act`` / ``gate_part`` and before any global write (P17).
  group A                 every thread: ACT_COPIES ``cp.async`` of 16 B of the act rows (src-size 0 for rows >= T, the source row clamped to
                          0: in-bounds in the [32, I] scratch, its stale bytes ignored by the zero-fill), ``commit_group``.  Ledger: 2 groups.
  bar.sync 0, THREADS2    the ONE CTA barrier: every thread after ``cp.async.wait_group 0`` and after the gate fold wrote ``sig_s[t]``
                          (warp w lane 0: tokens w, w + WARPS2, ..).  SUM THREADS2.  It publishes wd_s, act_s AND sig_s.
  exit                    plain ``st.global`` of y; ``wait_group 0`` already taken; nothing outstanding.

SMEM TABLES (no descriptor reads any buffer: every row pitch is 128 n + 16 B, so consecutive rows shift by 4 banks and the 8 row addresses of
one ``ldmatrix`` phase (8 rows x 16 B) cover all 32 banks; ``cp.async`` / bulk writes land consecutive 16-B granules of ONE row per warp
instruction: conflict-free).  Launch 1 (allocation order; the defaults: K_SPLIT 2, I 256, TILES 1 / 2):
  w_s      8 rows x K_SLICE bf16 at pitch K_SLICE x 2 + 16 (8,208 B)            65,664 B   written by the bulk copies (async proxy), read by
                                                                                           ``ldmatrix.x2`` (the mbarrier wait orders async ->
                                                                                           generic; no fence_proxy owed in this direction)
  x_s      X_STAGES x TPAD x 512 bf16 at pitch 1,040 B                 133,120 / 133,120 B  ``cp.async`` 16 B (generic proxy; wait_group +
                                                                                           bar.sync order it), read by ``ldmatrix.x4``; chunk i
                                                                                           at stage i % X_STAGES, WAR by the next bar.sync
  red_s    7 warps x TILES x 128 fp32                                   3,584 / 7,168 B    lanes of warps 1-7, 16 B per lane at +16 l (32 x
                                                                                           16 B = 512 B contiguous per tile); read by warp 0
  mail_s   (K_SPLIT - 1) x TILES x 128 fp32                                512 / 1,024 B    PEER ``st.async.v4`` (DSMEM), 16 B per lane at
                                                                                           +16 l; lifetime = cluster init sync .. the leader's
                                                                                           mb_kred wait (the SENDER's timeline): its own
                                                                                           bytes, aliases nothing
  mb_w, mb_kred  u64 slots                                                        32 B
  total                                                                202,912 / 206,496 B <= 232,448 B opt-in (cc 10.x); one CTA per SM
  (I 128, K_SPLIT 4: w_s 32,896 + x_s 4 x 16,640 = 66,560 (all 4 chunks resident) + red_s + mail_s 1,536 + 32 -> ~104,600 B: two CTAs per
  SM by shared memory, 128 CTAs on 152 SMs need one.)
Launch 2 (I 256; TPAD 16 / 32): wd_s 64 rows x pitch 528 B = 33,792 (``cp.async``, ``ldmatrix.x2``; 528 = 4 x 128 + 16); act_s TPAD x 528 =
  8,448 / 16,896 (``cp.async``, ``ldmatrix.x4``); sig_s 32 fp32 = 128; total 42,368 / 50,816 B (I 128: pitch 272 = 2 x 128 + 16).

PDL SITES (P17).  Launch 1: ``launch_dependents`` at entry under PDL_RELEASE; ``griddepcontrol.wait`` (unconditional) AFTER the mbarrier
inits, the W expect_tx + bulk copies and BEFORE the first x ``cp.async``, the gate-logit x loads and every global write.  Launch 2:
``launch_dependents`` at entry under PDL_RELEASE (inert for a consumer launched without the attribute); ``griddepcontrol.wait`` AFTER the
W_d prefetch + commit and BEFORE the act copies, the gate_part loads and every global write.  Neither kernel allocates TMEM, so the
release-above-alloc deadlock class does not arise; launch 2's CTAs (<= 51 KB) cannot share an SM with a live launch-1 CTA at the default
form (~203 KB), so under PDL_PAIR its first CTAs start on the idle SMs and the rest as launch-1 CTAs retire.
"""

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm
from cutlass._mlir.dialects import arith

from cudnn.frost.tile_dsl.barrier import launch_dependent_grids, wait, wait_cluster_acquire, wait_on_dependent_grids
from cudnn.frost.tile_dsl.pointwise import f16x2_to_f32, fp32_to_fp16, opaque_f32_zero, opaque_i32_zero, sigmoid
from cudnn.frost.tile_dsl.tma import (
    cp_async_commit,
    cp_async_gather_row16,
    cp_async_wait,
    ld_global,
    ld_shared_v4,
    mapa_u32,
    st_async_v4_f32,
    st_global,
    st_shared_v4,
)

from .shared_expert_decode_config import BUTTERFLY_STEPS, CHUNK_K, KRED_MBAR_ARRIVERS, MMA_K, MMA_M, PAIRS_PER_CLUSTER, W_MBAR_ARRIVERS, W_ROWS

FP32 = cutlass.Float32
BF16 = cutlass.BFloat16
INT32 = cutlass.Int32
INT64 = cutlass.Int64

MMA_PTX = "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};"


# --------------------------------------------------------------------------------------------------------------------------
# straight-line helpers (plain Python: traced inline, no DSL control flow inside)
# --------------------------------------------------------------------------------------------------------------------------
def _smem_addr(arr):
    """32-bit shared-window address of a shared-memory ``cutlass.Array``."""
    return INT32(arr.data_ptr().toint())


def _gmem_ptr(tensor, elem_offset, dtype=BF16):
    """A 16-byte-aligned GLOBAL pointer to ``tensor[elem_offset]`` for the async copies (``cp.async`` / ``cp.async.bulk`` want an explicit
    gmem address space; a generic pointer -- what a host trace of CPU tensors carries -- is refused by the primitive)."""
    return cute.make_ptr(dtype, (tensor.iterator + elem_offset).toint(), cute.AddressSpace.gmem, assumed_align=16)


def _select_i32(cond, a, b):
    """``cond ? a : b`` on Int32 values as one select (no branch)."""
    return INT32(arith.select(cond.ir_value(), a.ir_value(), b.ir_value()))


def _select_f32(cond, a, b):
    return FP32(arith.select(cond.ir_value(), a.ir_value(), b.ir_value()))


def _fma_f32(a, b, c):
    """``fma.rn.f32``: one rounding, spelled so the fold order of the gate logit is the one documented (never a compiler contraction)."""
    return nvvm.inline_ptx("fma.rn.f32 $0, $1, $2, $3;", write_only_types=[FP32], read_only_args=[a, b, c])


def _mma_bf16(a, b, c):
    """One ``mma.sync.m16n8k16`` bf16 -> fp32: ``a`` = 4 A-fragment words, ``b`` = 2 B-fragment words, ``c`` = 4 fp32; returns the 4 new fp32."""
    return list(
        nvvm.inline_ptx(
            MMA_PTX,
            write_only_types=[FP32, FP32, FP32, FP32],
            read_only_args=[a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], c[2], c[3]],
        )
    )


def _ldmatrix_a(arr, row0, col0, pitch_elems, lane):
    """The A fragment (16 x 16 bf16, rows = tokens, k contiguous) by one ``ldmatrix.x4``: lane ``l`` addresses row ``row0 + l % 16``,
    column ``col0 + (l // 16) x 8`` -- matrices (rows 0-7, k 0-7), (rows 8-15, k 0-7), (rows 0-7, k 8-15), (rows 8-15, k 8-15) = a0 .. a3."""
    ptr = arr.subview((row0 + lane % 16) * pitch_elems + col0 + (lane // 16) * 8)
    v = nvvm.ldmatrix(ptr.data_ptr(), 4, nvvm.MMALayout.ROW)
    return [v[0], v[1], v[2], v[3]]


def _ldmatrix_b(arr, row0, col0, pitch_elems, lane):
    """The B fragment (8 n rows x 16 k, k contiguous) by one ``ldmatrix.x2``: lane ``l`` addresses row ``row0 + l % 8``, column
    ``col0 + ((l % 16) // 8) x 8`` (lanes 16-31 repeat valid addresses) -- b0 = k 0-7, b1 = k 8-15 of the lane's n column."""
    ptr = arr.subview((row0 + lane % 8) * pitch_elems + col0 + ((lane % 16) // 8) * 8)
    v = nvvm.ldmatrix(ptr.data_ptr(), 2, nvvm.MMALayout.ROW)
    return [v[0], v[1]]


def _zeros4():
    return [opaque_f32_zero(), opaque_f32_zero(), opaque_f32_zero(), opaque_f32_zero()]


def _butterfly(v):
    """The 5-step xor butterfly (offsets 1, 2, 4, 8, 16): every lane ends with the warp sum in one fixed tree."""
    for step in range(BUTTERFLY_STEPS):
        v = v + cute.arch.shuffle_sync_bfly(v, offset=1 << step)
    return v


def _silu(v):
    return v * sigmoid(v)


def _issue_x_chunk(cfg, x, x_s, chunk, stage, k_base, tid, T, XS):
    """Every thread's X_COPIES 16-B ``cp.async`` of chunk ``chunk`` into ring stage ``stage``: copy ``q = it x THREADS + tid`` is row
    ``q // 64``, granule ``q % 64``; a row past ``T`` is a zero-filled copy of a CLAMPED (valid) source address (row 0)."""
    gpr = cfg.X_GRANULES_PER_ROW
    pitch = cfg.X_PITCH_ELEMS
    stage_base = stage * cfg.TPAD * pitch
    for it in range(cfg.X_COPIES):
        q = it * cfg.THREADS + tid
        row = q // gpr
        g16 = q % gpr
        live = row < T
        srow = _select_i32(live, row, INT32(0))
        dst = x_s.subview(stage_base + row * pitch + g16 * 8)
        src = _gmem_ptr(x, srow * XS + k_base + chunk * CHUNK_K + g16 * 8)
        cp_async_gather_row16(dst, src, live)


def _add4(acc, m, v):
    """``acc[4m .. 4m+3] += v[0 .. 3]`` on the flat accumulator list (fp32 adds in index order)."""
    for j in range(4):
        acc[4 * m + j] = acc[4 * m + j] + v[j]


@cute.jit
def _gate_up_epilogue(cfg: cutlass.Constexpr, acc, act: cute.Tensor, cluster, lane, T):
    """Leader warp 0: ``act[t][4c + i] = bf16(silu(gate_i[t]) * up_i[t])`` for ``t < T`` (lane (g, i): c0 / c1 = gate / up of token g, c2 / c3
    of token g + 8, per m-tile; ``acc`` = the flat ``4 x TOKEN_TILES`` accumulator list)."""
    g = lane // 4
    i = lane % 4
    col = cluster * PAIRS_PER_CLUSTER + i
    for m in cutlass.range_constexpr(cfg.TOKEN_TILES):
        tok0 = m * MMA_M + g
        tok1 = tok0 + 8
        if tok0 < T:
            act[tok0 * cfg.I + col] = (_silu(acc[4 * m]) * acc[4 * m + 1]).to(BF16)
        if tok1 < T:
            act[tok1 * cfg.I + col] = (_silu(acc[4 * m + 2]) * acc[4 * m + 3]).to(BF16)


@cute.jit
def _leader_finish(cfg: cutlass.Constexpr, acc, mail_base, mb_kred, act: cute.Tensor, cluster, lane, T):
    """Rank 0's warp 0 (K_SPLIT > 1): arm ``mb_kred`` (lane 0), wait the peers' ships with the cluster-scope acquire, add their
    accumulators in rank order, write ``act``."""
    mine = []  # the leader's OWN list: this arm's values must not leak into the peer arm's reads through the shared list object
    for j in cutlass.range_constexpr(4 * cfg.TOKEN_TILES):
        mine.append(acc[j])
    if lane == 0:
        nvvm.mbarrier_arrive_expect_tx(mb_kred, cfg.KRED_BYTES)
    wait_cluster_acquire(mb_kred, 0)
    for r in cutlass.range_constexpr(1, cfg.K_SPLIT):
        for m in cutlass.range_constexpr(cfg.TOKEN_TILES):
            v = ld_shared_v4(mail_base + (((r - 1) * cfg.TOKEN_TILES + m) * 128 + lane * 4) * 4, FP32)
            _add4(mine, m, v)
    _gate_up_epilogue(cfg, mine, act, cluster, lane, T)


@cute.jit
def _peer_ship(cfg: cutlass.Constexpr, acc, mail_base, mb_kred, rank, lane):
    """Rank >= 1's warp 0: one ``st.async.v4`` per lane and m-tile into the LEADER's ``mail_s`` slot, completing 16 tx bytes on the leader's
    ``mb_kred`` (both addresses ``mapa``-translated to rank 0)."""
    leader = opaque_i32_zero()  # rank 0 as a register operand (an immediate would take the asm's `n` constraint)
    peer_mbar = mapa_u32(_smem_addr(mb_kred), leader)
    for m in cutlass.range_constexpr(cfg.TOKEN_TILES):
        slot = mail_base + (((rank - 1) * cfg.TOKEN_TILES + m) * 128 + lane * 4) * 4
        st_async_v4_f32(mapa_u32(slot, leader), acc[4 * m : 4 * m + 4], peer_mbar)


@cute.jit
def _warp0_finish(cfg: cutlass.Constexpr, acc, red_base, mail_base, mb_kred, act: cute.Tensor, cluster, rank, lane, T):
    """Warp 0 after the mainloop: the intra-CTA fold (warps 1 .. 7 in order from ``red_s``), then the cluster fold (K_SPLIT > 1: the leader
    collects, the peers ship) or the epilogue."""
    for w in cutlass.range_constexpr(1, cfg.WARPS):
        for m in cutlass.range_constexpr(cfg.TOKEN_TILES):
            v = ld_shared_v4(red_base + (((w - 1) * cfg.TOKEN_TILES + m) * 128 + lane * 4) * 4, FP32)
            _add4(acc, m, v)
    if cutlass.const_expr(cfg.K_SPLIT > 1):
        if rank == 0:
            _leader_finish(cfg, acc, mail_base, mb_kred, act, cluster, lane, T)
        else:
            _peer_ship(cfg, acc, mail_base, mb_kred, rank, lane)
    else:
        _gate_up_epilogue(cfg, acc, act, cluster, lane, T)


@cute.jit
def _gate_logit(cfg: cutlass.Constexpr, x: cute.Tensor, wg: cute.Tensor, gate_part: cute.Tensor, cluster, rank, lane, k_base, T, XS: cutlass.Constexpr):
    """Warp 1 of CTA (c, ks): partial ``q = ks x CLUSTERS + c`` of the gate logit over its GATE_COLS columns ``k_base + c x GATE_COLS ..``:
    lane ``l`` walks column pairs ``2l + 64p`` (predicated on ``< GATE_COLS``; a dead pair leaves the chain untouched = +0.0 at a fixed
    position; its source column is clamped to a valid one), ``acc = fma(x_hi, w_hi, fma(x_lo, w_lo, acc))`` per pair in order, the 5-step
    butterfly, lane 0 writes ``gate_part[t][q]``; after ``griddepcontrol.wait`` (x is the predecessor's output)."""
    q = rank * cfg.CLUSTERS + cluster
    gk0 = k_base + cluster * cfg.GATE_COLS
    cols = []
    lives = []
    wlo = []
    whi = []
    for p in cutlass.range_constexpr(cfg.GATE_PAIRS_PER_LANE):
        colp = lane * 2 + 64 * p
        live = colp < cfg.GATE_COLS
        colc = _select_i32(live, colp, INT32(0))
        lo, hi = f16x2_to_f32(ld_global((wg.iterator + (gk0 + colc)).toint(), INT32), dtype=BF16)
        cols.append(colc)
        lives.append(live)
        wlo.append(lo)
        whi.append(hi)
    for t in cutlass.range_constexpr(cfg.TPAD):
        if t < T:
            acc = opaque_f32_zero()
            for p in cutlass.range_constexpr(cfg.GATE_PAIRS_PER_LANE):
                xlo, xhi = f16x2_to_f32(ld_global((x.iterator + (t * XS + gk0 + cols[p])).toint(), INT32), dtype=BF16)
                nxt = _fma_f32(xhi, whi[p], _fma_f32(xlo, wlo[p], acc))
                acc = _select_f32(lives[p], nxt, acc)
            acc = _butterfly(acc)
            if lane == 0:
                gate_part[t * cfg.GATE_PARTS + q] = acc


# --------------------------------------------------------------------------------------------------------------------------
# launch 1: gate_up
# --------------------------------------------------------------------------------------------------------------------------
@cute.kernel
def frost_shared_expert_gate_up(
    cfg: cutlass.Constexpr,
    x: cute.Tensor,
    wgu: cute.Tensor,
    wg: cute.Tensor,
    act: cute.Tensor,
    gate_part: cute.Tensor,
    T: cutlass.Int32,
    XS: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    S = cfg.K_SPLIT
    TILES = cfg.TOKEN_TILES
    STAGES = cfg.X_STAGES_EFF
    warp = tid // 32
    lane = tid % 32
    cluster = bx // S if cutlass.const_expr(S > 1) else bx
    rank = bx % S if cutlass.const_expr(S > 1) else INT32(0)
    k_base = rank * cfg.K_SLICE
    if cutlass.const_expr(cfg.PDL_RELEASE):
        launch_dependent_grids()

    SMEM = cutlass.AddressSpace.smem
    w_s = cutlass.Array(BF16, W_ROWS * cfg.W_PITCH_ELEMS, space=SMEM, alignment=16)
    x_s = cutlass.Array(BF16, STAGES * cfg.TPAD * cfg.X_PITCH_ELEMS, space=SMEM, alignment=16)
    red_s = cutlass.Array(FP32, (cfg.WARPS - 1) * TILES * 128, space=SMEM, alignment=16)
    mail_s = cutlass.Array(FP32, (S - 1) * TILES * 128, space=SMEM, alignment=16) if cutlass.const_expr(S > 1) else red_s
    mb_w = cutlass.Array(INT64, cfg.W_PHASES, space=SMEM, alignment=16)
    mb_kred = cutlass.Array(INT64, 1, space=SMEM, alignment=16) if cutlass.const_expr(S > 1) else mb_w

    # ---- init + the W slice (parameters: never a predecessor's output) -- ONE warp, ONE lane ----
    if warp == 0:
        if lane == 0:
            for p in cutlass.range_constexpr(cfg.W_PHASES):
                nvvm.mbarrier_init(mb_w.subview(p), W_MBAR_ARRIVERS)
            if cutlass.const_expr(S > 1):
                nvvm.mbarrier_init(mb_kred, KRED_MBAR_ARRIVERS)
            nvvm.fence_mbarrier_init()
            for p in cutlass.range_constexpr(cfg.W_PHASES):
                nvvm.mbarrier_arrive_expect_tx(mb_w.subview(p), cfg.W_PHASE_BYTES)
                for i in cutlass.range_constexpr(PAIRS_PER_CLUSTER):
                    for h in cutlass.range_constexpr(2):  # h 0 = the gate row, 1 = the up row -> w_s rows 2i, 2i + 1
                        srow = h * cfg.I + cluster * PAIRS_PER_CLUSTER + i
                        nvvm.cp_async_bulk_shared_cluster_global(
                            w_s.subview((2 * i + h) * cfg.W_PITCH_ELEMS + p * cfg.W_PHASE_K),
                            _gmem_ptr(wgu, srow * cfg.H + k_base + p * cfg.W_PHASE_K),
                            mb_w.subview(p),
                            cfg.W_PHASE_ROW_BYTES,
                        )
    if cutlass.const_expr(S > 1):
        # P4 step 3: every thread of every CTA arrives (relaxed; the init lane's fence orders its inits ahead); the matching wait sits
        # right before the first cross-CTA op
        nvvm.barrier_cluster_arrive_relaxed()

    # ---- the predecessor's output (x) is read only past this point; every global write sits below it ----
    wait_on_dependent_grids()

    # ---- the x ring prologue: chunks 0 .. STAGES - 2 (X_PROLOGUE_GROUPS groups, every thread) ----
    for c in cutlass.range_constexpr(STAGES - 1):
        _issue_x_chunk(cfg, x, x_s, c, c, k_base, tid, T, XS)
        cp_async_commit()

    # ---- the gate logit partial (warp 1; overlaps the ring) ----
    if warp == 1:
        _gate_logit(cfg, x, wg, gate_part, cluster, rank, lane, k_base, T, XS)

    # ---- mainloop: warp w runs k16 steps 4w .. 4w + 3 of every chunk, one accumulator chain per (m-tile, lane) ----
    acc = []
    for m in cutlass.range_constexpr(TILES):
        acc.extend(_zeros4())
    kw = warp * 4 * MMA_K  # the warp's first k column inside a chunk
    for i in cutlass.range_constexpr(cfg.X_CHUNKS):
        cp_async_wait(STAGES - 2)
        nvvm.barrier_cta_sync(0, thread_count=cfg.THREADS)
        nxt = i + STAGES - 1
        if cutlass.const_expr(nxt < cfg.X_CHUNKS):
            _issue_x_chunk(cfg, x, x_s, nxt, nxt % STAGES, k_base, tid, T, XS)
        cp_async_commit()  # every thread, every path (an empty group on the tail)
        if cutlass.const_expr(i % cfg.CHUNKS_PER_W_PHASE == 0):
            wait(mb_w.subview(i // cfg.CHUNKS_PER_W_PHASE), 0)  # the W phase this chunk reads has landed (all 256 threads)
        stage_rows = (i % STAGES) * cfg.TPAD
        for s in cutlass.range_constexpr(4):
            kk = kw + s * MMA_K
            b = _ldmatrix_b(w_s, 0, i * CHUNK_K + kk, cfg.W_PITCH_ELEMS, lane)
            for m in cutlass.range_constexpr(TILES):
                a = _ldmatrix_a(x_s, stage_rows + m * MMA_M, kk, cfg.X_PITCH_ELEMS, lane)
                c = _mma_bf16(a, b, acc[4 * m : 4 * m + 4])
                acc[4 * m] = c[0]
                acc[4 * m + 1] = c[1]
                acc[4 * m + 2] = c[2]
                acc[4 * m + 3] = c[3]
    cp_async_wait(0)

    # ---- the intra-CTA fold: warps 1 .. 7 publish, warp 0 adds in warp order, then the cluster fold / the epilogue ----
    red_base = _smem_addr(red_s)
    mail_base = _smem_addr(mail_s)
    if warp > 0:
        for m in cutlass.range_constexpr(TILES):
            st_shared_v4(red_base + (((warp - 1) * TILES + m) * 128 + lane * 4) * 4, acc[4 * m : 4 * m + 4], FP32)
    nvvm.barrier_cta_sync(0, thread_count=cfg.THREADS)
    if cutlass.const_expr(S > 1):
        nvvm.barrier_cluster_wait()  # every thread: the peers' inits are published before any cross-CTA op below
    if warp == 0:
        _warp0_finish(cfg, acc, red_base, mail_base, mb_kred, act, cluster, rank, lane, T)


frost_shared_expert_gate_up.set_name_prefix("cudnn", remove_cutlass_symbol=True)


# --------------------------------------------------------------------------------------------------------------------------
# launch 2: down
# --------------------------------------------------------------------------------------------------------------------------
@cute.kernel
def frost_shared_expert_down(
    cfg: cutlass.Constexpr,
    act: cute.Tensor,
    gate_part: cute.Tensor,
    wd: cute.Tensor,
    y: cute.Tensor,
    T: cutlass.Int32,
    YS: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    TILES = cfg.TOKEN_TILES
    I = cfg.I  # noqa: E741
    warp = tid // 32
    lane = tid % 32
    n0 = bx * cfg.ROWS_PER_CTA
    if cutlass.const_expr(cfg.PDL_RELEASE):
        launch_dependent_grids()

    SMEM = cutlass.AddressSpace.smem
    wd_s = cutlass.Array(BF16, cfg.ROWS_PER_CTA * cfg.WD_PITCH_ELEMS, space=SMEM, alignment=16)
    act_s = cutlass.Array(BF16, cfg.TPAD * cfg.WD_PITCH_ELEMS, space=SMEM, alignment=16)
    sig_s = cutlass.Array(FP32, 32, space=SMEM, alignment=16)
    gpr = cfg.WD_GRANULES_PER_ROW
    pitch = cfg.WD_PITCH_ELEMS

    # ---- group W: the CTA's W_d rows (static weights) -- BEFORE the wait ----
    for it in cutlass.range_constexpr(cfg.WD_COPIES):
        q = it * cfg.THREADS2 + tid
        row = q // gpr
        g16 = q % gpr
        nvvm.cp_async_shared_global(wd_s.subview(row * pitch + g16 * 8), _gmem_ptr(wd, (n0 + row) * I + g16 * 8), 16, nvvm.LoadCacheModifier.CG)
    cp_async_commit()

    # ---- launch 1's outputs (act, gate_part) are read only past this point; every global write sits below it ----
    wait_on_dependent_grids()

    # ---- group A: the act rows (rows past T zero-filled from a clamped, in-bounds source) ----
    for it in cutlass.range_constexpr(cfg.ACT_COPIES):
        q = it * cfg.THREADS2 + tid
        row = q // gpr
        g16 = q % gpr
        live = row < T
        srow = _select_i32(live, row, INT32(0))
        cp_async_gather_row16(act_s.subview(row * pitch + g16 * 8), _gmem_ptr(act, srow * I + g16 * 8), live)
    cp_async_commit()

    # ---- the gate fold: warp w folds tokens w, w + WARPS2, ..: lane l reads its GATE_PARTS / 32 partials in order, the butterfly, sigmoid ----
    PP = cfg.GATE_PARTS_PER_LANE
    for j in cutlass.range_constexpr(cfg.TOKENS_PER_WARP2):
        t = warp + cfg.WARPS2 * j
        if t < T:
            base = t * cfg.GATE_PARTS + lane * PP
            logit = gate_part[base]
            for k in cutlass.range_constexpr(1, PP):
                logit = logit + gate_part[base + k]
            logit = _butterfly(logit)
            if lane == 0:
                sig_s[t] = sigmoid(logit)

    cp_async_wait(0)
    nvvm.barrier_cta_sync(0, thread_count=cfg.THREADS2)  # the ONE CTA barrier: wd_s, act_s and sig_s are published

    # ---- the mma chain: warp w owns n-tile w over the full K = I ----
    acc = []
    for m in cutlass.range_constexpr(TILES):
        acc.extend(_zeros4())
    for s in cutlass.range_constexpr(cfg.K_STEPS2):
        b = _ldmatrix_b(wd_s, warp * 8, s * MMA_K, pitch, lane)
        for m in cutlass.range_constexpr(TILES):
            a = _ldmatrix_a(act_s, m * MMA_M, s * MMA_K, pitch, lane)
            c = _mma_bf16(a, b, acc[4 * m : 4 * m + 4])
            acc[4 * m] = c[0]
            acc[4 * m + 1] = c[1]
            acc[4 * m + 2] = c[2]
            acc[4 * m + 3] = c[3]

    # ---- epilogue: y[t][n0 + 8w + 2 i .. + 1] = bf16(acc x sigmoid[t]) for t < T (one 4-B store per lane and token) ----
    g = lane // 4
    i = lane % 4
    n = n0 + warp * 8 + i * 2
    for m in cutlass.range_constexpr(TILES):
        tok0 = m * MMA_M + g
        tok1 = tok0 + 8
        if tok0 < T:
            sg = sig_s[tok0]
            st_global((y.iterator + (tok0 * YS + n)).toint(), fp32_to_fp16(acc[4 * m] * sg, acc[4 * m + 1] * sg, dtype=BF16), INT32)
        if tok1 < T:
            sg = sig_s[tok1]
            st_global((y.iterator + (tok1 * YS + n)).toint(), fp32_to_fp16(acc[4 * m + 2] * sg, acc[4 * m + 3] * sg, dtype=BF16), INT32)


frost_shared_expert_down.set_name_prefix("cudnn", remove_cutlass_symbol=True)


# --------------------------------------------------------------------------------------------------------------------------
# the host-side launch of the pair (one compiled entry; both kernels on one stream)
# --------------------------------------------------------------------------------------------------------------------------
@cute.jit
def launch(
    cfg: cutlass.Constexpr,
    x: cute.Tensor,
    wgu: cute.Tensor,
    wd: cute.Tensor,
    wg: cute.Tensor,
    act: cute.Tensor,
    gate_part: cute.Tensor,
    y: cute.Tensor,
    T: cutlass.Int32,
    XS: cutlass.Constexpr,
    YS: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    """Launch 1: grid ``(CLUSTERS x K_SPLIT, 1, 1)`` in clusters of ``K_SPLIT``, 256 threads, the programmatic-stream-serialization attribute
    under ``cfg.PDL``; launch 2: grid ``(H / ROWS_PER_CTA, 1, 1)``, ``THREADS2`` threads, the attribute under ``cfg.PDL_PAIR``.  ``T`` is a
    runtime argument (rows past it are zero-filled and never stored); the strides and the config are the compile key."""
    k1 = frost_shared_expert_gate_up(cfg, x, wgu, wg, act, gate_part, T, XS)
    if cutlass.const_expr(cfg.K_SPLIT > 1):
        k1.launch(grid=(cfg.GRID1, 1, 1), block=(cfg.THREADS, 1, 1), cluster=(cfg.K_SPLIT, 1, 1), stream=stream, use_pdl=cfg.PDL)
    else:
        k1.launch(grid=(cfg.GRID1, 1, 1), block=(cfg.THREADS, 1, 1), stream=stream, use_pdl=cfg.PDL)
    k2 = frost_shared_expert_down(cfg, act, gate_part, wd, y, T, YS)
    k2.launch(grid=(cfg.GRID2, 1, 1), block=(cfg.THREADS2, 1, 1), stream=stream, use_pdl=cfg.PDL_PAIR)
