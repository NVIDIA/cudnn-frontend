# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shared-expert DECODE kernel pair (bf16 in / out, fp32 accumulation in a fixed order per output element): launch 1 ``gate_up``
(``act = bf16(silu(x W_gate^T) * (x W_up^T))`` plus the gate logit ``x W_g^T`` as per-group partial sums), launch 2 ``down``
(``y = bf16((act W_d^T) * sigmoid(gate))``); config and knobs in ``shared_expert_decode_config.py``, the host entry in
``moe/frost/shared_expert_decode.py``.  Math path: ``mma.sync.m16n8k16`` bf16 -> fp32 in both launches, tokens on M (``T`` zero-padded
to ``16 x TOKEN_TILES`` rows), weight rows on N, K walked in k16 steps; operands reach the tensor core by ``ldmatrix`` off 16-B-padded
shared-memory rows.  Arch-portable PTX (``mma.sync``, ``cp.async``, ``cp.async.bulk``, ``st.async``, clusters, ``griddepcontrol``); the
arch axis is an admission fact of the host entry (cc 10.x), not a directory.

TWO FORMS of launch 1 share one kernel body (``cutlass.const_expr`` knobs): the DEFAULT one-wave form (``K_SPLIT`` 8, ``X_BULK_ROWS``: the
``T`` live x rows of the CTA's 1024-column K slice resident by one bulk copy each, no ring; ``N_GROUPS`` n8 tiles per CTA by the one-wave
rule; ``GATE_CLUSTER_FOLD``: the gate logit's leaves added on the cluster leader) and the v1 RING form (``K_SPLIT`` 2 / 4, a ``cp.async``
ring of 512-column x chunks, the leaves folded by launch 2), kept as the A/B arm whose goldens are the ``ksplit2`` / ``ksplit4`` sets.

THE FIXED REDUCTION ORDER (bitwise replay, batch invariance).  Per output element, with ``mma_k16(A_row, B_col, C)`` the tensor core's own
fixed-order k16 contraction of ONE row of A against ONE column of B:

* ``G[t][j]`` (``j = 4g + i``, group ``g``) ``= sum_{ks} ( sum_{w = 0..7} ( chain over (chunk, step in 4w .. 4w+3) of mma_k16(x[t, k16], W_gu[j, k16]) ) )``
  -- the innermost chain is warp ``w``'s single accumulator in (chunk, step) order over the 512-column chunks of the K slice; the warp sum is
  the sequence ``((C_0 + C_1) + C_2) + ..`` through ``red_s``; the slice sum ``((C^{ks=0} + C^{ks=1}) + ..)`` in cluster-rank order through
  ``mail_s``.  ``U[t][j]`` identically with the up row (the n column next to it).  ``act[t][j] = bf16(silu(G) * U)`` with ``silu(v) = v *
  sigmoid(v)``.  ``K_SPLIT`` sets the slice count (a rounding-order axis); ``N_GROUPS`` (how many n8 tiles a CTA owns) and ``X_BULK_ROWS``
  (who loads x and how) enter no sum.
* gate leaf ``p_q``, ``q = ks GROUPS + g`` ``= butterfly_5( fma chain over the lane's column pair(s) of the GATE_COLS columns ks K_SLICE + g GATE_COLS .. )``.
  Under ``GATE_CLUSTER_FOLD`` (default): slot ``s_g = ((p_{0,g} + p_{1,g}) + ..) + p_{K_SPLIT-1,g}`` on the leader in rank order,
  ``gate[t] = sigmoid( butterfly_5( sequence over the lane's GROUPS / 32 slots ) )``.  The v1 path: every leaf is a slot,
  ``gate[t] = sigmoid( butterfly_5( sequence over the lane's GATE_PARTS / 32 leaves ) )``.  Both are fixed trees per token; they differ, so
  ``GATE_CLUSTER_FOLD`` is a rounding-order axis of the gate (the ``ksplit2`` / ``ksplit4`` goldens: off; the ``ksplit8`` goldens: on).
* ``y[t][n] = bf16( gate[t] x chain_{s = 0 .. I/16 - 1} mma_k16(act[t, k16(s)], W_d[n, k16(s)]) )`` -- one warp, one chain.

Why T-independent and batch-invariant: (1) the grid, ``K_SPLIT``, the chunking, the warp -> k-step map and the n-tile map depend on the
geometry alone; (2) a token occupies a ROW of the A operand (m-tile ``t // 16``, row ``t % 16``): the mma computes row ``g`` of C from
row ``g`` of A alone, so the rows past ``T`` (zero-filled on the ring form, CLAMPED to row 0 -- never loaded, never addressed -- on the bulk
form) and the other tokens cannot touch it; (3) ROW-POSITION INVARIANCE (a premise, not a proof): the ``m16n8k16`` datapath's rounding of a
row's k16 contraction does not depend on WHICH row of the fragment the token sits in (row 0 of m-tile 0 alone at T = 1, row ``t % 16`` of
m-tile ``t // 16`` inside T = 24) -- the batch-invariance cells of the determinism suite (a token alone vs inside T 6 / 24, the permuted
rows, the duplicated token) are its tripwire; if it ever fails, the contingency is one m-tile per token at row 0 (``TILES = T``, every other
row zero; 24 mma per warp per chunk instead of 1-2) or the warp-FMA path -- NOT ``TOKEN_TILES = 2`` for every T, which keeps a token's row
position tied to its batch index; (4) no atomics, no T-dependent scheduling, every reduction a fixed tree.  Knobs ``TOKEN_TILES``,
``N_GROUPS``, ``X_BULK_ROWS``, ``ROWS_PER_CTA``, ``X_STAGES``, ``W_PHASES``, ``PDL*`` and ``MIN_CTAS_PER_SM`` move no addition; ``K_SPLIT``
changes the slice order AND the gate tree's leaf count, ``GATE_CLUSTER_FOLD`` the gate tree's shape (both pinned by the host entry's default).

BARRIER TABLE -- launch 1 (``gate_up``), 256 threads, cluster ``(K_SPLIT, 1, 1)``; ``SUM(issuing lanes) == init count`` on every path:

  mb_w[p], p < W_PHASES   tx mbarrier (TMA_LOAD class), init W_MBAR_ARRIVERS = 1.  Arrive: warp 0 lane 0 (``if warp == 0: if lane == 0``,
                          the warp gate AND the lane gate, P3 / P4): ``arrive.expect_tx(W_PHASE_BYTES = W_ROWS x K_SLICE / W_PHASES x 2)``;
                          then, after ``bar.warp.sync`` (the inits and the expect published to the warp), warp 0 lanes r < W_ROWS (8 N:
                          the lane IS the w_s row) each issue ONE ``cp.async.bulk.shared::cluster.global`` of W_PHASE_ROW_BYTES per phase
                          (local destination: bytes stay local, P9) -- multi-lane issue, at entry BEFORE ``griddepcontrol.wait`` (weights
                          are never a predecessor's output).  Wait: all 256 threads, phase 0, right before the first ``ldmatrix`` of the
                          phase's K range (after the CTA barrier that publishes the inits).  SUM: 1 arrive = init 1; tx W_ROWS x
                          W_PHASE_ROW_BYTES = expect; a copy landing before the expect leaves tx transiently negative and the phase open
                          until the arrive (PTX permits; the GDN ``mb_norm`` pattern).  Bit-neutral in W_PHASES.
  mb_x (X_BULK_ROWS)      tx mbarrier, init X_MBAR_ARRIVERS = 1.  Arrive: warp 0 lane 0 ``arrive.expect_tx(T x X_ROW_BYTES)`` (a RUNTIME
                          operand) AFTER ``griddepcontrol.wait`` (x is the predecessor's output); then warp 0 lanes t < T (T <= X_ROWS <= 24
                          < 32: every live row has a lane) ONE bulk copy of X_ROW_BYTES of row t's K slice (``x + t XS + k_base``, 16-B
                          aligned: XS is a multiple of 8 elements, k_base of 1024) into x_s row t.  Wait: all 256 threads, phase 0, once,
                          before the first ``ldmatrix.x4`` of x_s (after the publishing CTA barrier).  SUM: 1 arrive = init 1; tx = T x
                          X_ROW_BYTES = expect (T >= 1 by the host check).
  mb_kred (K_SPLIT > 1)   tx-only mbarrier on the LEADER (rank 0), init KRED_MBAR_ARRIVERS = 1 (every CTA initialises it with the same
                          constant, P8; the peers' copies are never waited).  Arrive: leader warp 0 lane 0 ``arrive.expect_tx(KRED_BYTES =
                          (K_SPLIT - 1) x (ACC_TILES x 32 x 16 + [GATE_CLUSTER_FOLD and not KRED_SPLIT] N_GROUPS x TPAD x 4))`` after the
                          cluster wait; peers (rank >= 1): warp 0, ALL 32 lanes, ACC_TILES ``st.async ... complete_tx::bytes.v4.f32`` (16 B
                          each) onto ``mapa(mail_s slot, 0)``; under GATE_CLUSTER_FOLD without KRED_SPLIT warps 1 + n (n < N_GROUPS), lanes
                          t < TPAD, ONE ``st.async ... .f32`` (4 B) of leaf p[t] onto ``mapa(gmail_s[rank-1][n][t], 0)`` -- every ship after
                          the cluster wait (P4 step 3: no cross-CTA op before every CTA's init is published), with ``mapa(mb_kred, 0)``.
                          Wait: leader warp 0 (32 lanes) and, under GATE_CLUSTER_FOLD without KRED_SPLIT, leader warps 1 + n (32 lanes
                          each), phase 0, with the CLUSTER-scope acquire (``wait_cluster_acquire``: the peers' complete_tx releases at
                          cluster scope; several warps may wait one phase; waiting lanes are not issuing lanes).  SUM: 1 arrive = init 1;
                          tx = (K_SPLIT - 1) x (ACC_TILES x 32 x 16 + [not split] N_GROUPS x TPAD x 4) = expect.
  mb_kred_gate            (KRED_SPLIT, implies GATE_CLUSTER_FOLD and K_SPLIT > 1) tx mbarrier on the LEADER, init KRED_GATE_MBAR_ARRIVERS =
                          N_GROUPS (the same constant in every CTA, P8).  Arrive: leader warps 1 + n, lane 0 each (``if lane == 0`` inside
                          the warp's own arm), ``arrive.expect_tx(KRED_GATE_GROUP_BYTES = (K_SPLIT - 1) x TPAD x 4)`` after the cluster
                          wait; peers: warps 1 + n, lanes t < TPAD, ONE ``st.async ... .f32`` of leaf p[t] onto ``mapa(gmail_s[rank-1][n][t],
                          0)`` with ``mapa(mb_kred_gate, 0)``.  Wait: leader warps 1 + n (32 lanes each), phase 0, cluster-scope acquire.
                          SUM: N_GROUPS arrives (one lane per gate warp) = init N_GROUPS; tx = N_GROUPS x (K_SPLIT - 1) x TPAD x 4 =
                          GMAIL_SMEM_BYTES = the sum of the expects.  A leaf landing before its warp's expect leaves tx transiently negative
                          (PTX permits; the mb_w pattern).  Warp 0 no longer waits for any gate leaf, the gate warps for no accumulator ship.
  init                    warp 0 lane 0: ``mbarrier.init(mb_w[p], 1)`` for every p, ``init(mb_x, 1)`` (bulk form), ``init(mb_kred, 1)``
                          (K_SPLIT > 1), ``init(mb_kred_gate, N_GROUPS)`` (KRED_SPLIT), ``fence.mbarrier_init``; ``bar.warp.sync`` (warp 0:
                          the W-issuing lanes see the inits); then
                          ``barrier.cluster.arrive.relaxed`` by EVERY thread (P4 step 3, ``const_expr(K_SPLIT > 1)``), and
                          ``barrier.cluster.wait`` by every thread right before the first cross-CTA op (after the red_s ``bar.sync``); the
                          CTA barrier after the x issue (bulk form) / of chunk 0 (ring form) publishes the inits to the local waiters.
  x ring (ring form, P18) ``cp.async`` groups + ``bar.sync 0, 256``, no mbarrier.  EVERY thread: the prologue commits X_PROLOGUE_GROUPS =
                          X_STAGES - 1 groups (chunks 0 .. X_STAGES - 2; X_COPIES copies each, a dead row = a zero-filled copy of a CLAMPED,
                          valid source address, never a skipped copy); iteration i: ``cp.async.wait_group(X_WAIT_PENDING = X_STAGES - 2)``
                          [ledger: committed = (X_STAGES - 1) + i, pending <= X_STAGES - 2 -> chunks <= i landed], ``bar.sync 0, 256`` [RAW:
                          everyone's copies visible; WAR: the stage refilled next, (i + X_STAGES - 1) % X_STAGES = (i - 1) % X_STAGES, was
                          last read in iteration i - 1, before THIS barrier], issue chunk i + X_STAGES - 1 if < X_CHUNKS and COMMIT (an
                          empty group on the tail path: every thread commits on every path), compute chunk i; after the loop ``wait_group 0``
                          (nothing outstanding at exit, P18 rule 5).  SUM at every ``bar.sync``: 256; X_CHUNKS barriers per CTA under
                          ``range_constexpr`` (uniform control flow, P18 rule 4).
  publish (bulk form)     ``bar.sync 0, 256`` (#1): every thread after warp 0's x issue -- the three inits are published; everyone then waits
                          mb_x and mb_w[0].  SUM 256.
  red_s publish           ``bar.sync 0, 256`` (#2 bulk / #X_CHUNKS + 1 ring): every thread, after warps 1 .. 7 stored their ACC_TILES v4.
  gate region             warp 1 + n, after the red_s barrier (and the cluster wait): the x pairs of group n from x_s (bulk: row t clamped
                          to a live row; ring all-resident: stage gate_chunk; ring: global x, L2); no barrier inside (warp-uniform shuffles).
                          The leader's gate warp adds its own leaf (a register) and the shipped ones after ITS OWN mb_kred (mb_kred_gate
                          under KRED_SPLIT) wait: no cross-warp hand-off is owed.
  exit                    leader: its mb_kred waits ARE the drain (every shipped byte landed before its epilogue; the bulk copies completed
                          at the mb_w / mb_x waits, the cp.async groups at ``wait_group 0``); peers: ``st.async`` has a register source and
                          nobody addresses a peer's SMEM, so no drain is owed (P15).

BARRIER TABLE -- launch 2 (``down``), THREADS2 = 4 x ROWS_PER_CTA threads, no cluster, no mbarrier:

  group W                 every thread: WD_COPIES ``cp.async`` of 16 B (the CTA's W_d rows into wd_s), ``commit_group`` -- BEFORE the wait.
  griddepcontrol.wait     every thread, after group W, before any read of ``act`` / ``gate_part`` and before any global write (P17).
  group A                 every thread: ACT_COPIES ``cp.async`` of 16 B of the act rows (src-size 0 for rows >= T, the source row clamped to
                          0: in-bounds in the [32, I] scratch, its stale bytes ignored by the zero-fill), ``commit_group``.  Ledger: 2 groups.
  bar.sync 0, THREADS2    the ONE CTA barrier: every thread after ``cp.async.wait_group 0`` and after the gate fold wrote ``sig_s[t]``
                          (warp w lane 0: tokens w, w + WARPS2, ..; lane l reads GATE_SLOTS / 32 slots).  SUM THREADS2.  It publishes wd_s,
                          act_s AND sig_s.
  exit                    plain ``st.global`` of y; ``wait_group 0`` already taken; nothing outstanding.

SMEM TABLES (no descriptor reads any buffer: every row pitch is 128 n + 16 B, so consecutive rows shift by 4 banks and the 8 row addresses of
one ``ldmatrix`` phase (8 rows x 16 B) cover all 32 banks; ``cp.async`` / bulk writes land consecutive 16-B granules of ONE row per warp
instruction: conflict-free).  Launch 1 (I 256, K_SPLIT 8; the one-group one-tile form ; the two-group two-tile form; allocation order --
the I 256 DEFAULT is two groups at every T: 81,328 B at TILES 1 (w_s 33,024 + x_s 33,024 + red_s 7,168 + mail_s 7,168 + gmail_s 896 + 48),
113,072 B at TILES 2, two CTAs per SM; the one-group one-tile form is the I 128 default and the N_GROUPS 1 arm):
  w_s      W_ROWS = 8 N rows (row 8 n + 2 i + h: group n, pair i, h = gate / up) x pitch 2,064 B      16,512 ; 33,024 B   bulk copies (async
                                                                                                            proxy) -> ldmatrix.x2 (the mbarrier wait
                                                                                                            orders async -> generic)
  x_s      X_ROWS (16 ; 24) rows x pitch 2,064 B; row t = token t's K slice; rows >= T never written,  33,024 ; 49,536 B   bulk copies -> ldmatrix.x4
           never addressed (clamped)                                                                                       and ld.shared.b32 (gate)
  red_s    [(w - 1)][n][m][lane] x 16 B                                                                    3,584 ; 14,336 B   warps 1-7 st.shared.v4 ->
                                                                                                                           warp 0 (bar.sync #2)
  mail_s   [(rank - 1)][n][m][lane] x 16 B                                                                 3,584 ; 14,336 B   PEER st.async.v4 (DSMEM) ->
                                                                                                                           leader warp 0 (mb_kred)
  gmail_s  [(rank - 1)][n][t] x 4 B, t < TPAD                                                               448 ;  1,792 B   PEER st.async.f32 -> leader
                                                                                                                           warp 1 + n (mb_kred)
  mb_w[W_PHASES], mb_x, mb_kred   u64 slots in 16-B arrays (+ mb_kred_gate's 16 B under KRED_SPLIT)                      48 B
  total                                                                                              57,200 ; 113,072 B   4 ; 2 CTAs per SM (228 KB /
                                                                                                                           4 - 1 KB = 57,344 ; 115,712)
  Co-residence of launch 2 beside launch 1 (233,472 B per SM, 1,024 B reserve per CTA; matters only under PDL_RELEASE, off by default --
  with the implicit trigger a launch-2 CTA starts after a launch-1 CTA has exited): I 256 TILES 1 = two launch-1 CTAs (2 x 82,352) + one
  launch-2 CTA (43,392) = 208,096 B; I 256 TILES 2 = two launch-1 CTAs (2 x 114,096 = 228,192) + NONE (a launch-2 CTA needs 51,840 more:
  launch 2 waits for a launch-1 CTA to retire); I 128 TILES 1 = three launch-1 CTAs (3 x 58,224) + one launch-2 (22,912) = 197,584 B
  (four launch-1 CTAs = 232,896 B would leave 576 B: never "4 + 1").
  (``smem_table1(cfg)``'s figures, pinned by the host test ``test_default_form_figures_of_record``.  I 128 (N 1 at every T): TILES 1 57,200 B
  (256 CTAs, 4 per SM); TILES 2 81,328 B (2 per SM).  The v1 ring arm (K_SPLIT 2, I 256): w_s 65,664 + x_s 133,120 + red_s 3,584 / 7,168 + mail_s
  512 / 1,024 + 32 = 202,912 / 207,008 B, one CTA per SM, unchanged.)
Launch 2 (I 256; TPAD 16 / 32): wd_s 64 rows x pitch 528 B = 33,792 (``cp.async``, ``ldmatrix.x2``; 528 = 4 x 128 + 16); act_s TPAD x 528 =
  8,448 / 16,896 (``cp.async``, ``ldmatrix.x4``); sig_s 32 fp32 = 128; total 42,368 / 50,816 B (I 128: pitch 272 = 2 x 128 + 16; wd_s 17,408,
  act_s 4,352 / 8,704; total 21,888 / 26,240 B).

PDL SITES (P17).  Launch 1: ``launch_dependents`` at entry under PDL_RELEASE (measured a loss in every form; an A/B arm); ``griddepcontrol.wait``
(unconditional) AFTER the mbarrier inits, the W expect_tx + bulk copies and the ``w_gate`` word loads (every lane; parameters, never a
predecessor's output -- the CALLER OBLIGATION of the config docstring covers ``w_gate_up`` AND ``w_gate``) and BEFORE the x bulk copies /
the first x ``cp.async``, the gate-logit x reads and every global write.  Launch 2: ``launch_dependents`` at entry under PDL_RELEASE;
``griddepcontrol.wait`` AFTER the W_d prefetch + commit and BEFORE the act copies, the gate_part loads and every global write.  An early
release creates an obligation for the CONSUMER, not for this kernel: a grid launched after launch 2 WITHOUT the programmatic-stream-
serialization attribute (the stock add of ``y`` into the routed output) is inert to it; a consumer launched WITH the attribute may start
while launch 2 runs and must issue its own ``griddepcontrol.wait`` before it reads ``y`` (or touches anything either launch reads or
writes).  Neither kernel allocates TMEM, so the release-above-alloc deadlock class does not arise.
"""

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
    ld_shared,
    ld_shared_v4,
    mapa_u32,
    st_async_f32,
    st_async_v4_f32,
    st_global,
    st_shared_v4,
)

from .shared_expert_decode_config import (
    BUTTERFLY_STEPS,
    CHUNK_K,
    KRED_MBAR_ARRIVERS,
    MMA_K,
    MMA_M,
    PAIRS_PER_GROUP,
    ROWS_PER_GROUP,
    W_MBAR_ARRIVERS,
    X_MBAR_ARRIVERS,
)

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
    return _ldmatrix_a_row(arr, row0 + lane % 16, col0, pitch_elems, lane)


def _ldmatrix_a_row(arr, row, col0, pitch_elems, lane):
    """:func:`_ldmatrix_a` with the lane's row given (the bulk form clamps a dead row to a live one before the mainloop)."""
    ptr = arr.subview(row * pitch_elems + col0 + (lane // 16) * 8)
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
    """Ring form: every thread's X_COPIES 16-B ``cp.async`` of chunk ``chunk`` into ring stage ``stage``: copy ``q = it x THREADS + tid`` is
    row ``q // 64``, granule ``q % 64``; a row past ``T`` is a zero-filled copy of a CLAMPED (valid) source address (row 0)."""
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


def _add4(acc, tile, v):
    """``acc[4 tile .. 4 tile + 3] += v[0 .. 3]`` on the flat accumulator list (fp32 adds in index order)."""
    for j in range(4):
        acc[4 * tile + j] = acc[4 * tile + j] + v[j]


def _pick_lane_value(vals, lane):
    """``vals[lane]`` of a list of warp-uniform registers (``vals[t]`` identical across lanes): a chain of selects, no shared memory."""
    v = vals[0]
    for t in range(1, len(vals)):
        v = _select_f32(lane == t, vals[t], v)
    return v


@cute.jit
def _gate_up_epilogue(cfg: cutlass.Constexpr, acc, act: cute.Tensor, group0, lane, T):
    """Leader warp 0: ``act[t][4 (group0 + n) + i] = bf16(silu(gate_i[t]) * up_i[t])`` for ``t < T`` (lane (g, i): c0 / c1 = gate / up of
    token g, c2 / c3 of token g + 8, per (n-tile, m-tile); ``acc`` = the flat ``4 x ACC_TILES`` accumulator list, tile ``n TILES + m``)."""
    g = lane // 4
    i = lane % 4
    for n in cutlass.range_constexpr(cfg.N_GROUPS_EFF):
        col = (group0 + n) * PAIRS_PER_GROUP + i
        for m in cutlass.range_constexpr(cfg.TOKEN_TILES):
            tile = n * cfg.TOKEN_TILES + m
            tok0 = m * MMA_M + g
            tok1 = tok0 + 8
            if tok0 < T:
                act[tok0 * cfg.I + col] = (_silu(acc[4 * tile]) * acc[4 * tile + 1]).to(BF16)
            if tok1 < T:
                act[tok1 * cfg.I + col] = (_silu(acc[4 * tile + 2]) * acc[4 * tile + 3]).to(BF16)


@cute.jit
def _leader_finish(cfg: cutlass.Constexpr, acc, mail_base, mb_kred, act: cute.Tensor, group0, lane, T):
    """Rank 0's warp 0 (K_SPLIT > 1): arm ``mb_kred`` (lane 0), wait the peers' ships with the cluster-scope acquire, add their
    accumulators in rank order, write ``act``."""
    mine = []  # the leader's OWN list: this arm's values must not leak into the peer arm's reads through the shared list object
    for j in cutlass.range_constexpr(4 * cfg.ACC_TILES):
        mine.append(acc[j])
    if lane == 0:
        nvvm.mbarrier_arrive_expect_tx(mb_kred, cfg.KRED_BYTES)
    wait_cluster_acquire(mb_kred, 0)
    for r in cutlass.range_constexpr(1, cfg.K_SPLIT):
        for tile in cutlass.range_constexpr(cfg.ACC_TILES):
            v = ld_shared_v4(mail_base + (((r - 1) * cfg.ACC_TILES + tile) * 128 + lane * 4) * 4, FP32)
            _add4(mine, tile, v)
    _gate_up_epilogue(cfg, mine, act, group0, lane, T)


@cute.jit
def _peer_ship(cfg: cutlass.Constexpr, acc, mail_base, mb_kred, rank, lane):
    """Rank >= 1's warp 0: one ``st.async.v4`` per lane and (n-tile, m-tile) into the LEADER's ``mail_s`` slot, completing 16 tx bytes on the
    leader's ``mb_kred`` (both addresses ``mapa``-translated to rank 0)."""
    leader = opaque_i32_zero()  # rank 0 as a register operand (an immediate would take the asm's `n` constraint)
    peer_mbar = mapa_u32(_smem_addr(mb_kred), leader)
    for tile in cutlass.range_constexpr(cfg.ACC_TILES):
        slot = mail_base + (((rank - 1) * cfg.ACC_TILES + tile) * 128 + lane * 4) * 4
        st_async_v4_f32(mapa_u32(slot, leader), acc[4 * tile : 4 * tile + 4], peer_mbar)


@cute.jit
def _warp0_finish(cfg: cutlass.Constexpr, acc, red_base, mail_base, mb_kred, act: cute.Tensor, group0, rank, lane, T):
    """Warp 0 after the mainloop: the intra-CTA fold (warps 1 .. 7 in order from ``red_s``), then the cluster fold (K_SPLIT > 1: the leader
    collects, the peers ship) or the epilogue."""
    for w in cutlass.range_constexpr(1, cfg.WARPS):
        for tile in cutlass.range_constexpr(cfg.ACC_TILES):
            v = ld_shared_v4(red_base + (((w - 1) * cfg.ACC_TILES + tile) * 128 + lane * 4) * 4, FP32)
            _add4(acc, tile, v)
    if cutlass.const_expr(cfg.K_SPLIT > 1):
        if rank == 0:
            _leader_finish(cfg, acc, mail_base, mb_kred, act, group0, lane, T)
        else:
            _peer_ship(cfg, acc, mail_base, mb_kred, rank, lane)
    else:
        _gate_up_epilogue(cfg, acc, act, group0, lane, T)


def _gate_x_word(cfg, resident, xs_base, x, XS, k_base, gcol0, t, trow, col):
    """The bf16x2 word of token ``t``'s two gate columns ``gcol0 + col ..`` (``gcol0`` relative to the K slice): bulk form from x_s row
    ``trow`` (the row clamped to a live one); ring all-resident form from stage ``gcol0 // CHUNK_K`` (rows past T zero-filled: row ``t``);
    ring form from global ``x`` at the CLAMPED row (the rows the ring streamed sit in L2).  ``resident`` is a Python bool: one arm traces."""
    if cfg.X_BULK_ROWS:
        return ld_shared(xs_base + (trow * cfg.X_ROW_PITCH_ELEMS + gcol0 + col) * 2, INT32)
    if resident:
        return ld_shared(xs_base + (((gcol0 // CHUNK_K) * cfg.TPAD + t) * cfg.X_PITCH_ELEMS + gcol0 % CHUNK_K + col) * 2, INT32)
    return ld_global((x.iterator + (trow * XS + k_base + gcol0 + col)).toint(), INT32)


@cute.jit
def _gate_logit(
    cfg: cutlass.Constexpr,
    n: cutlass.Constexpr,
    resident: cutlass.Constexpr,
    x_s,
    x: cute.Tensor,
    XS: cutlass.Constexpr,
    k_base,
    group0,
    cols,
    lives,
    wlo,
    whi,
    gate_part: cute.Tensor,
    gmail_base,
    mb_gate,
    rank,
    lane,
    T,
):
    """Warp ``1 + n`` of CTA (c, ks), after the mainloop: leaf ``ks GROUPS + g`` (``g = group0 + n``) of the gate logit over the group's
    GATE_COLS gate columns ``k_base + g GATE_COLS ..`` for every one of the TPAD rows (a row past ``T`` reads a clamped live row -- or the
    ring's zero fill -- and its chain is never stored).  Lane ``l`` walks its column pairs in order, ``acc = fma(x_hi, w_hi, fma(x_lo, w_lo,
    acc))`` per pair (a dead pair leaves the chain untouched through the select), one 5-step butterfly per token -- all TPAD chains in ONE
    straight-line block.  Then: under GATE_CLUSTER_FOLD a peer ships lane t's leaf (4 B, ``st.async``) to the leader's gmail_s and the
    leader adds the K_SPLIT leaves in rank order after its own wait on ``mb_gate`` (``mb_kred``, or ``mb_kred_gate`` under KRED_SPLIT,
    which this warp's lane 0 arms with its group's share first; cluster acquire) and stores slot ``g``; on the launch-2-fold path lane 0
    stores the leaf as slot ``ks GROUPS + g``."""
    xs_base = _smem_addr(x_s)
    g = group0 + n
    gcol0 = g * cfg.GATE_COLS
    vals = []
    for t in cutlass.range_constexpr(cfg.TPAD):
        acc = opaque_f32_zero()
        trow = _select_i32(t < T, INT32(t), INT32(0))
        for p in cutlass.range_constexpr(cfg.GATE_PAIRS_PER_LANE):
            xlo, xhi = f16x2_to_f32(_gate_x_word(cfg, resident, xs_base, x, XS, k_base, gcol0, t, trow, cols[p]), dtype=BF16)
            nxt = _fma_f32(xhi, whi[p], _fma_f32(xlo, wlo[p], acc))
            acc = _select_f32(lives[p], nxt, acc)
        vals.append(_butterfly(acc))
    if cutlass.const_expr(cfg.GATE_CLUSTER_FOLD_EFF):
        if rank == 0:
            # the leader's OWN list: this arm's folded values must not leak into the peer arm's reads through the shared list object
            # (the same guard as _leader_finish's `mine`; the leaves `vals` stay untouched)
            folded = []
            for t in cutlass.range_constexpr(cfg.TPAD):
                folded.append(vals[t])
            if cutlass.const_expr(cfg.KRED_SPLIT_EFF):
                if lane == 0:
                    nvvm.mbarrier_arrive_expect_tx(mb_gate, cfg.KRED_GATE_GROUP_BYTES)
            wait_cluster_acquire(mb_gate, 0)
            for r in cutlass.range_constexpr(1, cfg.K_SPLIT):
                for t in cutlass.range_constexpr(cfg.TPAD):
                    folded[t] = folded[t] + ld_shared(gmail_base + (((r - 1) * cfg.N_GROUPS_EFF + n) * cfg.TPAD + t) * 4, FP32)
            if lane == 0:
                for t in cutlass.range_constexpr(cfg.TPAD):
                    if t < T:
                        gate_part[t * cfg.GATE_SLOTS + g] = folded[t]
        else:
            leader = opaque_i32_zero()
            peer_mbar = mapa_u32(_smem_addr(mb_gate), leader)
            if lane < cfg.TPAD:
                slot = gmail_base + (((rank - 1) * cfg.N_GROUPS_EFF + n) * cfg.TPAD + lane) * 4
                st_async_f32(mapa_u32(slot, leader), _pick_lane_value(vals, lane), peer_mbar)
    else:
        if lane == 0:
            q = rank * cfg.GROUPS + g
            for t in cutlass.range_constexpr(cfg.TPAD):
                if t < T:
                    gate_part[t * cfg.GATE_SLOTS + q] = vals[t]


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
    N = cfg.N_GROUPS_EFF
    BULK = cfg.X_BULK_ROWS
    STAGES = cfg.X_STAGES_EFF
    warp = tid // 32
    lane = tid % 32
    cluster = bx // S if cutlass.const_expr(S > 1) else bx
    rank = bx % S if cutlass.const_expr(S > 1) else INT32(0)
    k_base = rank * cfg.K_SLICE
    group0 = cluster * N
    if cutlass.const_expr(cfg.PDL_RELEASE):
        launch_dependent_grids()

    SMEM = cutlass.AddressSpace.smem
    w_s = cutlass.Array(BF16, cfg.W_ROWS * cfg.W_PITCH_ELEMS, space=SMEM, alignment=16)
    if cutlass.const_expr(BULK):
        x_s = cutlass.Array(BF16, cfg.X_ROWS * cfg.X_ROW_PITCH_ELEMS, space=SMEM, alignment=16)
    else:
        x_s = cutlass.Array(BF16, STAGES * cfg.TPAD * cfg.X_PITCH_ELEMS, space=SMEM, alignment=16)
    red_s = cutlass.Array(FP32, (cfg.WARPS - 1) * cfg.ACC_TILES * 128, space=SMEM, alignment=16)
    mail_s = cutlass.Array(FP32, (S - 1) * cfg.ACC_TILES * 128, space=SMEM, alignment=16) if cutlass.const_expr(S > 1) else red_s
    gmail_s = cutlass.Array(FP32, (S - 1) * N * cfg.TPAD, space=SMEM, alignment=16) if cutlass.const_expr(cfg.GATE_CLUSTER_FOLD_EFF) else red_s
    mb_w = cutlass.Array(INT64, cfg.W_PHASES, space=SMEM, alignment=16)
    mb_x = cutlass.Array(INT64, 1, space=SMEM, alignment=16) if cutlass.const_expr(BULK) else mb_w
    mb_kred = cutlass.Array(INT64, 1, space=SMEM, alignment=16) if cutlass.const_expr(S > 1) else mb_w
    mb_kred_gate = cutlass.Array(INT64, 1, space=SMEM, alignment=16) if cutlass.const_expr(cfg.KRED_SPLIT_EFF) else mb_kred

    # ---- init (ONE lane), then the W slice (parameters: never a predecessor's output) -- one issuing lane per w_s row and phase ----
    if warp == 0:
        if lane == 0:
            for p in cutlass.range_constexpr(cfg.W_PHASES):
                nvvm.mbarrier_init(mb_w.subview(p), W_MBAR_ARRIVERS)
            if cutlass.const_expr(BULK):
                nvvm.mbarrier_init(mb_x, X_MBAR_ARRIVERS)
            if cutlass.const_expr(S > 1):
                nvvm.mbarrier_init(mb_kred, KRED_MBAR_ARRIVERS)
            if cutlass.const_expr(cfg.KRED_SPLIT_EFF):
                nvvm.mbarrier_init(mb_kred_gate, cfg.KRED_GATE_MBAR_ARRIVERS)
            nvvm.fence_mbarrier_init()
            for p in cutlass.range_constexpr(cfg.W_PHASES):
                nvvm.mbarrier_arrive_expect_tx(mb_w.subview(p), cfg.W_PHASE_BYTES)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)  # the inits and the expects are published to the issuing lanes of this warp
        if lane < cfg.W_ROWS:
            # w_s row r = 8 n + 2 i + h: group n, pair i, h 0 = the gate row, 1 = the up row
            n_ = lane // ROWS_PER_GROUP
            rem = lane % ROWS_PER_GROUP
            srow = (rem % 2) * cfg.I + (group0 + n_) * PAIRS_PER_GROUP + rem // 2
            for p in cutlass.range_constexpr(cfg.W_PHASES):
                nvvm.cp_async_bulk_shared_cluster_global(
                    w_s.subview(lane * cfg.W_PITCH_ELEMS + p * cfg.W_PHASE_K),
                    _gmem_ptr(wgu, srow * cfg.H + k_base + p * cfg.W_PHASE_K),
                    mb_w.subview(p),
                    cfg.W_PHASE_ROW_BYTES,
                )
    if cutlass.const_expr(S > 1):
        # P4 step 3: every thread of every CTA arrives (relaxed; the init lane's fence orders its inits ahead); the matching wait sits
        # right before the first cross-CTA op
        nvvm.barrier_cluster_arrive_relaxed()

    # ---- the gate-logit weights (parameters, never a predecessor's output): every lane's words of every group of the CTA, BEFORE the
    # wait (N x GATE_PAIRS_PER_LANE words; the CTA's gate columns lie inside its own K slice) ----
    gcols = []
    glives = []
    wlo = []
    whi = []
    for n in cutlass.range_constexpr(N):
        gcol0 = (group0 + n) * cfg.GATE_COLS
        cols_n, lives_n, lo_n, hi_n = [], [], [], []
        for p in cutlass.range_constexpr(cfg.GATE_PAIRS_PER_LANE):
            colp = lane * 2 + 64 * p
            live = colp < cfg.GATE_COLS
            colc = _select_i32(live, colp, INT32(0))
            lo, hi = f16x2_to_f32(ld_global((wg.iterator + (k_base + gcol0 + colc)).toint(), INT32), dtype=BF16)
            cols_n.append(colc)
            lives_n.append(live)
            lo_n.append(lo)
            hi_n.append(hi)
        gcols.append(cols_n)
        glives.append(lives_n)
        wlo.append(lo_n)
        whi.append(hi_n)

    # ---- the predecessor's output (x) is read only past this point; every global write sits below it ----
    wait_on_dependent_grids()

    if cutlass.const_expr(BULK):
        # ---- x: the T live rows of the K slice, one bulk copy each (lane t = row t), on mb_x ----
        if warp == 0:
            if lane == 0:
                nvvm.mbarrier_arrive_expect_tx(mb_x, T * cfg.X_ROW_BYTES)
            if lane < T:
                nvvm.cp_async_bulk_shared_cluster_global(x_s.subview(lane * cfg.X_ROW_PITCH_ELEMS), _gmem_ptr(x, lane * XS + k_base), mb_x, cfg.X_ROW_BYTES)
        nvvm.barrier_cta_sync(0, thread_count=cfg.THREADS)  # publish: the inits reach every local waiter
        wait(mb_x, 0)
    else:
        # ---- the x ring prologue: chunks 0 .. X_STAGES - 2 (X_PROLOGUE_GROUPS groups, every thread) ----
        for c in cutlass.range_constexpr(cfg.X_PROLOGUE_GROUPS):
            _issue_x_chunk(cfg, x, x_s, c, c, k_base, tid, T, XS)
            cp_async_commit()

    # ---- mainloop: warp w runs k16 steps 4w .. 4w + 3 of every 512-column chunk, one accumulator chain per (n-tile, m-tile, lane) ----
    acc = []
    for _ in cutlass.range_constexpr(cfg.ACC_TILES):  # ACC_TILES repeats; the index is not used
        acc.extend(_zeros4())
    kw = warp * 4 * MMA_K  # the warp's first k column inside a chunk
    arow = []  # bulk form: the lane's A row per m-tile, a dead row clamped to row 0 (never loaded)
    for m in cutlass.range_constexpr(TILES):
        r = m * MMA_M + lane % 16
        arow.append(_select_i32(r < T, r, INT32(0)))
    for i in cutlass.range_constexpr(cfg.X_CHUNKS):
        if cutlass.const_expr(not BULK):
            cp_async_wait(cfg.X_WAIT_PENDING)
            nvvm.barrier_cta_sync(0, thread_count=cfg.THREADS)
            nxt = i + cfg.X_PROLOGUE_GROUPS
            if cutlass.const_expr(nxt < cfg.X_CHUNKS):
                _issue_x_chunk(cfg, x, x_s, nxt, nxt % STAGES, k_base, tid, T, XS)
            cp_async_commit()  # every thread, every path (an empty group on the tail)
        if cutlass.const_expr(i % cfg.CHUNKS_PER_W_PHASE == 0):
            wait(mb_w.subview(i // cfg.CHUNKS_PER_W_PHASE), 0)  # the W phase this chunk reads has landed (all 256 threads)
        for s in cutlass.range_constexpr(4):
            kk = kw + s * MMA_K
            b = []
            for n in cutlass.range_constexpr(N):
                b.append(_ldmatrix_b(w_s, n * ROWS_PER_GROUP, i * CHUNK_K + kk, cfg.W_PITCH_ELEMS, lane))
            for m in cutlass.range_constexpr(TILES):
                if cutlass.const_expr(BULK):
                    a = _ldmatrix_a_row(x_s, arow[m], i * CHUNK_K + kk, cfg.X_ROW_PITCH_ELEMS, lane)
                else:
                    a = _ldmatrix_a(x_s, (i % STAGES) * cfg.TPAD + m * MMA_M, kk, cfg.X_PITCH_ELEMS, lane)
                for n in cutlass.range_constexpr(N):
                    tile = n * TILES + m
                    c = _mma_bf16(a, b[n], acc[4 * tile : 4 * tile + 4])
                    acc[4 * tile] = c[0]
                    acc[4 * tile + 1] = c[1]
                    acc[4 * tile + 2] = c[2]
                    acc[4 * tile + 3] = c[3]
    if cutlass.const_expr(not BULK):
        cp_async_wait(0)

    # ---- the intra-CTA fold: warps 1 .. 7 publish, warp 0 adds in warp order, then the cluster fold / the epilogue ----
    red_base = _smem_addr(red_s)
    mail_base = _smem_addr(mail_s)
    gmail_base = _smem_addr(gmail_s)
    if warp > 0:
        for tile in cutlass.range_constexpr(cfg.ACC_TILES):
            st_shared_v4(red_base + (((warp - 1) * cfg.ACC_TILES + tile) * 128 + lane * 4) * 4, acc[4 * tile : 4 * tile + 4], FP32)
    nvvm.barrier_cta_sync(0, thread_count=cfg.THREADS)
    if cutlass.const_expr(S > 1):
        nvvm.barrier_cluster_wait()  # every thread: the peers' inits are published before any cross-CTA op below
    if warp == 0:
        _warp0_finish(cfg, acc, red_base, mail_base, mb_kred, act, group0, rank, lane, T)
    # ---- the gate logit leaves (warps 1 + n, one instantiation per group, overlapping warp 0's fold / cluster exchange) ----
    for n in cutlass.range_constexpr(N):
        if warp == 1 + n:
            _gate_logit(
                cfg,
                n,
                STAGES == cfg.X_CHUNKS,
                x_s,
                x,
                XS,
                k_base,
                group0,
                gcols[n],
                glives[n],
                wlo[n],
                whi[n],
                gate_part,
                gmail_base,
                mb_kred_gate,
                rank,
                lane,
                T,
            )


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

    # ---- the gate fold: warp w folds tokens w, w + WARPS2, ..: lane l reads its GATE_SLOTS / 32 slots in order (every token's loads
    # issued up front from a CLAMPED row -- row 0 for a token past T, in-bounds in the [32, GATE_SLOTS] scratch, its chain never stored),
    # the butterfly, then lane 0 stores sigmoid of the live tokens ----
    PP = cfg.GATE_SLOTS_PER_LANE
    parts = []
    for j in cutlass.range_constexpr(cfg.TOKENS_PER_WARP2):
        t = warp + cfg.WARPS2 * j
        tc = _select_i32(t < T, t, INT32(0))
        base = tc * cfg.GATE_SLOTS + lane * PP
        for k in cutlass.range_constexpr(PP):
            parts.append(gate_part[base + k])
    logits = []
    for j in cutlass.range_constexpr(cfg.TOKENS_PER_WARP2):
        logit = parts[j * PP]
        for k in cutlass.range_constexpr(1, PP):
            logit = logit + parts[j * PP + k]
        logits.append(_butterfly(logit))
    if lane == 0:
        for j in cutlass.range_constexpr(cfg.TOKENS_PER_WARP2):
            t = warp + cfg.WARPS2 * j
            if t < T:
                sig_s[t] = sigmoid(logits[j])

    cp_async_wait(0)
    nvvm.barrier_cta_sync(0, thread_count=cfg.THREADS2)  # the ONE CTA barrier: wd_s, act_s and sig_s are published

    # ---- the mma chain: warp w owns n-tile w over the full K = I ----
    acc = []
    for _ in cutlass.range_constexpr(TILES):  # TILES repeats; the index is not used
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
    runtime argument (rows past it are never loaded / zero-filled and never stored); the strides and the config are the compile key."""
    k1 = frost_shared_expert_gate_up(cfg, x, wgu, wg, act, gate_part, T, XS)
    if cutlass.const_expr(cfg.K_SPLIT > 1):
        k1.launch(grid=(cfg.GRID1, 1, 1), block=(cfg.THREADS, 1, 1), cluster=(cfg.K_SPLIT, 1, 1), stream=stream, use_pdl=cfg.PDL)
    else:
        k1.launch(grid=(cfg.GRID1, 1, 1), block=(cfg.THREADS, 1, 1), stream=stream, use_pdl=cfg.PDL)
    k2 = frost_shared_expert_down(cfg, act, gate_part, wd, y, T, YS)
    k2.launch(grid=(cfg.GRID2, 1, 1), block=(cfg.THREADS2, 1, 1), stream=stream, use_pdl=cfg.PDL_PAIR)
