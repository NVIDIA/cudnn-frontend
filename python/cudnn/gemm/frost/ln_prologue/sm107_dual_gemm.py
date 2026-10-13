# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""LayerNorm-prologue gated dual GEMM -- the FROST GEMM template with a LayerNorm PROLOGUE on the A stages in SMEM and a
sigmoid(gate) * value * mask EPILOGUE with a channel-major (transposed) TMA store.  Host side, parameters and the
row-block tiling: ``cudnn.gemm.frost.ln_prologue`` (this file is that package's sm_107a BODY).

Per pair row ``r`` (``bi * n_res + k`` of a ``[bn * n_res, K]`` bf16 activation) and pass ``p``::

    x[r, :]       = bf16( LayerNorm_K(z[r, :]; gamma, beta, eps) )                 fp32 two-pass statistics, in SMEM, in place
    acc[r, :]     = x[r, :] @ W[p]^T                                                 W[p] = [2 * c_out, K], rows interleaved per 128-block
    out[p][n][bi][k] = bf16( sigmoid(acc[r, gate(n)] + b_g[n]) * (acc[r, value(n)] + b_p[n]) * mask[r] )     exactly +0.0 where mask[r] == 0
    stats[r]      = (mean, rstd)                                                     pass 0 only

This file is the rendered dense-bf16 expansion of ``gemm/frost/sm100/kernel_templates/sm100_matmul.py`` for the sm_107a config
``CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma`` (a cta_group::2 pair: 256 pair rows x 256 columns per tile, 128 rows and
the full 256 accumulator columns per CTA, K in 64-wide stages, a 9-deep A/B ring, 2 TMEM accumulator stages, persistent CLC
scheduler) plus the two arms below.  KEEP IN SYNC with that template: any mainloop / scheduler / TMEM-pipeline fix applies to both
(diff a fresh rendering against this file; the generator that produced it keeps every edit as an anchored block replacement).
It is a RENDERING for ONE arch (sm_107a: ~309 KiB SMEM, 576 TMEM columns), not a registry template, so it lives beside its host
module in ``gemm/frost/ln_prologue/`` under the arch trees' ``sm<NNN>_`` naming; the sm_100a body goes beside it as
``sm100_dual_gemm.py`` (the host picks the body by arch).

WARP LAYOUT (12 warps x 32 = 384 threads per CTA; the register pool is the ENTRY count x warps = 168 x 12 = 2016 per lane slot)
  warps 0-3  : epilogue (warp 0 also allocates TMEM)                                   requests setmaxnreg.inc 232
  warp  4    : MMA driver (the pair leader issues; the follower CLC-consumes only)      requests setmaxnreg.dec 24
  warp  5    : TMA producer (both CTAs load their A rows and their half of B)          requests setmaxnreg.dec 24
  warp  6    : CLC scheduler                                                           requests setmaxnreg.dec 24
  warp  7    : unused donor                                                            requests setmaxnreg.dec 24
  warps 8-11 : LayerNorm prologue, ONE PAIR ROW PER LANE (lane = this CTA's row 0-127)  requests setmaxnreg.inc 232
  REGISTER SPLIT -- WHAT THE BINARY CARRIES.  The six setmaxnreg requests above are DROPPED by ptxas on this rendering: the PTX carries
  ``.reqntid 384`` and no ``.maxnreg`` directive (the public CuTe DSL emits none), which is the C7508 "setmaxnreg ignored" condition, so
  the SASS has NO USETMAXREG and every warp runs FLAT at the entry allocation -- REG 168 for K <= 256, REG 80 for K = 384 (ptxas lowers
  the entry count to fit a 1.7 KB stack).  The intended sum 4 x 232 + 4 x 24 + 4 x 232 = 1952 <= 2016 describes the day the DSL emits
  ``.maxnreg``; until then the roles are sized for entry x warps = 168 x 12 and the row the LN lane holds in registers between its
  statistics passes (K / 2 packed words) is what grows with K.  Measured spills of the sm_107a renderings (STL / LDL, nvdisasm of the
  trace-time cubin; the K1 test module's SASS pins hold the K = 128 / 256 counts): K1 K = 128 1 / 1, K1 K = 256 240 / 241 (664 B stack),
  K3 K = 128 1 / 1, K3 K = 256 499 / 504 (872 B), K = 384 K1 740 / 743 and K3 1375 / 1381 (1.7 KB, REG 80) -- K = 384 is NOT served by the
  triangle multiplicative update until a rendering without the register residency exists.
  THE SECOND DIRECTIVE THAT LIFTS C7508: ``.minnctapersm 1`` = the public DSL's ``launch(..., min_blocks_per_mp=1)``
  (``LnDualGemmParams.min_blocks_per_mp``, 0 by default).  With it ptxas HONOURS the six requests above -- USETMAXREG 6, epilogue + LN warps at
  232, the producers at 24: 4 x 232 + 4 x 24 + 4 x 232 = 1952 <= 2016, the DECREASE warps release 4 x (168 - 24) = 576 >= the 8 x (232 - 168)
  = 512 the INCREASE warps take, so no warp parks on the pool -- and the cubin is BYTE-IDENTICAL to a ``.maxnreg 168`` build; the
  accumulators-arm c_out 256 renderings then carry STL / LDL 0 / 0 at STACK 0 (the columns arm's K = 128 `interleaved` renderings carry 4 / 13
  of role-prologue values -- a per-rendering sign).  WITHOUT either directive the flat
  allocation is ptxas's occupancy heuristic's, which may set it at 80 = 65536 / (2 x 384 threads) -- two CTAs per SM -- and spill a few values
  to reach it when the body's demand sits near 80: the bias-OFF accumulators-arm c_out 256 rolled rendering (REG 80 / STACK 24: 5 STL / 7 LDL =
  three swizzled sD staging addresses + the two row-decode constants, reloaded per tile on the epilogue's hot path, +7-8 % on the kernel)
  while its bias-ON twin -- 32 more live bias words -- is allocated 112 with no spill.  The kernel is one CTA per SM by its SMEM footprint
  either way, so the hint states a fact; it is a per-rendering choice a consumer sets where measured (the triangle multiplicative update's
  ``heuristics.register_split_hint``), never library-wide -- the default keeps every rendering's cubin as shipped.

ROW-BLOCK TILING.  A 256-row pair tile is ``r_i`` consecutive ``bi`` x ``kb`` consecutive ``k`` (``r_i * kb == 256``, ``kb`` a power
of two, host-chosen: ``cudnn.gemm.frost.ln_prologue.row_block_geometry``).  The scheduler's tile index ``tile_m`` decodes
as ``ib = tile_m // n_kb``, ``kb_idx = tile_m % n_kb``; pair row ``r`` (0-255; this CTA holds ``128 * pair_member + 0..127``) is
``(bi, k) = (ib * r_i + (r >> lkb), kb_idx * kb + (r & (kb - 1)))``.  Rows past ``n_res`` or ``bn`` are TMA zero-filled on the load,
treated as masked (``mask = 0``, no statistics) and clipped or zero-written by the store (the output ``k`` axis is padded to
``n_pad``).  A operand: one 3-D TMA box ``(64 channels, kbc = min(kb, 128) k, 128 // kbc bi)`` per CTA -- in SMEM exactly the
128-row x 64-channel SWIZZLE_128B stage the MMA descriptor reads.  Output: a 4-D ``(k, bi, channel, pass)`` box ``(kbc, 128 // kbc,
32, 1)`` from a ``[32 channels][128 rows]`` staging buffer, unswizzled (inner run 2 * kbc <= 256 bytes).

BARRIER TABLE (one row per mbarrier; ``SUM(issuing lanes) == init`` written out; ``s`` = A/B ring slot, ``a`` = TMEM acc stage)
  ab_full[s]        TMA_LOAD   init 1   leader TMA warp, ONE elected lane: arrive.expect_tx(2 x sB_bytes) ... the two CTAs' B
                    loads (cta_group::2 -> the leader's mbar) deliver the bytes.  Waited by the leader MMA warp.  [A bytes NO
                    LONGER counted here -- they go to a_full.]
  a_full[s]  NEW    TMA_LOAD   init 1   this CTA's TMA warp, ONE elected lane: arrive.expect_tx(sA_bytes); the CTA-LOCAL A load
                    (``cp.async.bulk.tensor`` without .cta_group, so complete_tx lands on THIS CTA's mbar) delivers the bytes.
                    Waited by this CTA's 4 LN warps (all lanes).
  a_norm_full[s] NEW  LEADER_RELEASE  init 8 = num_ln_warps x cta_group:  4 leader LN warps x 1 elected lane (local arrive) +
                    4 follower LN warps x 1 elected lane (``mbarrier.arrive.release.cta.shared::cluster`` on the mapa'd leader
                    mbar -- the RELEASE form, because the arrive publishes LANE-WRITTEN SMEM (the normalised A stage) to the
                    leader-issued cta_group::2 MMA: every lane fence.proxy.async.shared::cta, then bar.warp.sync (the 32 lanes'
                    stores and warp 0's flag store are ORDERED before the one arriving lane -- a release covers the arriving
                    thread's own stores and what it has observed, and elect_sync does not say which lane wins), then the
                    release arrive; P1 / P11 / frost-tile-dsl.md S 3).  Waited by the leader MMA warp before the stage's MMAs.
                    The follower's copy is never waited (init 8 too, unconditional).
  ab_empty[s]       MMA_COMMIT init (cluster_m // cta_group) + cluster_n - 1 = 1   one elected lane of the leader MMA warp
                    commits with the pair multicast mask -> 1 arrive per CTA.  Waited by BOTH TMA warps before refilling A and
                    B (the LN warps' generic reads of the stage precede the MMA that precedes this commit -- the refill cannot
                    overtake them).
  acc_full[a]       MMA_COMMIT init 1   leader elected lane, pair multicast (ALSO committed when the tile's MMA was skipped: an
                    empty commit completes at once, so the epilogue's wait always returns).  Waited by the 4 epilogue warps.
  acc_empty[a]      THREAD     init 8 = 4 epilogue warps x 2 CTAs, one elected lane each (remote relaxed arrive on the leader).
                    Waited by the leader MMA warp.
  clc_full / clc_empty[1]   as the template; clc_empty init = 11 consumer warps per CTA x cluster_size (the 4 LN warps are
                    CLC consumers too: they decode the tile for the mask / statistics addresses and the loop end).
  tmem_dealloc      as the template (init 32).
  named barriers: TMEM_ALLOC (id 2, 160 threads: MMA warp + 4 epilogue warps), EPI_SYNC (id 1, 128: the 4 epilogue warps,
                    the TMA-store hand-over), LN_SYNC (id 3, 128: the 4 LN warps -- the
                    leader's pair-level mask vote).

SMEM TABLE (what writes it, what reads it, the per-lane stride, the swizzle and WHY)
  sA[9 x 128 x 64 bf16 = 144 KiB]   TMA (async) -> LN lanes read + write IN PLACE (generic; lane = row, 16-B chunks) -> MMA
                    descriptor.  SWIZZLE_128B: the DESCRIPTOR reads it (correctness) AND the lanes address it (per-lane row stride
                    128 B -> the 16-B chunk index is XOR (row & 7): 8 consecutive lanes cover all 32 banks, a 32-lane 16-B
                    access is 4 wavefronts = the minimum) -- both jobs, one swizzle.
  sB[9 x 128 x 64 bf16 = 144 KiB]   TMA -> MMA descriptor.  SWIZZLE_128B (descriptor-read).
  sD[2 x 32 x 128 bf16 = 16 KiB]    epilogue lanes write (lane = row r: one bf16 at [channel][r], 32 lanes = 64 contiguous bytes
                    per instruction, per-lane stride 2 B -> conflict-free as is) -> TMA store (unswizzled box: the inner run is
                    2 * kbc <= 256 B > the 128-B swizzle span).  NO swizzle: the per-lane stride is not a multiple of 128 B and the
                    unswizzled box needs the dense layout -- the two jobs agree.
  sLn[2 x K fp32 <= 2 KiB]          LN lanes write once at start (gamma | beta, bf16 -> fp32) -> LN lanes read (broadcast
                    LDS.128, every lane the same address: conflict-free).  No swizzle.
  sBias[passes x 2 c_out fp32 <= 4 KiB]  written once at start -> epilogue lanes read (broadcast).  No swizzle.
  sFlag[9 Int32]                    leader LN warp 0, its ELECTED lane (the arriving one) writes the pair-level "all rows masked"
                    flag for the tile's FIRST ring slot -> the leader MMA warp reads it after its a_norm_full wait (generic ->
                    generic through the release / acquire pair; the bar.warp.sync before the arrive orders it too).  Rewritten only
                    after the slot was refilled, i.e. after the previous MMA on it committed.
  sVote[8 Int32]                    the leader LN warps' per-warp mask ballots, double-buffered by tile parity.
  mbarriers (ab_full, ab_empty, a_full, a_norm_full x 9; acc_full, acc_empty x 2; clc x 2; tmem_dealloc), the CLC response.
  Total ~ 309 KiB of the 327 KiB cc 10.7 carveout (the rendering's ab_stages = 9 keeps its SMEM-D 16400 B budget line; the extra
  buffers here are ~6.3 KiB).  The operand rings keep the rendering's placement (first in SMEM), so the > 256 KiB tcgen05
  descriptor line is where the K2 instance already runs bitwise.

INVARIANTS THE EPILOGUE OWES (sdpa-invariants.md): a masked or padded row stores EXACTLY +0.0 through a SELECT on ``mask == 0``,
never a multiply by zero of a residue; a tile whose 256 rows are all masked skips its MMA (the leader LN warps' vote) -- its
epilogue reads the TMEM residue like any other tile and the select discards every value; with ``skip_masked_tiles=False`` the
control arm computes everything and must be BITWISE equal (the epilogue arithmetic is identical on both arms).

TILE ORDER (``LnDualGemmParams.pass_order``, compile-time; the same function, bitwise, under either value): ``"major"`` = the
rendering's -- the pass is the grid's z axis, every row block of pass 0 before pass 1; ``"interleaved"`` = the pass folded into the
scheduler's M axis (``tile_m = passes * row_block + pass``, grid z = 1, decoded by ``_decode_pass`` at every CLC consumer), so the
tiles that read the SAME z rows are adjacent in the tile list and the second pass's A load is an L2 hit -- the HBM-side saving of
the report's shared-z-load form (one z read for both passes) without resident weights.

THE K3 ARM (``LnDualGemmParams.pairing = "accumulators"``; its barrier and SMEM tables are the columns arm's above with the two
``expect_tx`` byte counts re-derived at the end of this paragraph).  The same body
with the module constants resolved differently at load time: TWO GEMMs of N = 128 each (``mma_inst = 256 x 128 x 16``, ``cta_tile =
128 x 64 x 64`` -- each CTA holds its 64-row half of BOTH B tiles), two A operands -- operand 0 (the triangle update's ``y`` rows) lands
MN-major (two ``(64 j, 1 bi, 64 c)`` SWIZZLE_128B boxes per stage = the canonical MN-major operand, LBO 8192 / SBO 1024 / k-step 2048 B) and
is LayerNormed by the LN warps in THAT layout (lane = j: one ``ld.shared.b32`` per channel + one ``prmt`` per channel pair, the normalised
value written back ``st.shared.b16`` in place; ``a0_prologue = "transpose"`` writes the row K-major instead after a 4-warp barrier and the
MMA reads it K-major), operand 1 (``z`` rows) is K1's K-major operand; the ring stage is one 64-wide k-tile of EVERY operand (A0 16 KiB +
A1 16 KiB + B0 8 KiB + B1 8 KiB = 48 KiB, ``ab_stages = 6``); GEMM g accumulates at TMEM columns ``[128 g, 128 g + 128)`` of the stage, so
the epilogue's value / gate pairing is the SAME code as K1's interleaved columns; the output is pair-major (``[rows, c_out]`` bf16, the
rendering's own ``store_swizzled(Swizzle(2, 4, 3))`` + s64b TMA store, a 3-D ``(channel, k, bi)`` box so the tail rows are clipped).  Knobs:
``affine_fold`` (operand 0's LayerNorm is the pure ``(x - mean) * rstd``; the host folded gamma / beta into W / bias), ``fuse_residual``
(``out = bf16(resid[row] + gate * value)`` in one rounding), ``prefetch_a_l2`` (the TMA warp prefetches the tile's A boxes into L2 before
waiting for a ring slot).  The arrival ledger of every mbarrier is unchanged; ``a_full`` counts ``num_a_operands x sA_bytes`` and ``ab_full``
``num_b_operands x sB_bytes x 2``.

LN PASS FORMS (``LnDualGemmParams.ln_pass_form``; performance only -- the SAME operand, bitwise for every form but ``"smem_rolled_tree"``: the
same bf16 words summed in the same order).
``"registers"`` (the default = this rendering's form): pass A reads the row's 16-B chunks into K / 2 packed words and sums, pass B squares the
deviations from the registers, pass C normalises from them and writes back -- the residency that spills from K = 256 up at the flat entry
allocation (REGISTER SPLIT above).  ``"smem"``: no residency -- passes B and C RE-READ the chunks from the unchanged stage at the same swizzled
offsets (each lane re-reads only its own row: no cross-lane hazard; K3's MN-major operand 0 re-reads its channel-pair words, whose only other reader, lane j ^ 1,
extracts its own half by ``prmt``).  ``"smem_rolled"``: the same with the per-stage chunk loops as runtime loops (the XOR and the gamma / beta
offsets computed per chunk) -- the code-size lever.  ``"smem_rolled_pf"``: the rolled form with a 2-DEEP SOFTWARE PIPELINE over every chunk
loop -- chunk (q + 1) & 7's words (pass C: its data word and its 8 gamma + 8 beta) are loaded at the top of iteration q and iteration q computes
on the words loaded one iteration earlier (loop-carried scalars; the prologue loads chunk 0, iteration 7's load re-reads chunk 0 and is
discarded), so the ``ld.shared`` latency overlaps the previous chunk's dependent FADD chain instead of preceding it; the same words into the same
additions in the same order: bitwise.  ``"smem_rolled_tree"``: the pipelined form with FOUR partial sums per statistics pass (word w of every
chunk -- pair e on operand 0 -- accumulates into partial w, combined once per pass per operand as (t0 + t1) + (t2 + t3); pass B the same on the
squared deviations) -- a 2-deep dependent chain per chunk instead of 8, at a DIFFERENT fp32 summation order: the one form NOT bitwise the
register form (the fp64 oracle's bound decides; never a silent default).  Under ``pairing = "accumulators"`` the re-read forms take ``a0_prologue = "mn_major"`` only
(typed decline on the host): the transpose form's K-major row writes overwrite channel rows other lanes still re-read; the register form reads
everything before its LN_SYNC barrier.  Barrier table and SMEM table UNCHANGED: no arrive / wait site moves, no new buffer, the re-reads are the
pass-A reads repeated.

EPILOGUE MATH (``LnDualGemmParams.epi_math``; performance only).  ``"scalar"`` (the default): one element at a time.  ``"packed"``: channel
PAIRS on the packed fp32x2 ALU -- ``add.f32x2`` for the two bias adds, ``sigmoid2`` (``mul.f32x2`` by an opaque 0.5, two ``tanh.approx``, one
``fma.rn.f32x2``), ``mul.f32x2`` for ``gate * value`` and for ``* mask``, ``add.f32x2`` for the fused residual; the mask SELECT stays per element and
AFTER the arithmetic (never a multiply by zero of a residue).  Each packed op is the IEEE fp32 op per lane in the scalar form's association.
``"packed_hoist"``: the packed form with the subtile's 16 bias ``LDS.128`` issued BEFORE its two ``tcgen05.ld`` (the SMEM latency under the TMEM
latency; the same words into the same ops: bitwise; nothing changes on a rendering without biases).  ``"packed_biasfold"``: the hoisted form with
the GATE bias staged HALVED into ``sBias`` (``b_g / 2``, one exact fp32 multiply per entry at kernel start) and the tanh argument formed by ONE
``fma.rn.f32x2`` (``g / 2 + b_g / 2``) in place of the bias ``add.f32x2`` + the halving ``mul.f32x2`` (``tile_dsl.pointwise.sigmoid2_biased_half``;
OUTPUT-BITWISE: the fp32 grid is self-similar under a power-of-two scale, so ``RN(g / 2 + b_g / 2) == RN(g + b_g) / 2`` wherever the sum is normal
and both arguments give exactly 0.5 below that) -- 16 FADD2 fewer per 32-channel subtile per lane on the renderings with biases, whose epilogue
stream is ~282 instructions per subtile with the bias path 48 of them (16 LDS.128 + 32 FADD2 = the measured +16 % bias cost at c_out = 128).

LN ONCE PER ROW BLOCK (``LnDualGemmParams.ln_once``; performance only, bitwise by construction: the SAME normalised bf16 words feed every GEMM
tile of the work item, where the shipped item recomputes them from the same input).  OFF (``""``): the work item is one GEMM tile and a 256-row
block's A rows are re-streamed and re-LayerNormed per (pass, N tile) -- 4x at K1 c_out 256, 2x at K1 c_out 128 and K3 c_out 256.  ON (``"block"``:
the item = the whole row block, every pass x every N tile; ``"passes"``: the item = (row block, one N tile) across the passes), the shipped A/B
ring is SPLIT: the **A REGION** = ``_A_SLOTS`` slots x ``_NK`` stages (region stage ``r * _NK + s``; a whole item's A operand(s), loaded by TMA and
LayerNormed ONCE per item, read by the MMA on every tile of the item, released by one commit after the item's last tile) and the **B RING** =
``_B_STAGES`` = the shipped ``ab_stages`` stages of one k-tile of B, streamed per k-step per tile exactly as before.  Grid ``(row blocks x cluster_m,
N tiles / _NTI, 1)``; every consumer role keeps its per-tile loop and consumes the CLC ring once per ITEM (``sub`` = the item's sub-tile; ``tile_n =
item_n * _NTI + (sub & (_NTI - 1))``, ``pass = sub >> log2(_NTI)``); the LN warps run once per item.  TMEM, the accumulator ring, the epilogue and
``acc_*`` are per tile as before.  The barrier rows that change (SUM(issuing lanes) == init written out; the others are the table above):
  a_full[r * _NK + s]    TMA_LOAD   init 1   this CTA's TMA warp, ONE elected lane, arrive.expect_tx(num_a_operands x sA_bytes) once per ITEM per
                    stage at the item's first sub-tile; the CTA-local A loads of stage s deliver.  Waited by the 4 LN warps (all lanes); phase =
                    the LN warps' item counter modulo _A_SLOTS (one phase per slot: a slot's _NK stages are filled and released together).
  a_norm_full[r * _NK + s]  LEADER_RELEASE  init 8 = 4 leader + 4 follower LN warps, one elected lane each, once per ITEM per stage (the same
                    fence -> warp sync -> release arrive as the per-tile form).  Waited by the leader MMA warp on the item's FIRST sub-tile only
                    (the region is immutable until a_empty[r] and the same thread observed the release); phase = its item counter modulo _A_SLOTS.
  a_empty[r]  NEW   MMA_COMMIT init (cluster_m // cta_group) + cluster_n - 1 = 1 per CTA   one elected lane of the leader MMA warp commits with
                    the pair multicast mask after the LAST k-step's MMAs of the item's LAST sub-tile -> 1 arrive on each CTA (a commit fires after
                    the MMAs drained; an empty commit on a fully masked item completes at once).  Waited by each CTA's TMA warp before refilling
                    slot r (phase pre-armed: the first _A_SLOTS fills wait nothing); drained over _A_SLOTS after the persistent loop on BOTH CTAs.
  ab_full[b] / ab_empty[b]  the B RING's rows, unchanged in producer, count and cadence (per k-step per tile); _B_STAGES deep.
  clc_full / clc_empty      unchanged counts; one response per ITEM (the scheduler issues one query per item, every consumer arrives once per item).
  sFlag[r * _NK]            the item's pair-level vote (the leader LN warp 0's elected lane writes it before its release arrive; the leader MMA warp
                    reads it after its a_norm_full wait on the item's first sub-tile and keeps tile_skip for the whole item).
  SMEM: sA = the region (num_a_operands x _A_SLOTS x _NK x 16 KiB: K1 c_out 128 64 KiB, K1 c_out 256 128 KiB, K3 c_out 256 128 KiB at ONE slot),
                    SWIZZLE_128B for the two jobs exactly as the ring's sA; sB = the ring (unchanged); a_full / a_norm_full / sFlag sized
                    _A_SLOTS x _NK; a_empty[_A_SLOTS] appended (ON only -- the OFF arm's layout is byte-for-byte the shipped one).  _A_SLOTS = 2
                    where the two-slot region plus the B ring fit in the shipped ring's bytes (K1 both widths: the next item's load + LayerNorm
                    overlap this item's tiles), else 1 (K3 c_out 256: ``cudnn.gemm.frost.ln_prologue.ln_once_a_slots``, one source for the host,
                    the validator and the tests).

Parameters come from the FROST template loader as ``FROST_TEMPLATE_PARAMS`` (a
:class:`cudnn.gemm.frost.ln_prologue.LnDualGemmParams`); a plain import gets the ``k = 128, c_out = 128, passes = 2``
geometry so the file runs standalone.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Optional

from cutlass._mlir.dialects import arith

import cutlass.experimental.primitives as nvvm
from cudnn.frost.tile_dsl.barrier import arrive_on_leader_release
from cudnn.frost.tile_dsl.pointwise import f16x2_to_f32, fadd2, fmul2, fp32_to_fp16, sigmoid, sigmoid2, sigmoid2_biased_half
from cudnn.frost.tile_dsl.tma import ld_global_v4, ld_shared_v4, st_global_v2, st_shared_v4
from cudnn.gemm.frost.ln_prologue import (
    PAIR_TILE_ROWS,
    RING_STAGES,
    LnDualGemmParams,
    ln_once_a_slots,
    ln_once_n_tiles_per_item,
    ln_once_tiles_per_item,
    validate_ln_dual_gemm_params,
)
from cudnn.gemm.frost.tile_helpers import (
    epi_subtile_spans as _epi_subtile_spans,
    l2_swizzle_tile as _l2_swizzle_tile,
    tcgen05_alloc as _tcgen05_alloc,
    tcgen05_dealloc as _tcgen05_dealloc,
    tcgen05_mma as _tcgen05_mma,
)
import cutlass.experimental.cuda.tensor_map as _tma
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.cute.runtime import make_fake_stream
from cuda.bindings import driver as _cuda
from cutlass.cute.arch import clc as cute_clc

# Tile config: CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma
mma_inst_shape_mnk = (256, 256, 16)
cta_group = 2
cgrp_tile_mnk = (256, 256, 64)
cta_tile_mnk = (128, 128, 64)
epi_tile_mn = (128, 32)
threads_per_cta = 384  # 8 template warps + the 4 LayerNorm-prologue warps
cluster_shape_mnk = (2, 1, 1)
matmul_a_batch = 1
matmul_b_batch = 0  # set below from PARAMS.passes: the pass axis is the GEMM batch of B and of the output
a_is_m_major = False
b_is_n_major = False
mma_a_major = 0
mma_b_major = 0
ab_stages = 10
b_collector_ok = True
multicast_a = False
multicast_b = False
a_mcast_slices = 1
b_mcast_slices = 1
ab_empty_full_mask = False
ab_smem_swizzle = cutlass.experimental.primitives.Tcgen05SmemSwizzle.SWIZZLE_128B
a_smem_desc_leading_byte_offset = 16
a_smem_desc_stride_byte_offset = 1024
a_smem_k_step_bytes = 32
a_smem_m_step_bytes = 16384
a_tma_group_elems = 1
b_smem_desc_leading_byte_offset = 16
b_smem_desc_stride_byte_offset = 1024
b_smem_k_step_bytes = 32
b_tma_group_elems = 1
mma_size_m = 1
mma_size_n = 1
mma_size_k = 4
ab_tma_swizzle = _tma.TensorMapSwizzle.s128b

# Dtype family: A=bf16->MMAbf16, B=bf16->MMAbf16, out=bf16 (K_BYTES=128)
ab_dtype = cutlass.BFloat16
cd_dtype = cutlass.BFloat16
epi_store_dtype = cutlass.BFloat16
mma_a_dtype = cutlass.BFloat16
mma_b_dtype = cutlass.BFloat16
mma_c_dtype = cutlass.Float32
acc_widen_to_fp32 = False
ab_tma_dtype = cutlass.BFloat16
mma_kind = nvvm.Tcgen05MMAKind.F16
epi_n = 32
epi_row_elems = 32
tile_swizzle_n = 0
swizzle_l2_budget_bytes = 41943040
num_gemms = 1
num_a_operands = 1
num_b_operands = 1
gemm_a_idx = (0,)
gemm_b_idx = (0,)
num_tmem_alloc_cols = 576
tmem_alloc_exclusive = True
acc_stages = 2  # 256 acc cols/stage
vec_bytes_epi = 32
split_k_slices = 1
frost_compile_options = "--enable-tvm-ffi --gpu-arch sm_107a"
n_tma_outputs = 1
moe_aligned_offsets = False
epi_slot_widen = 1
epi_packed_lanes = False
epi_dp22 = False
epi_stage_rows = 128
epi_chunk_elems = 32
ab_stages = 9  # SMEM-D 16400B fixed + cast LOAD 0B/stage + multi-GEMM 0B/stage
fallback_cluster_shape_mnk = None
mixed_a_pattern_pref = 1
mixed_b_pattern_pref = 1
mixed_a_pattern_fb = 1
mixed_b_pattern_fb = 1

# ---------------------------------------------------------------------------
# LN-PROLOGUE DUAL GEMM: geometry + knobs (compile-time; every use is const_expr)
# ---------------------------------------------------------------------------
PARAMS: LnDualGemmParams = globals().get("FROST_TEMPLATE_PARAMS", LnDualGemmParams())
validate_ln_dual_gemm_params(PARAMS)
_K: int = PARAMS.k  # the LayerNorm width = GEMM K
_NK: int = PARAMS.k_stages  # K stages per tile (64 wide); the LN warps hold a whole row = all of them at once
# The tcgen05 SMEM-descriptor VERSION, declared ONCE and passed at EVERY ``Tcgen05SmemDesc.build`` below -- never a literal, never the
# DSL default.  Both arms' operand rings end at 288 KiB (9 x 32 KiB; 6 x 48 KiB), so the last stages start past 256 KiB = 262144 B: a
# version-0 descriptor's 14-bit start-address field (16-B units) wraps there and the MMA reads stage 0 instead -- EXACTLY zero
# accumulators, no fault.  Version 1 carries the wide field.  (The DSL's build() defaults to 1 today; a default change would
# otherwise break this body silently.)
DESC_VERSION: int = 1
_C_OUT: int = PARAMS.c_out  # output channels per pass; N = 2 * c_out, interleaved [value block | gate block] per 256
_N_OUT: int = PARAMS.n_out  # the GEMM N per tile: 2 c_out (value | gate interleaved) in the columns arm, c_out per GEMM in the accumulators arm
_BIAS_PER_PASS: int = 2 * _C_OUT  # the interleaved [value block | gate block] bias entries per pass (BOTH arms: value + gate per output channel)
_PASSES: int = PARAMS.passes
HAS_BIAS: bool = PARAMS.has_bias
WANT_STATS: bool = PARAMS.want_stats
HAS_MASK: bool = PARAMS.has_mask
SKIP_MASKED_TILES: bool = PARAMS.skip_masked_tiles
_EPS: float = PARAMS.eps
_DIAG: str = PARAMS.diag  # diagnostic relief arms ("" in production)
_INV_K: float = 1.0 / PARAMS.k
_PASS_IN_M: bool = PARAMS.pass_order == "interleaved"  # the pass rides the scheduler's M axis (module docstring, TILE ORDER)
_LOG2_PASSES: int = _PASSES.bit_length() - 1  # the validator pins passes to a power of two when the pass is folded into M
matmul_b_batch = _PASSES
_VALUE_BLOCK: int = 128  # value columns [0, 128) and gate columns [128, 256) of every 256-wide N tile
_EPI_SUBTILES: int = _VALUE_BLOCK // epi_n  # 4 subtiles of 32 output channels per tile
# The facts the arms stand on, checked against THIS rendering's constants (the K3 arm re-resolves the tile below, from the same base).
if cta_group != 2 or cgrp_tile_mnk != (256, 256, 64) or cta_tile_mnk != (128, 128, 64) or cluster_shape_mnk != (2, 1, 1) or epi_dp22:
    raise ValueError(f"{__name__}: the LN-prologue dual GEMM is written for the cta_group::2 256x256x64 pair tile at cluster 2x1; re-render for another config")
if cd_dtype is not cutlass.BFloat16 or ab_dtype is not cutlass.BFloat16 or epi_packed_lanes or mma_size_m != 1 or num_gemms != 1 or split_k_slices != 1:
    raise ValueError(f"{__name__}: this fork was rendered for a plain dense bf16 single-GEMM, no split-K; the rendering's constants say otherwise")
if _K % cta_tile_mnk[2] != 0 or _NK > ab_stages:
    raise ValueError(
        f"{__name__}: K={_K} must be a multiple of the {cta_tile_mnk[2]}-wide stage and span at most ab_stages={ab_stages} stages (a row is normalised with every stage resident)"
    )
if epi_n != 32 or epi_stage_rows != 128 or epi_row_elems != 32:
    raise ValueError(f"{__name__}: the transposed staging buffer is laid out for 128-row x 32-column subtiles; the rendering says otherwise")
_IO_BPE: int = cd_dtype.width // 8
# ---- THE K3 ARM (pairing = "accumulators"): the module constants the rendering fixed are re-resolved from PARAMS (module docstring) ----
_DUAL: bool = PARAMS.pairing == "accumulators"
_A0_TRANSPOSE: bool = _DUAL and PARAMS.a0_prologue == "transpose"  # operand 0 normalised INTO the K-major layout (the fallback form)
_AFFINE_FOLD: bool = _DUAL and PARAMS.affine_fold  # operand 0's LayerNorm without gamma / beta (folded into W / bias on the host)
_FUSE_RESIDUAL: bool = _DUAL and PARAMS.fuse_residual
_PREFETCH_A_L2: bool = PARAMS.prefetch_a_l2
# LN PASS FORMS (module docstring): the re-read forms hold NO row in registers -- every LayerNorm pass re-reads the stage's chunks at the same
# swizzled offsets (bitwise the same operand: the same values in the same order); the rolled form makes the per-stage chunk loops runtime loops.
_LN_SMEM: bool = PARAMS.ln_pass_form != "registers"
_LN_ROLLED: bool = PARAMS.ln_pass_form == "smem_rolled"
# The PIPELINED rolled forms (module docstring): the chunk loops prefetch chunk (q + 1) & 7 one iteration ahead; "smem_rolled_tree" also splits every
# statistics pass into four partial sums (a different fp32 ORDER -- the oracle's form, never the bitwise pin's).  Both arms are const_expr-dead in
# every other rendering: the "smem_rolled" arm's predicate above is untouched.
_LN_PF: bool = PARAMS.ln_pass_form in ("smem_rolled_pf", "smem_rolled_tree")
_LN_TREE: bool = PARAMS.ln_pass_form == "smem_rolled_tree"
# EPILOGUE MATH (module docstring): the gated pairing on packed fp32x2 pairs (the same fp32 operations per element, half the ALU issue).
_EPI_PACKED: bool = PARAMS.epi_math in ("packed", "packed_hoist", "packed_biasfold")
# The BIAS-PATH forms of the packed epilogue (module docstring): both are const_expr-dead on a rendering without biases (the three packed forms
# render one cubin there).  `_EPI_BIAS_HOIST`: the subtile's bias words are loaded before its TMEM loads; `_EPI_BIASFOLD`: the gate bias is
# staged halved and folded into the tanh argument's fma (`sigmoid2_biased_half`).
_EPI_BIAS_HOIST: bool = PARAMS.epi_math in ("packed_hoist", "packed_biasfold")
_EPI_BIASFOLD: bool = PARAMS.epi_math == "packed_biasfold"
# REGISTER SPLIT (module docstring): the `.minnctapersm` hint of the launch -- 0 = no directive (the rendering as shipped: the six setmaxnreg
# requests dropped by ptxas, one flat allocation per warp), 1 = the hint under which ptxas honours them (USETMAXREG 6; the cubin byte-identical
# to a `.maxnreg 168` build).  Handed to the launch as the DSL's `min_blocks_per_mp`; at 0 the DSL emits no kernel attribute, so the default
# rendering's IR -- and its cubin -- is exactly today's.
_MIN_BLOCKS_PER_MP: int = PARAMS.min_blocks_per_mp
if _DUAL:
    mma_inst_shape_mnk = (256, 128, 16)  # two GEMMs of N = 128 per pair tile
    cgrp_tile_mnk = (256, 128, 64)
    cta_tile_mnk = (128, 64, 64)  # each CTA loads its 64-row half of BOTH B tiles
    num_gemms = 2
    num_a_operands = 2
    num_b_operands = 2
    gemm_a_idx = (0, 1)
    gemm_b_idx = (0, 1)
    ab_stages = 6  # 48 KiB per stage (A0 + A1 + B0 + B1) -- the same 288 KiB ring footprint as the single-GEMM rendering's 9 x 32 KiB
    if _K % cta_tile_mnk[2] != 0 or _NK > ab_stages:
        raise ValueError(f"{__name__}: the K3 arm keeps every K stage of a tile resident: K={_K} must be a multiple of 64 and at most {64 * ab_stages}")
# Per A operand: how it LANDS (MN-major = two 64-row boxes per stage) and how the MMA READS it (the descriptor constants of the FROST GEMM
# compiler's _smem_desc_params: K-major LBO 16 / SBO 1024 / k-step 32 B; MN-major SW128 bf16 LBO 8192 / SBO 1024 / k-step 2048 B).
_A_LANDS_MN: tuple = tuple((i == 0 and PARAMS.a0_major == "mn") for i in range(num_a_operands))
_A_MMA_MAJOR: tuple = tuple(1 if (_A_LANDS_MN[i] and not _A0_TRANSPOSE) else 0 for i in range(num_a_operands))
_A_LBO: tuple = tuple(8192 if mj else a_smem_desc_leading_byte_offset for mj in _A_MMA_MAJOR)
_A_SBO: tuple = tuple(a_smem_desc_stride_byte_offset for _ in _A_MMA_MAJOR)
_A_KSTEP: tuple = tuple(2048 if mj else a_smem_k_step_bytes for mj in _A_MMA_MAJOR)
_MN_GROUP: int = 64  # the 64-element (128-B) MN group of the SWIZZLE_128B atom; one TMA box per group
# LN ONCE PER ROW BLOCK (module docstring): the work item becomes the row block [x the N-tile group]; the A operand(s) of its 256 pair rows are
# loaded and LayerNormed ONCE into a resident region of _A_SLOTS slots x _NK stages (region stage r * _NK + s -- every LayerNorm pass addresses it
# exactly as it addressed the ring; only the slot list changes) while the B half-slab streams per k-step per tile through a ring of _B_STAGES = the
# shipped ab_stages.  Every constant is derived from PARAMS through the host module's ONE source (never a literal); the OFF arm resolves to the
# shipped sizes, so its cubin is byte-identical.
_LN_ONCE: bool = PARAMS.ln_once != ""
_N_TILES: int = PARAMS.n_tiles  # GEMM N tiles per row block (c_out / 128 in both arms)
_NTI: int = ln_once_n_tiles_per_item(PARAMS)  # N tiles per work item (every one under "block", one under "passes"; 1 when off)
_LOG2_NTI: int = _NTI.bit_length() - 1
_TPI: int = ln_once_tiles_per_item(PARAMS)  # GEMM tiles per work item (passes x _NTI; 1 when off)
_A_SLOTS: int = ln_once_a_slots(PARAMS)  # resident A region slots (2 = the next item's load + LayerNorm overlap this item's tiles; 1 at K3 c_out 256; 0 off)
_A_REGION_STAGES: int = _A_SLOTS * _NK if _LN_ONCE else ab_stages  # the A buffers' stage count: the region's, or the shipped ring's
_B_STAGES: int = ab_stages  # the B ring's depth (both arms)
if _LN_ONCE:
    if RING_STAGES[PARAMS.pairing] != ab_stages:
        raise ValueError(
            f"{__name__}: the host module's RING_STAGES[{PARAMS.pairing!r}] = {RING_STAGES[PARAMS.pairing]} but this body's ring is {ab_stages} deep -- the resident region was sized against a ring the body does not have"
        )
    if b_mcast_slices != 1 or multicast_b or b_is_n_major or matmul_a_batch != 1 or split_k_slices != 1 or fallback_cluster_shape_mnk is not None:
        raise ValueError(
            f"{__name__}: the LayerNorm-once arm is written for the rendering's live operand path (one B slice per CTA, no B multicast, K-major B, one A batch, no split-K, one cluster shape)"
        )

# Rank decomposition below uses shifts and masks instead of runtime integer
# division.  The catalog satisfies this; keep synthesized configs from silently
# taking the fast path with a non-power-of-two cluster dimension.
if any(_d <= 0 or (_d & (_d - 1)) != 0 for _d in cluster_shape_mnk[:2]):
    raise NotImplementedError(f"{__name__}: cluster M/N dimensions must be powers of two")
if fallback_cluster_shape_mnk is not None and any(_d <= 0 or (_d & (_d - 1)) != 0 for _d in fallback_cluster_shape_mnk[:2]):
    raise NotImplementedError(f"{__name__}: fallback cluster M/N dimensions must be powers of two")

# Keep the two launch alternatives as host constants and spell the preferred /
# fallback operations at each use site. This exposes constant masks and shift
# alternatives before backend canonicalization.
_preferred_cluster_m_shift = cluster_shape_mnk[0].bit_length() - 1
_preferred_cluster_n_shift = cluster_shape_mnk[1].bit_length() - 1
_fallback_cluster_m_shift = _preferred_cluster_m_shift if fallback_cluster_shape_mnk is None else fallback_cluster_shape_mnk[0].bit_length() - 1
_fallback_cluster_n_shift = _preferred_cluster_n_shift if fallback_cluster_shape_mnk is None else fallback_cluster_shape_mnk[1].bit_length() - 1
_CTA_GROUP = nvvm.CTAGroup.CTA_2 if cta_group == 2 else nvvm.CTAGroup.CTA_1
_cta_group_shift = cta_group.bit_length() - 1


# Scheduler ring depth: PARAMS.clc_stages (the rendering's value is 1; see LnDualGemmParams.clc_stages).
CLC_SCHED_STAGES = PARAMS.clc_stages

# Programmatic Dependent Launch (PDL, sm_90+).
USE_PDL = True

# Double-buffer for the TMA-store epilogue path.
EPI_SMEM_STAGES = 2

# Named barrier id for the 4-warp epilogue hand-over around the TMA store.
EPI_SYNC_BAR_ID = 1

# Named barrier id for the TMEM-alloc hand-over.
TMEM_ALLOC_BARRIER_ID = 2


@cute.jit
def _auto_swizzle_w(m, n, k, nt_n):
    """N-super-block width for the tile rasterization, resolved per launch.

    ``tile_swizzle_n > 0`` pins it. Otherwise: the walk keeps one operand slice
    resident and re-reads the other every super-block, so block along the SHORTER
    problem side. Once that side outgrows what L2 can hold onto while C streams
    through it, keeping it is no longer free -- fall back to the widest N block the
    budget does cover.
    """
    if cutlass.const_expr(tile_swizzle_n > 0):
        return tile_swizzle_n
    budget = cutlass.Int64(swizzle_l2_budget_bytes)
    row_bytes = (cutlass.Int64(ab_dtype.width) * k) // 8
    cap = cutlass.max(budget // (row_bytes * cgrp_tile_mnk[1]), cutlass.Int64(1))
    w = cutlass.min(cutlass.Int64(nt_n), cap)
    if cutlass.min(m, n) * row_bytes <= budget and m <= n:
        w = cutlass.Int64(1)
    return cutlass.Int32(w)


def _a_collector_op(g):
    if cutlass.const_expr(num_gemms == 1 or num_a_operands != 1 or mma_size_m != 1):
        return None
    if cutlass.const_expr(g == 0):
        return nvvm.Tcgen05MMACollectorOp.FILL
    if cutlass.const_expr(g == num_gemms - 1):
        return nvvm.Tcgen05MMACollectorOp.LASTUSE
    return nvvm.Tcgen05MMACollectorOp.USE


def _b_collector_op(mi):
    if cutlass.const_expr(not b_collector_ok or mma_size_m == 1):
        return None
    if cutlass.const_expr(mi == 0):
        return nvvm.Tcgen05MMACollectorOp.FILL
    if cutlass.const_expr(mi == mma_size_m - 1):
        return nvvm.Tcgen05MMACollectorOp.LASTUSE
    return nvvm.Tcgen05MMACollectorOp.USE


def _a0_chunk_words(a0w, wb, xow, chunk):
    """K3 ARM, the pipelined re-read forms: the eight b32 words of this thread's 8-channel chunk ``chunk`` of operand 0's MN-major stage -- per
    channel pair e the word holding (c, j), (c, j ^ 1) at the pre-XORed lane offset ``xow[2 e]`` and channel c + 1's word at ``xow[2 e + 1]``
    (c = 8 chunk + 2 e); the ``prmt`` of the two into row j's packed (c, c + 1) word is :func:`_a0_pair_f32`.  Plain Python: traced inline."""
    words = []
    for e in range(4):
        ca = chunk * 8 + 2 * e
        words.append(a0w.data_ptr(wb + xow[2 * e] + ca * 32).load())
        words.append(a0w.data_ptr(wb + xow[2 * e + 1] + (ca + 1) * 32).load())
    return tuple(words)


def _a0_pair_f32(wa, wc, sel):
    """The packed (c, c + 1) bf16 pair of row j out of the two channel words (``prmt`` with the thread's half selector), unpacked to two fp32."""
    w0 = nvvm.inline_ptx("prmt.b32 $0, $1, $2, $3;", write_only_types=[cutlass.Int32], read_only_args=[wa, wc, sel])
    return f16x2_to_f32(w0, dtype=cutlass.BFloat16)


@cute.jit
def _row_coords(tile_m, n_kb, lkb, r_pair):
    """Pair row ``r_pair`` (0-255) of scheduler tile ``tile_m`` -> ``(k, bi)`` (module docstring, ROW-BLOCK TILING)."""
    ib = tile_m // n_kb
    kb_idx = tile_m - ib * n_kb
    kbw = cutlass.Int32(1) << lkb
    k_idx = kb_idx * kbw + (r_pair & (kbw - 1))
    bi_idx = ib * (cutlass.Int32(PAIR_TILE_ROWS) >> lkb) + (r_pair >> lkb)
    return k_idx, bi_idx


@cute.jit
def _cta_box_origin(tile_m, n_kb, lkb, pair_member):
    """The ``(k, bi)`` origin of the TMA box covering rows ``128 * pair_member .. + 127`` of the pair tile."""
    return _row_coords(tile_m, n_kb, lkb, pair_member * 128)


@cute.jit
def _decode_pass(tile_m, tile_l):
    """The scheduler's ``(tile_m, tile_l)`` -> ``(row-block tile, pass)`` (module docstring, TILE ORDER).  ``pass_order =
    "interleaved"`` folds the pass into the M axis (``tile_m = passes * row_block + pass``: the two tiles reading one row block's z
    rows are adjacent in the tile list); ``"major"`` keeps the rendering's z-axis pass.  Called at every CLC consumer and at init."""
    row_tile = (tile_m >> _LOG2_PASSES) if cutlass.const_expr(_PASS_IN_M) else tile_m
    pass_idx = (tile_m & (_PASSES - 1)) if cutlass.const_expr(_PASS_IN_M) else tile_l
    return row_tile, pass_idx


@cute.jit
def _item_tile(item_n, sub):
    """LN ONCE (module docstring): the work item's sub-tile ``sub`` (0 .. _TPI - 1) -> ``(tile_n, pass)``: the item's N tiles are the low log2(_NTI)
    bits of ``sub``, the pass the rest (both powers of two by the validator; ``_NTI = 1`` under "passes" -> ``tile_n = item_n``, ``pass = sub``)."""
    tile_n = item_n * _NTI + (sub & (_NTI - 1))
    pass_idx = sub >> _LOG2_NTI
    return tile_n, pass_idx


@cute.jit
def _item_decode(m_idx, n_idx, gridx, gridy, swizzle_w):
    """LN ONCE: a CLC response -> ``(row block, N-tile group)`` = the shipped per-tile decode (the cluster-rank shift, the L2 tile swizzle) without
    the pass fold (the pass lives inside the item; the validator pins ``pass_order = "major"``)."""
    raw_m = m_idx >> _preferred_cluster_m_shift
    raw_n = n_idx >> _preferred_cluster_n_shift
    nt_m = gridx >> _preferred_cluster_m_shift
    nt_n = gridy >> _preferred_cluster_n_shift
    tile_m, item_n = _l2_swizzle_tile(raw_m, raw_n, nt_m, nt_n, swizzle_w, identity=tile_swizzle_n == 1)
    return tile_m, item_n


@cute.jit
def _row_mask(mask, row_valid, bi_idx, k_idx, n_res):
    """The row's mask as fp32: the bf16 ``mask[bi * n_res + k]`` for a valid row, 0 for a padded one (1 with no mask tensor).
    ONE 16-bit global load of exactly that element (``ld.global.u16`` zero-extended into a 32-bit register = the LOW half of a packed
    bf16x2 word; bf16 is the top half of an fp32, so the unpack of that low half is the value).  Never the aligned 32-bit word holding the half: for the LAST element of a mask with
    an odd number of rows that word spans 2 bytes past the ``[bn * n_res]`` tensor -- a red-zone read under compute-sanitizer and a fault
    for a mask carved at the end of a buffer -- and no aligned word INSIDE the tensor covers it, so a clamp cannot fix the word form."""
    m_val = cutlass.Float32(0.0)
    if row_valid:
        if cutlass.const_expr(HAS_MASK):
            _byte_off = (bi_idx * n_res + k_idx).to(cutlass.Int64) * cutlass.Int64(2)
            _half = nvvm.inline_ptx("ld.global.u16 $0, [$1];", write_only_types=[cutlass.Int32], read_only_args=[mask.iterator.toint() + _byte_off])
            m_val, _ = f16x2_to_f32(_half, dtype=cutlass.BFloat16)  # the low half of the zero-extended word IS the element
        else:
            m_val = cutlass.Float32(1.0)
    return m_val


@cute.kernel
def frost_ln_dual_gemm_128x256x128_cluster2x1_2ctamma(
    m: cutlass.Int64,
    n: cutlass.Int64,
    k: cutlass.Int64,
    tma_a_desc_0: cutlass.GridConstant[_tma.TensorMap],
    tma_a_desc_1: cutlass.GridConstant[_tma.TensorMap],  # K3: operand 1 (z rows); K1 passes operand 0's descriptor again (unused)
    tma_b_desc_0: cutlass.GridConstant[_tma.TensorMap],
    tma_b_desc_1: cutlass.GridConstant[_tma.TensorMap],  # K3: the gate weights; K1 passes operand 0's descriptor again (unused)
    tma_c_desc_0: cutlass.GridConstant[_tma.TensorMap],
    # ---- LN-PROLOGUE DUAL GEMM: appended, so the rendered prefix stays diffable ----
    n_res: cutlass.Int32,  # inner row index extent (k < n_res is a valid row)
    bn: cutlass.Int32,  # outer row index extent (bi < bn)
    n_pad: cutlass.Int32,  # output k extent (multiple of 8); k in [n_res, n_pad) is written as zeros
    lkb: cutlass.Int32,  # log2(k's per 256-row pair tile)
    n_kb: cutlass.Int32,  # k blocks per bi row
    ln_w: cute.Tensor,  # [K] bf16 LayerNorm weight of operand 0
    ln_b: cute.Tensor,  # [K] bf16 LayerNorm bias of operand 0
    ln_w1: Optional[cute.Tensor],  # [K] bf16 LayerNorm weight of operand 1 (K3); None in the single-operand arm
    ln_b1: Optional[cute.Tensor],  # [K] bf16 LayerNorm bias of operand 1 (K3); None in the single-operand arm
    bias: Optional[cute.Tensor],  # [passes * 2 * c_out] bf16, interleaved like the weight rows; None iff not HAS_BIAS
    mask: Optional[cute.Tensor],  # [bn * n_res] bf16 per-row mask; None iff not HAS_MASK (every valid row counts as 1)
    stats: Optional[cute.Tensor],  # [bn * n_res * 2] fp32 (mean, rstd) of operand 0's LayerNorm; None iff not WANT_STATS
    resid: Optional[cute.Tensor],  # [bn * n_res * c_out] bf16 pair-major residual added before the rounding (K3 fuse_residual); else None
) -> None:
    tma_a_descs = [tma_a_desc_0, tma_a_desc_1][:num_a_operands]
    tma_b_descs = [tma_b_desc_0, tma_b_desc_1][:num_b_operands]
    tma_c_descs = [tma_c_desc_0]
    ln_ws = [ln_w, ln_w1][:num_a_operands]
    ln_bs = [ln_b, ln_b1][:num_a_operands]

    mma_warp_id = 4
    tma_warp_id = 5
    scheduler_warp_id = 6
    unused_warp_id = 7
    num_epilogue_warps = 4
    epi_reg_count = 232
    prod_reg_count = 24
    # The LayerNorm prologue warpgroup (warps 8-11): one pair row per lane; a whole row (K/2 packed words) stays in registers
    # between its two statistics passes and the normalise pass, hence the same INCREASE as the epilogue -- REQUESTED, not applied:
    # ptxas drops every setmaxnreg of this rendering (module docstring, REGISTER SPLIT), so the residency runs at the flat entry count
    # and spills from K = 256 up.
    ln_warp_base = 8
    num_ln_warps = 4
    ln_reg_count = 232
    LN_SYNC_BAR_ID = 3

    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    elect_one = nvvm.elect_sync()

    tidx = cute.arch.thread_idx()[0]
    bidx = cute.arch.block_idx()[0]
    bidy = cute.arch.block_idx()[1]
    bidz = cute.arch.block_idx()[2]
    gridx = cute.arch.grid_dim()[0]
    gridy = cute.arch.grid_dim()[1]

    # Mixed CGA: the launch carries a preferred (wide) cluster plus a smaller
    # fallback one, and the device picks per cluster — a CTA can only tell which
    # by reading the hardware cluster dims. Everything cluster-shaped below then
    # follows from those, so the two kinds share one body; only the multicast bit
    # patterns are loop-built and come in precomputed per shape.
    a_mcast_pattern = mixed_a_pattern_pref
    if cutlass.const_expr(cta_group == 2):
        b_mcast_pattern = mixed_b_pattern_pref
    if cutlass.const_expr(fallback_cluster_shape_mnk is None):
        cluster_m = cluster_shape_mnk[0]
        cluster_n = cluster_shape_mnk[1]
    else:
        cdim_x, cdim_y, _cdim_z = cute.arch.block_in_cluster_dim()
        cluster_m = cdim_x
        cluster_n = cdim_y
        a_mcast_pattern = cutlass.Int32(mixed_a_pattern_pref)
        if cutlass.const_expr(cta_group == 2):
            b_mcast_pattern = cutlass.Int32(mixed_b_pattern_pref)
        # Bitwise, not `or`: both operands are runtime Booleans (this is the form
        # cutlass.cute.experimental.is_preferred_cluster uses).
        if (cdim_x != cluster_shape_mnk[0]) | (cdim_y != cluster_shape_mnk[1]):
            a_mcast_pattern = cutlass.Int32(mixed_a_pattern_fb)
            if cutlass.const_expr(cta_group == 2):
                b_mcast_pattern = cutlass.Int32(mixed_b_pattern_fb)
    cluster_size = cluster_m * cluster_n * cluster_shape_mnk[2]

    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    # Every catalog cluster dimension is a power of two.  Mixed-CGA makes the
    # divisor runtime-visible, so spelling rank decomposition as div/mod would
    # otherwise lower to reciprocal-based integer division in every warp.
    m_rank = cta_rank_in_cluster & (cluster_shape_mnk[0] - 1)
    n_rank = cta_rank_in_cluster >> _preferred_cluster_m_shift
    if cutlass.const_expr(fallback_cluster_shape_mnk is not None):
        if (cluster_m != cluster_shape_mnk[0]) | (cluster_n != cluster_shape_mnk[1]):
            m_rank = cta_rank_in_cluster & (fallback_cluster_shape_mnk[0] - 1)
            n_rank = cta_rank_in_cluster >> _fallback_cluster_m_shift

    if cutlass.const_expr(cta_group == 2):
        pair_member = m_rank % cta_group
        pair_m_idx = m_rank // cta_group
        is_pair_leader = pair_member == 0
        pair_leader_rank = pair_m_idx * cta_group + n_rank * cluster_m
    else:
        pair_member = 0
        pair_m_idx = m_rank
        is_pair_leader = True
        pair_leader_rank = cta_rank_in_cluster

    is_cluster_leader_cta = cta_rank_in_cluster == 0

    if warp_idx == mma_warp_id:
        for _i in cutlass.range_constexpr(num_a_operands):
            nvvm.prefetch_tensormap(tma_a_descs[_i].get_ptr())
        for _j in cutlass.range_constexpr(num_b_operands):
            nvvm.prefetch_tensormap(tma_b_descs[_j].get_ptr())

        for _ci in cutlass.range_constexpr(n_tma_outputs):
            nvvm.prefetch_tensormap(tma_c_descs[_ci].get_ptr())

    init_raw_m = bidx >> _preferred_cluster_m_shift
    init_raw_n = bidy >> _preferred_cluster_n_shift
    init_nt_m = gridx >> _preferred_cluster_m_shift
    init_nt_n = gridy >> _preferred_cluster_n_shift
    if cutlass.const_expr(fallback_cluster_shape_mnk is not None):
        if (cluster_m != cluster_shape_mnk[0]) | (cluster_n != cluster_shape_mnk[1]):
            init_raw_m = bidx >> _fallback_cluster_m_shift
            init_raw_n = bidy >> _fallback_cluster_n_shift
            init_nt_m = gridx >> _fallback_cluster_m_shift
            init_nt_n = gridy >> _fallback_cluster_n_shift
    swizzle_w = _auto_swizzle_w(m, n, k, init_nt_n)
    init_tile_m, init_tile_n = _l2_swizzle_tile(
        init_raw_m,
        init_raw_n,
        init_nt_m,
        init_nt_n,
        swizzle_w,
        identity=tile_swizzle_n == 1,
    )
    init_tile_l = bidz
    init_tile_m, init_tile_l = _decode_pass(init_tile_m, init_tile_l)  # TILE ORDER (module docstring)

    if cutlass.const_expr(cta_group == 1):
        a_pattern = a_mcast_pattern
        if cutlass.const_expr(fallback_cluster_shape_mnk is None):
            b_pattern = (1 << cluster_m) - 1
        else:
            b_pattern = (cutlass.Int32(1) << cluster_m) - 1

        if cutlass.const_expr(multicast_a):
            tma_mcast_mask_a = cutlass.Int16(a_pattern) << m_rank
        else:
            tma_mcast_mask_a = cutlass.Int16(1) << cta_rank_in_cluster
        if cutlass.const_expr(multicast_b):
            tma_mcast_mask_b = cutlass.Int16(b_pattern) << (n_rank * cluster_m)
        else:
            tma_mcast_mask_b = cutlass.Int16(1) << cta_rank_in_cluster
    else:
        if cutlass.const_expr(multicast_a):
            tma_mcast_mask_a = cutlass.Int16(a_mcast_pattern << m_rank)
        else:
            tma_mcast_mask_a = cutlass.Int16(1 << cta_rank_in_cluster)
        if cutlass.const_expr(multicast_b):
            tma_mcast_mask_b = cutlass.Int16((b_mcast_pattern << pair_member) << (n_rank * cluster_m))
        else:
            tma_mcast_mask_b = cutlass.Int16(1 << cta_rank_in_cluster)

    _smem_sys_reserved = cutlass.Array(cutlass.Int8, 1024, space=cutlass.AddressSpace.smem, alignment=1)

    ab_full_mbar_ptr = cutlass.Array(cutlass.Int64, _B_STAGES, space=cutlass.AddressSpace.smem)
    ab_empty_mbar_ptr = cutlass.Array(cutlass.Int64, _B_STAGES, space=cutlass.AddressSpace.smem)
    acc_empty_mbar_ptr = cutlass.Array(cutlass.Int64, acc_stages, space=cutlass.AddressSpace.smem)
    acc_full_mbar_ptr = cutlass.Array(cutlass.Int64, acc_stages, space=cutlass.AddressSpace.smem)
    if cutlass.const_expr(cta_group == 2):
        tmem_dealloc_mbar_ptr = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)

    # CLC scheduler SMEM — 2-stage ring.
    _clc_response_raw = cutlass.Array(cutlass.Int128, CLC_SCHED_STAGES, space=cutlass.AddressSpace.smem, alignment=16)
    clc_response_ptr_base = cute.make_ptr(
        cutlass.Int128,
        _clc_response_raw.data_ptr(),
        mem_space=cute.AddressSpace.smem,
    )
    clc_full_mbar_ptr = cutlass.Array(cutlass.Int64, CLC_SCHED_STAGES, space=cutlass.AddressSpace.smem, alignment=8)
    clc_empty_mbar_ptr = cutlass.Array(cutlass.Int64, CLC_SCHED_STAGES, space=cutlass.AddressSpace.smem, alignment=8)
    clc_full_mbar_cute_base = cute.make_ptr(
        cutlass.Int64,
        clc_full_mbar_ptr.data_ptr(),
        mem_space=cute.AddressSpace.smem,
    )

    sA_elems = cta_tile_mnk[0] * cta_tile_mnk[2]
    sB_elems = cta_tile_mnk[1] * cta_tile_mnk[2]
    smem_a_list = [
        cutlass.Array(
            ab_dtype,
            sA_elems * _A_REGION_STAGES,  # LN ONCE: the resident A region (_A_SLOTS x _NK stages); the shipped ring when off
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        for _ in range(num_a_operands)
    ]
    smem_b_list = [
        cutlass.Array(
            ab_dtype,
            sB_elems * _B_STAGES,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        for _ in range(num_b_operands)
    ]

    # One epilogue subtile = one MMA-M block x 32 cols; the M blocks reuse it.
    # The ring slot is indexed by `tidx`, so its row count is the EPILOGUE THREAD
    # count -- which is epi_tile_mn[0] only when the MMA M block is 128.
    epi_subtile_elems = epi_stage_rows * epi_row_elems * epi_slot_widen
    smem_d_ptr = cutlass.Array(
        epi_store_dtype,
        epi_subtile_elems * EPI_SMEM_STAGES,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    # ---- LN-PROLOGUE DUAL GEMM: the two new rings and the small parameter / vote buffers (SMEM table in the module docstring) ----
    a_full_mbar_ptr = cutlass.Array(cutlass.Int64, _A_REGION_STAGES, space=cutlass.AddressSpace.smem)
    a_norm_full_mbar_ptr = cutlass.Array(cutlass.Int64, _A_REGION_STAGES, space=cutlass.AddressSpace.smem)
    smem_ln_params = cutlass.Array(cutlass.Float32, 2 * _K * num_a_operands, space=cutlass.AddressSpace.smem, alignment=16)  # per operand: gamma | beta, fp32
    smem_bias = cutlass.Array(cutlass.Float32, _PASSES * _BIAS_PER_PASS if HAS_BIAS else 4, space=cutlass.AddressSpace.smem, alignment=16)
    smem_flags = cutlass.Array(cutlass.Int32, _A_REGION_STAGES, space=cutlass.AddressSpace.smem, alignment=16)
    smem_vote = cutlass.Array(cutlass.Int32, 8, space=cutlass.AddressSpace.smem, alignment=16)
    if cutlass.const_expr(_LN_ONCE):
        # LN ONCE: the A region slots' release (the leader MMA warp's pair-multicast commit after the item's last tile; waited by each CTA's TMA
        # warp before the refill) -- declared under the knob only, so the OFF arm's SMEM layout is byte-for-byte the shipped one
        a_empty_mbar_ptr = cutlass.Array(cutlass.Int64, _A_SLOTS, space=cutlass.AddressSpace.smem)
    ln_norm_arrivers = num_ln_warps * cta_group  # 4 leader LN warps + 4 follower LN warps, one elected lane each

    acc_empty_count = num_epilogue_warps * cta_group
    if cutlass.const_expr(ab_empty_full_mask):
        if cutlass.const_expr(cta_group == 1):
            ab_empty_count = cluster_size
        else:
            ab_empty_count = cluster_size // cta_group
    else:
        if cutlass.const_expr(cta_group == 1):
            ab_empty_count = cluster_m + cluster_n - 1
        else:
            ab_empty_count = (cluster_m // cta_group) + cluster_n - 1
    num_consumer_warps_per_cta = 7 + num_ln_warps  # the LN warps consume the CLC ring too
    clc_empty_count = num_consumer_warps_per_cta * cluster_size
    if warp_idx == 0:
        if cutlass.const_expr(cta_group == 2):
            if elect_one:
                nvvm.mbarrier_init(tmem_dealloc_mbar_ptr, 32)
        if cutlass.const_expr(_LN_ONCE):
            # LN ONCE: the B ring's rows, then the A region's (a_full / a_norm_full per region stage), then the slots' release barrier
            for i in range(_B_STAGES):
                if elect_one:
                    nvvm.mbarrier_init(ab_full_mbar_ptr.subview(i), 1)
                if elect_one:
                    nvvm.mbarrier_init(ab_empty_mbar_ptr.subview(i), ab_empty_count)
            for i in range(_A_REGION_STAGES):
                if elect_one:
                    nvvm.mbarrier_init(a_full_mbar_ptr.subview(i), 1)
                if elect_one:
                    nvvm.mbarrier_init(a_norm_full_mbar_ptr.subview(i), ln_norm_arrivers)
            for i in range(_A_SLOTS):
                if elect_one:
                    nvvm.mbarrier_init(a_empty_mbar_ptr.subview(i), ab_empty_count)
        else:
            for i in range(ab_stages):
                if elect_one:
                    nvvm.mbarrier_init(ab_full_mbar_ptr.subview(i), 1)
                if elect_one:
                    nvvm.mbarrier_init(ab_empty_mbar_ptr.subview(i), ab_empty_count)
                if elect_one:
                    nvvm.mbarrier_init(a_full_mbar_ptr.subview(i), 1)
                if elect_one:
                    nvvm.mbarrier_init(a_norm_full_mbar_ptr.subview(i), ln_norm_arrivers)
        for i in range(acc_stages):
            if elect_one:
                nvvm.mbarrier_init(acc_full_mbar_ptr.subview(i), 1)
            if elect_one:
                nvvm.mbarrier_init(acc_empty_mbar_ptr.subview(i), acc_empty_count)
        for i in range(CLC_SCHED_STAGES):
            if elect_one:
                nvvm.mbarrier_init(clc_full_mbar_ptr.subview(i), 1)
            if elect_one:
                nvvm.mbarrier_init(clc_empty_mbar_ptr.subview(i), clc_empty_count)
    # ---- LN-PROLOGUE DUAL GEMM: stage gamma | beta (and the biases) as fp32 in SMEM, once, by the LN warps, BEFORE the
    # CTA-wide sync below publishes them to every warp.  One 16-byte chunk (8 bf16) per lane; the parameters are weights, not
    # a producer kernel's output, but the read still sits behind the PDL wait.
    if warp_idx >= ln_warp_base:
        if cutlass.const_expr(USE_PDL):
            nvvm.griddepcontrol("wait")
        ln_tid_stage = tidx - ln_warp_base * 32
        for _op in cutlass.range_constexpr(num_a_operands):  # operand op's gamma | beta at [2 K op, 2 K op + 2 K)
            for _rep in cutlass.range_constexpr((2 * _K // 8 + num_ln_warps * 32 - 1) // (num_ln_warps * 32)):
                _chunk = ln_tid_stage + _rep * (num_ln_warps * 32)
                if _chunk < 2 * _K // 8:
                    _is_beta = _chunk >= _K // 8
                    _src_base = cutlass.Int64(arith.select(_is_beta.ir_value(), ln_bs[_op].iterator.toint().ir_value(), ln_ws[_op].iterator.toint().ir_value()))
                    _src_chunk = cutlass.Int32(arith.select(_is_beta.ir_value(), (_chunk - _K // 8).ir_value(), _chunk.ir_value()))
                    _words = ld_global_v4(_src_base + _src_chunk.to(cutlass.Int64) * cutlass.Int64(16), cutlass.Int32)
                    _f = []
                    for _w in _words:
                        _lo, _hi = f16x2_to_f32(_w, dtype=cutlass.BFloat16)
                        _f.append(_lo)
                        _f.append(_hi)
                    st_shared_v4(smem_ln_params.data_ptr(2 * _K * _op + _chunk * 8), _f[0:4], cutlass.Float32)
                    st_shared_v4(smem_ln_params.data_ptr(2 * _K * _op + _chunk * 8 + 4), _f[4:8], cutlass.Float32)
        if cutlass.const_expr(HAS_BIAS):
            for _rep in cutlass.range_constexpr((_PASSES * _BIAS_PER_PASS // 8 + num_ln_warps * 32 - 1) // (num_ln_warps * 32)):
                _chunk = ln_tid_stage + _rep * (num_ln_warps * 32)
                if _chunk < _PASSES * _BIAS_PER_PASS // 8:
                    _words = ld_global_v4(bias.iterator.toint() + _chunk.to(cutlass.Int64) * cutlass.Int64(16), cutlass.Int32)
                    _f = []
                    for _w in _words:
                        _lo, _hi = f16x2_to_f32(_w, dtype=cutlass.BFloat16)
                        _f.append(_lo)
                        _f.append(_hi)
                    if cutlass.const_expr(_EPI_BIASFOLD):
                        # "packed_biasfold" (module docstring): the GATE entries -- chunks 16..31 of every 32-chunk [value 128 | gate 128] block
                        # (8 entries per chunk; 128 % 8 == 0, so a chunk never straddles the boundary) -- are staged HALVED: `b_g / 2` is exact in
                        # fp32 for every bf16 value (a bf16 subnormal's lowest set bit is fp32 mantissa bit 16), the value entries are scaled by 1.0.
                        _gate_half = cutlass.Float32(
                            arith.select(((_chunk % 32) >= 16).ir_value(), cutlass.Float32(0.5).ir_value(), cutlass.Float32(1.0).ir_value())
                        )
                        _f = [_x * _gate_half for _x in _f]
                    st_shared_v4(smem_bias.data_ptr(_chunk * 8), _f[0:4], cutlass.Float32)
                    st_shared_v4(smem_bias.data_ptr(_chunk * 8 + 4), _f[4:8], cutlass.Float32)
    nvvm.fence_mbarrier_init()
    if cutlass.const_expr(cta_group == 1):

        if cutlass.const_expr(cluster_shape_mnk[0] * cluster_shape_mnk[1] > 1):
            nvvm.barrier_cluster_arrive_relaxed()
            nvvm.barrier_cluster_wait()
        else:
            nvvm.barrier_cta_sync(0)
    else:
        nvvm.barrier_cluster_arrive_relaxed()

    sA_bytes = sA_elems * (ab_dtype.width // 8)
    sB_bytes = sB_elems * (ab_dtype.width // 8)
    # LN-PROLOGUE DUAL GEMM: ab_full counts the B bytes only (both CTAs' halves land on the leader's mbar); each CTA's A bytes
    # complete on its own a_full (the LN warps must see THEIR rows land, and a cta_group::2 load signals the leader alone).
    if cutlass.const_expr(cta_group == 1):
        num_tma_copy_bytes = num_b_operands * sB_bytes
    else:
        num_tma_copy_bytes = num_b_operands * sB_bytes * 2

    # One descriptor for every MMA instruction of the tile — the CTA tile spans
    # mma_size_m of them, all the same shape.
    # K3 ARM: one instruction descriptor per GEMM -- its A operand's major (operand 0 MN-major in the K3 "mn_major" form, K-major else).
    idescs = [
        cutlass.experimental.primitives.Tcgen05InstrDesc.build(
            a_dtype=mma_a_dtype,
            b_dtype=mma_b_dtype,
            c_dtype=mma_c_dtype,
            n_dim=mma_inst_shape_mnk[1],
            m_dim=mma_inst_shape_mnk[0],
            a_major=_A_MMA_MAJOR[gemm_a_idx[_g]],
            b_major=mma_b_major,
        )
        for _g in range(num_gemms)
    ]

    # Per-CTA logical tile — the cluster cancels out, so these stay compile-time
    # constants even when the cluster shape is only known at runtime.
    logical_cta_tile_m = cgrp_tile_mnk[0] // cluster_shape_mnk[0]
    logical_cta_tile_n = cgrp_tile_mnk[1] // cluster_shape_mnk[1]
    pair_n_size = logical_cta_tile_n
    # Per-CTA output rows one MMA-M block covers. A 2-CTA pair splits M, so this
    # is the per-CTA mma_inst_m — half the instruction's hardware M.
    epi_rows_per_mma_m = cta_tile_mnk[0] // mma_size_m
    # TMEM accumulator layout, per acc stage: gemm g, M block mi -> columns
    # [g*cols_per_acc_stage + mi*epi_cols_per_mma_m, +epi_cols_per_mma_m), all at
    # TMEM lane base 0. N is NOT split across instructions, so the epilogue drains
    # a whole M block as one contiguous span.
    if cutlass.const_expr(epi_dp22):
        # cluster-MMA m=128: the pair also splits N, so each CTA drains N/2.
        epi_cols_per_mma_m = pair_n_size // 2
    else:
        epi_cols_per_mma_m = pair_n_size
    cols_per_acc_stage = mma_size_m * epi_cols_per_mma_m
    acc_region_cols = num_gemms * cols_per_acc_stage
    tmem_alloc_bar_count = (num_epilogue_warps + 1) * 32

    if cutlass.const_expr(cta_group == 2):
        nvvm.barrier_cluster_wait()
        nvvm.barrier_cta_sync(0)

    pass

    vsize = epi_chunk_elems

    M = m
    N = n
    num_k_tiles = cute.ceil_div(k, cta_tile_mnk[2])
    # The tile this cluster owns spans its OWN cluster shape; both shapes walk
    # the grid as the identity map (tile == blockIdx), so they tile the problem
    # identically and every output tile is still covered exactly once.
    cgrp_tile_m_cur = logical_cta_tile_m * cluster_m
    cgrp_tile_n_cur = logical_cta_tile_n * cluster_n
    num_k_blocks = cta_tile_mnk[2] // mma_inst_shape_mnk[2]

    if warp_idx == scheduler_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)
        if cutlass.const_expr(USE_PDL):
            nvvm.griddepcontrol("wait")
        sched_iter = cutlass.Int32(0)
        clc_empty_phase = cutlass.Int32(1)
        clc_full_phase = cutlass.Int32(0)
        is_valid_sched = cutlass.Int32(1)
        while is_valid_sched != 0:
            stage = sched_iter % CLC_SCHED_STAGES
            if stage == 0 and sched_iter != 0:
                clc_empty_phase = clc_empty_phase ^ 1
                clc_full_phase = clc_full_phase ^ 1

            if is_cluster_leader_cta:
                while not nvvm.mbarrier_try_wait_parity(clc_empty_mbar_ptr.subview(stage), clc_empty_phase, time_limit=10_000_000):
                    pass

            if elect_one:
                nvvm.mbarrier_arrive_expect_tx(clc_full_mbar_ptr.subview(stage), 16)

            if is_cluster_leader_cta:
                if elect_one:
                    cute_clc.issue_clc_query(
                        clc_full_mbar_cute_base + stage,
                        clc_response_ptr_base + stage,
                        multicast=True,
                    )

            while not nvvm.mbarrier_try_wait_parity(clc_full_mbar_ptr.subview(stage), clc_full_phase, time_limit=10_000_000):
                pass

            _m_idx, _n_idx, _l_idx, vld = cute_clc.clc_response(clc_response_ptr_base + stage)
            cute.arch.fence_proxy("async.shared", space="cta")
            is_valid_sched = vld

            nvvm.bar_warp_sync(0xFFFFFFFF)
            if elect_one:
                empty_remote = nvvm.mapa(clc_empty_mbar_ptr.subview(stage), 0)
                nvvm.mbarrier_arrive(empty_remote, scope=nvvm.MemScope.CLUSTER, relaxed=True)

            sched_iter += 1

        if cutlass.const_expr(cluster_shape_mnk[0] * cluster_shape_mnk[1] > 1):
            if is_cluster_leader_cta:
                for _ in range(CLC_SCHED_STAGES):
                    stage = sched_iter % CLC_SCHED_STAGES
                    if stage == 0 and sched_iter != 0:
                        clc_empty_phase = clc_empty_phase ^ 1
                    while not nvvm.mbarrier_try_wait_parity(
                        clc_empty_mbar_ptr.subview(stage),
                        clc_empty_phase,
                        time_limit=10_000_000,
                    ):
                        pass
                    sched_iter += 1

    if warp_idx == tma_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)
        if cutlass.const_expr(USE_PDL):
            nvvm.griddepcontrol("wait")
        if cutlass.const_expr(_LN_ONCE):
            # LN ONCE PER ROW BLOCK (module docstring): the work item is the row block [x the N-tile group].  At the item's first sub-tile the A
            # operand(s) of all _NK k-stages are loaded ONCE into region slot r (after the slot's a_empty: the leader MMA warp's release of the item
            # that used it -- pre-armed, so the first _A_SLOTS fills wait nothing); per TILE of the item the B ring streams one k-tile per k-step
            # exactly as the shipped loop in the else arm; the CLC response is consumed once per ITEM.  The ONLY arrive site added to the barrier
            # table is the a_empty commit (the MMA warp's); a_full / a_norm_full keep their producers and counts at the per-item cadence.
            ab_empty_phase_bit = cutlass.Int32(1)
            ab_iter = cutlass.Int32(0)
            a_empty_phase_tma = cutlass.Int32(1)  # pre-armed (P5): parity 1 on a fresh barrier returns at once
            a_item_iter_tma = cutlass.Int32(0)
            a_slot_tma = cutlass.Int32(0)
            tile_m = init_tile_m
            item_n = init_tile_n
            sub = cutlass.Int32(0)
            item_iter = cutlass.Int32(0)
            is_valid = cutlass.Int32(1)
            clc_full_phase_tma = cutlass.Int32(0)
            while is_valid != 0:
                if sub == 0:
                    a_slot_tma = a_item_iter_tma % _A_SLOTS
                    if a_slot_tma == 0 and a_item_iter_tma != 0:
                        a_empty_phase_tma = a_empty_phase_tma ^ 1
                    while not nvvm.mbarrier_try_wait_parity(a_empty_mbar_ptr.subview(a_slot_tma), a_empty_phase_tma, time_limit=10_000_000):
                        pass
                    # this CTA's 128 rows of the pair tile as a (k, bi) box origin (module docstring, ROW-BLOCK TILING); an MN-major operand lands as
                    # two 64-row boxes -- the (k, bi) of pair rows 128 pm + 0 and 128 pm + 64 (one bi each)
                    a_coord_k, a_coord_bi = _cta_box_origin(tile_m, n_kb, lkb, pair_member)
                    a_mn_k = []
                    a_mn_bi = []
                    for _mg in cutlass.range_constexpr(cta_tile_mnk[0] // _MN_GROUP):
                        _mk, _mbi = _row_coords(tile_m, n_kb, lkb, pair_member * cta_tile_mnk[0] + _mg * _MN_GROUP)
                        a_mn_k.append(_mk)
                        a_mn_bi.append(_mbi)
                    for _kt in cutlass.range_constexpr(_NK):
                        a_reg = a_slot_tma * _NK + _kt
                        if elect_one:
                            nvvm.mbarrier_arrive_expect_tx(a_full_mbar_ptr.subview(a_reg), num_a_operands * sA_bytes)
                        for _ai in cutlass.range_constexpr(num_a_operands):
                            sA_stage = smem_a_list[_ai].subview(sA_elems * a_reg)
                            if cutlass.const_expr(_A_LANDS_MN[_ai]):
                                for _mg in cutlass.range_constexpr(cta_tile_mnk[0] // _MN_GROUP):
                                    if elect_one:
                                        nvvm.cp_async_bulk_tensor_shared_cta_global(
                                            sA_stage.subview(_mg * _MN_GROUP * cta_tile_mnk[2]),
                                            tma_a_descs[_ai].get_ptr(),
                                            (a_mn_k[_mg], a_mn_bi[_mg], _kt * cta_tile_mnk[2]),
                                            a_full_mbar_ptr.subview(a_reg),
                                        )
                            else:
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cta_global(
                                        sA_stage,
                                        tma_a_descs[_ai].get_ptr(),
                                        (_kt * cta_tile_mnk[2], a_coord_k, a_coord_bi),
                                        a_full_mbar_ptr.subview(a_reg),
                                    )
                    a_item_iter_tma += 1
                tile_n, tile_l = _item_tile(item_n, sub)
                coord_n_per_cta = tile_n * cgrp_tile_n_cur + n_rank * logical_cta_tile_n + pair_member * cta_tile_mnk[1]
                if cutlass.const_expr(matmul_b_batch == 1):
                    tile_l_b = cutlass.Int32(0)
                else:
                    tile_l_b = tile_l
                for _kt in cutlass.range_constexpr(_NK):
                    stage = ab_iter % _B_STAGES
                    if stage == 0 and ab_iter != 0:
                        ab_empty_phase_bit = ab_empty_phase_bit ^ 1
                    while not nvvm.mbarrier_try_wait_parity(ab_empty_mbar_ptr.subview(stage), ab_empty_phase_bit, time_limit=10_000_000):
                        pass
                    if is_pair_leader:
                        if elect_one:
                            nvvm.mbarrier_arrive_expect_tx(ab_full_mbar_ptr.subview(stage), num_tma_copy_bytes)
                    for _bj in cutlass.range_constexpr(num_b_operands):
                        if elect_one:
                            nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                smem_b_list[_bj].subview(sB_elems * stage),
                                tma_b_descs[_bj].get_ptr(),
                                (_kt * cta_tile_mnk[2], coord_n_per_cta, tile_l_b),
                                ab_full_mbar_ptr.subview(stage),
                                [],
                                multicast_mask=tma_mcast_mask_b,
                                group=_CTA_GROUP,
                            )
                    ab_iter += 1
                if sub == _TPI - 1:
                    # the item's last sub-tile: the per-ITEM CLC consume (the shipped per-tile block, once per item)
                    sub = cutlass.Int32(0)
                    consumer_stage = item_iter % CLC_SCHED_STAGES
                    if consumer_stage == 0 and item_iter != 0:
                        clc_full_phase_tma = clc_full_phase_tma ^ 1
                    while not nvvm.mbarrier_try_wait_parity(clc_full_mbar_ptr.subview(consumer_stage), clc_full_phase_tma, time_limit=10_000_000):
                        pass
                    m_idx, n_idx, l_idx, vld = cute_clc.clc_response(clc_response_ptr_base + consumer_stage)
                    cute.arch.fence_proxy("async.shared", space="cta")
                    is_valid = vld
                    tile_m, item_n = _item_decode(m_idx, n_idx, gridx, gridy, swizzle_w)
                    nvvm.bar_warp_sync(0xFFFFFFFF)
                    if elect_one:
                        empty_remote = nvvm.mapa(clc_empty_mbar_ptr.subview(consumer_stage), 0)
                        nvvm.mbarrier_arrive(empty_remote, scope=nvvm.MemScope.CLUSTER, relaxed=True)
                    item_iter += 1
                else:
                    sub += 1
            # P15 drains, both CTAs: the B ring's cross-CTA empty commits (the shipped tail), then the A region slots' (the a_empty commit is multicast
            # to the pair; a slot never filled returns at once on its pre-armed parity)
            tail_stage = ab_iter % _B_STAGES
            tail_phase = ab_empty_phase_bit
            if tail_stage == 0 and ab_iter != 0:
                tail_phase = tail_phase ^ 1
            for _ in range(_B_STAGES):
                while not nvvm.mbarrier_try_wait_parity(ab_empty_mbar_ptr.subview(tail_stage), tail_phase, time_limit=10_000_000):
                    pass
                tail_stage = tail_stage + 1
                if tail_stage == _B_STAGES:
                    tail_stage = cutlass.Int32(0)
                    tail_phase = tail_phase ^ 1
            a_tail_slot = a_item_iter_tma % _A_SLOTS
            a_tail_phase = a_empty_phase_tma
            if a_tail_slot == 0 and a_item_iter_tma != 0:
                a_tail_phase = a_tail_phase ^ 1
            for _ in range(_A_SLOTS):
                while not nvvm.mbarrier_try_wait_parity(a_empty_mbar_ptr.subview(a_tail_slot), a_tail_phase, time_limit=10_000_000):
                    pass
                a_tail_slot = a_tail_slot + 1
                if a_tail_slot == _A_SLOTS:
                    a_tail_slot = cutlass.Int32(0)
                    a_tail_phase = a_tail_phase ^ 1
        else:
            ab_empty_phase_bit = cutlass.Int32(1)
            ab_iter = cutlass.Int32(0)
            tile_m = init_tile_m
            tile_n = init_tile_n
            tile_l = init_tile_l
            tile_iter = cutlass.Int32(0)
            is_valid = cutlass.Int32(1)
            clc_full_phase_tma = cutlass.Int32(0)
            while is_valid != 0:
                coord_m_per_cta = tile_m * cgrp_tile_m_cur + m_rank * cta_tile_mnk[0]
                # LN-PROLOGUE DUAL GEMM: this CTA's 128 rows of the pair tile as a (k, bi) box origin (module docstring, ROW-BLOCK TILING)
                a_coord_k, a_coord_bi = _cta_box_origin(tile_m, n_kb, lkb, pair_member)
                # K3 ARM: an MN-major operand lands as two 64-row boxes -- the (k, bi) of pair rows 128 pm + 0 and 128 pm + 64 (one bi each)
                a_mn_k = []
                a_mn_bi = []
                for _mg in cutlass.range_constexpr(cta_tile_mnk[0] // _MN_GROUP):
                    _mk, _mbi = _row_coords(tile_m, n_kb, lkb, pair_member * cta_tile_mnk[0] + _mg * _MN_GROUP)
                    a_mn_k.append(_mk)
                    a_mn_bi.append(_mbi)
                if cutlass.const_expr(_PREFETCH_A_L2):
                    # the idle-TMA-warp L2 prefetch: every A box of this tile BEFORE the first ring-slot wait (the loads below then hit L2)
                    for _kt in cutlass.range_constexpr(_NK):
                        for _ai in cutlass.range_constexpr(num_a_operands):
                            if cutlass.const_expr(_A_LANDS_MN[_ai]):
                                for _mg in cutlass.range_constexpr(cta_tile_mnk[0] // _MN_GROUP):
                                    if elect_one:
                                        nvvm.cp_async_bulk_tensor_prefetch(tma_a_descs[_ai].get_ptr(), (a_mn_k[_mg], a_mn_bi[_mg], _kt * cta_tile_mnk[2]), [])
                            else:
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_prefetch(tma_a_descs[_ai].get_ptr(), (_kt * cta_tile_mnk[2], a_coord_k, a_coord_bi), [])
                if cutlass.const_expr(cta_group == 1):
                    coord_n_per_cta = tile_n * cgrp_tile_n_cur + n_rank * cta_tile_mnk[1]
                else:
                    coord_n_per_cta = tile_n * cgrp_tile_n_cur + n_rank * logical_cta_tile_n + pair_member * cta_tile_mnk[1]
                # Split-K: grid z carries batch*S
                if cutlass.const_expr(split_k_slices > 1):
                    batch_tile_l = tile_l // split_k_slices
                    split_idx = cutlass.Int64(tile_l % split_k_slices)
                    k_tiles_per_split = num_k_tiles // split_k_slices
                    k_tiles_remainder = num_k_tiles % split_k_slices
                    k_begin = split_idx * k_tiles_per_split + cutlass.min(split_idx, k_tiles_remainder)
                    k_end = (split_idx + 1) * k_tiles_per_split + cutlass.min(split_idx + 1, k_tiles_remainder)
                else:
                    batch_tile_l = tile_l
                    k_begin = cutlass.Int64(0)
                    k_end = num_k_tiles
                if cutlass.const_expr(matmul_a_batch == 1):
                    tile_l_a = cutlass.Int32(0)
                else:
                    tile_l_a = batch_tile_l
                if cutlass.const_expr(matmul_b_batch == 1):
                    tile_l_b = cutlass.Int32(0)
                else:
                    tile_l_b = batch_tile_l

                for k_tile_idx in range(k_begin, k_end):
                    stage = ab_iter % ab_stages
                    if stage == 0 and ab_iter != 0:
                        ab_empty_phase_bit = ab_empty_phase_bit ^ 1

                    while not nvvm.mbarrier_try_wait_parity(ab_empty_mbar_ptr.subview(stage), ab_empty_phase_bit, time_limit=10_000_000):
                        pass

                    coord_k = k_tile_idx * cta_tile_mnk[2]
                    if is_pair_leader:
                        if elect_one:
                            nvvm.mbarrier_arrive_expect_tx(ab_full_mbar_ptr.subview(stage), num_tma_copy_bytes)
                    # LN-PROLOGUE DUAL GEMM: the A stage is THIS CTA's 128 pair rows x 64 channels per operand, on the CTA-LOCAL a_full (no
                    # .cta_group: complete_tx lands here, where the LN warps wait).  K-major: ONE 3-D box (channel, k, bi) at the tile's (k, bi)
                    # origin; MN-major (K3 operand 0): TWO (64 j, 1 bi, 64 c) boxes, one per 64-row group, at SMEM offsets of 8 KiB.
                    if elect_one:
                        nvvm.mbarrier_arrive_expect_tx(a_full_mbar_ptr.subview(stage), num_a_operands * sA_bytes)
                    for _ai in cutlass.range_constexpr(num_a_operands):
                        sA_stage = smem_a_list[_ai].subview(sA_elems * stage)
                        if cutlass.const_expr(_A_LANDS_MN[_ai]):
                            for _mg in cutlass.range_constexpr(cta_tile_mnk[0] // _MN_GROUP):
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cta_global(
                                        sA_stage.subview(_mg * _MN_GROUP * cta_tile_mnk[2]),
                                        tma_a_descs[_ai].get_ptr(),
                                        (a_mn_k[_mg], a_mn_bi[_mg], coord_k),
                                        a_full_mbar_ptr.subview(stage),
                                    )
                        else:
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cta_global(
                                    sA_stage,
                                    tma_a_descs[_ai].get_ptr(),
                                    (coord_k, a_coord_k, a_coord_bi),
                                    a_full_mbar_ptr.subview(stage),
                                )

                    for _bj in cutlass.range_constexpr(num_b_operands):
                        sB_stage = smem_b_list[_bj].subview(sB_elems * stage)
                        tma_b_desc = tma_b_descs[_bj]
                        if cutlass.const_expr(b_mcast_slices > 1):
                            _b_rows = cta_tile_mnk[1] // b_mcast_slices
                            if cutlass.const_expr(fallback_cluster_shape_mnk is None):
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                        sB_stage.subview(pair_m_idx * _b_rows * cta_tile_mnk[2]),
                                        tma_b_desc.get_ptr(),
                                        (coord_k, coord_n_per_cta + pair_m_idx * _b_rows, tile_l_b),
                                        ab_full_mbar_ptr.subview(stage),
                                        [],
                                        multicast_mask=tma_mcast_mask_b,
                                        group=_CTA_GROUP,
                                    )
                            else:
                                _b_per_cta = b_mcast_slices >> (_preferred_cluster_m_shift - _cta_group_shift)
                                if (cluster_m != cluster_shape_mnk[0]) | (cluster_n != cluster_shape_mnk[1]):
                                    _b_per_cta = b_mcast_slices >> (_fallback_cluster_m_shift - _cta_group_shift)
                                for _bsl in cutlass.range(_b_per_cta):
                                    _b_idx = pair_m_idx * _b_per_cta + _bsl
                                    if elect_one:
                                        nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                            sB_stage.subview(_b_idx * _b_rows * cta_tile_mnk[2]),
                                            tma_b_desc.get_ptr(),
                                            (coord_k, coord_n_per_cta + _b_idx * _b_rows, tile_l_b),
                                            ab_full_mbar_ptr.subview(stage),
                                            [],
                                            multicast_mask=tma_mcast_mask_b,
                                            group=_CTA_GROUP,
                                        )
                        elif cutlass.const_expr(multicast_b):
                            if pair_m_idx == 0:
                                if cutlass.const_expr(b_is_n_major):
                                    for n_group in cutlass.range_constexpr(cta_tile_mnk[1] // b_tma_group_elems):
                                        if elect_one:
                                            nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                                sB_stage.subview(n_group * b_tma_group_elems * cta_tile_mnk[2]),
                                                tma_b_desc.get_ptr(),
                                                (
                                                    coord_n_per_cta + n_group * b_tma_group_elems,
                                                    coord_k,
                                                    tile_l_b,
                                                ),
                                                ab_full_mbar_ptr.subview(stage),
                                                [],
                                                multicast_mask=tma_mcast_mask_b,
                                                group=_CTA_GROUP,
                                            )
                                else:
                                    if elect_one:
                                        nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                            sB_stage,
                                            tma_b_desc.get_ptr(),
                                            (coord_k, coord_n_per_cta, tile_l_b),
                                            ab_full_mbar_ptr.subview(stage),
                                            [],
                                            multicast_mask=tma_mcast_mask_b,
                                            group=_CTA_GROUP,
                                        )
                        else:
                            if cutlass.const_expr(b_is_n_major):
                                for n_group in cutlass.range_constexpr(cta_tile_mnk[1] // b_tma_group_elems):
                                    if elect_one:
                                        nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                            sB_stage.subview(n_group * b_tma_group_elems * cta_tile_mnk[2]),
                                            tma_b_desc.get_ptr(),
                                            (
                                                coord_n_per_cta + n_group * b_tma_group_elems,
                                                coord_k,
                                                tile_l_b,
                                            ),
                                            ab_full_mbar_ptr.subview(stage),
                                            [],
                                            multicast_mask=tma_mcast_mask_b,
                                            group=_CTA_GROUP,
                                        )
                            else:
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                        sB_stage,
                                        tma_b_desc.get_ptr(),
                                        (coord_k, coord_n_per_cta, tile_l_b),
                                        ab_full_mbar_ptr.subview(stage),
                                        [],
                                        multicast_mask=tma_mcast_mask_b,
                                        group=_CTA_GROUP,
                                    )

                    ab_iter += 1

                consumer_stage = tile_iter % CLC_SCHED_STAGES
                if consumer_stage == 0 and tile_iter != 0:
                    clc_full_phase_tma = clc_full_phase_tma ^ 1
                while not nvvm.mbarrier_try_wait_parity(
                    clc_full_mbar_ptr.subview(consumer_stage),
                    clc_full_phase_tma,
                    time_limit=10_000_000,
                ):
                    pass
                m_idx, n_idx, l_idx, vld = cute_clc.clc_response(clc_response_ptr_base + consumer_stage)
                cute.arch.fence_proxy("async.shared", space="cta")
                is_valid = vld
                tma_raw_m = m_idx >> _preferred_cluster_m_shift
                tma_raw_n = n_idx >> _preferred_cluster_n_shift
                tma_nt_m = gridx >> _preferred_cluster_m_shift
                tma_nt_n = gridy >> _preferred_cluster_n_shift
                if cutlass.const_expr(fallback_cluster_shape_mnk is not None):
                    if (cluster_m != cluster_shape_mnk[0]) | (cluster_n != cluster_shape_mnk[1]):
                        tma_raw_m = m_idx >> _fallback_cluster_m_shift
                        tma_raw_n = n_idx >> _fallback_cluster_n_shift
                        tma_nt_m = gridx >> _fallback_cluster_m_shift
                        tma_nt_n = gridy >> _fallback_cluster_n_shift
                tile_m, tile_n = _l2_swizzle_tile(
                    tma_raw_m,
                    tma_raw_n,
                    tma_nt_m,
                    tma_nt_n,
                    swizzle_w,
                    identity=tile_swizzle_n == 1,
                )
                tile_l = l_idx
                tile_m, tile_l = _decode_pass(tile_m, tile_l)  # TILE ORDER (module docstring)
                nvvm.bar_warp_sync(0xFFFFFFFF)
                if elect_one:
                    empty_remote = nvvm.mapa(clc_empty_mbar_ptr.subview(consumer_stage), 0)
                    nvvm.mbarrier_arrive(empty_remote, scope=nvvm.MemScope.CLUSTER, relaxed=True)
                tile_iter += 1

            tail_stage = ab_iter % ab_stages
            tail_phase = ab_empty_phase_bit
            if tail_stage == 0 and ab_iter != 0:
                tail_phase = tail_phase ^ 1
            if cutlass.const_expr(cluster_shape_mnk[0] * cluster_shape_mnk[1] > 1):
                for _ in range(ab_stages):
                    while not nvvm.mbarrier_try_wait_parity(ab_empty_mbar_ptr.subview(tail_stage), tail_phase, time_limit=10_000_000):
                        pass
                    tail_stage = tail_stage + 1
                    if tail_stage == ab_stages:
                        tail_stage = cutlass.Int32(0)
                        tail_phase = tail_phase ^ 1

    if cutlass.const_expr(fallback_cluster_shape_mnk is None):
        b_arrive_pattern = (1 << cluster_m) - 1
    else:
        b_arrive_pattern = (cutlass.Int32(1) << cluster_m) - 1
    a_part = a_mcast_pattern << m_rank
    if cutlass.const_expr(cta_group == 2):
        a_part = a_part | (a_part << 1)
    b_part = b_arrive_pattern << (n_rank * cluster_m)
    if cutlass.const_expr(ab_empty_full_mask):
        ab_empty_arrive_mask = cutlass.Int16((1 << cluster_size) - 1)
    else:
        ab_empty_arrive_mask = cutlass.Int16(a_part | b_part)
    if cutlass.const_expr(cta_group == 2):
        acc_full_mcast = cutlass.Int16(3) << pair_leader_rank
    else:
        acc_full_mcast = None
    if warp_idx == mma_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)
        _tcgen05_alloc(
            tmem_ptr_i32,
            cutlass.Int32(num_tmem_alloc_cols),
            is_exclusive=tmem_alloc_exclusive,
            group=_CTA_GROUP,
        )
        nvvm.bar_warp_sync(0xFFFFFFFF)
        nvvm.barrier_cta_arrive(barrier_id=TMEM_ALLOC_BARRIER_ID, thread_count=tmem_alloc_bar_count)
        tmem_raw_addr = tmem_ptr_i32.load()
        base_col_id_root = tmem_raw_addr & 0xFFFF
        base_row_id = tmem_raw_addr >> 16
        if cutlass.const_expr(cta_group == 2):
            peer_cta_rank = cta_rank_in_cluster ^ 1

        if is_pair_leader:
            if cutlass.const_expr(_LN_ONCE):
                # LN ONCE PER ROW BLOCK (module docstring): per TILE the accumulator stage and the B ring exactly as the shipped loop in the else arm;
                # the A operand(s) come from the item's resident region slot r -- waited (a_norm_full, the LN warps' 8 release arrives per region
                # stage) on the item's FIRST sub-tile only, the vote flag read with it and kept for the item, and RELEASED to both TMA warps by ONE
                # pair-multicast commit after the item's LAST sub-tile's MMAs (a_empty[r]: a commit fires after the MMAs drained, P13; on a fully
                # masked item no MMA was issued and the empty commit completes at once, P5); the CLC response is consumed once per ITEM.
                ab_full_phase_bit = cutlass.Int32(0)
                ab_iter = cutlass.Int32(0)
                acc_empty_phase_bit = cutlass.Int32(1)
                acc_iter = cutlass.Int32(0)
                a_item_iter_mma = cutlass.Int32(0)
                a_slot_mma = cutlass.Int32(0)
                a_norm_phase_mma = cutlass.Int32(0)
                sub = cutlass.Int32(0)
                tile_skip = cutlass.Int32(0)
                item_iter = cutlass.Int32(0)
                is_valid = cutlass.Int32(1)
                clc_full_phase_mma = cutlass.Int32(0)
                acc_stage = cutlass.Int32(0)
                desc_a_roots = [
                    cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                        start_address=smem_a_list[i],
                        leading_byte_offset=_A_LBO[i],
                        stride_byte_offset=_A_SBO[i],
                        version=DESC_VERSION,
                        layout=ab_smem_swizzle,
                    )
                    for i in range(num_a_operands)
                ]
                desc_b_roots = [
                    cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                        start_address=smem_b_list[j],
                        leading_byte_offset=b_smem_desc_leading_byte_offset,
                        stride_byte_offset=b_smem_desc_stride_byte_offset,
                        version=DESC_VERSION,
                        layout=ab_smem_swizzle,
                    )
                    for j in range(num_b_operands)
                ]
                while is_valid != 0:
                    acc_stage = acc_iter % acc_stages
                    if acc_stage == 0 and acc_iter != 0:
                        acc_empty_phase_bit = acc_empty_phase_bit ^ 1
                    while not nvvm.mbarrier_try_wait_parity(acc_empty_mbar_ptr.subview(acc_stage), acc_empty_phase_bit, time_limit=10_000_000):
                        pass
                    acc_base_col = base_col_id_root + acc_stage * acc_region_cols
                    tmem_addr_mmas = [
                        [
                            cutlass.inttoptr(
                                (base_row_id << 16) | (acc_base_col + g * cols_per_acc_stage + mi * epi_cols_per_mma_m),
                                6,
                                cutlass.Int32,
                            )
                            for mi in range(mma_size_m)
                        ]
                        for g in range(num_gemms)
                    ]
                    if sub == 0:
                        a_slot_mma = a_item_iter_mma % _A_SLOTS
                        if a_slot_mma == 0 and a_item_iter_mma != 0:
                            a_norm_phase_mma = a_norm_phase_mma ^ 1
                    scale_d = cutlass.Boolean(False)
                    for _kt in cutlass.range_constexpr(_NK):
                        stage = ab_iter % _B_STAGES
                        if stage == 0 and ab_iter != 0:
                            ab_full_phase_bit = ab_full_phase_bit ^ 1
                        while not nvvm.mbarrier_try_wait_parity(ab_full_mbar_ptr.subview(stage), ab_full_phase_bit, time_limit=10_000_000):
                            pass
                        a_reg = a_slot_mma * _NK + _kt
                        if sub == 0:
                            # the item's first sub-tile: both CTAs' A rows of this k-stage normalised in place (8 release arrives on the region stage)
                            while not nvvm.mbarrier_try_wait_parity(a_norm_full_mbar_ptr.subview(a_reg), a_norm_phase_mma, time_limit=10_000_000):
                                pass
                            if cutlass.const_expr(SKIP_MASKED_TILES and _kt == 0):
                                # the item's pair-level vote (the leader LN warps wrote the first region stage's flag before their release arrive);
                                # a fully masked item issues no MMA on any of its tiles -- every commit below still fires
                                tile_skip = smem_flags.data_ptr(a_reg).load()
                        issue_mma = elect_one & (tile_skip == 0)
                        for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                            for g in cutlass.range_constexpr(num_gemms):
                                desc_a_k = desc_a_roots[gemm_a_idx[g]].advance_start_address(sA_bytes * a_reg + _A_KSTEP[gemm_a_idx[g]] * k_block_idx)
                                desc_b = desc_b_roots[gemm_b_idx[g]].advance_start_address(sB_bytes * stage + b_smem_k_step_bytes * k_block_idx)
                                for mi in cutlass.range_constexpr(mma_size_m):
                                    desc_a = desc_a_k.advance_start_address(a_smem_m_step_bytes * mi)
                                    if issue_mma:
                                        _tcgen05_mma(
                                            mma_kind,
                                            _CTA_GROUP,
                                            tmem_addr_mmas[g][mi],
                                            desc_a,
                                            desc_b,
                                            idescs[g],
                                            scale_d,
                                            collector_op=_a_collector_op(g),
                                            b_collector_op=_b_collector_op(mi),
                                        )
                            scale_d = cutlass.Boolean(True)
                        if elect_one:
                            nvvm.tcgen05_commit(
                                ab_empty_mbar_ptr.subview(stage),
                                multicast_mask=ab_empty_arrive_mask,
                                group=_CTA_GROUP,
                            )
                        ab_iter += 1
                    if elect_one:
                        nvvm.tcgen05_commit(
                            acc_full_mbar_ptr.subview(acc_stage),
                            multicast_mask=acc_full_mcast,
                            group=_CTA_GROUP,
                        )
                    acc_iter += 1
                    if sub == _TPI - 1:
                        # the item's last sub-tile: release the region slot to BOTH TMA warps (one pair-multicast commit = 1 arrive per CTA = a_empty's init)
                        if elect_one:
                            nvvm.tcgen05_commit(
                                a_empty_mbar_ptr.subview(a_slot_mma),
                                multicast_mask=ab_empty_arrive_mask,
                                group=_CTA_GROUP,
                            )
                        a_item_iter_mma += 1
                        sub = cutlass.Int32(0)
                        consumer_stage = item_iter % CLC_SCHED_STAGES
                        if consumer_stage == 0 and item_iter != 0:
                            clc_full_phase_mma = clc_full_phase_mma ^ 1
                        while not nvvm.mbarrier_try_wait_parity(clc_full_mbar_ptr.subview(consumer_stage), clc_full_phase_mma, time_limit=10_000_000):
                            pass
                        _m_idx, _n_idx, _l_idx, vld = cute_clc.clc_response(clc_response_ptr_base + consumer_stage)
                        cute.arch.fence_proxy("async.shared", space="cta")
                        is_valid = vld
                        nvvm.bar_warp_sync(0xFFFFFFFF)
                        if elect_one:
                            empty_remote = nvvm.mapa(clc_empty_mbar_ptr.subview(consumer_stage), 0)
                            nvvm.mbarrier_arrive(empty_remote, scope=nvvm.MemScope.CLUSTER, relaxed=True)
                        item_iter += 1
                    else:
                        sub += 1
            else:
                ab_full_phase_bit = cutlass.Int32(0)
                ab_iter = cutlass.Int32(0)
                acc_empty_phase_bit = cutlass.Int32(1)
                tile_iter = cutlass.Int32(0)
                is_valid = cutlass.Int32(1)
                clc_full_phase_mma = cutlass.Int32(0)
                acc_stage = cutlass.Int32(0)
                if cutlass.const_expr(split_k_slices > 1):
                    tile_l = bidz  # this tile's z coord
                # Descriptor metadata and the SMEM allocation base are invariant
                # across persistent tiles.  Only the encoded start address advances.
                # K3 ARM: per-operand descriptor constants (an MN-major operand: LBO 8192 / SBO 1024, the FROST GEMM compiler's values)
                desc_a_roots = [
                    cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                        start_address=smem_a_list[i],
                        leading_byte_offset=_A_LBO[i],
                        stride_byte_offset=_A_SBO[i],
                        version=DESC_VERSION,
                        layout=ab_smem_swizzle,
                    )
                    for i in range(num_a_operands)
                ]
                desc_b_roots = [
                    cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                        start_address=smem_b_list[j],
                        leading_byte_offset=b_smem_desc_leading_byte_offset,
                        stride_byte_offset=b_smem_desc_stride_byte_offset,
                        version=DESC_VERSION,
                        layout=ab_smem_swizzle,
                    )
                    for j in range(num_b_operands)
                ]
                while is_valid != 0:
                    acc_stage = tile_iter % acc_stages
                    if acc_stage == 0 and tile_iter != 0:
                        acc_empty_phase_bit = acc_empty_phase_bit ^ 1

                    while not nvvm.mbarrier_try_wait_parity(
                        acc_empty_mbar_ptr.subview(acc_stage),
                        acc_empty_phase_bit,
                        time_limit=10_000_000,
                    ):
                        pass

                    acc_base_col = base_col_id_root + acc_stage * acc_region_cols
                    # One accumulator per (gemm, M block). Column arithmetic stays on
                    # the encoded (row << 16) | col integer.
                    tmem_addr_mmas = [
                        [
                            cutlass.inttoptr(
                                (base_row_id << 16) | (acc_base_col + g * cols_per_acc_stage + mi * epi_cols_per_mma_m),
                                6,
                                cutlass.Int32,
                            )
                            for mi in range(mma_size_m)
                        ]
                        for g in range(num_gemms)
                    ]

                    if cutlass.const_expr(split_k_slices > 1):
                        split_idx = cutlass.Int64(tile_l % split_k_slices)
                        k_tiles_per_split = num_k_tiles // split_k_slices
                        k_tiles_remainder = num_k_tiles % split_k_slices
                        k_begin = split_idx * k_tiles_per_split + cutlass.min(split_idx, k_tiles_remainder)
                        k_end = (split_idx + 1) * k_tiles_per_split + cutlass.min(split_idx + 1, k_tiles_remainder)
                    else:
                        k_begin = cutlass.Int64(0)
                        k_end = num_k_tiles
                    scale_d = cutlass.Boolean(False)
                    tile_skip = cutlass.Int32(0)
                    for k_tile_idx in range(k_begin, k_end):
                        stage = ab_iter % ab_stages
                        if stage == 0 and ab_iter != 0:
                            ab_full_phase_bit = ab_full_phase_bit ^ 1

                        while not nvvm.mbarrier_try_wait_parity(
                            ab_full_mbar_ptr.subview(stage),
                            ab_full_phase_bit,
                            time_limit=10_000_000,
                        ):
                            pass
                        # LN-PROLOGUE DUAL GEMM: both CTAs' A rows normalised in place (8 release arrives, same slot / phase as ab_full)
                        while not nvvm.mbarrier_try_wait_parity(
                            a_norm_full_mbar_ptr.subview(stage),
                            ab_full_phase_bit,
                            time_limit=10_000_000,
                        ):
                            pass
                        if cutlass.const_expr(SKIP_MASKED_TILES):
                            # The leader LN warps' pair-level vote for this tile sits in the FIRST stage's slot, written before their
                            # release arrive.  A fully masked tile issues no MMA; its commits still fire (empty commits complete).
                            if k_tile_idx == k_begin:
                                tile_skip = smem_flags.data_ptr(stage).load()
                        issue_mma = elect_one & (tile_skip == 0)

                        for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                            for g in cutlass.range_constexpr(num_gemms):
                                desc_a_k = desc_a_roots[gemm_a_idx[g]].advance_start_address(sA_bytes * stage + _A_KSTEP[gemm_a_idx[g]] * k_block_idx)
                                desc_b = desc_b_roots[gemm_b_idx[g]].advance_start_address(sB_bytes * stage + b_smem_k_step_bytes * k_block_idx)
                                for mi in cutlass.range_constexpr(mma_size_m):
                                    # The M sub-block offset is a whole SMEM swizzle atom
                                    # (mma_inst_m x cta_tile_k_bytes), so the descriptor's
                                    # swizzle phase is preserved. B is shared by every M block.
                                    desc_a = desc_a_k.advance_start_address(a_smem_m_step_bytes * mi)
                                    if issue_mma:
                                        _tcgen05_mma(
                                            mma_kind,
                                            _CTA_GROUP,
                                            tmem_addr_mmas[g][mi],
                                            desc_a,
                                            desc_b,
                                            idescs[g],
                                            scale_d,
                                            collector_op=_a_collector_op(g),
                                            b_collector_op=_b_collector_op(mi),
                                        )
                            # Every accumulator sees scale_d=False on exactly the first
                            # k_block of the tile, so the flip stays outside mi.
                            scale_d = cutlass.Boolean(True)

                        if elect_one:
                            nvvm.tcgen05_commit(
                                ab_empty_mbar_ptr.subview(stage),
                                multicast_mask=ab_empty_arrive_mask,
                                group=_CTA_GROUP,
                            )
                        ab_iter += 1

                    if elect_one:
                        nvvm.tcgen05_commit(
                            acc_full_mbar_ptr.subview(acc_stage),
                            multicast_mask=acc_full_mcast,
                            group=_CTA_GROUP,
                        )

                    consumer_stage = tile_iter % CLC_SCHED_STAGES
                    if consumer_stage == 0 and tile_iter != 0:
                        clc_full_phase_mma = clc_full_phase_mma ^ 1
                    while not nvvm.mbarrier_try_wait_parity(
                        clc_full_mbar_ptr.subview(consumer_stage),
                        clc_full_phase_mma,
                        time_limit=10_000_000,
                    ):
                        pass
                    _m_idx, _n_idx, _l_idx, vld = cute_clc.clc_response(clc_response_ptr_base + consumer_stage)
                    cute.arch.fence_proxy("async.shared", space="cta")
                    is_valid = vld
                    if cutlass.const_expr(split_k_slices > 1):
                        tile_l = _l_idx
                    nvvm.bar_warp_sync(0xFFFFFFFF)
                    if elect_one:
                        empty_remote = nvvm.mapa(clc_empty_mbar_ptr.subview(consumer_stage), 0)
                        nvvm.mbarrier_arrive(empty_remote, scope=nvvm.MemScope.CLUSTER, relaxed=True)
                    tile_iter += 1

            # Dense only: an early-launched reducer would pile its many small CTAs
            # onto the few idle SMs, so split-K leaves the trigger to grid exit.
            if cutlass.const_expr(USE_PDL and split_k_slices == 1):
                nvvm.griddepcontrol("launch_dependents")

            tail_stage = acc_stage
            tail_phase = acc_empty_phase_bit
            for _ in range(acc_stages):
                tail_stage = tail_stage + 1
                if tail_stage == acc_stages:
                    tail_stage = cutlass.Int32(0)
                    tail_phase = tail_phase ^ 1
                while not nvvm.mbarrier_try_wait_parity(
                    acc_empty_mbar_ptr.subview(tail_stage),
                    tail_phase,
                    time_limit=10_000_000,
                ):
                    pass
            nvvm.tcgen05_relinquish_alloc_permit(group=_CTA_GROUP)
            if cutlass.const_expr(cta_group == 2):
                peer_mbar = nvvm.mapa(tmem_dealloc_mbar_ptr, peer_cta_rank)
                while not nvvm.mbarrier_try_wait_parity(tmem_dealloc_mbar_ptr, 0, time_limit=10_000_000):
                    pass
                nvvm.mbarrier_arrive(peer_mbar, scope=nvvm.MemScope.CLUSTER, relaxed=True)
            alloc_ptr = cutlass.inttoptr(tmem_raw_addr, 6, cutlass.Int32)
            _tcgen05_dealloc(
                alloc_ptr,
                cutlass.Int32(num_tmem_alloc_cols),
                is_exclusive=tmem_alloc_exclusive,
                group=_CTA_GROUP,
            )
        else:
            if cutlass.const_expr(cta_group == 2):
                tile_iter = cutlass.Int32(0)
                is_valid = cutlass.Int32(1)
                clc_full_phase_mma = cutlass.Int32(0)
                while is_valid != 0:
                    consumer_stage = tile_iter % CLC_SCHED_STAGES
                    if consumer_stage == 0 and tile_iter != 0:
                        clc_full_phase_mma = clc_full_phase_mma ^ 1
                    while not nvvm.mbarrier_try_wait_parity(
                        clc_full_mbar_ptr.subview(consumer_stage),
                        clc_full_phase_mma,
                        time_limit=10_000_000,
                    ):
                        pass
                    _m_idx, _n_idx, _l_idx, vld = cute_clc.clc_response(clc_response_ptr_base + consumer_stage)
                    cute.arch.fence_proxy("async.shared", space="cta")
                    is_valid = vld
                    nvvm.bar_warp_sync(0xFFFFFFFF)
                    if elect_one:
                        empty_remote = nvvm.mapa(clc_empty_mbar_ptr.subview(consumer_stage), 0)
                        nvvm.mbarrier_arrive(empty_remote, scope=nvvm.MemScope.CLUSTER, relaxed=True)
                    tile_iter += 1
                # Dense only, same reason as above.
                if cutlass.const_expr(USE_PDL and split_k_slices == 1):
                    nvvm.griddepcontrol("launch_dependents")
                nvvm.tcgen05_relinquish_alloc_permit(group=_CTA_GROUP)
                peer_mbar = nvvm.mapa(tmem_dealloc_mbar_ptr, peer_cta_rank)
                nvvm.mbarrier_arrive(peer_mbar, scope=nvvm.MemScope.CLUSTER, relaxed=True)
                while not nvvm.mbarrier_try_wait_parity(tmem_dealloc_mbar_ptr, 0, time_limit=10_000_000):
                    pass
                alloc_ptr = cutlass.inttoptr(tmem_raw_addr, 6, cutlass.Int32)
                _tcgen05_dealloc(
                    alloc_ptr,
                    cutlass.Int32(num_tmem_alloc_cols),
                    is_exclusive=tmem_alloc_exclusive,
                    group=_CTA_GROUP,
                )

    if warp_idx < num_epilogue_warps:
        nvvm.setmaxregister(epi_reg_count, nvvm.SetMaxRegisterAction.INCREASE)
        nvvm.barrier_cta_sync(barrier_id=TMEM_ALLOC_BARRIER_ID, thread_count=tmem_alloc_bar_count)
        tmem_raw_addr = tmem_ptr_i32.load()
        base_col_id_root = tmem_raw_addr & 0xFFFF
        base_row_id = tmem_raw_addr >> 16

        if cutlass.const_expr(USE_PDL):
            nvvm.griddepcontrol("wait")

        tile_iter = cutlass.Int32(0)
        acc_full_phase_bit = cutlass.Int32(0)
        tile_m = init_tile_m
        tile_n = init_tile_n
        tile_l = init_tile_l
        is_valid = cutlass.Int32(1)
        clc_full_phase_epi = cutlass.Int32(0)
        if cutlass.const_expr(_LN_ONCE):
            # LN ONCE (module docstring): the item's N-tile group and sub-tile counter; the acc ring stays per tile (tile_iter), the CLC per item
            item_n = init_tile_n
            sub = cutlass.Int32(0)
            item_iter = cutlass.Int32(0)
            tile_n, tile_l = _item_tile(item_n, sub)

        # @@EPILOGUE_SETUP:BEGIN@@
        if cutlass.const_expr(epi_packed_lanes):
            row_id_with_warp_offset = base_row_id
        else:
            row_id_with_warp_offset = base_row_id + warp_idx * 32

        epi_spans = _epi_subtile_spans(epi_cols_per_mma_m, epi_n)
        subtile_cnt = len(epi_spans)
        if cutlass.const_expr(epi_packed_lanes):
            shape = nvvm.Tcgen05LdStShape.SHAPE_16X32BX2
            ld_half_off = 0
        else:
            shape = nvvm.Tcgen05LdStShape.SHAPE_32X32B
            ld_half_off = None
        lane = tidx % 32
        # @@EPILOGUE_SETUP:END@@

        epi_stage_idx = cutlass.Int32(EPI_SMEM_STAGES - 1)

        while is_valid != 0:
            coord_m_tile = tile_m * cgrp_tile_m_cur + m_rank * cta_tile_mnk[0]
            # @@EPILOGUE_DRAIN:BEGIN@@
            # ==== LN-PROLOGUE DUAL GEMM epilogue: out[n] = sigmoid(acc[128 + n] + b_g[n]) * (acc[n] + b_p[n]) * mask[row] ====
            # Row geometry (module docstring, ROW-BLOCK TILING): this thread owns pair row 128 * pair_member + tidx.
            k_idx, bi_idx = _row_coords(tile_m, n_kb, lkb, pair_member * cta_tile_mnk[0] + tidx)
            row_valid = (k_idx < n_res) & (bi_idx < bn)
            m_val = _row_mask(mask, row_valid, bi_idx, k_idx, n_res)
            # A tile whose MMA was skipped (the LN warps' pair-level vote) holds RESIDUE in TMEM: it is read like any other and
            # every value is discarded by the per-row select below (a select, not arithmetic -- a NaN residue cannot leak).
            # The store box origin of this CTA's 128 rows (the same decode the TMA warp used for the load).
            st_coord_k, st_coord_bi = _cta_box_origin(tile_m, n_kb, lkb, pair_member)
            out_ch_base = tile_n * _VALUE_BLOCK

            acc_stage = tile_iter % acc_stages
            if acc_stage == 0 and tile_iter != 0:
                acc_full_phase_bit = acc_full_phase_bit ^ 1

            while not nvvm.mbarrier_try_wait_parity(acc_full_mbar_ptr.subview(acc_stage), acc_full_phase_bit, time_limit=10_000_000):
                pass

            acc_base_col = base_col_id_root + acc_stage * acc_region_cols
            tmem_col_addr = (row_id_with_warp_offset << 16) | acc_base_col

            for subtile_idx in cutlass.range_constexpr(_EPI_SUBTILES):
                _col0 = subtile_idx * epi_n
                _outs = []
                _bp = []
                _bg = []
                if cutlass.const_expr(HAS_BIAS and _EPI_BIAS_HOIST and _DIAG != "no_epi_math"):
                    # "packed_hoist" / "packed_biasfold" (module docstring): the subtile's bias words BEFORE the TMEM loads -- the same 16 broadcast
                    # LDS.128 the shipped form issues after its tcgen05.wait; under the fold form the gate words are the halved `b_g / 2`.
                    _bias_base = tile_l * _BIAS_PER_PASS + tile_n * (2 * _VALUE_BLOCK) + _col0
                    for _q in cutlass.range_constexpr(epi_n // 4):
                        _bp.extend(ld_shared_v4(smem_bias.data_ptr(_bias_base + _q * 4), cutlass.Float32))
                    for _q in cutlass.range_constexpr(epi_n // 4):
                        _bg.extend(ld_shared_v4(smem_bias.data_ptr(_bias_base + _VALUE_BLOCK + _q * 4), cutlass.Float32))
                if cutlass.const_expr(_DIAG == "no_epi_math"):
                    for _i in cutlass.range_constexpr(epi_n):
                        _outs.append(cutlass.Float32(0.0) * m_val)  # DIAGNOSTIC: no TMEM read, no sigmoid; the store path stays
                else:
                    _tm_v = cutlass.inttoptr(tmem_col_addr + _col0, 6, mma_c_dtype)
                    _tm_g = cutlass.inttoptr(tmem_col_addr + _VALUE_BLOCK + _col0, 6, mma_c_dtype)
                    _v = nvvm.tcgen05_ld(shape, _tm_v, num=epi_n, offset=ld_half_off)
                    _g = nvvm.tcgen05_ld(shape, _tm_g, num=epi_n, offset=ld_half_off)
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                if cutlass.const_expr(HAS_BIAS and not _EPI_BIAS_HOIST and _DIAG != "no_epi_math"):
                    _bias_base = (
                        tile_l * _BIAS_PER_PASS + tile_n * (2 * _VALUE_BLOCK) + _col0
                    )  # the interleaved [value block | gate block] per 128 output channels
                    for _q in cutlass.range_constexpr(epi_n // 4):
                        _bp.extend(ld_shared_v4(smem_bias.data_ptr(_bias_base + _q * 4), cutlass.Float32))
                    for _q in cutlass.range_constexpr(epi_n // 4):
                        _bg.extend(ld_shared_v4(smem_bias.data_ptr(_bias_base + _VALUE_BLOCK + _q * 4), cutlass.Float32))
                _rz = []
                if cutlass.const_expr(_FUSE_RESIDUAL and _DIAG != "no_epi_math"):
                    # K3 ARM: the residual row (pair-major bf16 [row][c_out]), 32 channels of this subtile; a padded row reads row 0 and the
                    # select below discards it (the address is clamped, never a divergent load into a list -- the DSL's type join).
                    _rrow = cutlass.Int32(arith.select(row_valid.ir_value(), (bi_idx * n_res + k_idx).ir_value(), cutlass.Int32(0).ir_value()))
                    _raddr = resid.iterator.toint() + _rrow.to(cutlass.Int64) * cutlass.Int64(_C_OUT * _IO_BPE) + cutlass.Int64((out_ch_base + _col0) * _IO_BPE)
                    for _q in cutlass.range_constexpr(epi_n // 8):
                        for _w in ld_global_v4(_raddr + cutlass.Int64(_q * 16), cutlass.Int32):
                            _lo, _hi = f16x2_to_f32(_w, dtype=cutlass.BFloat16)
                            _rz.append(_lo)
                            _rz.append(_hi)
                if cutlass.const_expr(_EPI_PACKED and _DIAG != "no_epi_math"):
                    # EPILOGUE MATH "packed" (module docstring): channel pairs on the fp32x2 ALU -- the scalar form's operations in its association
                    # (gate = sigmoid(g + b_g); o = (gate * (v + b_p)) * m [+ resid]); the mask SELECT stays per element, after the arithmetic.
                    for _h in cutlass.range_constexpr(epi_n // 2):
                        _i = 2 * _h
                        if cutlass.const_expr(HAS_BIAS and _EPI_BIASFOLD):
                            # "packed_biasfold": the value add as shipped; the gate bias (staged halved) folded into the tanh argument's fma.
                            _vl, _vh = fadd2(_v[_i], _v[_i + 1], _bp[_i], _bp[_i + 1])
                            _sl, _sh = sigmoid2_biased_half(_g[_i], _g[_i + 1], _bg[_i], _bg[_i + 1])
                        else:
                            _gl, _gh = fadd2(_g[_i], _g[_i + 1], _bg[_i], _bg[_i + 1]) if cutlass.const_expr(HAS_BIAS) else (_g[_i], _g[_i + 1])
                            _vl, _vh = fadd2(_v[_i], _v[_i + 1], _bp[_i], _bp[_i + 1]) if cutlass.const_expr(HAS_BIAS) else (_v[_i], _v[_i + 1])
                            _sl, _sh = sigmoid2(_gl, _gh)
                        _pl, _ph = fmul2(_sl, _sh, _vl, _vh)
                        _ol, _oh = fmul2(_pl, _ph, m_val, m_val)
                        if cutlass.const_expr(_FUSE_RESIDUAL):
                            _ol, _oh = fadd2(_ol, _oh, _rz[_i], _rz[_i + 1])
                        # A masked / padded row is EXACTLY +0.0 (a select, never x * 0: the residue may be anything).
                        _outs.append(cutlass.Float32(arith.select((m_val == cutlass.Float32(0.0)).ir_value(), cutlass.Float32(0.0).ir_value(), _ol.ir_value())))
                        _outs.append(cutlass.Float32(arith.select((m_val == cutlass.Float32(0.0)).ir_value(), cutlass.Float32(0.0).ir_value(), _oh.ir_value())))
                else:
                    for _i in cutlass.range_constexpr(epi_n if _DIAG != "no_epi_math" else 0):
                        if cutlass.const_expr(HAS_BIAS):
                            _gate = sigmoid(_g[_i] + _bg[_i])
                            _o = _gate * (_v[_i] + _bp[_i]) * m_val
                        else:
                            _gate = sigmoid(_g[_i])
                            _o = _gate * _v[_i] * m_val
                        if cutlass.const_expr(_FUSE_RESIDUAL):
                            _o = _o + _rz[_i]  # one rounding for update + residual (the fused block form; the API's unfused form rounds twice)
                        # A masked / padded row is EXACTLY +0.0 (a select, never x * 0: the residue may be anything).
                        _outs.append(cutlass.Float32(arith.select((m_val == cutlass.Float32(0.0)).ir_value(), cutlass.Float32(0.0).ir_value(), _o.ir_value())))

                if subtile_idx == _EPI_SUBTILES - 1:
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
                    if elect_one:
                        if cutlass.const_expr(cta_group == 2):
                            nvvm.mbarrier_arrive(
                                nvvm.mapa(acc_empty_mbar_ptr.subview(acc_stage), pair_leader_rank),
                                scope=nvvm.MemScope.CLUSTER,
                                relaxed=True,
                            )
                        else:
                            nvvm.mbarrier_arrive(acc_empty_mbar_ptr.subview(acc_stage))

                epi_stage_idx = (epi_stage_idx + 1) % EPI_SMEM_STAGES
                _tsv_0 = cutlass.Array(base=smem_d_ptr.data_ptr(epi_stage_idx * epi_subtile_elems), shape=4096, dtype=cutlass.BFloat16)
                if cutlass.const_expr(_DUAL):
                    # K3 ARM: pair-major staging sD[row][32 channels] bf16 -- the rendering's own layout: lane = row writes its 64 bytes as four
                    # 16-byte chunks under Swizzle(2, 4, 3) = the s64b TMA store pattern (descriptor-read AND bank-spread: the two jobs agree).
                    _tsv_0.data_ptr(tidx * 32).store_swizzled(
                        cutlass.Vector.from_elements(tuple(_outs), cutlass.Float32).to(cutlass.BFloat16), alignment=64, swizzle=cutlass.Swizzle(2, 4, 3)
                    )
                else:
                    # Transposed staging: sD[channel][row] bf16 -- 32 lanes of a warp write 64 contiguous bytes per channel.
                    for _i in cutlass.range_constexpr(epi_n):
                        _tsv_0.data_ptr(_i * epi_stage_rows + tidx).store(_outs[_i].to(cutlass.BFloat16))
                cute.arch.fence_view_async_shared()
                nvvm.barrier_cta_sync(barrier_id=EPI_SYNC_BAR_ID, thread_count=num_epilogue_warps * 32)
                if warp_idx == 0:
                    if elect_one:
                        if cutlass.const_expr(_DUAL):
                            # the 3-D (channel, k, bi) box of the pair-major [bn][n_res][c_out] output: rows k >= n_res / bi >= bn are clipped
                            nvvm.cp_async_bulk_tensor_global_shared_cta(
                                tma_c_descs[0].get_ptr(),
                                _tsv_0.data_ptr(),
                                (out_ch_base + _col0, st_coord_k, st_coord_bi),
                            )
                        else:
                            nvvm.cp_async_bulk_tensor_global_shared_cta(
                                tma_c_descs[0].get_ptr(),
                                _tsv_0.data_ptr(),
                                (st_coord_k, st_coord_bi, out_ch_base + _col0, tile_l),
                            )
                    if elect_one:
                        nvvm.cp_async_bulk_commit_group()
                    nvvm.cp_async_bulk_wait_group(EPI_SMEM_STAGES - 1, read=True)
                nvvm.barrier_cta_sync(barrier_id=EPI_SYNC_BAR_ID, thread_count=num_epilogue_warps * 32)
            # ==== LN-PROLOGUE DUAL GEMM epilogue: END ====
            # @@EPILOGUE_DRAIN:END@@
            if cutlass.const_expr(_LN_ONCE):
                # LN ONCE: the next sub-tile of the item, or -- after its last -- the per-ITEM CLC consume (the shipped block in the else arm, once per item)
                if sub == _TPI - 1:
                    sub = cutlass.Int32(0)
                    consumer_stage = item_iter % CLC_SCHED_STAGES
                    if consumer_stage == 0 and item_iter != 0:
                        clc_full_phase_epi = clc_full_phase_epi ^ 1
                    while not nvvm.mbarrier_try_wait_parity(clc_full_mbar_ptr.subview(consumer_stage), clc_full_phase_epi, time_limit=10_000_000):
                        pass
                    m_idx, n_idx, l_idx, vld = cute_clc.clc_response(clc_response_ptr_base + consumer_stage)
                    cute.arch.fence_proxy("async.shared", space="cta")
                    is_valid = vld
                    tile_m, item_n = _item_decode(m_idx, n_idx, gridx, gridy, swizzle_w)
                    nvvm.bar_warp_sync(0xFFFFFFFF)
                    if elect_one:
                        empty_remote = nvvm.mapa(clc_empty_mbar_ptr.subview(consumer_stage), 0)
                        nvvm.mbarrier_arrive(empty_remote, scope=nvvm.MemScope.CLUSTER, relaxed=True)
                    item_iter += 1
                else:
                    sub += 1
                tile_n, tile_l = _item_tile(item_n, sub)
                tile_iter += 1
            else:
                consumer_stage = tile_iter % CLC_SCHED_STAGES
                if consumer_stage == 0 and tile_iter != 0:
                    clc_full_phase_epi = clc_full_phase_epi ^ 1
                while not nvvm.mbarrier_try_wait_parity(
                    clc_full_mbar_ptr.subview(consumer_stage),
                    clc_full_phase_epi,
                    time_limit=10_000_000,
                ):
                    pass
                m_idx, n_idx, l_idx, vld = cute_clc.clc_response(clc_response_ptr_base + consumer_stage)
                cute.arch.fence_proxy("async.shared", space="cta")
                is_valid = vld
                epi_raw_m = m_idx >> _preferred_cluster_m_shift
                epi_raw_n = n_idx >> _preferred_cluster_n_shift
                epi_nt_m = gridx >> _preferred_cluster_m_shift
                epi_nt_n = gridy >> _preferred_cluster_n_shift
                if cutlass.const_expr(fallback_cluster_shape_mnk is not None):
                    if (cluster_m != cluster_shape_mnk[0]) | (cluster_n != cluster_shape_mnk[1]):
                        epi_raw_m = m_idx >> _fallback_cluster_m_shift
                        epi_raw_n = n_idx >> _fallback_cluster_n_shift
                        epi_nt_m = gridx >> _fallback_cluster_m_shift
                        epi_nt_n = gridy >> _fallback_cluster_n_shift
                tile_m, tile_n = _l2_swizzle_tile(
                    epi_raw_m,
                    epi_raw_n,
                    epi_nt_m,
                    epi_nt_n,
                    swizzle_w,
                    identity=tile_swizzle_n == 1,
                )
                tile_l = l_idx
                tile_m, tile_l = _decode_pass(tile_m, tile_l)  # TILE ORDER (module docstring)
                nvvm.bar_warp_sync(0xFFFFFFFF)
                if elect_one:
                    empty_remote = nvvm.mapa(clc_empty_mbar_ptr.subview(consumer_stage), 0)
                    nvvm.mbarrier_arrive(empty_remote, scope=nvvm.MemScope.CLUSTER, relaxed=True)

                tile_iter += 1

        if warp_idx == 0:
            nvvm.cp_async_bulk_wait_group(0, read=True)

    if warp_idx == unused_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)

    # =====================================================================================================================
    # LN-PROLOGUE DUAL GEMM: the LayerNorm warps.  Per tile: wait every K stage of the tile (a_full), two-pass fp32 statistics
    # over the row held in registers, normalise + write back in place as bf16, (pass 0) store (mean, rstd), (leader) the
    # pair-level mask vote into the first stage's flag slot, fence the generic stores to the async proxy, then ONE elected
    # release arrive per LN warp on every stage's a_norm_full in the LEADER (8 per stage = its init), then consume the CLC.
    # =====================================================================================================================
    if warp_idx >= ln_warp_base:
        nvvm.setmaxregister(ln_reg_count, nvvm.SetMaxRegisterAction.INCREASE)
        ln_tid = tidx - ln_warp_base * 32  # this CTA's row 0..127 == the SMEM A-stage row normalised by the calling lane
        ln_warp = warp_idx - ln_warp_base
        ln_lane = ln_tid & 31
        ln_swz = ln_tid & 7  # the SWIZZLE_128B chunk XOR of this row
        _KOP = 1 if _DUAL else 0  # the K-MAJOR operand this thread normalises as a row: K3's operand 1 (z); K1's only operand
        if cutlass.const_expr(_DUAL):
            # K3 ARM: lane constants of operand 0's MN-major stage (lane = pair row j of this CTA; box mg = j >> 6, j_l = j & 63).  Element
            # (c, j) sits at mg * 4096 + 64 c + (j_l XOR ((c & 7) << 3)) [bf16 units]; the 32-bit word holding (c, j) and (c, j ^ 1) at
            # mg * 2048 + 32 c + ((j_l >> 1) XOR ((c & 7) << 2)).  The XOR constant takes 8 values -> 8 pre-XORed lane offsets; the per-channel
            # term (64 c resp. 32 c) is an immediate: no address ALU per channel.  32 consecutive lanes read 64 contiguous bytes of one 128-B
            # row (16 words, 2 lanes each): conflict-free; the in-place b16 writes cover the same span.
            _xw = (ln_tid >> 6) * 2048 + ((ln_tid & 63) >> 1)
            _xe = (ln_tid >> 6) * 4096 + (ln_tid & 63)
            _xow = []
            _xoe = []
            for _kx in cutlass.range_constexpr(8):
                _xow.append(_xw ^ (_kx << 2))
                _xoe.append(_xe ^ (_kx << 3))
            # prmt selector: an even j takes the LOW halves of the two channel words, an odd j the HIGH halves -> the packed (c, c + 1) pair of row j
            _sel = cutlass.Int32(arith.select(((ln_tid & 1) != 0).ir_value(), cutlass.Int32(0x7632).ir_value(), cutlass.Int32(0x5410).ir_value()))
            _a0w = cutlass.Array(base=smem_a_list[0].data_ptr(), shape=sA_elems * ab_stages // 2, dtype=cutlass.Int32)
        ab_iter = cutlass.Int32(0)
        a_full_phase_ln = cutlass.Int32(0)
        tile_m = init_tile_m
        tile_n = init_tile_n
        tile_l = init_tile_l
        tile_iter = cutlass.Int32(0)
        is_valid = cutlass.Int32(1)
        clc_full_phase_ln = cutlass.Int32(0)
        while is_valid != 0:
            r_pair = pair_member * cta_tile_mnk[0] + ln_tid
            k_idx, bi_idx = _row_coords(tile_m, n_kb, lkb, r_pair)
            row_valid = (k_idx < n_res) & (bi_idx < bn)
            # (1) the ring slots / phases of this tile's _NK stages (the TMA warp's stepping), then wait for all of them
            _sts = []
            _phs = []
            if cutlass.const_expr(_LN_ONCE):
                # LN ONCE (module docstring): the item's resident region slot and its phase -- one per slot, the slot's _NK stages are filled and
                # released together; the item counter is tile_iter (one CLC consume per iteration = per item).  Everything below addresses the
                # region stages through _sts exactly as it addressed the ring slots.
                a_slot_ln = tile_iter % _A_SLOTS
                if a_slot_ln == 0 and tile_iter != 0:
                    a_full_phase_ln = a_full_phase_ln ^ 1
                for _j in cutlass.range_constexpr(_NK):
                    _sts.append(a_slot_ln * _NK + _j)
                    _phs.append(a_full_phase_ln)
            else:
                for _j in cutlass.range_constexpr(_NK):
                    _st = ab_iter % ab_stages
                    if _st == 0 and ab_iter != 0:
                        a_full_phase_ln = a_full_phase_ln ^ 1
                    _sts.append(_st)
                    _phs.append(a_full_phase_ln)
                    ab_iter += 1
            for _j in cutlass.range_constexpr(_NK):
                while not nvvm.mbarrier_try_wait_parity(a_full_mbar_ptr.subview(_sts[_j]), _phs[_j], time_limit=10_000_000):
                    pass
            # DIAGNOSTIC "ln_first_tile_only" (LnDualGemmParams.diag): the LayerNorm passes (2)-(5b) run on the row block's FIRST (pass, N tile)
            # tile only -- its other tiles forward the stage UNNORMALISED (the output is WRONG by design, like "no_ln"); every mbarrier wait / arrive
            # below is unchanged (the K3 transpose form's LN_SYNC named barrier inside the block is reached by all four LN warps or by none: the
            # condition is CTA-uniform).  The kernel time against the production arm BOUNDS the "LayerNorm once per row block" lever (K1 streams a row
            # block 2 c_out / 128 times, K3 c_out / 128 times).  Every other rendering takes a constant True here (folded away: the cubin is unchanged).
            if cutlass.const_expr(_DIAG == "ln_first_tile_only"):
                _ln_this_tile = (tile_n == 0) & (tile_l == 0)
            else:
                _ln_this_tile = cutlass.Boolean(True)
            if _ln_this_tile:
                # (2) pass A on the K-MAJOR operand: the row into registers (K/2 packed bf16x2 words, chunks in LOGICAL channel order), fp32 sum
                _ln_work = cutlass.const_expr(_DIAG != "no_ln")
                _row_words = []
                _sum = cutlass.Float32(0.0)
                _sumsq = cutlass.Float32(0.0)
                if cutlass.const_expr(_LN_PF):
                    # LN PASS FORMS "smem_rolled_pf" / "smem_rolled_tree" (module docstring): the rolled re-read with a 2-deep software pipeline -- the
                    # words of chunk (q + 1) & 7 are loaded at the top of iteration q (the prologue loads chunk 0; iteration 7 re-reads chunk 0, discarded)
                    # and iteration q sums the words loaded one iteration earlier (loop-carried scalars) -- the same words into the same additions in the
                    # same order as "smem_rolled" (bitwise); the tree form sums word w of every chunk into partial _ta<w> (a 2-deep chain per chunk) and
                    # combines the four once per pass: a different fp32 order, the oracle's form.
                    _ta0 = cutlass.Float32(0.0)
                    _ta1 = cutlass.Float32(0.0)
                    _ta2 = cutlass.Float32(0.0)
                    _ta3 = cutlass.Float32(0.0)
                    for _j in cutlass.range_constexpr(_NK if _ln_work else 0):
                        _row_base = sA_elems * _sts[_j] + ln_tid * cta_tile_mnk[2]
                        _pa0, _pa1, _pa2, _pa3 = ld_shared_v4(smem_a_list[_KOP].data_ptr(_row_base + ln_swz * 8), cutlass.Int32)  # chunk 0 (0 ^ swz = swz)
                        for _q in cutlass.range(cta_tile_mnk[2] // 8, unroll=1):  # unroll=1 = NO unrolling (the rolled form's point)
                            _nqa = (_q + 1) & (cta_tile_mnk[2] // 8 - 1)
                            _na0, _na1, _na2, _na3 = ld_shared_v4(smem_a_list[_KOP].data_ptr(_row_base + (_nqa ^ ln_swz) * 8), cutlass.Int32)
                            if cutlass.const_expr(_LN_TREE):
                                _tl0, _th0 = f16x2_to_f32(_pa0, dtype=cutlass.BFloat16)
                                _tl1, _th1 = f16x2_to_f32(_pa1, dtype=cutlass.BFloat16)
                                _tl2, _th2 = f16x2_to_f32(_pa2, dtype=cutlass.BFloat16)
                                _tl3, _th3 = f16x2_to_f32(_pa3, dtype=cutlass.BFloat16)
                                _ta0 = _ta0 + _tl0 + _th0
                                _ta1 = _ta1 + _tl1 + _th1
                                _ta2 = _ta2 + _tl2 + _th2
                                _ta3 = _ta3 + _tl3 + _th3
                                if cutlass.const_expr(_DIAG == "one_pass_stats"):
                                    for _tlx, _thx in ((_tl0, _th0), (_tl1, _th1), (_tl2, _th2), (_tl3, _th3)):
                                        _sumsq = _sumsq + _tlx * _tlx + _thx * _thx
                            else:
                                for _pax in (_pa0, _pa1, _pa2, _pa3):
                                    _plo, _phi = f16x2_to_f32(_pax, dtype=cutlass.BFloat16)
                                    _sum = _sum + _plo + _phi
                                    if cutlass.const_expr(_DIAG == "one_pass_stats"):
                                        _sumsq = _sumsq + _plo * _plo + _phi * _phi
                            _pa0, _pa1, _pa2, _pa3 = _na0, _na1, _na2, _na3
                    if cutlass.const_expr(_LN_TREE):
                        _sum = (_ta0 + _ta1) + (_ta2 + _ta3)
                elif cutlass.const_expr(_LN_SMEM):
                    # LN PASS FORMS "smem" / "smem_rolled" (module docstring): pass A only SUMS -- the row is not kept; passes B / C re-read the chunks
                    for _j in cutlass.range_constexpr(_NK if _ln_work else 0):
                        _row_base = sA_elems * _sts[_j] + ln_tid * cta_tile_mnk[2]
                        if cutlass.const_expr(_LN_ROLLED):
                            # a runtime loop carries SCALARS only (the DSL joins every assignment of a loop-body name to one type: no list names here)
                            for _q in cutlass.range(cta_tile_mnk[2] // 8, unroll=1):  # unroll=1 = NO unrolling (the rolled form's point)
                                _aw0, _aw1, _aw2, _aw3 = ld_shared_v4(smem_a_list[_KOP].data_ptr(_row_base + (_q ^ ln_swz) * 8), cutlass.Int32)
                                for _awx in (_aw0, _aw1, _aw2, _aw3):
                                    _alo, _ahi = f16x2_to_f32(_awx, dtype=cutlass.BFloat16)
                                    _sum = _sum + _alo + _ahi
                                    if cutlass.const_expr(_DIAG == "one_pass_stats"):
                                        _sumsq = _sumsq + _alo * _alo + _ahi * _ahi
                        else:
                            for _q in cutlass.range_constexpr(cta_tile_mnk[2] // 8):
                                _words = ld_shared_v4(smem_a_list[_KOP].data_ptr(_row_base + (_q ^ ln_swz) * 8), cutlass.Int32)
                                for _w in _words:
                                    _lo, _hi = f16x2_to_f32(_w, dtype=cutlass.BFloat16)
                                    _sum = _sum + _lo + _hi
                                    if cutlass.const_expr(_DIAG == "one_pass_stats"):
                                        _sumsq = _sumsq + _lo * _lo + _hi * _hi
                else:
                    for _j in cutlass.range_constexpr(_NK if _ln_work else 0):
                        _row_base = sA_elems * _sts[_j] + ln_tid * cta_tile_mnk[2]
                        for _q in cutlass.range_constexpr(cta_tile_mnk[2] // 8):
                            _phys = _q ^ ln_swz
                            _words = ld_shared_v4(smem_a_list[_KOP].data_ptr(_row_base + _phys * 8), cutlass.Int32)
                            for _w in _words:
                                _row_words.append(_w)
                                _lo, _hi = f16x2_to_f32(_w, dtype=cutlass.BFloat16)
                                _sum = _sum + _lo + _hi
                                if cutlass.const_expr(_DIAG == "one_pass_stats"):
                                    _sumsq = _sumsq + _lo * _lo + _hi * _hi
                _mean = _sum * cutlass.Float32(_INV_K)
                # (3) pass B: sum of squared deviations (two-pass: the mean is subtracted before squaring)
                _sq = cutlass.Float32(0.0)
                if cutlass.const_expr(_DIAG == "one_pass_stats"):
                    _sq = _sumsq - _sum * _mean  # DIAGNOSTIC: E[x^2] - mean^2 from the first pass
                else:
                    if cutlass.const_expr(_LN_PF):
                        # the pipelined re-read of pass A's chunks (the stage is unchanged until pass C): the same words, the same order -> bitwise;
                        # the tree form: partial _tb<w> takes word w of every chunk (fma(dh, dh, fma(dl, dl, t)): a 2-deep chain per chunk)
                        _tb0 = cutlass.Float32(0.0)
                        _tb1 = cutlass.Float32(0.0)
                        _tb2 = cutlass.Float32(0.0)
                        _tb3 = cutlass.Float32(0.0)
                        for _j in cutlass.range_constexpr(_NK if _ln_work else 0):
                            _row_base = sA_elems * _sts[_j] + ln_tid * cta_tile_mnk[2]
                            _pb0, _pb1, _pb2, _pb3 = ld_shared_v4(smem_a_list[_KOP].data_ptr(_row_base + ln_swz * 8), cutlass.Int32)  # chunk 0
                            for _q in cutlass.range(cta_tile_mnk[2] // 8, unroll=1):  # unroll=1 = NO unrolling (the rolled form's point)
                                _nqb = (_q + 1) & (cta_tile_mnk[2] // 8 - 1)
                                _nb0, _nb1, _nb2, _nb3 = ld_shared_v4(smem_a_list[_KOP].data_ptr(_row_base + (_nqb ^ ln_swz) * 8), cutlass.Int32)
                                if cutlass.const_expr(_LN_TREE):
                                    _ul0, _uh0 = f16x2_to_f32(_pb0, dtype=cutlass.BFloat16)
                                    _ul1, _uh1 = f16x2_to_f32(_pb1, dtype=cutlass.BFloat16)
                                    _ul2, _uh2 = f16x2_to_f32(_pb2, dtype=cutlass.BFloat16)
                                    _ul3, _uh3 = f16x2_to_f32(_pb3, dtype=cutlass.BFloat16)
                                    _udl0 = _ul0 - _mean
                                    _udh0 = _uh0 - _mean
                                    _udl1 = _ul1 - _mean
                                    _udh1 = _uh1 - _mean
                                    _udl2 = _ul2 - _mean
                                    _udh2 = _uh2 - _mean
                                    _udl3 = _ul3 - _mean
                                    _udh3 = _uh3 - _mean
                                    _tb0 = _tb0 + _udl0 * _udl0 + _udh0 * _udh0
                                    _tb1 = _tb1 + _udl1 * _udl1 + _udh1 * _udh1
                                    _tb2 = _tb2 + _udl2 * _udl2 + _udh2 * _udh2
                                    _tb3 = _tb3 + _udl3 * _udl3 + _udh3 * _udh3
                                else:
                                    for _pbx in (_pb0, _pb1, _pb2, _pb3):
                                        _qlo, _qhi = f16x2_to_f32(_pbx, dtype=cutlass.BFloat16)
                                        _qdl = _qlo - _mean
                                        _qdh = _qhi - _mean
                                        _sq = _sq + _qdl * _qdl + _qdh * _qdh
                                _pb0, _pb1, _pb2, _pb3 = _nb0, _nb1, _nb2, _nb3
                        if cutlass.const_expr(_LN_TREE):
                            _sq = (_tb0 + _tb1) + (_tb2 + _tb3)
                    elif cutlass.const_expr(_LN_SMEM):
                        # the re-read of pass A's chunks (the stage is unchanged until pass C): the same words, the same order -> bitwise the register form
                        for _j in cutlass.range_constexpr(_NK if _ln_work else 0):
                            _row_base = sA_elems * _sts[_j] + ln_tid * cta_tile_mnk[2]
                            if cutlass.const_expr(_LN_ROLLED):
                                for _q in cutlass.range(cta_tile_mnk[2] // 8, unroll=1):  # unroll=1 = NO unrolling (the rolled form's point)
                                    _bw0, _bw1, _bw2, _bw3 = ld_shared_v4(smem_a_list[_KOP].data_ptr(_row_base + (_q ^ ln_swz) * 8), cutlass.Int32)
                                    for _bwx in (_bw0, _bw1, _bw2, _bw3):
                                        _blo, _bhi = f16x2_to_f32(_bwx, dtype=cutlass.BFloat16)
                                        _bdl = _blo - _mean
                                        _bdh = _bhi - _mean
                                        _sq = _sq + _bdl * _bdl + _bdh * _bdh
                            else:
                                for _q in cutlass.range_constexpr(cta_tile_mnk[2] // 8):
                                    _words = ld_shared_v4(smem_a_list[_KOP].data_ptr(_row_base + (_q ^ ln_swz) * 8), cutlass.Int32)
                                    for _w in _words:
                                        _lo, _hi = f16x2_to_f32(_w, dtype=cutlass.BFloat16)
                                        _dl = _lo - _mean
                                        _dh = _hi - _mean
                                        _sq = _sq + _dl * _dl + _dh * _dh
                    else:
                        for _w in _row_words:
                            _lo, _hi = f16x2_to_f32(_w, dtype=cutlass.BFloat16)
                            _dl = _lo - _mean
                            _dh = _hi - _mean
                            _sq = _sq + _dl * _dl + _dh * _dh
                _rstd = cute.math.rsqrt(_sq * cutlass.Float32(_INV_K) + cutlass.Float32(_EPS), fastmath=True)
                # (4) the statistics of OPERAND 0, once per row (pass 0, N tile 0), valid rows only -- K1's only operand is K-major (here);
                #     K3's operand 0 is the MN-major one (its statistics are written in (5b) below)
                if cutlass.const_expr(WANT_STATS and not _DUAL):
                    if row_valid & (tile_l == 0) & (tile_n == 0):
                        _stats_addr = stats.iterator.toint() + (bi_idx * n_res + k_idx).to(cutlass.Int64) * cutlass.Int64(8)
                        st_global_v2(_stats_addr, [_mean, _rstd], cutlass.Float32)
                # (5) pass C on the K-major operand: normalise with its gamma / beta (fp32, SMEM broadcast at 2 K op), round once to bf16, write
                #     back IN PLACE
                _wi = 0
                for _j in cutlass.range_constexpr(_NK if (_DIAG != "no_ln_norm" and _ln_work) else 0):
                    _row_base = sA_elems * _sts[_j] + ln_tid * cta_tile_mnk[2]
                    if cutlass.const_expr(_LN_SMEM and _LN_ROLLED):
                        # the re-read form, rolled: the chunk index is a loop variable; its XOR and its gamma / beta offset are computed per chunk
                        for _q in cutlass.range(cta_tile_mnk[2] // 8, unroll=1):  # unroll=1 = NO unrolling (the rolled form's point)
                            _cphys = _q ^ ln_swz
                            _cc0 = 2 * _K * _KOP + _j * cta_tile_mnk[2] + _q * 8
                            _cg0, _cg1, _cg2, _cg3 = ld_shared_v4(smem_ln_params.data_ptr(_cc0), cutlass.Float32)
                            _cg4, _cg5, _cg6, _cg7 = ld_shared_v4(smem_ln_params.data_ptr(_cc0 + 4), cutlass.Float32)
                            _cb0, _cb1, _cb2, _cb3 = ld_shared_v4(smem_ln_params.data_ptr(_K + _cc0), cutlass.Float32)
                            _cb4, _cb5, _cb6, _cb7 = ld_shared_v4(smem_ln_params.data_ptr(_K + _cc0 + 4), cutlass.Float32)
                            _cw0, _cw1, _cw2, _cw3 = ld_shared_v4(smem_a_list[_KOP].data_ptr(_row_base + _cphys * 8), cutlass.Int32)
                            _cl0, _ch0 = f16x2_to_f32(_cw0, dtype=cutlass.BFloat16)
                            _cl1, _ch1 = f16x2_to_f32(_cw1, dtype=cutlass.BFloat16)
                            _cl2, _ch2 = f16x2_to_f32(_cw2, dtype=cutlass.BFloat16)
                            _cl3, _ch3 = f16x2_to_f32(_cw3, dtype=cutlass.BFloat16)
                            _cpk0 = fp32_to_fp16((_cl0 - _mean) * _rstd * _cg0 + _cb0, (_ch0 - _mean) * _rstd * _cg1 + _cb1, dtype=cutlass.BFloat16)
                            _cpk1 = fp32_to_fp16((_cl1 - _mean) * _rstd * _cg2 + _cb2, (_ch1 - _mean) * _rstd * _cg3 + _cb3, dtype=cutlass.BFloat16)
                            _cpk2 = fp32_to_fp16((_cl2 - _mean) * _rstd * _cg4 + _cb4, (_ch2 - _mean) * _rstd * _cg5 + _cb5, dtype=cutlass.BFloat16)
                            _cpk3 = fp32_to_fp16((_cl3 - _mean) * _rstd * _cg6 + _cb6, (_ch3 - _mean) * _rstd * _cg7 + _cb7, dtype=cutlass.BFloat16)
                            st_shared_v4(smem_a_list[_KOP].data_ptr(_row_base + _cphys * 8), [_cpk0, _cpk1, _cpk2, _cpk3], cutlass.Int32)
                    elif cutlass.const_expr(_LN_PF):
                        # the pipelined rolled form: iteration q normalises chunk q from the 20 words loaded one iteration earlier (4 data words, the 8 gamma
                        # and 8 beta of its channels) while chunk (q + 1) & 7's 20 words load; the arithmetic per element is the rolled form's, in its order
                        _pcc0 = 2 * _K * _KOP + _j * cta_tile_mnk[2]
                        _pg0, _pg1, _pg2, _pg3 = ld_shared_v4(smem_ln_params.data_ptr(_pcc0), cutlass.Float32)
                        _pg4, _pg5, _pg6, _pg7 = ld_shared_v4(smem_ln_params.data_ptr(_pcc0 + 4), cutlass.Float32)
                        _pk0, _pk1, _pk2, _pk3 = ld_shared_v4(smem_ln_params.data_ptr(_K + _pcc0), cutlass.Float32)
                        _pk4, _pk5, _pk6, _pk7 = ld_shared_v4(smem_ln_params.data_ptr(_K + _pcc0 + 4), cutlass.Float32)
                        _pw0, _pw1, _pw2, _pw3 = ld_shared_v4(smem_a_list[_KOP].data_ptr(_row_base + ln_swz * 8), cutlass.Int32)  # chunk 0
                        for _q in cutlass.range(cta_tile_mnk[2] // 8, unroll=1):  # unroll=1 = NO unrolling (the rolled form's point)
                            _nqc = (_q + 1) & (cta_tile_mnk[2] // 8 - 1)
                            _ncc0 = 2 * _K * _KOP + _j * cta_tile_mnk[2] + _nqc * 8
                            _ng0, _ng1, _ng2, _ng3 = ld_shared_v4(smem_ln_params.data_ptr(_ncc0), cutlass.Float32)
                            _ng4, _ng5, _ng6, _ng7 = ld_shared_v4(smem_ln_params.data_ptr(_ncc0 + 4), cutlass.Float32)
                            _nk0, _nk1, _nk2, _nk3 = ld_shared_v4(smem_ln_params.data_ptr(_K + _ncc0), cutlass.Float32)
                            _nk4, _nk5, _nk6, _nk7 = ld_shared_v4(smem_ln_params.data_ptr(_K + _ncc0 + 4), cutlass.Float32)
                            _nw0, _nw1, _nw2, _nw3 = ld_shared_v4(smem_a_list[_KOP].data_ptr(_row_base + (_nqc ^ ln_swz) * 8), cutlass.Int32)
                            _pphys = _q ^ ln_swz
                            _pl0, _ph0 = f16x2_to_f32(_pw0, dtype=cutlass.BFloat16)
                            _pl1, _ph1 = f16x2_to_f32(_pw1, dtype=cutlass.BFloat16)
                            _pl2, _ph2 = f16x2_to_f32(_pw2, dtype=cutlass.BFloat16)
                            _pl3, _ph3 = f16x2_to_f32(_pw3, dtype=cutlass.BFloat16)
                            _ppk0 = fp32_to_fp16((_pl0 - _mean) * _rstd * _pg0 + _pk0, (_ph0 - _mean) * _rstd * _pg1 + _pk1, dtype=cutlass.BFloat16)
                            _ppk1 = fp32_to_fp16((_pl1 - _mean) * _rstd * _pg2 + _pk2, (_ph1 - _mean) * _rstd * _pg3 + _pk3, dtype=cutlass.BFloat16)
                            _ppk2 = fp32_to_fp16((_pl2 - _mean) * _rstd * _pg4 + _pk4, (_ph2 - _mean) * _rstd * _pg5 + _pk5, dtype=cutlass.BFloat16)
                            _ppk3 = fp32_to_fp16((_pl3 - _mean) * _rstd * _pg6 + _pk6, (_ph3 - _mean) * _rstd * _pg7 + _pk7, dtype=cutlass.BFloat16)
                            st_shared_v4(smem_a_list[_KOP].data_ptr(_row_base + _pphys * 8), [_ppk0, _ppk1, _ppk2, _ppk3], cutlass.Int32)
                            _pg0, _pg1, _pg2, _pg3, _pg4, _pg5, _pg6, _pg7 = _ng0, _ng1, _ng2, _ng3, _ng4, _ng5, _ng6, _ng7
                            _pk0, _pk1, _pk2, _pk3, _pk4, _pk5, _pk6, _pk7 = _nk0, _nk1, _nk2, _nk3, _nk4, _nk5, _nk6, _nk7
                            _pw0, _pw1, _pw2, _pw3 = _nw0, _nw1, _nw2, _nw3
                    else:
                        for _q in cutlass.range_constexpr(cta_tile_mnk[2] // 8):
                            _phys = _q ^ ln_swz
                            _c0 = 2 * _K * _KOP + _j * cta_tile_mnk[2] + _q * 8
                            _gam = []
                            _bet = []
                            _gam.extend(ld_shared_v4(smem_ln_params.data_ptr(_c0), cutlass.Float32))
                            _gam.extend(ld_shared_v4(smem_ln_params.data_ptr(_c0 + 4), cutlass.Float32))
                            _bet.extend(ld_shared_v4(smem_ln_params.data_ptr(_K + _c0), cutlass.Float32))
                            _bet.extend(ld_shared_v4(smem_ln_params.data_ptr(_K + _c0 + 4), cutlass.Float32))
                            if cutlass.const_expr(_LN_SMEM):
                                _words = ld_shared_v4(smem_a_list[_KOP].data_ptr(_row_base + _phys * 8), cutlass.Int32)  # the re-read form, unrolled
                            else:
                                _words = _row_words[_wi : _wi + 4]
                            _packed = []
                            for _e in cutlass.range_constexpr(4):
                                _lo, _hi = f16x2_to_f32(_words[_e], dtype=cutlass.BFloat16)
                                _ylo = (_lo - _mean) * _rstd * _gam[2 * _e] + _bet[2 * _e]
                                _yhi = (_hi - _mean) * _rstd * _gam[2 * _e + 1] + _bet[2 * _e + 1]
                                _packed.append(fp32_to_fp16(_ylo, _yhi, dtype=cutlass.BFloat16))
                            _wi += 4
                            st_shared_v4(smem_a_list[_KOP].data_ptr(_row_base + _phys * 8), _packed, cutlass.Int32)
                # (5b) K3 ARM: operand 0 (y rows) in its MN-major stage -- lane = j, loop over the channels.  Pass A: per channel pair two 32-bit
                #      loads (the words holding (c, j), (c, j ^ 1)) + ONE prmt -> the packed (c, c + 1) word of row j (64 words, K1's footprint);
                #      pass B from the registers; the LN2 statistics; pass C: (x - mean) * rstd [* gamma + beta] -> bf16 -> written back IN
                #      PLACE as b16 ("mn_major": the MMA reads the stage MN-major) or as the K-major row after LN_SYNC ("transpose": lane r's
                #      row r overlaps channel-row r of box 0 that every lane still reads -- all reads before any write, P11).
                if cutlass.const_expr(_DUAL and _ln_work):
                    _row_words0 = []
                    _sum0 = cutlass.Float32(0.0)
                    if cutlass.const_expr(_LN_PF):
                        # the pipelined re-read forms on operand 0: chunk (q + 1) & 7's eight channel words load (`_a0_chunk_words`) while chunk q's four
                        # pairs are prmt'd, unpacked and summed -- serially in "smem_rolled_pf" (bitwise the rolled form), into partial _tp<e> per pair e
                        # in "smem_rolled_tree" (combined once: (t0 + t1) + (t2 + t3))
                        _tp0 = cutlass.Float32(0.0)
                        _tp1 = cutlass.Float32(0.0)
                        _tp2 = cutlass.Float32(0.0)
                        _tp3 = cutlass.Float32(0.0)
                        for _j in cutlass.range_constexpr(_NK):
                            _wb = (sA_elems // 2) * _sts[_j]
                            _pya0, _pyc0, _pya1, _pyc1, _pya2, _pyc2, _pya3, _pyc3 = _a0_chunk_words(_a0w, _wb, _xow, 0)
                            for _q in cutlass.range(cta_tile_mnk[2] // 8, unroll=1):  # unroll=1 = NO unrolling (the rolled form's point)
                                _nqy = (_q + 1) & (cta_tile_mnk[2] // 8 - 1)
                                _nya0, _nyc0, _nya1, _nyc1, _nya2, _nyc2, _nya3, _nyc3 = _a0_chunk_words(_a0w, _wb, _xow, _nqy)
                                if cutlass.const_expr(_LN_TREE):
                                    _yl0, _yh0 = _a0_pair_f32(_pya0, _pyc0, _sel)
                                    _yl1, _yh1 = _a0_pair_f32(_pya1, _pyc1, _sel)
                                    _yl2, _yh2 = _a0_pair_f32(_pya2, _pyc2, _sel)
                                    _yl3, _yh3 = _a0_pair_f32(_pya3, _pyc3, _sel)
                                    _tp0 = _tp0 + _yl0 + _yh0
                                    _tp1 = _tp1 + _yl1 + _yh1
                                    _tp2 = _tp2 + _yl2 + _yh2
                                    _tp3 = _tp3 + _yl3 + _yh3
                                else:
                                    for _pyax, _pycx in ((_pya0, _pyc0), (_pya1, _pyc1), (_pya2, _pyc2), (_pya3, _pyc3)):
                                        _ylx, _yhx = _a0_pair_f32(_pyax, _pycx, _sel)
                                        _sum0 = _sum0 + _ylx + _yhx
                                _pya0, _pyc0, _pya1, _pyc1, _pya2, _pyc2, _pya3, _pyc3 = _nya0, _nyc0, _nya1, _nyc1, _nya2, _nyc2, _nya3, _nyc3
                        if cutlass.const_expr(_LN_TREE):
                            _sum0 = (_tp0 + _tp1) + (_tp2 + _tp3)
                    elif cutlass.const_expr(_LN_SMEM):
                        # LN PASS FORMS: operand 0 by 8-channel chunks q (a runtime loop in the rolled form) x channel pairs e -- pair (q * 8 + 2 e)'s
                        # XOR offsets are the constants xo[2 e] / xo[2 e + 1] either way; the pair order equals the register form's -> bitwise
                        for _j in cutlass.range_constexpr(_NK):
                            _wb = (sA_elems // 2) * _sts[_j]
                            if cutlass.const_expr(_LN_ROLLED):
                                for _q in cutlass.range(cta_tile_mnk[2] // 8, unroll=1):  # unroll=1 = NO unrolling (the rolled form's point)
                                    for _e in cutlass.range_constexpr(4):
                                        _oaca = _q * 8 + 2 * _e
                                        _oawa = _a0w.data_ptr(_wb + _xow[2 * _e] + _oaca * 32).load()
                                        _oawc = _a0w.data_ptr(_wb + _xow[2 * _e + 1] + (_oaca + 1) * 32).load()
                                        _oaw0 = nvvm.inline_ptx(
                                            "prmt.b32 $0, $1, $2, $3;", write_only_types=[cutlass.Int32], read_only_args=[_oawa, _oawc, _sel]
                                        )
                                        _oalo, _oahi = f16x2_to_f32(_oaw0, dtype=cutlass.BFloat16)
                                        _sum0 = _sum0 + _oalo + _oahi
                            else:
                                for _q in cutlass.range_constexpr(cta_tile_mnk[2] // 8):
                                    for _e in cutlass.range_constexpr(4):
                                        _ca = _q * 8 + 2 * _e
                                        _wa = _a0w.data_ptr(_wb + _xow[2 * _e] + _ca * 32).load()
                                        _wc = _a0w.data_ptr(_wb + _xow[2 * _e + 1] + (_ca + 1) * 32).load()
                                        _w0 = nvvm.inline_ptx("prmt.b32 $0, $1, $2, $3;", write_only_types=[cutlass.Int32], read_only_args=[_wa, _wc, _sel])
                                        _lo, _hi = f16x2_to_f32(_w0, dtype=cutlass.BFloat16)
                                        _sum0 = _sum0 + _lo + _hi
                    else:
                        for _j in cutlass.range_constexpr(_NK):
                            _wb = (sA_elems // 2) * _sts[_j]
                            for _p in cutlass.range_constexpr(cta_tile_mnk[2] // 2):
                                _ca = 2 * _p
                                _wa = _a0w.data_ptr(_wb + _xow[_ca & 7] + _ca * 32).load()
                                _wc = _a0w.data_ptr(_wb + _xow[(_ca + 1) & 7] + (_ca + 1) * 32).load()
                                _w0 = nvvm.inline_ptx("prmt.b32 $0, $1, $2, $3;", write_only_types=[cutlass.Int32], read_only_args=[_wa, _wc, _sel])
                                _row_words0.append(_w0)
                                _lo, _hi = f16x2_to_f32(_w0, dtype=cutlass.BFloat16)
                                _sum0 = _sum0 + _lo + _hi
                    _mean0 = _sum0 * cutlass.Float32(_INV_K)
                    _sq0 = cutlass.Float32(0.0)
                    if cutlass.const_expr(_LN_PF):
                        _tq0 = cutlass.Float32(0.0)
                        _tq1 = cutlass.Float32(0.0)
                        _tq2 = cutlass.Float32(0.0)
                        _tq3 = cutlass.Float32(0.0)
                        for _j in cutlass.range_constexpr(_NK):
                            _wb = (sA_elems // 2) * _sts[_j]
                            _pza0, _pzc0, _pza1, _pzc1, _pza2, _pzc2, _pza3, _pzc3 = _a0_chunk_words(_a0w, _wb, _xow, 0)
                            for _q in cutlass.range(cta_tile_mnk[2] // 8, unroll=1):  # unroll=1 = NO unrolling (the rolled form's point)
                                _nqz = (_q + 1) & (cta_tile_mnk[2] // 8 - 1)
                                _nza0, _nzc0, _nza1, _nzc1, _nza2, _nzc2, _nza3, _nzc3 = _a0_chunk_words(_a0w, _wb, _xow, _nqz)
                                if cutlass.const_expr(_LN_TREE):
                                    _zl0, _zh0 = _a0_pair_f32(_pza0, _pzc0, _sel)
                                    _zl1, _zh1 = _a0_pair_f32(_pza1, _pzc1, _sel)
                                    _zl2, _zh2 = _a0_pair_f32(_pza2, _pzc2, _sel)
                                    _zl3, _zh3 = _a0_pair_f32(_pza3, _pzc3, _sel)
                                    _zdl0 = _zl0 - _mean0
                                    _zdh0 = _zh0 - _mean0
                                    _zdl1 = _zl1 - _mean0
                                    _zdh1 = _zh1 - _mean0
                                    _zdl2 = _zl2 - _mean0
                                    _zdh2 = _zh2 - _mean0
                                    _zdl3 = _zl3 - _mean0
                                    _zdh3 = _zh3 - _mean0
                                    _tq0 = _tq0 + _zdl0 * _zdl0 + _zdh0 * _zdh0
                                    _tq1 = _tq1 + _zdl1 * _zdl1 + _zdh1 * _zdh1
                                    _tq2 = _tq2 + _zdl2 * _zdl2 + _zdh2 * _zdh2
                                    _tq3 = _tq3 + _zdl3 * _zdl3 + _zdh3 * _zdh3
                                else:
                                    for _pzax, _pzcx in ((_pza0, _pzc0), (_pza1, _pzc1), (_pza2, _pzc2), (_pza3, _pzc3)):
                                        _zlx, _zhx = _a0_pair_f32(_pzax, _pzcx, _sel)
                                        _zdlx = _zlx - _mean0
                                        _zdhx = _zhx - _mean0
                                        _sq0 = _sq0 + _zdlx * _zdlx + _zdhx * _zdhx
                                _pza0, _pzc0, _pza1, _pzc1, _pza2, _pzc2, _pza3, _pzc3 = _nza0, _nzc0, _nza1, _nzc1, _nza2, _nzc2, _nza3, _nzc3
                        if cutlass.const_expr(_LN_TREE):
                            _sq0 = (_tq0 + _tq1) + (_tq2 + _tq3)
                    elif cutlass.const_expr(_LN_SMEM):
                        for _j in cutlass.range_constexpr(_NK):
                            _wb = (sA_elems // 2) * _sts[_j]
                            if cutlass.const_expr(_LN_ROLLED):
                                for _q in cutlass.range(cta_tile_mnk[2] // 8, unroll=1):  # unroll=1 = NO unrolling (the rolled form's point)
                                    for _e in cutlass.range_constexpr(4):
                                        _obca = _q * 8 + 2 * _e
                                        _obwa = _a0w.data_ptr(_wb + _xow[2 * _e] + _obca * 32).load()
                                        _obwc = _a0w.data_ptr(_wb + _xow[2 * _e + 1] + (_obca + 1) * 32).load()
                                        _obw0 = nvvm.inline_ptx(
                                            "prmt.b32 $0, $1, $2, $3;", write_only_types=[cutlass.Int32], read_only_args=[_obwa, _obwc, _sel]
                                        )
                                        _oblo, _obhi = f16x2_to_f32(_obw0, dtype=cutlass.BFloat16)
                                        _obdl = _oblo - _mean0
                                        _obdh = _obhi - _mean0
                                        _sq0 = _sq0 + _obdl * _obdl + _obdh * _obdh
                            else:
                                for _q in cutlass.range_constexpr(cta_tile_mnk[2] // 8):
                                    for _e in cutlass.range_constexpr(4):
                                        _ca = _q * 8 + 2 * _e
                                        _wa = _a0w.data_ptr(_wb + _xow[2 * _e] + _ca * 32).load()
                                        _wc = _a0w.data_ptr(_wb + _xow[2 * _e + 1] + (_ca + 1) * 32).load()
                                        _w0 = nvvm.inline_ptx("prmt.b32 $0, $1, $2, $3;", write_only_types=[cutlass.Int32], read_only_args=[_wa, _wc, _sel])
                                        _lo, _hi = f16x2_to_f32(_w0, dtype=cutlass.BFloat16)
                                        _dl = _lo - _mean0
                                        _dh = _hi - _mean0
                                        _sq0 = _sq0 + _dl * _dl + _dh * _dh
                    else:
                        for _w0 in _row_words0:
                            _lo, _hi = f16x2_to_f32(_w0, dtype=cutlass.BFloat16)
                            _dl = _lo - _mean0
                            _dh = _hi - _mean0
                            _sq0 = _sq0 + _dl * _dl + _dh * _dh
                    _rstd0 = cute.math.rsqrt(_sq0 * cutlass.Float32(_INV_K) + cutlass.Float32(_EPS), fastmath=True)
                    if cutlass.const_expr(WANT_STATS):
                        if row_valid & (tile_n == 0):
                            _stats_addr0 = stats.iterator.toint() + (bi_idx * n_res + k_idx).to(cutlass.Int64) * cutlass.Int64(8)
                            st_global_v2(_stats_addr0, [_mean0, _rstd0], cutlass.Float32)
                    if cutlass.const_expr(_A0_TRANSPOSE):
                        nvvm.barrier_cta_sync(barrier_id=LN_SYNC_BAR_ID, thread_count=num_ln_warps * 32)
                    _wi = 0
                    for _j in cutlass.range_constexpr(_NK if _DIAG != "no_ln_norm" else 0):
                        _row_base0 = sA_elems * _sts[_j] + ln_tid * cta_tile_mnk[2]
                        _wb = (sA_elems // 2) * _sts[_j]
                        if cutlass.const_expr(_LN_SMEM and _LN_ROLLED):
                            # the re-read form, rolled ("mn_major" only -- the host declines the transpose form under the re-read forms): per chunk the
                            # gamma / beta broadcast loads, then per pair the two b32 re-reads + prmt, normalise, the in-place b16 stores of the element this thread owns (row j)
                            for _q in cutlass.range(cta_tile_mnk[2] // 8, unroll=1):  # unroll=1 = NO unrolling (the rolled form's point)
                                _occg = _j * cta_tile_mnk[2] + _q * 8
                                if cutlass.const_expr(_AFFINE_FOLD):
                                    _hg0 = _hg1 = _hg2 = _hg3 = _hg4 = _hg5 = _hg6 = _hg7 = _rstd0  # unused under the fold (the pure (x - mean) * rstd)
                                    _hb0 = _hb1 = _hb2 = _hb3 = _hb4 = _hb5 = _hb6 = _hb7 = _rstd0
                                else:
                                    _hg0, _hg1, _hg2, _hg3 = ld_shared_v4(smem_ln_params.data_ptr(_occg), cutlass.Float32)
                                    _hg4, _hg5, _hg6, _hg7 = ld_shared_v4(smem_ln_params.data_ptr(_occg + 4), cutlass.Float32)
                                    _hb0, _hb1, _hb2, _hb3 = ld_shared_v4(smem_ln_params.data_ptr(_K + _occg), cutlass.Float32)
                                    _hb4, _hb5, _hb6, _hb7 = ld_shared_v4(smem_ln_params.data_ptr(_K + _occg + 4), cutlass.Float32)
                                for _e in cutlass.range_constexpr(4):
                                    _occl = _q * 8 + 2 * _e
                                    _ocwa = _a0w.data_ptr(_wb + _xow[2 * _e] + _occl * 32).load()
                                    _ocwc = _a0w.data_ptr(_wb + _xow[2 * _e + 1] + (_occl + 1) * 32).load()
                                    _ocw0 = nvvm.inline_ptx("prmt.b32 $0, $1, $2, $3;", write_only_types=[cutlass.Int32], read_only_args=[_ocwa, _ocwc, _sel])
                                    _oclo, _ochi = f16x2_to_f32(_ocw0, dtype=cutlass.BFloat16)
                                    if cutlass.const_expr(_AFFINE_FOLD):
                                        _ocylo = (_oclo - _mean0) * _rstd0
                                        _ocyhi = (_ochi - _mean0) * _rstd0
                                    else:
                                        _ocgl = (_hg0, _hg1, _hg2, _hg3, _hg4, _hg5, _hg6, _hg7)[2 * _e]
                                        _ocgh = (_hg0, _hg1, _hg2, _hg3, _hg4, _hg5, _hg6, _hg7)[2 * _e + 1]
                                        _ocbl = (_hb0, _hb1, _hb2, _hb3, _hb4, _hb5, _hb6, _hb7)[2 * _e]
                                        _ocbh = (_hb0, _hb1, _hb2, _hb3, _hb4, _hb5, _hb6, _hb7)[2 * _e + 1]
                                        _ocylo = (_oclo - _mean0) * _rstd0 * _ocgl + _ocbl
                                        _ocyhi = (_ochi - _mean0) * _rstd0 * _ocgh + _ocbh
                                    smem_a_list[0].data_ptr(sA_elems * _sts[_j] + _xoe[2 * _e] + _occl * cta_tile_mnk[2]).store(_ocylo.to(cutlass.BFloat16))
                                    smem_a_list[0].data_ptr(sA_elems * _sts[_j] + _xoe[2 * _e + 1] + (_occl + 1) * cta_tile_mnk[2]).store(
                                        _ocyhi.to(cutlass.BFloat16)
                                    )
                        elif cutlass.const_expr(_LN_PF):
                            # the pipelined rolled form on operand 0 ("mn_major" only): chunk (q + 1) & 7's eight channel words and (outside the affine fold)
                            # its 16 gamma / beta load while chunk q's four pairs are normalised and stored in place as b16 -- the rolled form's arithmetic
                            if cutlass.const_expr(_AFFINE_FOLD):
                                _pyg0 = _pyg1 = _pyg2 = _pyg3 = _pyg4 = _pyg5 = _pyg6 = _pyg7 = _rstd0  # unused under the fold (the pure (x - mean) * rstd)
                                _pyb0 = _pyb1 = _pyb2 = _pyb3 = _pyb4 = _pyb5 = _pyb6 = _pyb7 = _rstd0
                            else:
                                _pyg0, _pyg1, _pyg2, _pyg3 = ld_shared_v4(smem_ln_params.data_ptr(_j * cta_tile_mnk[2]), cutlass.Float32)
                                _pyg4, _pyg5, _pyg6, _pyg7 = ld_shared_v4(smem_ln_params.data_ptr(_j * cta_tile_mnk[2] + 4), cutlass.Float32)
                                _pyb0, _pyb1, _pyb2, _pyb3 = ld_shared_v4(smem_ln_params.data_ptr(_K + _j * cta_tile_mnk[2]), cutlass.Float32)
                                _pyb4, _pyb5, _pyb6, _pyb7 = ld_shared_v4(smem_ln_params.data_ptr(_K + _j * cta_tile_mnk[2] + 4), cutlass.Float32)
                            _pva0, _pvc0, _pva1, _pvc1, _pva2, _pvc2, _pva3, _pvc3 = _a0_chunk_words(_a0w, _wb, _xow, 0)
                            for _q in cutlass.range(cta_tile_mnk[2] // 8, unroll=1):  # unroll=1 = NO unrolling (the rolled form's point)
                                _nqv = (_q + 1) & (cta_tile_mnk[2] // 8 - 1)
                                if cutlass.const_expr(_AFFINE_FOLD):
                                    _nyg0 = _nyg1 = _nyg2 = _nyg3 = _nyg4 = _nyg5 = _nyg6 = _nyg7 = _rstd0
                                    _nyb0 = _nyb1 = _nyb2 = _nyb3 = _nyb4 = _nyb5 = _nyb6 = _nyb7 = _rstd0
                                else:
                                    _nccg = _j * cta_tile_mnk[2] + _nqv * 8
                                    _nyg0, _nyg1, _nyg2, _nyg3 = ld_shared_v4(smem_ln_params.data_ptr(_nccg), cutlass.Float32)
                                    _nyg4, _nyg5, _nyg6, _nyg7 = ld_shared_v4(smem_ln_params.data_ptr(_nccg + 4), cutlass.Float32)
                                    _nyb0, _nyb1, _nyb2, _nyb3 = ld_shared_v4(smem_ln_params.data_ptr(_K + _nccg), cutlass.Float32)
                                    _nyb4, _nyb5, _nyb6, _nyb7 = ld_shared_v4(smem_ln_params.data_ptr(_K + _nccg + 4), cutlass.Float32)
                                _nva0, _nvc0, _nva1, _nvc1, _nva2, _nvc2, _nva3, _nvc3 = _a0_chunk_words(_a0w, _wb, _xow, _nqv)
                                for _e in cutlass.range_constexpr(4):
                                    _pvcl = _q * 8 + 2 * _e
                                    _pvlo, _pvhi = _a0_pair_f32((_pva0, _pva1, _pva2, _pva3)[_e], (_pvc0, _pvc1, _pvc2, _pvc3)[_e], _sel)
                                    if cutlass.const_expr(_AFFINE_FOLD):
                                        _pvylo = (_pvlo - _mean0) * _rstd0
                                        _pvyhi = (_pvhi - _mean0) * _rstd0
                                    else:
                                        _pvgl = (_pyg0, _pyg1, _pyg2, _pyg3, _pyg4, _pyg5, _pyg6, _pyg7)[2 * _e]
                                        _pvgh = (_pyg0, _pyg1, _pyg2, _pyg3, _pyg4, _pyg5, _pyg6, _pyg7)[2 * _e + 1]
                                        _pvbl = (_pyb0, _pyb1, _pyb2, _pyb3, _pyb4, _pyb5, _pyb6, _pyb7)[2 * _e]
                                        _pvbh = (_pyb0, _pyb1, _pyb2, _pyb3, _pyb4, _pyb5, _pyb6, _pyb7)[2 * _e + 1]
                                        _pvylo = (_pvlo - _mean0) * _rstd0 * _pvgl + _pvbl
                                        _pvyhi = (_pvhi - _mean0) * _rstd0 * _pvgh + _pvbh
                                    smem_a_list[0].data_ptr(sA_elems * _sts[_j] + _xoe[2 * _e] + _pvcl * cta_tile_mnk[2]).store(_pvylo.to(cutlass.BFloat16))
                                    smem_a_list[0].data_ptr(sA_elems * _sts[_j] + _xoe[2 * _e + 1] + (_pvcl + 1) * cta_tile_mnk[2]).store(
                                        _pvyhi.to(cutlass.BFloat16)
                                    )
                                _pyg0, _pyg1, _pyg2, _pyg3, _pyg4, _pyg5, _pyg6, _pyg7 = _nyg0, _nyg1, _nyg2, _nyg3, _nyg4, _nyg5, _nyg6, _nyg7
                                _pyb0, _pyb1, _pyb2, _pyb3, _pyb4, _pyb5, _pyb6, _pyb7 = _nyb0, _nyb1, _nyb2, _nyb3, _nyb4, _nyb5, _nyb6, _nyb7
                                _pva0, _pvc0, _pva1, _pvc1, _pva2, _pvc2, _pva3, _pvc3 = _nva0, _nvc0, _nva1, _nvc1, _nva2, _nvc2, _nva3, _nvc3
                        else:
                            for _q in cutlass.range_constexpr(cta_tile_mnk[2] // 8):
                                _cg = _j * cta_tile_mnk[2] + _q * 8  # the global channel of this 8-chunk (operand 0's gamma / beta sit at [0, 2 K))
                                _gam = []
                                _bet = []
                                if cutlass.const_expr(not _AFFINE_FOLD):
                                    _gam.extend(ld_shared_v4(smem_ln_params.data_ptr(_cg), cutlass.Float32))
                                    _gam.extend(ld_shared_v4(smem_ln_params.data_ptr(_cg + 4), cutlass.Float32))
                                    _bet.extend(ld_shared_v4(smem_ln_params.data_ptr(_K + _cg), cutlass.Float32))
                                    _bet.extend(ld_shared_v4(smem_ln_params.data_ptr(_K + _cg + 4), cutlass.Float32))
                                _packed = []
                                for _e in cutlass.range_constexpr(4):
                                    _cl = _q * 8 + 2 * _e  # the stage-local channel of _lo; _hi is the next channel
                                    if cutlass.const_expr(_LN_SMEM):
                                        _wa = _a0w.data_ptr(_wb + _xow[2 * _e] + _cl * 32).load()  # the re-read form, unrolled
                                        _wc = _a0w.data_ptr(_wb + _xow[2 * _e + 1] + (_cl + 1) * 32).load()
                                        _w0 = nvvm.inline_ptx("prmt.b32 $0, $1, $2, $3;", write_only_types=[cutlass.Int32], read_only_args=[_wa, _wc, _sel])
                                    else:
                                        _w0 = _row_words0[_wi + _e]
                                    _lo, _hi = f16x2_to_f32(_w0, dtype=cutlass.BFloat16)
                                    if cutlass.const_expr(_AFFINE_FOLD):
                                        _ylo = (_lo - _mean0) * _rstd0
                                        _yhi = (_hi - _mean0) * _rstd0
                                    else:
                                        _ylo = (_lo - _mean0) * _rstd0 * _gam[2 * _e] + _bet[2 * _e]
                                        _yhi = (_hi - _mean0) * _rstd0 * _gam[2 * _e + 1] + _bet[2 * _e + 1]
                                    if cutlass.const_expr(_A0_TRANSPOSE):
                                        _packed.append(fp32_to_fp16(_ylo, _yhi, dtype=cutlass.BFloat16))
                                    else:
                                        smem_a_list[0].data_ptr(sA_elems * _sts[_j] + _xoe[_cl & 7] + _cl * cta_tile_mnk[2]).store(_ylo.to(cutlass.BFloat16))
                                        smem_a_list[0].data_ptr(sA_elems * _sts[_j] + _xoe[(_cl + 1) & 7] + (_cl + 1) * cta_tile_mnk[2]).store(
                                            _yhi.to(cutlass.BFloat16)
                                        )
                                _wi += 4
                                if cutlass.const_expr(_A0_TRANSPOSE):
                                    st_shared_v4(smem_a_list[0].data_ptr(_row_base0 + (_q ^ ln_swz) * 8), _packed, cutlass.Int32)
            # (6) the pair-level mask vote (leader CTA): are ALL 256 rows of the pair tile masked or padded?  Each lane reads
            # its own row's mask and the follower's row with the same index; a nonzero anywhere keeps the MMA.
            if cutlass.const_expr(SKIP_MASKED_TILES):
                if is_pair_leader:
                    _m_self = _row_mask(mask, row_valid, bi_idx, k_idx, n_res)
                    _k2, _bi2 = _row_coords(tile_m, n_kb, lkb, r_pair ^ cta_tile_mnk[0])
                    _valid2 = (_k2 < n_res) & (_bi2 < bn)
                    _m_peer = _row_mask(mask, _valid2, _bi2, _k2, n_res)
                    _ballot = cute.arch.vote_ballot_sync((_m_self != cutlass.Float32(0.0)) | (_m_peer != cutlass.Float32(0.0)))
                    _vslot = (tile_iter & 1) * 4 + ln_warp
                    if ln_lane == 0:
                        smem_vote.data_ptr(_vslot).store(cutlass.Int32(_ballot))
                    nvvm.barrier_cta_sync(barrier_id=LN_SYNC_BAR_ID, thread_count=num_ln_warps * 32)
                    _vbase = (tile_iter & 1) * 4
                    _any = (
                        smem_vote.data_ptr(_vbase).load()
                        | smem_vote.data_ptr(_vbase + 1).load()
                        | smem_vote.data_ptr(_vbase + 2).load()
                        | smem_vote.data_ptr(_vbase + 3).load()
                    )
                    if ln_warp == 0:
                        if elect_one:  # the arriving lane stores the flag: its release covers its own store directly
                            smem_flags.data_ptr(_sts[0]).store(
                                cutlass.Int32(arith.select((_any == 0).ir_value(), cutlass.Int32(1).ir_value(), cutlass.Int32(0).ir_value()))
                            )
            # (7) publish: every lane fences its generic SMEM stores to the async proxy (the MMA reads the stage through a descriptor);
            # bar.warp.sync then orders ALL 32 lanes' in-place stores (and warp 0's flag store) before the ONE arriving lane -- a
            # release arrive covers the arriving thread's own stores plus what it has observed, and PTX does not say which lane
            # elect_sync picks (P11: the arrive depends on multi-lane data; the same fence -> warp sync -> elected arrive shape as every
            # other publish in this file); then the release arrive on the LEADER's a_norm_full for every stage of the tile.
            cute.arch.fence_proxy("async.shared", space="cta")
            nvvm.bar_warp_sync(0xFFFFFFFF)
            for _j in cutlass.range_constexpr(_NK):
                if elect_one:
                    arrive_on_leader_release(a_norm_full_mbar_ptr.subview(_sts[_j]), pair_leader_rank, cta_group)
            # (8) consume the CLC response (the same protocol as every consumer warp)
            consumer_stage = tile_iter % CLC_SCHED_STAGES
            if consumer_stage == 0 and tile_iter != 0:
                clc_full_phase_ln = clc_full_phase_ln ^ 1
            while not nvvm.mbarrier_try_wait_parity(
                clc_full_mbar_ptr.subview(consumer_stage),
                clc_full_phase_ln,
                time_limit=10_000_000,
            ):
                pass
            m_idx, n_idx, l_idx, vld = cute_clc.clc_response(clc_response_ptr_base + consumer_stage)
            cute.arch.fence_proxy("async.shared", space="cta")
            is_valid = vld
            ln_raw_m = m_idx >> _preferred_cluster_m_shift
            ln_raw_n = n_idx >> _preferred_cluster_n_shift
            ln_nt_m = gridx >> _preferred_cluster_m_shift
            ln_nt_n = gridy >> _preferred_cluster_n_shift
            tile_m, tile_n = _l2_swizzle_tile(
                ln_raw_m,
                ln_raw_n,
                ln_nt_m,
                ln_nt_n,
                swizzle_w,
                identity=tile_swizzle_n == 1,
            )
            tile_l = l_idx
            tile_m, tile_l = _decode_pass(tile_m, tile_l)  # TILE ORDER (module docstring)
            nvvm.bar_warp_sync(0xFFFFFFFF)
            if elect_one:
                empty_remote = nvvm.mapa(clc_empty_mbar_ptr.subview(consumer_stage), 0)
                nvvm.mbarrier_arrive(empty_remote, scope=nvvm.MemScope.CLUSTER, relaxed=True)
            tile_iter += 1


frost_ln_dual_gemm_128x256x128_cluster2x1_2ctamma.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _host(
    geom: tuple,
    a0: cute.Tensor,  # operand 0: K1 = z [bn * n_res * K] bf16 (pair-major rows); K3 = y [c * bn * n_pad] bf16 (channel-major, MN-major rows)
    a1: Optional[cute.Tensor],  # operand 1 (K3): z [bn * n_res * K] bf16; None in the single-operand arm
    w0: cute.Tensor,  # K1: [passes * 2 c_out * K] (per pass: rows interleaved per 128-channel block); K3: W_value [c_out * K]
    w1: Optional[cute.Tensor],  # K3: W_gate [c_out * K]; None in the single-operand arm
    out: cute.Tensor,  # K1: [passes * c_out * bn * n_pad] channel-major planes; K3: [bn * n_res * c_out] pair-major
    ln_w: cute.Tensor,
    ln_b: cute.Tensor,
    ln_w1: Optional[cute.Tensor],
    ln_b1: Optional[cute.Tensor],
    bias: Optional[cute.Tensor],
    mask: Optional[cute.Tensor],
    stats: Optional[cute.Tensor],
    resid: Optional[cute.Tensor],
    stream: _cuda.CUstream,
) -> None:
    m_rows = geom[0]  # the caller's flat M: pair tiles x 256
    n = geom[1]  # the GEMM N: K1 = 2 * c_out (value | gate interleaved), K3 = c_out per GEMM
    k_sym = geom[2]  # K
    batch = geom[3]  # passes
    # TILE ORDER (module docstring): "interleaved" folds the pass into the scheduler's M axis (passes tiles per row block, grid z = 1)
    m = (m_rows * _PASSES) if cutlass.const_expr(_PASS_IN_M) else m_rows
    grid_z = 1 if cutlass.const_expr(_PASS_IN_M or _LN_ONCE) else batch  # LN ONCE: the pass lives inside the work item
    n_res = cutlass.Int32(geom[4])
    bn = cutlass.Int32(geom[5])
    n_pad = cutlass.Int32(geom[6])
    lkb = cutlass.Int32(geom[7])
    n_kb = cutlass.Int32(geom[8])
    kbc = cutlass.Int32(geom[9])  # k's per CTA box (min(kb, 128))
    rc = cutlass.Int32(geom[10])  # bi's per CTA box (128 // kbc)
    bpe = ab_dtype.width // 8

    if cutlass.const_expr(_DUAL):
        # K3 operand 0: y viewed [c][bn][n_pad] -- dims innermost first (j, bi, channel) with the j extent n_res (j in [n_res, n_pad) is never
        # read: zero-filled); the (64 j, 1 bi, 64 c) box = one 64-row MN group of the SWIZZLE_128B stage, issued twice per stage.
        tma_a_desc_0 = _tma.create_tensor_map_tiled(
            global_address=a0.iterator.toint(),
            dtype=ab_tma_dtype,
            global_dims=[n_res, bn, k_sym],
            global_strides=[n_pad.to(cutlass.Int64) * bpe // 16, bn.to(cutlass.Int64) * n_pad.to(cutlass.Int64) * bpe // 16],
            box_dims=[_MN_GROUP, 1, cta_tile_mnk[2]],
            swizzle=ab_tma_swizzle,
        )
        # K3 operand 1: z viewed [bn][n_res][K] (channel, k, bi) -- the K-major box (64, kbc, rc) lands as the 128 x 64 stage (K1's A).
        tma_a_desc_1 = _tma.create_tensor_map_tiled(
            global_address=a1.iterator.toint(),
            dtype=ab_tma_dtype,
            global_dims=[k_sym, n_res, bn],
            global_strides=[k_sym * bpe // 16, n_res.to(cutlass.Int64) * k_sym * bpe // 16],
            box_dims=[cta_tile_mnk[2], kbc, rc],
            swizzle=ab_tma_swizzle,
        )
    else:
        # A: z viewed [bn][n_res][K] -- dims innermost first (channel, k, bi); the CTA box (64, kbc, rc) lands as the 128 x 64 stage.
        tma_a_desc_0 = _tma.create_tensor_map_tiled(
            global_address=a0.iterator.toint(),
            dtype=ab_tma_dtype,
            global_dims=[k_sym, n_res, bn],
            global_strides=[k_sym * bpe // 16, n_res.to(cutlass.Int64) * k_sym * bpe // 16],
            box_dims=[cta_tile_mnk[2], kbc, rc],  # dynamic box dims: the builder takes them as Int32 (an Int8 cannot hold 128)
            swizzle=ab_tma_swizzle,
        )
        tma_a_desc_1 = tma_a_desc_0  # unused by the single-operand arm (the kernel signature is shared)
    # B: K1 = the stacked weights [passes][2 c_out][K], K-major, each CTA loads its 128-row half of the 256-wide N tile; K3 = W_value and
    # W_gate [c_out][K], each CTA loads its 64-row half of the 128-wide N tile of BOTH.
    tma_b_desc_0 = _tma.create_tensor_map_tiled(
        global_address=w0.iterator.toint(),
        dtype=ab_tma_dtype,
        global_dims=[k_sym, n, batch],
        global_strides=[k_sym * bpe // 16, n * k_sym * bpe // 16],
        box_dims=[cta_tile_mnk[2], cta_tile_mnk[1] // b_mcast_slices, 1],
        swizzle=ab_tma_swizzle,
    )
    if cutlass.const_expr(_DUAL):
        tma_b_desc_1 = _tma.create_tensor_map_tiled(
            global_address=w1.iterator.toint(),
            dtype=ab_tma_dtype,
            global_dims=[k_sym, n, batch],
            global_strides=[k_sym * bpe // 16, n * k_sym * bpe // 16],
            box_dims=[cta_tile_mnk[2], cta_tile_mnk[1] // b_mcast_slices, 1],
            swizzle=ab_tma_swizzle,
        )
    else:
        tma_b_desc_1 = tma_b_desc_0
    if cutlass.const_expr(_DUAL):
        # C (K3): out viewed [bn][n_res][c_out] -- dims (channel, k, bi); the s64b box (32, kbc, rc) from the [128 rows][32 ch] swizzled staging.
        tma_c_desc_0 = _tma.create_tensor_map_tiled(
            global_address=out.iterator.toint(),
            dtype=cutlass.BFloat16,
            global_dims=[n, n_res, bn],
            global_strides=[n * bpe // 16, n_res.to(cutlass.Int64) * n * bpe // 16],
            box_dims=[epi_n, kbc, rc],
            swizzle=_tma.TensorMapSwizzle.s64b,
        )
    else:
        # C: out viewed [passes][c_out][bn][n_pad] -- dims (k, bi, channel, pass); the staging buffer is [32 channels][rc][kbc] dense.
        c_out_i64 = n // 2
        tma_c_desc_0 = _tma.create_tensor_map_tiled(
            global_address=out.iterator.toint(),
            dtype=cutlass.BFloat16,
            global_dims=[n_pad, bn, cutlass.Int32(c_out_i64), cutlass.Int32(batch)],
            global_strides=[
                n_pad.to(cutlass.Int64) * bpe // 16,
                bn.to(cutlass.Int64) * n_pad.to(cutlass.Int64) * bpe // 16,
                c_out_i64 * bn.to(cutlass.Int64) * n_pad.to(cutlass.Int64) * bpe // 16,
            ],
            box_dims=[kbc, rc, epi_n, 1],
            swizzle=_tma.TensorMapSwizzle.none,
        )

    cluster_m = cluster_shape_mnk[0]
    cluster_n = cluster_shape_mnk[1]
    cgrp_tile_m = cgrp_tile_mnk[0]
    cgrp_tile_n = cgrp_tile_mnk[1]
    num_tile_m_host = (m + cgrp_tile_m - 1) // cgrp_tile_m
    num_tile_n_host = (n + cgrp_tile_n - 1) // cgrp_tile_n
    if cutlass.const_expr(_LN_ONCE):
        num_tile_n_host = num_tile_n_host // _NTI  # LN ONCE: one scheduler item per (row block, N-tile group)
    grid_x = num_tile_m_host * cluster_m
    grid_y = num_tile_n_host * cluster_n
    grid_shape = (grid_x, grid_y, grid_z)
    launch = frost_ln_dual_gemm_128x256x128_cluster2x1_2ctamma(
        m,
        n,
        k_sym,
        tma_a_desc_0,
        tma_a_desc_1,
        tma_b_desc_0,
        tma_b_desc_1,
        tma_c_desc_0,
        n_res,
        bn,
        n_pad,
        lkb,
        n_kb,
        ln_w,
        ln_b,
        ln_w1,
        ln_b1,
        bias,
        mask,
        stats,
        resid,
    )
    launch.launch(
        grid=grid_shape,
        block=(threads_per_cta, 1, 1),
        cluster=cluster_shape_mnk,
        use_pdl=USE_PDL,
        min_blocks_per_mp=_MIN_BLOCKS_PER_MP,  # the `.minnctapersm` hint (REGISTER SPLIT, module docstring); 0 = no attribute = today's IR
        stream=stream,
    )


@lru_cache(maxsize=None)
def compile() -> Callable:
    """cute-compile ``_host`` for this module's PARAMS.  Shapes are symbolic; the geometry tuple is
    ``(m_sched, gemm_n, K, passes, n_res, bn, n_pad, lkb, n_kb, kbc, rc)`` -- the host module builds it."""
    sym = [cute.sym_int64() for _ in range(11)]
    flat = lambda dt, align: make_fake_compact_tensor(dt, (cute.sym_int64(),), stride_order=(0,), assumed_align=align)  # noqa: E731
    fake_a0 = flat(cutlass.BFloat16, 16)
    fake_a1 = flat(cutlass.BFloat16, 16) if _DUAL else None
    fake_w0 = flat(cutlass.BFloat16, 16)
    fake_w1 = flat(cutlass.BFloat16, 16) if _DUAL else None
    fake_out = flat(cutlass.BFloat16, 16)
    fake_ln_w = make_fake_compact_tensor(cutlass.BFloat16, (_K,), stride_order=(0,), assumed_align=16)
    fake_ln_b = make_fake_compact_tensor(cutlass.BFloat16, (_K,), stride_order=(0,), assumed_align=16)
    fake_ln_w1 = make_fake_compact_tensor(cutlass.BFloat16, (_K,), stride_order=(0,), assumed_align=16) if _DUAL else None
    fake_ln_b1 = make_fake_compact_tensor(cutlass.BFloat16, (_K,), stride_order=(0,), assumed_align=16) if _DUAL else None
    fake_bias = make_fake_compact_tensor(cutlass.BFloat16, (_PASSES * _BIAS_PER_PASS,), stride_order=(0,), assumed_align=16) if HAS_BIAS else None
    fake_mask = flat(cutlass.BFloat16, 16) if HAS_MASK else None
    fake_stats = flat(cutlass.Float32, 16) if WANT_STATS else None
    fake_resid = flat(cutlass.BFloat16, 16) if _FUSE_RESIDUAL else None
    _fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)
    return cute.compile(
        _host,
        tuple(sym),
        fake_a0,
        fake_a1,
        fake_w0,
        fake_w1,
        fake_out,
        fake_ln_w,
        fake_ln_b,
        fake_ln_w1,
        fake_ln_b1,
        fake_bias,
        fake_mask,
        fake_stats,
        fake_resid,
        stream=_fake_stream,
        options=frost_compile_options,
    )
