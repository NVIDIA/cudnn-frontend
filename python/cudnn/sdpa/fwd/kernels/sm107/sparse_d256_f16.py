# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Rubin (SM107) index-list SPARSE prefill SDPA forward -- d_qk = d_v = 256, FP16/BF16, swap-AB, gathered K/V.

HEADER FIRST: this module carries its pipeline tables (warp map, barrier table with the issuing-lane ledger, SMEM table with
swizzle + why, TMEM map, the one-helper invariant, the degenerate-input matrix) ahead of the body, so a reader checks the
body against them instead of reconstructing the pipeline from the code.  The body is built from the SM100 swap-AB decode
tile (``sm100/decode_d256_f16.py``: the MMA descriptors, the softmax orientation, the epilogue) plus the ``tma_gather4``
loader; the config twin is ``config_sm107.make_cfg_d256_sparse``.

WHAT IT COMPUTES.  For query row ``r`` of sequence ``b`` at position ``p`` with ``n_vis = min(p + 1, L_b)`` visible tokens
and the caller's block list ``ids(r)`` (``[T_q, BLOCK_TOPK]`` int32, valid prefix then -1, blocks of BLOCK_SIZE = 4 tokens):

    V(r) = { t < n_vis : floor(t / 4) in ids(r) }  UNION  { t : 4 floor(n_vis / 4) <= t < n_vis }     (the open tail block, always visible)
    O[r] = softmax_fp32(q_r . K[V(r)]^T * scale) . V[V(r)];   LSE[r] = max + log(sum)  (natural log, the exact fp32 sum)

A listed block at or past floor(n_vis / 4) contributes nothing (masked key by key); -1 entries at index < count are "no key"
(zero rows, -inf) and still cost fabric bytes; entries at index >= count are never read.  count = clamp(block_lens[r], 0,
n_sel_default) when block_lens is given, else n_sel_default = min(BLOCK_TOPK, floor((p + 1) / 4)) -- a device min / max,
never a host read.  An empty V(r) (L_b = 0, p < 0, a padded Q row, an empty selection with no tail) lands O = 0 / LSE = -inf
through a SELECT on the emptiness predicate, never ``residue * inv_sum``; padded Q rows store nothing.

WORK ITEM = (query token t, KV head h): the G query heads of t (12 at 24/2, G <= 16) form the MMA N axis of the swap-AB tile,

    BMM1  S^T[128 keys, 16] = K_tile[128 x 256] (A, K-major SW128) . Q^T[16 x 256] (B, K-major SW128)   16 k-steps of 16
    BMM2  O^T[128 d, 16]   += V^T[128 d x 128 keys] (A, MN-major) . P^T[128 keys x 16] (B, MN-major)    2 d-blocks x 8 k-steps

over n_tiles 128-key tiles, each 32 four-token blocks GATHERED by ``tma_gather4`` (one issue = 4 rows x one 128-B column box;
4 boxes per 256-wide row per operand; 128 issues = 64 KiB per operand tile).  One CTA per SM, cga1, 16 warps, persistent
through the shared scheduler (BSHD: the grid is the work list, 2 x T_q items decoded arithmetically; THD: the device-counter
form on an occupancy-sized grid).  Q^T and the block list of item i+1 are prefetched under item i's KV loop; the epilogue gate
of item i (``O *= sigmoid(G)``, compiled out unless CFG.EPILOGUE_GATE) is TMA-staged into the freed sQ slot of item i.

WARP MAP (16 warps = 512 threads; every warpgroup is role-homogeneous because ``setmaxnreg`` is warpgroup-collective)

    warps 0-3   WG0    softmax + epilogue: 128 lanes = the 128 key lanes of S^T / P^T = the 128 d lanes of O^T
    warps 4-11  WG1-2  GATHER issuers w = warp - 4: blocks 4w..4w+3 of every K and V tile; ids LDS from sIds BEFORE the ring
                       wait, shuffled to the elected lane; 4 blocks x 4 boxes = 16 gather4 per operand tile; ONE
                       arrive_expect_tx(KV_TX_BYTES_PER_WARP) per operand stage; kv_state carried across items; ring drain at exit
    warp 12     WG3    MMA issue (+ tmem_alloc / dealloc once per CTA): 4 commits per tile (s_full, kv_empty[K], bmm2_done,
                       kv_empty[V]) + 1 per item (q_empty)
    warp 13     WG3    TMA-LDG: Q^T(i+1) (4 subtiles, Q_TX_BYTES = the BOX bytes), the BLOCK_TOPK x 4 B ids bulk copy of
                       item i+1, the gate of item i into its freed sQ slot (gated arm); NO K/V bytes; drains mb_q_empty /
                       mb_ids_empty / mb_gate_empty at exit
    warp 14     WG3    scheduler (scheduler_warp_loop for BSHD; the persistent claim-counter form for THD) -- never credits
    warp 15     WG3    spare: passes the init sync, then exits (no CTA-wide barrier-0 sync may follow that one in the body)

Every warp except the scheduler credits the scheduler slot (read_tile_id_arrive) at the TOP of its item, right after reading
the payload -> READ_TILE_ARRIVERS = 4 + 1 + 1 + GATHER_WARPS = 14.  Registers: a 512-thread launch gives ptxas 128 per thread
FLAT (``.reqntid 512`` and no ``.maxnreg`` from the DSL; every setmaxnreg is dropped).  Host trace-compile of the decode body
under a forced 512-thread rendering (sm_107a): REG 104, STL 0, LDL 0 on the dense-padded and the causal specializations.  The
declared split for when ``.maxnreg`` lands: 4 x 240 + 8 x 96 + 4 x 80 = 2048 == 128 x 16 (SUM(role regs x warps) == entry x
warps; every count % 8 == 0).  The named fallback ``make_cfg_d256_sparse(gather_warps=4)``: 12 warps, 168 flat,
4 x 240 + 4 x 96 + 4 x 168 = 2016 == 168 x 12.  13 warps is not a config (warpgroup homogeneity, 32 % 6 != 0).

BARRIER TABLE (per CTA; init = the EXACT per-phase arrival sum; "lanes" resolved from the guard form at the call site --
nothing in tile_dsl.barrier elects for you; cga1: every commit is cta_group::1, no cluster fence, no cross-CTA arrive)

    #  name [stages]        producer     issuing lanes (guard form)                                   SUM == init  consumer / wait                   start  carried state
    1  mb_q_full[2]         TMA_LOAD     warp 13, arrive(n_bytes=Q_TX_BYTES, pred=elect_sync()); tma_load_tile x 4 subtiles
                                         Q_TX_BYTES = Q_BOX_ROWS x TILE_K x BPE = G x 512 B (6144 at G=12; the BOX, never the 8 KiB slot)
                                                                                                        1 == 1       MMA before the item's first BMM1     0      q_state(2), once per item
    2  mb_q_empty[2]        MMA_COMMIT   warp 12, arrive(cta_group=1, pred=elect_p) once per item after the item's LAST BMM1
                                                                                                        1 == 1       warp 13 before gate(i) / Q(i+2)      1      per item; drained at exit by warp 13
    3  mb_ids_full[2]       TMA_LOAD     warp 13, arrive(n_bytes=IDS_TX_BYTES, pred=elect_sync()); bulk_copy(sIds[slot], block_ids[row], IDS_TX_BYTES)
                                         IDS_TX_BYTES = BLOCK_TOPK x 4 (a 16-B multiple); the copy runs for EVERY item, dead ones included
                                                                                                        1 == 1       8 gather + 4 softmax warps, item top 0      ids_state(2), once per item
    4  mb_ids_empty[2]      THREAD       `if elect_sync(): arrive()` from each of the 12 consuming warps after its last read of the slot
                                                                                                        12 == 12     warp 13 before refilling (item i+2)  1      per item; drained at exit by warp 13
    5  mb_kv_full[3]        TMA_LOAD     each gather warp: arrive(n_bytes=KV_TX_BYTES_PER_WARP, pred=elect_sync()) once per stage fill,
                                         then its 16 gather4 x 512 B = 8192 B (OOB / -1 rows zero-filled AND credited); 8 x 8192 = the 64 KiB stage
                                                                                                        8 == 8       MMA before BMM1 (K) / BMM2 (V)       0      kv_state(3) per LOAD, order K(0) K(1) V(0) K(2) V(1) ... V(n-1),
                                                                                                                                                                 CARRIED across items in the gather warps AND the MMA warp
    6  mb_kv_empty[3]       MMA_COMMIT   warp 12, pred=elect_p after the last MMA reading the stage (K: with s_full's commit; V: with bmm2_done's)
                                                                                                        1 == 1       all 8 gather warps before refilling 1      the same kv_state; each gather warp drains STAGES_KV at exit
    7  mb_s_full[2]         MMA_COMMIT   warp 12, pred=elect_p right after BMM1(t)                     1 == 1       4 softmax warps, top of tile t       0      s_state(2) per TILE, never reset at an item boundary
    8  mb_s_empty[2]        THREAD       bare arrive() from all 128 softmax lanes after tcgen05_ld -> tcgen05_wait(LOAD) -> tcgen05_fence
                                                                                                        128 == 128   MMA before BMM1(t+2) reuses the slot 1      s_empty_state(2) per tile, carried
    9  mb_p_full[2]         THREAD       bare arrive() from 128 lanes after the P^T store_swizzled + fence_proxy + the O^T rescale + tcgen05_fence
                                                                                                        128 == 128   MMA before BMM2(t)                   0      p_state(2) per tile, carried
   10  mb_bmm2_done[2]      MMA_COMMIT   warp 12, pred=elect_p after BMM2(t)'s two d-blocks           1 == 1       softmax before the O^T rescale (tile t+1) and once in the epilogue (last tile)
                                                                                                                                                          0      bmm2_state(2) per tile; the epilogue's wait advances it.  Also the P^T slot's
                                                                                                                                                                 WAR gate: P^T(t) -> sP[t % 2] is stored AFTER the wait on bmm2_done(t-2) taken at iteration t-1
   11  mb_gate_full[1]      TMA_LOAD     warp 13, arrive(n_bytes=GATE_TX_BYTES, pred=elect_sync()); the gate box (1, 1 token, G heads, 64) x 4 subtiles
                                         into the freed sQ[i % 2]; GATE_TX_BYTES = Q_BOX_ROWS x TILE_O x GATE_BPE = G x 512 B (the BOX)   [gated arm]
                                                                                                        1 == 1       128 softmax lanes, epilogue(i)       0      gate_state(1), once per item
   12  mb_gate_empty[1]     THREAD       bare arrive() from 128 lanes after the item's last gate LDS     [gated arm]
                                                                                                        128 == 128   warp 13 before Q(i+2) into the slot 1      per item; drained at exit by warp 13
   13  sched.mb_scheduler[2]             the scheduler warp's elected arrive_expect_tx(16) (CLC) / elected arrive after the payload stores (THD)
                                                                                                        1 == 1       every consuming warp reads the payload      tile_dsl.scheduler
   14  sched.mb_read_tile_id[2]  THREAD  read_tile_id_arrive(mb, cga_size=1) = one elected lane per calling warp; callers = 4 + 1 + 1 + 8
                                                                                                        14 == 14     the scheduler before refilling the slot     tile_dsl.scheduler
   15  mb_tmem_dealloc[1]   THREAD       bare arrive() from 128 softmax lanes at exit                   128 == 128   MMA before tmem_dealloc               0      once
   16  named barrier 1      bar          MMA warp barrier_cta_arrive (32) + 4 softmax warps barrier_cta_sync (128), thread_count 160   the TMEM-base hand-off, once
   17  named barrier 2      bar          the 4 softmax warps, thread_count 128: the column-max exchange per tile + the per-lane sum per item
   18  barrier 0            bar          all 512 lanes once, after fence_mbarrier_init (the spare warp exits only after it)

Init: ``if warp_idx == 0: if nvvm.elect_sync():`` every stage of every ring (``.init()`` is per stage) + the two scheduler rings
(ONE_LANE / READ_TILE_ARRIVERS); fence_mbarrier_init + barrier_cta_sync OUTSIDE the branch.  Phase discipline: every ring
slot and parity above comes from a PipelineState CARRIED across work items in every role that touches it and advanced at
every producer fire / consumer wait -- NEVER recomputed from a per-item tile index.  The decode tile's arithmetic parities
(``_kv_slot``, ``(t // 2) & 1``, ``((t // 2) - 1) & 1``) are correct only because that kernel is not persistent (fresh
barriers per unit); in a persistent CTA they HANG at n_tiles = 1 with two items (s_empty), read a stale S^T slot at
n_tiles = 3 (s_full) and read a stale K/V slot whenever 2 x n_tiles % 3 != 0.  Every item runs >= 1 tile, so the per-item
rings (1-4, 11-14) advance exactly once per item and the per-tile rings (5-10) at least once; the epilogue runs for every item.
Exit drains (every arrive has a reachable wait): warp 13 waits mb_q_empty x STAGES_Q, mb_ids_empty x STAGES_IDS,
mb_gate_empty x 1; every gather warp waits mb_kv_empty x STAGES_KV -- each with its carried state.

THE ONE HELPER ``_item_bounds``: n_tiles, count, has_open, dead and the appended tail block id(s) are computed by ONE pure
helper at the item-start site of EVERY role that loops over tiles (softmax, MMA, the 8 gather warps, the TMA warp) over the
same inputs (payload, block_lens pointer, seq lens); nothing about them is staged through SMEM.

    pos           = s  (top-left)  |  s + (eff_seqlen_kv_b - eff_seqlen_q_b)  (CFG.BOTTOM_RIGHT)
    n_sel_default = min(CFG.BLOCK_TOPK, floor((pos + 1) / 4))
    count         = clamp(block_lens[t], 0, n_sel_default)  when given, else n_sel_default
    has_open      = ((pos + 1) % 4 != 0) & (pos >= 0)          (dropped under the pure-list test arm, CFG.INCLUDE_OPEN_BLOCK = 0)
    dead          = (pos < 0) | (pos >= eff_seqlen_q_b) | (eff_seqlen_kv_b == 0) | (count + has_open == 0)
    n_tiles       = max(1, ceil((count + has_open) / 32))        CLAMPED >= 1: a dead item runs ONE tile of -1 rows

A dead item's 32 block ids read as -1 (the gather and softmax lanes SELECT -1 over the loaded word), TMA zero-fills and
credits the 128 KiB, every lane is masked, l_tot = 0 and the row_dead SELECT lands O = 0 / LSE = -inf (no store for a padded
Q row) -- every ring advances exactly as on a live item.  Per key lane l of tile i: blk = sIds[32 i + l // 4] (or the tail
id / -1 by the SELECT), key_abs = 4 blk + l % 4, valid = (blk >= 0) & (key_abs <= pos) & (key_abs < eff_seqlen_kv_b);
s[l, j] = valid ? s_raw * scale_log2 : -inf for the 16 columns -- the open block's future rows hold REAL keys of other
tokens, so this mask is mandatory (TMA-OOB zero-fill does not do its job).

SMEM TABLE (declaration order = address order; descriptor-read operands first so every MMA operand starts below 256 KiB,
lane-only buffers last so no alignment padding is spent; 1024-B alignment modelled by the config validator)

    name [stages]   bytes     WRITER, how                       READER, how                      lane stride  swizzle + WHY                                   start
    sQ[2]           16,384    TMA: Q(i+1) 4 subtiles of (G rows x 64), then gate(i) into the freed slot (same box orientation);
                              rows G..15 of BOTH slots zeroed by STS once per CTA + fence_proxy, never written again
                                                                MMA descriptor (B of BMM1, K-major, SBO 8 x 128 B); the gate: 128 lanes, lane = d,
                                                                G x LDS.16 at load_swizzled offsets
                                                                                                 n/a / 2 B    SW128 -- a descriptor reads it AND the per-lane
                                                                                                              reads follow the same SW128 offsets             0 (1024-aligned slots)
    sP[2]            8,192    128 softmax lanes, store_swizzled(16-B chunks, Swizzle(1, 4, 3)) at lane x 32 B
                                                                MMA descriptor (B of BMM2, MN-major, 32-B atom, SBO 8 x 32 B)
                                                                                                 32 B         32-B atom Swizzle(1, 4, 3) -- BOTH jobs: the
                                                                                                              descriptor decodes it (P_SWZ_BYTES = N_Q x BPE)
                                                                                                              and it spreads the 32-B lane stride            16,384
    sKV[3]         196,608    8 gather warps, tma_gather4 x 128 per stage: box b of quad q at b x 16 KiB + q x 512 B (512-B quads inside
                              1024-B sub-boxes = the tiled SW128 layout)
                                                                MMA descriptors over the same bytes: K-major A of BMM1 (SBO 1 KiB) and
                                                                MN-major A of BMM2 (V^T: LBO 16 KiB between d atoms, SBO 1 KiB)
                                                                                                 n/a          SW128 -- both descriptors read it; the gather
                                                                                                              map writes under the same SW128                24,576; sKV[2] at 155,648 = 152 KiB
    sIds[2]          4,224    warp 13, ONE cp.async.bulk of IDS_TX_BYTES from block_ids[row]; no STS (slot = IDS_TX_BYTES + 64 B reserve, unread)
                                                                gather lanes 0-3: LDS.32 of block 32 i + 4 w + lane; softmax lanes: LDS.32 of
                                                                block 32 i + lane // 4 (4 adjacent lanes share a word)
                                                                                                 4 B          none -- no descriptor reads it and 32 lanes x 4 B
                                                                                                              is one bank cycle: neither justification applies  221,184 (16-B aligned bulk-copy destination)
    red                768    softmax warps: one warp-uniform fp32 per (slot, warp, column), 3 slots (tile parity 0 / 1, epilogue) x 4 x 16
                                                                the group's 4 warps after named barrier 2
                                                                                                 4 B          none (one address per warp per column)          225,408
    misc               512    tmem_ptr (16 B), 15 mbarrier arrays (29 stages, 16-B padded: 272 B), the payload ring (64 B) -- 352 B in a 512 B reserve  226,176
    total          226,688 B = 221.4 KiB  < 232,448 B (227 KiB, the standard carveout; L1 keeps its 100 KiB) by 5,760 B

Descriptor version: the LAST MMA-operand start is sKV[STAGES_KV - 1] = 155,648 B < 262,144 -> DESC_VERSION = 0, ONE module
constant passed to every SmemTile (a version-0 descriptor's 14-bit start_address wraps at 256 KiB; version 1 is not a
transparent widening, so it is derived, never "set to be safe").  A separate gate slot (+8 KiB) or ring depth 4 (+64 KiB)
needs the oversized carveout (ALLOW_OVERSIZED_SHARED_MEMORY, L1 -> 8 kB) -- measured levers, not defaults.  Proxy fences:
after the P^T generic stores (before mb_p_full), after the once-per-CTA sQ tail-row zeroing; the TMEM path is ordered by
tcgen05_fence + the commits; the sIds bulk copy -> LDS and gate TMA -> LDS edges are ordered by their mbarrier waits.

TMEM MAP (64 columns per CTA, allocated once, is_exclusive=False -- <= 512 columns): S^T slot 0 at [0, 16), slot 1 at
[16, 32), O^T d-block 0 at [32, 48), d-block 1 at [48, 64).  No TMEM slot is assumed zero: BMM2(0) of every item overwrites
(accumulate=False) and every dead column ends in the SELECT, never residue * 0.

GATHER TENSOR MAP: one 2-D map per operand, tokens OUTER, the token's FULL ROW inner, box (64 elems, 1 row) = one 128-B
SW128 span; the head and the d-box live in the COLUMN coordinate (h x D + box x 64; + o_k / o_v on the block's slab), every
batch / sequence / page term in the ROW coordinate (b x S + 4 blk + r; cu_seqlens_k[b] + 4 blk + r; page_table[b, 4 blk //
page_size] x page_size + (4 blk) % page_size + r; HND pools fold the head into the row).  Stride contract: last dim
contiguous, head stride a multiple of 64 elements, token stride a multiple of 8 elements, extents < 2^32, page_size % 4 == 0.

DEGENERATE-INPUT MATRIX (every row names its handling site): empty selection / pos < 0 -> dead -> one -1 tile -> the
row_dead SELECT; a query with 0 complete blocks (pos in {0, 1, 2}) -> count 0, has_open 1, the key_abs <= pos term;
(pos + 1) % 4 == 0 -> has_open 0; n_tiles in {1, 3, 4, 17} -> the carried PipelineStates; -1 inside the valid prefix -> "no
key" (blk >= 0), bytes still fetched; a block id past the sequence -> key_abs < eff_seqlen_kv_b, TMA zero-fills past the
tensor; a block violating block-causality -> key_abs <= pos; duplicates -> counted twice (contract); S_kv = 0 -> dead (one
tile, no host read); padded Q rows -> dead, no store; B x H > 1 with n_tiles = 1 -> carried states; block_lens out of range
-> clamped on device; B >= 2 / the slab stride -> the ROW coordinate carries b x S, the COLUMN coordinate the head and the
column offset; one CTA / one item -> cga1 init counts; page_size % 4 != 0, top_k outside [4, 512] or % 4 != 0, G > 16 ->
typed declines at config / check_support.  The denominator floor (1e-30) sits inside the reciprocal and the log only and
is always followed by the row_dead SELECT, so neither LSE nor O ever carries it.
"""
