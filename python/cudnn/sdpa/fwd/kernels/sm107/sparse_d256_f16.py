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

A listed block strictly past the open one (4 blk >= n_vis) contributes nothing (masked key by key); an entry that lists the
OPEN block itself (blk == floor(n_vis / 4) with n_vis % 4 != 0) or repeats another entry is attended TWICE -- a duplicate of
the appended tail / of the other entry (the softmax over the multiset; the list contract forbids both -- the library's host
check is FORM-only and never reads a list, the test tree's ``block_ids_contract_violations`` detector is what rejects them);
-1 entries at index < count are "no key" (zero rows, -inf) and still cost fabric bytes; entries at index
>= count are never read, so an extra entry placed at index count is never read either.  count = clamp(block_lens[r], 0,
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
    warps 4-11  WG1-2  GATHER issuers w = warp - 4: blocks 4w..4w+3 of every K and V tile; the quad of ids by ONE warp-uniform
                       16-B LDS from sIds BEFORE the ring wait (the quad is 16-B aligned: BLOCK_TOPK % 4 == 0), the SELECT per
                       index; PAGED arm: ONE block-table lookup per block of the quad right there (page = block_table[b, 4 blk //
                       PAGE_SIZE], an in-bounds read, -1 for a dead / past-the-length block, a page index past the table or a -1
                       page); 4 blocks x 4 boxes = 16 gather4 per operand tile from the elected lane; ONE
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
4 x 240 + 4 x 96 + 4 x 168 = 2016 == 168 x 12.  Every other population is declined by the config: a gather count that does
not divide the 32 blocks of a tile (5 -> 13 warps, 6 -> 14 warps) by the divisibility predicate, one that divides them but
does not fill a warpgroup (2 -> 10 warps) by the warpgroup predicate; the factory itself admits gather_warps in {4, 8} only.

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
   13  sched.mb_scheduler[2]  TMA_LOAD (CLC response form) | THREAD (persistent form)
                                         the scheduler warp's elected arrive_expect_tx(16) + the CLC response (CLC) / its elected arrive after the
                                         4 payload stores (persistent; no expect_tx at cga1)
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
    n_vis         = clamp(min(pos + 1, eff_seqlen_kv_b), 0)     the VISIBLE range (a padded sequence shorter than pos keeps its own tail)
    b_open        = floor(n_vis / 4);  has_open = (n_vis % 4 != 0)   (dropped under the pure-list test arm, CFG.INCLUDE_OPEN_BLOCK = 0)
    dead          = (pos < 0) | (pos >= eff_seqlen_q_b) | (eff_seqlen_kv_b == 0) | (count + has_open == 0)
    n_tiles       = max(1, ceil((count + has_open) / 32))        CLAMPED >= 1: a dead item runs ONE tile of -1 rows

A dead item's 32 block ids read as -1 (the gather and softmax lanes SELECT -1 over the loaded word, read at an address
clamped into the slot), TMA zero-fills and credits the 128 KiB, every lane is masked, l_tot = 0 and the row_dead SELECT lands
O = 0 / LSE = -inf (no store for a padded Q row) -- every ring advances exactly as on a live item.  A gather ROW coordinate
is batch x S_kv + 4 blk + r for a live row and -1 (TMA zero-fill, bytes credited) for a -1 block, a block past the batch's
length and the straddling rows of its last block, so no other batch's row -- or its NaN -- reaches an MMA under P = 0.  Per key lane l of tile i: blk = sIds[32 i + l // 4] (or the tail
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
                                                                gather warps: ONE uniform LDS.128 of blocks 32 i + 4 w .. + 3; softmax lanes: LDS.32 of
                                                                block 32 i + lane // 4 (4 adjacent lanes share a word); both clamped into the slot
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
batch / sequence / page term in the ROW coordinate (b x S + 4 blk + r; cu_seqlens_k[b] + 4 blk + r; the paged form below).
Stride contract: last dim contiguous, head stride a multiple of 64 elements, token stride a multiple of 8 elements, extents
< 2^32, page_size % 4 == 0.

PAGED ARM (CFG.PAGED_KV; a const_expr branch of the gather warps only -- the softmax, MMA and TMA warps and every barrier /
SMEM row are the dense arm's): K / V are page POOLS [num_pages, H_kv, PAGE_SIZE, D] addressed through a (B, max_pages) int32
block table; the per-batch KV length (seq_kv_lens, REQUIRED) is the visible range.  The 2-D map's rows are the pool's token
rows at the TOKEN stride (pitch = the page_size axis' stride): rows = num_pages x rows_per_page, inner = (H_kv - 1) x
col_head_stride + D.  The row of block blk's token r = page x rows_per_page + h x rows_per_head + (4 blk) % PAGE_SIZE + r
with page = block_table[b, (4 blk) // PAGE_SIZE] -- PAGE_SIZE % 4 == 0, so a block never straddles pages and ONE lookup per
block suffices; the lookup is hoisted into the gather warps' ids quad before the ring wait.  NHD pools (token stride
H_kv x D, head stride D): rows_per_page = PAGE_SIZE, rows_per_head = 0, the head in the column (h x D + box x 64); HND pools
(token stride D, head stride PAGE_SIZE x D): rows_per_page = H_kv x PAGE_SIZE, rows_per_head = PAGE_SIZE, the column box x
64 -- the adapter derives the three per-operand numbers from the strides (one compiled kernel serves both).  A -1 block, a
block whose first token is at or past the batch's KV length (its page index may name another sequence's stale page), a page
index past the table and a -1 table entry (the page -1 convention of the dense paged tiles) all take row -1: TMA zero-fill,
bytes credited -- so the rows of a partially filled last page beyond the KV length (another sequence's tokens, or garbage)
never reach an MMA, and the straddling rows of the open block are masked exactly as in the dense arm (key_abs < the length).

DEGENERATE-INPUT MATRIX (every row names its handling site): empty selection / pos < 0 -> dead -> one -1 tile -> the
row_dead SELECT; a query with 0 complete blocks (pos in {0, 1, 2}) -> count 0, has_open 1, the key_abs <= pos term;
(pos + 1) % 4 == 0 -> has_open 0; n_tiles in {1, 3, 4, 17} -> the carried PipelineStates; -1 inside the valid prefix -> "no
key" (blk >= 0), bytes still fetched; a block id past the sequence -> key_abs < eff_seqlen_kv_b, TMA zero-fills past the
tensor; a block violating block-causality -> key_abs <= pos (a future block contributes nothing; the open block listed inside
the count duplicates the tail); duplicates -> counted twice (contract); S_kv = 0 -> dead (one
tile, no host read); padded Q rows -> dead, no store; B x H > 1 with n_tiles = 1 -> carried states; block_lens out of range
-> clamped on device; B >= 2 / the slab stride -> the ROW coordinate carries b x S, the COLUMN coordinate the head and the
column offset; one CTA / one item -> cga1 init counts; a paged block whose page is -1 / past the table -> row -1 (zero rows,
the keys masked by the length); page_size % 4 != 0, top_k outside [4, 512] or % 4 != 0, G > 16 -> typed declines at
config / check_support.  The denominator floor (1e-30) sits inside the reciprocal and the log only and
is always followed by the row_dead SELECT, so neither LSE nor O ever carries it.
"""

# Config comes from the FROST template loader (FROST_TEMPLATE_PARAMS is injected before this body runs); the plain-import
# default is the 24/2 geometry (12 query heads per KV head) at top_k = 512, bf16, so a plain import stays usable standalone.
from cudnn.sdpa.fwd.config_sm107 import TemplateParams, make_cfg_d256_sparse

PARAMS: TemplateParams = globals().get(
    "FROST_TEMPLATE_PARAMS", TemplateParams(dtype_qkv=2, cta_mma=1, pack_gqa=True, qh_per_kh=12, qsa_block_topk=512, qsa_block_size=4)
)
CFG, _TMA = make_cfg_d256_sparse(PARAMS)

# tcgen05 SMEM-descriptor version for EVERY SmemTile in this module -- ONE decision point, DERIVED by the config from the
# layout (the start of the last K/V stage against the 14-bit version-0 window), wired into every construction below and
# never re-literalled at a tile (test_sm107_every_smem_tile_takes_the_module_desc_version's convention).
DESC_VERSION: int = CFG.DESC_VERSION
# Retry form of the per-KV-iteration RING waits: every such site is spelled ``.wait(..., spin=SPIN_RING_WAITS)``, never a
# literal; the sign is a MEASURED per-kernel fact, so this stays False until this kernel's own A/B.
SPIN_RING_WAITS: bool = False
Cfg = type(CFG)

# CuTe DSL version gate (python/cudnn/AGENTS.md Rule 7): sm_107a needs the public 4.8.0 wheel; below it every compile on a
# Rubin box dies inside the DSL (``KeyError: 'sm_107a'``).  The adapter declines first (SparseGqaFwdDslSm107.check_support,
# BEFORE this module is loaded); this is the typed backstop for a direct template load, placed before the kernel body so the
# message names the installed version, never a DSL internal.
from cudnn.frost.buffers import cutedsl_arch_requirement_error

_DSL_ARCH_ERROR = cutedsl_arch_requirement_error((10, 7))
if _DSL_ARCH_ERROR is not None:
    raise NotImplementedError(f"sparse_d256_f16 (sm107): {_DSL_ARCH_ERROR}")

from cudnn.frost.compiled_cache import compile_cached as _compile_cached, template_key as _template_key
from functools import lru_cache
from typing import Callable, NamedTuple, Optional, Tuple

from cutlass.experimental import primitives as nvvm
from cutlass.experimental.cuda import tensor_map as tmap
from cutlass._mlir.dialects import arith
from cutlass.base_dsl.typing import Pointer

import cutlass
from cutlass.experimental import primitives as prims
import cutlass.cute as cute
import cuda.bindings.driver as _cuda_driver  # noqa: F401

from cudnn.frost.tile_dsl.barrier import MBarrier, PipelineState, Producer, advance, wait
from cudnn.frost.tile_dsl.mma import mma_ss
from cudnn.frost.tile_dsl.tma import TMA_L2_EVICT_LAST, bulk_copy, opaque_i64, tma_gather4, tma_load_tile
from cudnn.frost.tile_dsl.handles import GmemTileTma, MmaDesc, SmemTile
from cudnn.frost.tile_dsl.tmem import tmem_alloc, tmem_dealloc
from cudnn.frost.tile_dsl.scheduler import Sched, read_clc_payload, read_tile_id_arrive, scheduler_warp_loop
from cudnn.frost.tile_dsl.pointwise import fp32_to_fp16
from cudnn.sdpa.fwd.kernels._common_blackwell import _bshd, _vec, row_max_for_exp2

if CFG.DTYPE_QKV == 2:
    STORAGE_DTYPE = cutlass.BFloat16
else:
    STORAGE_DTYPE = cutlass.Float16
MMA_KIND = nvvm.Tcgen05MMAKind.F16
CTA_GROUP_KIND = nvvm.CTAGroup.CTA_1

# === geometry, every value from CFG (frost-kernels.md section 6: derive, never re-literal) ===
N_Q = CFG.N_Q
TILE_N = CFG.TILE_N
TILE_K = CFG.TILE_K
TILE_O = CFG.TILE_O
BPE = CFG.BPE
STAGES_KV = CFG.STAGES_KV
STAGES_Q = CFG.STAGES_Q
STAGES_IDS = CFG.STAGES_IDS
SCHEDULER_STAGES = CFG.SCHEDULER_STAGES
BLOCK_SIZE = CFG.BLOCK_SIZE
BLOCKS_PER_TILE = CFG.BLOCKS_PER_TILE
BLOCKS_PER_WARP = CFG.BLOCKS_PER_WARP
BLOCK_TOPK = CFG.BLOCK_TOPK
GATHER_BOXES = CFG.GATHER_BOXES
GATHER_BOX_ELEMS = CFG.GATHER_BOX_ELEMS
# The paged arm: PAGE_SIZE tokens per pool page (0 on the dense arm, where no paged code is traced); PAGE_SIZE % BLOCK_SIZE == 0
# (the config's predicate) is what makes ONE block-table lookup per block sufficient.
PAGED_KV = CFG.PAGED_KV
PAGE_SIZE = CFG.PAGE_SIZE

# The work item's Q^T box: ONE token x its G query heads (Q_BOX_ROWS = G rows of the N_Q = 16 tile; rows G..15 are the
# once-zeroed tail whose columns are computed and never stored).
G = CFG.QH_PER_KH
Q_BOX_TOKENS = CFG.Q_BOX_TOKENS
Q_BOX_ROWS = CFG.Q_BOX_ROWS

# TMA granule for Q (the K-major B operand of BMM1): one 128 B swizzle atom of the head dim = 64 half elements.  The gather
# box of K/V is the same 128-B span, so one SW128 layout serves the TMA-loaded Q and the gathered K/V tiles alike.
GRANU_ELEMS = _TMA.QK_GRANU_ELEMS
QK_ITERS = _TMA.QK_ITERS
if _TMA.VO_GRANU_ELEMS != GRANU_ELEMS or GATHER_BOX_ELEMS != GRANU_ELEMS:
    raise ValueError("sparse_d256_f16: Q, K and V must share the 128 B swizzle granule (the gather box is one such span)")

qBufferElems = N_Q * TILE_K
kvBufferElems = TILE_N * TILE_K
pBufferWords = TILE_N * N_Q // 2  # P^T as packed half pairs: N_Q / 2 words per key row
SUB_BOX_ELEMS = TILE_N * GATHER_BOX_ELEMS  # one 128-B column box of a K/V tile = TILE_N rows x 128 B = 16 KiB
IDS_SLOT_WORDS = CFG.IDS_SLOT_BYTES // 4

# Transaction bytes -- every one the BOX / the copy length, never a buffer size (barrier rows 1, 3, 5).
Q_TX_BYTES = CFG.Q_TX_BYTES
IDS_TX_BYTES = CFG.IDS_TX_BYTES
KV_TX_BYTES_PER_WARP = CFG.KV_TX_BYTES_PER_WARP

_SWZ_ENUM = {128: 2, 64: 4, 32: 6}
_SWZ_BITS = {128: 3, 64: 2, 32: 1}
SMEM_LAYOUT_QK = _SWZ_ENUM[CFG.Q_SWZ_BYTES]
SMEM_LAYOUT_P = _SWZ_ENUM[CFG.P_SWZ_BYTES]
# The generic-proxy P^T stores apply the same XOR pattern the UMMA descriptor decodes (Swizzle<B, 4, 3>).
_P_SMEM_SWIZZLE = cutlass.Swizzle(_SWZ_BITS[CFG.P_SWZ_BYTES], 4, 3)
P_ROW_WORDS = N_Q // 2

# K-major operands (K as A of BMM1, Q^T as B of BMM1): SBO = 8 rows x 128 B.
STRIDE_BYTE_OFFSET_QK = 8 * CFG.K_SWZ_BYTES
# V^T as the MN-major A of BMM2: 128 d columns span two 64-wide swizzle atoms, LBO steps between them (TILE_N rows x 128 B
# each); SBO = 8 key rows.
LEADING_BYTE_OFFSET_VT = TILE_N * CFG.V_SWZ_BYTES
STRIDE_BYTE_OFFSET_VT = 8 * CFG.V_SWZ_BYTES
# P^T as the MN-major B of BMM2: one atom wide (N_Q * 2 B), SBO = 8 key rows.
STRIDE_BYTE_OFFSET_P = 8 * CFG.P_SWZ_BYTES

# TMEM: two S^T slots then the two O^T d-blocks, N_Q fp32 columns each (CFG.S_ACC_OFF / CFG.O_OFF, allocated once per CTA).
S_ACC_OFF = CFG.S_ACC_OFF
O_OFF = CFG.O_OFF
TMEM_COLS = CFG.TMEM_COLS
O_BLOCKS = TILE_O // 128

# One softmax column group: 4 warps x 16 columns (= the 16 Q rows of the item).
COLS = N_Q
SOFTMAX_WARPS = CFG.SOFTMAX_WARPS
SOFTMAX_LANES = CFG.SOFTMAX_LANES
GATHER_WARP_BASE = CFG.GATHER_WARP_BASE
GATHER_WARPS = CFG.GATHER_WARPS
MMA_WARP_ID = CFG.MMA_WARP_ID
TMALDG_WARP_ID = CFG.TMALDG_WARP_ID
SCHED_WARP_ID = CFG.SCHED_WARP_ID
P_CHUNKS = (COLS * BPE) // 16  # 16-B chunks of one key row's P^T slice
# Named barriers: 1 = TMEM base hand-off (softmax warps + MMA warp), 2 = the softmax warps' own cross-warp reductions.
_BAR_TMEM = 1
_BAR_SOFTMAX = 2
BAR_TMEM_THREADS = CFG.BAR_TMEM_THREADS
BAR_SOFTMAX_THREADS = CFG.BAR_SOFTMAX_THREADS
# Cross-warp reduction scratch: [2 tile parities + 1 epilogue][softmax warp][COLS].
RED_SLOTS = 3
RED_WORDS = RED_SLOTS * SOFTMAX_WARPS * COLS

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453


class SparseBars(NamedTuple):
    mb_q_full: object
    mb_q_empty: object
    mb_ids_full: object
    mb_ids_empty: object
    mb_kv_full: object
    mb_kv_empty: object
    mb_s_full: object
    mb_s_empty: object
    mb_p_full: object
    mb_bmm2_done: object
    mb_tmem_dealloc: object


def make_sparse_bars() -> SparseBars:
    """The barrier table's rows 1-10 and 15 as typed MBarriers; init counts from CFG (the EXACT per-phase arrival sums the
    header's ledger derives: ONE_LANE for every TMA_LOAD / MMA_COMMIT producer, KV_FULL_ARRIVERS = one expect_tx per gather
    warp, IDS_EMPTY_ARRIVERS = one elected lane per list-reading warp, SOFTMAX_LANES for the bare lane arrives).  The gate
    rows 11 / 12 belong to the EPILOGUE_GATE arm, which this body does not carry yet."""

    def _alloc(n):
        return cutlass.Array(cutlass.Int64, n, alignment=16, space=cutlass.AddressSpace.smem)

    return SparseBars(
        mb_q_full=MBarrier(_alloc(STAGES_Q), stages=STAGES_Q, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_q_empty=MBarrier(_alloc(STAGES_Q), stages=STAGES_Q, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_ids_full=MBarrier(_alloc(STAGES_IDS), stages=STAGES_IDS, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_ids_empty=MBarrier(_alloc(STAGES_IDS), stages=STAGES_IDS, init_count=CFG.IDS_EMPTY_ARRIVERS, producer=Producer.THREAD),
        mb_kv_full=MBarrier(_alloc(STAGES_KV), stages=STAGES_KV, init_count=CFG.KV_FULL_ARRIVERS, producer=Producer.TMA_LOAD),
        mb_kv_empty=MBarrier(_alloc(STAGES_KV), stages=STAGES_KV, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_s_full=MBarrier(_alloc(2), stages=2, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_s_empty=MBarrier(_alloc(2), stages=2, init_count=SOFTMAX_LANES, producer=Producer.THREAD),
        mb_p_full=MBarrier(_alloc(2), stages=2, init_count=SOFTMAX_LANES, producer=Producer.THREAD),
        mb_bmm2_done=MBarrier(_alloc(2), stages=2, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_tmem_dealloc=MBarrier(_alloc(1), stages=1, init_count=SOFTMAX_LANES, producer=Producer.THREAD),
    )


@cute.jit
def _select_i32(cond, a, b):
    return cutlass.Int32(arith.select(cond.ir_value(), cutlass.Int32(a).ir_value(), cutlass.Int32(b).ir_value()))


@cute.jit
def _select_f32(cond, a, b):
    return cutlass.Float32(arith.select(cond.ir_value(), a.ir_value(), b.ir_value()))


@cute.jit
def _item_bounds(tok, batch, block_lens_tensor: Optional[cute.Tensor], seq_kv_lens_tensor, seqlen_q, seqlen_kv):
    """THE ONE HELPER (header): the item's position, the entries read from its list, the open tail block, the dead flag and
    the tile count -- computed by EVERY role that loops over tiles from the SAME inputs, nothing staged through SMEM.

    pos = the token's position (top-left causal; the bottom-right arm is not carried yet); eff_seqlen_kv = seq_kv_lens[b]
    under SEQ_KV_LENS_PRESENT else the dense extent; n_vis = clamp(min(pos + 1, eff_seqlen_kv), 0): the visible range.
    n_sel_default = min(BLOCK_TOPK, floor((pos + 1) / BLOCK_SIZE)); count = clamp(block_lens[row], 0, n_sel_default) when the
    pointer is given (a device min / max, never a host read, never a fault), else n_sel_default.  The open tail block is
    b_open = floor(n_vis / BLOCK_SIZE) when n_vis % BLOCK_SIZE != 0 (the tail of the VISIBLE range, so a padded sequence
    shorter than the position keeps its own tail); the per-lane mask removes its rows past pos / past the length.
    dead = (pos < 0) | (pos >= eff_seqlen_q) | (eff_seqlen_kv == 0) | (count + has_open == 0).
    n_tiles = max(1, ceil((count + has_open) / BLOCKS_PER_TILE)) -- CLAMPED >= 1: a dead item runs ONE tile of -1 rows.
    """
    pos = tok
    eff_seqlen_kv = seqlen_kv
    if cutlass.const_expr(CFG.SEQ_KV_LENS_PRESENT == 1):
        eff_seqlen_kv = cutlass.Int32(cutlass.make_array_view(seq_kv_lens_tensor)[batch])
    eff_seqlen_q = seqlen_q
    zero = cutlass.Int32(0)
    n_vis = cute.math.max(cute.math.min(pos + cutlass.Int32(1), eff_seqlen_kv), zero)
    n_sel_default = cute.math.min(cutlass.Int32(BLOCK_TOPK), cute.math.max(pos + cutlass.Int32(1), zero) // cutlass.Int32(BLOCK_SIZE))
    count = n_sel_default
    if cutlass.const_expr(block_lens_tensor is not None):
        row = batch * seqlen_q + tok
        given = cutlass.Int32(cutlass.make_array_view(block_lens_tensor)[row])
        count = cute.math.min(cute.math.max(given, zero), n_sel_default)
    b_open = n_vis // cutlass.Int32(BLOCK_SIZE)
    has_open = zero
    if cutlass.const_expr(CFG.INCLUDE_OPEN_BLOCK == 1):
        has_open = _select_i32((n_vis % cutlass.Int32(BLOCK_SIZE)) != zero, 1, 0)
    n_listed = count + has_open
    dead = (pos < zero) | (pos >= eff_seqlen_q) | (eff_seqlen_kv <= zero) | (n_listed == zero)
    n_tiles = cute.math.max(cutlass.Int32(1), (n_listed + cutlass.Int32(BLOCKS_PER_TILE - 1)) // cutlass.Int32(BLOCKS_PER_TILE))
    return pos, count, has_open, b_open, dead, n_tiles, eff_seqlen_kv


@cute.jit
def _block_id_at(idx, loaded, count, has_open, b_open, dead):
    """The block id at list index ``idx`` of the current item: the staged word while ``idx < count``, the open tail block at
    ``idx == count`` when the item has one, -1 past that -- and -1 for every index of a dead item.  ``loaded`` is the word
    the caller read unconditionally at an in-bounds address; this SELECT decides whether it means anything."""
    tail = _select_i32((idx == count) & (has_open != cutlass.Int32(0)), b_open, cutlass.Int32(-1))
    blk = _select_i32(idx < count, loaded, tail)
    return _select_i32(dead, cutlass.Int32(-1), blk)


@cute.jit
def _decode_payload(t0, t1):
    """CLC response -> (token, KV head, batch): the grid is (S_q, H_kv, B), the response packs ctaid.y | ctaid.z << 16."""
    tok = cute.arch.make_warp_uniform(t0)
    head = cute.arch.make_warp_uniform(t1 & cutlass.Int32(0xFFFF))
    batch = cute.arch.make_warp_uniform((t1 >> cutlass.Int32(16)) & cutlass.Int32(0xFFFF))
    return tok, head, batch


# === TMA-LDG warp (13): Q^T and the block list of the NEXT item, one item ahead ===========================================


@cute.jit
def _tma_issue_item(tma_q, sQ, sIds_raw, bars, ids_base, tok, head, batch, seqlen_q, q_idx, q_phase, qe_idx, qe_phase, i_idx, i_phase, ie_idx, ie_phase):
    """Issue the two loads of one work item: the ``BLOCK_TOPK x 4`` B list row by ONE bulk copy into ``sIds[slot]``
    (barrier row 3; unconditional -- dead items included, the row exists for every token) and the ``Q^T`` box of the token's
    G heads into ``sQ[slot]`` (row 1; ``Q_TX_BYTES`` = the BOX bytes).  Each slot is taken only after its ``_empty`` wait
    (rows 2 / 4, pre-armed states).  Returns the four advanced PipelineStates as (idx, phase) pairs."""
    row = batch * seqlen_q + tok
    # --- the list (row 4 wait -> row 3 arm + copy)
    bars.mb_ids_empty[ie_idx].wait(ie_phase)
    bars.mb_ids_full[i_idx].arrive(n_bytes=IDS_TX_BYTES, pred=nvvm.elect_sync())
    bulk_copy(
        sIds_raw.subview(i_idx * cutlass.Int32(IDS_SLOT_WORDS)), ids_base + row * cutlass.Int32(BLOCK_TOPK), IDS_TX_BYTES, bars.mb_ids_full[i_idx].smem_ptr
    )
    # --- Q^T (row 2 wait -> row 1 arm + 4 subtiles)
    bars.mb_q_empty[qe_idx].wait(qe_phase)
    bars.mb_q_full[q_idx].arrive(n_bytes=Q_TX_BYTES, pred=nvvm.elect_sync())
    tma_load_tile(sQ[q_idx], tma_q(cutlass.Int32(0), head * cutlass.Int32(G), tok, batch), bars.mb_q_full[q_idx].smem_ptr, cta_group=1)
    q_state = advance(PipelineState(idx=q_idx, phase=q_phase), STAGES_Q)
    qe_state = advance(PipelineState(idx=qe_idx, phase=qe_phase), STAGES_Q)
    i_state = advance(PipelineState(idx=i_idx, phase=i_phase), STAGES_IDS)
    ie_state = advance(PipelineState(idx=ie_idx, phase=ie_phase), STAGES_IDS)
    return q_state.idx, q_state.phase, qe_state.idx, qe_state.phase, i_state.idx, i_state.phase, ie_state.idx, ie_state.phase


@cute.jit
def _tmaldg_warp_group(tma_q_desc, sQ, sIds_raw, bars, sched, block_ids_tensor, seqlen_q, tok0, head0, batch0):
    tma_q = GmemTileTma(tma_q_desc)
    ids_base = Pointer(block_ids_tensor.iterator.raw_ptr(), dtype=cutlass.Int32)
    # Producer states (slot to fill next) and the pre-armed consumer states of the matching _empty rings (P5b: Q(0), Q(1)
    # and ids(0), ids(1) pass fresh barriers) -- all four carried across items.
    q_state = PipelineState.start(phase=0)
    qe_state = PipelineState.start(phase=1)
    i_state = PipelineState.start(phase=0)
    ie_state = PipelineState.start(phase=1)

    # Prologue: the first item's loads (its coordinates are this CTA's grid position).
    qi, qp, qei, qep, ii, ip, iei, iep = _tma_issue_item(
        tma_q,
        sQ,
        sIds_raw,
        bars,
        ids_base,
        tok0,
        head0,
        batch0,
        seqlen_q,
        q_state.idx,
        q_state.phase,
        qe_state.idx,
        qe_state.phase,
        i_state.idx,
        i_state.phase,
        ie_state.idx,
        ie_state.phase,
    )
    q_state = PipelineState(idx=qi, phase=qp)
    qe_state = PipelineState(idx=qei, phase=qep)
    i_state = PipelineState(idx=ii, phase=ip)
    ie_state = PipelineState(idx=iei, phase=iep)

    is_valid = cutlass.Int32(1)
    sched_state = PipelineState.start()
    while is_valid > cutlass.Int32(0):
        # Credit the scheduler slot at the TOP of the item (row 14), then learn the NEXT item and prefetch its Q^T and list
        # under the current item's KV loop.  The _empty waits inside the issue bound this warp at one item ahead.
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), 1)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        t0, t1, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        tok, head, batch = _decode_payload(t0, t1)
        is_valid = cute.arch.make_warp_uniform(nxt_v)
        sched_state = advance(sched_state, SCHEDULER_STAGES)
        if is_valid > cutlass.Int32(0):
            qi, qp, qei, qep, ii, ip, iei, iep = _tma_issue_item(
                tma_q,
                sQ,
                sIds_raw,
                bars,
                ids_base,
                tok,
                head,
                batch,
                seqlen_q,
                q_state.idx,
                q_state.phase,
                qe_state.idx,
                qe_state.phase,
                i_state.idx,
                i_state.phase,
                ie_state.idx,
                ie_state.phase,
            )
            q_state = PipelineState(idx=qi, phase=qp)
            qe_state = PipelineState(idx=qei, phase=qep)
            i_state = PipelineState(idx=ii, phase=ip)
            ie_state = PipelineState(idx=iei, phase=iep)

    # Exit drains (P6): the last STAGES_Q q_empty commits and the last STAGES_IDS ids_empty arrives need a waiter; the
    # carried states name exactly those phases.
    for _ in cutlass.range_constexpr(STAGES_Q):
        bars.mb_q_empty[qe_state.idx].wait(qe_state.phase)
        qe_state = advance(qe_state, STAGES_Q)
    for _ in cutlass.range_constexpr(STAGES_IDS):
        bars.mb_ids_empty[ie_state.idx].wait(ie_state.phase)
        ie_state = advance(ie_state, STAGES_IDS)


# === GATHER warps (4-11): the K/V ring, block by block ===================================================================


@cute.jit
def _gather_block_ids(sIds_raw, slot_base, tile, w, count, has_open, b_open, dead):
    """This warp's BLOCKS_PER_WARP block ids of tile ``tile``: list indices ``32 tile + BLOCKS_PER_WARP w + q``.  The staged
    words are read with warp-uniform 16-B vector loads (the quad is 16-B aligned: BLOCK_TOPK % 4 == 0 and the slot start is)
    BEFORE the ring wait, at an address clamped into the slot; the SELECT (``_block_id_at``) substitutes the tail id / -1 /
    the dead item's -1 per index.  Returns a Vector of BLOCKS_PER_WARP Int32."""
    first = tile * cutlass.Int32(BLOCKS_PER_TILE) + w * cutlass.Int32(BLOCKS_PER_WARP)
    ld_first = cute.math.min(first, cutlass.Int32(BLOCK_TOPK - BLOCKS_PER_WARP))
    ids = []
    for qq in cutlass.range_constexpr(BLOCKS_PER_WARP // 4):
        v = sIds_raw.load(slot_base + ld_first + cutlass.Int32(qq * 4), vector_size=4, alignment=16)
        for r in cutlass.range_constexpr(4):
            q = qq * 4 + r
            ids.append(_block_id_at(first + cutlass.Int32(q), cutlass.Int32(v[r]), count, has_open, b_open, dead))
    return cutlass.Vector.from_elements(tuple(ids), cutlass.Int32)


@cute.jit
def _gather_block_pages(blks, block_table_tensor, batch, max_pages, eff_seqlen_kv):
    """PAGED arm: the pool PAGE of each block of this warp's quad -- ONE block-table lookup per block, hoisted here (the ids
    step, before the ring wait) so the fill does arithmetic only: ``page = block_table[batch, (4 blk) // PAGE_SIZE]`` read at
    an index clamped into the table row (always in bounds), then -1 unless the block is live (``blk >= 0``), its first token
    is inside the batch's KV length (a block at or past it may name another sequence's stale page), the page index is inside
    the table and the entry is not -1 (the page -1 convention of the dense paged tiles).  A -1 page becomes row -1 at the
    fill: TMA zero-fill, bytes credited.  Returns a Vector of BLOCKS_PER_WARP Int32."""
    bt = cutlass.make_array_view(block_table_tensor)
    last = cute.math.max(max_pages - cutlass.Int32(1), cutlass.Int32(0))
    pages = []
    for q in cutlass.range_constexpr(BLOCKS_PER_WARP):
        blk = cutlass.Int32(blks[q])
        key0 = blk * cutlass.Int32(BLOCK_SIZE)
        page_idx = key0 // cutlass.Int32(PAGE_SIZE)
        rd = cute.math.min(cute.math.max(page_idx, cutlass.Int32(0)), last)
        page = cutlass.Int32(bt[batch, rd])
        ok = (blk >= cutlass.Int32(0)) & (key0 < eff_seqlen_kv) & (page_idx < max_pages) & (page >= cutlass.Int32(0))
        pages.append(_select_i32(ok, page, cutlass.Int32(-1)))
    return cutlass.Vector.from_elements(tuple(pages), cutlass.Int32)


@cute.jit
def _gather_fill(desc, sKV_raw, stage_idx, mbar, w, blks, pages, batch_rows, rows_per_page, head_rows, eff_seqlen_kv, col_base, hint):
    """ONE warp's share of one K or V stage: BLOCKS_PER_WARP blocks x GATHER_BOXES column boxes = 16 ``tma_gather4`` from
    the ELECTED lane (the caller elects).  Block ``q`` of this warp = quad ``BLOCKS_PER_WARP w + q`` of the tile, landing at
    ``box * 16 KiB + quad * 512 B`` inside the stage -- a 512-B-aligned quad in a 1024-B-aligned sub-box, so the swizzle the
    hardware applies equals a tiled load's (the tile then reads under the dense descriptors).  Row coordinates carry the
    batch term (``batch x S_kv``, the 2-D map's rows are tokens of every batch) on the dense arm, and on the PAGED arm the
    pool row ``page x rows_per_page + head_rows + (4 blk) % PAGE_SIZE`` (``pages`` from ``_gather_block_pages``; ``head_rows``
    = ``head x rows_per_head``, 0 for an NHD pool whose head lives in the column); a ``-1`` block (or page), a block past
    the batch's length or the straddling rows of its last block take row ``-1`` -> TMA zero-fills and still credits the
    bytes, so no foreign batch's / sequence's row (or its NaN) ever reaches an MMA under P = 0."""
    stage_off = stage_idx * cutlass.Int32(kvBufferElems)
    quad0 = w * cutlass.Int32(BLOCKS_PER_WARP)
    for q in cutlass.range_constexpr(BLOCKS_PER_WARP):
        blk = cutlass.Int32(blks[q])
        key0 = blk * cutlass.Int32(BLOCK_SIZE)
        if cutlass.const_expr(PAGED_KV):
            page = cutlass.Int32(pages[q])
            in_page = key0 - (key0 // cutlass.Int32(PAGE_SIZE)) * cutlass.Int32(PAGE_SIZE)
            base = page * rows_per_page + head_rows + in_page
            live = page >= cutlass.Int32(0)
        else:
            base = batch_rows + key0
            live = blk >= cutlass.Int32(0)
        rows = []
        for r in cutlass.range_constexpr(BLOCK_SIZE):
            rows.append(_select_i32(live & ((key0 + cutlass.Int32(r)) < eff_seqlen_kv), base + cutlass.Int32(r), cutlass.Int32(-1)))
        quad_elems = (quad0 + cutlass.Int32(q)) * cutlass.Int32(BLOCK_SIZE * GATHER_BOX_ELEMS)
        for b in cutlass.range_constexpr(GATHER_BOXES):
            tma_gather4(
                desc,
                sKV_raw.subview(stage_off + cutlass.Int32(b * SUB_BOX_ELEMS) + quad_elems),
                mbar,
                col_base + cutlass.Int32(b * GATHER_BOX_ELEMS),
                rows[0],
                rows[1],
                rows[2],
                rows[3],
                cta_group=1,
                l2_hint=hint,
            )


@cute.jit
def _gather_warp_group(
    tma_k_desc,
    tma_v_desc,
    sKV_raw,
    sIds_raw,
    bars,
    sched,
    block_lens_tensor,
    seq_kv_lens_tensor,
    seqlen_q,
    seqlen_kv,
    k_head_stride,
    v_head_stride,
    w,
    tok0,
    head0,
    batch0,
    block_table_tensor: Optional[cute.Tensor] = None,
    max_pages=None,
    k_rows_per_page=None,
    k_rows_per_head=None,
    v_rows_per_page=None,
    v_rows_per_head=None,
):
    hint = opaque_i64(TMA_L2_EVICT_LAST)
    # kv_state: the stage to fill next (producer of mb_kv_full) == the mb_kv_empty wait (pre-armed: the first STAGES_KV
    # fills pass fresh barriers); advanced per LOAD in the K(0), K(1), V(0), K(2), V(1), ... order the MMA warp consumes,
    # CARRIED across items (never recomputed from a per-item tile index -- barrier rows 5 / 6).
    kv_state = PipelineState.start(phase=1)
    ids_state = PipelineState.start(phase=0)
    tok, head, batch = tok0, head0, batch0
    is_valid = cutlass.Int32(1)
    sched_state = PipelineState.start()
    while is_valid > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), 1)
        pos, count, has_open, b_open, dead, n_tiles, eff_seqlen_kv = _item_bounds(tok, batch, block_lens_tensor, seq_kv_lens_tensor, seqlen_q, seqlen_kv)
        batch_rows = batch * seqlen_kv
        col_k = head * k_head_stride
        col_v = head * v_head_stride
        # PAGED arm: the per-item row terms of the head (0 for an NHD pool: its head lives in the column coordinate).
        head_rows_k = head * k_rows_per_head if cutlass.const_expr(PAGED_KV) else None
        head_rows_v = head * v_rows_per_head if cutlass.const_expr(PAGED_KV) else None
        slot_base = ids_state.idx * cutlass.Int32(IDS_SLOT_WORDS)
        bars.mb_ids_full[ids_state.idx].wait(ids_state.phase)

        # Ring order K(0), K(1), V(0), K(2), V(1), ...: K runs one tile ahead of V so BMM1(t+1) overlaps the softmax of t.
        # PAGED arm: the block-table lookups of a tile's quad ride with its ids (before the ring wait), K and V share them.
        blks_cur = _gather_block_ids(sIds_raw, slot_base, cutlass.Int32(0), w, count, has_open, b_open, dead)
        pages_cur = _gather_block_pages(blks_cur, block_table_tensor, batch, max_pages, eff_seqlen_kv) if cutlass.const_expr(PAGED_KV) else None
        bars.mb_kv_empty[kv_state.idx].wait(kv_state.phase, spin=SPIN_RING_WAITS)
        bars.mb_kv_full[kv_state.idx].arrive(n_bytes=KV_TX_BYTES_PER_WARP, pred=nvvm.elect_sync())
        if nvvm.elect_sync():
            _gather_fill(
                tma_k_desc,
                sKV_raw,
                kv_state.idx,
                bars.mb_kv_full[kv_state.idx].smem_ptr,
                w,
                blks_cur,
                pages_cur,
                batch_rows,
                k_rows_per_page,
                head_rows_k,
                eff_seqlen_kv,
                col_k,
                hint,
            )
        kv_state = advance(kv_state, STAGES_KV)
        for i in cutlass.range(0, n_tiles, 1, unroll=1):
            # The next tile's ids are read before this iteration's ring waits (an in-bounds address even past the last tile).
            t_next = cute.math.min(i + cutlass.Int32(1), n_tiles - cutlass.Int32(1))
            blks_next = _gather_block_ids(sIds_raw, slot_base, t_next, w, count, has_open, b_open, dead)
            pages_next = _gather_block_pages(blks_next, block_table_tensor, batch, max_pages, eff_seqlen_kv) if cutlass.const_expr(PAGED_KV) else None
            if i + cutlass.Int32(1) < n_tiles:
                bars.mb_kv_empty[kv_state.idx].wait(kv_state.phase, spin=SPIN_RING_WAITS)
                bars.mb_kv_full[kv_state.idx].arrive(n_bytes=KV_TX_BYTES_PER_WARP, pred=nvvm.elect_sync())
                if nvvm.elect_sync():
                    _gather_fill(
                        tma_k_desc,
                        sKV_raw,
                        kv_state.idx,
                        bars.mb_kv_full[kv_state.idx].smem_ptr,
                        w,
                        blks_next,
                        pages_next,
                        batch_rows,
                        k_rows_per_page,
                        head_rows_k,
                        eff_seqlen_kv,
                        col_k,
                        hint,
                    )
                kv_state = advance(kv_state, STAGES_KV)
            bars.mb_kv_empty[kv_state.idx].wait(kv_state.phase, spin=SPIN_RING_WAITS)
            bars.mb_kv_full[kv_state.idx].arrive(n_bytes=KV_TX_BYTES_PER_WARP, pred=nvvm.elect_sync())
            if nvvm.elect_sync():
                _gather_fill(
                    tma_v_desc,
                    sKV_raw,
                    kv_state.idx,
                    bars.mb_kv_full[kv_state.idx].smem_ptr,
                    w,
                    blks_cur,
                    pages_cur,
                    batch_rows,
                    v_rows_per_page,
                    head_rows_v,
                    eff_seqlen_kv,
                    col_v,
                    hint,
                )
            kv_state = advance(kv_state, STAGES_KV)
            blks_cur = blks_next
            pages_cur = pages_next

        # The last read of this item's list slot (row 4: ONE lane per consuming warp).
        if nvvm.elect_sync():
            bars.mb_ids_empty[ids_state.idx].arrive()
        ids_state = advance(ids_state, STAGES_IDS)

        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        t0, t1, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        tok, head, batch = _decode_payload(t0, t1)
        is_valid = cute.arch.make_warp_uniform(nxt_v)
        sched_state = advance(sched_state, SCHEDULER_STAGES)

    # Exit drain (P6): the MMA warp's last STAGES_KV kv_empty commits need a waiter; the carried state names their phases.
    for _ in cutlass.range_constexpr(STAGES_KV):
        bars.mb_kv_empty[kv_state.idx].wait(kv_state.phase)
        kv_state = advance(kv_state, STAGES_KV)


# === MMA warp (12) ========================================================================================================


@cute.jit
def _mma_warp_group(sQ, sK, sVt, sP, tmem_ptr_i32, bars, sched, block_lens_tensor, seq_kv_lens_tensor, seqlen_q, seqlen_kv, tok0, head0, batch0):
    tmem_alloc(tmem_ptr_i32, TMEM_COLS, CTA_GROUP_KIND)
    nvvm.barrier_cta_arrive(_BAR_TMEM, BAR_TMEM_THREADS)

    tmem_raw = nvvm.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)

    idesc_qk = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=N_Q,
        m_dim=TILE_N,
        k_dim=0,
    )
    idesc_pv = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=N_Q,
        m_dim=128,
        a_major=1,
        b_major=1,
        k_dim=0,
    )
    # BMM1: S^T = K . Q^T -- K [128 keys x TILE_K] K-major, Q^T [N_Q x TILE_K] K-major.
    bmm1_desc = MmaDesc(
        M=TILE_N,
        N=N_Q,
        K=TILE_K,
        bpe_a=BPE,
        bpe_b=BPE,
        tile_k_hw=CFG.TILE_K_HW,
        btranspose=False,
        cta_group=1,
        idesc=idesc_qk,
        kind=MMA_KIND,
    )
    # BMM2: O^T = V^T . P^T -- V^T [128 d x 128 keys] MN-major (d contiguous), P^T [N_Q x 128 keys] MN-major.
    bmm2_desc = MmaDesc(
        M=128,
        N=N_Q,
        K=TILE_N,
        bpe_a=BPE,
        bpe_b=BPE,
        tile_k_hw=CFG.TILE_K_HW,
        atranspose=True,
        btranspose=True,
        cta_group=1,
        idesc=idesc_pv,
        kind=MMA_KIND,
    )

    # Every ring slot and parity below comes from a PipelineState CARRIED across work items (barrier rows 1, 2, 5-10):
    #   q_state      -- mb_q_full wait (consumer) and the slot of the mb_q_empty commit, once per item;
    #   kv_state     -- mb_kv_full wait + the slot of the mb_kv_empty commit, per LOAD in the gather warps' order;
    #   s_state      -- the S^T slot of BMM1(t) = the mb_s_full commit slot, per tile;
    #   s_empty_state-- mb_s_empty wait (pre-armed: BMM1(0), BMM1(1) pass fresh), per tile;
    #   p_state      -- mb_p_full wait, per tile;  bmm2_state -- the mb_bmm2_done commit slot, per tile.
    q_state = PipelineState.start(phase=0)
    kv_state = PipelineState.start(phase=0)
    s_state = PipelineState.start(phase=0)
    s_empty_state = PipelineState.start(phase=1)
    p_state = PipelineState.start(phase=0)
    bmm2_state = PipelineState.start(phase=0)

    tok, head, batch = tok0, head0, batch0
    is_valid = cutlass.Int32(1)
    sched_state = PipelineState.start()
    while is_valid > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), 1)
        pos, count, has_open, b_open, dead, n_tiles, eff_seqlen_kv = _item_bounds(tok, batch, block_lens_tensor, seq_kv_lens_tensor, seqlen_q, seqlen_kv)

        bars.mb_q_full[q_state.idx].wait(q_state.phase)
        desc_Q = sQ[q_state.idx].desc()

        # Prologue: BMM1(0).  The item's LAST BMM1 (here when n_tiles == 1) also commits mb_q_empty (row 2): the commit tracks
        # every prior MMA of this thread, so it fires once the last read of sQ[slot] is done.
        bars.mb_kv_full[kv_state.idx].wait(kv_state.phase, spin=SPIN_RING_WAITS)
        bars.mb_s_empty[s_empty_state.idx].wait(s_empty_state.phase, spin=SPIN_RING_WAITS)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        mma_ss(bmm1_desc, sK[kv_state.idx].desc(), desc_Q, tmem_raw.subview(cutlass.Int32(S_ACC_OFF[0]) + s_state.idx * cutlass.Int32(N_Q)))
        elect_p = nvvm.elect_sync()
        bars.mb_s_full[s_state.idx].arrive(cta_group=1, pred=elect_p)
        bars.mb_kv_empty[kv_state.idx].arrive(cta_group=1, pred=elect_p)
        bars.mb_q_empty[q_state.idx].arrive(cta_group=1, pred=elect_p & (n_tiles == cutlass.Int32(1)))
        kv_state = advance(kv_state, STAGES_KV)
        s_state = advance(s_state, 2)
        s_empty_state = advance(s_empty_state, 2)

        for i in cutlass.range(0, n_tiles, 1, unroll=1):
            # BMM1(i + 1) ahead of BMM2(i): the softmax of tile i overlaps the next score tile, and the K slot is released as
            # early as possible.
            if i + cutlass.Int32(1) < n_tiles:
                bars.mb_kv_full[kv_state.idx].wait(kv_state.phase, spin=SPIN_RING_WAITS)
                bars.mb_s_empty[s_empty_state.idx].wait(s_empty_state.phase, spin=SPIN_RING_WAITS)
                nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
                mma_ss(bmm1_desc, sK[kv_state.idx].desc(), desc_Q, tmem_raw.subview(cutlass.Int32(S_ACC_OFF[0]) + s_state.idx * cutlass.Int32(N_Q)))
                elect_p1 = nvvm.elect_sync()
                bars.mb_s_full[s_state.idx].arrive(cta_group=1, pred=elect_p1)
                bars.mb_kv_empty[kv_state.idx].arrive(cta_group=1, pred=elect_p1)
                bars.mb_q_empty[q_state.idx].arrive(cta_group=1, pred=elect_p1 & ((i + cutlass.Int32(2)) == n_tiles))
                kv_state = advance(kv_state, STAGES_KV)
                s_state = advance(s_state, 2)
                s_empty_state = advance(s_empty_state, 2)

            # BMM2(i): V(i) is the next load of the ring, P^T(i) the softmax's publish.
            bars.mb_kv_full[kv_state.idx].wait(kv_state.phase, spin=SPIN_RING_WAITS)
            bars.mb_p_full[p_state.idx].wait(p_state.phase, spin=SPIN_RING_WAITS)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
            desc_P = sP[p_state.idx].desc()
            accum = cutlass.Boolean(i > cutlass.Int32(0))
            for blk in cutlass.range_constexpr(O_BLOCKS):
                mma_ss(
                    bmm2_desc,
                    sVt[kv_state.idx].shifted(blk * 2 * TILE_N * GRANU_ELEMS).desc(),
                    desc_P,
                    tmem_raw.subview(cutlass.Int32(O_OFF[blk])),
                    accumulate=accum,
                )
            elect_p2 = nvvm.elect_sync()
            bars.mb_bmm2_done[bmm2_state.idx].arrive(cta_group=1, pred=elect_p2)
            bars.mb_kv_empty[kv_state.idx].arrive(cta_group=1, pred=elect_p2)
            kv_state = advance(kv_state, STAGES_KV)
            p_state = advance(p_state, 2)
            bmm2_state = advance(bmm2_state, 2)

        q_state = advance(q_state, STAGES_Q)

        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        t0, t1, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        tok, head, batch = _decode_payload(t0, t1)
        is_valid = cute.arch.make_warp_uniform(nxt_v)
        sched_state = advance(sched_state, SCHEDULER_STAGES)

    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, TMEM_COLS, CTA_GROUP_KIND)


# === SOFTMAX warps (0-3): 128 key lanes, 16 columns = the item's query heads ============================================


@cute.jit
def _warp_reduce_max(x):
    for off in cutlass.range_constexpr(5):
        x = cute.math.max(x, cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, x, 16 >> off, 31, kind=nvvm.Shfl.BFLY)))
    return x


@cute.jit
def _warp_reduce_sum(x):
    for off in cutlass.range_constexpr(5):
        x = x + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, x, 16 >> off, 31, kind=nvvm.Shfl.BFLY))
    return x


@cute.jit
def _softmax_warp_group(
    tmem_ptr_i32,
    bars,
    sched,
    red_smem,
    sP_raw,
    sIds_raw,
    o_tensor,
    lse_tensor: Optional[cute.Tensor],
    block_lens_tensor: Optional[cute.Tensor],
    seq_kv_lens_tensor,
    seqlen_q,
    seqlen_kv,
    tok0,
    head0,
    batch0,
    scale_log2: cutlass.Float32,
):
    nvvm.barrier_cta_sync(barrier_id=_BAR_TMEM, thread_count=BAR_TMEM_THREADS)
    tidx = cute.arch.thread_idx()[0]
    sm_warp = tidx // cutlass.Int32(32)  # 0 .. SOFTMAX_WARPS - 1
    lane = tidx  # key row of S^T / P^T, d row of O^T (one 16-column group)
    tmem_base = tmem_ptr_i32.load()
    red_ptr = Pointer(red_smem.data_ptr(), dtype=cutlass.Float32)

    NEG_INF = cutlass.Float32(float("-inf"))
    ZERO = cutlass.Float32(0.0)
    ONE = cutlass.Float32(1.0)
    LN2_F = cutlass.Float32(LN2)
    TINY = cutlass.Float32(1e-30)

    # Carried states (barrier rows 3, 7-10): t_state = the per-tile slot / parity of mb_s_full (wait), mb_s_empty and
    # mb_p_full (this group's arrives); bmm2_state = the mb_bmm2_done wait, one per tile, one iteration behind; ids_state =
    # the mb_ids_full wait and the mb_ids_empty arrive slot, once per item.  None is ever recomputed from a per-item index.
    t_state = PipelineState.start(phase=0)
    bmm2_state = PipelineState.start(phase=0)
    ids_state = PipelineState.start(phase=0)

    tok, head, batch = tok0, head0, batch0
    is_valid = cutlass.Int32(1)
    sched_state = PipelineState.start()
    while is_valid > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), 1)
        pos, count, has_open, b_open, dead, n_tiles, eff_seqlen_kv = _item_bounds(tok, batch, block_lens_tensor, seq_kv_lens_tensor, seqlen_q, seqlen_kv)
        slot_base = ids_state.idx * cutlass.Int32(IDS_SLOT_WORDS)
        bars.mb_ids_full[ids_state.idx].wait(ids_state.phase)

        # Running column state: max in the scaled log2 domain (-inf = no live key yet), the thread's partial sum.
        m_vec = cutlass.Vector.from_elements(tuple(NEG_INF for _ in range(COLS)), cutlass.Float32)
        l_vec = cutlass.Vector.from_elements(tuple(ZERO for _ in range(COLS)), cutlass.Float32)

        for i in cutlass.range(0, n_tiles, 1, unroll=1):
            par = t_state.idx
            s_off = cutlass.Int32(S_ACC_OFF[0]) + par * cutlass.Int32(N_Q)

            bars.mb_s_full[par].wait(t_state.phase, spin=SPIN_RING_WAITS)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
            s_raw = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(tmem_base + s_off, cutlass.Float32), num=COLS)
            # The LOAD wait is what ORDERS the arrive after the read (frost-kernels.md section 3).
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            bars.mb_s_empty[par].arrive()

            # Per-thread validity (header: the in-block mask depends on the key index alone -- all 16 columns share pos).
            # Lane l of tile i holds key 4 blk + l % 4 with blk = the list entry 32 i + l // 4 (tail / -1 by the SELECT).
            idx = i * cutlass.Int32(BLOCKS_PER_TILE) + (lane >> cutlass.Int32(2))
            ld_idx = cute.math.min(idx, cutlass.Int32(BLOCK_TOPK - 1))
            word = cutlass.Int32(sIds_raw.load(slot_base + ld_idx))
            blk = _block_id_at(idx, word, count, has_open, b_open, dead)
            key_abs = blk * cutlass.Int32(BLOCK_SIZE) + (lane & cutlass.Int32(BLOCK_SIZE - 1))
            valid = (blk >= cutlass.Int32(0)) & (key_abs <= pos) & (key_abs < eff_seqlen_kv)
            s_cols = []
            for j in cutlass.range_constexpr(COLS):
                s_cols.append(_select_f32(valid, cutlass.Float32(s_raw[j]) * scale_log2, NEG_INF))

            # Column max over the tile's 128 keys: butterfly inside the warp, then the group's four warps exchange through
            # SMEM (slot = the tile's ring parity, so the single barrier per tile also orders the next reuse of the slot).
            red_base = par * cutlass.Int32(SOFTMAX_WARPS * COLS) + sm_warp * cutlass.Int32(COLS)
            col_max = []
            for j in cutlass.range_constexpr(COLS):
                cm = _warp_reduce_max(s_cols[j])
                (red_ptr + (red_base + cutlass.Int32(j))).store(cm)
                col_max.append(cm)
            nvvm.barrier_cta_sync(barrier_id=_BAR_SOFTMAX, thread_count=BAR_SOFTMAX_THREADS)
            red_read = par * cutlass.Int32(SOFTMAX_WARPS * COLS)
            tile_max = []
            for j in cutlass.range_constexpr(COLS):
                tm = col_max[j]
                for w in cutlass.range_constexpr(SOFTMAX_WARPS):
                    tm = cute.math.max(tm, cutlass.Float32((red_ptr + (red_read + cutlass.Int32(w * COLS + j))).load()))
                tile_max.append(tm)

            # Online update per column (identical in every lane: alpha is uniform, so O^T's rescale is exact).
            m_new = []
            l_new = []
            alpha = []
            all_one = None
            p_cols = []
            for j in cutlass.range_constexpr(COLS):
                m_old_j = cutlass.Float32(m_vec[j])
                m_new_j = cute.math.max(m_old_j, tile_max[j])
                ms_old = row_max_for_exp2(m_old_j)
                ms_new = row_max_for_exp2(m_new_j)
                alpha_j = cute.math.exp2(cute.math.min(ms_old - ms_new, ZERO), fastmath=True)
                p_j = cute.math.exp2(s_cols[j] - ms_new, fastmath=True)
                l_new.append(cutlass.Float32(l_vec[j]) * alpha_j + p_j)
                m_new.append(m_new_j)
                alpha.append(alpha_j)
                p_cols.append(p_j)
                is_one = alpha_j == ONE
                all_one = is_one if all_one is None else (all_one & is_one)
            m_vec = cutlass.Vector.from_elements(tuple(m_new), cutlass.Float32)
            l_vec = cutlass.Vector.from_elements(tuple(l_new), cutlass.Float32)

            # P^T(t) -> sP[par] as packed half pairs, row = the thread's key, swizzled the way the BMM2 B descriptor decodes
            # it.  WAR on the slot: BMM2(t-2) last read it, and its completion was waited at iteration t-1 (below), BEFORE
            # this store -- the store-then-wait order inside an iteration is exactly sufficient (header row 10).
            p_words = []
            for w in cutlass.range_constexpr(P_ROW_WORDS):
                p_words.append(fp32_to_fp16(p_cols[2 * w], p_cols[2 * w + 1], dtype=STORAGE_DTYPE))
            p_row_base = par * cutlass.Int32(pBufferWords) + lane * cutlass.Int32(P_ROW_WORDS)
            for c in cutlass.range_constexpr(P_CHUNKS):
                chunk = cutlass.Vector.from_elements(tuple(p_words[4 * c + k] for k in range(4)), cutlass.Int32)
                p_ptr = Pointer(sP_raw.subview(p_row_base + cutlass.Int32(4 * c)).data_ptr(), dtype=cutlass.Int32)
                p_ptr.store_swizzled(chunk, _P_SMEM_SWIZZLE, alignment=16)
            nvvm.fence_proxy("async.shared", space="cta")

            # O^T(t-1) is complete once BMM2(t-1) commits; rescale this group's columns by alpha before BMM2(t) accumulates
            # on top.  Skipped when no column's max moved.
            if i > cutlass.Int32(0):
                bars.mb_bmm2_done[bmm2_state.idx].wait(bmm2_state.phase, spin=SPIN_RING_WAITS)
                bmm2_state = advance(bmm2_state, 2)
                nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
                if ~all_one:
                    for blk_o in cutlass.range_constexpr(O_BLOCKS):
                        o_ptr = nvvm.make_tmem_ptr(tmem_base + cutlass.Int32(O_OFF[blk_o]), cutlass.Float32)
                        o_vals = nvvm.tcgen05_ld("32x32b", o_ptr, num=COLS)
                        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                        o_scaled = cutlass.Vector.from_elements(tuple(cutlass.Float32(o_vals[j]) * alpha[j] for j in range(COLS)), cutlass.Float32)
                        nvvm.tcgen05_st("32x32b", o_ptr, o_scaled)
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
            # Publish P^T(t) (and the rescaled O^T) to the MMA warp.
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            bars.mb_p_full[par].arrive()
            t_state = advance(t_state, 2)

        # The last read of this item's list slot (row 4: ONE lane per consuming warp).
        if nvvm.elect_sync():
            bars.mb_ids_empty[ids_state.idx].arrive()
        ids_state = advance(ids_state, STAGES_IDS)

        # --- epilogue: the item's last BMM2 must land before O^T is read (one wait per tile: this is the last tile's).
        bars.mb_bmm2_done[bmm2_state.idx].wait(bmm2_state.phase)
        bmm2_state = advance(bmm2_state, 2)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)

        # Column sums over the 128 lanes (once per item), via the epilogue slot.
        red_epi = cutlass.Int32(2 * SOFTMAX_WARPS * COLS) + sm_warp * cutlass.Int32(COLS)
        for j in cutlass.range_constexpr(COLS):
            (red_ptr + (red_epi + cutlass.Int32(j))).store(_warp_reduce_sum(cutlass.Float32(l_vec[j])))
        nvvm.barrier_cta_sync(barrier_id=_BAR_SOFTMAX, thread_count=BAR_SOFTMAX_THREADS)
        red_epi_read = cutlass.Int32(2 * SOFTMAX_WARPS * COLS)

        lse_cols = []
        inv_cols = []
        dead_cols = []
        for j in cutlass.range_constexpr(COLS):
            l_tot = ZERO
            for w in cutlass.range_constexpr(SOFTMAX_WARPS):
                l_tot = l_tot + cutlass.Float32((red_ptr + (red_epi_read + cutlass.Int32(w * COLS + j))).load())
            m_raw = cutlass.Float32(m_vec[j])  # -inf while the column has no live key
            m_nat = row_max_for_exp2(m_raw) * LN2_F
            row_dead = l_tot <= ZERO
            # The denominator floor sits inside the log and the reciprocal ONLY and is always followed by the row_dead
            # SELECT, so neither LSE nor O ever carries it (sdpa-invariants.md sections 2-4): keep these lines together.
            lse_j = m_nat + cute.math.log(cute.math.max(l_tot, TINY), fastmath=True)
            inv_j = ONE / cute.math.max(l_tot, TINY)
            lse_j = _select_f32(row_dead, NEG_INF, lse_j)
            inv_j = _select_f32(row_dead, ZERO, inv_j)
            if cutlass.const_expr(CFG.STATS_LOG2):
                lse_j = lse_j * cutlass.Float32(LOG2E)
            lse_cols.append(lse_j)
            inv_cols.append(inv_j)
            dead_cols.append(row_dead)

        # Stores: column j = query head (head * G + j) of token tok; the zero tail columns j >= G are never stored, and a
        # padded Q row (pos past the sequence) stores nothing.
        live_row = pos < seqlen_q
        head_base = head * cutlass.Int32(G)
        if cutlass.const_expr(lse_tensor is not None):
            lse_arr = cutlass.make_array_view(lse_tensor)
            for j in cutlass.range_constexpr(Q_BOX_ROWS):
                if (lane == cutlass.Int32(j)) & live_row:
                    lse_arr[batch, head_base + cutlass.Int32(j), tok] = lse_cols[j]

        # O[q, d] = O^T[d, q] / l[q]: the thread index is d (and d + 128).
        oo = cutlass.make_array_view(o_tensor)
        D_V = cutlass.const_expr(o_tensor.shape[3])
        for blk_o in cutlass.range_constexpr(O_BLOCKS):
            if cutlass.const_expr(blk_o * 128 < D_V):
                d_idx = lane + cutlass.Int32(blk_o * 128)
                o_vals = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(tmem_base + cutlass.Int32(O_OFF[blk_o]), cutlass.Float32), num=COLS)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                if (d_idx < cutlass.Int32(D_V)) & live_row:
                    for j in cutlass.range_constexpr(Q_BOX_ROWS):
                        val = _select_f32(dead_cols[j], ZERO, cutlass.Float32(o_vals[j]) * inv_cols[j])
                        o_row = oo[batch, tok, head_base + cutlass.Int32(j), :]
                        o_row[d_idx] = val.to(o_tensor.element_type)

        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        t0, t1, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        tok, head, batch = _decode_payload(t0, t1)
        is_valid = cute.arch.make_warp_uniform(nxt_v)
        sched_state = advance(sched_state, SCHEDULER_STAGES)

    bars.mb_tmem_dealloc.arrive()


# === the kernel ===========================================================================================================


@cute.kernel
def _kernel(
    tma_q_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_k_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_v_desc: cutlass.GridConstant[tmap.TensorMap],
    o_tensor: cute.Tensor,
    lse_tensor: Optional[cute.Tensor],
    block_ids_tensor: cute.Tensor,
    block_lens_tensor: Optional[cute.Tensor],
    seq_kv_lens_tensor: cute.Tensor,
    seqlen_q: cutlass.Int32,
    seqlen_kv: cutlass.Int32,
    k_head_stride: cutlass.Int32,
    v_head_stride: cutlass.Int32,
    scale_softmax_log2: cutlass.Float32,
    block_table_tensor: Optional[cute.Tensor],
    max_pages: cutlass.Int32,
    k_rows_per_page: cutlass.Int32,
    k_rows_per_head: cutlass.Int32,
    v_rows_per_page: cutlass.Int32,
    v_rows_per_head: cutlass.Int32,
) -> None:
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx, _, _ = cute.arch.thread_idx()
    # The grid IS the work list: x = query token, y = KV head, z = batch; a CTA's first item is its own position, later
    # ones come from the CLC scheduler's cancelled CTAs (decoded by _decode_payload with the same (x, y, z) meaning).
    bidx = cute.arch.block_idx()[0]
    bidy = cute.arch.block_idx()[1]
    bidz = cute.arch.block_idx()[2]

    # SMEM in the header table's order: the descriptor-read operands first (1024-B aligned), the register-addressed buffers last.
    sQ_raw = cutlass.Array(STORAGE_DTYPE, STAGES_Q * qBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sP_raw = cutlass.Array(cutlass.Int32, 2 * pBufferWords, alignment=1024, space=cutlass.AddressSpace.smem)
    sKV_raw = cutlass.Array(STORAGE_DTYPE, STAGES_KV * kvBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sIds_raw = cutlass.Array(cutlass.Int32, STAGES_IDS * IDS_SLOT_WORDS, alignment=16, space=cutlass.AddressSpace.smem)
    red_smem = cutlass.Array(cutlass.Float32, RED_WORDS, alignment=16, space=cutlass.AddressSpace.smem)
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, alignment=16, space=cutlass.AddressSpace.smem)

    # Q^T: the K-major B operand of BMM1 (N_Q rows x TILE_K per slot), one TMA sub-tile (N_Q rows x 128 B) per 64-wide granule.
    sQ = SmemTile(
        base=sQ_raw,
        elems_per_stage=qBufferElems,
        stages=STAGES_Q,
        leading_byte_offset=0,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_QK,
        tma_loads_per_tile=QK_ITERS,
        tma_granu_elems=GRANU_ELEMS,
        tma_subtile_stride_elems=N_Q * GRANU_ELEMS,
        desc_version=DESC_VERSION,
    )
    # The gathered K/V ring, seen as the K-major A operand of BMM1 (TILE_N rows x TILE_K) ...
    sK = SmemTile(
        base=sKV_raw,
        elems_per_stage=kvBufferElems,
        stages=STAGES_KV,
        leading_byte_offset=0,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_QK,
        desc_version=DESC_VERSION,
    )
    # ... and as the MN-major A operand of BMM2 (V^T: d contiguous, LBO steps the 64-wide d atoms).  Same bytes, two
    # descriptors -- the gather map writes under the same SW128 the descriptors decode.
    sVt = SmemTile(
        base=sKV_raw,
        elems_per_stage=kvBufferElems,
        stages=STAGES_KV,
        leading_byte_offset=LEADING_BYTE_OFFSET_VT,
        stride_byte_offset=STRIDE_BYTE_OFFSET_VT,
        layout=SMEM_LAYOUT_QK,
        desc_version=DESC_VERSION,
    )
    # P^T: the MN-major B operand of BMM2, two buffers of TILE_N key rows.
    sP = SmemTile(
        base=sP_raw,
        elems_per_stage=pBufferWords,
        stages=2,
        leading_byte_offset=0,
        stride_byte_offset=STRIDE_BYTE_OFFSET_P,
        layout=SMEM_LAYOUT_P,
        desc_version=DESC_VERSION,
    )

    bars = make_sparse_bars()
    sched = Sched(
        **{
            "mb_scheduler": cutlass.Array(cutlass.Int64, SCHEDULER_STAGES, alignment=16, space=cutlass.AddressSpace.smem),
            "mb_read_tile_id": cutlass.Array(cutlass.Int64, SCHEDULER_STAGES, alignment=16, space=cutlass.AddressSpace.smem),
            "tile_id_smem": cutlass.Array(cutlass.Int32, SCHEDULER_STAGES * 8, alignment=16, space=cutlass.AddressSpace.smem),
            "bidx_init": bidx,
            "bidy_init": bidy,
            "bidz_init": bidz,
        }
    )

    # P4: ONE warp, ONE lane initialises every stage of every ring (.init() is per stage); fence + CTA sync outside.
    if warp_idx == 0:
        if nvvm.elect_sync():
            for s in cutlass.range_constexpr(STAGES_Q):
                bars.mb_q_full[s].init()
                bars.mb_q_empty[s].init()
            for s in cutlass.range_constexpr(STAGES_IDS):
                bars.mb_ids_full[s].init()
                bars.mb_ids_empty[s].init()
            for s in cutlass.range_constexpr(STAGES_KV):
                bars.mb_kv_full[s].init()
                bars.mb_kv_empty[s].init()
            for p in cutlass.range_constexpr(2):
                bars.mb_s_full[p].init()
                bars.mb_s_empty[p].init()
                bars.mb_p_full[p].init()
                bars.mb_bmm2_done[p].init()
            bars.mb_tmem_dealloc.init()
            for s in cutlass.range_constexpr(SCHEDULER_STAGES):
                nvvm.mbarrier_init(sched.mb_scheduler.subview(s), CFG.ONE_LANE)
                nvvm.mbarrier_init(sched.mb_read_tile_id.subview(s), CFG.READ_TILE_ARRIVERS)

    if cutlass.const_expr(Q_BOX_ROWS < N_Q):
        # Zero the Q^T tail rows the TMA box never covers (rows G..15 of BOTH slots) once per CTA, so their never-stored
        # columns stay finite; the generic stores are published to the async proxy before any MMA reads the slots.
        if warp_idx < SOFTMAX_WARPS:
            zero4 = cutlass.Vector.from_elements(tuple(cutlass.Int32(0) for _ in range(4)), cutlass.Int32)
            tail_chunks = (N_Q - Q_BOX_ROWS) * 8  # 16 B chunks per sub-tile
            n_chunks = STAGES_Q * QK_ITERS * tail_chunks
            for it in cutlass.range_constexpr((n_chunks + SOFTMAX_LANES - 1) // SOFTMAX_LANES):
                chunk = cutlass.Int32(it * SOFTMAX_LANES) + tidx
                if chunk < cutlass.Int32(n_chunks):
                    slot = chunk // cutlass.Int32(QK_ITERS * tail_chunks)
                    rem = chunk % cutlass.Int32(QK_ITERS * tail_chunks)
                    sub = rem // cutlass.Int32(tail_chunks)
                    within = rem % cutlass.Int32(tail_chunks)
                    elem_off = (
                        slot * cutlass.Int32(qBufferElems)
                        + sub * cutlass.Int32(N_Q * GRANU_ELEMS)
                        + cutlass.Int32(Q_BOX_ROWS * GRANU_ELEMS)
                        + within * cutlass.Int32(8)
                    )
                    Pointer(sQ_raw.subview(elem_off).data_ptr(), dtype=cutlass.Int32).store(zero4, alignment=16)
            nvvm.fence_proxy("async.shared", space="cta")

    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()
    # No CTA-wide barrier-0 sync may follow this one: the spare warp exits here.

    if warp_idx < SOFTMAX_WARPS:
        _softmax_warp_group(
            tmem_ptr_i32=tmem_ptr_i32,
            bars=bars,
            sched=sched,
            red_smem=red_smem,
            sP_raw=sP_raw,
            sIds_raw=sIds_raw,
            o_tensor=o_tensor,
            lse_tensor=lse_tensor,
            block_lens_tensor=block_lens_tensor,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            tok0=bidx,
            head0=bidy,
            batch0=bidz,
            scale_log2=scale_softmax_log2,
        )
    elif warp_idx < GATHER_WARP_BASE + GATHER_WARPS:
        _gather_warp_group(
            tma_k_desc=tma_k_desc,
            tma_v_desc=tma_v_desc,
            sKV_raw=sKV_raw,
            sIds_raw=sIds_raw,
            bars=bars,
            sched=sched,
            block_lens_tensor=block_lens_tensor,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            k_head_stride=k_head_stride,
            v_head_stride=v_head_stride,
            w=warp_idx - cutlass.Int32(GATHER_WARP_BASE),
            tok0=bidx,
            head0=bidy,
            batch0=bidz,
            block_table_tensor=block_table_tensor,
            max_pages=max_pages,
            k_rows_per_page=k_rows_per_page,
            k_rows_per_head=k_rows_per_head,
            v_rows_per_page=v_rows_per_page,
            v_rows_per_head=v_rows_per_head,
        )
    elif warp_idx == MMA_WARP_ID:
        _mma_warp_group(
            sQ=sQ,
            sK=sK,
            sVt=sVt,
            sP=sP,
            tmem_ptr_i32=tmem_ptr_i32,
            bars=bars,
            sched=sched,
            block_lens_tensor=block_lens_tensor,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            tok0=bidx,
            head0=bidy,
            batch0=bidz,
        )
    elif warp_idx == TMALDG_WARP_ID:
        nvvm.prefetch_tensormap(tma_q_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_k_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_v_desc.get_ptr())
        _tmaldg_warp_group(
            tma_q_desc=tma_q_desc,
            sQ=sQ,
            sIds_raw=sIds_raw,
            bars=bars,
            sched=sched,
            block_ids_tensor=block_ids_tensor,
            seqlen_q=seqlen_q,
            tok0=bidx,
            head0=bidy,
            batch0=bidz,
        )
    elif warp_idx == SCHED_WARP_ID:
        cta_id_x = cutlass.Int32(0)
        is_cga_first_cta = cta_id_x == cutlass.Int32(0)
        scheduler_warp_loop(sched, SCHEDULER_STAGES, is_cga_first_cta, 1)
    # else: the spare warp (15) exits.


_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


# === host =================================================================================================================


@cute.jit
def _host(
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    v_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    lse_ptr: Optional[cute.Pointer],
    block_ids_ptr: cute.Pointer,
    block_lens_ptr: Optional[cute.Pointer],
    seq_kv_lens_ptr: cute.Pointer,
    problem_size: Tuple[int, int, int, int, int],
    q_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    k_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    v_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    o_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    lse_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    scale_softmax_log2: cutlass.Float32,
    block_table_ptr: Optional[cute.Pointer],
    paged_geom: Tuple[int, int, int, int, int, int],
    stream: _cuda_driver.CUstream = None,
) -> None:
    """Bind the explicit pointer ABI and launch the sparse core.

    ``problem_size`` = (B, QH, KH, SQ, SKV); strides are the caller's (batch, seq, head) ELEMENT strides (Int64 leaves);
    ``block_ids_ptr`` -> int32 [B x SQ, BLOCK_TOPK] contiguous, ``block_lens_ptr`` -> int32 [B x SQ] or None,
    ``seq_kv_lens_ptr`` -> int32 [B] (read only under SEQ_KV_LENS_PRESENT).  The K/V gather maps are 2-D (tokens of EVERY
    batch as rows, the token's full row as columns), so the batch stride must be SKV x the token stride (or B == 1) -- the
    adapter validates that contract; this entry assumes validated operands.

    PAGED arm (CFG.PAGED_KV): ``k_strides`` / ``v_strides`` = (page stride, TOKEN stride, COLUMN head stride -- 0 for an HND
    pool whose head folds into the row), ``block_table_ptr`` -> int32 (B, max_pages) contiguous, ``paged_geom`` =
    (num_pages, max_pages, k rows_per_page, k rows_per_head, v rows_per_page, v rows_per_head); the gather maps' rows are
    the pool's token rows (num_pages x rows_per_page); SKV is the table's capacity (max_pages x PAGE_SIZE), the visible range
    comes from ``seq_kv_lens``.  On the dense arm ``block_table_ptr`` is None and ``paged_geom`` is unread.
    """
    B, QH, KH, SQ, SKV = problem_size
    num_pages, max_pages, k_rows_per_page, k_rows_per_head, v_rows_per_page, v_rows_per_head = paged_geom
    q_tensor = _bshd(q_ptr, B, SQ, QH, TILE_K, q_strides, False)
    o_tensor = _bshd(o_ptr, B, SQ, QH, TILE_O, o_strides, False)
    lse_tensor = None
    if cutlass.const_expr(lse_ptr is not None):
        l0, l1, l2 = lse_strides
        lse_tensor = cute.make_tensor(lse_ptr, cute.make_layout((B, QH, SQ), stride=(l0, l1, l2)))
    block_ids_tensor = _vec(block_ids_ptr, B * SQ * BLOCK_TOPK)
    block_lens_tensor = None
    if cutlass.const_expr(block_lens_ptr is not None):
        block_lens_tensor = _vec(block_lens_ptr, B * SQ)
    seq_kv_lens_tensor = _vec(seq_kv_lens_ptr, B)
    block_table_tensor = None
    if cutlass.const_expr(block_table_ptr is not None):
        block_table_tensor = cute.make_tensor(block_table_ptr, cute.make_layout((B, max_pages), stride=(max_pages, 1)))

    def _tma_swz(byte_w: int):
        return tmap.TensorMapSwizzle.s128b if byte_w == 128 else tmap.TensorMapSwizzle.s64b if byte_w == 64 else tmap.TensorMapSwizzle.s32b

    # Q^T box: ONE token x G heads of one KV group (row = head within the group), the 4-D map over (D, QH, SQ, B).
    stride_order = (3, 2, 1, 0)
    box_q = (1, Q_BOX_TOKENS, Q_BOX_ROWS, GRANU_ELEMS)
    tma_q_desc = tmap.create_tensor_map_tiled(
        global_address=q_tensor.iterator.toint(),
        dtype=q_tensor.element_type,
        global_dims=tuple(q_tensor.shape[i] for i in stride_order),
        global_strides=tuple(cutlass.Int64(q_tensor.stride[i]) * q_tensor.element_type.width // 128 for i in stride_order[1:]),
        box_dims=tuple(box_q[i] for i in stride_order),
        swizzle=_tma_swz(CFG.Q_SWZ_BYTES),
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )

    # K / V gather maps: ONE 2-D map per operand, rows = the tokens of every batch (B x SKV, pitch = the token stride;
    # PAGED arm: the pool's token rows, num_pages x rows_per_page, at the same pitch), columns = the token's full row
    # ((KH - 1) x head stride + D elements, contiguous); box (64 elements, 1 row) = one 128-B SW128 span.  The head and
    # the column box live in the COLUMN coordinate (gather warps: head x head_stride + box x 64), the batch / page term in
    # the ROW coordinate (batch x SKV + 4 blk + r; page x rows_per_page + head x rows_per_head + (4 blk) % PAGE_SIZE + r).
    def _gather_desc(ptr, strides, rows_per_page):
        _bs, ss, hs = strides
        rows = cutlass.Int32(num_pages) * cutlass.Int32(rows_per_page) if cutlass.const_expr(PAGED_KV) else cutlass.Int32(B) * cutlass.Int32(SKV)
        inner = (cutlass.Int32(KH) - cutlass.Int32(1)) * cutlass.Int32(hs) + cutlass.Int32(TILE_K)
        return tmap.create_tensor_map_tiled(
            global_address=ptr.toint(),
            dtype=STORAGE_DTYPE,
            global_dims=(inner, rows),
            global_strides=(cutlass.Int64(ss) * BPE // 16,),
            box_dims=(GATHER_BOX_ELEMS, 1),
            swizzle=_tma_swz(CFG.K_SWZ_BYTES),
            l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
        )

    tma_k_desc = _gather_desc(k_ptr, k_strides, k_rows_per_page)
    tma_v_desc = _gather_desc(v_ptr, v_strides, v_rows_per_page)

    # The grid is the work list: one CTA per (query token, KV head, batch); the CLC scheduler hands cancelled CTAs to the
    # resident ones (one CTA per SM through the 221 KiB SMEM footprint).
    grid_shape = (SQ, KH, B)
    _kernel(
        tma_q_desc,
        tma_k_desc,
        tma_v_desc,
        o_tensor,
        lse_tensor,
        block_ids_tensor,
        block_lens_tensor,
        seq_kv_lens_tensor,
        cutlass.Int32(SQ),
        cutlass.Int32(SKV),
        cutlass.Int32(k_strides[2]),
        cutlass.Int32(v_strides[2]),
        scale_softmax_log2,
        block_table_tensor,
        cutlass.Int32(max_pages),
        cutlass.Int32(k_rows_per_page),
        cutlass.Int32(k_rows_per_head),
        cutlass.Int32(v_rows_per_page),
        cutlass.Int32(v_rows_per_head),
    ).launch(
        grid=grid_shape,
        block=[CFG.THREADS_PER_CTA, 1, 1],
        cluster=(1, 1, 1),
        stream=stream,
    )


EXPLICIT_ABI = True  # pointer/int host entry; the adapter builds the argument list itself


@lru_cache(maxsize=None)
def compile(  # noqa: A001
    has_lse: bool = True,
    has_block_lens: bool = False,
) -> Callable:
    """Compile the pointer ABI: ``has_lse`` folds the LSE store in / out, ``has_block_lens`` the per-row count read; the
    paged arm's block table is present exactly when the config is paged (CFG.PAGED_KV, part of the template's identity).
    Head dims are the config's (d_qk = d_v = 256, exact); shapes and strides are runtime arguments."""
    _cache_key = _template_key(globals(), locals(), "compile")
    gmem = cute.AddressSpace.gmem

    def P(dtype, align=16):
        return cute.runtime.make_ptr(dtype, 16, gmem, assumed_align=align)  # fake: type only

    i64_3 = (cutlass.Int64(0),) * 3  # stride slots: Int64 leaves, see _host
    return _compile_cached(
        _host,
        P(STORAGE_DTYPE),
        P(STORAGE_DTYPE),
        P(STORAGE_DTYPE),
        P(STORAGE_DTYPE),
        P(cutlass.Float32, 4) if has_lse else None,
        P(cutlass.Int32, 16),
        P(cutlass.Int32, 4) if has_block_lens else None,
        P(cutlass.Int32, 4),
        (0, 0, 0, 0, 0),
        i64_3,
        i64_3,
        i64_3,
        i64_3,
        i64_3,
        cutlass.Float32(0.0),
        P(cutlass.Int32, 4) if PAGED_KV else None,
        (0, 0, 0, 0, 0, 0),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
        cache_key=_cache_key,
        symbol="frost_sdpa_fwd_sparse",
    )
