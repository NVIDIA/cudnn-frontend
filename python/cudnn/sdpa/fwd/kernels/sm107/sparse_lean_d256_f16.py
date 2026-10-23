# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Rubin (SM107) index-list SPARSE d256 DECODE forward, the LEAN tile -- ``mma.sync`` m16n8k16 + ``cp.async``, 128 threads, no TMEM,
no TMA, no CLC, no mbarrier, no warp specialization.  The second body of the function ``sparse_d256_f16.py`` (the tcgen05 tile) computes,
for the serving step's S_q = 1 rows (the shared list per sequence, bottom-right anchored), writing the SAME fp32 split-major partial slab
and LSE as that tile's split store so ``sm100/split_combine.py`` reduces it unchanged.

HEADER FIRST: the barrier table (the accumulator initialisation stated), the SMEM table, the register plan and the fragment plan below
are what the body is checked against.  Every geometry value is CFG's (``config_sm107.make_cfg_d256_sparse_lean``), never a literal.

WHAT IT COMPUTES (= the sparse tile's DECODE form, ``sparse_d256_f16.py`` header + ``_sparse_item_common.visible_bounds``; UNCHANGED).
Sequence b (KV length L_b = seq_kv_lens[b]), S_q = 1 row at pos_0 = L_b - 1, KV head h, the shared list ids[b, :] ([B, BLOCK_TOPK] int32,
a valid prefix then -1; block_lens[b] or None):

    n_sel_default = min(BLOCK_TOPK, floor((pos_0 + 1) / 4));  count = clamp(block_lens[b], 0, n_sel_default)       (device values)
    tail_lo = floor((pos_0 + 1) / 4);  n_tail = ceil((pos_0 + 1) / 4) - tail_lo  in {0, 1}  (S_q 1: the one open block)
    V = { t <= pos_0 : floor(t / 4) in ids[b, :count] }  UNION  { t : 4 tail_lo <= t <= pos_0 }
    O[b, 0, h G + g] = softmax_fp32(q . K[V]^T x scale) . V[V];  LSE = m + log(sum)  (natural log; the exact fp32 sum of the UNQUANTIZED P)

-1 inside the count = "no key" (zero rows, -inf); a listed block past pos_0 contributes nothing (masked key by key); duplicates / the open
block listed inside the count are attended twice (the list contract forbids both; the host check is form-only); an empty V (L_b = 0) lands
O = 0 / LSE = -inf through a SELECT.  Served: bottom_right + list_per_sequence at S_q 1, dense BSHD K / V or paged pools (HND / NHD), bf16 /
f16, G in 1..16, top_k a multiple of 4 in [8, 512]; every other arm a typed decline of the config (THD, per-row lists, sink, band, log2
stats, padded-Q trim, the epilogue gate -- under the lean split the gate rides the combine).

WORK ITEM, GRID, SPLIT.  Sub-tile t of (b, h) = list indices [8 t, 8 t + 8) = 8 four-token blocks = N_KEYS = 32 keys; an index at or
past count + n_tail reads -1.  n_sub = ceil((BLOCK_TOPK + 2) x 4 / 32) is the HOST bound (65 at top_k 512; the n_tail_max of 2 is the
S_q <= 4 bound, kept so the S_q 2..4 rendering changes no grid); k = SUB_TILES_PER_CTA (a template axis, range_constexpr-unrolled); SPLIT = ceil(n_sub / k);
grid = (SPLIT, H_kv, B) = the work list (blockIdx = (c, h, b); no scheduler, no CLC, no persistence).  Chunk c runs sub-tiles [k c, k c + k);
a sub-tile with 8 t >= count + n_tail is DEAD (every row zero-filled, every key masked); a chunk whose first sub-tile is dead is a DEAD
ITEM -- it still runs k uniform sub-tiles and writes 0 / -inf into every (chunk, row) slot the combine reads (the combine's identity,
``split_combine.py``).  Partials: o_partial[b + c B, 0, h G + g, d] fp32 = acc / l (the dead SELECT per element); lse_partial[b + c B,
h G + g, 0] = m_nat + log l or -inf -- byte-identical to ``sparse_gqa_sm107._split_workspace_layout`` at ``split`` = SPLIT, strides
(SQ H D, H D, D) / (H SQ, SQ, 1).

THREADS, WARPS, FRAGMENTS (the "split-D" plan).  128 threads = 4 warps (N_MT = 1 m16 tile at S_q 1; THREADS = 128 x N_MT is the config
pin).  Warp w owns the d-slice [64 w, 64 w + 64) of BOTH the BMM1 K-reduction and the BMM2 O columns; lane l: quad g = l / 4 = query rows
g and g + 8 (the m16n8 C / A row pair), t = l % 4 = the column pair.  Rows >= G (12..15 at G 12) are ZERO padding: Q rows SELECTed to 0
at the load, never stored.  mma.sync.aligned.m16n8k16.row.col.f32.{bf16|f16}.{bf16|f16}.f32 (``tile_dsl/mma.py::mma_m16n8k16_f32``):
A 16 x 16 = 4 x b32 per lane (a0 row g k 2t+{0,1}; a1 row g+8; a2 row g k +8; a3 row g+8 k +8), B 16 x 8 = 2 x b32 (b0 k 2t+{0,1} n g;
b1 k +8), C / D 4 x f32 (c0, c1 row g n 2t+{0,1}; c2, c3 row g+8).

    operand                      source                                  per lane            how
    Q A-fragments (BMM1 A)       the Q slab q[b, 0, h G + row, 64 w ..]   16 b32, resident    16 LDG.32 (ld_global): per k-step ks and row r in {g, g+8}:
                                                                                              d = 64 w + 16 ks + 2 t and +8 (two halves each); rows >= G read
                                                                                              row min(r, G-1) and SELECT 0 (no OOB, no divergence); issued
                                                                                              after the PDL wait and after the cp.async issue (they overlap the gather)
    K B-fragments (BMM1 B = K^T) sK[stage] [32 keys][256 d], 512-B rows   2 b32 per (ks, nb)  load_b_smem_x4(sK, k_step=ks, N=32, b_trans=False, col_base=64 w, swizzle):
                                                                                              2 ldmatrix.x4 per k-step (n-frag pairs), 4 k-steps = 8 per sub-tile
    V B-fragments (BMM2 B = V)   sV[stage] [32 keys][256 d]               per (ks, nb)        load_b_smem_x4(sV, k_step=ks, N=64, b_trans=True, col_base=64 w, swizzle):
                                                                                              4 ldmatrix.x4.trans per k-step, 2 k-steps = 8 per sub-tile
    P A-fragments (BMM2 A)       the S accumulators IN PLACE after exp    8 b32               k-step ks (keys 16 ks..+15): a0 = pack(c0, c1 of nb 2ks), a1 = pack(c2, c3 of nb 2ks),
                                                                                              a2 = pack(c0, c1 of nb 2ks+1), a3 = pack(c2, c3 of nb 2ks+1); fp32_to_fp16(lo, hi)
                                                                                              (cvt.rn.{bf16,f16}x2.f32) -- the FA2 identity: the m16n8 C layout of keys IS the
                                                                                              m16n8k16 A layout for k = keys

Per sub-tile per warp: BMM1 16 MMAs (4 n-blocks x 4 k-steps) into S_w[16 x 32] (16 f32), BMM2 16 MMAs (8 n-blocks x 2 k-steps) into
O_w[16 x 64] (32 f32); 128 MMAs per CTA per sub-tile.

THE S EXCHANGE AND THE FIXED-ORDER SUM.  Each warp holds a PARTIAL S_w (its 64 d of the 256-d reduction).  sS[4 warps][4 n-blocks][32
lanes][4 f32]: warp w writes its 4 n-block C fragments as 4 x STS.128 at ((w 4 + nb) 32 + l) x 16 B; bar.sync 0; EVERY warp reads all 16
slots (LDS.128 x 16) and sums S = s0 + s1 + s2 + s3 in that order -- every warp computes the bitwise-identical S, so the softmax is
redundant across the 4 warps and alpha, m, l agree without another exchange.

ONLINE SOFTMAX (fp32, exp2 domain; the sparse tile's spelling).  Per lane 2 rows x 8 scores (4 n-blocks x the column pair).  Key j of
the sub-tile at column pair (nb, e): j = 8 nb + 2 t + e; block q(j) = j / 4; key_abs = 4 blk_q + j % 4.
  * The mask is a 32-bit VALID WORD (frost-tile-dsl 10d), warp-uniform, from the 8 block ids: n_live_q = (blk_q >= 0) ? clamp(n_vis -
    4 blk_q, 0, 4) : 0 with n_vis = the shared helper's clamp(min(pos_0 + 1, L_b), 0); word |= ((1 << n_live_q) - 1) << 4 q.  Per lane and
    n-block: bits = word >> (8 nb + 2 t); key e valid iff bits & (1 << e) -- tile_dsl.mask.apply_membership_fragment, the fragment-order
    form of the shared membership-word op.  This IS the sparse tile's valid = (blk >= 0) & (key_abs <= pos) & (key_abs < eff_seqlen_kv)
    at S_q 1 (pos = pos_0 = L_b - 1); a zero-filled row whose block id is -1, dead or at / past L_b is masked by the word too.  NOT masked
    (zero-filled, attended as a zero key with weight exp(-m) over a zero V row): a -1 / out-of-range TABLE entry under a live block on the
    paged arm, and a dense row at / past S_kv when L_b > S_kv -- PARITY with the tcgen05 tile, whose valid word ignores the page too and
    whose TMA clips both rows to zeros; a malformed table / length outside the contract, never a NaN.
  * s = valid ? s_raw x scale_log2 : -inf (a NEGATIVE scale is served signed -- applied BEFORE the max, as the sparse tile); the raw tile
    max over the thread's 8 scores, then the QUAD reduction over the 4 lanes of the row (quad_reduce_max: shfl.sync.bfly 2, 1) -> tile_max
    (-inf when every key of the row is masked).
  * m_new = max(m_old, tile_max) (ftz); ms_old = row_max_for_exp2(m_old), ms_new = row_max_for_exp2(m_new) (the -inf -> 0 SELECT at the
    point of use); alpha = exp2(min(ms_old - ms_new, 0)); p = exp2(s - ms_new) (a masked -inf -> exactly 0; a sub-tile whose every key is
    masked: tile_max = -inf, m unchanged, alpha = 1, P = 0 -- the dead-tile SELECT by construction, sdpa-invariants section 1 row 3);
    l = l x alpha + quad_reduce_sum(sum of the thread's 8 p) (fp32, the UNQUANTIZED p: the LSE contract).  k > 1: O_w *= alpha on all 32
    regs (EXACT, never lazy; alpha is quad-uniform and warp-identical, so the 4 d-slices of a row rescale consistently); at sub-tile 0
    alpha is exactly 1 and O is exactly 0, so the multiply is compiled out there.
  * P -> {bf16,f16}x2 in place (the FA2 identity above); no shuffle.

EPILOGUE (= the sparse tile's split store, the same lines):
    m_nat = row_max_for_exp2(m) x LN2;   row_dead = (l_tot <= 0)
    lse = m_nat + log(max(l_tot, TINY));   inv = 1 / max(l_tot, TINY);   TINY = 1e-30 defined ONCE, used at these two sites only
    lse = SELECT(row_dead, -inf, lse);     inv = SELECT(row_dead, 0, inv)
    O_s[row, d] = SELECT(row_dead, 0, acc x inv)  per element -> 16 x STG.64 per lane (d pairs 64 w + 8 nb + 2 t + {0,1} of rows g, g+8)
    LSE: lanes with t == 0 of warp 0 store rows g and g+8
    store predicate: row < G (padding rows never stored); EVERY live-row slot of EVERY chunk is written, dead chunks included
    addresses: Int64 (part_batch = b + c x B widened BEFORE the stride multiply, Rule S7); the slab strides are runtime arguments
Numerics: fp32 accumulation; P rounded once to the storage half; bitwise DIFFERENT from the tcgen05 tile BY CONSTRUCTION (a 4 x 64-d split-K
sum in fixed order vs one 256-d MMA chain; 32- vs 128-key tiles); DETERMINISTIC (fixed-order sums, quad shuffles, the online sequence over
sub-tiles, no atomics): two launches bitwise, replay == eager bitwise, paged == dense bitwise on the lean route.

THE GATHER (the only global reads besides Q, the ids, the lengths and the table; parity with the sparse tile's ``_gather_block_pages`` /
``_gather_fill``).  Loader = all 4 warps, every sub-tile.  Pass p (0..7), lane l: 16 B of operand row 8 w + p (block 2 w + p / 4, token
p % 4) at column chunk l (32 chunks x 16 B = the 512-B row): a warp copies 8 whole rows = 2 blocks, so it needs TWO block ids and TWO page
ids (a SELECT chain on w over the 8 ids every lane already holds for the valid word); the 8 ids per lane come by 2 x LDG.128 (ldg_int32x4;
ids[b, 8 t ..] is 16-B aligned: BLOCK_TOPK % 4 == 0, the quad start clamped into the row), kv_lens[b] and block_lens[b] once per CTA.
16 cp.async.cg.shared.global ... 16, src-size per lane per sub-tile (cp_async_gather_row16), ONE commit group, zero waits between the issues.

    blk  = block_id_at(idx = 8 t + q, loaded, count, n_tail, tail_lo, dead)        -- the shared helper (_sparse_item_common)
    dense: addr = k_base + b x s_batch + (4 blk + r) x s_tok + h x s_head + 8 l            live = blk >= 0 & (4 blk + r) < L_b & (4 blk + r) < S_kv
    paged: page = table[b, (4 blk) / P] (a clamped read; -1 unless blk >= 0, 4 blk < L_b, the index inside the table, the entry >= 0 and
           below num_pages); addr = k_base + page x s_page + h x s_head + ((4 blk) % P + r) x s_tok + 8 l      live = page >= 0 & (4 blk + r) < L_b
    cp.async [sK[stage] + swz(row 8 w + p, chunk l)], [live ? addr : k_base], 16, src-size = live ? 16 : 0  (zero-fill; the dummy src = the base)

A row at or past L_b inside the last live page is zero-filled, never read -- PARITY with the sparse tile's row -1 (stricter only than the
dense paged decode tile).  On the dense arm a row at or past S_kv (the tensor's extent) is zero-filled too, whatever L_b says (the lengths
are device values no host checks): the TMA form clips such a row to zeros, this form must not read it -- the extent clamp is the one global
read the length alone would not bound.  s_head on the paged arm = rows_per_head x s_tok + col_head_stride: ONE address form for HND and NHD pools.
Stride contract (the adapter's): d contiguous; s_tok, s_head, s_batch / s_page multiples of 8 elements (16-B cp.async); 16-B aligned bases.
A DEAD sub-tile / item issues the same 16 zero-filled copies and the same commit (uniform ledger).

BARRIER TABLE (cp.async groups + bar.sync 0 only; T = THREADS = 128 = the launch's thread count; no mbarrier, no phase, no cross-CTA
arrive, no early exit anywhere).  k and the sub-tile loop are COMPILE-TIME (cutlass.range_constexpr(k)): every stage index, every
cp.async.wait_group immediate and every predicate t + 2 < k folds.  STAGES = min(2, k).  Group ledger: the prologue commits G_0 = {K(0),
V(0)} into stage 0 and, at k >= 2, G_1 = {K(1), V(1)} into stage 1; at F(t) with t + 2 < k the loader commits G_{t+2} into stage t % 2.
Groups committed before A(t) = min(k, t + 2), so A(t)'s wait leaves min(k, t + 2) - (t + 1) groups in flight = wait_group 1 while t + 1 <
k, wait_group 0 on the last sub-tile.

    #     site                       instruction                                         issuing  waiting  guard                                              SUM(issuing) == count
    init  the accumulator init        before sub-tile 0: O_w[32] = 0 (opaque_f32_zero), l[2] = 0, m[2] = -inf per lane; S_w[16] = 0
          (sdpa-invariants section 2) (opaque) before EVERY sub-tile's BMM1; ONE item per CTA -- no state carried between items or CTAs    all 128  --       none                                               n/a (no barrier)
    P     PDL wait + release          griddepcontrol.wait then griddepcontrol.launch_dependents, after the blockIdx arithmetic,
                                      BEFORE the first global read                                                                          all 128  --       cutlass.const_expr(use_pdl)                        n/a
    A(t)  stage t % 2 FULL + visible  cp.async.wait_group (1 if t + 1 < k else 0), then bar.sync 0                                        all 128  all 128  none: every thread every sub-tile; a dead sub-tile  128 == 128
                                                                                                                                                              issued 16 zero-filled copies + the commit too
    X(t)  the S exchange              4 STS.128 -> bar.sync 0 -> 16 LDS.128                                                               all 128  all 128  none (the slot's previous reads ended before A(t):   128 == 128
                                                                                                                                                              X(t-1)'s reads precede BMM2(t-1) precede F(t-1) / A(t))
    F(t)  stage t % 2 FREE            bar.sync 0 after BMM2(t); THEN the loader issues G_{t+2} into stage t % 2 and commits               all 128  all 128  the row EXISTS iff t + 2 < k (a constexpr: the      128 == 128
          (a refill follows)          -- the barrier and the refill fold together: no refill, no barrier                                                      barrier guards a refill; the last two sub-tiles
                                                                                                                                                              have none); inside it, no guard
    exit  --                          nothing outstanding: the last A waited wait_group 0; no arrive outlives the CTA (P15 has no object)

k = 1: A and X only (two bar.sync per CTA); k = 2: both sub-tiles are prefetched in the prologue, no F (four bar.sync); k >= 3: 2 k + (k - 2) bar.sync.  Divergence hazard: none -- the only lane-varying control
is the cp.async src-size operand, the per-row SELECTs and the row < G / t == 0 store predicates; no `if` encloses a barrier.  Config pin:
THREADS == 128 x N_MT against the loader's 8 w + p row map and the exchange's [4 warps] slot count (sdpa-invariants section 6).

PDL SITES (P17): under use_pdl (a cutlass.Constexpr[bool] of _kernel and _host; the launch takes the programmatic-stream-serialization
attribute under the same flag) EVERY warp issues wait_on_dependent_grids() after its blockIdx / lane arithmetic and BEFORE its first global
read -- kv_lens / block_lens (the stack's lengths, the selector's counts), the ids (the top-k's output), the table, the Q slab (the
projection / prepare), the pool rows (the write-through) -- then launch_dependent_grids() right after the wait (no TMEM: the P17 alloc
exception cannot arise; the dependent combine's own wait blocks on this grid's full completion wherever the trigger sits).  The OFF
rendering carries no griddepcontrol token.

SMEM TABLE (declaration order = address order; every byte lane-written and lane-read; NO descriptor anywhere; dynamic .extern .shared as
the DSL lowers cutlass.Array(..., space=smem); 128-B alignment -- ldmatrix / cp.async rows; 1024 is not needed, no TMA)

    name [stages]                 bytes            WRITER, how                                   READER, how                               lifetime            lane stride  swizzle + WHY
    sK[STAGES] (32 keys x 512 B)  STAGES x 16,384  128 lanes x 8 cp.async.cg 16 B (zero-fill):  ldmatrix.x4 .row (B of BMM1): an 8-lane     fill(t) -> BMM1(t)  16 B         XOR-128B on the 16-B chunk index by row & 7
                                                   lane l writes chunk l ^ (row & 7) of row 8w+p   phase addresses 8 consecutive key rows                                      (swizzle_xor_128b = col ^ ((row & 7) x 8)):
                                                                                                 at ONE 16-B chunk                                                             a 512-B pitch puts 8 rows' chunk c on ONE
                                                                                                                                                                               16-B slot of the 128-B bank cycle -> 8-way
                                                                                                                                                                               conflict unswizzled; XORed, 8 distinct slots.
                                                                                                                                                                               The writes: 32 lanes x 16 B = one 512-B row per
                                                                                                                                                                               instruction (a permutation within the row)
    sV[STAGES]                    STAGES x 16,384  the same                                     ldmatrix.x4.trans (B of BMM2): 8 lanes      fill(t) -> BMM2(t)  16 B         the same (the sm80 / sm120 kernels' layout)
                                                                                                 address 8 consecutive key rows at one chunk
    sS[4 warps][4 nb][32][4 f32]  8,192            each lane 4 STS.128 at ((w 4 + nb) 32 + l) 16 B   every lane 16 LDS.128, fixed order w = 0..3  X(t)                16 B         none needed: 32 lanes x 16 B = 512 B contiguous
                                                                                                                                                                               per instruction = four full 128-B bank cycles
    total                         40,960 B at k = 1 (STAGES 1); 73,728 B = 72 KiB at k >= 2 (STAGES 2) < the 227 KiB standard carveout; CTAs / SM by SMEM 5 / 3; by registers 4 at k 1 (REG 70 / 72), 3 at k 2..4 (REG <= 168; the register plan below)

No sQ (Q in registers), no sIds (uniform / vector LDG), no sP (P in registers), no TMEM, no mbarrier arrays, no tensormap.

REGISTER PLAN (target <= 128 per lane at k = 1; the host sm_107a trace's REG / STL / LDL is the detector -- the DSL emits .reqntid 128 and
no .maxnreg, so ptxas sizes the flat body): Q A-fragments 16 (resident) + S accumulator 16 (-> the packed P, 8) + O accumulator 32 +
B-fragment staging 8 + softmax state (m, l x 2 rows, alpha, tile max, scale_log2, the valid word) ~10 + gather / addressing (8 ids, 2 pages,
lens, row bases, lane constants, stage bases) ~22 = ~104 modeled.  MEASURED (the test module's host trace): k = 1 REG 70 dense / 72 paged /
69 at G 1, STL 0, LDL 0 (4 CTAs / SM by registers; 5 by the 40-KiB SMEM); k >= 2: ptxas software-pipelines the unrolled sub-tiles (the next
sub-tile's fragments hoisted over the current chain) -- 136 dense / 104 paged at k 2, 141 at k 3, 164 at k 4, pinned <= 168 (3 CTAs / SM
by registers at <= 170 = the SMEM bound of the 72-KiB two-stage form, so no occupancy is lost at k <= 4); 204 at k 8 (2 CTAs / SM; outside
the fill rule, not pinned).  A -maxrregcount=128 build is a measured lever, not a default.  No runtime-indexed local array anywhere (the
valid-WORD mask; every loop range_constexpr).

DEGENERATE INPUTS (sdpa-invariants section 1; each with its site in the body): L_b = 0 / pos_0 < 0 -> dead (visible_bounds) -> every blk
-1 -> 16 zero-filled copies -> word 0 -> P = 0, l = 0 -> the row_dead SELECT writes 0 / -inf into EVERY chunk slot of the sequence; count = 0
with a tail -> the tail block only (block_id_at: idx == count -> tail_lo); (pos_0 + 1) % 4 == 0 -> n_tail = 0 (no open block); -1 inside the
count -> blk = -1 -> zero rows (zero-fill) and masked (word bit 0); an id past the length / past the tensor -> n_live_q = 0 and the paged
4 blk < L_b / dense (4 blk + r) < L_b predicate -> zero-filled + masked; blk > (2^31 - 1) / 4 -> -1 (block_id_at's domain bound); a future
block (4 blk > pos_0) -> word bits 0; duplicates -> counted twice (the contract); block_lens out of range -> clamped on device; block_lens
absent -> count = n_sel_default (the has_block_lens compile axis); a chunk past the item's real n_sub (dead item) -> every sub-tile dead ->
0 / -inf in every slot; k > the item's live sub-tiles -> the trailing sub-tiles dead (zero-filled, masked; the ledger uniform); the paged
page -1 / past the table / at or past num_pages / 4 blk >= L_b -> page = -1 -> zero-filled rows -- UNMASKED when the block itself is live
(4 blk < L_b): a zero key attended, parity with the tcgen05 tile, a malformed table; max_pages = 0 (a zero-column table) -> the clamped
table read is steered onto the ids row (a valid address) and every page is -1 (page_idx < max_pages fails): no out-of-bounds read, the
listed blocks are zero keys; L_b > S_kv on the dense arm -> rows at / past S_kv zero-filled by the extent clamp (never read), attended as
zero keys where the word marks them (parity with the tcgen05 tile's TMA-clipped rows); stale rows past L_b in the last live page
(NaN-poisoned) -> zero-filled by the per-row predicate, never read; B x H > 1, k = 1 -> no carried state exists to drift; G = 1 (MHA) /
G = 16 -> rows 1..15 padding / none, the row < G store predicate; top_k < 8 -> a typed decline (one sub-tile = split 1 -> the tcgen05 tile).
"""

# Config comes from the FROST template loader (FROST_TEMPLATE_PARAMS is injected before this body runs); the plain-import default is the
# 24/2 geometry (12 query heads per KV head) at top_k = 512, bf16, dense, k = 1 (split 65), so a plain import stays usable standalone.
from cudnn.sdpa.fwd.config_sm107 import TemplateParams, make_cfg_d256_sparse_lean

PARAMS: TemplateParams = globals().get(
    "FROST_TEMPLATE_PARAMS",
    TemplateParams(
        dtype_qkv=2,
        cta_mma=1,
        pack_gqa=True,
        qh_per_kh=12,
        qsa_block_topk=512,
        qsa_block_size=4,
        seq_kv_lens_present=True,
        bottom_right=True,
        qsa_list_per_sequence=True,
    ),
)
CFG = make_cfg_d256_sparse_lean(PARAMS)

# CuTe DSL version gate (python/cudnn/AGENTS.md Rule 7): sm_107a needs the public 4.8.0 wheel.  The adapter declines first; this is the
# typed backstop for a direct template load, placed before the kernel body so the message names the installed version.
from cudnn.frost.buffers import cutedsl_arch_requirement_error

_DSL_ARCH_ERROR = cutedsl_arch_requirement_error((10, 7))
if _DSL_ARCH_ERROR is not None:
    raise NotImplementedError(f"sparse_lean_d256_f16 (sm107): {_DSL_ARCH_ERROR}")

from cudnn.frost.compiled_cache import compile_cached as _compile_cached, template_key as _template_key
from functools import lru_cache
from typing import Callable, Optional, Tuple

from cutlass._mlir.dialects import arith
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as _cuda_driver  # noqa: F401

from cudnn.frost.tile_dsl.barrier import launch_dependent_grids, wait_on_dependent_grids
from cudnn.frost.tile_dsl.mask import apply_membership_fragment
from cudnn.frost.tile_dsl.mma import load_b_smem_x4, mma_m16n8k16_f32
from cudnn.frost.tile_dsl.pointwise import fp32_to_fp16, opaque_f32_zero, quad_reduce_max, quad_reduce_sum
from cudnn.frost.tile_dsl.tma import cp_async_commit, cp_async_gather_row16, cp_async_wait, ld_global, ldg_int32x4, st_global, st_global_v2
from cudnn.sdpa.fwd.kernels._common_blackwell import _bshd, _vec, row_max_for_exp2

# The visible-set helper shared with the tcgen05 tile (ONE helper for both bodies; the SELECT spellings travel with it).
from cudnn.sdpa.fwd.kernels.sm107._sparse_item_common import _select_f32, _select_i32, block_id_at, visible_bounds

if CFG.DTYPE_QKV == 2:
    STORAGE_DTYPE = cutlass.BFloat16
else:
    STORAGE_DTYPE = cutlass.Float16

# === geometry, every value from CFG (frost-kernels.md section 6: derive, never re-literal) ===
D = CFG.D
N_KEYS = CFG.N_KEYS
BLOCK_SIZE = CFG.BLOCK_SIZE
BLOCKS_PER_SUB = CFG.BLOCKS_PER_SUB
BLOCK_TOPK = CFG.BLOCK_TOPK
N_SUB = CFG.N_SUB
SUB_TILES_PER_CTA = CFG.SUB_TILES_PER_CTA
SPLIT = CFG.SPLIT
STAGES = CFG.STAGES
THREADS = CFG.THREADS
WARPS = CFG.WARPS
G = CFG.QH_PER_KH
PAGED_KV = CFG.PAGED_KV
PAGE_SIZE = CFG.PAGE_SIZE
BPE = CFG.BPE
# The warp's d-slice (the split-D plan), the MMA k-step and the n-block counts of the two products.
D_SLICE = D // WARPS  # 64
BMM1_K_STEPS = D_SLICE // 16  # 4: the warp's 64 d in k16 steps
BMM1_N_BLOCKS = N_KEYS // 8  # 4: 32 keys in n8 blocks
BMM2_K_STEPS = N_KEYS // 16  # 2: 32 keys in k16 steps
BMM2_N_BLOCKS = D_SLICE // 8  # 8: the warp's 64 d in n8 blocks
CHUNK_ELEMS = 16 // BPE  # 8 half elements per 16-B cp.async chunk
ROW_ELEMS = D  # a K / V row is D elements = 512 B
STAGE_ELEMS = N_KEYS * ROW_ELEMS  # 8192 elements = 16 KiB per operand stage
ROWS_PER_PASS = THREADS // (ROW_ELEMS // CHUNK_ELEMS)  # 4: 128 lanes x 16 B = four 512-B rows per pass
PASSES = N_KEYS // ROWS_PER_PASS  # 8 passes per operand per sub-tile
ROWS_PER_WARP = PASSES  # warp w owns rows 8 w .. 8 w + 7 (the `8 w + p` map) = BLOCKS_PER_WARP blocks
BLOCKS_PER_WARP = ROWS_PER_WARP // BLOCK_SIZE  # 2
S_SLOT_WORDS = 4  # one C fragment = 4 f32 = 16 B per lane
S_EXCHANGE_WORDS = WARPS * BMM1_N_BLOCKS * 32 * S_SLOT_WORDS  # 2048 words = 8 KiB
if CFG.SMEM_S_BYTES != S_EXCHANGE_WORDS * 4 or CFG.SMEM_K_BYTES != STAGES * STAGE_ELEMS * BPE or CFG.N_MT != 1 or ROWS_PER_WARP != 8 or BLOCKS_PER_WARP != 2:
    raise ValueError("sparse_lean_d256_f16: the body's geometry (the 8 w + p loader map, the [4 warps] exchange, one m16 tile) disagrees with CFG")

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
_I32_MAX_BLK = (2**31 - 1) // BLOCK_SIZE


@cute.jit
def _select_i64(cond, a, b):
    return cutlass.Int64(arith.select(cond.ir_value(), cutlass.Int64(a).ir_value(), cutlass.Int64(b).ir_value()))


@cute.jit
def _sel4(w, x0, x1, x2, x3):
    """The warp's own value out of four warp-indexed ones (a SELECT chain on the warp id: no runtime-indexed local array)."""
    return _select_i32(w == cutlass.Int32(0), x0, _select_i32(w == cutlass.Int32(1), x1, _select_i32(w == cutlass.Int32(2), x2, x3)))


@cute.jit
def _subtile_blocks(ids_base_i64, batch, sub, count, n_tail, tail_lo, dead):
    """The 8 block ids of sub-tile ``sub`` of the sequence's list (list indices 8 sub .. 8 sub + 7): two 16-B vector loads (the quad start
    clamped into the row; a clamped quad's words are never selected -- their index is at or past the count) and the shared
    ``block_id_at`` SELECT per index.  Returns a Python list of 8 Int32 (constexpr-indexed)."""
    first = sub * cutlass.Int32(BLOCKS_PER_SUB)
    row_base = cutlass.Int64(batch) * cutlass.Int64(BLOCK_TOPK)
    blks = []
    for qq in cutlass.range_constexpr(BLOCKS_PER_SUB // 4):
        ld_first = cute.math.min(first + cutlass.Int32(qq * 4), cutlass.Int32(BLOCK_TOPK - 4))
        w0, w1, w2, w3 = ldg_int32x4(ids_base_i64 + (row_base + cutlass.Int64(ld_first)) * cutlass.Int64(4))
        for r, word in enumerate((w0, w1, w2, w3)):
            blks.append(block_id_at(first + cutlass.Int32(qq * 4 + r), word, count, n_tail, tail_lo, dead, list_per_sequence=1, block_size=BLOCK_SIZE))
    return blks


@cute.jit
def _valid_word(blks, n_vis):
    """The sub-tile's 32-bit VALID word: bit j set iff key j (block j / 4, token j % 4) is a real key of the row's visible range --
    n_live_q = (blk >= 0) ? clamp(n_vis - 4 blk, 0, 4) : 0 live tokens of block q, as bits 4 q .. 4 q + n_live_q - 1."""
    zero = cutlass.Int32(0)
    one = cutlass.Int32(1)
    word = zero
    for q in cutlass.range_constexpr(BLOCKS_PER_SUB):
        n_live = cute.math.min(cute.math.max(n_vis - blks[q] * cutlass.Int32(BLOCK_SIZE), zero), cutlass.Int32(BLOCK_SIZE))
        n_live = _select_i32(blks[q] >= zero, n_live, 0)
        bits = (one << n_live) - one
        word = word | (bits << cutlass.Int32(q * BLOCK_SIZE))
    return word


@cute.jit
def _page_of(table_base_i64, batch, max_pages, num_pages, blk, eff_seqlen_kv):
    """PAGED arm: the pool page of block ``blk`` -- table[batch, (4 blk) / PAGE_SIZE] read at an index clamped into the row (always in
    bounds), then -1 unless the block is live (blk >= 0), its first token is inside the sequence's KV length (a block at or past it may
    name another sequence's stale page), the page index is inside the table, the entry is not -1 and below num_pages (the sparse tile's
    ``_gather_block_pages`` rule).  A -1 page becomes zero-filled rows at the copy."""
    zero = cutlass.Int32(0)
    key0 = blk * cutlass.Int32(BLOCK_SIZE)
    page_idx = key0 // cutlass.Int32(PAGE_SIZE)
    last = cute.math.max(max_pages - cutlass.Int32(1), zero)
    rd = cute.math.min(cute.math.max(page_idx, zero), last)
    page = ld_global(table_base_i64 + (cutlass.Int64(batch) * cutlass.Int64(max_pages) + cutlass.Int64(rd)) * cutlass.Int64(4), cutlass.Int32)
    ok = (blk >= zero) & (key0 < eff_seqlen_kv) & (page_idx < max_pages) & (page >= zero) & (page < num_pages)
    return _select_i32(ok, page, cutlass.Int32(-1))


@cute.jit
def _issue_gather(
    sK_raw,
    sV_raw,
    stage: cutlass.Constexpr[int],
    k_base,
    v_base,
    blk_a,
    blk_b,
    page_a,
    page_b,
    batch,
    head,
    lane,
    warp,
    eff_seqlen_kv,
    seqlen_kv,
    k_s0,
    k_st,
    k_sh,
    v_s0,
    v_st,
    v_sh,
):
    """ONE sub-tile's K + V gather into ``stage``: 8 passes, row 8 w + p (block 2 w + p / 4 = blk_a / blk_b, token p % 4), chunk ``lane``;
    16 ``cp_async_gather_row16`` per lane (zero-fill for a dead row: src = the operand base, src-size 0); the caller commits ONE group.
    The destination chunk is XOR-swizzled by the row (row & 7 = p here): the ``ldmatrix`` phases read it back conflict-free."""
    zero = cutlass.Int32(0)
    lane_elems = cutlass.Int64(lane) * cutlass.Int64(CHUNK_ELEMS)
    head64 = cutlass.Int64(head)
    for p in cutlass.range_constexpr(PASSES):
        blk = blk_a if cutlass.const_expr(p < BLOCK_SIZE) else blk_b
        page = page_a if cutlass.const_expr(p < BLOCK_SIZE) else page_b
        r = p % BLOCK_SIZE
        key0 = blk * cutlass.Int32(BLOCK_SIZE)
        tok = key0 + cutlass.Int32(r)
        in_range = tok < eff_seqlen_kv
        if cutlass.const_expr(PAGED_KV):
            live = (page >= zero) & in_range
            in_page = key0 - (key0 // cutlass.Int32(PAGE_SIZE)) * cutlass.Int32(PAGE_SIZE) + cutlass.Int32(r)
            row_k = cutlass.Int64(page) * k_s0 + head64 * k_sh + cutlass.Int64(in_page) * k_st
            row_v = cutlass.Int64(page) * v_s0 + head64 * v_sh + cutlass.Int64(in_page) * v_st
        else:
            # the extent clamp: a row at / past S_kv is zero-filled whatever L_b says (the TMA form clips it; this form must not read it)
            live = (blk >= zero) & in_range & (tok < seqlen_kv)
            row_k = cutlass.Int64(batch) * k_s0 + cutlass.Int64(tok) * k_st + head64 * k_sh
            row_v = cutlass.Int64(batch) * v_s0 + cutlass.Int64(tok) * v_st + head64 * v_sh
        src_k = _select_i64(live, row_k + lane_elems, cutlass.Int64(0))
        src_v = _select_i64(live, row_v + lane_elems, cutlass.Int64(0))
        row = warp * cutlass.Int32(ROWS_PER_WARP) + cutlass.Int32(p)
        dst = cutlass.Int32(stage * STAGE_ELEMS) + row * cutlass.Int32(ROW_ELEMS) + ((lane ^ cutlass.Int32(p)) * cutlass.Int32(CHUNK_ELEMS))
        cp_async_gather_row16(sK_raw.subview(dst), k_base + src_k, live)
        cp_async_gather_row16(sV_raw.subview(dst), v_base + src_v, live)


@cute.jit
def _issue_subtile(
    t: cutlass.Constexpr[int],
    sK_raw,
    sV_raw,
    k_base,
    v_base,
    ids_base_i64,
    table_base_i64,
    batch,
    head,
    lane,
    warp,
    sub0,
    count,
    n_tail,
    tail_lo,
    dead,
    n_vis,
    max_pages,
    num_pages,
    eff_seqlen_kv,
    seqlen_kv,
    k_s0,
    k_st,
    k_sh,
    v_s0,
    v_st,
    v_sh,
):
    """Sub-tile ``t`` of the chunk: its 8 block ids (the shared helper), the warp's two blocks and -- on the paged arm -- their pages, the
    16 zero-filled copies into stage ``t % STAGES`` and ONE commit; returns the sub-tile's valid word for its softmax."""
    blks = _subtile_blocks(ids_base_i64, batch, sub0 + cutlass.Int32(t), count, n_tail, tail_lo, dead)
    word = _valid_word(blks, n_vis)
    blk_a = _sel4(warp, blks[0], blks[2], blks[4], blks[6])
    blk_b = _sel4(warp, blks[1], blks[3], blks[5], blks[7])
    page_a = cutlass.Int32(-1)
    page_b = cutlass.Int32(-1)
    if cutlass.const_expr(PAGED_KV):
        page_a = _page_of(table_base_i64, batch, max_pages, num_pages, blk_a, eff_seqlen_kv)
        page_b = _page_of(table_base_i64, batch, max_pages, num_pages, blk_b, eff_seqlen_kv)
    _issue_gather(
        sK_raw,
        sV_raw,
        t % STAGES,
        k_base,
        v_base,
        blk_a,
        blk_b,
        page_a,
        page_b,
        batch,
        head,
        lane,
        warp,
        eff_seqlen_kv,
        seqlen_kv,
        k_s0,
        k_st,
        k_sh,
        v_s0,
        v_st,
        v_sh,
    )
    cp_async_commit()
    return word


@cute.kernel
def _kernel(
    q_tensor: cute.Tensor,
    k_tensor: cute.Tensor,
    v_tensor: cute.Tensor,
    o_tensor: cute.Tensor,
    lse_tensor: cute.Tensor,
    block_ids_tensor: cute.Tensor,
    block_lens_tensor: Optional[cute.Tensor],
    seq_kv_lens_tensor: cute.Tensor,
    block_table_tensor: Optional[cute.Tensor],
    n_batch: cutlass.Int32,
    seqlen_kv: cutlass.Int32,
    max_pages: cutlass.Int32,
    num_pages: cutlass.Int32,
    q_sb: cutlass.Int64,
    q_sh: cutlass.Int64,
    k_s0: cutlass.Int64,
    k_st: cutlass.Int64,
    k_sh: cutlass.Int64,
    v_s0: cutlass.Int64,
    v_st: cutlass.Int64,
    v_sh: cutlass.Int64,
    o_sb: cutlass.Int64,
    o_sh: cutlass.Int64,
    lse_sb: cutlass.Int64,
    lse_sh: cutlass.Int64,
    scale_softmax_log2: cutlass.Float32,
    use_pdl: cutlass.Constexpr[bool],
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    chunk, head, batch = cute.arch.block_idx()
    warp = cute.arch.make_warp_uniform(tidx // cutlass.Int32(32))
    lane = tidx % cutlass.Int32(32)
    quad = lane >> cutlass.Int32(2)  # g: the thread's row pair g, g + 8
    tq = lane & cutlass.Int32(3)  # t: the thread's column pair
    zero = cutlass.Int32(0)
    one = cutlass.Int32(1)
    NEG_INF = cutlass.Float32(float("-inf"))
    ZERO = cutlass.Float32(0.0)
    ONE = cutlass.Float32(1.0)
    LN2_F = cutlass.Float32(LN2)
    TINY = cutlass.Float32(1e-30)

    # SMEM in the header table's order: sK[STAGES] | sV[STAGES] | sS (128-B aligned rows; no descriptor).
    sK_raw = cutlass.Array(STORAGE_DTYPE, STAGES * STAGE_ELEMS, alignment=128, space=cutlass.AddressSpace.smem)
    sV_raw = cutlass.Array(STORAGE_DTYPE, STAGES * STAGE_ELEMS, alignment=128, space=cutlass.AddressSpace.smem)
    sS_raw = cutlass.Array(cutlass.Float32, S_EXCHANGE_WORDS, alignment=16, space=cutlass.AddressSpace.smem)

    if cutlass.const_expr(use_pdl):
        # PDL (header, PDL SITES): every global read below -- the lengths, the counts, the ids, the table, the Q slab, the pool rows --
        # is a predecessor's output; the release follows at once (no TMEM, no resource a dependent could contend for).
        wait_on_dependent_grids()
        launch_dependent_grids()

    q_base = cutlass.make_array_view(q_tensor).data_ptr()
    k_base = cutlass.make_array_view(k_tensor).data_ptr()
    v_base = cutlass.make_array_view(v_tensor).data_ptr()
    o_base_i64 = cutlass.make_array_view(o_tensor).data_ptr().toint()
    lse_base_i64 = cutlass.make_array_view(lse_tensor).data_ptr().toint()
    ids_base_i64 = cutlass.make_array_view(block_ids_tensor).data_ptr().toint()
    q_base_i64 = q_base.toint()

    # --- the item: ONE decode per CTA (blockIdx = (chunk, head, batch)); the visible set from the shared helper at S_q 1 (tok 0).
    pos, count, n_tail, tail_lo, dead, n_listed, eff_seqlen_kv, n_vis = visible_bounds(
        zero,
        batch,
        batch,
        one,
        block_lens_tensor,
        seq_kv_lens_tensor,
        seqlen_kv,
        seq_kv_lens_present=1,
        bottom_right=1,
        list_per_sequence=1,
        include_open_block=1,
        block_size=BLOCK_SIZE,
        block_topk=BLOCK_TOPK,
    )
    sub0 = chunk * cutlass.Int32(SUB_TILES_PER_CTA)
    table_base_i64 = cutlass.Int64(0)
    if cutlass.const_expr(PAGED_KV):
        # A zero-column table (max_pages == 0) holds no readable word: the clamped table read of _page_of is steered onto the ids row (a
        # valid address; the value is discarded because page_idx < max_pages fails and every page reads -1).  The adapter declines it too.
        table_base_i64 = _select_i64(max_pages > zero, cutlass.make_array_view(block_table_tensor).data_ptr().toint(), ids_base_i64)

    # --- the prologue: G_0 (and G_1 at k >= 2) issued before anything else, so the gather round trip overlaps the Q loads.
    words = [None] * SUB_TILES_PER_CTA
    words[0] = _issue_subtile(
        0,
        sK_raw,
        sV_raw,
        k_base,
        v_base,
        ids_base_i64,
        table_base_i64,
        batch,
        head,
        lane,
        warp,
        sub0,
        count,
        n_tail,
        tail_lo,
        dead,
        n_vis,
        max_pages,
        num_pages,
        eff_seqlen_kv,
        seqlen_kv,
        k_s0,
        k_st,
        k_sh,
        v_s0,
        v_st,
        v_sh,
    )
    if cutlass.const_expr(SUB_TILES_PER_CTA >= 2):
        words[1] = _issue_subtile(
            1,
            sK_raw,
            sV_raw,
            k_base,
            v_base,
            ids_base_i64,
            table_base_i64,
            batch,
            head,
            lane,
            warp,
            sub0,
            count,
            n_tail,
            tail_lo,
            dead,
            n_vis,
            max_pages,
            num_pages,
            eff_seqlen_kv,
            seqlen_kv,
            k_s0,
            k_st,
            k_sh,
            v_s0,
            v_st,
            v_sh,
        )

    # --- Q A-fragments (resident): rows quad / quad + 8 of the token's G heads, the warp's 64 d; rows >= G read row G - 1 and are zeroed.
    row_lo = quad
    row_hi = quad + cutlass.Int32(8)
    keep_lo = row_lo < cutlass.Int32(G)
    keep_hi = row_hi < cutlass.Int32(G)
    g_last = cutlass.Int32(G - 1)
    head_row = head * cutlass.Int32(G)
    q_row_lo = q_base_i64 + (cutlass.Int64(batch) * q_sb + cutlass.Int64(head_row + cute.math.min(row_lo, g_last)) * q_sh) * cutlass.Int64(BPE)
    q_row_hi = q_base_i64 + (cutlass.Int64(batch) * q_sb + cutlass.Int64(head_row + cute.math.min(row_hi, g_last)) * q_sh) * cutlass.Int64(BPE)
    d_lane = warp * cutlass.Int32(D_SLICE) + tq * cutlass.Int32(2)
    q_frag = []
    for ks in cutlass.range_constexpr(BMM1_K_STEPS):
        d0 = cutlass.Int64(d_lane + cutlass.Int32(ks * 16)) * cutlass.Int64(BPE)
        d1 = d0 + cutlass.Int64(8 * BPE)
        a0 = ld_global(q_row_lo + d0, cutlass.Int32)
        a1 = ld_global(q_row_hi + d0, cutlass.Int32)
        a2 = ld_global(q_row_lo + d1, cutlass.Int32)
        a3 = ld_global(q_row_hi + d1, cutlass.Int32)
        q_frag.append(_select_i32(keep_lo, a0, 0))
        q_frag.append(_select_i32(keep_hi, a1, 0))
        q_frag.append(_select_i32(keep_lo, a2, 0))
        q_frag.append(_select_i32(keep_hi, a3, 0))

    # --- the accumulator initialisation (barrier table row `init`): O = 0, l = 0, m = -inf per lane; nothing carried between items.
    o_acc = []
    for i in cutlass.range_constexpr(BMM2_N_BLOCKS * 4):
        o_acc.append(opaque_f32_zero())
    m_lo = NEG_INF
    m_hi = NEG_INF
    l_lo = ZERO
    l_hi = ZERO
    col_base = warp * cutlass.Int32(D_SLICE)
    x_slot = (warp * cutlass.Int32(BMM1_N_BLOCKS) * cutlass.Int32(32) + lane) * cutlass.Int32(S_SLOT_WORDS)  # + nb x 32 x 4 words

    for t in cutlass.range_constexpr(SUB_TILES_PER_CTA):
        stage = t % STAGES
        # A(t): this thread's copies of G_t landed (G_{t+1} may still fly), then every thread's -- the stage is full and visible.
        if cutlass.const_expr(t + 1 < SUB_TILES_PER_CTA):
            cp_async_wait(1)
        else:
            cp_async_wait(0)
        nvvm.barrier_cta_sync()

        # BMM1: S_w[16 x 32] = Q[16 x 64_w] . K[32 x 64_w]^T -- 4 k-steps x 4 n-blocks, the B fragments by ldmatrix.x4 (.row) out of sK.
        sK_stage = sK_raw.subview(stage * STAGE_ELEMS)
        s_acc = []
        for i in cutlass.range_constexpr(BMM1_N_BLOCKS * 4):
            s_acc.append(opaque_f32_zero())
        for ks in cutlass.range_constexpr(BMM1_K_STEPS):
            kb = load_b_smem_x4(
                sK_stage, k_step=ks, N=N_KEYS, sB_elems_per_row=ROW_ELEMS, b_trans=False, lane=lane, swizzle=True, elem_bytes=BPE, col_base=col_base
            )
            for nb in cutlass.range_constexpr(BMM1_N_BLOCKS):
                c0, c1, c2, c3 = mma_m16n8k16_f32(
                    q_frag[ks * 4 + 0],
                    q_frag[ks * 4 + 1],
                    q_frag[ks * 4 + 2],
                    q_frag[ks * 4 + 3],
                    kb[nb * 2 + 0],
                    kb[nb * 2 + 1],
                    s_acc[nb * 4 + 0],
                    s_acc[nb * 4 + 1],
                    s_acc[nb * 4 + 2],
                    s_acc[nb * 4 + 3],
                    STORAGE_DTYPE,
                )
                s_acc[nb * 4 + 0] = c0
                s_acc[nb * 4 + 1] = c1
                s_acc[nb * 4 + 2] = c2
                s_acc[nb * 4 + 3] = c3

        # X(t): the S exchange -- the warp's 4 partial C fragments to its slots, bar.sync, every warp sums the 4 slices in the FIXED order 0..3.
        for nb in cutlass.range_constexpr(BMM1_N_BLOCKS):
            off = x_slot + cutlass.Int32(nb * 32 * S_SLOT_WORDS)
            Pointer(sS_raw.subview(off).data_ptr(), dtype=cutlass.Float32).store(
                cutlass.Vector.from_elements(tuple(s_acc[nb * 4 + i] for i in range(4)), cutlass.Float32), alignment=16
            )
        nvvm.barrier_cta_sync()
        s_sum = [None] * (BMM1_N_BLOCKS * 4)
        for ww in cutlass.range_constexpr(WARPS):
            for nb in cutlass.range_constexpr(BMM1_N_BLOCKS):
                off = cutlass.Int32((ww * BMM1_N_BLOCKS + nb) * 32 * S_SLOT_WORDS) + lane * cutlass.Int32(S_SLOT_WORDS)
                v = Pointer(sS_raw.subview(off).data_ptr(), dtype=cutlass.Float32).load(alignment=16, count=4)
                for i in cutlass.range_constexpr(4):
                    if cutlass.const_expr(ww == 0):
                        s_sum[nb * 4 + i] = cutlass.Float32(v[i])
                    else:
                        s_sum[nb * 4 + i] = s_sum[nb * 4 + i] + cutlass.Float32(v[i])

        # softmax(t): the valid word -> masked scaled scores -> the quad row max -> the exact online update -> P packed in place.
        word = words[t]
        s_m = [None] * (BMM1_N_BLOCKS * 4)
        tile_lo = NEG_INF
        tile_hi = NEG_INF
        for nb in cutlass.range_constexpr(BMM1_N_BLOCKS):
            # each lane's four cells of n-block nb are keys 8 nb + 2 t + {0, 1} on rows g and g + 8: the shared membership op in fragment order
            frag = apply_membership_fragment(
                (
                    s_sum[nb * 4 + 0] * scale_softmax_log2,
                    s_sum[nb * 4 + 1] * scale_softmax_log2,
                    s_sum[nb * 4 + 2] * scale_softmax_log2,
                    s_sum[nb * 4 + 3] * scale_softmax_log2,
                ),
                word,
                cutlass.Int32(nb * 8) + tq * cutlass.Int32(2),
                (0, 1, 0, 1),
            )
            s_m[nb * 4 + 0] = frag[0]
            s_m[nb * 4 + 1] = frag[1]
            s_m[nb * 4 + 2] = frag[2]
            s_m[nb * 4 + 3] = frag[3]
            tile_lo = cute.math.max(tile_lo, cute.math.max(s_m[nb * 4 + 0], s_m[nb * 4 + 1], ftz=True), ftz=True)
            tile_hi = cute.math.max(tile_hi, cute.math.max(s_m[nb * 4 + 2], s_m[nb * 4 + 3], ftz=True), ftz=True)
        tile_lo = quad_reduce_max(tile_lo)
        tile_hi = quad_reduce_max(tile_hi)
        m_lo_new = cute.math.max(m_lo, tile_lo, ftz=True)
        m_hi_new = cute.math.max(m_hi, tile_hi, ftz=True)
        ms_lo_old = row_max_for_exp2(m_lo)
        ms_hi_old = row_max_for_exp2(m_hi)
        ms_lo = row_max_for_exp2(m_lo_new)
        ms_hi = row_max_for_exp2(m_hi_new)
        alpha_lo = cute.math.exp2(cute.math.min(ms_lo_old - ms_lo, ZERO, ftz=True), fastmath=True)
        alpha_hi = cute.math.exp2(cute.math.min(ms_hi_old - ms_hi, ZERO, ftz=True), fastmath=True)
        m_lo = m_lo_new
        m_hi = m_hi_new
        p = [None] * (BMM1_N_BLOCKS * 4)
        psum_lo = ZERO
        psum_hi = ZERO
        for nb in cutlass.range_constexpr(BMM1_N_BLOCKS):
            p[nb * 4 + 0] = cute.math.exp2(s_m[nb * 4 + 0] - ms_lo, fastmath=True)
            p[nb * 4 + 1] = cute.math.exp2(s_m[nb * 4 + 1] - ms_lo, fastmath=True)
            p[nb * 4 + 2] = cute.math.exp2(s_m[nb * 4 + 2] - ms_hi, fastmath=True)
            p[nb * 4 + 3] = cute.math.exp2(s_m[nb * 4 + 3] - ms_hi, fastmath=True)
            psum_lo = psum_lo + (p[nb * 4 + 0] + p[nb * 4 + 1])
            psum_hi = psum_hi + (p[nb * 4 + 2] + p[nb * 4 + 3])
        l_lo = l_lo * alpha_lo + quad_reduce_sum(psum_lo)
        l_hi = l_hi * alpha_hi + quad_reduce_sum(psum_hi)
        if cutlass.const_expr(t > 0):
            # The EXACT rescale of the running O (never lazy): rows g by alpha_lo, rows g + 8 by alpha_hi.
            for nb in cutlass.range_constexpr(BMM2_N_BLOCKS):
                o_acc[nb * 4 + 0] = o_acc[nb * 4 + 0] * alpha_lo
                o_acc[nb * 4 + 1] = o_acc[nb * 4 + 1] * alpha_lo
                o_acc[nb * 4 + 2] = o_acc[nb * 4 + 2] * alpha_hi
                o_acc[nb * 4 + 3] = o_acc[nb * 4 + 3] * alpha_hi
        p_frag = []
        for ks in cutlass.range_constexpr(BMM2_K_STEPS):
            lo_nb = 2 * ks
            hi_nb = 2 * ks + 1
            p_frag.append(fp32_to_fp16(p[lo_nb * 4 + 0], p[lo_nb * 4 + 1], dtype=STORAGE_DTYPE))
            p_frag.append(fp32_to_fp16(p[lo_nb * 4 + 2], p[lo_nb * 4 + 3], dtype=STORAGE_DTYPE))
            p_frag.append(fp32_to_fp16(p[hi_nb * 4 + 0], p[hi_nb * 4 + 1], dtype=STORAGE_DTYPE))
            p_frag.append(fp32_to_fp16(p[hi_nb * 4 + 2], p[hi_nb * 4 + 3], dtype=STORAGE_DTYPE))

        # BMM2: O_w[16 x 64_w] += P[16 x 32] . V[32 x 64_w] -- 2 k-steps x 8 n-blocks, the B fragments by ldmatrix.x4.trans out of sV.
        sV_stage = sV_raw.subview(stage * STAGE_ELEMS)
        for ks in cutlass.range_constexpr(BMM2_K_STEPS):
            vb = load_b_smem_x4(
                sV_stage, k_step=ks, N=D_SLICE, sB_elems_per_row=ROW_ELEMS, b_trans=True, lane=lane, swizzle=True, elem_bytes=BPE, col_base=col_base
            )
            for nb in cutlass.range_constexpr(BMM2_N_BLOCKS):
                c0, c1, c2, c3 = mma_m16n8k16_f32(
                    p_frag[ks * 4 + 0],
                    p_frag[ks * 4 + 1],
                    p_frag[ks * 4 + 2],
                    p_frag[ks * 4 + 3],
                    vb[nb * 2 + 0],
                    vb[nb * 2 + 1],
                    o_acc[nb * 4 + 0],
                    o_acc[nb * 4 + 1],
                    o_acc[nb * 4 + 2],
                    o_acc[nb * 4 + 3],
                    STORAGE_DTYPE,
                )
                o_acc[nb * 4 + 0] = c0
                o_acc[nb * 4 + 1] = c1
                o_acc[nb * 4 + 2] = c2
                o_acc[nb * 4 + 3] = c3

        # F(t) (exists iff a refill follows, t + 2 < k): the stage is free once every warp's BMM2 reads are behind the barrier; the
        # loader then refills it with G_{t+2}.  The barrier and the issue fold together: the last two sub-tiles refill nothing.
        if cutlass.const_expr(t + 2 < SUB_TILES_PER_CTA):
            nvvm.barrier_cta_sync()
            words[t + 2] = _issue_subtile(
                t + 2,
                sK_raw,
                sV_raw,
                k_base,
                v_base,
                ids_base_i64,
                table_base_i64,
                batch,
                head,
                lane,
                warp,
                sub0,
                count,
                n_tail,
                tail_lo,
                dead,
                n_vis,
                max_pages,
                num_pages,
                eff_seqlen_kv,
                seqlen_kv,
                k_s0,
                k_st,
                k_sh,
                v_s0,
                v_st,
                v_sh,
            )

    # --- the epilogue (= the sparse tile's split store): the per-row sums are quad-uniform and warp-identical already.
    l_tot_lo = l_lo
    l_tot_hi = l_hi
    m_nat_lo = row_max_for_exp2(m_lo) * LN2_F
    m_nat_hi = row_max_for_exp2(m_hi) * LN2_F
    row_dead_lo = l_tot_lo <= ZERO
    row_dead_hi = l_tot_hi <= ZERO
    # The denominator floor sits inside the log and the reciprocal ONLY and is always followed by the row_dead SELECT, so neither LSE
    # nor O ever carries it (sdpa-invariants sections 2-4): keep these lines together.
    lse_lo = m_nat_lo + cute.math.log(cute.math.max(l_tot_lo, TINY), fastmath=True)
    lse_hi = m_nat_hi + cute.math.log(cute.math.max(l_tot_hi, TINY), fastmath=True)
    inv_lo = ONE / cute.math.max(l_tot_lo, TINY)
    inv_hi = ONE / cute.math.max(l_tot_hi, TINY)
    lse_lo = _select_f32(row_dead_lo, NEG_INF, lse_lo)
    lse_hi = _select_f32(row_dead_hi, NEG_INF, lse_hi)
    inv_lo = _select_f32(row_dead_lo, ZERO, inv_lo)
    inv_hi = _select_f32(row_dead_hi, ZERO, inv_hi)

    # Stores: the split-major partial batch b + c x B (widened BEFORE the stride multiply), rows h G + g / h G + g + 8 of token 0.
    part_batch = cutlass.Int64(batch) + cutlass.Int64(chunk) * cutlass.Int64(n_batch)
    o_row_lo = o_base_i64 + (part_batch * o_sb + cutlass.Int64(head_row + row_lo) * o_sh) * cutlass.Int64(4)
    o_row_hi = o_base_i64 + (part_batch * o_sb + cutlass.Int64(head_row + row_hi) * o_sh) * cutlass.Int64(4)
    for nb in cutlass.range_constexpr(BMM2_N_BLOCKS):
        d_off = cutlass.Int64(d_lane + cutlass.Int32(nb * 8)) * cutlass.Int64(4)
        v0 = _select_f32(row_dead_lo, ZERO, o_acc[nb * 4 + 0] * inv_lo)
        v1 = _select_f32(row_dead_lo, ZERO, o_acc[nb * 4 + 1] * inv_lo)
        v2 = _select_f32(row_dead_hi, ZERO, o_acc[nb * 4 + 2] * inv_hi)
        v3 = _select_f32(row_dead_hi, ZERO, o_acc[nb * 4 + 3] * inv_hi)
        if keep_lo:
            st_global_v2(o_row_lo + d_off, (v0, v1), cutlass.Float32)
        if keep_hi:
            st_global_v2(o_row_hi + d_off, (v2, v3), cutlass.Float32)
    # The LSE partial: one lane per row (t == 0 of warp 0; l and m are quad-uniform and warp-identical).
    lse_lane = (tq == zero) & (warp == zero)
    if lse_lane & keep_lo:
        st_global(lse_base_i64 + (part_batch * lse_sb + cutlass.Int64(head_row + row_lo) * lse_sh) * cutlass.Int64(4), lse_lo, cutlass.Float32)
    if lse_lane & keep_hi:
        st_global(lse_base_i64 + (part_batch * lse_sb + cutlass.Int64(head_row + row_hi) * lse_sh) * cutlass.Int64(4), lse_hi, cutlass.Float32)


# === host =================================================================================================================


@cute.jit
def _host(
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    v_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    lse_ptr: cute.Pointer,
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
    use_pdl: cutlass.Constexpr[bool],
    stream: _cuda_driver.CUstream,
) -> None:
    """Bind the explicit pointer ABI and launch the lean tile -- the tcgen05 sparse tile's ``_host`` contract, decode form, minus the arms
    the lean tile does not serve (THD, the epilogue gate).

    ``problem_size`` = (B, QH, KH, SQ, SKV) with SQ = 1 (the S_q = 1 rendering: the config pins S_Q 1; a caller passing SQ > 1 is the adapter's
    bug -- the token coordinate is 0); strides are the caller's (batch, seq, head) ELEMENT strides (Int64 leaves); ``block_ids_ptr`` -> int32
    [B, BLOCK_TOPK] contiguous (ONE list per sequence), ``block_lens_ptr`` -> int32 [B] or None, ``seq_kv_lens_ptr`` -> int32 [B] (REQUIRED:
    the decode form's visible range).  ``o_ptr`` -> the fp32 O PARTIAL slab (B x SPLIT, 1, QH, D) and ``lse_ptr`` -> the fp32 LSE partial
    slab (B x SPLIT, QH, 1), both split-major on the batch axis (b + c x B) at the strides given -- the SAME layout the tcgen05 tile's split
    store writes, so ``sm100/split_combine.py`` reduces them unchanged (the adapter's second launch; the gate rides it).  The grid is
    (SPLIT, KH, B) with SPLIT = CFG.SPLIT = ceil(n_sub / k).

    PAGED arm (CFG.PAGED_KV): ``k_strides`` / ``v_strides`` = (page stride, TOKEN stride, COLUMN head stride -- 0 for an HND pool whose head
    folds into the row), ``block_table_ptr`` -> int32 (B, max_pages) contiguous, ``paged_geom`` = (num_pages, max_pages, k rows_per_page, k
    rows_per_head, v rows_per_page, v rows_per_head) -- the adapter's per-operand numbers; the kernel folds them into ONE element head
    stride per operand (rows_per_head x token stride + column head stride) so HND and NHD pools take one address form.  SKV is the table's
    capacity; the visible range comes from ``seq_kv_lens``.  On the dense arm ``block_table_ptr`` is None and ``paged_geom`` is unread.

    PDL (``use_pdl``, a compile-time constant of this entry -- the adapter's ``compile(use_pdl=)`` key; NO default: a defaulted slot ahead of
    ``stream`` breaks the compiled entry, frost-gotchas): the launch takes the programmatic-stream-serialization attribute and the kernel
    carries the griddepcontrol pair (header, PDL SITES).  The same math, bitwise.
    """
    B, QH, KH, SQ, SKV = problem_size
    num_pages, max_pages, k_rows_per_page, k_rows_per_head, v_rows_per_page, v_rows_per_head = paged_geom
    n_o_batches = B * SPLIT  # the split-major partial slabs' batch axis (b + c x B)
    q_tensor = _bshd(q_ptr, B, SQ, QH, D, q_strides, False)
    o_tensor = _bshd(o_ptr, n_o_batches, SQ, QH, D, o_strides, False)
    l0, l1, l2 = lse_strides
    lse_tensor = cute.make_tensor(lse_ptr, cute.make_layout((n_o_batches, QH, SQ), stride=(l0, l1, l2)))
    block_ids_tensor = _vec(block_ids_ptr, B * BLOCK_TOPK)
    block_lens_tensor = None
    if cutlass.const_expr(block_lens_ptr is not None):
        block_lens_tensor = _vec(block_lens_ptr, B)
    seq_kv_lens_tensor = _vec(seq_kv_lens_ptr, B)
    block_table_tensor = None
    if cutlass.const_expr(PAGED_KV):
        block_table_tensor = cute.make_tensor(block_table_ptr, cute.make_layout((B, max_pages), stride=(max_pages, 1)))
        # The pool as a (pages, page rows, heads, d) tensor: only its base pointer is read; the strides below do the addressing.
        k_tensor = _bshd(k_ptr, num_pages, k_rows_per_page, KH, D, k_strides, False)
        v_tensor = _bshd(v_ptr, num_pages, v_rows_per_page, KH, D, v_strides, False)
        k_sh = cutlass.Int64(k_rows_per_head) * cutlass.Int64(k_strides[1]) + cutlass.Int64(k_strides[2])
        v_sh = cutlass.Int64(v_rows_per_head) * cutlass.Int64(v_strides[1]) + cutlass.Int64(v_strides[2])
    else:
        k_tensor = _bshd(k_ptr, B, SKV, KH, D, k_strides, False)
        v_tensor = _bshd(v_ptr, B, SKV, KH, D, v_strides, False)
        k_sh = cutlass.Int64(k_strides[2])
        v_sh = cutlass.Int64(v_strides[2])
    _kernel(
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse_tensor,
        block_ids_tensor,
        block_lens_tensor,
        seq_kv_lens_tensor,
        block_table_tensor,
        cutlass.Int32(B),
        cutlass.Int32(SKV),
        cutlass.Int32(max_pages),
        cutlass.Int32(num_pages),
        cutlass.Int64(q_strides[0]),
        cutlass.Int64(q_strides[2]),
        cutlass.Int64(k_strides[0]),
        cutlass.Int64(k_strides[1]),
        k_sh,
        cutlass.Int64(v_strides[0]),
        cutlass.Int64(v_strides[1]),
        v_sh,
        cutlass.Int64(o_strides[0]),
        cutlass.Int64(o_strides[2]),
        cutlass.Int64(lse_strides[0]),
        cutlass.Int64(lse_strides[1]),
        scale_softmax_log2,
        use_pdl,
    ).launch(
        grid=(SPLIT, KH, B),
        block=[THREADS, 1, 1],
        cluster=(1, 1, 1),
        stream=stream,
        use_pdl=use_pdl,  # the programmatic-stream-serialization attribute, paired with the kernel's griddepcontrol sites
    )


EXPLICIT_ABI = True  # pointer/int host entry; the adapter builds the argument list itself


@lru_cache(maxsize=None)
def compile(  # noqa: A001
    has_lse: bool = True,
    has_block_lens: bool = False,
    use_pdl: bool = False,
) -> Callable:
    """Compile the pointer ABI: ``has_lse`` must be True (the per-split LSE partial drives the combine; the parameter mirrors the tcgen05
    module's key), ``has_block_lens`` folds the per-sequence count read in / out; the paged arm's block table is present exactly when the
    config is paged (CFG.PAGED_KV, part of the template's identity).  Head dims are the config's (d_qk = d_v = 256, exact); shapes and
    strides are runtime arguments.  ``use_pdl`` (in the key) traces the programmatic-dependent-launch form (the launch attribute plus the
    griddepcontrol pair before the first global read) -- the same math, bitwise; the False rendering carries no griddepcontrol token."""
    if not has_lse:
        raise ValueError("sparse_lean_d256_f16: has_lse must be True (the per-split LSE partial drives the combine)")
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
        P(cutlass.Float32),  # the fp32 partial slab
        P(cutlass.Float32, 4),
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
        bool(use_pdl),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
        cache_key=_cache_key,
        symbol="frost_sdpa_fwd_sparse_lean",
    )
