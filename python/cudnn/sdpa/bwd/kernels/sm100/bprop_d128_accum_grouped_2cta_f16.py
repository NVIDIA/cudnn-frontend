# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""SM100 SDPA backward d=128x128 bf16/fp16 -- grouped THD, cta_group::2 datapath  (section 1: identity)

HEADER FIRST: the tables below come before the body; a reader checks the body against them instead of
reconstructing the pipeline from code.  Layout skeleton: resource-manager (2026-10-26); ``Bars`` / ``BAR_LAYOUT`` / the init
block / the ``# BARRIER`` punch lists: barrier-inspector (2026-10-26, SKELETON mode); the role bodies: code-writer-dsl
(2026-10-26), forked from ``bprop_d128_accum_thd_f16.py`` / ``bprop_d128_accum_grouped_f16.py`` onto this layout.  Config as
truth: ``config_sm100_d128_accum_2cta.CFG`` (every number below is derived there and asserted at import).

2. WHAT IT COMPUTES (per (batch, kv head, q head); GQA: one q-head owner per pair, as today's grouped loaders)
    S^T = K Q^T * scale        P^T = exp2(S^T * log2e - LSE[q])        dP^T = V dO^T
    dS^T = P^T * (dP^T - delta[q])        dV += P^T dO        dK += dS^T Q        dQ = dS K  (fp32, tma_reduce_add)
    delta = rowsum(dO * O) comes from the chain's delta launch (THD: setup_thd -> delta_zero_thd -> MAIN -> cast_dq_thd
    -> the fp32 group fold); LSE in the natural-log unit of the forward's Stats.

3. WORK ITEM + MMA SHAPES (cluster (2,1,1) = one cga2 PAIR; CTA rank r owns kv rows [128 r, +128) of a 256-row block)
    one item = 256 kv rows of one (sequence, kv head) x ONE q head; the pair walks the identical q tiles (128 rows)
    S^T  [256 kv, 128 q]  = K1[own 128 kv x 128 d] (A, K-major SW128, two boxes [128 x 64]) . Q1^T (B N-split:
                              Q[64 r.., +64 q x 128 d], two boxes [64 x 64])          cta_group::2, M 256, 8 k-steps
    dP^T [256 kv, 128 q]  = V[own 128 kv x 128 d] (A) . dO1^T (B N-split: dO[64 q x 128 d])             M 256
    dV   [256 kv, 128 d] += P^T (A, TS: own TMEM, packed bf16) . dO2 (B N-split along d: dO[128 q x 64 d], MN-major,
                              one box [128 x 64])                                                   M 256, k = q
    dK   [256 kv, 128 d] += dS^T (A, TS: own TMEM) . Q2 (B N-split along d: Q[128 q x 64 d], MN-major)  M 256
    dQ   [128 q, 128 d]   = dS (A, M-split: own 64 q rows): sDS = dS^T [256 kv x own 64 q] MN-major SW128 (128 B per
                              kv row; own 128 kv rows by the epilogue's 16-B STS, the peer's 128 rows by ONE
                              cp.async.bulk DSMEM copy of its XCHG) . K2 (B N-split along d: K[256 kv x 64 d],
                              MN-major, two 128-row boxes)                          cta_group::2, M 128, K 256 (16 steps)
    The D image of the M 128 pair MMA is the half-row atom: dQ row m -> lane m % 64 + 64 * (d // 64), 64 columns.
    Operand bytes per CTA per q tile: Q1 16 + Q2 16 + dO1 16 + dO2 16 KiB (= today's 64 KiB); per kv block
    K1 32 + K2 32 + V 32 KiB (+32 vs today: K is read twice, the user's "TMA loads Q twice" applies to K as well).
    TS k-step order: the packed P^T / dS^T columns follow the softmax quarter ownership (group g owns q quarters {g, g+2}),
    so TS k-step k (8 packed columns = 16 q) pairs with the B q block CFG.ts_k_perm(k) = [0, 1, 4, 5, 2, 3, 6, 7][k].

4. WARP MAP        role | warps | regs (setmaxnreg) | what it credits (names = the inspector's table)
    softmax   | 0-7   | 144 | p_ready, ds_ready (pair-wide on the leader), ds_tmem (local), stats consumption
    epilogue  | 8-11  | 144 | ds_smem (own half by STS), ds_recv_full (expect_tx) + the XCHG ship, dp_empty (leader), dq_full x 4,
                             dv_full, dk_full
    mma       | 12    |  80 | LEADER only: s_done, dp_done, dq_done (commit, pair multicast), *_empty releases; the
                             follower's MMA warp runs the quiet body (alloc, the ds_pair_ready relay, dealloc)
    tma_ldg   | 13    |  80 | q1_full[2], q2_full, do1_full[2], do2_full, k1_full, k2_full, v_full (expect_tx on the
                             LEADER for 2x bytes; the follower issues its half naming the leader's mbar)
    tma_stg   | 14    |  80 | dq_empty x 4 (reduce-add drained), dvk_empty (leader), k2_empty
    sched     | 15    |  80 | sched_full / sched_empty (CLC payload delivered to the pair); never credits its own ring
    SUM(role regs x warps) = (8 x 144 + 4 x 144 + 4 x 80) x 32 = 65536 == entry 128 x 16 warps x 32; every count % 8 == 0.
    KNOB reg_split (CFG.REG_SPLITS, typed ``reg_split`` field -> ``reg_split_code`` Constexpr; default 144/144/80 = this table):
    136/136/104 = FA4's working point (M5': softmax / epilogue -8 each, the 4 service warps +24; the same formula, asserted in the
    config); the counts reach every role as ``regs``, the setmaxnreg DIRECTIONS (inc / inc / dec) are fixed by s, e >= 128 >= v.
    KNOB store_wait_form (``store_wait_spin`` Constexpr; default retry): the store warp's six mbarrier waits (dq_full x 4, dv_full,
    dk_full) in the hint-less 2-instruction spin form (`_wait_store`) instead of the declared parity-wait retry loop; sched_full and
    tma_store_wait (one spelling) unchanged.  Both knobs: the same function bitwise; the default rendering unchanged (c28e6013).

5. BARRIER TABLE  (barrier-inspector 2026-10-26, SKELETON mode; ``config_sm100_d128_accum_2cta.BAR_LAYOUT`` = offsets / counts / kinds;
   32 names / 43 slots in [230400, 230944) of the 944-B bar block).  L = pair LEADER (rank 0, the only MMA issuer), F = FOLLOWER (rank 1);
   "(-)" = the follower's copy is initialised with the SAME count and never armed / never waited; x N = per q tile, x T = per CLC item
   (256-row kv block).  Issuing lanes come from the guard form ONLY: bare 1 warp = 32, bare 8 warps = 256, pred / elect = 1 per warp,
   pred=is_leader & elect = 1 for the pair, a tcgen05.commit = 1 arrive per TARGET CTA; x the CTAs that reach the site.
    name[st]         kind              init L / F     SUM(issuing lanes) == init        waiter (own copy unless "L only")   start  fires
    q1_full[2]       TMA_LOAD          1 / 1(-)       1 (L, pred=is_leader & elect) == 1  MMA(L) before S^T(t)              st(0)  x N
        arm: loader `q1_full[s].arrive(n_bytes=Q1_TX, pred=is_leader & elect_sync())`; BOTH CTAs issue tma_load_tile(cta_group=2, mcast_mask=1 << rank)
        (tma_load_form=plain: mcast_mask=None, the same bytes on the LEADER's mbar by the wrapper's bit-24 routing: no row changes)
    q1_empty[2]      MMA_COMMIT        1 / 1          1 commit x 2 target CTAs == 1 each  loader L+F before the Q1 reload     st(1)  x N
        site: MMA(L) right after S^T(t) issue, `_commit_pair(bars.q1_empty[s], elect_p)` (same site as s_done); DRAIN x Q1_STAGES
    q2_full[1]       TMA_LOAD          1 / 1(-)       1 (L) == 1  (Q2_TX)                 MMA(L) before dK(t)               st(0)  x N
    q2_empty[1]      MMA_COMMIT        1 / 1          1 x 2 == 1 each                      loader L+F                        st(1)  x N  (after dK(t); DRAIN x1)
    do1_full[2]      TMA_LOAD          1 / 1(-) /slot 1 (L) per half slot == 1 (DO1_HALF_TX each, range_constexpr(2))  MMA(L) dP^T k-half 0 | 1  st(0)  x N
    do1_empty[1]     MMA_COMMIT        1 / 1          1 x 2 == 1 each                      loader L+F                        st(1)  x N
            (after dP^T(t); DRAIN x1)
    do2_full[1]      TMA_LOAD          1 / 1(-)       1 (L) == 1  (DO2_TX)                MMA(L) before dV(t)               st(0)  x N
    do2_empty[1]     MMA_COMMIT        1 / 1          1 x 2 == 1 each                      loader L+F                        st(1)  x N  (after dV(t); DRAIN x1)
    k1_full[1]       TMA_LOAD          1 / 1(-)       1 (L) == 1  (K1_TX)                 MMA(L) before S^T(lo)             st(0)  x T
    k1_empty[1]      MMA_COMMIT        1 / 1          1 x 2 == 1 each                      loader L+F                        st(1)  x T  (block end; DRAIN x1)
    v_full[1]        TMA_LOAD          1 / 1(-)       1 (L) == 1  (V_TX)                  MMA(L) before dP^T(lo)            st(0)  x T
    v_empty[1]       MMA_COMMIT        1 / 1          1 x 2 == 1 each                      loader L+F                        st(1)  x T  (block end; DRAIN x1)
    k2_full[1]       TMA_LOAD          1 / 1(-)       1 (L) == 1  (K2_TX)                 MMA(L) before dQ(lo)              st(0)  x T
    k2_empty[1]      THREAD            32 / 32        store warp bare after tma_store_wait(0) = 32 == 32   loader (own) before K2  st(1)  x T (every item)
        the MMA's read of K2 is covered by dq_done(t_last) (observed by the epilogue before dv_full) -> dV store -> drain -> this arrive: barrier
            chain, no issue-order invariant
        BACK-EDGE (barrier-inspector AUDIT 2026-10-26 fix A): the loader waits it right before the K2 reload (live item) or once at the
            item top (zero-trip item), then arrives dv_full (bare 32); the store's NEXT k2_empty arrive follows its dv_full wait, so the
            single waiter can never be lapped by a zero-trip item (P14-B); only the K2 reload is gated by the previous dV store drain
    s_done[1]        MMA_COMMIT        1 / 1          1 x 2 == 1 each                      softmax L+F                       st(0)  x N
    dp_done[1]       MMA_COMMIT        1 / 1          1 x 2 == 1 each                      softmax L+F                       st(0)  x N
    dq_done[1]       MMA_COMMIT        1 / 1          1 x 2 == 1 each (+1 empty commit per zero-trip item)   epilogue L+F   st(0)  x N (+1 dead)
    p_ready[1]       LEADER (relaxed)  512 / 512(-)   softmax 8 warps bare x 2 CTAs = 256 x 2 = 512 == 512    MMA(L) before dV(t)    st(0)  x N
        site: after tcgen05_wait(STORE) + tcgen05_fence(BEFORE_THREAD_SYNC): `_arrive_leader(bars.p_ready)` (TMEM data: relaxed is right, P8)
    ds_ready[1]      LEADER (relaxed)  512 / 512(-)   same lanes, after the dS^T pack = 512 == 512            MMA(L) before dK(t)    st(0)  x N
    ds_tmem[1]       THREAD            256 / 256      softmax 8 warps bare, local, same site = 256 == 256     epilogue (own)         st(0)  x N
    dp_empty[1]      LEADER (relaxed)  256 / 256(-)   epilogue 4 warps bare x 2 CTAs = 128 x 2 = 256 == 256   MMA(L) before dP^T(t+1)  st(1)  x N
            ; DRAIN x1 (MMA(L))
        site: after the dQ tcgen05.ld + tcgen05_wait(LOAD) + fence: `_arrive_leader(bars.dp_empty)` (dQ sits in dP's dead half: dp_empty covers it)
    ds_smem[1]       THREAD            128 / 128      epilogue 4 warps bare after own-half STS + fence_proxy = 128 == 128
            MMA warp (own): L dQ issue, F relay   st(0) x N
    ds_recv_full[1]  TMA_LOAD (dsmem)  1 / 1          epilogue warp 8, pred=is_lead_warp & elect = 1 == 1; tx = the PEER's cp.async.bulk XCHG_TX
            MMA warp (own)  st(0) x N
    ds_pair_ready[1] THREAD (remote)   1 / 1(-)       MMA(F) `arrive_on_peer(0, pred=elect_p
            , release_cta=True)` after ds_smem + ds_recv_full = 1 -> L == 1   MMA(L) before dQ(t)  st(0) x N
    dq_empty[4]      THREAD            32 / 32 /slot  store warp bare: 1 bootstrap per item + 1 per tile after tma_store_wait = 32 == 32
            epilogue (own)   st(0)  x N (+1 / item)
    dq_full[4]       THREAD            64 / 64 /slot  epilogue, the TWO warps of d half i // 2, bare = 2 x 32 = 64 == 64 (DQ_SLICE_LANES)
            store (own)  st(0)  x N
    dv_full[1]       THREAD            160 / 160      epilogue 4 warps bare after the dV staging STS + fence = 128 + loader bare after its
            k2_empty observation = 32 -> 160 == DV_FULL_LANES   store (own)   st(0)  x T   (AUDIT 2026-10-26 fix A: the k2_empty back-edge)
    dk_full[1]       THREAD            128 / 128      epilogue 4 warps bare after the dK staging STS + fence = 128 == 128    store (own)          st(0)  x T
    dvk_empty[1]     LEADER (relaxed)  64 / 64(-)     store warp bare x 2 CTAs after tma_store_wait(0) = 32 x 2 = 64 == 64   MMA(L) item top
            st(1)  x T; DRAIN x1 (MMA(L))
    sched_full[2]    TMA_LOAD (CLC)    1 / 1          scheduler(L) elect: local arrive(n_bytes=CLC_TX) + mapa(F) mbarrier_arrive_expect_tx(CLC_TX
            , CTA) = 1 per CTA == 1
        completion: clusterlaunchcontrol.try_cancel(multicast=1) lands 16 B in EACH CTA's slot
            ; waiters: every consumer warp (own copy) + scheduler(L)  st(0) x T
    sched_empty[2]   THREAD (remote)   30 / 30(-)     read_tile_id_arrive_on_leader(mb, CGA_M) from 15 warps per CTA (all but the scheduler) x 2 = 30 == 30
        waiter: scheduler(L) before re-arming the slot (every phase observed in-loop: the credits of tile n gate payload n + 1)   st(0) x T
    tmem_dealloc[1]  THREAD (+peer)    2 / 2          MMA warp, each CTA, after the final CTA sync: `if elect: arrive(); arrive_on_peer(peer rank)` = 1 + 1 == 2
        waiter: the MMA warp of each CTA, wait(0) -> tcgen05_fence -> tcgen05.dealloc.cta_group::2 (the wait IS the drain of the peer's arrive)  x 1
    Lane ledger: every row above written as SUM == init on BOTH CTAs; unresolved: none.  Non-mbarrier sites: named barrier 0 (3 x
    barrier_cta_sync: init, TMEM-base publish, final), CFG.SOFTMAX_BARRIER (256 lanes, 2 per _stats), CFG.XCHG_BARRIER (128 lanes, 1 per tile).
    SMEM lifetime seams (Rule S3: each alias reuse is an explicit barrier chain): dQ staging over ds_own + XCHG <- dq_done(t) (the dQ
    MMA read sDS and the bulk copy read XCHG: the copy completed before ds_recv_full on the peer, before ds_pair_ready, before the dQ
    issue) -> dq_empty[i] (slice i's reduce-add read) -> the next dS / XCHG write; recv half <- dq_done(t) on the PEER's timeline (its
    copy(t + 1) follows its dq_done(t) wait); dK staging over the same bytes <- the item-end dq_empty x 4 observation, released by the
    NEXT item's dq_empty bootstrap (store warp, after tma_store_wait(0)); dV staging over K2 <- dq_done(t_last) (K2 read) + the
    dq_empty bootstrap (previous dV store drained), released to the loader by k2_empty.
    BODY STATEMENT (code-writer 2026-10-26): table unchanged from the SKELETON -- no barrier added, removed or re-counted; the
    scheduler warp is the library's ``scheduler_warp_loop`` (the same arm / try_cancel / wait sites as the punch list).
    AUDIT FOLD (barrier-inspector 23:10 UTC, folded by the code-writer): dv_full init 128 -> DV_FULL_LANES 160 with the loader's bare
    arrive after its k2_empty wait (fix A, no new row) and the k2_empty wait moved from the item top to the K2 reload (perf note);
    every other row, site and count unchanged.  Test-only lever 4 (``DEBUG_LOADER_DELAY_US``, default 0 = no traced code) makes the
    pre-fix lap deterministic.  Skew levers 1-3 (code-writer 2026-10-27, the AUDIT's forms, same house form, default 0): the
    FOLLOWER epilogue between the XCHG ship and its own-half STS (ds_pair_ready), the LEADER epilogue before its XCHG ship
    (ds_recv_full + P9 routing), the FOLLOWER epilogue before the item-end dV / dK readout (dvk_empty's pair-wide 64); no row, site
    or count changed by a lever: each is a sleep on one role of one CTA inside an existing phase.
    DIAGNOSTIC PROBES (code-writer 2026-10-27, typed field ``diag_2cta``, section 11; default None = this table's body): D7 removes
    the leader's per-tile dp_empty WAIT (row dp_empty: the 256 arrives per tile and the DRAIN x1 stay; the drain's parity test passes
    once the kernel's N arrives landed -- no hang path), D8 removes the leader's ds_smem / ds_recv_full / ds_pair_ready WAITs before
    dQ(t) (rows unchanged: the epilogue's 128 / 1 arrives and the relay's remote arrive stay; no drain reads them), D2-pair removes
    the store warp's four reduce-add ISSUES (rows dq_full / dq_empty unchanged: waits, commits, wait_groups and arrives stay).  No
    init count, arrive site, kind, scope or alias seam moves under any probe; each is output-wrong by design (never validated).
    MMA ISSUE ORDER (leader warp, per q tile t; code-writer 2026-10-26, from profiler-ncu 23:05 UTC: the leader waited 42 % at the dS
    gates [ds_smem + ds_recv_full] while the softmax waited 29 % for dP^T(t+1) -- one bubble seen from both ends.  FAv4's hdim-128
    2-CTA schedule read as the reference ONLY (sass-checker 23:20 UTC census: S(t+1), dK(t), dP(t+1), dQ(t), dV(t+1), dQ in its own
    TMEM columns); nothing copied.  The order is the typed adapter field ``pair_mma_order`` -> ``s_lookahead_after_dk`` (Constexpr).
      slot | s_first (THE default: PAIR_MMA_ORDER_DEFAULT) | dk_first (the A/B arm frost_thd_2cta_dk_first)  | what gates the issue
       1   | dP^T(t)  [dp_empty(t-1), do1_full x2]         | dP^T(t)   (same)                               | the epilogue's dQ(t-1) readout
       2   | dV(t)    [p_ready(t), do2_full]               | dV(t)     (same)                               | the softmax's P^T(t)
       3   | S^T(t+1) [q1_full(t+1)] if t+1 < hi           | dK(t)     [ds_ready(t), q2_full]               | the softmax's dS^T(t)
       4   | dK(t)    [ds_ready(t), q2_full]               | S^T(t+1)  [q1_full(t+1)] if t+1 < hi           | the Q1 ring (loaded long before)
       5   | dQ(t)    [ds_smem ds_recv_full ds_pair_ready] | dQ(t)     (the same three gates)               | the dS exchange of tile t
    Why: the exchange chain (ds_tmem -> dS^T ld -> XCHG STS -> 16 KiB ship -> own STS -> relay) starts at ds_ready(t); s_first covers it
    with dK(t) alone (512 clk: S^T(t+1) was already issued before ds_ready), dk_first with dK(t) + S^T(t+1) (1024 clk) while dV(t) still
    covers the softmax's dS^T(t) after dP^T(t).  The tensor pipe is in-order per issuing thread: S^T(t+1) after dV(t) (the P^T(t) read)
    is legal in both orders, dQ(t) after dK(t) keeps ds_ready(t)'s cover of dP's dead half.  dP^T(t+1) BEFORE dQ(t) (FAv4's full
    order) is impossible on THIS TMEM map: the dQ half-row image is dP's dead half [192, +64) and dp_empty is arrived after the
    epilogue's dQ readout; the full order needs the image moved (S^T's tail after P^T(t+1) is packed, a new S-tail release before
    S^T(t+2), dp_empty arrived after the dS^T readout) = TMEM map + barrier table change, the next arm (RESOURCE-NOTE owed).
    Numerics identical by construction (the same accumulation chain per GEMM; only S^T(t+1)'s issue position moves): the test module
    pins dK / dV bitwise between the orders.  Barriers: every row, init, arrive / wait site and alias seam is the same under both
    orders (one `vs-barrier-grep` census for the file); only the ORDER of the leader's waits inside tile t changes.
    MEASURED (benchmarker ORDER A/B/A, 2026-10-27 00:05 UTC, gb200-perf-152, ONE process per cell, main-kernel CUPTI us, band 1 %):
    dk_first is a main-kernel REGRESSION against s_first -- -2.78 % at 16K (17405 vs 16921 us) and -1.55 % at 32K (69157 vs 68086 us),
    indistinguishable at the step (-0.04 % / -0.15 %).  The reasoning above (1024 clk of cover) did not survive the measurement:
    the default is s_first again (CFG.PAIR_MMA_ORDER_DEFAULT; the 2026-10-26 flip to dk_first at 2c951980 rode a validate ladder
    only).  dk_first stays selectable as the A/B arm ``frost_thd_2cta_dk_first`` (the per-role phase table below describes it).
    PER-ROLE PHASE TABLE under dk_first (every PipelineState still advances exactly once per producer fire on every path -- no phase
    arithmetic changes; what moves is the order of the leader's waits inside tile t):
      role     | state (start)               | advance          | tile t waits in issue order                          | bootstrap / tail
      MMA(L)   | q1 st(0) adv Q1_STAGES      | +1 per S^T issue | dp_empty(dpe) do1_full[0] do1_full[1]                | S^T(lo) in the item prologue;
               | q2 / do1 / do2 st(0)        | +1 / tile        | p_ready(ready) do2_full | ds_ready(ready) q2_full    | last tile: lookahead skipped,
               | ready st(0) (the 5 gates)   | +1 / tile        | q1_full[q1_next] iff t+1 < hi                        | dQ(t_last) still issued -> dq_done;
               | dpe st(1) / out_empty st(1) | +1 / tile, +1/it | ds_smem ds_recv_full ds_pair_ready (ready) -> dQ(t)  | drains after the loop as before
      softmax  | tile st(0)                  | +1 / tile        | s_done(t) -> P^T | dp_done(t) -> dS^T                | s_done(t+1) lands LATER than under
               |                             |                  |                                                      | s_first (after dK(t)); dp_done same
      epilogue | ds / dq / empty st(0)       | +1 / tile (+1it) | ds_tmem, dq_empty x4, dq_done -- unchanged            | unchanged
      MMA(F)   | relay st(0)                 | +1 / tile        | ds_smem, ds_recv_full -> ds_pair_ready -- unchanged   | unchanged
      loader   | q1 st(1) adv Q1_STAGES, ... | unchanged        | q1_empty[s(t+1)] fires later (S^T(t+1) after dK(t)) | Q1_STAGES 2 keeps Q1(t+2) ahead
      store    | dq / out st(0)              | unchanged        | unchanged                                            | unchanged
      P14 paths: zero-trip item (no tile: no S^T, the empty dq_done commit as before); half-dead pair (CTA 1 runs every tile with
      P = 0: the same order, every barrier flips once per tile); n_q == 1 (lo + 1 == hi: the lookahead is skipped on the only tile,
      dQ(lo) issued after dK(lo); the prologue's S^T(lo) is the item's only S^T).

6. INIT + PHASE DISCIPLINE + EXIT DRAINS  (P4 / P5 / P6 / P14 / P15; code: the kernel entry + the helpers above the role bodies)
    init: warp 0, `if elect_sync():` every stage of every Bars field (ONE lane, both gates) -> fence_mbarrier_init -> barrier_cta_sync
          -> cga_arrive / cga_wait (before ANY cross-CTA arrive) -> MMA warp: tmem_alloc(TMEM_COLS, cta_group::2) on BOTH CTAs
          -> barrier_cta_sync (publishes the TMEM base) -> role dispatch.  The only bootstrap arrives are the store warp's dq_empty x 4
          per item (THREAD, counted in init: P5 (a)); every MMA-fed or pair-fed consumer starts pre-armed where the table says st(1).
    phase trackers (ONE PipelineState per ring, carried across CLC items, advanced on EVERY path; never bit math -- P2 / P14):
      softmax L+F:  tile st(0) [s_done, dp_done; p_ready / ds_ready / ds_tmem arrives] +1 / tile; sched st(0) adv SCHED_STAGES
      epilogue L+F: ds st(0) [ds_tmem] +1 / tile; dq st(0) [dq_done] +1 / tile +1 / zero-trip item; empty st(0) [dq_empty x4] +1 / tile
                    +1 / item (the item-end observation of all four slots before the native stores -- Rule S9); sched
      MMA(L):       q1 st(0) adv Q1_STAGES [q1_full]; q2 / do (do1 halves share ONE state) / do2 st(0) +1 / tile; kv st(0) [k1_full, v_full,
                    k2_full] +1 / live block; ready st(0) [p_ready, ds_ready, ds_smem, ds_recv_full, ds_pair_ready] +1 / tile;
                    dp_empty st(1) +1 / tile; out_empty st(1) [dvk_empty] +1 / item; sched
      MMA(F):       relay st(0) [ds_smem, ds_recv_full] +1 / tile; sched.  No MMA, no expect_tx, no wait on a leader copy.
      loader L+F:   q1 st(1) adv Q1_STAGES [q1_empty]; q2 / do1 / do2 st(1) +1 / tile; kv st(1) [k1_empty, v_empty] +1 / LIVE block
                    (lo < hi, as the baseline); k2e st(1) [k2_empty] +1 / item, waited right before the K2 reload (live) or once at
                    the item top (zero-trip) and answered by a bare dv_full arrive on BOTH paths (fix A); sched
      store L+F:    dq st(0) [dq_full x4] +1 / tile; out st(0) [dv_full, dk_full] +1 / item; sched
      scheduler(L): st(0) adv SCHED_STAGES: wait sched_empty[s] -> arm BOTH copies -> try_cancel -> wait sched_full[s] -> valid.
      scheduler(F): no loop (falls through to the final CTA sync): never armed, never credited, never waits.
    zero-trip item (lo >= hi, PAIR-UNIFORM from _item_bounds; a padded CTA inside a live pair runs every tile with P = 0): MMA(L) one
      empty dq_done commit (live pipe, multicast); both epilogues wait it; the store warps bootstrap dq_empty, run the native zero
      stores, arrive dvk_empty (-> L) + k2_empty; the loaders wait k2_empty; softmax / MMA(F) only credit the scheduler.
    P15 drains (OUTSIDE the persistent loop, const_expr(CFG.CTA_MMA == 2), whole ring from the carried state -- never min(total, stages)):
      loader L+F: q1_empty x Q1_STAGES, q2_empty, do1_empty, do2_empty, k1_empty, v_empty (the leader's last multicast commits in flight)
      MMA(L): dp_empty x1, dvk_empty x1 (the follower's last relaxed arrives in flight); tmem_dealloc: the wait is the drain.
      Observed in-loop on every phase, no drain: s_done / dp_done / dq_done / sched_full (every consumer waits every phase),
      p_ready / ds_ready / ds_pair_ready / ds_recv_full (the MMA warps wait every tile), sched_empty (the credits of tile n gate
      payload n + 1, the terminal payload carries no credit).  Remote arrives target only a CTA that still waits them: resident.
    SEAM NOTE -- dQ staging over ds_own + XCHG (profiler-ncu 20:27 UTC on the 1-CTA body: the epilogue's hottest wait is
      dq_empty[0] (SASS 0x4060 -> mbar 0x38500), the store warp waits the reduce-add SOURCE READ 77 % of its time, the MMA's late wait
      is ds_smem (0x7810 -> 0x384f0)): the alias CANNOT move inside 231424 B.  Every per-tile slab dead before dq_done(t) (sQ1[s(t)],
      sQ2, sDO1, sDO2) is a TMA-load target the MMA needs 0 .. 1536 clk after dq_done(t) at the 2560-clk ideal period, so a staging
      alias there would gate the MMA's OPERAND path by the drain (sQ1[s(t)]: the Q(t+2) prefetch lands ~1000 clk late; sQ2 / sDO2: dK /
      dV of t+1 wait the whole drain).  ds_own + XCHG are the bytes with the LATEST next writer (the dS(t+1) STS, ~1200 clk after
      dq_done(t)).  Per-CTA drain = 32 KiB per tile, half of today's 64 KiB: ~1620 clk at the l1tex->xbar path peak implied by the
      measured 77.3 % duty (17.99 GB / 4205.8 us / 152 SMs / 1805 MHz = 15.6 B/clk/SM -> 20.2 B/clk/SM), ~2100 clk at the measured
      duty.  REQUIRED streaming order (the writer's punch list): the store warp issues slices 2, 3 (the XCHG bytes) FIRST, then 0, 1
      (the own-half bytes), and arrives dq_empty[2], [3], [0], [1] after tma_store_wait(3), (2), (1), (0); the epilogue writes XCHG(t+1)
      after dq_empty[2], [3] and ships it at once (the longer exchange path drains first), then the own half after dq_empty[0], [1].
      Expected dQ(t+1) issue slip at the ideal period: ~0 clk at the path peak, ~500 clk at the measured duty (today: ~1600).  The
      waits each role eats: epilogue dq_empty[2|3] then [0|1]; MMA(L) ds_pair_ready (exchange) and ds_smem (own); store tma_store_wait.
      Optimizer lever, not a table change: issuing dV(t+1) before dP^T(t+1) adds 512 clk of slack on dp_empty (the readout path).
      MEASURED INPUT (benchmarker lane B, 2026-10-26, runs/aba/frost_s{8192,32768}_uniform_d2): deleting the dQ fp32 reduce-add on the
      1-CTA body makes its main kernel 23 % faster at 8K (4626 -> 3749 us) and 20 % at 32K (74403 -> 61828 us), the step +23 % at both
      cells -- the whole FAv4 gap fits inside that chain, so the pair's dQ path is THE lever.  Body compliance: the store warp is the
      only role that waits a reduce-add (its own `tma_store_wait(3 - i)` = the SOURCE READ, `cp.async.bulk.wait_group.read`) right
      before it re-arms dq_empty; every other role waits credits only (epilogue: dq_empty XCHG half first, then own; MMA(L): ds_smem /
      ds_recv_full / ds_pair_ready); the reduce-add is issued by the store warp, never between the MMA warp and its next S^T / dP^T
      issue (S^T(t+1) is gated by q1_full alone, dP^T(t+1) by dp_empty, arrived right after the dQ readout).  First diagnostic to wire
      once green (NOT in this body): a const_expr twin of the 1-CTA `diag_delete_dq_reduce_add` around the four reduce-add issues of
      `_store_loop`, threaded from the adapter record, so the benchmarker can measure how much of the drain the pair already removed.

7. THE ONE HELPER
    _item_bounds(pair, meta, batch, n_batch, causal) -> (q_lo, q_hi, seq_q, seq_k, q_start): CLUSTER-UNION bounds
    (the pair walks the identical q range; causal trim per 256-row PAIR: lo = pair * Q_TILES_PER_PAIR; CTA 1's diagonal tile is
    fully MASKED, never skipped; a dead pair (256 pair >= seq_k) is lo = hi = 0 on both CTAs).  Every role on both CTAs derives
    its loop bounds from this pure helper over the SAME inputs.

8. SMEM TABLE  (REQUIRED; per CTA, DECLARATION order; vs-smem predict PASS: total 231424 B of cap 232448 B (cc 10.0,
   227 KiB opt-in, SKU card 2026-10-15), slack 1024 B; DESC_VERSION implied 0, baseline's 1 inherited)
    name[stages]   | bytes  | 1024-al | desc | swizzle + WHY                                  | lifetime / alias   | offset
    sQ1[2]         | 16384  | 32768   | yes  | SW128 K-major: MATCH the S^T B descriptor       | tile; q1 ring      | 0
    sQ2[1]         | 16384  | 16384   | yes  | SW128 MN-major (64-d rows = one atom): MATCH dK B | tile; freed by dK | 32768
    sDO1[1]        | 16384  | 16384   | yes  | SW128 K-major: MATCH the dP^T B descriptor      | tile; freed by dP^T| 49152
    sDO2[1]        | 16384  | 16384   | yes  | SW128 MN-major: MATCH the dV B descriptor       | tile; freed by dV  | 65536
    sK1[1]         | 32768  | 32768   | yes  | SW128 K-major: MATCH the S^T A descriptor       | kv block           | 81920
    sK2[1]         | 32768  | 32768   | yes  | SW128 MN-major, 256 rows (two 128-row boxes):   | kv block; dV store | 114688
                   |        |         |      | MATCH the dQ B desc                             | staging at the end |
    sV[1]          | 32768  | 32768   | yes  | SW128 K-major: MATCH the dP^T A descriptor      | kv block           | 147456
    sDS[1]         | 32768  | 32768   | yes  | SW128 MN-major (q contiguous, 128 B per kv row, | own half: STS ..   | 180224
                   |        |         |      | kv-global row order): MATCH the dQ A desc        | dQ MMA; recv half: |
                   |        |         |      | (atranspose); today's 16-B `_store_words`, no   | the PEER's timeline|
                   |        |         |      | bank hit (vs-smem banks PASS form)              |                    |
    sDS_xchg[1]    | 16384  | 16384   | no   | SW128 image of the peer's recv half (byte copy) | STS .. bulk copy   | 212992
    sSTATS (LSE)   | 512    | 512     | no   | none: 4-B lane stride                            | tile               | 229376
    sDELTA         | 512    | 512     | no   | none                                             | tile               | 229888
    scaffold       | 1024   |         |      | bars [230400, 231344) + TMEM handle 16 + CLC payload 64           | 230400
    aliases: dQ fp32 staging 4 x 8 KiB [64 x 32] SW128 (lane stride 128 B: SPREAD + MATCH the reduce-add box) over
    ds_own(rank) + XCHG (rank-dependent: the kv-global row order puts CTA r's own 128 kv rows at DS + r * 16 KiB);
    dK native store (2 x 16 KiB) over the same bytes at the block end; dV native store (2 x 16 KiB) over sK2.
    total 231424 B = today's.  The user's claim ("Q + dQ staging at today's bytes pays K2") does not close as stated:
    today's dQ staging is a tenant of a 64 KiB ALIAS = max(dS 32, dQ 64, dV/dK 64), so halving it alone saves nothing.
    What closes: Q 48 KiB (Q2 at ONE stage, -16), dO 32 (=), K 64 (+32), V 32 (=), ALIAS 48 (dS 32 + XCHG 16 with the
    dQ staging 32 aliased and the dV / dK stores moved to K2 / the dS bytes, -16), stats 1 = 225 KiB + 1 KiB scaffold.
    WRITER DEVIATION from the layout skeleton (same bytes, same aliases, same barriers): the skeleton spelled sDS as a K-major
    A operand (4 K-chunks of [64 q x 64 kv], written by a transposed 2-B-per-lane STS = 128 scalar stores per lane per tile);
    the body keeps today's MN-major dS^T form ([kv row][64 q], 16 vector stores per lane per tile) and flips the dQ A
    descriptor to atranspose -- the M = 128 cta_group::2 MMA takes a 64-q (one SW128 atom) MN-major A slab per CTA.
    proxy-fence sites: after every lane-written slab the async proxy reads (sDS own half, XCHG before the bulk copy,
    dQ slices before the reduce-add, dV / dK staging before the TMA store) -- the inspector's rows.

9. TMEM MAP (fp32 columns; one tcgen05.alloc.cta_group::2 of TMEM_COLS = 512 on BOTH CTAs, the same constant deallocates)
    S^T    @   0 + 128  | P^T packed bf16 in place @ 0 + 64 (softmax group g packs into 32 g .. +32); 64..128 dead
    dP^T   @ 128 + 128  | dS^T packed @ 128 + 64; dQ (M 128 half-row image) @ 192 + 64 = dP's dead fp32 half
    dV     @ 256 + 128  | dK @ 384 + 128   -> used 512 / 512.  No TMEM slot is assumed zero.
    Softmax ownership: group g (128 lanes = kv rows) owns fp32 q quarters [32 g, +32) and [64 + 32 g, +32), so both
    groups pack into their OWN columns (no cross-group barrier) and the packed q order is the fixed permutation
    CFG.ts_k_perm = [0, 1, 4, 5, 2, 3, 6, 7] (16-q blocks) that the dV / dK B descriptors walk via k_idx.
    Packed column c of a plane: group c // 32, quarter (c // 32) + 2 * ((c % 32) // 16): the epilogue's own q half
    (q [64 r, +64)) is the packed columns {16 r + [0, 16), 32 + 16 r + [0, 16)}; the XCHG half the complement.

10. TENSOR-MAP / STRIDE CONTRACT (CFG.MAPS_PER_SEQUENCE = 9 per-sequence runtime maps, built by setup_thd_d128_2cta_kernel)
    Q1 / dO1: 64-row maps, two [64 rows x 64 d] SW128 boxes per CTA stage (rows = the CTA's q half); Q2 / dO2: 128-row
    maps, one [128 q x 64 d] box at d = 64 * rank; K1 / V: 128-row maps, two [128 kv x 64 d] boxes (own rows); K2: two
    [128 kv x 64 d] boxes at d = 64 * rank over the PAIR's rows (the 128-row K map, contiguous in sK2); dQ reduce-add:
    a 64-row fp32 map, [64 q x 32 d] boxes (4 per tile per CTA; one reduce per PAIR tile = half of today's L2 reduce
    traffic); dV / dK: two [128 kv x 64 d] native boxes per CTA (128-row maps).  expect_tx = FULL box bytes ON THE
    LEADER for both CTAs' halves (2 x); TMA OOB zero-fills padded rows; sequence-local maps clip the tails.
    LOAD FORM (typed adapter field ``tma_load_form`` -> ``tma_load_plain`` Constexpr; CFG.TMA_LOAD_FORM_DEFAULT = multicast):
    multicast = every cga2 load carries mcast_mask = 1 << rank (`.multicast::cluster.cta_group::2`, the validated form); plain =
    mcast_mask None (`.cta_group::2` alone, FAv4's form at cluster (2,1); sass-checker diff item 8).  The DSL wrapper routes the
    complete_tx to the pair leader's mbarrier (bit 24) in BOTH forms, so expect_tx and every barrier row are identical; same bytes,
    same SMEM image (bitwise on dK / dV by construction).  The plain form is the A/B arm ``frost_thd_2cta_plain_tma``: UNMEASURED.

11. OPTIONAL ARMS (const_expr; each states "the dense rendering is unchanged")
    THD (this file's arm: the unit is 256 kv rows per pair) | GQA grouped loaders (head // ratio) | causal |
    pair_mma_order (``s_lookahead_after_dk``: dk_first | s_first, section 5 MMA ISSUE ORDER; the pair body has no dense rendering --
    each order's rendering is its own PTX (sm_100a host render: s_first = the default c28e6013, dk_first f50cbcfe, 2026-10-27;
    the census of both is identical, pinned by the test module) | tma_load_form (``tma_load_plain``: multicast | plain, section 10
    LOAD FORM; the default rendering is unchanged (c28e6013); plain renders 12 `cp.async.bulk.tensor.4d...cta_group::2` loads
    without `.multicast::cluster` and 0 with it, pinned by the test module's plain-form census) |
    diag_2cta (DIAGNOSTIC; ``diag_2cta_code``: None = the default rendering, unchanged (c28e6013) | dq_reduce_add_off (D2-pair:
    the store warp's four reduce-add ISSUES skipped, PTX census reduce_add 4 -> 0, dQ stays zero) | dp_empty_wait_off (D7: the
    leader's per-tile dp_empty WAIT skipped, PTX wait census 63 -> 62; the P15 drain stays) | ds_gates_off (D8: the leader's three
    dS-gate WAITs before dQ(t) skipped, PTX wait census 63 -> 60); every arrive site, init count and seam of section 5 is identical under every
    value (mbarrier.arrive 41 pinned); OUTPUT WRONG BY DESIGN -- a timing CEILING for arm (ii) (D7), the dS exchange family L1 /
    DS (D8) and L3 (D2-pair); the driver's `--validate` refuses the arms by name; never shipped armed.  Hazard note for D8: with
    the leader no longer waiting ds_recv_full(t), the epilogue's expect_tx arm of tile t+1 (after dp_empty(t), ~1500+ clk later)
    could in principle land before the peer's 16 KiB copy(t) completed on the same barrier (an over-arrive window on a barrier
    nobody waits on the leader; on the follower the relay waits its copy in order) -- a probe-only artefact to read, not a design) |
    store_wait_form (KNOB; ``store_wait_spin``: retry | two_instr, section 4; the default rendering is unchanged (c28e6013); two_instr
    renders the store warp's six waits as the inline-PTX hint-less parity-wait spin (`.acquire.cta`; PTX wait census 63 unchanged,
    every other row identical, pinned by the test module); the arm ``frost_thd_2cta_two_instr_wait``) |
    reg_split (KNOB; ``reg_split_code``: 144/144/80 | 136/136/104, section 4; the default rendering is unchanged (c28e6013);
    136/136/104 renders `setmaxnreg.inc 136` x 2 + `setmaxnreg.dec 104` x 5 with every census row identical, pinned by the test
    module; KILLED on any STL / LDL in the SASS census; the arm ``frost_thd_2cta_regs_136``).

12. DEGENERATE-INPUT MATRIX  (row -> the handling LINE; absence is the bug)
    S_kv = 0 (dead pair: lo = hi = 0, stores skipped: _item_bounds / _store_loop `live_rows`) | kv rows >= seq_k in the
    pair's SECOND half (whole CTA dead: masked columns P = 0 exactly (_p_values `hi = 0`), dK / dV stores skipped by
    `live_rows`, dS recv half still exchanged = zeros) | q tile fully masked for CTA 1 at the causal diagonal (mask words,
    never a skip) | n_q tiles = 1 | S % 256 != 0 (the 256-row THD unit; the sequence-local maps clip) | one pair per
    launch | padded Q rows (stats 0, mask word `hi = seq_q`, no store: the 64-row dQ map clips) | zero-trip owner (CLC
    envelope) on BOTH CTAs (the zero-trip protocol of section 6).

13. MEASURED LEVERS  (A/B/A x 3 x 100, control twin, band max(1 %, |pair|))
    lever | node class / SMs / lock | result | verdict | stands?
    pair MMA order dk_first vs s_first (S^T(t+1) after dK(t)) | gb200-perf-152 / 152 / 1830 (cap-degraded, sampled 1680 / 1657 MHz;
    in-process ratios are the figures) | MAIN kernel dk_first vs s_first: -2.78 % @ 16K, -1.55 % @ 32K (step -0.04 % / -0.15 %, control
    pairs -0.29 % / -0.05 %, band 1 %); benchmarker 2026-10-27 00:05 UTC, runs/aba/frost_s{16384,32768}_uniform_order | REGRESSION for
    dk_first; default = s_first (restored 2026-10-27), dk_first = the A/B arm | stands
    tma_load_form plain vs multicast (section 10) | gb200-perf-152 | PENDING (the knob process: HEAD + plain + two_instr + 136/136/104 + canary, one
    process per bar cell, PLAN § 6.3 step 6) | -- | --
    store_wait_form two_instr vs retry (section 4, lever L3's wait-form half) | gb200-perf-152 | PENDING (the same knob process) | -- | --
    reg_split 136/136/104 vs 144/144/80 (section 4, M5') | gb200-perf-152 | PENDING (the same knob process; SASS census STL / LDL == 0 first) | -- | --

14. LAUNCH ABI
    positional args as the grouped THD host (maps, lse, delta, meta, origins, n_batch, stats_origin, scale, causal,
    dtype, ratio) + cluster (CFG.CGA_M, CFG.CGA_N, 1); grid x = ceil(max_skv / CFG.THD_UNIT_ROWS) * CFG.CGA_M, the
    declared envelope; d128_grouped_contract's Int32 envelope holds (ceil(skv / 256) * batch * heads <= the validated
    128-row count) with the rounding bound tightened to 2**31 - 256 (CFG.INT32_ROUNDING_BOUND).

PDL SITES: none in the accum family today (CLC-scheduled persistent envelope); unchanged.

Sources: config_sm100_d128_accum.py (the fork's baseline, fc472d3d); python/cudnn/sdpa/AGENTS.md § 2x2 datapath
lessons; bprop_d512_f16_2x2.py (pair-leader expect_tx, pair-symmetric dealloc, cluster CLC); FAv4 flash_bwd_sm100.py
NOTE [2CTA schedules] (read as the cross-check only: Q_stage 1 under 2-CTA, sdS_xchg = dS / 2, sdK in sK / sdV in sV).
"""

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm

from cudnn.frost.tile_dsl.barrier import MBarrier, PipelineState, Producer, Scope, advance, cga_arrive, cga_wait, named_barrier_fence
from cudnn.frost.tile_dsl.barrier import wait as mbar_wait
from cudnn.frost.tile_dsl.handles import MmaDesc, SmemTile, tma_slice_runtime_desc
from cudnn.frost.tile_dsl.mask import apply_mask_words, band_mask_words
from cudnn.frost.tile_dsl.mma import desc_opaque, mma_ss, mma_ts_step
from cudnn.frost.tile_dsl.pointwise import ffma2, fmul2, fp32_to_fp16
from cudnn.frost.tile_dsl.scheduler import Sched, read_clc_payload, read_tile_id_arrive_on_leader, scheduler_warp_loop
from cudnn.frost.tile_dsl.thd import TENSOR_MAP_QWORDS, thd_port_origin
from cudnn.frost.tile_dsl.tma import (
    cp_async_bulk_shared_cluster_shared_cta,
    tma_load_subtiles,
    tma_load_tile,
    tma_reduce_add_subtile,
    tma_store_commit,
    tma_store_tile,
    tma_store_wait,
    tma_tensormap_acquire,
)
from cudnn.frost.tile_dsl.tmem import tmem_alloc, tmem_dealloc

from ...config_sm100_d128_accum_2cta import ALIASES, BAR_LAYOUT, BAR_LAYOUT_STATUS, CFG, diag_2cta_code, reg_split_counts

LOG2E = 1.4426950408889634
# The 2-CTA per-sequence tensor-map set (header section 10); box rows per slot = CFG.MAP_BOX_ROWS, in this order.
MAP_Q1, MAP_K, MAP_V, MAP_DO1, MAP_DQ, MAP_DK, MAP_DV, MAP_Q2, MAP_DO2 = range(CFG.MAPS_PER_SEQUENCE)
MAP_COUNT = CFG.MAPS_PER_SEQUENCE

# Role warp ids (CFG.SOFTMAX_WARPS softmax, CFG.EPILOGUE_WARPS epilogue, then the four service warps).
SOFTMAX_WARP_END = CFG.SOFTMAX_WARPS
EPILOGUE_WARP_END = SOFTMAX_WARP_END + CFG.EPILOGUE_WARPS
EPILOGUE_LEAD_WARP = SOFTMAX_WARP_END  # the one warp whose elected lane arms ds_recv_full and ships XCHG
MMA_WARP = EPILOGUE_WARP_END
TMA_LOAD_WARP = MMA_WARP + 1
TMA_STORE_WARP = MMA_WARP + 2
SCHED_WARP = MMA_WARP + 3
# Test-only skew lever (barrier-inspector AUDIT 2026-10-26, lever 4 = the k2_empty back-edge detector): the LOADER of both CTAs sleeps
# ~DEBUG_LOADER_DELAY_US us at the top of EVERY item right after its scheduler credit, so a zero-trip item's k2_empty completes twice
# before the loader observes it (RED = HANG on the pre-fix body, GREEN with dv_full = DV_FULL_LANES).  Default 0 = no traced code (the
# default rendering carries no nanosleep: the test module's census pins it); the test sets the module global BEFORE the module executes
# (spec loader + sys.modules, the d512 2x2 DEBUG_STG_DELAY_US form).  Never an environment variable, never a knob.
DEBUG_LOADER_DELAY_US: int = int(globals().get("FROST_D128_2CTA_DEBUG_LOADER_DELAY_US", 0))
# Test-only skew levers 1-3 (the same AUDIT, the same house form, default 0 = no traced code; one role of ONE CTA, selected by rank at the
# site): FOLLOWER_OWN_HALF = the follower epilogue between the XCHG ship and its own-half STS (isolates ds_pair_ready: only the relay's
# release.cta arrive holds dQ(t)); LEADER_SHIP = the leader epilogue right before its XCHG ship (isolates the relay's ds_recv_full wait +
# the P9 routing of the ship); FOLLOWER_EPILOGUE_ITEM_END = the follower epilogue before its item-end dV / dK TMEM readout (isolates
# dvk_empty's pair-wide init 64).  Detectors: test_sdpa_bwd_d128_2cta_skew_sm100.py (fresh process, watchdog, fp64 gate).
DEBUG_FOLLOWER_OWN_HALF_DELAY_US: int = int(globals().get("FROST_D128_2CTA_DEBUG_FOLLOWER_OWN_HALF_DELAY_US", 0))
DEBUG_LEADER_SHIP_DELAY_US: int = int(globals().get("FROST_D128_2CTA_DEBUG_LEADER_SHIP_DELAY_US", 0))
DEBUG_FOLLOWER_EPILOGUE_ITEM_END_DELAY_US: int = int(globals().get("FROST_D128_2CTA_DEBUG_FOLLOWER_EPILOGUE_ITEM_END_DELAY_US", 0))
# DIAGNOSTIC probes (optimizer day-2 plan rank 1; header section 11): the typed adapter field ``diag_2cta`` (``CFG.DIAG_2CTA_NAMES`` or
# None) reaches the kernel as the ``diag_2cta_code`` Constexpr[int]; DIAG_NONE = the default (no traced code: the rendering is
# PTX-identical, pinned).  A probe removes ONE issue or ONE wait and keeps EVERY arrive site, so the barrier table and the lane ledger
# are unchanged; the output is WRONG BY DESIGN (a timing ceiling); `--validate` refuses the arm by name; never shipped armed.
#   DIAG_DQ_REDUCE_ADD_OFF (D2-pair): _store_loop skips the four tma_reduce_add issues (commit / wait / dq_empty arrives stay)
#   DIAG_DP_EMPTY_WAIT_OFF (D7):      _mma_loop_leader skips the per-tile dp_empty wait (the P15 drain after the loop stays)
#   DIAG_DS_GATES_OFF      (D8):      _mma_loop_leader skips the ds_smem / ds_recv_full / ds_pair_ready waits before dQ(t)
DIAG_NONE = diag_2cta_code(None)
DIAG_DQ_REDUCE_ADD_OFF = diag_2cta_code("dq_reduce_add_off")
DIAG_DP_EMPTY_WAIT_OFF = diag_2cta_code("dp_empty_wait_off")
DIAG_DS_GATES_OFF = diag_2cta_code("ds_gates_off")


class Bars(NamedTuple):
    """One MBarrier per ``BAR_LAYOUT`` row, in ``BAR_LAYOUT`` order (header section 5 = the table and the lane ledger).

    Built by :func:`_bars` on BOTH CTAs with identical counts; a LEADER-scope field's follower copy is never armed / waited."""

    q1_full: MBarrier
    q1_empty: MBarrier
    q2_full: MBarrier
    q2_empty: MBarrier
    do1_full: MBarrier
    do1_empty: MBarrier
    do2_full: MBarrier
    do2_empty: MBarrier
    k1_full: MBarrier
    k1_empty: MBarrier
    v_full: MBarrier
    v_empty: MBarrier
    k2_full: MBarrier
    k2_empty: MBarrier
    s_done: MBarrier
    dp_done: MBarrier
    dq_done: MBarrier
    p_ready: MBarrier
    ds_ready: MBarrier
    ds_tmem: MBarrier
    dp_empty: MBarrier
    ds_smem: MBarrier
    ds_recv_full: MBarrier
    ds_pair_ready: MBarrier
    dq_empty: MBarrier
    dq_full: MBarrier
    dv_full: MBarrier
    dk_full: MBarrier
    dvk_empty: MBarrier
    sched_full: MBarrier
    sched_empty: MBarrier
    tmem_dealloc: MBarrier


_BARS_STATUS = BAR_LAYOUT_STATUS
# BAR_LAYOUT kind -> (producer, scope).  "peer" rows are driven by arrive_on_peer / read_tile_id_arrive_on_leader (never .arrive()).
_KINDS = {
    "tma_leader": (Producer.TMA_LOAD, Scope.LEADER),
    "tma": (Producer.TMA_LOAD, Scope.LOCAL),
    "tma_dsmem": (Producer.TMA_LOAD, Scope.LOCAL),
    "mma": (Producer.MMA_COMMIT, Scope.LOCAL),
    "thread": (Producer.THREAD, Scope.LOCAL),
    "leader": (Producer.LEADER, Scope.LEADER),
    "peer": (Producer.THREAD, Scope.LEADER),
}
assert tuple(Bars._fields) == tuple(row[0] for row in BAR_LAYOUT), "Bars fields must follow BAR_LAYOUT (config = truth)"
MCAST_PAIR = CFG.MCAST_PAIR_MASK  # tcgen05.commit multicast to both CTAs of the pair; the leader is cluster rank 0 under cluster (2,1,1)
LEADER_RANK = 0


def _view(slab, offset, dtype, count, alignment=16):
    ptr = Pointer(slab.data_ptr() + offset, dtype=dtype)
    return cutlass.Array(ptr.ir_value(), shape=count, dtype=dtype, alignment=alignment)


def _bars(slab):
    """The inspector's table as typed barriers: one MBarrier per BAR_LAYOUT row, producer and scope from the row's kind."""
    return Bars(
        *(
            MBarrier(
                _view(slab, off, cutlass.Int64, slots),
                stages=slots,
                init_count=count,
                producer=_KINDS[kind][0],
                scope=_KINDS[kind][1],
                try_wait=True,
            )
            for _, off, slots, count, kind, _scope in BAR_LAYOUT
        )
    )


# ---------------------------------------------------------------------------------------------------------------------
# Barrier helpers (the guard forms of header section 5 as code; each role body below names the sites that call them).
# ---------------------------------------------------------------------------------------------------------------------


@cute.jit
def _wait(bar, phase):
    bar.wait(phase)
    # All full-warp consumers reconverge before any elected credit / issue (the baseline's form).
    nvvm.bar_warp_sync(cute.arch.FULL_MASK)


@cute.jit
def _wait_tcgen(bar, phase):
    _wait(bar, phase)
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)


@cute.jit
def _wait_store(bar, phase, spin: cutlass.Constexpr[bool]):
    """The STORE warp's mbarrier wait in the ``store_wait_form`` knob's form (header section 4 / 11; the same phase test either way):
    spin = ``tile_dsl.barrier.wait`` with ``spin=True``, the hint-less 2-instruction parity-wait loop in inline PTX (the form the MMA warp's
    waits render as); else the barrier's declared untimed parity-wait form = :func:`_wait`.  Both reconverge the warp before the issue."""
    if cutlass.const_expr(spin):
        mbar_wait(bar.smem_ptr, phase, spin=True)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
    else:
        _wait(bar, phase)


@cute.jit
def _commit_pair(bar, elect_p):
    """MMA_COMMIT from the LEADER's MMA warp: ONE lane (pred) commits, the commit lands ONE arrive on EACH CTA of the pair (P13)."""
    bar.arrive(mcast_mask=cutlass.Int32(MCAST_PAIR), cta_group=CFG.CTA_MMA, pred=elect_p)


@cute.jit
def _arrive_leader(bar):
    """LEADER (relaxed): EVERY calling lane lands one arrive on the LEADER's copy (init = lanes x CTA_MMA); for TMEM data already
    ordered by tcgen05_wait + tcgen05_fence(BEFORE_THREAD_SYNC) or a count-only credit (P8).  Call it bare from every lane."""
    bar.arrive(cta_group=CFG.CTA_MMA, leader_cta_id=cutlass.Int32(LEADER_RANK))


@cute.jit
def _credit(bar):
    """sched_empty: ONE .release.cta arrive per calling warp on the LEADER's copy (init PAIR_SCHED_CREDIT_WARPS); at the TOP of the item."""
    read_tile_id_arrive_on_leader(bar.smem_ptr, CFG.CGA_M)


@cute.jit
def _drain(bar, state, stages: cutlass.Constexpr[int]):
    """P15 / whole-ring drain from the CARRIED state: unused slots pass at their pre-armed parity, used slots wait their last release."""
    for _ in cutlass.range_constexpr(stages):
        _wait(bar[state.idx], state.phase)
        state = advance(state, stages)
    return state


@cute.jit
def _ship_xchg(slab, bars, rank, ship_pred):
    """Epilogue, after named barrier CFG.XCHG_BARRIER (all 4 warps' XCHG stores + fence_proxy done): ONE lane (pred=is_lead_warp & elect)
    ships OUR XCHG (dS^T[our 128 kv x the peer's 64 q]) into the PEER's recv half -- the peer's bytes at OUR ds_own offset (kv-global row
    order) -- completing XCHG_TX on the PEER's ds_recv_full (mapa BOTH the pointer and the mbar; the d512 role-split's _xfer form)."""
    peer = rank ^ cutlass.Int32(1)
    local_src = _view(slab, CFG.XCHG, cutlass.Uint8, CFG.XCHG_BYTES, alignment=1024)
    peer_dst = nvvm.mapa(_view(slab, CFG.DS, cutlass.Uint8, CFG.DS_BYTES, alignment=1024).subview(rank * cutlass.Int32(CFG.DS_HALF_BYTES)), peer, addrspace=7)
    peer_mbar = nvvm.mapa(bars.ds_recv_full.smem_ptr, peer, addrspace=7)
    cp_async_bulk_shared_cluster_shared_cta(peer_dst, local_src, peer_mbar, CFG.XCHG_TX, pred=ship_pred)


@cute.jit
def _tmem_dealloc_pair(bars, tmem_slot, rank):
    """MMA warp of EACH CTA after the final barrier_cta_sync (every local TMEM reader done): one elected arrive on our tmem_dealloc and one
    on the peer's (init TMEM_DEALLOC_ARRIVERS = 2), wait(0), tcgen05 fence, tcgen05.dealloc.cta_group::2 -- the dealloc frees the PAIR's
    columns, so neither CTA frees them while the other's warps may still read; the wait is the drain of the peer's arrive."""
    if nvvm.elect_sync():
        bars.tmem_dealloc.arrive()
        bars.tmem_dealloc.arrive_on_peer(rank ^ cutlass.Int32(1))
    bars.tmem_dealloc.wait(cutlass.Int32(0))
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
    tmem_dealloc(tmem_slot, CFG.TMEM_COLS, nvvm.CTAGroup.CTA_2)


# ---------------------------------------------------------------------------------------------------------------------
# Descriptor declarations.  Every SmemTile is SW128; rows are one atom (128 B) wide per TMA box.
# RESOURCE-NOTE: cta_group::2 loads complete on the PAIR LEADER's mbar: leader arms 2x bytes, follower never.
# ---------------------------------------------------------------------------------------------------------------------


def _operand_k_major(slab, offset, dtype, rows):
    """A / B operand with d along K (Q1, dO1, K1, V): ``rows`` x 128 d as two [rows x HEAD_BOX] SW128 boxes."""
    return SmemTile(
        base=_view(slab, offset, dtype, rows * CFG.D, alignment=1024),
        elems_per_stage=rows * CFG.D,
        leading_byte_offset=16,
        stride_byte_offset=8 * CFG.SWIZZLE_BYTES,
        layout=CFG.SWIZZLE_LAYOUT,
        tma_loads_per_tile=CFG.D // CFG.HEAD_BOX,
        tma_granu_elems=CFG.HEAD_BOX,
        tma_subtile_stride_elems=rows * CFG.HEAD_BOX,
        desc_version=CFG.DESC_VERSION,
    )


def _operand_mn_major(slab, offset, dtype, rows):
    """B operand with d along N (Q2, dO2: rows = 128 q; K2: rows = 256 kv): one [rows x D_HALF] SW128 box, MN-major."""
    return SmemTile(
        base=_view(slab, offset, dtype, rows * CFG.D_HALF, alignment=1024),
        elems_per_stage=rows * CFG.D_HALF,
        leading_byte_offset=rows * CFG.SWIZZLE_BYTES,
        stride_byte_offset=8 * CFG.SWIZZLE_BYTES,
        layout=CFG.SWIZZLE_LAYOUT,
        tma_loads_per_tile=1,
        tma_granu_elems=CFG.D_HALF,
        tma_subtile_stride_elems=rows * CFG.D_HALF,
        desc_version=CFG.DESC_VERSION,
    )


def _store_half(slab, offset, dtype):
    """One native-store box [M kv x HEAD_BOX d] (a dK staging half at a rank-dependent alias offset)."""
    return SmemTile(
        base=_view(slab, offset, dtype, CFG.M * CFG.HEAD_BOX, alignment=1024),
        elems_per_stage=CFG.M * CFG.HEAD_BOX,
        leading_byte_offset=16,
        stride_byte_offset=8 * CFG.SWIZZLE_BYTES,
        layout=CFG.SWIZZLE_LAYOUT,
        tma_loads_per_tile=1,
        tma_granu_elems=CFG.HEAD_BOX,
        tma_subtile_stride_elems=CFG.M * CFG.HEAD_BOX,
        desc_version=CFG.DESC_VERSION,
    )


def _ds_operand(slab, dtype):
    """dQ's A operand: dS^T [PAIR_KV_ROWS kv rows x Q_HALF q] MN-major (q contiguous: one 64-q SW128 atom per CTA, 128 B per kv
    row, 8-row swizzle period) -- the layout today's `_store_words` writes; the MN-major k-step walks 16 kv rows = 2 KiB."""
    return SmemTile(
        base=_view(slab, CFG.DS, dtype, CFG.Q_HALF * CFG.PAIR_KV_ROWS, alignment=1024),
        elems_per_stage=CFG.Q_HALF * CFG.PAIR_KV_ROWS,
        leading_byte_offset=CFG.DS_A_LBO,
        stride_byte_offset=8 * CFG.SWIZZLE_BYTES,
        layout=CFG.SWIZZLE_LAYOUT,
        tma_loads_per_tile=1,
        tma_granu_elems=CFG.Q_HALF,
        tma_subtile_stride_elems=CFG.Q_HALF * CFG.PAIR_KV_ROWS,
        desc_version=CFG.DESC_VERSION,
    )


def _dq_slice(slab, rank, index):
    """fp32 reduce-add staging slice ``index`` ([Q_HALF x DQ_BOX] SW128) at its rank-dependent alias offset."""
    return SmemTile(
        base=_view(slab, CFG.dq_slice(rank, index), cutlass.Float32, CFG.Q_HALF * CFG.DQ_BOX, alignment=1024),
        elems_per_stage=CFG.Q_HALF * CFG.DQ_BOX,
        leading_byte_offset=16,
        stride_byte_offset=8 * CFG.SWIZZLE_BYTES,
        layout=CFG.SWIZZLE_LAYOUT,
        tma_loads_per_tile=1,
        tma_granu_elems=CFG.DQ_BOX,
        tma_subtile_stride_elems=CFG.Q_HALF * CFG.DQ_BOX,
        desc_version=CFG.DESC_VERSION,
    )


def _mma_desc_kv(dtype, *, atranspose=False, btranspose=False):
    """S^T / dP^T (SS) and dV / dK (TS) : collective M = 256 (own 128 kv rows per CTA), N = 128, K = 128."""
    return MmaDesc(
        M=CFG.COLLECTIVE_M_KV,
        N=CFG.N,
        K=CFG.D,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.K_HW,
        atranspose=atranspose,
        btranspose=btranspose,
        cta_group=CFG.CTA_MMA,
        idesc=nvvm.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32,
            a_dtype=dtype,
            b_dtype=dtype,
            m_dim=CFG.COLLECTIVE_M_KV,
            n_dim=CFG.N,
            a_major=int(atranspose),
            b_major=int(btranspose),
        ),
        kind=nvvm.Tcgen05MMAKind.F16,
    )


def _mma_desc_dq(dtype):
    """dQ = dS K (SS): collective M = 128 (64 q rows per CTA), N = 128 d, K = the pair's 256 kv rows; A and B MN-major."""
    return MmaDesc(
        M=CFG.COLLECTIVE_M_Q,
        N=CFG.D,
        K=CFG.PAIR_KV_ROWS,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.K_HW,
        atranspose=True,
        btranspose=True,
        cta_group=CFG.CTA_MMA,
        idesc=nvvm.Tcgen05InstrDesc.build(c_dtype=cutlass.Float32, a_dtype=dtype, b_dtype=dtype, m_dim=CFG.COLLECTIVE_M_Q, n_dim=CFG.D, a_major=1, b_major=1),
        kind=nvvm.Tcgen05MMAKind.F16,
    )


# ---------------------------------------------------------------------------------------------------------------------
# Body helpers (forked from bprop_d128_accum_thd_f16.py onto the quarter-interleaved TMEM map and the pair geometry).
# ---------------------------------------------------------------------------------------------------------------------


@cute.jit
def _tmem_load(address, n: cutlass.Constexpr[int], dtype: cutlass.Constexpr = cutlass.Float32):
    return nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(address, dtype), num=n)


@cute.jit
def _tmem_store(address, values):
    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(address, cutlass.Int32), values)


@cute.jit
def _store_words(slab, byte_offset, row, words):
    """One SW128 atom row (32 Int32 = 128 B) of ``row`` at ``byte_offset``: today's dS^T / dV / dK staging form (16-B vectors, XOR swizzle)."""
    for vec in cutlass.range_constexpr(8):
        off = byte_offset + row * CFG.SWIZZLE_BYTES + ((row % 8) ^ vec) * 16
        values = cutlass.Vector.from_elements(tuple(words[vec * 4 + k] for k in range(4)), cutlass.Int32)
        Pointer(slab.data_ptr() + off, dtype=cutlass.Int32).store(values, alignment=16)


@cute.jit
def _store_dq_slice(slab, byte_base, row, values, first: cutlass.Constexpr[int]):
    """fp32 slice row: 32 values ``values[first .. first + 32)`` of q row ``row`` into the SW128 [Q_HALF x DQ_BOX] slice at ``byte_base``."""
    for vec in cutlass.range_constexpr(8):
        off = byte_base + row * CFG.SWIZZLE_BYTES + ((row % 8) ^ vec) * 16
        v = cutlass.Vector.from_elements(tuple(values[first + vec * 4 + k] for k in range(4)), cutlass.Float32)
        Pointer(slab.data_ptr() + off, dtype=cutlass.Float32).store(v, alignment=16)


@cute.jit
def _concat32(a, b):
    """Two 16-word tcgen05.ld results -> one 32-word row (q ascending across the two quarters)."""
    return cutlass.Vector.from_elements(tuple(a[i] for i in range(CFG.PACK_QUARTER_COLS)) + tuple(b[i] for i in range(CFG.PACK_QUARTER_COLS)), cutlass.Int32)


@cute.jit
def _map_ptr(maps, batch, n_batch, slot: cutlass.Constexpr[int]):
    return (maps.iterator.raw_ptr() + (cutlass.Int64(slot) * n_batch + batch) * TENSOR_MAP_QWORDS).tospace(cutlass.AddressSpace.generic)


@cute.jit
def _next_tile(sched, bars, state):
    """The pair's next CLC payload: x = the first ctaid.x of the cancelled cluster (a multiple of CGA_M) -> pair = x // CGA_M."""
    _wait(bars.sched_full[state.idx], state.phase)
    x, yz, valid = read_clc_payload(sched, state.idx * 8, warp_broadcast=True)
    return x // CFG.CGA_M, yz & 0xFFFF, (yz >> 16) & 0xFFFF, valid, advance(state, CFG.SCHED_STAGES)


@cute.jit
def _item_bounds(pair, meta, batch, n_batch, causal: cutlass.Constexpr[bool]):
    """Section 7: cluster-union (q_lo, q_hi, seq_q, seq_k, q_start) for the pair's 256-row kv block; pure, no SMEM.

    # BARRIER contract: both CTAs derive IDENTICAL (lo, hi) from the same payload words, so every barrier flips the same number of
    # times on both CTAs (P14); a dead owner (pair * 256 >= seq_k) is lo = hi = 0 on both; a padded SECOND half is NOT a skip.
    """
    data = cutlass.make_array_view(meta)
    q_start = cutlass.Int32(data[n_batch + batch])
    seq_q = cutlass.Int32(data[n_batch + batch + 1]) - q_start
    seq_k = cutlass.Int32(data[batch])
    lo = pair * cutlass.Int32(CFG.Q_TILES_PER_PAIR) if cutlass.const_expr(causal) else cutlass.Int32(0)
    hi = (seq_q + CFG.N - 1) // CFG.N
    if pair * CFG.PAIR_KV_ROWS >= seq_k:
        lo = cutlass.Int32(0)
        hi = cutlass.Int32(0)
    return lo, hi, seq_q, seq_k, q_start


@cute.jit
def _stats(slab, lse, delta, row, group, qbase, batch, head, scale, q_start, seq_q, origins, stats_origin: cutlass.Constexpr[int]):
    # Full 256-lane named barriers also cover inactive tail rows (both CTAs hold all 128 q stats). Invalid rows
    # never dereference a Stats/Delta address, including an all-empty carrier.
    named_barrier_fence(CFG.SOFTMAX_BARRIER, CFG.SOFTMAX_LANES)
    local_row = qbase + row
    if group == 0:
        stat = cutlass.Float32(0.0)
        if local_row < seq_q:
            token = thd_port_origin(origins, stats_origin, batch, q_start) + cutlass.Int64(local_row)
            if cutlass.const_expr(len(lse.shape) == 2):
                val = cutlass.Float32(lse[token, head])
            else:
                val = cutlass.Float32(lse[0, head, token])
            stat = -cutlass.Float32(LOG2E) * val
            if val == cutlass.Float32(float("-inf")):
                stat = cutlass.Float32(0.0)
        _view(slab, CFG.STATS, cutlass.Float32, CFG.N).subview(row).store(stat)
    else:
        stat = cutlass.Float32(0.0)
        if local_row < seq_q:
            stat = -scale * cutlass.Float32(delta[0, head, cutlass.Int64(q_start) + local_row])
        _view(slab, CFG.DELTA, cutlass.Float32, CFG.N).subview(row).store(stat)
    named_barrier_fence(CFG.SOFTMAX_BARRIER, CFG.SOFTMAX_LANES)


@cute.jit
def _p_values(slab, tmem_base, group, key, qtile, seq_q, seq_k, scale, causal: cutlass.Constexpr[bool], dtype: cutlass.Constexpr):
    """S^T -> P^T for the lane's kv row ``key`` over softmax group ``group``'s two q quarters; the packed bf16 P^T goes in place into
    the group's PACK_GROUP_COLS columns.  Returns the 64 fp32 P values in packed order (quarter g, then quarter g + 2)."""
    q_lo = group * CFG.GROUP_QUARTER
    a = _tmem_load(tmem_base + CFG.S_COL + q_lo, CFG.GROUP_QUARTER)
    b = _tmem_load(tmem_base + CFG.S_COL + q_lo + CFG.N // 2, CFG.GROUP_QUARTER)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    scaled = scale * cutlass.Float32(LOG2E)
    # KV-major transpose of the shared mask band: Q columns >= key for causal, below the actual Q length, and entirely
    # dead for a padded KV row (key >= seq_k). Select exact zero AFTER exp2 (also safe for a no-key row whose LSE is -inf).
    lo = key if cutlass.const_expr(causal) else None
    hi = cutlass.Int32(arith.select((key < seq_k).ir_value(), seq_q.ir_value(), cutlass.Int32(0).ir_value()))
    p_all = []
    packed = []
    for half in cutlass.range_constexpr(CFG.QUARTERS_PER_GROUP):
        scores = (a, b)[half]
        col0 = q_lo + half * (CFG.N // 2)
        values = []
        for chunk in cutlass.range_constexpr(CFG.GROUP_QUARTER // 4):
            stat = _view(slab, CFG.STATS, cutlass.Float32, CFG.N).subview(col0 + chunk * 4).load(vector_size=4, alignment=16)
            for pair in cutlass.range_constexpr(2):
                i = chunk * 4 + pair * 2
                e0, e1 = ffma2(scaled, scaled, scores[i], scores[i + 1], stat[pair * 2], stat[pair * 2 + 1])
                values.extend((cute.math.exp2(e0, fastmath=True), cute.math.exp2(e1, fastmath=True)))
        words = band_mask_words(lo, hi, qtile * CFG.N + col0, CFG.GROUP_QUARTER)
        p = apply_mask_words(cutlass.Vector.from_elements(tuple(values), cutlass.Float32), words, mask_value=0.0, n_cols=CFG.GROUP_QUARTER)
        p_all.extend(p[i] for i in range(CFG.GROUP_QUARTER))
        packed.extend(fp32_to_fp16(p[2 * i], p[2 * i + 1], dtype=dtype) for i in range(CFG.GROUP_QUARTER // 2))
    _tmem_store(tmem_base + CFG.P_COL + group * CFG.PACK_GROUP_COLS, cutlass.Vector.from_elements(tuple(packed), cutlass.Int32))
    return cutlass.Vector.from_elements(tuple(p_all), cutlass.Float32)


@cute.jit
def _ds_values(slab, tmem_base, group, p, scale, dtype: cutlass.Constexpr):
    """dP^T -> dS^T = P^T * (dP^T - delta) * scale for the group's two quarters; packed bf16 in place into the group's columns."""
    q_lo = group * CFG.GROUP_QUARTER
    a = _tmem_load(tmem_base + CFG.DP_COL + q_lo, CFG.GROUP_QUARTER)
    b = _tmem_load(tmem_base + CFG.DP_COL + q_lo + CFG.N // 2, CFG.GROUP_QUARTER)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    packed = []
    for half in cutlass.range_constexpr(CFG.QUARTERS_PER_GROUP):
        dp = (a, b)[half]
        col0 = q_lo + half * (CFG.N // 2)
        for chunk in cutlass.range_constexpr(CFG.GROUP_QUARTER // 4):
            stat = _view(slab, CFG.DELTA, cutlass.Float32, CFG.N).subview(col0 + chunk * 4).load(vector_size=4, alignment=16)
            for pair in cutlass.range_constexpr(2):
                i = chunk * 4 + pair * 2
                d0, d1 = ffma2(scale, scale, dp[i], dp[i + 1], stat[pair * 2], stat[pair * 2 + 1])
                d0, d1 = fmul2(d0, d1, p[half * CFG.GROUP_QUARTER + i], p[half * CFG.GROUP_QUARTER + i + 1])
                packed.append(fp32_to_fp16(d0, d1, dtype=dtype))
    _tmem_store(tmem_base + CFG.DS_COL + group * CFG.PACK_GROUP_COLS, cutlass.Vector.from_elements(tuple(packed), cutlass.Int32))


@cute.jit
def _native_out(slab, tmem_base, row, live, col, bases, dtype: cutlass.Constexpr):
    """dV / dK native staging: two [128 kv x 64 d] bf16 halves of the lane's kv row from TMEM columns ``col`` (zeros when the item ran
    no tile) into the two alias bases; one proxy fence + tcgen05 fence for both halves (the caller arrives)."""
    for half in cutlass.range_constexpr(CFG.D_HALVES):
        vals = cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(CFG.D_HALF)), cutlass.Float32)
        if live:
            vals = _tmem_load(tmem_base + col + half * CFG.D_HALF, CFG.D_HALF)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
        words = cutlass.Vector.from_elements(tuple(fp32_to_fp16(vals[2 * i], vals[2 * i + 1], dtype=dtype) for i in range(CFG.D_HALF // 2)), cutlass.Int32)
        _store_words(slab, bases[half], row, words)
    nvvm.fence_proxy("async.shared", space="cta")
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)


@cute.jit
def _ts(desc, plane_col, bdesc, dst, accum):
    """TS MMA over the q (K) dimension: A = the packed P^T / dS^T plane at TMEM column ``plane_col`` (8 columns per k-step, PACKED order);
    the B descriptor walks the q blocks in CFG.ts_k_perm order (the quarter-interleaved pack of header section 9)."""
    for k in cutlass.range_constexpr(CFG.N // CFG.K_HW):
        qblock = CFG.ts_k_perm(k)
        bstep = bdesc + ((desc.smem_advance_B_intra * qblock) >> 4)
        mma_ts_step(
            desc,
            nvvm.make_tmem_ptr(plane_col + k * desc.tmem_advance_A, cutlass.Int8),
            bstep,
            dst,
            k_idx=0,
            accumulate=accum if cutlass.const_expr(k == 0) else True,
        )


@cute.jit
def _s_lookahead(slab, bars, score, k1d, s_acc, q1_next, elect_p, qtile, hi, dtype: cutlass.Constexpr):
    """S^T(t + 1) into the S slot -- free once dV(t) was ISSUED (the in-order tensor pipe reads P^T(t) first): _wait_tcgen(q1_full[next])
    -> mma_ss -> _commit_pair(s_done), _commit_pair(q1_empty[next]).  Skipped on the item's last tile (nothing waits those flips).
    Its position in the tile (before or after dK(t)) is the leader's ``s_lookahead_after_dk`` Constexpr (header section 5)."""
    if qtile + 1 < hi:
        _wait_tcgen(bars.q1_full[q1_next.idx], q1_next.phase)
        q1d_next = _operand_k_major(slab, CFG.Q1 + q1_next.idx * CFG.Q1_BYTES, dtype, CFG.Q_HALF).desc()
        mma_ss(score, k1d, q1d_next, s_acc)
        _commit_pair(bars.s_done, elect_p)
        _commit_pair(bars.q1_empty[q1_next.idx], elect_p)


# ---------------------------------------------------------------------------------------------------------------------
# Role bodies (section 4).  The ``# BARRIER`` punch lists are the inspector's issue-order contract; the code follows them line by line.
# ---------------------------------------------------------------------------------------------------------------------


@cute.jit
def _softmax_loop(
    slab,
    bars,
    sched,
    tmem_base,
    lse,
    delta,
    origins,
    stats_origin: cutlass.Constexpr[int],
    pair,
    head,
    batch,
    meta,
    n_batch,
    rank,
    scale,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    regs: cutlass.Constexpr[int],
):
    """Warps 0-7, BOTH CTAs (lane = own kv row; group g owns q quarters {g, g+2}): S^T / dP^T -> P^T / dS^T packed in place.

    # BARRIER (per item, FIRST thing): _credit(bars.sched_empty[sched_state.idx])  -- before any work (every role: the top-of-item
    #   credit is what keeps the peer provably resident for the remote credits, P15)
    # BARRIER (per tile t, tile_state st(0)):
    #   _wait_tcgen(bars.s_done, tile_state.phase) -> tcgen05.ld S^T -> P^T pack (tcgen05.st) -> tcgen05_wait(STORE)
    #   -> tcgen05_fence(BEFORE_THREAD_SYNC) -> _arrive_leader(bars.p_ready)                 [LEADER, bare, 256 x 2 CTAs = 512]
    #   _wait_tcgen(bars.dp_done, tile_state.phase) -> dS^T pack -> tcgen05_wait(STORE) -> fence
    #   -> _arrive_leader(bars.ds_ready)  [LEADER, 512]  and  bars.ds_tmem.arrive()          [THREAD local, bare, 256]
    #   tile_state = advance(tile_state, 1)
    # Required waits: s_done, dp_done (own copies of the leader's multicast commits).  Zero-trip item: credit only (pair-uniform lo / hi).
    # Non-mbarrier: _stats -> named_barrier_fence(CFG.SOFTMAX_BARRIER, CFG.SOFTMAX_LANES) x 2 per call (both CTAs hold all 128 q stats).
    """
    nvvm.setmaxregister(regs, nvvm.SetMaxRegisterAction.INCREASE)  # CFG.REG_SPLITS[reg_split][0] (knob; default SOFTMAX_REGS)
    tid = cute.arch.thread_idx()[0]
    row = tid % CFG.M
    group = tid // CFG.M
    tile_state = PipelineState.start()
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        lo, hi, seq_q, seq_k, q_start = _item_bounds(pair, meta, batch, n_batch, causal)
        key = pair * CFG.PAIR_KV_ROWS + rank * CFG.M + row
        if lo < hi:
            _stats(slab, lse, delta, row, group, lo * CFG.N, batch, head, scale, q_start, seq_q, origins, stats_origin)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            _wait_tcgen(bars.s_done, tile_state.phase)
            p = _p_values(slab, tmem_base, group, key, qtile, seq_q, seq_k, scale, causal, dtype)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            _arrive_leader(bars.p_ready)
            _wait_tcgen(bars.dp_done, tile_state.phase)
            _ds_values(slab, tmem_base, group, p, scale, dtype)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            _arrive_leader(bars.ds_ready)
            bars.ds_tmem.arrive()
            if qtile + 1 < hi:
                _stats(slab, lse, delta, row, group, (qtile + 1) * CFG.N, batch, head, scale, q_start, seq_q, origins, stats_origin)
            tile_state = advance(tile_state, 1)
        pair, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state)


@cute.jit
def _epilogue_loop(
    slab,
    bars,
    sched,
    tmem_base,
    pair,
    head,
    batch,
    meta,
    n_batch,
    rank,
    warp,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    regs: cutlass.Constexpr[int],
):
    """Warps 8-11, BOTH CTAs (lane = own kv row for dS; lane = q row of d half (warp // 2) % 2 for dQ): dS^T -> sDS own half + XCHG
    (today's 16-B STS form), the DSMEM ship, the dQ fp32 slices, the native dV / dK stores.

    # BARRIER (per item, first): _credit(bars.sched_empty[sched_state.idx]);  ship_pred = is_lead_warp & nvvm.elect_sync()
    # BARRIER (per tile t): _wait_tcgen(bars.ds_tmem, ds_state.phase) -> tcgen05.ld dS^T (4 x 16 packed cols) -> tcgen05_wait(LOAD)
    #   1. recv arm:  bars.ds_recv_full.arrive(n_bytes=CFG.XCHG_TX, pred=ship_pred)   [1 == init; the PEER's copy delivers the bytes]
    #   2. XCHG half (the peer's 64 q): _wait(bars.dq_empty[2], e.phase); _wait(bars.dq_empty[3], e.phase)   [XCHG bytes drained FIRST]
    #      -> STS into XCHG -> nvvm.fence_proxy("async.shared", space="cta")
    #      -> named_barrier_fence(CFG.XCHG_BARRIER, CFG.EPILOGUE_LANES)   [P11 KEEP: the ship's operand is all 4 warps' stores]
    #      -> [test-only DEBUG_LEADER_SHIP_DELAY_US sleep, LEADER only, default none]
    #      -> _ship_xchg(slab, bars, rank, ship_pred)                      [one cp.async.bulk XCHG_TX -> the peer's recv half + ds_recv_full]
    #   3. own half:  _wait(bars.dq_empty[0], e.phase); _wait(bars.dq_empty[1], e.phase)
    #      -> [test-only DEBUG_FOLLOWER_OWN_HALF_DELAY_US sleep, FOLLOWER only, default none] -> STS into ds_own(rank) -> fence_proxy
    #      -> tcgen05_fence(BEFORE_THREAD_SYNC) -> bars.ds_smem.arrive()   [THREAD local, bare, 128: the local MMA warp waits it]
    #   4. _wait_tcgen(bars.dq_done, dq_state.phase) -> tcgen05.ld dQ (64 cols at CFG.DQ_COL, half-row image) -> tcgen05_wait(LOAD)
    #      -> tcgen05_fence(BEFORE_THREAD_SYNC) -> _arrive_leader(bars.dp_empty)   [LEADER, bare, 128 x 2 = 256]
    #   5. slices (same dq_empty phase e as steps 2-3: ONE completion per slot per tile):
    #      warps 10-11 (d half 1): slice 2 then 3: STS -> fence_proxy -> bars.dq_full[i].arrive()   [THREAD local, 2 warps = 64 == init]
    #      warps 8-9  (d half 0): slice 0 then 1: STS -> fence_proxy -> bars.dq_full[i].arrive()   [64]
    #   ds_state, dq_state, e = advance(.., 1)
    # BARRIER (zero-trip item, lo >= hi): _wait_tcgen(bars.dq_done, dq_state.phase); dq_state = advance(dq_state, 1)
    # BARRIER (item end): for i in range_constexpr(4): _wait(bars.dq_empty[i], e.phase)   [terminal observation of every slot, Rule S9]
    #   -> [test-only DEBUG_FOLLOWER_EPILOGUE_ITEM_END_DELAY_US sleep, FOLLOWER only, default none]
    #   -> native dV over K2 (CFG.dv_store(half); live=False writes zeros) -> fence_proxy + fence -> bars.dv_full.arrive()   [128 of
    #      DV_FULL_LANES 160: the loader adds 32 after its k2_empty observation of the item, AUDIT fix A]
    #   -> native dK over ds_own(rank) + XCHG (CFG.dk_store(rank, half)) -> bars.dk_full.arrive()   [128];  e = advance(e, 1)
    # No drain: every arrive of this role is local or waited by the leader in-loop (dp_empty's last phase is the leader MMA's drain).
    """
    # RESOURCE-NOTE: XCHG -> peer DS_RECV by one cp.async.bulk (16 KiB), complete_tx on the peer's ds_recv_full.
    nvvm.setmaxregister(regs, nvvm.SetMaxRegisterAction.INCREASE)  # CFG.REG_SPLITS[reg_split][1] (knob; default EPILOGUE_REGS)
    lane = cute.arch.thread_idx()[0] % CFG.M  # own kv row for dS^T; TMEM lane of the dQ half-row image
    qrow = lane % CFG.Q_HALF  # dQ row of the half-row image
    dhalf = lane // CFG.Q_HALF  # d half this warp pair holds (warp-uniform): slices 2 dhalf, 2 dhalf + 1
    is_lead_warp = warp == cutlass.Int32(EPILOGUE_LEAD_WARP)
    own_base = cutlass.Int32(CFG.DS) + rank * cutlass.Int32(CFG.DS_HALF_BYTES)  # CFG.ds_own(rank)
    peer_q = cutlass.Int32(1) - rank  # the q half the PEER needs = the XCHG half
    own_col = cutlass.Int32(CFG.DS_COL) + rank * cutlass.Int32(CFG.PACK_QUARTER_COLS)
    xchg_col = cutlass.Int32(CFG.DS_COL) + peer_q * cutlass.Int32(CFG.PACK_QUARTER_COLS)
    slice_base = cutlass.Int32(arith.select((dhalf == 0).ir_value(), own_base.ir_value(), cutlass.Int32(CFG.XCHG).ir_value()))
    ds_state = PipelineState.start()
    dq_state = PipelineState.start()
    empty_state = PipelineState.start()
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        ship_pred = is_lead_warp & nvvm.elect_sync()
        lo, hi, seq_q, seq_k, q_start = _item_bounds(pair, meta, batch, n_batch, causal)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            _wait_tcgen(bars.ds_tmem, ds_state.phase)
            x0 = _tmem_load(tmem_base + xchg_col, CFG.PACK_QUARTER_COLS, cutlass.Int32)
            x1 = _tmem_load(tmem_base + xchg_col + CFG.PACK_GROUP_COLS, CFG.PACK_QUARTER_COLS, cutlass.Int32)
            o0 = _tmem_load(tmem_base + own_col, CFG.PACK_QUARTER_COLS, cutlass.Int32)
            o1 = _tmem_load(tmem_base + own_col + CFG.PACK_GROUP_COLS, CFG.PACK_QUARTER_COLS, cutlass.Int32)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            bars.ds_recv_full.arrive(n_bytes=CFG.XCHG_TX, pred=ship_pred)
            _wait(bars.dq_empty[CFG.DQ_SLICE_ORDER[0]], empty_state.phase)
            _wait(bars.dq_empty[CFG.DQ_SLICE_ORDER[1]], empty_state.phase)
            _store_words(slab, CFG.XCHG, lane, _concat32(x0, x1))
            nvvm.fence_proxy("async.shared", space="cta")
            named_barrier_fence(CFG.XCHG_BARRIER, CFG.EPILOGUE_LANES)
            if cutlass.const_expr(DEBUG_LEADER_SHIP_DELAY_US > 0):
                if rank == cutlass.Int32(LEADER_RANK):  # skew lever 2: the LEADER holds its XCHG ship (the half is written)
                    for _us in cutlass.range(DEBUG_LEADER_SHIP_DELAY_US, unroll=1):
                        nvvm.inline_ptx("nanosleep.u32 1000;", read_only_args=[])
            _ship_xchg(slab, bars, rank, ship_pred)
            _wait(bars.dq_empty[CFG.DQ_SLICE_ORDER[2]], empty_state.phase)
            _wait(bars.dq_empty[CFG.DQ_SLICE_ORDER[3]], empty_state.phase)
            if cutlass.const_expr(DEBUG_FOLLOWER_OWN_HALF_DELAY_US > 0):
                if rank != cutlass.Int32(LEADER_RANK):  # skew lever 1: the FOLLOWER holds its own half (its XCHG is shipped)
                    for _us in cutlass.range(DEBUG_FOLLOWER_OWN_HALF_DELAY_US, unroll=1):
                        nvvm.inline_ptx("nanosleep.u32 1000;", read_only_args=[])
            _store_words(slab, own_base, lane, _concat32(o0, o1))
            nvvm.fence_proxy("async.shared", space="cta")
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            bars.ds_smem.arrive()
            _wait_tcgen(bars.dq_done, dq_state.phase)
            dq = _tmem_load(tmem_base + CFG.DQ_COL, CFG.DQ_COLS)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            _arrive_leader(bars.dp_empty)
            for j in cutlass.range_constexpr(CFG.SLICES_PER_D_HALF):
                _store_dq_slice(slab, slice_base + j * CFG.DQ_SLICE_BYTES, qrow, dq, j * CFG.DQ_BOX)
                nvvm.fence_proxy("async.shared", space="cta")
                bars.dq_full[dhalf * CFG.SLICES_PER_D_HALF + j].arrive()
            ds_state = advance(ds_state, 1)
            dq_state = advance(dq_state, 1)
            empty_state = advance(empty_state, 1)
        if lo >= hi:
            _wait_tcgen(bars.dq_done, dq_state.phase)
            dq_state = advance(dq_state, 1)
        # Observe every terminal source-credit phase before the next CLC owner bootstraps these barriers again (Rule S9).
        for i in cutlass.range_constexpr(CFG.DQ_SLICES):
            _wait(bars.dq_empty[i], empty_state.phase)
        live = lo < hi
        if cutlass.const_expr(DEBUG_FOLLOWER_EPILOGUE_ITEM_END_DELAY_US > 0):
            if rank != cutlass.Int32(LEADER_RANK):  # skew lever 3: the FOLLOWER holds its dV / dK TMEM readout at the item end
                for _us in cutlass.range(DEBUG_FOLLOWER_EPILOGUE_ITEM_END_DELAY_US, unroll=1):
                    nvvm.inline_ptx("nanosleep.u32 1000;", read_only_args=[])
        _native_out(slab, tmem_base, lane, live, CFG.DV_COL, (CFG.dv_store(0), CFG.dv_store(1)), dtype)
        bars.dv_full.arrive()
        _native_out(slab, tmem_base, lane, live, CFG.DK_COL, (own_base, CFG.XCHG), dtype)
        bars.dk_full.arrive()
        empty_state = advance(empty_state, 1)
        pair, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state)


@cute.jit
def _mma_loop_leader(
    slab,
    bars,
    sched,
    tmem_base,
    pair,
    head,
    batch,
    meta,
    n_batch,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    s_lookahead_after_dk: cutlass.Constexpr[bool],
    diag_2cta_code: cutlass.Constexpr[int],
    regs: cutlass.Constexpr[int],
):
    """Warp 12 on the pair LEADER: the five cta_group::2 GEMMs; every commit multicast to the pair.  elect_p = nvvm.elect_sync() once per item.
    ``s_lookahead_after_dk`` (header section 5 MMA ISSUE ORDER): True = dk_first (S^T(t+1) after dK(t)), False = s_first (before).
    ``diag_2cta_code`` (DIAGNOSTIC, default DIAG_NONE = this body): DIAG_DP_EMPTY_WAIT_OFF skips the per-tile dp_empty wait (D7: dP^T(t+1)
    may overwrite the dQ(t) image before the epilogue read it; the epilogue's arrives and the P15 drain stay, so no wait is ever starved:
    the drain's parity passes once the N arrives of the kernel landed); DIAG_DS_GATES_OFF skips the three dS gates before dQ(t) (D8: the dQ
    A operand may be stale; the epilogue's / relay's arrives stay and no drain reads these barriers; the peer's in-flight bulk copy into the
    recv half is a RACE against the MMA read, never a fault).  Output WRONG BY DESIGN under either; `--validate` refuses the arm.

    # BARRIER (per item, first): _credit(bars.sched_empty[sched_state.idx])
    #   if lo < hi: _wait_tcgen(bars.k1_full, kv.phase); _wait_tcgen(bars.q1_full[q1.idx], q1.phase); S^T(lo) = mma_ss(M 256)
    #               -> _commit_pair(bars.s_done, elect_p); _commit_pair(bars.q1_empty[q1.idx], elect_p)
    #               _wait_tcgen(bars.v_full, kv.phase); _wait_tcgen(bars.k2_full, kv.phase)
    #   _wait_tcgen(bars.dvk_empty, out_empty.phase); out_empty = advance(out_empty, 1)   [pre-armed st(1); EVERY item, zero-trip too]
    # BARRIER (per tile t):
    #   _wait_tcgen(bars.dp_empty, dpe.phase)  [st(1)] -> _wait_tcgen(bars.do1_full[0], do.phase) -> dP^T k-half 0 (k_count 4)
    #   -> _wait_tcgen(bars.do1_full[1], do.phase) -> dP^T k-half 1 -> _commit_pair(dp_done); _commit_pair(do1_empty)
    #   _wait_tcgen(bars.p_ready, ready.phase) -> _wait_tcgen(bars.do2_full, do2.phase) -> dV TS (A = P^T in BOTH CTAs' own TMEM,
    #   k_idx via CFG.ts_k_perm) -> _commit_pair(do2_empty)
    #   q1_next = advance(q1, CFG.Q1_STAGES);  [s_first: _s_lookahead HERE]
    #   _wait_tcgen(bars.ds_ready, ready.phase) -> _wait_tcgen(bars.q2_full, q2.phase) -> dK TS -> _commit_pair(q2_empty)
    #   [dk_first: _s_lookahead HERE] = if t + 1 < hi: _wait_tcgen(bars.q1_full[q1_next.idx], q1_next.phase) -> S^T(t + 1)
    #   -> _commit_pair(s_done); _commit_pair(bars.q1_empty[q1_next.idx])   (the same sites in both orders, one position)
    #   _wait_tcgen(bars.ds_smem, ready.phase); _wait_tcgen(bars.ds_recv_full, ready.phase); _wait_tcgen(bars.ds_pair_ready, ready.phase)
    #   -> dQ = mma_ss(M 128, K 256: A = sDS [256 kv x own 64 q] MN-major, B = K2) -> _commit_pair(dq_done)
    #   q1 = q1_next; do, do2, q2, ready, dpe = advance(.., 1)
    # BARRIER (block end): if lo < hi: _commit_pair(k1_empty); _commit_pair(v_empty); kv = advance(kv, 1)
    #                      else: _commit_pair(dq_done)   [the zero-trip empty commit on a live pipe: both epilogues wait it]
    # BARRIER (after the persistent loop, P15): _wait_tcgen(bars.dp_empty, dpe.phase); _wait_tcgen(bars.dvk_empty, out_empty.phase)
    #   then the final barrier_cta_sync and _tmem_dealloc_pair (kernel entry).
    """
    nvvm.setmaxregister(regs, nvvm.SetMaxRegisterAction.DECREASE)  # CFG.REG_SPLITS[reg_split][2] (knob; default SERVICE_REGS)
    score = _mma_desc_kv(dtype)
    grad = _mma_desc_kv(dtype, btranspose=True)
    dqdesc = _mma_desc_dq(dtype)
    s_acc = nvvm.make_tmem_ptr(tmem_base + CFG.S_COL, cutlass.Float32)
    dp_acc = nvvm.make_tmem_ptr(tmem_base + CFG.DP_COL, cutlass.Float32)
    dq_acc = nvvm.make_tmem_ptr(tmem_base + CFG.DQ_COL, cutlass.Float32)
    dv_acc = nvvm.make_tmem_ptr(tmem_base + CFG.DV_COL, cutlass.Float32)
    dk_acc = nvvm.make_tmem_ptr(tmem_base + CFG.DK_COL, cutlass.Float32)
    q1_state = PipelineState.start()
    q2_state = PipelineState.start()
    do1_state = PipelineState.start()
    do2_state = PipelineState.start()
    kv_state = PipelineState.start()
    ready_state = PipelineState.start()
    dp_empty = PipelineState.start(phase=1)
    out_empty = PipelineState.start(phase=1)
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        elect_p = nvvm.elect_sync()
        lo, hi, seq_q, seq_k, q_start = _item_bounds(pair, meta, batch, n_batch, causal)
        # The 1-stage slabs have STATIC addresses: pin their descriptor bases per item (desc_opaque + the item anchor), or LLVM hoists
        # every `base + k_step` add to the persistent loop's preheader and the 80-register MMA warp spills (the d256 2x2 lesson).
        k1d = desc_opaque(_operand_k_major(slab, CFG.K1, dtype, CFG.M).desc(), anchor=pair)
        vd = desc_opaque(_operand_k_major(slab, CFG.V, dtype, CFG.M).desc(), anchor=pair)
        k2d = desc_opaque(_operand_mn_major(slab, CFG.K2, dtype, CFG.PAIR_KV_ROWS).desc(), anchor=pair)
        do1d = desc_opaque(_operand_k_major(slab, CFG.DO1, dtype, CFG.Q_HALF).desc(), anchor=pair)
        do2d = desc_opaque(_operand_mn_major(slab, CFG.DO2, dtype, CFG.N).desc(), anchor=pair)
        q2d = desc_opaque(_operand_mn_major(slab, CFG.Q2, dtype, CFG.N).desc(), anchor=pair)
        dsd = desc_opaque(_ds_operand(slab, dtype).desc(), anchor=pair)
        if lo < hi:
            _wait_tcgen(bars.k1_full, kv_state.phase)
            _wait_tcgen(bars.q1_full[q1_state.idx], q1_state.phase)
            q1d = _operand_k_major(slab, CFG.Q1 + q1_state.idx * CFG.Q1_BYTES, dtype, CFG.Q_HALF).desc()
            mma_ss(score, k1d, q1d, s_acc)
            _commit_pair(bars.s_done, elect_p)
            _commit_pair(bars.q1_empty[q1_state.idx], elect_p)
            _wait_tcgen(bars.v_full, kv_state.phase)
            _wait_tcgen(bars.k2_full, kv_state.phase)
        _wait_tcgen(bars.dvk_empty, out_empty.phase)
        out_empty = advance(out_empty, 1)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            if cutlass.const_expr(diag_2cta_code != DIAG_DP_EMPTY_WAIT_OFF):  # DIAGNOSTIC D7: the WAIT only; the arrive sites stay
                _wait_tcgen(bars.dp_empty, dp_empty.phase)
            _wait_tcgen(bars.do1_full[0], do1_state.phase)
            mma_ss(score, vd, do1d, dp_acc, k_start=0, k_count=CFG.K_STEPS_PER_BOX)
            _wait_tcgen(bars.do1_full[1], do1_state.phase)
            mma_ss(score, vd, do1d, dp_acc, k_start=CFG.K_STEPS_PER_BOX, k_count=CFG.K_STEPS_PER_BOX, accumulate=True)
            _commit_pair(bars.dp_done, elect_p)
            _commit_pair(bars.do1_empty, elect_p)
            _wait_tcgen(bars.p_ready, ready_state.phase)
            _wait_tcgen(bars.do2_full, do2_state.phase)
            _ts(grad, tmem_base + CFG.P_COL, do2d, dv_acc, qtile > lo)
            _commit_pair(bars.do2_empty, elect_p)
            q1_next = advance(q1_state, CFG.Q1_STAGES)
            if cutlass.const_expr(not s_lookahead_after_dk):
                _s_lookahead(slab, bars, score, k1d, s_acc, q1_next, elect_p, qtile, hi, dtype)  # s_first: before the dS^T(t) wait
            _wait_tcgen(bars.ds_ready, ready_state.phase)
            _wait_tcgen(bars.q2_full, q2_state.phase)
            _ts(grad, tmem_base + CFG.DS_COL, q2d, dk_acc, qtile > lo)
            _commit_pair(bars.q2_empty, elect_p)
            if cutlass.const_expr(s_lookahead_after_dk):
                _s_lookahead(slab, bars, score, k1d, s_acc, q1_next, elect_p, qtile, hi, dtype)  # dk_first: covers the dS exchange
            if cutlass.const_expr(diag_2cta_code != DIAG_DS_GATES_OFF):  # DIAGNOSTIC D8: the three WAITs only; every arrive site stays
                _wait_tcgen(bars.ds_smem, ready_state.phase)
                _wait_tcgen(bars.ds_recv_full, ready_state.phase)
                _wait_tcgen(bars.ds_pair_ready, ready_state.phase)
            mma_ss(dqdesc, dsd, k2d, dq_acc)
            _commit_pair(bars.dq_done, elect_p)
            q1_state = q1_next
            do1_state = advance(do1_state, 1)
            do2_state = advance(do2_state, 1)
            q2_state = advance(q2_state, 1)
            ready_state = advance(ready_state, 1)
            dp_empty = advance(dp_empty, 1)
        if lo < hi:
            _commit_pair(bars.k1_empty, elect_p)
            _commit_pair(bars.v_empty, elect_p)
            kv_state = advance(kv_state, 1)
        else:
            _commit_pair(bars.dq_done, elect_p)
        pair, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state)
    # P15: the follower's last relaxed arrives (dp_empty of the last tile, dvk_empty of the last item) land on a resident leader.
    _wait_tcgen(bars.dp_empty, dp_empty.phase)
    _wait_tcgen(bars.dvk_empty, out_empty.phase)


@cute.jit
def _mma_loop_follower(bars, sched, pair, head, batch, meta, n_batch, causal: cutlass.Constexpr[bool], regs: cutlass.Constexpr[int]):
    """Warp 12 on the FOLLOWER: the quiet body + the dS relay.  Never issues an MMA, never arms expect_tx, never waits a leader copy
    (q*_full / do*_full / k*_full / v_full / p_ready / ds_ready / dp_empty / dvk_empty / ds_pair_ready are LEADER-only waits).

    # BARRIER (per item, first): _credit(bars.sched_empty[sched_state.idx])   [this warp is one of this CTA's 15 credit warps]
    # BARRIER (per tile t, relay st(0)): _wait(bars.ds_smem, relay.phase); _wait(bars.ds_recv_full, relay.phase)
    #   -> bars.ds_pair_ready.arrive_on_peer(cutlass.Int32(LEADER_RANK), pred=elect_p, release_cta=True)
    #      [1 lane -> the leader's copy == init 1; the cumulative .release.cta publishes our epilogue's fenced own-half stores to the
    #       leader-issued cta_group::2 dQ MMA (P8 / P12); relaxed would NOT]
    #   relay = advance(relay, 1)
    # Zero-trip item: credit only.  No drain: the leader waits every ds_pair_ready phase in-loop.  Teardown: _tmem_dealloc_pair.
    """
    nvvm.setmaxregister(regs, nvvm.SetMaxRegisterAction.DECREASE)  # CFG.REG_SPLITS[reg_split][2] (knob; default SERVICE_REGS)
    relay_state = PipelineState.start()
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        elect_p = nvvm.elect_sync()
        lo, hi, seq_q, seq_k, q_start = _item_bounds(pair, meta, batch, n_batch, causal)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            _wait(bars.ds_smem, relay_state.phase)
            _wait(bars.ds_recv_full, relay_state.phase)
            bars.ds_pair_ready.arrive_on_peer(cutlass.Int32(LEADER_RANK), pred=elect_p, release_cta=True)
            relay_state = advance(relay_state, 1)
        pair, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state)


@cute.jit
def _load_loop(
    slab,
    bars,
    sched,
    maps,
    pair,
    head,
    batch,
    meta,
    n_batch,
    rank,
    is_leader,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    ratio: cutlass.Constexpr[int],
    tma_load_plain: cutlass.Constexpr[bool],
    regs: cutlass.Constexpr[int],
):
    """Warp 13, BOTH CTAs: Q1 / Q2 / dO1 / dO2 per q tile, K1 / K2 / V per kv block (grouped head // ratio), each CTA its own boxes.

    # arm_pred = is_leader & nvvm.elect_sync()  (P9: a cga2 tensor load completes on the pair LEADER's mbar whoever issued it; the
    # leader arms BOTH CTAs' bytes = the *_TX constants; the follower NEVER arms).  tma_mask = cutlass.Int16(1) << rank (self-only)
    # under the multicast form, None under ``tma_load_plain`` (header section 10 LOAD FORM: the plain `.cta_group::2` load, same bytes,
    # same leader mbar); every load: tma_load_tile / tma_load_subtiles(..., bars.X_full[..].smem_ptr, cta_group=CFG.CTA_MMA, mcast_mask=tma_mask).
    # BARRIER (per item, first): _credit(bars.sched_empty[sched_state.idx])   [then the test-only DEBUG_LOADER_DELAY_US sleep, default none]
    #   if lo < hi: acquire the maps; _wait(bars.k1_empty, kv.phase) -> bars.k1_full.arrive(n_bytes=CFG.K1_TX, pred=arm_pred) -> K1
    #               (2 boxes, own 128 rows); _wait(bars.v_empty, kv.phase) -> v_full.arrive(CFG.V_TX, pred=arm_pred) -> V;
    #               _wait(bars.k2_empty, k2e.phase) -> bars.dv_full.arrive() [bare 32, fix A] -> k2_full.arrive(CFG.K2_TX, pred=arm_pred)
    #               -> K2 (two [128 kv x 64 d] boxes at d = 64 * rank, the PAIR's rows); kv = advance(kv, 1)
    #   else (zero-trip): _wait(bars.k2_empty, k2e.phase) -> bars.dv_full.arrive()   [the same single observation on every path, P14]
    #   k2e = advance(k2e, 1)   [k2_empty is fired by the store warp EVERY item, so it is waited every item; st(1); only the K2 reload
    #   needs the previous item's dV store drain (AUDIT perf note): K1 / V and the q-tile loads no longer wait for it]
    # BARRIER (per tile t):
    #   _wait(bars.q1_empty[s], q1.phase) -> q1_full[s].arrive(CFG.Q1_TX, pred=arm_pred) -> Q1 (rows 64 * rank .., 2 boxes)
    #   _wait(bars.do1_empty, do1.phase) -> for half in range_constexpr(2): do1_full[half].arrive(CFG.DO1_HALF_TX, pred=arm_pred)
    #   -> tma_load_subtiles(sDO1, .., bars.do1_full[half].smem_ptr, half, 1, cta_group=2, mcast_mask=tma_mask) x 2
    #   _wait(bars.do2_empty, do2.phase) -> do2_full.arrive(CFG.DO2_TX, pred=arm_pred) -> dO2 ([128 q x 64 d] at d = 64 * rank)
    #   _wait(bars.q2_empty, q2.phase) -> q2_full.arrive(CFG.Q2_TX, pred=arm_pred) -> Q2 ([128 q x 64 d] at d = 64 * rank)
    #   q1 = advance(q1, CFG.Q1_STAGES); do1, do2, q2 = advance(.., 1)
    # BARRIER (after the persistent loop, P15, if const_expr(CFG.CTA_MMA == 2)): q1 = _drain(bars.q1_empty, q1, CFG.Q1_STAGES);
    #   _drain(bars.q2_empty, q2, 1); _drain(bars.do1_empty, do1, 1); _drain(bars.do2_empty, do2, 1); _drain(bars.k1_empty, kv, 1);
    #   _drain(bars.v_empty, kv, 1)   [the leader's last multicast commits must land on a resident follower]
    """
    # RESOURCE-NOTE: both CTAs hold all 128 q stats (D image spans N=128): STATS/DELTA 512 B each, 1 slot.
    nvvm.setmaxregister(regs, nvvm.SetMaxRegisterAction.DECREASE)  # CFG.REG_SPLITS[reg_split][2] (knob; default SERVICE_REGS)
    q1_state = PipelineState.start(phase=1)
    q2_state = PipelineState.start(phase=1)
    do1_state = PipelineState.start(phase=1)
    do2_state = PipelineState.start(phase=1)
    kv_state = PipelineState.start(phase=1)
    k2e_state = PipelineState.start(phase=1)
    sched_state = PipelineState.start()
    if cutlass.const_expr(tma_load_plain):
        tma_mask = None  # the plain `.cta_group::2` load: no multicast modifier; the bytes still complete on the leader's mbar
    else:
        tma_mask = cutlass.Int16(1) << rank
    d_half = rank * cutlass.Int32(CFG.D_HALF)
    q_half = rank * cutlass.Int32(CFG.Q_HALF)
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        if cutlass.const_expr(DEBUG_LOADER_DELAY_US > 0):
            for _us in cutlass.range(DEBUG_LOADER_DELAY_US, unroll=1):
                nvvm.inline_ptx("nanosleep.u32 1000;", read_only_args=[])
        arm_pred = is_leader & nvvm.elect_sync()
        lo, hi, seq_q, seq_k, q_start = _item_bounds(pair, meta, batch, n_batch, causal)
        q1_desc = _map_ptr(maps, batch, n_batch, MAP_Q1)
        q2_desc = _map_ptr(maps, batch, n_batch, MAP_Q2)
        k_desc = _map_ptr(maps, batch, n_batch, MAP_K)
        v_desc = _map_ptr(maps, batch, n_batch, MAP_V)
        do1_desc = _map_ptr(maps, batch, n_batch, MAP_DO1)
        do2_desc = _map_ptr(maps, batch, n_batch, MAP_DO2)
        kv_head = head // ratio
        kv_row0 = pair * CFG.PAIR_KV_ROWS + rank * CFG.M
        if lo < hi:
            tma_tensormap_acquire(q1_desc)
            tma_tensormap_acquire(q2_desc)
            tma_tensormap_acquire(k_desc)
            tma_tensormap_acquire(v_desc)
            tma_tensormap_acquire(do1_desc)
            tma_tensormap_acquire(do2_desc)
            _wait(bars.k1_empty, kv_state.phase)
            bars.k1_full.arrive(n_bytes=CFG.K1_TX, pred=arm_pred)
            tma_load_tile(
                _operand_k_major(slab, CFG.K1, dtype, CFG.M),
                tma_slice_runtime_desc(k_desc, 0, kv_row0, kv_head, 0),
                bars.k1_full.smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mask,
                acquire=False,
            )
            _wait(bars.v_empty, kv_state.phase)
            bars.v_full.arrive(n_bytes=CFG.V_TX, pred=arm_pred)
            tma_load_tile(
                _operand_k_major(slab, CFG.V, dtype, CFG.M),
                tma_slice_runtime_desc(v_desc, 0, kv_row0, kv_head, 0),
                bars.v_full.smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mask,
                acquire=False,
            )
            # Only the K2 reload is gated by the previous item's dV store drain (dV staging aliases K2); the observation is answered
            # by a bare dv_full arrive (fix A): the store's next k2_empty arrive waits dv_full, so this waiter is never lapped.
            _wait(bars.k2_empty, k2e_state.phase)
            bars.dv_full.arrive()
            bars.k2_full.arrive(n_bytes=CFG.K2_TX, pred=arm_pred)
            for h in cutlass.range_constexpr(CFG.KV_HALVES):
                tma_load_tile(
                    _operand_mn_major(slab, CFG.K2 + h * CFG.K2_HALF_BYTES, dtype, CFG.M),
                    tma_slice_runtime_desc(k_desc, d_half, pair * CFG.PAIR_KV_ROWS + h * CFG.M, kv_head, 0),
                    bars.k2_full.smem_ptr,
                    cta_group=CFG.CTA_MMA,
                    mcast_mask=tma_mask,
                    acquire=False,
                )
            kv_state = advance(kv_state, 1)
        else:
            # zero-trip item: the same single k2_empty observation and dv_full answer (P14: every path flips every barrier once)
            _wait(bars.k2_empty, k2e_state.phase)
            bars.dv_full.arrive()
        k2e_state = advance(k2e_state, 1)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            q_row = qtile * CFG.N
            _wait(bars.q1_empty[q1_state.idx], q1_state.phase)
            bars.q1_full[q1_state.idx].arrive(n_bytes=CFG.Q1_TX, pred=arm_pred)
            tma_load_tile(
                _operand_k_major(slab, CFG.Q1 + q1_state.idx * CFG.Q1_BYTES, dtype, CFG.Q_HALF),
                tma_slice_runtime_desc(q1_desc, 0, q_row + q_half, head, 0),
                bars.q1_full[q1_state.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mask,
                acquire=False,
            )
            _wait(bars.do1_empty, do1_state.phase)
            for half in cutlass.range_constexpr(CFG.D_HALVES):
                bars.do1_full[half].arrive(n_bytes=CFG.DO1_HALF_TX, pred=arm_pred)
            for half in cutlass.range_constexpr(CFG.D_HALVES):
                tma_load_subtiles(
                    _operand_k_major(slab, CFG.DO1, dtype, CFG.Q_HALF),
                    tma_slice_runtime_desc(do1_desc, 0, q_row + q_half, head, 0),
                    bars.do1_full[half].smem_ptr,
                    half,
                    1,
                    cta_group=CFG.CTA_MMA,
                    mcast_mask=tma_mask,
                    acquire=False,
                )
            _wait(bars.do2_empty, do2_state.phase)
            bars.do2_full.arrive(n_bytes=CFG.DO2_TX, pred=arm_pred)
            tma_load_tile(
                _operand_mn_major(slab, CFG.DO2, dtype, CFG.N),
                tma_slice_runtime_desc(do2_desc, d_half, q_row, head, 0),
                bars.do2_full.smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mask,
                acquire=False,
            )
            _wait(bars.q2_empty, q2_state.phase)
            bars.q2_full.arrive(n_bytes=CFG.Q2_TX, pred=arm_pred)
            tma_load_tile(
                _operand_mn_major(slab, CFG.Q2, dtype, CFG.N),
                tma_slice_runtime_desc(q2_desc, d_half, q_row, head, 0),
                bars.q2_full.smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mask,
                acquire=False,
            )
            q1_state = advance(q1_state, CFG.Q1_STAGES)
            do1_state = advance(do1_state, 1)
            do2_state = advance(do2_state, 1)
            q2_state = advance(q2_state, 1)
        pair, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state)
    if cutlass.const_expr(CFG.CTA_MMA == 2):
        # P15: whole-ring drains from the CARRIED states (the leader's last multicast commits must land on a resident follower).
        q1_state = _drain(bars.q1_empty, q1_state, CFG.Q1_STAGES)
        q2_state = _drain(bars.q2_empty, q2_state, 1)
        do1_state = _drain(bars.do1_empty, do1_state, 1)
        do2_state = _drain(bars.do2_empty, do2_state, 1)
        _drain(bars.k1_empty, kv_state, 1)
        _drain(bars.v_empty, kv_state, 1)


@cute.jit
def _store_loop(
    slab,
    bars,
    sched,
    maps,
    pair,
    head,
    batch,
    meta,
    n_batch,
    rank,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    diag_2cta_code: cutlass.Constexpr[int],
    store_wait_spin: cutlass.Constexpr[bool],
    regs: cutlass.Constexpr[int],
):
    """Warp 14, BOTH CTAs: dQ reduce-add (4 fp32 slices [64 q x 32 d] per tile, one reduce per PAIR tile), dV / dK native stores.
    ``diag_2cta_code`` (DIAGNOSTIC, default DIAG_NONE = this body): DIAG_DQ_REDUCE_ADD_OFF skips the four tma_reduce_add ISSUES per tile
    (D2-pair, the 1-CTA ``diag_delete_dq_reduce_add`` form); the dq_full waits, the commit / wait groups and the dq_empty arrives stay,
    so the barrier table is unchanged and dQ stays ZERO (output WRONG BY DESIGN; `--validate` refuses the arm).
    ``store_wait_spin`` (KNOB ``store_wait_form``, default retry = False): the six mbarrier waits of this warp (dq_full x 4, dv_full, dk_full)
    through :func:`_wait_store` -- the same phase tests, only the retry mechanics move; the sched_full wait (``_next_tile``, a whole-item
    park) and tma_store_wait (one spelling) stay.  Bitwise the same function under both forms.

    # BARRIER (per item, first): _credit(bars.sched_empty[sched_state.idx]); for i in range_constexpr(4): bars.dq_empty[i].arrive()
    #   [bootstrap, bare 32 lanes per slot == init; the epilogue's dS write of the item waits it: it also publishes the previous
    #    item's dV / dK store drain]
    # BARRIER (per tile t) -- SLICE ORDER 2, 3, 0, 1 (the XCHG bytes drain first; seam note in header section 6):
    #   for i in (2, 3, 0, 1): _wait(bars.dq_full[i], dq.phase) -> tma_reduce_add_subtile(_dq_slice(slab, rank, i), ..) -> tma_store_commit()
    #   tma_store_wait(3) -> bars.dq_empty[2].arrive(); tma_store_wait(2) -> dq_empty[3].arrive(); tma_store_wait(1) -> dq_empty[0].arrive();
    #   tma_store_wait(0) -> dq_empty[1].arrive()   [bare, 32 each, after the corresponding SOURCE READ];  dq = advance(dq, 1)
    # BARRIER (item end): _wait(bars.dv_full, out.phase) -> dV store (2 halves from CFG.dv_store(half); skipped for a padded CTA);
    #   _wait(bars.dk_full, out.phase) -> dK store (CFG.dk_store(rank, half)); out = advance(out, 1); tma_store_commit(); tma_store_wait(0)
    #   -> _arrive_leader(bars.dvk_empty)   [LEADER, bare, 32 x 2 = 64]   -> bars.k2_empty.arrive()   [THREAD local, bare, 32; its back-edge:
    #      this arrive follows _wait(dv_full), whose 160 include the loader's 32 = the loader observed the previous k2_empty phase (fix A)]
    # No drain: dvk_empty's last phase is the leader MMA's drain; k2_empty / dq_empty are local.
    """
    nvvm.setmaxregister(regs, nvvm.SetMaxRegisterAction.DECREASE)  # CFG.REG_SPLITS[reg_split][2] (knob; default SERVICE_REGS)
    dq_state = PipelineState.start()
    out_state = PipelineState.start()
    sched_state = PipelineState.start()
    q_half = rank * cutlass.Int32(CFG.Q_HALF)
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        for i in cutlass.range_constexpr(CFG.DQ_SLICES):
            bars.dq_empty[i].arrive()  # all 32 lanes, one bootstrap per item
        lo, hi, seq_q, seq_k, q_start = _item_bounds(pair, meta, batch, n_batch, causal)
        dq_desc = _map_ptr(maps, batch, n_batch, MAP_DQ)
        if lo < hi:
            tma_tensormap_acquire(dq_desc)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            q_row = qtile * CFG.N + q_half
            for i in cutlass.range_constexpr(CFG.DQ_SLICES):
                slot = CFG.DQ_SLICE_ORDER[i]
                _wait_store(bars.dq_full[slot], dq_state.phase, store_wait_spin)
                if cutlass.const_expr(diag_2cta_code != DIAG_DQ_REDUCE_ADD_OFF):  # DIAGNOSTIC D2-pair: the ISSUE only (commit / wait / arrives stay)
                    tma_reduce_add_subtile(_dq_slice(slab, rank, slot), tma_slice_runtime_desc(dq_desc, slot * CFG.DQ_BOX, q_row, head, 0), 0, acquire=False)
                tma_store_commit()
            for i in cutlass.range_constexpr(CFG.DQ_SLICES):
                tma_store_wait(CFG.DQ_SLICES - 1 - i)
                bars.dq_empty[CFG.DQ_SLICE_ORDER[i]].arrive()  # all 32 after the corresponding source read
            dq_state = advance(dq_state, 1)
        # Invalid owners and the padded SECOND half of a live pair still consume both native publications and return every source
        # credit, but never acquire / use a map for rows past the sequence.
        kv_row0 = pair * CFG.PAIR_KV_ROWS + rank * CFG.M
        live_rows = kv_row0 < seq_k
        _wait_store(bars.dv_full, out_state.phase, store_wait_spin)
        if live_rows:
            dv_desc = _map_ptr(maps, batch, n_batch, MAP_DV)
            tma_store_tile(_operand_k_major(slab, CFG.K2, dtype, CFG.M), tma_slice_runtime_desc(dv_desc, 0, kv_row0, head, 0))
        _wait_store(bars.dk_full, out_state.phase, store_wait_spin)
        if live_rows:
            dk_desc = _map_ptr(maps, batch, n_batch, MAP_DK)
            for half in cutlass.range_constexpr(CFG.D_HALVES):
                tma_store_tile(_store_half(slab, CFG.dk_store(rank, half), dtype), tma_slice_runtime_desc(dk_desc, half * CFG.HEAD_BOX, kv_row0, head, 0))
        out_state = advance(out_state, 1)
        tma_store_commit()
        tma_store_wait(0)
        _arrive_leader(bars.dvk_empty)
        bars.k2_empty.arrive()
        pair, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state)


@cute.jit
def _scheduler_loop(sched, is_leader, regs: cutlass.Constexpr[int]):
    """Warp 15: CLC try_cancel over the 256-row-unit envelope; the LEADER's scheduler publishes to the pair, the follower's has no loop.

    # LEADER only (the follower's warp 15 falls through to the final barrier_cta_sync), st(0) adv SCHED_STAGES -- the library's
    # scheduler_warp_loop IS the punch list: _wait(bars.sched_empty[st.idx], st.phase) [30 credits: 15 local + 15 remote .release.cta];
    # if elect: bars.sched_full[st.idx].arrive(n_bytes=CFG.CLC_TX) [our copy, 1 == init] + nvvm.mbarrier_arrive_expect_tx(mapa(
    # bars.sched_full[st.idx].smem_ptr, 1), CFG.CLC_TX, scope=CTA) [the follower's copy, 1 == init] + clusterlaunchcontrol_try_cancel(
    # multicast=1); fence_proxy; warp sync; _wait(bars.sched_full[st.idx], st.phase); read_clc_payload -> valid; advance.
    # Every sched_empty phase is observed in-loop (the credits of tile n gate payload n + 1; the terminal payload carries no credit):
    # no drain.  The payload decodes the PAIR's unit (x // CGA_M); rank = the low bit of block_idx_in_cluster selects each CTA's kv half.
    """
    # RESOURCE-NOTE: THD unit = 256 kv rows per pair; causal trim per pair; CTA 1's diagonal tile is masked.
    nvvm.setmaxregister(regs, nvvm.SetMaxRegisterAction.DECREASE)  # CFG.REG_SPLITS[reg_split][2] (knob; default SERVICE_REGS)
    scheduler_warp_loop(sched, CFG.SCHED_STAGES, is_leader, CFG.CGA_M)


@cute.kernel
def bprop_d128_accum_grouped_2cta_kernel(
    maps: cute.Tensor,
    lse: cute.Tensor,
    delta: cute.Tensor,
    meta: cute.Tensor,
    origins: cute.Tensor,
    n_batch: cutlass.Int32,
    stats_origin: cutlass.Constexpr[int],
    scale: cutlass.Float32,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    ratio: cutlass.Constexpr[int],
    s_lookahead_after_dk: cutlass.Constexpr[bool],
    tma_load_plain: cutlass.Constexpr[bool],
    diag_2cta_code: cutlass.Constexpr[int],
    store_wait_spin: cutlass.Constexpr[bool],
    reg_split_code: cutlass.Constexpr[int],
):
    """Slab, scaffold views, the barrier init block (P4), the pair TMEM allocation, the role dispatch, the pair-symmetric teardown.
    ``s_lookahead_after_dk`` = the leader's MMA issue order (header section 5), from the adapter's typed ``pair_mma_order`` field;
    ``tma_load_plain`` = the cga2 TMA load form (header section 10), from the typed ``tma_load_form`` field; ``diag_2cta_code`` = the
    DIAGNOSTIC probe (header section 11; DIAG_NONE = this body), from the typed ``diag_2cta`` field; ``store_wait_spin`` = the store
    warp's wait form (KNOB ``store_wait_form``) and ``reg_split_code`` = the setmaxnreg split (KNOB ``reg_split``, header section 4), both
    perf-only (bitwise the same function under every value)."""
    softmax_regs, epilogue_regs, service_regs = reg_split_counts(reg_split_code)  # config = truth: the split's three counts, asserted there
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    x, y, z = cute.arch.block_idx()
    rank = cute.arch.block_idx_in_cluster() & cutlass.Int32(1)  # pair rank under cluster (2,1,1): 0 = LEADER, 1 = FOLLOWER
    is_leader = rank == cutlass.Int32(LEADER_RANK)  # the loader's arm predicate and the scheduler / MMA role select
    pair, head, batch = x // CFG.CGA_M, y, z  # grid x = pairs x CGA_M: both CTAs of a cluster decode the same pair
    slab = cutlass.Array(cutlass.Uint8, CFG.SMEM_BYTES, alignment=1024, space=cutlass.AddressSpace.smem)
    bars = _bars(slab)
    tmem_slot = _view(slab, CFG.TMEM_HANDLE, cutlass.Int32, 1, alignment=4)
    sched_words = _view(slab, CFG.SCHED_PAYLOAD, cutlass.Int32, CFG.SCHED_PAYLOAD_BYTES // 4, alignment=16)
    sched = Sched(bars.sched_full.base_ptr, bars.sched_empty.base_ptr, sched_words, x, y, z)
    # --- P4: ONE warp, ONE lane initialises every stage of every ring; fence -> CTA sync -> CLUSTER sync, all OUTSIDE the branch.
    if warp == 0:
        if nvvm.elect_sync():
            for i in cutlass.range_constexpr(len(bars)):
                for stage in cutlass.range_constexpr(bars[i].stages):
                    bars[i][stage].init()
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()
    cga_arrive()  # required before ANY cross-CTA arrive (multicast commits, leader arrives, the remote CLC arm, the DSMEM ship)
    cga_wait()
    # --- TMEM: one tcgen05.alloc.cta_group::2 of TMEM_COLS per CTA (BOTH CTAs allocate; the leader's collective MMAs write both halves).
    if warp == MMA_WARP:
        tmem_alloc(tmem_slot, CFG.TMEM_COLS, nvvm.CTAGroup.CTA_2)
    nvvm.barrier_cta_sync()  # publishes the TMEM base to every warp
    tmem_base = tmem_slot.load()
    # BARRIER: role dispatch: warp < 8 softmax | < 12 epilogue | == MMA_WARP (leader: _mma_loop_leader, follower: _mma_loop_follower)
    #   | == TMA_LOAD_WARP | == TMA_STORE_WARP | SCHED_WARP (leader only runs the loop).
    if warp < SOFTMAX_WARP_END:
        _softmax_loop(
            slab, bars, sched, tmem_base, lse, delta, origins, stats_origin, pair, head, batch, meta, n_batch, rank, scale, causal, dtype, softmax_regs
        )
    elif warp < EPILOGUE_WARP_END:
        _epilogue_loop(slab, bars, sched, tmem_base, pair, head, batch, meta, n_batch, rank, warp, causal, dtype, epilogue_regs)
    elif warp == MMA_WARP:
        if is_leader:
            _mma_loop_leader(slab, bars, sched, tmem_base, pair, head, batch, meta, n_batch, causal, dtype, s_lookahead_after_dk, diag_2cta_code, service_regs)
        else:
            _mma_loop_follower(bars, sched, pair, head, batch, meta, n_batch, causal, service_regs)
    elif warp == TMA_LOAD_WARP:
        _load_loop(slab, bars, sched, maps, pair, head, batch, meta, n_batch, rank, is_leader, causal, dtype, ratio, tma_load_plain, service_regs)
    elif warp == TMA_STORE_WARP:
        _store_loop(slab, bars, sched, maps, pair, head, batch, meta, n_batch, rank, causal, dtype, diag_2cta_code, store_wait_spin, service_regs)
    else:
        _scheduler_loop(sched, is_leader, service_regs)
    # BARRIER (teardown, both CTAs): the final CTA sync after every role returned, then the MMA warp's pair-symmetric dealloc.
    #   No remote arrive targets an exited CTA: the pair-symmetric dealloc wait is the last cross-CTA hand-off.
    nvvm.barrier_cta_sync()
    if warp == MMA_WARP:
        _tmem_dealloc_pair(bars, tmem_slot, rank)


bprop_d128_accum_grouped_2cta_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def main_thd_2cta_host(
    maps,
    lse,
    delta,
    meta,
    origins,
    n_batch,
    nheads,
    max_skv,
    stats_origin: cutlass.Constexpr[int],
    scale,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    ratio: cutlass.Constexpr[int],
    s_lookahead_after_dk: cutlass.Constexpr[bool],
    tma_load_plain: cutlass.Constexpr[bool],
    diag_2cta_code: cutlass.Constexpr[int],
    store_wait_spin: cutlass.Constexpr[bool],
    reg_split_code: cutlass.Constexpr[int],
    stream,
):
    """One owner PAIR per Q head and declared 256-row KV-envelope unit (header section 14); no host length read."""
    n_pairs = cute.math.max((max_skv + CFG.THD_UNIT_ROWS - 1) // CFG.THD_UNIT_ROWS, cutlass.Int32(1))
    bprop_d128_accum_grouped_2cta_kernel(
        maps,
        lse,
        delta,
        meta,
        origins,
        n_batch,
        stats_origin,
        scale,
        causal,
        dtype,
        ratio,
        s_lookahead_after_dk,
        tma_load_plain,
        diag_2cta_code,
        store_wait_spin,
        reg_split_code,
    ).launch(
        grid=(n_pairs * CFG.CGA_M, nheads, n_batch),
        block=(CFG.THREADS, 1, 1),
        cluster=(CFG.CGA_M, CFG.CGA_N, 1),
        stream=stream,
        min_blocks_per_mp=1,
    )


__all__ = [
    "ALIASES",
    "BAR_LAYOUT",
    "Bars",
    "CFG",
    "LEADER_RANK",
    "MAP_COUNT",
    "MAP_DK",
    "MAP_DO1",
    "MAP_DO2",
    "MAP_DQ",
    "MAP_DV",
    "MAP_K",
    "MAP_Q1",
    "MAP_Q2",
    "MAP_V",
    "MCAST_PAIR",
    "bprop_d128_accum_grouped_2cta_kernel",
    "main_thd_2cta_host",
]
