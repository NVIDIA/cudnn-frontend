# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""SM100 SDPA backward d=128x128 bf16/fp16 -- grouped THD, cta_group::2 datapath, ARM (ii): dQ in S^T's tail + FAv4's rotated order
(section 1: identity)

HEADER FIRST: the tables below come before the body; a reader checks the body against them instead of reconstructing the
pipeline from code.  This is the SECOND pair body: a copy of ``bprop_d128_accum_grouped_2cta_f16.py`` (UNTOUCHED, the first body,
the ``tmem_map = "dp_tail"`` default) re-planned onto the TMEM map of resource-manager 2026-10-27 00:53 UTC (tables 1-6) and the
34-row barrier table of barrier-inspector 2026-10-27 01:05 UTC (resources/arm2-barrier-table.md, SKELETON; its deviations D1-D5
OVERRIDE the resource-manager's table 3).  Config as truth: ``config_sm100_d128_accum_2cta_stail.CFG`` (every number below is
derived there and asserted at import; the adapter's typed field ``tmem_map = "s_tail"`` selects this body).  FAv4's
flash_bwd_sm100.py was READ as the structural reference only (NOTE [2CTA schedules], the pipelined MMA loop, the deferred dS
commit, ``Q_stage = 1``); nothing copied.  Written by code-writer-dsl 2026-10-27 (lane A step 5).

2. WHAT IT COMPUTES (per (batch, kv head, q head); GQA: one q-head owner per pair, as today's grouped loaders)
    S^T = K Q^T * scale        P^T = exp2(S^T * log2e - LSE[q])        dP^T = V dO^T
    dS^T = P^T * (dP^T - delta[q])        dV += P^T dO        dK += dS^T Q        dQ = dS K  (fp32, tma_reduce_add | red.global.add.v4.f32:
    the typed field ``dq_route_pair`` = "tma" (THE default) | "red_global" | "red_global_pair", section 11; the fp32 partial lands in the SAME
    dq_accum workspace either way, summed in a non-deterministic fp32 order under every route; cast_dq folds it in the NEXT launch)
    delta = rowsum(dO * O) comes from the chain's delta launch (THD: setup_thd -> delta_zero_thd -> MAIN -> cast_dq_thd
    -> the fp32 group fold); LSE in the natural-log unit of the forward's Stats.  Numerics = the first body's by construction
    (the same accumulation chain per GEMM, the same k-orders; only the ISSUE positions and the dQ image's TMEM columns move):
    dK / dV are pinned BITWISE against the first body by the test module; dQ (reduce-add order across pairs) by the fp64 gate.
    mask_form (section 11, lever L5): P^T / dS^T are BITWISE between "fsel" (the default: the keep-word mask selected on every tile) and
    "predicated" (the same mask body only on a tile that can hold a masked cell: a masked cell is exactly +0.0 in both, an interior tile has
    none), hence dQ (per reduce-add order) / dK / dV; "diag_mask_off" is the DIAGNOSTIC ceiling, OUTPUT WRONG BY DESIGN.

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
    The D image of the M 128 pair MMA is the half-row atom: dQ row m -> lane m % 64 + 64 * (d // 64), 64 columns -- the SAME
    lane map as the first body; only its column base moves (CFG.DQ_COL = 64 = S^T's dead tail).
    Operand bytes per CTA per q tile: Q1 16 + Q2 16 + dO1 16 + dO2 16 KiB; per kv block K1 32 + K2 32 + V 32 KiB.
    TS k-step order: the packed P^T / dS^T columns follow the softmax quarter ownership (group g owns q quarters {g, g+2}),
    so TS k-step k (8 packed columns = 16 q) pairs with the B q block CFG.ts_k_perm(k) = [0, 1, 4, 5, 2, 3, 6, 7][k].

4. WARP MAP        role | warps | regs (setmaxnreg) | what it credits (names = the inspector's table)
    softmax   | 0-7   | 144 | s_read (NEW: pair-wide on the leader, after the S^T ld pair, EVERY tile), p_ready, ds_ready (pair-wide),
                             ds_tmem (local), stats consumption; mask_form predicated (section 11): the keep-word mask body of _p_values
                             only on `tile_may_mask` tiles (a CTA-uniform per-tile scalar: no live register added, no barrier touched)
    epilogue  | 8-11  | 144 | dp_empty (leader; MOVED: after the dS^T readout, before the recv arm), ds_smem (own half by STS),
                             ds_recv_full (expect_tx) + the XCHG ship, s_tail_empty (NEW: leader, after the dQ readout), dq_full x 2
                             slots x 2 rounds per tile, dv_full, dk_full
    mma       | 12    |  80 | LEADER only: s_done, dp_done, dq_done (commit, pair multicast; dq_done in-tile for dQ(t-1) AND at the
                             tail for dQ(hi-1)), *_empty releases; the follower's MMA warp runs the quiet body (alloc, the
                             ds_pair_ready relay, dealloc)
    tma_ldg   | 13    |  80 | q1_full[1] (ONE stage), q2_full, do1_full[2], do2_full, k1_full, k2_full, v_full (expect_tx on the
                             LEADER for 2x bytes; the follower issues its half naming the leader's mbar); dv_full (32, fix A)
    tma_stg   | 14    |  80 | dq_empty x 2 slots per ROUND (reduce-add source read drained; D4 closed per-round release), dvk_empty
                             (leader), k2_empty
    sched     | 15    |  80 | sched_full / sched_empty (CLC payload delivered to the pair); never credits its own ring
    SUM(role regs x warps) = (8 x 144 + 4 x 144 + 4 x 80) x 32 = 65536 == entry 128 x 16 warps x 32; every count % 8 == 0.
    The epilogue carries 32 fp32 of dQ(t-1)'s second slice across the dS(t) staging (~96 live of 144): the vs-sass STL / LDL == 0
    pin is owed on this body (the PTX census pins `.local` == 0).  KNOBS reg_split / store_wait_form as the first body (section 11).
    STAMPS (the typed field ``debug_clk``, section 11): ONE representative warp per role -- softmax warp 0, the epilogue LEAD warp 8, the
    single service warps 12 / 13 / 14 -- brackets each of ITS mbarrier waits with two %clock64 reads (the wait itself unmodified) and its
    elected lane accumulates per (role, barrier) into the Int32 record in the bar block's spare (config DBG_CLK); no barrier, no fence,
    no SMEM layout change; the scheduler warp's library loop is not instrumented.  Default off = no traced code.
    X-FA4 ROLE SHAPE (typed field ``warp_roles = "16w_fa4"``; code-writer B 2026-10-27 on day-3 rank 1; the SAME 16 warps, the SAME split):
    softmax   | 0-7   | 144 | + the dS STS: 4 own-half + 4 XCHG STS.128 per lane per tile straight from its 32 packed dS^T words (= today's LDTM
                             image word for word, resource-manager table 1; group g writes vec 4g..4g+3 of BOTH 64-q halves), ONE fence.proxy.async,
                             ds_smem (NOW 256 = SOFTMAX_LANES, bare), the 256-lane XCHG named barrier, warp SOFTMAX_LEAD_WARP's elected lane arms
                             ds_recv_full + ships XCHG; a NEW WAR wait on dq_done(t-1) (`dq_war` st(1), +1 per wait) before the first dS STS of every
                             tile.  ds_tmem GONE.  +4..8 transient registers (the words are live for the STTM anyway); STL / LDL 0 = the gate
    reduce    | 8-11  | 144 | (the former epilogue, `_reduce_loop`): dq_done(t) -> the x64 dQ readout -> s_tail_empty -> rounds A, B (no `carried`
                             half, no item-end round) -> the item-end dV / dK native staging (dv_full 160, dk_full 128).  No dS work: ~80-90 live
    mma / tma_ldg / tma_stg / sched: unchanged (dp_empty GONE from the leader: no slot-1 wait, no state, no drain)
    RESOURCE-NOTE: split 144/144/80 unchanged (pool 2048); softmax +4..8 regs; fallback 152/128/80

5. BARRIER TABLE  (barrier-inspector 2026-10-27 01:05 UTC, SKELETON for ARM (ii); ``config_sm100_d128_accum_2cta_stail.BAR_LAYOUT`` =
   offsets / counts / kinds; 34 names / 39 slots in [230400, 230944) of the 944-B bar block, re-packed, 16 B per row).  L = pair LEADER
   (rank 0, the only MMA issuer), F = FOLLOWER (rank 1); "(-)" = the follower's copy is initialised with the SAME count and never armed /
   never waited; N = hi - lo q tiles of the item; x T = per CLC item.  Issuing lanes come from the guard form ONLY: bare 1 warp = 32, bare
   8 warps = 256, pred / elect = 1 per warp, pred=is_leader & elect = 1 for the pair, a tcgen05.commit = 1 arrive per TARGET CTA; x the
   CTAs that reach the site.  Rows marked NEW / MOVED / SPLIT are the arm's; every other row is the first body's AUDIT (23:10) line.
    name[st]         kind              init L / F     SUM(issuing lanes) == init        waiter (own copy unless "L only")   start  fires
    q1_full[1]       TMA_LOAD          1 / 1(-)       1 (L, pred=is_leader & elect) == 1  MMA(L) before every S^T issue     st(0)  x N
        arm: loader `q1_full[s].arrive(n_bytes=Q1_TX, pred=is_leader & elect_sync())` (s = the single slot); BOTH CTAs issue tma_load_tile(cta_group=2)
        ONE stage (Q1_STAGES 1: FAv4's Q_stage = 1 under 2-CTA funds the sDQ ring; Q1(t+1) has ~2000 clk after q1_empty(t): 4 GEMMs between S^T issues)
    q1_empty[1]      MMA_COMMIT        1 / 1          1 commit x 2 target CTAs == 1 each  loader L+F before the Q1 reload     st(1)  x N
        site: MMA(L) right after each S^T issue (prologue + lookahead), `_commit_pair(bars.q1_empty[s], elect_p)`; DRAIN x Q1_STAGES = 1
    q2_full[1]       TMA_LOAD          1 / 1(-)       1 (L) == 1  (Q2_TX)                 MMA(L) before dK(t)               st(0)  x N
    q2_empty[1]      MMA_COMMIT        1 / 1          1 x 2 == 1 each                      loader L+F                        st(1)  x N  (after dK(t); DRAIN x1)
    do1_full[2]      TMA_LOAD          1 / 1(-) /slot 1 (L) per half slot == 1 (DO1_HALF_TX each, range_constexpr(2))  MMA(L) dP^T k-half 0 | 1  st(0)  x N
    do1_empty[1]     MMA_COMMIT        1 / 1          1 x 2 == 1 each                      loader L+F                        st(1)  x N
            (after dP^T(t); DRAIN x1)
    do2_full[1]      TMA_LOAD          1 / 1(-)       1 (L) == 1  (DO2_TX)                MMA(L) before dV(t)               st(0)  x N
    do2_empty[1]     MMA_COMMIT        1 / 1          1 x 2 == 1 each                      loader L+F                        st(1)  x N  (after dV(t); DRAIN x1)
    k1_full[1]       TMA_LOAD          1 / 1(-)       1 (L) == 1  (K1_TX)                 MMA(L) before S^T(lo)             st(0)  x T (live)
    k1_empty[1]      MMA_COMMIT        1 / 1          1 x 2 == 1 each                      loader L+F                        st(1)  x T
            (block end, after the tail dQ; DRAIN x1)
    v_full[1]        TMA_LOAD          1 / 1(-)       1 (L) == 1  (V_TX)                  MMA(L) in the prologue            st(0)  x T (live)
    v_empty[1]       MMA_COMMIT        1 / 1          1 x 2 == 1 each                      loader L+F                        st(1)  x T  (block end; DRAIN x1)
    k2_full[1]       TMA_LOAD          1 / 1(-)       1 (L) == 1  (K2_TX)                 MMA(L) in the prologue            st(0)  x T (live)
    k2_empty[1]      THREAD            32 / 32        store warp bare after tma_store_wait(0) = 32 == 32   loader (own) before K2  st(1)  x T (every item)
        the MMA's read of K2 is covered by dq_done(t_last) (observed by the epilogue before dv_full) -> dV store -> drain -> this arrive;
        BACK-EDGE (fix A carried): the loader observes it right before the K2 reload (live) or once at the item top (zero-trip), then arrives
            dv_full (bare 32); the store's NEXT k2_empty arrive follows its dv_full wait (P14-B)
    s_done[1]        MMA_COMMIT        1 / 1          1 x 2 == 1 each                      softmax L+F                       st(0)  x N
    dp_done[1]       MMA_COMMIT        1 / 1          1 x 2 == 1 each                      softmax L+F                       st(0)  x N
    dq_done[1]       MMA_COMMIT        1 / 1          1 x 2 == 1 each: after dQ(t-1) [tile t slot 2], after the tail dQ(hi-1), + one EMPTY
                                                      commit per zero-trip item            epilogue L+F (dq_state)           st(0)  x N (+1 dead)
    p_ready[1]       LEADER (relaxed)  512 / 512(-)   softmax 8 warps bare x 2 CTAs = 256 x 2 = 512 == PAIR_SOFTMAX_LANES   MMA(L) before dV(t)  st(0)  x N
        site: after tcgen05_wait(STORE) + tcgen05_fence(BEFORE_THREAD_SYNC): `_arrive_leader(bars.p_ready)` (TMEM data: relaxed, P8)
    ds_ready[1]      LEADER (relaxed)  512 / 512(-)   same lanes, after the dS^T pack = 512 == 512            MMA(L) before dK(t)    st(0)  x N
    s_read[1] NEW    LEADER (relaxed)  512 / 512(-)   softmax 8 warps bare x 2 CTAs after the S^T ld pair (`_s_values`) + tcgen05_wait(LOAD)
                                                      + tcgen05_fence(BEFORE_THREAD_SYNC), BEFORE the exp body, EVERY tile (D2: no runtime
                                                      guard on a 512-lane arrive) = 256 x 2 = 512 == PAIR_SOFTMAX_LANES
                                                      waiter MMA(L) at slot 2 of EVERY tile, before the conditional dQ(t-1) (ready_state); at
                                                      t == lo dominated by the p_ready(lo) wait that follows             st(0)  x N  (in-loop)
    ds_tmem[1]       THREAD            256 / 256      softmax 8 warps bare, local, at the ds_ready site = 256 == 256   epilogue (own)   st(0)  x N
    dp_empty[1] MOVED LEADER (relaxed) 256 / 256(-)   epilogue 4 warps bare x 2 CTAs right after the dS^T 4 x tcgen05.ld + tcgen05_wait(LOAD)
                                                      + fence, BEFORE `ds_recv_full.arrive` (moved off the dQ readout) = 128 x 2 = 256 ==
                                                      PAIR_EPILOGUE_LANES                  MMA(L) before dP^T(t+1) (dpe) st(1)  x N; DRAIN x1 (MMA(L))
        RESOURCE-NOTE: dp_empty = dS^T readout only (arrive before the XCHG STS); dP^T(t+1) never waits dQ
    s_tail_empty[1] NEW LEADER (relaxed) 256 / 256(-) epilogue 4 warps bare x 2 CTAs right after `_tmem_load(DQ_COL, 64)` + tcgen05_wait(LOAD) +
                                                      tcgen05_fence(BEFORE_THREAD_SYNC) (the first body's dp_empty site) = 128 x 2 = 256 ==
                                                      PAIR_EPILOGUE_LANES.  waiter MMA(L) (s_tail_state, +1 per wait) at the THREE sites of
                                                      D1: the prologue S^T(lo) [always; pre-armed for the kernel's first item], the lookahead
                                                      S^T(t+1) iff qtile > lo, the tail dQ(hi-1) iff hi - 1 > lo       st(1)  x N (one per dQ readout)
                                                      ; DRAIN x1 (MMA(L), after the loop: the last item's tail readout, P15)
        RESOURCE-NOTE: s_tail_empty right after wait(LOAD): S^T(t+1) waits the readout, never the STS
    ds_smem[1]       THREAD            128 / 128      epilogue 4 warps bare after own-half STS + fence_proxy + tcgen05 fence = 128 == 128;
                                                      NO dq_empty wait before the STS any more (the staging alias is deleted)
            MMA warp (own): L dQ issue on xchg_state (in-tile + tail), F relay (relay_state)   st(0) x N
    ds_recv_full[1]  TMA_LOAD (dsmem)  1 / 1          epilogue warp 8, pred=is_lead_warp & elect = 1 == 1; tx = the PEER's cp.async.bulk XCHG_TX
            MMA warp (own): L on xchg_state, F relay  st(0) x N
    ds_pair_ready[1] THREAD (remote)   1 / 1(-)       MMA(F) `arrive_on_peer(0, pred=elect_p, release_cta=True)` after ds_smem + ds_recv_full = 1
                                                      -> L == 1                            MMA(L) before each dQ issue (xchg_state)  st(0) x N
        SPLIT (P2): the three dS gates ride `xchg_state` (+1 per dQ ISSUE: slot 2 of tiles lo+1 .. hi-1 and the tail), ONE tile behind
            `ready_state` (p_ready / ds_ready / s_read, +1 per tile).  On ready_state they would park on completion t while the epilogue's
            dS(t) STS waits dq_done(t-1) = the dQ(t-1) issue BEHIND the wait: deterministic HANG at tile lo+1 (section 8 of the table, RED 4)
    dq_empty[2]      THREAD            32 / 32 /slot  store warp bare after each ROUND's source reads: tma_store_wait(1) -> the slot-0 arrive,
                                                      tma_store_wait(0) -> the slot-1 arrive = 32 == STORE_LANES; NO bootstrap (D3)
            epilogue (own, empty_state st(1)): all 4 warps wait BOTH slots before each round (D4 pair: closed per-round release + both-slots
            wait form; "change both or neither")       x 2N per slot (2 rounds per tile)
    dq_full[2]       THREAD            64 / 64 /slot  epilogue warp pair dhalf (warps 8-9 -> slot 0, 10-11 -> slot 1) bare after the slice STS +
                                                      fence_proxy = 2 x 32 = 64 == DQ_SLICE_LANES, per round
            store (own, dq_state 2 stages, +1 per USE): uses per tile (slot 0, slice 0), (slot 1, slice 2), (slot 0, slice 1), (slot 1, slice 3)
                                                                                                          st(0)  x 2N per slot
    dv_full[1]       THREAD            160 / 160      epilogue 4 warps bare after the dV staging STS + fence = 128 + loader bare after its
            k2_empty observation = 32 -> 160 == DV_FULL_LANES   store (own)   st(0)  x T   (fix A carried)
    dk_full[1]       THREAD            128 / 128      epilogue 4 warps bare after the dK staging STS + fence = 128 == 128    store (own)   st(0)  x T
    dvk_empty[1]     LEADER (relaxed)  64 / 64(-)     store warp bare x 2 CTAs after tma_store_wait(0) = 32 x 2 = 64 == 64   MMA(L) item top
            st(1)  x T; DRAIN x1 (MMA(L))
    sched_full[2]    TMA_LOAD (CLC)    1 / 1          scheduler(L) elect: local arrive(n_bytes=CLC_TX) + mapa(F) mbarrier_arrive_expect_tx(CLC_TX
            , CTA) = 1 per CTA == 1; completion: clusterlaunchcontrol.try_cancel(multicast=1) lands 16 B in EACH CTA's slot
            ; waiters: every consumer warp (own copy) + scheduler(L)  st(0) x T
    sched_empty[2]   THREAD (remote)   30 / 30(-)     read_tile_id_arrive_on_leader(mb, CGA_M) from 15 warps per CTA (all but the scheduler) x 2 = 30 == 30
        waiter: scheduler(L) before re-arming the slot (every phase observed in-loop)   st(0) x T
    tmem_dealloc[1]  THREAD (+peer)    2 / 2          MMA warp, each CTA, after the final CTA sync: `if elect: arrive(); arrive_on_peer(peer rank)` = 1 + 1 == 2
        waiter: the MMA warp of each CTA, wait(0) -> tcgen05_fence -> tcgen05.dealloc.cta_group::2  x 1
    dQ ROUTE red_global / red_global_pair (step 8): NO row changes -- the epilogue still waits dq_empty[0] + dq_empty[1] per round and arrives
    dq_full[dhalf] per round; the store warp still waits dq_full x 4 uses per tile, keeps `tma_store_wait(1) / (0)` (no bulk group in flight:
    a no-op) and arrives dq_empty[s] per round; only the sDQ STS + fence_proxy (epilogue) and the UTMAREDG issue + per-use `tma_store_commit`
    (store) are skipped under `const_expr`.  The reds carry no barrier: fire-and-forget on the LSU global path; the dq_accum consumer is the
    NEXT launch (cast_dq, stream order), so no fence orders them against anything in this kernel (the TMA route's proxy fences cover SMEM
    only).  PTX census pin: arrive sites 40 / wait sites 68 IDENTICAL under every route value.
    DIAGNOSTIC PROBES vs this table (day 3, code-writer 2026-10-27 on barrier-inspector 08:01): D8s `exchange_off` moves NO row -- the
    epilogue's `ds_recv_full.arrive(n_bytes=..)` keeps its one elected lane and arms CFG.DIAG_EXCHANGE_OFF_TX = 0 bytes, so each phase
    completes AT ITS OWN ARM (1 == 1, nothing in flight: no tx-pending over-arrival window exists), the XCHG ship is not issued, and the
    three dS gates (xchg_state), the follower relay, the XCHG STS / proxy fence / CFG.XCHG_BARRIER and every other wait and arrive stay;
    D10 `stats_one_chunk` and D9 `sdq_sts_off` touch no barrier (LDS / STS bytes only); STAMPS `debug_clk` adds clock reads and SMEM
    words only.  D8 `ds_gates_off` is RETIRED on this map (declined by name): skipping the three gates removed the ONLY back-edge of
    the ds_recv_full arm (K-site arrive(t+1) <- dq_done(t) <- the ds_recv_full(t) / ds_pair_ready(t) waits) and the arm of tile t+1
    landed on a tx-pending phase = P3 over-arrival -> CUDA 719 on every launch (inspector 08:01; RULE-GAPS 527).
    dkv_accum fp32_reduce (C2b, code-writer C 2026-10-27; the typed op-level field, ``dkv_fp32`` Constexpr[bool], default False = this table
    byte-identical; resource-manager 08:25 table 9 form (i)): the item-end dV / dK publications carry fp32 slices instead of bf16 halves --
    dv_full (160, the same lanes and sites) publishes ROUND 1 = dV's four slices + dK's slice 0 (five 16 KiB slots: sK2's halves, the dS slab's
    halves, XCHG), dk_full (128) publishes ROUND 2 = dK's slices 1..3 (three of those slots).  NEW ROW dvk_stage_empty[1] THREAD 32 / 32 LOCAL
    (BAR_LAYOUT @ 230944, appended after tmem_dealloc; INIT ONLY under dkv_fp32): the store warp arrives it bare (32 == init) after round 1's
    bulk group drained (`tma_store_commit` -> `tma_store_wait(0)`, P7); the epilogue's 4 warps wait it (own, stage_state st(0), +1 per item
    incl. zero-trip) between round 1's dv_full arrive and round 2's STS = the WAR edge of the five slots; x 1 per item; no drain (every phase
    observed in-loop by the epilogue before its round 2).  The dq_empty ring is NOT touched (D3 / D4 hold): its pre-armed accounting (wait k
    <-> release k-1) cannot carry an item-end credit -- the first build tried it and round 2 overwrote round 1 under the reduce reads.  Every
    other row unchanged (dvk_empty / k2_empty still after the final drain = round 2's).  PTX census: arrive sites 41 / wait sites 69 under
    fp32_reduce (40 / 68 under the default); reduce-adds 4 -> 12, bf16 tile stores 4 -> 0; the default rendering has no instruction on the row.
    Lane ledger: every row above written as SUM == init on BOTH CTAs (the inspector's section 2, PASS 0); unresolved: none.  Non-mbarrier
    sites: named barrier 0 (3 x barrier_cta_sync: init, TMEM-base publish, final), CFG.SOFTMAX_BARRIER (256 lanes, 2 per _stats),
    CFG.XCHG_BARRIER (128 lanes, 1 per tile, P11 KEEP: the ship's operand is all four warps' XCHG stores).
    X-FA4 (``warp_roles = "16w_fa4"``; barrier-inspector SKELETON resources/x-fa4-barrier-table.md 2026-10-27 08:45 UTC, sections 0-14, COPIED;
    ``config_sm100_d128_accum_2cta_stail.BAR_LAYOUT_XFA4`` = 32 names / 37 slots / END 230912, re-packed; its rows = the 34 above MINUS:
      ds_tmem[1]   DEL  (THREAD 256: producer and consumer both vanish -- the softmax writes sDS from registers, nobody reads dS^T out of TMEM)
      dp_empty[1]  DEL  (LEADER 256: its ONLY arrive site was the epilogue's dS^T readout; dP^T(t+1) over the packed dS^T(t) [128,192) is ordered by the
                        in-order tensor pipe after dK(t) (issued first by the same thread); over dP^T(t)'s fp32 [128,256) by the softmax's ld +
                        wait(LOAD) -> STTM -> wait(STORE) -> fence -> ds_ready(t) (512) -> the leader's dK(t) wait -> dK(t) -> dP^T(t+1): both edges
                        hold in-tile, across the item boundary, around a zero-trip item and for the kernel's first dP^T (inspector section 6))
    and with these rows CHANGED (every other row: the same object, count, kind, waiter and cadence; the epilogue's rows RENAMED to the reduce group):
    name[st]         kind              init L / F     SUM(issuing lanes) == init        waiter (own copy unless "L only")   start  fires
    ds_smem[1] CHG   THREAD            256 / 256      softmax 8 warps bare after the 4 own-half + 4 XCHG STS + ONE fence.proxy.async (no tcgen05 fence:
                                                      the words come from registers) = 8 x 32 = 256 == SOFTMAX_LANES; BEFORE the XCHG bar.sync
            MMA warp (own): L dQ issue on xchg_state (in-tile + tail), F relay (relay_state)   st(0) x N   (the epilogue's 128-lane site DELETED)
    ds_recv_full[1] CHG TMA_LOAD (dsmem) 1 / 1        softmax warp SOFTMAX_LEAD_WARP (0), `arrive(n_bytes=XCHG_TX, pred=ship_pred)`, ship_pred =
                                                      (warp == 0) & elect_sync() ONCE per item = 1 == 1; AFTER the 256-lane XCHG bar.sync, IMMEDIATELY
                                                      before `_ship_xchg` (form B); tx = the PEER softmax's cp.async.bulk of ITS XCHG = 16384 == XCHG_TX
            MMA warp (own): L on xchg_state, F relay  st(0) x N.  The arm's BACK-EDGE on every path (the 08:01 rule) = the softmax's dq_done(t-1) WAR
            wait: dq_done(t-1) <- dQ(t-1) issued by the leader after ds_recv_full(t-1) [copy(t-1) landed on CTA 0] AND ds_pair_ready(t-1) <- the relay's
            ds_recv_full(t-1) wait [copy(t-1) landed on CTA 1]; item top: the tail dq_done(hi-1) of item k; after a zero-trip item: the empty commit,
            issued by the leader AFTER the previous live tail in program order.  ds_gates_off (D8) would delete the edge on both CTAs: declined BY NAME.
    dq_done[1]       MMA_COMMIT        1 / 1          UNCHANGED from the row above (inspector section 15: the softmax waiter set of the first build is
                                                      REMOVED): producer dQ(t-1) at slot 2 / the tail dQ(hi-1) / the zero-trip empty commit; waiter the
                                                      REDUCE group `dq_state` st(0) (the readout + the zero-trip observation) ONLY   N (+1) / N (+1)
    dq_war_done[1] NEW MMA_COMMIT      1 / 1          MMA(L) `_commit_pair(bars.dq_war_done, elect_p)` INSIDE _dq_issue at the TWO dQ issue sites ONLY
                                                      (in-tile dQ(t-1), the tail dQ(hi-1)), IMMEDIATELY BEFORE the dq_done commit; NEVER at the zero-trip
                                                      empty-commit site = 1 arrive per target CTA.  Waiter: the SOFTMAX ONLY, `dq_war` st(1) PRE-ARMED,
                                                      +1 per wait, `_wait` (plain) before the first dS STS of every LIVE tile (after the STTM + wait(STORE)
                                                      + fence + ds_ready: dK(t) never waits dQ(t-1)); NO zero-trip wait, NO zero-trip advance; ONE
                                                      unconditional drain `_wait` after the persistent loop.  Flips per item: live N / N (wait k of the item
                                                      consumes flip k-1: the first consumes the previous live item's tail flip or the pre-arm); zero-trip
                                                      0 / 0 (no site executes); n_q == 1: 1 / 1; kernel: W flips, W + 1 waits (the drain consumes the last
                                                      tail; W = 0 passes on the pre-arm).  LAP BOUND: flip w <- dQ(w) <- ds_smem <- the softmax's STS after
                                                      wait w; wait w-1 returned only after flip w-2 landed -> F in {w-1, w} at EVERY wait: the pre-armed
                                                      waiter can never find a flip landed early (the first build's shared dq_done waiter raced the
                                                      leader's zero-trip EMPTY commit: a zero-trip item as a cluster's first item HUNG at 10:24, RULE-GAPS
                                                      649; consecutive zero-trips = a lap).  The arm's back-edge (section 5.3) on every path = a DATA edge
                                                      through the previous live tile's / item's dq_war_done flip = a real dQ issued behind the leader's gates.
    s_tail_empty[1] RENAMED  LEADER (relaxed) 256 / 256(-)  the REDUCE group (the same lanes) after the x64 dQ ld + wait(LOAD) + fence, BEFORE round A;
                                                      waits / D1 sites / drain unchanged
    dq_empty[2] / dq_full[2] RENAMED  THREAD           32 / 32 per slot ; 64 / 64 per slot  the REDUCE group waits BOTH slots per round (D4 pair) / arrives
                                                      dq_full[dhalf] per round; rounds A then B inside the tile (no carried half): 2N / 2N per slot
    dv_full[1] / dk_full[1] RENAMED  THREAD            160 / 160 ; 128 / 128  the REDUCE group's item-end native staging (X9) + the loader's 32 (fix A)
    named XCHG_BARRIER (id 2) CHG                      bar.sync 2, 256 = CFG.xchg_barrier_lanes("16w_fa4") = SOFTMAX_LANES, ONE per tile by all 8 softmax
                                                      warps after the 8 STS + the proxy fence + the ds_smem arrive; P11 KEEP (the ship's operand is all
                                                      8 warps' XCHG stores); the inspector REJECTED folding it into _stats' SOFTMAX_BARRIER (X3)
    s_read[1]        KEPT exactly as above (X1; the merge into a LEADER_RELEASE deferred s_read is a STAMPS-decided follow-up, inspector section 10)
    dQ ring          FORM A kept (X6): the store warp issues the UTMAREDG; the ring rows unchanged in count, kind, waiter and the D4 pair
    BAR_KIND_COUNTS_XFA4 {tma_leader 7, mma 10, thread 7, leader 5, peer 2, tma_dsmem 1, tma 1} = 33 (BAR_LAYOUT_XFA4: 33 names / 38 slots / END
      230928, dq_war_done appended after tmem_dealloc).  Lane ledger (inspector sections 2 + 15.3, PASS 0): ds_smem 256 == 256 (8 x 32 bare, every
      tile) | ds_recv_full 1 == 1 (warp 0 elect, tx 16384 from the peer) | dq_done 1 == 1 per CTA (the reduce group) | dq_war_done 1 == 1 per CTA (the
      softmax) | s_tail_empty 128 x 2 == 256 | dq_empty 32 == 32 | dq_full 64 == 64 | dv_full 128 + 32 == 160 | dk_full 128 == 128 | s_read 512 == 512 |
      ds_pair_ready 1 == 1 | XCHG id 2: 256 threads == 256 | SOFTMAX id 1: 256 == 256; DELETED ds_tmem / dp_empty: no init, arrive, wait, drain, state.
    PER-ITEM FLIPS under 16w_fa4 (both CTAs): ds_smem N / N; ds_recv_full N / N; ds_pair_ready N / N; dq_done N (+1) / N (+1) (the reduce group);
      dq_war_done N / N (the softmax; + the ONE kernel-end drain); s_read N / N; s_tail_empty N / N; dq rings 2N / 2N per slot; p_ready / ds_ready /
      s_done / dp_done / do* / q2 / q1 N / N; k1 / v / k2 / dvk_empty / k2_empty / dv_full / dk_full / sched 1 / 1; named XCHG N; SOFTMAX 2N.  Zero-trip
      item (any position: first, consecutive, after live, THD empty): dq_done 1 / 1 (the reduce group), dq_war_done 0 / 0, dvk_empty / k2_empty /
      dv_full / dk_full / sched 1 / 1, everything else 0.  N == 1: ds_smem 1 / 1, dq_done 1 / 1, dq_war_done 1 (the tail) / 1 (tile lo consumes the
      previous tail or the pre-arm; its own tail flip goes to the next live item's first wait or the drain), s_tail_empty 1 / 1 (prologue), xchg gates
      1 / 1 (tail).  Half-dead pair: CTA 1's softmax STS zeros, ships zeros, arrives everything: identical counts.
    FENCE TABLE under 16w_fa4 (inspector section 7): F1 softmax after the 8 STS: ONE fence.proxy.async (generic STS -> the bulk copy's read of XCHG +
      the UMMA's read of sDS own) -> ds_smem.arrive -> bar.sync 2,256 -> arm -> ship [replaces the epilogue's two fences + its tcgen05 fence]; F2 / F3
      the S^T ld pair / the STTMs: tcgen05_wait + tcgen05_fence(BEFORE) before s_read / p_ready / ds_ready (unchanged); F4 the dq_war wait: plain
      `_wait` (a WAR of an async read by a generic write: the commit -> mbarrier acquire IS the order; no proxy fence exists or is needed); F5 the reduce
      group's x64 ld: wait(LOAD) + fence before s_tail_empty (unchanged); F6 each round's slice STS: fence.proxy.async before dq_full (unchanged); F7
      _native_out (unchanged); F8 the ship: none after (async -> async); F9 the store warp's wait_group.read before every credit (unchanged); F10 init.
    PER-ROLE STUB ORDER under 16w_fa4 (inspector sections 9 + 15.4 = the `# BARRIER` punch lists of _softmax_loop / _reduce_loop / _mma_loop_leader
      below): softmax per live tile: ... ds_ready -> `_wait(dq_war_done, dq_war)` -> advance -> 8 STS -> ONE fence -> ds_smem -> bar.sync 2,256 -> arm ->
      ship; nothing on dq_war at a zero-trip item; the ONE drain `_wait(dq_war_done)` after the loop.  MMA(L) `_dq_issue`: gates -> dQ MMA ->
      `_commit_pair(dq_war_done)` -> `_commit_pair(dq_done)` -> xchg += 1; the zero-trip site: `_commit_pair(dq_done)` only.
    PER-ITEM FLIPS (N = hi - lo; every path flips every ring the same number of times on BOTH CTAs: lo / hi are pair-uniform):
      s_tail_empty N fires (one readout per dQ issue) / N waits [prologue 1 + lookaheads with qtile > lo (N-2 for N >= 2) + the tail iff
      N >= 2 (1)]; s_read N / N; dp_empty N / N; the dS gates N each / N (N-1 in-tile + the tail); dq_done N (+1 zero-trip) / N (+1);
      q1 N / N; dq_empty[s] / dq_full[s] 2N / 2N per slot; p_ready, ds_ready, ds_tmem, s_done, dp_done, the do* / q2 rings N / N;
      dvk_empty, k2_empty, dv_full, dk_full, sched 1 / 1 per item.  Zero-trip item: 0 everywhere but dq_done / dvk_empty / k2_empty /
      dv_full / dk_full / sched 1 / 1.
    THE s_tail_empty WALK (the only row whose wait sites carry runtime guards; the inspector's section 4):
      N == 1: S^T(lo) waits [prev tail / pre-armed]; tile lo: no dQ, no lookahead; tail dQ(lo): hi - 1 > lo FALSE -> no wait (the
        write before it in [64, 128) is S^T(lo); the softmax's read of S^T(lo) is implied by the dS gates (lo) on both CTAs); the
        epilogue's dQ(lo) readout fires 1.  1 == 1.
      N == 2: S^T(lo) [1]; tile lo nothing; tile lo+1: dQ(lo) at slot 2, no S^T(lo+2); tail dQ(lo+1) waits dQ(lo)'s readout [1]; fires 2.
      N == 3: S^T(lo) [1]; tile lo nothing; tile lo+1: dQ(lo), then S^T(lo+2) waits dQ(lo)'s readout [1]; tile lo+2: dQ(lo+1), no
        S^T(lo+3); tail dQ(lo+2) waits dQ(lo+1)'s readout [1]; fires 3.
      Table 3's "before EVERY S^T" form would park tile lo's lookahead on the readout of a dQ not yet issued = HANG on item 1 (D1).
    DELAYED OBSERVER (ABA, the inspector's section 5): PASS for every pair-scope object; the LOCAL dS gates' and ds_pair_ready's back-edge
      rests on the epilogue's tile-end dq_done(t) wait (the round-A placement): a lever that moves the dQ readout out of the epilogue
      re-audits ds_pair_ready / ds_smem / ds_recv_full.  Under 16w_fa4 that back-edge is the SOFTMAX's dq_war wait (the X-FA4 block above;
      RULE-GAPS 314 closed for this body by the inspector's section 5.2): the reduce group's readout wait is no longer load-bearing for the gates.
    MMA ISSUE ORDER (leader warp, per q tile t; FAv4's loop rotated onto our tile body; the typed order field accepts `fa4` ONLY on this map):
      slot | GEMM              | waits (state)                                                   | commits                | why here
      pro  | S^T(lo)           | k1_full, q1_full (q1), s_tail_empty (s_tail: the previous item's tail readout) | s_done, q1_empty | hidden behind K1 / V / K2
       1   | dP^T(t)           | dp_empty(t-1) (dpe: the dS^T(t-1) readout), do1_full x 2 (do1)  | dp_done, do1_empty     | ~+1300 clk slack: ds_tmem(t-1) + ld
       2   | dQ(t-1) if t > lo | s_read(t) (ready) [every tile]; ds_smem, ds_recv_full, ds_pair_ready (xchg) | dq_done | the exchange gets dV + S^T +
           |                   |                                                                  |         | dK + dP^T of cover (~1350 clk)
       3   | dV(t)             | p_ready(t) (ready), do2_full (do2)                               | do2_empty              | unchanged
       4   | S^T(t+1) if t+1<hi| q1_full (q1); s_tail_empty iff t > lo (s_tail: dQ(t-1)'s readout) | s_done, q1_empty | the TIGHTEST edge (~+300 clk:
           |                   |                                                                  |                  | dq_done(t-1) + ld vs dV(t))
       5   | dK(t)             | ds_ready(t) (ready), q2_full (q2)                                | q2_empty               | unchanged
      tail | dQ(hi-1)          | s_tail_empty iff hi - 1 > lo; the three dS gates (hi-1) (xchg)    | dq_done; then k1_empty, v_empty | once per item
      What it attacks (profiler-ncu 23:40 UTC @ 16K on the first body): the MMA leader's 43 % at the dS gates and the softmax's 30 % dP wait
      were ONE lock-step loop (dQ(t) -> readout -> dp_empty -> dP^T(t+1) -> softmax -> exchange -> dQ(t+1)); here dP^T(t+1) no longer sits
      behind dQ(t)'s exchange chain and dQ(t) is issued one GEMM later with nothing queued behind it.  The gates removed from the critical
      path: [0x38550] + [0x38560] (ds_smem own + peer) and [0x384f0] (softmax dP).  MEASURED verdict = the benchmarker's A/B/A (section 13).
    MMA ISSUE ORDER fa4b (code-writer step 6, 2026-10-27: the typed `pair_mma_order = "fa4b"` -> `dq_after_dv` Constexpr[bool]; the default
      stays fa4 (= False) until MEASURED; the SAME barrier table, the SAME sites: every wait keeps its object, its guard and its count -- the
      move changes WHEN inside tile t the leader consumes s_read(t) and the three dS gates, never WHETHER (P14)):
      slot | GEMM              | waits (state)                                                   | commits            | edge cover vs fa4
      pro  | S^T(lo)           | unchanged                                                       | unchanged          | =
       1   | dP^T(t)           | dp_empty(t-1) (dpe), do1_full x 2 (do1)                         | dp_done, do1_empty | =
       2   | dV(t)             | p_ready(t) (ready), do2_full (do2)                               | do2_empty          | issued BEFORE the exchange gates:
           |                   |                                                                  |                    | never queued behind them
       3   | dQ(t-1) if t > lo | s_read(t) (ready) [EVERY tile; DOMINATED by p_ready(t): the same   | dq_done            | the exchange (t-1) gets dV(t) =
           |                   | 16 softmax warps arrive s_read before p_ready in program order,   |                    | +512 clk of cover at zero SMEM
           |                   | so the wait passes at once and still consumes its phase]; then   |                    | (profiler-ncu 04:30 @ 16K: the leader
           |                   | ds_smem, ds_recv_full, ds_pair_ready (xchg)                      |                    | at ds_recv_full 23.6 % + ds_smem 5.6 %)
       4   | S^T(t+1) if t+1<hi| q1_full (q1); s_tail_empty iff t > lo (s_tail: dQ(t-1)'s readout) | s_done, q1_empty   | -512 clk: NOTHING sits between dQ(t-1)
           |                   |                                                                  |                    | and S^T(t+1); the readout latency
           |                   |                                                                  |                    | (dq_done + 64-col ld + arrive) reaches
           |                   |                                                                  |                    | the pipe unless the queue is deep (fa4
           |                   |                                                                  |                    | hid it behind dV(t): 5.4 % of the
           |                   |                                                                  |                    | leader's samples)
       5   | dK(t)             | ds_ready(t) (ready), q2_full (q2)                                | q2_empty           | =
      tail | dQ(hi-1)          | unchanged: s_tail_empty iff hi - 1 > lo; the three gates (xchg)  | unchanged          | =
      TMEM under fa4b: dV(t) reads P^T(t) at [P_COL, +PACK_COLS) = [0, 64) (its A operand) and writes [DV_COL, +DV_COLS); dQ(t-1) writes
      [DQ_COL, +DQ_COLS) = [64, 128): DISJOINT, and the tensor pipe is in-order, so S^T(t+1)'s [S_COL, +S_COLS) write still follows both.
      SMEM under fa4b: dV reads sDO2 (do2_full), dQ reads sDS (the three gates) + sK2 (k2_full): disjoint, each behind its own barrier.
      P14 under fa4b: per-item flips, the N = 1 / 2 / 3 walk (waits 1 / 2 / 3 == readouts; N == 1's tail dQ(lo) under the s_read(lo) wait, now
      consumed after dV(lo)) and the P15 drains are IDENTICAL to fa4; the first tile of every live item consumes s_read(lo) after dV(lo) and
      issues no dQ, the same as fa4.  ABA under fa4b: s_read -- c+1 <- S^T(t+1) at slot 4, after the s_read(t) wait at slot 3 (in-order issue);
      s_tail_empty -- c+1 <- dQ(x+1), issued at tile x+2 slot 3, after the lookahead's wait at tile x+1 slot 4 consumed c (or the tail / the next
      prologue consumed it); the dS gates / ds_pair_ready / ds_recv_full -- the back-edge is unchanged (the epilogue's tile-t dq_done(t) wait).
      Expected net (INFERRED, to be MEASURED): the exchange stall shrinks by up to 512 clk per tile, the s_tail_empty edge grows by the readout
      latency; a loss on this edge names the next variant (dK(t) before S^T(t+1) = FAv4's own placement, at the softmax's cost).  Code: ONE
      helper `_dq_slot` (the s_read wait + the guarded `_dq_issue`) at TWO `const_expr(dq_after_dv)` sites of `_mma_loop_leader`; the prologue,
      the tail and every other role are shared by both orders.
    PER-ROLE PHASE TABLE (ONE PipelineState per consumption cadence, carried across CLC items, advanced on EVERY path -- P2 / P14):
      role     | state (start)                 | advance                        | feeds
      MMA(L)   | q1 st(0), 1 stage             | +1 per tile (q1_next = the lookahead's slot; the skipped last lookahead pre-positions the next
               |                               |   item's prologue, N per item == N S^T issues) | q1_full
               | q2 / do1 / do2 st(0)          | +1 per tile                    | q2_full, do1_full[0..1] (ONE do1_state), do2_full
               | kv st(0)                      | +1 per live block              | k1_full, v_full, k2_full
               | ready st(0)                   | +1 per tile                    | p_ready, ds_ready, s_read (all indexed by tile t)
               | xchg st(0) NEW (the P2 split) | +1 per dQ ISSUE (slot 2 of tiles lo+1 .. hi-1, the tail) | ds_smem, ds_recv_full, ds_pair_ready
               | s_tail st(1) NEW              | +1 per s_tail_empty wait (D1's three sites) | s_tail_empty
               | dpe st(1) / out_empty st(1)   | +1 per tile / +1 per item      | dp_empty / dvk_empty
      MMA(F)   | relay st(0)                   | +1 per tile                    | ds_smem, ds_recv_full (own copies) -> ds_pair_ready arrive_on_peer
      softmax  | tile st(0)                    | +1 per tile                    | s_done, dp_done (arrives: s_read, p_ready, ds_ready, ds_tmem)
      epilogue | ds st(0) / dq st(0)           | +1 per tile / +1 per tile, +1 per zero-trip item | ds_tmem / dq_done
               | empty st(1)                   | +1 per ROUND (both slots waited at one phase: the inspector's 2-stage "+2 per round" collapsed to
               |                               |   its one flip per round)      | dq_empty[0], dq_empty[1]
      store    | dq st(0), 2 stages            | +1 per USE (4 per tile)        | dq_full[slot]
               | out st(0)                     | +1 per item                    | dv_full, dk_full
      loader   | q1 st(1), 1 stage; q2 / do1 / do2 st(1); kv st(1); k2e st(1) | +1 per tile / per live block / per item | q1_empty .. k2_empty
      scheduler(L) | sched st(0), 2 stages     | library loop                   | sched_empty, sched_full
    P14 paths: zero-trip item (no tile: no S^T, no dQ, the empty dq_done commit; no dq_empty traffic: 0 / 0); half-dead pair (CTA 1 runs
      every tile with P = 0 exactly: reads S^T (s_read), reads its 64 dQ rows out (s_tail_empty), ships zeros, relays -- identical flip
      counts); n_q == 1 (lo + 1 == hi: the lookahead is skipped, no in-tile dQ, the tail dQ(lo) without an s_tail_empty wait; the
      prologue's S^T(lo) is the item's only S^T); the first tile of every live item: dP^T(lo) [dp_empty: the previous item's last dS^T
      readout / pre-armed]; s_read(lo); NO dQ; dV(lo); S^T(lo+1) with q1_full only; dK(lo).

6. INIT + PHASE DISCIPLINE + EXIT DRAINS  (P4 / P5 / P6 / P14 / P15; code: the kernel entry + the helpers above the role bodies)
    init: warp 0, `if elect_sync():` every stage of every Bars field (ONE lane, both gates) -> fence_mbarrier_init -> barrier_cta_sync
          -> cga_arrive / cga_wait (before ANY cross-CTA arrive) -> MMA warp: tmem_alloc(TMEM_COLS, cta_group::2) on BOTH CTAs
          -> barrier_cta_sync (publishes the TMEM base) -> role dispatch.  NO bootstrap arrive anywhere (D3: the store warp's dq_empty
          bootstrap of the first body left with the staging alias); every MMA-fed or pair-fed consumer starts pre-armed where the table says
          st(1): the leader's dp_empty / s_tail_empty / dvk_empty states, the epilogue's dq_empty state, the loader's *_empty states.
          16w_fa4: the SAME block over the 33 fields of BAR_LAYOUT_XFA4 (`_bars(slab, fa4_roles)` builds `BarsXFA4`: ds_tmem / dp_empty absent,
          an AttributeError if touched, never a silent dead row; dq_war_done present); pre-armed st(1) ALSO the softmax's NEW `dq_war`; the leader's
          dp_empty state is gone; dispatch warp < 8 softmax | < 12 REDUCE (`_reduce_loop`) | 12 MMA | 13 | 14 | 15; P15 drains MMA(L): dvk_empty x1,
          s_tail_empty x1 (dp_empty's gone); SOFTMAX: the ONE unconditional `dq_war_done` drain after the loop (section 15.3: the leader's last tail
          commit is issued before it leaves its loop and depends on nothing of the softmax: the drain cannot hang; a CTA with no live item passes on
          the pre-arm).  Everything else of this section (seams, zero-trip, drains) holds unchanged; the ds_own / XCHG seams now end at the
          SOFTMAX's dS STS, which waits dq_war_done(t-1) directly (the WAR) and dp_done(t) (dvk_empty's dK-staging seam, RULE-GAPS 547).
    phase trackers: section 5's PER-ROLE PHASE TABLE (never bit math -- P2 / P14).
    zero-trip item (lo >= hi, PAIR-UNIFORM from _item_bounds; a padded CTA inside a live pair runs every tile with P = 0): MMA(L) one
      empty dq_done commit (live pipe, multicast); both epilogues wait it (dq_state +1); the store warps run the native zero stores,
      arrive dvk_empty (-> L) + k2_empty; the loaders wait k2_empty + arrive dv_full; softmax / MMA(F) only credit the scheduler.
    P15 drains (OUTSIDE the persistent loop, const_expr(CFG.CTA_MMA == 2), whole ring from the carried state):
      loader L+F: q1_empty x Q1_STAGES (= 1), q2_empty, do1_empty, do2_empty, k1_empty, v_empty (the leader's last multicast commits in flight)
      MMA(L): dp_empty x1, dvk_empty x1, s_tail_empty x1 (the follower's last relaxed arrives in flight: the last item's dS^T readout,
      the last dV / dK drain, the last tail readout); tmem_dealloc: the wait is the drain.
      Observed in-loop on every phase, no drain: s_done / dp_done / dq_done / sched_full, p_ready / ds_ready / s_read (every tile),
      ds_pair_ready / ds_recv_full / ds_smem (every dQ issue, in-tile and tail), dq_empty / dq_full (every round), sched_empty.
    SMEM LIFETIME SEAMS (Rule S3: an explicit barrier chain per reuse, never an issue-order invariant; the inspector's section 6):
      sDQ slot s (DEDICATED): written by the epilogue warp pair s per round after dq_empty[0] + dq_empty[1] (the round's phase), read by the
        store's reduce-add of slot s, released by dq_empty[s] after `tma_store_wait(1)` (slot 0) / `(0)` (slot 1); two cycles per tile; no
        other tenant: the dS / XCHG STS never waits dq_empty.
      ds_own(rank) (the dS^T(t+1) STS) <- the dQ(t) MMA's A read (the previous tenant) -> dq_done(t), waited by the SAME epilogue at the end
        of its tile t (the dQ(t) readout) before it begins tile t+1: program order after a barrier wait.  The gate is the MMA's COMPLETION,
        never the reduce-add drain ("the exchange is not gated").
      XCHG (the dS^T(t+1) peer-half STS) <- our ship(t)'s cp.async.bulk source read: completes before the PEER's ds_recv_full(t) -> (L's XCHG)
        the follower relay -> ds_pair_ready(t) -> dQ(t) -> dq_done(t) / (F's XCHG) the leader's own ds_recv_full(t) wait -> dQ(t) -> dq_done(t)
        -> our epilogue's dq_done(t) wait at its tile-t end.  The first body's dq_empty leg is gone.
      ds_recv(rank) <- ONLY the peer's ship(t+1), which follows the peer epilogue's dq_done(t) wait; dQ(t) read the recv half before dq_done(t).
      dK native staging over ds_own(rank) + XCHG (item end) <- dS(hi-1) own half: the dQ(hi-1) MMA -> dq_done(hi-1) waited by the epilogue (its
        last tile's readout); XCHG <- the ship(hi-1) read <- dq_done(hi-1) via the chains above.  Next tenant: the dS(lo') STS of item k+1 <-
        ds_tmem(lo') <- dp_done(lo') <- dP^T(lo') issued after the MMA's item-top dvk_empty(k) wait <- both store warps' tma_store_wait(0)
        (the dK store's source read).  The first body's second gate (the dq_empty bootstrap before the dS STS) is gone: the dvk_empty chain
        carries the seam alone (ALIASES row dk_store).
      dV native staging over sK2 (item end) <- the dQ(hi-1) MMA's B read -> dq_done(hi-1) waited by the epilogue; the previous item's dV store
        drain -> dvk_empty -> MMA -> dq_done (live) / the empty commit (zero-trip); the K2 reload <- k2_empty (store after tma_store_wait(0)),
        fix A back-edge intact.
      sQ1 (1 stage) <- the S^T(t) MMA's B read -> q1_empty commit after S^T(t) -> q1_full(t+1) before S^T(t+1); window ~2000 clk.
      sQ2 / sDO1 / sDO2 / sK1 / sV: unchanged rings, roots shifted -16384.
    THE ~150-clk CLAIM (resource-manager; why the staging alias is deleted): at the 512-clk-per-GEMM ideal, dq_done(t) lands at ~T + 3584 and
      ds_tmem(t+1) at ~T + 3300..3500, so the epilogue's dS(t+1) STS starts after its dq_done(t) wait + readout (~T + 3800), 300..500 clk after
      the exchange could start; the sDQ round-B drain (A(t)'s reads, ~800..1050 clk after round A) completes by ~T + 4850, before the
      dq_done(t+1) wait at ~T + 6144: the reduce-add drain gates nothing on the ideal timeline, ~1200 clk of margin at the measured duty.

7. THE ONE HELPER
    _item_bounds(pair, meta, batch, n_batch, causal) -> (q_lo, q_hi, seq_q, seq_k, q_start): CLUSTER-UNION bounds
    (the pair walks the identical q range; causal trim per 256-row PAIR: lo = pair * Q_TILES_PER_PAIR; CTA 1's diagonal tile is
    fully MASKED, never skipped; a dead pair (256 pair >= seq_k) is lo = hi = 0 on both CTAs).  Every role on both CTAs derives
    its loop bounds from this pure helper over the SAME inputs -- the pair-uniform `qtile > lo` / `hi - 1 > lo` guards of the
    s_tail_empty sites and the dQ issue sites read the same (lo, hi) on the leader as the softmax / epilogue do on both CTAs.
    tile_may_mask(pair, rank, qtile, seq_q, seq_k, causal) -> Boolean (lever L5, section 11; plain Python over the traced scalars = the
    SAME inputs _item_bounds reads + the CTA rank; the test module runs the identical function on host ints against the per-cell mask):
    TRUE iff a cell of THIS CTA's [128 kv x 128 q] tile can be masked = ((qtile + 1) * N > seq_q) | (kv_end > seq_k) | (causal & (qtile * N <
    kv_end)) with kv_end = pair * PAIR_KV_ROWS + (rank + 1) * M (the CTA's exclusive last kv row + 1).  CTA-uniform by construction (no lane
    term), so the `if mask_tile:` branch of _p_values never diverges.  EXACT, not merely conservative: _p_values keeps column c of the lane with
    key k iff c >= k (causal) and c < (seq_q if k < seq_k else 0), monotone in c on both sides, so a masked cell exists iff the tile's FIRST
    column fails the causal edge for the CTA's largest key (qtile * N < kv_end - 1, i.e. qtile * N < kv_end as both are multiples of N = M) or
    its LAST column fails the hi edge for some lane ((qtile + 1) * N - 1 >= seq_q, or some key >= seq_k, i.e. kv_end > seq_k).  Under causal the
    pair's live tiles start at lo = 2 pair, so CTA 0 runs the mask body on tile 2 pair only and CTA 1 on 2 pair (fully masked) and 2 pair + 1
    (its diagonal): ~2 of 64.5 tiles per item at 16K; every other tile of a full-length sequence is interior.

8. SMEM TABLE  (REQUIRED; per CTA, DECLARATION order; vs-smem predict PASS (resources/vs-smem-predict-arm2.log): total 231424 B of cap
   232448 B (cc 10.0, 227 KiB opt-in), slack 1024 B (scaffold); banks PASS 0 / 7; DESC_VERSION implied 0, the baseline's 1 inherited)
    name[stages]   | bytes  | 1024-al | desc | swizzle + WHY                                  | lifetime / alias   | offset
    sQ1[1]         | 16384  | 16384   | yes  | SW128 K-major: MATCH the S^T B descriptor       | tile; q1 ring (ONE stage, FAv4 Q_stage 1) | 0
    sQ2[1]         | 16384  | 16384   | yes  | SW128 MN-major (64-d rows = one atom): MATCH dK B | tile; freed by dK | 16384
    sDO1[1]        | 16384  | 16384   | yes  | SW128 K-major: MATCH the dP^T B descriptor      | tile; freed by dP^T| 32768
    sDO2[1]        | 16384  | 16384   | yes  | SW128 MN-major: MATCH the dV B descriptor       | tile; freed by dV  | 49152
    sK1[1]         | 32768  | 32768   | yes  | SW128 K-major: MATCH the S^T A descriptor       | kv block           | 65536
    sK2[1]         | 32768  | 32768   | yes  | SW128 MN-major, 256 rows (two 128-row boxes):   | kv block; dV store | 98304
                   |        |         |      | MATCH the dQ B desc                             | staging at the end |
    sV[1]          | 32768  | 32768   | yes  | SW128 K-major: MATCH the dP^T A descriptor      | kv block           | 131072
    sDS[1]         | 32768  | 32768   | yes  | SW128 MN-major (q contiguous, 128 B per kv row, | own half: STS ..   | 163840
                   |        |         |      | kv-global row order): MATCH the dQ A desc        | dQ MMA; recv half: |
                   |        |         |      | (atranspose); the 16-B `_store_words` form        | the PEER's timeline|
    sDS_xchg[1]    | 16384  | 16384   | no   | SW128 image of the peer's recv half (byte copy) | STS .. bulk copy   | 196608
    sDQ[2] NEW     |  8192  | 16384   | no   | fp32 [64 q x 32 d] SW128 (lane stride 128 B:     | DEDICATED ring: epilogue | 212992
                   | x 2    |         |      | SPREAD + MATCH the reduce-add box); slot s = d   | STS per round -> store   |
                   |        |         |      | half s's warp pair                               | reduce-add -> dq_empty   |
    sSTATS (LSE)   | 512    | 512     | no   | none: 4-B lane stride                            | tile               | 229376
    sDELTA         | 512    | 512     | no   | none                                             | tile               | 229888
    scaffold       | 1024   |         |      | bars [230400, 230944) 544 B + 400 spare [230944, 231344) = the STAMPS record (config DBG_CLK: 100 x Int32,
                   |        |         |      | written only under `debug_clk`) + TMEM handle 16 @ 231344 + CLC payload 64 @ 231360 | 230400
    aliases (3; config ALIASES): dK native store (2 x 16 KiB) over ds_own(rank) + XCHG at the item end (seam: dq_done(t_last) + dvk_empty);
    dV native store (2 x 16 KiB) over sK2 (seam: dq_done(t_last) + k2_empty); ds_recv(rank) = the peer's half (the peer's timeline).
    dkv_accum fp32_reduce (form (i) of resource-manager 08:25 table 9): round 1 = dV's four fp32 [128 kv x 32 d] slices + dK's slice 0 over
    CFG.dkv_acc_r1_slot(0..4) = sK2's two halves, the dS slab's two halves (own + recv: dead from dq_done(hi-1) -- the recv bytes landed before
    it -- until the next item's dS writes, which the dvk_empty chain gates), XCHG (80 KiB); round 2 = dK's slices 1..3 over
    CFG.dkv_acc_r2_slot(0..2) = sK2's halves + the dS slab's first half (48 KiB) after dvk_stage_empty (the one new seam: a WAR edge inside
    the item end); the entry seams (dq_done(t_last), the ship(hi-1) read) and the exit seams (dvk_empty, k2_empty after the final drain) are
    unchanged; the sDQ ring is never a staging slot.
    The first body's dq_staging alias (4 x 8 KiB over ds_own + XCHG) is DELETED: under the rotated order the alias is issue-order sensitive
    (dq_done(t) -> dS(t+1) STS gap ~1100 clk under s_first -> ~150 clk here) and a drain-gated dS write would re-create the lock-step.
    What pays for the ring: sQ1 2 -> 1 stage (-16384) = +sDQ 16384; every root from sQ2 on shifts -16384; total unchanged.
    Under dq_route_pair red_global / red_global_pair the sDQ ring is NEVER WRITTEN (its 16 KiB stay allocated: the Medium form of the arm
    deletes the ring and its rows; this Lightweight form keeps the layout byte-identical) -- the SMEM data pipe loses the 256 STS + 256
    UTMAREDG-read wavefronts per tile per CTA (resource-manager 04:52 table 2 rows 4 / 7; RULE-GAPS 395 decides whether the LSU global path
    shares the port).
    proxy-fence sites: after every lane-written slab the async proxy reads (sDS own half, XCHG before the bulk copy, each sDQ slot before the
    reduce-add, dV / dK staging before the TMA store) -- the inspector's rows.
    16w_fa4: the SMEM table is UNCHANGED (every slab, root, alias; resources/vs-smem-predict-xfa4.log PASS 231424 / 232448; the softmax's STS image =
    the same bytes at the same offsets: resources/vs-smem-banks-xfa4.log PASS 0 / 7, 4 wavefronts per STS.128 = the floor, resource-manager table 2);
    only the bar block's used span shrinks to [230400, 230912).

9. TMEM MAP (fp32 columns; one tcgen05.alloc.cta_group::2 of TMEM_COLS = 512 on BOTH CTAs, the same constant deallocates)
    S^T    @   0 + 128  | P^T packed bf16 in place @ 0 + 64 (softmax group g packs into 32 g .. +32); dQ (M 128 half-row image) @ 64 + 64 =
                          S^T's dead tail (CFG.DQ_COL = S_COL + PACK_COLS); two alias edges: WRITE dQ(t-1) <- the softmax's S^T(t) read (s_read),
                          REUSE S^T(t+1) <- the epilogue's dQ(t-1) readout (s_tail_empty)
    dP^T   @ 128 + 128  | dS^T packed @ 128 + 64; [192, 256) DEAD (was the first body's dQ image); dp_empty covers the dS^T readout only
    dV     @ 256 + 128  | dK @ 384 + 128   -> used 512 / 512.  No TMEM slot is assumed zero.
    RESOURCE-NOTE: TMEM 512/512 unchanged; 192..256 DEAD (was dQ); one alloc/dealloc constant
    16w_fa4: the TMEM map is UNCHANGED; dS^T packed @ 128 + 64 KEEPS its STTM (dK's TS A operand); the epilogue's 4 x LDTM.x16 of it are GONE
    (the softmax stores the same words to SMEM); dp_empty's job (the readout) is gone with them (header section 5, the X-FA4 block).
    Softmax ownership: group g (128 lanes = kv rows) owns fp32 q quarters [32 g, +32) and [64 + 32 g, +32), so both groups pack into their
    OWN columns (no cross-group barrier) and the packed q order is the fixed permutation CFG.ts_k_perm = [0, 1, 4, 5, 2, 3, 6, 7] (16-q
    blocks) that the dV / dK B descriptors walk via k_idx.  Quarter b of both groups ([64, 128)) is the dQ image's home: without s_read the
    follower's P^T(t) q columns 64..127 would be exp2 of dQ values (g03 Rule 6's "second half" signature; RED 1 of the test module).

10. TENSOR-MAP / STRIDE CONTRACT (CFG.MAPS_PER_SEQUENCE = 9 per-sequence runtime maps, built by setup_thd_d128_2cta_kernel; unchanged)
    Q1 / dO1: 64-row maps, two [64 rows x 64 d] SW128 boxes per CTA stage (rows = the CTA's q half); Q2 / dO2: 128-row maps, one
    [128 q x 64 d] box at d = 64 * rank; K1 / V: 128-row maps, two [128 kv x 64 d] boxes (own rows); K2: two [128 kv x 64 d] boxes at
    d = 64 * rank over the PAIR's rows; dQ reduce-add: a 64-row fp32 map, [64 q x 32 d] boxes (4 per tile per CTA in two rounds; one
    reduce per PAIR tile); dV / dK: two [128 kv x 64 d] native boxes per CTA (128-row maps).  expect_tx = FULL box bytes ON THE LEADER
    for both CTAs' halves (2 x); TMA OOB zero-fills padded rows; sequence-local maps clip the tails.  ONE tensor-map acquire per
    consuming warp per work item (`acquire=False` on every per-tile load / reduce).  LOAD FORM (typed field ``tma_load_form`` ->
    ``tma_load_plain``): multicast (default) | plain, as the first body's section 10.
    dQ ROUTE red_global / red_global_pair: the dQ map is neither acquired nor used; each epilogue lane addresses its own 64 fp32 words of
    dq_accum (header section 3's image: lane m % 64 + 64 * (d // 64) = q row m of the CTA's q half, d half dhalf) from the SAME two carriers
    the TMA route's per-sequence descriptor is built from (helpers `setup_thd_d128_2cta_kernel` / `tile_dsl.thd.emit_seq_descs`): the meta
    prefix cu_q[b] (= `_item_bounds`' q_start, the descriptor's GLOBAL_ADDRESS origin) and the dq_accum tensor's strides (`cast_dq_thd`'s
    spelling: token stride `stride[1]` = heads x D fp32, head stride `stride[2]`); a q row >= seq_q (the descriptor's GLOBAL_DIM clip) is
    skipped by a branch (never written).  The dq_accum tensor reaches the kernel as an Optional argument that is None (dropped from the
    specialised signature) under the default route, so the default rendering is byte-identical.

11. OPTIONAL ARMS (const_expr; each states what the default rendering does)
    tmem_map (the ADAPTER's typed field: "dp_tail" = the first body, the default, its rendering byte-identical (c28e6013 @ 4/32/4,
    28b0eba0 @ 2/8/2) | "s_tail" = THIS body; declines by name; the compile key carries it) | THD (the unit is 256 kv rows per pair) |
    GQA grouped loaders (head // ratio) | causal | pair_mma_order (on this map `fa4` = THE default | `fa4b` = dQ(t-1) issued after dV(t), section 5,
    the `dq_after_dv` Constexpr, the same barrier table; `s_first` / `dk_first` decline BY NAME: their s_read / s_tail_empty site sets differ --
    inspector section 11; the field is resolved per map by config_sm100_d128_accum_2cta_stail.resolve_pair_mma_order_for_map /
    dq_after_dv_for_map) | tma_load_form (``tma_load_plain``: multicast | plain, section 10) |
    dq_route_pair (code-writer step 8, optimizer 05:15 rank 1 "#3-dry"; ``dq_route_code`` Constexpr[int] from
    config_sm100_d128_accum_2cta_stail.DQ_ROUTE_PAIR_CODES: "tma" = THE default, the ring STS + the store warp's four reduce-adds |
    "red_global" = right after the dQ tcgen05.ld + wait(LOAD) + the s_tail_empty arrive, `if row < seq_q:` CFG.DQ_RED_V4_PER_LANE = 16 x
    `red.global.add.v4.f32` of the lane's 64 words (tile_dsl.atomic.red_global_add_v4_f32; 2048 per CTA per tile = 64 warp-level instructions);
    the ring PROTOCOL kept with NO STS (section 5), the store warp skips the UTMAREDG issue + its commit | "red_global_pair" = the same words
    through tile_dsl.atomic.red_global_add_f32x8_sector_paired: lanes 2k / 2k+1 swap one v4 per 32-B sector by shfl.sync.bfly so every
    warp-level red carries FULL sectors (RULE-GAPS 396: the plain form's 32 x 16-B half-sector ops per instruction become 16 x 32 B);
    perf-only on dK / dV (bitwise under every value: they never touch the route), dQ by the fp64 gate (the fp32 add order is non-deterministic
    under every route; the test module's dQ twin is a TOLERANCE twin at the suite's gates, never bitwise); declines BY NAME on dp_tail;
    D2-pair (dq_reduce_add_off) under a red route is declined by name at compile_host (nothing to skip)) |
    warp_roles (code-writer B 2026-10-27, day 3 rank 1 "X-FA4"; ``warp_roles_code`` Constexpr[int] from config_sm100_d128_accum_2cta_stail.
    WARP_ROLES_CODES: "16w" = THE default = today's rendering, byte-identical (md5 4b3d0bbb @ 4/32/4, 645f8a24 @ 2/8/2) | "16w_fa4" = FAv4's
    role shape on this body: the softmax writes sDS own + XCHG from its dS^T registers and ships, warps 8-11 = the REDUCE group (`_reduce_loop`),
    ds_tmem + dp_empty deleted and dq_war_done added (BAR_LAYOUT_XFA4, 33 rows), the softmax's dq_war_done WAR wait, XCHG_BARRIER at 256 lanes, s_read kept, dQ ring form A
    (header sections 4 / 5 / 6).  Resolved PER MAP ("16w" only on dp_tail); under "16w_fa4" the red dQ routes are NOT BUILT and ds_gates_off is
    RETIRED: both decline BY NAME (config.validate_fields_for_roles at compile_host + the trace-time backstop in the kernel).  Perf-only on
    dK / dV (BITWISE vs 16w: the same accumulation chain per GEMM, the same k-orders; nothing of the softmax's arithmetic moves), dQ by the fp64
    gate (the reduce order is the ring's, unchanged).  The 20-warp lever-2 form is NOT a value here (rank 2: a separate step on the corrected
    144/64/64/64 split)) |
    (the stack fold 2026-10-27, on top of the above: under "16w_fa4" ALSO ``debug_clk`` (no STAMPS record layout for the fa4 roles) and
    ``dkv_accum = "fp32_reduce"`` (the fp32 staging rounds live in the 16w epilogue loop) decline BY NAME at compile_host and at the trace;
    ``exchange_off`` (D8s) is RE-SITED to the softmax's arm + ship under 16w_fa4 (the same two guards: the arm's bytes 0, no ship); the
    C2b row dvk_stage_empty is UNSERVED under 16w_fa4 = absent from BAR_LAYOUT_XFA4 / BarsXFA4; mask_form, stats_one_chunk, sdq_sts_off
    and dq_reduce_add_off serve both role shapes) |
    mask_form (lever L5, code-writer D 2026-10-27 on optimizer day-3 rank 6; ``mask_form_code`` Constexpr[int] from
    config_sm100_d128_accum_2cta_stail.MASK_FORM_CODES, resolved per map by resolve_mask_form_for_map; FAv4's compute body was READ for the
    idea only -- its transposed r2p mask selects -inf per cell (97 SEL in the 23:20 dump) on EVERY m_block, so this arm skips MORE than the
    reference does: "fsel" = THE default, today's rendering: _p_values applies the keep-word band mask (band_mask_words + apply_mask_words;
    CFG.MASK_WORDS_PER_LANE_PER_TILE = 2 keep-words, CFG.MASK_SELECTS_PER_LANE_PER_TILE = 64 selects = R2P + FSEL per cell) on EVERY q tile |
    "predicated" = the SAME mask body under the CTA-uniform runtime branch `if mask_tile:` of _p_values (one branch per q quarter = 2 per tile),
    mask_tile = tile_may_mask(pair, rank, qtile, seq_q, seq_k, causal) (section 7) computed once per tile in _softmax_loop: TRUE on the causal
    straddle (qtile * N < kv_end: CTA 0's tile 2 pair, CTA 1's tiles 2 pair and 2 pair + 1), on the last partial q tile ((qtile + 1) * N > seq_q)
    and on a CTA whose kv rows reach seq_k (kv_end > seq_k: the padded second half, a partial last kv block); an interior tile packs the exp2
    values straight -- 0 selects, 0 keep-words, 0 R2P and no `hi` select per lane (~90 fewer instructions per lane per tile; the ledger's
    <= 128-clk-per-tile model) -- and EXACTLY the fsel values (the exactness argument of section 7; test module `tile_mask_predicate_is_exact`),
    so P / dS and therefore dQ (per reduce-add order) / dK / dV are BITWISE between the two forms; no live register added (the predicate is a
    per-tile scalar, the keep-words and `hi` live inside the branch); the dS path (_ds_values) is untouched and inherits the exact zero |
    "diag_mask_off" = DIAGNOSTIC: the mask body on NO tile (the lever's ceiling; OUTPUT WRONG BY DESIGN on the diagonal / edge tiles and the
    dead second half; resolved like the diag_2cta probes: on the adapter INSTANCE, keyed apart in the plan cache, the driver's `--validate`
    refuses its arm `frost_thd_2cta_stail_l5off`); declines BY NAME on dp_tail (the first body keeps its own per-cell select form).  No barrier,
    SMEM or TMEM change under any value (the PTX mbarrier / MMA / commit census identical to the default's = the test module's _EXPECTED row,
    nanosleep 0, .local 0; the setmaxnreg split unchanged); the driver arms `frost_thd_2cta_stail_l5` (predicated) / `frost_thd_2cta_stail_l5off` (the probe)) |
    dkv_accum (C2b; the typed OP-LEVEL field ``SdpaBwdDslSm100D128.dkv_accum`` -> ``dkv_fp32`` Constexpr[bool]: "bf16_partials" = THE default,
    the bf16 dV / dK halves staged by `_native_out` + `tma_store_tile` into the per-Q-head partials + the fold launch; "fp32_reduce" = the fp32
    tiles staged by `_stage_fp32_slice` as [128 x 32] fp32 slices over the same bytes (+ the dS slab) in two rounds (5 + 3, dvk_stage_empty between) and reduce-added by the store warp
    (`tma_reduce_add_subtile` add.f32, the dQ slices' primitive) into the per-KV-head fp32 workspace maps the MAP_DV / MAP_DK slots carry
    (128-row fp32 maps, kv head = head // ratio), the fold launch gone, the one bf16 rounding in cast_dq; a ROUNDING CHANGE (fp32 group
    accumulation in hardware order), an op attribute, never a knob; dQ untouched; declines BY NAME on dp_tail, dense and MHA) |
    diag_2cta (DIAGNOSTIC; ``diag_2cta_code``, the shared code space of config_sm100_d128_accum_2cta.CFG.DIAG_2CTA_NAMES; the probes
    THIS map defines = config_sm100_d128_accum_2cta_stail.CFG.DIAG_2CTA_NAMES_ON_MAP, every other name declines BY NAME before the trace:
    None = this body | dq_reduce_add_off (D2-pair: the store warp's four reduce-add ISSUES skipped; the per-round `tma_store_wait` returns
    at once and the two dq_empty arrives per round stay, so the 2-slot ring balance holds with dQ staying zero) | exchange_off (D8s, the
    SAFE exchange-family ceiling = the inspector's "D8x bytes faked", 2026-10-27 08:01: the epilogue's `_ship_xchg` is NOT issued and
    `ds_recv_full.arrive(n_bytes=CFG.DIAG_EXCHANGE_OFF_TX = 0, pred=ship_pred)` completes each phase at its own arm; the three dS gates,
    the follower relay, the XCHG STS / fence / named barrier and EVERY other wait stay; the recv half is never written: dQ wrong; what it
    deletes = the ship + its landing latency + the peer dependence = the room of lever 1 / lever 2's exchange part / #4) |
    stats_one_chunk (D10: `_p_values` / `_ds_values` load ONE 16-B stats / delta chunk per q quarter (CFG.DIAG_STATS_CHUNK) and reuse it
    for all CFG.DIAG_STATS_CHUNKS_PER_QUARTER chunks: LDS.128 per lane per tile CFG.DIAG_STATS_LDS_PER_LANE 32 -> 4, registers / exp2 /
    FFMA / waits unchanged = the byte-removal-only residual test of the SMEM-pipe reading and I1's ceiling; stats wrong by design) |
    sdq_sts_off (D9: `_dq_round` skips the sDQ slice STS + its proxy fence in every round; the dq_empty waits, the dq_full arrive and
    the store warp's UTMAREDG (reading stale staging) stay = -256 wavefronts per tile per CTA, the second point of the wavefront -> time
    slope; declined by name under a red route: nothing to skip) | ds_gates_off (D8) RETIRED on this map, DECLINED BY NAME (section 5: the
    arm's only back-edge; CUDA 719) | dp_empty_wait_off (D7) DECLINED BY NAME: dp_empty is arrived at the dS^T readout, four GEMMs before
    dP^T(t+1) needs it, so the probe measures nothing); OUTPUT WRONG BY DESIGN under a probe; the driver's `--validate` refuses them) |
    debug_clk (STAMPS; the typed bool field -> the ``debug_clk`` Constexpr + the ``dbg_clk`` Int32 dump tensor, None = dropped from the
    signature when off: the default rendering is byte-identical; the d512 bodies' `debug_clk` pattern): every mbarrier wait of the role's
    representative warp (section 4) is bracketed by %clock64 and its delta accumulated per (role, barrier) into the record of config
    DBG_CLK (the leader also counts waits above DBG_CLK.BLOCKED_THRESH); the epilogue lead warp adds per tile: ds_tmem pass -> ship issue
    (sum, the DBG_CLK.HIST_EDGES histogram = the median the optimizer's kill line reads, the max), ds_tmem pass -> dq_done pass, dq_done
    pass -> s_tail_empty arrive; the ship -> recv handshake (the lead writes the ship clock + the CTA's tile count as a tag, the MMA warps
    read it after their ds_recv_full wait: the SAME-CTA proxy of the exchange flight, both CTAs being symmetric; a tag mismatch counts a
    miss); the store warp times its `tma_store_wait` drains; warp 0 zeroes the record in the P4 init block and dumps it after the final
    CTA sync to dbg_clk[(linear CTA) x DBG_CLK.WORDS ..] (the adapter carves the region past the chain's workspace, config
    dbg_clk_region_offset; the driver's decoder reads it; a CLC-cancelled cluster never runs, so its records stay unwritten and the magic word
    marks the ones that did).  Output CORRECT (bitwise dK / dV: the test module's twin), timing diagnostic:
    a ~30-60 clk issue cost per wait in the representative lane only) |
    store_wait_form (KNOB; the store warp's six mbarrier waits = dq_full x 4 USES on 2 slots + dv_full + dk_full in the hint-less
    two-instruction spin (`_wait_store`); the per-round `tma_store_wait` spelling is unchanged) | reg_split (KNOB; 144/144/80 |
    136/136/104: the epilogue carries 32 fp32 across the dS staging, so at 136 the STL / LDL == 0 pin is the gate for that arm).
    Test-only skew levers (house form `DEBUG_<SITE>_DELAY_US = int(globals().get("FROST_D128_2CTA_DEBUG_<SITE>_DELAY_US", 0))`, a
    `nanosleep.u32 1000` loop under const_expr, default 0 = no traced code; the test module's census pins nanosleep == 0 on the default
    rendering): LOADER (lever 4, k2_empty back-edge), FOLLOWER_OWN_HALF (1, ds_pair_ready), LEADER_SHIP (2, ds_recv_full + P9 routing),
    FOLLOWER_EPILOGUE_ITEM_END (3, dvk_empty 64) carried; NEW SOFTMAX_S_READ (5: the FOLLOWER softmax between its s_done wait and the S^T
    ld, every tile -- isolates s_read: with the follower asleep dQ(t-1) is held by s_read ALONE; RED = the wait removed before dQ(t-1): the
    follower's P^T(t) q columns 64..127 = exp2 of dQ values -> dV / dK RED) and EPILOGUE_DQ_READOUT (6: the FOLLOWER epilogue between its
    dq_done wait and the dQ ld -- isolates s_tail_empty: with the follower asleep S^T(t+1) is held by s_tail_empty ALONE; RED = the wait
    removed at the lookahead: the follower's 64 dQ rows of tile t-1 = S^T values; at the tail: dQ rows of tile hi-2 wrong; at the
    prologue: dQ rows of tile hi-1 of the previous item wrong).  Deterministic-hang topologies are SOURCE assertions in the test module
    (RED 3: dp_empty at the first body's site; RED 4: the dS gates on ready_state; RED 5: table 3's unguarded lookahead wait; D4: the
    streaming release with the both-slots wait form).

12. DEGENERATE-INPUT MATRIX  (row -> the handling LINE; absence is the bug)
    S_kv = 0 (dead pair: lo = hi = 0, stores skipped: _item_bounds / _store_loop `live_rows`) | kv rows >= seq_k in the pair's SECOND half
    (whole CTA dead: masked columns P = 0 exactly (_p_values `hi = 0`), dK / dV stores skipped by `live_rows`, dS recv half still
    exchanged = zeros, s_read / s_tail_empty still arrived by its softmax / epilogue: the pair-wide counts need both CTAs; mask_form predicated:
    tile_may_mask is TRUE on every tile of a dead CTA (kv_end > seq_k) so the mask body runs there) | q tile fully
    masked for CTA 1 at the causal diagonal (mask words, never a skip; under predicated tile_may_mask is TRUE on it: the mask body runs) |
    interior tile under mask_form predicated (tile_may_mask FALSE: (qtile + 1) * N <= seq_q, kv_end <= seq_k, and qtile * N >= kv_end under
    causal): no cell masked by construction, the exp2 values packed straight = what the fsel form selects through (bitwise) |
    n_q tiles = 1 (N == 1: no in-tile dQ, the tail dQ(lo) without an
    s_tail_empty wait, the readout fires once for the next item's prologue) | N == 2 / N == 3 (the walk of section 5) | S % 256 != 0 (the
    256-row THD unit; the sequence-local maps clip) | one pair per launch | padded Q rows (stats 0, mask word `hi = seq_q`, no store: the
    64-row dQ map clips; mask_form predicated: the last partial q tile has (qtile + 1) * N > seq_q, so its mask body runs) |
    zero-trip owner (CLC envelope) on BOTH CTAs (the zero-trip protocol of section 6: no dq_empty traffic at all) |
    a zero-trip item between two live items (the pending s_tail_empty / dp_empty completions of the previous live item's tail stay for
    the next live item's prologue: the states are never re-armed per item) | the kernel's first item (s_tail_empty / dp_empty / dvk_empty
    pre-armed st(1); dq_empty pre-armed on the epilogue side).

13. MEASURED LEVERS  (A/B/A x 3 x 100, control twin, band max(1 %, |pair|))
    lever | node class / SMs / lock | result | verdict | stands?
    ARM (ii) = this body (`frost_thd_2cta_stail`: tmem_map s_tail + order fa4 + the sDQ ring) vs the first body (`frost_thd_2cta`, s_first,
    dp_tail) | gb200-perf-152 / 152 / 1830 | PENDING (the benchmarker's A/B/A at 16K / 32K; the vs-aba line in the driver README's ARM (ii)
    block; success per profiler-ncu 23:40: period per pair tile 4307 -> 3700..4000 clk, the leader's gate share [0x38550]+[0x38560] well
    below 20 %, the softmax's dP wait ~10 %) | -- | --
    the map's own cost (s_first on the new map vs the old map) | -- | NOT built: a separate skeleton turn (inspector section 11) | -- | --
    fa4b on this body (`frost_thd_2cta_stail_fa4b`: dQ(t-1) after dV(t) = +512 clk exchange cover, -512 clk s_tail_empty cover, zero SMEM) vs fa4
    (`frost_thd_2cta_stail`) | gb200-perf-152 / 152 / 1830 | PENDING (the benchmarker's A/B/A at 16K / 32K; validated 6 / 6 cells + the module's
    rows under fa4b, code-writer step 6) | -- | the default stays fa4 until measured
    dq_route_pair red_global / red_global_pair on this body (`frost_thd_2cta_stail_redq` / `_redq_pair`: -256 certain .. -512 SMEM-pipe wavefronts
    per tile per CTA, RULE-GAPS 395; the #3-dry probe = the residual test of the SMEM-feed regime, optimizer 05:15) vs tma (`frost_thd_2cta_stail`)
    | gb200-perf-152 / 152 / 1830 | PENDING (validate first, then the benchmarker's A/B/A at 16K / 32K; ncu success metrics
    `l1tex__data_pipe_lsu_wavefronts_mem_shared` 1048 -> ~792, `_mem_global` <= +256, `lts__t_sectors_op_red` per tile vs the TMA form) | -- |
    the default stays tma until measured
    day-3 PROBES D8s `exchange_off` / D10 `stats_one_chunk` / D9 `sdq_sts_off` + STAMPS `debug_clk` on this body (driver arms
    `frost_thd_2cta_stail_d8s` / `_d10` / `_d9` DIAGNOSTIC, `frost_thd_2cta_stail_clk` real) | gb200-perf-152 / 152 / 1830 | PENDING (the
    benchmarker's probe process at 16K / 32K + the stamps run at 16K; ceilings and a decomposition, never wins; kill lines in PLAN § 11.1
    rank 0: D8s < +2.5 % step at both bar cells kills lever 1 + #4; D10 <= +1.5 % kills I1 / I2; STAMPS ship - ds_tmem < 100 clk median =
    lever 2's exchange part dead) | -- | --
    L5 mask_form predicated on this body (`frost_thd_2cta_stail_l5`: the softmax's 64 FSEL + 8 R2P + 2 keep-words per lane per tile skipped on every
    interior tile by the CTA-uniform `tile_may_mask` branch; ~2 of 64.5 tiles per item run the mask body at 16K causal; bitwise on dQ / dK / dV) vs
    fsel (`frost_thd_2cta_stail`) | gb200-perf-152 / 152 / 1830 | PENDING (validated first on the shared dev B200; the benchmarker's A/B/A at 16K /
    32K; the ceiling = `frost_thd_2cta_stail_l5off` in the probe process; kill line (ledger § DAY-3 ARMS rank 6): a WIN >= band + 1 pt at a bar
    cell ships it, NEUTRAL inside the band at both -> fsel stays the default, a loss beyond the band -> KILLED) | -- | the default stays fsel until
    measured
    warp_roles 16w_fa4 on this body (`frost_thd_2cta_stail_xfa4`: X-FA4 = lever 1 + lever 2, the exchange ship moves to dS-pack time on the softmax and
    the dQ readout + rounds leave the exchange path; POINT +4.2 / LOW -1.2 / HIGH +8.3 % step at 16K, +3.8 / -1.4 / +7.6 at 32K per the ledger; kill line:
    < +2 % step at both bar cells or a loss beyond the band at either -> kept as a selectable value) vs 16w (`frost_thd_2cta_stail`) | gb200-perf-152 /
    152 / 1830 | PENDING (AUDIT of the committed body first, then validate, then the benchmarker's A/B/A at 16K / 32K; success = the leader's
    ds_recv_full share (23.6 % of its samples at 16K) and the epilogue's dq_done share (38 %) fall; D8s is the ceiling) | -- | the default stays 16w

14. LAUNCH ABI
    positional args as the first body's host with THIS body's Constexprs (maps, lse, delta, meta, origins, dq_accum [Optional: the fp32
    workspace tensor under a red route, None = dropped under tma], n_batch, nheads, max_skv, stats_origin, scale, causal, dtype, ratio,
    dq_after_dv, dq_route_code, tma_load_plain, diag_2cta_code, store_wait_spin, reg_split_code, warp_roles_code, mask_form_code, dkv_fp32,
    debug_clk, dbg_clk [Optional: the Int32 STAMPS dump of config dbg_clk_records(n_batch, nheads, max_skv) x DBG_CLK.WORDS words, None =
    dropped unless debug_clk], stream) + cluster
    (CFG.CGA_M, CFG.CGA_N, 1); grid x = ceil(max_skv / CFG.THD_UNIT_ROWS) * CFG.CGA_M, the declared envelope; the prepared host's RUNTIME
    signature is unchanged (the body selector is a Constexpr behind the typed field, keyed in the compile cache).

PDL SITES: none in the accum family today (CLC-scheduled persistent envelope); unchanged.

Sources: bprop_d128_accum_grouped_2cta_f16.py (the first body, 3ef6b36f); config_sm100_d128_accum_2cta_stail.py; resources/arm2-barrier-table.md
(barrier-inspector 2026-10-27 01:05 UTC); .agent-report.md resource-manager 2026-10-27 00:53 UTC (tables 1-6); X-FA4: resources/x-fa4-barrier-table.md
(barrier-inspector 2026-10-27 08:45 UTC, sections 0-14, COPIED) + resources/xfa4-budgets.md (resource-manager 08:15 UTC, tables 1-4 + the RESOURCE-NOTE
lines; FAv4 flash_bwd_sm100.py :3936-3990 / :446-469 / :4181 read as the structural reference only, nothing copied); FAv4 flash_bwd_sm100.py
NOTE [2CTA schedules] + the pipelined MMA loop (read as the structural cross-check only: S(t+1) first, dQ before dV, Q_stage 1,
sdQaccum not aliased, the deferred dS commit = our dS gates + s_read, pipeline_dQ empty before S(t+1) = our s_tail_empty).
"""

from typing import NamedTuple, Optional

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm

from cudnn.frost.tile_dsl.atomic import RED_V4_WORDS, SECTOR_F32_WORDS, red_global_add_f32x8_sector_paired, red_global_add_v4_f32
from cudnn.frost.tile_dsl.barrier import MBarrier, PipelineState, Producer, Scope, advance, cga_arrive, cga_wait, named_barrier_fence
from cudnn.frost.tile_dsl.barrier import wait as mbar_wait
from cudnn.frost.tile_dsl.handles import MmaDesc, SmemTile, tma_slice_runtime_desc
from cudnn.frost.tile_dsl.mask import MASK_WORD_COLS, apply_mask_words, band_mask_words
from cudnn.frost.tile_dsl.mma import desc_opaque, mma_ss, mma_ts_step
from cudnn.frost.tile_dsl.pointwise import ffma2, fmul2, fp32_to_fp16, opaque_i32
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

from ...config_sm100_d128_accum_2cta_stail import (
    ALIASES,
    BAR_LAYOUT,
    BAR_LAYOUT_STATUS,
    ADDED_BARRIERS_XFA4,
    BAR_LAYOUT_XFA4,
    BAR_LAYOUT_XFA4_STATUS,
    CFG,
    DBG_CLK,
    DELETED_BARRIERS_XFA4,
    UNSERVED_BARRIERS_XFA4,
    bar_layout_for_roles,
    diag_2cta_code,
    dq_route_pair_code,
    mask_form_code,
    reg_split_counts,
    warp_roles_code,
)

LOG2E = 1.4426950408889634
# The 2-CTA per-sequence tensor-map set (header section 10); box rows per slot = CFG.MAP_BOX_ROWS, in this order (== the first body's).
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
# Test-only skew levers (house form; default 0 = no traced code: the test module's census pins nanosleep == 0 on the default rendering; the
# test sets the module global BEFORE the module executes -- spec loader + sys.modules; never an environment variable, never a knob).
# Carried from the first body (barrier-inspector AUDIT 2026-10-26): LOADER (lever 4, the k2_empty back-edge: both CTAs' loaders sleep at the
# top of every item), FOLLOWER_OWN_HALF (1, ds_pair_ready), LEADER_SHIP (2, the relay's ds_recv_full wait + the P9 routing of the ship),
# FOLLOWER_EPILOGUE_ITEM_END (3, dvk_empty's pair-wide 64).
DEBUG_LOADER_DELAY_US: int = int(globals().get("FROST_D128_2CTA_DEBUG_LOADER_DELAY_US", 0))
DEBUG_FOLLOWER_OWN_HALF_DELAY_US: int = int(globals().get("FROST_D128_2CTA_DEBUG_FOLLOWER_OWN_HALF_DELAY_US", 0))
DEBUG_LEADER_SHIP_DELAY_US: int = int(globals().get("FROST_D128_2CTA_DEBUG_LEADER_SHIP_DELAY_US", 0))
DEBUG_FOLLOWER_EPILOGUE_ITEM_END_DELAY_US: int = int(globals().get("FROST_D128_2CTA_DEBUG_FOLLOWER_EPILOGUE_ITEM_END_DELAY_US", 0))
# NEW for ARM (ii) (barrier-inspector 2026-10-27 section 8, RED 1 / RED 2): SOFTMAX_S_READ = the FOLLOWER softmax between its s_done wait and
# the S^T ld pair, every tile (isolates s_read: dQ(t-1) is held by s_read ALONE); EPILOGUE_DQ_READOUT = the FOLLOWER epilogue between its
# dq_done wait and the dQ ld (isolates s_tail_empty: S^T(t+1) / the tail dQ / the next prologue are held by s_tail_empty ALONE).
DEBUG_SOFTMAX_S_READ_DELAY_US: int = int(globals().get("FROST_D128_2CTA_DEBUG_SOFTMAX_S_READ_DELAY_US", 0))
DEBUG_EPILOGUE_DQ_READOUT_DELAY_US: int = int(globals().get("FROST_D128_2CTA_DEBUG_EPILOGUE_DQ_READOUT_DELAY_US", 0))
# NEW for X-FA4 (barrier-inspector x-fa4 table section 12 item 12; the resource-manager's RED form): SOFTMAX_DQ_WAR = the LEADER's MMA warp between
# the three dS gates and the dQ(t-1) issue (inside _dq_issue, both sites, leader-only by construction): with the issue late, both softmaxes' dS(t)
# STS are held under 16w_fa4 by their dq_done(t-1) WAR wait ALONE -- RED = the `dq_war` wait removed: the dS(t) STS lands under dQ(t-1)'s A read of
# sDS -> dQ rows wrong, dK / dV green.  Under 16w the same lever only delays the epilogue's dq_done(t-1) wait (SKEW_OK).
DEBUG_SOFTMAX_DQ_WAR_DELAY_US: int = int(globals().get("FROST_D128_2CTA_DEBUG_SOFTMAX_DQ_WAR_DELAY_US", 0))
# DIAGNOSTIC probes (header section 11): the typed adapter field ``diag_2cta`` reaches the kernel as the ``diag_2cta_code`` Constexpr[int] in
# the shared code space (config_sm100_d128_accum_2cta.diag_2cta_code); the probes this map does not define decline BY NAME before the trace
# (D7 dp_empty_wait_off; D8 ds_gates_off RETIRED: it removed the ds_recv_full arm's only back-edge and FAULTED 719, barrier-inspector 08:01).
#   DIAG_DQ_REDUCE_ADD_OFF (D2-pair): _store_loop skips the four tma_reduce_add issues (commit / wait / the per-round dq_empty arrives stay)
#   DIAG_EXCHANGE_OFF      (D8s):     _epilogue_loop skips the XCHG ship and arms ds_recv_full with CFG.DIAG_EXCHANGE_OFF_TX = 0 bytes; EVERY wait kept
#   DIAG_STATS_ONE_CHUNK   (D10):     _p_values / _ds_values load ONE stats / delta chunk per quarter and reuse it (LDS.128 32 -> 4 per lane per tile)
#   DIAG_SDQ_STS_OFF       (D9):      _dq_round skips the sDQ slice STS + its proxy fence (the reduce-add reads stale staging; waits / arrives stay)
DIAG_NONE = diag_2cta_code(None)
DIAG_DQ_REDUCE_ADD_OFF = diag_2cta_code("dq_reduce_add_off")
DIAG_EXCHANGE_OFF = diag_2cta_code("exchange_off")
DIAG_STATS_ONE_CHUNK = diag_2cta_code("stats_one_chunk")
DIAG_SDQ_STS_OFF = diag_2cta_code("sdq_sts_off")
DIAG_DS_GATES_OFF = diag_2cta_code(
    "ds_gates_off"
)  # RETIRED on this map (writer A) and under every role shape: the code of the shared space, named by the kernel backstop only
assert set(CFG.DIAG_2CTA_NAMES_ON_MAP) == {"dq_reduce_add_off", "exchange_off", "stats_one_chunk", "sdq_sts_off"}, "this body renders exactly these probes"
# STAMPS (header section 11): the typed adapter field ``debug_clk`` reaches the kernel as the ``debug_clk`` Constexpr[bool] + the ``dbg_clk`` dump
# tensor (Optional: None = dropped when off); the record layout is config truth (DBG_CLK: _W["wait:<role>:<barrier>"], _W["blocked:leader:<barrier>"],
# the header and segment names; the kernel never spells a word index).
_W = DBG_CLK.WORD
# dQ ROUTE (header section 11): the typed adapter field ``dq_route_pair`` reaches the kernel as the ``dq_route_code`` Constexpr[int]
# (config_sm100_d128_accum_2cta_stail.DQ_ROUTE_PAIR_CODES).  DQ_ROUTE_TMA = the default (sDQ STS + the store warp's reduce-adds);
# DQ_ROUTE_RED = red.global.add.v4.f32 from the dQ readout registers; DQ_ROUTE_RED_PAIR = the same with lane pairs forming full 32-B sectors.
DQ_ROUTE_TMA = dq_route_pair_code("tma", "s_tail")
DQ_ROUTE_RED = dq_route_pair_code("red_global", "s_tail")
DQ_ROUTE_RED_PAIR = dq_route_pair_code("red_global_pair", "s_tail")
# MASK FORM (header section 11, lever L5): the typed adapter field ``mask_form`` reaches the kernel as the ``mask_form_code`` Constexpr[int]
# (config_sm100_d128_accum_2cta_stail.MASK_FORM_CODES).  MASK_FORM_FSEL = the default (the keep-word mask selected on every tile);
# MASK_FORM_PREDICATED = the same mask body only on `tile_may_mask` tiles; MASK_FORM_DIAG_OFF = no mask on any tile (OUTPUT WRONG BY DESIGN).
MASK_FORM_FSEL = mask_form_code("fsel", "s_tail")
MASK_FORM_PREDICATED = mask_form_code("predicated", "s_tail")
MASK_FORM_DIAG_OFF = mask_form_code("diag_mask_off", "s_tail")
if CFG.MASK_WORD_COLS != MASK_WORD_COLS:  # config = truth: the config's keep-word width is the mask helper's (both 32; the config stays cutlass-free)
    raise ValueError(f"CFG.MASK_WORD_COLS {CFG.MASK_WORD_COLS} != tile_dsl.mask.MASK_WORD_COLS {MASK_WORD_COLS}")
# ROLE SHAPE (header sections 4 / 5 / 11; day 3 rank 1 "X-FA4"): the typed adapter field ``warp_roles`` reaches the kernel as the ``warp_roles_code``
# Constexpr[int] (config_sm100_d128_accum_2cta_stail.WARP_ROLES_CODES).  WARP_ROLES_16W = the default (today's rendering, byte-identical);
# WARP_ROLES_16W_FA4 = FAv4's role shape: the softmax writes sDS own + XCHG from its dS^T registers and ships, warps 8-11 = the REDUCE group
# (_reduce_loop), ds_tmem / dp_empty deleted (BAR_LAYOUT_XFA4), the softmax's dq_done WAR wait (barrier-inspector resources/x-fa4-barrier-table.md).
WARP_ROLES_16W = warp_roles_code("16w", "s_tail")
WARP_ROLES_16W_FA4 = warp_roles_code("16w_fa4", "s_tail")
SOFTMAX_LEAD_WARP = CFG.SOFTMAX_LEAD_WARP  # 16w_fa4: the softmax warp whose elected lane arms ds_recv_full and ships XCHG (inspector X4; FAv4's tidx 0)
XCHG_BARRIER_LANES_16W = CFG.xchg_barrier_lanes("16w")  # bar.sync 2, 128: the 4 epilogue warps (today's form)
XCHG_BARRIER_LANES_16W_FA4 = CFG.xchg_barrier_lanes("16w_fa4")  # bar.sync 2, 256: the 8 softmax warps (inspector X3)
assert (WARP_ROLES_16W, WARP_ROLES_16W_FA4) == (0, 1) and XCHG_BARRIER_LANES_16W == CFG.EPILOGUE_LANES and XCHG_BARRIER_LANES_16W_FA4 == CFG.SOFTMAX_LANES


class Bars(NamedTuple):
    """One MBarrier per ``BAR_LAYOUT`` row, in ``BAR_LAYOUT`` order (header section 5 = the table and the lane ledger; 34 rows) -- the ``16w`` bundle.

    Built by :func:`_bars` on BOTH CTAs with identical counts; a LEADER-scope field's follower copy is never armed / waited.  Under
    ``warp_roles = "16w_fa4"`` the bundle is :data:`BarsXFA4` (``BAR_LAYOUT_XFA4``: ds_tmem / dp_empty ABSENT -- an AttributeError if a path
    touches them, never a silent dead row)."""

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
    s_read: MBarrier
    ds_tmem: MBarrier
    dp_empty: MBarrier
    s_tail_empty: MBarrier
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
    dvk_stage_empty: MBarrier  # C2b (dkv_accum = fp32_reduce): store warp after round 1's fp32 reduce drain -> the epilogue's round 2; init only under dkv_fp32


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
DVK_STAGE_EMPTY_ROW = Bars._fields.index("dvk_stage_empty")  # C2b: the one row whose init is guarded by dkv_fp32 (no user under the default rendering)
# The 16w_fa4 bundle: the inspector's 32 rows + dq_war_done (BAR_LAYOUT_XFA4) in order; the two deleted names have no field, no init, no arrive, no
# wait, no state; the C2b row dvk_stage_empty is UNSERVED there (fp32_reduce declines by name under 16w_fa4) and has no field either.
BarsXFA4 = NamedTuple("BarsXFA4", [(row[0], MBarrier) for row in BAR_LAYOUT_XFA4])
_BARS_XFA4_STATUS = BAR_LAYOUT_XFA4_STATUS
assert (
    tuple(BarsXFA4._fields)
    == tuple(row[0] for row in BAR_LAYOUT_XFA4)
    == tuple(n for n in Bars._fields if n not in DELETED_BARRIERS_XFA4 and n not in UNSERVED_BARRIERS_XFA4) + ADDED_BARRIERS_XFA4
)
assert CFG.TMEM_MAP == "s_tail" and CFG.PAIR_MMA_ORDERS == ("fa4", "fa4b"), "this body renders the s_tail map under the fa4 / fa4b orders only (`dq_after_dv`)"
assert CFG.DQ_ROUTES_PAIR == ("tma", "red_global", "red_global_pair") and (DQ_ROUTE_TMA, DQ_ROUTE_RED, DQ_ROUTE_RED_PAIR) == (0, 1, 2)
MCAST_PAIR = CFG.MCAST_PAIR_MASK  # tcgen05.commit multicast to both CTAs of the pair; the leader is cluster rank 0 under cluster (2,1,1)
LEADER_RANK = 0


class Dbg(NamedTuple):
    """STAMPS context of a role (header section 11; ``None`` = the lever is off = no traced code): ``words`` = the CTA's Int32 record in the
    bar block's spare (DBG_CLK.SMEM), ``rep`` = this warp is the role's representative (softmax warp 0, the epilogue lead warp; a single
    service warp passes True), so ONE elected lane per role accumulates into the role's own words (no atomics)."""

    words: object
    rep: object


class DbgLeader(NamedTuple):
    """The leader MMA warp's STAMPS context: the same record, plus the per-barrier BLOCKED counts (waits above DBG_CLK.BLOCKED_THRESH clk)
    DBG_CLK.LEADER_BLOCKED_STRIDE words past each wait sum (``_clk_wait_add`` tells the two contexts apart at trace time)."""

    words: object
    rep: object


def _view(slab, offset, dtype, count, alignment=16):
    ptr = Pointer(slab.data_ptr() + offset, dtype=dtype)
    return cutlass.Array(ptr.ir_value(), shape=count, dtype=dtype, alignment=alignment)


def _bars(slab, fa4_roles: bool = False):
    """The inspector's table as typed barriers: one MBarrier per row of the role shape's layout (``BAR_LAYOUT`` -> :class:`Bars` under ``16w``;
    ``BAR_LAYOUT_XFA4`` -> :data:`BarsXFA4` under ``16w_fa4``), producer and scope from the row's kind."""
    bundle = BarsXFA4 if fa4_roles else Bars
    return bundle(
        *(
            MBarrier(
                _view(slab, off, cutlass.Int64, slots),
                stages=slots,
                init_count=count,
                producer=_KINDS[kind][0],
                scope=_KINDS[kind][1],
                try_wait=True,
            )
            for _, off, slots, count, kind, _scope in bar_layout_for_roles("16w_fa4" if fa4_roles else "16w")
        )
    )


# ---------------------------------------------------------------------------------------------------------------------
# Barrier helpers (the guard forms of header section 5 as code; each role body below names the sites that call them).
# ---------------------------------------------------------------------------------------------------------------------


@cute.jit
def _clk_add(dbg, word: cutlass.Constexpr[int], v):
    """STAMPS: the role's representative warp's elected lane adds the Int32 ``v`` to record word ``word`` (its own words: no atomics)."""
    if dbg.rep:
        if nvvm.elect_sync():
            w = dbg.words.subview(word)
            w.store(w.load() + v)


@cute.jit
def _clk_wait_add(dbg, word: cutlass.Constexpr[int], dt):
    """STAMPS: ``dt`` (Int64 clk spent inside ONE mbarrier wait) into the wait sum ``word`` (u32 wrap); on the leader's context also the
    blocked count DBG_CLK.LEADER_BLOCKED_STRIDE words further when the wait took more than DBG_CLK.BLOCKED_THRESH clk."""
    if dbg.rep:
        if nvvm.elect_sync():
            w = dbg.words.subview(word)
            w.store(w.load() + cutlass.Int32(dt))
            if cutlass.const_expr(isinstance(dbg, DbgLeader)):
                if dt > cutlass.Int64(DBG_CLK.BLOCKED_THRESH):
                    b = dbg.words.subview(word + DBG_CLK.LEADER_BLOCKED_STRIDE)
                    b.store(b.load() + cutlass.Int32(1))


@cute.jit
def _wait(bar, phase, dbg, word: cutlass.Constexpr[int]):
    """The barrier's declared try_wait form + the warp reconvergence (the baseline's form).  ``dbg`` None = exactly that (the shipped
    statements); a STAMPS context brackets the UNMODIFIED wait with two %clock64 reads and accumulates the delta into ``word`` -- no
    barrier, no fence, no change to the wait."""
    if cutlass.const_expr(dbg is None):
        bar.wait(phase)
        # All full-warp consumers reconverge before any elected credit / issue (the baseline's form).
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
    else:
        t0 = cute.arch.clock64()
        bar.wait(phase)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        _clk_wait_add(dbg, word, cute.arch.clock64() - t0)


@cute.jit
def _wait_tcgen(bar, phase, dbg, word: cutlass.Constexpr[int]):
    _wait(bar, phase, dbg, word)
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)


@cute.jit
def _wait_store(bar, phase, spin: cutlass.Constexpr[bool], dbg, word: cutlass.Constexpr[int]):
    """The STORE warp's mbarrier wait in the ``store_wait_form`` knob's form (header section 11; the same phase test either way): spin =
    ``tile_dsl.barrier.wait`` with ``spin=True``, the hint-less 2-instruction parity-wait loop in inline PTX; else the barrier's declared
    untimed parity-wait form = :func:`_wait`.  Both reconverge the warp before the issue; ``dbg`` / ``word`` = the STAMPS bracket of _wait."""
    if cutlass.const_expr(spin):
        if cutlass.const_expr(dbg is None):
            mbar_wait(bar.smem_ptr, phase, spin=True)
            nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        else:
            t0 = cute.arch.clock64()
            mbar_wait(bar.smem_ptr, phase, spin=True)
            nvvm.bar_warp_sync(cute.arch.FULL_MASK)
            _clk_wait_add(dbg, word, cute.arch.clock64() - t0)
    else:
        _wait(bar, phase, dbg, word)


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
def _drain(bar, state, stages: cutlass.Constexpr[int], dbg, word: cutlass.Constexpr[int]):
    """P15 / whole-ring drain from the CARRIED state: unused slots pass at their pre-armed parity, used slots wait their last release."""
    for _ in cutlass.range_constexpr(stages):
        _wait(bar[state.idx], state.phase, dbg, word)
        state = advance(state, stages)
    return state


@cute.jit
def _clk_ship(dbg, d):
    """STAMPS, the epilogue lead warp per tile: ``d`` = clk from the ds_tmem pass to the ship issue -> the sum, the DBG_CLK.HIST_EDGES
    histogram (bin 0 = the optimizer's "< 100 clk" line) and the max; then the ship stamp (clock lo32 + the CTA's tile count as the tag)
    the MMA warps read back after their ds_recv_full wait (:func:`_clk_recv`)."""
    if dbg.rep:
        if nvvm.elect_sync():
            s = dbg.words.subview(_W["seg_ship_sum"])
            s.store(s.load() + cutlass.Int32(d))
            if d < cutlass.Int64(DBG_CLK.HIST_EDGES[0]):
                h = dbg.words.subview(_W["seg_ship_hist0"])
                h.store(h.load() + cutlass.Int32(1))
            elif d < cutlass.Int64(DBG_CLK.HIST_EDGES[1]):
                h = dbg.words.subview(_W["seg_ship_hist1"])
                h.store(h.load() + cutlass.Int32(1))
            elif d < cutlass.Int64(DBG_CLK.HIST_EDGES[2]):
                h = dbg.words.subview(_W["seg_ship_hist2"])
                h.store(h.load() + cutlass.Int32(1))
            else:
                h = dbg.words.subview(_W["seg_ship_hist3"])
                h.store(h.load() + cutlass.Int32(1))
            m = dbg.words.subview(_W["seg_ship_max"])
            if cutlass.Int32(d) > m.load():
                m.store(cutlass.Int32(d))
            dbg.words.subview(_W["ship_clk"]).store(cutlass.Int32(cute.arch.clock64()))
            dbg.words.subview(_W["ship_tag"]).store(dbg.words.subview(_W["tiles"]).load())


@cute.jit
def _clk_recv(
    dbg,
    tiles_word: cutlass.Constexpr[int],
    sum_word: cutlass.Constexpr[int],
    matched_word: cutlass.Constexpr[int],
    missed_word: cutlass.Constexpr[int],
    tag_back: cutlass.Constexpr[int],
):
    """STAMPS, an MMA warp right after its ds_recv_full wait passed: the handshake with the epilogue lead's ship stamp (header section 11).
    The stamp's tag is the lead's tile count when it shipped; this warp's own count minus ``tag_back`` names the tile whose copy the wait
    consumed (DBG_CLK.TAG_BACK_*).  A match accumulates pass - ship (the SAME-CTA proxy of the exchange flight: our ship vs the peer's
    landing, both CTAs symmetric) and counts; a mismatch (the lead has not shipped this tile yet, or is a tile ahead) counts a miss."""
    if dbg.rep:
        if nvvm.elect_sync():
            expected = dbg.words.subview(tiles_word).load() - cutlass.Int32(tag_back)
            if dbg.words.subview(_W["ship_tag"]).load() == expected:
                s = dbg.words.subview(sum_word)
                s.store(s.load() + (cutlass.Int32(cute.arch.clock64()) - dbg.words.subview(_W["ship_clk"]).load()))
                m = dbg.words.subview(matched_word)
                m.store(m.load() + cutlass.Int32(1))
            else:
                x = dbg.words.subview(missed_word)
                x.store(x.load() + cutlass.Int32(1))


@cute.jit
def _clk_init(words):
    """STAMPS, warp 0's elected lane inside the P4 init block (published by the CTA sync that follows): zero the record, stamp the body
    start (clock lo32, %globaltimer lo32; the dump closes both spans -> clk / ns = the SM clock the body actually ran at)."""
    for w in cutlass.range_constexpr(DBG_CLK.WORDS):
        words.subview(w).store(cutlass.Int32(0))
    words.subview(_W["body_clk"]).store(cutlass.Int32(cute.arch.clock64()))
    words.subview(_W["body_ns"]).store(cutlass.Int32(cute.arch.globaltimer()))


@cute.jit
def _clk_dump(words, dbg_clk, rank):
    """STAMPS, warp 0's elected lane after the final CTA sync (every role returned, every record word written): close the body clock and
    its %globaltimer span, write the magic and the rank, copy the record to dbg_clk[(linear CTA) x DBG_CLK.WORDS ..] (plain stores)."""
    c = words.subview(_W["body_clk"])
    c.store(cutlass.Int32(cute.arch.clock64()) - c.load())
    n = words.subview(_W["body_ns"])
    n.store(cutlass.Int32(cute.arch.globaltimer()) - n.load())
    words.subview(_W["magic"]).store(cutlass.Int32(DBG_CLK.MAGIC))
    words.subview(_W["rank"]).store(rank)
    x, y, z = cute.arch.block_idx()
    nx, ny, nz = cute.arch.grid_dim()
    base = (x + nx * (y + ny * z)) * cutlass.Int32(DBG_CLK.WORDS)
    out = cutlass.make_array_view(dbg_clk)
    for w in cutlass.range_constexpr(DBG_CLK.WORDS):
        out[base + cutlass.Int32(w)] = words.subview(w).load()


@cute.jit
def _skew_leader_ship(rank):
    """Test-only skew lever 2 (LEADER_SHIP): the LEADER holds its XCHG ship -- the half is written, fenced and the XCHG named barrier passed.  ONE site
    for both role shapes (16w: the epilogue; 16w_fa4: the softmax, right before the elected arm + ship); default 0 = no traced code."""
    if cutlass.const_expr(DEBUG_LEADER_SHIP_DELAY_US > 0):
        if rank == cutlass.Int32(LEADER_RANK):
            for _us in cutlass.range(DEBUG_LEADER_SHIP_DELAY_US, unroll=1):
                nvvm.inline_ptx("nanosleep.u32 1000;", read_only_args=[])


@cute.jit
def _skew_follower_own_half(rank):
    """Test-only skew lever 1 (FOLLOWER_OWN_HALF): the FOLLOWER holds its own-half dS STS (16w: between the ship and the own STS -- its XCHG is shipped;
    16w_fa4: right after the softmax's dq_war wait, before its 8 STS -- its ds_smem AND its ship are held together).  ONE site for both shapes."""
    if cutlass.const_expr(DEBUG_FOLLOWER_OWN_HALF_DELAY_US > 0):
        if rank != cutlass.Int32(LEADER_RANK):
            for _us in cutlass.range(DEBUG_FOLLOWER_OWN_HALF_DELAY_US, unroll=1):
                nvvm.inline_ptx("nanosleep.u32 1000;", read_only_args=[])


@cute.jit
def _skew_follower_dq_readout(rank):
    """Test-only skew lever 6 (EPILOGUE_DQ_READOUT): the FOLLOWER's warps 8-11 hold their dQ readout between the dq_done wait and the dQ ld (s_tail_empty
    alone holds S^T(t+1) / the tail dQ / the next prologue).  ONE site for both shapes (the epilogue under 16w, the reduce group under 16w_fa4)."""
    if cutlass.const_expr(DEBUG_EPILOGUE_DQ_READOUT_DELAY_US > 0):
        if rank != cutlass.Int32(LEADER_RANK):
            for _us in cutlass.range(DEBUG_EPILOGUE_DQ_READOUT_DELAY_US, unroll=1):
                nvvm.inline_ptx("nanosleep.u32 1000;", read_only_args=[])


@cute.jit
def _skew_follower_item_end(rank):
    """Test-only skew lever 3 (FOLLOWER_EPILOGUE_ITEM_END): the FOLLOWER's warps 8-11 hold their dV / dK TMEM readout at the item end (dvk_empty's
    pair-wide 64).  ONE site for both shapes."""
    if cutlass.const_expr(DEBUG_FOLLOWER_EPILOGUE_ITEM_END_DELAY_US > 0):
        if rank != cutlass.Int32(LEADER_RANK):
            for _us in cutlass.range(DEBUG_FOLLOWER_EPILOGUE_ITEM_END_DELAY_US, unroll=1):
                nvvm.inline_ptx("nanosleep.u32 1000;", read_only_args=[])


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
    row, 8-row swizzle period) -- the layout `_store_words` writes; the MN-major k-step walks 16 kv rows = 2 KiB."""
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


def _dq_slot_tile(slab, slot):
    """fp32 reduce-add staging slot ``slot`` of the DEDICATED sDQ ring ([Q_HALF x DQ_BOX] SW128; header section 8): one source box per reduce."""
    return SmemTile(
        base=_view(slab, CFG.dq_slot(slot), cutlass.Float32, CFG.Q_HALF * CFG.DQ_BOX, alignment=1024),
        elems_per_stage=CFG.Q_HALF * CFG.DQ_BOX,
        leading_byte_offset=16,
        stride_byte_offset=8 * CFG.SWIZZLE_BYTES,
        layout=CFG.SWIZZLE_LAYOUT,
        tma_loads_per_tile=1,
        tma_granu_elems=CFG.DQ_BOX,
        tma_subtile_stride_elems=CFG.Q_HALF * CFG.DQ_BOX,
        desc_version=CFG.DESC_VERSION,
    )


def _dkv_acc_tile(slab, base):
    """One fp32 [M kv x DQ_BOX d] SW128 reduce-add source slice at byte ``base`` (CFG.dkv_acc_slot(rank, i); dkv_accum = fp32_reduce): the
    sDQ slot form over 128 rows."""
    # RESOURCE-NOTE: fp32 box [128 kv x 32 d] = 16 KiB SW128 = _dq_slot_tile with M rows; same STS form
    return SmemTile(
        base=_view(slab, base, cutlass.Float32, CFG.M * CFG.DQ_BOX, alignment=1024),
        elems_per_stage=CFG.M * CFG.DQ_BOX,
        leading_byte_offset=16,
        stride_byte_offset=8 * CFG.SWIZZLE_BYTES,
        layout=CFG.SWIZZLE_LAYOUT,
        tma_loads_per_tile=1,
        tma_granu_elems=CFG.DQ_BOX,
        tma_subtile_stride_elems=CFG.M * CFG.DQ_BOX,
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
# Body helpers (the first body's forms onto the s_tail map; the S^T ld pair is split out of _p_values for the s_read arrive).
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
def _store_words_quarter(slab, row_base, words, first: cutlass.Constexpr[int], xor_lane):
    """16w_fa4 (X-FA4, resource-manager table 1): ONE 32-q quarter of the lane's dS^T row -- the CFG.PACK_QUARTER_COLS = 16 packed Int32 words
    ``words[first, first + 16)`` -- into the SW128 slab row at ``row_base`` (= the own-half or XCHG base + row * SWIZZLE_BYTES) as
    CFG.DS_STS_VECS_PER_QUARTER = 4 16-B vectors at vec ``vec0 + v`` of the row, ``xor_lane = (row % 8) ^ vec0`` (``vec0`` = 4 x the softmax group:
    group g owns vec 4g .. 4g+3 of the 64-q half; ``vec0 + v == vec0 | v == vec0 ^ v`` for v < 4): the same ``(row % 8) ^ vec`` XOR of
    :func:`_store_words`, so the softmax writes today's epilogue LDTM image word for word (dK / dV bitwise, dQ's A descriptor unchanged);
    conflict-free: 4 wavefronts per STS.128 = the floor (resource-manager table 2, resources/vs-smem-banks-xfa4.log).  The caller passes
    ``xor_lane`` through ``opaque_i32(.., anchor=qtile)`` so the 8 address adds live inside the tile (the d256 2x2 lesson: hoisted, they spilled)."""
    for v in cutlass.range_constexpr(CFG.DS_STS_VECS_PER_QUARTER):
        off = row_base + (xor_lane ^ v) * 16
        values = cutlass.Vector.from_elements(tuple(words[first + v * 4 + k] for k in range(4)), cutlass.Int32)
        Pointer(slab.data_ptr() + off, dtype=cutlass.Int32).store(values, alignment=16)


@cute.jit
def _store_dq_slice(slab, byte_base, row, values):
    """fp32 slice row: the lane's DQ_BOX values of q row ``row`` into the SW128 [Q_HALF x DQ_BOX] slot at ``byte_base`` (one ROUND's half of
    the 64-column dQ readout; the same 16-B vector / XOR-swizzle form as _store_words)."""
    for vec in cutlass.range_constexpr(CFG.DQ_BOX // 4):
        off = byte_base + row * CFG.SWIZZLE_BYTES + ((row % 8) ^ vec) * 16
        v = cutlass.Vector.from_elements(tuple(values[vec * 4 + k] for k in range(4)), cutlass.Float32)
        Pointer(slab.data_ptr() + off, dtype=cutlass.Float32).store(v, alignment=16)


@cute.jit
def _dq_half(values, first: cutlass.Constexpr[int]):
    """Columns [first, first + DQ_BOX) of the lane's 64-column dQ readout as one vector: round A = the first DQ_BOX (first 0), round B = the
    second (first DQ_BOX), CARRIED in registers across the next tile's dS staging (header section 4: ~96 live of 144)."""
    return cutlass.Vector.from_elements(tuple(values[first + i] for i in range(CFG.DQ_BOX)), cutlass.Float32)


@cute.jit
def _dq_zero_half():
    """The carried half before the item's first readout (never stored: round B runs only after a readout)."""
    return cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(CFG.DQ_BOX)), cutlass.Float32)


@cute.jit
def _concat32(a, b):
    """Two 16-word tcgen05.ld results -> one 32-word row (q ascending across the two quarters)."""
    return cutlass.Vector.from_elements(tuple(a[i] for i in range(CFG.PACK_QUARTER_COLS)) + tuple(b[i] for i in range(CFG.PACK_QUARTER_COLS)), cutlass.Int32)


@cute.jit
def _map_ptr(maps, batch, n_batch, slot: cutlass.Constexpr[int]):
    return (maps.iterator.raw_ptr() + (cutlass.Int64(slot) * n_batch + batch) * TENSOR_MAP_QWORDS).tospace(cutlass.AddressSpace.generic)


@cute.jit
def _next_tile(sched, bars, state, dbg, word: cutlass.Constexpr[int]):
    """The pair's next CLC payload: x = the first ctaid.x of the cancelled cluster (a multiple of CGA_M) -> pair = x // CGA_M."""
    _wait(bars.sched_full[state.idx], state.phase, dbg, word)
    x, yz, valid = read_clc_payload(sched, state.idx * 8, warp_broadcast=True)
    return x // CFG.CGA_M, yz & 0xFFFF, (yz >> 16) & 0xFFFF, valid, advance(state, CFG.SCHED_STAGES)


@cute.jit
def _item_bounds(pair, meta, batch, n_batch, causal: cutlass.Constexpr[bool]):
    """Section 7: cluster-union (q_lo, q_hi, seq_q, seq_k, q_start) for the pair's 256-row kv block; pure, no SMEM.

    # BARRIER contract: both CTAs derive IDENTICAL (lo, hi) from the same payload words, so every barrier flips the same number of
    # times on both CTAs (P14) -- including the leader's pair-uniform `qtile > lo` / `hi - 1 > lo` guards on the s_tail_empty waits and
    # the dQ issue sites; a dead owner (pair * 256 >= seq_k) is lo = hi = 0 on both; a padded SECOND half is NOT a skip.
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


def tile_may_mask(pair, rank, qtile, seq_q, seq_k, causal: bool):
    """Section 7 (lever L5): TRUE iff a cell of CTA ``rank``'s [M kv x N q] tile ``qtile`` of pair ``pair`` can be masked by _p_values' band --
    the causal straddle (qtile * N < kv_end), the last partial q tile ((qtile + 1) * N > seq_q), kv rows at or past seq_k (kv_end > seq_k) --
    with kv_end = pair * PAIR_KV_ROWS + (rank + 1) * M.  Plain Python over the traced scalars (CTA-uniform: no lane term) AND over host ints
    (the test module's exactness row runs this very function against the per-cell mask).  EXACT (section 7): FALSE means every column of the
    tile keeps for every lane, so the fsel form's selects are all pass-through there and skipping them is bitwise."""
    kv_end = pair * CFG.PAIR_KV_ROWS + (rank + 1) * CFG.M
    edge = ((qtile + 1) * CFG.N > seq_q) | (kv_end > seq_k)
    if causal:
        edge = edge | (qtile * CFG.N < kv_end)
    return edge


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
def _s_values(tmem_base, group):
    """The S^T ld pair of the lane's kv row over softmax group ``group``'s two q quarters (a = quarter g, b = quarter g + 2 = the [64, 128)
    half that is also the dQ image's home), completed by tcgen05_wait(LOAD).  The caller fences and arrives s_read BEFORE the exp body:
    once these registers hold S^T(t), dQ(t-1) may land in [64, 128)."""
    q_lo = group * CFG.GROUP_QUARTER
    a = _tmem_load(tmem_base + CFG.S_COL + q_lo, CFG.GROUP_QUARTER)
    b = _tmem_load(tmem_base + CFG.S_COL + q_lo + CFG.N // 2, CFG.GROUP_QUARTER)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    return a, b


@cute.jit
def _p_values(
    slab,
    tmem_base,
    group,
    key,
    qtile,
    seq_q,
    seq_k,
    scale,
    a,
    b,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    one_chunk: cutlass.Constexpr[bool],
    mask_form_code: cutlass.Constexpr[int],
    mask_tile,
):
    """S^T (registers ``a``, ``b`` from _s_values) -> P^T for the lane's kv row ``key`` over softmax group ``group``'s two q quarters; the
    packed bf16 P^T goes in place into the group's PACK_GROUP_COLS columns of [0, 64).  Returns the 64 fp32 P values in packed order.
    ``one_chunk`` (DIAGNOSTIC D10 `stats_one_chunk`): ONE 16-B LSE chunk per quarter (CFG.DIAG_STATS_CHUNK) is loaded and reused for all
    CFG.DIAG_STATS_CHUNKS_PER_QUARTER chunks -- the same registers, exp2 and FFMA count, 7 of 8 LDS.128 gone; the stats are WRONG BY DESIGN.
    ``mask_form_code`` (header section 11): MASK_FORM_FSEL = the keep-word band mask selected on EVERY tile (today's rendering);
    MASK_FORM_PREDICATED = the SAME mask body under the CTA-uniform runtime branch ``if mask_tile:`` (``mask_tile`` = tile_may_mask of section 7,
    read under this form only: FALSE on an interior tile = no cell of this CTA's tile can be masked, so the exp2 values are packed straight);
    MASK_FORM_DIAG_OFF = no mask on any tile (the DIAGNOSTIC ceiling: OUTPUT WRONG BY DESIGN)."""
    q_lo = group * CFG.GROUP_QUARTER
    scaled = scale * cutlass.Float32(LOG2E)
    # KV-major transpose of the shared mask band: Q columns >= key for causal, below the actual Q length, and entirely
    # dead for a padded KV row (key >= seq_k). Select exact zero AFTER exp2 (also safe for a no-key row whose LSE is -inf).
    lo = key if cutlass.const_expr(causal) else None
    if cutlass.const_expr(mask_form_code == MASK_FORM_FSEL):
        hi = cutlass.Int32(arith.select((key < seq_k).ir_value(), seq_q.ir_value(), cutlass.Int32(0).ir_value()))
    p_all = []
    packed = []
    for half in cutlass.range_constexpr(CFG.QUARTERS_PER_GROUP):
        scores = (a, b)[half]
        col0 = q_lo + half * (CFG.N // 2)
        values = []
        if cutlass.const_expr(one_chunk):  # DIAGNOSTIC D10: the one chunk every column of the quarter reuses
            stat_one = _view(slab, CFG.STATS, cutlass.Float32, CFG.N).subview(col0 + CFG.DIAG_STATS_CHUNK * 4).load(vector_size=4, alignment=16)
        for chunk in cutlass.range_constexpr(CFG.DIAG_STATS_CHUNKS_PER_QUARTER):
            if cutlass.const_expr(one_chunk):
                stat = stat_one
            else:
                stat = _view(slab, CFG.STATS, cutlass.Float32, CFG.N).subview(col0 + chunk * 4).load(vector_size=4, alignment=16)
            for pair in cutlass.range_constexpr(2):
                i = chunk * 4 + pair * 2
                e0, e1 = ffma2(scaled, scaled, scores[i], scores[i + 1], stat[pair * 2], stat[pair * 2 + 1])
                values.extend((cute.math.exp2(e0, fastmath=True), cute.math.exp2(e1, fastmath=True)))
        if cutlass.const_expr(mask_form_code == MASK_FORM_FSEL):
            words = band_mask_words(lo, hi, qtile * CFG.N + col0, CFG.GROUP_QUARTER)
            p = apply_mask_words(cutlass.Vector.from_elements(tuple(values), cutlass.Float32), words, mask_value=0.0, n_cols=CFG.GROUP_QUARTER)
        else:
            p = cutlass.Vector.from_elements(tuple(values), cutlass.Float32)
            if cutlass.const_expr(mask_form_code == MASK_FORM_PREDICATED):
                if mask_tile:  # CTA-uniform (section 7): the mask body only on a tile that can hold a masked cell; `hi` and the words live in here
                    hi_m = cutlass.Int32(arith.select((key < seq_k).ir_value(), seq_q.ir_value(), cutlass.Int32(0).ir_value()))
                    words = band_mask_words(lo, hi_m, qtile * CFG.N + col0, CFG.GROUP_QUARTER)
                    p = apply_mask_words(p, words, mask_value=0.0, n_cols=CFG.GROUP_QUARTER)
        p_all.extend(p[i] for i in range(CFG.GROUP_QUARTER))
        packed.extend(fp32_to_fp16(p[2 * i], p[2 * i + 1], dtype=dtype) for i in range(CFG.GROUP_QUARTER // 2))
    _tmem_store(tmem_base + CFG.P_COL + group * CFG.PACK_GROUP_COLS, cutlass.Vector.from_elements(tuple(packed), cutlass.Int32))
    return cutlass.Vector.from_elements(tuple(p_all), cutlass.Float32)


@cute.jit
def _ds_values(slab, tmem_base, group, p, scale, dtype: cutlass.Constexpr, one_chunk: cutlass.Constexpr[bool]):
    """dP^T -> dS^T = P^T * (dP^T - delta) * scale for the group's two quarters; packed bf16 in place into the group's columns.  Returns the 32
    packed Int32 words (quarter g = words [0, 16), quarter g + 2 = words [16, 32), q ascending inside each): under 16w_fa4 the softmax stores
    them to sDS own + XCHG itself (:func:`_store_words_quarter`); under 16w the value is unused (the epilogue reads the same words back by LDTM).
    ``one_chunk`` (DIAGNOSTIC D10): ONE delta chunk per quarter reused for all eight, as in :func:`_p_values`."""
    q_lo = group * CFG.GROUP_QUARTER
    a = _tmem_load(tmem_base + CFG.DP_COL + q_lo, CFG.GROUP_QUARTER)
    b = _tmem_load(tmem_base + CFG.DP_COL + q_lo + CFG.N // 2, CFG.GROUP_QUARTER)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    packed = []
    for half in cutlass.range_constexpr(CFG.QUARTERS_PER_GROUP):
        dp = (a, b)[half]
        col0 = q_lo + half * (CFG.N // 2)
        if cutlass.const_expr(one_chunk):  # DIAGNOSTIC D10: the one chunk every column of the quarter reuses
            stat_one = _view(slab, CFG.DELTA, cutlass.Float32, CFG.N).subview(col0 + CFG.DIAG_STATS_CHUNK * 4).load(vector_size=4, alignment=16)
        for chunk in cutlass.range_constexpr(CFG.DIAG_STATS_CHUNKS_PER_QUARTER):
            if cutlass.const_expr(one_chunk):
                stat = stat_one
            else:
                stat = _view(slab, CFG.DELTA, cutlass.Float32, CFG.N).subview(col0 + chunk * 4).load(vector_size=4, alignment=16)
            for pair in cutlass.range_constexpr(2):
                i = chunk * 4 + pair * 2
                d0, d1 = ffma2(scale, scale, dp[i], dp[i + 1], stat[pair * 2], stat[pair * 2 + 1])
                d0, d1 = fmul2(d0, d1, p[half * CFG.GROUP_QUARTER + i], p[half * CFG.GROUP_QUARTER + i + 1])
                packed.append(fp32_to_fp16(d0, d1, dtype=dtype))
    words = cutlass.Vector.from_elements(tuple(packed), cutlass.Int32)
    _tmem_store(tmem_base + CFG.DS_COL + group * CFG.PACK_GROUP_COLS, words)
    return words


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
def _stage_fp32_slice(slab, tmem_base, row, live, col, index: cutlass.Constexpr[int], base):
    """dkv_accum = fp32_reduce: fp32 slice ``index`` (DQ_BOX columns) of the accumulator plane at TMEM column ``col``, the lane's kv row -> the
    [M x DQ_BOX] fp32 SW128 slot at byte ``base`` (the `_store_dq_slice` form over 128 rows).  Zeros when the item ran no tile (the store warp
    skips a dead CTA's reduce-adds; the workspace is zero-filled).  The caller fences and arrives once per round."""
    vals = cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(CFG.DQ_BOX)), cutlass.Float32)
    if live:
        vals = _tmem_load(tmem_base + col + index * CFG.DQ_BOX, CFG.DQ_BOX)
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    _store_dq_slice(slab, base, row, vals)


@cute.jit
def _dkv_fp32_stage(slab, bars, tmem_base, lane, live, stage_state, dbg):
    """dkv_accum = fp32_reduce, the epilogue's item end (header section 11; resource-manager 08:25 table 9 form (i)): ROUND 1 = dV's four fp32
    slices + dK's slice 0 over the five slots CFG.dkv_acc_r1_slot(0..4) (sK2's two halves, the dS slab's two halves, XCHG) -> dv_full; then the
    NEW wait dvk_stage_empty (the store warp drained round 1's five source reads) -> ROUND 2 = dK's slices 1..3 over CFG.dkv_acc_r2_slot(0..2)
    -> dk_full.  The dq_empty ring is untouched (its pre-armed wait k <-> release k-1 accounting cannot carry an item-end credit: the first
    build's attempt let round 2 overwrite round 1 under the reduce reads).  Returns the advanced stage_state (one flip per item incl. zero-trip)."""
    # RESOURCE-NOTE: C2b: R1 dV 4 + dK 0 over K2 + ALIAS (80 KiB); R2 dK 1-3 after dvk_stage_empty
    for s in cutlass.range_constexpr(CFG.DKV_ACC_SLICES):
        _stage_fp32_slice(slab, tmem_base, lane, live, CFG.DV_COL, s, CFG.dkv_acc_r1_slot(s))
    _stage_fp32_slice(slab, tmem_base, lane, live, CFG.DK_COL, 0, CFG.dkv_acc_r1_slot(CFG.DKV_ACC_SLICES))
    nvvm.fence_proxy("async.shared", space="cta")
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
    bars.dv_full.arrive()
    # WAR: round 1's five slots were read out by the store warp's reduce-adds.  STAMPS: no record word for this row (DBG_CLK.WORD is the
    # 82-word default-body layout), so the wait is passed dbg=None = the shipped statements, never timed; the item's other waits keep their words.
    _wait(bars.dvk_stage_empty, stage_state.phase, None, 0)
    for s in cutlass.range_constexpr(CFG.DKV_ACC_R2_SLICES):
        _stage_fp32_slice(slab, tmem_base, lane, live, CFG.DK_COL, s + 1, CFG.dkv_acc_r2_slot(s))
    nvvm.fence_proxy("async.shared", space="cta")
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
    bars.dk_full.arrive()
    return advance(stage_state, 1)


@cute.jit
def _dkv_fp32_reduce_rounds(slab, bars, maps, batch, n_batch, kv_row0, kv_head, live_rows, out_phase, store_wait_spin: cutlass.Constexpr[bool], dbg):
    """dkv_accum = fp32_reduce, the store warp's item end (after its dv_full wait): ROUND 1 = five reduce-adds (dV's four slices + dK's slice 0)
    into the per-KV-head fp32 workspace maps (the MAP_DV / MAP_DK slots: 128-row fp32 [M x DQ_BOX]-box maps; kv head = head // ratio) as ONE
    bulk group -> drain (P7) -> dvk_stage_empty.arrive() (bare 32: the epilogue's round 2 may overwrite the slots) -> the dk_full wait -> ROUND 2
    = three reduce-adds (dK's slices 1..3; their drain = the caller's item-end `tma_store_wait(0)` before dvk_empty / k2_empty).  A dead CTA
    (kv_row0 >= seq_k) issues nothing and acquires no map (the workspace is zero-filled); every commit / wait / arrive is kept."""
    # RESOURCE-NOTE: 8 UTMAREDG fp32 boxes per item replace 4 UTMASTG; issue under live_rows, flips not
    dv_desc = _map_ptr(maps, batch, n_batch, MAP_DV)
    dk_desc = _map_ptr(maps, batch, n_batch, MAP_DK)
    if live_rows:
        tma_tensormap_acquire(dv_desc)
        tma_tensormap_acquire(dk_desc)
        for i in cutlass.range_constexpr(CFG.DKV_ACC_SLICES):
            tma_reduce_add_subtile(
                _dkv_acc_tile(slab, CFG.dkv_acc_r1_slot(i)), tma_slice_runtime_desc(dv_desc, i * CFG.DQ_BOX, kv_row0, kv_head, 0), 0, acquire=False
            )
        tma_reduce_add_subtile(
            _dkv_acc_tile(slab, CFG.dkv_acc_r1_slot(CFG.DKV_ACC_SLICES)), tma_slice_runtime_desc(dk_desc, 0, kv_row0, kv_head, 0), 0, acquire=False
        )
    tma_store_commit()
    tma_store_wait(0)
    bars.dvk_stage_empty.arrive()  # all 32 after round 1's five source reads: the epilogue's round 2 may overwrite the slots
    _wait_store(bars.dk_full, out_phase, store_wait_spin, dbg, _W["wait:store:dk_full"])
    if live_rows:
        for i in cutlass.range_constexpr(CFG.DKV_ACC_R2_SLICES):
            tma_reduce_add_subtile(
                _dkv_acc_tile(slab, CFG.dkv_acc_r2_slot(i)), tma_slice_runtime_desc(dk_desc, (i + 1) * CFG.DQ_BOX, kv_row0, kv_head, 0), 0, acquire=False
            )


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
def _s_lookahead(slab, bars, score, k1d, s_acc, q1_next, s_tail_state, elect_p, qtile, lo, hi, dtype: cutlass.Constexpr, dbg):
    """S^T(t + 1) into the S slot at slot 4 of tile t (after dV(t): the in-order tensor pipe reads P^T(t) first): _wait_tcgen(q1_full[next])
    -> [iff qtile > lo: _wait_tcgen(s_tail_empty) + advance -- dQ(t-1) was issued into [64, 128) at slot 2 of THIS tile and the epilogue
    must have read it out (D1 ii); at tile lo nothing was issued into the tail since S^T(lo), whose softmax read p_ready(lo) already covers]
    -> mma_ss -> _commit_pair(s_done), _commit_pair(q1_empty[next]).  Skipped on the item's last tile (nothing waits those flips).
    Returns the s_tail state (one flip per wait).  RESOURCE-NOTE: each S^T issue after a dQ waits s_tail_empty (dQ read out of the S tail)."""
    if qtile + 1 < hi:
        _wait_tcgen(bars.q1_full[q1_next.idx], q1_next.phase, dbg, _W["wait:leader:q1_full"])
        if qtile > lo:
            _wait_tcgen(bars.s_tail_empty, s_tail_state.phase, dbg, _W["wait:leader:s_tail_empty"])
            s_tail_state = advance(s_tail_state, 1)
        q1d_next = _operand_k_major(slab, CFG.Q1 + q1_next.idx * CFG.Q1_BYTES, dtype, CFG.Q_HALF).desc()
        mma_ss(score, k1d, q1d_next, s_acc)
        _commit_pair(bars.s_done, elect_p)
        _commit_pair(bars.q1_empty[q1_next.idx], elect_p)
    return s_tail_state


@cute.jit
def _dq_issue(bars, dqdesc, dsd, k2d, dq_acc, xchg_state, elect_p, fa4_roles: cutlass.Constexpr[bool], dbg, tag_back: cutlass.Constexpr[int]):
    """ONE dQ issue (dQ(t-1) at slot 2 of tile t, or the tail dQ(hi-1)): the three dS gates of the exchange on ``xchg_state`` (the P2 split:
    one flip per dQ ISSUE, one tile behind ready_state), the M 128 pair MMA into [DQ_COL, +64) = S^T's dead tail, the dq_done commit.
    The three waits are UNCONDITIONAL under every probe: the ds_recv_full wait (CTA 0) / the ds_pair_ready wait fed by the relay (CTA 1) is
    the ONLY back-edge of the ds_recv_full expect_tx ARM (P14-B: arm(t+1) <- dq_done(t) <- this site); the D8 probe that skipped
    them let arm(t+1) land on a tx-pending phase and FAULTED (CUDA 719) -- it is declined by name on this map and under every role shape
    (barrier-inspector 2026-10-27 08:01; header section 5).  ``dbg`` / ``tag_back`` = STAMPS: the wait sums and the ship -> recv handshake
    (:func:`_clk_recv`).  ``fa4_roles`` (16w_fa4; barrier-inspector x-fa4 table section 15): ALSO `_commit_pair(bars.dq_war_done)` IMMEDIATELY
    BEFORE the dq_done commit -- the softmax's sDS WAR signal, flipped at the two dQ issue sites ONLY (never at the zero-trip empty-commit site),
    so every flip of the softmax's pre-armed waiter depends on its own preceding dS STS (ds_smem) and can never land early."""
    _wait_tcgen(bars.ds_smem, xchg_state.phase, dbg, _W["wait:leader:ds_smem"])
    _wait_tcgen(bars.ds_recv_full, xchg_state.phase, dbg, _W["wait:leader:ds_recv_full"])
    if cutlass.const_expr(dbg is not None):
        _clk_recv(dbg, _W["tiles_leader"], _W["recv_leader_sum"], _W["recv_leader_matched"], _W["recv_leader_missed"], tag_back)
    _wait_tcgen(bars.ds_pair_ready, xchg_state.phase, dbg, _W["wait:leader:ds_pair_ready"])
    # skew lever 7 (SOFTMAX_DQ_WAR): the LEADER holds the dQ issue -- under 16w_fa4 both softmaxes' dS STS wait dq_war_done ALONE
    if cutlass.const_expr(DEBUG_SOFTMAX_DQ_WAR_DELAY_US > 0):
        for _us in cutlass.range(DEBUG_SOFTMAX_DQ_WAR_DELAY_US, unroll=1):
            nvvm.inline_ptx("nanosleep.u32 1000;", read_only_args=[])
    mma_ss(dqdesc, dsd, k2d, dq_acc)
    if cutlass.const_expr(fa4_roles):
        _commit_pair(bars.dq_war_done, elect_p)  # section 15: the softmax's WAR signal (the two dQ sites only), BEFORE dq_done
    _commit_pair(bars.dq_done, elect_p)
    return advance(xchg_state, 1)


@cute.jit
def _dq_slot(bars, dqdesc, dsd, k2d, dq_acc, xchg_state, ready_state, elect_p, qtile, lo, fa4_roles: cutlass.Constexpr[bool], dbg):
    """The in-tile dQ SLOT of tile t (slot 2 under fa4, slot 3 under fa4b: ONE helper at the two `const_expr(dq_after_dv)` sites of the leader
    so the orders cannot drift): _wait_tcgen(s_read(t)) on ready_state EVERY tile (D2: both softmaxes hold S^T(t)'s tail in registers before
    dQ(t-1) overwrites [DQ_COL, +DQ_COLS); under fa4b the wait is DOMINATED by the p_ready(t) wait just consumed -- the same 16 softmax warps
    arrive s_read before p_ready in program order -- and is still consumed: P14, N / N), then iff qtile > lo the dQ(t-1) issue behind the three
    dS gates (_dq_issue: xchg_state +1 per issue).  Returns xchg_state.  RESOURCE-NOTE: the dQ image never overlaps dV's P^T read [P_COL, +64)."""
    _wait_tcgen(bars.s_read, ready_state.phase, dbg, _W["wait:leader:s_read"])
    if qtile > lo:
        xchg_state = _dq_issue(bars, dqdesc, dsd, k2d, dq_acc, xchg_state, elect_p, fa4_roles, dbg, DBG_CLK.TAG_BACK_IN_TILE)
    return xchg_state


@cute.jit
def _dq_round(slab, bars, slot_base, dhalf, qrow, values, empty_state, route_red: cutlass.Constexpr[bool], sts_off: cutlass.Constexpr[bool], dbg):
    """One dQ staging ROUND of the epilogue warp pair of d half ``dhalf`` (D4 pair: the store's closed per-round release + this both-slots
    wait form, change both or neither): every warp waits BOTH slots' release at the round's phase, STS the lane's DQ_BOX fp32 into slot
    dhalf, fence_proxy, publish dq_full[dhalf] (2 warps x 32 = DQ_SLICE_LANES); ONE flip of empty_state per round.
    ``route_red`` (dq_route_pair red_global | red_global_pair): the words already left through the LSU global path (``_dq_red_out``), so the
    STS and its proxy fence are skipped; ``sts_off`` (DIAGNOSTIC D9 `sdq_sts_off`) skips the SAME two statements with the TMA route kept (the
    store warp's reduce-add then reads stale staging: -256 wavefronts, dQ WRONG BY DESIGN); under both the waits and the arrive stay -- the
    ring PROTOCOL is unchanged (header section 5).  ``dbg`` = the STAMPS bracket of the two dq_empty waits (one word per round)."""
    for s in cutlass.range_constexpr(CFG.DQ_SLOTS):
        _wait(bars.dq_empty[s], empty_state.phase, dbg, _W["wait:epilogue:dq_empty"])
    if cutlass.const_expr(not route_red and not sts_off):
        _store_dq_slice(slab, slot_base, qrow, values)
        nvvm.fence_proxy("async.shared", space="cta")
    bars.dq_full[dhalf].arrive()
    return advance(empty_state, 1)


@cute.jit
def _dq_red_out(dq, row_addr_even, row_stride_bytes, parity, live, live_even, live_odd, dq_route_code: cutlass.Constexpr[int]):
    """The register-direct dQ route (header section 11; ``dq_route_pair`` red_global | red_global_pair): the lane's CFG.DQ_RED_ROW_WORDS = 64 fp32
    of ONE q row (the half-row image: d = D_HALF * dhalf + c) leave the CTA as ``red.global.add.v4.f32`` straight from the tcgen05.ld registers.
    ``row_addr_even`` = the byte address (dq_accum, column D_HALF * dhalf) of the EVEN row of the lane pair (2k, 2k+1) -- the lane's own row
    is ``parity`` token rows further; ``live`` / ``live_even`` / ``live_odd`` = the rows are inside the sequence (the TMA descriptor's GLOBAL_DIM
    clip as a BRANCH: a padded q row is never written).  Plain form: 16 x v4 per lane (each a 16-B half of a 32-B sector); paired form: 8 x
    the sector-paired pair (tile_dsl.atomic), every warp-level red a full sector.  No barrier, no fence: fire-and-forget (section 5)."""
    if cutlass.const_expr(dq_route_code == DQ_ROUTE_RED):
        row_addr = row_addr_even + cutlass.Int64(parity) * row_stride_bytes
        if live:
            for g in cutlass.range_constexpr(CFG.DQ_RED_V4_PER_LANE):
                red_global_add_v4_f32(row_addr + cutlass.Int64(g * RED_V4_WORDS * CFG.BPE_ACC), tuple(dq[g * RED_V4_WORDS + k] for k in range(RED_V4_WORDS)))
    else:
        for sct in cutlass.range_constexpr(CFG.DQ_RED_SECTORS_PER_LANE):
            red_global_add_f32x8_sector_paired(
                row_addr_even + cutlass.Int64(sct * SECTOR_F32_WORDS * CFG.BPE_ACC), row_stride_bytes, parity, live_even, live_odd, dq, sct * SECTOR_F32_WORDS
            )


# ---------------------------------------------------------------------------------------------------------------------
# Role bodies (section 4).  The ``# BARRIER`` punch lists are the inspector's issue-order contract (resources/arm2-barrier-table.md
# section 7); the code follows them line by line.
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
    warp,
    scale,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    diag_2cta_code: cutlass.Constexpr[int],
    fa4_roles: cutlass.Constexpr[bool],
    regs: cutlass.Constexpr[int],
    mask_form_code: cutlass.Constexpr[int],
    dbg,
):
    """Warps 0-7, BOTH CTAs (lane = own kv row; group g owns q quarters {g, g+2}): S^T / dP^T -> P^T / dS^T packed in place; under
    ``fa4_roles`` (warp_roles 16w_fa4, X-FA4) ALSO the dS exchange: sDS own half + XCHG from the same registers, the ship, the sDS WAR wait.
    ``diag_2cta_code`` (DIAGNOSTIC): DIAG_STATS_ONE_CHUNK (D10) = `_p_values` / `_ds_values` load one stats / delta chunk per quarter and
    reuse it (stats WRONG BY DESIGN; no barrier touched); DIAG_EXCHANGE_OFF (D8s) under 16w_fa4 = this role's arm expects 0 bytes and the ship
    is not issued (the 16w site is the epilogue's).  ``mask_form_code`` (header section 11, lever L5): under MASK_FORM_PREDICATED the
    per-tile CTA-uniform scalar ``mask_tile`` = tile_may_mask(pair, rank, qtile, seq_q, seq_k, causal) (section 7) gates the mask body of
    _p_values; the barrier program is identical under every form (no wait, arrive or fence moves).  ``dbg`` (STAMPS, warp 0 = the
    representative; None under 16w_fa4 -- the lever declines there): the s_done / dp_done / sched_full wait sums; None = no traced code.

    # BARRIER (per item, FIRST thing): _credit(bars.sched_empty[sched_state.idx])  -- before any work (every role: the top-of-item
    #   credit is what keeps the peer provably resident for the remote credits, P15)
    # BARRIER (per tile t, tile_state st(0)):
    #   _wait_tcgen(bars.s_done, tile_state.phase) -> [test-only DEBUG_SOFTMAX_S_READ_DELAY_US sleep, FOLLOWER only, default none]
    #   -> _s_values: tcgen05.ld S^T quarters a, b -> tcgen05_wait(LOAD) -> tcgen05_fence(BEFORE_THREAD_SYNC)
    #   -> _arrive_leader(bars.s_read)   [NEW; LEADER, bare, EVERY tile (D2), 256 x 2 CTAs = 512: the S^T registers are held, dQ(t-1) may land]
    #   -> _p_values (exp2 / mask / pack, tcgen05.st into [0, 64)) -> tcgen05_wait(STORE) -> tcgen05_fence(BEFORE_THREAD_SYNC)
    #   -> _arrive_leader(bars.p_ready)                 [LEADER, bare, 512]
    #   _wait_tcgen(bars.dp_done, tile_state.phase) -> dS^T pack -> tcgen05_wait(STORE) -> fence
    #   -> _arrive_leader(bars.ds_ready)  [LEADER, 512]  and  bars.ds_tmem.arrive()          [THREAD local, bare, 256]
    #   tile_state = advance(tile_state, 1)
    # Required waits: s_done, dp_done (own copies of the leader's multicast commits).  Zero-trip item: credit only (pair-uniform lo / hi).
    # Non-mbarrier: _stats -> named_barrier_fence(CFG.SOFTMAX_BARRIER, CFG.SOFTMAX_LANES) x 2 per call (both CTAs hold all 128 q stats).
    # BARRIER under fa4_roles (16w_fa4; barrier-inspector x-fa4 table section 9, the softmax stub order; dq_war = PipelineState.start(phase=1)):
    #   per item, first: _credit; ship_pred = (warp == SOFTMAX_LEAD_WARP) & nvvm.elect_sync()   [ONCE per item, P16]
    #   per tile t, steps 1-3 as above up to _arrive_leader(bars.ds_ready) [STTM + wait(STORE) + fence BEFORE the WAR wait: dK(t) never waits dQ(t-1)];
    #     bars.ds_tmem.arrive() DELETED (no field);
    #   4. _wait(bars.dq_war_done, dq_war.phase); dq_war = advance(dq_war, 1)   [NEW: plain _wait, no tcgen05 fence; the sDS WAR: dQ(t-1) read own +
    #      recv, both ships read XCHG; wait k consumes flip k-1 (the previous live tile's / item's dQ): the kernel's first dS(lo) passes on the pre-arm;
    #      section 15: the barrier is dq_war_done (flipped by dQ issues ONLY), so no flip can land before its wait and no lap is possible]
    #   5. _store_words_quarter(slab, lo_base, row, words, 0, vec0) [quarter g: q half 0 -> own on rank 0 / XCHG on rank 1];
    #      _store_words_quarter(slab, hi_base, row, words, PACK_QUARTER_COLS, vec0) [quarter g+2: q half 1 -> XCHG on rank 0 / own on rank 1]
    #      -> nvvm.fence_proxy("async.shared", space="cta") [ONE] -> bars.ds_smem.arrive()   [THREAD local, bare, 8 warps = 256 == SOFTMAX_LANES]
    #   6. named_barrier_fence(CFG.XCHG_BARRIER, XCHG_BARRIER_LANES_16W_FA4) [bar.sync 2, 256; P11 KEEP: the ship reads all 8 warps' XCHG stores]
    #      -> [test-only LEADER_SHIP sleep, LEADER only] -> bars.ds_recv_full.arrive(n_bytes=CFG.XCHG_TX, pred=ship_pred) [1 == init; the PEER's
    #      copy delivers the bytes; back-edge = step 4] -> _ship_xchg(slab, bars, rank, ship_pred)
    #   7. if qtile + 1 < hi: _stats(qtile + 1); tile_state = advance(tile_state, 1)
    #   item end: NOTHING on dq_war (section 15: a zero-trip item has 0 flips / 0 waits on dq_war_done; a live item's tail flip is consumed by the
    #     next live item's tile lo' or by the drain)
    #   after the persistent loop: _wait(bars.dq_war_done, dq_war.phase)   [the ONE unconditional drain; no advance]
    """
    # RESOURCE-NOTE: s_read (512, every tile) after S^T ld + wait(LOAD): dQ(t-1) lands in S^T's tail
    nvvm.setmaxregister(regs, nvvm.SetMaxRegisterAction.INCREASE)  # CFG.REG_SPLITS[reg_split][0] (knob; default SOFTMAX_REGS)
    tid = cute.arch.thread_idx()[0]
    row = tid % CFG.M
    group = tid // CFG.M
    one_chunk = cutlass.const_expr(diag_2cta_code == DIAG_STATS_ONE_CHUNK)  # DIAGNOSTIC D10: one LSE / delta chunk per quarter reused
    # DIAGNOSTIC D8s (exchange_off) RE-SITED for 16w_fa4 (the stack fold): this role arms ds_recv_full with 0 bytes and issues no ship; every wait stays
    exchange_off = cutlass.const_expr(diag_2cta_code == DIAG_EXCHANGE_OFF)
    xchg_tx = CFG.DIAG_EXCHANGE_OFF_TX if cutlass.const_expr(diag_2cta_code == DIAG_EXCHANGE_OFF) else CFG.XCHG_TX
    tile_state = PipelineState.start()
    sched_state = PipelineState.start()
    if cutlass.const_expr(fa4_roles):
        # RESOURCE-NOTE: lo_base=rank?XCHG:DS, hi_base=rank?DS+16K:XCHG; vec 4g..4g+3; (row%8)^vec: floor
        own_base = cutlass.Int32(CFG.DS) + rank * cutlass.Int32(CFG.DS_HALF_BYTES)  # CFG.ds_own(rank): our 128 kv rows of sDS
        xchg_base = cutlass.Int32(CFG.XCHG)  # the peer's 64 q of our rows: the ship's source
        is_leader_cta = rank == cutlass.Int32(LEADER_RANK)
        lo_base = cutlass.Int32(arith.select(is_leader_cta.ir_value(), own_base.ir_value(), xchg_base.ir_value()))  # quarter g = q half 0 (rank 0's own)
        hi_base = cutlass.Int32(arith.select(is_leader_cta.ir_value(), xchg_base.ir_value(), own_base.ir_value()))  # quarter g+2 = q half 1 (rank 1's own)
        lo_row = lo_base + row * cutlass.Int32(CFG.SWIZZLE_BYTES)  # the lane's 128-B row of the quarter-g slab (item-invariant: 1 register)
        hi_row = hi_base + row * cutlass.Int32(CFG.SWIZZLE_BYTES)  # the lane's row of the quarter-(g+2) slab (1 register)
        xor_lane = (row % 8) ^ (group * cutlass.Int32(CFG.DS_STS_VECS_PER_QUARTER))  # (row % 8) ^ vec0: group g writes vec 4g .. 4g+3 of BOTH halves
        is_lead_warp = warp == cutlass.Int32(SOFTMAX_LEAD_WARP)
        dq_war = PipelineState.start(phase=1)  # pre-armed (P5): the kernel's first dS STS has no predecessor; +1 per wait, every tile + zero-trip
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        if cutlass.const_expr(fa4_roles):
            ship_pred = is_lead_warp & nvvm.elect_sync()  # ONE lane of warp SOFTMAX_LEAD_WARP per item (P16): the arm + the ship
        lo, hi, seq_q, seq_k, q_start = _item_bounds(pair, meta, batch, n_batch, causal)
        key = pair * CFG.PAIR_KV_ROWS + rank * CFG.M + row
        if lo < hi:
            _stats(slab, lse, delta, row, group, lo * CFG.N, batch, head, scale, q_start, seq_q, origins, stats_origin)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            _wait_tcgen(bars.s_done, tile_state.phase, dbg, _W["wait:softmax:s_done"])
            if cutlass.const_expr(DEBUG_SOFTMAX_S_READ_DELAY_US > 0):
                if rank != cutlass.Int32(LEADER_RANK):  # skew lever 5: the FOLLOWER softmax holds its S^T read (s_read alone holds dQ(t-1))
                    for _us in cutlass.range(DEBUG_SOFTMAX_S_READ_DELAY_US, unroll=1):
                        nvvm.inline_ptx("nanosleep.u32 1000;", read_only_args=[])
            a, b = _s_values(tmem_base, group)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            _arrive_leader(bars.s_read)
            mask_tile = None  # read under MASK_FORM_PREDICATED only (section 7's CTA-uniform per-tile scalar; the other forms trace no predicate)
            if cutlass.const_expr(mask_form_code == MASK_FORM_PREDICATED):
                mask_tile = tile_may_mask(pair, rank, qtile, seq_q, seq_k, causal)
            p = _p_values(slab, tmem_base, group, key, qtile, seq_q, seq_k, scale, a, b, causal, dtype, one_chunk, mask_form_code, mask_tile)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            _arrive_leader(bars.p_ready)
            _wait_tcgen(bars.dp_done, tile_state.phase, dbg, _W["wait:softmax:dp_done"])
            words = _ds_values(slab, tmem_base, group, p, scale, dtype, one_chunk)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            _arrive_leader(bars.ds_ready)
            if cutlass.const_expr(not fa4_roles):
                bars.ds_tmem.arrive()
            else:
                # RESOURCE-NOTE: STTM + ds_ready BEFORE the dq_done WAR wait: dK(t) never waits dQ(t-1)
                # RESOURCE-NOTE: WAR: wait dq_war_done (dQ issues only, never the zero-trip commit); st(1); drain after the loop
                # section 15: dQ(t-1)'s read of sDS own + recv (and, via the leader's gates, both ships' XCHG reads) done.  STAMPS declines under
                # 16w_fa4 (no record word for this row): dbg=None = the shipped statements, untimed.
                _wait(bars.dq_war_done, dq_war.phase, None, 0)
                dq_war = advance(dq_war, 1)
                _skew_follower_own_half(rank)  # skew lever 1 (16w_fa4 site): the FOLLOWER holds its 8 STS (its ds_smem AND its ship)
                # RESOURCE-NOTE: dS STS from the 32 packed words = today's LDTM image; 4 own + 4 XCHG STS.128/lane
                xor_t = opaque_i32(xor_lane, anchor=qtile)  # pins the 8 address adds inside the tile (hoisted per item they spilled 4 words at 144 regs)
                _store_words_quarter(slab, lo_row, words, 0, xor_t)
                _store_words_quarter(slab, hi_row, words, CFG.PACK_QUARTER_COLS, xor_t)
                nvvm.fence_proxy("async.shared", space="cta")
                # RESOURCE-NOTE: ds_smem 256 THREAD bare after own STS + fence.proxy; no tcgen05 fence (registers)
                bars.ds_smem.arrive()
                # RESOURCE-NOTE: XCHG_BARRIER 256 (8 softmax warps) -> warp 0 elect: arm ds_recv_full, ship XCHG
                named_barrier_fence(CFG.XCHG_BARRIER, XCHG_BARRIER_LANES_16W_FA4)
                _skew_leader_ship(rank)  # skew lever 2 (16w_fa4 site): the LEADER holds its arm + ship (the half is written, the barrier passed)
                # BARRIER (the arm's back-edge on EVERY path, section 15.2 row 3): arm(t) follows this warp's dq_war_done wait above = dQ(t-1) issued
                # after the leader's ds_recv_full(t-1) / ds_pair_ready(t-1) waits (both copies of the previous phase landed); across items the first
                # tile's wait consumes the previous LIVE item's tail flip (a real dQ behind the leader's gates); a zero-trip item adds no phase
                # (no tile, no arm, no ship) and no flip -- the edge is a DATA edge through a real dQ, never through the zero-trip empty commit.
                bars.ds_recv_full.arrive(n_bytes=xchg_tx, pred=ship_pred)  # XCHG_TX; DIAGNOSTIC D8s arms CFG.DIAG_EXCHANGE_OFF_TX = 0 (completes at the arm)
                if cutlass.const_expr(not exchange_off):  # DIAGNOSTIC D8s (re-sited): the ship only; the arm above expects 0 bytes, every wait stays
                    _ship_xchg(slab, bars, rank, ship_pred)
            if qtile + 1 < hi:
                _stats(slab, lse, delta, row, group, (qtile + 1) * CFG.N, batch, head, scale, q_start, seq_q, origins, stats_origin)
            tile_state = advance(tile_state, 1)
        # section 15: NO zero-trip wait and NO zero-trip advance on dq_war -- dq_war_done flips at dQ issues only (0 / 0 on every zero-trip path)
        pair, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, dbg, _W["wait:softmax:sched_full"])
    if cutlass.const_expr(fa4_roles):
        # P15 / section 15.3: the ONE unconditional drain of dq_war_done (the kernel's last tail flip, or the pre-arm when this CTA ran no live item);
        # the leader's last commit is issued before it leaves its loop and depends on nothing of this warp: the drain cannot hang.
        _wait(bars.dq_war_done, dq_war.phase, None, 0)  # STAMPS declines under 16w_fa4: no record word for this row (passed dbg=None, untimed)


@cute.jit
def _epilogue_loop(
    slab,
    bars,
    sched,
    tmem_base,
    dq_accum,
    pair,
    head,
    batch,
    meta,
    n_batch,
    rank,
    warp,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    dq_route_code: cutlass.Constexpr[int],
    diag_2cta_code: cutlass.Constexpr[int],
    regs: cutlass.Constexpr[int],
    dkv_fp32: cutlass.Constexpr[bool],
    dbg,
):
    """Warps 8-11, BOTH CTAs (lane = own kv row for dS; lane = q row of d half (warp // 2) % 2 for dQ): dS^T -> sDS own half + XCHG
    (the 16-B STS form), the DSMEM ship, the dQ fp32 rounds into the DEDICATED sDQ ring, the native dV / dK stores.
    ``diag_2cta_code`` (DIAGNOSTIC, header section 11): DIAG_EXCHANGE_OFF (D8s) = the ship is NOT issued and the ds_recv_full arm expects
    CFG.DIAG_EXCHANGE_OFF_TX = 0 bytes (each phase completes at its own arm; every wait, the relay, the XCHG STS / fence / named barrier
    stay; the recv half is never written); DIAG_SDQ_STS_OFF (D9) = `_dq_round` skips the slice STS + its fence (the reduce-add reads stale
    staging).  ``dbg`` (STAMPS; the LEAD warp = the representative): the wait sums, the per-tile hand-off segments, the ship stamp.

    # BARRIER (per item, first): _credit(bars.sched_empty[sched_state.idx]);  ship_pred = is_lead_warp & nvvm.elect_sync();  carried = zeros
    # BARRIER (per tile t):
    #   dS(t) staging: _wait_tcgen(bars.ds_tmem, ds_state.phase) -> tcgen05.ld dS^T (4 x 16 packed cols) -> tcgen05_wait(LOAD)
    #     -> tcgen05_fence(BEFORE_THREAD_SYNC) -> _arrive_leader(bars.dp_empty)   [MOVED HERE: LEADER, bare, 128 x 2 = 256; dP^T(t+1) never waits dQ]
    #     -> bars.ds_recv_full.arrive(n_bytes=CFG.XCHG_TX, pred=ship_pred)   [1 == init; the PEER's copy delivers the bytes]
    #     -> STS XCHG (the peer's 64 q) -> fence_proxy -> named_barrier_fence(CFG.XCHG_BARRIER, CFG.EPILOGUE_LANES)   [P11 KEEP]
    #     -> [test-only DEBUG_LEADER_SHIP_DELAY_US sleep, LEADER only] -> _ship_xchg(slab, bars, rank, ship_pred)
    #     -> [test-only DEBUG_FOLLOWER_OWN_HALF_DELAY_US sleep, FOLLOWER only] -> STS own half -> fence_proxy -> tcgen05 fence
    #     -> bars.ds_smem.arrive()   [THREAD local, bare, 128].  NO dq_empty wait anywhere in this block (the staging alias is deleted).
    #   round B of dQ(t-1): if qtile > lo: _dq_round(carried) = _wait(dq_empty[0]) + _wait(dq_empty[1]) at empty_state.phase -> STS slot dhalf
    #     -> fence_proxy -> bars.dq_full[dhalf].arrive()   [THREAD local, 2 warps = 64 == DQ_SLICE_LANES] -> empty_state += 1
    #   dQ(t) readout: _wait_tcgen(bars.dq_done, dq_state.phase) -> [test-only DEBUG_EPILOGUE_DQ_READOUT_DELAY_US sleep, FOLLOWER only]
    #     -> tcgen05.ld dQ (64 cols at CFG.DQ_COL = 64, the half-row image) -> tcgen05_wait(LOAD) -> tcgen05_fence(BEFORE_THREAD_SYNC)
    #     -> _arrive_leader(bars.s_tail_empty)   [NEW: LEADER, bare, 128 x 2 = 256: S^T(t+1) / the tail dQ / the next prologue may overwrite]
    #     -> [dq_route_pair red_global | red_global_pair ONLY: _dq_red_out = `if row < seq_q:` 16 x red.global.add.v4.f32 of the lane's 64 words
    #        straight from `dq` (no barrier, no fence, no SMEM); every _dq_round below then runs WITHOUT its STS: the waits + arrives stay]
    #   round A of dQ(t): _dq_round(dq[0, 32)) (the same form) -> carried = dq[32, 64);  ds_state, dq_state = advance(.., 1)
    # BARRIER (item end): if lo < hi: round B of dQ(hi-1) (_dq_round(carried))  else: _wait_tcgen(bars.dq_done, dq_state.phase) + advance
    #   -> [test-only DEBUG_FOLLOWER_EPILOGUE_ITEM_END_DELAY_US sleep, FOLLOWER only]
    #   -> native dV over K2 (CFG.dv_store(half); live=False writes zeros) -> fence_proxy + fence -> bars.dv_full.arrive()   [128 of
    #      DV_FULL_LANES 160: the loader adds 32 after its k2_empty observation of the item, fix A]
    #   -> native dK over ds_own(rank) + XCHG (CFG.dk_store(rank, half)) -> bars.dk_full.arrive()   [128]
    #   NO terminal dq_empty observation, NO item-end empty_state flip (D3: 2N rounds == 2N releases per slot per item; zero-trip 0 / 0).
    #   dkv_fp32 (dkv_accum = fp32_reduce, `_dkv_fp32_stage`): round 1 = dV slices 0-3 + dK slice 0 over CFG.dkv_acc_r1_slot(0..4) -> fence -> bars.dv_full.arrive()
    #   -> _wait(bars.dvk_stage_empty, stage_state.phase) [NEW row: the store warp's post-round-1 drain] -> round 2 = dK slices 1-3 over
    #   CFG.dkv_acc_r2_slot(0..2) -> fence -> bars.dk_full.arrive() -> stage_state += 1   [1 wait / 1 release per item incl. zero-trip; the dq_empty ring untouched].
    # No drain: every arrive of this role is local or waited by the leader in-loop (dp_empty's / s_tail_empty's last phases are the leader's drains).
    """
    # RESOURCE-NOTE: XCHG -> peer DS_RECV by one cp.async.bulk (16 KiB), complete_tx on the peer's ds_recv_full.
    # RESOURCE-NOTE: epilogue: dS(t) stage -> dQ(t-1) round B -> dQ(t) ld + round A; 32 fp32 carried
    nvvm.setmaxregister(regs, nvvm.SetMaxRegisterAction.INCREASE)  # CFG.REG_SPLITS[reg_split][1] (knob; default EPILOGUE_REGS)
    lane = cute.arch.thread_idx()[0] % CFG.M  # own kv row for dS^T; TMEM lane of the dQ half-row image
    qrow = lane % CFG.Q_HALF  # dQ row of the half-row image
    dhalf = lane // CFG.Q_HALF  # d half this warp pair holds (warp-uniform) = its sDQ slot
    is_lead_warp = warp == cutlass.Int32(EPILOGUE_LEAD_WARP)
    own_base = cutlass.Int32(CFG.DS) + rank * cutlass.Int32(CFG.DS_HALF_BYTES)  # CFG.ds_own(rank)
    peer_q = cutlass.Int32(1) - rank  # the q half the PEER needs = the XCHG half
    own_col = cutlass.Int32(CFG.DS_COL) + rank * cutlass.Int32(CFG.PACK_QUARTER_COLS)
    xchg_col = cutlass.Int32(CFG.DS_COL) + peer_q * cutlass.Int32(CFG.PACK_QUARTER_COLS)
    slot_base = cutlass.Int32(CFG.DQ) + dhalf * cutlass.Int32(CFG.DQ_SLICE_BYTES)  # CFG.dq_slot(dhalf): the warp pair's slot of the ring
    route_red = cutlass.const_expr(dq_route_code != DQ_ROUTE_TMA)
    sts_off = cutlass.const_expr(diag_2cta_code == DIAG_SDQ_STS_OFF)  # DIAGNOSTIC D9: the sDQ slice STS + its proxy fence skipped in every round
    exchange_off = cutlass.const_expr(diag_2cta_code == DIAG_EXCHANGE_OFF)  # DIAGNOSTIC D8s: no ship; the arm expects 0 bytes; every wait stays
    xchg_tx = CFG.DIAG_EXCHANGE_OFF_TX if cutlass.const_expr(diag_2cta_code == DIAG_EXCHANGE_OFF) else CFG.XCHG_TX
    if cutlass.const_expr(route_red):
        # RESOURCE-NOTE: red route address = dq_accum + ((cu_q[b] + row) * stride_t + head * stride_h + dhalf * D_HALF) fp32: cast_dq's spelling
        q_half = rank * cutlass.Int32(CFG.Q_HALF)  # the CTA's q half of the tile = the store warp's box row coordinate
        parity = qrow % 2  # lane pair (2k, 2k+1) = q rows (2k, 2k+1): the paired form's sector partner
        dq_stride_t = cutlass.Int64(dq_accum.stride[1])  # fp32 per token row (heads x D): the SAME row stride emit_seq_descs gave the dQ descriptor
        dq_stride_h = cutlass.Int64(dq_accum.stride[2])
        dq_row_stride_bytes = dq_stride_t * cutlass.Int64(CFG.BPE_ACC)
    ds_state = PipelineState.start()
    dq_state = PipelineState.start()
    empty_state = PipelineState.start(phase=1)  # pre-armed (P5 b): both slots of the ring are free before the first round; NO bootstrap
    if cutlass.const_expr(dkv_fp32):
        stage_state = (
            PipelineState.start()
        )  # C2b: dvk_stage_empty, +1 per item (incl. zero-trip); defined under the field only: the default rendering is byte-identical
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        if cutlass.const_expr(dbg is not None):
            _clk_add(dbg, _W["items"], cutlass.Int32(1))
        ship_pred = is_lead_warp & nvvm.elect_sync()
        lo, hi, seq_q, seq_k, q_start = _item_bounds(pair, meta, batch, n_batch, causal)
        carried = _dq_zero_half()
        if cutlass.const_expr(route_red):
            dq_item_base = (
                dq_accum.iterator.raw_ptr() + (cutlass.Int64(q_start) * dq_stride_t + cutlass.Int64(head) * dq_stride_h + cutlass.Int64(dhalf * CFG.D_HALF))
            ).toint(
                cutlass.Int64
            )  # byte address of the sequence's row 0 at this lane's d half (the descriptor's GLOBAL_ADDRESS + the head / d offsets)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            _wait_tcgen(bars.ds_tmem, ds_state.phase, dbg, _W["wait:epilogue:ds_tmem"])
            if cutlass.const_expr(dbg is not None):
                t_ds = cute.arch.clock64()  # STAMPS: the dS^T(t) readout may begin = the hand-off chain's origin
            x0 = _tmem_load(tmem_base + xchg_col, CFG.PACK_QUARTER_COLS, cutlass.Int32)
            x1 = _tmem_load(tmem_base + xchg_col + CFG.PACK_GROUP_COLS, CFG.PACK_QUARTER_COLS, cutlass.Int32)
            o0 = _tmem_load(tmem_base + own_col, CFG.PACK_QUARTER_COLS, cutlass.Int32)
            o1 = _tmem_load(tmem_base + own_col + CFG.PACK_GROUP_COLS, CFG.PACK_QUARTER_COLS, cutlass.Int32)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            # RESOURCE-NOTE: dp_empty = dS^T readout only (arrive before the XCHG STS); dP^T(t+1) never waits dQ
            _arrive_leader(bars.dp_empty)
            bars.ds_recv_full.arrive(n_bytes=xchg_tx, pred=ship_pred)  # XCHG_TX; DIAGNOSTIC D8s arms CFG.DIAG_EXCHANGE_OFF_TX = 0 (completes at the arm)
            _store_words(slab, CFG.XCHG, lane, _concat32(x0, x1))
            nvvm.fence_proxy("async.shared", space="cta")
            named_barrier_fence(CFG.XCHG_BARRIER, CFG.EPILOGUE_LANES)
            _skew_leader_ship(rank)  # skew lever 2 (ONE site for both role shapes): the LEADER holds its XCHG ship (the half is written)
            # DIAGNOSTIC D8s (exchange_off): the ship only; the arm above expects 0 bytes, every wait and arrive stays
            if cutlass.const_expr(not exchange_off):
                _ship_xchg(slab, bars, rank, ship_pred)
            if cutlass.const_expr(dbg is not None):
                _clk_ship(dbg, cute.arch.clock64() - t_ds)  # STAMPS: ds_tmem pass -> ship issue (+ the stamp the MMA warps read back)
            _skew_follower_own_half(rank)  # skew lever 1: the FOLLOWER holds its own half (its XCHG is shipped)
            _store_words(slab, own_base, lane, _concat32(o0, o1))
            nvvm.fence_proxy("async.shared", space="cta")
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            bars.ds_smem.arrive()
            if qtile > lo:
                # round B of dQ(t-1): slice 2 dhalf + 1
                empty_state = _dq_round(slab, bars, slot_base, dhalf, qrow, carried, empty_state, route_red, sts_off, dbg)
            _wait_tcgen(bars.dq_done, dq_state.phase, dbg, _W["wait:epilogue:dq_done"])
            if cutlass.const_expr(dbg is not None):
                t_dq = cute.arch.clock64()  # STAMPS: dQ(t) landed
                _clk_add(dbg, _W["seg_dq_done_sum"], cutlass.Int32(t_dq - t_ds))
            _skew_follower_dq_readout(rank)  # skew lever 6: the FOLLOWER holds its dQ readout (s_tail_empty alone holds S^T(t+1))
            dq = _tmem_load(tmem_base + CFG.DQ_COL, CFG.DQ_COLS)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            # RESOURCE-NOTE: s_tail_empty right after wait(LOAD): S^T(t+1) waits the readout, never the STS
            _arrive_leader(bars.s_tail_empty)
            if cutlass.const_expr(dbg is not None):
                _clk_add(dbg, _W["seg_readout_sum"], cutlass.Int32(cute.arch.clock64() - t_dq))  # STAMPS: dq_done pass -> the readout's arrive
                _clk_add(dbg, _W["tiles"], cutlass.Int32(1))
            if cutlass.const_expr(route_red):
                row_local = qtile * cutlass.Int32(CFG.N) + q_half + qrow  # the lane's q row inside the sequence (= the store warp's box row + qrow)
                row_even = row_local - parity
                _dq_red_out(
                    dq,
                    dq_item_base + cutlass.Int64(row_even) * dq_row_stride_bytes,
                    dq_row_stride_bytes,
                    parity,
                    row_local < seq_q,
                    row_even < seq_q,
                    row_even + cutlass.Int32(1) < seq_q,
                    dq_route_code,
                )
            # round A of dQ(t): slice 2 dhalf
            empty_state = _dq_round(slab, bars, slot_base, dhalf, qrow, _dq_half(dq, 0), empty_state, route_red, sts_off, dbg)
            carried = _dq_half(dq, CFG.DQ_BOX)
            ds_state = advance(ds_state, 1)
            dq_state = advance(dq_state, 1)
        if lo < hi:
            empty_state = _dq_round(slab, bars, slot_base, dhalf, qrow, carried, empty_state, route_red, sts_off, dbg)  # round B of dQ(hi-1)
        else:
            # the zero-trip empty commit (both epilogues observe it: dq_state +1 per item)
            _wait_tcgen(bars.dq_done, dq_state.phase, dbg, _W["wait:epilogue:dq_done"])
            dq_state = advance(dq_state, 1)
        live = lo < hi
        _skew_follower_item_end(rank)  # skew lever 3: the FOLLOWER holds its dV / dK TMEM readout at the item end
        if cutlass.const_expr(dkv_fp32):
            # dkv_accum = fp32_reduce (header section 11): round 1 (dV 4 + dK 0) -> dv_full -> dvk_stage_empty wait -> round 2 (dK 1-3) -> dk_full
            stage_state = _dkv_fp32_stage(slab, bars, tmem_base, lane, live, stage_state, dbg)
        else:
            _native_out(slab, tmem_base, lane, live, CFG.DV_COL, (CFG.dv_store(0), CFG.dv_store(1)), dtype)
            bars.dv_full.arrive()
            _native_out(slab, tmem_base, lane, live, CFG.DK_COL, (own_base, CFG.XCHG), dtype)
            bars.dk_full.arrive()
        pair, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, dbg, _W["wait:epilogue:sched_full"])


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
    dq_after_dv: cutlass.Constexpr[bool],
    fa4_roles: cutlass.Constexpr[bool],
    regs: cutlass.Constexpr[int],
    dbg,
):
    """Warp 12 on the pair LEADER: the five cta_group::2 GEMMs in FAv4's rotated order (header section 5 MMA ISSUE ORDER: ``dq_after_dv``
    False = fa4, the default: dP^T(t), dQ(t-1), dV(t), S^T(t+1), dK(t); True = fa4b: dP^T(t), dV(t), dQ(t-1), S^T(t+1), dK(t) -- the in-tile
    dQ slot one GEMM later, the same waits / guards / counts, ONE helper `_dq_slot` at two const_expr sites); every commit multicast to the
    pair.  elect_p = nvvm.elect_sync() once per item.
    No DIAGNOSTIC probe reaches this warp: D7 is declined BY NAME on this map and D8 (`ds_gates_off`) is RETIRED here -- the three dS gates
    are the ds_recv_full arm's only back-edge (_dq_issue; barrier-inspector 2026-10-27 08:01).  ``dbg`` (STAMPS; DbgLeader): every wait of
    this warp bracketed by %clock64 into its own word + the blocked count, the tile count, the ship -> recv handshake.

    # BARRIER (per item, first): _credit(bars.sched_empty[sched_state.idx])
    #   if lo < hi: _wait_tcgen(bars.k1_full, kv.phase); _wait_tcgen(bars.q1_full[q1.idx], q1.phase);
    #               _wait_tcgen(bars.s_tail_empty, s_tail.phase) [ALWAYS: the previous item's tail dQ readout / pre-armed, D1 i] -> s_tail += 1
    #               -> S^T(lo) = mma_ss(M 256) -> _commit_pair(bars.s_done, elect_p); _commit_pair(bars.q1_empty[q1.idx], elect_p)
    #               _wait_tcgen(bars.v_full, kv.phase); _wait_tcgen(bars.k2_full, kv.phase)
    #   _wait_tcgen(bars.dvk_empty, out_empty.phase); out_empty = advance(out_empty, 1)   [pre-armed st(1); EVERY item, zero-trip too]
    # BARRIER (per tile t):
    #   1. _wait_tcgen(bars.dp_empty, dpe.phase)  [st(1): the dS^T(t-1) readout] -> _wait_tcgen(bars.do1_full[0], do1.phase) -> dP^T k-half 0
    #      -> _wait_tcgen(bars.do1_full[1], do1.phase) -> dP^T k-half 1 -> _commit_pair(dp_done); _commit_pair(do1_empty)
    #   2. [fa4, dq_after_dv False] _dq_slot = _wait_tcgen(bars.s_read, ready.phase)  [EVERY tile: both softmaxes hold S^T(t) cols 64..127 in registers]
    #      if qtile > lo: _dq_issue = _wait_tcgen(ds_smem, xchg.phase); _wait_tcgen(ds_recv_full, xchg.phase); _wait_tcgen(ds_pair_ready, xchg.phase)
    #         -> dQ(t-1) = mma_ss(M 128, K 256: A = sDS [256 kv x own 64 q] MN-major, B = K2) into [DQ_COL, +64) -> _commit_pair(dq_done) -> xchg += 1
    #   3. _wait_tcgen(bars.p_ready, ready.phase) -> _wait_tcgen(bars.do2_full, do2.phase) -> dV TS (accumulate = qtile > lo) -> _commit_pair(do2_empty)
    #      [fa4b, dq_after_dv True] _dq_slot HERE instead (the same waits, guard and count; s_read(t) dominated by p_ready(t)): dV(t) covers the exchange
    #   4. q1_next = advance(q1, Q1_STAGES); _s_lookahead: if t + 1 < hi: _wait_tcgen(q1_full[q1_next.idx]); if t > lo: _wait_tcgen(s_tail_empty,
    #      s_tail.phase) -> s_tail += 1 [D1 ii: dQ(t-1)'s readout]; S^T(t+1) -> _commit_pair(s_done); _commit_pair(q1_empty[q1_next.idx])
    #   5. _wait_tcgen(bars.ds_ready, ready.phase) -> _wait_tcgen(bars.q2_full, q2.phase) -> dK TS -> _commit_pair(q2_empty)
    #   q1 = q1_next; do1, do2, q2, ready, dpe = advance(.., 1)
    # BARRIER (block end): if lo < hi: if hi - 1 > lo: _wait_tcgen(bars.s_tail_empty, s_tail.phase) -> s_tail += 1 [D1 iii: dQ(hi-2)'s readout]
    #                                  _dq_issue (the three gates on xchg, dQ(hi-1), dq_done, xchg += 1); _commit_pair(k1_empty); _commit_pair(v_empty); kv += 1
    #                      else: _commit_pair(dq_done)   [the zero-trip empty commit on a live pipe: both epilogues wait it]
    # BARRIER (after the persistent loop, P15): _wait_tcgen(bars.dp_empty, dpe.phase); _wait_tcgen(bars.dvk_empty, out_empty.phase);
    #   _wait_tcgen(bars.s_tail_empty, s_tail.phase)   [the last item's tail readout]; then the final barrier_cta_sync and _tmem_dealloc_pair.
    # BARRIER under fa4_roles (16w_fa4; x-fa4 table section 15): `_dq_issue` commits dq_war_done IMMEDIATELY BEFORE dq_done at BOTH dQ issue sites (the
    #   softmax's WAR signal); the zero-trip site commits dq_done ONLY (unchanged).
    # BARRIER under fa4_roles (16w_fa4; x-fa4 table X5 / section 6): dp_empty is DELETED -- no slot-1 wait, no dpe state, no P15 drain; dP^T(t+1)'s
    #   overwrite of the packed dS^T(t) is ordered by the in-order tensor pipe after dK(t) (issued first by this thread), of dP^T(t)'s fp32 by the
    #   softmax's ld + wait(LOAD) -> ds_ready(t) -> this warp's dK(t) wait -> dK(t) -> dP^T(t+1).  Every other line of this role is unchanged.
    """
    # RESOURCE-NOTE: fa4 order: dP^T(t), dQ(t-1), dV(t), S^T(t+1), dK(t), tail dQ(hi-1); xchg_state
    # RESOURCE-NOTE: fa4b order (dq_after_dv): dP^T(t), dV(t), dQ(t-1), S^T(t+1), dK(t); the same sites, dV(t) covers the exchange
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
    xchg_state = PipelineState.start()  # the P2 split: +1 per dQ ISSUE (one tile behind ready_state)
    s_tail_state = PipelineState.start(phase=1)  # pre-armed: the kernel's first S^T(lo) finds the S tail free
    if cutlass.const_expr(not fa4_roles):
        dp_empty_state = PipelineState.start(phase=1)
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
            _wait_tcgen(bars.k1_full, kv_state.phase, dbg, _W["wait:leader:k1_full"])
            _wait_tcgen(bars.q1_full[q1_state.idx], q1_state.phase, dbg, _W["wait:leader:q1_full"])
            # RESOURCE-NOTE: each S^T issue waits s_tail_empty (dQ(t-1) read out of the S tail); st(1)
            # D1 (i): the previous item's tail dQ(hi_prev - 1) has been read out of [64, 128)
            _wait_tcgen(bars.s_tail_empty, s_tail_state.phase, dbg, _W["wait:leader:s_tail_empty"])
            s_tail_state = advance(s_tail_state, 1)
            q1d = _operand_k_major(slab, CFG.Q1 + q1_state.idx * CFG.Q1_BYTES, dtype, CFG.Q_HALF).desc()
            mma_ss(score, k1d, q1d, s_acc)
            _commit_pair(bars.s_done, elect_p)
            _commit_pair(bars.q1_empty[q1_state.idx], elect_p)
            _wait_tcgen(bars.v_full, kv_state.phase, dbg, _W["wait:leader:v_full"])
            _wait_tcgen(bars.k2_full, kv_state.phase, dbg, _W["wait:leader:k2_full"])
        _wait_tcgen(bars.dvk_empty, out_empty.phase, dbg, _W["wait:leader:dvk_empty"])
        out_empty = advance(out_empty, 1)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            if cutlass.const_expr(dbg is not None):
                _clk_add(dbg, _W["tiles_leader"], cutlass.Int32(1))  # STAMPS: this warp's tile count (the handshake's tag base)
            # slot 1: dP^T(t) -- 16w: gated by the dS^T(t-1) READOUT (dp_empty's moved arrive), never by a dQ readout; 16w_fa4: dp_empty OUT
            if cutlass.const_expr(not fa4_roles):
                _wait_tcgen(bars.dp_empty, dp_empty_state.phase, dbg, _W["wait:leader:dp_empty"])
            # RESOURCE-NOTE: dp_empty OUT: packed dS^T(t) read by dK(t) in-order; dP^T(t) fp32 by ds_ready(t)
            _wait_tcgen(bars.do1_full[0], do1_state.phase, dbg, _W["wait:leader:do1_full"])
            mma_ss(score, vd, do1d, dp_acc, k_start=0, k_count=CFG.K_STEPS_PER_BOX)
            _wait_tcgen(bars.do1_full[1], do1_state.phase, dbg, _W["wait:leader:do1_full"])
            mma_ss(score, vd, do1d, dp_acc, k_start=CFG.K_STEPS_PER_BOX, k_count=CFG.K_STEPS_PER_BOX, accumulate=True)
            _commit_pair(bars.dp_done, elect_p)
            _commit_pair(bars.do1_empty, elect_p)
            # slot 2 (fa4): dQ(t-1) into S^T's tail -- after both softmaxes read S^T(t) (s_read, every tile), behind the exchange gates of t-1
            if cutlass.const_expr(not dq_after_dv):
                xchg_state = _dq_slot(bars, dqdesc, dsd, k2d, dq_acc, xchg_state, ready_state, elect_p, qtile, lo, fa4_roles, dbg)
            # slot 3 (fa4) / slot 2 (fa4b): dV(t) -- reads P^T(t) at [P_COL, +PACK_COLS), disjoint from the dQ image [DQ_COL, +DQ_COLS)
            _wait_tcgen(bars.p_ready, ready_state.phase, dbg, _W["wait:leader:p_ready"])
            _wait_tcgen(bars.do2_full, do2_state.phase, dbg, _W["wait:leader:do2_full"])
            _ts(grad, tmem_base + CFG.P_COL, do2d, dv_acc, qtile > lo)
            _commit_pair(bars.do2_empty, elect_p)
            # slot 3 (fa4b): dQ(t-1) one GEMM later -- dV(t) covers the exchange of t-1 (+512 clk at zero SMEM); the same waits, guard and count
            if cutlass.const_expr(dq_after_dv):
                xchg_state = _dq_slot(bars, dqdesc, dsd, k2d, dq_acc, xchg_state, ready_state, elect_p, qtile, lo, fa4_roles, dbg)
            # slot 4: S^T(t+1) -- LEADS as in FAv4; waits dQ(t-1)'s readout (s_tail_empty) only when a dQ was issued in this tile
            q1_next = advance(q1_state, CFG.Q1_STAGES)
            s_tail_state = _s_lookahead(slab, bars, score, k1d, s_acc, q1_next, s_tail_state, elect_p, qtile, lo, hi, dtype, dbg)
            # slot 5: dK(t)
            _wait_tcgen(bars.ds_ready, ready_state.phase, dbg, _W["wait:leader:ds_ready"])
            _wait_tcgen(bars.q2_full, q2_state.phase, dbg, _W["wait:leader:q2_full"])
            _ts(grad, tmem_base + CFG.DS_COL, q2d, dk_acc, qtile > lo)
            _commit_pair(bars.q2_empty, elect_p)
            q1_state = q1_next
            do1_state = advance(do1_state, 1)
            do2_state = advance(do2_state, 1)
            q2_state = advance(q2_state, 1)
            ready_state = advance(ready_state, 1)
            if cutlass.const_expr(not fa4_roles):
                dp_empty_state = advance(dp_empty_state, 1)
        if lo < hi:
            # tail: dQ(hi-1) -- the S tail holds the UNREAD dQ(hi-2) image iff a dQ was issued in the loop (N >= 2): wait its readout (D1 iii);
            # at N == 1 the previous tenant is S^T(lo), whose softmax read is implied by the dS gates (lo) on both CTAs
            if hi - 1 > lo:
                _wait_tcgen(bars.s_tail_empty, s_tail_state.phase, dbg, _W["wait:leader:s_tail_empty"])
                s_tail_state = advance(s_tail_state, 1)
            xchg_state = _dq_issue(bars, dqdesc, dsd, k2d, dq_acc, xchg_state, elect_p, fa4_roles, dbg, DBG_CLK.TAG_BACK_TAIL)
            _commit_pair(bars.k1_empty, elect_p)
            _commit_pair(bars.v_empty, elect_p)
            kv_state = advance(kv_state, 1)
        else:
            _commit_pair(bars.dq_done, elect_p)
        pair, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, dbg, _W["wait:leader:sched_full"])
    # P15: the follower's last relaxed arrives (dp_empty of the last dS^T readout, dvk_empty of the last item, s_tail_empty of the last tail
    # readout) land on a resident leader.
    if cutlass.const_expr(not fa4_roles):
        _wait_tcgen(bars.dp_empty, dp_empty_state.phase, dbg, _W["wait:leader:dp_empty"])
    _wait_tcgen(bars.dvk_empty, out_empty.phase, dbg, _W["wait:leader:dvk_empty"])
    _wait_tcgen(bars.s_tail_empty, s_tail_state.phase, dbg, _W["wait:leader:s_tail_empty"])


@cute.jit
def _mma_loop_follower(bars, sched, pair, head, batch, meta, n_batch, causal: cutlass.Constexpr[bool], regs: cutlass.Constexpr[int], dbg):
    """Warp 12 on the FOLLOWER: the quiet body + the dS relay.  Never issues an MMA, never arms expect_tx, never waits a leader copy
    (q*_full / do*_full / k*_full / v_full / p_ready / ds_ready / s_read / dp_empty / s_tail_empty / dvk_empty / ds_pair_ready are
    LEADER-only waits).

    # BARRIER (per item, first): _credit(bars.sched_empty[sched_state.idx])   [this warp is one of this CTA's 15 credit warps]
    # BARRIER (per tile t, relay st(0)): _wait(bars.ds_smem, relay.phase); _wait(bars.ds_recv_full, relay.phase)
    #   -> bars.ds_pair_ready.arrive_on_peer(cutlass.Int32(LEADER_RANK), pred=elect_p, release_cta=True)
    #      [1 lane -> the leader's copy == init 1; the cumulative .release.cta publishes our epilogue's fenced own-half stores to the
    #       leader-issued cta_group::2 dQ MMA (P8 / P12); relaxed would NOT]
    #   relay = advance(relay, 1)   [the leader consumes it one tile later (xchg_state) or at the tail: the counts are the same, N per item]
    # Zero-trip item: credit only.  No drain: the leader waits every ds_pair_ready phase in-loop.  Teardown: _tmem_dealloc_pair.
    # ``dbg`` (STAMPS): the relay's two wait sums, its tile count and the ship -> recv handshake after the ds_recv_full wait (_clk_recv).
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
            if cutlass.const_expr(dbg is not None):
                _clk_add(dbg, _W["tiles_follower"], cutlass.Int32(1))
            _wait(bars.ds_smem, relay_state.phase, dbg, _W["wait:follower:ds_smem"])
            _wait(bars.ds_recv_full, relay_state.phase, dbg, _W["wait:follower:ds_recv_full"])
            if cutlass.const_expr(dbg is not None):
                _clk_recv(dbg, _W["tiles_follower"], _W["recv_follower_sum"], _W["recv_follower_matched"], _W["recv_follower_missed"], DBG_CLK.TAG_BACK_RELAY)
            bars.ds_pair_ready.arrive_on_peer(cutlass.Int32(LEADER_RANK), pred=elect_p, release_cta=True)
            relay_state = advance(relay_state, 1)
        pair, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, dbg, _W["wait:follower:sched_full"])


@cute.jit
def _reduce_loop(
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
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    diag_2cta_code: cutlass.Constexpr[int],
    regs: cutlass.Constexpr[int],
):
    """Warps 8-11 under warp_roles 16w_fa4 (X-FA4): the REDUCE group = FAv4's dQ readout warps on the S-tail body (lane = q row of d half
    (warp // 2) % 2 for dQ; lane = own kv row for the item-end dV / dK staging).  No dS work: the softmax writes sDS own + XCHG and ships.
    ``diag_2cta_code`` (DIAGNOSTIC): DIAG_SDQ_STS_OFF (D9) = the two `_dq_round` sites skip the sDQ slice STS + its proxy fence (the stack fold:
    the probe serves both role shapes).  STAMPS (`debug_clk`) declines under 16w_fa4: every wait here is passed dbg=None (untimed).
    `dkv_accum = "fp32_reduce"` declines under 16w_fa4 too: this group stages bf16 halves only (`_native_out`); dvk_stage_empty has no field here.

    # BARRIER (per item, first): _credit(bars.sched_empty[sched_state.idx])
    # BARRIER (per tile t; dq_state st(0), empty_state st(1); barrier-inspector x-fa4 table section 9, the reduce stub order):
    #   1. _wait_tcgen(bars.dq_done, dq_state.phase)   [the AUDIT's K:1144 / eff5f58a K:1277 site STAYS on warps 8-11]
    #      -> [test-only EPILOGUE_DQ_READOUT sleep, FOLLOWER only] -> dq = tcgen05.ld x64 @ CFG.DQ_COL -> tcgen05_wait(LOAD)
    #      -> tcgen05_fence(BEFORE_THREAD_SYNC) -> _arrive_leader(bars.s_tail_empty)   [LEADER, bare, 128 x 2 = 256; BEFORE round A (X8)]
    #   2. round A: _dq_round(dq[0, 32)) = _wait(dq_empty[0]) + _wait(dq_empty[1]) at empty_state.phase -> STS slot dhalf -> fence_proxy
    #      -> bars.dq_full[dhalf].arrive()   [THREAD local, 2 warps = 64 == DQ_SLICE_LANES] -> empty_state += 1
    #   3. round B: _dq_round(dq[32, 64)) -- back-to-back inside the tile: no `carried` half, no item-end round (resource-manager table 3)
    #   4. dq_state = advance(dq_state, 1)
    # BARRIER (item end): if lo >= hi: _wait_tcgen(bars.dq_done, dq_state.phase) + advance   [the zero-trip empty commit; both CTAs' groups observe it]
    #   -> [test-only FOLLOWER_EPILOGUE_ITEM_END sleep, FOLLOWER only] -> native dV over K2 -> bars.dv_full.arrive() [128 of 160: + the loader's 32, fix A]
    #   -> native dK over ds_own(rank) + XCHG -> bars.dk_full.arrive() [128]   (X9: the reduce group keeps the staging; the store warp keeps the TMA stores)
    # Flips: dq_done N (+1 zero-trip) observed; dq_empty / dq_full 2N / 2N per slot (A then B per tile); s_tail_empty N; dv_full / dk_full 1 per item.
    # No drain: every arrive of this role is local or waited by the leader in-loop (s_tail_empty's last phase is the leader's P15 drain).
    """
    # RESOURCE-NOTE: reduce group: dq_done -> ld x64 -> s_tail_empty -> rounds A, B; no carried half
    nvvm.setmaxregister(regs, nvvm.SetMaxRegisterAction.INCREASE)  # CFG.REG_SPLITS[reg_split][1] (knob; default EPILOGUE_REGS: the same 144)
    lane = cute.arch.thread_idx()[0] % CFG.M  # own kv row for the dV / dK staging; TMEM lane of the dQ half-row image
    qrow = lane % CFG.Q_HALF  # dQ row of the half-row image
    dhalf = lane // CFG.Q_HALF  # d half this warp pair holds (warp-uniform) = its sDQ slot
    own_base = cutlass.Int32(CFG.DS) + rank * cutlass.Int32(CFG.DS_HALF_BYTES)  # CFG.ds_own(rank): the dK staging's first alias half
    slot_base = cutlass.Int32(CFG.DQ) + dhalf * cutlass.Int32(CFG.DQ_SLICE_BYTES)  # CFG.dq_slot(dhalf): the warp pair's slot of the ring
    dq_state = PipelineState.start()
    empty_state = PipelineState.start(phase=1)  # pre-armed (P5 b): both slots of the ring are free before the first round; NO bootstrap (D3)
    sched_state = PipelineState.start()
    sts_off = cutlass.const_expr(diag_2cta_code == DIAG_SDQ_STS_OFF)  # DIAGNOSTIC D9: the sDQ slice STS + its proxy fence skipped in every round
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        lo, hi, seq_q, seq_k, q_start = _item_bounds(pair, meta, batch, n_batch, causal)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            _wait_tcgen(bars.dq_done, dq_state.phase, None, 0)
            _skew_follower_dq_readout(rank)  # skew lever 6 (16w_fa4 site): the FOLLOWER holds its dQ readout (s_tail_empty alone holds S^T(t+1))
            dq = _tmem_load(tmem_base + CFG.DQ_COL, CFG.DQ_COLS)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            # RESOURCE-NOTE: s_tail_empty right after wait(LOAD): S^T(t+1) waits the readout, never the STS
            _arrive_leader(bars.s_tail_empty)
            # round A of dQ(t): slice 2 dhalf; round B: slice 2 dhalf + 1 (the tma route only; the sdq_sts_off probe's flag; no stamps under 16w_fa4)
            empty_state = _dq_round(slab, bars, slot_base, dhalf, qrow, _dq_half(dq, 0), empty_state, False, sts_off, None)
            empty_state = _dq_round(slab, bars, slot_base, dhalf, qrow, _dq_half(dq, CFG.DQ_BOX), empty_state, False, sts_off, None)
            dq_state = advance(dq_state, 1)
        if lo >= hi:
            _wait_tcgen(bars.dq_done, dq_state.phase, None, 0)  # the zero-trip empty commit (both reduce groups observe it: dq_state +1 per item)
            dq_state = advance(dq_state, 1)
        live = lo < hi
        _skew_follower_item_end(rank)  # skew lever 3 (16w_fa4 site): the FOLLOWER holds its dV / dK TMEM readout at the item end
        _native_out(slab, tmem_base, lane, live, CFG.DV_COL, (CFG.dv_store(0), CFG.dv_store(1)), dtype)
        bars.dv_full.arrive()
        _native_out(slab, tmem_base, lane, live, CFG.DK_COL, (own_base, CFG.XCHG), dtype)
        bars.dk_full.arrive()
        pair, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, None, 0)


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
    dbg,
):
    """Warp 13, BOTH CTAs: Q1 / Q2 / dO1 / dO2 per q tile, K1 / K2 / V per kv block (grouped head // ratio), each CTA its own boxes.
    Unchanged from the first body except the Q1 ring depth (Q1_STAGES = 1: one slot, st(1) adv 1, drain x 1).  ``dbg`` (STAMPS): the
    *_empty wait sums (the drains included) and sched_full.

    # arm_pred = is_leader & nvvm.elect_sync()  (P9: a cga2 tensor load completes on the pair LEADER's mbar whoever issued it; the
    # leader arms BOTH CTAs' bytes = the *_TX constants; the follower NEVER arms).  tma_mask = cutlass.Int16(1) << rank (self-only)
    # under the multicast form, None under ``tma_load_plain`` (header section 10 LOAD FORM); every load: tma_load_tile / tma_load_subtiles(...,
    # bars.X_full[..].smem_ptr, cta_group=CFG.CTA_MMA, mcast_mask=tma_mask).
    # BARRIER (per item, first): _credit(bars.sched_empty[sched_state.idx])   [then the test-only DEBUG_LOADER_DELAY_US sleep, default none]
    #   if lo < hi: acquire the maps; _wait(bars.k1_empty, kv.phase) -> bars.k1_full.arrive(n_bytes=CFG.K1_TX, pred=arm_pred) -> K1
    #               (2 boxes, own 128 rows); _wait(bars.v_empty, kv.phase) -> v_full.arrive(CFG.V_TX, pred=arm_pred) -> V;
    #               _wait(bars.k2_empty, k2e.phase) -> bars.dv_full.arrive() [bare 32, fix A] -> k2_full.arrive(CFG.K2_TX, pred=arm_pred)
    #               -> K2 (two [128 kv x 64 d] boxes at d = 64 * rank, the PAIR's rows); kv = advance(kv, 1)
    #   else (zero-trip): _wait(bars.k2_empty, k2e.phase) -> bars.dv_full.arrive()   [the same single observation on every path, P14]
    #   k2e = advance(k2e, 1)
    # BARRIER (per tile t):
    #   _wait(bars.q1_empty[s], q1.phase) -> q1_full[s].arrive(CFG.Q1_TX, pred=arm_pred) -> Q1 (rows 64 * rank .., 2 boxes)   [s = 0: ONE stage]
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
    # RESOURCE-NOTE: Q1 single stage (FAv4 2-CTA form) funds sDQ 16 KiB; Q1(t+1) window ~2000 clk
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
            _wait(bars.k1_empty, kv_state.phase, dbg, _W["wait:loader:k1_empty"])
            bars.k1_full.arrive(n_bytes=CFG.K1_TX, pred=arm_pred)
            tma_load_tile(
                _operand_k_major(slab, CFG.K1, dtype, CFG.M),
                tma_slice_runtime_desc(k_desc, 0, kv_row0, kv_head, 0),
                bars.k1_full.smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mask,
                acquire=False,
            )
            _wait(bars.v_empty, kv_state.phase, dbg, _W["wait:loader:v_empty"])
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
            _wait(bars.k2_empty, k2e_state.phase, dbg, _W["wait:loader:k2_empty"])
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
            _wait(bars.k2_empty, k2e_state.phase, dbg, _W["wait:loader:k2_empty"])
            bars.dv_full.arrive()
        k2e_state = advance(k2e_state, 1)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            q_row = qtile * CFG.N
            _wait(bars.q1_empty[q1_state.idx], q1_state.phase, dbg, _W["wait:loader:q1_empty"])
            bars.q1_full[q1_state.idx].arrive(n_bytes=CFG.Q1_TX, pred=arm_pred)
            tma_load_tile(
                _operand_k_major(slab, CFG.Q1 + q1_state.idx * CFG.Q1_BYTES, dtype, CFG.Q_HALF),
                tma_slice_runtime_desc(q1_desc, 0, q_row + q_half, head, 0),
                bars.q1_full[q1_state.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mask,
                acquire=False,
            )
            _wait(bars.do1_empty, do1_state.phase, dbg, _W["wait:loader:do1_empty"])
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
            _wait(bars.do2_empty, do2_state.phase, dbg, _W["wait:loader:do2_empty"])
            bars.do2_full.arrive(n_bytes=CFG.DO2_TX, pred=arm_pred)
            tma_load_tile(
                _operand_mn_major(slab, CFG.DO2, dtype, CFG.N),
                tma_slice_runtime_desc(do2_desc, d_half, q_row, head, 0),
                bars.do2_full.smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mask,
                acquire=False,
            )
            _wait(bars.q2_empty, q2_state.phase, dbg, _W["wait:loader:q2_empty"])
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
        pair, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, dbg, _W["wait:loader:sched_full"])
    if cutlass.const_expr(CFG.CTA_MMA == 2):
        # P15: whole-ring drains from the CARRIED states (the leader's last multicast commits must land on a resident follower).
        q1_state = _drain(bars.q1_empty, q1_state, CFG.Q1_STAGES, dbg, _W["wait:loader:q1_empty"])
        q2_state = _drain(bars.q2_empty, q2_state, 1, dbg, _W["wait:loader:q2_empty"])
        do1_state = _drain(bars.do1_empty, do1_state, 1, dbg, _W["wait:loader:do1_empty"])
        do2_state = _drain(bars.do2_empty, do2_state, 1, dbg, _W["wait:loader:do2_empty"])
        _drain(bars.k1_empty, kv_state, 1, dbg, _W["wait:loader:k1_empty"])
        _drain(bars.v_empty, kv_state, 1, dbg, _W["wait:loader:v_empty"])


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
    dq_route_code: cutlass.Constexpr[int],
    store_wait_spin: cutlass.Constexpr[bool],
    regs: cutlass.Constexpr[int],
    ratio: cutlass.Constexpr[int],
    dkv_fp32: cutlass.Constexpr[bool],
    dbg,
):
    """Warp 14, BOTH CTAs: dQ reduce-add out of the DEDICATED sDQ ring (4 fp32 slices [64 q x 32 d] per tile in TWO rounds of two slots; one
    reduce per PAIR tile), dV / dK native stores.
    ``diag_2cta_code`` (DIAGNOSTIC, default DIAG_NONE = this body): DIAG_DQ_REDUCE_ADD_OFF skips the four tma_reduce_add ISSUES per tile
    (D2-pair); the dq_full waits, the commit / wait groups and the per-round dq_empty arrives stay, so the 2-slot ring balance holds (2N per
    slot per item) and dQ stays ZERO (output WRONG BY DESIGN; `--validate` refuses the arm).
    ``store_wait_spin`` (KNOB ``store_wait_form``, default retry = False): the six mbarrier waits of this warp (dq_full x 4 USES, dv_full,
    dk_full) through :func:`_wait_store` -- the same phase tests, only the retry mechanics move; the sched_full wait and tma_store_wait (one
    spelling) stay.  Bitwise the same function under both forms.
    ``dq_route_code`` (``dq_route_pair``; DQ_ROUTE_TMA = this path): under a red route the epilogue's registers carried the words out, so the
    dQ map is not acquired, the UTMAREDG issue and its per-use commit are skipped; the dq_full waits, the per-round `tma_store_wait` (a no-op:
    no bulk group in flight) and the two dq_empty arrives per round stay = the ring protocol of header section 5, every flip unchanged.

    # BARRIER (per item, first): _credit(bars.sched_empty[sched_state.idx]); NO dq_empty bootstrap (D3: the epilogue's empty_state is pre-armed)
    # BARRIER (per tile t; q_row = qtile * N + q_half) -- the D4 closed per-round release:
    #   for round r in (A = 0, B = 1): for slot in (0, 1): (slot, slice) = CFG.dq_use(r * DQ_SLOTS + slot) = (0, 0), (1, 2), (0, 1), (1, 3):
    #     _wait_store(bars.dq_full[slot], dq.phase) -> tma_reduce_add_subtile(sDQ[slot] -> dQ map @ (slice * DQ_BOX, q_row)) -> tma_store_commit()
    #     -> dq = advance(dq, DQ_SLOTS)   [2 stages, +1 per USE: idx == slot by construction, asserted in the config]
    #     then tma_store_wait(1) -> bars.dq_empty[0].arrive(); tma_store_wait(0) -> bars.dq_empty[1].arrive()   [bare, 32 each, after the
    #     corresponding SOURCE READ: `cp.async.bulk.wait_group.read`]
    # BARRIER (item end): _wait_store(bars.dv_full, out.phase) -> dV store (2 halves from CFG.dv_store(half); skipped for a padded CTA);
    #   _wait_store(bars.dk_full, out.phase) -> dK store (CFG.dk_store(rank, half)); out = advance(out, 1); tma_store_commit(); tma_store_wait(0)
    #   -> _arrive_leader(bars.dvk_empty)   [LEADER, bare, 32 x 2 = 64]   -> bars.k2_empty.arrive()   [THREAD local, bare, 32; its back-edge:
    #      this arrive follows the dv_full wait, whose 160 include the loader's 32 = the loader observed the previous k2_empty phase (fix A)]
    # BARRIER (item end, dkv_fp32 = dkv_accum fp32_reduce, `_dkv_fp32_reduce_rounds`): _wait_store(bars.dv_full) -> [live_rows: acquire MAP_DV + MAP_DK;
    #   4 x tma_reduce_add_subtile of the round-1 slots CFG.dkv_acc_r1_slot(0..3) -> dV map @ (i * DQ_BOX, kv_row0, head // ratio) + 1 of slot 4 -> dK map @ (0, ..)]
    #   -> tma_store_commit() -> tma_store_wait(0) -> bars.dvk_stage_empty.arrive()   [NEW row, bare 32 == init: the round-2 WAR release] -> _wait_store(bars.dk_full)
    #   -> [live_rows: 3 x tma_reduce_add_subtile of CFG.dkv_acc_r2_slot(0..2) -> dK map @ ((i + 1) * DQ_BOX, ..)] -> out += 1; tma_store_commit(); tma_store_wait(0)
    #   -> dvk_empty / k2_empty as above.
    # No drain: dvk_empty's last phase is the leader MMA's drain; k2_empty / dq_empty are local.
    # ``dbg`` (STAMPS): the dq_full / dv_full / dk_full / sched_full wait sums + the clk inside the per-round and item-end `tma_store_wait`.
    """
    nvvm.setmaxregister(regs, nvvm.SetMaxRegisterAction.DECREASE)  # CFG.REG_SPLITS[reg_split][2] (knob; default SERVICE_REGS)
    dq_state = PipelineState.start()
    out_state = PipelineState.start()
    sched_state = PipelineState.start()
    q_half = rank * cutlass.Int32(CFG.Q_HALF)
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        lo, hi, seq_q, seq_k, q_start = _item_bounds(pair, meta, batch, n_batch, causal)
        dq_desc = _map_ptr(maps, batch, n_batch, MAP_DQ)
        if cutlass.const_expr(dq_route_code == DQ_ROUTE_TMA):
            if lo < hi:
                tma_tensormap_acquire(dq_desc)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            q_row = qtile * CFG.N + q_half
            for r in cutlass.range_constexpr(CFG.DQ_ROUNDS):
                for s in cutlass.range_constexpr(CFG.DQ_SLOTS):
                    slot, slice_index = CFG.dq_use(r * CFG.DQ_SLOTS + s)
                    _wait_store(bars.dq_full[slot], dq_state.phase, store_wait_spin, dbg, _W["wait:store:dq_full"])
                    if cutlass.const_expr(dq_route_code == DQ_ROUTE_TMA):  # a red route carried the words out of the epilogue's registers: no source read
                        if cutlass.const_expr(diag_2cta_code != DIAG_DQ_REDUCE_ADD_OFF):  # DIAGNOSTIC D2-pair: the ISSUE only (commit / wait / arrives stay)
                            tma_reduce_add_subtile(
                                _dq_slot_tile(slab, slot), tma_slice_runtime_desc(dq_desc, slice_index * CFG.DQ_BOX, q_row, head, 0), 0, acquire=False
                            )
                        tma_store_commit()
                    dq_state = advance(dq_state, CFG.DQ_SLOTS)
                for s in cutlass.range_constexpr(CFG.DQ_SLOTS):
                    if cutlass.const_expr(dbg is not None):
                        t_wg = cute.arch.clock64()
                    tma_store_wait(CFG.DQ_SLOTS - 1 - s)  # a no-op under a red route (no bulk group in flight): the arrive below keeps the ring's release
                    if cutlass.const_expr(dbg is not None):
                        _clk_add(dbg, _W["store_wait_group_rounds"], cutlass.Int32(cute.arch.clock64() - t_wg))  # STAMPS: the reduce-add source drain
                    bars.dq_empty[s].arrive()  # all 32 after this slot's source read: the round's release of slot s
        # Invalid owners and the padded SECOND half of a live pair still consume both native publications and return every source
        # credit, but never acquire / use a map for rows past the sequence.
        kv_row0 = pair * CFG.PAIR_KV_ROWS + rank * CFG.M
        live_rows = kv_row0 < seq_k
        _wait_store(bars.dv_full, out_state.phase, store_wait_spin, dbg, _W["wait:store:dv_full"])
        if cutlass.const_expr(dkv_fp32):
            # dkv_accum = fp32_reduce: 5 + 3 fp32 reduce-adds into the per-KV-head workspace maps, round 1's drain -> dvk_stage_empty before the
            # dk_full wait (`_dkv_fp32_reduce_rounds`; kv head = head // ratio); the item-end commit / wait(0) below drain round 2
            _dkv_fp32_reduce_rounds(slab, bars, maps, batch, n_batch, kv_row0, head // ratio, live_rows, out_state.phase, store_wait_spin, dbg)
        else:
            if live_rows:
                dv_desc = _map_ptr(maps, batch, n_batch, MAP_DV)
                tma_store_tile(_operand_k_major(slab, CFG.K2, dtype, CFG.M), tma_slice_runtime_desc(dv_desc, 0, kv_row0, head, 0))
            _wait_store(bars.dk_full, out_state.phase, store_wait_spin, dbg, _W["wait:store:dk_full"])
            if live_rows:
                dk_desc = _map_ptr(maps, batch, n_batch, MAP_DK)
                for half in cutlass.range_constexpr(CFG.D_HALVES):
                    tma_store_tile(_store_half(slab, CFG.dk_store(rank, half), dtype), tma_slice_runtime_desc(dk_desc, half * CFG.HEAD_BOX, kv_row0, head, 0))
        out_state = advance(out_state, 1)
        tma_store_commit()
        if cutlass.const_expr(dbg is not None):
            t_wg = cute.arch.clock64()
        tma_store_wait(0)
        if cutlass.const_expr(dbg is not None):
            _clk_add(dbg, _W["store_wait_group_item"], cutlass.Int32(cute.arch.clock64() - t_wg))  # STAMPS: the dV / dK store drain
        _arrive_leader(bars.dvk_empty)
        bars.k2_empty.arrive()
        pair, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, dbg, _W["wait:store:sched_full"])


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
def bprop_d128_accum_grouped_2cta_stail_kernel(
    maps: cute.Tensor,
    lse: cute.Tensor,
    delta: cute.Tensor,
    meta: cute.Tensor,
    origins: cute.Tensor,
    dq_accum: Optional[cute.Tensor],
    n_batch: cutlass.Int32,
    stats_origin: cutlass.Constexpr[int],
    scale: cutlass.Float32,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    ratio: cutlass.Constexpr[int],
    dq_after_dv: cutlass.Constexpr[bool],
    dq_route_code: cutlass.Constexpr[int],
    tma_load_plain: cutlass.Constexpr[bool],
    diag_2cta_code: cutlass.Constexpr[int],
    store_wait_spin: cutlass.Constexpr[bool],
    reg_split_code: cutlass.Constexpr[int],
    warp_roles_code: cutlass.Constexpr[int],
    mask_form_code: cutlass.Constexpr[int],
    dkv_fp32: cutlass.Constexpr[bool],
    debug_clk: cutlass.Constexpr[bool],
    dbg_clk: Optional[cute.Tensor],
):
    """Slab, scaffold views, the barrier init block (P4), the pair TMEM allocation, the role dispatch, the pair-symmetric teardown.
    ``warp_roles_code`` = the ROLE SHAPE (header sections 4 / 5 / 11; WARP_ROLES_16W = the default, today's rendering; WARP_ROLES_16W_FA4 =
    X-FA4: the softmax ships the exchange from its registers, warps 8-11 run ``_reduce_loop``, the bar bundle is ``BarsXFA4``), from the typed
    ``warp_roles`` field resolved PER MAP.
    ``mask_form_code`` = the softmax's mask form (header section 11, lever L5: MASK_FORM_FSEL = the default; MASK_FORM_PREDICATED = the
    interior-tile mask skip; MASK_FORM_DIAG_OFF = the DIAGNOSTIC ceiling, output wrong by design), from the typed ``mask_form`` field resolved
    PER MAP; read by the softmax role only.
    ``dq_after_dv`` = the leader's MMA issue order (header section 5: False = fa4, the default; True = fa4b, dQ(t-1) issued after dV(t)),
    from the typed ``pair_mma_order`` field resolved PER MAP (other names decline on the s_tail map before the trace).  ``dq_route_code`` = the
    dQ partial's route (header section 11: DQ_ROUTE_TMA = the default; DQ_ROUTE_RED / DQ_ROUTE_RED_PAIR = register-direct red.global.add from the
    epilogue), from the typed ``dq_route_pair`` field; ``dq_accum`` = the fp32 dQ workspace tensor the red routes address (None = dropped
    from the signature under the default: the default rendering is byte-identical).  ``tma_load_plain``
    = the cga2 TMA load form (header section 10), from the typed ``tma_load_form``
    field; ``diag_2cta_code`` = the DIAGNOSTIC probe (header section 11; DIAG_NONE = this body; D7 declined by name on this map), from the
    typed ``diag_2cta`` field; ``store_wait_spin`` = the store warp's wait form (KNOB ``store_wait_form``) and ``reg_split_code`` = the
    setmaxnreg split (KNOB ``reg_split``), both perf-only (bitwise the same function under every value); ``debug_clk`` / ``dbg_clk`` = the
    STAMPS lever (header section 11) and its Int32 dump tensor (None = dropped from the signature when off: byte-identical rendering).
    ``dkv_fp32`` = the op-level ``dkv_accum`` field (header section 11: False = bf16 partials + the fold, the default rendering; True = fp32
    reduce-adds of dK / dV into the per-KV-head workspace, a ROUNDING CHANGE)."""
    softmax_regs, epilogue_regs, service_regs = reg_split_counts(reg_split_code)  # config = truth: the split's three counts, asserted there
    fa4_roles = warp_roles_code == WARP_ROLES_16W_FA4  # the role shape (trace-time)
    # Trace-time invariants (const_expr: a Python raise, never a staged early exit); the config's validate_fields_for_roles declines first by name.
    if cutlass.const_expr(warp_roles_code not in (WARP_ROLES_16W, WARP_ROLES_16W_FA4)):
        raise ValueError(f"warp_roles_code must be one of {(WARP_ROLES_16W, WARP_ROLES_16W_FA4)}; got {warp_roles_code!r}")
    if cutlass.const_expr(fa4_roles and dq_route_code != DQ_ROUTE_TMA):
        raise ValueError("warp_roles 16w_fa4 serves the tma dQ route only (the reduce group has no register-direct red path)")
    if cutlass.const_expr(fa4_roles and diag_2cta_code == DIAG_DS_GATES_OFF):
        raise ValueError("diag_2cta ds_gates_off is RETIRED under warp_roles 16w_fa4 (the ds_recv_full arm's back-edge runs through the leader's gate waits)")
    if cutlass.const_expr(fa4_roles and debug_clk):  # the stack fold: the STAMPS record layout is the 16w roles' (no words for the fa4 waits)
        raise ValueError("debug_clk is not defined under warp_roles 16w_fa4 (the STAMPS record layout names the 16w roles' waits)")
    if cutlass.const_expr(fa4_roles and dkv_fp32):  # the stack fold: the fp32 dK / dV staging rounds live in the 16w epilogue loop
        raise ValueError("dkv_accum fp32_reduce is not built under warp_roles 16w_fa4 (the reduce group stages bf16 halves only)")
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    x, y, z = cute.arch.block_idx()
    rank = cute.arch.block_idx_in_cluster() & cutlass.Int32(1)  # pair rank under cluster (2,1,1): 0 = LEADER, 1 = FOLLOWER
    is_leader = rank == cutlass.Int32(LEADER_RANK)  # the loader's arm predicate and the scheduler / MMA role select
    pair, head, batch = x // CFG.CGA_M, y, z  # grid x = pairs x CGA_M: both CTAs of a cluster decode the same pair
    slab = cutlass.Array(cutlass.Uint8, CFG.SMEM_BYTES, alignment=1024, space=cutlass.AddressSpace.smem)
    bars = _bars(slab, fa4_roles)  # 34 fields (16w) or the inspector's 32 (16w_fa4): the init loop below runs over the bundle it was handed
    tmem_slot = _view(slab, CFG.TMEM_HANDLE, cutlass.Int32, 1, alignment=4)
    sched_words = _view(slab, CFG.SCHED_PAYLOAD, cutlass.Int32, CFG.SCHED_PAYLOAD_BYTES // 4, alignment=16)
    sched = Sched(bars.sched_full.base_ptr, bars.sched_empty.base_ptr, sched_words, x, y, z)
    # --- STAMPS contexts (header section 11): the record view in the bar block's spare + ONE representative warp per role; None when off.
    if cutlass.const_expr(debug_clk):
        dbg_words = _view(slab, DBG_CLK.SMEM, cutlass.Int32, DBG_CLK.WORDS, alignment=16)
        dbg_softmax = Dbg(dbg_words, warp == cutlass.Int32(0))
        dbg_epilogue = Dbg(dbg_words, warp == cutlass.Int32(EPILOGUE_LEAD_WARP))
        dbg_single = Dbg(dbg_words, True)
        dbg_leader = DbgLeader(dbg_words, True)
    else:
        dbg_words = None
        dbg_softmax = None
        dbg_epilogue = None
        dbg_single = None
        dbg_leader = None
    # --- P4: ONE warp, ONE lane initialises every stage of every ring; fence -> CTA sync -> CLUSTER sync, all OUTSIDE the branch.
    if warp == 0:
        if nvvm.elect_sync():
            for i in cutlass.range_constexpr(len(bars)):
                # C2b: dvk_stage_empty has no user under the default rendering (16w, dkv_fp32 False); under 16w_fa4 the bundle has no such row
                if cutlass.const_expr(fa4_roles or dkv_fp32 or i != DVK_STAGE_EMPTY_ROW):
                    for stage in cutlass.range_constexpr(bars[i].stages):
                        bars[i][stage].init()
            if cutlass.const_expr(debug_clk):
                _clk_init(dbg_words)  # STAMPS: the record zeroed + the body start stamped before the CTA sync publishes it
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()
    cga_arrive()  # required before ANY cross-CTA arrive (multicast commits, leader arrives, the remote CLC arm, the DSMEM ship)
    cga_wait()
    # --- TMEM: one tcgen05.alloc.cta_group::2 of TMEM_COLS per CTA (BOTH CTAs allocate; the leader's collective MMAs write both halves).
    if warp == MMA_WARP:
        tmem_alloc(tmem_slot, CFG.TMEM_COLS, nvvm.CTAGroup.CTA_2)
    nvvm.barrier_cta_sync()  # publishes the TMEM base to every warp
    tmem_base = tmem_slot.load()
    # BARRIER: role dispatch: warp < 8 softmax | < 12 epilogue (16w) / REDUCE group (16w_fa4: _reduce_loop) | == MMA_WARP (leader: _mma_loop_leader,
    #   follower: _mma_loop_follower) | == TMA_LOAD_WARP | == TMA_STORE_WARP | SCHED_WARP (leader only runs the loop).
    if warp < SOFTMAX_WARP_END:
        _softmax_loop(
            slab,
            bars,
            sched,
            tmem_base,
            lse,
            delta,
            origins,
            stats_origin,
            pair,
            head,
            batch,
            meta,
            n_batch,
            rank,
            warp,
            scale,
            causal,
            dtype,
            diag_2cta_code,
            fa4_roles,
            softmax_regs,
            mask_form_code,
            dbg_softmax,
        )
    elif warp < EPILOGUE_WARP_END:
        if cutlass.const_expr(fa4_roles):
            _reduce_loop(slab, bars, sched, tmem_base, pair, head, batch, meta, n_batch, rank, causal, dtype, diag_2cta_code, epilogue_regs)
        else:
            _epilogue_loop(
                slab,
                bars,
                sched,
                tmem_base,
                dq_accum,
                pair,
                head,
                batch,
                meta,
                n_batch,
                rank,
                warp,
                causal,
                dtype,
                dq_route_code,
                diag_2cta_code,
                epilogue_regs,
                dkv_fp32,
                dbg_epilogue,
            )
    elif warp == MMA_WARP:
        if is_leader:
            _mma_loop_leader(slab, bars, sched, tmem_base, pair, head, batch, meta, n_batch, causal, dtype, dq_after_dv, fa4_roles, service_regs, dbg_leader)
        else:
            _mma_loop_follower(bars, sched, pair, head, batch, meta, n_batch, causal, service_regs, dbg_single)
    elif warp == TMA_LOAD_WARP:
        _load_loop(slab, bars, sched, maps, pair, head, batch, meta, n_batch, rank, is_leader, causal, dtype, ratio, tma_load_plain, service_regs, dbg_single)
    elif warp == TMA_STORE_WARP:
        _store_loop(
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
            causal,
            dtype,
            diag_2cta_code,
            dq_route_code,
            store_wait_spin,
            service_regs,
            ratio,
            dkv_fp32,
            dbg_single,
        )
    else:
        _scheduler_loop(sched, is_leader, service_regs)
    # BARRIER (teardown, both CTAs): the final CTA sync after every role returned, then the MMA warp's pair-symmetric dealloc.
    #   No remote arrive targets an exited CTA: the pair-symmetric dealloc wait is the last cross-CTA hand-off.
    nvvm.barrier_cta_sync()
    if cutlass.const_expr(debug_clk):
        if warp == 0:
            if nvvm.elect_sync():
                _clk_dump(dbg_words, dbg_clk, rank)  # STAMPS: every role returned and every record word is published by the CTA sync above
    if warp == MMA_WARP:
        _tmem_dealloc_pair(bars, tmem_slot, rank)


bprop_d128_accum_grouped_2cta_stail_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def main_thd_2cta_stail_host(
    maps,
    lse,
    delta,
    meta,
    origins,
    dq_accum,
    n_batch,
    nheads,
    max_skv,
    stats_origin: cutlass.Constexpr[int],
    scale,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    ratio: cutlass.Constexpr[int],
    dq_after_dv: cutlass.Constexpr[bool],
    dq_route_code: cutlass.Constexpr[int],
    tma_load_plain: cutlass.Constexpr[bool],
    diag_2cta_code: cutlass.Constexpr[int],
    store_wait_spin: cutlass.Constexpr[bool],
    reg_split_code: cutlass.Constexpr[int],
    warp_roles_code: cutlass.Constexpr[int],
    mask_form_code: cutlass.Constexpr[int],
    dkv_fp32: cutlass.Constexpr[bool],
    debug_clk: cutlass.Constexpr[bool],
    dbg_clk,
    stream,
):
    """One owner PAIR per Q head and declared 256-row KV-envelope unit (header section 14); no host length read.  ``dq_accum`` (the fp32 dQ
    workspace tensor) is passed to the kernel ONLY under a red ``dq_route_code``; under the default it is None = dropped from the specialised
    signature (the default rendering stays byte-identical).  ``dbg_clk`` (the Int32 STAMPS dump, config dbg_clk_records x DBG_CLK.WORDS
    words) is passed ONLY under ``debug_clk``, the same way.  ``warp_roles_code`` = the role shape (header section 11; 0 = 16w, the default)."""
    n_pairs = cute.math.max((max_skv + CFG.THD_UNIT_ROWS - 1) // CFG.THD_UNIT_ROWS, cutlass.Int32(1))
    dq_accum_red = dq_accum if cutlass.const_expr(dq_route_code != DQ_ROUTE_TMA) else None
    dbg_clk_on = dbg_clk if cutlass.const_expr(debug_clk) else None
    bprop_d128_accum_grouped_2cta_stail_kernel(
        maps,
        lse,
        delta,
        meta,
        origins,
        dq_accum_red,
        n_batch,
        stats_origin,
        scale,
        causal,
        dtype,
        ratio,
        dq_after_dv,
        dq_route_code,
        tma_load_plain,
        diag_2cta_code,
        store_wait_spin,
        reg_split_code,
        warp_roles_code,
        mask_form_code,
        dkv_fp32,
        debug_clk,
        dbg_clk_on,
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
    "BAR_LAYOUT_XFA4",
    "Bars",
    "BarsXFA4",
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
    "bprop_d128_accum_grouped_2cta_stail_kernel",
    "main_thd_2cta_stail_host",
]
