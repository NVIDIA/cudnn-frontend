# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""SM107 (Rubin) SDPA backward, d_qk = d_v = 256, MXFP8 (E4M3 payloads + E8M0 block scale factors): dV in-kernel + the dS workspace
(P-b, the default: block-scaled e4m3 x 2 with E8M0 atoms | P-c: bf16).

The per-tensor FP8 body ``bprop_d256_fp8.py`` with block scaling plumbed in (exactly seven deltas, listed at the end of this
docstring; the e4m3 P ring stays in TMEM and every scale-factor atom aliases its dead slot).  One cga2 pair (CGA_M=2, CGA_N=1, CTA_MMA=2) owns one
256-row kv block of one (batch, head) and walks the q tiles that attend it.  It computes **dV in-kernel** (TMEM accumulator over the
q loop, TRUE units, stored per Q-head as bf16 / fp16) and writes **dS** = the fp32 ``attn_scale * P * (dP - delta)`` to GMEM
workspaces ``[B, H_chunk, S_kv, S_q]`` under the load-time policy ``CFG.DS_SF_POLICY``: **P-b** (``DS_SF_P_B`` =
``DS_SF_POLICY_DEFAULT``, what ships: exact 1x32 block-scaled e4m3 BOTH ways) writes TWO e4m3 payloads plus their E8M0 atoms --
``ds_dk`` scaled per 32-q block of a kv row (the dK GEMM's A, K-major) and ``ds_dq`` scaled per 32-kv block of a q column (the dQ
GEMM's A, MN-major), ``sf_ds_dk`` / ``sf_ds_dq`` in cuDNN's F8_128x4 atom order -- for the block-scale stage-3 GEMM arm; **P-c**
(``DS_SF_P_C``, the bf16-dS oracle twin, selectable) rounds it once to **bf16** (the bf16 stage-3 GEMM renderings consume it over
dequantized Q_T / K_T).  Every MMA is a
``tcgen05.mma.block_scale`` (kind MXF8F6F4,
BLOCK32, K64 per instruction): the accumulators are in TRUE units, so the body has NO per-tensor scale, NO amax and NO atomics.  lane =
kv row (S = [kv, q]).  Twelve warps per CTA, ONE warp-specialized body:

    MMA leader -- the S issue order is a COMPILE-TIME property of the mask arm (``S_LOOKAHEAD``, derived from the mask flags next
    to ``SPIN_RING_WAITS``; the two orders are bitwise identical and MEASURED there -- dense faster at the top, causal with the lookahead):
        masked arms (S_LOOKAHEAD True, the fp8 body's lookahead):  Q.K[q_lo] -> { dO.V[i] ; Q.K[i+1] ; P.dO[i] } -> dO.V[last] ; P.dO[last]
        dense arm   (S_LOOKAHEAD False, Q.K at the iteration top):  Q.K[q_lo] -> dO.V[q_lo] ; P.dO[q_lo] -> { Q.K[i] ; dO.V[i] ; P.dO[i] }
        BMM1 S  = K . Q^T    -> S_acc[kv, q]   TMEM [  0, 128)   A = K (per tile), B = Q[i]        mma_ss  sf_a = SF_K, sf_b = SF_Q
        BMM1 dP = V . dO^T   -> dP[kv, q]      TMEM [128, 256)   A = V (per tile), B = dO[i]       mma_ss  sf_a = SF_V, sf_b = SF_dO
        BMM2 dV = P . dO_T   -> dV[kv, d_v]    TMEM [256, 512)   A = e4m3 P (TMEM ring slot p), B = dO_T[i] (BT)  mma_ts  sf_a = SF_P (constant 119), sf_b = SF_dOT
        Before each MMA the warp UTCCPs (``tcgen05.cp 32x128b``, one call per 512-B F8_128x4 atom = 4 TMEM columns) the scale factors
        that MMA reads INTO THE DEAD P-RING SLOT of the iteration (slot s = p ^ 1, the TMEM table below): 11 atoms per q iteration --
        V-SF + dO-SF before dO.V[i], K-SF + Q-SF before the iteration's Q.K (Q.K[i+1] under the lookahead, Q.K[i] at the top otherwise),
        P-SF + dO_T-SF before P.dO[i] -- plus K-SF + Q-SF before the prologue Q.K.
    8 compute warps (2 warpgroups x 4, each wg owns a 64-wide q half), per q iteration:
        softmax : P = exp2(S * attn_scale_log2e - lse * log2e)   (TRUE units: SF_K / SF_Q dequantize S in-MMA; the oracle's spelling)
                  -> transposed bit-word mask (+ the q < seqlen_q_real band under MASK_Q_PAD) -> e4m3(P * 2^8) by the scaled cvt with the
                  CONSTANT byte 119 (``fp32_to_fp8_pack_scaled``; ``fused=False`` = one FMUL by an opaque 256.0, bit-identical) ->
                  wait mb_p_sf_consumed (P.dO[i-1] no longer reads the scale factors aliased in this slot) -> tcgen05_st into P slot p ->
                  tcgen05_wait(STORE) -> the relaxed arrive on mb_p_ready (the dV BMM2 starts NOW)
        dsoftmax: dS = (dP * attn_scale - delta * attn_scale) * P   (fp32, the UNQUANTIZED P; dP is TRUE units)
                  P-c: -> bf16 -> sdS SMEM ring (2 stages) -> TMASTG -> GMEM workspace
                  P-b: -> (a) per 32-q block of a lane's kv row: abs_max_tree -> e8m0_pair byte -> the scaled cvt (fused on cc 10.7)
                          -> e4m3 ds_dk into the sdS ring slot + the two bytes into the sf_ds_dk atom (one st.u16 per lane);
                       (b) per q column: warp_abs_max_f32 (ONE redux.sync.max.abs.f32 over the warp = the column's 32-kv block) ->
                          e8m0_pair_u (the column pair's bytes, the e8m0_pair rule as one packed multiply per pair) -> the LONE
                          fused scaled cvt per element with its column's byte (fp32_to_fp8x4_scaled_pairs) -> e4m3 ds_dq into the
                          sdS_kv ring slot; lane l stores the bytes of columns 2l, 2l+1 out of the same pair words into the
                          sf_ds_dq atom.  Four buffers, ONE proxy fence, the same arrive.
    per kv tile (post q loop): dV epilogue = dV_acc (TRUE units) -> bf16 / fp16 -> sdV SMEM (aliases K+V, dead by then) -> TMASTG -> GMEM
    TMALDG: K + K-SF, V + V-SF once per kv tile; Q + Q-SF, dO + dO-SF (dP view) and dO_T + dO_T-SF (dV view, the COLUMNWISE-quantized
        dO payload through the BT box) per q iteration -- the SF loads ride the operand's ``_full`` mbarrier (tx grown by the SF bytes).
    TMASTG: dS ring slot -> workspace per q iteration (P-b: both payload slots + the two 512-B atoms, ONE bulk group); dV -> GMEM per kv tile.
    Scheduler warp: try_cancel protocol FUSED with the lse / delta SMEM prefetch ring (lane = q col); folds lse * log2e and delta * attn_scale.

MXFP8 contract (cuDNN ``sdpa_mxfp8_backward``): Q, K, V, dO are E4M3 with ROWWISE E8M0 scale factors (32 consecutive d share a byte;
``sf_v`` is rowwise in the backward: V is BMM1 dP's A operand contracting over d); dO_T is the COLUMNWISE quantization of dO (32
consecutive q share a byte, the BMM2 contraction axis) with its own SF; every SF tensor is in cuDNN's F8_128x4 atom order (atom byte
``(r % 32) * 16 + (r // 32) * 4 + c``; ``tile_dsl.sf_layout``).  P is quantized with ONE fixed power-of-two scale ``P_q = e4m3(P * 2^8)``
descaled in BMM2 by the constant E8M0 byte ``127 - 8 = 119`` (``CFG.P_SF_BYTE``; the SM100 chain's ``p_scale_log2`` convention; the
oracle ``mxfp8_ref.compute_ref_backward`` models it as ``e4m3(p * 256) / 256``).  dS: NOT quantized under P-c; under P-b quantized
from the fp32 value per 32-element block along BOTH axes with ``e8m0_pair`` / ``e8m0_pair_u`` (= the oracle's
``e8m0_ceil(amax * fp32(1/448))``; an all-zero block -> byte 0x00 -> payload 0) and ``e4m3(x * 2^(127 - e))``
(``fp32_to_fp8_pack_scaled`` along q, ``fp32_to_fp8x4_scaled_pairs`` along kv; their ``fused=False`` arms are ``e8m0_rcp`` + the
plain cvt) -- the oracle's ``quantize_ds=True`` arm.  P-a (the 32x32 tile scale) is REFUSED at load time (optional
validation only, never built here).  No ``descale_*`` / ``scale_*`` scalars, no ``amax_*`` outputs (a graph requesting them is declined, typed).

Workspace head-chunking, GQA, the masked q range rounded to the stage-3 pair (``_q_loop_bounds``), the persistent scheduler and the
Launch ABI's ``head_base`` / ``seqlen_kv_real`` are the fp8 body's, unchanged.

--------------------------------------------------------------------------------------------------------------------
TMEM (one ``tcgen05.alloc.cta_group::2`` of TOTAL_COLS = 576 per CTA, ``is_exclusive=True``; ``config_sm107.tmem_layout(FAMILY_MXFP8)`` ==
the fp8 body's map; alloc AND dealloc pass ``LAYOUT.TOTAL_COLS``)
    [  0, 128)  S_acc   fp32 [kv, q]     BMM1 S  ->  softmax  ``tmem_load_tile(S_OFF + q_half_off, 64)``
    [128, 256)  dP      fp32 [kv, q]     BMM1 dP ->  dsoftmax ``tmem_load_tile(dP_OFF + q_half_off, 64)``
    [256, 512)  dV_acc  fp32 [kv, d_v]   BMM2 dV (accumulate=(q_iter > q_lo)), TRUE units -> epilogue
    [512, 544)  P slot 0  e4m3 P[kv, q]: P_COLS = 32 = TILE_N * BPE / 4 (wg0 cols 0-15 = q 0-63, wg1 16-31)  softmax ``tcgen05_st`` -> BMM2 ``mma_ts`` A
    [544, 576)  P slot 1                                                                                  ZERO dedicated SF columns, RSVD 0
    THE SLOT RULE (one statement; every site derives from it).  n = the kernel-global q-iteration count = the ``mb_p_ready``
    PipelineState (2 stages, advanced once per q iteration by the softmax AND by the MMA warp, continuing across kv tiles):
        p(n) = n % 2 = p_ready_state.idx   the P slot of iteration n (the softmax stores P[n] there; P.dO[n] reads it)
        s(n) = p(n) ^ 1                     the SF slot of iteration n: EVERY UTCCP of iteration n lands there and every MMA of
                                            iteration n (dO.V[n], P.dO[n] and the Q.K the iteration issues -- Q.K[n+1] under
                                            S_LOOKAHEAD, Q.K[n] at the top otherwise) reads its scale factors there
    Inside the SF slot (slot-relative ``LAYOUT.SF_*_OFF``, derived from the atom counts; ``sf_base = P_OFF + s(n) * P_COLS``):
        BMM1 band  K [0, 8) | V [8, 16) | Q [16, 24) | dO [24, 32)  = SF_BMM1_COLS 32 = the whole slot   -> dO.V[n] (V, dO), the iteration's Q.K (K, Q)
        BMM2 band  P [0, 4) | dOT [4, 12)                            = SF_BMM2_COLS 12, written over the dead K / V columns AFTER
                                                                       the iteration's Q.K and dO.V[n] were issued      -> P.dO[n]
    The prologue Q.K[q_lo] of a tile whose first iteration is n0 takes K-SF + Q-SF in s(n0) (= p(n0-1), the previous tile's last P
    slot, read by P.dO[n0-1] which precedes the copies in the stream); iteration n0 refills the same slot after this MMA (V, dO for
    dO.V[n0]; K again + Q[q_lo + 1] too under S_LOOKAHEAD; P, dOT over the K / V columns).
    SF column walk inside one MMA (``tile_dsl/mma.py``, identical in ``mma_ss`` and ``mma_ts``: ``sf_id = (k * 2) % 4``, column group
    +4 every two K64 steps): BMM1 (K = 256 = 4 steps) k = 0 / 1 read atom 0 (sf_id 0 / 2), k = 2 / 3 atom 1; BMM2 (K = 128 = 2 steps)
    k = 0 / 1 read the one atom.
    Ordering of the alias (the in-order tcgen05 stream of the MMA warp's elected lane orders every copy against every MMA it issued):
      (1) the BMM1 copies into s(n) follow P.dO[n-1], which read P[n-1] from that very slot; (2) softmax[n-1]'s store of P[n-1] there
      completed before mb_p_ready[s(n)] was waited (tcgen05_wait(STORE) precedes the arrive); (3) the BMM2 copies over K / V's
      columns follow the iteration's Q.K / dO.V[n]; (4) softmax[n+1]'s store of P[n+1] into s(n) follows mb_s_acc_full[n+1] (the
      commit after Q.K[n+1]: under S_LOOKAHEAD it also covers dO.V[n]; issued at the top of iteration n+1 it covers P.dO[n] as well)
      AND mb_p_sf_consumed (the commit after P.dO[n], covering P.dO[n] and the BMM2 copies -- what the lookahead arm needs; the
      dense arm's s_acc_full[n+1] alone would do, the ring is kept under both orders) -- THE ONE RACE the stream cannot close
      (another thread's tcgen05.st) and the one new ring (BARRIER TABLE).

SMEM (declaration order == ``config_sm107.smem_layout(FAMILY_MXFP8)``: rows 1-11 under every dS policy, the dS rows per ``CFG.DS_SF_POLICY``;
every slab 1024-B aligned; KiB @ KiB offset; descriptor
roots in bytes.  The CONFIG model is pinned by ``test_mxfp8_design_a_smem_table_offsets_roots_and_desc_version``; that the BODY lands its
slabs at those offsets is MEASURED on the sm_107a PTX: every slab base below appears as an immediate in the dumped ``kernel.ptx`` --
``test_sm107a_ptx_carries_the_config_slab_offsets`` (2026-09-30, host trace-compile == the Rubin node cubin of the SMEM-P-ring predecessor,
REG 168 / SHARED 1024 + 320 KiB dynamic there; 288 KiB (P-c) / 290 KiB (P-b) dynamic here))
    #  buffer      dtype x elems               KiB @off   writer (how)                     reader (how)                lane stride    swizzle + WHY                        roots
    1  sQ          e4m3 x 3 x (64 x 256)        48 @  0   TMA (box 1,64,1,128 x2)           MMA desc, B of BMM1 S       --             128 B s128b: TMA write + MMA desc   0 / 16384 / 32768
                                                                                                                                       read (job 1)
    2  sdO         e4m3 x 3 x (64 x 256)        48 @ 48   TMA (rowwise dO, same box)        MMA desc, B of BMM1 dP      --             128 B, job 1                        49152 / 65536 / 81920
    3  sdOdv       e4m3 x 3 x (128 x 128)       48 @ 96   TMA (box 1,128,1,128) over dO_T   MMA desc, B (BT) of BMM2    --             128 B, job 1                        98304 / 114688 / 131072
    4a sK          e4m3 x 128 x 256             32 @144   TMA (box 1,128,1,128 x2)          MMA desc, A of BMM1 S       --             128 B, job 1                        147456
    4b sV          e4m3 x 128 x 256             32 @176   TMA                               MMA desc, A of BMM1 dP      --             128 B, job 1 (ELEMENT alias off.)   180224
    4c sdV (=4a+4b post-loop) OUT x 128 x 256   64 @144   compute lanes store_swizzled      TMA store                   128 B (bf16)   Swizzle(3,4,3): row 128 B -> banks  --
                                                          (Swizzle(3,4,3))                                                            spread (job 2) AND the s128b store
                                                                                                                                       descriptor (job 1)
    5  sStats      fp32 x 2 x 256                2 @208   scheduler lanes (4 B stride)      compute lanes, one address  4 B / bcast    LINEAR (job 2 by arithmetic; no    --
                                                                                            on 32 lanes (broadcast)                    descriptor)
    6  sK_SF       Int8 x 1024 (2 atoms)         1 @210   TMA: this CTA's OWN kv tile,      UTCCP desc (32x128b: leading --            NONE: the UTCCP atom layout IS the  215040
                                                          whole slab, self-multicast        16 / stride 128 / layout 0)                F8_128x4 atom
    7  sV_SF       Int8 x 1024                   1 @211   TMA, as 6                         UTCCP desc                  --             none (atom)                         216064
    8  sP_SF       Int8 x 512 (1 KiB slab)       1 @212   every thread, byte 119, then      UTCCP desc, per q iter      1 B            none (atom)                         217088
                                                          fence_proxy BEFORE the init sync
    9  sQ_SF ring  Int8 x 3 x 1024               3 @213   TMA: each CTA ONE atom (its       UTCCP desc                  --             none (atom)                         218112 / 219136 / 220160
                                                          d-chunk), mcast 3 -> both CTAs
                                                          hold the full N tile
    10 sdO_SF ring Int8 x 3 x 1024               3 @216   TMA, as 9                         UTCCP desc                  --             none                                221184 / 222208 / 223232
    11 sdOT_SF ring Int8 x 3 x 1024              3 @219   TMA: each CTA its D-plane (512    UTCCP desc                  --             none                                224256 / 225280 / 226304
                                                          B), mcast 3
    12 sdS ring    bf16 x 2 x (128 x 128)       64 @222   compute lanes store_swizzled:     TMA store box (1,1,128,64)  128 B per lane Swizzle(3,4,3) (job 2 + the s128b   -- (no descriptor reads it)
       (P-c)                                              two 64-col subtiles per row, wg   x2 subtiles                                store descriptor)
                                                          w -> subtile w at slot + w*8192
                                                          + tid*64 elems
    (The e4m3 P ring is NOT an SMEM slab: it is the TMEM ring [512, 576) above, whose dead slot also carries the SF atoms.)
    ~0.6 KiB scaffolding (25 mbarrier rings, scheduler slots, TMEM pointer)  ->  286 KiB slabs + 2 KiB budget = 288 KiB <= 327 KiB
    (Rubin oversized cap; the launcher sets ALLOW_OVERSIZED_SHARED_MEMORY), 39 KiB free of the 325 usable.
    P-b (the default; config_sm107.smem_layout(FAMILY_MXFP8) at DS_SF_P_B) replaces row 12 by THREE slabs -- the dS slot is FOUR
    buffers (two payloads + two atoms); no rcp gather slab is declared (the column scales stay in registers):
    12 sdS_SF      Int8 x 2 x 2 x 512 (atoms)    2 @222   compute lanes: sf_ds_dk atom      TMASTG TMA store, 5-D byte   16 B (the     NONE: the F8_128x4 atom layout IS  -- (no descriptor reads it)
       staging                                            (+0): ONE st.u16 per lane at      descriptor, box [128 B, 4    atom's 16-B   what the stage-3 GEMM's UTCCP
                                                          sf_atom_byte(kv row, 2*wg);       rows, 1, 1, 1] = one atom     line per row) consumes (job 1 forbids a
                                                          sf_ds_dq atom (+512): TWO st.u8   per op                                      permutation; job 2: 4 wavefronts
                                                          per lane at sf_atom_byte(q col                                                per sub-word store, 3 stores per
                                                          2*lane, warp) and +16                                                         lane per iteration -- negligible)
    13 sdS ring    e4m3 x 2 x (128 x 128)       32 @224   compute lanes store_swizzled at   TMA store box (1,1,128,128) 128 B per lane Swizzle(3,4,3) (job 2 + the s128b   -- (no descriptor reads it)
       = ds_dk                                            slot + tid*128 + wg*64 (both      (the fp8 body's e4m3 dS box)              store descriptor)
                                                          wgs' halves in ONE row: P_D_BLOCK
                                                          128)
    14 sdS_kv ring e4m3 x 2 x (128 x 128)       32 @256   as row 13 (the ds_dq payload)     TMA store, the ds_dq          128 B per lane as row 13                            --
       = ds_dq                                                                              descriptor, same box
    ~0.6 KiB scaffolding (25 rings)  ->  288 KiB slabs + 2 KiB budget = 290 KiB <= 327 KiB, 37 KiB free of the 325 usable.  Under BOTH
    policies the six SF slabs are declared BEFORE the dS slabs because tcgen05 descriptors read them (rules/mma-tma-matrix.md S6): the
    highest ROOT is sdOT_SF[2] at 226304 B (221 KiB) < 262144 -> DESC_VERSION = 0 (derived, never a literal; the dS slabs carry no
    root, so the policy does not move it); a root past the line would make the UTCCP copy DATA bytes into
    the SF columns (LSE = +inf / O = NaN on the d512 MXFP8 forward) with no crash.

BARRIER TABLE (lane ledger: SUM(issuing lanes) == init, per CTA.  A phase = one q iteration for the per-q rings, one kv tile for the
per-tile bars.  ``ldr+elect`` = ``pred=is_leader & elect_sync()``; ``pred`` = ``pred=elect_p`` (one lane); ``bare`` = every lane of
the calling warps; ``elect`` = ``if elect_sync():``.  cga2 tensor-TMA bytes land on the LEADER's mbar (P9), so only the leader arms
``expect_tx``; the follower's TMA_LOAD bars are initialised but never armed or waited.  Rows marked ``*`` CHANGED vs the fp8 body,
the row marked ``+`` is ADDED (one commit ring); every other equation is the fp8 body's.)

  bar                 stages  producer            guard        issuing lanes / phase    init            consumer (waits)          scope
* mb_q_full           3       TMA_LOAD (TMALDG)   ldr+elect    1 (leader CTA only)      ONE_LANE=1      MMA leader                LOCAL  tx = 32768 (Q: 2 CTAs x 16 KiB)
                                                                                                                                         + 2048 (Q-SF: 2 CTAs x 512 B x 2 dest) = 34816
* mb_do_full          3       TMA_LOAD            ldr+elect    1                        1               MMA leader                LOCAL  tx 34816 (dO + dO-SF)
* mb_dodv_full        3       TMA_LOAD            ldr+elect    1                        1               MMA leader                LOCAL  tx 34816 (dO_T + dO_T-SF: 2 CTAs x 512-B plane x 2 dest)
* mb_k_full / v_full  1       TMA_LOAD            ldr+elect    1                        1               MMA leader                LOCAL  tx 65536 + 2048 (K-SF / V-SF: 2 CTAs x 1024 B,
                                                                                                                                         self-multicast) = 67584 each
  mb_q_empty          3       MMA_COMMIT mcast    pred         1 per target CTA         1               TMALDG (both CTAs)        LOCAL  drained x3 at exit (P15)
  mb_do_empty         3       MMA_COMMIT mcast    pred         1 per target CTA         1               TMALDG (both)             LOCAL  drained x3
  mb_dodv_empty       3       MMA_COMMIT mcast    pred         1 per target CTA         1               TMALDG (both)             LOCAL  drained x3
  mb_k_empty/v_empty  1       MMA_COMMIT mcast    pred         1 per target CTA         1               TMALDG (both)             LOCAL  drained x1
  mb_s_acc_full       1       MMA_COMMIT mcast    pred         1 per target CTA         1               softmax 256 lanes (both)  LOCAL
  mb_dp_full          1       MMA_COMMIT mcast    pred         1 per target CTA         1               softmax 256 (both)        LOCAL
  mb_dv_ready         1       MMA_COMMIT mcast    pred         1 per target CTA         1               softmax 256 (both)        LOCAL
  mb_s_acc_empty      1       LEADER (softmax)    bare         256 lanes x 2 CTAs       512             MMA leader (pre-armed)    LEADER drained x1 at exit.  Gates the loop's
                                                                                                                                         Q.K: Q.K[i+1] after dO.V[i] (S_LOOKAHEAD) or
                                                                                                                                         Q.K[i] at the top of iteration i, after the
                                                                                                                                         mb_dp_empty wait (dense) -- N waits + N
                                                                                                                                         arrives per tile under either order.
  mb_dp_empty         1       LEADER (softmax)    bare         256 x 2                  512             MMA leader (pre-armed)    LEADER drained x1
  mb_p_ready          2       LEADER (softmax)    bare         256 x 2                  512             MMA leader                LEADER  the fp8 body's row: relaxed arrive after
                                                                                                                                         tcgen05_wait(STORE) (TMEM data needs no
                                                                                                                                         release).  No p_empty: P slot p(n) is
                                                                                                                                         rewritten at iteration n + 2 only after
                                                                                                                                         mb_s_acc_full[n + 2] (commit after Q.K[n + 2],
                                                                                                                                         after P.dO[n] in the stream under BOTH S
                                                                                                                                         issue orders) AND after
                                                                                                                                         mb_p_sf_consumed[n + 1] (next row).
+ mb_p_sf_consumed    1       MMA_COMMIT mcast    pred         1 per target CTA         1               softmax 256 lanes of      LOCAL  ONE commit per q iteration, right after
                              (after P.dO[n])     (elect_p)    (one elected lane x                      EACH CTA (pre-armed:              P.dO[n]: "P.dO[n] no longer reads slot
                                                               one commit per target)                   start(phase=1)); drained          s(n) = p(n + 1)".  Every softmax lane
                                                                                                        x1 after the loop (P15)           waits it before its tcgen05_st of P[n + 1]
                                                                                                                                         into that slot -- the one write the
                                                                                                                                         in-order tcgen05 stream cannot order (a
                                                                                                                                         different thread's store).  Lane ledger:
                                                                                                                                         SUM(issuing lanes) = 1 x 1 commit per
                                                                                                                                         target CTA = 1 == init 1.  Phase count:
                                                                                                                                         one commit + one wait per q iteration on
                                                                                                                                         every path (the forced N >= 1 tile too),
                                                                                                                                         wait k consumes commit k - 1, the drain
                                                                                                                                         consumes the last; commit k follows wait k
                                                                                                                                         (P.dO[k] needs mb_p_ready[p(k)] needs the
                                                                                                                                         store after wait k), so the 1-stage ring
                                                                                                                                         never runs two phases ahead.
  mb_dv_acc_empty     1       LEADER (softmax)    bare         256 x 2                  512             MMA leader                LEADER
  mb_stats_full       2       THREAD (scheduler)  bare         32 (one warp)            ONE_WARP=32     softmax 256               LOCAL
  mb_stats_empty      2       THREAD (softmax)    bare         256                      256             scheduler (pre-armed)     LOCAL
* mb_ds_smem_full     2       THREAD (softmax)    bare         256                      256             TMASTG                    LOCAL  XFER_STAGES = 2 under P-c AND P-b (neither
* mb_ds_smem_empty    2       THREAD (TMASTG)     elect        1                        1               softmax (pre-armed)       LOCAL  the bf16 ring nor two e4m3 rings fit 3-deep;
                                                                                                                                         the fp8 body only ran 3-deep).  P-b: the slot
                                                                                                                                         carries FOUR buffers (ds_dk, ds_dq, the two
                                                                                                                                         atoms) -- every lane's stores precede its ONE
                                                                                                                                         fence_proxy, then the SAME arrive; the TMASTG
                                                                                                                                         frees the slot after ONE bulk group (4 TMA
                                                                                                                                         stores).  Same lanes, same counts.
  mb_dv_stg_full      1       THREAD (softmax)    bare         256                      256             TMASTG                    LOCAL
  mb_dv_stg_empty     1       THREAD (TMASTG)     elect        1                        1               TMALDG (pre-armed; F1)    LOCAL  drained x1
  mb_tmem_dealloc     1       THREAD (softmax)    bare local + bare arrive_on_peer  256 local + 256 peer = 512  MMA warp of each CTA  LOCAL
  sched.mb_scheduler  2       expect_tx 16 B      elect (cga-first CTA arms BOTH)  1 per CTA          1               every persistent warp     LOCAL
  sched.mb_read_tile_id 2     read_tile_id_arrive one predicated arrive per calling warp on EVERY CTA   READ_TILE_ARRIVERS_TOT=21  scheduler
        = leader (8 softmax + MMA + TMALDG + TMASTG = 11) + follower (8 + TMALDG + TMASTG = 10); the scheduler warp never credits.

Named barrier 1 (288 = 32 x 9 threads): MMA warp ``barrier_cta_arrive`` <-> softmax ``barrier_cta_sync`` publishes the TMEM base
after ``tmem_alloc``.  Every LEADER-scope arrive (mb_s_acc_empty / mb_dp_empty / mb_dv_acc_empty) is preceded by the
``tcgen05_wait(LOAD)`` that orders it after the slot's last LDTM; mb_p_ready by the ``tcgen05_wait(STORE)`` that completes the P
store (the fp8 body's pairs).  The constant P-SF fill is a generic store
the async-proxy UTCCP reads: its ``fence_proxy("async.shared")`` sits BEFORE ``fence_mbarrier_init`` / ``barrier_cta_sync`` /
``cga_arrive`` (the forward's order).  P15 drains at the tail of the CONSUMER: TMALDG drains the five multicast ``_empty`` rings,
the MMA leader drains the two pre-armed LEADER-scope rings, the softmax warps drain ``mb_p_sf_consumed``, the softmax's
``mb_tmem_dealloc`` is waited by both CTAs' MMA warps.
The UTCCP sites need no barrier of their own against the MMAs: ``tcgen05.cp`` and ``tcgen05.mma`` are one in-order async stream per
issuing thread, so a copy issued before the MMA that reads it (and after the MMA that read the previous contents) is ordered by
issue order -- the forward's per-stage refill pattern, in production.  What the stream CANNOT order is the softmax lanes'
``tcgen05.st`` into the slot the copies and P.dO[n] used: that is ``mb_p_sf_consumed``.  That argument -- and every commit-based
``_empty`` / ``_full`` ring here -- rests on the SAME lane issuing the UTCCP (``if nvvm.elect_sync():``), the ``mma_ss`` k-steps
(its internal per-step ``elect_sync``) and the ``tcgen05.commit`` (``pred=elect_p``): ``tcgen05.commit`` tracks only the executing
thread's prior tcgen05 ops.  It holds because the MMA warp is a converged full-mask warp at every site (``elect_sync`` picks the
lowest active lane; the predicated arrives branch round the native op and reconverge) -- checked lane by lane in the barrier-table audit.

The seven deltas vs ``bprop_d256_fp8.py``: (1) the fp8 P pipeline is KEPT (TMEM ring, ``tcgen05_st`` + ``tcgen05_wait(STORE)`` +
the relaxed LEADER arrive, BMM2 = ``mma_ts``); every SF atom ALIASES the dead P slot of its iteration (the slot rule above) and ONE
commit ring ``mb_p_sf_consumed`` closes the softmax-store-vs-SF-read race; (2) six SF TMA loads on the existing ``_full`` bars, tx
grown by ``CFG.*_SF_TX``; (3) UTCCP in the MMA warp right before the MMA that reads the atoms, into the alias slot,
``if elect_sync():``-gated (11 per q iteration + 4 per tile prologue); (4) block-scale descriptors (``Tcgen05MxInstrDesc`` k_dim=1,
``MmaDesc`` MXF8F6F4 / BLOCK32 / ``sf_blocks_per_step=CFG.SF_BLOCKS_PER_STEP``) and ``mma_ss`` / ``mma_ts(..., tmem_sf_a=, tmem_sf_b=)``
on all three MMAs;
(5) the ``sdO_dv`` ring loads dO_T through the SAME BT box over a new descriptor; (6) TRUE-unit softmax (``lse * log2e``, no shift;
``dp_scale = dot_scale = attn_scale``), P quantized by the scaled cvt with the constant byte 119, dS bf16 (P-c) or block-scaled e4m3
x 2 + two E8M0 atoms per (kv tile, q tile) (P-b: in-lane ``abs_max_tree`` along q, ``warp_abs_max_f32`` along kv, ``e8m0_pair`` /
``e8m0_pair_u`` bytes),
dV bf16 / fp16, no descales / amax / atomics; (7) the launch ABI below (payloads + six SF tensors + the appended P-b dS operands; no
scalars, no amax).

## Launch ABI

``compile(b, qh, kh, sq, skv, qh_chunk=0)`` (all ``lru_cache``-d Python ints; ``qh_chunk=0`` means ``qh``) returns a callable
taking, POSITIONALLY (torch tensors bind through tvm-ffi):

    fn(q, k, v, do, do_T, dv, ds_ws, lse, delta,
       sf_q, sf_k, sf_v, sf_do, sf_do_T,
       (b, qh, kh, sq, skv, qh_chunk),          # the compile-time problem_size tuple, repeated at the call
       attn_scale, attn_scale_log2e,            # cutlass.Float32: softmax scale, and attn_scale * log2(e)
       head_base, seqlen_kv_real, seqlen_q_real,  # cutlass.Int32: first full-tensor head of this chunk; REAL S_kv (<= skv); REAL S_q (<= sq)
       ds_dk, ds_dq, sf_ds_dk, sf_ds_dq,        # APPENDED (P-b, the DEFAULT path): the two block-scaled payloads + their E8M0 atoms; None under P-c
       seq_kv_lens | None = None,               # APPENDED: int32 [B] per-batch REAL kv lengths under ``seq_kv_lens_present`` (the PADDED arm
                                                #   reads seq_kv_lens[b] in place of seqlen_kv_real); None-specialized (pass nothing) otherwise
       stream=<CUstream>)

    The dS policy (``CFG.DS_SF_POLICY``, a load-time constant) decides which dS operands are LIVE; the others are passed as None and
    None-specialized away (the fp8 body's Optional amax idiom) -- ONE positional shape for both policies:
        P-b (the default): ds_ws None; ds_dk / ds_dq / sf_ds_dk / sf_ds_dq live.
        P-c:               ds_ws (bf16) live; ds_dk / ds_dq / sf_ds_dk / sf_ds_dq None.

    q, k        e4m3  [B, S_q, H_q, 256] / [B, S_kv, H_kv, 256]   BSHD, d contiguous, seq/head strides 16-B multiples; ROWWISE-quantized
    v           e4m3  [B, S_kv, H_kv, 256]   ROWWISE-quantized (BMM1 dP's A operand contracts over d)
    do          e4m3  [B, S_q, H_q, 256]     ROWWISE-quantized (the dP view)
    do_T        e4m3  [B, S_q, H_q, 256]     the COLUMNWISE quantization of the same dO (32 consecutive q share a scale; the dV view),
                                             dO's BSHD-physical layout
    dv          OUT   [B, S_kv, H_q, 256]    bf16 / fp16 per Q-HEAD partial in TRUE units (GQA fold is the adapter's dkv_reduce)
    ds_ws       bf16  [B, qh_chunk, S_kv, S_q]   P-c: chunk-local head axis; dS = attn_scale * P * (dP - delta) rounded to bf16
    ds_dk       e4m3  [B, qh_chunk, S_kv, S_q]   P-b: dS scaled per 32-q block of a kv row (q contiguous: the dK GEMM's A, [M = kv, K = q] K-major)
    ds_dq       e4m3  [B, qh_chunk, S_kv, S_q]   P-b: dS scaled per 32-kv block of a q column (same geometry; the dQ GEMM's A, read [M = q, K = kv]
                                             MN-major)
    sf_ds_dk    uint8 [B, qh_chunk, S_kv/128, S_q/128, 512]  P-b: one F8_128x4 atom per (kv_tile, q_tile), byte (r % 32) * 16 + (r // 32) * 4 + c
                                             with r = kv within the tile, c = q-block within the 128-q tile (the dK GEMM's SFA)
    sf_ds_dq    uint8 [B, qh_chunk, S_q/128, S_kv/128, 512]  P-b: one atom per (q_tile, kv_tile), r = q within the tile, c = kv-block (the dQ GEMM's
                                             SFA).  An atom is written iff its payload tile is (the [q_lo, q_hi) range rounded to the 256-row
                                             q pair, per CTA over its own 128-kv tile), so the GEMM arm's K-trim reads no unwritten atom.
    lse         fp32  [B, H_q, S_q]  NATURAL-log LSE of the forward (the kernel applies log2e); rows past the real S_q must read +inf
    delta       fp32  [B, H_q, S_q]  rowsum(dO * O) in TRUE units, UNSCALED (the half row's host arm on o_f16 / dO_f16); rows past the real
                                 S_q must be FINITE (0): a pad q column's P is a SELECT-zero but dS = (dP * s - delta) * P multiplies it,
                                 so a NaN delta pad lands NaN in the workspace's pad columns (MEASURED 2026-09-30 on Rubin with a NaN-poisoned delta pad).
                                 ``dot_do_o_host`` over zero-padded o / dO yields exactly that 0.
    sf_q, sf_do uint8 [B, H_q, ceil128(S_q), 8]    rowwise F8_128x4 atoms, per-(b, h, 128-row tile) contiguous (1024 B per tile)
    sf_k, sf_v  uint8 [B, H_kv, S_kv, 8]           rowwise at the PADDED S_kv (skv % 256 == 0): the descriptors span S_kv / 128 tiles per head
                                                   (the head / batch strides follow), so the pad tile's atoms are read -- they must be
                                                   FINITE (zero-filled) whenever seqlen_kv_real < S_kv; the rows are P-select-dead
    sf_do_T     uint8 [B, H_q, 8, ceil128(S_q)]    COLUMNWISE F8_128x4 atoms, D-plane-major (plane stride = B*H*tiles atoms: grows with S)
    The SF tensors are consumed by base address + the F8_128x4 atom rule only (``sdpa/kernels/_mxfp8_sf.py``); the shapes above fix
    their rank and byte count for the tvm-ffi binding.  ``seqlen_q_real`` feeds the ``q < seqlen_q_real`` band of the transposed mask
    (``CFG.MASK_Q_PAD``, set by the adapter iff ``S_q % 128 != 0``; folded out otherwise) AND the bottom-right causal diagonal
    (``kv <= q + (S_kv_real - S_q_real)``, the REAL lengths -- a ragged S_q is served, not declined).  Under the PADDED arm the kv
    length is per batch entry (``seq_kv_lens[b]``) and so is the diagonal.
    Shapes: sq % 128 == 0, skv % 256 == 0 (the adapter pads; a padded S_kv REQUIRES the MASK_PADDED specialization with
    ``seqlen_kv_real`` the uniform real length or the per-batch ``seq_kv_lens`` AND zero-filled sf_k / sf_v pad atoms; a padded
    S_q REQUIRES MASK_Q_PAD AND zero-filled SF pad rows / groups; both measured RED-then-green on Rubin, 2026-09-30),
    qh % kh == 0, qh % qh_chunk == 0.  K / V rows between a batch entry's real kv length and skv (and their SF atoms) must hold
    FINITE data under the per-batch arm (P-select-dead, but dS = (dP * s - delta) * P multiplies them).
    Grid: NATURAL ``(skv/256 * 2, qh_chunk, b)``; LPT / LPT_L2 the flat ``(skv/256 * qh_chunk * b * 2, 1, 1)``; cluster (2, 1, 1);
    384 threads; SMEM ``config_sm107.kernel_smem_bytes(CFG)`` (oversized mode).
    Masks are TemplateParams (module-load time): ``window_right=0`` causal, ``window_left=W`` SWA, ``bottom_right``;
    ``scaled_fp8_pack=True`` (cc 10.7 only) selects the fused scaled cvt (P and the ds_dk payload); ``mask_q_pad=True`` the q band;
    ``ds_sf_policy`` P-b (the default, ``DS_SF_POLICY_DEFAULT``) or P-c -- P-a is refused.
"""

from functools import lru_cache
from typing import Callable, NamedTuple, Optional, Tuple

import cuda.bindings.driver as _cuda_driver  # noqa: F401
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith
from cutlass.experimental import primitives as nvvm
from cutlass.experimental import primitives as prims
from cutlass.experimental.cuda import tensor_map as tmap

from cudnn.frost.compiled_cache import compile_cached as _compile_cached, template_key as _template_key
from cudnn.frost.tile_dsl.barrier import MBarrier, PipelineState, Producer, Scope, advance, arrive_expect_tx, cga_arrive, cga_wait, wait
from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_E4M3, DTYPE_FP16
from cudnn.frost.tile_dsl.handles import GmemTileTma, MmaDesc, SmemTile
from cudnn.frost.tile_dsl.mask import (
    MASK_CAUSAL,
    MASK_NONE,
    MASK_PADDED,
    MASK_SWA,
    apply_mask_words,
    band_mask_words,
    compute_q_loop_bounds,
)
from cudnn.frost.tile_dsl.mma import mma_ss, mma_ts
from cudnn.frost.tile_dsl.pointwise import (
    abs_max_tree,
    e8m0_pair,
    e8m0_pair_u,
    fp32_to_fp8_pack_scaled,
    fp32_to_fp8x4_scaled_pairs,
    opaque_e4m3_max_rcp_in_lane,
    tmem_load_tile,
    warp_abs_max_f32,
)
from cudnn.frost.tile_dsl.scheduler import SCHED_NATURAL, Sched, read_clc_payload, read_tile_id_arrive
from cudnn.frost.tile_dsl.sf_layout import SF_ATOM_LINE_BYTES, sf_atom_byte
from cudnn.frost.tile_dsl.tma import tma_load_tile, tma_store_commit, tma_store_tile, tma_store_wait
from cudnn.frost.tile_dsl.tmem import tmem_alloc, tmem_dealloc
from cudnn.sdpa.bwd.config_sm107 import (
    DS_SF_P_A,
    DS_SF_P_B,
    DS_SF_P_C,
    FAMILY_MXFP8,
    SF_ATOM_BYTES,
    SF_ATOM_COLS,
    SF_ATOM_ROWS,
    SF_TMEM_COLS_PER_ATOM,
    TMEM_IS_EXCLUSIVE,
    TemplateParams,
    buffer_elems,
    desc_version,
    make_cfg_d256_bwd,
    q_write_tiles,
    tmem_layout,
)
from cudnn.sdpa.kernels._mxfp8_sf import build_columnwise_sf_desc, build_ds_sf_atom_desc, build_rowwise_sf_desc, sf_peer_split, sf_tma_rows

# Config comes from the FROST template loader, never an environment variable: the loader injects FROST_TEMPLATE_PARAMS
# before this body runs; the default keeps a plain `import` usable as a standalone driver (E4M3 is the only io this body
# serves).  The bare record traces P-c: bf16 dS, a 2-deep dS ring, bf16 dV, SCALED_FP8_PACK 0 (the FMUL arm), MASK_Q_PAD 0.
PARAMS: TemplateParams = globals().get("FROST_TEMPLATE_PARAMS", TemplateParams(dtype_qkv=DTYPE_E4M3))
CFG = make_cfg_d256_bwd(PARAMS, FAMILY_MXFP8)
if not CFG.IS_MXFP8:
    raise ValueError(f"{__name__}: this body is the MXFP8 family (make_cfg_d256_bwd(..., FAMILY_MXFP8)); got IS_MXFP8={CFG.IS_MXFP8}")
# THD / varlen is admitted by the shared config record (the f16 and fp8 bodies serve it) but this body has no THD arm yet: its
# scale factors are per-(batch, head, 128-row tile) atoms with no per-sequence packing / pad staging, so a THD record here would
# trace a dense body reading per-sequence lengths it never addresses.  Refuse at template load (a flag a body does not read is a
# claim it cannot honour); the adapter declines THD on this row before reaching here.
if PARAMS.thd_varlen:
    raise ValueError(
        f"{__name__}: the MXFP8 body has no THD / varlen arm (per-sequence scale-factor packing and pad staging are a follow-up); "
        f"the f16 body (sm107/bprop_d256_f16.py) and the fp8 body (sm107/bprop_d256_fp8.py) serve THD"
    )
# The dS scale-factor policy is a FAMILY constant (numerics-changing, never a knob).  This body serves P-b (the default: exact 1x32
# block-scaled e4m3 dS both ways -- two payload rings, two E8M0 atoms per stage, the along-kv warp redux) and P-c (the fp32 dS
# rounded to bf16, the oracle twin) and REFUSES P-a (the 32x32 tile scale: optional validation only, never a target) at load
# time -- a typed refusal beats an arm that traces dead code reading tensors the ABI does not carry.
if CFG.DS_SF_POLICY not in (DS_SF_P_B, DS_SF_P_C):
    raise NotImplementedError(
        f"{__name__}: serves DS_SF_POLICY = P-c ({DS_SF_P_C}, bf16 dS) and P-b ({DS_SF_P_B}, block-scaled e4m3 dS both ways); the P-a tile-scale "
        f"policy ({DS_SF_P_A}) is not built; got {CFG.DS_SF_POLICY}"
    )
_B = buffer_elems(CFG)
LAYOUT = tmem_layout(CFG)


def _needs_desc_v1(cfg) -> bool:
    """True when any tcgen05 descriptor ROOT of the SMEM table (``config_sm107.desc_roots``) starts at or past the 14-bit
    version-0 window (256 KiB).  This body's last root is sdOT_SF[2] at 221 KiB (the six SF slabs are declared BEFORE sdS for
    exactly this reason; the P ring is in TMEM), so this is False -- derived, not assumed."""
    return desc_version(cfg) == 1


# tcgen05 SMEM-descriptor version for EVERY SmemTile in this module: ONE decision point wired into every construction
# below, never a per-tile literal (rules/mma-tma-matrix.md S6; the d512 MXFP8 sibling shipped NaN on 100 % of cells from
# one SmemTile declared under a comment claiming the version it did not pass).
DESC_VERSION: int = 1 if _needs_desc_v1(CFG) else 0

# Retry form of the per-q-iteration RING waits (q / dO / dO_dv _full/_empty, s_acc_full/_empty, dp_full/_empty, p_ready,
# p_sf_consumed, stats_*, ds_smem_*): every such site is spelled ``.wait(..., spin=SPIN_RING_WAITS)``.  Whole-tile waits (k/v, dv_*,
# tmem_dealloc, the scheduler payload) and the end-of-kernel drains keep the default sleeping form.  The sign of the
# hint-less spin is a MEASURED per-kernel fact (rules/frost-tile-dsl.md S8b); False until this body has its own A/B/A.
SPIN_RING_WAITS: bool = False

# S issue order of the MMA warp's q loop -- a COMPILE-TIME property of the MASK ARM, never a knob (the two orders compute the same
# MMAs on the same operands: bitwise identical, only the order of issue moves).  True = the fp8 body's LOOKAHEAD: Q.K[i+1] issued
# between dO.V[i] and P.dO[i].  False = Q.K[i] issued at the TOP of iteration i, right after the mb_dp_empty wait (dsoftmax[i-1]
# released dP[i-1]), so S[i] is never computed while dsoftmax[i-1] runs.  MEASURED on Rubin (cc 10.7, 212 SMs, SM clock 2376 MHz),
# B=1 H=128/128, the block-scaled dS chain, main kernel / whole row, speed-up = base_ms / new_ms - 1.  On THIS body at S=8K the
# top-of-iteration order is FASTER on the DENSE row (+1.7..1.8 % / +0.9 %; control pairs within 0.11 %) and the CAUSAL row is unchanged
# (its arm keeps the lookahead: the cubin is byte-identical).  The sweep that decided the rule ran on the predecessor body (before the
# per-iteration K / V scale-factor copies): dense +2.6..3.0 % / +2.0 % @4K, +4.0 % / +2.2 % @8K, +4.8 % / +3.3 % @16K with the
# lookahead removed, causal -0.9 % / -0.5 % @8K and -0.9 % / -0.3 % @16K (the short q loops near the diagonal expose the S latency the
# lookahead hides); ncu on that dense cell: identical instruction count, short_scoreboard stalls -23 %, issue-active 55.5 -> 58.0 % --
# the lookahead's 4-k-step S MMA sat in the tensor pipe between dO.V[i] and P.dO[i], competing with the softmax warps' TMEM / SMEM
# loads and queueing P.dO[i].  So the dense arm (no mask at all) takes the top-of-iteration order and every MASKED arm -- causal /
# SWA / kv-padded, and the q-pad band alone (unmeasured, kept with its class) -- keeps the lookahead; derived from the SAME predicate
# that folds the mask IR (_mask_p_chunk), never a literal.
S_LOOKAHEAD: bool = bool(CFG.MASK_FLAGS != MASK_NONE or CFG.MASK_Q_PAD)


# --- dtype dispatch (the config validated the codes; these are the DSL types they name) ---------------------------------
STORAGE_DTYPE = cutlass.Float8E4M3FN  # CFG.DTYPE_QKV == DTYPE_E4M3 (the MXFP8 backward is E4M3-only)
# Every MMA is a block-scale one: kind MXF8F6F4, 32-element blocks (the F8_128x4 SF tensors' block, CFG.SF_BLOCK).
MMA_KIND = nvvm.MMABlockScaleKind.MXF8F6F4
SCALE_VEC_SIZE = nvvm.Tcgen05MMAScaleVecSize.BLOCK32
# dS workspace storage: BF16 under P-c (the fp32 dS rounded once; the bf16 stage-3 renderings read it unchanged), E4M3 under P-b
# (two block-scaled payloads; the block-scale stage-3 GEMM arm).  The config validated the code against the policy; every dS-ring
# constant below is BPE_DS-driven.
DS_STORAGE_DTYPE = {DTYPE_BF16: cutlass.BFloat16, DTYPE_E4M3: cutlass.Float8E4M3FN}[CFG.DTYPE_DS]
# dV storage: the graph's half-precision gradient (TRUE units, no quantization, no amax).
_OUT_DTYPES = {DTYPE_BF16: cutlass.BFloat16, DTYPE_FP16: cutlass.Float16}
OUT_STORAGE_DTYPE = _OUT_DTYPES[CFG.DTYPE_O]

# Rubin's MXFP8 block-scale MMA runs K=64 per instruction and every idesc below passes k_dim=CFG.IDESC_K_DIM (validated == 1 with
# TILE_K_HW == 64 by the config; two 32-element SF blocks per k-step).  The pairing is arch-OPPOSITE of Blackwell and a mismatch
# scrambles accumulator ROWS silently, so the tripwire is repeated here where the idescs are built (rules/mma-tma-matrix.md S1).
if CFG.TILE_K_HW_BMM1 != 64 or CFG.TILE_K_HW_BMM2 != 64 or CFG.IDESC_K_DIM != 1 or CFG.SF_BLOCKS_PER_STEP != 2:
    raise ValueError(
        f"{__name__}: Rubin MXFP8 needs TILE_K_HW=64 with idesc k_dim=1 (2 SF blocks per k-step); got {CFG.TILE_K_HW_BMM1}/{CFG.TILE_K_HW_BMM2}, "
        f"k_dim={CFG.IDESC_K_DIM}, SF_BLOCKS_PER_STEP={CFG.SF_BLOCKS_PER_STEP}"
    )

# --- per-CTA buffer geometry (config_sm107.buffer_elems; never re-derived here) ------------------------------------------
_M_PER_CTA = _B._M_PER_CTA  # 64 q rows per CTA of the Q / dO N-split
qBufferElems = _B.qBufferElems  # 64 x 256
dOBufferElems = _B.dOBufferElems  # 64 x 256 (== TILE_N x TILE_O / CTA_MMA for the dV view)
kBufferElems = _B.kBufferElems  # 128 x 256
vBufferElems = _B.vBufferElems  # 128 x 256
dSBufferElems = _B.dSBufferElems  # 128 kv x 128 q, elements of DS_STORAGE_DTYPE
dVBufferElems = _B.dVBufferElems  # 128 kv x 256 d_v, elements of OUT_STORAGE_DTYPE
qTmaTransactionBytes = _B.qTmaTransactionBytes  # both peers' bytes land on the leader's mbar (x CTA_MMA)
dOTmaTransactionBytes = _B.dOTmaTransactionBytes  # also the dO_dv view: TILE_N x (TILE_O / CTA_MMA) x BPE x CTA_MMA
kTmaTransactionBytes = _B.kTmaTransactionBytes
vTmaTransactionBytes = _B.vTmaTransactionBytes
TMA_QK_ITERS = _B.TMA_QK_ITERS  # 2 x 128-elem subtiles per 256-B fp8 row
TMA_VO_ITERS = _B.TMA_VO_ITERS
TMA_QK_GRANU_ELEMS = _B.TMA_QK_GRANU_ELEMS
TMA_VO_GRANU_ELEMS = _B.TMA_VO_GRANU_ELEMS
TMA_VO_SG1_ITERS = _B.TMA_VO_SG1_ITERS  # the dV view: 128 d_v per CTA = 1 subtile
TMA_VO_SG1_GRANU_ELEMS = _B.TMA_VO_SG1_GRANU_ELEMS
DV_D_BLOCK = _B.DV_D_BLOCK  # d_v elems per 128-B store subtile: 128 at e4m3 out, 64 at bf16 / fp16 out
TMA_DV_ITERS = _B.TMA_DV_ITERS  # 2 (e4m3) / 4 (bf16) subtiles per dV row
DV_BLOCK_SLAB = _B.DV_BLOCK_SLAB  # TILE_M x DV_D_BLOCK: one store subtile's slab
P_TMA_ITERS = _B.P_TMA_ITERS  # dS store subtiles per 128-col row: 1 at e4m3 (one 128-B row), 2 at bf16
P_D_BLOCK = _B.P_D_BLOCK  # q cols per dS subtile: 128 at e4m3 (both warpgroups' halves in one row), 64 at bf16 (one half)
P_BLOCK_ELEMS = _B.P_BLOCK_BYTES  # TILE_M x P_D_BLOCK elems per dS subtile (the pre-port misnomer is the config's)
STATS_SLOT_ELEMS = _B.STATS_SLOT_ELEMS
STATS_LSE_OFF = _B.STATS_LSE_OFF
STATS_DOT_OFF = _B.STATS_DOT_OFF
_SMX_CHUNK = _B._SMX_CHUNK  # 64 q cols per compute warpgroup
_LDTM_NUM = _B._LDTM_NUM  # tcgen05.ld.32x32b.x64 granule
_KV_BLOCK_ROWS = _B._KV_BLOCK_ROWS  # 256 kv rows per cga2 pair
# The q tiles a kv block's masked q range is rounded OUTWARD to (2 = the kv block = the stage-3 GEMMs' cluster M tile and
# their K-trim granularity `causal_gran`): stage 2 writes dS in 256-row q PAIRS so the GEMMs never read an unwritten tile.
_Q_WRITE_TILES = q_write_tiles(CFG)
CGA_SIZE = _B.CGA_SIZE  # 2
LEADING_BYTE_OFFSET_QK = _B.LEADING_BYTE_OFFSET_QK
STRIDE_BYTE_OFFSET_QK = _B.STRIDE_BYTE_OFFSET_QK
LEADING_BYTE_OFFSET_dO = _B.LEADING_BYTE_OFFSET_dO
STRIDE_BYTE_OFFSET_dO = _B.STRIDE_BYTE_OFFSET_dO
LEADING_BYTE_OFFSET_dS = _B.LEADING_BYTE_OFFSET_dS
STRIDE_BYTE_OFFSET_dS = _B.STRIDE_BYTE_OFFSET_dS
LEADING_BYTE_OFFSET_dO_SG1 = _B.LEADING_BYTE_OFFSET_dO_SG1  # BT=true B operand: K x swz (B_PC_COLS // 8 > 8)
STRIDE_BYTE_OFFSET_dO_SG1 = _B.STRIDE_BYTE_OFFSET_dO_SG1
SMEM_LAYOUT_Q = _B.SMEM_LAYOUT_Q
SMEM_LAYOUT_dO = _B.SMEM_LAYOUT_dO
SMEM_LAYOUT_K = _B.SMEM_LAYOUT_K
SMEM_LAYOUT_V = _B.SMEM_LAYOUT_V
SMEM_LAYOUT_dS = _B.SMEM_LAYOUT_dS
SMEM_LAYOUT_dV = _B.SMEM_LAYOUT_dV

# --- the P ring + the SF alias (config_sm107.tmem_layout(FAMILY_MXFP8) == the fp8 body's map) ---------------------------------
# P_COLS = TILE_N * BPE / 4: one e4m3 P[kv, q] tile per slot, 2 slots at [P_OFF, TOTAL_COLS).  The scale-factor atoms of iteration n
# alias the slot the softmax is NOT writing this iteration (slot p ^ 1, p = the mb_p_ready ring index); each band must fit ONE slot.
# The config validates all of it; the tripwire is repeated here because an atom past its band is an SF read of P bytes (garbage dV,
# no crash).
_P_SLOT_ALIAS_BANDS = (
    (
        "BMM1",
        LAYOUT.SF_BMM1_COLS,
        (
            ("K", LAYOUT.SF_K_OFF, CFG.SF_TMEM_COLS_K),
            ("V", LAYOUT.SF_V_OFF, CFG.SF_TMEM_COLS_V),
            ("Q", LAYOUT.SF_Q_OFF, CFG.SF_TMEM_COLS_Q),
            ("dO", LAYOUT.SF_dO_OFF, CFG.SF_TMEM_COLS_dO),
        ),
    ),
    ("BMM2", LAYOUT.SF_BMM2_COLS, (("P", LAYOUT.SF_P_OFF, CFG.SF_TMEM_COLS_P), ("dOT", LAYOUT.SF_dOT_OFF, CFG.SF_TMEM_COLS_dOT))),
)
if LAYOUT.P_COLS != (CFG.TILE_N * CFG.BPE) // 4 or CFG.STAGES_TMEM_P != 2 or LAYOUT.P_OFF + CFG.STAGES_TMEM_P * LAYOUT.P_COLS != LAYOUT.TOTAL_COLS:
    raise ValueError(
        f"{__name__}: the e4m3 P ring is 2 TMEM slots of TILE_N*BPE/4 = {(CFG.TILE_N * CFG.BPE) // 4} cols at the tail (the slot rule p / p^1 needs "
        f"exactly two); got P_OFF={LAYOUT.P_OFF}, P_COLS={LAYOUT.P_COLS}, STAGES_TMEM_P={CFG.STAGES_TMEM_P}, TOTAL_COLS={LAYOUT.TOTAL_COLS}"
    )
for _band, _width, _atoms in _P_SLOT_ALIAS_BANDS:
    _col = 0
    for _name, _off, _cols in _atoms:
        if _off != _col:
            raise ValueError(f"{__name__}: SF_{_name} must start at slot-relative column {_col} of the {_band} alias band; got {_off}")
        _col += _cols
    if _col != _width or _width > LAYOUT.P_COLS:
        raise ValueError(
            f"{__name__}: the {_band} scale-factor band ({_col} cols, SF_{_band}_COLS={_width}) must fit one P slot of {LAYOUT.P_COLS} cols -- an atom "
            f"past the slot lands in the LIVE P slot (an SF read of P bytes: garbage dV, no crash)"
        )

# --- scale-factor slabs: F8_128x4 atoms (512 B = 128 rows x 4 K-groups = 4 TMEM columns each), UTCCP'd 32x128b ------------------
# The UTCCP descriptor of an SF slab: leading 16 B / stride 128 B / layout 0 (no swizzle) -- the atom layout IS the F8_128x4 atom
# (the forward's SF_LEADING_BYTE_OFFSET / SF_STRIDE_BYTE_OFFSET / SMEM_LAYOUT_SF, sm107/prefill_d256_mxfp8.py).
SF_LEADING_BYTE_OFFSET = 16
SF_STRIDE_BYTE_OFFSET = 128
SMEM_LAYOUT_SF = 0
_SF_ATOM_DESC_STEP = SF_ATOM_BYTES // 16  # `desc + n` advances n x 16 B: one atom per step (the mma_ss convention)
# Atoms per slab, DERIVED from the config's slab bytes and pinned against its TMEM column counts (4 columns per atom).
_SF_ATOMS_K = CFG.SF_SMEM_K // SF_ATOM_BYTES  # 2: d-chunks 0-3 / 4-7 of this CTA's 128 kv rows
_SF_ATOMS_V = CFG.SF_SMEM_V // SF_ATOM_BYTES  # 2
_SF_ATOMS_Q = CFG.SF_SMEM_Q // SF_ATOM_BYTES  # 2: the full N = 128 q tile
_SF_ATOMS_dO = CFG.SF_SMEM_dO // SF_ATOM_BYTES  # 2
_SF_ATOMS_P = CFG.SF_SMEM_P // SF_ATOM_BYTES  # 1: 128 kv x 128 q, the constant byte
_SF_ATOMS_dOT = CFG.SF_SMEM_dOT // SF_ATOM_BYTES  # 2: D-planes 0 / 1 (d_v 0-127 / 128-255) x the 128-q K chunk
for _name, _atoms, _cols in (
    ("K", _SF_ATOMS_K, CFG.SF_TMEM_COLS_K),
    ("V", _SF_ATOMS_V, CFG.SF_TMEM_COLS_V),
    ("Q", _SF_ATOMS_Q, CFG.SF_TMEM_COLS_Q),
    ("dO", _SF_ATOMS_dO, CFG.SF_TMEM_COLS_dO),
    ("P", _SF_ATOMS_P, CFG.SF_TMEM_COLS_P),
    ("dOT", _SF_ATOMS_dOT, CFG.SF_TMEM_COLS_dOT),
):
    if _atoms * SF_TMEM_COLS_PER_ATOM != _cols or _atoms * SF_ATOM_BYTES != getattr(CFG, f"SF_SMEM_{_name}"):
        raise ValueError(
            f"{__name__}: SF_{_name}: {_atoms} atoms x {SF_TMEM_COLS_PER_ATOM} cols != SF_TMEM_COLS_{_name}={_cols}, or the slab is not whole atoms -- "
            f"one UTCCP per atom would leave SF columns stale (wrong dV / dS, no crash)"
        )
if CFG.TILE_M != SF_ATOM_ROWS or CFG.TILE_N != SF_ATOM_ROWS:
    raise ValueError(
        f"{__name__}: the SF tile coordinate is the 128-row tile index (TILE_M == TILE_N == SF_ATOM_ROWS == {SF_ATOM_ROWS}); got {CFG.TILE_M} / {CFG.TILE_N}"
    )
# How each SF slab is TMA-loaded across the cga2 pair (both CTAs' bytes land on the LEADER's mbar, P9; every tx == CFG.*_SF_TX):
#   K / V (the A operands, M-split: each CTA its OWN 128 kv rows): the whole slab of THIS CTA's tile, self-multicast ->
#     2 CTAs x 1024 B = 2048.  Q / dO (the B operands: the FULL N = 128 q tile in both CTAs): each CTA loads ONE atom (its
#     d-chunk, sf_peer_split) with mcast 3 so both CTAs hold both atoms -> 2 CTAs x 512 B x 2 destinations = 2048.  dO_T (the BT
#     B operand, full N = 256 d_v in both CTAs): each CTA loads ITS D-plane (512 B) with mcast 3 -> 2048.
SF_ROWS_K = sf_tma_rows(CFG.SF_SMEM_K)  # 8 x 128-B rows: the whole slab per CTA
SF_ROWS_V = sf_tma_rows(CFG.SF_SMEM_V)
Q_SF_SPLIT = sf_peer_split(CFG.SF_SMEM_Q, CFG.CTA_MMA)  # (512 B, 4 rows) per peer
dO_SF_SPLIT = sf_peer_split(CFG.SF_SMEM_dO, CFG.CTA_MMA)
if _SF_ATOMS_dOT % CFG.CTA_MMA:
    raise ValueError(f"{__name__}: the dO_T SF D-plane count {_SF_ATOMS_dOT} must split across CTA_MMA={CFG.CTA_MMA} peers (one box per peer)")
_dOT_SF_PLANES_PER_PEER = _SF_ATOMS_dOT // CFG.CTA_MMA  # 1
_SF_D_GROUPS = CFG.TILE_K // CFG.SF_BLOCK  # 8 E8M0 bytes per rowwise SF row (d = 256 / 32); the rowwise SF tensors' last dim
for _name, _tx, _bytes in (
    ("K", CFG.K_SF_TX, SF_ROWS_K * 128 * CFG.CTA_MMA),
    ("V", CFG.V_SF_TX, SF_ROWS_V * 128 * CFG.CTA_MMA),
    ("Q", CFG.Q_SF_TX, Q_SF_SPLIT.bytes_per_peer * CFG.CTA_MMA * CFG.CTA_MMA),
    ("dO", CFG.dO_SF_TX, dO_SF_SPLIT.bytes_per_peer * CFG.CTA_MMA * CFG.CTA_MMA),
    ("dOT", CFG.dOT_SF_TX, _dOT_SF_PLANES_PER_PEER * SF_ATOM_BYTES * CFG.CTA_MMA * CFG.CTA_MMA),
):
    if _tx != _bytes:
        raise ValueError(
            f"{__name__}: the {_name}-SF box routing delivers {_bytes} B to the leader per phase but CFG.{_name}_SF_TX is {_tx} -- a short expect_tx "
            f"completes the phase before the scale factors land (stale SF), a long one hangs"
        )
# expect_tx per _full mbarrier = the payload's bytes + the SF bytes (BARRIER TABLE; config_sm107 pins 34816 / 67584).
qFullTxBytes = qTmaTransactionBytes + CFG.Q_SF_TX
dOFullTxBytes = dOTmaTransactionBytes + CFG.dO_SF_TX
dOTFullTxBytes = dOTmaTransactionBytes + CFG.dOT_SF_TX
kFullTxBytes = kTmaTransactionBytes + CFG.K_SF_TX
vFullTxBytes = vTmaTransactionBytes + CFG.V_SF_TX
# The softmax quantizes P in 16-value packs (the ONE shape in which the constant scale byte assembles as the cvt.u8.u32 immediate,
# tile_dsl/pointwise.py header; a lone x2 cvt with the immediate ICEs ptxas).
_P_PACK = 16
if _SMX_CHUNK % _P_PACK:
    raise ValueError(f"{__name__}: a warpgroup's {_SMX_CHUNK} P values must be whole {_P_PACK}-value packs")

# --- dS scale-factor policy P-b: exact 1x32 block-scaled e4m3 dS BOTH ways ---------------------------------
# A lane's 64 fp32 dS values (its kv row x the warpgroup's 64 q columns) are quantized twice per q iteration: (a) ds_dk per 32-q
# BLOCK of its row (IN-LANE abs_max_tree; _DS_BLOCKS_PER_WG blocks per warpgroup half, ONE e8m0_pair for both bytes) and
# (b) ds_dq per 32-kv block of each q COLUMN = the warp's 32 lanes (ONE warp_abs_max_f32 = redux.sync.max.abs.f32 per column; the
# result is warp-uniform, the bytes are per-lane data; a column pair's bytes per e8m0_pair_u word, one lone scaled cvt per element).
# ds_dk packs 16 values at a time (the fp32_to_fp8_pack_scaled shape), ds_dq 4 (fp32_to_fp8x4_scaled_pairs, two pair words per
# payload word) into the fp8 body's e4m3 dS ring geometry; the E8M0 bytes go to two staged F8_128x4 atoms per ring stage (sf_ds_dk at +0, sf_ds_dq at
# +SF_ATOM_BYTES) the TMASTG stores with the payloads.  The config validates the same couplings (_validate_cfg_mxfp8, P-b rows);
# the body re-checks exactly what its byte arithmetic assumes (the body is the truth, sdpa-invariants S6).
_IS_P_B: bool = CFG.DS_SF_POLICY == DS_SF_P_B
_DS_PACK = 16
_DS_BLOCKS_PER_WG = (_SMX_CHUNK // CFG.SF_BLOCK) if _IS_P_B else 0  # 2: the two in-lane q-blocks of a warpgroup half
_DS_SF_STAGE_BYTES = _B.dSSfStagingBytes  # DS_SF_ATOMS x 512 B per dS ring stage (0 under P-c)
_DS_SF_RING_BYTES = CFG.XFER_STAGES * _DS_SF_STAGE_BYTES
_DS_KV_RING_ELEMS = (CFG.XFER_STAGES * dSBufferElems) if _IS_P_B else 0  # the ds_dq payload ring, sdS's twin
_DS_SF_ATOM_DK_OFF = 0  # sf_ds_dk atom: r = kv row (tid_in_wg), c4 = q-block (2 * wg + blk)
_DS_SF_ATOM_DQ_OFF = SF_ATOM_BYTES  # sf_ds_dq atom: r = q column (q_half_off + col), c4 = kv-block (= the warp within its wg)
if _IS_P_B and (
    _DS_BLOCKS_PER_WG != 2
    or CFG.SF_BLOCK % _DS_PACK
    or _SMX_CHUNK % _DS_PACK
    or _DS_PACK % 4
    or (_SMX_CHUNK // 2) != 1 << 5
    or P_D_BLOCK != CFG.TILE_N
    or CFG.SOFTMAX_WG_WARPS * 32 != CFG.TILE_M
    or _DS_SF_STAGE_BYTES != 2 * SF_ATOM_BYTES
    or CFG.TILE_N // CFG.SF_BLOCK != SF_ATOM_COLS
    or CFG.TILE_M // CFG.SF_BLOCK != SF_ATOM_COLS
):
    raise ValueError(
        f"{__name__}: the P-b byte arithmetic assumes two {CFG.SF_BLOCK}-blocks per warpgroup half of {_SMX_CHUNK} columns (one e8m0_pair), whole "
        f"{_DS_PACK}-packs of whole 4-column words (32 column-pair scale words = the 5-bit lane select of the atom bytes), both halves in ONE 128-B ring row (P_D_BLOCK={P_D_BLOCK} == TILE_N={CFG.TILE_N}), a warp = one 32-kv block of the CTA's "
        f"TILE_M={CFG.TILE_M} rows, {SF_ATOM_COLS} q-blocks and {SF_ATOM_COLS} kv-blocks per 128-row tile (the atoms' columns) and two staged "
        f"atoms per stage (got {_DS_SF_STAGE_BYTES} B) -- a mismatch lands an E8M0 byte in another block's slot (wrong dK / dQ, no crash)"
    )

# Lane-written staging buffers (dS ring, dV epilogue) are laid out as 128-B-row subtiles under Swizzle(3, 4, 3):
# MBase + SShift = 4 + 3 = 7 = log2(128 B), so 32 lanes writing one row each hit 32 different bank groups (job 2), and it
# is the s128b pattern the TMA-store descriptors of both buffers decode (job 1).  Both jobs by ONE swizzle.
STAGING_SMEM_SWIZZLE = cutlass.Swizzle(3, 4, 3)
_DV_EPI_CHUNK = 64  # d_v cols per epilogue TMEM load / store (register cap); each wg owns TILE_O / SOFTMAX_WARPGROUPS
_DV_CHUNKS_PER_WG = (CFG.TILE_O // CFG.SOFTMAX_WARPGROUPS) // _DV_EPI_CHUNK  # 2
if (CFG.TILE_O // CFG.SOFTMAX_WARPGROUPS) % _DV_EPI_CHUNK or DV_D_BLOCK % _DV_EPI_CHUNK:
    raise ValueError(
        f"{__name__}: the dV epilogue walks {_DV_EPI_CHUNK}-col chunks; TILE_O/WGS={CFG.TILE_O // CFG.SOFTMAX_WARPGROUPS}, DV_D_BLOCK={DV_D_BLOCK}"
    )
if P_D_BLOCK % _SMX_CHUNK:
    raise ValueError(
        f"{__name__}: a dS store subtile ({P_D_BLOCK} q cols) must be whole warpgroup q halves ({_SMX_CHUNK}); each wg stores its half at col_in_blk = q_half % P_D_BLOCK"
    )

# log2(e): the softmax uses exp2, so P = exp2(S_acc * (attn_scale * log2e) - lse * log2e) in TRUE units (S_acc is dequantized
# in-MMA by the block scale factors; NO constant folded into the shift -- the P scale 2^8 is applied by the scaled cvt after
# the exp2).  attn_scale * log2e rides in the `attn_scale_log2e` scalar; the lse fold happens IN-KERNEL in the scheduler-stats
# warp (the host passes natural-log lse).
_LOG2E = 1.4426950408889634
# CLC response payload the scheduler ring's expect_tx arms (tile_dsl/scheduler.py uses the same 16).
_CLC_RESPONSE_BYTES = 16

CTA_GROUP_KIND = nvvm.CTAGroup.CTA_2  # CTA_MMA == 2 (validated by the config)

# --- named arrival-count constants (P3) --------------------------------------------------------------------------------------
ONE_LANE = CFG.ONE_LANE  # 1
ONE_WARP = CFG.ONE_WARP  # 32
SOFTMAX_LANES = CFG.SOFTMAX_LANES  # 256 = every compute lane of ONE CTA (both warpgroups)
SOFT_X_CTA_MMA = CFG.SOFT_X_CTA_MMA  # 512 = every compute lane of BOTH CTAs (the LEADER-scope counts)
MMA_COMMIT_ARRIVES = CFG.MMA_COMMIT_ARRIVES  # 1 per target CTA per multicast commit (one elected lane)
READ_TILE_ARRIVERS_TOT = CFG.READ_TILE_ARRIVERS_TOT  # 21, derivation in config_sm107.read_tile_arrivers_tot
_COMPUTE_WARPS = CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS  # 8
_NAMED_BAR_TMEM_ID = 1  # MMA warp <-> compute warps: TMEM base publish
_NAMED_BAR_TMEM_THREADS = 32 * (_COMPUTE_WARPS + 1)  # 288


# === P-b helpers: pick each lane's element out of warp-uniform candidates without a runtime-indexed register array =========


def _sel_i32(pred, a, b):
    """``a if pred else b`` on traced ``Int32`` (one ``SEL``); ``pred`` a traced comparison."""
    return cutlass.Int32(arith.select(pred.ir_value(), a.ir_value(), b.ir_value()))


def _select_by_lane(vals, lane_bits):
    """``vals[lane]`` from ``2 ** len(lane_bits)`` WARP-UNIFORM ``Int32`` candidates: a select tree on a lane's index bits
    (``lane_bits[k]`` = bit k of its lane id, LSB first), ``len(vals) - 1`` ``SEL``s per lane.  The alternative -- a Python list
    indexed by a traced lane id -- is a local-memory array (``STL`` / ``LDL`` on every access, rules/frost-kernels.md S2).  Trace-time
    helper (plain Python over traced values), like ``abs_max_tree``."""
    level = list(vals)
    if len(level) != 1 << len(lane_bits):
        raise ValueError(f"_select_by_lane: {len(level)} candidates need exactly log2 = {len(lane_bits)} lane bits")
    for bit in lane_bits:
        level = [_sel_i32(bit, level[2 * i + 1], level[2 * i]) for i in range(len(level) // 2)]
    return level[0]


# === Bars -- the mbarrier inventory (see the BARRIER TABLE in the module docstring) ==================================
class Bars(NamedTuple):
    mb_q_full: object
    mb_q_empty: object
    mb_do_full: object
    mb_do_empty: object
    mb_dodv_full: object
    mb_dodv_empty: object
    mb_k_full: object
    mb_k_empty: object
    mb_v_full: object
    mb_v_empty: object
    mb_s_acc_full: object
    mb_s_acc_empty: object
    mb_dp_full: object
    mb_dp_empty: object
    mb_p_ready: object
    mb_p_sf_consumed: object
    mb_stats_full: object
    mb_stats_empty: object
    mb_ds_smem_full: object
    mb_ds_smem_empty: object
    mb_dv_ready: object
    mb_dv_acc_empty: object
    mb_dv_stg_full: object
    mb_dv_stg_empty: object
    mb_tmem_dealloc: object


def _make_bars(CFG) -> Bars:
    """Allocate the SMEM mbarriers with the init counts of the BARRIER TABLE.  The LEADER-scope bars are initialised to
    the cluster-wide count on BOTH CTAs (the follower never waits its copy; an unconditional constant beats a runtime
    select, P8).  Depths come from the config (``config_sm107.mbar_stage_counts`` lists the same inventory)."""

    def _alloc(n):
        return cutlass.Array(cutlass.Int64, n, alignment=16, space=cutlass.AddressSpace.smem)

    return Bars(
        mb_q_full=MBarrier(_alloc(CFG.STAGES_Q), stages=CFG.STAGES_Q, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_q_empty=MBarrier(_alloc(CFG.STAGES_Q), stages=CFG.STAGES_Q, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        mb_do_full=MBarrier(_alloc(CFG.STAGES_dO), stages=CFG.STAGES_dO, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_do_empty=MBarrier(_alloc(CFG.STAGES_dO), stages=CFG.STAGES_dO, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        mb_dodv_full=MBarrier(_alloc(CFG.STAGES_dO_DV), stages=CFG.STAGES_dO_DV, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_dodv_empty=MBarrier(_alloc(CFG.STAGES_dO_DV), stages=CFG.STAGES_dO_DV, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        mb_k_full=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_k_empty=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        mb_v_full=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_v_empty=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        # S / dP accumulators are single-buffered (STAGES_TMEM_S == 1): the cross-iteration overlap is the MMA order.
        mb_s_acc_full=MBarrier(_alloc(CFG.STAGES_TMEM_S), stages=CFG.STAGES_TMEM_S, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        mb_s_acc_empty=MBarrier(_alloc(CFG.STAGES_TMEM_S), stages=CFG.STAGES_TMEM_S, init_count=SOFT_X_CTA_MMA, producer=Producer.LEADER, scope=Scope.LEADER),
        mb_dp_full=MBarrier(_alloc(CFG.STAGES_TMEM_S), stages=CFG.STAGES_TMEM_S, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        mb_dp_empty=MBarrier(_alloc(CFG.STAGES_TMEM_S), stages=CFG.STAGES_TMEM_S, init_count=SOFT_X_CTA_MMA, producer=Producer.LEADER, scope=Scope.LEADER),
        # e4m3 P ring (2 TMEM stages, the fp8 body's): softmax -> MMA "P[slot] stored".  Relaxed LEADER arrive after tcgen05_wait(STORE)
        # (TMEM data needs no release; same 512 lanes, same init as the fp8 body).  No p_empty: slot p(n) is rewritten at iteration
        # n + 2 only after mb_s_acc_full[n + 2], whose commit orders after P.dO[n] read the slot (in-order tcgen05 stream), AND after
        # mb_p_sf_consumed[n + 1] below.
        mb_p_ready=MBarrier(_alloc(CFG.STAGES_TMEM_P), stages=CFG.STAGES_TMEM_P, init_count=SOFT_X_CTA_MMA, producer=Producer.LEADER, scope=Scope.LEADER),
        # THE ONE NEW RING (BARRIER TABLE row mb_p_sf_consumed): the scale-factor atoms of iteration n live in P slot s(n) = p(n + 1), the
        # slot the softmax writes NEXT.  The in-order tcgen05 stream orders the MMA warp's own copies and MMAs; it cannot order the
        # softmax lanes' tcgen05.st, so the MMA warp commits this mbar (multicast, one elected lane = 1 arrive per CTA) right after
        # P.dO[n] and every softmax lane waits it before its store of P[n + 1].  1 stage; the consumer state is pre-armed (start(phase=1)).
        mb_p_sf_consumed=MBarrier(_alloc(1), stages=1, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        # lse / delta SMEM prefetch ring: scheduler warp (32 lanes) -> compute lanes (256).
        mb_stats_full=MBarrier(_alloc(CFG.STATS_STAGES), stages=CFG.STATS_STAGES, init_count=ONE_WARP, producer=Producer.THREAD),
        mb_stats_empty=MBarrier(_alloc(CFG.STATS_STAGES), stages=CFG.STATS_STAGES, init_count=SOFTMAX_LANES, producer=Producer.THREAD),
        # dS SMEM ring: compute lanes store_swizzled -> TMASTG TMA store (one elected lane releases the slot).
        mb_ds_smem_full=MBarrier(_alloc(CFG.XFER_STAGES), stages=CFG.XFER_STAGES, init_count=SOFTMAX_LANES, producer=Producer.THREAD),
        mb_ds_smem_empty=MBarrier(_alloc(CFG.XFER_STAGES), stages=CFG.XFER_STAGES, init_count=ONE_LANE, producer=Producer.THREAD),
        # dV epilogue (once per kv tile): MMA -> compute warps -> TMASTG -> (F1) TMALDG.
        mb_dv_ready=MBarrier(_alloc(1), stages=1, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        mb_dv_acc_empty=MBarrier(_alloc(1), stages=1, init_count=SOFT_X_CTA_MMA, producer=Producer.LEADER, scope=Scope.LEADER),
        mb_dv_stg_full=MBarrier(_alloc(1), stages=1, init_count=SOFTMAX_LANES, producer=Producer.THREAD),
        mb_dv_stg_empty=MBarrier(_alloc(1), stages=1, init_count=ONE_LANE, producer=Producer.THREAD),
        # 256 local bare arrives + 256 relaxed cluster arrives from the partner's compute lanes, on EACH CTA.
        mb_tmem_dealloc=MBarrier(_alloc(1), stages=1, init_count=SOFT_X_CTA_MMA, producer=Producer.THREAD),
    )


# === Mask plumbing (the backward is the TRANSPOSE of the forward) =====================================================
# lane = kv row, inner loop = q tile.  A fixed kv block [kvb, kvb + 256) bounds WHICH q tiles attend (`_q_loop_bounds`
# skips the rest) and `_mask_p_chunk` zeroes P on the masked (kv, q) cells of the boundary tiles.  P = 0 makes dV / dS /
# dK / dQ inherit the mask.  Every MASK_* arm is a const_expr: the dense specialization emits no mask IR at all.


def _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv_real):
    """Real kv length of this batch entry: ``seq_kv_lens[batch_idx]`` under the PADDED arm, else the uniform scalar.  No
    batch chunking on this body (the whole batch is in-grid; ``head_base`` walks the head chunks), so the grid batch IS
    the full-tensor batch."""
    if cutlass.const_expr(CFG.MASK_FLAGS & MASK_PADDED):
        arr = cutlass.make_array_view(seq_kv_lens_tensor)
        return cutlass.Int32(arr[batch_idx])
    return seqlen_kv_real


def _causal_diag(seqlen_q_real, eff_seqlen_kv):
    """Bottom-right diagonal offset (S_kv - S_q) of this tile's batch entry, on the REAL lengths; 0 for top-left / dense."""
    if cutlass.const_expr(CFG.CAUSAL_BOTTOM_RIGHT):
        return eff_seqlen_kv - seqlen_q_real
    return cutlass.Int32(0)


def _q_loop_bounds(kv_block_base, seqlen_q, seqlen_q_real, eff_seqlen_kv):
    """``[q_lo, q_hi)``: the q tiles that attend this kv block, via ``tile_dsl.mask.compute_q_loop_bounds`` (the same band
    the forward applied: causal keeps kv <= q + diag, SWA keeps kv >= q + diag - W with the bottom-right anchor on BOTH
    edges; padding masks kv ROWS per lane and leaves the q range alone), rounded OUTWARD to ``_Q_WRITE_TILES`` (the 256-row
    pair the stage-3 GEMMs trim at, so every dS tile a GEMM can reach was written here), then the empty-kv-block clamp: a block no q attends
    (top-left causal with S_kv > S_q) would run zero q iterations, and an N = 0 tile hangs the UNCONDITIONAL MMA prologue on
    mb_q_full (the TMA loop made no load) -- a runtime `if` cannot skip it without breaking every ring's per-tile balance.
    So q_lo is clamped in range and N is FORCED >= 1: the single forced tile is fully masked (P = 0 -> dV += 0, dS = 0:
    correct zeros, dV_acc overwritten via accumulate=False) and every per-iteration ring stays balanced (P14).  A no-op for
    non-empty blocks.  Uniform across the pair (kv_block_base is the cluster's kv base).  Every warp derives the SAME bounds
    from the same (kv_block_base, lengths) -- the P14 balance of six loop bodies depends on it.

    ``seqlen_q_real`` / ``eff_seqlen_kv`` are the REAL lengths the bottom-right diagonal is anchored on (the uniform q length
    and the batch entry's kv length); the padded extent ``seqlen_q`` is the envelope the q loop walks (its pad tiles are
    +inf-LSE / q-band dead)."""
    n_q_tiles = seqlen_q // cutlass.Int32(CFG.TILE_N)
    if cutlass.const_expr(CFG.MASK_FLAGS == MASK_NONE):
        return cutlass.Int32(0), n_q_tiles
    b = compute_q_loop_bounds(
        kv_block_base,
        seqlen_q_real,
        eff_seqlen_kv,
        n_q_tiles,
        CFG.SWA_WINDOW,
        CFG.MASK_FLAGS,
        CFG.TILE_N,
        _KV_BLOCK_ROWS,
        bottom_right=bool(CFG.CAUSAL_BOTTOM_RIGHT),
        window_right=0,
    )
    # Round the band OUTWARD to `_Q_WRITE_TILES` (a 256-row q pair = the stage-3 GEMMs' cluster M tile and K-trim
    # granularity): the extra tile is fully masked (P = 0 -> dS = 0, dV += 0), and a dQ M tile that spans the pair then reads
    # only kv blocks that wrote BOTH its q tiles -- what lets the adapter drop the workspace zero-fill
    # (`config_sm107.q_write_tiles`; `bprop_matmul_blackwell._causal_k_range`).  Plain top-left causal pays nothing (the
    # diagonal's tile is even); the dense arm returned above.
    pair = cutlass.Int32(_Q_WRITE_TILES)
    lo = (b.lo // pair) * pair
    hi = cute.math.min(((b.hi + pair - cutlass.Int32(1)) // pair) * pair, n_q_tiles)
    q_lo = cute.math.min(lo, n_q_tiles - cutlass.Int32(1))
    q_hi = cute.math.max(hi, q_lo + cutlass.Int32(1))
    return q_lo, q_hi


def _mask_p_chunk(reg_P, kv_abs, q_col_base, eff_seqlen_kv, causal_diag, N: int, seqlen_q_real=None):
    """Zero P on masked (kv = lane, q = col) cells of an ``N``-wide q chunk starting at absolute q ``q_col_base``.

    The TRANSPOSE of ``tile_dsl.mask.apply_mask_chunk`` (row = kv, col = q), with ZERO as the masked value:
      causal : kv_abs > q + diag          (key past the query; diag = S_kv - S_q under bottom-right, else 0)
      SWA    : kv_abs < q + diag - W      (key left of the window; the same bottom-right anchor as the forward)
      padded : kv_abs >= seq_kv_len       (per-lane pad row -> the whole row is masked)
    The bit-word form (rules/frost-tile-dsl.md S10d): the terms map onto ONE q band [lo, hi) per lane -- causal lo =
    kv_abs - diag, SWA hi = kv_abs - diag + W + 1, padded hi = q_col_base (all masked) -- and the library's
    ``band_mask_words`` / ``apply_mask_words`` do the rest (one keep-word per 32 q columns, R2P + 1 FSEL per cell).  The
    masked set is the per-cell compare's, so dS / dV are bitwise what the pre-port body produced.  MASK_NONE without the q
    band returns reg_P unchanged (no IR).
    MXFP8: under ``CFG.MASK_Q_PAD`` (the adapter sets it iff S_q % 128 != 0) the band also takes
    ``hi = min(hi, seqlen_q_real)`` -- P is a SELECT-zero on every q pad column regardless of S.  The Q-side SF pad rows (sf_q /
    sf_dO / sf_dO_T past S_q_real) are producer-defined bytes: a 0xFF there is an E8M0 NaN, S[kv, q_pad] = 0 x NaN = NaN, and the
    +inf-LSE trick (P = exp2(NaN - inf) = NaN) no longer zeroes P -- so BMM2 would fold a NaN P into dV on EVERY kv row.  The band is
    ONE half of the fix (P = 0); the other half is the host's zero-filled SF pad staging (finite dequantized dO / dO_T, and a finite
    dP so the dS multiply is 0, not NaN).  A SEPARATE const_expr arm, not a MASK_FLAGS bit: a MASK_FLAGS-keyed dispatch folds it out.
    TODO: hoist this transposed arm into tile_dsl.mask as the kv-major twin of apply_mask_chunk."""
    if cutlass.const_expr(CFG.MASK_FLAGS == MASK_NONE and not CFG.MASK_Q_PAD):
        return reg_P
    lo = None
    hi = None
    if cutlass.const_expr(CFG.MASK_FLAGS & MASK_CAUSAL):
        lo = kv_abs - causal_diag
    if cutlass.const_expr(CFG.MASK_FLAGS & MASK_SWA):
        hi = kv_abs - causal_diag + cutlass.Int32(CFG.SWA_WINDOW + 1)
    if cutlass.const_expr(CFG.MASK_FLAGS & MASK_PADDED):
        row_dead = kv_abs >= eff_seqlen_kv
        hi_pad = cutlass.Int32(arith.select(row_dead.ir_value(), q_col_base.ir_value(), (q_col_base + cutlass.Int32(N)).ir_value()))
        hi = hi_pad if hi is None else cute.math.min(hi, hi_pad)
    if cutlass.const_expr(CFG.MASK_Q_PAD):
        hi = seqlen_q_real if hi is None else cute.math.min(hi, seqlen_q_real)
    words = band_mask_words(lo, hi, q_col_base, N)
    return apply_mask_words(reg_P, words, mask_value=0.0, n_cols=N)


# === Tile decode (one cga2 cluster per (kv block, head, batch) tile) ===================================================


@cute.jit
def _decode_linear_bprop(linear, n_qh_grid, n_batch):
    """Flat 1-D (LPT / LPT_L2) decode: linear cluster id -> (kv_super, head, batch).  kv_super is the OUTER axis so the
    heaviest causal kv blocks (small kv_super, attended by the most queries) land in the first SM wave.  ``n_qh_grid`` is
    the GRID head extent (== qh_chunk under head-chunking)."""
    hb = n_qh_grid * n_batch
    kv_super = linear // hb
    within = linear % hb
    head = within % n_qh_grid
    batch = within // n_qh_grid
    return kv_super, head, batch


@cute.jit
def _boot_tile(sched):
    """The FIRST tile (kv_super, head, batch) from the launch blockIdx: natural 3-D grid (bidx // CGA_M, bidy, bidz); LPT /
    LPT_L2 flat grid: bidx is the linear cluster base and bidy_init / bidz_init are REPURPOSED to carry (n_qh_grid,
    n_batch) -- dead under an (N, 1, 1) grid (see _kernel)."""
    linear = sched.bidx_init // cutlass.Int32(CFG.CGA_M)
    if cutlass.const_expr(CFG.SCHEDULER_POLICY == SCHED_NATURAL):
        kv = linear
        h = sched.bidy_init
        b = sched.bidz_init
    else:
        kv, h, b = _decode_linear_bprop(linear, sched.bidy_init, sched.bidz_init)
    return cute.arch.make_warp_uniform(kv), cute.arch.make_warp_uniform(h), cute.arch.make_warp_uniform(b)


@cute.jit
def _decode_tile_payload(sched, sched_idx):
    """Decode (kv_super, head, batch, is_valid) from the try_cancel response slot (ONE 128-bit load, ``read_clc_payload``).
    Natural grid: word 0 = kv_super * CGA_M, word 1 = head (low 16) | batch (high 16).  Flat grid: word 0 = linear * CGA_M,
    decoded with the (n_qh_grid, n_batch) stashed in bidy_init / bidz_init."""
    t0, t1, valid = read_clc_payload(sched, sched_idx * cutlass.Int32(8))
    t0 = cute.arch.make_warp_uniform(t0)
    t1 = cute.arch.make_warp_uniform(t1)
    valid = cute.arch.make_warp_uniform(valid)
    linear = t0 // cutlass.Int32(CFG.CGA_M)
    if cutlass.const_expr(CFG.SCHEDULER_POLICY == SCHED_NATURAL):
        kv = linear
        h = t1 & cutlass.Int32(0xFFFF)
        b = (t1 >> cutlass.Int32(16)) & cutlass.Int32(0xFFFF)
    else:
        kv, h, b = _decode_linear_bprop(linear, sched.bidy_init, sched.bidz_init)
    return kv, h, b, valid


# === Kernel entry ========================================================================================================


@cute.kernel
def _kernel(
    # TMA descriptors -- loads (payloads)
    tma_q_desc: cutlass.GridConstant[tmap.TensorMap],  # Q    (BMM1 S B, q-split box; rowwise-quantized)
    tma_do_desc: cutlass.GridConstant[tmap.TensorMap],  # dO   (BMM1 dP B, q-split box; the ROWWISE quantization)
    tma_do_T_desc: cutlass.GridConstant[tmap.TensorMap],  # dO_T (BMM2 dV B, d_v-split box, BT; the COLUMNWISE quantization of dO)
    tma_k_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_v_desc: cutlass.GridConstant[tmap.TensorMap],
    # TMA descriptors -- loads (scale factors: F8_128x4 atoms, sdpa/kernels/_mxfp8_sf.py)
    tma_q_sf_desc: cutlass.GridConstant[tmap.TensorMap],  # rowwise; box = one atom (each CTA its d-chunk, mcast 3)
    tma_k_sf_desc: cutlass.GridConstant[tmap.TensorMap],  # rowwise; box = the whole slab of this CTA's own kv tile
    tma_v_sf_desc: cutlass.GridConstant[tmap.TensorMap],  # rowwise (V contracts over d in BMM1 dP); as K
    tma_do_sf_desc: cutlass.GridConstant[tmap.TensorMap],  # rowwise dO; as Q
    tma_do_T_sf_desc: cutlass.GridConstant[tmap.TensorMap],  # COLUMNWISE dO_T, D-plane-major; box = one plane (this CTA's, mcast 3)
    # TMA descriptors -- stores
    tma_dv_desc: cutlass.GridConstant[tmap.TensorMap],  # dV -> [B, S_kv, H_q, d_v] OUT (bf16 / fp16, TRUE units)
    tma_ds_desc: cutlass.GridConstant[
        tmap.TensorMap
    ],  # dS payload 0 -> workspace [B, H_chunk, S_kv, S_q]: P-c bf16 dS | P-b ds_dk (e4m3, per-32-q-block scale)
    tma_ds_kv_desc: cutlass.GridConstant[tmap.TensorMap],  # P-b: ds_dq (e4m3, per-32-kv-block scale), same geometry; P-c: tma_ds_desc again, never read
    tma_sf_ds_dk_desc: cutlass.GridConstant[tmap.TensorMap],  # P-b: sf_ds_dk atoms [B, H_chunk, S_kv/128, S_q/128, 512], one atom per box; P-c: unread
    tma_sf_ds_dq_desc: cutlass.GridConstant[tmap.TensorMap],  # P-b: sf_ds_dq atoms [B, H_chunk, S_q/128, S_kv/128, 512]; P-c: unread
    # GMEM vectors
    lse_tensor: cute.Tensor,  # [B, H_q, S_q] fp32, natural log
    do_dot_tensor: cute.Tensor,  # [B, H_q, S_q] fp32 delta = rowsum(dO * O), true units, unscaled
    # Scalars (no per-tensor scales, no amax: the block scale factors dequantize inside every MMA)
    seqlen_q: cutlass.Int32,  # the PADDED S_q (whole q tiles: the loop bounds; the LSE +inf pad zeroes P past the real length)
    seqlen_kv: cutlass.Int32,  # the REAL kv length (drives the PADDED mask)
    seqlen_q_real: cutlass.Int32,  # the REAL q length: the q < seqlen_q_real band under CFG.MASK_Q_PAD (unread when folded out)
    n_batch: cutlass.Int32,
    qh_per_kh: cutlass.Int32,
    n_qh: cutlass.Int32,  # FULL Q-head count: the (batch * n_qh + head) index of the columnwise dO_T SF descriptor
    attn_scale: cutlass.Float32,
    attn_scale_log2e: cutlass.Float32,
    head_base: cutlass.Int32,  # workspace chunking: full-tensor head = grid head + head_base
    n_qh_grid: cutlass.Int32,  # grid head extent (== qh_chunk); the flat-grid decode
    seq_kv_lens_tensor: Optional[cute.Tensor],  # [B] int32 per-batch REAL kv lengths (the PADDED arm reads seq_kv_lens[b]); None otherwise
) -> None:
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx = cute.arch.thread_idx()[0]

    bidx = cute.arch.block_idx()[0]
    bidy = cute.arch.block_idx()[1]
    bidz = cute.arch.block_idx()[2]

    # --- TRUE-unit softmax scalars (delta 6): SF_K / SF_Q dequantize S and SF_V / SF_dO dequantize dP INSIDE the MMAs, so the
    # exp2 argument is S_acc * attn_scale_log2e - lse * log2e (the oracle's spelling, test/python/sdpa/mxfp8_ref.py; NO constant
    # folded into the shift -- the P scale 2^8 is applied by the scaled cvt AFTER the exp2, never folded into the argument) and
    # dS = (dP_acc * attn_scale - delta * attn_scale) * P.  The stats warp folds lse * log2e and delta * dot_scale.
    s_scale_log2 = attn_scale_log2e
    dp_scale = attn_scale
    dot_scale = attn_scale

    # --- SharedStorage: DECLARATION ORDER == config_sm107.smem_layout (the desc-root tally describes this layout) ------
    # Q ring (3 stages): B operand of BMM1 S (q-split, 64 q x 256 d_qk per CTA).
    sQ_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_Q * qBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    # dO ring #1 (3 stages): B operand of BMM1 dP (BT=false, q-split, 64 q x 256 d_v; leading 0, 2 swizzle subtiles).
    sdO_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_dO * dOBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    # dO ring #2 (3 stages): B operand of BMM2 dV (BT=true, d_v-split, 128 q x 128 d_v; leading TILE_N x swz, 1 subtile).
    # The SAME dO GMEM data with a DIFFERENT per-CTA box + swizzle layout: it cannot share sdO_raw (the s128b XOR maps
    # cells to different bytes for the 256-B-row and the 128-B-row interpretations).
    sdOdv_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_dO_DV * dOBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    # K + V backing: sK [0, kBufferElems) | sV [kBufferElems, +vBufferElems) during the q loop; the dV epilogue staging
    # ALIASES it post-loop (K, V dead).  Byte-sized for max(K + V, dV @ BPE_O) -- equal at bf16 out, dV smaller at e4m3.
    _KV_ALIAS_ELEMS = max((kBufferElems + vBufferElems) * CFG.BPE, dVBufferElems * CFG.BPE_O)
    sExcl_raw = cutlass.Array(STORAGE_DTYPE, _KV_ALIAS_ELEMS, alignment=1024, space=cutlass.AddressSpace.smem)
    sK_raw = cutlass.Array(sExcl_raw.data_ptr(), shape=kBufferElems, dtype=STORAGE_DTYPE)
    # ELEMENT offset (subview is element-addressed; a `* BPE` here is a no-op at BPE = 1 and doubles at BPE = 2).
    sV_raw = cutlass.Array(sExcl_raw.subview(kBufferElems).data_ptr(), shape=vBufferElems, dtype=STORAGE_DTYPE)
    sdV_raw = cutlass.Array(sExcl_raw.data_ptr(), shape=dVBufferElems, dtype=OUT_STORAGE_DTYPE)
    # lse / delta prefetch ring (fp32, ~2 KiB).  K + V are live through the whole q loop, so it has its own backing.
    sStats_raw = cutlass.Array(cutlass.Float32, CFG.STATS_STAGES * STATS_SLOT_ELEMS, alignment=1024, space=cutlass.AddressSpace.smem)
    # --- MXFP8 slabs, EVERY one a tcgen05 descriptor source, declared BEFORE sdS so their roots stay under the 256 KiB
    # version-0 window (rules/mma-tma-matrix.md S6; DESC_VERSION derives from exactly this order via config_sm107.smem_layout).
    # Scale-factor slabs (F8_128x4 atoms, UTCCP sources; no swizzle: the atom layout IS the UTCCP 32x128b layout):
    # K / V once per kv tile (this CTA's own 128 kv rows x 8 d-groups), P the constant byte (filled once, below), Q / dO / dO_T
    # rings in lock-step with their payload rings (the full N tile in both CTAs).
    sK_SF_raw = cutlass.Array(cutlass.Int8, CFG.STAGES_KV * CFG.SF_SMEM_K, alignment=1024, space=cutlass.AddressSpace.smem)
    sV_SF_raw = cutlass.Array(cutlass.Int8, CFG.STAGES_KV * CFG.SF_SMEM_V, alignment=1024, space=cutlass.AddressSpace.smem)
    sP_SF_raw = cutlass.Array(cutlass.Int8, CFG.SF_SMEM_P, alignment=1024, space=cutlass.AddressSpace.smem)
    sQ_SF_raw = cutlass.Array(cutlass.Int8, CFG.STAGES_Q * CFG.SF_SMEM_Q, alignment=1024, space=cutlass.AddressSpace.smem)
    sdO_SF_raw = cutlass.Array(cutlass.Int8, CFG.STAGES_dO * CFG.SF_SMEM_dO, alignment=1024, space=cutlass.AddressSpace.smem)
    sdOT_SF_raw = cutlass.Array(cutlass.Int8, CFG.STAGES_dO_DV * CFG.SF_SMEM_dOT, alignment=1024, space=cutlass.AddressSpace.smem)
    # (The e4m3 P ring is the TMEM ring [P_OFF, TOTAL_COLS): no SMEM slab.)
    # P-b only: the dS scale-factor staging -- DS_SF_ATOMS (2) F8_128x4 atoms per dS ring stage (sf_ds_dk at +0, sf_ds_dq at
    # +512 B), byte-addressed by the compute lanes (sf_layout.sf_atom_byte), TMA-stored by the TMASTG with the slot.  No swizzle
    # (the atom layout IS what the stage-3 GEMM's UTCCP reads); declared BEFORE the payload rings = the config's slab order.  None
    # under P-c (no slab, no bytes).
    sdS_SF_raw = cutlass.Array(cutlass.Int8, _DS_SF_RING_BYTES, alignment=1024, space=cutlass.AddressSpace.smem) if cutlass.const_expr(_IS_P_B) else None
    # dS SMEM ring (DS_STORAGE_DTYPE, XFER_STAGES deep; 2 under P-c / P-b): compute lanes store_swizzled dS[kv, q] (P-b: the ds_dk
    # payload), the TMASTG TMA-stores each slot to the GMEM workspace.  LOCAL to each CTA (each CTA owns its 128 kv rows of the pair's
    # block).  No descriptor reads it (222 KiB @ P-c, 224 KiB @ P-b -- after the 2 KiB atom staging).
    sdS_raw = cutlass.Array(DS_STORAGE_DTYPE, CFG.XFER_STAGES * dSBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    # P-b only: the second e4m3 payload ring (ds_dq: dS scaled per 32-kv block, the dQ GEMM's A), sdS's geometry twin.  None under P-c.
    sdS_kv_raw = cutlass.Array(DS_STORAGE_DTYPE, _DS_KV_RING_ELEMS, alignment=1024, space=cutlass.AddressSpace.smem) if cutlass.const_expr(_IS_P_B) else None
    # 16-bit alias of the SF staging for the sf_ds_dk pair store (a lane's two adjacent bytes as one st.u16): ELEMENT-addressed, so an
    # index is a byte offset / 2 (rules/frost-tile-dsl.md S6).
    sdS_SF16_raw = cutlass.Array(sdS_SF_raw.data_ptr(), shape=_DS_SF_RING_BYTES // 2, dtype=cutlass.Int16) if cutlass.const_expr(_IS_P_B) else None

    # --- SmemTile wrappers (every one takes desc_version=DESC_VERSION) --------------------------------------------------
    sQ = SmemTile(
        base=sQ_raw,
        elems_per_stage=qBufferElems,
        stages=CFG.STAGES_Q,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_Q,
        tma_loads_per_tile=TMA_QK_ITERS,
        tma_granu_elems=TMA_QK_GRANU_ELEMS,
        tma_subtile_stride_elems=_M_PER_CTA * TMA_QK_GRANU_ELEMS,
        desc_version=DESC_VERSION,
    )
    sdO = SmemTile(
        base=sdO_raw,
        elems_per_stage=dOBufferElems,
        stages=CFG.STAGES_dO,
        leading_byte_offset=LEADING_BYTE_OFFSET_dO,
        stride_byte_offset=STRIDE_BYTE_OFFSET_dO,
        layout=SMEM_LAYOUT_dO,
        tma_loads_per_tile=TMA_VO_ITERS,
        tma_granu_elems=TMA_VO_GRANU_ELEMS,
        tma_subtile_stride_elems=_M_PER_CTA * TMA_VO_GRANU_ELEMS,
        desc_version=DESC_VERSION,
    )
    sdO_dv = SmemTile(
        base=sdOdv_raw,
        elems_per_stage=dOBufferElems,
        stages=CFG.STAGES_dO_DV,
        leading_byte_offset=LEADING_BYTE_OFFSET_dO_SG1,
        stride_byte_offset=STRIDE_BYTE_OFFSET_dO_SG1,
        layout=SMEM_LAYOUT_dO,
        tma_loads_per_tile=TMA_VO_SG1_ITERS,
        tma_granu_elems=TMA_VO_SG1_GRANU_ELEMS,
        tma_subtile_stride_elems=CFG.TILE_N * TMA_VO_SG1_GRANU_ELEMS,
        desc_version=DESC_VERSION,
    )
    sK = SmemTile(
        base=sK_raw,
        elems_per_stage=kBufferElems,
        stages=CFG.STAGES_KV,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_K,
        tma_loads_per_tile=TMA_QK_ITERS,
        tma_granu_elems=TMA_QK_GRANU_ELEMS,
        tma_subtile_stride_elems=CFG.TILE_M * TMA_QK_GRANU_ELEMS,
        desc_version=DESC_VERSION,
    )
    sV = SmemTile(
        base=sV_raw,
        elems_per_stage=vBufferElems,
        stages=CFG.STAGES_KV,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_V,
        tma_loads_per_tile=TMA_VO_ITERS,
        tma_granu_elems=TMA_VO_GRANU_ELEMS,
        tma_subtile_stride_elems=CFG.TILE_M * TMA_VO_GRANU_ELEMS,
        desc_version=DESC_VERSION,
    )
    # dS staging: TMA-store source only (no MMA reads it -> leading / stride are irrelevant, kept at the dS constants).
    # A 128-col row is P_TMA_ITERS subtiles of P_D_BLOCK cols (e4m3: 1 x 128, bf16: 2 x 64); subtile s of stage k starts at
    # k * dSBufferElems + s * P_BLOCK_ELEMS.
    sdS = SmemTile(
        base=sdS_raw,
        elems_per_stage=dSBufferElems,
        stages=CFG.XFER_STAGES,
        leading_byte_offset=LEADING_BYTE_OFFSET_dS,
        stride_byte_offset=STRIDE_BYTE_OFFSET_dS,
        layout=SMEM_LAYOUT_dS,
        tma_loads_per_tile=P_TMA_ITERS,
        tma_granu_elems=P_D_BLOCK,
        tma_subtile_stride_elems=P_BLOCK_ELEMS,
        desc_version=DESC_VERSION,
    )
    # P-b: the ds_dq payload ring, sdS's twin (same subtile walk, the ds_dq store descriptor); None under P-c.
    sdS_kv = (
        SmemTile(
            base=sdS_kv_raw,
            elems_per_stage=dSBufferElems,
            stages=CFG.XFER_STAGES,
            leading_byte_offset=LEADING_BYTE_OFFSET_dS,
            stride_byte_offset=STRIDE_BYTE_OFFSET_dS,
            layout=SMEM_LAYOUT_dS,
            tma_loads_per_tile=P_TMA_ITERS,
            tma_granu_elems=P_D_BLOCK,
            tma_subtile_stride_elems=P_BLOCK_ELEMS,
            desc_version=DESC_VERSION,
        )
        if cutlass.const_expr(_IS_P_B)
        else None
    )
    # P-b: the dS SF atom staging -- one stage = DS_SF_ATOMS x 512 B; a TMA store moves ONE atom (tma_loads_per_tile 1, the
    # 5-D byte descriptor's [128 B, 4 rows] box), the dq atom through .shifted(_DS_SF_ATOM_DQ_OFF).  No MMA reads it (the
    # leading / stride constants are the SF slabs', unused); None under P-c.
    sdS_SF = (
        SmemTile(
            base=sdS_SF_raw,
            elems_per_stage=_DS_SF_STAGE_BYTES,
            stages=CFG.XFER_STAGES,
            leading_byte_offset=SF_LEADING_BYTE_OFFSET,
            stride_byte_offset=SF_STRIDE_BYTE_OFFSET,
            layout=SMEM_LAYOUT_SF,
            tma_loads_per_tile=1,
            tma_granu_elems=SF_ATOM_BYTES,
            tma_subtile_stride_elems=SF_ATOM_BYTES,
            desc_version=DESC_VERSION,
        )
        if cutlass.const_expr(_IS_P_B)
        else None
    )
    # dV staging: TMA_DV_ITERS subtiles of (TILE_M kv x DV_D_BLOCK d_v) under the 128-B swizzle; aliases K + V.
    sdV = SmemTile(
        base=sdV_raw,
        elems_per_stage=dVBufferElems,
        stages=1,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=SMEM_LAYOUT_dV,
        tma_loads_per_tile=TMA_DV_ITERS,
        tma_granu_elems=DV_D_BLOCK,
        tma_subtile_stride_elems=DV_BLOCK_SLAB,
        desc_version=DESC_VERSION,
    )
    # Scale-factor slabs: UTCCP sources (32x128b atoms: leading 16 / stride 128 / layout 0) and one-box TMA destinations
    # (tma_loads_per_tile 1, whole box).  The K / V / P slabs are one stage; Q / dO / dO_T rings pace their payload rings.
    sK_SF = SmemTile(
        base=sK_SF_raw,
        elems_per_stage=CFG.SF_SMEM_K,
        stages=CFG.STAGES_KV,
        leading_byte_offset=SF_LEADING_BYTE_OFFSET,
        stride_byte_offset=SF_STRIDE_BYTE_OFFSET,
        layout=SMEM_LAYOUT_SF,
        desc_version=DESC_VERSION,
    )
    sV_SF = SmemTile(
        base=sV_SF_raw,
        elems_per_stage=CFG.SF_SMEM_V,
        stages=CFG.STAGES_KV,
        leading_byte_offset=SF_LEADING_BYTE_OFFSET,
        stride_byte_offset=SF_STRIDE_BYTE_OFFSET,
        layout=SMEM_LAYOUT_SF,
        desc_version=DESC_VERSION,
    )
    sP_SF = SmemTile(
        base=sP_SF_raw,
        elems_per_stage=CFG.SF_SMEM_P,
        stages=1,
        leading_byte_offset=SF_LEADING_BYTE_OFFSET,
        stride_byte_offset=SF_STRIDE_BYTE_OFFSET,
        layout=SMEM_LAYOUT_SF,
        desc_version=DESC_VERSION,
    )
    sQ_SF = SmemTile(
        base=sQ_SF_raw,
        elems_per_stage=CFG.SF_SMEM_Q,
        stages=CFG.STAGES_Q,
        leading_byte_offset=SF_LEADING_BYTE_OFFSET,
        stride_byte_offset=SF_STRIDE_BYTE_OFFSET,
        layout=SMEM_LAYOUT_SF,
        desc_version=DESC_VERSION,
    )
    sdO_SF = SmemTile(
        base=sdO_SF_raw,
        elems_per_stage=CFG.SF_SMEM_dO,
        stages=CFG.STAGES_dO,
        leading_byte_offset=SF_LEADING_BYTE_OFFSET,
        stride_byte_offset=SF_STRIDE_BYTE_OFFSET,
        layout=SMEM_LAYOUT_SF,
        desc_version=DESC_VERSION,
    )
    sdOT_SF = SmemTile(
        base=sdOT_SF_raw,
        elems_per_stage=CFG.SF_SMEM_dOT,
        stages=CFG.STAGES_dO_DV,
        leading_byte_offset=SF_LEADING_BYTE_OFFSET,
        stride_byte_offset=SF_STRIDE_BYTE_OFFSET,
        layout=SMEM_LAYOUT_SF,
        desc_version=DESC_VERSION,
    )

    bars = _make_bars(CFG)
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, alignment=16, space=cutlass.AddressSpace.smem)
    sched = Sched(
        **{
            "mb_scheduler": cutlass.Array(cutlass.Int64, CFG.SCHEDULER_STAGES, alignment=16, space=cutlass.AddressSpace.smem),
            "mb_read_tile_id": cutlass.Array(cutlass.Int64, CFG.SCHEDULER_STAGES, alignment=16, space=cutlass.AddressSpace.smem),
            "tile_id_smem": cutlass.Array(cutlass.Int32, CFG.SCHEDULER_STAGES * 8, alignment=16, space=cutlass.AddressSpace.smem),
            "bidx_init": bidx,
            # NATURAL: bidy / bidz = blockIdx.{y, z} (head, batch).  LPT / LPT_L2 flat 1-D grid: blockIdx.{y, z} = 0 (dead)
            # -> REPURPOSE these slots to carry (n_qh_grid, n_batch) for the linear decode (no extra Sched fields).
            "bidy_init": (n_qh_grid if cutlass.const_expr(CFG.SCHEDULER_POLICY != SCHED_NATURAL) else bidy),
            "bidz_init": (n_batch if cutlass.const_expr(CFG.SCHEDULER_POLICY != SCHED_NATURAL) else bidz),
        }
    )

    # --- cluster role identity (one cga2 pair) ----------------------------------------------------------------------------
    cta_id_x = cute.arch.block_idx_in_cluster()
    cta_in_pair = cta_id_x & cutlass.Int32(1)
    leader_cta_id = cta_id_x & cutlass.Int32(~1 & 0xFFFFFFFF)
    partner_cta_id = cta_id_x ^ cutlass.Int32(1)
    is_leader = cta_in_pair == cutlass.Int32(0)

    # --- mbarrier init: ONE warp, ONE lane (P4); every stage of every ring; then fence -> CTA sync -> cluster sync ------
    if warp_idx == 0:
        if nvvm.elect_sync():
            for s in cutlass.range_constexpr(CFG.STAGES_Q):
                bars.mb_q_full[s].init()
                bars.mb_q_empty[s].init()
            for s in cutlass.range_constexpr(CFG.STAGES_dO):
                bars.mb_do_full[s].init()
                bars.mb_do_empty[s].init()
            for s in cutlass.range_constexpr(CFG.STAGES_dO_DV):
                bars.mb_dodv_full[s].init()
                bars.mb_dodv_empty[s].init()
            for s in cutlass.range_constexpr(CFG.STAGES_KV):
                bars.mb_k_full[s].init()
                bars.mb_v_full[s].init()
                bars.mb_k_empty[s].init()
                bars.mb_v_empty[s].init()
            for p in cutlass.range_constexpr(CFG.STAGES_TMEM_S):
                bars.mb_s_acc_full[p].init()
                bars.mb_dp_full[p].init()
                bars.mb_s_acc_empty[p].init()
                bars.mb_dp_empty[p].init()
            for p in cutlass.range_constexpr(CFG.STAGES_TMEM_P):
                bars.mb_p_ready[p].init()
            bars.mb_p_sf_consumed.init()
            bars.mb_dv_ready.init()
            bars.mb_dv_acc_empty.init()
            bars.mb_dv_stg_full.init()
            bars.mb_dv_stg_empty.init()
            bars.mb_tmem_dealloc.init()
            for p in cutlass.range_constexpr(CFG.STATS_STAGES):
                bars.mb_stats_full[p].init()
                bars.mb_stats_empty[p].init()
            for p in cutlass.range_constexpr(CFG.XFER_STAGES):
                bars.mb_ds_smem_full[p].init()
                bars.mb_ds_smem_empty[p].init()
            # Scheduler rings: init on EVERY CTA (the try_cancel multicast targets all of them).
            for s in cutlass.range_constexpr(CFG.SCHEDULER_STAGES):
                nvvm.mbarrier_init(sched.mb_scheduler.subview(s), ONE_LANE)
                nvvm.mbarrier_init(sched.mb_read_tile_id.subview(s), READ_TILE_ARRIVERS_TOT)

    # SF_P: the constant E8M0 byte CFG.P_SF_BYTE (119 = 2^-8, descaling e4m3(P * 2^8)) over the whole 512-B sP_SF atom, every
    # thread of EACH CTA (the cga2 UTCCP self-fill copies each CTA's OWN slab), BEFORE cga_arrive / cga_wait so both peers hold
    # it before any cta_group::2 op.  A generic store the async-proxy tcgen05.cp reads: it needs its own fence_proxy --
    # fence_mbarrier_init and barrier_cta_sync publish nothing to the async proxy (the forward's order, rules/frost-tile-dsl.md S1;
    # without it a first launch can copy a partially visible atom and pass on the second).
    for _i in cutlass.range_constexpr((CFG.SF_SMEM_P + CFG.THREADS_PER_CTA - 1) // CFG.THREADS_PER_CTA):
        _off = tidx + cutlass.Int32(_i * CFG.THREADS_PER_CTA)
        if _off < cutlass.Int32(CFG.SF_SMEM_P):
            sP_SF_raw.subview(_off).store(cutlass.Int8(CFG.P_SF_BYTE))
    nvvm.fence_proxy("async.shared", space="cta")

    # P4 order: init -> fence -> within-CTA sync -> cluster sync (BEFORE any cross-CTA arrive).  No bootstrap arrives: every
    # first wait on a consumer-side ring is pre-armed by PipelineState.start(phase=1) (P5b).
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()
    cga_arrive()
    cga_wait()

    # Pair-scoped multicast mask for the MMA commits (both CTAs of the pair); TMA multicast = this CTA only.
    mcast_mask = cutlass.Int32(3) << leader_cta_id
    tma_mcast_mask = cutlass.Int16(1) << cta_in_pair
    # Q / dO / dO_T SF boxes: each CTA loads its share and multicasts it to BOTH CTAs (both hold the full N tile's atoms).  The literal
    # 3 == (3 << leader_cta_id) as an Int16 because the config validates the cluster as (CGA_M, CGA_N, 1) = (2, 1, 1): the pair IS the
    # cluster, so leader_cta_id is 0 in every CTA (the forward's spelling, prefill_d256_mxfp8.py sf_mcast_mask).
    sf_mcast_mask = cutlass.Int16(3)
    is_cga_first_cta = cta_id_x == cutlass.Int32(0)

    # --- warp dispatch: 0..7 compute (2 wg) ; 8 MMA ; 9 TMALDG ; 10 TMASTG ; 11 scheduler + stats ------------------------
    if warp_idx < cutlass.Int32(_COMPUTE_WARPS):
        nvvm.setmaxregister(CFG.SOFTMAX_REGS, nvvm.SetMaxRegisterAction.INCREASE)
        _softmax_warp_group(
            warp_idx=warp_idx,
            tmem_ptr_i32=tmem_ptr_i32,
            bars=bars,
            sched=sched,
            sdS_raw=sdS_raw,
            sdS_kv_raw=sdS_kv_raw,
            sdS_SF_raw=sdS_SF_raw,
            sdS_SF16_raw=sdS_SF16_raw,
            sStats_raw=sStats_raw,
            sdV_raw=sdV_raw,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            seqlen_q_real=seqlen_q_real,
            s_scale_log2=s_scale_log2,
            dp_scale=dp_scale,
            cta_in_pair=cta_in_pair,
            leader_cta_id=leader_cta_id,
            partner_cta_id=partner_cta_id,
        )

    elif warp_idx == cutlass.Int32(CFG.MMA_WARP_ID):
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        if is_leader:
            _mma_warp(
                sQ=sQ,
                sdO=sdO,
                sdO_dv=sdO_dv,
                sK=sK,
                sV=sV,
                sK_SF=sK_SF,
                sV_SF=sV_SF,
                sP_SF=sP_SF,
                sQ_SF=sQ_SF,
                sdO_SF=sdO_SF,
                sdOT_SF=sdOT_SF,
                tmem_ptr_i32=tmem_ptr_i32,
                bars=bars,
                sched=sched,
                seq_kv_lens_tensor=seq_kv_lens_tensor,
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
                seqlen_q_real=seqlen_q_real,
                mcast_mask=mcast_mask,
            )
        else:
            _mma_warp_quiet(tmem_ptr_i32, bars)

    elif warp_idx == cutlass.Int32(CFG.TMALDG_WARP_ID):
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        nvvm.prefetch_tensormap(tma_q_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_do_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_do_T_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_k_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_v_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_q_sf_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_k_sf_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_v_sf_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_do_sf_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_do_T_sf_desc.get_ptr())
        _tmaldg_warp(
            tma_q_desc=tma_q_desc,
            tma_do_desc=tma_do_desc,
            tma_do_T_desc=tma_do_T_desc,
            tma_k_desc=tma_k_desc,
            tma_v_desc=tma_v_desc,
            tma_q_sf_desc=tma_q_sf_desc,
            tma_k_sf_desc=tma_k_sf_desc,
            tma_v_sf_desc=tma_v_sf_desc,
            tma_do_sf_desc=tma_do_sf_desc,
            tma_do_T_sf_desc=tma_do_T_sf_desc,
            sQ=sQ,
            sdO=sdO,
            sdO_dv=sdO_dv,
            sK=sK,
            sV=sV,
            sQ_SF=sQ_SF,
            sK_SF=sK_SF,
            sV_SF=sV_SF,
            sdO_SF=sdO_SF,
            sdOT_SF=sdOT_SF,
            bars=bars,
            sched=sched,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            seqlen_q_real=seqlen_q_real,
            qh_per_kh=qh_per_kh,
            n_qh=n_qh,
            head_base=head_base,
            is_leader=is_leader,
            cta_in_pair=cta_in_pair,
            tma_mcast_mask=tma_mcast_mask,
            sf_mcast_mask=sf_mcast_mask,
        )

    elif warp_idx == cutlass.Int32(CFG.TMASTG_WARP_ID):
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        nvvm.prefetch_tensormap(tma_dv_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_ds_desc.get_ptr())
        if cutlass.const_expr(_IS_P_B):
            nvvm.prefetch_tensormap(tma_ds_kv_desc.get_ptr())
            nvvm.prefetch_tensormap(tma_sf_ds_dk_desc.get_ptr())
            nvvm.prefetch_tensormap(tma_sf_ds_dq_desc.get_ptr())
        _tmastg_warp(
            tma_dv_desc=tma_dv_desc,
            tma_ds_desc=tma_ds_desc,
            tma_ds_kv_desc=tma_ds_kv_desc,
            tma_sf_ds_dk_desc=tma_sf_ds_dk_desc,
            tma_sf_ds_dq_desc=tma_sf_ds_dq_desc,
            sdV=sdV,
            sdS=sdS,
            sdS_kv=sdS_kv,
            sdS_SF=sdS_SF,
            bars=bars,
            sched=sched,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            seqlen_q_real=seqlen_q_real,
            cta_in_pair=cta_in_pair,
            head_base=head_base,
            n_qh_grid=n_qh_grid,
        )

    else:  # warp_idx == CFG.SCHED_WARP_ID
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        _scheduler_stats_warp(
            sched=sched,
            is_cga_first_cta=is_cga_first_cta,
            bars=bars,
            sStats_raw=sStats_raw,
            lse_tensor=lse_tensor,
            do_dot_tensor=do_dot_tensor,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            dot_scale=dot_scale,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            seqlen_q_real=seqlen_q_real,
            head_base=head_base,
        )


_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


# === Compute warps: softmax + dsoftmax per q iteration, dV epilogue per kv tile ========================================


@cute.jit
def _softmax_warp_group(
    warp_idx,
    tmem_ptr_i32,
    bars,
    sched,
    sdS_raw,
    sdS_kv_raw,
    sdS_SF_raw,
    sdS_SF16_raw,
    sStats_raw,
    sdV_raw,
    seq_kv_lens_tensor,
    seqlen_q,
    seqlen_kv,
    seqlen_q_real,
    s_scale_log2,
    dp_scale,
    cta_in_pair,
    leader_cta_id,
    partner_cta_id,
) -> None:
    """8 compute warps (2 wg x 4), lane = kv row; wg0 -> q[0:64], wg1 -> q[64:128] (q_half_off).  Per kv tile, per q iter:
      softmax : wait mb_s_acc_full; tmem_load S[q half]; tcgen05_wait(LOAD); arrive mb_s_acc_empty (frees S for Q.K[i+1]);
                P = exp2(S * attn_scale_log2e - lse * log2e) (TRUE units); mask (+ the q < seqlen_q_real band); e4m3(P * 2^8) by
                the scaled cvt with the constant byte 119 -> wait mb_p_sf_consumed (P.dO[n-1] finished reading the scale factors
                aliased in this P slot) -> tcgen05_st into P slot p(n) (this wg's 16 columns); tcgen05_wait(STORE); the relaxed
                arrive on mb_p_ready (EARLY, before dsoftmax, so the dV BMM2 overlaps the whole dsoftmax below).
      dsoftmax: wait mb_dp_full; tmem_load dP[q half]; tcgen05_wait(LOAD); arrive mb_dp_empty (frees dP for dO.V[i+1]);
                dS = (dP * attn_scale - delta * attn_scale) * P (the fp32 P); P-c: -> bf16 -> sdS ring slot (this wg's 64 q cols =
                its own subtile).  P-b: -> ds_dk (a lane's two 32-q blocks: abs_max_tree -> e8m0_pair -> the scaled 16-pack) into
                the sdS slot + the two bytes into the sf_ds_dk atom (one st.u16); ds_dq (per column: warp_abs_max_f32 = ONE
                redux.sync.max.abs.f32 over the warp's 32 kv rows -> e8m0_pair_u, the column pair's bytes in one word -> the lone
                scaled cvt per element with its column's byte, fp32_to_fp8x4_scaled_pairs) into the sdS_kv slot + lane l's bytes of
                columns 2l, 2l+1 (the same pair words) into the sf_ds_dq atom (two st.u8).  Then ONE fence_proxy; arrive
                mb_ds_smem_full; arrive mb_stats_empty.
    Post q loop (per kv tile): wait mb_dv_ready; per 64-col chunk of this wg's d_v half: tmem_load dV; tcgen05_wait(LOAD);
    the TRUE-unit accumulator -> OUT dtype -> sdV (store_swizzled); fence_proxy; arrive mb_dv_stg_full; arrive mb_dv_acc_empty
    (frees dV TMEM for the next tile's P.dO[0], accumulate=False).  No amax folds, no atomics (the row produces no amax)."""
    # Pair with the MMA warp's barrier_cta_arrive: the TMEM base is published after its tmem_alloc.
    nvvm.barrier_cta_sync(barrier_id=_NAMED_BAR_TMEM_ID, thread_count=_NAMED_BAR_TMEM_THREADS)

    kv_super_idx, head_idx, batch_idx = _boot_tile(sched)

    # tid_in_wg: 0..127 = a lane's kv row within its warpgroup; wg_id (0 / 1) -> q half.
    tid_in_wg = cute.arch.thread_idx()[0] & cutlass.Int32(127)
    wg_id = (warp_idx - cutlass.Int32(CFG.SOFTMAX_WG0_BASE)) // cutlass.Int32(CFG.SOFTMAX_WG_WARPS)
    q_half_off = wg_id * cutlass.Int32(_SMX_CHUNK)  # 0 or 64 q cols
    # This wg's 16 columns inside a P ring slot: 64 q x 1 B / 4 B per column (wg0 0, wg1 16); lane = kv row = TMEM lane.
    p_col_off = wg_id * cutlass.Int32(_SMX_CHUNK * CFG.BPE // 4)
    # This wg's 64 q cols inside a dS ring slot: subtile blk = q_half // P_D_BLOCK (its slab is blk * P_BLOCK_ELEMS), column
    # col_in_blk = q_half % P_D_BLOCK inside a lane's 128-B swizzle row.  e4m3 (P_D_BLOCK = 128): blk 0, col 0 | 64 -- both
    # halves in ONE row; bf16 (P_D_BLOCK = 64): blk = wg, col 0 -- one subtile per wg (the pre-port `wg * P_BLOCK_ELEMS`).
    ds_blk = q_half_off // cutlass.Int32(P_D_BLOCK)
    ds_wg_off = ds_blk * cutlass.Int32(P_BLOCK_ELEMS) + (q_half_off - ds_blk * cutlass.Int32(P_D_BLOCK))
    dv_col_base = wg_id * cutlass.Int32(CFG.TILE_O // CFG.SOFTMAX_WARPGROUPS)  # this wg's d_v half
    # P-b per-lane invariants (hoisted out of the q loop): a lane's id bits for the pair select, and the two atom byte offsets.
    #   sf_ds_dk atom: r = tid_in_wg (a lane's kv row within the CTA's 128), c4 = 2 * wg + blk -> the two bytes are adjacent and
    #   the offset is even: kept as a 16-bit INDEX (byte / 2) into the Int16 alias.
    #   sf_ds_dq atom: r = q_half_off + 2 * lane (the first of its two columns), c4 = this warp's index within the wg (= its 32-kv block);
    #   column 2 * lane + 1 is the next row = +SF_ATOM_LINE_BYTES.
    lane_in_warp = tid_in_wg & cutlass.Int32(31)
    warp_in_wg = tid_in_wg >> cutlass.Int32(5)
    lane_bits = [(lane_in_warp & cutlass.Int32(1 << _k)) != cutlass.Int32(0) for _k in range(5)] if cutlass.const_expr(_IS_P_B) else None
    # fp32(1/448) in a VECTOR register ptxas cannot fold to an immediate (one FFMA, once per kernel): e8m0_pair_u multiplies a column
    # PAIR of CREDUX amaxes by it in ONE packed FMUL2 (32 per iteration instead of 64 scalar FMULs), and the packed multiply has no
    # immediate operand form.  The `MOV R, UR` per column stays: sm_107a ptxas moves a redux result into the vector file before any
    # ALU op reads it, in every spelling tried (13 forms, 64 moves each) -- the floor, not a cost this hoist removes.
    dq_rcp_max = opaque_e4m3_max_rcp_in_lane(lane_in_warp.to(cutlass.Float32)) if cutlass.const_expr(_IS_P_B) else None
    dk_sf_off16 = (
        sf_atom_byte(tid_in_wg, wg_id * cutlass.Int32(_DS_BLOCKS_PER_WG), base=_DS_SF_ATOM_DK_OFF) >> cutlass.Int32(1) if cutlass.const_expr(_IS_P_B) else None
    )
    dq_sf_off = sf_atom_byte(q_half_off + lane_in_warp * cutlass.Int32(2), warp_in_wg, base=_DS_SF_ATOM_DQ_OFF) if cutlass.const_expr(_IS_P_B) else None

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    s_full_state = PipelineState.start()  # consume mb_s_acc_full
    dp_full_state = PipelineState.start()  # consume mb_dp_full
    ds_empty_state = PipelineState.start(phase=1)  # dS ring slot free (pre-armed)
    p_ready_state = PipelineState.start()  # produce mb_p_ready (the 2-stage TMEM P ring; its idx is the slot rule's p(n))
    p_sf_state = PipelineState.start(phase=1)  # consume mb_p_sf_consumed (pre-armed: the kernel's first P store has no prior P.dO)
    dv_ready_state = PipelineState.start()  # consume mb_dv_ready
    stats_full_state = PipelineState.start()  # consume mb_stats_full

    # Per-lane absolute kv row base: this CTA's M slice of the pair's kv block.
    kv_lane_base0 = cta_in_pair * cutlass.Int32(CFG.TILE_M) + tid_in_wg

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        tmem_base = tmem_ptr_i32.load()  # TMEM col base (published by the MMA warp's alloc)

        kv_block_base = kv_super_idx * cutlass.Int32(_KV_BLOCK_ROWS)
        # This tile's REAL kv length (the uniform scalar, or seq_kv_lens[b] under the PADDED arm), the bottom-right causal
        # diagonal kv <= q + (S_kv - S_q) it anchors on the REAL lengths (0 for top-left / dense) and the q range it bounds.
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        causal_diag = _causal_diag(seqlen_q_real, eff_seqlen_kv)
        q_lo, q_hi = _q_loop_bounds(kv_block_base, seqlen_q, seqlen_q_real, eff_seqlen_kv)
        kv_abs = kv_block_base + kv_lane_base0

        for q_iter in cutlass.range(q_lo, q_hi, 1, unroll=1):
            # ---- stats ring: this q tile's lse_s / delta_s (lane = kv row reads the SAME q cols -> broadcast LDS) ----
            stats_slot = stats_full_state.idx
            bars.mb_stats_full[stats_slot].wait(stats_full_state.phase, spin=SPIN_RING_WAITS)
            stats_base = stats_slot * cutlass.Int32(STATS_SLOT_ELEMS) + q_half_off

            # ---- 1) softmax: S[q half] -> P_s (registers) ----
            bars.mb_s_acc_full[s_full_state.idx].wait(s_full_state.phase, spin=SPIN_RING_WAITS)
            s_full_state = advance(s_full_state, CFG.STAGES_TMEM_S)
            reg_S = tmem_load_tile(tmem_base + cutlass.Int32(LAYOUT.S_OFF) + q_half_off, num_elems=_SMX_CHUNK, ld_num=_LDTM_NUM)
            # The LOAD-side wait ORDERS the arrive after the last LDTM (frost-kernels S3): the arrive has no data dependency
            # on the loaded registers, and without it ptxas may schedule it between two LDTMs -> a parked Q.K[i+1] overwrites
            # S under the pending second read.
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            # Free S for Q.K[i+1] (the lookahead gate) -- every compute lane of both CTAs arrives on the LEADER (512).
            bars.mb_s_acc_empty.arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
            lse_elems = []
            for j in cutlass.range_constexpr(_SMX_CHUNK // 4):
                lse_elems.extend(sStats_raw.load(stats_base + cutlass.Int32(STATS_LSE_OFF + 4 * j), vector_size=4, alignment=16).to_elements())
            lse_vec = cutlass.Vector.from_elements(tuple(lse_elems), cutlass.Float32)
            # P in TRUE units: S_acc is dequantized in-MMA and the stats warp folded lse * log2e with NO constant shift (the oracle's
            # arithmetic order, mxfp8_ref.py; folding the P scale's +8 into this argument costs 2^-21 of argument rounding against the oracle).
            chunk_P = cute.math.exp2(reg_S.vec * s_scale_log2 - lse_vec, fastmath=True)
            # Transposed mask: zero P on masked (kv = lane, q = col) cells -> the e4m3 P (dV) AND dS inherit it; plus the
            # q < seqlen_q_real band under MASK_Q_PAD (a SEPARATE const_expr arm).  No IR at NONE without the band.
            if cutlass.const_expr(CFG.MASK_FLAGS != MASK_NONE or CFG.MASK_Q_PAD):
                chunk_P = _mask_p_chunk(
                    chunk_P, kv_abs, q_iter * cutlass.Int32(CFG.TILE_N) + q_half_off, eff_seqlen_kv, causal_diag, _SMX_CHUNK, seqlen_q_real=seqlen_q_real
                )

            # ---- 2) e4m3(P * 2^8) -> the P ring slot p(n) in TMEM -> tcgen05_wait(STORE) -> relaxed arrive (EARLY, before dsoftmax, so
            #         the dV BMM2 overlaps the dP load + dS compute + dS store below).  Each wg stores its 16 columns (64 q x 1 B).  The
            #         scale byte is CFG.P_SF_BYTE as a PYTHON INT: the cvt.u8.u32 IMMEDIATE feeding the 16-pack is the ONE constant
            #         form the scaled cvt assembles with (a provable-constant register or a kernel parameter ICEs ptxas C7907,
            #         tile_dsl/pointwise.py header); fused=False is one FMUL by an opaque 256.0 then the plain pack, bit-identical. ----
            p_slot = p_ready_state.idx
            p_words = []
            for _j in cutlass.range_constexpr(_SMX_CHUNK // _P_PACK):
                _vals = []
                for _i in cutlass.range_constexpr(_P_PACK):
                    _vals.append(chunk_P[_j * _P_PACK + _i])
                _w = fp32_to_fp8_pack_scaled(_vals, CFG.P_SF_BYTE, dtype=STORAGE_DTYPE, fused=bool(CFG.SCALED_FP8_PACK))
                for _i in cutlass.range_constexpr(4):
                    p_words.append(_w[_i])
            # The words are BIT PATTERNS (element i in byte i) = exactly the 32-bit TMEM words tcgen05.st takes: 16 words = this wg's
            # 16 columns of the slot (the layout the fp8 body's mma_ts A operand has always read).
            chunk_P_words = cutlass.Vector.from_elements(tuple(p_words), cutlass.Int32)
            # BARRIER TABLE row mb_p_sf_consumed (consumer): P.dO[n-1] read its scale factors from THIS slot (s(n-1) = p(n)) -- and the
            # MMA warp's UTCCPs wrote them there -- so the store below may not enter it before that MMA's commit fired.  Pre-armed
            # for the kernel's first store; one wait per q iteration on every path, the drain after the loop consumes the last commit.
            bars.mb_p_sf_consumed.wait(p_sf_state.phase, spin=SPIN_RING_WAITS)
            p_sf_state = advance(p_sf_state, 1)
            nvvm.tcgen05_st(
                "32x32b",
                nvvm.make_tmem_ptr(tmem_base + cutlass.Int32(LAYOUT.P_OFF) + p_slot * cutlass.Int32(LAYOUT.P_COLS) + p_col_off, cutlass.Float32),
                chunk_P_words,
            )
            # The STORE-side wait completes the async TMEM store before the arrive publishes the slot (the fp8 body's pair).
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
            # BARRIER TABLE row mb_p_ready (producer): relaxed LEADER arrive, every compute lane of both CTAs = 512.
            bars.mb_p_ready[p_slot].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
            p_ready_state = advance(p_ready_state, CFG.STAGES_TMEM_P)

            # ---- 3) dsoftmax: dP -> free the dP slot -> dS = (dP * dp_scale - delta_s) * P_s -> P-c: bf16 | P-b: block-scaled e4m3 x 2 +
            #         two E8M0 atoms -> the dS ring slot ----
            bars.mb_dp_full[dp_full_state.idx].wait(dp_full_state.phase, spin=SPIN_RING_WAITS)
            dp_full_state = advance(dp_full_state, CFG.STAGES_TMEM_S)
            reg_dP = tmem_load_tile(tmem_base + cutlass.Int32(LAYOUT.dP_OFF) + q_half_off, num_elems=_SMX_CHUNK, ld_num=_LDTM_NUM)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            # dP read into registers -> the slot is free for dO.V[i+1] (every lane arrives on the leader).
            bars.mb_dp_empty.arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
            dot_elems = []
            for j in cutlass.range_constexpr(_SMX_CHUNK // 4):
                dot_elems.extend(sStats_raw.load(stats_base + cutlass.Int32(STATS_DOT_OFF + 4 * j), vector_size=4, alignment=16).to_elements())
            dot_vec = cutlass.Vector.from_elements(tuple(dot_elems), cutlass.Float32)
            # dS = attn_scale * P * (dP - delta) in fp32 from the UNQUANTIZED P (the oracle's and the SM100 chain's spelling; the e4m3
            # P touches dV only).  dP is TRUE units (SF_V / SF_dO dequantize in-MMA); a masked cell is P = 0 times a FINITE dP = 0.
            chunk_dS = (reg_dP.vec * dp_scale - dot_vec) * chunk_P
            ds_slot = ds_empty_state.idx
            # dS SMEM [kv, q]: 128-B-row subtiles per slot under Swizzle(3, 4, 3) (job 2 bank spread, job 1 the s128b store
            # descriptor); a lane's kv row at its wg's columns (ds_wg_off).  e4m3: 64 B = half the row (the XOR permutes
            # the row's 16-B chunks, so the two wgs' halves stay disjoint); bf16: 64 x 2 B = the whole row of wg's own subtile.
            ds_lane_elem = ds_slot * cutlass.Int32(dSBufferElems) + ds_wg_off + tid_in_wg * cutlass.Int32(P_D_BLOCK)
            if cutlass.const_expr(_IS_P_B):
                # ---- P-b (a) ds_dk = e4m3(dS * 2^(127 - e_blk)), e_blk per 32-q block of each lane's kv row (in-lane amax).  Two
                #      blocks per warpgroup half (validated), ONE cvt.rp.satfinite.ue8m0x2 for both bytes (e8m0_pair == the oracle's
                #      e8m0_ceil(amax * fp32(1/448)); amax 0 -> byte 0 -> payload 0).  The bytes are per-lane DATA -- the provenance the
                #      fused scaled cvt accepts (tile_dsl/pointwise.py header); fused=False is the bit-identical FMUL arm. ----
                blk_max = []
                for _b in cutlass.range_constexpr(_DS_BLOCKS_PER_WG):
                    blk_max.append(abs_max_tree([chunk_dS[_b * CFG.SF_BLOCK + _i] for _i in range(CFG.SF_BLOCK)]))
                _rcp_dk0, _rcp_dk1, dk_pair = e8m0_pair(blk_max[0], blk_max[1])
                dk_bytes = [dk_pair & cutlass.Int32(0xFF), (dk_pair >> cutlass.Int32(8)) & cutlass.Int32(0xFF)]
                dk_words = []
                for _b in cutlass.range_constexpr(_DS_BLOCKS_PER_WG):
                    for _j in cutlass.range_constexpr(CFG.SF_BLOCK // _DS_PACK):
                        _vals = [chunk_dS[_b * CFG.SF_BLOCK + _j * _DS_PACK + _i] for _i in range(_DS_PACK)]
                        _w = fp32_to_fp8_pack_scaled(_vals, dk_bytes[_b], dtype=DS_STORAGE_DTYPE, fused=bool(CFG.SCALED_FP8_PACK))
                        for _i in cutlass.range_constexpr(4):
                            dk_words.append(_w[_i])
                # The payload stays a vector of Int32 WORDS across the ring wait below and is bitcast at the store: a 64-element fp8
                # vector live across the wait's basic block is legalized element-wise (16-bit registers: 4 PRMT + 4 zero-extends to
                # unpack every word, 3 PRMT to rebuild it -- ~420 instructions per iteration for the two payloads, MEASURED on
                # 987eb02b), while a bitcast in the store's own block folds into the 16-B stores.
                chunk_dS_dk_words = cutlass.Vector.from_elements(tuple(dk_words), cutlass.Int32)
                # ---- P-b (b) ds_dq = e4m3(dS * 2^(127 - e_col)), e_col per 32-kv block of each q COLUMN = this warp's 32 lanes: ONE
                #      redux.sync.max.abs.f32 per column (warp_abs_max_f32; a warp-uniform result), the column pair's bytes by
                #      e8m0_pair_u (the e8m0_pair rule op for op -- amax * fp32(1/448), then the ceil cvt -- as ONE packed FMUL2 per
                #      column pair; the MOV R,UR per CREDUX result is the toolchain's floor), and every element converted by the LONE
                #      fused scaled cvt with ITS column's byte out of the pair word (fp32_to_fp8x4_scaled_pairs: no fp32 scale build,
                #      no FMUL; the fused=False twin is the e8m0_rcp FMUL + the plain pair cvt).  Per 16-column group, so <= 16 column
                #      amaxes are live at once; the 32 pair words stay live for the atom bytes below. ----
                dq_words = []
                dq_pairs = []
                for _g in cutlass.range_constexpr(_SMX_CHUNK // _DS_PACK):
                    col_max = [warp_abs_max_f32(chunk_dS[_g * _DS_PACK + _i]) for _i in range(_DS_PACK)]
                    for _q in cutlass.range_constexpr(_DS_PACK // 4):
                        _c0 = _g * _DS_PACK + _q * 4
                        _p01 = e8m0_pair_u(col_max[_q * 4], col_max[_q * 4 + 1], dq_rcp_max)
                        _p23 = e8m0_pair_u(col_max[_q * 4 + 2], col_max[_q * 4 + 3], dq_rcp_max)
                        dq_pairs.append(_p01)
                        dq_pairs.append(_p23)
                        dq_words.append(
                            fp32_to_fp8x4_scaled_pairs(
                                [chunk_dS[_c0 + _i] for _i in range(4)], _p01, _p23, dtype=DS_STORAGE_DTYPE, fused=bool(CFG.SCALED_FP8_PACK)
                            )
                        )
                chunk_dS_dq_words = cutlass.Vector.from_elements(tuple(dq_words), cutlass.Int32)
                # Lane l publishes the bytes of columns 2l, 2l+1 = pair l of the 32 warp-uniform pair words -- the SAME words the
                # payload was scaled by, no second derivation (a select tree on its lane bits, never a runtime-indexed register
                # array): structural election, 32 lanes x 2 bytes = the warp's 64 bytes of the atom, no duplicate store.
                dq_pair_mine = _select_by_lane(dq_pairs, lane_bits)
            else:
                # P-c: the workspace value is the fp32 dS rounded to bf16 -- no scale, no E8M0.
                chunk_dS_ws = chunk_dS.to(DS_STORAGE_DTYPE)
            # ---- the slot: wait it free (ONE wait site for both policies), then the buffers -> ONE proxy fence -> the arrive ----
            bars.mb_ds_smem_empty[ds_slot].wait(ds_empty_state.phase, spin=SPIN_RING_WAITS)
            ds_empty_state = advance(ds_empty_state, CFG.XFER_STAGES)
            if cutlass.const_expr(_IS_P_B):
                # P-b: the four buffers of the slot -- two payloads, two atoms.
                # The words are BIT PATTERNS (element i in byte i): bitcast to the storage dtype HERE, in the store's block (above),
                # and before the store -- store_swizzled VALUE-casts to the pointer's dtype (rules/frost-gotchas.md, the packed-fp8 row).
                chunk_dS_dk = chunk_dS_dk_words.bitcast(DS_STORAGE_DTYPE)
                chunk_dS_dq = chunk_dS_dq_words.bitcast(DS_STORAGE_DTYPE)
                (sdS_raw.subview(ds_lane_elem)).data_ptr().store_swizzled(chunk_dS_dk, alignment=_SMX_CHUNK * CFG.BPE_DS, swizzle=STAGING_SMEM_SWIZZLE)
                (sdS_kv_raw.subview(ds_lane_elem)).data_ptr().store_swizzled(chunk_dS_dq, alignment=_SMX_CHUNK * CFG.BPE_DS, swizzle=STAGING_SMEM_SWIZZLE)
                sf_stage_off = ds_slot * cutlass.Int32(_DS_SF_STAGE_BYTES)
                # sf_ds_dk atom (r = kv row, c4 = 2 * wg + blk): a lane's two adjacent bytes as ONE 16-bit store (Int16 index = byte / 2).
                sdS_SF16_raw.subview((sf_stage_off >> cutlass.Int32(1)) + dk_sf_off16).store(dk_pair.to(cutlass.Int16))
                # sf_ds_dq atom (r = q column, c4 = kv-block = warp): columns 2l and 2l+1 are consecutive rows = one 16-B line apart.
                sdS_SF_raw.subview(sf_stage_off + dq_sf_off).store((dq_pair_mine & cutlass.Int32(0xFF)).to(cutlass.Int8))
                sdS_SF_raw.subview(sf_stage_off + dq_sf_off + cutlass.Int32(SF_ATOM_LINE_BYTES)).store(
                    ((dq_pair_mine >> cutlass.Int32(8)) & cutlass.Int32(0xFF)).to(cutlass.Int8)
                )
            else:
                (sdS_raw.subview(ds_lane_elem)).data_ptr().store_swizzled(chunk_dS_ws, alignment=_SMX_CHUNK * CFG.BPE_DS, swizzle=STAGING_SMEM_SWIZZLE)
            # Generic SMEM writes (P-b: all four buffers of the slot) -> async-proxy TMA store: the real proxy fence (rules/frost-tile-dsl.md S1).
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_ds_smem_full[ds_slot].arrive()
            bars.mb_stats_empty[stats_slot].arrive()
            stats_full_state = advance(stats_full_state, CFG.STATS_STAGES)

        # ---- dV epilogue (per kv tile): dV TMEM -> dV_true -> OUT dtype -> sdV SMEM (aliases K + V, dead by now) ----
        # sdV is TMA_DV_ITERS subtiles of (TILE_M kv x DV_D_BLOCK d_v) under the 128-B swizzle; this wg owns
        # TILE_O / SOFTMAX_WARPGROUPS d_v cols, walked in 64-col chunks: chunk cols [gcol, gcol + 64) land at
        # (gcol // DV_D_BLOCK) * DV_BLOCK_SLAB + row * DV_D_BLOCK + (gcol % DV_D_BLOCK), with the swizzle XOR.
        bars.mb_dv_ready.wait(dv_ready_state.phase)
        dv_ready_state = advance(dv_ready_state, 1)
        for _c in cutlass.range_constexpr(_DV_CHUNKS_PER_WG):
            gcol = dv_col_base + cutlass.Int32(_c * _DV_EPI_CHUNK)
            reg_dV = tmem_load_tile(tmem_base + cutlass.Int32(LAYOUT.dV_OFF) + gcol, num_elems=_DV_EPI_CHUNK)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            # dV_acc = sum e4m3(P * 2^8) . dO_T, dequantized in-MMA by SF_P (2^-8) and SF_dOT: TRUE units -> the half-precision output.
            dv_out = reg_dV.vec.to(OUT_STORAGE_DTYPE)
            dv_blk = gcol // cutlass.Int32(DV_D_BLOCK)
            dv_col_in_blk = gcol - dv_blk * cutlass.Int32(DV_D_BLOCK)
            (sdV_raw.subview(dv_blk * cutlass.Int32(DV_BLOCK_SLAB) + tid_in_wg * cutlass.Int32(DV_D_BLOCK) + dv_col_in_blk)).data_ptr().store_swizzled(
                dv_out, alignment=_DV_EPI_CHUNK * CFG.BPE_O, swizzle=STAGING_SMEM_SWIZZLE
            )
        nvvm.fence_proxy("async.shared", space="cta")
        bars.mb_dv_stg_full.arrive()
        # Free the dV TMEM for the next kv tile's P.dO[0] (accumulate=False) overwrite.
        bars.mb_dv_acc_empty.arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

        # ---- scheduler tail ----
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        kv_super_idx, head_idx, batch_idx, is_valid_tile = _decode_tile_payload(sched, sched_state.idx)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)

    # P15 drain, BARRIER TABLE row mb_p_sf_consumed: the last P.dO's commit is an ASYNC multicast onto a possibly-exiting peer and
    # nothing above waited it (wait k consumes commit k - 1).  Every softmax lane of both CTAs waits it here, BEFORE the dealloc
    # arrives below (which keep both CTAs' MMA warps, and so both CTAs, resident until it has landed).
    bars.mb_p_sf_consumed.wait(p_sf_state.phase)
    # Release the MMA warps' TMEM allocation on BOTH CTAs (256 local + 256 from the partner's compute lanes = 512 each).
    bars.mb_tmem_dealloc.arrive()
    bars.mb_tmem_dealloc.arrive_on_peer(partner_cta_id)


# === MMA warp (pair leader) + the follower's quiet MMA warp ==============================================================


def _utccp_sf_atoms(tmem_sf, desc_sf, n_atoms: int) -> None:
    """SMEM -> TMEM copy of ``n_atoms`` F8_128x4 scale-factor atoms: one ``tcgen05.cp 32x128b`` per 512-B atom = 4 TMEM columns
    (``SF_TMEM_COLS_PER_ATOM``), ``cta_group::2`` self-fill (the leader-issued copy moves EACH CTA's own SMEM into its own TMEM;
    the follower issues nothing).  Descriptor arithmetic is in 16-byte units: ``desc + _SF_ATOM_DESC_STEP`` is the next atom.
    A trace-time macro that emits IR at the call site: the CALLER elect-gates it (``if nvvm.elect_sync():``) -- nothing here elects,
    and an un-gated call is 32 redundant copies (rules/frost-tile-dsl.md S10).  No fence before it: a TMA-written atom is async ->
    async; the constant P atom was proxy-fenced once at kernel start.  Ordering against the MMAs that read / previously read the
    same columns is the in-order tcgen05 stream of the issuing thread (the forward's per-stage refill pattern).  The alias adds ONE
    writer of those columns the stream cannot order -- the softmax lanes' ``tcgen05.st`` of the next P -- and that is
    ``mb_p_sf_consumed`` (BARRIER TABLE)."""
    for a in range(n_atoms):
        nvvm.tcgen05_cp(
            nvvm.Tcgen05CpShape.SHAPE_32X128B,
            tmem_sf.subview(a * SF_TMEM_COLS_PER_ATOM),
            desc_sf + a * _SF_ATOM_DESC_STEP,
            group=CTA_GROUP_KIND,
            multicast=nvvm.Tcgen05CpMulticast.WARPX4,
        )


def _sf_alias_views(tmem_P, p_slot):
    """The six scale-factor TMEM bases of ONE q iteration, aliased into the dead P slot ``s = p ^ 1`` (the slot rule of the module
    docstring; ``p`` = the ``mb_p_ready`` ring index of the iteration): ``P_OFF + s * P_COLS + LAYOUT.SF_*_OFF`` (slot-relative,
    config-derived).  A trace-time macro that emits IR at the call site -- the SLOT is a runtime parity, the offsets are constants."""
    sf_base = tmem_P.subview((p_slot ^ cutlass.Int32(1)) * cutlass.Int32(LAYOUT.P_COLS))
    return (
        sf_base.subview(cutlass.Int32(LAYOUT.SF_K_OFF)),
        sf_base.subview(cutlass.Int32(LAYOUT.SF_V_OFF)),
        sf_base.subview(cutlass.Int32(LAYOUT.SF_Q_OFF)),
        sf_base.subview(cutlass.Int32(LAYOUT.SF_dO_OFF)),
        sf_base.subview(cutlass.Int32(LAYOUT.SF_P_OFF)),
        sf_base.subview(cutlass.Int32(LAYOUT.SF_dOT_OFF)),
    )


@cute.jit
def _mma_warp(
    sQ,
    sdO,
    sdO_dv,
    sK,
    sV,
    sK_SF,
    sV_SF,
    sP_SF,
    sQ_SF,
    sdO_SF,
    sdOT_SF,
    tmem_ptr_i32,
    bars,
    sched,
    seq_kv_lens_tensor,
    seqlen_q,
    seqlen_kv,
    seqlen_q_real,
    mcast_mask,
) -> None:
    """MMA leader: the 3-matmul stream, every MMA block-scaled.  Per kv tile (K, V one-shot) the issue order is FIXED per MASK ARM
    (``S_LOOKAHEAD``, the module constant next to ``SPIN_RING_WAITS``: the two orders are bitwise identical -- the same MMAs on the
    same operands, only the order of issue moves -- and MEASURED there, the dense row faster with Q.K at the iteration top, the
    causal row with the lookahead):

        [UTCCP K-SF, Q-SF -> s]  Q.K[q_lo]                                                          (both arms: the prologue)
        S_LOOKAHEAD (every masked arm, the fp8 body's order):
          for i in q_lo .. q_hi-2:  [UTCCP V-SF, dO-SF -> s] dO.V[i] ; [UTCCP K-SF, Q-SF -> s] Q.K[i+1] ; [UTCCP P-SF, dO_T-SF -> s] P.dO[i] ; commit p_sf_consumed
          [UTCCP V-SF, dO-SF -> s] dO.V[q_hi-1] ; [UTCCP P-SF, dO_T-SF -> s] P.dO[q_hi-1] ; commit p_sf_consumed
        not S_LOOKAHEAD (the dense arm):
          for i in q_lo .. q_hi-1:  [wait dp_empty] ( i > q_lo: [UTCCP K-SF, Q-SF -> s] Q.K[i] ) ; [UTCCP V-SF, dO-SF -> s] dO.V[i] ;
                                    [UTCCP P-SF, dO_T-SF -> s] P.dO[i] ; commit p_sf_consumed
          Q.K[i] follows the mb_dp_empty wait: it issues only once dsoftmax[i-1] has released dP[i-1], so S[i] is never computed
          while dsoftmax[i-1] runs.  The lookahead instead put the 4-k-step S MMA in the tensor pipe between dO.V[i] and P.dO[i],
          where it competed with the softmax warps' TMEM loads and queued P.dO[i] behind it -- the measured dense loss; on causal
          tiles the short q loops near the diagonal need S[i+1] early, so they keep the lookahead.
        The loop's Q.K block is ONE block spelled once per arm (the same two waits, two copies, MMA, two commits and two advances,
        N - 1 instances per tile under either order: ``q_iter > q_lo`` <=> ``q_iter + 1 < q_hi`` over the same loop), selected by
        ``cutlass.const_expr(S_LOOKAHEAD)`` -- exactly one Q.K issue site per iteration traces, plus the prologue.

    with s = the SF slot of the iteration = P slot p ^ 1 (p = p_ready_state.idx; the slot rule of the module docstring): 11 UTCCPs per
    q iteration into the P-ring slot the softmax is NOT writing, plus 4 for the prologue.  Every UTCCP sits between the ``_full``
    wait of the operand it scales and the MMA that reads it (one in-order tcgen05 stream: the copy lands before the MMA, and after
    the MMA that last read those columns -- P.dO[i-1] read P[i-1] from slot s; the iteration's Q.K / dO.V[i] read K / V from the
    columns the BMM2 band then overwrites).
    Handshakes (single-buffer S / dP; the e4m3 P ring is the fp8 body's 2-stage TMEM ring):
      mb_s_acc_empty  gates the loop's Q.K (softmax loaded the previous S)   pre-armed (PipelineState.start(phase=1))
      mb_s_acc_full   Q.K  -> softmax "S[i] ready"
      mb_dp_full      dO.V -> softmax "dP[i] ready"
      mb_dp_empty     gates dO.V[i+1] (dsoftmax loaded dP[i])     pre-armed
      mb_p_ready[2]   softmax -> P.dO[i] "P[i] in TMEM slot p" (relaxed arrive after tcgen05_wait(STORE))
      mb_p_sf_consumed  P.dO[i] -> softmax "slot s = p(i+1) is free of scale-factor readers" (the ONE new ring; committed right
                      after P.dO[i], waited by every softmax lane before its store of P[i+1])
    P-slot reuse is gated by the 2-stage ring + the in-order tcgen05 stream (s_acc_full[i+2] fires after P.dO[i] read slot
    p, under both S issue orders) -- no p_empty.  At exit the two pre-armed LEADER-scope rings hold one completed, un-waited
    phase each (256 of its 512 arrives cross-CTA); they are drained here so the follower's last cluster arrive lands on a
    resident CTA (P15, port fix F3); the softmax warps drain mb_p_sf_consumed."""
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND, is_exclusive=TMEM_IS_EXCLUSIVE)
    # Publish the TMEM base: pair with the compute warps' barrier_cta_sync at their top of body.
    nvvm.barrier_cta_arrive(_NAMED_BAR_TMEM_ID, _NAMED_BAR_TMEM_THREADS)

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    k_full_state = PipelineState.start()
    v_full_state = PipelineState.start()
    q_full_state = PipelineState.start()
    do_full_state = PipelineState.start()  # dO ring #1 (BMM1 dP)
    dodv_full_state = PipelineState.start()  # dO ring #2 (BMM2 dV)
    # Consumer handshakes (single buffer): pre-armed so the prologue Q.K[q_lo] / the first dO.V pass with no softmax load
    # yet.  p_ready is a real consumer wait.
    s_acc_empty_state = PipelineState.start(phase=1)
    dp_empty_state = PipelineState.start(phase=1)
    p_ready_state = PipelineState.start(phase=0)
    dv_empty_state = PipelineState.start(phase=0)  # epilogue drained dV_acc

    # batch_idx is KEPT: the per-batch kv length (PADDED arm) bounds this tile's q range.
    kv_super_idx, _, batch_idx = _boot_tile(sched)

    tmem_raw = nvvm.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)
    tmem_S = tmem_raw.subview(cutlass.Int32(LAYOUT.S_OFF))
    tmem_dP = tmem_raw.subview(cutlass.Int32(LAYOUT.dP_OFF))
    tmem_dV = tmem_raw.subview(cutlass.Int32(LAYOUT.dV_OFF))
    # e4m3 P 2-stage ring (the dV mma_ts A operand); the SF atoms of each iteration alias its dead slot (_sf_alias_views).
    tmem_P = tmem_raw.subview(cutlass.Int32(LAYOUT.P_OFF))

    # ---- block-scale descriptors (delta 4; Rubin MXFP8: K = 64 per instruction, idesc k_dim = 1, two 32-blocks per k-step;
    #      mma_ss walks sf_id 0 / 2 and advances the SF column group by 4 every two steps -- the spellings of the Rubin forward) ----
    # BMM1 S = K . Q^T   (A = K, M = kv; B = Q, N = q; K = d_qk).  sf_a = SF_K (this CTA's kv rows), sf_b = SF_Q (the full N tile).
    idesc_bmm1_s = prims.Tcgen05MxInstrDesc.build(
        a_dtype=STORAGE_DTYPE, b_dtype=STORAGE_DTYPE, n_dim=CFG.TILE_N, m_dim=CFG.TILE_M * CFG.CTA_MMA, k_dim=CFG.IDESC_K_DIM
    )
    bmm1_s_desc = MmaDesc(
        M=CFG.TILE_M * CFG.CTA_MMA,
        N=CFG.TILE_N,
        K=CFG.TILE_K,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.TILE_K_HW_BMM1,
        btranspose=False,
        cta_group=CFG.CTA_MMA,
        idesc=idesc_bmm1_s,
        kind=MMA_KIND,
        is_block_scale=True,
        sf_blocks_per_step=CFG.SF_BLOCKS_PER_STEP,
        scale_vec_size=SCALE_VEC_SIZE,
    )
    # BMM1 dP = V . dO^T  (A = V, M = kv; B = dO, N = q; K = d_v).  sf_a = SF_V, sf_b = SF_dO.
    idesc_bmm1_dp = prims.Tcgen05MxInstrDesc.build(
        a_dtype=STORAGE_DTYPE, b_dtype=STORAGE_DTYPE, n_dim=CFG.TILE_N, m_dim=CFG.TILE_M * CFG.CTA_MMA, k_dim=CFG.IDESC_K_DIM
    )
    bmm1_dp_desc = MmaDesc(
        M=CFG.TILE_M * CFG.CTA_MMA,
        N=CFG.TILE_N,
        K=CFG.TILE_O,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.TILE_K_HW_BMM1,
        btranspose=False,
        cta_group=CFG.CTA_MMA,
        idesc=idesc_bmm1_dp,
        kind=MMA_KIND,
        is_block_scale=True,
        sf_blocks_per_step=CFG.SF_BLOCKS_PER_STEP,
        scale_vec_size=SCALE_VEC_SIZE,
    )
    # BMM2 dV = P . dO_T  (A = e4m3 P in TMEM, K-major (a_major=0), M = kv, K = q; B = dO_T in SMEM, N = d_v, BT (b_major=1)).
    # sf_a = SF_P (the constant 2^-8), sf_b = SF_dOT (both D-planes of the full N = 256 tile; K = 128 = one atom, no column advance).
    idesc_bmm2_dv = prims.Tcgen05MxInstrDesc.build(
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=CFG.TILE_O,
        m_dim=CFG.TILE_M * CFG.CTA_MMA,
        a_major=0,
        b_major=1,
        k_dim=CFG.IDESC_K_DIM,
    )
    bmm2_dv_desc = MmaDesc(
        M=CFG.TILE_M * CFG.CTA_MMA,
        N=CFG.TILE_O,
        K=CFG.TILE_N,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.TILE_K_HW_BMM2,
        btranspose=True,
        cta_group=CFG.CTA_MMA,
        idesc=idesc_bmm2_dv,
        kind=MMA_KIND,
        is_block_scale=True,
        sf_blocks_per_step=CFG.SF_BLOCKS_PER_STEP,
        scale_vec_size=SCALE_VEC_SIZE,
    )

    # ---- SF_P: the constant byte 119 (each CTA's OWN sP_SF, filled and proxy-fenced before the init sync; the leader-issued
    #      cta_group::2 self-fill copies it into each CTA's own TMEM, the follower issues nothing).  Its descriptor is kernel-invariant;
    #      the copy itself happens per q iteration now (the alias slot alternates), right before P.dO. ----
    desc_P_SF = sP_SF[0].desc()

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)
        # ONE elect per work item for every predicated commit of this tile: the warp is converged here and the
        # predicated arrives below never diverge it (each is a branch round the native op that reconverges).
        elect_p = nvvm.elect_sync()

        # The q range that attends this kv block (uniform across the pair); the prologue handles q_lo, the loop runs
        # [q_lo, q_hi); the dV accumulate restarts at q_lo, NOT 0 (rules/frost-tile-dsl.md S2).
        kv_block_base = kv_super_idx * cutlass.Int32(_KV_BLOCK_ROWS)
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        q_lo, q_hi = _q_loop_bounds(kv_block_base, seqlen_q, seqlen_q_real, eff_seqlen_kv)

        # K + V (+ their scale-factor slabs) one-shot per kv tile; the atoms are UTCCP'd into the alias slot before EVERY MMA that
        # reads them (the slot alternates per q iteration), so only the descriptors are hoisted here.
        bars.mb_k_full[k_full_state.idx].wait(k_full_state.phase)
        bars.mb_v_full[v_full_state.idx].wait(v_full_state.phase)
        desc_K = sK[k_full_state.idx].desc()
        desc_V = sV[v_full_state.idx].desc()
        desc_K_SF = sK_SF[k_full_state.idx].desc()
        desc_V_SF = sV_SF[v_full_state.idx].desc()

        # ---- prologue: Q.K[q_lo] -> S_acc.  Its K-SF / Q-SF go to the SF slot of the tile's FIRST iteration (the slot rule: p ^ 1 of
        #      the current mb_p_ready index = the previous tile's last P slot, whose reader P.dO precedes these copies in the stream);
        #      iteration n0 refills the same slot after this MMA (V, dO for dO.V[q_lo]; K again + Q[q_lo + 1] too under
        #      S_LOOKAHEAD, the lookahead's Q.K[q_lo + 1]; at the top order the next K / Q copies are iteration q_lo + 1's). ----
        tmem_SF_K, tmem_SF_V, tmem_SF_Q, tmem_SF_dO, tmem_SF_P, tmem_SF_dOT = _sf_alias_views(tmem_P, p_ready_state.idx)
        bars.mb_s_acc_empty[s_acc_empty_state.idx].wait(s_acc_empty_state.phase, spin=SPIN_RING_WAITS)
        s_acc_empty_state = advance(s_acc_empty_state, CFG.STAGES_TMEM_S)
        bars.mb_q_full[q_full_state.idx].wait(q_full_state.phase, spin=SPIN_RING_WAITS)
        if nvvm.elect_sync():
            _utccp_sf_atoms(tmem_SF_K, desc_K_SF, _SF_ATOMS_K)
            _utccp_sf_atoms(tmem_SF_Q, sQ_SF[q_full_state.idx].desc(), _SF_ATOMS_Q)
        mma_ss(bmm1_s_desc, desc_K, sQ[q_full_state.idx].desc(), tmem_S, tmem_sf_a=tmem_SF_K, tmem_sf_b=tmem_SF_Q)
        bars.mb_s_acc_full.arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
        bars.mb_q_empty[q_full_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
        q_full_state = advance(q_full_state, CFG.STAGES_Q)

        for q_iter in cutlass.range(q_lo, q_hi, 1, unroll=1):
            # The slot rule for THIS iteration: P slot p = p_ready_state.idx (P[i] lands there), SF slot s = p ^ 1 (dead: P.dO[i-1],
            # which read P[i-1] from it, precedes every copy below in the in-order tcgen05 stream).
            p_slot = p_ready_state.idx
            tmem_SF_K, tmem_SF_V, tmem_SF_Q, tmem_SF_dO, tmem_SF_P, tmem_SF_dOT = _sf_alias_views(tmem_P, p_slot)

            # ----- the dP slot must be free (dsoftmax[i-1] loaded dP[i-1]); pre-armed for i = q_lo.  Gates dO.V[i] under both orders
            #       and, in the dense arm, the Q.K[i] issued right below it: S[i] is then never computed while dsoftmax[i-1] runs. -----
            bars.mb_dp_empty[dp_empty_state.idx].wait(dp_empty_state.phase, spin=SPIN_RING_WAITS)
            dp_empty_state = advance(dp_empty_state, CFG.STAGES_TMEM_S)

            # ----- Q.K[i] -> S_acc at the TOP of the iteration (S_LOOKAHEAD False = the dense arm; not for i = q_lo: the prologue
            #       issued it).  The SAME block as the lookahead arm's below -- the same two waits, two copies, MMA, two commits and
            #       two advances, N - 1 instances per tile either way (q_iter > q_lo <=> the lookahead's q_iter + 1 < q_hi over the
            #       same loop) -- moved from "after dO.V[i-1]" to "before dO.V[i]"; const_expr: exactly one of the two blocks traces. -----
            if cutlass.const_expr(not S_LOOKAHEAD):
                if q_iter > q_lo:
                    bars.mb_s_acc_empty[s_acc_empty_state.idx].wait(s_acc_empty_state.phase, spin=SPIN_RING_WAITS)
                    s_acc_empty_state = advance(s_acc_empty_state, CFG.STAGES_TMEM_S)
                    bars.mb_q_full[q_full_state.idx].wait(q_full_state.phase, spin=SPIN_RING_WAITS)
                    if nvvm.elect_sync():
                        _utccp_sf_atoms(tmem_SF_K, desc_K_SF, _SF_ATOMS_K)
                        _utccp_sf_atoms(tmem_SF_Q, sQ_SF[q_full_state.idx].desc(), _SF_ATOMS_Q)
                    mma_ss(bmm1_s_desc, desc_K, sQ[q_full_state.idx].desc(), tmem_S, tmem_sf_a=tmem_SF_K, tmem_sf_b=tmem_SF_Q)
                    bars.mb_s_acc_full.arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                    bars.mb_q_empty[q_full_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                    q_full_state = advance(q_full_state, CFG.STAGES_Q)

            # ----- dO.V[i] -> dP (BMM1 dP) -----
            bars.mb_do_full[do_full_state.idx].wait(do_full_state.phase, spin=SPIN_RING_WAITS)
            if nvvm.elect_sync():
                _utccp_sf_atoms(tmem_SF_V, desc_V_SF, _SF_ATOMS_V)
                _utccp_sf_atoms(tmem_SF_dO, sdO_SF[do_full_state.idx].desc(), _SF_ATOMS_dO)
            mma_ss(bmm1_dp_desc, desc_V, sdO[do_full_state.idx].desc(), tmem_dP, tmem_sf_a=tmem_SF_V, tmem_sf_b=tmem_SF_dO)
            bars.mb_dp_full.arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            bars.mb_do_empty[do_full_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            do_full_state = advance(do_full_state, CFG.STAGES_dO)

            # ----- Q.K[i+1] -> S_acc as a LOOKAHEAD between dO.V[i] and P.dO[i] (S_LOOKAHEAD True = every masked arm; not on the last
            #       iteration): the fp8 body's order -- S[i+1] is in TMEM when dsoftmax[i] releases dP[i], which the short q loops of a
            #       causal tile need (the module constant's measurement). -----
            if cutlass.const_expr(S_LOOKAHEAD):
                if (q_iter + cutlass.Int32(1)) < q_hi:
                    bars.mb_s_acc_empty[s_acc_empty_state.idx].wait(s_acc_empty_state.phase, spin=SPIN_RING_WAITS)
                    s_acc_empty_state = advance(s_acc_empty_state, CFG.STAGES_TMEM_S)
                    bars.mb_q_full[q_full_state.idx].wait(q_full_state.phase, spin=SPIN_RING_WAITS)
                    if nvvm.elect_sync():
                        _utccp_sf_atoms(tmem_SF_K, desc_K_SF, _SF_ATOMS_K)
                        _utccp_sf_atoms(tmem_SF_Q, sQ_SF[q_full_state.idx].desc(), _SF_ATOMS_Q)
                    mma_ss(bmm1_s_desc, desc_K, sQ[q_full_state.idx].desc(), tmem_S, tmem_sf_a=tmem_SF_K, tmem_sf_b=tmem_SF_Q)
                    bars.mb_s_acc_full.arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                    bars.mb_q_empty[q_full_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                    q_full_state = advance(q_full_state, CFG.STAGES_Q)

            # ----- P.dO[i] -> dV += P[i] . dO_T[i] (BMM2 dV, mma_ts; A = the TMEM P slot p the softmax filled EARLY; B = dO_T with its
            #       scale factors -- the constant P atom and the two D-plane dO_T atoms -- copied first into the SF slot s OVER the dead
            #       K / V columns: Q.K[i+1] and dO.V[i] precede these copies in the stream) -----
            bars.mb_p_ready[p_slot].wait(p_ready_state.phase, spin=SPIN_RING_WAITS)
            p_ready_state = advance(p_ready_state, CFG.STAGES_TMEM_P)
            bars.mb_dodv_full[dodv_full_state.idx].wait(dodv_full_state.phase, spin=SPIN_RING_WAITS)
            if nvvm.elect_sync():
                _utccp_sf_atoms(tmem_SF_P, desc_P_SF, _SF_ATOMS_P)
                _utccp_sf_atoms(tmem_SF_dOT, sdOT_SF[dodv_full_state.idx].desc(), _SF_ATOMS_dOT)
            mma_ts(
                bmm2_dv_desc,
                tmem_P.subview(p_slot * cutlass.Int32(LAYOUT.P_COLS)),
                sdO_dv[dodv_full_state.idx].desc(),
                tmem_dV,
                tmem_sf_a=tmem_SF_P,
                tmem_sf_b=tmem_SF_dOT,
                accumulate=(q_iter > q_lo),
            )
            bars.mb_dodv_empty[dodv_full_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            # BARRIER TABLE row mb_p_sf_consumed (producer): P.dO[i] -- and the copies before it -- are done with slot s = p(i+1) once
            # this commit fires; the softmax lanes of both CTAs wait it before storing P[i+1] there.  One elected lane, one commit
            # per target CTA (multicast) = 1 arrive per CTA per q iteration.
            bars.mb_p_sf_consumed.arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            dodv_full_state = advance(dodv_full_state, CFG.STAGES_dO_DV)

        # dV accumulation complete for this kv tile -> epilogue (compute warps).
        bars.mb_dv_ready.arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
        # The epilogue drained dV TMEM before the next tile's P.dO[0] (accumulate=False) overwrites it -- gate here.
        bars.mb_dv_acc_empty[dv_empty_state.idx].wait(dv_empty_state.phase)
        dv_empty_state = advance(dv_empty_state, 1)

        # End of tile: release K + V (the commit orders after every Q.K / dO.V of the tile that read them).
        bars.mb_k_empty[k_full_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
        bars.mb_v_empty[v_full_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
        k_full_state = advance(k_full_state, CFG.STAGES_KV)
        v_full_state = advance(v_full_state, CFG.STAGES_KV)

        # ---- scheduler tail ----
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        kv_super_idx, _, batch_idx, is_valid_tile = _decode_tile_payload(sched, sched_state.idx)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)

    # P15 (F3): the two pre-armed LEADER-scope rings each hold ONE completed phase this warp never waited (the protocol runs
    # one phase ahead); drain it so the follower's relaxed cluster arrives have landed before either CTA can exit.
    bars.mb_s_acc_empty[s_acc_empty_state.idx].wait(s_acc_empty_state.phase)
    bars.mb_dp_empty[dp_empty_state.idx].wait(dp_empty_state.phase)

    # ---- TMEM dealloc (after every compute lane of both CTAs has arrived) ----
    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


@cute.jit
def _mma_warp_quiet(tmem_ptr_i32, bars) -> None:
    """The follower CTA's MMA warp: collective tmem_alloc (the leader's collective MMAs write this CTA's TMEM half), publish
    the TMEM base to this CTA's compute warps, then wait mb_tmem_dealloc and release TMEM.  No persistent loop, no
    scheduler credit (READ_TILE_ARRIVERS_TOT counts it out)."""
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND, is_exclusive=TMEM_IS_EXCLUSIVE)
    nvvm.barrier_cta_arrive(_NAMED_BAR_TMEM_ID, _NAMED_BAR_TMEM_THREADS)
    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


# === TMA-store warp ====================================================================================================


@cute.jit
def _tmastg_warp(
    tma_dv_desc,
    tma_ds_desc,
    tma_ds_kv_desc,
    tma_sf_ds_dk_desc,
    tma_sf_ds_dq_desc,
    sdV,
    sdS,
    sdS_kv,
    sdS_SF,
    bars,
    sched,
    seq_kv_lens_tensor,
    seqlen_q,
    seqlen_kv,
    seqlen_q_real,
    cta_in_pair,
    head_base,
    n_qh_grid,
) -> None:
    """Per q iteration: store the dS ring slot -> GMEM workspace [B, H_chunk, S_kv, S_q] (mb_ds_smem_full wait -> TMA store
    -> commit -> wait(0) -> mb_ds_smem_empty).  P-b: the slot is FOUR buffers -- the ds_dk payload (tma_ds), the ds_dq payload
    (tma_ds_kv, same box), and the two 512-B F8_128x4 atoms (5-D byte descriptors, one atom per op: sf_ds_dk at (kv_tile, q_tile),
    sf_ds_dq at (q_tile, kv_tile), the head axis the workspace's CHUNK-local one) -- ONE bulk group, one wait(0), one release.
    Per kv tile: store dV -> GMEM [B, S_kv, H_q, d_v] (mb_dv_stg_full wait ->
    TMA store -> commit -> wait(0) -> mb_dv_stg_empty, which the TMALDG waits before it reloads K over the same SMEM).
    Both stores are per-CTA (this CTA's 128 kv rows of the pair's block).  The lse / delta prefetch lives on the scheduler
    warp so it runs concurrently with these stores."""
    tma_dv = GmemTileTma(tma_dv_desc)
    tma_ds = GmemTileTma(tma_ds_desc)
    tma_ds_kv = GmemTileTma(tma_ds_kv_desc) if cutlass.const_expr(_IS_P_B) else None
    tma_sf_ds_dk = GmemTileTma(tma_sf_ds_dk_desc) if cutlass.const_expr(_IS_P_B) else None
    tma_sf_ds_dq = GmemTileTma(tma_sf_ds_dq_desc) if cutlass.const_expr(_IS_P_B) else None

    kv_super_idx, head_idx, batch_idx = _boot_tile(sched)

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    ds_full_state = PipelineState.start()  # consume mb_ds_smem_full
    dv_full_state = PipelineState.start()  # consume mb_dv_stg_full
    KV_ROW_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_M)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)
        kv_block_base = kv_super_idx * cutlass.Int32(_KV_BLOCK_ROWS)
        # Only the in-range q tiles produce dS; the skipped (out-of-band) q tiles' workspace regions stay ZERO (the adapter
        # zero-initialises the workspace under a mask) so dK / dQ = dS.Q / dS^T.K are correct.
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        q_lo, q_hi = _q_loop_bounds(kv_block_base, seqlen_q, seqlen_q_real, eff_seqlen_kv)
        # P-b atom coordinates: this CTA's own 128-kv tile and the workspace's chunk-local (batch, head) index (per tile).
        kv_sf_tile = (kv_block_base + KV_ROW_OFFSET_PEER) // cutlass.Int32(SF_ATOM_ROWS)
        ds_bh = batch_idx * n_qh_grid + head_idx

        for q_iter in cutlass.range(q_lo, q_hi, 1, unroll=1):
            q_col_base = q_iter * cutlass.Int32(CFG.TILE_N)
            ds_slot = ds_full_state.idx
            bars.mb_ds_smem_full[ds_slot].wait(ds_full_state.phase, spin=SPIN_RING_WAITS)
            ds_full_state = advance(ds_full_state, CFG.XFER_STAGES)
            # Workspace [B, H_chunk, S_kv, S_q] -> coords innermost-first (S_q, S_kv, H, B); box (1, 1, TILE_M, P_D_BLOCK) =
            # 128-B rows, walked over the P_TMA_ITERS subtiles by tma_store_tile.  dS stays CHUNK-local (grid head, no head_base).
            tma_store_tile(sdS[ds_slot], tma_ds(q_col_base, kv_block_base + KV_ROW_OFFSET_PEER, head_idx, batch_idx))
            if cutlass.const_expr(_IS_P_B):
                # The ds_dq payload (same box, its own workspace) and the two atoms: coords (byte 0, row 0, inner tile, outer tile,
                # b * H_chunk + h) -- sf_ds_dk inner = q tile, outer = kv tile; sf_ds_dq the transpose.  Same bulk group as the payload.
                tma_store_tile(sdS_kv[ds_slot], tma_ds_kv(q_col_base, kv_block_base + KV_ROW_OFFSET_PEER, head_idx, batch_idx))
                q_sf_tile = q_col_base // cutlass.Int32(SF_ATOM_ROWS)
                tma_store_tile(sdS_SF[ds_slot], tma_sf_ds_dk(cutlass.Int32(0), q_sf_tile, kv_sf_tile, ds_bh, coord_0=cutlass.Int32(0)))
                tma_store_tile(
                    sdS_SF[ds_slot].shifted(_DS_SF_ATOM_DQ_OFF), tma_sf_ds_dq(cutlass.Int32(0), kv_sf_tile, q_sf_tile, ds_bh, coord_0=cutlass.Int32(0))
                )
            tma_store_commit()
            tma_store_wait(0)
            if nvvm.elect_sync():
                bars.mb_ds_smem_empty[ds_slot].arrive()

        # dV for THIS kv tile (after the epilogue) -> the FULL output [B, S_kv, H_q, d_v]: full-tensor head = head + head_base.
        bars.mb_dv_stg_full.wait(dv_full_state.phase)
        dv_full_state = advance(dv_full_state, 1)
        tma_store_tile(sdV[0], tma_dv(cutlass.Int32(0), head_idx + head_base, kv_block_base + KV_ROW_OFFSET_PEER, batch_idx))
        tma_store_commit()
        # wait_group.read: the SMEM source has been READ (not: the GMEM write has landed) -- exactly what the alias needs.
        tma_store_wait(0)
        if nvvm.elect_sync():
            bars.mb_dv_stg_empty.arrive()

        # ---- scheduler tail ----
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        kv_super_idx, head_idx, batch_idx, is_valid_tile = _decode_tile_payload(sched, sched_state.idx)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)


# === TMA-load warp =====================================================================================================


@cute.jit
def _tmaldg_warp(
    tma_q_desc,
    tma_do_desc,
    tma_do_T_desc,
    tma_k_desc,
    tma_v_desc,
    tma_q_sf_desc,
    tma_k_sf_desc,
    tma_v_sf_desc,
    tma_do_sf_desc,
    tma_do_T_sf_desc,
    sQ,
    sdO,
    sdO_dv,
    sK,
    sV,
    sQ_SF,
    sK_SF,
    sV_SF,
    sdO_SF,
    sdOT_SF,
    bars,
    sched,
    seq_kv_lens_tensor,
    seqlen_q,
    seqlen_kv,
    seqlen_q_real,
    qh_per_kh,
    n_qh,
    head_base,
    is_leader,
    cta_in_pair,
    tma_mcast_mask,
    sf_mcast_mask,
) -> None:
    """Per kv tile: K, V one-shot (M-split kv, full d) with their scale-factor slabs (this CTA's own kv tile, whole 1024-B slab,
    self-multicast).  Per q iteration: Q (BMM1 S B: TILE_N / CTA_MMA q rows x full d_qk), dO (BMM1 dP B: same N-split) and dO_T
    (BMM2 dV B: full TILE_N q rows x TILE_O / CTA_MMA d_v cols, BT -- the COLUMNWISE quantization of dO, its own e4m3 tensor, through
    the fp8 body's dV-box geometry) -- each with its SF box: Q-SF / dO-SF one 512-B atom per CTA (its d-chunk) multicast to BOTH
    CTAs so both hold the full N tile's two atoms; dO_T-SF this CTA's 512-B D-plane, multicast to both.  Every SF load rides the
    operand's ``_full`` mbarrier: the leader's expect_tx is the payload's bytes + CFG.*_SF_TX (delta 2; BARRIER TABLE).  Only the
    pair LEADER arms expect_tx (the cga2 tensor TMA routes both CTAs' bytes to its mbar, P9); both CTAs issue their own loads.

    Port fix F1: the dV epilogue staging ALIASES K + V.  The MMA's K/V release (mb_k_empty / mb_v_empty) fires when the
    epilogue has finished READING dV from TMEM (mb_dv_acc_empty), not when the TMASTG has finished reading the staged dV
    from SMEM -- so this warp additionally waits its CTA's mb_dv_stg_empty (armed after the dV store's wait_group.read)
    before the K load, or tile t+1's K could land under tile t's dV store.  Pre-armed so tile 0 passes.  At exit every
    cross-CTA multicast _empty ring is drained here (its consumer), the K/V rings included (F2)."""
    tma_q = GmemTileTma(tma_q_desc)
    tma_k = GmemTileTma(tma_k_desc)
    tma_do = GmemTileTma(tma_do_desc)
    tma_do_T = GmemTileTma(tma_do_T_desc)
    tma_v = GmemTileTma(tma_v_desc)
    tma_q_sf = GmemTileTma(tma_q_sf_desc)
    tma_k_sf = GmemTileTma(tma_k_sf_desc)
    tma_v_sf = GmemTileTma(tma_v_sf_desc)
    tma_do_sf = GmemTileTma(tma_do_sf_desc)
    tma_do_T_sf = GmemTileTma(tma_do_T_sf_desc)
    # cga2 share of the B-operand SF slabs: this CTA's atom (bytes in SMEM, 128-B rows in the descriptor) -- sf_peer_split; the
    # dO_T (columnwise) split walks D-PLANES: this CTA's plane index and its 512-B plane slot in both CTAs' slabs.
    q_sf_peer_off = cta_in_pair * cutlass.Int32(Q_SF_SPLIT.bytes_per_peer)
    q_sf_peer_row = cta_in_pair * cutlass.Int32(Q_SF_SPLIT.rows_per_peer)
    do_sf_peer_off = cta_in_pair * cutlass.Int32(dO_SF_SPLIT.bytes_per_peer)
    do_sf_peer_row = cta_in_pair * cutlass.Int32(dO_SF_SPLIT.rows_per_peer)
    doT_sf_peer_off = cta_in_pair * cutlass.Int32(_dOT_SF_PLANES_PER_PEER * SF_ATOM_BYTES)
    doT_sf_peer_plane = cta_in_pair * cutlass.Int32(_dOT_SF_PLANES_PER_PEER)

    kv_super_idx, head_idx, batch_idx = _boot_tile(sched)
    # Q / K / V / dO are the FULL [B, H, ...] tensors: index by the full-tensor head (grid head + head_base).
    full_head = cute.arch.make_warp_uniform(head_idx + head_base)
    kv_head_idx = cute.arch.make_warp_uniform(full_head // qh_per_kh)

    K_ROW_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_M)
    Q_ROW_OFFSET_PEER = cta_in_pair * cutlass.Int32(_M_PER_CTA)
    DO_DV_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_O // CFG.CTA_MMA)  # dO_dv: per-CTA d_v offset, full q rows

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    k_empty_state = PipelineState.start(phase=1)
    v_empty_state = PipelineState.start(phase=1)
    q_empty_state = PipelineState.start(phase=1)
    do_empty_state = PipelineState.start(phase=1)
    dodv_empty_state = PipelineState.start(phase=1)
    dv_stg_empty_state = PipelineState.start(phase=1)  # F1: this CTA's dV staging has left SMEM

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        kv_block_base = kv_super_idx * cutlass.Int32(_KV_BLOCK_ROWS)
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        q_lo, q_hi = _q_loop_bounds(kv_block_base, seqlen_q, seqlen_q_real, eff_seqlen_kv)
        # This CTA's own 128-row kv tile in the K / V SF tensors (rowwise atoms are per-(b, h, 128-row tile) contiguous); the
        # (batch, head) index of the columnwise dO_T SF descriptor is batch * n_qh + full head (heads inner).
        kv_sf_tile = (kv_block_base + K_ROW_OFFSET_PEER) // cutlass.Int32(SF_ATOM_ROWS)
        doT_sf_bh = batch_idx * n_qh + full_head

        # ---- K + V (+ K-SF, V-SF): one-shot per kv tile (after the previous tile's dV has left the aliased SMEM, F1) ----
        bars.mb_dv_stg_empty.wait(dv_stg_empty_state.phase)
        dv_stg_empty_state = advance(dv_stg_empty_state, 1)

        bars.mb_k_empty[k_empty_state.idx].wait(k_empty_state.phase)
        bars.mb_k_full[k_empty_state.idx].arrive(n_bytes=kFullTxBytes, pred=is_leader & nvvm.elect_sync())
        tma_load_tile(
            sK[k_empty_state.idx],
            tma_k(cutlass.Int32(0), kv_head_idx, kv_block_base + K_ROW_OFFSET_PEER, batch_idx),
            bars.mb_k_full[k_empty_state.idx].smem_ptr,
            cta_group=CFG.CTA_MMA,
            mcast_mask=tma_mcast_mask,
        )
        # K-SF: the whole slab (both d-chunk atoms) of THIS CTA's kv tile, self-multicast (the A operand is M-split: each CTA's TMEM
        # SF tile scales its own 128 rows).  A kv pad tile's atoms are READ (the descriptor spans the tensor's SKV / 128 tiles): the
        # adapter zero-fills them; the rows are P-select-dead under MASK_PADDED, so any FINITE byte there is harmless.
        tma_load_tile(
            sK_SF[k_empty_state.idx],
            tma_k_sf(cutlass.Int32(0), kv_sf_tile, kv_head_idx, batch_idx, coord_0=cutlass.Int32(0)),
            bars.mb_k_full[k_empty_state.idx].smem_ptr,
            cta_group=CFG.CTA_MMA,
            mcast_mask=tma_mcast_mask,
        )
        k_empty_state = advance(k_empty_state, CFG.STAGES_KV)

        bars.mb_v_empty[v_empty_state.idx].wait(v_empty_state.phase)
        bars.mb_v_full[v_empty_state.idx].arrive(n_bytes=vFullTxBytes, pred=is_leader & nvvm.elect_sync())
        tma_load_tile(
            sV[v_empty_state.idx],
            tma_v(cutlass.Int32(0), kv_head_idx, kv_block_base + K_ROW_OFFSET_PEER, batch_idx),
            bars.mb_v_full[v_empty_state.idx].smem_ptr,
            cta_group=CFG.CTA_MMA,
            mcast_mask=tma_mcast_mask,
        )
        # V-SF: rowwise in the backward (V contracts over d in BMM1 dP), as K.
        tma_load_tile(
            sV_SF[v_empty_state.idx],
            tma_v_sf(cutlass.Int32(0), kv_sf_tile, kv_head_idx, batch_idx, coord_0=cutlass.Int32(0)),
            bars.mb_v_full[v_empty_state.idx].smem_ptr,
            cta_group=CFG.CTA_MMA,
            mcast_mask=tma_mcast_mask,
        )
        v_empty_state = advance(v_empty_state, CFG.STAGES_KV)

        # ---- Q + dO + dO_T (each with its SF box) per q iteration (the mask-bounded range) ----
        for q_iter in cutlass.range(q_lo, q_hi, 1, unroll=1):
            q_row_base = q_iter * cutlass.Int32(CFG.TILE_N)
            q_sf_tile = q_row_base // cutlass.Int32(SF_ATOM_ROWS)  # == q_iter (TILE_N == SF_ATOM_ROWS)

            bars.mb_q_empty[q_empty_state.idx].wait(q_empty_state.phase, spin=SPIN_RING_WAITS)
            bars.mb_q_full[q_empty_state.idx].arrive(n_bytes=qFullTxBytes, pred=is_leader & nvvm.elect_sync())
            tma_load_tile(
                sQ[q_empty_state.idx],
                tma_q(cutlass.Int32(0), full_head, q_row_base + Q_ROW_OFFSET_PEER, batch_idx),
                bars.mb_q_full[q_empty_state.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )
            # Q-SF: this CTA's atom (its d-chunk) of the FULL q tile into the peer slot of BOTH CTAs (the B operand's TMEM SF tile
            # holds the whole N = 128 in each CTA).
            tma_load_tile(
                sQ_SF[q_empty_state.idx].shifted(q_sf_peer_off),
                tma_q_sf(q_sf_peer_row, q_sf_tile, full_head, batch_idx, coord_0=cutlass.Int32(0)),
                bars.mb_q_full[q_empty_state.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=sf_mcast_mask,
            )
            q_empty_state = advance(q_empty_state, CFG.STAGES_Q)

            bars.mb_do_empty[do_empty_state.idx].wait(do_empty_state.phase, spin=SPIN_RING_WAITS)
            bars.mb_do_full[do_empty_state.idx].arrive(n_bytes=dOFullTxBytes, pred=is_leader & nvvm.elect_sync())
            tma_load_tile(
                sdO[do_empty_state.idx],
                tma_do(cutlass.Int32(0), full_head, q_row_base + Q_ROW_OFFSET_PEER, batch_idx),
                bars.mb_do_full[do_empty_state.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )
            tma_load_tile(
                sdO_SF[do_empty_state.idx].shifted(do_sf_peer_off),
                tma_do_sf(do_sf_peer_row, q_sf_tile, full_head, batch_idx, coord_0=cutlass.Int32(0)),
                bars.mb_do_full[do_empty_state.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=sf_mcast_mask,
            )
            do_empty_state = advance(do_empty_state, CFG.STAGES_dO)

            bars.mb_dodv_empty[dodv_empty_state.idx].wait(dodv_empty_state.phase, spin=SPIN_RING_WAITS)
            bars.mb_dodv_full[dodv_empty_state.idx].arrive(n_bytes=dOTFullTxBytes, pred=is_leader & nvvm.elect_sync())
            # dO_T (the COLUMNWISE quantization) through the fp8 body's dV-view box: this CTA's d_v half, full q rows (BT).
            tma_load_tile(
                sdO_dv[dodv_empty_state.idx],
                tma_do_T(DO_DV_OFFSET_PEER, full_head, q_row_base, batch_idx),
                bars.mb_dodv_full[dodv_empty_state.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )
            # dO_T-SF: this CTA's D-plane (d_v half) of the q tile, columnwise atoms (D-plane-major GMEM: the plane stride grows
            # with S -- the descriptor's, not the kernel's, concern), into the plane slot of BOTH CTAs.
            tma_load_tile(
                sdOT_SF[dodv_empty_state.idx].shifted(doT_sf_peer_off),
                tma_do_T_sf(cutlass.Int32(0), doT_sf_peer_plane, q_sf_tile, doT_sf_bh, coord_0=cutlass.Int32(0)),
                bars.mb_dodv_full[dodv_empty_state.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=sf_mcast_mask,
            )
            dodv_empty_state = advance(dodv_empty_state, CFG.STAGES_dO_DV)

        # ---- scheduler tail ----
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        kv_super_idx, head_idx, batch_idx, is_valid_tile = _decode_tile_payload(sched, sched_state.idx)
        full_head = cute.arch.make_warp_uniform(head_idx + head_base)
        kv_head_idx = cute.arch.make_warp_uniform(full_head // qh_per_kh)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)

    # P15: drain every cross-CTA multicast _empty ring OUTSIDE the persistent loop, on its consumer (this warp), so the
    # leader's last commits land on a resident CTA.  Q / dO / dO_dv (3 deep each) and -- port fix F2 -- K / V (1 deep).
    if cutlass.const_expr(CFG.CTA_MMA == 2):
        for _qs in cutlass.range_constexpr(CFG.STAGES_Q):
            bars.mb_q_empty[q_empty_state.idx].wait(q_empty_state.phase)
            q_empty_state = advance(q_empty_state, CFG.STAGES_Q)
        for _ds in cutlass.range_constexpr(CFG.STAGES_dO):
            bars.mb_do_empty[do_empty_state.idx].wait(do_empty_state.phase)
            do_empty_state = advance(do_empty_state, CFG.STAGES_dO)
        for _ds in cutlass.range_constexpr(CFG.STAGES_dO_DV):
            bars.mb_dodv_empty[dodv_empty_state.idx].wait(dodv_empty_state.phase)
            dodv_empty_state = advance(dodv_empty_state, CFG.STAGES_dO_DV)
        for _ks in cutlass.range_constexpr(CFG.STAGES_KV):
            bars.mb_k_empty[k_empty_state.idx].wait(k_empty_state.phase)
            k_empty_state = advance(k_empty_state, CFG.STAGES_KV)
            bars.mb_v_empty[v_empty_state.idx].wait(v_empty_state.phase)
            v_empty_state = advance(v_empty_state, CFG.STAGES_KV)
    # The LOCAL dV-staging ring: the last tile's arrive is unconsumed (its wait guards the NEXT tile's K load).  Harmless,
    # but a symmetric 1-deep drain keeps an init-count imbalance a localizable hang.
    bars.mb_dv_stg_empty.wait(dv_stg_empty_state.phase)


# === Scheduler warp FUSED with the lse / delta prefetch =================================================================


@cute.jit
def _scheduler_stats_warp(
    sched, is_cga_first_cta, bars, sStats_raw, lse_tensor, do_dot_tensor, seq_kv_lens_tensor, dot_scale, seqlen_q, seqlen_kv, seqlen_q_real, head_base
) -> None:
    """Persistent tile scheduler (try_cancel protocol, the shape of ``tile_dsl.scheduler.scheduler_warp_loop``) fused with
    the lse / delta stats prefetch.  The compute lanes (lane = kv row) all read the SAME TILE_N q values of lse and delta
    per q tile; straight from GMEM that was 128x redundant and the dominant long-scoreboard stall, so this otherwise idle
    warp prefetches them into a STATS_STAGES-deep SMEM ring (LDG -> STS -> arrive) and the compute lanes read SMEM.

    Per loop iteration: (A) prefetch the CURRENTLY processed tile's rows -- one ring slot per q iteration, tile context
    tracked one behind the scheduling (bootstrap = blockIdx, then the decoded payload); the folds ``lse * log2e`` (NO constant
    shift: P is in TRUE units, the P scale's 2^8 is applied by the scaled cvt AFTER the exp2, never folded into the argument) and
    ``delta * attn_scale`` happen here (the host passes RAW natural-log lse and RAW TRUE-unit delta).
    (B) the try_cancel protocol for the NEXT tile: the cga-first CTA's elected lane arms expect_tx(16) on EVERY CTA's
    mb_scheduler (program-ordered before its multicast try_cancel; CTA scope suffices -- a cluster-scope release is a
    GPU drain), then every CTA's warp waits its own response.  This warp does NOT credit mb_read_tile_id."""
    lane = cute.arch.thread_idx()[0] & cutlass.Int32(31)
    _PER_LANE = CFG.TILE_N // 32  # 4 q cols per lane per q tile

    # Context of the tile the consumers are CURRENTLY processing (one behind the scheduling).
    cur_kv_super, cur_head, cur_batch = _boot_tile(sched)

    state = PipelineState.start()
    stats_empty_state = PipelineState.start(phase=1)
    is_valid = cutlass.Int32(1)

    while is_valid > cutlass.Int32(0):
        # lse / delta are the FULL [B, H_q, S_q] tensors: index by the full-tensor head.
        cur_full_head = cute.arch.make_warp_uniform(cur_head + head_base)
        # Prefetch ONLY the q tiles the consumers process, so the ring count matches their [q_lo, q_hi).
        kv_block_base = cur_kv_super * cutlass.Int32(_KV_BLOCK_ROWS)
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, cur_batch, seqlen_kv)
        q_lo, q_hi = _q_loop_bounds(kv_block_base, seqlen_q, seqlen_q_real, eff_seqlen_kv)
        # ---- (A) stats prefetch for the CURRENT tile ----
        for q_iter in cutlass.range(q_lo, q_hi, 1, unroll=1):
            q_col_base = q_iter * cutlass.Int32(CFG.TILE_N)
            slot = stats_empty_state.idx
            bars.mb_stats_empty[slot].wait(stats_empty_state.phase, spin=SPIN_RING_WAITS)
            stats_empty_state = advance(stats_empty_state, CFG.STATS_STAGES)
            slot_base = slot * cutlass.Int32(STATS_SLOT_ELEMS)
            for j in cutlass.range_constexpr(_PER_LANE):
                col = lane + cutlass.Int32(j * 32)
                lse_s = lse_tensor[cur_batch, cur_full_head, q_col_base + col] * cutlass.Float32(_LOG2E)
                dot_s = do_dot_tensor[cur_batch, cur_full_head, q_col_base + col] * dot_scale
                sStats_raw.subview(slot_base + cutlass.Int32(STATS_LSE_OFF) + col).store(lse_s)
                sStats_raw.subview(slot_base + cutlass.Int32(STATS_DOT_OFF) + col).store(dot_s)
            # Generic STS -> generic LDS on the consumer: the mbarrier orders it; the proxy fence here is harmless (carried).
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_stats_full[slot].arrive()

        # ---- (B) try_cancel protocol for the NEXT tile ----
        wait(sched.mb_read_tile_id.subview(state.idx), state.phase)
        if nvvm.elect_sync() and is_cga_first_cta:
            for i in cutlass.range_constexpr(CGA_SIZE):
                if cutlass.const_expr(i == 0):
                    arrive_expect_tx(sched.mb_scheduler.subview(state.idx), _CLC_RESPONSE_BYTES)
                else:
                    peer_mb = nvvm.mapa(sched.mb_scheduler.subview(state.idx), cutlass.Int32(i))
                    nvvm.mbarrier_arrive_expect_tx(peer_mb, _CLC_RESPONSE_BYTES, scope=nvvm.MemScope.CTA)
            nvvm.clusterlaunchcontrol_try_cancel(sched.tile_id_smem.subview(state.idx * cutlass.Int32(8)), sched.mb_scheduler.subview(state.idx), multicast=1)
        nvvm.fence_proxy("async.shared", space="cta")
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(state.idx), state.phase)
        # The NEXT tile's context becomes "current" for the next loop.
        cur_kv_super, cur_head, cur_batch, is_valid = _decode_tile_payload(sched, state.idx)
        state = advance(state, CFG.SCHEDULER_STAGES)


# === Host launcher =========================================================================================================


@cute.jit
def _host(
    q_tensor: cute.Tensor,  # [B, S_q, H_q, d_qk]   e4m3, rowwise-quantized
    k_tensor: cute.Tensor,  # [B, S_kv, H_kv, d_qk] e4m3, rowwise
    v_tensor: cute.Tensor,  # [B, S_kv, H_kv, d_v]  e4m3, rowwise (BMM1 dP contracts over d)
    do_tensor: cute.Tensor,  # [B, S_q, H_q, d_v]    e4m3, rowwise (the dP view)
    do_T_tensor: cute.Tensor,  # [B, S_q, H_q, d_v]    e4m3, the COLUMNWISE quantization of dO (the dV view), dO's BSHD-physical layout
    dv_tensor: cute.Tensor,  # out [B, S_kv, H_q, d_v] OUT dtype (per Q-head partial, TRUE units)
    ds_tensor: Optional[cute.Tensor],  # out [B, H_chunk, S_kv, S_q] bf16 workspace (P-c: the bf16 dK / dQ GEMMs' A operand); None under P-b
    lse_tensor: cute.Tensor,  # [B, H_q, S_q] fp32 natural-log LSE
    do_dot_tensor: cute.Tensor,  # [B, H_q, S_q] fp32 delta, true units
    sf_q_tensor: cute.Tensor,  # uint8 rowwise F8_128x4 atoms   [B, H_q, ceil128(S_q), 8]
    sf_k_tensor: cute.Tensor,  # uint8 rowwise                  [B, H_kv, ceil128(S_kv), 8]
    sf_v_tensor: cute.Tensor,  # uint8 rowwise                  [B, H_kv, ceil128(S_kv), 8]
    sf_do_tensor: cute.Tensor,  # uint8 rowwise                  [B, H_q, ceil128(S_q), 8]
    sf_do_T_tensor: cute.Tensor,  # uint8 COLUMNWISE, D-plane-major [B, H_q, 8, ceil128(S_q)]
    problem_size: Tuple[int, int, int, int, int, int],  # (B, QH, KH, SQ, SKV, QH_CHUNK)
    attn_scale: cutlass.Float32,
    attn_scale_log2e: cutlass.Float32,
    head_base: cutlass.Int32,
    seqlen_kv_real: cutlass.Int32,  # REAL kv length (the PADDED mask's bound); == SKV when dense.  NOT a descriptor extent.
    seqlen_q_real: cutlass.Int32,  # REAL q length (the MASK_Q_PAD band); == SQ when S_q % 128 == 0
    # --- APPENDED (P-b; None under P-c, None-specialized away): the two block-scaled dS payloads and their E8M0 atoms ---
    ds_dk_tensor: Optional[cute.Tensor] = None,  # out [B, H_chunk, S_kv, S_q] e4m3, scaled per 32-q block (the dK GEMM's A, K-major)
    ds_dq_tensor: Optional[cute.Tensor] = None,  # out [B, H_chunk, S_kv, S_q] e4m3, scaled per 32-kv block (the dQ GEMM's A, MN-major)
    sf_ds_dk_tensor: Optional[cute.Tensor] = None,  # out uint8 [B, H_chunk, S_kv/128, S_q/128, 512]: one F8_128x4 atom per (kv_tile, q_tile)
    sf_ds_dq_tensor: Optional[cute.Tensor] = None,  # out uint8 [B, H_chunk, S_q/128, S_kv/128, 512]: one atom per (q_tile, kv_tile)
    # --- APPENDED (append-only ABI): the per-batch REAL kv lengths of the PADDED arm; None-specialized (pass nothing) otherwise ---
    seq_kv_lens_tensor: Optional[cute.Tensor] = None,  # [B] int32; the PADDED arm reads seq_kv_lens[b] in place of seqlen_kv_real
    stream: _cuda_driver.CUstream = None,
) -> None:
    B, QH, KH, SQ, SKV, QH_CHUNK = problem_size

    # TMA boxes ([B, S, H, D] layout: box = (1 batch, S rows, 1 head, D cols); the workspace is [B, H, S_kv, S_q]).
    qk_box_q = (1, _M_PER_CTA, 1, TMA_QK_GRANU_ELEMS)  # Q    (BMM1 S B, q-split)
    qk_box_k = (1, CFG.TILE_M, 1, TMA_QK_GRANU_ELEMS)  # K    (M-split kv)
    do_box = (1, _M_PER_CTA, 1, TMA_VO_GRANU_ELEMS)  # dO   (BMM1 dP B, q-split)
    v_box = (1, CFG.TILE_M, 1, TMA_VO_GRANU_ELEMS)  # V    (M-split kv)
    do_dv_box = (1, CFG.TILE_N, 1, CFG.TILE_O // CFG.CTA_MMA)  # dO_T (BMM2 dV B, BT): full TILE_N q x TILE_O / CTA_MMA d_v
    dv_box = (1, CFG.TILE_M, 1, DV_D_BLOCK)  # dV store subtile: TILE_M kv x DV_D_BLOCK d_v = 128 B rows
    ds_box = (1, 1, CFG.TILE_M, P_D_BLOCK)  # dS store subtile: TILE_M kv x P_D_BLOCK q (x BPE_DS) = 128 B rows
    stride_order = (3, 2, 1, 0)

    def _tma_swz(byte_w: int):
        return tmap.TensorMapSwizzle.s128b if byte_w == 128 else tmap.TensorMapSwizzle.s64b if byte_w == 64 else tmap.TensorMapSwizzle.s32b

    tma_q_desc = tmap.create_tensor_map_tiled_from_view(
        q_tensor, box_dims=qk_box_q, stride_order=stride_order, swizzle=_tma_swz(CFG.Q_SWZ_BYTES), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
    )
    tma_do_desc = tmap.create_tensor_map_tiled_from_view(
        do_tensor, box_dims=do_box, stride_order=stride_order, swizzle=_tma_swz(CFG.dO_SWZ_BYTES), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
    )
    # The dV view reads the COLUMNWISE payload dO_T (delta 5) through the fp8 body's BT box geometry.
    tma_do_T_desc = tmap.create_tensor_map_tiled_from_view(
        do_T_tensor, box_dims=do_dv_box, stride_order=stride_order, swizzle=_tma_swz(CFG.dO_SWZ_BYTES), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
    )
    tma_k_desc = tmap.create_tensor_map_tiled_from_view(
        k_tensor, box_dims=qk_box_k, stride_order=stride_order, swizzle=_tma_swz(CFG.K_SWZ_BYTES), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
    )
    tma_v_desc = tmap.create_tensor_map_tiled_from_view(
        v_tensor, box_dims=v_box, stride_order=stride_order, swizzle=_tma_swz(CFG.V_SWZ_BYTES), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
    )
    # dV / dS stores: the compute lanes store_swizzled with the 128-B pattern, so both descriptors decode s128b (one unit).
    tma_dv_desc = tmap.create_tensor_map_tiled_from_view(
        dv_tensor, box_dims=dv_box, stride_order=stride_order, swizzle=_tma_swz(CFG.dV_SWZ_BYTES), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
    )
    # dS payload 0 = P-c's bf16 workspace | P-b's ds_dk (the same box shape at BPE_DS 1 = the fp8 body's e4m3 dS descriptor).  P-b adds
    # the ds_dq payload (its own tensor, the same box) and the two SF atom descriptors (one 512-B atom per box, sf_ds_dk inner = q
    # tile / outer = kv tile, sf_ds_dq the transpose; the head axis is the launch's chunk of QH_CHUNK heads); under P-c the three
    # appended kernel descriptors are tma_ds_desc again and never dereferenced (the arm is folded out).
    ds_payload0 = ds_dk_tensor if cutlass.const_expr(_IS_P_B) else ds_tensor
    tma_ds_desc = tmap.create_tensor_map_tiled_from_view(
        ds_payload0, box_dims=ds_box, stride_order=stride_order, swizzle=_tma_swz(CFG.dS_SWZ_BYTES), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
    )
    n_q_sf_tiles = SQ // SF_ATOM_ROWS
    n_kv_sf_tiles = SKV // SF_ATOM_ROWS
    tma_ds_kv_desc = (
        tmap.create_tensor_map_tiled_from_view(
            ds_dq_tensor, box_dims=ds_box, stride_order=stride_order, swizzle=_tma_swz(CFG.dS_SWZ_BYTES), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
        )
        if cutlass.const_expr(_IS_P_B)
        else tma_ds_desc
    )
    tma_sf_ds_dk_desc = (
        build_ds_sf_atom_desc(sf_ds_dk_tensor, inner_tiles=n_q_sf_tiles, outer_tiles=n_kv_sf_tiles, num_bh=B * QH_CHUNK, sf_atom_bytes=SF_ATOM_BYTES)
        if cutlass.const_expr(_IS_P_B)
        else tma_ds_desc
    )
    tma_sf_ds_dq_desc = (
        build_ds_sf_atom_desc(sf_ds_dq_tensor, inner_tiles=n_kv_sf_tiles, outer_tiles=n_q_sf_tiles, num_bh=B * QH_CHUNK, sf_atom_bytes=SF_ATOM_BYTES)
        if cutlass.const_expr(_IS_P_B)
        else tma_ds_desc
    )

    # Scale-factor descriptors (sdpa/kernels/_mxfp8_sf.py; delta 2).  Rowwise Q / dO: box = ONE atom (each CTA loads its d-chunk of
    # the full q tile).  Rowwise K / V: box = the whole slab of this CTA's own kv tile.  EVERY tile count below is the SF TENSOR's
    # (the ABI's padded extents SQ / SKV in 128-row atoms), never the real length's: `build_rowwise_sf_desc` derives the HEAD and
    # BATCH strides from `num_tiles`, so a count smaller than the tensor's tiles per head reads head h's atoms at (h * real_tiles)
    # instead of (h * padded_tiles) -- every head > 0 (and batch > 0) gets a neighbour's scale factors, silently, on exactly the
    # blocks whose E8M0 byte differs (MEASURED 2026-09-30 on Rubin, S_kv 800 / H_kv 2: dV cos 0.93, 12 % of dS outside; H_kv 1 clean).
    # The kv pad tile's atoms are therefore READ, not TMA-OOB: the adapter zero-fills sf_k / sf_v past S_kv_real whenever
    # S_kv_real < SKV (a NaN byte there lands NaN in the dead rows' dS: measured at S_kv 800 with poisoned kv pads), and the P-select-dead rows keep dV / dS exact 0.
    # Columnwise dO_T: D-plane-major atoms, plane stride = B * QH * tiles (it GROWS with S -- rules/mma-tma-matrix.md S7), one
    # plane per box (this CTA's d_v half).
    sq_sf_tiles = (SQ + SF_ATOM_ROWS - 1) // SF_ATOM_ROWS
    kv_sf_tiles = (SKV + SF_ATOM_ROWS - 1) // SF_ATOM_ROWS
    tma_q_sf_desc = build_rowwise_sf_desc(
        sf_q_tensor, num_tiles=sq_sf_tiles, sf_smem_size=CFG.SF_SMEM_Q, num_rows_box=Q_SF_SPLIT.rows_per_peer, num_heads=QH, num_batches=B
    )
    tma_k_sf_desc = build_rowwise_sf_desc(sf_k_tensor, num_tiles=kv_sf_tiles, sf_smem_size=CFG.SF_SMEM_K, num_rows_box=SF_ROWS_K, num_heads=KH, num_batches=B)
    tma_v_sf_desc = build_rowwise_sf_desc(sf_v_tensor, num_tiles=kv_sf_tiles, sf_smem_size=CFG.SF_SMEM_V, num_rows_box=SF_ROWS_V, num_heads=KH, num_batches=B)
    tma_do_sf_desc = build_rowwise_sf_desc(
        sf_do_tensor, num_tiles=sq_sf_tiles, sf_smem_size=CFG.SF_SMEM_dO, num_rows_box=dO_SF_SPLIT.rows_per_peer, num_heads=QH, num_batches=B
    )
    tma_do_T_sf_desc = build_columnwise_sf_desc(
        sf_do_T_tensor,
        num_tiles=sq_sf_tiles,
        num_heads=QH,
        num_batches=B,
        num_planes=_SF_ATOMS_dOT,
        planes_per_box=_dOT_SF_PLANES_PER_PEER,
        sf_bytes_per_block=SF_ATOM_BYTES,
        sf_smem_size=CFG.SF_SMEM_dOT,
        thd_varlen=False,
    )

    # Grid: one cga2 cluster per (kv block, head, batch) tile; the head axis spans QH_CHUNK heads per launch.  NATURAL =
    # 3-D grid; LPT / LPT_L2 = flat 1-D (kv_super OUTER: the heaviest causal kv blocks first).  Both ride the same persistent
    # try_cancel scheduler -- the policy only sets the launch shape and the per-tile decode.
    kv_blocks = (SKV + _KV_BLOCK_ROWS - 1) // _KV_BLOCK_ROWS
    if cutlass.const_expr(CFG.SCHEDULER_POLICY == SCHED_NATURAL):
        grid_shape = (kv_blocks * CFG.CGA_M, QH_CHUNK, B)
    else:
        grid_shape = (kv_blocks * QH_CHUNK * B * CFG.CGA_M, 1, 1)

    _kernel(
        tma_q_desc,
        tma_do_desc,
        tma_do_T_desc,
        tma_k_desc,
        tma_v_desc,
        tma_q_sf_desc,
        tma_k_sf_desc,
        tma_v_sf_desc,
        tma_do_sf_desc,
        tma_do_T_sf_desc,
        tma_dv_desc,
        tma_ds_desc,
        tma_ds_kv_desc,
        tma_sf_ds_dk_desc,
        tma_sf_ds_dq_desc,
        lse_tensor,
        do_dot_tensor,
        cutlass.Int32(SQ),
        # kernel seqlen_kv = the REAL length (drives the PADDED mask); grid / kv_blocks / descriptors stay on the padded SKV.
        seqlen_kv_real,
        seqlen_q_real,
        cutlass.Int32(B),
        cutlass.Int32(QH // KH),
        cutlass.Int32(QH),
        attn_scale,
        attn_scale_log2e,
        head_base,
        cutlass.Int32(QH_CHUNK),
        seq_kv_lens_tensor,
    ).launch(
        grid=grid_shape,
        block=[CFG.THREADS_PER_CTA, 1, 1],
        cluster=(CFG.CGA_M, CFG.CGA_N, 1),
        stream=stream,
    )


@lru_cache(maxsize=None)
def compile(  # noqa: A001
    b: int = 1,
    qh: int = 1,
    kh: int = 1,
    sq: int = 256,
    skv: int = 256,
    qh_chunk: int = 0,
) -> Callable:
    """Compile the kernel with ALL dims concrete (pins the TMA descriptor strides); see the module docstring's Launch ABI.

    ``qh`` is the FULL Q-head count (the Q / K / V / dO / dO_T / dV / lse / delta / SF descriptors span it); ``qh_chunk`` (0 = qh)
    is the per-launch head extent: the grid head axis and the dS workspace head dim.  The SF tensors are uint8 F8_128x4 atoms
    (rowwise ``[B, H, S, 8]``, columnwise dO_T ``[B, H, 8, S]``) at the PADDED S: the kernel reads them by base address + the atom
    rule, the shapes fix the rank and the byte count for the binding.  The dS operands follow ``CFG.DS_SF_POLICY`` (Launch ABI):
    P-c binds the bf16 ``ds_ws`` and None for the four appended operands; P-b binds None for ``ds_ws`` and the two e4m3 payloads
    ``[B, qh_chunk, S_kv, S_q]`` + the two atom tensors ``[B, qh_chunk, S_kv/128, S_q/128, 512]`` / ``[B, qh_chunk, S_q/128, S_kv/128, 512]``."""
    _cache_key = _template_key(globals(), locals(), "compile")
    if qh_chunk == 0:
        qh_chunk = qh
    if sq % CFG.TILE_N or skv % _KV_BLOCK_ROWS or sq <= 0 or skv <= 0:
        raise ValueError(f"{__name__}: sq must be a multiple of {CFG.TILE_N} and skv of {_KV_BLOCK_ROWS} (the adapter pads); got sq={sq}, skv={skv}")
    if kh <= 0 or qh % kh:
        raise ValueError(f"{__name__}: qh ({qh}) must be a positive multiple of kh ({kh})")
    if qh_chunk <= 0 or qh % qh_chunk:
        raise ValueError(f"{__name__}: qh_chunk ({qh_chunk}) must be a positive divisor of qh ({qh})")

    def _fake_bshd(shape, dtype):
        return cute.runtime.make_fake_compact_tensor(dtype, shape, stride_order=(3, 2, 1, 0), assumed_align=16)

    fake_q = _fake_bshd((b, sq, qh, CFG.TILE_K), STORAGE_DTYPE)
    fake_k = _fake_bshd((b, skv, kh, CFG.TILE_K), STORAGE_DTYPE)
    fake_v = _fake_bshd((b, skv, kh, CFG.TILE_O), STORAGE_DTYPE)
    fake_do = _fake_bshd((b, sq, qh, CFG.TILE_O), STORAGE_DTYPE)
    fake_do_T = _fake_bshd((b, sq, qh, CFG.TILE_O), STORAGE_DTYPE)
    fake_dv = _fake_bshd((b, skv, qh, CFG.TILE_O), OUT_STORAGE_DTYPE)

    # dS workspace(s) [B, H_chunk, S_kv, S_q]: the kv-major layout the dK = dS.Q GEMM reads un-permuted.  P-c: ONE bf16 tensor; P-b:
    # the two e4m3 payloads + the two SF atom tensors (5-D, 512 contiguous bytes per (tile, tile)); the unused operands are None.
    def _fake_sf_atoms(shape):
        return cute.runtime.make_fake_compact_tensor(cutlass.Uint8, shape, stride_order=(4, 3, 2, 1, 0), assumed_align=16)

    if _IS_P_B:
        fake_ds = None
        fake_ds_dk = _fake_bshd((b, qh_chunk, skv, sq), DS_STORAGE_DTYPE)
        fake_ds_dq = _fake_bshd((b, qh_chunk, skv, sq), DS_STORAGE_DTYPE)
        fake_sf_ds_dk = _fake_sf_atoms((b, qh_chunk, skv // SF_ATOM_ROWS, sq // SF_ATOM_ROWS, SF_ATOM_BYTES))
        fake_sf_ds_dq = _fake_sf_atoms((b, qh_chunk, sq // SF_ATOM_ROWS, skv // SF_ATOM_ROWS, SF_ATOM_BYTES))
    else:
        fake_ds = _fake_bshd((b, qh_chunk, skv, sq), DS_STORAGE_DTYPE)
        fake_ds_dk = fake_ds_dq = fake_sf_ds_dk = fake_sf_ds_dq = None
    fake_lse = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (b, qh, sq), stride_order=(2, 1, 0), assumed_align=16)
    fake_dot = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (b, qh, sq), stride_order=(2, 1, 0), assumed_align=16)
    # Scale factors: rowwise [B, H, S, d/32] and the columnwise dO_T [B, H, d/32, S] (the same byte count; the atom order is the
    # producer's -- test/python/sdpa/mxfp8_quant.py's `sf_*_swizzle`).
    fake_sf_q = _fake_bshd((b, qh, sq, _SF_D_GROUPS), cutlass.Uint8)
    fake_sf_k = _fake_bshd((b, kh, skv, _SF_D_GROUPS), cutlass.Uint8)
    fake_sf_v = _fake_bshd((b, kh, skv, _SF_D_GROUPS), cutlass.Uint8)
    fake_sf_do = _fake_bshd((b, qh, sq, _SF_D_GROUPS), cutlass.Uint8)
    fake_sf_do_T = _fake_bshd((b, qh, _SF_D_GROUPS, sq), cutlass.Uint8)
    # The per-batch kv lengths exist only on the PADDED arm; elsewhere the appended operand is None-specialized away.
    fake_seq_kv_lens = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (b,), stride_order=(0,), assumed_align=16) if CFG.SEQ_KV_LENS_PRESENT else None

    return _compile_cached(
        _host,
        fake_q,
        fake_k,
        fake_v,
        fake_do,
        fake_do_T,
        fake_dv,
        fake_ds,
        fake_lse,
        fake_dot,
        fake_sf_q,
        fake_sf_k,
        fake_sf_v,
        fake_sf_do,
        fake_sf_do_T,
        (b, qh, kh, sq, skv, qh_chunk),
        cutlass.Float32(0.0),  # attn_scale
        cutlass.Float32(0.0),  # attn_scale_log2e
        cutlass.Int32(0),  # head_base
        cutlass.Int32(skv),  # seqlen_kv_real (default = the allocated SKV; the adapter passes the real length under padding)
        cutlass.Int32(sq),  # seqlen_q_real (default = the allocated SQ; the adapter passes the real length under MASK_Q_PAD)
        fake_ds_dk,
        fake_ds_dq,
        fake_sf_ds_dk,
        fake_sf_ds_dq,
        fake_seq_kv_lens,
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
        cache_key=_cache_key,
        symbol="frost_sdpa_bwd_d256_mxfp8",
    )


def _main():
    """Minimal CLI for a standalone compile check."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--b", type=int, default=1)
    parser.add_argument("--hq", type=int, default=1)
    parser.add_argument("--hk", type=int, default=1)
    parser.add_argument("--sq", type=int, default=256)
    parser.add_argument("--skv", type=int, default=256)
    args = parser.parse_args()
    print(f"[bprop_d256_mxfp8] compile b={args.b} qh={args.hq} kh={args.hk} sq={args.sq} skv={args.skv}", flush=True)
    fn = compile(args.b, args.hq, args.hk, args.sq, args.skv)
    print(f"[bprop_d256_mxfp8] compile OK: {fn}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
