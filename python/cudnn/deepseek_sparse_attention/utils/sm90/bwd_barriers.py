# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Copyright (c) 2026, Jerry Chen
"""SM90 named-barrier id table for the FA / DSA backward kernels.

SM90 hardware named barriers are addressed by a 4-bit operand: only ids in
[0..15] are legal for ``bar.sync $0`` / ``bar.arrive $0``.  Any id >= 16 makes
ptxas reject the PTX with::

    Barrier number 'N' out of range, expected to be in range [0..15]

Because we have more *logical* barriers than the 16 hardware slots, ids must
be packed by hand and reused across mutually-exclusive kernels.  Rule:

  * Barriers used by the **same launched kernel** must have **distinct** ids.
  * Barriers in **different kernels** may safely **share** ids.

Mutually-exclusive kernel groups in this codebase:

  A. FA backward dS / dQ warp-split kernel ........ ids  1..7
  B. bwd sparse score 2-WG kernel
     (SparseScoreRecomputeSm90, ``num_threads=256``) ..... ids  8..13
  C. bwd score 3-WG kernel
     (DenseScoreRecomputeSm90, ``num_threads=384``) ...... ids  8..15
     — REUSES ids 8..13 with group B because sparse and dense score
       kernels are compiled / launched independently.

Implemented as plain ``int`` class attributes (not ``enum.IntEnum``) so that
distinct names can deliberately share the same numeric value without the
auto-aliasing surprise that ``IntEnum`` imposes.  All call sites use
``int(NamedBarrierBwd.X)`` which keeps working unchanged.

When adding a new barrier: pick an explicit id < 16, document which kernel
group it belongs to, and verify reuse is safe (i.e. no kernel uses the same
id for two different logical barriers).
"""


class NamedBarrierBwd:
    # ---- Group A: FA bwd dS / dQ warp-split kernel (ids 1..7) -------- #
    sP_ready = 1  # WG0 -> WG1: sP R2S done
    sdS_ready = 2  # WG0 -> WG1: sdS R2S done
    sP_consumed = 3  # WG1 -> WG0: all GEMM3 read sP done (PdS_stage==1)
    sdS_consumed = 4  # WG1 -> WG0: all GEMM5/G4 read sdS done + sQ safe for epilogue
    G4_half_ready = 5  # WG1 -> WG0: WG1 G4_half committed, WG0 can issue G4_half
    Epilogue_WG0 = 6  # WG0 internal sync (epilogue_dQ R2S)
    Epilogue_WG1 = 7  # WG1 internal sync (epilogue_dQ R2S)

    # ---- Group B: bwd sparse score 2-WG kernel (ids 8..13) ----------- #
    WG0_producer_sync = 8  # WG0 internal sync — warp0 TMA KV done, all 128 threads can proceed
    KV_empty = 9  # sKV consumed — WG0 sync(256) + WG1 arrive(256), gate cp.async KV load
    WG1_consumer_sync = 10  # WG1 internal sync (softmax cross-warp reduce)
    QueryWeightsReady = 11  # after WG0 loads Q+Weights → SMEM
    KVReady = 12  # after WG0 loads KV → SMEM (prologue + pipeline)
    TileComplete = 13  # after WG1 finishes softmax + output

    # ---- Group C: bwd score 3-WG kernel (ids 8..15) ------------------ #
    # REUSES ids 8..13 with group B — sparse and dense score kernels are
    # separate launches / compile keys.  The 3-WG dense kernel does not
    # launch any Group B barrier, and vice versa, so the same underlying
    # slot carries a different semantic meaning between launches.
    PingMmaWG0_3WG = 8  # 256-thread pingpong barrier: WG0 sync / WG1 arrive
    PingMmaWG1_3WG = 9  # 256-thread pingpong barrier: WG1 sync / WG0 arrive
    DenseConsumer0Sync_3WG = 10  # WG0 internal 128-thread sync (per-iter sReduce fence)
    DenseConsumer1Sync_3WG = 11  # WG1 internal 128-thread sync (per-iter sReduce fence)
    QueryWeightsReady3WG = 12  # 384-thread: producer -> consumers, Q + Weights / LSE in SMEM
    TileComplete3WG = 13  # 384-thread: consumers -> producer, tile epilogue done
    ConsumersDone3WG = 14  # 256-thread: WG0 + WG1, partial state exchanged in sReduce
    ProducerHTileSync_3WG = 15  # 128-thread producer-WG private sync at h_tile boundary
    # (only used when num_head_tiles > 1; warp-split
    # producer otherwise races across h_tile edges)
