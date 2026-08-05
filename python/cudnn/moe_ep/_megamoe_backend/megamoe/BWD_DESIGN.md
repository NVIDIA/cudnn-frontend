# fp8 bprop megakernel — design

Goal: replace `megamoe/fp8_bwd.py` (manual torch mxfp8 backward, 7.34 ms hot)
with a CuTe DSL backward megakernel in the mold of the forward
(`cutedsl_megamoe/moe_mxfp8_glu/megamoe_kernel_mxfp8.py`). Budget from
`bench/bench_fair_bprop` @ T=16384 H=2048 I=1024 E=8 K=2, GB200 world=1:

    fp8 bwd today: gemm 0.58 | comm-adj 1.84 | dispatch 1.33 | elemwise 1.30
                   | prep 1.21 | act-quant 1.04  = 7.34 ms
    bf16-EP bwd baseline: 3.82 ms      fused-equivalent floor: ~3.7 ms

Everything outside the GEMMs is glue the megakernel absorbs, exactly like the
forward did (0.45 vs 2.45 ms).

## The adjoint is the forward dataflow

Per dispatched copy the forward computes (tw folded into fc1 epilogue,
`apply_topk_in_fc1=True`):

    g = x@Wg, u = x@Wu   (one interleaved GEMM vs w13)
    A = tw * silu(g) * u
    y_copy = A @ W2                       out = sum_k y_copy   (TopkReduce)

Backward for the same copy, given dout at the source rank:

    dA  = tw * (dout @ W2^T)              <- GEMM1, tw via apply_topk_in_fc1
    dg  = dA * u * s(g)(1 + g(1-s(g)))    <- elemwise, s = sigmoid
    du  = dA * silu(g)                       needs RAW g,u  == fc1_c stash
    dx_copy = [du,dg] @ W13               <- GEMM2 (contraction over 2I)
    dx  = sum_k dx_copy                   <- plain topk sum == TopkReduce
    dtw = <dout, y_copy> = <dout@W2^T, silu(g)*u>   (pre-tw acc dot — no /tw)

So the backward megakernel IS the forward megakernel — dispatch(dout) ->
grouped GEMM -> elemwise -> grouped GEMM -> token-back + TopkReduce — with
swapped operands and a different elemwise. Shape walk per phase:

    forward:  H --fc1(N=2I,K=H)--> 2I --SwiGLU--> I --fc2(N=H,K=I)--> H
    backward: H --g1 (N=I, K=H)--> I --SwiGLU'--> 2I --g2 (N=H,K=2I)--> H

## Verified contracts (2026-07-19, source review)

1. **fc1_c stash is RAW pre-SwiGLU, pre-clamp, pre-tw gate+up**
   (`epilogue_mxfp8.py`: `_store_fc1_c_subtile` at L711 runs before the clamp
   at L725 and the tw multiply inside `swiglu_act` at L741). Exactly the g,u
   the elemwise needs. bf16, gate/up interleaved in 32-col blocks.
2. **fc1_c row layout**: grouped by local expert, expert e's arrivals start at
   `doff[e] = cumsum(round_up(valid_tokens, 128))` (`mega_runner.py` L609-626;
   `token_padding_block=128` when generate_c). Token order WITHIN an expert
   slot is non-deterministic in the multi-rank dispatch path (the runner's own
   validator sorts before comparing, L465).
3. **token_src_metadata**: one i64 per pool row in the persistent local
   workspace — lo32 = src_token, hi32 = (src_rank<<16)|flags|src_topk
   (`src/token_comm.py` L150). This is the exact routing record the backward
   needs to invert dispatch and to address the dx push-back.
4. **Local workspace persists across launches**: the in-kernel tail reset only
   zeros the counter prefix BEFORE `l1_token_buffer`
   (`megamoe_kernel_mxfp8.py` L477); data regions (`l1_token_buffer` = the
   dispatched quantized X pool, `l1_topk_weights_buffer`, `token_src_metadata`,
   `fc1_output`) survive until the next forward. Region byte offsets are
   host-visible (`kernel._local_offsets` / `_local_region_by_name`) so the
   wrapper can build torch views with zero kernel changes.
5. **TopkReduce is already a plain top-k sum** (weighting folded into fc1), so
   the dx combine needs NO adjoint variant — the forward token-back + reduce
   path applies verbatim to dx_copy.
6. **Weight operands already exist in host code**: `fp8_bwd.quant_weights_3d`
   produces W2^T (E,I,H) quantized along H and W13^T (E,H,2I) along 2I — same
   (data, swizzled-SF) format `megamoe/weights.py` feeds the kernel. GEMM1's B
   is layout-identical to forward fc1's B (K=H), GEMM2's B to forward fc2's B
   (K widened to 2I).

## Kernel deltas vs forward (the actual work)

D1. **Dispatch -> metadata-driven gather.** A second launch that re-runs the
    routing walk would pull dout into a DIFFERENT pool order than forward
    (contract 2) and misalign every fc1_c row. Instead the backward dispatch
    warps iterate the forward's pool rows and pull `dout[src_token]` from
    `src_rank` per `token_src_metadata` — inheriting forward's pool ordering,
    128-aligned expert offsets, and recv counts (all still in the persistent
    workspace). No routing recount, no expert_send_count exchange.
    Source side: dout is staged quantized (fp8+SF along H) by the existing
    `DataPreprocess` into a sym-heap dout pool (routing repack skipped).

D2. **GEMM1 shape**: N = I instead of 2I, B = W2^T. Same K=H mainloop.

D3. **Epilogue elemwise**: forward reads a (gate,up) interleaved acc PAIR and
    writes one 32-block of quantized A (2:1 contraction). Backward reads ONE
    32-block of dA acc + the matching (g,u) pair from fc1_c, computes (du,dg),
    requants, writes an interleaved PAIR into a 2I-wide fc1_output pool
    (1:2 expansion). The tw multiply is reused as-is (dA *= tw). This inverts
    the epilogue's hardcoded 2:1 gate/up ratio — main surgery site
    (`epilogue_mxfp8.py` subtile path + `fc1_fc2_fuse_sched.py` N/K
    bookkeeping + fc1_output/fc1_output_sf region widths).
    Concretely (source-verified): a forward subtile is one (gate,up) pair of
    EpilogueTileN=32 tmem cols each (`_subtile_local_tmem_tensor_pair`,
    subtile_col_off = subtile_idx*64; subtile_cnt = cta_tile_n/2/32), and the
    generate_c store stages a (cta_tile_m=128, 2*32=64) TMA tile per subtile
    (`_epi_tile_c`). The backward subtile inverts exactly this: 32 tmem acc
    cols (single-tensor loader replaces the pair loader) + one (128, 64)
    fc1_c TMA LOAD (same tile geometry as the C store, reversed direction,
    same SMEM stage) -> SwiGLU' -> two quant_sfd_row calls -> a 64-col
    interleaved store. fc1_output width and its SF cols double; per-warp
    subtile count doubles for the same cta_tile_n.

D4. **GEMM2 shape**: K = 2I instead of I, B = W13^T (interleaved column order
    matching the DFC1 pool layout). N = H unchanged, so token-back + combine
    staging + TopkReduce run verbatim -> dx.

D5. **Stashes for wgrad** (generate_c mechanism reused): the backward's own
    `generate_c` writes raw bf16 (du,dg) = DFC1 to a caller buffer. Together
    with fc1_c (recompute A elemwise) and the persistent l1 X pool, all wgrad
    operands are on the owner rank with no extra comm.

## dtw

`dtw = <dout, y_copy> = <GEMM1 pre-tw accumulator row, silu(g)*u row>` — all
present in the backward epilogue; a per-row fp32 dot written to a
(pool_rows,) side buffer, returned to sources with a scalar-payload a2a (tiny,
torch for v1; possibly folded into the dx token-back payload later).
v0 alternative (no kernel change): reuse forward's per-copy combine staging
(shared region, (T, K, H)) — dtw = einsum(dout, staged)/tw; needs a tw floor,
so v1's in-epilogue dot is the keeper.

## wgrad

    dW2 [h,i] = sum_m dout_pool[m,h] * (tw*A)[m,i]
    dW13[2i,h] = sum_m DFC1[m,2i]   * X_pool[m,h]

Contractions over TOKENS -> both operands need token-axis (trans)quant, a
different animal than the megakernel's row-quant world.

- **v1**: keep `fp8_bwd.gmm_wgrad_2d2d` (torch `_scaled_grouped_mm`, 2d-2d)
  fed by the kernel stashes: DFC1 (D5), A = elemwise(fc1_c), X = dequant of
  the persistent l1 pool, dout_pool = dequant of the backward's l1 pool. The
  128-row expert slots line up with GroupedPad by construction. GEMM+quant
  cost today: well under 1 ms.
  Staleness caveat: pool/stash PAD rows are zero on a fresh buffer (probe C5)
  but are NOT re-zeroed per launch — a pad row that was valid under an
  earlier routing keeps its old bytes. Token-K contractions must therefore
  zero/mask pad rows explicitly (bwd_v0 sidesteps this by rebuilding all
  wgrad operands zero-padded on the host side).
- **v2**: second small kernel (or tail phase): fused transquant + grouped
  wgrad GEMMs consuming the same pools.

## Status (2026-07-19)

- M0/M1 done (probe all-PASS; bwd_v0 `bwd_impl="pool"` parity-PASS, bwd
  5.80 ms vs fp8 7.89 / bf16-EP 3.83).
- M2.A done: `megamoe/bwd_kernel/` backward FC12 kernel — dgrad chain in one
  launch, `test_bwd_fc12` PASS (dXG rel_l2 1.6e-2, DFC1 bf16 stash 1.7e-3).
  The base kernel/scheduler/TMA paths needed zero changes; only two
  `//2`→`*2` host derivations + the epilogue inversion.
- M2.B done (`bwd_impl="mega"`, `bwd_kernel/backward.py`): compile-once
  wrapper feeding the kernel from the forward pools (zero-copy tw + fc1_c),
  uint8 gathers, wgrads on the kernel's DFC1 stash, dtw from combine
  staging. Parity PASS 1+4-rank (vs replay <= 5.3% on all grads). Bench:
  mega bwd 5.64 ms (pool 5.84, fp8 7.89, bf16-EP 3.83); best total step
  6.13 ms vs bf16-EP 7.18. Breakdown: the WHOLE dgrad chain = 0.46 ms
  in-kernel; the rest is glue — elemwise 1.81 (wgrad operand prep) + prep
  1.34 (host routing) + comm-adj 0.90 (dtw einsum + dx index_add) +
  dispatch 0.50 + act-quant 0.43.
  Gotchas hit: fp8/e8m0 tensors need uint8 views for indexing/NCCL; the
  kernel specializes activation_sf's dtype from the baked tensor — it must
  be `float8_e8m0fnu`, a uint8 buffer makes `cute.compile` die with a
  SILENT exit(1) (no traceback); `python -m` imports the package before
  `__main__` runs, so `MEGA_NO_DIST` must be set in the shell env.
- M2.C (in-kernel comm) — split into two halves; the dx token-back half is
  DONE, the dout dispatch half is next.
  - **dx token-back DONE (2026-07-20).** No new kernel code: the backward
    kernel + `epilogue_bwd.py` were copied from the forward WITH the full
    token-back machinery intact (`Fc2OutputDest`, the `fc2_in_kernel_topk_reduce`
    REDG path at `epilogue_bwd.py` ~L1038 `_red_add_relaxed_sys_v2_bf16x2`).
    Turning it on: set `fc2_in_kernel_topk_reduce=True` and pass a
    `token_comm_args` bundle (only 3 live fields — `combine_output`,
    `token_src_metadata`, `peer_rank_ptr_mapper`; everything else None,
    skipped by TokenCommArgs' MLIR serialization). The epilogue pushes each
    dx_copy row to `peer(src_rank).dx_pool[src_token, 0, :]` and REDG-adds
    the top-k copies on the fly (`reduce_topk_in_kernel` collapses src_topk→0)
    = dx. Replaces the wrapper's torch `index_add_ + all_reduce` (part of
    comm-adj 0.90 ms). Wiring is `bwd_kernel/backward.py` (sym-heap `dx_pool`
    `(T,1,H)`, peer mapper via `_compute_peer_offsets`) + a small
    `token_comm_args` build at the top of `kernel_bwd_fc12.py::__call__`
    (gated on the peer-mapper kwarg; None keeps the lean dgrad path byte-identical).
    Verified: world=1 `repro_mega` `|dx|=231.071` == baseline `231.072`
    (fp summation-order diff 1e-3); `test_bwd_fc12` still PASS (lean path
    unchanged); 4-rank `test_hybrid_training_dist` mega vs reference PASS.
    KEY FACTS for the next half: token_src_metadata is READ from the FORWARD's
    persistent local-workspace region (data region, NOT zeroed — survives),
    in the SAME pool order the backward GEMM walks (offs_padded matches), so
    `pool_token_global` in the epilogue == the forward metadata row. The
    per-expert counts (`expert_recv_count_sum`) are in the SHARED counter
    prefix which IS zeroed at the forward's kernel_tail — so the backward must
    supply counts itself (host `offs`), it cannot read them back from the
    forward workspace.
  - **dout dispatch (D1) DONE (2026-07-20).** `bwd_kernel/kernel_bwd_mega.py`:
    `BwdGatherComm` (a trimmed `TokenInPullTokenBackPush`) + `Sm100MegaMoEMxfp8-
    BwdKernel` (enable_token_comm=True, 12-warp topology, delegates the token-
    comm hooks like the forward). Per pool row the dispatch warps (8-11) read
    `token_src_metadata` → `peer_rank_ptr_mapper.map` → TMA peer-pull the
    quantized dout row + ldg the SF from a sym-heap `my_dout`/`my_dout_sf`
    into `act_pool`, then publish `fc1_ready_counter[expert_task_tile_offset +
    token_idx_in_expert // cluster_tile_tokens]` — the SAME slot the GEMM's
    TMA-B `fc1_tma_b_predispatch_spin` waits on. REUSED verbatim: the pull/store
    TMA ops, SF ldg/store, the `GpuReleaseFlagBatchTracker`, the `pull_buffer`
    SMEM, the spin + sched-ext threshold. DROPPED vs the forward: the routing
    walk (ChooseToken), dedup, `src_token_topk_idx`, `expert_send_count`,
    `dispatch_prep`/`dispatch_barrier`/NVLink barrier (my_dout is host-staged
    before the collective launch; no cross-rank sends), and the metadata WRITE.
    Per-expert counts come from host `offs` (staged into `expert_recv_count_sum`;
    the forward's are zeroed at kernel_tail). Removes the wrapper's
    `mxfp8_rowquant(dout)` + allgather + scatter (dispatch 0.50). Default ON
    (`MEGA_BWD_INKERNEL_DISPATCH=0` falls back to the torch gather) — bench:
    mega bwd beats M2.B at world=1 (4.95 vs 5.12 ms; bf16-EP 3.86) AND 4-rank
    (3.46 vs 3.50 ms, T/rank=2048; fp8 3.92). 4-rank margin is modest because
    the WGRAD path still allgathers dout+x — the next win.
    Verified: world=1 `repro_mega` |dx|=231.071 == baseline (act_pool filled
    ONLY by the in-kernel gather); 4-rank `test_hybrid_training_dist`
    `HYBRID DIST PASS` (mega dX=0.051 vs ref, real cross-rank pull) — first
    compile modulo one fix (NVSHMEM has no fp8 dtype → allocate `my_dout` via
    `_sym_zeros_byte_view_1b`, the forward's my_activation trick).
    KEY GOTCHA: `act_pool` doubles as GEMM-A AND the dispatch DEST
    (`fc1_input_token_buffer=act_pool`), with `my_dout` (T rows) the separate
    SOURCE — so no GEMM-A/pool restructure was needed. All sym-heap allocs share
    one peer delta, so the dx peer mapper covers `my_dout` pulls too.
  - Remaining M2.C — profiled priorities (mega bwd phase breakdown, D1 default,
    world=1 T=16384: sum 4.74 ms, was fp8 7.48):
    * **elemwise 1.84 ms (38.7%)** — THE target. wgrad operand prep: the
      `A = tw·silu(g)·u` recompute (fp32 over Mp×I) + `X`/`dout` dequant.
      Kill via an in-kernel **A stash** (the fc1 epilogue at
      `epilogue_bwd.py::_run_fc1_subtile` L736 already has `r_gate`,`r_up`,`topk`
      right after `swiglu_bwd` — add a 2nd generate_c-style TMA stash of
      `swiglu_act(g)·u·tw`, mirroring the DFC1 stash). Note the wgrad `dout`/`x`
      pools carry SWIZZLED SF, so `dequant_mxfp8_pool` (row-major) can't read
      them directly — the allgather-free operand path needs an SF unswizzle or
      the raw stash.
    * **prep 1.16 ms (24.5%)** — host routing (bincount + `.tolist()` sync +
      offs/metadata/`_col_atom_order`). Move on-device (dynamic-shape: Mp/doffs
      are consumed host-side for slicing, so this needs a restructure).
    * **comm-adj 0.47 ms (10%)** — in-epilogue dtw dot: accumulate
      `sum_i r_dacc[i]·silu(g[i])·u[i]` (pre-tw) per row across the fc1 subtiles
      → a (pool_rows,) side buffer; kills the torch einsum + combine-staging dep.
    * dispatch is DONE (0.20 ms, was 1.40). Each item above is a dedicated
      kernel/host milestone with GPU iteration — bench-guided, do elemwise first.

## Execution plan

- **M0 probe** (`megamoe/tests/probe_bwd_contracts.py`): validate contracts
  1-5 on GPU — fc1_c content vs dequant-pool recompute, metadata decode,
  pool persistence after launch, order (in)stability across launches.
  Wrapper accessor `megamoe/pools.py` builds the region views.
- **M1 v0 backward, no new kernel**: metadata-driven torch backward — gather
  dout by metadata (replaces dispatch+GroupedPad), reuse pools for X/g/u
  (kills prep + FC1-recompute GEMM + Y GEMM), grouped GEMMs via
  `_scaled_grouped_mm`, dx return via a2a + index_add. Validates the whole
  adjoint-by-metadata story and should already cut several ms.
- **M2 dgrad megakernel**: kernel deltas D1-D4, `bwd_kernel/` package
  subclassing/copying from `cutedsl_megamoe` (clone stays unmodified), wrapper
  `megamoe/backward.py` mirroring `forward.py` (persistent buffers, one
  compile, `load_weights_bwd` from the fp8_bwd wcache quantizers).
- **M3 wgrad v1 + dtw** around the kernel; integrate as `bwd_impl="mega"` in
  `MegaMoeHybridMxfp8Layer`; 4-rank parity vs `fp8_backward`.
- **M4 bench** vs bf16-EP 3.82 ms; then wgrad v2 / dtw-in-kernel / knob sweep.
