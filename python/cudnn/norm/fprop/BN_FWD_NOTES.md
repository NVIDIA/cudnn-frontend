# BatchNorm / InstanceNorm / GroupNorm forward (sm_100) — design & progress

Kernels are **cutlass primitives** (`cutlass.primitives as nvvm`). Goal: BN/IN/GN forward
for NCHW **and** NHWC, hitting the copy-kernel ceiling; benchmark BN/IN vs cuDNN (GN has
no cuDNN reference). Problem sizes: RN50 BN layers (`levels.label:RN50_FP8_BN_LAYERS_FULL`,
N shrinkable). Benchmark: `benchmark/norms/benchmark_bn_in.py`
(`--backend {copy,cudnn,frost} --variant {bn,in} --layout {nchw,nhwc} --n N`, GB/s at the
minimal 1R+1W, profiler device-time, run each backend in a separate process).

## Baseline (bf16, N=128)
Copy ceiling 5000-6155 GB/s. **cuDNN BN-NHWC 2200-3448 (0.44-0.56 of ceiling — even cuDNN
isn't near it: BN's NHW reduction set is too big to fully cache, so it's ~2-pass).**
cuDNN BN-NCHW is a slow generic fallback (~0.16 ceiling); IN-NCHW cuDNN is also slow — so
NHWC is BatchNorm's real layout, and cuDNN's fast path (SGBN) is NHWC-only.

## BN NHWC forward — WIP status
Two validated de-risk kernels (`benchmark/norms/bn_fwd_*_wip.py`, standalone, correct
yerr~0.03). NHWC flattens to `[M=N·H·W, C]`; reduce over M per channel.

**3-kernel (bn_fwd_3kernel_wip.py): 0.85-0.98× cuDNN, all RN50 shapes — the current best.**
stats → 2D-parallel-finalize → normalize. The wins, most impactful first:
1. partials buffer `[mparts,2,C]` (NOT atomics) for the cross-CTA stats.
2. **normalize as a channel-tile map that stages the affine (mean/rstd/γ/β) once/thread
   into regs**, then streams pixels — vs per-element global affine reads (biggest win).
3. **2D-parallel finalize** (grid=(cgrid,nchunk)+atomic) — a 1-block finalize ran the whole
   mparts reduction on 1 SM = 19µs for 600KB (same pitfall as the LN-bwd finalize).
4. **M-aware split-K**: mparts=clamp(M//64 pixels/CTA, NSM, 4·NSM).
5. **C_PER_CTA knob** (channel-tile ≠ C, cap 256 → TPP≤32, PPL≥8): whole-C-per-CTA gave
   PPL=1 (no pixel-parallelism) for C=2048 → 0.74×; CPC=256 → 0.93×.
Thread map (cuDNN's): TPP=C_PER_CTA/V threads span the contiguous C-tile (128-bit coalesced),
PPL=BT/TPP pixels in parallel; WARPS_M(cuDNN)=PIXELS_PER_LDG=our PPL. Dead ends: register-cache
/unroll of the stats loop (not ILP-bound); over-splitting small-M.

**Fused single kernel (bn_fwd_fused_wip.py): now 0.65-0.94× cuDNN (STEP 1 done).** One
cooperative kernel: stats → atomic-spin grid barrier (gpu scope, validated no deadlock) →
finalize → normalize, KC-pixel register cache.
- Original blocker: in-kernel finalize was REDUNDANT across all mparts CTAs → O(mparts²).
- **STEP 1 fix (non-redundant finalize): each CTA atomic-adds its LOCAL reduction to global
  gsum/gsq[C], grid-barrier, then all threads just READ gsum → mean/rstd. O(mparts) atomics,
  no partials buffer.** Took it 0.11-0.37× → **0.65-0.94×** (C=2048 0.94×). Correct.
- Remaining vs 3-kernel: atomic contention (mparts/channel, ~296 for small C) is the mid-shape gap.
- **STEP 3 (partials buffer, NO atomics) = the current fused (_d_bn_sp.py).**
  The cross-CTA reduce writes each CTA's partial to its OWN slot in a
  `[cblks,2,mparts,C-tile]` buffer (plain 128-bit stores, no atomics), inter-block spin-barrier,
  then a REDUNDANT finalize where every part-CTA re-reduces all mparts slots for its channel tile
  (partials L2-resident), parallelized across the PPL pixel-groups. Ported that structure verbatim
  (replacing the atomic finalize). Result at N=128: **0.68-0.99× cuDNN** (C=2048 **0.99×**, matches
  cuDNN; C=64 0.73×, C=256/56 0.68×, C=128 0.72× at occ=1). Slightly beats the atomic version and
  removes the small-C atomic contention, but small/mid C STILL lags the 3-kernel (0.85-0.98×).
- **PRIMITIVE PERF TRAP (cost me a 15-55% phantom regression):** the whole kernel is
  `cutlass.primitives` (`nvvm.atomicrmw` / `nvvm.fence_acq_rel(MemScope.GPU)` for the barrier;
  `nvvm.barrier_cta_sync_aligned(0)` for __syncthreads). Using the NON-aligned
  `nvvm.barrier_cta_sync(0)` is ~15% slower AND register-heavier — it dropped C=64 2113→1797 at the
  same occ/mparts and tipped C=256/56 occ=2→1 (→0.29×). Always the ALIGNED variant. occ is
  config-dependent → run_bn_sp tries occ=2 then falls back to occ=1 on COOPERATIVE_LAUNCH_TOO_LARGE
  (mparts is a runtime arg, so the same compiled kernel just relaunches with a smaller grid).
- **Remaining gap to cuDNN/3-kernel at small/mid C = redundant-finalize L2 traffic (~44MB for C=64,
  mparts=296) + the 2-pass re-read (KC=8 caches only ~8/42 pixels of the slice → pass2 re-reads
  ~80%). cuDNN closes both by smem-caching more of the slice WHILE budgeting smem to hold occ=2 —
  the exact smem-vs-cooperative-occupancy tension that made our naive KS cache a dead end.** The
  3-kernel (0.85-0.98×) avoids the redundant-finalize term via a separate 2D-parallel finalize.
- **STEP 4 (BT=512) = the real cuDNN lever, a broad WIN.** cuDNN uses THREADS_PER_CTA=512; that
  doubles PPL → HALVES the per-thread slice (21 vs our 42 at BT=256), so much more of it is
  register-cacheable. Just bumping BT 256→512 (KS=0) took N=128 to **0.71-0.99×** (C=128/28²
  0.72→0.89×, C=256/28² 0.78→0.92×, C=512 0.81→0.91×, C=2048 0.99×) — matching/beating the 3-kernel
  on 4/6 shapes. BT=512 forces occ=1 (reg cache blows the 2-CTA register budget), but 1×512 threads
  hide latency fine and occ=1 leaves ~238KB of the smem/L1 carveout as L1, which auto-caches re-reads.
- **STEP 4b (add smem cache on top of BT=512) = DEAD END #2 (the occ-2 smem-cache shot the user
  asked for).** Auto-sized KS to cover the slice (KS=14-25, dynamic smem 130-218KB — the >48KB
  opt-in DOES work via `.launch(smem=...)`). Result: SLOWER on EVERY shape (C=2048 0.99→0.87×,
  C=256/28² 0.92→0.87×), monotonically worse with more smem. **Why: KS=0 already leaves ~238KB as
  L1, which caches re-reads better than an explicit smem cache; a 130-218KB smem carveout STARVES
  L1, and the smem store(pass1)+load(pass2) overhead exceeds the DRAM-re-read saving.** So for OUR
  kernel structure, relying on the big L1 (KS=0) beats explicit smem caching — the opposite of
  cuDNN, whose autotuned smem/L1 balance at occ=2 is what buys its extra ~0.16 ceil on the largest-M
  (56²) shapes. **Best fused = BT=512, KS=0: 0.71-0.99× cuDNN (geomean ~0.85×).** The only shapes
  still lagging are the two largest-M 56² (M=401408) at 0.71× — the uncacheable-slice wall.
- **STEP 5 (force occ=2 at BT=512 via `min_blocks_per_mp=2`) = the large-M unlock.** The per-thread
  slice is `M·TPP/(occ·BT·NSM)` — it depends on occ×BT, not BT alone. cuDNN runs BT=512 at occ=2
  (occ×BT=1024, slice=21); we were stuck at occ=1 (BT=512 reg cache blows the 2-CTA budget → slice=42,
  double the re-read). `.launch(min_blocks_per_mp=2)` FORCES occ=2 (caps regs at 64/thread), which
  needs a SMALL reg cache to avoid spills: KC=8→spills (C64 0.60×), KC=2→0.83×, **KC=0→0.85-0.89×**
  (monotonic: the win is less spill, not more cache). Took the 56² shapes 0.71→0.85-0.89×.
- **STEP 5b (occ=2 + MODERATE smem cache) = the win that makes smem caching finally pay.** At occ=2
  the slice is 21, so a small KS covers a real fraction; and critically KS≈6 (64KB/CTA → 2×64=128KB,
  leaving ~100KB L1) does NOT starve L1 (unlike the 216KB occ=1 cache of STEP 4b). Result: C=64 56²
  0.85→**0.92×**, C=2048 **1.03× (BEATS cuDNN)**. KS=11 (106KB/CTA, ~16KB L1) regresses — L1 starve
  again. **The occ/KC/KS optimum is per-shape** (the usual autotuner axes -- occupancy, register-cache depth,
  and smem-cache depth): occ=1/KC=8 for small/mid-M (big L1), occ=2/KC=0/KS≈6 for large-M and C=2048.
- **AUTOTUNED best-per-shape (the deliverable): 0.89-1.02× cuDNN, geomean ~0.92×, BEATS cuDNN at
  C=2048 (1.02×).** cfgs: 56² & 7² → (mbpm2,KC0,KS6); 28²/C512 → (mbpm0,KC8,KS0). run_bn_sp exposes
  (mbpm,KC,KS); benchmark AUTO=1 sweeps CANDS and picks per shape. This is a single fused cooperative
  kernel matching cuDNN within ~10% on the hard shapes and beating it on the small-M shape.
- **STEP 6 (non-redundant finalize) = DEAD END.** Replaced the redundant finalize with a
  non-redundant one (only CTA my==0 reduces its tile once → global mMR → 2nd grid barrier → all read).
  Correct (yerr 0.031) but SLOWER: C=64 56² 0.92→0.72×, C=256 0.89→0.84×. **Why: for cblks=1 only ONE
  CTA finalizes while ~295 idle at barrier2, with element-wise serial L2 loads + an extra grid barrier
  — worse than the redundant finalize which is fully parallel across all part-CTAs, vectorized, and
  needs no 2nd barrier.** So the finalize is NOT the bottleneck (the 2-pass re-read is); redundant wins.
- **STEP 2 (reg+SMEM cache) = DEAD END for RN50 huge-M BN.** Added a smem tile cache (KS pixels/
  thread, `store_ext`/`load_ext` shared — those DO work) so more of the NHW slice is cached on the
  single read. Result: KS=0 (reg-only) 0.65-0.95×, KS=4 ≈ same, KS=8 worse (C2048 0.95→0.86), KS=16
  much worse (occupancy 2→1, cooperative co-residency shrinks mparts). **Why: for huge M the slice
  is ~85 pixels/thread; caching 8-24 is <30%, a marginal 2nd-read saving, while any real smem cache
  cuts cooperative occupancy — the loss dominates.** This is exactly why cuDNN also tops out at ~0.5
  of ceiling here. **Best fused config = KS=0 (register-only).**
- **STEP 1b (small-N, "does the fused kernel beat cuDNN when M is small enough to fully cache?")
  = DEAD END.** Swept N=8/16/32, KC=8 register-only (fully covers the small-N slice → true 1R+1W).
  Fused only reaches **0.09-0.30 of the (lower) copy ceiling** — *relatively worse* than N=128
  (0.30-0.42). vs cuDNN measured live at small N: fused is **0.40-0.79×** (N=8), 0.41-0.75× (N=16),
  0.44-0.80× (N=32) — worse than N=128's 0.65-0.95×. **Why: cuDNN ALSO collapses at small N (551→
  overhead-bound), but the fused kernel's fixed overheads — cooperative launch + atomic-spin grid
  barrier + atomic finalize — do NOT shrink with M, so as M drops the constant cost dominates and
  the ratio worsens.** KC=24 (192-reg cache) can't even launch: `CUDA_ERROR_COOPERATIVE_LAUNCH`
  (occupancy→<1, grid can't co-reside). Small-batch BN does not rescue the fused approach.
- **STEP 2/3 (TMEM cache tier) = DEAD END (concluded from primitives, not built).** Probed the
  Blackwell TMEM primitives: `alloc_tmem(num_columns, smem_ptr)` + `retrieve_tmem_ptr`; capacity 512
  cols × 128 lanes × 4B = 256KB/SM; only data path is `nvgpu.tcgen05` ld/st (MMA-operand lane×column
  layout). Three blockers, all fatal: (1) same cacheable-fraction wall as smem — <30% of the huge-M
  slice; (2) `is_tmem_allocation_exclusive` — TMEM is an **exclusive per-SM resource**, so allocating
  it drops cooperative occupancy exactly like the smem cache did (NOT "free" separate memory as first
  hoped); (3) tcgen05 ld/st layout is MMA-tiled, a poor fit for the per-thread (TPP-on-C × PPL-pixel)
  BN cache map — high complexity, no MMA in the pipeline. Expected payoff ≤ smem's (i.e. none).
- **CONCLUSION: the 3-kernel (0.85-0.98× cuDNN) is the overall BN fprop result.** The fused single
  kernel tops out at 0.65-0.95× (cooperative overhead + per-channel atomic contention), and NONE of
  {reg+smem cache, small-N, TMEM} beats the fundamental bandwidth-bound-with-uncacheable-reduction
  wall — the same wall that caps cuDNN at 0.44-0.56 of ceiling. Further BN gains would need a
  different axis (e.g. fusing BN into an adjacent conv/activation), not a better standalone BN kernel.

## The fast fused kernel — design (next build)
1. **Persistent + reg+SMEM cache → effective 1R+1W.** Each CTA grid-strides its NHW range,
   caching pixels into registers **and a smem tile** on the single read; normalize from that
   cache after stats (no 2nd global read). The smem tier is what makes the cached fraction
   large enough to matter. (cuDNN caches NHW across smem+regs for the same reason.)
2. **Non-redundant finalize.** After the barrier, parallelize the cross-CTA reduce across the
   tile's CTAs (each finalizes a channel subset → global mean/rstd), 2nd barrier, all read —
   the 3-kernel's 2D-parallel finalize, folded in-kernel. Kills the O(mparts²) term.
3. **THEN** TMA `cp.async.bulk` (async double-buffered load into the smem cache) + **TMEM**
   (Blackwell 512 cols, `cute.arch.alloc_tmem`) as an extra cache tier beyond smem+reg.

## Tiers (all variants)
IN-NCHW & GN-NCHW are contiguous LN rows (R=N·C M=HW; R=N·G M=Cg·HW) → route through the LN
warp+pipeline kernel + an affine-mode switch. BN both layouts = the split-K/persistent design
above. IN/GN-NHWC (C fast/strided) = net-new column-reduction. Then **modularize** the shared
pieces (staged-affine normalize, 2D finalize, partials, vec load/store, DMA pipeline) into
`_common_sm100.py` and apply across all norms.


---

# Landed in the module + the ceiling analysis (2026-08-25)

## What landed

The two WIP kernels are now real module kernels, routed from `fprop/api.py` by layout:

* `fprop/kernels/batchnorm_nhwc_sm100.py` -- the fused cooperative kernel from
  `bn_fwd_fused_wip.py`, generalised: dtype-generic (V = 16/elem_bytes, so fp32 works),
  `has_beta`, saved mean/rstd outputs, running-stat update (guarded to `my == 0`),
  NSM from device properties, and a separate non-cooperative **inference** kernel.
* `fprop/kernels/batchnorm_nchw_sm100.py` -- new. NCHW is the *easy* layout: per channel
  the elements of an image are `S = H*W` **contiguous** values, so a warp-per-row map is
  perfectly coalesced, and with `CT = WPB*CPW` channels per CTA each warp owns fixed
  channels -> per-channel accumulators live in registers and the cross-CTA reduce needs
  **no atomics at all**. Pass2's affine is a *scalar per row*, staged once.
* `api.py` no longer calls `.contiguous()` on a channels-last BatchNorm input (that was a
  hidden transpose); NHWC is viewed as `[M=N*H*W, C]` with `permute(0,2,3,1).reshape(...)`,
  a pure view.

Correct on all 7 RN50 shapes x {fp32, fp16, bf16} x {training, inference}, y/mean/rstd and
both running stats checked against PyTorch.

**bf16, N=128, GB/s at the minimal 1R+1W:**

| shape | ceiling | NCHW | x ceil | vs cuDNN | NHWC | x ceil | vs cuDNN |
|---|---|---|---|---|---|---|---|
| 64x56x56   | 5489 | 2984 | 0.54 | 3.53x | 2625 | 0.48 | 0.91x |
| 256x56x56  | 6155 | 3073 | 0.50 | 3.01x | 3048 | 0.50 | 0.88x |
| 128x28x28  | 5082 | 2493 | 0.49 | 3.08x | 1865 | 0.37 | 0.84x |
| 256x28x28  | 5546 | 2903 | 0.52 | 3.31x | 2248 | 0.41 | 0.92x |
| 512x14x14  | 5097 | 2347 | 0.46 | 2.84x | 1991 | 0.39 | 0.86x |
| 512x7x7    | 3211 |  883 | 0.28 | 2.05x |  766 | 0.24 | 0.82x |
| 2048x7x7   | 5065 | 1798 | 0.35 | 2.50x | 2137 | 0.42 | 0.95x |

NCHW went from the naive kernel's 132-1016 GB/s to 883-3073 (**2.0-3.5x cuDNN's NCHW**,
which is its slow generic fallback). NHWC reproduces the WIP result at 0.82-0.95x cuDNN
(slightly under the WIP's 0.89-1.04x because the landed kernel also writes saved mean/rstd
and updates the running stats, which the WIP did not).

## Two MLP bugs worth remembering (both cost >2x)

1. **A strip-mined loop bound must not depend on `lane`.** `kv = lane; while kv + (UF-1)*32
   < NV` lets only the first few lanes into the unrolled body -- for `NV=98, UF=4` exactly
   2 of 32 lanes took the fast path and the other 30 fell into the one-load-at-a-time tail.
   Use a uniform strip counter: `kb = 0; while kb + UF*32 <= NV`, index `kb + j*32 + lane`.
   Also size `UF <= NV//32` or the strip loop never fires at all.
2. **An unroll only creates MLP if the loads share a basic block.** The channel unroll was
   written as `for cw: while kv < NV: ...`, i.e. each channel in its own dynamic loop --
   the loads could not overlap and CPW bought nothing. Hoisting `for cw` *inside* the
   element loop (so UF*CPW loads issue back to back) is what made it work.

Also fixed: the NHWC inference kernel read `grid_dim()[0]` (= cblks) as its grid-stride
when the M-split is `.y`, so every CTA walked the whole tensor -- 21 GB/s, and *correct*,
because the redundant writes are idempotent. Idempotent bugs do not show up in a
correctness test; only the bandwidth number exposed it.

## The ceiling: what actually limits BN (measured, not assumed)

**1. Phase separation costs only 8-15%, not 40%.** A fused BN is structurally
read-everything-then-write-everything, which cannot overlap reads and writes the way a copy
kernel does. Measured with real cute kernels (flat, 128-bit, unrolled) rather than
`torch.sum` (which is *not* bandwidth-bound and understates read bandwidth by ~2x):

| size | copy (R+W) | read-only | write-only | best 2-phase | / copy |
|---|---|---|---|---|---|
| 25.7 MB | 5099 | 3710 | 5148 | 4312 | 0.85 |
| 51.4 MB | 5519 | 4359 | 6142 | 5099 | 0.92 |
| 205 MB  | 6137 | 5005 | 6220 | 5547 | 0.90 |

So **~0.85-0.92 of the copy ceiling is the real target** for a fused BN, and the read phase
is the expensive half. We are at 0.24-0.54, so the gap is ours, not the machine's.

**2. The per-row thread map is the elementwise bottleneck, not the reduction.** A pure 1R1W
pass with the row map reaches only 0.26-0.86 of ceiling; the same work with a **flat,
channel-indexed** map (grid-stride over the whole tensor, affine looked up from a smem
`[2C]` table by `channel = (elem // S) % C`) reaches **0.85-0.90** on the big shapes:

| shape | ceiling | row map | flat map |
|---|---|---|---|
| 64x56x56  | 5485 | 0.73 | **0.90** |
| 256x56x56 | 6142 | 0.86 | **0.89** |
| 128x28x28 | 5097 | 0.60 | **0.87** |
| 256x28x28 | 5527 | 0.73 | **0.85** |
| 512x14x14 | 5049 | 0.44 | **0.70** |
| 512x7x7   | 3198 | 0.33 | **0.60** |
| 2048x7x7  | 5026 | 0.44 | **0.45** |

The row map wastes lanes whenever `S/VS` is not a multiple of 32 (S=784 -> 98 vectors ->
3.06 lane steps, so one warp instruction in four is nearly empty). The flat map has no such
quantisation. The remaining 7x7 gap is the straddle path: `S=49` is not divisible by
`VS=8`, so most vectors span two channels and take the per-element lookup.

**3. TMEM is viable after all -- the earlier "dead end" call was WRONG.** Re-probed with
`cutlass.primitives`:

* `tcgen05_alloc` / `tcgen05_st` / `tcgen05_ld` (`SHAPE_32X32B`) round-trip correctly
  outside any MMA context, at 32 and 128 columns.
* **`is_exclusive=False` works**, and 2 CTAs/SM each holding 128 columns (grid 296) is
  *faster* than 1 CTA/SM. So TMEM does **not** cost occupancy -- that was the fatal
  objection in the old STEP 2/3 note and it does not hold.
* Sustained store+load: **39.5 TB/s at 1 CTA/SM, 67.3 TB/s at 2** -- an order of magnitude
  above the ~5-6 TB/s DRAM rate, so the tier is never the bottleneck.
* Capacity: 512 columns x 128 lanes x 4B = **256 KB/SM**, on top of the 228 KB smem, and it
  does *not* come out of the smem/L1 carveout. That is the key point the old note missed:
  every previous cache attempt failed because it stole L1; TMEM does not.

**Coverage with smem+TMEM = 484 KB/SM** (tensor bytes / 148 SMs), bf16 N=128:

| shape | bytes/SM | fits 484 KB? |
|---|---|---|
| 64x56x56  | 347 KB | yes (needs TMEM) |
| 256x56x56 | 1389 KB | **no** (~35%) |
| 128x28x28 | 174 KB | yes (smem alone) |
| 256x28x28 | 347 KB | yes (needs TMEM) |
| 512x14x14 | 174 KB | yes |
| 512x7x7   | 43 KB  | yes |
| 2048x7x7  | 174 KB | yes |

**6 of 7 RN50 shapes are fully cacheable on-chip**, i.e. a true single-read 1R+1W fused BN
is reachable for them. Only 256x56x56 (205 MB) overflows and stays partially 2-pass.

## TMEM tier -- built, measured (2026-08-25)

Wired into the NHWC kernel as a 4th cache tier under registers/smem (`KT` pixels per
thread, stored as **fp32 columns** -- `Vector.bitcast` preserves element COUNT, not width,
so packed-bf16 columns are not expressible, and pass2 wants fp32 anyway).

**Three traps, all of which cost real time:**

1. **`offset=` is only legal for `SHAPE_16X32BX2`.** For `SHAPE_32X32B` build a separate
   pointer per column with `make_tmem_ptr_from_warp_row_col(base, warp % 4, col, Float32)`.
2. **`tcgen05_ld/st` are warp-ALIGNED -- every lane must participate.** Guarding them with
   the per-pixel bound `if rk < r1` diverges whenever `TPP < 32` (C=64 -> TPP=8, four `tp`
   values per warp) and **HANGS the kernel**. Clamp the row and run the TMEM op
   unconditionally; predicate only the accumulate / the global store.
3. **TMEM capacity must scale with the launched occupancy.** 512 columns per SM total; two
   CTAs each asking for 512 both block in `tcgen05_alloc`, which in a cooperative grid is a
   hang, not an error. `_tmem_cap()` makes this a correctness constraint, not a knob.
   (Also: reusing the name `xv` for both an fp32 TMEM vector and a bf16 global vector trips
   `CONTAINER_STRUCTURE_CHANGED` -- the DSL treats it as loop-carried.)

**Result: the TMEM tier alone was worth only ~5-10%.** Caching the second read is NOT what
was limiting this kernel. The bigger win on the same pass was strip-mining the *remainder*
loop -- the pixels past the cache tiers were being read one load at a time, and that loop
dominates whenever the slice exceeds the cache depth. Strip-mining it (UR=4 pixels issued
before any is consumed, with a **tp-independent, warp-uniform** strip bound) took the
knob-sweep best from 0.89-1.02x cuDNN to **0.92-1.10x**.

**Final landed numbers (bf16, N=128, GB/s at the minimal 1R+1W):**

| shape | ceiling | NCHW | x ceil | vs cuDNN | NHWC | x ceil | vs cuDNN |
|---|---|---|---|---|---|---|---|
| 64x56x56   | 5527 | 3035 | 0.55 | 3.59x | 2678 | 0.48 | 0.93x |
| 256x56x56  | 6133 | 3067 | 0.50 | 3.01x | 3321 | 0.54 | 0.96x |
| 128x28x28  | 5065 | 2497 | 0.49 | 3.08x | 2118 | 0.42 | 0.96x |
| 256x28x28  | 5537 | 2907 | 0.53 | 3.31x | 2786 | 0.50 | **1.14x** |
| 512x14x14  | 5002 | 2349 | 0.47 | 2.84x | 2047 | 0.41 | 0.88x |
| 512x7x7    | 3064 |  878 | 0.29 | 2.04x |  771 | 0.25 | 0.83x |
| 2048x7x7   | 5081 | 1810 | 0.36 | 2.51x | 2292 | 0.45 | **1.02x** |

## The ceiling is still ~0.5 -- what is actually left

Both layouts plateau at 0.25-0.55 of the copy ceiling, which is where cuDNN sits too. The
measured evidence says the remaining gap is **the thread map, not the cache and not the
phase split**: a pure 1R1W pass with the *pixel/row* map reaches only 0.26-0.86, while the
same work with a **flat channel-indexed** map reaches 0.85-0.90. Neither the TMEM tier
(+5-10%) nor phase separation (-8-15%) accounts for a 2x.

## Remaining plan to the ceiling

1. **The flat channel-indexed map is the main remaining lever** (0.44 -> 0.87 measured on a
   pure 1R1W pass). The clean way to get it for *both* passes -- including the reduction --
   is a per-thread map whose **grid stride is one whole image (`S*C` elements)**: then each
   thread revisits the same position in every image, so its channel is **fixed and computed
   once**, accumulators stay in registers, 128-bit vectors and full coalescing hold for any
   `S`, and the odd-`S` straddle is also fixed per thread (two accumulators, split index
   computed once). Cross-thread combining becomes a single segmented warp reduction at the
   very END of the pass rather than per element.
2. With that map the existing smem+TMEM tiers become a true single-read cache (they are
   indexed per thread, so pass1 and pass2 line up), giving 1R+1W for the 6/7 shapes that
   fit in 484 KB/SM.
3. Only then revisit 256x56x56, which is genuinely capacity-bound and should target the
   2-phase bound (~0.9 of ceiling minus the unavoidable partial re-read).

Note the two passes must share a thread map for the cache to work, which is why swapping
only pass2 to a flat map is not sufficient -- it is the reduction map that has to change.


---

# The map change, attempted and measured (2026-08-25, session 2)

Three things were built and measured. **Two are negative results**, and they invalidate
the "remaining plan" written above -- keep that in mind when reading it.

## 1. The flat fixed-position map -- WORSE (kept as `batchnorm_nchw_flat_sm100.py`)

Built exactly as planned: each thread owns a fixed *position* inside an image and strides
by one whole image (`P = C*S`), so its channel is fixed, accumulators stay in registers,
128-bit vectors work for any `S`, and the odd-`S` straddle is a per-thread constant (two
accumulator pairs + a split index, computed once). Correct on all 7 shapes including the
7x7 straddle cases. Cross-CTA reduce is atomic-free: a CTA's window is contiguous so it
touches a contiguous run of `<= CPB` channels, partials go to `[pparts, mparts, 2, CPB]`,
and finalising a channel reads only the `<= NPX` tiles overlapping it.

**Result vs the row map (bf16, N=128): 0.59x, 1.09x, 0.80x, 0.62x, 0.73x, 0.82x, 0.95x.**
Worse on 6 of 7.

**Why the premise was wrong.** The 0.85-0.90 measured earlier for a "flat map" came from a
probe that streamed the tensor **linearly end to end** -- its speed was full contiguity,
NOT having a fixed channel. The fixed-position map buys the fixed channel by *giving up*
contiguity across `n`. Contiguous run per image visit:

* row map:  `WPB * CPW * S`  = 50 KB for 128x28x28
* flat map: `BT * VS * VPT`  =  4 KB

Under-utilisation explains only part of it (64x56x56 and 256x28x28 ran 98 CTAs on 148 SMs);
128x28x28, 512x14x14 and 2048x7x7 were at 97-99% occupancy and still lost. **The row map is
already the better-streaming structure.**

A genuinely linear map would need per-element channel dispatch, i.e. a segmented warp
reduction, and that is blocked anyway: **`nvvm.atomicrmw` on shared memory does not
compile** (internal DSL error), so segment leaders have nowhere cheap to accumulate.

## 2. TMEM in the NCHW kernel -- WORSE (built, gated off behind `_USE_TMEM = False`)

Isolated at N=128 by capping KTR: **2048x7x7 1794 -> 988 GB/s (-45%)**, 512x14x14 2365 ->
2053 (-13%), 64x56x56 neutral. Why it helps NHWC (+5-10%) but loses here:

* the traffic is far more fine-grained -- one fp32 column per *element*, and `VS=1` on odd
  `S`, so 7x7 writes a 4-byte column per 2-byte input;
* `tcgen05_ld/st` are warp-aligned, so the row loop must become a constexpr `LPT` sweep
  with clamped indices, which re-loads the tail lanes;
* enforcing the per-CTA TMEM budget needs shared memory padded to pin occupancy (below),
  and that costs L1 -- the same L1 starvation that capped the plain smem cache.

## 3. Two real bugs found on the way (both now fixed, both worth remembering)

**A cooperative launch does NOT guarantee an even CTA distribution.** It guarantees
co-residency only. A 3rd CTA can land on an SM whose 512 TMEM columns are already spoken
for; its `tcgen05_alloc` then fails and leaves a **garbage base address** -> illegal access,
not a clean error. The fix is to pad shared memory so smem is the occupancy limiter
(`smem_bytes > SMEM_PER_SM / (occ+1)`), making the per-CTA TMEM budget actually
enforceable. **This was a latent bug in the NHWC kernel too** -- it only had not fired
because its grids happened to distribute evenly.

Also: `tcgen05_alloc` requires `nCols` to be a **power of 2 in [32, 512]**, so the request
must be rounded up *and* still fit the budget after rounding.

The other bug was mine: the smem tier's loop still started at `n0` instead of `nt` after
the TMEM tier was inserted, so `r` overran `KSR` and the smem cache wrote straight through
into the TMEM pointer -- which is what produced "tensor memory not completely freed" and
the illegal accesses. A cache index that overflows into a *pointer* is indistinguishable
from a TMEM bug at the symptom level; check the smem allocator layout first.

## 4. The occupancy policy is not about caching

Choosing occupancy by *cached fraction* ("prefer whichever occ caches more") was tried and
measured worse: it pushed 64x56x56 to occ=1 to gain one extra cached image and lost
**3035 -> 2589 GB/s**. CTA count matters more than the marginal cached row. Reverted to
occ=2-first.

## Standing result

| shape | ceiling | NCHW | x ceil | vs cuDNN | NHWC | x ceil | vs cuDNN |
|---|---|---|---|---|---|---|---|
| 64x56x56   | 5522 | 3027 | 0.55 | 3.58x | 2673 | 0.48 | 0.93x |
| 256x56x56  | 6127 | 3067 | 0.50 | 3.01x | 3315 | 0.54 | 0.96x |
| 128x28x28  | 5097 | 2489 | 0.49 | 3.07x | 2121 | 0.42 | 0.96x |
| 256x28x28  | 5537 | 2903 | 0.52 | 3.31x | 2794 | 0.50 | **1.14x** |
| 512x14x14  | 5002 | 2342 | 0.47 | 2.84x | 2052 | 0.41 | 0.89x |
| 512x7x7    | 3064 |  879 | 0.29 | 2.04x |  770 | 0.25 | 0.82x |
| 2048x7x7   | 5073 | 1809 | 0.36 | 2.51x | 2290 | 0.45 | **1.01x** |

72/72 functional cases pass (both layouts x fp32/fp16/bf16 x train/infer x N in {16,128}).

## What the ~0.5 plateau now looks like

The map is not the lever, the cache tier is not the lever, and phase separation only costs
8-15%. cuDNN sits at the same 0.44-0.56. The two hypotheses left, in order of promise:

1. **Overlap the phases across channel tiles.** The grid barrier is already *per channel
   tile* (`mRet[cx]`), so tiles are independent -- but every CTA starts pass1 at the same
   moment, so the whole GPU is read-only then write-only. Giving each CTA several channel
   tiles and software-pipelining tile k's writes against tile k+1's reads would recover the
   read/write overlap a copy kernel gets. This is the only idea that attacks the measured
   8-15% *and* the serialisation, and it needs no new cache.
2. **Stop treating BN as standalone.** At 0.5 of ceiling with both cuDNN and this kernel at
   the same wall, the remaining factor of 2 plausibly is not addressable in a standalone
   BN at all; fusing into the adjacent conv/activation removes the pass entirely. The
   earlier note reached the same conclusion from a different direction.
