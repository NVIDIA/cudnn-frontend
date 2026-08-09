# SM120 per-tensor FP8 SDPA forward: performance, options, and tradeoffs

Engine `sdpa_fwd_prefill_sm120_fp8` (kernel
`python/cudnn/sdpa/fwd/kernels/prefill_fp8_sm120.py`) is the per-tensor FP8
(e4m3) sibling of the f16/bf16 SM120 prefill kernel, reachable through the
ordinary `graph.sdpa_fp8(...)` op with
`CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1`. This note records the measured
baseline, every design decision that trades performance against something
else, and the optimization options on the table — so the next change here
starts from data instead of archaeology.

## Measured baseline

RTX 5080 (84 SMs, SM120), CUDA 13.3, cutlass-dsl 4.7.0, bf16 f16-kernel as the
comparator, kernel time via CUPTI (min of 3 trials × 20 iters; host excluded):

| shape (d=128) | f16 kernel | fp8 kernel | speedup |
|---|---|---|---|
| b2 h24 s4096 non-causal | 3699 µs | 2129 µs | **1.74x** |
| b2 h24 s8192 non-causal | 14832 µs | 8266 µs | **1.79x** |
| b1 h32 s4096 causal | 1364 µs | 804 µs | **1.70x** |

RTX PRO 6000 Blackwell Server Edition (188 SMs, SM120, cuDNN 9.25.0.15),
against the **backend's own fp8 fprop** — the path these graphs take today —
at the tiles the shape rule picks, over 22 shapes it was not fitted on:
**1.15x–1.93x, ahead on all of them**, widest at small grids. Against the
sibling DKG SM120 fp8 kernel at matched tiles, best-vs-best: **1.05x–1.08x
behind** (it was 1.21x–1.34x before P moved out of SMEM).

Two caveats on reading any of these. The two paths overlap less than "fp8
sibling" suggests: this engine emits FP16 O and the backend's fp8 fprop
declines FP16 O, so most fp8 graphs cannot move either way. And the 1.74–1.79x
above is against the **bf16 kernel** — it answers "what does fp8 buy", not
"is this faster than what already runs".

Numerics (vs fp32 reference on dequantized inputs, scale-folded descales):
O max abs err 3e-4..3e-3 at std `1/sqrt(d)` inputs, ~1e-2 at full-scale (std
1.0) inputs — dominated by the in-kernel P→e4m3 quantization. LSE agrees to
~1e-6 (the softmax path is fp32 end to end). Amax_S matches the fp32
reference to float precision; Amax_O to P-quantization tolerance.

The mainloop is `QMMA.16832.F32.E4M3.E4M3` (half the f16 kernel's HMMA count
— the k-depth doubled), `MUFU.EX2` for the unchanged fp32 softmax,
`LDSM.8.MT1616` for the hardware 8-bit transposed V loads, and `LDSM.16.M88.4`
for K. P never reaches SMEM (see below). The amax atomics compile to one warp
`REDUX.MAX` + one global `RED` per warp — negligible.

## Design decisions and their tradeoffs

**e4m3 as Uint8 storage.** The kernel never does elementwise math on Q/K/V, so
bytes flow TMA → ldmatrix → MMA as bit patterns and no Float8 element support
is needed anywhere in the DSL plumbing (Array, TensorMap, pointer paths).
Tradeoff: the ABI is a `uint8` view (`torch` fp8 tensors pass as
`.view(torch.uint8)`), and dtype identity lives in the adapter, not the type
system.

**K loads via classic `ldmatrix.m8n8.x4.b16`.** sm_120a has no non-transposed
8-bit ldmatrix (the `m8n16 .b8` form in the ISA is the FP4 nibble-expansion
load — using it on e4m3 corrupts data silently). The b16 form is
byte-preserving and gathers the m16n8k32 B fragment exactly when each lane
points at one 16-byte K-segment. No tradeoff vs a hypothetical native
instruction; one extra issue per k32 step vs f16's per-k16 amortization.

**V loads via `ldmatrix.m16n16.x2.trans.b8`** (SASS `LDSM.8.MT1616`): the
hardware 8-bit transpose added for exactly this MMA family. One issue covers a
32(kv)×16(d-bytes) tile and feeds two MMAs. Without it, V would need a
transposed smem layout (second TMA descriptor bank conflicts) or in-register
`prmt` transposes — this instruction is why the port is clean.

**P reaches the PV MMA through shfl, not SMEM.** The QK C-fragment owns columns
`2*(t%4)+{0,1}` while the k32 A-fragment wants 4 consecutive bytes, so the f16
kernel's in-register repack does not apply. The two layouts differ only by an
exchange inside each thread quad, so `pack_f8x2_pairs` + two `shfl.sync.idx` +
one `prmt.b32` produce each A register directly from the `cvt.rn.satfinite.e4m3x2`
results left in registers by the softmax.

v1 of this kernel staged P through a per-warp `16 × kv_tile` SMEM tile instead
(32 `STS.U16` + 4 `LDSM.x4` + 2 warp barriers per warp per KV tile). Replacing
it measured **1.07–1.43x** across 24 shape × tile combinations on an RTX PRO
6000 Blackwell, largest at `kv_tile=128` where the restage traffic was largest.
Two things about that are worth recording, because the v1 notes predicted
neither:

- The kernel was **L1-bound, not issue-bound**. ncu put L1/TEX throughput at
  72–77% against the backend fp8 kernel's 45–51%, with DRAM at 5–13% on both.
  An estimate that counted instructions put the shfl route at 1.04–1.08x; the
  binding resource was the L1 port, and the measured gain was several times
  that.
- It freed 16 KB of SMEM (49 → 33 KB at 128×128), which **moved the tile
  optimum** — see below.

**Scale_S applied before the e4m3 cast.** cuDNN's `Scale_S`/`Descale_S` quantize
P — the softmax OUTPUT, not the scores: the graph applies `Scale_S` after
softmax (and after `Amax_S`) and hands `Descale_S` to bmm2. v1 of this kernel
converted P unscaled, so neither reached the math. That is exact for the
reciprocal pairs the standard contract supplies (see `test/python/sdpa/fp8.py`)
— nothing was applied, so nothing is owed back — but it silently ignored a
non-reciprocal pair, which asks for a different O. The kernel now multiplies P
by `scale_s` before the cast and folds `descale_s` into `o_scale_fused`,
matching the backend's FORT ordering (amax on the unscaled softmax result, then
scale, then cast). `tile_sum` keeps consuming the UNSCALED P so the denominator,
and hence `Amax_S`, are unaffected.

Cost: **~0.7–0.9%** at large shapes (six of six 8192 cells positive in a
0.65–0.92% band; the 30-cell geomean of 1.0089 is at the run-to-run floor and
not separately informative). It buys the non-reciprocal case and nothing else —
the predicted accuracy gain does **not** materialise. Measured on SM100 across
scale_s 1 → 64, `max|O-ref|` is flat to the digit (swa .0239 throughout),
because e4m3 is floating point, so relative precision does not move with scale,
and subtracting the row max already places P per ROW, which dominates any
per-tensor scale. A free version of the multiply exists: fold `log2(scale_s)`
into the exp2 addend, which is already an `fma2` operand, and compensate
`row_sum`/`Amax_S`/LSE once per row instead of per element.

**Why SM100 declines instead.** The d=128 SM100 FP8 kernel skips the running-max
refresh until a tile exceeds it by `RESCALE_THRESHOLD=8`, so its P is bounded by
2^8 = 256 rather than 1. e4m3 tops out at 448, so the headroom above 1.0 is
already spent on that skip, and only `scale_s <= 448/256 = 1.75` is provably
safe — empirically it survives to ~64, then swa degrades .0239 → .0807 at 448.
Since scaling buys no accuracy anyway, that row honours a reciprocal pair
analytically and declines a non-reciprocal one. This kernel has no lazy rescale,
so P <= 1 and the full range is available.

*Rejected: prefetching K or V fragments one step ahead, and loading Q coalesced
with a shfl transpose.* All three are in the sibling DKG kernel. Measured here
against the shfl-P baseline, K and V prefetch land inside ±0.5% (the run-to-run
floor at s≥2048 is ~1%), and the coalesced Q load is **reproducibly 0.5–2.3%
slower at s=512**, the shape it was supposed to help most — Q is small enough
to sit in L2, so the eight shuffles per d-fragment cost more than the saved
transactions. Absent the SMEM restage there is little ldmatrix latency left to
hide.

## Tile options

The capability row advertises `tile_ms/tile_ns ∈ {64,128}`. Since #528 the
engine proposes nothing: `sdpa/fwd/heuristics.py` names the config, listing this
cell in `_TILE_RULE_CELLS` so `_sm120_tiles` decides, and emits the runners-up
behind it for a caller that autotunes. **Every entry carries concrete knobs** —
there is no "entry 0 delegates" slot any more — and a caller pins one with
`create_execution_plan(engine_id, SdpaFwdKnobs(tile_m=..., tile_n=...))`.

The rule is shared with the f16 SM120 cell rather than fp8-specific, which is a
measurement and not a convenience: over 30 causal and non-causal shapes on a
188-SM part it gives regret **1.0046 geomean / 1.039 worst**, against the f16
rule's own 1.009 / 1.054.

Read any worst cell here with care. Most cells sit within the ~1% run-to-run
floor of each other, so one sweep's worst is often noise — an **unseeded** run
of this same code reported 1.155 at `B1xH16xS2048` causal, and the seeded
repeat shows that cell as a tie. Seed the sweep. What does survive repetition
is that the misses cluster on **causal** shapes, which is the same blind spot
the 22-shape fit below reports independently.

`kv_tile=128` in all 28 shapes measured. **This reversed when P moved out of
SMEM**: the restage tile was `(q_tile/16) x 16 x kv_tile`, so its traffic grew
with the KV tile and made 64 the better choice, by 11–25% at every size. The
f16 measurements this file used to extrapolate from said 128 was optimal only
at s>=4k; neither that nor the fp8 v1 rule survives a change to how P is
carried. A tile rule is a property of the kernel it was measured on.

`q_tile=64` while the grid cannot fill the machine *and* the sequence is long
enough to amortize the extra Q-tile loop — `grid*2 <= SMs`, or
`grid*2 <= 3*SMs` with at least 12 KV tiles. Held-out regret against the best
of the enumerated domain is 1.009x mean, 1.089x worst over 22 shapes; the
misses are causal, where the triangular mask changes the per-CTA balance in a
way grid alone does not capture.

The 1.5x-SM bound was moved in after a held-out shape at 320 CTAs missed by
1.19x while 240 CTAs was correct, so that pair is no longer independent
evidence — `test_the_grid_bound_sits_between_240_and_320_ctas` pins both
points, and a genuine re-validation needs fresh shapes.

## Bigger levers beyond this kernel

- **Multi-stage KV**: 1-byte KV halves smem per stage, and the 16 KB the P
  restage used to hold is now free, so a 2-stage sK/sV pipeline fits at
  128x128 (33 KB single-buffered today). This hides TMA latency behind
  compute, which is a different thing from the *register*-level prefetch of K
  and V fragments measured and rejected above. Largest remaining item.
- **NVFP4 / mixed-precision PV**: `MmaMXF8F6F4Op` supports mixed
  (e4m3, e2m1) operand pairs on SM120, and `LDSM.U4` (incl. the transposed
  form) hardware-unpacks fp4 from smem — a P=e4m3 × V=e2m1 PV MMA halves V
  bandwidth without quantizing P below 8 bits. That is a new kernel (SF
  channel, sub-byte addressing), not an upgrade of this one.
