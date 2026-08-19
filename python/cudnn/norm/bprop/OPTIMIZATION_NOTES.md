# LayerNorm/RMSNorm backward (sm_100) — optimization findings

Kernels are **cutlass primitives** (`cutlass.primitives as nvvm`), not high-level CuTe-DSL.

## Current state
Backward geomean **0.95×** vs cuDNN (bf16, Blackwell sm_100). Committed: `29fbdfd0`,
`a82f7cf2`, `de0b55ba`. Forward is 1.06×.

Structure: warp-specialized pipeline (dedicated DMA warp `cp.async.bulk`s dy+x into
double-buffered smem; compute warps register-cache dxhat/xhat, reduce, accumulate
dgamma/dbeta register partials) → separate finalize kernel reduces the `[ctas,C]`
partials (no atomics in the hot loop). Tiny-C (C≤128) uses a multi-row-tiled variant.

## What worked (the wins)
- **Occupancy-aware grid cap** (cached path): `min(occ, 2M//(NSM·C), R//(NSM·6))`. The
  finalize-traffic budget term is key — smaller C tolerates more CTAs. mixtral-8x22b
  0.89→0.99, mixtral 0.80→0.87, deepseek-2048 0.92→0.95, qwen-2048 0.98→1.00.
- **Occupancy-scaled cache gate**: cache xhat/dxhat when `R·C ≤ 24M·cached_occ` (not a
  flat 48M) — llama3-8b 0.88→0.92 while llama3-70b stays uncached.
- 3-stage pipeline gated on the cached path; single-launch dgamma+dbeta zeroing.

## What was ruled out (do not re-try without new information)
Measured on the register-limited laggards (gpt3 0.71, nemotronh 0.82, mixtral 0.87):

| approach | result |
|---|---|
| non-pipeline (direct global loads, higher occ) | worse everywhere — pipeline overlap wins |
| direct-atomic finalize (skip partials buffer) | much worse (atomic contention 148–296×) |
| gamma-direct (read γ from L2, free smem) | worse — gpt3 not occupancy-bound |
| fp16 partials | breaks fp32 tolerance; degrades dgamma |
| STAGES=1 / 4 | 2 (cached→3) is optimal |
| wn (warps/row) sweep | +0.01, within noise |
| **CGA split-row + DSMEM (the "2D-TMA" lever)** | **gpt3 0.71→0.48, nemotronh 0.82→0.69** — per-row cluster-barrier + DSMEM overhead exceeds the occupancy gain; row-splitting is not profitable at these column counts, so one CTA per row is the right structure here |
| **`min_blocks_per_mp` (launch bounds)** | compiler already max-occupancy; forcing higher **spills** (gpt3→0.44, nemotronh→0.15) |
| **`setmaxregister` (warp-spec reg realloc)** | freeing the single DMA warp (~40 regs) can't recover a CTA vs compute's ~96-reg cache across 5–9 warps; hangs. Works for GEMM (1 producer:1 consumer), not norm |

## Why the residual gap exists
cuDNN's `ln_tma_bwd_kernel.cu` is **structurally identical** to frost's cached pipeline
(same 1D `cp.async.bulk`, register-cache, register partials + finalize). Its edge on
gpt3 (N=1024, parallelism-starved) and nemotronh (large-C, register-heavy) is
micro-optimization *within* single-CTA-per-row (PDL, finalize overlap, instruction
scheduling) — not a structural lever reproducible in the DSL. **0.95× is the practical
ceiling for this kernel; the remaining sub-0.95 shapes are deferred.**
