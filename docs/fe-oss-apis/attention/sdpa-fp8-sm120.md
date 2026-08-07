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

RTX PRO 6000 Blackwell (SM120, more SMs / bandwidth): **pending** — numbers to
be added from the workstation run; the kernel is arch-identical so ratios are
expected to transfer, absolute times to scale with SM count and clocks.

Numerics (vs fp32 reference on dequantized inputs, scale-folded descales):
O max abs err 3e-4..3e-3 at std `1/sqrt(d)` inputs, ~1e-2 at full-scale (std
1.0) inputs — dominated by the in-kernel P→e4m3 quantization. LSE agrees to
~1e-6 (the softmax path is fp32 end to end). Amax_S matches the fp32
reference to float precision; Amax_O to P-quantization tolerance.

SASS profile (d=128, both mask instances): 384× `QMMA.16832.F32.E4M3.E4M3`
(exactly half the f16 kernel's 768× HMMA — the k-depth doubled), 196×
`MUFU.EX2` (unchanged fp32 softmax), 96× `LDSM.8.MT1616.2` (hardware 8-bit
transposed V loads), 108× `LDSM.16.M88.4` (K + P-reload), 96× `STS.U16` +
96× `F2FP.SATFINITE.E4M3.F32.PACK` (P restage), 168 registers, zero spills.
The amax atomics compile to one warp `REDUX.MAX` + one global `RED` per warp —
negligible.

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

**P restage through SMEM (the main open perf item).** The QK C-fragment owns
columns `2*(t%4)+{0,1}` but the PV A-fragment wants 4 *consecutive* bytes
`4*(t%4)..+3`, so the f16 kernel's in-register P repack cannot work at k32.
v1 does correctness-first: `cvt.rn.satfinite.e4m3x2` → per-warp 16×kv_tile
smem tile at C coordinates → `bar_warp_sync` → `ldmatrix.m8n8.x4.b16` back as
A fragments. Cost per warp per KV tile: 32 `STS.U16` + 4 `LDSM.x4` + 2 warp
barriers + addressing — roughly 8–15% of dynamic instruction issue.

*Option: shfl-permute restage.* The C→A ownership mismatch is exchangeable
within each thread-quad: each A register needs two 16-bit P pairs from two
statically-known quad lanes → 2 `shfl.sync` + 1 `prmt` per A register, ~48
warp-uniform ops per tile replacing the ~36 memory ops + 2 barriers + address
math. Expected gain **~4–8% kernel time (1.04–1.08x)** — instruction-count
bound, plus an unquantified ILP benefit from removing both warp barriers
between softmax and PV. Worth doing; not a step change.

*Option: double-buffered sP.* Removes one of the two `bar_warp_sync`s
without touching the fragment math; ~half the barrier cost for 16KB more smem
(still fits). Subsumed by the shfl option.

**Amax outputs always on.** Amax_S/Amax_O are computed unconditionally
(dummy buffers when the graph doesn't request them) instead of specializing
the template on amax presence. Tradeoff: one warp-REDUX + one global RED per
warp of pure overhead in the no-amax case — measured noise — versus 2× fewer
template specializations.

**FP16 O only.** An fp8 O needs a `scale_o`-quantizing epilogue store
(`F2FP.SATFINITE` pack + byte stmatrix path). Straightforward follow-up; cut
from v1 to keep the epilogue identical to the proven f16 one. Note the fp16
epilogue already does the `o_scale_fused` multiply, so adding fp8-O is a store
-path change only.

**E4M3 only.** The MMA tag is compile-time; an e5m2 variant is one more
template axis (the fragment math is identical). Deferred until a consumer
exists — e5m2 Q/K/V for *forward* is rare.

**Exact d128.** The f16 kernel's zero-padding envelope (d_qk/d_v up to the
tile) is not wired for the 8-bit fragment path (TMA zero-fill interacts with
the byte-level A/B fragment addressing untested). d ∈ {16..256, mult of 16}
generalizes the fragment loops the same way as f16; only d128 is validated,
so the engine gates exact-match.

**No sink, no THD, no seq_len_q (v1).** Sink: Amax_S (= max 1/row_sum) is
ill-defined with a virtual sink column — needs a semantics decision, not code.
THD: the packed-LSE plumbing is untested with the fp8 epilogue. seq_len_q:
mirrors the SM100 fp8 row (no epilogue trim); `lower_dsl_prefill` already
forces it off for the fp8 family.

## Tile options

The capability row advertises `tile_ms/tile_ns ∈ {64,128}` (same knob domain
as f16). f16 measurements showed the (128,128) default is optimal at s≥4k but
loses ~11% at s=1k to (64,64); the fp8 kernel inherits the same geometry, so
the same seqlen-dependent tile choice applies once a heuristic or autotune
proposes multiple plans. Not re-measured for fp8 yet.

## Bigger levers beyond this kernel

- **K-load batching**: issue `ldmatrix` x4 across two k32 steps (halve issue
  count) — small.
- **Software pipelining / multi-stage KV**: 1-byte KV halves smem per stage;
  a 2-stage sK/sV pipeline fits where f16 could not (32KB vs 64KB per stage
  pair). Hides TMA latency behind compute; the single-buffered design shows
  up at short seqlen where tiles don't amortize. Likely the largest remaining
  item after the shfl restage.
- **NVFP4 / mixed-precision PV**: `MmaMXF8F6F4Op` supports mixed
  (e4m3, e2m1) operand pairs on SM120, and `LDSM.U4` (incl. the transposed
  form) hardware-unpacks fp4 from smem — a P=e4m3 × V=e2m1 PV MMA halves V
  bandwidth without quantizing P below 8 bits. That is a new kernel (SF
  channel, sub-byte addressing), not an upgrade of this one.
