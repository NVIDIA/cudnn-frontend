# FROST SDPA — support matrix

Prepared scalar-output MXFP8 dense plans can specialize their declared Stats strides
at compilation and retain a generic compiled host branch for other valid runtime
Stats layouts. THD and split partials retain their existing binding contract.

What the shipped FROST SDPA engines actually serve, one table per architecture.
Columns are the kernel **flavors** (native head-dim geometry, with the model
class it was tuned for in brackets) crossed with the pass; rows are features.

Source of truth is the `Capabilities` row of each engine
(`python/cudnn/sdpa/fwd/engines.py`, `python/cudnn/sdpa/bwd/engines.py`) — a
cell here is ✅ only when that row admits it. Anything not listed as a row
(dropout, ALiBi, `block_mask`, `score_mod`, `rng_dump`,
`score_max`/`score_sum_exp`, tensor `attn_scale`, `unfuse_fma`, `Amax_S`) is
**declined by every FROST SDPA engine on every arch**.

**An explicit `attn_scale = 0`** (`zero_scale`) is a zero scale (uniform P), never the 1/sqrt(d) default.
It is served by `sdpa_fwd_prefill_sm80`, `sdpa_fwd_prefill_sm90` and the `sdpa_bwd_sm80`, `sdpa_bwd_sm100`, `sdpa_bwd_sm107`,
`sdpa_bwd_sm107_fp8`, `sdpa_bwd_sm107_mxfp8`, `sdpa_bwd_sm100_d256` and `sdpa_bwd_sm107_d512` rows. Every other row
declines it: the SM100/SM107/SM120 forward kernels fold the scale into exp2 after an unscaled, -inf-masked
running max, which a zero scale turns into NaN (#1435). SM80 and SM90 compile the scale's sign into the kernel
(SM80 `score_sign`, SM90 `scale_mode`), so they also serve negative scales under masks.

**A negative `attn_scale`** is served by the SM100/SM107/SM120 forward rows (half, FP8, MXFP8); the plan sets
`negate_scores` and the kernel runs at |scale|, so the raw-score row max still bounds P (#1435). SM100/SM107 BMM1
negates Q through the tcgen05 instruction descriptor; SM120 flips Q's sign bits as it loads Q. The per-tensor FP8
kernels also take the sign of descale_q · descale_k on the device (SM100/SM107 XOR it into the descriptor bit,
SM120 into the Q flip). An execute-time scale of the other sign is refused.

**Base-2 stats (`stats_use_log2`)** are served natively by the SM80, SM90, SM100,
SM107 and SM120 FROST forward engines: the request is a plan-time epilogue
specialization (natural-log LSE scaled by log2(e) right before the store; -inf
rows stay -inf). It is a
convention gate rather than a kernel feature — serving a base-2 request with
natural-log values would be correct O plus silently wrong Stats, so the
capability row must stay in step with the kernels. Under split-KV the per-split
partial LSEs stay natural (the combine merges them in that base) and only the
combine kernel's final LSE converts. Backward engines consume natural-log Stats
only (the graph attribute is forward-only).

`sdpa_fwd_prefill_sm100`, `sdpa_fwd_prefill_sm107`, `sdpa_fwd_prefill_sm120`, `sdpa_fwd_prefill_sm90` (f16/bf16),
`sdpa_fwd_prefill_sm100_fp8` and `sdpa_fwd_prefill_sm107_fp8` (per-tensor FP8) and `sdpa_fwd_prefill_sm107_mxfp8`
(block-scale MXFP8, exact cc 10.7) are default candidates, ranked against the backend per measured shard or qualification verdict
(`sdpa/fwd/placement.py`).
`sdpa_bwd_sm100` (f16/bf16, d in (256, 512]) is a default candidate too: the backend has no SM100
engine for that band, so it is the only provider there. Every other FROST SDPA engine is `opt_in=True`: set `CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1` before
`import cudnn` or the graph runs a cuDNN backend plan. The flag also ranks FROST first everywhere.

**Execute-time shape/stride overrides:** graphs created with
`is_override_shape_enabled=True` retain compatible prepared SM100/SM107 f16/bf16
plans: dense zero-copy layouts, split-KV with a non-overlapping final O layout,
and supported unsplit THD. Each runtime override must remain inside that plan's
compiled geometry, dtype, layout and workspace envelope. SM100/SM103 per-tensor
FP8 E4M3/E5M2 also supports prepared dense launches across its existing
D128, D192x128, D256 and D512 head envelopes, with FP16/BF16/E4M3/E5M2
output, including existing dense split-KV plans. THD retains its four native
head shapes. Its D128 flavor also prepares
existing paged KV graphs and split plans with NHD or HND pools and separate K/V
page tables. Page counts, table and pool strides bind at execution time; the
compiled pool layout and page size stay fixed.
K/V page tables on this prepared FP8 path bind independent Int64 batch/page
strides, including distinct declared layouts and runtime stride overrides.
Each table retains its own observed storage bound. Device scales rebind each call; requested
Amax_O is reset and unscaled on the launch stream. SM120/SM121 FP16/BF16 also
supports prepared dense, dense split-KV and unsplit THD launches, with native
KV-tail masking and bounded runtime geometry. Split partials retain the input
half dtype; final Stats conversion runs in the common combine. Split plans keep
the declared batch and Q length, while KV may shrink within its envelope. THD lengths and
metadata stay on device, including mixed length forms and padded Stats.
SM107 per-tensor FP8 at exact device cc 10.7 also uses prepared dense and native-shape THD launches
for its four existing flavors, plus its existing D128 split-KV path. Int64
strides reach descriptor setup; D256 keeps its optional Amax specialization.
The row's wider cc range does not admit prepared FP8 plans on cc 10.8–11.9;
the standalone adapter does not support those devices.
The existing D256 quantized epilogue gate also uses prepared launches, with
runtime BF16 gate pointers and Int64 strides. Amax remains the ungated SDPA
output's maximum. Existing conversion routes use prepared gather/scatter copies
around a compact prepared attention plan.
Fixed dense D128 block-scaled outputs use the prepared hosts described below.

SM120/SM121 per-tensor FP8 also supports prepared dense, dense split-KV and THD
across its general and D512 head envelopes, with device scales, all four scalar
output dtypes and native KV-tail masking. THD retains per-batch length inputs;
CU-prefix-sum graph inputs remain unsupported on this FP8 row.
SM100/SM103 MXFP8 uses prepared launches for scalar outputs with fixed dense or bounded THD geometry.
SM120/SM121 MXFP8 uses prepared launches for scalar outputs with fixed dense (unsplit and split) or THD geometry,
and for the block-scaled MXFP8 O on fixed dense unsplit plans;
dense conversion layouts use staged copies around a prepared compact plan, with the scale tensors bound unchanged.
SM107 MXFP8 at exact device cc 10.7 uses prepared launches for its four existing dense, unsplit native
head shapes and for bounded THD at d256 (`config_sm107.SM107_MXFP8_THD_SHAPES`); THD at the other
three shapes, split-KV and PackGQA remain unsupported. Dense MXFP8 runtime
shape overrides remain declined because SF batch/head pitches are plan-fixed.
SM107 D256 MXFP8 gates also use the prepared host. The existing direct-only SM100
D128/D192 PV-BF16 specialization prepares native layouts with BF16 V and no SF_V
operand; graph eligibility is unchanged. Conversion layouts and bias continue to
decline runtime shape/stride overrides; a mask-free `S_kv` off the KV tile takes them
through the KV-tail mask (#1425). Existing
conversion layouts use staged copies around a prepared compact plan;
explicit opt-in does not bypass the contract. The same pure capability predicate
filters candidate knobs and selects the prepared executor. Static-geometry graph
eligibility is unchanged.

### Explicit scalar KV-tail split

SM100/SM103/SM107 exact D128/V128 FP16/BF16 and per-tensor FP8 dense
forward plans can explicitly combine split-KV with a mask-free, non-tile-aligned
KV extent. The existing scalar-tail mask bounds each split; empty partitions
emit the combine identity. No per-batch length tensor or execute-time staging
is introduced. Native shape overrides may shrink KV within the declaration,
with the requested O, Stats and FP8 Amax contracts preserved. CGA domains are
unchanged (FP8 split uses CGA2). Other dimensions, MXFP8, dense padding and
sink combinations retain their existing restrictions. Automatic scalar-tail
selection remains unsplit pending independent tuning.

> **Keeping this current is a hard rule.** A change to any FROST SDPA
> `Capabilities` row, or adding/retiring an `EngineSpec`, updates this file in
> the same commit — see `python/cudnn/sdpa/AGENTS.md` **Rule S2** and
> `python/cudnn/frost/README.md` § "The rules" #14. This tracker is maintained
> by hand; nothing else catches it going stale. It lives beside the engines it
> tracks (`../fwd/engines.py`, `../bwd/engines.py`) rather than under `docs/`
> precisely so that "in the same commit" is the path of least resistance.

Legend: ✅ served natively · ⚠️ served, but **no native kernel for this head dim** —
the graph rides another flavor's envelope with TMA zero-padding (correct, but pays
the larger flavor's MMA cost) · ❔ **accepted by the capability row but not
validated on this path** — treat as untested, not as a guarantee · ❌ declined at
plan time · — not applicable · ⁿ footnote.

---

## SM90 (Hopper, cc 9.0 exactly)

Engine: `sdpa_fwd_prefill_sm90`, public ID **20517**, manifest slot 17 (default candidate,
ranked against the backend by `placement._place_sm90_f16`).
Dense and THD execute through the shared prepared pointer binder; graph shape/stride
overrides remain unsupported. The scheduler keeps its declared batch and dense
extents; THD workspace retains 128-byte tensor-map alignment. No plan-owned dummy
tensors or tensor-launch fallback.
Forward only; optional Stats do not provide a backward implementation. Q/K/V/O
all use FP16 or all BF16. The native head tile is **(512, 512)**. The row's
envelope is floored at 256 (`d_envelope_floors`): `D_QK` and `D_V`
independently accept positive multiples of 8 in **(256, 512]**, and the exact
native hit always passes. Pairs inside the envelope are **envelope-served at
the full D512 compute cost**, with TMA zero-fill and output clipping, so the
floor caps that at 2x padding.

The floor is a **routing** change, not only a claim change: SM90 is the only
cc-9.0 row, so an f16/bf16 Hopper graph with either head dim at or below 256
now gets no FROST plan at all and falls back to the cuDNN backend. Unfloored,
this row led the preference order and took such graphs at up to 64x
zero-padding. Deliberate asymmetry: `config_sm90.head_dims_mismatch` still
serves every multiple of 8 in `(0, 512]`, so a direct standalone-adapter caller
keeps the full envelope — the row is narrower than the adapter, which is the
safe direction.

H200 validation (2026-09-20, on `b100e342`; DSL 4.8.0.dev0, cuDNN 9.24.0,
torch 2.14.0+cu130). The SM90 L0/L1 conformance suites — 21 files, not yet in
this tree, run from the development checkout against this package — gave 1115
passed, 2 skipped (both the intentional protocol-fault diagnostic) and 1 failed
before the sink-at-decode refusal was dropped; seven of those files drive the
graph API, so the run covers `mismatch`, the lowering and the binding. After
the drop, the two files that pin refusals were re-run: 394 passed, 3 failed.
Every failure, before or after, is a test pinning the superseded
sink-at-decode behaviour, to be inverted when that suite lands here:
`test_shared_contract_validator_refusals[sink]` (the validator rule #1095
removed), `test_shared_contract_native_refusal_table[sink]` and
`test_standalone_wrapper_refusals[sq1-sink]` (the SM90 refusal). The sink at
`S_q == 1` itself: `test_sdpa_fwd_dsl_sm90.py` — landing here, repinning
`test_sdpa_fwd_dsl_sm100.py`'s `_ARCH` to run that suite at SM90's head dim plus
the SM90-only half (envelope floor, adapter declines, template, THD arm,
wrapper) — was RED with the refusal (five typed declines at build, under a
strict pin) and is green without it; a one-off FP64 probes matched 61/61 cases
(fp16/bf16, MHA/GQA/MQA, `S_kv` 1/63/64/65/129, unmasked / top-left /
bottom-right / left window / right band
/ both bounds, zero-KV and trimmed-Q batches, PackGQA with sinks ±20 / 0 /
−inf, Stats natural / base-2 / omitted, head dims (192, 128), and a THD
envelope of one in the sequence- and cumulative-length forms with both packed
Stats layouts), and the same graph ran on an explicit stream with torch's sync
detector in error mode, through rebound buffers, and under CUDA-graph replay,
bit-identical to eager. No kernel change was needed. Other DSL versions are
unverified.

| Feature | Native (512, 512) / envelope (256, 512], independently ×8ʷ |
|---|:--:|
| FP16 / BF16 Q/K/V/O | ✅ / ⚠️ |
| Dense layouts within `dense_flex`, with native TMA stride alignment | ✅ / ⚠️ |
| Packed token-major THD Q/K/V/O | ✅ / ⚠️ |
| Optional FP32 Stats, natural-log or base-2 (`stats_use_log2`); strided / permuted dense Stats within `dense_flex`; packed THD token-major / head-major Stats | ✅ / ⚠️ |
| Unmasked / top-left causal / bottom-right causal | ✅ / ⚠️ |
| Left/right diagonal bands; bottom-right requires a right bound | ✅ / ⚠️ |
| Bottom-right or left-window `S_q > S_kv` with padding or THD | ✅ / ⚠️ |
| Dense padding with INT32 per-batch Q/KV lengths and Stats | ✅ / ⚠️ |
| THD INT32 sequence or cumulative Q/KV lengths | ✅ / ⚠️ |
| Ragged `S_kv` (no tile rule; masked natively, no synthesized lengths) | ✅ / ⚠️ |
| GQA / MQA; dense PackGQA when the head ratio divides 64 | ✅ / ⚠️ |
| Decode-shaped (`S_q == 1`) | ✅ / ⚠️ |
| Per-head FP32 sink at any `S_q`, decode (`S_q == 1`, declared explicitly by the row) included: a keyless row is O = 0 / LSE = sink, a `seq_len_q`-trimmed row stays O = 0 / LSE = −inf | ✅ / ⚠️ |
| Host scale, including literal zero and negative values (the sign is a compile-time specialization) | ✅ / ⚠️ |
| Softmax precision unset / FLOAT | ✅ / ⚠️ |
| FP8/MXFP8, Amax_O output, mixed output dtype, bias, paged KV, fused epilogue gate, backward | ❌ |
| Dense-padded THD Stats; dense cumulative lengths or Q-only trim | ❌ |
| Either head dim ≤ 256 (below the envelope floor; served by the adapter directly, not by this row) | ❌ |

The following are **declined, pending**:

| Case | Current enforcement |
|---|---|
| Unpadded bottom-right `S_q > S_kv` | Shared validator; native declaration for direct callers |
| Unpadded left-window `S_q > S_kv`, including vacuous raw bounds | Shared validator; native declaration |
| Bottom-right band with only a left bound | Shared eligibility; contradictory direct constructor arguments |
| Tight-tail dense layouts outside `dense_flex` | Shared eligibility; native declaration |
| THD PackGQA | Shared eligibility; native declaration |
| Head-major THD data, including `S_max == 1` | Shared eligibility where detectable; native declaration always |
| THD with `SCHED_LPT` or `SCHED_LPT_L2` | Native declaration at build |
| Gapped THD sequences | Caller-contract violation; offset values are never read, so this is not detected |

Explicit records stay partial: `{}` exports `{}` before and after build.
The adapter defaults to unpacked execution. Dense causal and right-widened
declarations take LPT_L2 while one KV head's K+V (`S_kv·(D_QK+D_V)·2` bytes)
fits 50 MiB and LPT above it, the shared SM100/SM120 rule; others take NATURAL. Heuristics use the shared
ordering (for example, 8/2 GQA packs first at `S_q=32`, unpacked first at `S_q=1024`), and
THD offers NATURAL/unpacked only. Requested values remain unchanged; public
float and string knob keys follow shared integer coercion. Tiles are 64/64,
CGA and split-KV are 1; split-KV >1 declines. Schedulers are NATURAL, plain
LPT, and LPT_L2 blocks of a power of two of whole KV groups within 50 MiBʷ.

Dense launches attention with zero workspace. THD launches setup then attention
with `align128(16*B+16) + 128*(B+3)` bytes: the Int32 metadata, then `B+3`
tensor maps in SM100's slot order, carved as one plain chunk. The maps need an
absolute 128-byte boundary, so the workspace base must be 128-byte aligned (a
torch allocation is); the compiled launcher refuses any other base before
either launch. The shared carver copies a non-view-flattenable rank-2
workspace, while a 1-D strided workspace reaches the adapter and is refused.
Capture safety requires a contiguous or view-flattenable workspace.

THD sequences start at packed Q/KV prefixes times each port's token stride.
Ragged-offset declarations are required for THD Stats, but their values and
multipliers are never read. Stats packing uses the shared classifier; a
head-major head stride covering the live total is a caller contract, as on the
sibling adapters (Rule S1). Accessed padded V rows must be finite. O, Stats and
workspace must not overlap other operands (caller contract, unchecked as on the
sibling adapters), and storage must outlive the stream.

SM90 is an ordinary `ENGINE_SPECS` row using the unchanged shared validator,
matcher, heuristics, lowering and binding. Native refusals are owned by
`SdpaFwdDslSm90.check_support`, as in the sibling adapters; the adapter shares
`api_dsl.py` with SM80 and SM120 and is exported lazily from `cudnn.sdpa.fwd`.
The row is additive: `mismatch`, `recommend`, the validator, the lowering and
the binding are unchanged by it, so the nine other rows' eligibility and
ordering are what they were. That is a statement about the host code, not
numerical validation on those architectures.

ʷ **The kernel (`sm90/prefill_d512_f16.py`).** Three warpgroups (384 threads),
`setmaxnreg` (24, 240, 240). WG0 issues the TMA loads, WG1 runs QK (m64n64k512,
SS), the online softmax and PV on O columns [0, 256) (RS), WG2 PV on [256, 512)
(SS) from the P WG1 stages in SMEM — the head dim splits because a 64-row FP32 O
accumulator over 512 columns costs 256 registers per thread against a WGMMA `N`
of at most 256. Q, K and V/O are single-stage 64 × 512 regions of eight SW128
slabs, V and O aliasing one: 222,408 B of 232,448 B, so **one CTA per SM**, and
single-stage K/V means **the producer cannot run a KV tile ahead**. P is the one
multi-stage region, a three-stage WG1→WG2 ring; alpha and the inverse row sums
cross through FP32 SMEM arrays. The KV walk is reverse: boundary tiles mask,
interior ones do not. SW128 only: `wgmma_descriptor` hard-codes the 8-row ×
128-byte atom, so a narrower swizzle would be misread, and the head tile is whole
slabs (512 × 2 B is eight). The scale's sign is a compile-time specialization (a
negative scale anchors each row at its raw minimum), plus an FP64 path for scales
that would overflow the FP32 products.
**LPT_L2 departs from the shared rule**: SM120 passes the raw 50 MiB, so its
block is `floor(budget / per-group)` KV groups; SM90 floors that count to a power
of two. The raw width measured 3–14 % slower here (short last block under one
wave). The shared decode now spreads the groups evenly over the same block
count (`lpt_l2_tile_coords`), which shrinks the tail underfill (9 groups at
a width of 8 now run as 5 + 4 instead of 8 + 1); on
B300 that made LPT_L2 plans up to 5.2 % faster (median 0.4 %).
**Measured** on H200 (132 SMs), bf16, forward with O + Stats, against cuDNN
9.24.0 (FE 1.30.0, CUDA 13.0, DSL 4.8.0.dev0, 5 alternating rounds; SOL against
the 1071 TFLOPS BF16 MMA peak). DeepSeek-V4 Flash, the #991 set (b=2, 64/1 heads,
causal + SWA 128), `S` 2k / 4k / 8k / 16k: 3.57 / 3.70 / 3.82 / 3.95x, 201–217
TFLOPS against 55–57, 18.8–20.3 % SOL. Gemma4 full attention (b=1, 32/4 heads,
causal), same lengths: 5.46 / 5.72 / 5.78 / 5.72x, 491–577 TFLOPS against
90–100, 45.8–53.9 % SOL. The windowed shape's lower SOL is per-CTA fixed cost
over its three KV tiles.

---

## SM100 / SM103 (Blackwell, cc 10.0–10.6)

Engines: `sdpa_fwd_prefill_sm100` (f16/bf16), `sdpa_fwd_prefill_sm100_fp8`,
`sdpa_fwd_prefill_sm100_mxfp8`, `sdpa_bwd_sm100`, `sdpa_bwd_sm100_mxfp8`,
`sdpa_bwd_sm100_d256`. The half-precision d512 backward engine serves **only
the large-head-dim band, d ∈ (256, 512]**, the half-precision d256 backward
engine (`sdpa_bwd_sm100_d256`, the 2x2-datapath body, 2026-10-01, opt_in) serves
**exactly d = 256**, and the MXFP8 backward engine serves **exactly d = 256** —
every other `sdpa_backward()` / `sdpa_mxfp8_backward()` shape still falls
through to the cuDNN backend. The d512 BPROP column below reads ᵇ for the
half-precision d512 row and ᵍ for the MXFP8 row; the d256 BPROP column reads ᵇ²
for the half-precision d256 row and ᵍ for the MXFP8 row.

The MXFP8 backward is a **two-kernel chain with a scale-factor repack in
front**: the seven F8_128x4 scale tensors are repacked into the kernels'
2-CTA slot layout (eleven small launches into workspace), then a dQ kernel
(Q·Kᵀ, dO·Vᵀ, dS·K) and a fused dK/dV kernel (Q·Kᵀ, dO·Vᵀ, dSᵀ·Q, Pᵀ·dO) run,
both 2-CTA block-scaled MMA pipelines ported from Xinbo Zhao's
`fmha_mxfp8_large_head_dim`. dS is quantized in-kernel with an
online per-32-block E8M0 scale; P with a fixed 2⁻⁸ descale. The repack is a
documented exception to Hard Rule 2 (see `bwd/api_dsl_mxfp8_sm100.py`).
The existing chain is prepared as one pointer host: tensor views, workspace
offsets and SF layouts are fixed at plan time; execution binds current payload,
SF and gradient buffers, caller workspace and stream. This does not remove the
eleven device repacks or add new layouts or shapes.

The backward is a **three-stage chain**, not one fused kernel: a fused d=512
backward needs 512 TMEM columns for dV and 512 more for dK against 512 per CTA,
so S and dS go to a GMEM workspace and the gradients are three batched GEMMs
over it (`do_dot` → `sm100/bprop_d512_f16` → `bprop_matmul_blackwell`). Two
consequences a user can see: the workspace is `2·B·H_chunk·S_q·S_kv·2 B` (the
host loops over head chunks to hold it under a 4 GiB budget; under THD it is
`2·H_chunk·(T_q + B·128)·pad(S_kv_max)·2 B` instead, the budget charged on the token rows
so that equal-length THD gets the BSHD chunk and the padded slab may exceed 4 GiB by at most 1/8 — see ʰ), and everything in the band
is **envelope-served** — the tiles are fixed at 512, so d=264 costs the same
MMA as d=512.

| Feature | d64 (GPT-OSS)<br>FPROP | d128 (Llama)<br>FPROP | d192×d128 (DSv3 MLA)<br>FPROP | d256 (Qwen)<br>FPROP | d512 (DSv4)<br>FPROP | d512 (DSv4)<br>BPROP<br>d ∈ (256, 512] | d256 (Qwen)<br>BPROPᵇ² |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **Data types** | | | | | | |  |
| FP16 / BF16 | ✅ (native d64)⁷ | ✅ | ✅ | ✅ | ✅ | ✅ᵇ |  ✅ᵇ²  |
| FP8 E4M3 / E5M2 (per-tensor descale) | ✅ (native d64, cga1)⁷ | ✅ | ✅ | ✅ | ✅ | ❌ |  ❌  |
| MXFP8 (E4M3/E5M2 + per-32 E8M0 SF) | ✅ (native d64, cga1, dense / unsplit / unpaged)⁷ | ✅ | ✅ | ✅ | ✅ | ✅ᵍ (E4M3 only, d=256) |  ✅ᵍ (E4M3 only)  |
| O dtype ≠ QKV dtype — **quantized graphs only**¹ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ᵍ (fp16/bf16 gradients) |  ✅ᵍ · —ᵇ²  |
| Block-scaled O (`sdpa_fp8` / `sdpa_mxfp8` + `sf_o`): FP4_E2M1 O + E4M3 scale per 16 d, or E4M3 O + UE8M0 scale per 32 d — per-tensor FP8 and block-scale MXFP8 graphs, dense/unsplit/unpacked only; `scale_o` doubles as the FP4 global scale (a python-only input on `sdpa_mxfp8`) | ❌ | ✅ (FP8: SM100 / SM107 / SM120; MXFP8: SM100 / SM107, and SM120 for the E4M3 + UE8M0/32 O) | ❌ | ❌ | ❌ | ❌ |  ❌  |
| Head-dim envelope (zero-padded below native) | f16 ×8 · fp8 ×16 up to 64 (**native 64**: the d128 kernel files at `d_flavor=64`)⁷ · mxfp8 exact | f16 ×8 · fp8 ×16 · mxfp8 exact | f16 ×8 · **fp8 exact (192, 128) only**¹⁰ · mxfp8 exact | f16 ×8 · **fp8 exact 256 only**¹⁰ · mxfp8 exact | f16 ×8 · fp8 ×16, floor 256² · mxfp8 exact | f16 (256, 512] ×8ᵇ · mxfp8 exact 256ᵍ |  none — exact (256, 256)ᵇ² ᵍ  |
| **Layout** | | | | | | |  |
| BSHD | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ᵇ ᵍ |  ✅ᵇ² ᵍ  |
| Arbitrary dense B/H/S stride order (`dense_flex`) | f16 only | f16 only | f16 only | ✅ | f16 only | ✅ᵇ ᶜ · ❌ᵍ |  ❌ᵇ² · ❌ᵍ  |
| THD / ragged (packed varlen)ᵏ | f16 only⁹ | ✅ | f16 + fp8 + mxfp8³ | f16 + fp8 + mxfp8⁹ | f16 + fp8 + mxfp8³ | ✅ᵇ ʰ · ❌ᵍ |  ❌ᵇ² · ❌ᵍ  |
| `cu_seq_len_q/kv` prefix sums (THD only) | f16 only⁹ | ✅ | ✅ | ✅ | ✅ | ❌ʲ |  ❌ʲ  |
| **Masks / features** | | | | | | |  |
| Causal (top-left) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ᵇ ᵈ ᵍ |  ✅ᵇ² ᵍ  |
| Causal bottom-right | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ᵇ ᵈ · ❌ᵍ |  ✅ᵇ² · ❌ᵍ  |
| Causal right-band widening | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ᵇ · ❌ᵍ |  ❌ᵇ² · ❌ᵍ  |
| Sliding window (left) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ᵇ · ❌ᵍ |  ✅ᵇ² · ❌ᵍ  |
| Padding mask (`seq_len_q/kv`) | ✅ | ✅ | ✅ | ✅ | ✅ | THD onlyᵇ ʰ · ❌ᵍ |  ❌ᵇ² · ❌ᵍ  |
| THD + causal family (top-left / bottom-right / SWA / band) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ᵇ ʰ · ❌ᵍ |  ❌  |
| Padding mask + stats (per-batch LSE trim) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |  ❌  |
| Dense padded-Q trim (O:=0, LSE:=−inf) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |  ❌  |
| Attention sink (incl. `S_q == 1` decode)ˢ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |  ❌  |
| Base-2 stats (`stats_use_log2`) | ✅ | ✅ | ✅ | ✅ | ✅ | — |  —  |
| GQA / MQA (`H_q ≠ H_kv`) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ᵇ ᶠ ᵍ ʰ |  ✅ᵇ² ᵍ  |
| PackGQA (`PACK_GQA` knob: the GQA group packed into the Q tile)ᵐ | whole group onlyᵐ | ✅ᵐ partial | whole group onlyᵐ | ✅ᵐ partial | whole group onlyᵐ | — |  —  |
| THD prefill + PackGQA (f16/bf16, CGA2, unsplit)ᵐ | ❌ | ✅ native, partial | ❌ | ❌ | ❌ | — | — |
| Bias / dBias | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |  ❌  |
| `use_deterministic_algorithm` | — | — | — | — | — | ❌ᵇ · ✅ᵍ |  ❌ᵇ² · ✅ᵍ  |
| Ragged `S_kv` (non-multiple of 128) | ✅⁶ | ✅⁶ | ✅⁶ | ✅⁶ | ✅⁶ | ✅ᵇ ᵉ ᵍ |  ✅ᵇ² ᵍ  |
| Decode-shaped (`S_q == 1`; with sink / sliding window: ˢ) | ✅ | ✅ | ✅ | ✅ᵈ (decode tile) | ✅ | ❌ᵇ · ✅ᵍ |  ❌ᵇ² · ✅ᵍ  |
| **Decode tile** (decode + MTP; f16/bf16, dense or paged: d128 `S_q · PACK_G ≤ 128`ᵈᵗ, d256 `S_q · G` within the routed rowsᵈ) | ✅ᵈ⁶⁴ **native** (f16/bf16; an explicit cga2 runs the prefill tile) | ✅ᵈᵗ **native** | ❌ (prefill tile) | ✅ᵈ (d256 decode tile, swap-AB) | ❌ (prefill tile) | — |  —  |
| Paged KV cache (`paged_attention_k/v_table` + padding mask)ᵖ | ✅ᵖ f16 + fp8 (native d64; mxfp8 d64 is unpaged) | ✅ᵖ f16 + fp8 + mxfp8 | ✅ᵖ f16 + mxfp8 (native (192, 128)) | ✅ᵖ ᵈ f16 + mxfp8 | ✅ᵖ f16 + mxfp8 (native 512; (256, 512] envelope) | ❌ |  ❌  |
| Fused epilogue gate (sdpa virtual `O_v` → `mul(O_v, sigmoid(G))`; the SM107 rows serve it, see the SM107 table) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |  ❌  |

ᵈ⁶⁴ **d64 decode tile.** The same `sm100/decode_d128_f16.py` tile at
`d_flavor=64` (`config_sm100.CfgD64Decode`: the d128 reshape — one 128-row Q
tile per CTA, one softmax warpgroup, three KV stages — at the narrow extents,
112 KiB against d128 decode's 224). It is selected by
`TemplateParams.decode_tile`, NOT by `TILE_CGA_M=1` as on d128: cga1 is d64's
prefill width (a 512-row cga2 cluster wastes most of a narrow diagonal band), so
the two legs cannot be told apart by the cga knob, and one template record must
never name two kernels. `SdpaFwdDslSm100._d64_decode_tile` sets it, mirroring
`_decode_q_tile` for the d256 tile, and only with the cga knob unset or 1: the
tile is cga1-only (`make_cfg_d64_decode`), so an explicit `cga=2` selects the
prefill pipeline instead. Explicit paged THD FP16/BF16 split plans also use
this native D64 tile on SM100/SM103, with cga1, with or without packed GQA. The shared
packed partial/combine host supports packed token/head-major Stats; padded
Stats, sinks and other architectures retain their prior admission. Default
selection is unchanged.

ᵈᵗ **d128 decode tile (`sm100/decode_d128_f16.py`, f16/bf16).** The (128, 128) flavor of
`sdpa_fwd_prefill_sm100` has two tiles behind the `TILE_CGA_M` knob: `2` is the prefill
pipeline (TILES_Q=2 x TILE_M=128 x CTA_MMA=2 = 512 Q rows per cga2 cluster) and `1` the
decode tile — TILES_Q=1, one independent CTA per 128-row Q tile, one softmax warpgroup
(12 warps), the Q/O SMEM slab aliased so the K/V ring is three stages deep, two S/P
TMEM slots alternating per KV tile so BMM1(i+1) overlaps softmax(i). The heuristics
propose `TILE_CGA_M=1` exactly when one 128-row tile covers a packed head's live Q rows —
`S_q * pack_g <= 128` with `pack_g` the candidate's own packing: the packed group
`p = Cfg.PACK_G` on the packed leg (the whole group G when it divides 128, else its
largest divisor that does — partial PackGQAᵐ, 96/8 packs 4; S_q = 1 decode, MTP S_q in
[2, 8]), 1 on the unpacked leg, so where the
packed leg overflows one tile (`S_q * p > 128 >= S_q`) it keeps the prefill tile while
the unpacked runner-up rides the decode tile — and keep the prefill tile otherwise; a
pinned `TILE_CGA_M` is honored either
way (a pinned 1 on any dense S_q runs the decode tile). Bottom-right causal decode
shapes walk `SCHED_NATURAL` first (every unit has the same work). Same contract as the
prefill tile on dense and paged graphs: padding mask + per-batch lengths, causal /
bottom-right / SWA band, dense padded-Q trim, sink (dense and paged, incl. `S_q == 1`
— ˢ) — a keyless row with a sink (above the bottom-right diagonal, or `seq_len_kv[b] ==
0`) writes `O = 0`, `LSE = sink` whatever the sink's magnitude: the fold's operands are
selected for such a row, as the prefill kernels do since PR #1095 (`exp(sink − max)`
underflows in fp32 for sink ≤ −104, which used to give `O = NaN`, `LSE = −inf`;
`test_decode_kernel_keyless_rows_sink_magnitude` at −120 / −5 / +3, Stats on / base-2 /
off, and its graph-path twin pin it) — Stats incl. base-2, PackGQA incl. partial
packing (ᵐ: `HEADS_PER_TILE = PACK_G`, `G / PACK_G` packed heads per KV head), KV split
+ combine, NATURAL
/ LPT / LPT_L2. **THD on the decode tile: the ragged-Q-over-paged-KV leg only** (ʳᵠ:
ragged Q/O/Stats + page pools at `S_q(max) == 1` — FlashInfer's prefill-style paged graph
at one token per sequence, nvbug 6607857). Exact D128 FP16/BF16 paged THD
prefill at declared `S_q > 1` also accepts an explicit `TILE_CGA_M=1` on cc 10.0:
it uses the existing two-slab prefill body (256 packed rows), distinct from the
128-row decode/split tile. Default CGA selection remains unchanged, and cc 10.3
retains its existing domain. Other unsplit ragged prefill graphs keep `TILE_CGA_M=2`
and a pinned 1 declines. The decode tile remains unavailable for fp8 / mxfp8 (their (128, 128) flavors keep
`cgas={2}`, their other flavors their own width), and the d192x128 / d512 f16 flavors
(no decode tile yet — their decode graphs run the prefill kernel as before; the d256
f16/bf16 flavor has its own swap-AB decode tile, ᵈ). d64 has its own native decode tile (ᵈ⁶⁴). Measured on B200 (graph path,
CUDA-graph replay, b=32, d=128, S_kv=4096, page 16, bf16): 64/4 S_q=1 119.2 us on the
prefill tile → 48.9 us on the decode tile (the cuDNN backend's decode engine: 44 us);
64/4 MTP S_q=4 127.5 → 50.6 us; 64/8 S_q=1 234.6 → 96.4 us; 96/8 S_q=1 (G=12 packs 4,
partial PackGQAᵐ on both tiles) 615 → 225 us (the same shape unpacked on the decode tile:
875 us); d64 64/8 230.5 → 80.2 us. The kernel docstring carries the full table. Since issue
#1472 the cc 10.7 half row serves the same tile on DENSE d128 graphs (the body compiled for
sm_107a; THD, dense paged queries and the pre-folded scale keep the Rubin prefill pipeline at
cga2; the ragged-Q leg stays SM100-line only).

ʳᵠ **Ragged Q over paged KV on the d128 decode tile (`TemplateParams.ragged_q`, nvbug
6607857).** FlashInfer's prefill-style paged graph — ragged Q/O/Stats (ragged offsets +
`seq_len_q`) over page pools + block tables — at `S_q(max) == 1` used to be THD to every
FROST row and so kept the cga2 prefill tile unsplit and unpacked: on B200 (b=4, 64/8,
d128, page 16, bf16) 206 us at `S_kv=2048` and 1641 us at 16384, 9–22× the dense decode
graph of the same geometry (the cuDNN backend's `heur_mode.A` served it with the sm80
WMMA engine, 66 / 517 us). It now rides the decode tile's **ragged-Q leg**: the dense
grid over the declared batch is kept (one unit per packed head × batch × split), the
TMA-LDG warp reads each batch's Q ragged offset on device and uses it as the row
coordinate over the packed `[1, T, H, D]` Q view (no host read — Rule 3), PackGQA
composes (the row base is token-unit), and the split path is **mandatory**: the fp32
partials stay dense in the workspace and `split_combine_sm100` places the recombined O
/ Stats rows at their ragged offsets, skipping every row of a zero-length sequence. No
THD setup launch, no per-sequence O descriptors. Same shape after: 23.6 / 73.5 us
(device time incl. the 6–7 us combine) — the dense decode graph's 22.7 / 73.3 us — with
bit-clean Stats. Contract: the native (128, 128) f16/bf16 flavor on cc10.0/10.3 (the
ragged-Q leg is not wired on cc 10.7), `S_q(max) == 1`, page pools (a ragged K/V needs the prefill THD
leg's clamped descriptors), ragged Stats when Stats are requested (a per-batch padded
Stats has no ragged base), int32 **or** int64 offsets of one width whose multiplier
divides the row (`engines._thd_decode_leg`), per-batch `seq_len_kv` (not the cu form),
an attention sink admitted, split included (the split partials are sink-free and `split_combine`'s ragged sink
entry folds the per-head logit once; keyless rows O := 0 / LSE := sink -- see the sink + split-KV paragraph under ˢ),
no gate. Twins: `engines._thd_decode_leg` / `SdpaFwdDslSm100.thd_decode_leg` /
`config_sm100._validate_params(ragged_q)` / `CfgD128Decode.RAGGED_Q`. **Not yet:**
MTP-THD (`S_q(max) > 1` needs the per-sequence Q length in the bottom-right diagonal —
the prefill THD leg keeps it), ragged K/V (non-paged THD decode), d192×d128 / d256 /
d512 ragged decode (their decode graphs keep the prefill THD leg), fp8.

ᵖ **Paged KV (issue #920): f16/bf16 on the d64 / d128 / d192×d128 / d256 / d512 flavors, per-tensor
FP8 on the d64 / d128 flavors, MXFP8 on its four d ≥ 128 native flavors** (`Capabilities.paged_d_shapes`:
the f16/bf16 row `{(64, 64), (128, 128), (192, 128), (256, 256), (512, 512)}`, the SM100 per-tensor FP8 row
`{(64, 64), (128, 128)}`, the MXFP8 row `{(128, 128), (192, 128), (256, 256), (512, 512)}`). The head-dim
gate is the flavor the lowering SELECTS — the smallest covering envelope — not the raw
dims: d=64 is **native** (paged wired on the d64 flavor), d=192/192 and mixed dims such as (256, 128) or
(64, 192) ride the d256 one, (192, 128) is **native** on d192×d128 and shapes below it
such as (136, 72) ride its envelope; a d512-envelope selection is served on f16/bf16 —
(512, 512) **native** (DSv4-class), the (256, 512] band such as 384/384 and pairs straddling
the d256 / d512 envelopes such as (256, 512), (512, 256) or (512, 128) zero-padded on the
narrower side at the d512 kernel's MMA cost (absorbed-MLA 576/512 has no flavor envelope at
all). On the d512 flavor the loader is role-split across the cga4 cluster: the sub-group 0
CTAs issue the K boxes and the sub-group 1 CTAs the V boxes, each pair walking its own
block table; decode-shaped launches (`S_q` in [1, 8], MQA / GQA) and paged prefill (`S_q`
up to 128 validated) both run, with KV split as on d128, and THD queries over d512 pools
ride the same path. A decode-shaped paged d512 graph runs that prefill tile today (no d512
decode tile yet — the measured gap, the default placement and the follow-up are a
gaps-table row below). On the FP8 row the
d128 and native d64 selections are wired (`d_qk, d_v <= 128`: d ≤ 64 selects the (64, 64) flavor, larger
the d128 one — both exact in FP8); a d192×d128 or d256 FP8 selection is declined. The K and V pools may
differ in row width (d192×d128: a 192-wide K pool and a 128-wide V pool behind
separate block tables) and must share the in-page layout. The graph is cuDNN's own
paged-cache contract: K/V are page pools `[num_pages, H_kv, page_size, D]` — HND
compact, or NHD (`[num_pages, page_size, H_kv, D]` storage) declared through the
strides — plus `(B, 1, max_pages, 1)` int32 block tables and `use_padding_mask` with
`seq_len_q` / `seq_len_kv` (the per-batch KV length is read on device;
`paged_attention_max_seq_len_kv` defaults to `max_pages * page_size`). `page_size` is
a multiple of 8 that divides the 128-row KV tile or is a multiple of it. Any `S_q`
(decode or paged prefill), GQA (PackGQAᵐ — the whole group, or on the f16/bf16 kernels
its largest divisor of the tile; the FP8 kernel packs the whole group only), Stats out,
and — f16/bf16 only — **THD queries**: ragged Q/O (ragged offsets + `seq_len_q`)
over the same pools — chunked prefill — with the THD scheduler walking the Q units (no
KV split there), including top-left causal + a left window (on d192x128 that band takes
the plain decode path: the predecoded THD+SWA scheduler is folded off under `PAGED_KV`);
THD queries over FP8 pools are declined (the FP8 THD path clamps its runtime K/V
descriptors to a packed total a pool does not have). At `S_q(max) == 1` a ragged d128 f16/bf16 graph with ragged Stats instead rides the d128 decode tile's ragged-Q legʳᵠ (PackGQA + KV split + combine, the offsets read on device) — FlashInfer's prefill-style paged graph at one token per sequence.
KV split is proposed on dense-Q
paged graphs by the same wave-cost model as on dense graphs (they are padded by
construction: the per-batch lengths bound the walk on device and the split composes
with them; it pays when `B * H_kv` leaves SMs idle) and recombined by
`split_combine_sm100`, which on the FP8 row also owns the `Amax_O` of the recombined O.
The declared `paged_attention_max_seq_len_kv` only sizes that cost model — a maximum
that is not a multiple of the 128-row KV tile (FlashInfer passes its true max verbatim,
e.g. 4000) does not withhold the split, unlike a mask-free dense `S_kv` off the tile, which rides
the KV-tail mask. Exact D128 now permits that scalar-tail split explicitly;
automatic scalar-tail selection stays conservative. Decode-shaped d128 units
(`S_q * PACK_G <= 128`, one decode tile's Q rowsᵈᵗ) launch one CTA per `(batch, packed
head)` unit on the plain scheduler; the packed head holds `PACK_G` Q heads — the KV
head's whole group when it divides the tile, its largest divisor that doesᵐ, a single Q
head when nothing packs — so a serving batch such as `B=32, H_kv=4` fills one wave and
runs unsplit, while a small batch splits the KV loop across the idle SMs (`B=8, H_kv=4,
S_q=1`: `SPLIT_KV=4`; the wave-cost model charges the unsplit leg no combine and a lone
combine block its latency floor, `heuristics.choose_split_kv`). Units of up to one cga2
cluster (`S_q * PACK_G <= 512`, e.g. a 16-token speculative chunk at 64/4) keep the
prefill tile and still walk `SCHED_NATURAL` first under a causal band (the one-cluster
rule). The attention sink (incl. `S_q == 1`) and
a left sliding window under the bottom-right causal diagonal ride the same paged graph
on every wired f16/bf16 flavorˢ; on the FP8 row the left window rides it; the sink and
the block-scaled O epilogue (`sf_o`) over pools are not validated, so those two pairs
stay declined. The FP8 graph is `sdpa_fp8`
with scalar `descale_q/k/v` and `scale_o` (`scale_s`/`descale_s` accepted and ignored),
O in FP16/BF16/E4M3/E5M2, `Amax_O` out. **The graph must not declare
`Amax_S`**: every FROST FP8 row declines a graph that requests that output (the kernels
do not produce it), so a graph that binds `Amax_S` as a non-virtual output stays on
the backend engine, and the paged FP8 capability is reachable only by leaving it
virtual (FlashInfer's `sdpa_fp8` prefill graph already does: `set_output(False)`). Masks validated over FP8 pools: a causal
upper bound top-left or bottom-right (MTP `S_q <= 8`, each batch's diagonal anchored
at its own KV length, rows left without a key write O := 0 / LSE := -inf) and a left
sliding window (`test_sdpa_fwd_paged_sm100.py` fp8 causal / sliding-window tests, pinned
on the FROST plan, and the `test_mhas_v2.py` fp8 paged decode fuzz, which draws the same
masks over the default walk and asserts the row served every draw). Not yet: packed (ragged-offset) block tables, the f16/bf16 and FP8 d512 flavors, the SM107 (Rubin)
half / per-tensor FP8 siblings (the SM107 MXFP8 d128 / d256 siblings serve pools -- SM107 table), sink over FP8 pools (and a sink + KV split on the FP8 / MXFP8 pool rows: their epilogues fold the sink per tile, so a quantized sink graph runs unsplit; the half rows split with a sink -- see ˢ), block-scaled O
(`sf_o`) over FP8 and MXFP8 pools.
Served by the `PAGED_KV` specialization of
`sm100/prefill_d128_f16.py`, `sm100/prefill_d192_d128_f16.py`, `sm100/prefill_d256_f16.py`
and `sm100/prefill_d128_fp8.py`, and of the four `sm100/prefill_d*_mxfp8.py` kernels
(**MXFP8 pools**, below) (block-table indirection on the K/V TMA loads; boxes past
a sequence's live pages are TMA-OOB zero-filled; under paged KV the d192×d128 kernel
takes the plain scheduler decode — its predecoded THD+SWA segment path folds off);
every other kernel file refuses `paged_kv=True` at module scope. For decode / MTP shapes
the f16/bf16 decode tiles serve the same paged contract: on the d128 flavor (`S_q *
PACK_G <= 128`) the d128 decode tile `sm100/decode_d128_f16.py` (ᵈᵗ), on d256 the d256
decode tileᵈ when the graph is decode-shaped (`S_q * G <= 32`); there is no quantized
decode tile — paged FP8 decode runs the prefill tile. The d192×d128 flavor
has no decode tile yet: at decode shapes its paged graphs run the prefill geometry (one
live row per 128-row Q tile), measured behind the backend's paged decode plan (B200,
`b = 32`, `S_q = 1`, page 16, bf16, default plan: 32/32 MHA 788.7 µs vs 476.9 µs, 32/8
GQA 275.8 vs 199.6 µs — see the gaps table); a d192×d128 decode tile is the follow-up,
as ᵈᵗ was for d128. Validated 2026-09-16 on B200 (`test_sdpa_fwd_paged_sm100.py`:
d192×d128 graph and kernel level at cga1 / cga2, page 16 / 32 / 128, HND / NHD, forced
splits, sink at `S_q` 1 / 4 with the bottom-right causal diagonal;
`test_mhas_v2.py::test_sdpa_fwd_paged_mla_decode_L0` /
`test_sdpa_fwd_paged_mla_frost_L0` (decode- and prefill-shaped fuzz, FROST serves every
draw) and the pinned `test_sdpa_fwd_paged_d192x128_decode_frost_L0` /
`test_sdpa_fwd_paged_d192x128_prefill_frost_L0` /
`test_sdpa_fwd_paged_envelope_decode_frost_L0`). **Decode-shaped paged FP8 is served — FROST-first under the opt-in — and measured
behind the backend's decode engine (B200 / SM100, cuDNN 9.26, B=32, S_q=1, S_kv=4096
mixed per-batch lengths, page 16, e4m3 pools, bf16 O; probe omitting `Amax_S` — with it
declared FROST declines and the 64/4 graph still fails to build on the backend):** the
FROST row leads for every graph it accepts, and the d128 paged FP8 kernel is a prefill
tile (one 128-row Q tile per batch and KV head; there is no quantized decode tile), so
decode-shaped graphs pay for it. FP8 64/4 heads (PackGQA, 16 live rows per tile) is a
capability win: the backend engine accepts that graph at plan time and fails to build it
(`cudnnFinalize`: runtime kernel compilation failure — at B=32 with no Stats output; a
batch of 2 or a Stats output builds), and FROST serves it at 137-161 us. FP8 96/8 heads
(group 12 does not divide the tile, PackGQA off, one live row per tile) runs 807 us on
FROST against 54.9 us on the backend engine. A prefill-shaped paged FP8 graph can pay
too: B=4, 16/4 heads, d128, S_q=64, page 16, max KV 2048, e4m3 Q/K/V, bf16 O defaults to
the FROST `PACK_GQA=1` / `SPLIT_KV=2` plan at 50.0 us GPU (186-189 us CPU enqueue) against
the backend engine's 28.7 us (15-16 us) — 148-SM SM100, cuDNN 9.25.1, CuTe DSL 4.8,
independent review measurement; on B200 / cuDNN 9.26 (this lane's probe, same kernel
bytes) the same chunked-prefill graph measured 204.7 us on FROST vs 30.4 us on the
backend engine at `S_q=64`, 200.5 vs 30.7 us at `S_q=128`, 135.6 vs 31.9 us at
`S_q=512`, and 96/8 `S_q=9` 808 vs 369 us. Both gaps are closed by kernels — an fp8 d128 decode tile,
the quantized twin of ᵈᵗ, and the prefill tile's own short-`S_q` tuning — not by a
backend-relative ordering rule, which encodes a performance snapshot that goes stale
(gaps table); a caller that needs the backend plan for such a shape today deselects the
FROST row by engine name (`graph.deselect_engines([...])`).

**MXFP8 pools** (`sdpa_mxfp8` with the same paged arguments; the `PAGED_KV` specialization of all four
`sm100/prefill_d*_mxfp8.py` kernels, `paged_d_shapes` = every native MXFP8 shape on that row): the block-scale
descale tensors are page pools too and page with K/V through the same block tables —
`descale_k` is `[num_pages, H_kv, page_size, D/32]` (scaled along D) and
`descale_v` is `[num_pages, H_kv, page_size/32, D]` (scaled along the page's rows), both F8_128x4-reordered,
so `page_size % 128 == 0` (a page holds whole 128-row SF atoms and TMA loads one SF tile per KV tile).
Q keeps the dense `[B, H_q, S_q_padded, D/32]` F8_128x4 descale. E4M3/E5M2 in, O in FP16 / BF16 / E4M3 / E5M2, Stats,
GQA (no PackGQA), KV split, sink; exact native head dims (d128 / d192×d128 / d256 / d512), dense Q only
(no THD queries), no block-scaled O (`sf_o`) over pools, SM100/SM103 only. The cuDNN backend declines these graphs (no paged load for block-scale
pools), so a FROST engine is the only server.

ˢ **Attention sink at `S_q == 1` (decode), incl. paged KV and sliding window.**
Served natively by this row: the sink is a per-Q-row epilogue fold (`max(m, sink)`
lifts the running max, `exp(sink − max)` joins the denominator, `LSE = max + log(sum)`)
that is independent of `S_q`, of the mask and of the paged loader. Hardware-validated
on B200 (SM100) — `test/python/sdpa/frost/test_sdpa_fwd_paged_sm100.py`,
`test_sdpa_fwd_dsl_sm100.py`, the `test/python/sdpa/graph/test_mhas_v2.py` `S_q = 1` sweeps
(`test_sdpa_random_sq1_L0`, `test_sdpa_random_sq1_unified_L1`,
`test_sdpa_random_lean_attn_L0`, `test_sdpa_random_lean_attn_unified_L1` draw
`with_sink_token` 1:3 when FROST engines are enabled, sink-free otherwise) and the
pinned `test_sdpa_paged_decode_sink_sliding_window_frost_L0` /
`test_sdpa_paged_decode_sink_keyless_rows_frost_L0` (both ride the d128 decode tileᵈᵗ
since #1094 — `TILE_CGA_M=1` asserted — whose fold carries the same keyless-row
select) and their d256 twin
`test_sdpa_paged_decode_sink_d256_frost_L0` (since the d256 decode tileᵈ its two
decode-shaped cases ride that tile — `decode_d256_f16` asserted — and a third case with
more packed Q rows than the tile routes, `S_q · G > 32`, keeps the d256 prefill kernel's
fold covered — `prefill_d256_f16` asserted) — per flavor: d128 at `S_q` in
{1, 2, 4} (dense and paged, f16/bf16, PackGQA on and off, and partialᵐ -- 96/8 and 48/8,
`sinks[row_head_idx]` stays the Q head under `PACK_G < G`); d64 on its **native**
decode tile (ᵈ⁶⁴) at `S_q` in {1, 4} paged (bf16, 64/8 heads, sink + left window 128) and `S_q = 1`
dense; d256 at `S_q` in {1, 2, 4} paged (f16, and bf16 at `S_q` 1 and 4: the
keyless-row sink case and the pinned `test_mhas_v2` twin); d192x128 and d512 at
`S_q` 4 dense (keyless rows, sink −120). Across them: HND and NHD pools, dense
unpadded / dense padded / paged, sink + `diagonal_band_left_bound` + bottom-right
causal (`right_bound = 0`); the `test_mhas_v2` sweeps add their own geometry at
`S_q = 1` (d in 1..128 incl. mixed dims, GQA up to 32 heads, dense / padded /
packed-THD, both alignments, `S_kv` up to 8192). A keyless row (`seq_len_kv[b] == 0`,
or above the bottom-right diagonal) holds the sink's mass alone and writes `O = 0`,
`LSE = sink` whatever the sink's magnitude: the four sm100 f16 prefill kernels (d128,
d192×128, d256, d512) — and the d128 decode tileᵈᵗ, which #1094 gave the same
select — select the fold's operands for such a row instead of computing
them (`exp(sink − max)` underflows in fp32 for sink ≤ −104, which used to give
`O = NaN`, `LSE = −inf`; `test_paged_graph_keyless_rows_sink_magnitude` at −120 / −5 /
+3 and `test_dsl_sm100_keyless_rows_very_negative_sink` on every flavor pin it). What
changed: the python-native validator (`cudnn/_sdpa_validate.py`) no longer rejects
`sink_token` at `s_q == 1` — that was the backend engines' support-surface rule,
which the C++ surface keeps for the backend-only path — and the row's `paged KV with
an attention sink is not validated` decline is gone. Scope of the lift: the removed
rule sat in the validator's `NodeType.SDPA` branch, so it only ever gated f16/bf16
`sdpa()` graphs — this row (SM100 f16/bf16, dense and paged, hardware-validated
above) and, unvalidated, the other f16/bf16 forward rows that declare `sink=True`
with the default `decode=True` (SM120 f16/bf16: sink at `S_q == 1` is ❔ there — same
epilogue fold, not run here; SM107 f16/bf16 is validated on cc 10.7, paged THD included —
see the SM107 table). The quantized rows are untouched by
it: `sdpa_fp8()` / `sdpa_mxfp8()` build `SDPA_FP8` / `SDPA_MXFP8` nodes that never
passed through that branch, so the sink-at-decode status of SM100 / SM107 per-tensor
FP8, SM100 / SM107 MXFP8 and SM120 FP8 (❔) is pre-existing and unchanged, not newly
exposed. **Sink + split-KV on the SM100/SM103 f16/bf16 rows (cc 10.0 / 10.3 only):** a split kernel
writes SINK-FREE partials (its epilogue fold is compiled out at `SPLIT_KV > 1`) and `split_combine`'s
sink entries fold the per-head logit once -- `M = max(max_s lse_s, sink)`, `den = Σ_s exp(lse_s − M) +
exp(sink − M)`, `LSE = M + log(den)`, keyless rows O := 0 / LSE := sink, the unsplit kernels' contract --
on dense and paged graphs, the d128 decode tile's ragged-Q legʳᵠ and the d256 decode tileᵈ; the native dense
binder hands the fp32 [H] logits to the combine, so a sink decode graph is no longer pinned to one cluster
per (batch, KV head). A PADDED query row (past its batch's `seq_len_q`, the dense padded-Q trim) keeps O := 0 /
LSE := −inf whatever the sink -- its partials equal a valid keyless row's, so a plan carrying `seq_len_q` also
hands the combine the int32 Q lengths (`split_combine.compile_ptr(has_q_lens=True)`, the binder's
`supports_combine_q_lens`) and the lean, serial and decode forms drop the sink for those rows. Validated on B200 (SM100): `test_sdpa_fwd_split_kv_sm100.py::test_split_kv_sink_matches_reference_dense`,
`test_sdpa_fwd_decode_d128_sm100.py::test_decode_kernel_sink_split_kv` (dense and paged) /
`test_graph_small_batch_decode_sink_splits_kv`, `test_sdpa_fwd_decode_d256_sm100.py::test_decode_graph_sink_small_batch_splits_and_recombines`
/ `test_decode_graph_sink_deep_split_keyless_rows`, `test_sdpa_split_combine_sm100.py::test_pointer_combine_folds_the_sink_once`;
the padded rows: `test_pointer_combine_sink_keeps_padded_query_rows_dead` (every combine form),
`test_decode_kernel_sink_split_kv_keeps_padded_query_rows_dead` (d128 direct / d64 quadrant) and
`test_split_sink_keeps_padded_query_rows_dead` (graph API: the d256 / d128 / d64 decode tiles and the d256 prefill tile).
Still unsplit with a sink: the per-tensor FP8 and MXFP8 rows (per-tile epilogue fold), a gated split (its
combine carries no sink) and the cc 10.7 dense legs (`test_mhas_v2.py::test_sdpa_fwd_cc107_sink_split_declines_L0`;
the Rubin exception stays the packed D128 half paged THD Q>1 combine ᵖˢ).

ᵈ **d256 decode tile (`sm100/decode_d256_f16.py`, 2026-09-16).** Same engine row,
same graph contract, a second template: f16/bf16 d256 graphs (d_qk = d_v in
(128, 256], the d256 envelope) whose packed Q rows fit 16 -- `S_q * G <= 16` with
`G` the PackGQA group (16:1 at S_q = 1, 8:1 at S_q <= 2, 4:1 at S_q <= 4, MHA at
S_q <= 16; `config_sm100.D256_DECODE_ROUTED_MAX_Q_ROWS`) -- lower onto a swap-AB
tile: KV tokens on the MMA M axis (128 keys per tile), the 16 Q rows on N, one
cta_group::1 CTA per (KV-head group, batch, split), softmax reducing over TMEM
lanes (4 warps per 16 Q columns), P^T through a swizzled SMEM tile, O^T in TMEM.
Dense padded, dense unpadded (KV-tail mask) and paged caches, every mask
the d256 row serves (padding / top-left and bottom-right causal / SWA / right
band, dense padded-Q trim), sink (dense), Stats natural or base-2, KV split
partials for `split_combine_sm100`. THD, fp8/mxfp8 and `S_q * G > 16` stay on the
prefill tile; `TILE_CGA_M` / `SCHED_POLICY` knobs are accepted and unused there.
The template also compiles a **32-column tile** (`N_Q = 32`: two softmax column
groups, 8 softmax warps over the same 128 TMEM lanes) that is driven and checked
at the template level but **not routed**: it is issue-bound per CTA -- 2.8 us per
KV tile against the 16-column tile's 1.75 us, 90 us unsplit at the b=32 x 2
KV-head x 4096-key S_q=2 MTP step where the prefill tile takes 66 us -- and its
split-2 plan (58 us of GPU time) costs an eager caller 96-103 us per execute, so
routing it would regress uncaptured MTP callers; `S_q * G` in (16, 32] stays on
the prefill tile (pinned by the routing-boundary tests) until its per-CTA issue
rate is fixed.

PackGQA on the decode tile is **whole-group**: `HEADS_PER_TILE = QH_PER_KH` (96/8
puts its 12 heads in the 16-column tile, four tail rows zero-filled), not the
prefill tile's partial `gcd(G, 128)`ᵐ, so the routing test and the split model
count `S_q * G` (adapter `_decode_q_tile`, heuristics `_decode_tile_pack_g`): 96/8
at S_q = 1 is a 12-row decode launch of `B * 8` units, at S_q = 2 (24 rows) the
prefill tile's graph.

Split policy (`heuristics.choose_decode_tile_split_kv`): the tile has its own cost
model -- per-CTA streaming until the concurrent K/V streams saturate HBM, plus the
combine pass -- and the LEADING plan charges the split path's second host launch
(a second CuTe-DSL launch plus slab carving: ~30 us more per eager `graph.execute`
on the Python launch path, 62 -> 92 us), so it splits only where the GPU saving
also covers that; the captured caller's optimum, when different, is the runner-up
plan (`select_plan`), and LEADS when the graph was created with
`is_cuda_graph_replay_expected=True` (`facts.cuda_graph_replay`: the second launch is
paid once at capture). On the 16-column tile the b=32 x 2 KV-head x 4096-key
serving shape runs unsplit, b=8 splits 8 ways, b=128 stays unsplit, s_kv=16384
splits 2 ways. The 32-column tile's fit (1.6x the per-tile cost) prices the launches that
route it (`config_sm100.d256_wide_decode_launch`, below).

Measured on B200 (SM100, 148 SMs), bf16, 32/2 heads, d256, page 16, prefill tile
-> decode tile as the heuristics lead, two numbers per cell: eager back-to-back
`graph.execute` (host time included: what an UNCAPTURED caller sees) / CUDA-graph
replay (GPU only). The eager readings are from an otherwise idle GPU: that number
is Python-host-bound and inflates 2-6x when other processes share the device (a
551 us reading of the s_kv=16384 case in a contended window re-measured at 164 us);
the replay numbers are stable across windows to about 1 us. b=32, s_kv=4096, S_q=1,
full lengths: 64.3-64.6 / 63.7 -> 57.2-59.2 / 56.2-56.9 us (the split-2 runner-up
replays at 51.3 us, 1.24x, but runs 95-107 us eager); mixed lengths: 57.0-60.7 /
57.5-58.4 -> 55.8-58.0 / 55.4 us (runner-up replay 44.9 us); b=8: 99.3-99.8 /
26.7-27.1 (the prefill model splits 4 ways there) -> 93.7-99.4 / 22.6 us (split 8;
the unsplit runner-up 57.6 / 55.3); b=128: 250.4-251.7 / 286.7-333.6 -> 161.4-164.7
/ 161.8-166.9 us; s_kv=16384: 245.1-252.4 / 265.2-343.7 -> 163.8-181.0 / 163.9-166.9
us (split 2, host 96 us per execute; unsplit 199.7 / 199.9); Qwen3-Next 16/2 b=32:
63.7-69.7 / 63.5-69.5 -> 56.8-60.0 / 55.6-56.0 us; Qwen3-Next 16/2 b=64 s_kv=8192
mixed: 209.6-222.7 / 218.9-240.2 -> 137.0-159.2 / 138.6-143.5 us; page 64:
62.7-64.4 / 62.1-70.0 -> 54.3-59.1 / 53.3 us. S_q=2 bottom-right at 32/2 (32
packed rows) stays on the prefill tile: 66.1-67.0 / 65.9-66.1 us before and after
(on the 32-column decode tile it measured 90.2 / 90.2 unsplit and 96-103 /
58.5 split 2). Dense 16-row graphs the rule newly takes, replay, prefill -> decode:
MHA 8/8 b=8 S_q=16 s_kv=2048 32.9 -> 29.6; MHA 32/32 b=1 S_q=16 s_kv=32768 243.5 ->
179.6 (split 4), causal 16.5 -> 12.5; GQA 32/2 b=4 S_q=1 s_kv=32768 68.0 -> 60.6;
GQA 32/2 b=32 S_q=1 s_kv=4096 61.8 -> 51.2; padded GQA 16/2 b=32 S_q=2 bottom-right
52.8 -> 51.2 and 32/4 b=16 52.8 -> 49.2; MHA 64/64 b=1 S_q=16 s_kv=4096 62.4 ->
51.4; synthesized KV tail S=300 14.4 -> 12.3, S=1000 20.5 -> 20.5. The main kernel
alone reaches the 46 us of the TensorRT-LLM decode kernel (via FlashInfer) at split 2;
the shared combine pass (~6 us) is the remaining captured-path gap. Not a Capabilities change (the row's claims are
unchanged; this documents the lowering), Rule S2.

**The 32-column tile on big launches (SM100 / SM103).** A lone CTA streams a KV tile of the 32-column
tile in 2.8 us against the 16-column tile's 1.75 us, so below ~130 concurrent streams the prefill tile
serves the (16, 32] packed rows better -- the numbers above. A launch that is HBM-bound whatever the
per-CTA rate is the other regime: the 256-row prefill tile reaches ~78 % of the HBM roofline on
S_q x G = 32 d256 steps, the 32-column decode tile the roofline. `config_sm100.d256_wide_decode_launch`
routes the tile for batch x KV-head groups >= 8 units streaming >= 4096 128-key tiles in all (never a
windowed graph); the heuristics' decode geometry, `engines.d256_decode_tile_selected` and the adapter's
`_decode_q_tile_for(b, s_kv)` read the same rule. Kernel time on B200, bf16, d256, S_q = 2 / 32 heads
over 2 KV heads (16 packed rows per group x 2 tokens), prefill tile -> decode tile: b=128 x 128K keys
5.91 -> 4.75 ms, b=32 1.54 -> 1.21 ms, b=16 0.767 -> 0.621, b=8 0.386 -> 0.332, b=4 0.202 -> 0.179;
b=128 x 32K 1.50 -> 1.20 ms, x 8K 0.401 -> 0.323, x 2K 0.120 -> 0.101; 8 heads over 1 KV head at
S_q = 3 / 4 (b=128 x 128K 2.93 -> 2.36 ms, b=16 0.385 -> 0.331, b=8 0.199 -> 0.181). Launches under the
boundary measured within +-2 % of the prefill tile (b=1 / 2 / 4 at one KV head, b=32 x 2K-8K keys,
b=8 x 2K-8K keys) and are unchanged: forced onto the wide tile they were +1..+10 % at b=1 (the serial
combine of its 64-128 splits) and +2..+7 % at 64 units x 16-32 tiles.

ᵐ **PackGQA — partial packing on the d128 and d256 f16/bf16 kernels**
(`Capabilities.pack_gqa_partial_d_shapes = {(128, 128), (256, 256)}`, `Cfg.PACK_G`).
A GQA group `G = H_q / H_kv` that divides the 128-row Q tile packs whole (64/4:
16 heads per token row-group); one that does not packs its largest divisor that
does, `gcd(G, 128)`: 96/8 (G = 12) packs 4 heads per token row-group with 3
packed heads per KV head, 48/8 (G = 6) packs 2. A group sharing no factor with
the tile (G = 3, 5, 7, …) cannot pack — a pinned `PACK_GQA=1` is **declined**
rather than silently run unpacked, and the heuristics propose only unpacked
plans; MHA (G = 1) is the identity. **Native** in the d128 and d256 f16 kernels
(`HEADS_PER_TILE = CFG.PACK_G`; `CFG.QH_PER_KH` stays the GQA ratio for the
bottom-right diagonal and the LPT_L2 head grouping, whose groups become the
`G / PACK_G` packed heads of one KV head); d64 packs the **whole group only**
(`pack_gqa_partial_d_shapes` leaves (64, 64) out, so `_pack_g(..., partial=False)`). The
d192×d128 and d512 f16 kernels and every fp8 / mxfp8 kernel pack
`HEADS_PER_TILE = G` and keep the **whole-group** contract (a non-divisor
group is declined there). Validated on B200: `test_sdpa_fwd_paged_sm100.py`
(graph + kernel, G = 12 / 6 / 5 / 3, S_q 1–8 bottom-right causal, HND / NHD),
`test_sdpa_fwd_dsl_sm100.py::test_dsl_sm100_pack_gqa_partial_group` (dense,
d128 / d256, Stats), and the `test_mhas_v2.py` paged partial-pack sweeps.

THD prefill also supports packing at **native (128, 128), f16/bf16,
CGA2, split-KV=1** on SM100/SM103, including paged KV. The capability field
`thd_pack_gqa_d_shapes` keeps other shapes and engine rows fail-closed;
the separate ragged-Q decode tile retains its existing contract. The GPU
worklist counts token tiles after packing and every packed Stats layout
indexes the actual query head. `test_dsl_sm100_thd_pack_gqa_stats_capture`
checks token-major and head-major O/LSE, partial GQA groups, non-tile-aligned
lengths, empty sequences and CUDA Graph replay with poisoned outputs.
`test_paged_graph_thd_pack_gqa` covers HND/NHD pools and partial groups;
the padded-Stats stride test compares packed and unpacked plans in all
storage orders. Heuristics prefer packing for causal THD with GQA4/GQA8, and
GQA2 and GQA16 on nonpaged SM100/SM103 (unpacked GQA16 ran 1.04-2.58x the backend on
B200, packed 0.53-0.99; packed GQA2 matched or beat unpacked on 72 B300 cases); other supported THD groups remain available as
explicit tuning candidates. On SM100/SM103, nonpaged d128 half THD graphs
with a packed first plan lead the backend unsplit when they are bottom-right
causal without window, sink or right band, declare KV > 512, and carry at
least 110 query rows per SM or Q >= 256 at KV >= 1024; GQA2 also needs KV >= 4096.
Both query gates use a declared `max_total_seq_len_q` when it is smaller than
b * s_q: rows are h_q * min(b * s_q, total), and the Q >= 256 test uses the
per-sequence mean of the total.

¹ **Reads as: on a quantized (fp8/mxfp8) graph in this column, O may be FP16,
BF16, E4M3 or E5M2.** It does NOT mean an f16/bf16 graph may convert O — the f16
row has no `out_dtypes` domain and `facts.uniform_dtype` requires O == Q there.
`—` marks a column with no quantized kernel at all.
² The d512 FP8 flavor serves head dims in (256, 512] on both axes; a smaller
graph is declined rather than routed onto it at >2× zero-padding cost.
³ Per-tensor FP8 and MXFP8 both serve THD at native d192×d128 and d512.
MXFP8 uses the packed per-sequence tile-padded scale-factor layout.
⁴ Every SM100 / SM103 flavor carries the `SEQ_Q_LENS_PRESENT` epilogue trim (f16, per-tensor FP8 and MXFP8 alike, #1037).
⁵ Every forward kernel trims dense padded Q natively; there is no `dense_seq_q_trim` capability any more -- a graph with per-batch Q lengths compiles the trim specialization on every row.
⁶ Served through the KV-tail mask (the padded mask compiled against the scalar
`S_kv`, no lengths buffer; #1425), or natively when the causal band covers the KV tail.
⁷ **d=64 is NATIVE on SM100/SM103 for every forward dtype.** `_SM100_FLAVORS`
leads with `(64, 64)`, so `_pick_flavor(64, 64)` returns `(64, 64)`: an f16/bf16
graph compiles `sm100/prefill_d128_f16.py`, a per-tensor FP8 graph
`sm100/prefill_d128_fp8.py` and an MXFP8 graph `sm100/prefill_d128_mxfp8.py`, each
at `TILE_K = TILE_O = 64` (`TemplateParams.d_flavor`, `config_sm100.CfgD64`; the
quantized kernel-file maps `_SM100_FP8_KERNEL_FILES` / `_SM100_MXFP8_KERNEL_FILES`
list `(64, 64)`) — the contraction is no longer zero-padded 64 → 128, which was
costing ~2× the MMA work on both BMM1 and BMM2. The f16/bf16 flavor carries the
same graph contract as the envelope it replaced — paged KV, split-KV and the
128-row decode tile are all wired at d64 (ᵖ, ᵈ⁶⁴) rather than handed back to
d128. The per-tensor FP8 leg is cga1 with paged KV and split-KV; the MXFP8 leg is
cga1 and dense / unsplit / unpaged (that row's `paged_d_shapes`, `split_d_shapes`
and `thd_d_shapes` leave `(64, 64)` out, and `check_support` declines the same).
The half THD path also honors LPT and LPT_L2 over its live work list. On
SM100/SM103, exact `(64, 64)` causal graphs prefer LPT when it is in the
flavor's scheduler domain and there is no sliding window, sink, or right-band
widening. Captured plans keep reading current sequence lengths, including
prefix chunks and empty requests.
Rubin (SM107) has no d64 sibling and keeps the d128 envelope (ⁱ).
⁸ MXFP8 sets `d_pad_multiple=0` (exact native shapes only — the scale-factor
plumbing is not audited for envelope zero-padding), so an MXFP8 graph off every
native flavor's exact head dims — `(64, 64)` is one since ⁷ — is declined at plan
time rather than padded.
ᵇ **`sdpa_bwd_sm100` — the d512 (DSv4) backward: d ∈ (256, 512], multiples of 8,
f16/bf16** (the f16/bf16 d = 256 backward is the ᵇ² row since 2026-10-01). It is
one engine over one head-dim band, not a per-flavor column, so it does not
follow the FPROP columns to its left: below 257 the d256 rows are the right
kernel and the engine declines, and every other `sdpa_backward()` shape on
SM100 falls through to the cuDNN backend. The whole band is **envelope-served** on 512-wide tiles, so d=264
pays d=512's MMA cost. Multiples of 8 rather than the forward's 16: the stage-3
epilogue narrows its store vector from 32 B to 16 B when d is not also a
multiple of 16.
ᵇ² **`sdpa_bwd_sm100_d256` — the d256 (Qwen) f16/bf16 backward on the 2x2 tcgen05
datapath (2026-10-01, opt_in; `python/cudnn/sdpa/bwd/api_dsl_sm100_d256.py`).**
The Rubin half chain (`api_dsl_sm107`: `delta = rowsum(dO·O)` → the main kernel
with dV in TMEM and a kv-major `[B, H_chunk, S_kv, S_q]` dS workspace → dK = dS·Q
and dQ = dSᵀ·K as `bprop_matmul_blackwell` at the (256, 256) cluster tile → the
GQA fold; ONE prepared pointer-host artifact per plan, `prepared_sm107.py` /
`kernels/sm107/prepared_host.py`) over the shared body
`kernels/bprop_d256_2x2_f16.py` at `datapath_2x2_profile = 1`
(`bwd/config_d256_2x2.py`): `tcgen05.mma.cta_group::2` with the collective M = 128
(64 kv rows per CTA, a 128-row kv block per cga2 pair), every operand an SMEM SS
operand (no UTCCP K-split, no TMEM P; P is lane-written into a 2-deep SMEM ring
and published with `fence.proxy.async` + a `.release.cta` leader arrive), 512
non-exclusive TMEM columns, 210 KiB of SMEM, descriptor version 0 — the footprint
that fits SM100's 227 KiB / 512 columns where the 4x1 Rubin body needs 576 /
327 KiB. The kv WRITE PAIR stays 256 rows: both 128-row blocks of a pair walk
one pair-rounded q range, so the stage-3 renderings, the two-sided K-trim and
the no-zero-fill contract are the Rubin row's (poisoned-workspace tests in
`test/python/sdpa/frost/test_sdpa_bwd_d256_sm100.py`). Exact (256, 256), f16/bf16;
dense, top-left and bottom-right causal (ragged S_q included), sliding window
(left), MHA/GQA/MQA, any S_q / S_kv (padded to 128 / 256 and masked), BSHD-physical
io, contiguous fp32 Stats. Declined, each asserted by a test: dense padding
masks, sink / dSink, bias / dBias, right-band widening, THD, `dense_flex`, decode
shapes, `use_deterministic_algorithm` (no atomics in the chain; the two-launch
bitwise test passes, the claim waits on the bring-up sweep). **The native
competitor on B200 / cuDNN 9.26 is backend engine 5**
(`eng5_k14=3_k24=2_k27=0_k38=0_k40=3_k41=2`, the FORT flash_bprop 128x128x256
cga2x1x1 three-kernel chain), not engine 17: pin the FROST plan by name when
validating or measuring (the plan list offers both). Perf (B=1 H_q=32 H_kv=2
S=8192 bf16, 1155 MHz power-capped B200, cuDNN 9.26.0.51, CUPTI medians of 30
trials, one process per measurement on an otherwise idle GPU, 2026-10-01): whole
backward **5673 us dense / 3256 us causal** against engine 5's 7265 / 3939 us in the
same session (0.78x / 0.83x — under the 1.25x gate and a win on both); chain
dense = stage-2 3781 + dK 842 + dQ 828 + dot_do_o 109 + dkv_reduce 110 us, causal =
2020 + 523 + 492 + 109 + 114 us. The stage-2 kernel runs at 2465 clk (dense) /
2554 clk (causal) per 128-kv x 128-q tile per CTA against the 1708-clk MMA floor
(1.44x / 1.50x). The 2-CTA profile re-fetches Q/dO/dO_dv per 128-row kv block (2x
the 4x1 body's L2 traffic per MAC), but Nsight Compute on the dense 8K shape puts
L2 throughput at 23% and DRAM at 11% with SM throughput at 66% (2431 clk per tile
at the locked 686 MHz): the gap to the floor is exposed latency on the 1-stage
Q/dO/dO_dv rings (210 of 227 KiB used) and the S -> softmax -> P -> BMM2 chain, not
L2 bandwidth. The 4-CTA multicast arm (two pairs sharing Q/dO/dO_dv, probe
mcast_twin PASS) halves those bytes and is the follow-up, with that caveat. MMA
order (`config_d256_2x2.MMA_LOOKAHEAD`, same session): the design's lookahead order
(S(i+1) issued between dP(i) and BMM2(i)) measured 6431 / 3211 us whole backward
(stage 2 4525 / 1974 us), the NATURAL order 5673 / 3256 us (3781 / 2020 us), so
profile 1 ships NATURAL and the lookahead stays the A/B arm (an A/B/A x3 with a control pair (spread 0.10 %) completed on the causal leg and confirms the delta: 3247 us NATURAL vs 3201 us lookahead whole backward (stage 2 2019 vs 1967 us); the dense leg landed 3 NATURAL / 1 lookahead clean slots before GPU contention ended the repeat (5678 vs 6447 us; 3789 vs 4527 us stage 2), consistent with the single-run numbers: the
causal leg is 1.4 % slower under NATURAL and the dense leg 11.8 % faster -- a real
per-mask trade-off, accepted); profile 2 keeps the design's lookahead. **The Rubin
twin -- the same body at profile 2 (two sub-blocks per CTA, 322 KiB, descriptor
version 1) behind `api_dsl_sm107.BWD_D256_2X2` with `BWD_D256_2X2_PROFILE` (2, or
1 = the SM100 body as-is), default off -- traced and ran on the Rubin board on
2026-10-01** (cc 10.7, the internal DSL with sm_107a and the v1
descriptor intrinsic): the fp64-oracle matrix passes on both profiles (13 accepts,
the 13 poisoned-workspace cases, two-launch bitwise, per profile) and dQ / dK / dV
are BITWISE the shipped 4x1 body's on both profiles (int16 views, three GQA shapes).
The board's ptxas spilled profile 2 at the 176 / 152 split (84-91 STL / 122-129 LDL,
stack 448 B: 64 q columns per compute lane), so the split is per profile since
then -- 224 / 56 on profile 2 (0 / 0; 208 / 88 was worse at 113 / 135), 176 / 152
on profile 1 (sm_100a / sm_103a 0 / 0 unchanged; its sm_107a build reads 18 / 43,
the A/B arm only). Board perf (CUDA events -- torch's CUPTI predates cc 10.7 --,
L2 flushed per trial, one process per slot, A/B/A x3 with a control pair whose
spread is <= 0.04 %, the 4x1 body as the in-session control; SM clock sampled
~2016 MHz under load): causal stage 2 600 us (4x1) / 753 us (profile 1) / 719 us (profile 2), whole backward 1308 / 1452 / 1428 us; dense stage 2 1110 / 1690 / 1355 us, whole 2144 / 2741 / 2395 us: profile 2 at 1.20x / 1.09x (causal stage 2 / whole) and 1.22x / 1.12x (dense); profile 1 at 1.25x / 1.52x stage 2 (causal / dense). **The flip gate
(profile 2 <= 1.00x on stage 2 AND the whole backward, both masks) is NOT met, so
`BWD_D256_2X2` stays False and the 4x1 body ships on Rubin**; the twin is a
validated, bitwise-equal A/B arm. The 4x1 body's 600 us causal stage 2 at the
sampled ~2016 MHz matches PR #1318's 546 us at 2376 MHz within the clock ratio.
ᶜ Dense I/O with legal native TMA strides, including BHSD-contiguous dO, is
addressed directly by the prepared SM100 path. Each operand must have unit D
stride and positive outer strides aligned to eight half-precision elements.
Q/K/V/O/dO and gradient pointers must also be 16-byte aligned; misaligned
bindings are rejected before launch. Dense layouts outside the native stride
requirements retain workspace staging (one copy in, and one back out for a
gradient). That existing staging path uses prepared bitwise gather/scatter
entries around the same pointer chain. It retains no tensor-ABI compiler,
per-call tensor staging views or DLPack wrapping; current pointers, Int64
strides, workspace storage origins and stream bind on each call. Potentially
overlapping gradient outputs preserve ordered copy-back. Native dense and
packed layout admission is unchanged.
ᵉ **Any S_q and S_kv, not just tile multiples.** The engine rounds the COMPILE
shape up to the tile (256 in q, 128 in kv), lets stage 2 compute the tail and
mask it, and hands stage 3 a real-extent slice so the padding never reaches a
GEMM's M/N/K. Note this is the UNIFORM length only -- a per-batch
`seq_len_q/kv` padding mask is still declined (`padded=False`).
ᶠ dK/dV are accumulated as one partial per Q head and folded onto the KV heads
by the shared `dkv_reduce` kernel (deterministic, fixed-order fp32). dQ runs ONE
GEMM per head chunk when the chunk is a whole number of GQA groups (the dQ record's
`b_head_group = group` indexes the shared K head by `h // group`; `api_dsl.DQ_SINGLE_LAUNCH`);
THD, a budget-limited chunk that splits a group and the `DQ_SINGLE_LAUNCH = False` twin
run one dQ GEMM per group member / per head -- bitwise the same dQ (the same k walk per
output tile). The head chunk is any divisor of H_q under the workspace budget.
ᵈ Top-left AND bottom-right. The empty kv range bottom-right admits needs no
special path: every ring is per-kv-iteration, so a zero-trip loop fires nothing.
Causal also skips whole kv tiles above the diagonal (~44 % of them at S=2048),
which means those workspace tiles are never written and stage 3 must trim its K
range to match — a correctness requirement, not just an optimization.
ʰ **THD / ragged backward** (`sdpa_bwd_sm100`, the ᵇ row). Q/K/V/O/dO and the gradients are PACKED
`[1, T, H, D]`; the S/dS workspace is **row-blocked** (each sequence owns a
128-row-aligned block, columns uniform at `pad(S_kv_max)`), which is where the
memory win over the dense `B · S_max²` rectangle comes from. A setup launch
publishes the metadata, the per-sequence block offsets and the clipped output
descriptors — all device-side, so nothing reads a length on the host. Both
packed Stats layouts the FROST forward emits are read (token-major `(T, H)` and
head-major `(1, QH, head_stride)`, the latter with a head stride wider than the
packed total); a DENSE per-batch Stats on a ragged graph is declined, because its
stride reads as head-major over storage that is not packed.
The **causal family is served** (top-left, bottom-right, sliding window, right-band
widening): stage 2 already masks from the per-sequence metadata lengths, including
a per-sequence bottom-right diagonal `S_kv[b] − S_q[b]`. Stage 3 renders the
**same per-sequence causal K-trim as the dense path** under THD (the template's THD
arm `_thd_causal_k_range`: every bound a sequence-relative row; the bottom-right diagonal is read
per sequence from the setup launch's metadata, `MatmulTemplateParams.thd_causal_bottom_right`;
`api_dsl.THD_STAGE3_TRIM`), and the blocked workspace is still zero-filled per
execute — the 512-row cluster M tile straddles two 256-row stage-2 blocks, so the
fill is the correctness and the trim the optimization, exactly as on the dense
path (where the trim forced off measured **−20 %** on the whole backward, A/B/A,
B=1 H=128 S=8192 d=512 bf16 causal: ~259 → ~207 TFLOPS). The one THD causal arm
still rendered **untrimmed** is the **sliding window** (`sliding_window_length`):
the window edge's per-sequence bound was not validated, so a windowed packed graph
reads every k tile of the group and pays for the ones the window skipped; the
**right-band widening** renders untrimmed too (the template's THD arm takes no
constant shift).
Pins: `test_sdpa_bwd_thd_sm100.py::test_graph_thd_causal_stage3_is_trimmed_per_sequence`
(record spy), `test_graph_thd_causal_trim_is_bitwise_the_untrimmed_rendering`
(`THD_STAGE3_TRIM = False` twin, int16 views), the PTX md5 list
`renderings/md5_stage3_sm100a.txt` (`*_thd_causal*`).
A sequence that is empty on ONE side only (`S_q[b] == 0` with `S_kv[b] > 0`, or
the reverse) is served and returns exactly zero for that sequence: its GEMM's
reduction axis is empty, so no MMA initialises the accumulator, and the epilogue
stores zeros rather than TMEM residue.
**GQA / MQA is served** as well: the stage-3 dK/dV GEMMs write one partial per Q
head over the PACKED kv axis and the shared reduce folds the group onto the KV
heads, so the packed path now matches the dense one feature for feature. With
that, BOTH THD conjunction flags (`thd_causal`, `thd_gqa`) are **deleted** rather
than set True — a conjunction flag is transitional by design and earns its place
only while some row genuinely cannot serve the pair.
What remains is a property of the path, not a feature it declines: a
non-BSHD-physical layout (the packed path has no staging copy), and a graph that
does not declare **`max_total_seq_len_q`/`_kv`** — those are REQUIRED here,
because `scratch_workspace_bytes()` is a build-time function and the blocked row
count comes from the packed totals before any buffer exists.
ʲ Not a gap in this engine: `cu_seq_len_q/kv` is a **forward-only** graph
attribute. `SDPA_backward_attributes` has no such input port and
`pygraph.sdpa_backward()` no such keyword, so no backward row can claim it and
none could be tested. Ragged backward lengths arrive as per-batch `seq_len_q/kv`.
⁹ `thd_d_shapes` is an **exact** membership test, not an envelope: the
SM100 per-tensor FP8 and MXFP8 rows both list
`{(128,128), (192,128), (256,256), (512,512)}`, so d=64 **THD on FP8/MXFP8 is declined**. f16/bf16 THD rides the
envelope (`thd_d_shapes=None`) and works.
¹⁰ The per-tensor FP8 d192×d128 and d256 flavors are **floored to their exact
shapes** (`d_envelope_floors` `((192,128),128), ((256,256),255)`, mirrored in
`fwd/api_dsl._SM100_FP8_ENVELOPE_FLOORS`): with d_qk zero-padded into d192×d128
the kernel's output is wrong (4–19 % of elements off by O(1) in `test_mhas_v2`),
and the d256 padded envelope is run-to-run nondeterministic on long causal
e5m2/GQA/sink graphs. So an FP8 graph with head dims in (128, 256) that is not
exactly (192, 128) or (256, 256) is declined by the engine and takes the classic
backend verdict — the same envelope the C++ `validate()` always enforced. The
d128 flavor's ×16 envelope and the d512 band² are unaffected. Lift the floors
once the kernels' padded paths pass the battery.
ᵍ **`sdpa_bwd_sm100_mxfp8` only — `sdpa_mxfp8_backward()` with E4M3 payloads,
d_qk = d_v = 256 exactly, fp16/bf16 `o_f16`/`dO_f16`/dQ/dK/dV.** Serves MHA /
GQA / MQA, any fixed S_q / S_kv (the kernels mask tile tails; S_q = 1 works),
dense and top-left causal, and honors `use_deterministic_algorithm` (both
kernels own their output tiles — nothing accumulates through atomics).
BSHD-physical storage only: the kernels derive head and batch strides rather
than reading them, so a BHSD-contiguous graph is declined, not staged. Declined
outright: E5M2, bottom-right / right-widened / sliding-window masks, padding,
THD, bias / dBias, sink / dSink, and any of `amax_dQ/dK/dV` requested as a real
output (the kernels write half-precision gradients and produce no amax).
Numerics: dS is quantized in-kernel with an online per-32-block E8M0 scale;
P with a fixed 2⁻⁸ descale (cuDNN's MXFP8 convention). Cost to know about: the
scale-factor repack in front of the kernels (eleven launches, ~1–2 % of the
backward) and its workspace (about one payload-equivalent of bytes).

---

## SM107 (Rubin, cc 10.7–11.9)

The half forward row is offered by default. On exact cc10.7 it leads the backend
when the shared paged D128 or nonpaged D128/D192-V128 selector proposes a
native packed split, or for the qualified packed paged prefill family below.
Other half graphs remain backend-first; the opt-in flag still ranks FROST first.
Selection uses declared bounds for override graphs and never reads device
lengths. The MXFP8 forward row is offered by default too (2026-10) and leads the
backend on exact cc 10.7 for every graph it admits (`placement._place_sm107_mxfp8`),
a qualification verdict: the backend's cc 10.7 MXFP8 engines mis-report Amax_O (the
harness xfails them), their heuristics crash the process while planning any
single-query MXFP8 graph without a sink (dense and THD, Stats on or off; cuDNN
9.26.0.51 and 9.27.0.28; `sdpa/fwd/backend_guard.py` keeps the backend out of
planning there on every known build -- `SQ1_MXFP8_PLANNING_CRASH_FIXED_IN = None`: cuDNN 9.28.0's heuristics
plan the 21-contract matrix but building the backend's BHSD single-query plan killed the test process on the cc
10.7 CI lane (2026-10-08), and the detector re-measures the matrix through plan, check_support and build on
every run of that lane; an
explicit backend pin, `create_execution_plan` with a backend engine id, on such a graph is
a typed decline too; `test_mhas_v2.py::test_sdpa_mxfp8_cc107_backend_planning_crash_guard_is_current_L0`
walks the 21-contract trigger matrix -- plan, then the backend's own plan through check_support and
build -- with the guard disabled in a child interpreter restarted after every crash (the per-contract, per-stage
record is printed under `measured` in the run's summary) and fails when the installed backend gets every
contract through cleanly -- the signal to re-measure and record the
version), and their d256 / d512 MXFP8 plans fail to build (NVRTC
compilation failure on both engines, same two backends), so without this row those
flavors have no provider on cc 10.7. THD/varlen is served at d256 only (#1488's
`thd_d_shapes`); a THD MXFP8 request at another head dim declines with `THD (ragged) rides the
packed native-tile leg on this engine (shapes [(256, 256)]); the head-dim envelope is dense-only`,
paged F8_128x4 pools with dense queries are served on d128 / d256 (the paged MXFP8 pools section
below), a THD query over pools declines with `paged MXFP8 KV with THD queries is not wired`, a
page that is not a multiple of 128 rows with `paged MXFP8 KV needs page_size to be a multiple of
128`, and a planning error names every side (the backend's text and each python engine's reason). The per-tensor FP8 forward row and the backward rows remain
opt-in.

SM107 half paged D128 causal THD candidates prefer the existing GQA4/GQA8
packing, sharing K/V across heads in the same Q tile. Default placement also
admits the qualified FP16/BF16 HND/page16 bottom-right family (B8–64, Hq4–64,
Q64–128, KV2K–32K, fixed declarations, no sink/window/right-band).
The paged split selector uses the same first-wave budget through B64/KV32K
on SM107; selected native splits retain priority. Other contracts keep their existing
placement; no kernel coverage or public knob is added.

These engines require a CuTe DSL build with the `sm_107a` target. Public
4.7.0 meets the shared DSL floor but lacks this target; graph admission and
standalone support checks decline it with the installed version before compile.
SM100/SM120 keep their existing 4.7.0 floor.

Engines: `sdpa_fwd_prefill_sm107` (f16/bf16), `sdpa_fwd_prefill_sm107_fp8`
(per-tensor FP8) and `sdpa_fwd_prefill_sm107_mxfp8` (block-scale) forward;
`sdpa_bwd_sm107` (f16/bf16), `sdpa_bwd_sm107_fp8` (per-tensor FP8 E4M3,
`sdpa_fp8_backward`) and `sdpa_bwd_sm107_mxfp8` (block-scale MXFP8 E4M3,
`sdpa_mxfp8_backward`, see ᵐˣ) backward — **d = 256 exactly**, see ᵇ — and, since
2026-10-01, `sdpa_bwd_sm107_d512` (f16/bf16, **d ∈ (256, 512]** envelope-served
on 512-wide tiles, opt_in, the 2x2-datapath stage 2 at the 8-stage ring arm, see
ᵇ³). Every other backward graph on the Rubin line (d < 256; d > 256 outside the
f16/bf16 d ∈ (256, 512] band; a quantized graph off d = 256) gets a typed
`cudnnGraphNotSupportedError` at plan creation — the backend has no d = 256 MXFP8
backward kernel on smVersion 1070 either, so the MXFP8 row is that graph's sole
provider (MEASURED 2026-09-29).

The f16/bf16 row is a separate engine from `sdpa_fwd_prefill_sm100` (which stops
at cc 10.6) because the lowerings diverge: the Rubin kernels build **version-1
tcgen05 SMEM descriptors**. A version-0 descriptor's `start_address` field is 14
bits — a 256 KiB window — and Rubin's 327 KiB per-CTA budget puts the d512
flavor's P transfer ring at exactly 256 KiB, where a version-0 descriptor wraps
to offset 0 and the MMA multiplies the untouched O staging slab (O comes out
exactly zero, no crash).

The bounded half THD split and paged paths also reuse the SM100 single-CTA
and paged pipelines, compiled natively for SM107. Their SMEM layout stays
within the version-0 descriptor window; this does not change the wider Rubin
sibling kernels. Dense d128 half graphs at `TILE_CGA_M=1` (the 128-row decode
tile, 224 KiB) and packed dense d128 at cga2 (the shared prefill body, 192 KiB)
reuse them too (issue #1472).

All three lines now carry d192×d128 (`sm107/prefill_d192_d128_{f16,fp8,mxfp8}.py`
— the d128 body with `make_cfg_d192` / `make_cfg_d192_mxfp8`), so a d=192 graph
lands on its NATIVE kernel in every dtype family. The FP8 row carries that
flavor's envelope **floor** (128) with it: the floors are arch-independent
(`fwd/api_dsl._SM100_FP8_ENVELOPE_FLOORS`), so d_qk zero-padded into d192 stays
declined here exactly as on SM100 — that is a kernel property, not an arch one.

The MXFP8 d192 flavor is **cga2 only** where SM100 serves it at both widths, and
that is a descriptor constraint rather than a tuning choice. At cga1 the K/V
rings are not halved, so its four scale-factor tiles start at 256–278 KiB —
past the 256 KiB version-0 tcgen05 descriptor window, where `start_address`
wraps to 0 and the UTCCP copies Q *data* bytes into the SF TMEM columns
(`LSE = +inf`, `O = NaN` on 100 % of cells, at every shape). The row expresses
this by leaving `(192, 128)` on its default `cgas={2}`, and the kernel body
raises at import if handed cga1 anyway. Lifting it needs `DESC_VERSION` derived
from the layout **and** the version-1 SF path validated on Rubin — not free:
putting the d128/d256 MXFP8 tiles on descriptor version 1 turned 21 green tests
red (2026-09-08).

| Feature | d64 (GPT-OSS)<br>FPROP | d128 (Llama)<br>FPROP | d192×d128 (DSv3 MLA)<br>FPROP | d256 (Qwen)<br>FPROP | d512 (DSv4)<br>FPROP | d256<br>BPROPᵇ | d512 (DSv4)<br>BPROPᵇ³<br>d ∈ (256, 512] |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **Data types** | | |  | | | |  |
| FP16 / BF16 | ⚠️ⁱ | ✅ | ✅ | ✅ | ✅ | ✅ᵇ |  ✅ᵇ³  |
| FP8 E4M3 / E5M2 (per-tensor) | ⚠️ⁱ | ✅ | ✅ | ✅ | ✅ | E4M3 ✅ᵇ · E5M2 ❌ |  ❌  |
| MXFP8 | ❌ | ✅ | ✅ˣ | ✅ | ⚠️ⁱᵛ | ✅ᵐˣ (E4M3 payloads, bf16 gradients, **no amax**) |  ❌  |
| O dtype ≠ QKV — **quantized graphs only** (fp16/bf16/fp8 out) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ᵇ (fp8 row: dQ/dK/dV e4m3, bf16 or fp16) · ✅ᵐˣ (mxfp8 row: bf16 only; fp16 declined, follow-up) |  —  |
| Block-scaled O (`sdpa_fp8` / `sdpa_mxfp8` + `sf_o`): FP4_E2M1 O + E4M3 scale per 16 d, or E4M3 O + UE8M0 scale per 32 d — dense/unsplit/unpacked only | ❌ | ✅ | ❌ | ❌ | ❌ | — |  —  |
| Head-dim envelope | none — runs the d128 kernelⁱ | f16 ×8 · fp8 ×16 | f16 ×8 · fp8 exact only (floor 128) | f16 ×8 · fp8 ×16 (floor 255) | f16 ×8 · fp8 ×16 (floor 256) | none — exact (256, 256) |  f16 ×8 — envelope-served on 512-wide tiles (floor 256, exclusive)ᵇ³  |
| **Layout** | | |  | | | |  |
| BSHD | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  ✅  |
| Arbitrary dense stride order (`dense_flex`) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |  ❌ᵇ³  |
| THD / ragged (packed varlen) | ✅ᶻ | ✅ᶻ | ✅ᶻ | ✅ᶻ | ✅ᶻ | f16 ✅ᵇ · fp8 ✅ᵇ · mxfp8 ✅ᵐˣ (the quantized rows standalone and graph: the `sdpa_fp8_backward` / `sdpa_mxfp8_backward` nodes carry `max_total_seq_len_q/kv`; a ragged quantized graph without them — or through a pybind extension built before the attribute, which cannot declare them — is a typed decline) |  ❌ᵇ³  |
| `cu_seq_len_q/kv` prefix sums (THD only) | ✅ᶻ | ✅ᶻ | ✅ᶻ | ✅ᶻ | ✅ᶻ | ❌ʲ (the backward node has no such port; the standalone adapter takes `(B+1,)` prefixes) |  ❌ʲ  |
| **Masks / features** | | |  | | | |  |
| Causal (top-left) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  ✅  |
| Causal bottom-right | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ any S_q on all three rows (every body takes `seqlen_q_real`)ᵇ ᵐˣ |  ✅  |
| Causal right-band widening | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |  ✅  |
| Sliding window (left) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  ✅  |
| Padding mask (`seq_len_kv`) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ dense graph form (carries `seq_len_q`); per-batch `seq_kv_lens` on every row's standalone adapter (the K / V rows past a length must be finite)ᵇ ᵐˣ; a RAGGED padded graph (THD) is served on all three rowsᵇ ᵐˣ |  ❌  |
| Padding mask + stats (per-batch LSE trim) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |  ❌  |
| Dense padded-Q trim (O:=0, LSE:=−inf) | ✅ | ✅ | ✅ | ✅ | ✅ | — |  —  |
| Attention sink (at `S_q == 1`: f16/bf16 ✅ — dense / padded / ragged via test_sdpa_sq1_cc107_half_L0's sink draw on the native bodies, paged THD via test_mhas_v2.py's P2 block on the shared bodies ᵖˢ; per-tensor FP8 ❔ — see SM100 ˢ; MXFP8 ✅ — `test_mhas_v2.py::test_sdpa_mxfp8_fwd_default_walk_cc107_L0` draws `s_q == 1` with an e4m3 sink on every exact flavor against the MXFP8 reference incl. `Amax_O`, cuDNN 9.26.0.51 on the cc 10.7 board and 9.28 on the Rubin CI lane, 2026-10-08) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |  ❌  |
| Base-2 stats (`stats_use_log2`) | ❔ | ❔ | ❔ | ❔ | ❔ | — |  —  |
| GQA / MQA (`H_q ≠ H_kv`) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  ✅  |
| **Decode tile** (f16/bf16, dense Q, `S_q · G` packed rows within the Rubin route: ≤ 16 on the 16-column tile, (16, 32] on the 32-column tile, a packed MTP step of up to two token units of itᵐᵗᵖ; dense or paged) | ✅ dense Q (the shared `sm100/decode_d128_f16.py` compiled for sm_107a, `S_q × packed group ≤ 128`, issue #1472 -- the d128 decode tile row below; THD keeps the prefill tile) | ❌ (prefill tile) | ❌ (prefill tile) | ✅ᵈʳ **native** (`sm107/decode_d256_f16.py`, swap-AB; whole-group PackGQA + dense / paged split-KV; the 32-column tile in token units for MTP `S_q ≤ 4`ᵐᵗᵖ; **default placement since cudnn-frontend 1.32.0**: a PAGED dense-Q decode graph at `S_q = 1` -- groups 4 / 6 / 8 / 12 / 16, ≥ 4 units (`b · h_kv`) and ≥ 2¹⁶ KV tokens in flight, declared KV 1k-128k, a page pool at page 16 / 32 / 64 in either in-page layout (both measured), per-request KV lengths (the padding mask), no mask / sink / Stats / gate -- LEADs the backend by default (`placement._sm107_paged_d256_decode_tile_leads`; the measured band in the default-placement paragraph of the SM107 section); since the follow-up to that arm, group 32 on 64/2 at page 16 LEADs from 4 units and 2¹⁷ KV tokens in flight and the MTP rows (`S_q` 2-4 on 24/2 and 32/2, the bottom-right diagonal) LEAD over their whole measured box at page 16 bf16 (the unmeasured 32/1, 12/1 and 16/1 geometries trail) -- the two further bands of the same paragraph; what stays unmeasured is in the gaps table) | ❌ (prefill tile) | — |  —  |
| PackGQA | fp8 only | fp8; half dense GQA (shared SM100 d128 bodies, group divides 128, not with the pre-folded scale); half paged THD / nonpaged split THD | ❌ | half paged unsplit THD (CGA2); and the d256 decode tile (`pack_gqa_d_shapes` ∋ (256, 256) for both routes: on the tile the WHOLE group within the Rubin route -- `S_q · G ≤ 16` on the 16-column tile (24/2 packs 12 live rows + 4 zero tail rows), (16, 32] and a packed MTP step of up to two token units on the 32-column tile (24/2 at `S_q = 4`: two units of 2 tokens × 12 heads)ᵐᵗᵖ -- dense or paged, whatever `tile_m`; the dense d256 prefill kernel runs unpacked and a packed dense graph past the route is a typed decline)ᵈʳ | ❌ | — |  —  |
| Split-KV | f16/bf16 + per-tensor fp8, envelopeᵛⁱⁱ | denseᵛⁱⁱ; half THD | denseᵛⁱⁱ; half nonpaged THD | half paged THD, CGA2; the decode tile's dense (unpadded) / paged split, packed or not (fp32 partials + the shared combine, `choose_decode_tile_split_kv`)ᵈʳ | ❌ᵛⁱⁱ | — |  —  |
| d128 decode tile (`TILE_CGA_M=1`, dense; the shared `sm100/decode_d128_f16.py` compiled for sm_107a; `S_q × packed group ≤ 128` proposes it first; past one tile the dense SM100 and cc 10.7 rows list both widths and lead with the decode tile while its waves stay within 1.6x the cga2 cluster's below a 16k cache, issue #1518; THD and the pre-folded scale keep the prefill tile) | envelope (d64 via d128) | ✅ (issue #1472) | ❌ (prefill) | ❌ (prefill) | ❌ (prefill) | — | — |
| Paged KV (half THD; the attention sink composes ᵖˢ -- keyless rows O := 0 / LSE := sink; dense paged queries ❌ except decode-shaped dense Q on the cc 10.7 d256 decode tile ᵈʳ; MXFP8 pools with dense queries, sinks compose)ᵖ | envelope | half THD ✅ (sink: unsplit, PackGQA on / off, cga1 / cga2, HND / NHD, page 8–1024) · mxfp8 pools ✅ (page 128 / 256 / 384 / 512, HND / NHD, cga2) | ❌ | half THD ✅, including unpacked split (sink: unsplit, PackGQA on / off) · mxfp8 pools ✅ (page 128 / 256 / 384 / 512, cga1) · decode-shaped dense Q on the d256 decode tile ✅ (sink, PackGQA and split included)ᵈʳ | ❌ | — | — |
| Fused epilogue gate (sdpa virtual `O_v` → `mul(O_v, sigmoid(G))`, `G = (B, H_q, S_q, D_v)`; graph tail + standalone `sample_gate`)ᵛⁱⁱⁱ | ❌ | ❌ | ❌ | f16/bf16 ✅ (dense; and THD / packed sequences on the d256 prefill through the STANDALONE adapter -- `SdpaFwdDslSm100(thd=True, sample_gate=...)`, the gated attention block's packed `fuse_gate`; the graph tail stays dense-onlyᵛⁱⁱⁱ; + over PAGED pools on the shared SM100 d256 half body compiled for sm_107a through the STANDALONE adapter -- packed queries (the gated attention block's EXTEND form, `kv_source='paged'` + `fuse_gate=True`), PackGQA on or off, and dense paged queries past the decode tile; unsplit; the graph rows keep declining paged x gate outside the decode tile's splitᵛⁱⁱⁱ; + on the d256 decode tile's SPLIT the gate is applied by the combine: dense or paged, packed or not; not under execute-time shape overridesᵈʳ) · fp8 ✅ (bf16 G; dense only) · mxfp8 ✅ (bf16 G; a gated e4m3 O is unscaled; dense only) | ❌ | — |  —  |
| Index-list sparse attention (`QsaSpec`: per-query lists of 4-token KV blocks + the open tail; frontend-only -- the gated attention block's sparse stage and the standalone adapter, no graph form; the `SM107 d256 index-list sparse forward` section below) | ❌ | ❌ | ❌ | f16/bf16 ✅ (dense BSHD or packed THD sequences -- through the gated attention block and the standalone adapter --, or paged K / V pools through the standalone adapter; per-batch KV lengths; GQA group 1..16; `top_k` 4..512; the decode form through the standalone adapter and, since 2026-10-14, through the gated attention block's DECODE / extend-with-prefix mode -- `kv_source="paged"` on a dense `[B, S_q]` `QsaSpec` block over the pools, `block_table` / `kv_lens`, one list per sequence at `S_q <= 4` and one per row past it --: bottom-right positions, split-KV with the fp32 combine, the gate and the LSE) | ❌ | ❌ |  ❌  |
| Optional stats (LSE store compiled out) | ✅ | ✅ | ✅ | ✅ | ✅ | — |  —  |
| Bias | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |  ❌  |
| Ragged `S_kv` (non-multiple of 128) | ✅ⁱˣ | ✅ⁱˣ | ✅ⁱˣ | ✅ⁱˣ | ✅ⁱˣ | ✅ᵇ (any S_q / S_kv on every row, every mask; padded to 128 / 256; the mxfp8 row also re-stages the scale-factor pads zero-filledᵐˣ) |  ✅ (any S_q / S_kv; padded to 256 / 128 and masked)  |
| FP16 softmax exponent (`sdpa(softmax_precision=HALF)` op attribute) | ❔ⁱⁱⁱ | fp8 ✅ · mxfp8 ✅ (f16x2 exponent arm) | fp8 ✅ · mxfp8 ✅ | fp8 ✅ · mxfp8 ✅ (f16 pair-sum denominator without Stats; O differs from the Stats build within the family bound) | fp8 ✅ · mxfp8 ✅ (same) | — |  —  |
| Pre-folded attention scale (`sdpa(attn_scale_prefolded=True)` op attribute: Q carries attn_scale · log2 e, no in-kernel scale; with HALF and no Stats the shift and f32→f16 convert fuse) | ❔ (d128 envelope) | mxfp8 ✅ · f16/bf16 ✅ · fp8 ❌ (descale fold) | mxfp8 ✅ · f16/bf16 ✅ · fp8 ❌ | mxfp8 ✅ · f16/bf16 ✅ · fp8 ❌ | mxfp8 ✅ · f16/bf16 ✅ (role-split and 2x2) · fp8 ❌ | — |  —  |

ᵈʳ **d256 decode tile on Rubin (`sm107/decode_d256_f16.py`).** The SM100 swap-AB
tile (SM100 ᵈ: KV tokens on the MMA M axis, 16 packed Q rows on N, one cta_group::1
CTA per (KV-head group, batch, split) unit, 6 warps) ported line-for-line as a Rubin
sibling: the same `CfgD256Decode` record re-validated by `config_sm107.make_cfg_d256_decode`
against the 320 KiB usable carveout (the tile's 209 / 226 KiB at N_Q = 16 / 32 fit the
STANDARD 227 KiB carveout, so no oversized-SMEM launch and no L1 shrink) and the 256 KiB
version-0 tcgen05 descriptor window (`DESC_VERSION` derived from the layout: 0, every
MMA-operand slab starts below it); `SPIN_RING_WAITS = False` until the tile's own A/B.
Routed by the Rubin route (`SdpaFwdDslSm100._decode_q_tile`, `engines.d256_decode_tile_selected`
through `engines.decode_d256_q_tile_for_row`, `config_sm107.decode_d256_q_tile` -- the SM100 rule
`config_sm100.D256_DECODE_ROUTED_MAX_Q_ROWS` widened on cc 10.7, footnote ᵐᵗᵖ): f16/bf16 (256, 256)
graphs with dense (non-THD) queries whose `S_q · G` packed rows fit 16 (one unit of the 16-column
tile, as on SM100), or fit the 32-column tile in one unit (rows in (16, 32]), or -- a PACKED group
only -- in two token units of it (the MTP step: 24/2 at `S_q = 3` / `4`), without the pre-folded
scale (the d256 prefill kernel keeps it) -- and a graph WITH the fused epilogue gate only through a SPLIT (`split_kv >= 2`): the
tile has no gate seams, so the gate moves into the split combine (below); unsplit, the gated
graph keeps the d256 prefill kernel's fused epilogue (and so does a gated graph whose caller
enabled execute-time shape overrides: the combine binds G to the declared (B, H_q, S_q, D_v),
so its split is a typed decline naming the overrides, never a bind failure at execute). Served on it: dense padded / unpadded and PAGED caches
(page 16 / 32 / 64 / 128, NHD / HND — the row's paged THD-only scope is lifted for exactly this
form, sink included: the tile walks the block table itself and folds the sink per Q row), every
mask the d256 row serves, the dense padded-Q trim, Stats natural or base-2, the CUDA-graph
replay under the D2H detector. **PackGQA and split-KV, as on SM100**: the row's
`pack_gqa_d_shapes` carries (256, 256) for exactly this tile -- the WHOLE group packed into
the 16-row Q tile (`HEADS_PER_TILE = QH_PER_KH`; 24/2 = 12 live rows + 4 zero tail rows per
(batch, KV head) unit, one CTA instead of twelve), dense or paged, whatever `tile_m` (the
prefill tiles' divisibility rule does not apply; the d256 prefill kernel runs unpacked, so a
packed d256 graph past the route -- 24/2 at `S_q = 5`, a third token unit -- is a typed decline;
the 32-row MTP shape, 32/2 at `S_q = 2`, that used to ride the tile UNPACKED now packs on the
32-column tile in one unit, ᵐᵗᵖ) -- and the decode tile's dense (unpadded) / paged split-KV through
`choose_decode_tile_split_kv` (fp32 partials into the split-major workspace, the shared
`sm100/split_combine`; the eager-safe choice leads, the captured caller's optimum follows,
`is_cuda_graph_replay_expected=True` swaps them; a padded dense cache and a sink keep the
shared no-split rules). Evidence on cc 10.7 (204 SMs, `test_sdpa_fwd_decode_d256_sm107.py`):
the SM100 case list re-run with the SM100 policy assertions (packed 8:1 / 16:1 / 32:2 / 12:1,
split 2 runner-up at the b=32 serving shape, split 8 at b=8, split 16 on the deep-split
shape, CUDA-graph replay at split 4 under the D2H detector), **paged == dense bitwise** on
the same tokens, **packed == unpacked bitwise** (24/2), and **split == unsplit within a
DERIVED budget**: per element one ulp of the output dtype at its own binade (the single cast
of two differently-associated fp32 sums -- the combine renormalises each partial by
`exp(lse_s - lse)`, the unsplit path rescales its running accumulator per tile) plus
`2 · eps_P · Σ_j p_j |v_j| / l` (the tile quantizes P to the half IO dtype at the running max of
ITS OWN split, `eps_P = 2^-8` bf16 / `2^-11` f16; each path within `eps_P ·` that mean of the
exact value, so their DIFFERENCE within twice it -- the term that lets a near-zero element differ
by far more than an ulp of its tiny magnitude); measured max |ΔO| = exactly one output ulp of
the top binade (`2^-9` bf16 / `2^-12` f16 over pages, `2^-11` bf16 on the unpadded dense cache),
~27-45 % of the elements differ, the fp32 LSE within fp32 rounding -- pinned as the derived
budget, never widened; the paged split evidenced at page 16 and page 64 HND, the unpacked split
on an MHA graph the model splits 16 ways. **The fused epilogue gate rides the tile's SPLIT with the
gate IN the combine** (`sm100/split_combine.compile_ptr(gate=True)`,
`SdpaFwdDslSm100._gate_in_combine`, `d256_decode_tile_selected(..., split_kv)`): the tile is
compiled UNGATED (`TemplateParams.epilogue_gate` False -- it has no gate seams) and writes its
fp32 partials as any split does; the combine applies `O *= sigmoid(G)` to the fp32 MERGED value
before the single cast with the gated prefill kernels' own arithmetic (`h = acc · inv_den / 2`,
`h · tanh(g / 2) + h`: one FMUL2, one MUFU.TANH, one FFMA2 per element, the dead-row SELECT per
element after the fma) -- FROST's ONE-rounding convention, so a gated split plan and the gated
unsplit plan (the d256 prefill kernel's epilogue) of one graph differ only by the attention's
summation order: pinned within the same derived budget with the P term scaled by `sigmoid(g)`; a
gated graph whose plan does not split keeps the prefill kernel (dense only: the paged prefill
kernel has no gate, so a paged gated decode graph lists split plans alone), packed or not over a
dense unpadded or a paged cache (`packed == unpacked` bitwise, gated). The unfused references --
vLLM's split merge and the gated attention block's torch reference -- round the merged O to the
IO dtype BEFORE the fp32 gate (two roundings); against that convention the pass is pinned within
`ulp(out) + sigmoid(g) · ulp(O) / 2 + |O| · 2^-11` (the intermediate rounding plus the approximate
tanh), and BITWISE where the arithmetic is identical: `sigmoid(+1e4) = 1` (the ungated O),
`sigmoid(0) = 1/2` (the ungated O halved), `sigmoid(-1e4) = 0` (exactly 0), LSE bitwise the ungated
one in every case; the gate reaches the combine through a per-launch binding (the native dense
binder knows no gate on a split and calls the combine with a fixed frame), captured and replayed
under the D2H detector. Past the route (ᵐᵗᵖ: a third token unit, an unpacked step above 32
tokens, a group wider than the tile), THD queries and the quantized families stay on the prefill
tiles.
**Default placement (cudnn-frontend 1.32.0; `placement._place_sm107_f16` ->
`_sm107_paged_d256_decode_tile_leads`).** FlashInfer's cuDNN decode graph shape -- a PAGED dense-Q
d256 graph at `S_q = 1` with the padding mask, a page pool, no causal mask / window / sink / Stats --
LEADs the backend's default plan by default inside the measured band: GQA groups 4 / 6 / 8 / 12 / 16
(16/4, 24/4, 8/1, 24/2 + 12/1, 32/2 + 16/1), `4 <= b · h_kv <= 512` units, declared KV 1k-128k with
`units × KV >= 2¹⁶` tokens in flight, page 16 / 32 / 64, f16 and bf16, the pool in either in-page layout -- the
token-major one the band measured (FlashInfer's benchmark pool: `[pages, H_kv, page, D]` dims with head stride `D`,
token stride `H_kv × D`) and the head-major one (each head's page contiguous, vLLM's `[blocks, 2, H_kv, block, D]`)
measured 2026-10-14 on the 24/2 core geometry with the same protocol on an EXCLUSIVE GPU of the same 212-SM cc 10.7 class (lock 2376 sampled: busy median 2364; cuDNN 9.27.0 build 9.27.0.21; b 1-128 x declared KV 1k-128k at page 16 bf16, 48 cells x 2 regimes): the tile 0.36-0.92 of the backend on EVERY in-band cell in both regimes (32 cells per regime, all quotable; medians 0.45 replay / 0.45 eager), the out-of-band cells mixed 0.73-2.00 as in the token-major band; per-request KV lengths (the padding mask every band cell carried). Measured 2026-10-13 on an
EXCLUSIVE GPU of the 212-SM cc 10.7 perf class (lock 2376 MHz, sampled busy median 2364; cuDNN 9.27.0
build 9.27.0.21; CUDA-graph replay AND eager; the backend's default plan and the first FROST plan pinned
by name on the SAME graph; 3 × 100 iterations with the arm order shuffled, control pair within 1 %): the
tile ran 0.43-0.90 (eager) / 0.43-0.81 (replay) of the backend on EVERY quotable cell of the band
(289 (cell, regime) pairs per regime; 0.59-0.90 / 0.67-0.81 on the frontier cells 64 × 1k, 32 × 2k,
16 × 4k, 8 × 8k; 0.53-0.63 at 512 units; page 32 / 64 0.43-0.85; the f16 twins 0.46-0.57); one step
below (2¹⁵ tokens in flight) the cells are mixed, 0.78-1.79, twelve of them launch-bound and
unverifiable; fewer than 4 units lose up to 1.79× below the 128k endpoint. The same measurement's two
further bands LEAD by default too (the follow-up to the first arm; each where the tile is strictly faster
than the backend with margin in both regimes): group 32 on its measured geometry 64/2 at page 16 (the 32-column tile in one unit, `S_q = 1`;
32/1 shares the group but was not measured and trails, as do pages 32 / 64, measured on 24/2 only):
0.27-0.79 of the backend on every quotable cell with `4 <= units <= 256` and `units × KV >= 2¹⁷` tokens
in flight -- 28 cells × 2 regimes, the f16 twin 0.38 inside; the 2¹⁶ step holds the 8 × 8k eager parity
cell, so it starts one step higher than the first band) and the MTP rows (`S_q` 2-4 on the measured geometries 24/2 and 32/2 -- 12/1 and 16/1 share the groups but were not measured at `S_q > 1` and trail -- =
the 32-column tile in up to two token units, FlashInfer's decode graph with the bottom-right causal
diagonal: 0.01-0.31 of the backend's plan -- which costs 6.8-142 ms per launch at KV 128k -- on all 576
(cell, regime) pairs over b 1-128 × KV 1k-128k, page 16 bf16, so the whole measured box incl. 2 units
leads; page 32 / 64 and the f16 twin unmeasured, backend-first). On both the tile trails FlashInfer's own
fa2 kernel on part of the band (7-9 % at the medians on group 32; 6 % at q 2, 21-22 % at q 3-4 on the MTP
rows -- the 32-column tile's issue-rate gap at 24-64 packed rows, the gaps table): FE's verdict is against
the backend's plan, the consumer's choice among its own kernels is the consumer's routing. Not measured,
backend-first: Stats, a sink, a mask or window at `S_q = 1`,
page 128, a dense (unpaged) d256 cache at `S_q = 1` (measured against the prefill body only), THD
queries, the fused gate, a paged decode without per-request KV lengths (no padding mask). Pinned by `test_sdpa_fwd_placement.py::test_sm107_paged_d256_decode_tile_*`
(the frontier cells per geometry in both regimes, RED on the pre-arm tree).

ᵐᵗᵖ **The MTP form on the Rubin decode tile: the 32-column tile in TOKEN UNITS
(`config_sm107.decode_d256_q_tile`, `D256_DECODE_ROUTED_MAX_Q_ROWS = 32`,
`D256_DECODE_ROUTED_MAX_TOKEN_UNITS = 2`).** The body of both decode twins gained a token-unit
axis (`Q_TOKEN_UNITS`): a unit covers one Q box of `Q_BOX_TOKENS = N_Q // HEADS_PER_TILE`
consecutive tokens, the host entry derives `ceil(S_q / Q_BOX_TOKENS)` units per (head group,
batch, split) from the runtime `S_q` and widens the grid's x axis by them, and every token
coordinate (the Q box, the causal / window mask, the O / LSE rows, the padded-Q trim) is offset by
the unit's first token -- the barrier table, the SMEM table and the per-unit pipeline are
unchanged (the axis moves rows between CTAs, it never changes the arithmetic). On cc 10.7 the
adapter routes: `S_q · G ≤ 16` -> the 16-column tile in one unit (as before); rows in (16, 32] ->
the 32-column tile in one unit (32/2 at `S_q = 2`, 24/2 at `S_q = 2` with 24 live + 8 tail rows,
an unpacked / MHA step of 17-32 tokens); a PACKED group with `ceil(S_q / (32 // G)) = 2` -> two
token units of the 32-column tile (24/2 at `S_q = 3` / `4`: 2 tokens × 12 heads per unit, the KV
range streamed once per unit = 2x the bytes of a single stream); anything past that (a third
unit, an unpacked step above 32 tokens, `G > 32`) -> the d256 prefill kernel, a packed request
there a typed decline. The decode split model counts the token units as streams
(`units = B × head groups × units`, the 32-column tile's per-tile cost fit on B200 -- a cc 10.7
refit is the heuristics owner's). The SM100 line routes the 32-column tile only for a launch past
`config_sm100.d256_wide_decode_launch` (`decode_d256_q_tile` with its defaults is the former rule
verbatim: 16 rows, one unit). Evidence on
cc 10.7 (204 SMs, `test_sdpa_fwd_decode_d256_sm107.py`): the 24/2 steps `S_q = 2 / 3 / 4` over
dense padded and paged caches, bottom-right, with per-batch Q lengths below `S_q` (the trim across
units), the split forms at the token-unit geometry, the 32/2 step at `S_q = 2`; **two token units
== four single-token units of the 16-column tile == the unpacked form BITWISE** (O and LSE), **paged
== dense bitwise** across the units; the token-unit axis driven at the template level too (two
units of the 32-column tile, four of the 16-column one, a half-filled second unit at `S_q = 3`,
unsplit and split 2); the other geometries inside the route -- two token units at `G = 8` (8:1 at
`S_q = 6`, the second unit half-filled, no tail rows) and at `G = 6` (12:2 at `S_q = 10`, two zero tail
rows per unit: the tail-row liveness guard's second instance), an MHA bottom-right step of 24 tokens
in one unit of the 32-column tile -- with per-batch Q lengths below `S_q` (one unit entirely past a
sequence's Q length), dense padded and paged, and the two-unit ones split by the decode model. The
measured form (Rubin cc 10.7, 212 SMs, lock 2376, the MTP table of the dense-decode PR): the
two-unit 32-column form against the prefill body, the pre-route lead (the tile unpacked) and the
single-token-unit 16-column form at `B` in {1, 4, 32}, `S_q` in {2, 4}, the vLLM / SGLang recipe
shape (`B = 4`, `next_n = 4`) among the cells. **Measured at the 24/2 geometry only**: the other
geometries inside the route (an unpacked / MHA step of 17-32 tokens, packed groups at `G != 12` in
one or two units) are the same tile with the same stream count per (head group, batch, split) and
carry the correctness cells above, but no perf cell yet -- one per class is owed to the
dense-decode PR table before any of them is quoted.

ᵇ **d=256 backward (`sdpa_bwd_sm107` f16/bf16, `sdpa_bwd_sm107_fp8` per-tensor
FP8 E4M3; `python/cudnn/sdpa/bwd/api_dsl_sm107.py`, kernels
`sm107/bprop_d256_{f16,fp8}.py`, config `bwd/config_sm107.py`).** Exact
`d_qk = d_v = 256` (the bodies hardcode the 256-wide tiles; no envelope). A
TWO-kernel chain per row: `delta = rowsum(dO·O)` → the main kernel (one cga2 pair
per 256-row kv block walks the q tiles; dV accumulates in TMEM and is stored per Q
head, dS goes to a kv-major `[B, H_chunk, S_kv, S_q]` GMEM workspace) → dK = dS·Q
and dQ = dSᵀ·K as the shared `bprop_matmul_blackwell` GEMMs over that workspace
(operand majors flipped relative to the SM100 chain's `[S_q, S_kv]` workspace; the
causal K-trim modes are not; the trim is TWO-SIDED — a sliding window bounds the
K range on its second side (`MatmulTemplateParams.causal_window`, appended,
default = no window: the SM100 renderings stay PTX-identical) — and the kernels
round every kv block's q range outward to the GEMMs' 256-row pair, so a GEMM reads
only tiles the kernel wrote and no mask needs the workspace zero-filled (kept only
for the untrimmed twin and a top-left window with `S_q > roundup(S_kv + W, 256)`;
poisoned-workspace tests `test_masked_stage3_reads_only_what_stage2_wrote` in
both suites); rendered at the d = 256 cluster tile
`MatmulTemplateParams.cgrp_tile_mn = (256, 256)` — cluster 2x1, 256 × 256 per pair,
no N padding at d = 256, accumulator double-buffered — selected by the sm107 adapter
only, the SM100 d512 renderings byte-identical: `test_stage3_tile_rows_render_their_upstream_constants`,
`test_stage3_d256_rendering_is_bitwise_the_padded_one`, `test_stage3_d256_rendering_sass_pins`;
the fp8 row writes `dS_q = e4m3(dS · scale_dP)` and renders the template's fp8 K64 arm
(`MatmulTemplateParams.dtype_qkv = DTYPE_E4M3`, 256x256x64 `F8F6F4`, idesc `k_dim=1`)
whose epilogue applies `descale_dP · descale_{q|k}` and quantizes dQ / MHA dK with
`scale_{dQ|dK}` + `amax` in place — no Q / K upcasts, half the dS bytes; the bf16-dS twin
`api_dsl_sm107.FP8_DS_DTYPE = DTYPE_BF16` is the A/B and oracle base, every fp8 accept case
runs on both: `test_sdpa_bwd_fp8_sm107.py::ds_knob`)
→ the GQA fold of the per-Q-head partials
(`dkv_reduce`, fixed order; the dK partials are fp32 under GQA on the fp8 and the
MXFP8 rows, so dK is rounded once; the fp8 row's dV partials are fp32 too, the MXFP8
row's stay bf16). Served: dense, top-left and bottom-right causal,
sliding window (left), MHA / GQA / MQA, **any** S_q / S_kv (a non-multiple of the
128-row q tile / 256-row kv block is padded through zero-filled staging copies,
`+inf` LSE rows, and the kernels' padded-mask arm at the uniform real S_kv), BSHD-
physical io, contiguous fp32 Stats. **Bottom-right causal at ANY S_q on both rows.**
The bottom-right diagonal is `S_kv − S_q` in REAL rows and every body takes the real
lengths on its ABI (`seqlen_q_real` / `seqlen_kv_real`; the fp8 body used to derive the
diagonal and the q-tile trim from its PADDED q extent and the row declined a ragged S_q
with `Capabilities.bottom_right_s_q_multiple = 128` — the field is 1 on every row now, the
former reject tests invert into accepts: `test_sdpa_bwd_fp8_sm107.py::test_reject_bottom_right_with_ragged_s_q`
self-inverts on the field, `::test_causal_bottom_right_{rectangular,ragged_kv}` and the
ragged-S_q accept cells cover it on Rubin, `test_sdpa_bwd_dsl_sm107.py::test_causal_bottom_right_ragged_s_q`
pins the f16 claim). The dS
workspace is head-chunked (and, on the
f16 row, batch-chunked) to the 8 GiB budget every sm107 row shares
(`_SM107_WS_BUDGET_BYTES`); one compiled artifact serves every
chunk. **FP8 row** = cuDNN's `sdpa_fp8_backward`: E4M3 Q/K/V/O/dO with the twelve
scalar descales / scales as 1-element fp32 device tensors (read in-kernel, never
host-folded), fp32 Stats, dQ/dK/dV in the graph's gradient dtype (E4M3 scaled by
`scale_dQ/dK/dV`, or bf16 / fp16), and the four `amax_dQ/dK/dV/dP` outputs when
requested (`amax_dP` reduces the fp32 dS before its cast, as the C++ node does).
dS rides an **E4M3** workspace (`dS_q = e4m3(dS · scale_dP)`) and the gradient GEMMs
run the template's fp8 K64 arm over it and the E4M3 Q / K payloads, undoing both
scalings in their epilogue (`· descale_dP · descale_k` for dQ, `· descale_q` for dK),
folding `amax_dQ` / `amax_dK` over the true-unit value, applying `scale_dQ` / `scale_dK`
and casting in place (dQ always, dK at MHA); dV always, and dK under GQA, take the
fold + quantize pass (`bprop_chain_common.fold_quant`: fixed-order partial sum, amax,
scale, cast -- ONE launch for both; under GQA the per-Q-head partials are fp32, so the
folded gradient is rounded once like the reference, not once per partial). The bf16-dS twin (`api_dsl_sm107.FP8_DS_DTYPE = DTYPE_BF16`: bf16 GEMMs
over exact E4M3 → bf16 upcasts of Q / K, three fold passes, `descale_dP` / `scale_dP`
bound and unused) is the A/B and oracle base. E5M2 payloads are declined (no body).
**Per-batch kv lengths and an external delta, both rows' STANDALONE adapters:**
`seq_kv_lens_present=True` + `execute(seq_kv_lens=)` — both bodies read `seq_kv_lens[b]`
under the same padded-mask arm (dead rows / batches exact zeros; on the fp8 row the amax
row gate follows the per-batch length, so a dead or shortened batch folds nothing into
`amax_dV` / `amax_dP`), bottom-right keeps the dS zero-fill for its per-batch diagonal —
ahead of every chunk on the batch-chunked f16 chain, once per execute on the fp8 chain —
with a sliding window dropped from the stage-3 trim, so the GEMMs read the plain
bottom-right band there; **the K / V rows at or past a length — and, on the mxfp8 row, their scale-factor atoms — must hold FINITE data** (the
kernels select P = 0 there, but `dS = (dP − delta) ∘ P` is `NaN × 0`; the dense rows get
finite pads from the zero-filled staging, the per-batch arm reads the caller's buffers).
`external_delta=True` + `execute(delta_tensor=)`: a contiguous fp32 `[B, H_q, S_q_pad]`
delta (zeros past S_q; under THD the packed head-major `[1, H_q, ceil128(T_q)]`, zeros
past the token capacity `T_q` — the same layout at `B = 1, S = T_q`) in place of the
chain's own `dot` launch and `delta` region — on
the f16 row the raw half-precision dot (bitwise the chain's own when the producer forms it
in `dot_do_o`'s order), on the fp8 row in **TRUE units, unscaled** (nobody applies
`descale_o · descale_dO` to a caller's delta, so it is not bitwise the row's own scaled
pre-pass of the e4m3 codes; its tests compare against an oracle fed the same delta).
Tests: `test_sdpa_bwd_dsl_sm107.py::test_adapter_per_batch_kv_lengths*`,
`::test_adapter_dead_kv_entry_is_exactly_zero`, `::test_padding_mask_graph_always_carries_seq_len_q`,
the external-delta pins of both suites. **Declined on both rows, each asserted by a
test:** graph padding masks (`seq_len_q/kv`: a padded `sdpa_backward` graph carries BOTH
lengths — the frontend requires them — and no body threads per-batch Q lengths, so the
graph form is declined rather than served while ignoring the q lengths), sink / dSink,
bias / dBias, right-band widening, `dense_flex`, decode shapes, and
`use_deterministic_algorithm` (the chain has no atomics; the claim waits on the bring-up
sweep). The bf16 d256 graph has a native backend competitor (engine 17): pin the engine
when validating or measuring. Both rows are `opt_in`. Tests: `test_sdpa_bwd_dsl_sm107.py`,
`test_sdpa_bwd_fp8_sm107.py`.
**THD / ragged on the fp8 row** (standalone `SdpaBwdDslSm107Fp8(thd=True, max_total_seq_len_q/kv=.., ...)`
and the graph tier: the `sdpa_fp8_backward` / `sdpa_mxfp8_backward` bindings take `max_total_seq_len_q/kv` as
trailing keywords and the node carries them; a ragged fp8 graph without them — or through a pybind extension
built before the attribute, which cannot declare them — is a typed decline at eligibility, `thd_declared_totals`): the f16 row's
mechanism (below, "THD / ragged") in e4m3 — packed e4m3 payloads through the
packed-total-clamped descriptors, the metadata buffer and the device claim counter,
per-sequence clipped dV stores, a kv-blocked **e4m3** dS workspace (`dS_q = e4m3(dS · scale_dP)`;
bf16 on the `FP8_DS_DTYPE = DTYPE_BF16` twin, which needs compact packed rows for its exact
e4m3 → bf16 upcast of Q / K), the fp8 K64 stage-3 arm trimmed PER SEQUENCE with its
descale / quantize epilogue, and the `fold_quant` passes bounded ON DEVICE at the live
totals (`cu_k[B]` for dV / dK, `cu_q[B]` for the twin's dQ: an unbounded fold would copy
the 0xFF-poisoned unwritten capacity tail into the caller's gradients AND into the amax).
**Every amax is the max over the packed LIVE region** — dead units, kv pad rows, q pad
columns and the capacity tail excluded at all four fold sites (the kernel's row gate and
fold values, the GEMM epilogue's per-row gate, the bounded fold passes) — and there is
ONE `scale_dP` per packed batch (the forward's one-scalar-per-operand convention over
packed tokens). An external delta is served under THD as on the f16 row (the packed
`[1, H_q, ceil128(T_q)]` form in TRUE units; no `dot` launch, no `delta` region). Declined
under THD as on the f16 row: right-band widening, bias. Tests: `test_sdpa_bwd_thd_fp8_sm107.py` (direct adapter + the graph-tier
probe), the THD pins of `test_sdpa_bwd_fp8_sm107.py`.
ᵐˣ **d=256 MXFP8 backward (`sdpa_bwd_sm107_mxfp8`, `sdpa_mxfp8_backward`;
`api_dsl_sm107.SdpaBwdDslSm107Mxfp8`, kernel `sm107/bprop_d256_mxfp8.py`, config
`bwd/config_sm107.py` `FAMILY_MXFP8`).**
ONE kernel plus scale-factor plumbing (no separate SF-repack or dequant kernels): the per-tensor fp8
body's pipeline with the F8_128x4 E8M0 scale factors dequantizing INSIDE every tcgen05
block-scale MMA (K / V / Q / dO / dO_T SF ride their operands' TMA barriers into TMEM
through UTCCP), P quantized to E4M3 with the fixed 2⁸ scale (byte 119, cuDNN's MXFP8
convention; `p_scale_log2` pinned to 8, any other value a typed decline), dV in TMEM
(bf16 TRUE-unit per-Q-head partials: the kernel stores them from its epilogue, an fp32
staging does not fit its 327 KiB SMEM, so under GQA dV carries one bf16 rounding per group
member; the block-scale dK GEMM's per-Q-head partials are **fp32** under GQA, so dK is
rounded ONCE by `dkv_reduce`, like the reference — pinned bitwise by
`test_mxfp8_gqa_dk_is_the_once_rounded_fold_of_its_fp32_partials`; the per-tensor fp8 row's
precedent), dS = attn_scale · P ∘ (dP − delta) from the
**fp32** P (never the e4m3 P — pinned by `test_ds_is_computed_from_the_fp32_p_not_the_e4m3_p`).
**dS policy P-b ships** (`config_sm107.DS_SF_POLICY_DEFAULT = DS_SF_P_B`, read by
`api_dsl_sm107.MXFP8_DS_SF_POLICY` when the adapter is built — a module constant, never a
knob): exact 1x32 block-scaled e4m3 dS in BOTH orientations — the kernel writes two payloads
(`ds_dk` per 32-q block of a kv row, `ds_dq` per 32-kv block of a q column) plus their
F8_128x4 E8M0 atoms, and dK = ds_dk·q_T / dQ = ds_dqᵀ·k_T render
`MatmulTemplateParams.block_scale` over them and the columnwise q_T / k_T scale factors,
dequantizing in the MMA (no dequant pass). The e4m3 P ring stays in TMEM and the UTCCP'd
scale factors of each q iteration alias its dead slot (no dedicated TMEM columns; one commit
ring `mb_p_sf_consumed` orders the softmax's P store after the MMA that read them). P-c (a
bf16 dS workspace, dK / dQ through the bf16 stage-3 renderings over the EXACTLY dequantized
bf16 q_T / k_T, `prepared_host.dequant_mxfp8_to_bf16_host` — the oracle twin at
`quantize_ds=False`, MORE accurate than cuDNN's 1x32-quantized dS) stays BUILT and selectable
through `MXFP8_DS_SF_POLICY = DS_SF_P_C`. The flip was a numerics change made on the Rubin
A/B (cc 10.7, 212 SMs, SM clock 2376 MHz; B=1 H=128/128 S=8192, whole row): P-b +22.0 % dense
/ +17.1 % causal over P-c at a shared 4 GiB stage-2 workspace budget, and the rows' 8 GiB budget
(`_SM107_WS_BUDGET_BYTES`, one constant for every sm107 row: 32-head chunks / 4 launches at
8K H=128 instead of 16 / 8) another +1.4 % / +9.3 %; every P-b accept cell within the
recipe below, the dS payloads and atoms bit-exact against the oracle's own 1x32 quantization
(`test_p_b_ds_payloads_and_atoms_dequantize_to_the_oracles_quantized_ds`), no dequant launch
under the default (`test_p_b_runs_no_dequant_pass_and_p_c_runs_two`). Served: E4M3
payloads (q / k / v / dO + the transposed-quantization q_T / k_T / dO_T) with **bf16**
`o_f16` / `dO_f16` / dQ / dK / dV (fp16 declined: the bf16 GEMM stores its io dtype; the
fp16 arm — a final cast pass per gradient — is a follow-up), contiguous fp32 Stats,
BSHD-physical layout, MHA / GQA / MQA, any S_q / S_kv (zero-filled staging of the
payloads AND of the scale-factor pads: the producer's SF pad bytes past S_q / S_kv are
undefined and the kernel reads them — a 0xFF there is an E8M0 NaN → NaN dV on every kv
row; both kv pad classes `S_kv % 256 ∈ (0, 128]` / `(128, 256)` are staged, poisoned-pad
RED-then-green tests `test_poisoned_sf_pads_*`), dense / top-left causal / bottom-right
causal at any `S_q` / sliding window (left) — WIDER than the SM100 MXFP8 row
(bottom-right, SWA). `descale_v` is the ROWWISE V scale in the backward (the C++ node's
own reference math dequantizes V like K); the adapter asserts that shape for `sf_v` and
byte counts only for the other six SF tensors (the node rewrites two SF strides before
lowering). **Declined, each asserted by a test:** `amax_dQ / dK / dV` requested as real
outputs — the backend's canonical MXFP8 backward graph declares them
(`test/python/sdpa/mxfp8.py`), so this is a documented parity gap (AGENTS Rule 9): the
row produces no amax; E5M2; fp16 gradients; the graph padding mask (it carries
`seq_len_q`, which no body threads); sink / dSink; bias / dBias; right-band widening;
`dense_flex`; decode shapes; `use_deterministic_algorithm` (no atomics anywhere in the
chain; the shared decline reason no longer blames "fp32 atomics" — it reads "this engine
has not claimed the two-run bitwise guarantee"; the two-run bitwise pin
`test_two_launches_are_bitwise` runs every session, the claim waits on the >= 12
fresh-process sweep across the accept matrix). Tolerances, per policy: dV under the fp8
suite's recipe on both (`assert_close_fp8_grad`, atol 0.08 / rtol 0.2 + the midpoint-flip
budget: the e4m3 P feeds BMM2); dK / dQ under the SAME fp8 recipe on the shipped P-b (the
kernel and the oracle each round dS to e4m3 per 32-block from fp32 values ~1e-6 apart, so a
midpoint flip of one dS cell moves one gradient row by one e4m3 step at the block's scale —
the fp8 row's dS-flip class, a new cell's recipe, never a widened bf16 tolerance; the oracle
runs at `quantize_ds=True`) and under the **bf16 row's** recipe on P-c
(`torch.testing.assert_close`, atol 5e-2 / rtol 5e-2, no flip budget: P-c forms dS from the
fp32 P and the GEMMs run over exactly dequantized bf16 operands, so no fp8 rounding reaches
them; `quantize_ds=False`). P-c calibrated ONCE on the bf16-dS bring-up ladder (Rubin, 212 SMs,
2026-09-30: dV max|diff| ≤ 3.9e-3, dK / dQ ≤ 1.6e-2, 0 cells outside on every shape; dS
within one bf16 rounding of the fp32-P value on every cell); P-b on the same node class
(2026-10-01: 0 cells outside the gate on every gradient of every cell, worst dV max|diff|
0.031 at max|ref| 5.78 and dK 0.0156 at max|ref| 3.45 on the head-chunked cell whose E8M0
bytes exceed 128, elsewhere ≤ 3.9e-3). **Served on the standalone adapter, as on the sibling rows:** bottom-right
causal at ANY S_q (the body derives the diagonal from `seqlen_q_real`; `bottom_right_s_q_multiple = 1`); per-batch kv
lengths (`seq_kv_lens_present=True` + `execute(seq_kv_lens=)`, the same padded arm, the K / V rows past a length and their scale-factor atoms finite;
under bottom-right the dS payloads are zero-filled once per execute and a window is dropped from the stage-3 trim); an
external delta (`external_delta=True` + `execute(delta_tensor=)`: bitwise the row's own `dot` over the `o_f16` / `dO_f16`
ports when the producer forms it in that order; its pad rows `[S_q, S_q_pad)` must be finite zeros — under P-b a 32-element
dS block straddling the q pad folds them into the real columns' E8M0 scale). The host-side dequant (`_dequant_mxfp8_to_bf16`) decodes the E8M0 byte
UNSIGNED (0..255: a block with amax > 448 carries a byte ≥ 128, legal MXFP8) — the
review of 2026-09-30 found it sign-extending the Int8 `raw_ptr` load (right magnitude,
flipped sign on every such block, finite and plausible); pinned by an all-256-bytes
Rubin cell, an `ld.global.u8` / `and.b32 255` PTX pin and the `q x 2^12` / `k x 2^12`
accept cells. `opt_in`. Tests: `test_sdpa_bwd_mxfp8_sm107.py` (host static + sm_107a
SASS pins, the row's `mismatch()` on real graphs, the Rubin accept matrix incl. top-left causal with S_kv > S_q
(tail rows exactly zero), SWA without causal, SWA 600, MQA 8/1, bottom-right + ragged
S_kv, n_q_tiles 3 / 8 at B·H = 4 with two launches bitwise, CUDA-graph replay over
poisoned outputs, head-chunked launches, poisoned SF pads at S_q 129 × S_kv 257 / 384,
and the FROST d256 MXFP8 forward's Stats fed end to end).
**THD / ragged on the MXFP8 row** (standalone `SdpaBwdDslSm107Mxfp8(thd=True, max_total_seq_len_q/kv=.., ...)`
with its MXFP8 operands, and the graph tier: the `sdpa_mxfp8_backward` binding takes
`max_total_seq_len_q/kv` as trailing keywords and the node carries them; a ragged MXFP8 graph
without them — or through a pybind extension built before the attribute, which cannot declare
them — is a typed decline at eligibility, `thd_declared_totals`): the f16 row's mechanism ("THD / ragged" below) over
the packed e4m3 payloads (q / k / v / dO and the transposed-quantization q_T / k_T / dO_T,
all packed `[1, T, H, D]` rows) — the metadata buffer and device claim counter,
per-sequence clipped dV stores, the kv-blocked dS workspace (bf16 under P-c, the two
block-scaled e4m3 payloads + E8M0 atoms under P-b), the stage-3 GEMMs trimmed PER
SEQUENCE (P-c: the bf16 renderings over the packed q_T / k_T dequantized EXACTLY to bf16
per token; P-b: the block-scale arm's THD leg over the kv-blocked payloads + atoms and the
packed columnwise q_T / k_T scale factors read through per-sequence SF tile prefixes, dQ
once per head chunk as on the dense P-b chain -- the block-scale dQ record takes
`b_head_group` = the GQA group, B and its scale factors indexed by `h // group`, bitwise
the per-member launches), the GQA fold bounded ON DEVICE at
`cu_k[B]`. **The seven scale-factor tensors travel PACKED per-sequence-TILE-padded**, the
forward's convention: per head, every sequence's `ceil(s_b / 128)` F8_128x4 tiles in
cu_seqlens order (sequence b's tiles start at `cu_sf[b] = Σ_{i<b} ceil(s_i / 128)`, NOT at
`cu[b] / 128`), 1024 B per (head, tile) at d = 256 — rowwise the tile's 128 rows × 8 groups,
columnwise BOTH D planes of a (head, tile) contiguous (plane stride one atom; the dense
D-plane-major view would read plane 1 from the wrong place by an S-dependent offset). The
packed tile count is a PER-CALL fact derived from the bound buffer's byte size (whole
`H × 1024`-byte tile rows, one count per side — `sf_q / sf_q_T / sf_dO / sf_dO_T` and
`sf_k / sf_k_T / sf_v` must each agree — at least `Σ_b ceil(s_b / 128)` tiles per head (the
maps' tile extent is the live total) and at most the plan's capacity, the larger of
`ceil(T_cap / 128) + B` tiles per head and the declared sample's own count, so the graph may
declare any dims with the right byte total, the dense capacity `B × ceil(S_max / 128)`
included, and bind exactly those bytes);
the chain builds every SF descriptor and SFB view at that count. **The producer's pad bytes
may hold anything** (a 0xFF is an E8M0 NaN): the five SF tensors whose pad positions are
READ — `sf_v / sf_dO / sf_dO_T` by the main kernel, and under P-b `sf_q_T / sf_k_T` by the
block-scale GEMMs (whole atoms) — are re-staged per execute into packed staging copies
with every byte scaling a position at or past its sequence's length zeroed, from the
device prefixes (`sf_meta = [cu_sf_q(B+1) | cu_sf_k(B+1)]`, a workspace region of its own
next to the shared metadata, written by the chain's setup launch); `sf_q / sf_k` pads are
harmless (an S NaN is select-dead) and bind as they are. An external delta is served under
THD as on the sibling rows (the packed `[1, H_q, ceil128(T_q)]` form over the `o_f16` /
`dO_f16` ports, bitwise the chain's own `dot` when formed in its order). Declined under THD
as on the sibling rows: right-band widening, bias. Stats comes from the caller
(the Rubin MXFP8 forward row serves THD at d256 since 2026-10-08 -- footnote ᶻ -- and
writes the packed Stats this row consumes). Tests: `test_sdpa_bwd_thd_mxfp8_sm107.py`
(direct adapter), the THD pins of `test_sdpa_bwd_mxfp8_sm107.py`.
ᵇ³ **d ∈ (256, 512] backward (`sdpa_bwd_sm107_d512`, f16/bf16, opt_in, 2026-10-01;
`python/cudnn/sdpa/bwd/api_dsl_sm107_d512.py`, stage-2 kernel
`sm107/bprop_d512_f16_2x2.py`, config `bwd/config_sm100.py::TemplateParams2x2`).**
The SM100 large-head-dim chain (ᵇ of the SM100 section: `delta = rowsum(dO·O)` →
stage 2 writes the S and dS `[B, H_chunk, S_q, S_kv]` workspaces → dV = Sᵀ·dO,
dK = dSᵀ·Q, dQ = dS·K as the shared `bprop_matmul_blackwell` GEMMs at the (512, 512)
cluster tile → the GQA fold) with stage 2 ALWAYS the **2x2-datapath** body at the
cc 10.7 ring arm: two independent `tcgen05.mma.cta_group::2` pairs per (4,1,1)
cluster, 64 q rows per CTA, both BMMs as SMEM SS operands, an **8-stage K/V chunk
ring** (SM100: 4) and **two cast stages** (SM100: 1) filling 320 of the line's 325 KiB
usable SMEM, 256 TMEM columns, tcgen05 descriptor version 0 at ZERO margin (the last
root ends at byte 262143; a 9th stage flips every `SmemTile` to version 1 — pinned
at import). The 4x1 role split the SM100 row renders by default never ran on this
line; the twin's own row keeps that honest (Capabilities cannot vary per cc inside
one row). The rendering is PTX-identical to the SM100 body at these parameters
(`test_sdpa_bwd_d512_sm107.py`, the committed md5 record
`renderings/md5_stage2_2x2_sm107a.txt`), so the SM100 twin's bitwise-vs-role-split
oracle carries over. The sm_107a md5 and SASS pins run only on the cc 10.7 board
with the internal DSL build the record names (the public 4.7.0 DSL has no sm_107a
target, so they SKIP in public CI — a skip there is not coverage); in CI the fork is
held by its code-diff allowlist, its source pins and its import-time `_require`s,
while the SM100 4x1 sm_100a record does run on the public 4.7.0. **Envelope-served:** d ∈ (256, 512], multiples of 8, on
512-wide tiles (d = 264 pays d = 512's MMA); the floor is exclusive at 256 (the d256
rows ᵇ own it). Served: dense, top-left and bottom-right causal, right-band
widening, sliding window (left), MHA / GQA / MQA, any S_q / S_kv (padded to 256 /
128 and masked), BSHD-physical io, contiguous fp32 Stats, bf16 / fp16; the
whole-workspace zero-fill under every causal-family mask (the 512-row stage-3 M
tile straddles two 256-row stage-2 blocks, SM100 ᵇ). **Declined on day one**, each
asserted by a reject probe on a faked cc 10.7 and flipped only with a board-run
accept + a line here: THD / ragged (the SM100 row serves it; the packed path has
not run on cc 10.7, and its multi-chunk correctness is open), `dense_flex`
staging layouts, dense padding masks, sink / dSink, bias / dBias, deterministic,
decode. **No backend competitor:** cuDNN 9.26.0.51 builds no d > 256 backward
plan on cc 10.7 (engines 17 and 7 decline `d_qk > 128` outside the 192×128 and
256×256 cases; probed 2026-10-01), so the row's gate is floor-relative
(stage-2 MMA floor 2502 clk per CTA-kv-tile at the board's 39.1-clk 2SM M=128
N=128 K=16 rate). Rule 7: the public DSL 4.7.0 lacks `sm_107a`; `check_support`
declines through `cutedsl_arch_requirement_error` first. Prepared launch like the
SM100 row (symbol `frost_sdpa_bwd_sm107_d512_prepared`, the SM100 contract below).
ⁱ No native d=64 Rubin kernel, so a d=64 graph rides the d128 envelope (64 is a
multiple of 8 at f16 and of 16 at fp8) at ~2× the MMA cost.
ⁱⁱ `thd_d_shapes={(128,128)}` on the FP8 row is exact — d=64 THD is declined.
ⁱⁱⁱ **Accepted, not validated.** `softmax_precision=HALF` (requested as the
`sdpa()` op attribute — numerics-changing, so it is a graph fact gated by the
row's `softmax_precisions`, not a tuning knob) is admitted per selected flavor
(`fwd/api_dsl.py`, every quantized flavor), and a d=64 graph's *flavor* IS
(128,128), so the request passes the probe and the kernel runs. Untested is the
f16x2 exponent arm over the zero-padded 64 → 128 region (the same holds for the
pre-folded scale riding the d128 envelope).
**d512 (DSv4) f16/bf16 forward datapath (both tables, 2026-10-06):** the half-precision d512 row lowers onto the
2x2-datapath kernel by default (`fwd/kernels/sm100/prefill_d512_f16_2x2.py`, cc 10.7: `sm107/…`; `TemplateParams.mma_2x2`
set by the call-time switch `api_dsl.D512_2X2 = True`): one pipeline per CTA on the `tcgen05.mma.cta_group::2` M = 128
atom, a (4,1,1) cluster of two pairs sharing K/V by multicast, instead of the cga4x1 role split. It won every measured
shape on both lines (B1 H128 S8192: +21 % dense / +21 % causal on B200, +21 / +18 % on cc 10.7; GQA 64x1 Sq16384 with
Skv 512 / 1024 / 2048 / 8192: +46 / +36 / +29 / +22 % on B200, +11 / +16 / +22 / +24 % on cc 10.7; short and small-batch
shapes re-measured on B200 the day of the flip: 11 shapes from B32 H8 S256 to B1 H128 S8192, +21..+36 %, see PR #1323). Plans the twin does not serve take the role split under
either value of the switch: split-KV (`split_kv > 1`), paged KV, PackGQA G = 128, and every fp8 / mxfp8 d512 graph; THD,
sinks, Stats (natural or base-2) and the whole causal family are served. `D512_2X2 = False` is the role-split A/B arm
(`test_sdpa_fwd_dsl_sm100.py` runs every d512 cell under both; `test_sdpa_fwd_d512_2x2_sm{100,107}.py` pin the served
template per cell). Dense graph and standalone calls on the twin bind natively like the role split (the native dense
binder gate in `fwd/prepared.py` names `prefill_d512_f16_2x2`; same host slot list; `test_two_by_two_default_plan_binds_natively`).
Not a Capabilities change: the same row, the same contract, a different default kernel.
ⁱᵛ **d512 MXFP8 is CORRECT but has no test module**, so it is ⚠️ not ✅: cos =
0.9997 / LSE exact at SQ ∈ {128, 256, 384, 512} from `frost_dev/_probe_d512_mxfp8.py`,
but SM100 has no d512 MXFP8 sibling, so the shared suite carries no d512 case to
widen — it needs new cases, not a widened gate. Every other quantized cell above
is covered by `test_sdpa_fwd_{fp8,mxfp8}_sm100.py` running on Rubin.
ᵛ **Superseded by ᶻ** — kept only so the marker resolves for readers of an older
revision. THD used to be unported on the f16 line (7-arg setup call against a
14-arg helper, 3B+2 metadata where the shared decode reads 4B+4); every f16 and
per-tensor FP8 flavor now carries the contract and serves it.
ᵛⁱ Every Rubin template carries the per-batch `seq_len_q` trim (bounds collapse for tiles past the length, O:=0 / LSE:=−inf on the rows past it; #1037), so the rows claim `padded_stats`.
ᵛⁱⁱ FP16/BF16 and per-tensor FP8 D128/V128 and D192/V128 serve dense,
unpadded split-KV with FP32 partial O and the shared combine. Smaller head
dimensions can use the D128 envelope; half also supports the existing
D192/V128 envelope. Dense split with PackGQA is served for per-tensor FP8 D128 only.
Half exact D128 THD also supports split with PackGQA, using the prepared packed
path described below; half nonpaged D192 THD supports unpacked split. Quantized
THD, sinks, and nonpaged dense padded split graphs remain declined.
ᵛⁱⁱⁱ **Fused epilogue gate `O := O * sigmoid(G)`** — a production feature of the
d256 f16/bf16, per-tensor FP8 and block-scale MXFP8 Rubin kernels
(`TemplateParams.epilogue_gate`; rows `sdpa_fwd_prefill_sm107`,
`sdpa_fwd_prefill_sm107_fp8` and `sdpa_fwd_prefill_sm107_mxfp8`,
`Capabilities.epilogue_gate_d_shapes = config_sm107.SM107_EPILOGUE_GATE_SHAPES`, the
same constant the standalone adapter's `check_support` twin reads) — and, over PAGED
pools, of the SHARED SM100 d256 f16/bf16 body compiled for sm_107a
(`sm100/prefill_d256_f16.py`: the paged / THD / PackGQA prefill a Rubin paged half
(256, 256) plan loads; `config_sm100._EPILOGUE_GATE_FLAVORS = {"d256"}` for half
inputs; its gate seams are the sm107 body's through the shared hook, the gate tile's
TMA box the Q / O box's packed-row convention, the staging tile declared after the V
ring so every MMA operand root stays under the 256 KiB version-0 window, ~258 KiB of
SMEM = a Rubin-only rendering). The standalone adapter admits it on that route
(unsplit; packed or dense paged queries past the decode tile; PackGQA on or off) and
both native binders bind G on a paged plan (`supports_paged_epilogue_gate`); the
graph rows keep declining paged x gate outside the decode tile's split (the graph's
THD lowering binds no gate; a row narrower than the adapter is the admitted direction).
The gated attention block's EXTEND form (`kv_source='paged'`, `fuse_gate=True`) runs
stage (5) inside the paged SDPA's epilogue: 5 -> 4 stages per layer. On the graph it
is the three-node tail `sdpa(virtual O_v) -> sigmoid(G) -> mul(O_v, s)`
(`cudnn._sdpa_tail.match_gate_tail`; either mul operand order), validated
python-natively and lowered with the mul output bound as the kernel's O — the
virtual `O_v` / `s` are never bound or materialised. Served **exactly** at
(256, 256) (not through the head-dim envelope), dense on the graph (the standalone
adapter adds THD on the f16/bf16 row, below), unsplit,
non-PackGQA, non-paged -- with ONE exception on the f16/bf16 row: a decode-shaped
graph (`S_q · G` packed rows within the Rubin route, footnote ᵐᵗᵖ) whose plan SPLITS rides the d256 decode tile, which
applies the gate in its split COMBINE (dense or paged, packed or not; the same
one-rounding arithmetic -- footnote ᵈʳ); the padding mask (`seq_len_kv`) composes. `O_v`, `G` and
the final O must be DECLARED (dim + stride, BSHD) — the classic C++
`pre_validate_node` needs both on the sdpa output and only user-assigned values
are pushed at lowering, so an undeclared `O_v` is a typed not-supported from
`validate()` rather than a bare `ValueError` out of planning. **THD (packed
sequences) x gate -- the standalone adapter only, f16/bf16 d256 prefill only
(the THD gate descriptor).** `SdpaFwdDslSm100(thd=True, sample_gate=G)` with G
declared like O -- the `(B, H_q, S_max, D_v)` envelope at G's (token, head)
strides (a gate sliced out of a fused projection slab at token stride N binds
zero-copy, exactly as under dense) -- and handed to `execute(gate=)` as the
packed `(T, H_q, D_v)` buffer: the kernel host maps G through the SAME packed
`[1, T, H_q, D_v]` tensor map Q / O take and the TMA-LDG warp's gate load runs
on the THD coordinates, so the kernel body is the dense gated body byte for
byte; the prepared THD binder binds G per call (validated like Q / O: Q's dtype,
16-byte-aligned base, TMA-expressible strides, a capacity covering every packed
Q token -- a shorter gate is a typed `ValueError`, never a shortened launch).
Validated on the cc 10.7 development class: per sequence against the fp32
oracle at the gate suite's bounds, within one output ulp of the ungated THD run
followed by the separate `O * sigmoid(G)` stage (the one-rounding convention
the dense gate keeps), `B = 1` packed BITWISE the dense gated plan (O and LSE),
LSE bitwise the ungated THD run's, a sequence with Q rows and ZERO keys exactly
0 / −inf under a ±1e4 gate, a zero-length sequence in the packing, storage past
the packed total untouched. NOT served under THD: the quantized (per-tensor FP8
and MXFP8) THD bodies (they build the dense gate map; declined by name), and the
GRAPH tail (the graph's THD lowering binds no gate operand: the rows keep the
"dense-only (no THD gate descriptor)" decline for a THD gated graph -- the
adapter is wider than the row here by design, never the reverse). A pybind
extension older than the binder's gate role (`_SdpaThdBinder.supports_epilogue_gate`)
declines a gated THD plan typed at `check_support`. `G` is Q's dtype on
the f16/bf16 row and BF16 on the FP8 and MXFP8 rows (the kernels stage it in bf16), and must
be **BSHD-physical with 16-byte-aligned, non-overlapping seq/head strides**: G is
TMA-loaded zero-copy (no normalisation copy, unlike Q/K/V/O), so the row declines
by message exactly the G the standalone adapter's `check_support` rejects
(`facts.epilogue_gate_layout_ok`, the twin of `_bshd_zero_copy_stride`). `O_v`
(the sdpa node's virtual output) must carry AT LEAST O's precision — FLOAT or Q's
dtype on the f16/bf16 row; FLOAT, a half dtype, or O's own dtype on the FP8 row
(never Q's fp8 dtype when O is half: that would spell "quantize, THEN gate", a
rounding the kernel does not do) — because the kernel gates the fp32 pre-cast
accumulator and casts once, so the intermediate dtype is descriptive. With
FROST engines enabled the tail is validated **python-natively on every arch**
that offers a python SDPA engine (the family validator does not look at the
rows); the backend's own verdict on it is deferred to planning, as for every
python-validated graph. Numerics: the gate multiplies the fp32
accumulator AFTER the dead-row / empty-KV select (a dead row stays exactly 0,
LSE −inf, and LSE is bit-independent of the gate); on FP8 the O quantization
applies to the GATED value, while `Amax_O`, when requested, is the amax of the
UNGATED normalised O (pre-gate, pre-quant, in `scale_o` units) — it is an output
of the sdpa node, which precedes the sigmoid/mul tail, so it is independent of
G; the standalone adapter's `sample_gate` path shares that one contract.
**Amax_O binding changed for every quantized forward
row:** only a `set_output(True)` `Amax_O` is a fact (`facts.amax_o_t =
_real_output(...)`) — an unrequested (virtual) `Amax_O` is no longer bound or
written, so a quantized graph that leaves `Amax_O` virtual now EXECUTES where it
used to raise "the variant pack is missing buffers"; on the Rubin FP8 and MXFP8 d256
kernels the graph path passes `has_amax_o=False` so the amax atomic folds out, while
the SM100 fp8 and the other mxfp8 kernels (no `has_amax` compile knob) keep their
legacy dummy amax slot. **MXFP8 gate (PR-B, 2026-09-15):** the d256 block-scale
kernel carries the same 16 seams through the shared `_common_blackwell` hook; the
gate `SmemTile` is declared AFTER the four scale-factor slabs (with a bf16 O at
`STAGES_KV=2` a gate declared before them would put `sQ_SF` at exactly 262144, past
the version-0 tcgen05 descriptor window the kernel pins, and the UTCCP would copy Q
data as scale factors — `config_sm107.d256_mxfp8_last_sf_tile_start` plus the
kernel's own `_LAST_SF_TILE_START` guard raise on any depth / O dtype that crosses
it). The MXFP8 kernel has no per-tensor `scale_o`, so a gated **e4m3 O is written
UNSCALED** — the same unscaled e4m3 O the ungated MXFP8 row writes; a scaled
quantized gated O needs a bf16 O and a downstream quantize. Numerics as on FP8:
gate on the fp32 accumulator, dead-row select after the gate fma, LSE bit-independent
of the gate, and `Amax_O`, when requested, is the amax of the UNGATED normalised O —
the sdpa node's output, pre-gate and pre-quant, independent of G (the kernel folds
|h| = |u/2| and doubles once per tile, bit-exact; the PR #1102 review-round-1
contract) — in the O's OWN units, since this path has no `scale_o` (the same units
the ungated MXFP8 row publishes; on FP8 they are `scale_o` units). Pinned by
`test_sm107_mxfp8_gate_amax_is_a_compile_time_fact` (standalone adapter: G = ±1e4, a
random G and the ungated specialization agree bit-for-bit) and
`test_mxfp8_gate_tail_graph_api` (graph path: Amax_O(G=+1e4) == Amax_O(G=−1e4) ==
Amax_O(no gate)). Validated on the standalone adapter on EVERY dtype member the row claims
with the gate -- inputs {e4m3, e5m2} x outputs {fp16, bf16, e4m3, e5m2}, one Rubin
launch each against the dequantized oracle (`test_sm107_mxfp8_gate_matches_the_dequant_oracle`,
e4m3-in also dense 512 + causal 1000 at bf16 and e4m3 O; e5m2-in at the suite's 8e-2
half-O bound), plus the padded dead entry (bf16 / e4m3 O), and on the graph path
(`test_mxfp8_gate_tail_graph_api`, e4m3 in, bf16 / e4m3 O). **One-time
gate-off == shipped proof (2026-09-15, re-run 2026-09-16 on the pre-gate-`Amax_O` kernel; Rubin dev node):** the working-tree kernel with
`epilogue_gate=False, has_amax=True` produced O, LSE and `Amax_O` BITWISE equal to the
pre-gate develop kernel (`18091c19`, #1059) at B=2 H=8 H_kv=2 S=1000 causal, e4m3 in /
bf16 O, through the same adapter marshalling
(`test_sm107_mxfp8_gate_off_is_bitwise_the_shipped_kernel`; the test SKIPS once develop
itself carries the gate, so this footnote is where the evidence lives). The ungated
sm_107a SASS is byte-identical to develop's apart from the param-space offsets of the
unread gate descriptor (STL/LDL 6 = 6); the GATED MXFP8 module carries 25 STL/LDL (an
L1-resident refill in the 40-register TMA-LDG / TMA-STG / scheduler warps around the
`setmaxnreg` boundary, 0 in the gate math; the f16 and per-tensor FP8 gated bodies stay
at 6) whose cost is UNMEASURED until a perf-node A/B/A -- a correctness claim, not yet a
perf one.

**THD / ragged (f16 and fp8 rows; the fp8 row's own paragraph above).** The packed path: Q/K/V/O/dO and the gradients are
PACKED `[1, T, H, D]` (packed BSHD rows -- element stride 1, head stride D, token stride
>= H·D and a multiple of 8; a padded token stride is served as declared), `seq_len_q/kv`
as `(B,)` tensors on the graph (`(B,)` or `(B+1,)` prefixes standalone), both declared
totals REQUIRED (`Capabilities.thd_declared_totals`: the kv-blocked dS workspace, delta
and the GQA partials are sized at build time), both packed Stats layouts the forward
emits.  Mechanics: a setup launch (`kernels/thd_helpers.thd_bwd_setup_host(kv_blocked=True)`)
builds the metadata with the dS workspace row offsets blocked over the KV lengths at the
kernel's 256-row kv block; the main kernel's own setup kernel clamps the five input
descriptors to the live packed totals (a NaN capacity tail is TMA-OOB zero), emits one
clipped dV descriptor per sequence and resets the claim counter per launch; the kernel
claims (sequence, kv block, head) units from the device counter, masks each sequence's kv
tail and q pad columns from its own lengths (the bottom-right diagonal `S_kv[b] − S_q[b]`
included) and reads the packed Stats past `S_q[b]` as `+inf`; stage 3 renders the THD arm
with `MatmulTemplateParams.thd_rows_kv` (the kv-major workspace flips the token side of
each reduction against the SM100 chain's q-major one) and the dense path's two-sided K-trim
PER SEQUENCE (`MatmulTemplateParams.thd_causal_bottom_right` spells the per-sequence
diagonal; every bound is sequence-local and derived from the sequence's real lengths; a
tile whose band is empty gets an empty K range and a select-zero store), so the kv-blocked
workspace is never zero-filled -- the poisoned-workspace THD cases and the host tile walk
`test_stage3_thd_band_arithmetic` are the proof; GQA via per-Q-head partials over the
packed kv axis up to the live total `cu_k[B]` (a device word; the SM80 row's contract that
nothing past the packed total is written into the caller's gradients holds on every path:
dQ / dK / dV through per-sequence clipped output descriptors, the fold bounded on device),
dQ once per head chunk (`b_head_group` = the GQA group on the packed K descriptor, whose
per-sequence clamp touches only the token extent; bitwise the per-member twin).  Degenerate
sequences exact (empty-KV: no unit, zero dQ by
select; empty-Q: one forced fully-masked tile whose every operand load is routed past the
clamped extent -- zero-filled, so an all-NaN Q / dO capacity with no live query row still
yields exact-zero dK/dV; the same routing for the spare units of the occupancy-sized grid).
An external delta is served under THD on every row (`external_delta=True` +
`execute(delta_tensor=)`: the packed head-major `[1, H_q, ceil128(T_q)]`, zeros past the
token capacity, read where the chain read its own region, one `dot` launch and the `delta`
region gone; the `external_delta` cells of the three THD suites). Declined under THD:
right-band widening (as dense), bias (every row; the
MXFP8 row's packed scale-factor contract is its own paragraph, ᵐˣ).  Perf (Rubin, cc 10.7, 212 SMs; B = 4, S_max = 8192,
bf16, the THD arms against the dense run of the same shape in one process): the THD
overhead at identical FLOPs (uniform 8192-token sequences) is +0.1 / +1.3 / +1.8 % under the
dense mask and +2.0 / +8.2 / +10.1 % under causal at H 32/2 / 64/8 / 16/16 (it was +14.6 / +17.2 /
+1.6 % and +73.3 / +63.1 / +54.1 % with the per-member dQ launches, the untrimmed GEMMs and the
zero-fill); the varlen cell ([2048, 4096, 6144, 8192]) runs at 51 / 51 / 52 % (dense) and 45 / 45 /
48 % (causal) of the bf16 peak on its exact FLOPs, +16-24 % above the FLOP-scaled dense time.
Remaining THD-specific costs, numerics-neutral: (1) the stage-3 grids sized on the KV / Q
ENVELOPE per (head, sequence) group -- B × ceil(S_max/256) M tiles per head where only
Σ ceil(s[b]/256) are live, the spare tiles clipped by the output descriptor (a B-fold waste
on skewed batches); (2) every spare cluster of the occupancy-sized grid runs one forced
masked tile whose zero dS tile lands in the same slack rows past `row_off[B]` (up to ~100
clusters on one 64 KiB region on a small problem).  Levers, in order: a per-group M-tile
early-out in the GEMM template, a THD grid bounded by the live units.  Tests: `test_sdpa_bwd_thd_sm107.py`
(direct adapter + pinned graph, per-sequence fp64 oracle under the dense suite's bounds;
empty sequences, NaN capacity tails, a finite sentinel on the gradient rows past the packed
totals, dead units, GQA, the causal family, both Stats packings, prepared rebind / replay /
length forms, every sequence empty on the Q side with a NaN dO capacity), the SM100 stage-3
renderings PTX-identical at `thd_rows_kv`'s default, the dense f16 PTX identical to before
the port.

ᶻ **f16/bf16 THD is served on EVERY flavor** as of 2026-09-09 (d128, d192×d128,
d256, d512), and per-tensor FP8 THD likewise on all four
(`config_sm107.SM107_FP8_THD_SHAPES`: d128, d192×d128, d256, d512). The f16 bodies were
moved onto the FROST THD contract: the 14-arg setup helper, the 4B+4 metadata
the shared decode already read, the persistent claim-counter scheduler, the
dead-unit O-store guard, the packed-total-clamped runtime K/V descriptors (a NaN
capacity tail otherwise wipes a tile through BMM2), the token-major Stats arm,
and per-sequence Q lengths for the bottom-right diagonal. Validated on
`w2u1g-lc-0030`: the whole SM100 f16 THD suite — **215 passed** where it
previously skipped entirely — the per-tensor FP8 THD suite at **43 passed / 0
failed**, and a per-flavor attribution sweep (B=1/2/3 × none/causal/
bottom-right, 9/9 per flavor). Whole-suite baseline on Rubin moved from
**829 passed / 11 failed** on unmodified `develop` to **1111 passed / 10
failed**; those 10 are the pre-existing set (9 ×
`test_q_loop_bounds_match_reference`, 1 × `test_torch_ops::test_padded_seq_lens`)
verified to reproduce at `60cb6cda`. **MXFP8 THD is served at d256** (2026-10-08:
`config_sm107.SM107_MXFP8_THD_SHAPES` is the row's `thd_d_shapes`, the standalone
adapter's Rubin THD gate and its `_can_prepare_mxfp8` -- one constant, rule 8b'; the
d256 MXFP8 body moved onto the same FROST THD contract at cga1 -- the 14-arg setup
launch, 4B+4 metadata, the persistent claim-counter scheduler, the dead-unit O-store
guard, the packed-total-clamped runtime K/V maps, the three Stats arms (token-major,
head-major, per-batch padded, so `thd_padded_stats` is claimed) -- with the PACKED
per-sequence-TILE-padded scale-factor layout the SM100 MXFP8 row and the SM107 MXFP8
backward already use: SF descriptors at B = 1 over the bound SF tensor's tile extent,
which the native binder derives from the buffer's byte size, so a producer may hand
zero-filled slack tiles past the live total and every kernel coordinate stays below
`cu_sf[B]`; both length forms (`cu_seq_len`); the dense cubin is byte-identical before
and after the port. Validated on Rubin (cc 10.7) through the SM100 MXFP8 THD suite's
d256 cells, which the per-shape Rubin skip now admits -- 37 THD cells green at the
suite's unchanged tolerances (E4M3 + E5M2; none / causal / bottom-right / sliding-window;
sink; GQA 8/2 and MQA 8/1 (eight Q heads over one K/V head); both length forms;
zero-length sequences first, middle and last;
scale-factor slack capacity bitwise the exact buffer; Stats VALUES in the token-major,
head-major and per-batch padded layouts; per-sequence attribution for one to three
sequences; a single-unit launch; the multi-unit claim loop) plus the 3 Rubin decline cells
at the unported shapes, the 25 THD prepared-launch cells of the SM107 entry point
(`TestPreparedSm107Mxfp8`: routes, rebind, capture, SF storage order, one bound frame, no
allocation or sync, the packed-SF binder's typed declines), and the dense d256 surface
unchanged-green (37 suite cells + 24 dense prepared cells) on a byte-identical cubin).
**Still declined: MXFP8 THD at d128 / d192×d128 / d512** — those bodies keep the pre-upstream
7-arg setup call and the `total_*_sf_tiles` host kwargs the prepared pointer host
never passes, so the row declines them through `thd_d_shapes` rather than
half-serving.

ʸ Per-tensor FP8 **THD** at d192×d128 came free with the DSv3 port and is
served as of 2026-09-09: that kernel *is* the shipped d128 FP8 body (only the
config factory differs), so its THD leg is the validated wiring — 5 THD cases
pass on `w2u1g-lc-0030`. Closing it needed both enforcement points widened
together, the row and the standalone adapter's gate; they now share one
constant (`config_sm107.SM107_FP8_THD_SHAPES`) so they cannot drift. d256 and
d512 were moved onto the same contract afterwards (the 14-arg setup helper, the
4B+4 metadata the shared decode reads; 43 passed / 0 failed across the per-tensor
FP8 THD suite), so the constant names all four shapes and no ported body raises
at `compile()` any more.

ˣ d192×d128 MXFP8 runs at **cga2 only** (SM100 serves the shape at cga1 and
cga2). See the paragraph above: at cga1 this flavor's scale-factor tiles cross
the 256 KiB version-0 tcgen05 descriptor window. Every other cell in that
column was measured on `w2u1g-lc-0030` (cc 10.7) by
`test_sdpa_fwd_{fp8,mxfp8}_sm100.py`, whose dense d192 cases stopped skipping on
Rubin in the same commit — 31 passed, and the 18 that still skip are the
PackGQA and THD families this line declines per-feature, not per-shape.
ⁱˣ Served through the KV-tail mask (`skv_tail_via_padding`: the padded mask
compiled against the scalar `S_kv`, #1425). REQUIRED, not an optimization: with
no mask the kernel's KV loop bound is a floor division, so an unmasked ragged
`S_kv` would silently drop the tail tile.

**Scheduler policy — `SCHED_LPT` now served on the f16 (256, 256) Rubin
flavor; NATURAL elsewhere.**

The earlier reading — "the ported decode does not honor SCHED_LPT" — had the
symptom right and the cause wrong. Every SM100 kernel calls
`make_sdpa_helpers(CFG, lpt_q_tiles_in_cga_units=True)`; the SM107 port omitted
that argument on 9 of 11 flavors. Without it the LPT linearization walks a row
range `CTA_MMA`× too large, no tile is ever claimed and the kernel writes
**nothing** (cosine 0.0000, dense and causal). The two flavors that kept it,
d128 FP8 and d192 FP8, are exactly the ones previously described as working.

The argument is restored on every 2-CTA SM107 flavor. It is a **no-op under
`SCHED_NATURAL`** (that decode branch never reads `q_tiles`), so the shipped
path is unchanged.

Only what is validated is advertised, via the new per-shape
`sched_policies_by_d_shape` (mirroring `cgas_by_d_shape`): the f16 row serves
`SCHED_LPT` at **(128, 128)** and **(256, 256)**. The D128 qualification
covers dense, ragged and paged half inputs, including PackGQA and changed
live full/prefix/empty requests under capture. D256 validation: cos 1.0000 at n_kv 2/3/4/8,
dense and causal. Measured causal SOL on Rubin at S = 4096/8192/32768:
53.0/71.1/76.7 % under NATURAL → **62.6/79.3/77.5 %** under LPT
(+18.2/+11.6/+1.1 %), recovering 40/51/29 % of the causal-vs-dense gap; dense is
neutral. The decay with S is the signature of scheduler imbalance.

**FP8 (256, 256) and (192, 128) join the LPT claim (2026-09-11).** Validated
through the standalone adapter on Rubin (E4M3 per-tensor scales, bf16 O, causal,
dense and padded, (B, S) ∈ {(1,256), (2,1000), (1,4096)}) against the fp64
kernel-mirroring `fp8_ref.compute_ref`: (256, 256) max|O−ref| 0.0078 causal /
≤ 0.0019 dense (tol 0.075); (192, 128) 0.0397 causal / ≤ 0.0060 dense (tol
0.04). On both flavors O and LSE under LPT are **bit-identical** to NATURAL (the
scheduler reorders whole (batch, head, q-tile) work items; each tile's KV loop is
unchanged), sentinel 0, two-launch 0. Perf node, d256 causal H32/2, LPT vs
NATURAL launch-interleaved: +5.2/+5.9/+5.6/+2.0/+2.3 % at S = 2K..32K (control
pair within 1.9 %). The FP8 row now carries
`sched_policies_by_d_shape = (((256, 256), {NATURAL, LPT}), ((192, 128), {NATURAL, LPT}))`.
One FP8 flavor stays out ((128, 128) joined on 2026-09-14, below): **(512, 512)** — the cga4×1
role-split kernel still calls `make_sdpa_helpers(CFG)` without
`lpt_q_tiles_in_cga_units` (the #1001 argument, left on the d512 line), so under
LPT it writes *nothing* (sentinel on 100 % of cells; the earlier "causal d512
FP8 → NaN" report was that unwritten output being read). In the same
change `heuristics._sched_points` ranks from the FLAVOR's effective domain
(`effective_sched_policies`) rather than the row-wide floor — before it, a
per-shape LPT claim was honoured only when a caller REQUESTED the knob and was
never proposed for the first plan (this also makes the f16 d256 claim reach
the graph path's ranking). Pinned by
`test_sm107_fp8_advertises_lpt_only_for_the_validated_d_shape`,
`test_sm107_fp8_lpt_knob_is_honored_or_ineligible_per_d_shape` and the Rubin
e2e `test_fp8_lpt_is_bit_identical_to_natural_on_the_claimed_flavors`.

**LPT_L2 joins on the flavors whose kernels thread its inputs; MXFP8 d128 and
d192×128 claim LPT + LPT_L2 (2026-09-14).** `SCHED_LPT_L2`'s decode needs
`qh_per_kh` and `seqlen_kv` at every call site (the shared decode raises at
trace time without them). The d128 and d192×128 per-tensor FP8 kernels already
threaded them through `make_split_helpers`; the two MXFP8 siblings
(`sm107/prefill_d128_mxfp8.py`, `sm107/prefill_d192_d128_mxfp8.py`) now thread
them into all ten `_dispatch_decode_*` sites (mirroring the FP8 form). Claims:
FP8 `(192, 128)` adds `LPT_L2` and FP8 `(128, 128)` claims `{NATURAL, LPT, LPT_L2}`
(bit-identical to NATURAL under both — the 0.041–0.048 its e5m2 causal path
reads against the suite's 0.04 comes out of the same bits under every policy;
perf node, kernel-level d128 H64/8 causal vs NATURAL: LPT_L2 +4.2/+7.6/+7.0/
+5.9/+5.4 %, LPT +5.3/+5.8/+4.2/+2.0/+1.1 % at S = 2K..32K); MXFP8 `(128, 128)`
and `(192, 128)` claim
`{NATURAL, LPT, LPT_L2}` — the row was NATURAL-only, and the "23 MXFP8 tests red
under LPT" report that kept it there has the #1001 signature (dense graphs rank
NATURAL and stayed correct; every masked graph took the LPT decode with the
dropped `lpt_q_tiles_in_cga_units` argument and read unwritten output).
Validation, Rubin, standalone adapter, sentinel-filled O and NaN-filled LSE: on
each claimed flavor O and LSE under LPT and under LPT_L2 are **bit-identical**
to NATURAL, dense and causal (multi-wave causal grid), and the MXFP8 suite —
whose causal cases now rank a remap first — stays green. **Heuristics change,
Rubin rows only** (`heuristics._sched_points`): with GQA the L2-budget rule
keeps leading with LPT_L2 (Llama above); without GQA (h_q == h_kv, the DSv3
layout) LPT_L2 has no K/V sharing to group and measured **−9.7 %** at S=2K, so
the rule proposes plain LPT while the grid is ≤ 24 waves of 2-CTA clusters
(+16 % at S=4K) and NATURAL beyond (LPT −6.5 / −13 / −7.9 % at S=8K/16K/32K) —
perf node, kernel-level, d192×128 FP8 H128 causal. Every domain member stays an
autotune runner. Pinned by
`test_sm107_causal_ranking_picks_the_policy_by_gqa_and_wave_count`,
`test_sm107_rows_serve_natural_scheduling_only`,
`test_sm107_mxfp8_advertises_lpt_and_lpt_l2_per_d_shape`,
`test_sm107_mxfp8_sched_knob_is_honored_or_ineligible_per_d_shape` and the
Rubin e2e `test_mxfp8_sched_policies_are_bit_identical_to_natural` (plus the
widened FP8 e2e).

Still declined, and why: d192/d512 f16 are **unvalidated** under LPT rather
than known-incorrect; d512 (f16, FP8, MXFP8)
**does not produce output** under LPT until the d512 kernels get the
`lpt_q_tiles_in_cga_units` argument and are re-validated (cga4×1 role-split, a
different scheduler shape — `prefill_d512_fp8.py:2533` notes the LPT range is
`q_clusters * CTA_MMA`). `SCHED_LPT_L2` stays declined on every f16 flavor and
on d256 / d512 of every dtype family — those kernels' decode call sites pass
neither `qh_per_kh` nor `seqlen_kv`, so the decode raises rather than
miscomputes. Threading them there is the same mechanical edit the MXFP8 pair
received.

**`STAGES_KV` on the d256 flavors is 2..4, not pinned to 2.** The old pin blamed
a body that conflated the KV ring index with the 2-slot `S_acc` parity; the body
in fact derives parity from the absolute `kv_loop & 1` and is depth-agnostic.
The real constraint was the >256 KiB tcgen05 descriptor wrap: at depth 4 the last
two V stages start at 256/288 KiB and a version-0 descriptor truncates
`start_address` to 14 bits. `prefill_d256_f16.DESC_VERSION` is now **derived**
from the layout, so every depth in range is correct by construction (cos 1.0000
at 2/3/4, dense and causal). Depth is perf-neutral — dense SOL moves 0.3 points
across all three — so the default stays 2.

**Coverage.** The Rubin cells above are exercised by the SM100 suites running on
cc 10.7 with an arch-aware engine pin — `test_sdpa_fwd_dsl_sm100.py` (f16/bf16),
`test_sdpa_fwd_fp8_sm100.py`, `test_sdpa_fwd_mxfp8_sm100.py` — plus the
device-independent `test_sdpa_fwd_dsl_sm107.py` / `test_sdpa_fp8_sm107.py` row
assertions. Every ❌ in the table is a `skipif(_SM == 107, ...)` naming the row
field that declines it, so the skip inverts when the feature lands.

---

Base-2 stats are wired across the Rubin dtype families, including the separate
Rubin config factories. Target-SM107 numerical validation is still required.

## SM120 / SM121 (Blackwell GeForce, cc 12.0–12.9)

Engines: `sdpa_fwd_prefill_sm120`, `sdpa_fwd_prefill_sm120_fp8`,
`sdpa_fwd_prefill_sm120_mxfp8`, `sdpa_bwd_sm120`. Head dims are a **continuum**, not per-model flavors: the
kernel picks Q/K and V head tiles independently (f16/bf16 head dims it would
tile at 256 on both sides run a dedicated template, `sm120/prefill_d256_f16.py`
with the same support).

D512 FP16/BF16 FPROP (`sm120/prefill_d512_f16.py`) supports **both D_QK and D_V
in (256, 512]**, in multiples of 8. It uses a 64 x 32 CTA and 512 x 512 head
tiles: (512, 512) is **native**; smaller shapes are **envelope-served** by
zero-padding, with the same MMA and on-chip storage cost.
FP8 FPROP (`sm120/prefill_d512_fp8.py`) adds the same independent band with
**multiples of 16** and a **64 x 64 CTA**. BPROP remains limited to 256.

MXFP8 FPROP (`sm120/prefill_mxfp8.py`) runs the FP8 pipeline on the block-scaled
`m16n8k32` MMA, with the same independent head band up to 256 in **multiples of
16**; both head dims in (256, 512] run `sm120/prefill_d512_mxfp8.py`, the d512
FP8 layout on block-scaled MMAs, with the same **64 x 64 CTA**.

| Feature | FPROP<br>d ≤ 256, any ×8;<br>both dims ∈ (256, 512], any ×8 | FPROP FP8 / MXFP8<br>d ≤ 512, any ×16 | BPROP<br>d ≤ 256, any ×8 |
|---|:--:|:--:|:--:|
| **Data types** | | | |
| FP16 / BF16 | ✅ | — | ✅ |
| FP8 E4M3 / E5M2 (per-tensor) | — | ✅ | ❌ |
| MXFP8 | — | ✅ | ❌ |
| O dtype ≠ QKV — **FP8 graphs only** (fp16/bf16/fp8 out) | ❌ | ✅ | — |
| Rectangular head dims (D_QK ≠ D_V) | ✅ (independent within each served band) | ✅ (independent) | ✅ (D_QK ≥ D_V) |
| Head-dim alignment (actual `D_QK`/`D_V`) | ×8, both ≤ 256ᵃ; or both ∈ (256, 512] | ×16, ≤ 512ᵃ | ×8, ≤ 256 |
| **Layout** | | | |
| BSHD / `dense_flex` | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| THD / ragged | ✅ | ✅ | ❌ |
| `cu_seq_len_q/kv` | ✅ | ❌ | ❌ |
| Strided / permuted Stats | ✅ | ✅ | ✅ |
| **Masks / features** | | | |
| Causal (top-left) | ✅ | ✅ | ✅ |
| Causal bottom-right | ✅ | ✅ | ✅ |
| Causal right-band widening | ✅ | ✅ | ✅ |
| Sliding window (left) | ✅ | ✅ | ✅ |
| Padding mask (+ stats, + padded-Q trim) | ✅ | ✅ | ✅ |
| Attention sink / dSink (forward at `S_q == 1`: ❔ — see SM100 ˢ) | ✅ | ✅ | ✅ / ✅ |
| Base-2 stats (`stats_use_log2`) | ❔ | ❔ | — |
| Bias / dBias | ❌ | ❌ | ✅ / ✅ |
| GQA / MQA (`H_q ≠ H_kv`) | ✅ | ✅ | ✅ |
| Deterministic (`use_deterministic_algorithm`) | — | — | ✅ |
| Ragged `S_kv` (no tile rule) | ✅ | ✅ | ✅ |
| Decode-shaped (`S_q == 1`) | ✅ | ✅ | ✅ |

Base-2 stats are wired into all three f16 templates and the FP8 and MXFP8
templates, including the final Split-KV combine. `sdpa_mxfp8` graphs take no
`stats_use_log2` input, so only the direct adapter reaches the MXFP8 wiring.
Target-SM120 numerical validation remains required; local SM80/SM100 results do
not validate these kernels.

ᵃ **Head TILE granule and head-DIM alignment are different numbers — the column
headers quote the head-dim rule.** `GENERAL_HEAD_TILES` steps by 16 (f16), and
`FP8_GENERAL_HEAD_TILES` steps by 32; the wide FP8 and MXFP8 flavors use fixed
512-wide head tiles. Those are the kernel's native tile sizes; an actual `D_QK`/`D_V`
only has to satisfy `d_pad_multiple` — **8** on the f16 row, **16** on the FP8
row (the TMA 16-byte global-stride rule at 2 and 1 bytes/elem) — and is
zero-padded up to the next tile. So a d=72 f16 graph is eligible and computes
on the 80-wide tile.

---

## SM80 (Ampere A100, cc 8.0 exactly)

Engines: `sdpa_fwd_prefill_sm80`, `sdpa_bwd_sm80`. Both use `mma.sync` (no
tcgen05) and assume the A100's 164 KiB opt-in SMEM — sm86/sm89 are declined.
The backward serves packed THD graphs (ᵏ); the forward serves native-flavor packed THD graphs
with per-batch or cumulative Q/KV lengths, compact BSHD Q/K/V/O, and packed
head-major or token-major Stats.
The packed caller contract still applies: every sequence length is within its
declared `S_max`, the Q/KV length sums fit the respective bound packed buffers
and any `max_total_seq_len_*`, and ragged offsets describe the same adjacent
sequences. Runtime lengths stay on device; the adapter cannot host-validate
these values without breaking graph capture.

Dense forward uses a prepared pointer host when V matches its flavor width,
Q/K head dimensions are multiples of eight, and every stepped Q/K/V/O outer
stride is a multiple of eight elements. These plans require 16-byte-aligned
Q/K/V/O base addresses and reject misaligned runtime bindings. They address
declared B/H/S permutations, padded strides, GQA/MQA and strided Stats directly,
with no layout staging or GQA expansion. Sink logits are converted to log2
units inside the attention kernel. Address strides and products retain Int64
width. Other admitted head dimensions and layouts use prepared bitwise
gather/pad and scatter kernels in caller workspace around the same native GQA
host. Q/K padding preserves the original attention scale; sink conversion remains
inside attention. Native operands bind directly without redundant copies. The
standalone dense wrapper keys plans by input strides and supplies current scratch.
The capability envelope is unchanged; standalone RoPE uses prepared angle-table
preprocessing and data copies. Its shared forward/backward table host fuses
full-precision FP32 sin/cos and interleaving, retaining the existing CPU/dtype/layout
normalization and launch-stream allocation. All forward entries share prepared host lowering;
the former tensor compilers and their fake-operand construction are removed.

| Feature | d64 (GPT-OSS) | d128 (Llama) | d192×d128 (DSv3) | d256 (Qwen) |
|---|:--:|:--:|:--:|:--:|
| | F / B | F / B | F / B | F / B |
| **Data types** | | | | |
| FP16 / BF16 | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| FP8 / MXFP8 | ❌ / ❌ | ❌ / ❌ | ❌ / ❌ | ❌ / ❌ |
| **Layout** | | | | |
| BSHD / `dense_flex` | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| THD / ragged | ✅ / ✅ᵏ | ✅ / ✅ᵏ | ✅ / ✅ᵏ | ✅ / ✅ᵏ |
| `cu_seq_len_q/kv` (THD forward only) | ✅ / ❌ʲ | ✅ / ❌ʲ | ✅ / ❌ʲ | ✅ / ❌ʲ |
| Strided / permuted Stats (dense; packed THD uses the two layouts above) | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| **Masks / features** | | | | |
| Causal (top-left) | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| Causal bottom-right | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| Causal right-band widening | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| Sliding window (left) | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| Padding mask (+ stats, + padded-Q trim) | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| Attention sink / dSink | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| Base-2 stats (`stats_use_log2`) | ✅ / — | ✅ / — | ✅ / — | ✅ / — |
| Bias / dBias | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| GQA / MQA | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| Deterministic | — / ✅ | — / ✅ | — / ✅ | — / ✅ |
| Ragged `S_kv` | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| Decode-shaped (`S_q == 1`) | ❌ / ❌ | ❌ / ❌ | ❌ / ❌ | ❌ / ❌ |

Native SM80 backward flavor widths with 16-byte-aligned Q/K/V/O/dO and
D-gradient bases and stepped outer strides divisible by eight elements use
one prepared pointer host. The host includes workspace initialization,
do-dot, optional dSink, the selected backward kernel, dQ conversion, GQA
reductions and auxiliary output copies. All stages use the current caller
stream; graph and standalone execution share the binding validator. Native
plans require caller workspace and do not build tensor views at execute.
Native-flavor THD graph/direct-adapter plans use the same binder and a packed
host chain, including device length-to-prefix setup. Packed capacities are
runtime host arguments; workspace views and offsets are formed from those
capacities without specializing the artifact on them. A fresh plan with the
same specialization and stepped strides reuses the process-local artifact for
different totals and bounds, even without persistent caching. The standalone
THD wrapper pads to the native flavor width and uses this prepared packed chain.
Off-flavor graph/direct-adapter widths, unaligned strides and older direct adapters
without complete optional-output declarations use prepared gather/pad, gradient
scatter and auxiliary cast recipes around the same pointer chain. These replace
the existing tensor copies within the same caller-workspace budget. Packed copy
capacities and stepped strides stay Int64 runtime arguments; disjoint dense gradients
share one scatter launch, while overlapping storage ranges retain ordered copy-back.
Standalone RoPE uses the same prepared data-copy recipes and the shared prepared
angle-table host on the launch stream, preserving the existing FP32 trigonometry.
Packed output casts and folds truncate flavor padding on device and leave capacity
tails untouched. The obsolete generic and d64 tensor compilers and fake builders
are removed. Standalone adapters require caller workspace, as native prepared
adapters do; convenience wrappers continue to provide it. No eligibility changes.

The standalone SM80 packed forward wrapper also uses a cached pointer host.
Packed capacities, Q/K/V token and head strides, and launch bounds bind as
runtime Int64 arguments; Sink logits remain in natural units. The wrapper
preserves its output allocation, capacity-tail zeroing and off-flavor padding.
All SM80 forward tensor-fake construction is removed. Dense conversion copies,
including standalone RoPE, share prepared gather/scatter plans with native GQA
and natural-unit sinks. RoPE preserves its angle-table values through the shared prepared host;
its obsolete tensor-staging executor and dedicated host compiler are removed.
Native-flavor THD forward graphs reuse this pointer host, with optional head-major
or token-major Stats and device-side length normalization in caller workspace.
Plan construction and execution allocate no device memory. Graph admission requires
compact packed Q/K/V/O, natural scheduling, and no bias, sink, or RoPE fusion.
The graph path does not run the standalone wrapper's output-capacity initialization.

Standalone packed forward/backward wrappers share prepared bitwise data copies
for their existing head-width padding, backward compaction and output trimming.
One fused copy handles the selected operands; native operands bind directly.
The bounded wrapper allocation-recipe memo contains only host metadata, while
compiled copy artifacts specialize on head/width geometry and bind Int64 token
capacities, strides and addresses per call. The former Torch padding helpers
are removed. Packed forward initializes O and Stats together inside its prepared
host; packed backward combines its existing dQ/GQA-gradient capacity zeroing with
workspace initialization. MHA dK/dV and direct graph-output tails retain their
previous write contract. Initialization capacities and offset products stay Int64.
Both packed wrappers bind the operand device around compilation and execution,
including the default-stream path, and restore the caller's current device.
CPU prefix/scalar conversion retains its existing convenience-wrapper contract;
graph eligibility is unchanged.

The SM80 backward additionally has a dedicated plain-dense **d=64 fast path**
(~2× on A100) that supports **no** features — it is selected only for a
feature-free d=64 graph.

ᵏ **THD / ragged backward** (`sdpa_bwd_sm80`). Q/K/V/O/dO and the gradients are
PACKED `[1, T, H, D]` **BSHD rows, each at its own token and head stride**:
element stride 1, head stride `>= D`, token stride `>= H * head_stride`, each a
multiple of 8 elements (every head base 16-byte aligned for the `cp.async`
loads). A compact port is the common case; a K/V view into an interleaved
`[T, 2, H, D]` record (token stride `2*H*D`, the fused-KV slicing layout) or a
head-interleaved record (head stride `D + gap`, what the ragged sweeps fuzz) is
served at those strides, the gap cells never read or written. Stepped strides
are plan-time specialization; packed token capacities remain dynamic in the
compiled host.
Lengths arrive as the graph's per-batch `seq_len_q/kv` (`use_padding_mask=True`)
and become `cu_seqlens` on device in the setup launch, which also reads **the
bound ragged offsets of every port** (issue #737): each of Q/K/V/O/dO, dQ/dK/dV
and Stats places sequence `b` at its own token origin `ro[b] × M / ts` (`M` the
port's `ragged_offset_multiplier`, `ts` its token stride; Int64 arithmetic), so
a padded layout with gaps between sequences (TE's `cu_seqlens_padded`, per side
or per port) is read and written where the caller put it and the gap rows are
never touched. Origins are recomputed from the bound values on every execute
and CUDA-graph replay. The internal accumulators stay packed at
`prefix(lengths)`. Whole-token offsets are a supported-input precondition of the
FROST engines (element-unit offsets are fine when `offset × M` is a whole
number of tokens); they are not checked on device. **Declared
`max_total_seq_len_q/kv` are required** (the fp32 dQ accumulator, the
per-query-head dK/dV partials and do_dot are sized from them at build time).
Ragged Stats is read in either packed packing — token-major `(T, H)` or
head-major `(1, H, head_stride)` with any head stride covering the packed
capacity (the compact `[1, H, T_q]` the SM80 forward wrapper emits, or the
FROST forwards' 64-rounded one). The kv-tile grid and the deterministic relay
counter are bounded by the envelope `S_max` (short tiles early-out; no length is
read on the host); the dQ cast and the dK/dV fold stop at `cu_*[B]` on device and
write the caller's head dim, so nothing past the packed total is ever written into
the caller's gradients (the ragged sweeps assert this with a NaN-filled tail). Served under THD: the causal family, GQA/MQA, sinks/dSink,
deterministic dQ, head-dim envelope padding (carved staging at the packed
capacities), zero-length and one-sided-empty sequences. Bias/dBias is dense-only (a packed graph has no `[B, H, S_q, S_kv]` bias); RoPE is a
wrapper-only fusion no engine row admits. The SM80 forward row now serves
native-flavor, compact packed THD with Q/KV lengths; its dense head-dim
envelope and fused bias/sinks remain unavailable to THD.

---

## Gaps at a glance

| Missing | Where |
|---|---|
| Gated attention block (frontend-only, no graph form): e4m3 (`kv_cache_dtype`) pools on the DENSE paged forms (the decode tile's in-kernel dequant arm), on the packed prefill's write-through, at `S_q > 4` and with an LSE / a split on the block-sparse decode mode (served 2026-10-14 at `S_q <= 4` through the gathered cast + the decode tile, and as the quantizing write-through on the dense prefill forms), the write-through folded into the projection epilogue, an LSE / a window / another head dim on the DENSE paged forms, a dense step wider than 32 rows (dense geometries), a `QsaSpec` on the PACKED EXTEND form, the in-block indexer over the pools | SM107 -- the block's paged KV-cache WRITE-THROUGH (`paged_kv_page_size`: bf16 / f16, dense and packed), the EXTEND form (`kv_source="paged"` + `token_capacity`: the packed step's attention over the pools through `block_table` / `kv_lens` on the bottom-right diagonal) AND the DECODE form (`kv_source="paged"` on a dense `[B, S_q <= 4]` block: the Rubin d256 decode tile, PackGQA + split-KV with the gate in the combine; the unpacked tile up to 32 rows declared, unmeasured) are served on HND / NHD / unified-row pools at pages 16 / 32 / 64 / 128, bf16 / f16, AND the BLOCK-SPARSE DECODE / extend-with-prefix mode (`kv_source="paged"` on a dense `[B, S_q]` `QsaSpec` block, 2026-10-14: the sparse core reading the pools, one list per sequence at `S_q <= 4` and one per row past it, the gate and the LSE, split-KV) on HND / NHD / unified-row pools at pages 16 / 32 / 64 (the `Rubin d256 gated attention block -- serving forms` section below) |
| Index-list (block-sparse) forward: a GRAPH form (no `sdpa` node carries a block-index list), a sink, a band, Q-length trimming of a dense batch, log2 stats; the in-block indexer (through the block); THD and paged pools in ONE declaration; the decode form's arms (bottom-right, the per-sequence list, split-KV) under THD; a shared list at `S_q > 4` | SM107 — frontend-only: the gated attention block's sparse stage (`GatedAttentionBlockGeometry(qsa=QsaSpec(...))`, `execute(block_ids=, block_lens=)`) and the standalone `sparse_gqa_sm107` adapter serve dense BSHD bf16 / f16, d = 256, caller lists at `top_k` in [4, 512] (multiples of 4), query-head groups 1..16, per-batch KV lengths, the fused epilogue gate (`epilogue_gate=` the gate operand on the adapter, `fuse_gate=True` through the block: O gated in place, the LSE untouched; it composes with the paged read and with THD), and packed THD sequences (`thd=True`: per-sequence `[B]` or cumulative `[B + 1]` lengths (`cu_seq_len`), sequence-relative block ids; through the block `GatedAttentionBlockFwd(thd=True, num_sequences=, max_seq_len=, cu_seqlens=)` + `execute(seq_lens=, block_ids=)`); the standalone adapter also reads paged K / V pools (`paged_kv=True`: `[num_pages, H_kv, page_size, D]` HND compact or NHD by strides, one `(B, max_pages)` block table, `page_size` a positive multiple of 4, per-batch lengths required) -- one of the two arms per declaration (`thd=True` with `paged_kv=True` is a typed decline) -- and the DECODE form (`bottom_right=True`: positions `kv_len - S_q + row`; `list_per_sequence=True`: `block_ids [B, top_k]` shared by the sequence's `S_q <= 4` rows, anchored at the step-0 position; `split_kv=S` up to `ceil((top_k + 1) / 32)`: fp32 partials in the caller's workspace + the split combine, the gate riding the combine) -- through the gated attention block since 2026-10-14 (`kv_source="paged"` on a dense `[B, S_q]` `QsaSpec` block: `block_table` / `kv_lens` / `slot_mapping` / the pools through `_SparseSdpa`, the list form derived from `S_q`); every other arm is a typed decline by name (see the SM107 index-list section below) |
| Backward pass entirely | SM90 |
| Backward outside d ∈ (256, 512] ∪ {256} (f16/bf16) or d = 256 (per-tensor FP8 E4M3 and block-scale MXFP8 E4M3): every other head dim; and on the d256 rows the dense GRAPH padding mask (per-batch kv lengths ride every row's standalone adapter), sink / dSink, bias / dBias, deterministic, `dense_flex`, right-band widening, decode; on the MXFP8 row also fp16 gradients and the `amax_dQ/dK/dV` outputs; on the d512 row THD, `dense_flex`, the dense padding mask, sink / dSink, bias / dBias, deterministic, decode | SM107 — the three d256 rows plus the d512 2x2-datapath row (`sdpa_bwd_sm107_d512`, opt_in, since 2026-10-01) are the whole Rubin backward (see the SM107 table, ᵇ ᵐˣ and ᵇ³) |
| Backward outside d ∈ (256, 512] ∪ {256} (f16/bf16) or d = 256 (MXFP8) | SM100, SM103 — the three backward engines there serve exactly those bands (f16/bf16 d = 256 via the 2x2-datapath row ᵇ², opt_in, since 2026-10-01) |
| Backward per-batch padding mask (`seq_len_q/kv`) on a DENSE graph | SM100, SM103 — a UNIFORM non-tile-multiple length is served, and the THD path carries per-sequence lengths; a per-batch mask on a dense graph is not |
| Backward sink / dSink, bias / dBias | SM100, SM103 |
| f16/bf16 d = 256 backward: dense padding mask, right-band widening, THD, `dense_flex`, decode, deterministic; the 4-CTA multicast arm (perf) | SM100, SM103 — the ᵇ² row serves dense / causal (top-left, bottom-right) / sliding window / GQA on BSHD only |
| Backward deterministic, decode | SM100, SM103 — served by the MXFP8 d=256 row only |
| MXFP8 backward: E5M2, bottom-right / band-widened / sliding-window masks, non-BSHD strides, `amax_*` outputs | SM100, SM103 |
| f16/bf16 forward split-KV | SM90, SM80; SM107 D512 and dense D256; THD outside D128/D256 and nonpaged D192 |
| f16/bf16 forward PackGQA | SM107 outside dense D128 GQA (shared bodies, not with the pre-folded scale), D128 paged THD / nonpaged D128 split THD and exact D256 paged unsplit THD with CGA2 |
| d192×d128 quantized PackGQA / split-KV, and d192 MXFP8 THD | SM107 — the shape is served in FP8 and MXFP8 as of 2026-09-09, and per-tensor FP8 **THD** with it; PackGQA and split-KV stay wired in the d128 flavor only (`pack_gqa_d_shapes` / `split_d_shapes`), and the MXFP8 line declines THD outside d256 (`SM107_MXFP8_THD_SHAPES`, ᶻ) |
| MXFP8 forward | SM90 and SM80 (SM107 is served — see the SM107 table; d512 is ⚠️ⁱᵛ, correct but with no test module -- covered through the graph API by `test_mhas_v2.py::test_sdpa_mxfp8_fwd_default_walk_cc107_L0` and `::test_sdpa_mxfp8_fwd_cc107_L0`, d512 being one of the four exact flavors drawn; a dedicated module is still open; SM120 serves d ≤ 512 — see the SM120 table) |
| Per-tensor FP8 backward | every arch except SM107 d = 256 E4M3 (`sdpa_bwd_sm107_fp8`, ᵇ) |
| MXFP8 backward outside SM100/SM103 d = 256 and SM107 d = 256 (`sdpa_bwd_sm107_mxfp8`, ᵐˣ) | every arch |
| THD / ragged backward | SM120, the SM107 d512 row (ᵇ³) and the SM100/SM103 MXFP8 and f16/bf16 d256 (ᵇ²) rows (the SM100/SM103 f16/bf16 d512 row serves it — see ʰ; the SM107 f16/bf16 and per-tensor FP8 rows — see ᵇ; the SM107 MXFP8 row — see ᵐˣ; SM80 — see ᵏ) |
| THD forward outside native-flavor dims or compact packed Q/K/V/O | SM80 — the graph path serves only the four native head-dim pairs and copy-free packed BSHD buffers |
| **Decode tile on the Rubin line outside the d256 f16/bf16 flavor** | **SM107** — d128 / d192×d128 / d512 decode and every fp8 / mxfp8 decode run their flavor's prefill kernel; the d256 f16/bf16 decode tile (ᵈʳ: whole-group PackGQA, dense / paged split-KV, the MTP step in token units of the 32-column tile ᵐᵗᵖ) is the only Rubin decode tile today |
| **Paged d256 decode-tile default placement outside its measured bands** | **SM107** — the MTP rows (`S_q` 2-4 on 24/2 and 32/2, the 32-column tile in up to two token units ᵐᵗᵖ) and group 32 (64/2, the 32-column tile in one unit) LEAD by default inside their measured bands since the follow-up to the first arm (the MTP box b 1-128 × KV 1k-128k at page 16 bf16: 0.01-0.31 of the backend's default plan, which costs 6.8-142 ms per launch at KV 128k; group 32 from 4 units and 2¹⁷ KV tokens in flight: 0.27-0.79) although the 32-column tile still trails FlashInfer's own fa2 decode kernel on part of them (6 % at q 2, 21-22 % at q 3-4; 7-9 % on group 32, at the medians) -- its per-CTA issue-rate gap at 24-64 packed rows (212-SM cc 10.7 perf class, cuDNN 9.27.0, 2026-10-13), a kernel lever, not a placement fact: FE's verdict is against the backend's plan. Still backend-first because unmeasured: the MTP rows at page 32 / 64 and in f16 and on the 12/1 / 16/1 geometries (the same groups, unmeasured at `S_q > 1`), group 32 on 32/1 (the same group, unmeasured), at page 32 / 64 (measured on 24/2 only), below 2¹⁷ tokens in flight (its 2¹⁶ step holds a parity cell) or with fewer than 4 units; Stats, sinks, a mask or window at `S_q = 1`, page 128, fewer than 4 units (1-2 units led only at the 128k endpoint), above 512 units or past 128k declared KV, a dense (unpaged) d256 cache at `S_q = 1` (measured against the prefill body only), THD queries, the fused gate, a paged decode without per-request KV lengths (no padding mask: every band cell carried one). The head-major in-page pool layout (each head's page contiguous, vLLM's `[blocks, 2, H_kv, block, D]`) is admitted beside the token-major one the band measured: measured 2026-10-14 on the 24/2 core geometry with the same protocol on an EXCLUSIVE GPU of the same 212-SM cc 10.7 class (lock 2376 sampled: busy median 2364; cuDNN 9.27.0 build 9.27.0.21; b 1-128 x declared KV 1k-128k at page 16 bf16, 48 cells x 2 regimes): the tile 0.36-0.92 of the backend on EVERY in-band cell in both regimes (32 cells per regime, all quotable; medians 0.45 replay / 0.45 eager), the out-of-band cells mixed 0.73-2.00 as in the token-major band. Follow-up: the kernel fix of the 32-column tile reopens both holds (a placement constant each); the Stats / page-128 twins are one measured chunk each (`placement._sm107_paged_d256_decode_tile_leads`) |
| **Native d=64 (GPT-OSS) forward kernel** | **SM107** — served via the d128 envelope at ~2× MMA cost. SM100/SM103 is native (⁷) for f16/bf16 (prefill, decode (ᵈ⁶⁴), paged and split-KV), per-tensor FP8 (prefill, paged, split-KV; cga1) and MXFP8 (dense / unsplit / unpaged; cga1) |
| Decode tile outside the d128 / d256 f16/bf16 flavors | SM107 — dense d128 f16/bf16 rides the shared decode tile since issue #1472, and decode-shaped dense-Q d256 f16/bf16 (dense or paged cache) rides the Rubin d256 decode tile (ᵈʳ); d192×128 / d512 and every quantized flavor run their prefill kernels, d256 graphs past the tile's route run the d256 prefill kernel, THD keeps the prefill tile (the ragged-Q leg is SM100-line only), dense paged queries outside the d256 decode tile are not wired. SM100, SM103 — dense/paged d192×128 and d512 decode and every fp8 / mxfp8 decode have no dedicated decode tile: each runs its flavor's prefill kernel at that flavor's own CGA width (f16 d512 and the quantized d128 flavors at `TILE_CGA_M=2`; per-tensor FP8 d256 and SM100 MXFP8 d256 / d512 are cga1 kernels; d192×128 selects 1 or 2 by shape). Nonpaged, unpacked D192 THD can select the shared single-Q pipeline described below. Unsplit THD queries on the d128 f16/bf16 flavor keep its prefill pipeline (`TILE_CGA_M=2`) too (ᵈᵗ); exact D128 split THD can use the single-CTA path; d256 f16/bf16 graphs the adapter does not route onto the d256 decode tile (THD, or more packed Q rows than it routes, ᵈ) run the d256 prefill tile; on cc 10.7 the MXFP8 pools likewise run the d128 (`TILE_CGA_M=2`) / d256 (cga1) prefill tiles at `S_q <= 8` (the paged-vs-dense measurement is the R15 note of the SM107 paged MXFP8 pools section below: paged within 2 % of dense at the qwen35 decode shape; a decode tile for the quantized flavors is a P3-class follow-up) |
| **d192×d128 paged decode tile** | SM100, SM103 — paged (192, 128) is served (ᵖ) but at `S_q ≤ 8` runs the prefill tile. Measured on B200 (`S_q = 1`, `b = 32`, page 16, bf16, mixed `S_kv ≤ 4096`, default plan): 32/32 MHA **788.7 µs on the prefill tile vs 476.9 µs on the backend**; 32/8 GQA 275.8 vs 199.6 µs. Follow-up: a d192×d128 decode tile behind `TILE_CGA_M=1`, as ᵈᵗ is for d128 |
| **d512 paged decode tile** | SM100, SM103 — paged d512 is served (ᵖ) but a decode-shaped (`S_q ≤ 8`) paged d512 graph runs the d512 PREFILL tile (cga4 role-split, 256 Q rows per cluster). Measured on the FlashInfer decode shape (`S_q = 1`, `B = 8`, `d_qk = d_v = 512`, page 16, bf16, mixed `S_kv ≤ 4096`, B200, CUDA-graph replay, kernel time): 64/1 MQA prefill tile **77.8 µs vs 65.1 µs on the backend's paged decode engine**; 64/8 GQA 136.9 vs 102.5 µs. `sdpa/fwd/placement.py` keeps the backend first for paged d512 at `S_q = 1` by default (the FROST plan serves it under the opt-in; multi-token paged d512 keeps the decode-shaped lead). Follow-up: a d512 decode tile, the d128 ᵈᵗ recipe on the d512 geometry (`test_sdpa_fwd_paged_d512_decode_frost_pinned_L0` pins the shape) |
| d=64 quantized THD; d=64 MXFP8 paged / split-KV | SM100/SM103 (`thd_d_shapes` of both quantized rows, the MXFP8 row's `paged_d_shapes` / `split_d_shapes`, mirrored by `check_support`); every d=64 MXFP8 graph on SM107 (exact-shape gates) |
| Bias forward | SM90, SM100, SM107, SM120 |
| Forward with more than 65535 heads, or batch × split_kv (#1573) | every arch: CUDA caps `gridDim.y/z` at 65535 and `SCHED_POLICY=NATURAL` launches heads and batch × split_kv there, so `mismatch` (`launch_grid_mismatch`) declines a NATURAL set and the heuristic re-emits each unsplit set under LPT (flat grid). Declined outright: a flavor whose scheduler domain is NATURAL only, the d256 decode tile past 65535 batch (batch on `gridDim.y` under every policy), ragged graphs on SM80 / SM90 (3-D grid under every policy) and split_kv > 1 past 65535 heads or batch (the combine launches both on `gridDim.y/z` under every policy); the packed THD split combine launches only heads on `gridDim.y`, so it is declined past 65535 heads only. Measured 2026-10-11, cuDNN 9.27, b or h = 65536: NATURAL fails to launch on B200 (d64 / d128, prefill and `S_q = 1`), A100 (d64 / d128) and H100 (d512); LPT and LPT_L2 match the fp64 reference (max error ≤ 5.3e-4) |
| Dropout, ALiBi, `block_mask`, `score_mod` | every arch, both passes |
| Paged KV cache | SM107 serves half THD D128/D256 with or without an attention sink, plus decode-shaped dense-Q half d256 graphs (`S_q · G` within the Rubin route ᵐᵗᵖ: ≤ 16, the 32-column tile, a packed MTP step of two token units; sink included) on the d256 decode tile (ᵈʳ) (other dense paged queries and the d192×128 / d512 half pools stay declined there) and MXFP8 pools on D128 / D256 with dense queries (`page_size % 128`, 128 / 256 / 384 / 512 validated, HND / NHD, sinks compose; THD queries over MXFP8 pools and the d192×d128 / d512 MXFP8 pools are the SM107 follow-ups below); otherwise every arch except SM100/SM103 forward on f16/bf16 d128 / d192×d128 / d256 / d512, per-tensor FP8 d128 and MXFP8 on every native flavor (see ᵖ); the FP8 d512 flavor, MXFP8 pools with unreordered SF or page_size < 128 (page 64 included: a 64-row page cannot hold an F8_128x4 atom set, so a product page-64 contract needs a different SF encoding, not a half-page re-addressing of a 128-declared pool), THD queries over MXFP8 pools, packed (ragged-offset) block tables everywhere (THD queries over f16/bf16 pools ARE served — see ᵖ); THD queries and the attention sink over FP8 pools, a block-scaled O (`sf_o`) over FP8 and MXFP8 pools |
| Fused epilogue gate (`O * sigmoid(G)` tail) | every arch and flavor except SM107 d256 f16/bf16, per-tensor FP8 and MXFP8, exact (256, 256), dense / unsplit / non-PackGQA / non-paged -- plus, f16/bf16 only, the d256 decode tile's split (`S_q · G` within the Rubin route ᵐᵗᵖ: ≤ 16 rows on the 16-column tile, (16, 32] on the 32-column tile, a packed MTP step of up to two token units; split_kv ≥ 2; not with execute-time shape overrides -- the combine binds G to the declared (B, H_q, S_q, D_v)), whose combine applies the gate over a dense or paged cache, packed or not (see the SM107 table, ᵈʳ) |
| PackGQA of a group sharing no factor with the 128-row tile (G = 3, 5, 7, …), and partial packing outside the SM100/SM103 f16/bf16 d128 / d256 kernels | every arch — such groups run unpacked (see ᵐ); the d192×d128 / d512 f16 and the fp8 / mxfp8 kernels pack the whole group only |
| Attention sink + split-KV outside the SM100/SM103 f16/bf16 rows (dense, paged, the d128 decode tile's ragged-Q leg and the d256 decode tile -- sink-free partials, one fold in the combine, see ˢ) and SM107 exact D128 half paged THD Q>1 | The per-tensor FP8 and MXFP8 rows (per-tile epilogue fold), gated splits and the cc 10.7 dense legs remain unsplit with a sink; the qualified packed-combine exception is described below. Measured cost on cc 10.7 dense d128 half (216 SMs, cuDNN 9.26.0.51, CUDA-graph replay, issue #1472's lever D): the sink graph's best listed plan (the unsplit packed decode tile under LPT, its default since the scheduler re-fit) at b1 / b4 64/8 over KV 16k / 32k takes 73 / 135 us (b1) and 75 / 137 us (b4) -- its NATURAL runner 82 / 154 and 83 / 154 -- while the sink-free twin's split tile (split 16 at b1, 4 at b4) takes 19 / 26 us and 34 / 57 us: 2.2-5.3x on the table against the best listed sink plan (2.4-6.0x against the runner), far past the 1.15x bar for a follow-up (remaining work: bring the SM100 dense sink combine to the cc 10.7 dense legs and qualify the still-declined layouts and shapes before extending their admission and selection; preserve Rule S4 natural-log partials and sink-free behavior; owner: the issue #1472 lane); the backend declines `S_q == 1` with a sink (FROST is the only provider there) and at `S_q == 8` its default is 191-674 us against FROST's 74-137 |
| Attention sink at `S_q == 1` validated | every row except SM100/SM103 f16/bf16 (see ˢ), SM90 f16/bf16, SM107 f16/bf16 (cc 10.7: the native bodies via the s_q == 1 sweep's sink draw, the shared paged bodies via the paged THD + sink block ᵖˢ) and SM107 MXFP8 (`test_mhas_v2.py::test_sdpa_mxfp8_fwd_default_walk_cc107_L0`: `s_q == 1` draws with an e4m3 sink on d128 / d192×d128 / d256 / d512 against the MXFP8 reference incl. `Amax_O`, cuDNN 9.26.0.51 on the cc 10.7 board and 9.28 on the Rubin CI lane, 2026-10-08): SM120 f16/bf16 accepts it since the validator lift (f16/bf16 `sdpa()` graphs only) but is ❔; the FP8 rows and the SM100/SM103 MXFP8 row were never gated by that rule and stay ❔ as before. SM90 f16/bf16 is validated too (H200, `test_sdpa_fwd_dsl_sm90.py`: the #1095 accept graphs under a strict SM90 pin — dense packed / unpacked with a keyless batch, a THD envelope of one in both packed Stats layouts, base-2 Stats); paged KV and split-KV stay declined there by their own rows |
| MXFP8 decode-shaped graphs on cc 10.7: the SM107 MXFP8 row has no decode tile, so a dense `S_q <= 8` graph runs its 512-row 2-CTA prefill tile with 1..8 live rows; it leads by qualification (the backend's cc 10.7 MXFP8 plans mis-report Amax_O, crash the planner at `S_q == 1` without a sink and do not build at d256 / d512) and trails the backend's decode-shaped engines where those plan at all -- w2u1g-lc-0614 (216 SMs, cuDNN 9.26.0.51, CUDA-graph replay, interleaved rounds, 2026-10-08): B64 H8/2 `S_q=1` KV4096 d128 sink FROST 113.3 us vs eng3 81.5 / eng16 70.1; B128 H32/8 `S_q=1` KV2048 sink 766.8 vs 423.3 / 375.8 (board throttling); B64 H8/2 `S_q=8` KV4096 114.2 vs 81.6 / 69.9; B64 `S_q=8` causal-BR sink 129.3 vs eng16 91.7; at B4 `S_q=1` the row is faster (16.0 vs eng3 20.0, whose Amax_O reads 0); prefill shapes sit at parity to 1.25x (`sdpa/fwd/placement.py` docstring, PR #1478); follow-up: an MXFP8 decode tile for cc 10.7 (the quantized twin of ᵈᵗ), never a backend-relative placement rule | SM107 MXFP8 d128 |
| Paged FP8 decode tile: the d128 paged FP8 kernel is a prefill tile (one 128-row Q tile per batch and KV head), so a decode-shaped (`S_q <= 8`) paged FP8 graph runs it FROST-first under the opt-in — fp8 d128 paged decode, S_q=1, B=32, 96/8 heads (group 12 does not divide the tile, PackGQA off, one live row per tile), B200: prefill tile 807 us vs the backend engine 54.9 us (ᵖ); follow-up: an fp8 d128 decode tile (the quantized twin of ᵈᵗ) | SM100, SM103 — per-tensor FP8 paged d128 |
| Paged FP8 short-`S_q` prefill: an S_q=64 fp8 paged prefill graph (B=4, 16/4 heads, d128, page 16, max KV 2048, e4m3, bf16 O) defaults to the FROST `PACK_GQA=1` / `SPLIT_KV=2` plan at 50.0 us GPU / 186-189 us CPU enqueue against the backend engine's 28.7 us / 15-16 us (148-SM SM100, cuDNN 9.25.1, independent review measurement; B200 / cuDNN 9.26: 204.7 vs 30.4 us at S_q=64, 135.6 vs 31.9 us at S_q=512, ᵖ); follow-up: the prefill tile's tile / split heuristics for short-`S_q` paged fp8 and the fp8 d128 decode tile's MTP reach | SM100, SM103 — per-tensor FP8 paged d128 |

ᵏ **THD / ragged forward layout.** Q/K/V/O must be BSHD-ordered over **(H, S, D)**
only — head dim innermost, then heads, then tokens (`graph_analyzer.packed_layout_ok`).
The batch stride is not gated: every sequence base comes from the ragged offsets and
the lowering binds the batch axis at extent 1, so its declared value is never read.
FlashInfer declares it equal to the token stride (`h * d`), which the previous
all-four-axes check refused at `b > 1`. Stats under THD are written in the caller's declared layout: packed `(T, H)` rows, head-major `(1, QH, head_stride)`, or -- a Stats tensor **without** ragged offsets -- the per-batch padded form -- the graph's logical `[b, h, s_max, 1]` Stats view over FlashInfer's physical, contiguous `(b, s_max, h)` `return_lse` buffer (declared strides `[s_max*h, 1, h, 1]`; the adapter rebuilds the view with `as_strided`, nothing is allocated in the logical order), stored per batch through the declared strides on every THD row (SM100 / SM107 / SM120, `Capabilities.thd_padded_stats`); the adapter seeds that buffer with `-inf` on the launch stream first, so the rows past a sequence's length read the backend's value. On SM100 / SM107 the THD templates compile with DYNAMIC batch and head extents (`compile(dynamic_bhk=True)`): one artifact per layout class (d, dtypes, masks, GQA ratio, packed vs declared strides), not per shape.


### Prepared SM100/SM103 MXFP8 forward launch contract

Scalar-output MXFP8 dense, native-shape THD and the existing dense split-KV
paths use prepared pointer launches for D128, D192x128, D256 and D512.
THD shape/stride overrides bind within the declared envelope; dense MXFP8
keeps fixed geometry because its opaque scale-factor batch/head pitches are
plan-specific. SF storage may be any dense physical-axis permutation and is
validated against the producer's observed byte span. Packed SF tile totals are
runtime metadata, never read from device lengths or used as compile keys.
D512 retains half split partials; the other three flavors use FP32 partials.
D128 block-scaled O and the existing direct-only D128/D192 PV-BF16
specializations also use prepared launches for native layouts. SM107 MXFP8
prepares its existing dense scalar-output and D256 gate paths as described above.
Standalone prepared calls require the declared
caller workspace, like graph execution; no plan owns device scratch.

### Prepared SM107 d256 backward launch contract

The cc 10.7 d512 row (`sdpa_bwd_sm107_d512`, ᵇ³ of the SM107 table) is NOT this
contract: it rides the SM100 large-head-dim chain's prepared host unchanged
(`bwd/prepared_sm100.py`, `kernels/sm100/prepared_host.py`; artifact symbol
`frost_sdpa_bwd_sm107_d512_prepared`) — dense BSHD-physical operands, contiguous
Stats, the S / dS head-chunk workspace carved from the caller's buffer.

The three Rubin d=256 backward rows (`sdpa_bwd_sm107`, `sdpa_bwd_sm107_fp8`, `sdpa_bwd_sm107_mxfp8`) use one
prepared pointer launch per plan (`bwd/prepared_sm107.py`, `kernels/sm107/prepared_host.py`):
dense BSHD-physical operands and contiguous Stats at the plan's fixed geometry (no
runtime shape overrides; a mismatching override or a changed layout is refused
before any stage launches), padding copies / fills / the fp8 fold passes (and, on
the bf16-dS twin, the Q / K upcasts) as kernels of the artifact, the caller-owned
workspace carved in a fixed order that
reproduces `get_workspace_size()` exactly.  The fp8 row binds its twelve scalar
descales / scales as 1-element fp32 device operands and ONLY the amax outputs the
graph requested (an unrequested amax is compiled out of the artifact).  Standalone
prepared calls require the declared caller workspace, like graph execution.

### Prepared block-scaled output launch contract

Existing dense D128 NVFP4 and MXFP8 outputs use prepared pointer launches on
SM100/SM103, SM107 and SM120/SM121 per-tensor FP8, and on SM100/SM103 and SM107
MXFP8 input paths; SM120/SM121 MXFP8 inputs serve the MXFP8 output. The declared
SF_O atom geometry stays fixed; each call binds current O, SF_O, optional Amax
and device scales with observed storage checks.
FP4 O uses packed byte geometry, while V keeps its full logical head dimension.
SF_O offsets and prepared copy strides preserve Int64 addressing.
THD, paged, split-KV, PackGQA, gated output and shape-override combinations keep
their existing admission boundaries. Remaining conversion layouts use prepared
gather/scatter copies around a compact prepared plan. Standalone prepared calls
require caller-owned workspace.

### SM100 / SM103 paged D128 THD split follow-up

SM103 shares the SM100 packed split admission and native binding contract for
this path and nonpaged D192/V128 below. Both use the existing half single-CTA
kernel and combine; no new template or execute-time preparation is introduced.

Native FP16/BF16 D128/V128 paged THD prefill may use one-CTA split-KV with
optional GQA packing. It requires a matching native binder, bounded packed-Q
workspace, and packed NH/HN Stats; output Stats may use ln or log2. Q=1
retains its existing ragged decode path. Automatic selection is narrower:
SM100/SM103 BF16 B1..4, HND page16, Hq4..64 with integral GQA ratios 1/2/4/8, ordinary bottom-right
causal attention, Q64–1024 and KV2048–16384, without shape overrides. Other
legal split records remain explicit tuning choices. The policy counts all
batches when estimating the grid and retains four KV tiles per split; grids
that already fill its bounded wave budget keep the existing selection.

### SM100 / SM103 nonpaged D192/V128 THD small-Q and fixed split

The existing FP16/BF16 D192/V128 nonpaged, unpacked THD single-CTA choice
uses the shared 128-row Q pipeline (two KV stages). The two-CTA prefill
pipeline and quantized paths retain their existing kernels. Exact SM100/SM103
D192/V128 graphs can explicitly select split-KV on the single-CTA path,
using caller-owned, bounded packed partials and the existing combine. NH/HN
Stats are optional and may use ln or log2. Sink and padded-Stats split
combinations remain declined. Shape overrides require a declared positive
`max_total_seq_len_q` within the graph's packed-Q capacity. This extension
adds no public tuning knob. A bounded BF16 automatic policy fills the first
wave of 128-row single-CTA work and retains at least four KV tiles per split.
It covers fixed SM100/SM103 nonpaged THD graphs with B1..4, equal Q/KV heads 4..64,
Q64..1024 and KV2K..32K, with KV at least four times Q, no window/sink/gate,
and either no causal mask or bottom-right causal masking. These measured
split choices lead the backend with or without packed Stats. Other graphs,
including shape overrides and full prefill, keep their previous automatic
policy; explicit legal splits remain available.

### SM107 D128 paged prefill cluster width

The half THD D128 paged path admits explicit CGA1 and CGA2 for unsplit
execution. CGA1 reuses the shared two-slab prefill template (256 packed query
rows per CTA), whereas native split uses the separate 128-row tile. The same
public knob describes the physical cluster width; no new tuning axis is added.
FP16/BF16, both page layouts, packed/unpacked heads, and changed-length/pointer
capture are covered by the paged prepared tests, including one-query unsplit
requests with ragged Stats. Dense, nonpaged and quantized
CGA domains are unchanged. The conservative BF16 preference chooses CGA1 only
when packed short queries fit one tile and remove a grid wave; split selection
retains its existing policy.

### SM107 bounded half serving layouts

Rubin's opt-in half row additionally serves nonpaged, unpacked D192/V128 THD
on the shared single-CTA pipeline, with bounded split partials and optional
packed NH/HN Stats. Paged D128/V128 and D256/V256 THD use the existing
Blackwell paged pipeline, compiled for SM107; D128 also supports PackGQA
and the single-CTA packed split path. D256 also serves explicit CGA2 packed
split with unpacked heads, using the matching native extension. The existing
packed binder, setup, caller-owned workspace and combine are reused.
Dynamic lengths and changed pointers are bound on each execution without
host readback or execute-time compilation. Split shape overrides need a
positive bounded `max_total_seq_len_q`; padded Stats remain declined. Split sinks
on cc 10.7 are supported only for exact D128 half paged THD Q>1, as described below (the SM100/SM103 half rows split with a sink on their dense, paged and ragged-Q decode legs -- SM100 table, ˢ). Paged dense queries and paged quantized inputs remain outside this
extension; paged sinks on the half THD legs are served (see "SM107 paged half
THD with attention sinks" ᵖˢ below). Unsplit D128 now admits NATURAL/LPT on Rubin;
MLA and native split paths retain NATURAL. Default placement is described
in the SM107 section above.

### Nonpaged D128 half packed split

SM100/SM103/SM107 FP16/BF16 exact D128/V128 nonpaged THD can explicitly
select one-CTA split-KV, with or without GQA packing. It reuses the existing
single-CTA pipeline, packed FP32 partials, setup and combine; no new template,
public knob, per-execute staging or allocation is added. The matching native
binder is required; older extensions decline this new domain.

Packed NH/HN Stats remain optional and may use ln or log2. Shape overrides
require the existing positive bounded packed-Q capacity. Padded Stats, sinks,
fused gates, quantized inputs and other head dimensions remain outside this
addition. Nonpaged D128 single-CTA unsplit is not admitted. Rubin's unsplit
nonpaged half PackGQA remains declined.

The existing nonpaged first-wave split rule additionally selects D128 on
SM100/SM103 FP16/BF16 graphs, fixed or bounded-override, with or without packed
NH/HN Stats, with B1..4, Hq4..64, integral GQA1/2/4/8/16, Q8..1024, KV2K..32K
and KV at least four times Q. Bounded graphs are judged on their declared
capacities. D128 counts the CTAs of each candidate: a packed GQA candidate
holds 128 / (Hq/Hkv) query tokens of one KV head's group per CTA, and a
declared `max_total_seq_len_q` bounds a ragged batch to at most one partial
tile per sequence. D128 candidates may fill two waves; among them it picks
the fewest waves times KV loop, then fewer partitions, then unpacked. It
excludes windows, sinks,
gates and right-band widening, accepts unmasked or bottom-right causal graphs,
retains at least four KV tiles per partition, and never overfills its wave
budget (two for D128, one for D192). Only an actual split selection leads the backend. Full prefill,
already-filled grids and other graph features keep their previous D128
automatic policy; explicit legal split records remain available. Rubin (SM107)
uses the same packed-CTA count, two-wave budget and Q8 floor for D128 since
2026-10-09 (gr100, 120 THD cases: mean regret 199.5% -> 10.6%, no lead-loss). The
D192 and paged selection rules retain their existing domains.

### D256 half packed split across Blackwell and Rubin

SM100/SM103/SM107 exact D256/V256 FP16/BF16 THD forward plans admit
explicit split-KV over both paged and packed nonpaged K/V. They reuse the
existing two-CTA D256 partial stores, prepared host and packed combine, with
NH/HN Stats, optional final log2 Stats, runtime device lengths and bounded
shape overrides. Nonpaged D256 requires the matching native extension. No
new runtime staging or scalar metadata readback is introduced.

CGA2 and unpacked heads remain required for split plans; sink, padded Stats,
fused gate and pre-folded-scale combinations remain declined. Quantized and
inexact head dimensions retain their existing limits. Automatic D256 split
selection is unchanged: the measured Rubin paged domain below is the only
one selected automatically; the new domains are explicit candidates.

### SM107 paged D256 half packed split

The SM107 half row additionally admits explicit paged D256/V256 THD split-KV
with CGA2 (explicit or default) and unpacked heads; explicit CGA1 remains
unsupported. It reuses the SM100 D256 main pipeline and the
shared packed host/combine; the native binder advertises this geometry
separately so older extensions decline it. FP16/BF16, NHD/HND page pools and
optional packed NH/HN Stats (ln/log2) retain the existing graph contract.
Shape-override plans require a bounded packed-Q capacity. Sinks, output gates,
padded Stats and dense queries remain outside this
extension. Automatic selection covers FP16/BF16, Hq4..64 with GQA groups
1/2/4/8/16, page16/128, bottom-right causal without a window or widened right
band, Q64..1024, and KV2K..32K with KV at least four times Q. FROST leads when
the declared two-CTA query grid fits one physical wave. Spare resident capacity
sets the split budget, capped at 16 partitions and at least four KV tiles per
partition on average; the final partition may contain a shorter tail. A full
first wave keeps the unsplit plan. Packed Stats and bounded shape overrides
use the same declared envelope; other requests retain their existing policy.

### SM107 paged D256 half PackGQA

The existing two-CTA D256 paged half pipeline also serves explicit PackGQA
with unsplit THD queries. Setup counts token tiles after head packing, and
Stats stores retain each row's true query-head coordinate. Graph and standalone
admission share the same exact-D256, SM107, paged-half, CGA2, split1 predicate.
Dense/nonpaged D256, quantized inputs, other architectures and D256 split+pack
remain outside this extension. No default selection is changed.

### SM107 paged half THD with attention sinks ᵖˢ

The cc 10.7 half row serves paged KV for packed (THD / ragged-Q) queries on the exact D128 and D256 flavors with
or without `sink_token` (2026-10-08). The two Rubin-specific sink gates (`engines.mismatch`'s paged rule and
`SdpaFwdDslSm100.check_support`'s Rubin paged clause) were lifted; no kernel changed: the legs run the shared
`sm100/prefill_d128_f16.py` / `sm100/prefill_d256_f16.py` bodies compiled for sm_107a, whose per-row sink fold
indexes sinks by packed query head (PackGQA composes) and selects a keyless row to O := 0 / LSE := sink. Validated
on cc 10.7 (216-SM board, cuDNN 9.26.0.51 and 9.27.0.28) by test_mhas_v2.py's P2 block: per-request Q 1 / 4 / 8
(and 0) mixed in one packed batch, chunked prefill (128 / 77 / 1 / 128), KV lengths 0 / 1 / page-1 / page / tile
boundaries up to 4096, page 16 / 64 / 128 pinned and 8..1024 swept, GQA 4 / 8 / 16 / MQA / MHA, PackGQA on / off and
cga1 / cga2 by explicit pin, HND / NHD pools, Stats off / token-major / head-major / base-2 (head-major on
single- and multi-request batches; a single-request head-major declaration below the packed capacity serves since
issue #1482), per-batch lengths or the cu_seq_len (B+1 prefix-sum) form, with or without the packed totals
declared (`max_total_seq_len_q/kv`), bottom-right / top-left causal, left window 128, right band and
band-around-diagonal masks, sinks -120 / -5 / +3 / +10 on keyless rows (O := 0 / LSE := sink asserted bit-exact),
dead pool pages NaN-poisoned on both routes (neither provider reads a dead table slot), f16 and bf16; the flag-free
common-API walk runs the same graphs (backend-served at s_q >= 2 on 9.26 / 9.27, row-served at s_q == 1 where the
backend has no sink engine). Still declined: dense (non-THD) paged queries, per-tensor FP8 pools (graph-level cells for
the THD and dense query forms) and MXFP8 pools (a separate change), d192×128 / d512 pools, and sink + split-KV outside exact D128 half paged THD Q>1. Default
placement keeps its sink exclusion on PAGED graphs (the paged packed-GQA prefill shard, Q 64-128, stays backend-first
with a sink); the cga1 two-slab preference and the LPT-for-packed-THD rule take sink graphs on cc 10.7 paged half since
the plan-ordering measurement (the next section: the B128 64/8 d128 Q 1 / 4 / 8 paged bf16 sink family proposes the
packed two-slab cga1 LPT set first, as its sink-free twin does) -- the paged THD + sink plan table is in the PR, not
asserted; the DENSE d128 sink decode / verify shard leads the backend inside its measured band (issue #1472).

### SM107 dense D128 decode tile and PackGQA (issue #1472)

Issue #1472 (dense BF16 d128 GQA with an attention sink under bottom-right causal masking at decode / verify
depth, B128 64/8 Q8 KV2056): the cc 10.7 half row ran the 512-row two-CTA Rubin prefill tile UNPACKED -- 8192
(batch, q-head) units of 8 live rows, the KV head streamed once per query head, the per-unit fixed cost (the
heuristics' own 21-KV-tile model) paid 8192 times -- 1236 us against the backend's 802 us default plan on the
board (216 SMs, cuDNN 9.26.0.51, CUDA-graph replay, kernel time).

**PackGQA on dense D128 half graphs.** A dense (not THD, not paged) cc 10.7 half graph on the (128, 128) flavor
-- the d64 envelope included -- without the pre-folded scale lowers its `PACK_GQA=1` plans at `TILE_CGA_M=2` onto
the shared SM100 prefill body (`sm100/prefill_d128_f16.py`, CfgD128 at cga2, 192 KiB, inside the version-0
descriptor window; loader tag `sdpa_fwd_sm107_d128_packed`), exactly as the paged legs borrow it: the Rubin
sibling carries no PACK_GQA arm, and the shared body's `row_head_idx` sink fold is the one the paged leg already
runs. `engines.rubin_dense_d128_shared_leg` is the predicate (twin `SdpaFwdDslSm100._rubin_shared_dense_leg`):
`heuristics._pack_gqa_eligible` packs such graphs, `engines.mismatch` and the adapter's PackGQA gate admit the pin,
the adapter's cc 10.7 "split_kv > 1 with PackGQA" clause is lifted for these legs (a heuristic-listed packed split
must build -- honored-or-never-listed), and the pre-folded scale declines on them (the shared bodies apply the
scale in-kernel; such graphs keep the Rubin body). Dense D256 PackGQA, partial PackGQA (96/8 -> 4) and MHA packing
as a proposal stay out (an MHA PackGQA pin is the bit-exact unpacked fold, honorable as on the SM100 row).

**The shared decode tile on dense D128 half graphs.** The same row now carries two tiles behind `TILE_CGA_M` on the
(128, 128) flavor, as the SM100 row does (`cgas_by_d_shape`): cga2 the Rubin prefill pipeline (512 rows per two-CTA
cluster), cga1 the shared 128-row decode tile `sm100/decode_d128_f16.py` (CfgD128Decode, 224 KiB, inside the version-0
descriptor window; already compiled for sm_107a as the `_single_q` THD leg; loader tag `sdpa_fwd_sm107_d128_decode`).
`heuristics._d128_f16_flavor` covers the cc 10.7 row, so `_d128_decode_tile_fits` proposes cga1 exactly when
`S_q × pack_g ≤ 128` (the candidate's own packing) and the packed leg goes first where packing wins (`_pack_gqa_wins`:
S_q < 512); the scheduler lead follows the cc 10.7 tables rather than the SM100 one-cluster rule (`_sched_points`, two
cc 10.7 arms keyed on `rubin_dense_d128_shared_leg`, `_SM107_DECODE_TILE_LPT_MIN_KV`): the packed decode tile walks
plain LPT first from a 2k cache on -- LPT / NATURAL median 0.989 over 224 (cell, policy) pairs of the dense sink decode /
verify tables at KV 2k-16k (58 cells more than 3 % ahead against 1), 0.944 at 32k (b1 / b4 64/8: 153 -> 135 us) --
while a short cache (26 pairs, median 1.000), a sliding window on either shared body (GPT-OSS d64 SWA128 on the decode
tile NATURAL 1.05-1.16 ahead; the packed prefill body under a window: b2 64/8 S4096 window 1024 NATURAL 214 us vs LPT
281, with a sink 215 vs 282, b8 64/8 S2048 window 512 373 vs 385, b1 64/8 S8192 window 2048 379 vs 393) and the
one-cluster packed cga2 band (30 pairs, median 1.013, 9 past 3 % for NATURAL, none for LPT) keep NATURAL first; the
other policy stays listed as the runner (per-cell exceptions exist both ways -- b8 32/4 q8 KV 2k ran LPT 1.32x slower,
b32 32/8 q8 KV 8k f16 0.80x -- and the unpacked MHA decode tile measured parity, so it keeps the Rubin no-GQA wave
rule). Dense only:
`engines.effective_cgas` narrows the domain back to `{2}` for THD (the tile has no THD_VARLEN leg; the paged cga1 THD
leg is the two-slab prefill body, the split cga1 THD leg the single-Q split), dense PAGED queries (not wired on
cc 10.7) and the pre-folded scale, so neither a proposal nor a pin reaches the tile there; the adapter's
`supported_cgas_for((128, 128))` admits `(1, 2)` on cc 10.7 and its THD cga1 decline covers the ragged case. The
standalone default width stays cga2 on cc 10.7 (the graph heuristics always pass an explicit cga); the d256 decode
tile, the ragged-Q leg over paged pools, dense paged decode and the quantized flavors stay where they were. Issue
#1472's B128 64/8 Q8 KV2056 becomes 1024 one-CTA 128-row units (64 live rows each) instead of 8192 two-CTA 512-row
clusters (8 live rows each).

**What stays out, and why.** A DENSE attention sink with split-KV stays declined (the dense combine folds no
sink; measured here as a lever-D candidate -- see the issue's follow-up -- and pinned by
`test_sdpa_fwd_cc107_sink_split_declines_L0`); dense D256 PackGQA and partial PackGQA need their own validation; the
ragged-Q decode leg over paged pools (sink-free, split-mandatory THD-binder path) and dense paged decode on cc 10.7
outside the d256 decode tile's route (ᵈʳ) are follow-ups; `_SM107_CGA_Q_ROWS = 256` in the Rubin no-GQA wave rule describes the d256 flavor (TILES_Q=1) rather
than the 512-row d128 / d192 clusters (TILES_Q=2) -- noted for the rule's next re-measurement, not changed here.

**Default placement (lever E, `placement._place_sm107_f16`).** Measured on a 216-SM cc 10.7 board (cuDNN 9.26.0.51 and
9.27.0.28, CUDA-graph replay, kernel time, every plan of the unified list timed round-robin, 2026-10-08): dense d128
half decode / verify WITH an attention sink -- bottom-right causal `1 <= S_q <= 16` or mask-free `S_q == 1`, GQA 4 / 8 /
16, `b * h_kv >= 32` units, caches 1k-16k, no window / right band / pre-folded scale -- runs its default plan (the
packed decode tile, or the packed cga2 prefill body once `S_q x G > 128`) at 0.06-0.42 of the backend's default on
every multi-token cell of the 150-cell shard family (b 8 / 32 / 128 x 64/8, 64/4 x q 1 / 4 / 8 / 16 x KV 1k-16k) and
at 0.06-0.30 on the 54 multi-token cells of the 90-cell grid (b 32 / 64 / 128 x 64/8, 64/4, 32/8 x q 1 / 4 / 8 / 16 x
KV 2k / 8k); 0.06-0.21 on the 36 multi-token cells of the 9.27.0.28 twin (b 32 / 128 x 64/8, 64/4 x q 1 / 4 / 8 / 16 x KV
2k / 8k / 16k).  The review-fix pass over the band's corners (9.26.0.51, ratios against the backend's default plan):
the head-dim envelope WITHOUT a window -- d64 64/8 at b 8 / 32 / 128 x q 1 / 4 / 8 x KV 2k / 4k 0.06-0.11 (an f16 twin
0.09), d96 at b 8 / 32 0.10-0.22 -- the 32-unit bound at G = 4 (32/8 b4, 16/4 b8, 4/1 b32, 8/2 b16; KV 1k-8k) 0.26-0.62
and at G = 8 / 16 (32/4 b8 0.19; 128/8 b4 q = 1 FROST-only), and f16 / Stats / padded twins at the band's corners
0.10-0.54 (the three families above are bf16, Stats-free and unpadded; these cells are the band's only measurements of
those attributes) -- every corner keeps the FROST default ahead.  The backend declines `S_q == 1` with a sink, so the
decode arm names the row's own default (8-695 us across the band).  Issue #1472's cells: B128 64/8 Q8 KV2056 102 us against the backend's 802 us
default / 752 us best (was 1236 us), B128 64/4 Q4 KV2052 58 us against 798 / 748 (was 1231 us), B1 64/4 Q512 KV512
16.3 us against 16.6 (prefill-shaped: unchanged, outside the shard).  Those graphs now LEAD without the opt-in flag
(`SM107_SINK_DECODE_*`; contract test `test_sm107_sink_decode_shard_stays_inside_its_configured_domain`); the GPT-OSS
d64 + SWA envelope cell (0.45-0.68 of the backend default on two cells) and GQA 2 / 32, MHA, partial groups, caches
past 16k and fewer units keep the backend first until measured.  Sink-free dense d128 decode on cc 10.7 (no sink, so
not in the shard): the split decode tile matches the backend's decode engine within 2-6 % (b 1 / 4 x KV 16k / 32k).

**Default placement, paged d256 decode (FlashInfer's cuDNN decode graph shape; `placement._sm107_paged_d256_decode_tile_leads`,
cudnn-frontend 1.32.0).** Measured 2026-10-13 on an EXCLUSIVE GPU of the 212-SM cc 10.7 perf class (SM clock locked at 2376 MHz and
SAMPLED, busy median 2364; cuDNN 9.27.0 build 9.27.0.21, the pre-release build the harness loads; CuTeDSL 4.8.0; FlashInfer's
`bench_gpu_time`, CUPTI kernel time with the L2 flushed per iteration; the backend's default plan and the first FROST plan each pinned
by name on the SAME FE graph; 3 rounds × 100 iterations per regime, the arm order shuffled, a second backend arm as the control pair
within 1 % or the cell is not quoted; CUDA-graph replay AND eager): seven geometries (24/2, 32/2, 16/4, 24/4, 8/1, 12/1, 16/1) × b
1-128 × declared KV 1k / 2k / 4k / 8k / 32k / 128k at page 16 bf16, page 32 / 64 at 24/2, one f16 twin per geometry; the pool as FlashInfer's benchmark strides it (`[pages, H_kv, page, D]` dims, token-major
in the page: head stride `D`, token stride `H_kv × D`). The Rubin d256
decode tile (whole-group PackGQA, one 16-row Q tile per (batch, KV head) unit) runs 0.43-0.90 (eager) / 0.43-0.81 (replay) of the
backend's default plan on EVERY quotable cell with `units = b · h_kv >= 4` and `units × KV >= 2¹⁶` tokens in flight -- 289 (cell,
regime) pairs per regime, 0.59-0.90 / 0.67-0.81 on the frontier cells (64 × 1k, 32 × 2k, 16 × 4k, 8 × 8k), 0.53-0.63 at the 512-unit
edge, page 32 / 64 0.43-0.85, the f16 twins 0.46-0.57 -- so that band LEADs by default (`SM107_PAGED_D256_DECODE_*`); one step below
(2¹⁵ tokens in flight: 32 × 1k, 16 × 2k, 8 × 4k, 4 × 8k, 1 × 32k) the cells are mixed, 0.78-1.79, and twelve are launch-bound and
unverifiable, so the band starts at 2¹⁶ (replay alone also led at 16 × 2k, 0.82-0.85, and 8 × 4k, 0.89-0.95: not admitted); fewer
than 4 units lose up to 1.79× below the 128k endpoint (1-2 units led 0.75-0.83 in that one column: not admitted); above 512 units
and past 128k declared KV unmeasured. The same measurement's two further bands, LEAD by default since the follow-up to the first
arm (the earlier hold on them is lifted; each band admits only cells where the tile is strictly faster than the backend with margin in
BOTH regimes): group 32 on its measured geometry 64/2 at page 16 (the 32-column tile in one unit, `S_q = 1`; 32/1 shares the group but was
not measured and trails; pages 32 / 64 were measured on 24/2 only and trail here) with `4 <= units <= 256` and `units × KV >= 2¹⁷` tokens in
flight -- the tile 0.27-0.79 of the backend on all 28 cells × 2 regimes there (every one quotable; the f16 twin 32 × 8k 0.38 both
regimes); one step below, the 8 × 8k eager cell sits at parity (1.00) and 16 × 4k / 32 × 2k eager are unverifiable, so the band starts
at 2¹⁷ although 32 × 1k (0.64 / 0.58), 16 × 2k (0.68 / 0.68) and 64 × 1k (0.54 / 0.79) led below it (not admitted), and 2 units led
only at the 128k endpoint (0.67; not admitted) -- `SM107_PAGED_D256_G32_*`; and the MTP rows (`S_q` 2-4 on the measured geometries 24/2 and
32/2 -- 12/1 and 16/1 share the groups but were not measured at `S_q > 1` and trail -- = the 32-column tile in up to two token units; FlashInfer's decode graph at `q_len_per_req` 2-4 declares the bottom-right causal diagonal with the
padding mask) over the WHOLE measured box b 1-128 × declared KV 1k-128k at page 16 bf16 -- the tile 0.01-0.31 of the backend's plan
(which costs 6.8-142 ms per launch at KV 128k) on all 576 (cell, regime) pairs, every one quotable, so no tokens-in-flight floor and the
2-unit corner included -- `SM107_PAGED_D256_MTP_*`; page 32 / 64 and the f16 twin were not measured on the MTP rows and keep the backend
first. On both bands the tile trails FlashInfer's own fa2 kernel on part of the cells (7-9 % at the medians on group 32; 6 % at q 2 and
21-22 % at q 3-4 on the MTP rows -- the 32-column tile's issue-rate gap at 24-64 packed rows, the gaps table's row): FE's verdict is
against the backend's plan on the same graph; which of its kernels a consumer routes to is the consumer's question, and the kernel
lever stays open. Not measured, backend-first:
Stats, a sink, a mask or window at `S_q = 1`, page 128, a dense (unpaged) d256 cache at `S_q = 1` (the dense table above measured it
against the prefill body), THD queries, the fused gate, a paged decode without per-request KV lengths (no padding mask: every band cell
carried one). The head-major in-page pool layout (each head's page contiguous, vLLM's cache; the same bytes per page as the token-major
one in a friendlier order) measured 2026-10-14 on the 24/2 core geometry with the same protocol on an EXCLUSIVE GPU of the same 212-SM cc 10.7 class (lock 2376 sampled: busy median 2364; cuDNN 9.27.0 build 9.27.0.21; b 1-128 x declared KV 1k-128k at page 16 bf16, 48 cells x 2 regimes): the tile 0.36-0.92 of the backend on EVERY in-band cell in both regimes (32 cells per regime, all quotable; medians 0.45 replay / 0.45 eager), the out-of-band cells mixed 0.73-2.00 as in the token-major band -- both in-page layouts are admitted. Contract tests
`test_sm107_paged_d256_decode_tile_leads_on_its_measured_corner_cells` (the frontier cells per geometry, both regimes; RED on the
pre-arm tree; the group-32 corners added with the follow-up, RED on the first arm's tree),
`test_sm107_paged_d256_mtp_rows_lead_on_the_measured_box` (the MTP box and its edges in both regimes, the diagonal as the keyed shape;
RED on the first arm's tree) and `test_sm107_paged_d256_decode_tile_shard_stays_inside_its_measured_shape`.

**Plan ordering from the measured tables (the same board and method).** Dense d128 half GQA under a diagonal band at
prefill depth packs first on the shared body (`heuristics._sm100_banded_gqa_packs` covers the cc 10.7 shared dense d128
leg: b2 64/8 S2048 66.7 us packed vs 76.8 unpacked, b4 64/4 S1024 50.8 vs 60.8, b1 32/8 S4096 60.5 vs 64.0, b1 64/4
S512 15.3 vs 16.3, with and without a sink; the unpacked Rubin tile stays listed).  A MASK-FREE dense d128 GQA graph
whose rows fit the decode tile unpacked while the packed unit overflows it (`S_q <= 128 < S_q x G`) rides the unpacked
decode tile first, unsplit (`_pack_gqa_points`; the opt-in order -- placement keeps the backend first there): b2 64/8
KV 4k q128 32.0 us against the packed cga2 body's 90.9 (split 2) / 98.1 (unsplit) and the backend's 103.7 (9.27: 30.0 vs
88.0 / 100.0), q64 31.2 vs 31.9 (the packed body's split-4 arm) and b8 64/4 q128 KV 2k 97.2 vs 97.3 at parity; past the
tile packing keeps the lead (b2 64/8 q256 KV 4k 93.9 packed vs 191.2 unpacked) and a band packs first.  The Stats twin
of the issue cell costs nothing extra on the decode tile: its first-pass 206 (NATURAL) / 151 (LPT) were intermittent GPU
sharing -- the 9.27 twin measured 101.3, and the re-measurement under a steady 2x share put the Stats cell at 206.6 /
206.7 (NATURAL / LPT) next to the no-Stats cell's 207.3 / 206.8 (parity across policies and cells).  Paged THD half (the shared paged
bodies; the backend offers no plan for packed queries over HND pools with or without a sink on 9.26 / 9.27, so the
row's first plan IS the serving plan): GQA16 joins the packed-first groups (`_prefer_thd_pack_gqa`: unpacked cga2 ran
2.8 ms against the packed set's 0.22 ms at b128 64/4 q 1 / 4 / 8 KV 2k, page 16 and 128; 184 vs 43 us at b24 mixed), and
the sink exclusions of the LPT-for-packed-THD rule and of the two-slab cga1 preference are dropped for cc 10.7 paged
half together with that rule's `s_q > 1` and GQA 4 / 8 terms (`_sm107_paged_half`): with every knob set pinned, the sink
cells ranked exactly like their sink-free twins -- 64/8 b128 q 1 / 4 / 8 default 363-366 us -> two-slab cga1 LPT
178-180 (the twins' own default, 182), b24 mixed 61 -> 41; 64/4 b128 q 4 / 8 2810-2816 -> 129-130, b24 mixed 184 -> 40
(cga2 LPT; one wave either way).  The SM100 line keeps its measured families untouched (contract test
`test_sm107_paged_thd_rules_do_not_move_the_sm100_row`); the paged packed-GQA prefill placement shard (Q 64-128) keeps
GQA 4 / 8 (GQA16 was not timed at that depth).

### SM107 paged MXFP8 pools (dense queries)

**Contract** -- the shipped SM100 paged MXFP8 contract (#1214) verbatim on cc 10.7, through
`graph.sdpa_mxfp8` (no new entry point): K/V pools `[num_pages, H_kv, page_size, D]` HND or NHD
(both pools alike); `descale_k [num_pages, H_kv, page_size, ceil4(D/32)]` F8_128x4 rowwise;
`descale_v [num_pages, H_kv, page_size/32, D]` F8_128x4 columnwise, D-plane-major across the WHOLE
pool for D > 128 (documented, not changed); `descale_q` dense `[B, H_q, ceil128(S_q), ceil4(D/32)]`;
`page_size % 128 == 0` (128 / 256 / 384 / 512 validated -- one to four 128-row tiles per page); `(B, 1,
max_pages, 1)` int32 tables, a separate V table and batch-innermost strides allowed; `seq_len_kv[b] <=
max_pages * page_size` is the CALLER's contract -- the kernel walks the block table for every page the
length covers and does not clamp it to the table's reach (inherited 1:1 from the SM100 pools); causal /
bottom-right / SWA / padding + sink composed; Stats optional; Amax_O; `softmax_precision=HALF`
composed; d128 (cga2) and d256 (cga1) with DENSE queries.
Page 64 is OUT and stays a typed decline: a 64-row page cannot hold an F8_128x4 atom set (the atom
interleaves rows `r // 32` at 4-byte granularity inside 16-byte TMA rows), so a product page-64
contract needs a different SF encoding.

**SF geometry per (page, head)** -- K: `page_size / 128` contiguous slabs of `SF_SMEM_SIZE_K` bytes
(512 B at d128, 1024 B at d256); V: `D / 128` planes of 512 B per 128-row tile, plane-major ACROSS
THE POOL: group `(page * KH + head) * (page_size / 128) + tile`, planes a whole plane of
`num_pages * KH * page_size / 128` atoms apart (what `_mxfp8_sf.build_columnwise_sf_desc` derives
from `num_batches = n_pages`, `num_tiles = page_size / 128`).

**Loader** -- the SM100 `PAGED_KV` specialization ported 1:1 into `sm107/prefill_d128_mxfp8.py` and
`sm107/prefill_d256_mxfp8.py` (never routed onto the SM100 bodies): the page id is the TMA batch
coordinate of the K/V AND SF descriptors, the tile inside the page the SF tile coordinate, page -1
past a sequence's live pages is TMA-OOB zero fill; the cga2 d128 K half-tile (64 rows) sits inside the
128-row tile of the page. No SMEM is added, so the version-0 descriptor window and the k_dim=1 /
TILE_K_HW=64 pin are untouched; the dense renderings are cubin-identical before / after the port
(16 sm_107a builds, md5 on w2u1g-lc-0614). `config_sm107._validate_params` backstops every other
flavor (`_PAGED_KV_FLAVORS_SM107`), and the d192x128 / d512 MXFP8 files refuse `paged_kv` at module
scope. The row: `paged_kv=True, paged_d_shapes={(128, 128), (256, 256)}`; the adapter's Rubin clause
is the twin. The cuDNN backend has no engine for MXFP8 pools, so the FROST plan is the only plan.

**Typed declines (unchanged dtype and paging)** -- `page_size % 128 != 0` ("multiple of 128");
THD queries over pools (the generic THD decline on this row; stage 2 below); the d192×d128 / d512
pools ("d128, d256 kernel flavors only"; stage 3); a block-scaled O (`sf_o`) over pools; the
pre-folded scale over paged KV (every row); split-KV (+ sink); PackGQA; per-tensor FP8 pools; pools
without a padding mask.

**Stage 2 (separate PR): THD packed queries over MXFP8 pools** -- the quantized THD setup clamps the
K/V descriptors to a packed total (`thd_helpers.build_thd_meta_o_kv_descs_kernel`), so the port is:
the 14-arg `build_thd_meta_o_descs_kernel(..., clamp_kv=not PAGED_KV)` in both sm107 MXFP8 hosts
(the d128 host still calls the 7-arg setup), the packed Q SF prefix (`cu_sf_q_base`) with the K/V SF
on page coordinates, `compile_host(thd_slots=True)` with paged, the THD binder's paged exclusion
lifted for quantized `sf_sizes == 3` unsplit scalar-output plans (`sdpa_thd_binding.cpp`, reusing
`bind_paged`), a per-role mode in `SdpaMxScaleBinding::bind` (Q packed = bytes / row, K/V pooled =
`n_pages * row * page_size/128`), the admission lifts (engines.py "paged MXFP8 KV with THD queries is
not wired" -> `not sm 107`, row `thd=True, thd_d_shapes={(128,128),(256,256)}`, the api_dsl twins,
the prepared gate, `compile_prepared`, the config rule's `thd_varlen` raise), and the tests
(`exec_sdpa_mxfp8_thd` paged arm, a `_cc107_sweep` with per-request Q 1 / 4 / 8, the mirror's
`mxfp8 thd paged -> {FLOAT, HALF}`). Go / no-go: the THD sweep strict-green on the board and the
binder frame-parity tests green. **Stage 3**: the same loader on `prefill_d192_d128_mxfp8.py` /
`prefill_d512_mxfp8.py`, `paged_d_shapes` and `_PAGED_KV_FLAVORS_SM107` extended, the guards flipped.

**Masked leading tile (review finding on this PR, fixed here)** -- at `|attn_scale * log2 e| >= 1` the
scaled max of a fully masked KV tile overflowed the finite mask sentinel to -inf, the first tile's select
seeded the running max with it and every later shift read `-inf - (-inf)` = NaN for each row whose legal
keys sit behind that tile (a left window at attn_scale 1: Q 65 x KV 193, bottom-right causal, left bound
34 -> 65,536 nonfinite O elements at d256, with a sink also 256 nonfinite Stats; the scale-1/16 twin was
finite); the same arithmetic was reported on the THD route in #1488, which fixed the cc 10.7 d256 body by
selecting a dead masked tile out of the running-max state (`running_max_step_finite_sentinel`).  The cc 10.7
d128 / d192x128 / d512 MXFP8 bodies and the SM100 d128 / d192x128 bodies (the same scaled-max line, the same
finite sentinel) now clamp the scaled tile max to the sentinel -- the state the pre-folded arm always
produced -- so a masked tile contributes exp2(-inf) = 0 and the next live tile re-seeds the max; the SM100
d256 / d512 bodies mask with -inf and shift through `row_max_for_exp2`, and the dense twin's d256 cell is
finite on them.  Regression cells with constant dequantized inputs (O exactly 0.5 on every row with a legal
key): over pools (page 128 / 256 x d128 / d256 x sink on / off, Stats on)
`test_sdpa_fwd_paged_mxfp8_sm107.py::test_masked_leading_tile_with_live_keys_behind_it` and the dense twin
in `test_sdpa_fwd_mxfp8_sm100.py` (d128 / d256; the d128 cell was the SM100 lanes' NaN before the SM100 clamp).
The per-tensor FP8 and half bodies multiply their raw max by the same scale and are a follow-up.

**Measured note (R15)** -- qwen35_decode (b32, 32/2 heads, KV 4096, page 128, s_q 1, e4m3, bf16 O),
the row's default plan over pools (the prefill tile) vs the dense MXFP8 graph of the same logical shape,
d128 and d256, on w2u1g-lc-0614 (cc 10.7, 216 SMs; `time_paged_mxfp8_decode.py` under the board's
timing lock, CUDA-graph replay, A/B/A x5 rounds of 50 replays, 150 ms gaps, clocks logged: 2364 MHz
through this run, no throttling; 2026-10-08): d128 paged 219.4 us vs dense 218.8 us (1.003x), both on
`sdpa_fwd_prefill_sm107_mxfp8[PACK_GQA=0, SCHED_POLICY=0, SPLIT_KV=1, TILE_CGA_M=2, TILE_M=128,
TILE_N=128]`; d256 paged 109.4 us vs dense 107.5 us (1.017x), both on the TILE_CGA_M=1 twin; the paged
O is bitwise the dense O on both (max |paged - dense| = 0). Page indirection costs nothing measurable
at this shape; the absolute time is the prefill tile's at s_q 1 (a 128-row tile with one live row per
request), so paged MXFP8 decode stays a P3-class follow-up (the quantized twin of the d128 decode tile).

**Validated on w2u1g-lc-0614** -- dead-page NaN data + 0xFF SF poison (TMA-OOB page -1), distinct K /
V tables (V behind its own page permutation), zero-length / single-key / whole-page / tile-boundary
per-batch lengths, keyless rows with sinks -120 / +3 (O := 0, LSE := sink), e4m3 (with and without
sink) / e5m2, f16 / bf16 / e4m3 O (e4m3 O declared as such through the harness and the module), Stats
on / off, HND / NHD, page 128 / 256 / 384 / 512, FLOAT and HALF arms, every scheduler policy the row
admits over pools by explicit plan pin (d128 NATURAL / LPT / LPT_L2, d256 NATURAL), Amax_O on every
served cell, CUDA-graph replay after mutating the SF bytes through the native binder, Rule 3 sync-debug,
sched NATURAL / LPT / LPT_L2 bit-identical over pools (d128):
`test_mhas_v2.py::test_sdpa_mxfp8_fwd_paged_cc107_L0` / `_paged_prefill_cc107_L0` /
`_paged_cc107_pinned_L0` / `test_sdpa_mxfp8_paged_cc107_plan_pins_L0` /
`test_sdpa_mxfp8_paged_cc107_declines_L0` / `_split_with_sink_declines_L0`,
`test_sdpa_native_mxfp8_binding.py::test_mxfp8_native_paged_scales_and_replay`,
`test_sdpa_fwd_paged_mxfp8_sm107.py`, `test_sdpa_fwd_dsl_sm107.py::test_rubin_mxfp8_row_serves_paged_pools_on_d128_d256`.
CI (PR #1481): the cc 10.7 FROST lane (cuDNN 9.28, CuTe DSL 4.8.0a0) runs the `test_mhas_v2.py` cells
green (the sm107 module and the native MXFP8 binder test are not on that lane's explicit file list --
a maintainers' follow-up); the SM100 / SM103 FROST lanes were red on the PR's pipeline with a
pre-existing, develop-wide failure under the public DSL 4.7.0 -- `TYPE_UNSTABLE_JOIN` on `k_row` /
`kv_row_base` ("has type None on one path and Int32 on another") in the untouched
`sm100/prefill_d256_fp8.py`, `sm100/prefill_d256_mxfp8.py` and `sm100/prefill_d512_mxfp8.py`, whose
K/V row bindings are first bound inside the dynamic empty-KV `else:`; the same cells fail on develop's
own pipeline. The sm107 ports copy that construct (it compiles under 4.8.0a0), so the binding hoist is
ONE shared follow-up for the sm100 kernels and their sm107 siblings (sibling lockstep, S6).


### SM107 D128 half paged packed split with attention sinks

Explicit CGA1 split-KV plans now compose `sink_token` with FP16/BF16 exact
D128/V128 paged THD and declared Q>1. Existing GQA packing, independent K/V
page tables, HND/NHD pools, bounded shape overrides and optional packed NH/HN
Stats (ln or log2) retain their contracts. The matching native extension is
required. Default split selection is unchanged.

Partial attention excludes the sink. Packed combine treats the per-query-head
sink logit as one zero-value virtual key in the stable normalizer, counted once
across all splits. A keyless row writes O=0 and LSE=sink; a sink of negative
infinity retains the ordinary keyless result. Runtime sink pointers remain
per-call bindings; no new setup, host readback or execute allocation is added.

Dense, nonpaged, Q1-only, other head dimensions/architectures, quantized inputs,
padded Stats and fused gates remain outside this extension. Tests cover GQA/MHA,
changed device lengths and sinks under graph replay, empty/keyless rows, runtime
pointer validation, output tail canaries, and combine split counts on both sides
of the warp reduction boundary. No timing or preferred split count is asserted.


### Rubin d256 gated attention block -- serving forms (frontend-only, no graph form)

The gated attention block (`cudnn.gated_attention_block`; `docs/fe-oss-apis/gated_attention_block.md`) has no engine row and no
`sdpa` graph form: its serving forms are claims of the BLOCK, recorded here one row per form x pool layout x page size x dtype,
each written in the same change as its accept cell (Rule S2). Rubin (cc 10.7) only. The pool contract is the SDPA adapter's own
paged contract -- a 4-D view `[num_pages, H_kv, page_size, D]`, `D` contiguous, the other three strides free: **L1** HND compact
`(H*P*D, P*D, D, 1)`; **L2** NHD by strides `(P*H*D, D, H*D, 1)`; **L3** a serving stack's unified `[num_pages, H_kv, P, 2D]`
content row (K at `+0`, V at `+D`, token stride `2D`) bound as its two halves -- nothing is repacked; every non-innermost stride a
whole number of 16-byte vectors (a typed `ValueError` otherwise), `page_size` a positive multiple of 16, `slot_mapping [T]` int32 or
int64 (a negative slot writes nothing; a slot past the pool writes nothing either).

**Served since 2026-10-12 -- the paged KV-cache WRITE-THROUGH** (`paged_kv_page_size > 0`, the block's `(4w)` stage: the post-RoPE
K / V of every token into the caller's pools at `execute(k_cache=, v_cache=, slot_mapping=)`, after norm + RoPE and before the SDPA,
which still attends over the block's own K / V; `out` bitwise the block's without it; `get_workspace_size()` unchanged; `quant`
and `save_for_backward` with it are typed declines; a CuTe DSL below the floor is declined at `check_support`, named with the
installed version). Validated on a Rubin development node (cc 10.7, 204 SMs; functional cells, no perf claim) at the 8 / 2 d256
geometry, T = 512 (dense B = 2 x S = 256; packed sequences 300 / 0 / 128 / 84 -- a zero-length sequence in the packing), the
default in-place pipeline, every 9th token padded (`-1`) on NaN-poisoned pools, int32 slots at pages 16 / 64 and int64 at 32 / 128:
the live pool rows BITWISE the slab's post-RoPE K / V bands attention consumed and within the dense suite's bars of the
per-sequence oracle (V at the band bar rtol 2^-7 / atol 1e-3; K at rtol 2^-7 / atol 8e-3), padded tokens unwritten, every
unwritten slot still NaN (`test/python/gated_attention_block/cutedsl/test_block_serving_extend.py`):

| form | pool layout | page 16 | page 32 | page 64 | page 128 |
|---|---|---|---|---|---|
| dense `[B, S, d_model]` write-through | L1 HND compact | bf16 ✅ f16 ✅ | bf16 ✅ f16 ✅ | bf16 ✅ f16 ✅ | bf16 ✅ f16 ✅ |
| dense | L2 NHD by strides | bf16 ✅ f16 ✅ | bf16 ✅ f16 ✅ | bf16 ✅ f16 ✅ | bf16 ✅ f16 ✅ |
| dense | L3 unified content row (K `+0`, V `+D`) | bf16 ✅ f16 ✅ | bf16 ✅ f16 ✅ | bf16 ✅ f16 ✅ | bf16 ✅ f16 ✅ |
| packed `thd=True` write-through (`B = 1, S = T` inside) | L1 HND compact | bf16 ✅ f16 ✅ | bf16 ✅ f16 ✅ | bf16 ✅ f16 ✅ | bf16 ✅ f16 ✅ |
| packed | L2 NHD by strides | bf16 ✅ f16 ✅ | bf16 ✅ f16 ✅ | bf16 ✅ f16 ✅ | bf16 ✅ f16 ✅ |
| packed | L3 unified content row | bf16 ✅ f16 ✅ | bf16 ✅ f16 ✅ | bf16 ✅ f16 ✅ | bf16 ✅ f16 ✅ |

Also validated (bf16, L3 at page 16): the other inference pipelines -- `fuse_norm_rope` (the fork-written slab; the pool's K
within one bf16 ulp of the fp32-accumulator oracle rounded once), out-of-place Q / K / V (`inplace_qkv=False`: the compact buffers
are the operands and the pool rows), `fuse_gate` (dense only) -- dense and, for the first two, packed; a second execute with
another slot mapping on the same compiled block re-places every row; two launches bitwise on the pools and on `out`. The indexer
band's raw-key pool (`index_k_raw`, `QsaSpec(index_band=True)`) is written by the same stage, bitwise `index_k_raw_view`, through
the block's isolated stage objects; that validation did not execute the sparse attention core. The standalone sparse
adapter's coverage is recorded separately in the index-list section below.

**Served since 2026-10-13 -- the EXTEND form** (`kv_source="paged"` on a packed, bounded declaration: `thd=True, num_sequences=B_max,
max_seq_len=token_capacity, token_capacity=T_max, paged_kv_page_size=P, kv_source="paged", geometry.causal_bottom_right=True` + the pools' and
the block table's samples; `execute(seq_lens=, k_cache=, v_cache=, slot_mapping=, block_table=, kv_lens=)`): the step's new tokens are written
through, then stage (4) reads EVERY sequence's K / V -- its prefix and the new tokens -- from the pools over `[0, kv_lens[b])` through
`block_table[b]` on the bottom-right diagonal, on the sm107 f16 row's paged THD leg (the shared SM100 d256 paged / THD body compiled for
sm_107a, CGA2, unsplit, unpacked, NATURAL, no gate, no LSE); five launches on one stream (the write ordered before the read by stream order:
no kernel on the chain launches with programmatic dependent launch -- pinned). Validated on a Rubin development node (cc 10.7, 204 SMs;
functional cells, no perf claim) at the 8 / 2 d256 geometry: one declaration per (dtype, layout, page) over four sequences at `T_max` 512 --
kv_len 128 (one KV tile; `B x H_kv > 1` with it), an 84-token step over a 300-row prefix ending mid-page (kv_len 384), a PADDED entry
(kv_len 0, the NULL block 0, no rows), a 200-token step over a 77-row prefix (kv_len 277: a tail tile, three KV tiles) -- `T_live` 412 of 512
(pad rows at the END, slot -1), the prefix rows written by the torch reference, `isfinite(out)` on every row first, every live sequence
within the dense suite's budget of the per-sequence oracle over its FULL history (`out` cos >= 0.999, the gated O within atol 2e-2), the new
tokens' pool rows bitwise the slab rows attention consumed, every unaddressed pool row still its finite stale value, the NULL page untouched
(`test/python/gated_attention_block/cutedsl/test_block_serving_extend.py`, the "EXTEND form" half):

| form | pool layout | page 16 | page 32 | page 64 | page 128 |
|---|---|---|---|---|---|
| EXTEND `kv_source="paged"` (the paged THD read; bottom-right; bounded tokens) | L1 HND compact | bf16 ✅ f16 ✅ | bf16 ✅ f16 ✅ | bf16 ✅ f16 ✅ | bf16 ✅ f16 ✅ |
| EXTEND | L2 NHD by strides | bf16 ✅ f16 ✅ | bf16 ✅ f16 ✅ | bf16 ✅ f16 ✅ | bf16 ✅ f16 ✅ |
| EXTEND | L3 unified content row (K `+0`, V `+D`) | bf16 ✅ f16 ✅ | bf16 ✅ f16 ✅ | bf16 ✅ f16 ✅ | bf16 ✅ f16 ✅ |

Also validated (bf16): a one-sequence EXTEND with an EMPTY prefix at a tight declaration (`T_live = T_max = 384`) against the dense `B = 1`
block on the same tokens -- the two bodies differ (the shared SM100 paged THD body vs the Rubin sibling): BITWISE on `out` and on the gated O (0 of 196 608 / 786 432 elements differ; both legs within the oracle budget, out cos 0.999995), pinned; a 512-token sequence
as TWO chunks (300 + 212, `T_max` 320 -- a bounded declaration) on one compiled block against the single-shot dense block and the oracle,
the pool after both chunks within the write-through's bars of the oracle's K / V over all 512 positions; the prefix-hit step on L3 with the
prefix rows written by the torch reference through the stack's own `[..., :D]` / `[..., D:]` views and read back the same way (the layout
proof, both directions); L1 / L2 / L3 declarations over pools holding the same values bitwise on `out`; a zero-length sequence in the
MIDDLE of the packing and a live-Q entry over kv_len 0 with the NULL block and slot -1 (its `out` rows exactly 0, nothing written to the
NULL block); kv_len = 128 x n_kv for n_kv in {1, 2, 3, 4, 8} and the tail twin 128 x n_kv - 37 (one tile, the ring depth, the wrap) on L2 at
page 64; a `[:, :max_pages]` row slice of a wider block table (row stride 50) bitwise the compact table; a finite stale tail inside the last
live page (777 vs 0) bitwise on `out`; the bounded token contract -- live rows BITWISE the tight declaration, pad rows of a sentinel-filled
`out` EXACTLY 0 over a once-zeroed workspace (the `o` slot aliased by no other stage's scratch, pinned on the carve), a shorter second step
leaving the previously live rows finite and the pad rows 0; two launches bitwise; with the projection GEMM driver's capture-safe route in
the tree (`build_proj_gemm(capture_safe=True)`; the cell skips without it), the FIRST execute of a fresh declaration captured into a
CUDA graph on a new stream with no warm-up under `set_sync_debug_mode("error")`, replayed bitwise an eager run and then replayed over the
next chunk's lengths / tables / slots written through the captured tensors; 16 fresh processes (one launch each, under `timeout`: n_kv in {1, 2, 4, 8} x 3 seeds at kv_len = 128 x n_kv, plus the tail twins 128 x n_kv - 37 at n_kv in {1, 3, 4, 8}; B_max 2, L2, page 64) exit 0 / 124 / other = 16 / 0 / 0. `serving_envelope(geometry, dtype, device)`
publishes the form's declines (every row reachable at declaration with the published message, pinned on the CPU) and its served verdict.
Also executed on the same node, each through the suite's own harness and check (`test_extend_admitted_spellings_execute_within_the_budget`:
5 / 5 within the dense suite's budget, worst `out` cos 0.999988 -- the fused pipeline -- and gated-O max|err| 3.906e-03 bf16 / 4.883e-04 f16;
nothing widened): the five spellings the declaration admits -- the `[B]` lengths form (`cu_seqlens=False`) on the matrix packing (L1, page
16), `fuse_norm_rope=True` (the fork-written slab the write-through reads; L2, page 32), the out-of-place Q / K / V pipeline
(`inplace_qkv=False`, f16, L3, page 64), a FULL table row (`kv_len = max_pages x P` = 256) on a `[:, :max_pages]` slice of a wider table
under the lengths form, and a block table WIDER than `max_pages` (`[2, 40]` at the declared row stride 50, `max_pages` 16).

**Served since 2026-10-13 -- the DECODE form** (`kv_source="paged"` on a dense `[B, S_q]` declaration over `paged_kv_page_size > 0`:
the step's tokens written through, then stage (4) over the first `kv_lens[b]` keys of every sequence's `block_table` row --
`seq_kv_lens = kv_lens`, `seq_q_lens = None`, the batch uniform, bottom-right causal -- on the Rubin d256 DECODE tile
(`sm107/decode_d256_f16.py`: PackGQA over the whole group, the decode split model's split-KV with fp32 partials in the block's
workspace and one combine launch, `fuse_gate` applied IN THE COMBINE), the same tile unpacked past the packed route (one head per
CTA, `S_q <= 32`); a wider dense step is a typed decline at declaration (the adapter serves dense half queries over paged K / V
on the decode tile only -- the paged prefill body is the EXTEND form's). `GatedAttentionBlockFwd.serving_envelope()` is the static record
(forms, page sizes, pool layouts, the route per `(group, S_q)`). Validated on a Rubin development node (cc 10.7, 204 SMs;
functional cells, no perf claim; `test/python/gated_attention_block/cutedsl/test_block_serving_decode.py`) at a shrunk
`d_model = 512` with `B = 4` batches holding one KV tile with a tail (`kv_lens` 100), the ring depth exactly (384), four tiles
with a tail (500) and a PADDED entry (`kv_lens = 0`, block table row 0, slot -1) -- `out` within the dense suite's budget of the
per-sequence fp32 oracle (the block on the full sequence, its last `S_q` rows; the prefix K / V of the pools ARE the oracle's bf16
rows; cos > 0.999 on every live row, the padded entry's rows EXACTLY 0, block 0's bytes untouched), the compiled kernel pinned
against the declared route (template, packing, split, `epilogue_gate` False):

| group (`H_q / H_kv`) | `S_q = 1` | `S_q = 2` | `S_q = 3` | `S_q = 4` | `S_q = 5` | `S_q > 32` |
|---|---|---|---|---|---|---|
| 16 (32 / 2) | tile 16, packed, 1 unit ✅ | tile 32, packed, 1 unit ✅ | tile 32, packed, 2 units ✅ | tile 32, packed, 2 units ✅ | tile 16, UNPACKED (declared, unmeasured) ✅ | typed decline (the EXTEND form) |
| 12 (24 / 2) | tile 16, packed, 1 unit ✅ | tile 32, packed, 1 unit ✅ | tile 32, packed, 2 units ✅ | tile 32, packed, 2 units ✅ | tile 16, UNPACKED (declared, unmeasured) ✅ | typed decline |
| 8 (8 / 1) | tile 16, packed, 1 unit ✅ | tile 16, packed, 1 unit ✅ | tile 32, packed, 1 unit ✅ | tile 32, packed, 1 unit ✅ | tile 32, packed, 2 units (declared, unmeasured) ✅ | typed decline |
| 6 (24 / 4) | tile 16, packed, 1 unit ✅ | tile 16, packed, 1 unit ✅ | tile 32, packed, 1 unit ✅ | tile 32, packed, 1 unit ✅ | tile 32, packed, 1 unit (declared, unmeasured) ✅ | typed decline |
| 4 (4 / 1, the 397B TP 8 shard; 2026-10-13) | tile 16, packed, 1 unit ✅ (also at `d_model` 4096, page 64) | tile 16, packed, 1 unit ✅ | tile 16, packed, 1 unit ✅ | tile 16, packed, 1 unit ✅ | tile 32, packed, 1 unit (declared, unmeasured) ✅ | typed decline |
| 16 (16 / 1, the 397B TP 2 shard, ONE KV head; 2026-10-13) | tile 16, packed, 1 unit ✅ (at `d_model` 4096, page 16) | the 32 / 2 row's route (declared; not re-run at 16 / 1) | same | same | same | typed decline |

The CLAIMED envelope (`DecodeRoute.claimed`) is the packed tile at `S_q <= 4`; the packed route reaches `S_q = 8` at 8 / 1,
`S_q = 10` at 24 / 4 and `S_q = 16` at 4 / 1 and the unpacked tile `S_q = 32` at every group -- routed, declared by the envelope,
correct on the cells run, unmeasured (the serving perf cells). The pool matrix on the decode tile (the three stride patterns holding the
same values BITWISE on `out`, each within the oracle's budget; `kv_lens` 200 / 0 / 77 / 1000 at 24 / 2, `S_q = 2`):

| form | pool layout | page 16 | page 32 | page 64 | page 128 |
|---|---|---|---|---|---|
| DECODE `[B, S_q]` over the pools | L1 HND compact | bf16 ✅ f16 ✅ | bf16 ✅ f16 ✅ | bf16 ✅ f16 ✅ | bf16 ✅ f16 ✅ |
| DECODE | L2 NHD by strides | bf16 ✅ f16 ✅ | bf16 ✅ f16 ✅ | bf16 ✅ f16 ✅ | bf16 ✅ f16 ✅ |
| DECODE | L3 unified content row (K `+0`, V `+D`) | bf16 ✅ f16 ✅ | bf16 ✅ f16 ✅ | bf16 ✅ f16 ✅ | bf16 ✅ f16 ✅ |

Also validated: the block's form BITWISE the standalone `SdpaFwdDslSm100` on the same slab Q views / pools / table / gate at the
same split and packing; paged == dense BITWISE (O and LSE) at the unsplit plan on the block's own operands (the padded dense
adapter has no split); split == unsplit within the combine's derived two-term budget on the pre-gate O (max |diff| / budget
0.26 at 32 / 2 and 0.16 at 8 / 1; the fp32 LSEs to 1e-5); the gate in the combine vs the separate gate launch at the same split
within the stated one-rounding budget (max |diff| on O 9.766e-04 bf16 / 1.221e-04 f16 at split 2, 0.73 / 0.83 of the budget; `out` cos > 0.9999 between them);
a block table that is a row SLICE of a wider table (row stride 37) bitwise the compact table (a block-table row slice); `kv_lens` 128 / 384 / 512 /
1024 (one tile, the ring depth, four, eight) and an entry whose whole content is the step (`kv_lens == S_q`); the FIRST execute
of a fresh declaration captured on a new stream with no warm-up under `set_sync_debug_mode("error")` (the capture launches
nothing, the replay bitwise the eager run; the replay over new `kv_lens` / table / slots / inputs bitwise the eager run over them);
`get_workspace_size()` honest (the split partials inside; a sentinel past it untouched); two launches bitwise; 12 fresh processes
at `kv_lens = 128 x n_kv`, `n_kv` in {1, 2, 4, 8} (ok 12 / hang 0 / other 0). The tail tile
READS the pool rows past `kv_lens[b]` (the last page's stale rows, block 0 for unlisted pages) and masks them to `P = 0`: they
must be FINITE, as a zero-initialized cache's are (a NaN there is `0 x NaN` -- the suite's pools carry a finite sentinel).

**Pinned 2026-10-13 -- the Qwen3.5-397B tensor-parallel shards end to end** (16 / 1 = TP 2, 8 / 1 = TP 4 -- the serving
deployment's shard --, 4 / 1 = TP 8; `d_model` 4096, `d_head` 256, the ONE replicated KV head from TP 4 on; the layout contract
`test/python/gated_attention_block/cutedsl/test_layout_contract.py` pins `n_qkvg` 8704 / 4608 / 2560 -- every band a whole number of
256 columns, so the forced 256-wide GEMM tile holds and `qkvg_tile_plan(64 / 128 / 256)` is well defined -- and the TMA norm kernel's
fitted tile 16 / 8 / 4 rows, the 4-row tile one row per warp). Validated on a Rubin development node (cc 10.7, 204 SMs; functional
cells, no perf claim), bf16, L1 HND pools, one cell per shard and form within the dense suite's budget (`out` cos >= 0.999, the gated
O within atol 2e-2 on the EXTEND cells; nothing widened), each shard on the kernel block size its deployment allocates:

| shard | DENSE `[B, S]` (`test_block_end_to_end.py`: B = 1, S = 512, HF-form weights; `out` cos vs the oracle on the block's weights / vs the HF-faithful fp32 `(1 + w)` oracle) | EXTEND (`test_block_serving_extend.py`: two sequences at `T_max` 256, a 100-token step over a 60-row prefix and a 156-token step over a 100-row prefix; worst `out` cos / gated-O max abs err) | DECODE (`test_block_serving_decode.py`: `S_q = 1` over `kv_lens` 100 / 384 / 500 / 0, the whole group packed per tile, the gate in the combine; `out` cos / max abs diff) |
|---|---|---|---|
| 16 / 1 (TP 2) | ✅ 0.999995 / 0.999987 (TMA 16-row norm tile) | ✅ page 16: 0.999994 / 7.812e-03 | ✅ page 16: 0.999988 / 2.823e-03 (tile 16, HEADS_PER_TILE 16) |
| 8 / 1 (TP 4) | ✅ 0.999996 / 0.999988 (TMA 8-row norm tile) | ✅ page 32: 0.999994 / 3.906e-03 | ✅ page 32: 0.999988 / 1.953e-03 (tile 16, HEADS_PER_TILE 8) |
| 4 / 1 (TP 8) | ✅ 0.999995 / 0.999988 (TMA 4-row norm tile) | ✅ page 64: 0.999995 / 3.906e-03 | ✅ page 64: 0.999987 / 1.221e-03 (tile 16, HEADS_PER_TILE 4) |

Group 4 also joined the shrunk-width DECODE matrix above (4 / 1 at `d_model` 512: `S_q` 1 .. 5 vs the oracle, the declaration facts,
the route table -- its rows of the route table); 16 / 1 rides group 16's route with one KV head (its `S_q = 1` cell above; the wider
`S_q` are the 32 / 2 cells' route, declared and not re-run at 16 / 1); a shard's route equals its group's shrunk-width route at every
`S_q`, pinned on the CPU. The shards' `S_q > 1` decode steps and every page / layout cell beyond the ones named are functional
claims only -- no perf claim before the serving perf cells land.

Not served (typed declines by name; each lands with its accept cell and flips this section in the same change): an LSE / a sliding window or a right band / a `QsaSpec` / another head dim on the paged forms, a dense step wider than 32 rows (the DECODE form's `decode_rows` row: the EXTEND form's packed business), `save_for_backward` / `quant` on the paged forms, e4m3 pools (`kv_cache_dtype`) on the DENSE paged forms (served on the block-sparse decode mode at `S_q <= 4` and as the quantizing write-through on the dense prefill forms -- the e4m3 paragraph below), the write-through folded into the projection epilogue, `quant` / `save_for_backward` together with the write-through attribute; the DECODE form's `S_q > 4` arms and every page / layout cell above are functional claims only -- no perf claim before the serving perf cells land.

**Served since 2026-10-14 -- the BLOCK-SPARSE DECODE / extend-with-prefix mode** (`kv_source="paged"` on a dense `[B, S_q]`
declaration of a `QsaSpec` geometry with `geometry.causal_bottom_right=True` + the pools' and the block table's samples, the table
CONTIGUOUS; `execute(block_ids=[, block_lens=], k_cache=, v_cache=, slot_mapping=, block_table=, kv_lens=)`): the step's tokens are
written through, then stage (4) is the index-list sparse core (`sparse_gqa_sm107`, the index-list section below) READING THE POOLS --
per sequence the listed 4-token blocks and the open tail over `[0, kv_lens[b])` through `block_table[b]`, on the bottom-right diagonal
(row `j` at `kv_lens[b] - S_q + j`; a negative position is a dead row, `kv_lens = 0` a dead entry: `out = 0` / `LSE = -inf` exactly).
The list form is a declaration fact derived from `S_q`: at `S_q <= 4` ONE list per sequence (`block_ids [B, top_k]` + `block_lens
[B]`, the adapter's `list_per_sequence` arm: selected at the step-0 position and shared by the step's rows, the MTP visible set of the
contract page), past four rows one list per row (`[B, S_q, top_k]`, the adapter's bottom-right per-token form) at any `S_q`; the split
(`decode_split_kv`: `None` = `ceil(SMs / (B x H_kv x S_q))` chunks capped at `min(8, ceil((top_k + 1) / 32))` -- the cap from a split sweep at B 4 x S_kv 32K x S_q 1 on a 212-SM part: the step 69.2 / 64.3 / 59.9 / 59.8 / 65.0 us at split 1 / 2 / 4 / 8 / 17 --, or a pinned int up to `ceil((top_k + 1) / 32)`)
puts fp32 partials in the block's workspace (`get_workspace_size()` holds them) and runs the split combine; `fuse_gate` is served at every
split (the core's epilogue unsplit, the combine otherwise), `return_lse` is served (`[B, H_q, S_q]` fp32).  Typed declines: the packed form
(the table's `sparse` row: THD and paged pools one at a time), the in-block indexer over the pools, a non-contiguous table sample, a split
past the item's tile count, `fuse_prepare`; `causal_bottom_right` on the block's own K / V names this mode.  Validated on a Rubin
development node (cc 10.7, 204 SMs; functional cells, no perf claim) against the fp32 oracle of the whole sequence (the dense block
reference's post-norm / post-RoPE Q, K, V and gate; the prefix rows fill NaN pools, the step rows are the block's own write-through; the
step rows' attention over `qsa_visible_mask(..., pos0=)` or per row; `cos(out) > 0.999`, the gated O and the LSE within atol 2e-2, dead
rows exact, the pools' step rows BITWISE the slab's K / V bands, the prefix rows and the NULL block untouched, two launches bitwise;
`test/python/gated_attention_block/cutedsl/test_block_qsa_decode.py`):

| cell | `S_q` / list form | `B`, `kv_lens` | pool | split | gate / LSE | dtype |
|---|---|---|---|---|---|---|
| ladder 1 | 1 / shared | 4: 2052, 8192, 0 (dead), 4097 | L1 HND page 16 | the fill rule | launch / LSE | bf16 ✅ |
| ladder 2 | 2 / shared | 4: 300, 0, 1000, 77 | L2 NHD page 64 | 1 | fused (epilogue) / -- ; `fuse_norm_rope` | bf16 ✅ |
| ladder 3 | 3 / shared, `block_lens` absent | 2: 8192, 2052 | L3 unified page 32 | 4 | fused (combine) / LSE | f16 ✅ |
| ladder 4 | 4 / shared, every residue of the step-0 position | 4: 8191, 8190, 8189, 8188 | L1 page 16 | 17 (one tile per chunk) | fused (combine) / LSE | bf16 ✅ |
| ladder 5 | 4 / shared, 24 / 2 at `d_model` 2560 | 1: 4096 | L1 page 16 | the fill rule | fused ; `fuse_norm_rope` | bf16 ✅ |
| ladder 6 | 4 / shared, rows at negative positions | 4: 2, 4, 0, 20000 | L1 page 16 | 2 | launch / LSE | bf16 ✅ |
| ladder 7 | 1 / shared, the indexer band's `index_k_raw` write-through | 2: 1024, 60 | L1 page 16 | 1 | launch / LSE | bf16 ✅ |
| ladder 8 | 8 / per row | 2: 300, 100 | L1 page 16 | 1 | launch / LSE | bf16 ✅ |
| ladder 9 | 64 / per row (the 2500 list omits blocks) | 2: 2500, 64 | L2 NHD page 64 | 2 | fused (combine) / -- | bf16 ✅ |
| MTP residues | 2 and 4 / shared, `pos_0 mod 4` in 0..3 on ~60-token sequences | 2 | L1 page 16 | 1 | fused / LSE | bf16 ✅ (the step-0-anchored oracle within budget; the per-row-tail reading farther on the `4k + 3` rows) |
| adapter twin | 2 / shared | 4: 700, 0, 2052, 130 | L1 page 16 | 1 and 4 | fused / LSE | bf16 ✅ BITWISE the standalone adapter on the block's own Q / gate views |
| workspace sentinel | 1 / shared | 3: 4096, 512, 0 | L1 page 16 | 8 | fused | bf16 ✅ (a sentinel past `get_workspace_size()` untouched; unsplit scratch 0) |
| capture / replay | 2 / shared | 4: 300, 0, 1000, 77 -> 301, 0, 1000, 77 | L1 page 16 | 4 | fused / LSE | bf16 ✅ (the first execute captured with no warm-up under `set_sync_debug_mode("error")`, the replay bitwise eager, the replay over new lengths / table / lists / slots / inputs bitwise eager) |

**Served since 2026-10-14 -- the e4m3 KV cache** (`kv_cache_dtype=torch.float8_e4m3fn`; bf16 activations; `paged_kv_page_size > 0`; the
serving stacks' `--kv-cache-dtype fp8` recipe: e4m3 codes under per-tensor fp32 DEscales `execute(k_scale=, v_scale=)`, value = code x
scale, a bf16 Q over the dequantized rows -- no fp8-Q form, no per-tensor FP8 SDPA row bound).  Two forms of the one attribute: the
quantizing WRITE-THROUGH on the dense prefill forms over the block's own K / V (two per-tensor casts `sat_e4m3(row x (1 / scale))` +
the bit copy through the pools' bf16 views; no new kernel; the raw indexer key stays bf16) and the BLOCK-SPARSE DECODE mode over e4m3
pools at `S_q <= 4` (stage (4) = `kernels/qsa_kv_upcast.py` -- the listed blocks and the open tail gathered per (token, KV head) item,
dequantized `bf16(fp32(code) x scale)` and compacted into the workspace scratch `[items, 2, 2176, 256]` + a per-item length -- then the
Rubin d256 decode tile over the scratch as a padded dense SDPA: no causal mask, the length on the KV side, the group packed, unsplit;
the sparse core untouched by fp8; the mode norms out of place and gates in its own launch).  Typed declines: e5m2 / another dtype, the
attribute without the write-through, `quant` / `save_for_backward` with it, f16 activations, the packed prefill's write-through (unvalidated),
the EXTEND form and the dense DECODE form over e4m3 pools (the paged THD body and the decode tile read bf16 / f16 pools: the in-kernel
dequant arm is the follow-up), `S_q > 4`, `return_lse`, `decode_split_kv > 1`, `fuse_norm_rope` / `inplace_qkv=True`, pools of the wrong
dtype both ways, the scales missing under the attribute or given without it.  Validated on a Rubin development node (cc 10.7, 204 SMs;
functional cells, no perf claim) against the fp32 oracle ON THE DEQUANTIZED POOL (the prefix rows quantized by torch with the per-tensor
scales, the step rows the block's own quantizing write; `cos(out) > 0.999`, the gated O within atol 2e-2, dead rows exact, the pools'
step rows BITWISE the torch cast, the prefix rows and the NULL block untouched, two launches bitwise; the distance to the pre-quantization
bf16 oracle reported without a gate; `test/python/gated_attention_block/cutedsl/test_block_qsa_fp8_kv.py`):

| cell | `S_q` | `B`, `kv_lens` | pool | gate | dtype |
|---|---|---|---|---|---|
| e4m3 ladder 1 | 1 | 4: 2052, 8192, 0 (dead), 4097 | L1 HND page 16 e4m3 | launch | bf16 act ✅ |
| e4m3 ladder 2 | 4, `block_lens` absent | 4: 300, 0, 1000, 77 | L2 NHD page 64 e4m3 | `fuse_gate` (runs as the launch) | bf16 act ✅ |
| e4m3 ladder 3 | 3, a row at a negative position | 3: 2, 20000, 7 | L3 unified page 32 e4m3 | `fuse_gate` | bf16 act ✅ |
| e4m3 ladder 4 | 1, 24 / 2 at `d_model` 2560 | 1: 4096 | L1 page 16 e4m3 | `fuse_gate` | bf16 act ✅ |
| e4m3 ladder 5 | 3, the indexer band's raw-key write-through (bf16 pool) | 2: 8192, 2052 | L1 page 16 e4m3 | launch | bf16 act ✅ |
| e4m3 MTP residues | 4 / shared, `pos_0 mod 4` in {1, 3} on ~60-token sequences | 2 | L1 page 16 e4m3 | `fuse_gate` | bf16 act ✅ |
| e4m3 prefill write-through | 64 (the sparse block over its own K / V, `kv_source="self"`) | 2 | L1 page 16 e4m3 | launch | bf16 act ✅ `out` BITWISE the bf16-pool twin, the pools BITWISE the torch cast, padded slots unwritten |
| e4m3 workspace sentinel | 2 | 3: 4096, 512, 0 | L1 page 16 e4m3 | `fuse_gate` | bf16 act ✅ (a sentinel past `get_workspace_size()` untouched; every scratch row past an item's length 0.0) |
| e4m3 capture / replay | 2 | 4: 300, 0, 1000, 77 -> 301, 0, 1000, 77 with new scales | L1 page 16 e4m3 | `fuse_gate` | bf16 act ✅ (the first execute captured with no warm-up under `set_sync_debug_mode("error")`, the replay bitwise eager, the replay over new scales / lengths / table / lists / slots / inputs bitwise eager) |

### SM107 d256 index-list sparse forward

Frontend-only (the gated attention block's sparse stage — `GatedAttentionBlockGeometry(qsa=QsaSpec(...))` routes stage (4) to
it and `execute(block_ids=, block_lens=)` carries the list — and the standalone adapter
`python/cudnn/sdpa/fwd/sparse_gqa_sm107.py`), no graph form: no `sdpa` node carries a block-index list, so there is NO engine
row and NO manifest slot — an `EngineSpec` whose `lower` cannot run would break the engine contract.  A future graph port
carries `QsaSpec` as a `python_only_attrs` attribute of the `sdpa` node (AGENTS Rule 9: one FE graph contract for the backend and
FROST, and a SET numerics-changing attribute makes the node backend-unlowerable), so no backend plan can ever serve such a graph
densely.  The kernel's claims live
in the adapter's frozen `SparseCapabilities` record, spelled in the `Capabilities` vocabulary where a field exists, and its
`check_support` is the enforcement point (the CuTe DSL `sm_107a` gate first, then every field as a typed decline); the block
reads the same record for its own declines (dtypes, head dim, block size, `top_k`, the GQA group) and transcribes nothing.
The block route is validated end to end on Rubin against the fp32 QSA oracle at the 24/2, 12/1, 6/1 and 32/2 geometries
(`test/python/gated_attention_block/cutedsl/test_block_qsa_end_to_end.py`).

Served (bring-up body, 2026-10-08): dense BSHD f16 / bf16, d_qk = d_v = 256 exactly, one query token per work item with its
packed query-head group (1..16 heads per KV head), caller block lists `block_ids` int32 `[B, S_q, top_k]` (contiguous, `top_k`
a multiple of 4 in [4, 512]; the valid prefix then -1; an optional `block_lens` count, clamped on device), the open tail block
appended on device from the token's position (top-left causal), per-batch KV lengths (`seq_kv_lens`: a 0 length is a dead
row, `O = 0` / `LSE = -inf`), and the fused epilogue gate (`epilogue_gate=` the gate operand on the adapter, `fuse_gate=True`
on the block: G in O's shape in Q's dtype at its own strides -- the slab's GATE columns bind in place; the kernel stages
each query's gate slice into its freed Q^T slot and writes `O * sigmoid(G)` after the dead-row select, the LSE untouched; a
module specialization, so gated and ungated are two artifacts; the gate composes with the paged read and with THD, where it is
the packed `[1, T_q, H, D]` operand like O).  The K / V gather map is 2-D (tokens of every batch as
rows), so the K / V batch stride must equal `S_kv x` the token stride (or B = 1); the head stride a multiple of 64 elements,
the token stride a multiple of 8.

Paged K / V (the standalone adapter, `paged_kv=True, page_size=, block_table=, seq_kv_lens=`; 2026-10-08): K / V are page
pools `[num_pages, H_kv, page_size, D]` -- HND compact, or NHD storage declared through the strides (the same contract as the
dense paged tiles) -- read through one `(B, max_pages)` int32 block table (the same table for K and V); `page_size` a positive
multiple of 4 (a 4-token block never straddles a page: one table lookup per block, hoisted into the gather warps' id step);
`seq_kv_lens` is REQUIRED as the per-batch logical KV length (the pool has no dense extent).  A `-1` table entry and a page
index past the table read as zero rows; the rows of a partially filled last page beyond the length (another sequence's
tokens, or garbage) are never gathered; the open block's rows past the length are masked exactly as in the dense form.  The
same tokens through a paged pool and through a dense tensor produce BITWISE the same `O` and `LSE` (validated on Rubin at
page 16 / 64 / 48, NHD and HND, bf16 and f16, B = 2 with different lengths and lists).  The gated attention block's paged-READ
mode (`block_table` / `kv_lens` at `execute`) is served since 2026-10-14 as the block-sparse DECODE / extend-with-prefix mode (the
`Rubin d256 gated attention block -- serving forms` section above).

Served since 2026-10-08 (`thd=True`; the standalone adapter, and the gated attention block since the same day): packed THD
sequences — Q / O `[1, T_q, H, 256]`, K / V `[1, T_kv, H_kv, 256]` (capacities; the live totals are device values), `block_ids`
`[T_q, top_k]` with ids RELATIVE TO THEIR SEQUENCE, `seq_q_lens` and `seq_kv_lens` as per-sequence lengths `[B]` or cumulative
`[B + 1]` (`cu_seq_q_lens` / `cu_seq_kv_lens`; the record's `cu_seq_len`), LSE `(1, H, T_q)`, a `workspace` of
`scratch_workspace_bytes()` for the shared THD metadata (`[seq_kv_lens | cu_q | cu_k | remap | live | ctr]`, written by the dense
forwards' setup launch); the kernel runs the persistent claim-counter scheduler (one unit per packed token per KV head, the
gather row carries `cu_k[b]`); a sequence with `S_kv_b = 0` is dead (`O = 0` / `LSE = -inf` stored), an empty sequence
contributes no work, a one-sequence packing is bitwise the dense `B = 1` run.  Through the gated attention block
(`GatedAttentionBlockFwd(thd=True, num_sequences=B, max_seq_len=S_max, cu_seqlens=)` on a `QsaSpec` geometry; `execute(seq_lens=,
block_ids=)`): the block is internally `B = 1, S = T`, its sparse stage binds the same `[1, T, H, D]` views the dense arm binds,
ONE lengths tensor serves both sides (self-attention), `block_ids` `[T, top_k]` carries each token's blocks numbered from its own
sequence's first token, and the adapter's metadata rides the block's engine workspace arm (`get_workspace_size()` reports it);
validated on Rubin per sequence against the per-sequence fp32 QSA oracle (bf16 and f16; packings with empty, 5-token and
non-multiple-of-4 sequences; a one-sequence packing BITWISE the dense `B = 1` sparse block;
`test/python/gated_attention_block/cutedsl/test_block_qsa_end_to_end.py`).  `thd=True` together with `paged_kv=True` is a typed
decline (the packed sequence's K / V row offset `cu_k[b]` composes with a dense `[1, T_kv, H_kv, D]` tensor only, never with a
page pool; the combination lands with its own accept cells when a serving stack asks for it); through the block, the in-block
indexer (`index_source="indexer"`) stays a typed decline naming the feature (a follow-up); `thd=True` together with the paged
write-through (`paged_kv_page_size`) is served since 2026-10-14 (the stage addresses tokens: the packed sparse block builds the same
`(4w)` stage at the packed token count, `slot_mapping` `[T]`, and its pools hold the slab's post-RoPE rows -- `out` / LSE bitwise the
packed sparse block without it).  `fuse_gate=True` composes with `thd=True`
through the block (the GATE is the packed `[1, T, H_q, D]` slab slice like O, gated in the sparse core's epilogue; validated on
Rubin, bf16 and f16: the gated packed block within one rounding of the activation dtype of the unfused packed block on O, the
LSE bitwise).

Served since 2026-10-08, standalone adapter only -- the DECODE FORM (`bottom_right=True`, `list_per_sequence=True`,
`split_kv=S`; three `TemplateParams` arms of the same kernel, each a `const_expr` fold: the dense rendering traces the same
program as before): the serving step's `S_q <= 4` rows per sequence sit at the END of the sequence (`bottom_right=True`: row `j`
of sequence `b` is at position `kv_len_b - S_q + j`; a sequence shorter than the query count makes the rows with a negative
position dead -- `O = 0` / `LSE = -inf`; also served with per-token lists), share ONE block list (`list_per_sequence=True`:
`block_ids` int32 `[B, top_k]`, `block_lens` `[B]` or None) anchored at the step-0 position `kv_len_b - S_q` -- the count comes
from it and the appended tail runs from the step-0 tail start to the row's own position, so the block completed between step 0
and the row stays visible, its own token included (the MTP shared-list semantics; the oracle is `qsa_visible_mask(..., pos0=)`) --
and `split_kv=S` (2 <= `S` <= `ceil((top_k + 1) / 32)` = 17 at `top_k` 512) cuts every item's tiles into `S` chunks, each its own
work item (grid `S_q x S` per (sequence, KV head)) writing fp32 O / natural-log LSE partials into the caller's `workspace`
(`scratch_workspace_bytes()`: the two split-major slabs, 16-byte aligned, plus a final-LSE scratch when no LSE is declared),
reduced by `sm100/split_combine.py` into O / LSE as a second launch on the same stream (an empty chunk is a dead item: an identity
partial, never residue); a split with the gate operand compiles the UNGATED kernel and the GATED combine (`O *= sigmoid(G)` on the
fp32 merged value: ONE rounding, the fused epilogue's convention).  Dense BSHD or paged pools; never THD.  Validated on Rubin
(dev node, functional): B in {1, 4}, S_kv in {2052, 8K, 32K}, bf16 and f16, page 16 HND / 64 NHD BITWISE the dense read at the
same split, splits 1 / 2 / 3 / 4 / 8 / 17 within the budget vs the oracle and split == unsplit within the derived two-term budget
(one output ulp + the half P quantisation), a dead sequence's partials exactly the combine's identity, the paged open block's
rows past the length masked (NaN fill == other-sequence fill == dense), and the MTP MATRIX of the shared list -- `S_q` in {1, 2, 3, 4}
x every residue of the step-0 position `pos_0 = kv_len - S_q` mod 4 x B in {1, 4} (a saturated 512-list sequence whose two tail ids
sit at list indices 512 / 513, the identity bound, a ~60-token sequence, a sequence shorter than the step): every row within the
budget of the step-0-anchored oracle; the per-row-tail reading (vLLM's expand form, the oracle at `pos0=None`) differs on exactly
the rows at or past `4k + 3` when `pos_0 = 4k + r` with `r < 3`, by exactly the block completed after step 0, and coincides
everywhere else -- shown with natural data on the short sequence (the hidden block's four keys move O by 0.2-0.6) and with a
dominant key planted in that block on the saturated sequence (the `4k + 3` row attends it, cos 1.000 with its V; the rows before
it never see it; the per-row-tail oracle sits at cos ~0); the shared list BITWISE the per-token form fed with the replicated step-0
list plus that block appended explicitly; paged (16 HND / 64 NHD) bitwise the dense read; split 2 within the derived budget.
Bytes per step: MTP rows run as `S_q` single-token work items, each re-gathering the sequence's list (2 KiB of ids + `(count +
n_tail) x 4 x 2 x 512 B` of K / V, ~2.1 MiB at a saturated list with two tail ids), so a step moves `S_q x` the single-row step's
bytes -- ~16 MiB per sequence per layer at `S_q = 4`, `top_k` 512 and 2 KV heads (~4 MiB at `S_q = 1`); the 64-wide single-item
form that would gather the shared list once is not built.  The gated attention block's decode mode (`kv_lens` /
`block_table` / `slot_mapping` / pools through `_SparseSdpa`) is served since 2026-10-14: `kv_source="paged"` on a dense `[B, S_q]`
`QsaSpec` declaration binds this form -- `list_per_sequence` at `S_q <= 4` (the block derives the list form from `S_q`; one list per row
past it), `bottom_right` from the geometry, the split from `decode_split_kv` or the block's fill rule (`ceil(SMs / (B x H_kv x S_q))`
capped at 8 and at the item's tile count), the pools and the whole table as declaration samples, the gate and the LSE through the block -- validated
end to end against the fp32 block oracle (the serving-forms section above, `test_block_qsa_decode.py`).

Not served (typed declines by name; each lands with its accept cell and flips this section in the same change): the in-block
indexer (through the block; its decode step over the pools with it), THD and paged pools in one declaration, any of the decode form's three arms under THD, a shared list at `S_q > 4`, a split past the item's tile
count, the kernel's own epilogue gate under a split (the combine carries it), a sink, a sliding window / band, Q-length trimming
of a dense batch, log2 stats, a pure caller list without the tail, head dims other than 256, FP8 / MXFP8.
