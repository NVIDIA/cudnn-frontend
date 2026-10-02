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

`sdpa_fwd_prefill_sm100` and `sdpa_fwd_prefill_sm120` (f16/bf16) are default candidates,
ranked against the backend per measured shard (`sdpa/fwd/placement.py`); every other FROST
SDPA engine is `opt_in=True`: set `CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1` before
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
SM107 MXFP8 at exact device cc 10.7 uses prepared launches for its four existing dense, unsplit native
head shapes; THD, split-KV and PackGQA remain unsupported. Dense MXFP8 runtime
shape overrides remain declined because SF batch/head pitches are plan-fixed.
SM107 D256 MXFP8 gates also use the prepared host. The existing direct-only SM100
D128/D192 PV-BF16 specialization prepares native layouts with BF16 V and no SF_V
operand; graph eligibility is unchanged. Synthesized KV-tail padding, conversion
layouts and bias continue to decline runtime shape/stride overrides. Existing
conversion layouts use staged copies around a prepared compact plan;
explicit opt-in does not bypass the contract. The same pure capability predicate
filters candidate knobs and selects the prepared executor. Static-geometry graph
eligibility is unchanged.

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

Engine: `sdpa_fwd_prefill_sm90`, public ID **20517**, manifest slot 17 (opt-in).
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
wave); a balanced width is the follow-up.
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
`sdpa_fwd_prefill_sm100_mxfp8`, `sdpa_bwd_sm100`, `sdpa_bwd_sm100_mxfp8`. The
half-precision backward engine serves **only the large-head-dim band,
d ∈ (256, 512]**, and the MXFP8 backward engine serves **exactly d = 256** —
every other `sdpa_backward()` / `sdpa_mxfp8_backward()` shape still falls
through to the cuDNN backend. The BPROP column below reads ᵇ for the
half-precision row and ᵍ for the MXFP8 row.

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
host loops over head chunks to hold it under 4 GiB; under THD it is
`2·H_chunk·(T_q + B·256)·pad(S_kv_max)·2 B` instead — see ʰ), and everything in the band
is **envelope-served** — the tiles are fixed at 512, so d=264 costs the same
MMA as d=512.

| Feature | d64 (GPT-OSS)<br>FPROP | d128 (Llama)<br>FPROP | d192×d128 (DSv3 MLA)<br>FPROP | d256 (Qwen)<br>FPROP | d512 (DSv4)<br>FPROP | d512 (DSv4)<br>BPROP<br>d ∈ (256, 512] |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| **Data types** | | | | | | |
| FP16 / BF16 | ✅ (native d64)⁷ | ✅ | ✅ | ✅ | ✅ | ✅ᵇ |
| FP8 E4M3 / E5M2 (per-tensor descale) | ✅ (native d64, cga1)⁷ | ✅ | ✅ | ✅ | ✅ | ❌ |
| MXFP8 (E4M3/E5M2 + per-32 E8M0 SF) | ✅ (native d64, cga1, dense / unsplit / unpaged)⁷ | ✅ | ✅ | ✅ | ✅ | ✅ᵍ (E4M3 only, d=256) |
| O dtype ≠ QKV dtype — **quantized graphs only**¹ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ᵍ (fp16/bf16 gradients) |
| Block-scaled O (`sdpa_fp8` / `sdpa_mxfp8` + `sf_o`): FP4_E2M1 O + E4M3 scale per 16 d, or E4M3 O + UE8M0 scale per 32 d — per-tensor FP8 and block-scale MXFP8 graphs, dense/unsplit/unpacked only; `scale_o` doubles as the FP4 global scale (a python-only input on `sdpa_mxfp8`) | ❌ | ✅ (FP8: SM100 / SM107 / SM120; MXFP8: SM100 / SM107) | ❌ | ❌ | ❌ | ❌ |
| Head-dim envelope (zero-padded below native) | f16 ×8 · fp8 ×16 up to 64 (**native 64**: the d128 kernel files at `d_flavor=64`)⁷ · mxfp8 exact | f16 ×8 · fp8 ×16 · mxfp8 exact | f16 ×8 · **fp8 exact (192, 128) only**¹⁰ · mxfp8 exact | f16 ×8 · **fp8 exact 256 only**¹⁰ · mxfp8 exact | f16 ×8 · fp8 ×16, floor 256² · mxfp8 exact | f16 (256, 512] ×8ᵇ · mxfp8 exact 256ᵍ |
| **Layout** | | | | | | |
| BSHD | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ᵇ ᵍ |
| Arbitrary dense B/H/S stride order (`dense_flex`) | f16 only | f16 only | f16 only | ✅ | f16 only | ✅ᵇ ᶜ · ❌ᵍ |
| THD / ragged (packed varlen)ᵏ | f16 only⁹ | ✅ | f16 + fp8 + mxfp8³ | f16 + fp8 + mxfp8⁹ | f16 + fp8 + mxfp8³ | ✅ᵇ ʰ · ❌ᵍ |
| `cu_seq_len_q/kv` prefix sums (THD only) | f16 only⁹ | ✅ | ✅ | ✅ | ✅ | ❌ʲ |
| **Masks / features** | | | | | | |
| Causal (top-left) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ᵇ ᵈ ᵍ |
| Causal bottom-right | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ᵇ ᵈ · ❌ᵍ |
| Causal right-band widening | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ᵇ · ❌ᵍ |
| Sliding window (left) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ᵇ · ❌ᵍ |
| Padding mask (`seq_len_q/kv`) | ✅ | ✅ | ✅ | ✅ | ✅ | THD onlyᵇ ʰ · ❌ᵍ |
| THD + causal family (top-left / bottom-right / SWA / band) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ᵇ ʰ · ❌ᵍ |
| Padding mask + stats (per-batch LSE trim) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Dense padded-Q trim (O:=0, LSE:=−inf) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Attention sink (incl. `S_q == 1` decode)ˢ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Base-2 stats (`stats_use_log2`) | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| GQA / MQA (`H_q ≠ H_kv`) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ᵇ ᶠ ᵍ ʰ |
| PackGQA (`PACK_GQA` knob: the GQA group packed into the Q tile)ᵐ | whole group onlyᵐ | ✅ᵐ partial | whole group onlyᵐ | ✅ᵐ partial | whole group onlyᵐ | — |
| THD prefill + PackGQA (f16/bf16, CGA2, unsplit)ᵐ | ❌ | ✅ native, partial | ❌ | ❌ | ❌ | — |
| Bias / dBias | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `use_deterministic_algorithm` | — | — | — | — | — | ❌ᵇ · ✅ᵍ |
| Ragged `S_kv` (non-multiple of 128) | ✅⁶ | ✅⁶ | ✅⁶ | ✅⁶ | ✅⁶ | ✅ᵇ ᵉ ᵍ |
| Decode-shaped (`S_q == 1`; with sink / sliding window: ˢ) | ✅ | ✅ | ✅ | ✅ᵈ (decode tile) | ✅ | ❌ᵇ · ✅ᵍ |
| **Decode tile** (decode + MTP; f16/bf16, dense or paged: d128 `S_q · PACK_G ≤ 128`ᵈᵗ, d256 `S_q · G` within the routed rowsᵈ) | ✅ᵈ⁶⁴ **native** (f16/bf16; an explicit cga2 runs the prefill tile) | ✅ᵈᵗ **native** | ❌ (prefill tile) | ✅ᵈ (d256 decode tile, swap-AB) | ❌ (prefill tile) | — |
| Paged KV cache (`paged_attention_k/v_table` + padding mask)ᵖ | ✅ᵖ f16 + fp8 (native d64; mxfp8 d64 is unpaged) | ✅ᵖ f16 + fp8 + mxfp8 | ✅ᵖ f16 + mxfp8 (native (192, 128)) | ✅ᵖ ᵈ f16 + mxfp8 | mxfp8 only | ❌ |
| Fused epilogue gate (sdpa virtual `O_v` → `mul(O_v, sigmoid(G))`; the SM107 rows serve it, see the SM107 table) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

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
prefill pipeline instead.

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
at one token per sequence, nvbug 6607857; every other ragged graph keeps `TILE_CGA_M=2`
and a pinned 1 declines), fp8 / mxfp8 (no quantized decode tile: their (128, 128) flavors keep
`cgas={2}`, their other flavors their own width), and the d192x128 / d512 f16 flavors
(no decode tile yet — their decode graphs run the prefill kernel as before; the d256
f16/bf16 flavor has its own swap-AB decode tile, ᵈ). d64 has its own native decode tile (ᵈ⁶⁴). Measured on B200 (graph path,
CUDA-graph replay, b=32, d=128, S_kv=4096, page 16, bf16): 64/4 S_q=1 119.2 us on the
prefill tile → 48.9 us on the decode tile (the cuDNN backend's decode engine: 44 us);
64/4 MTP S_q=4 127.5 → 50.6 us; 64/8 S_q=1 234.6 → 96.4 us; 96/8 S_q=1 (G=12 packs 4,
partial PackGQAᵐ on both tiles) 615 → 225 us (the same shape unpacked on the decode tile:
875 us); d64 64/8 230.5 → 80.2 us. The kernel docstring carries the full table.

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
bit-clean Stats. Contract: the native (128, 128) f16/bf16 flavor on cc10.0/10.3 (Rubin
has no decode tile), `S_q(max) == 1`, page pools (a ragged K/V needs the prefill THD
leg's clamped descriptors), ragged Stats when Stats are requested (a per-batch padded
Stats has no ragged base), int32 **or** int64 offsets of one width whose multiplier
divides the row (`engines._thd_decode_leg`), per-batch `seq_len_kv` (not the cu form),
no sink, no gate. Twins: `engines._thd_decode_leg` / `SdpaFwdDslSm100.thd_decode_leg` /
`config_sm100._validate_params(ragged_q)` / `CfgD128Decode.RAGGED_Q`. **Not yet:**
MTP-THD (`S_q(max) > 1` needs the per-sequence Q length in the bottom-right diagonal —
the prefill THD leg keeps it), ragged K/V (non-paged THD decode), d192×d128 / d256 /
d512 ragged decode (their decode graphs keep the prefill THD leg), fp8.

ᵖ **Paged KV (issue #920): f16/bf16 on the d64 / d128 / d192×d128 / d256 flavors, per-tensor
FP8 on the d64 / d128 flavors, MXFP8 on its four d ≥ 128 native flavors** (`Capabilities.paged_d_shapes`:
the f16/bf16 row `{(64, 64), (128, 128), (192, 128), (256, 256)}`, the SM100 per-tensor FP8 row
`{(64, 64), (128, 128)}`, the MXFP8 row `{(128, 128), (192, 128), (256, 256), (512, 512)}`). The head-dim
gate is the flavor the lowering SELECTS — the smallest covering envelope — not the raw
dims: d=64 is **native** (paged wired on the d64 flavor), d=192/192 and mixed dims such as (256, 128) or
(64, 192) ride the d256 one, (192, 128) is **native** on d192×d128 and shapes below it
such as (136, 72) ride its envelope; a d512-envelope selection ((512, 512), (512, 128),
...) is declined on f16/bf16 until that kernel wires the specialization. On the FP8 row the
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
e.g. 4000) does not withhold the split, unlike a mask-free dense `S_kv`, which rides
synthesized KV-tail padding the split cannot. The attention sink (incl. `S_q == 1`) and
a left sliding window under the bottom-right causal diagonal ride the same paged graph
on every wired f16/bf16 flavorˢ; on the FP8 row the left window rides it; the sink and
the block-scaled O epilogue (`sf_o`) over pools are not validated, so those two pairs
stay declined. The FP8 graph is `sdpa_fp8`
with scalar `descale_q/k/v` and `scale_o` (`scale_s`/`descale_s` accepted and ignored),
O in FP16/BF16/E4M3/E5M2, `Amax_O` out. **The graph must not declare
`Amax_S`**: every FROST FP8 row declines a graph that requests that output (the kernels
do not produce it), so a generic fp8 graph that binds `Amax_S` — the spelling a
FlashInfer-style wrapper produces today — stays on the backend engine; the paged FP8
capability is reachable only by omitting it. Masks validated over FP8 pools: a causal
upper bound top-left or bottom-right (MTP `S_q <= 8`, each batch's diagonal anchored
at its own KV length, rows left without a key write O := 0 / LSE := -inf) and a left
sliding window (`test_sdpa_fwd_paged_sm100.py` fp8 causal / sliding-window tests, pinned
on the FROST plan, and the `test_mhas_v2.py` fp8 paged decode fuzz, which draws the same
masks over the default walk and asserts the row served every draw). Not yet: packed (ragged-offset) block tables, the f16/bf16 and FP8 d512 flavors, the SM107 (Rubin)
siblings, sink + KV split (a sink graph runs unsplit — see ˢ), sink over FP8 pools, block-scaled O
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
with the default `decode=True` (SM107 f16/bf16, SM120 f16/bf16: sink at `S_q == 1`
is ❔ there — same epilogue fold, not run here). The quantized rows are untouched by
it: `sdpa_fp8()` / `sdpa_mxfp8()` build `SDPA_FP8` / `SDPA_MXFP8` nodes that never
passed through that branch, so the sink-at-decode status of SM100 / SM107 per-tensor
FP8, SM100 / SM107 MXFP8 and SM120 FP8 (❔) is pre-existing and unchanged, not newly
exposed. Sink + split-KV stays declined on
every row (`split_kv > 1 serves dense, unpadded, sink-free graphs only`): a sink
decode graph runs unsplit, one cluster per (batch, KV head), until a sink-aware
`split_combine` lands.

ᵈ **d256 decode tile (`sm100/decode_d256_f16.py`, 2026-09-16).** Same engine row,
same graph contract, a second template: f16/bf16 d256 graphs (d_qk = d_v in
(128, 256], the d256 envelope) whose packed Q rows fit 16 -- `S_q * G <= 16` with
`G` the PackGQA group (16:1 at S_q = 1, 8:1 at S_q <= 2, 4:1 at S_q <= 4, MHA at
S_q <= 16; `config_sm100.D256_DECODE_ROUTED_MAX_Q_ROWS`) -- lower onto a swap-AB
tile: KV tokens on the MMA M axis (128 keys per tile), the 16 Q rows on N, one
cta_group::1 CTA per (KV-head group, batch, split), softmax reducing over TMEM
lanes (4 warps per 16 Q columns), P^T through a swizzled SMEM tile, O^T in TMEM.
Dense padded, dense unpadded (synthesized KV tail) and paged caches, every mask
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
plan (`select_plan`). On the 16-column tile the b=32 x 2 KV-head x 4096-key
serving shape runs unsplit, b=8 splits 8 ways, b=128 stays unsplit, s_kv=16384
splits 2 ways. The unrouted 32-column tile's fit (1.6x the per-tile cost) stays in
the model for when it is routed.

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
(on the unrouted 32-column decode tile it measured 90.2 / 90.2 unsplit and 96-103 /
58.5 split 2). Dense 16-row graphs the rule newly takes, replay, prefill -> decode:
MHA 8/8 b=8 S_q=16 s_kv=2048 32.9 -> 29.6; MHA 32/32 b=1 S_q=16 s_kv=32768 243.5 ->
179.6 (split 4), causal 16.5 -> 12.5; GQA 32/2 b=4 S_q=1 s_kv=32768 68.0 -> 60.6;
GQA 32/2 b=32 S_q=1 s_kv=4096 61.8 -> 51.2; padded GQA 16/2 b=32 S_q=2 bottom-right
52.8 -> 51.2 and 32/4 b=16 52.8 -> 49.2; MHA 64/64 b=1 S_q=16 s_kv=4096 62.4 ->
51.4; synthesized KV tail S=300 14.4 -> 12.3, S=1000 20.5 -> 20.5. The main kernel
alone reaches the 46 us of the TensorRT-LLM decode kernel (via FlashInfer) at split 2;
the shared combine pass (~6 us) is the remaining captured-path gap. Not a Capabilities change (the row's claims are
unchanged; this documents the lowering), Rule S2.

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
storage orders. Heuristics prefer packing for causal THD with GQA4/GQA8;
other supported THD groups remain available as explicit tuning candidates.

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
⁶ Served through the padded path with synthesized full-length KV lengths, or
natively when the causal band covers the KV tail.
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
Rubin (SM107) has no d64 sibling and keeps the d128 envelope (ⁱ).
⁸ MXFP8 sets `d_pad_multiple=0` (exact native shapes only — the scale-factor
plumbing is not audited for envelope zero-padding), so an MXFP8 graph off every
native flavor's exact head dims — `(64, 64)` is one since ⁷ — is declined at plan
time rather than padded.
ᵇ **`sdpa_bwd_sm100` — the d512 (DSv4) backward, and the ONLY FROST backward on
this arch: d ∈ (256, 512], multiples of 8, f16/bf16.** It is one engine over one
head-dim band, not a per-flavor column, so it does not follow the FPROP columns
to its left: below 257 the d256 flavors are the right kernel and the engine
declines, and every other `sdpa_backward()` shape on SM100 falls through to the
cuDNN backend. The whole band is **envelope-served** on 512-wide tiles, so d=264
pays d=512's MMA cost. Multiples of 8 rather than the forward's 16: the stage-3
epilogue narrows its store vector from 32 B to 16 B when d is not also a
multiple of 16.
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
by the shared `dkv_reduce` kernel (deterministic, fixed-order fp32). dQ runs one
GEMM per group member so the shared K head lines up without an expand or a copy.
The head chunk is forced to a multiple of the group.
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
a per-sequence bottom-right diagonal `S_kv[b] − S_q[b]`. Stage 3 is rendered
**untrimmed** under THD — its K-trim is expressed in absolute workspace rows, which
the blocked layout renumbers per sequence — and the caller zero-fills the blocked
workspace instead, which is what makes the masked-and-therefore-unwritten tiles
read as zero. That costs the k-tiles causal would have skipped (see ᵈ) — measured
at **−20 %** on the whole backward (A/B/A, dense path with the trim forced off,
B=1 H=128 S=8192 d=512 bf16 causal: ~259 → ~207 TFLOPS). Correct, and a known
optimization gap: re-trimming per sequence needs `row_off[b]` folded into the
bounds and the bottom-right diagonal threaded per group.
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

These engines require a CuTe DSL build with the `sm_107a` target. Public
4.7.0 meets the shared DSL floor but lacks this target; graph admission and
standalone support checks decline it with the installed version before compile.
SM100/SM120 keep their existing 4.7.0 floor.

Engines: `sdpa_fwd_prefill_sm107` (f16/bf16), `sdpa_fwd_prefill_sm107_fp8`
(per-tensor FP8) and `sdpa_fwd_prefill_sm107_mxfp8` (block-scale) forward;
`sdpa_bwd_sm107` (f16/bf16), `sdpa_bwd_sm107_fp8` (per-tensor FP8 E4M3,
`sdpa_fp8_backward`) and `sdpa_bwd_sm107_mxfp8` (block-scale MXFP8 E4M3,
`sdpa_mxfp8_backward`, see ᵐˣ) backward — **d = 256 exactly**, see ᵇ. Every other
backward graph on the Rubin line (any other head dim) gets a typed
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

| Feature | d64 (GPT-OSS)<br>FPROP | d128 (Llama)<br>FPROP | d192×d128 (DSv3 MLA)<br>FPROP | d256 (Qwen)<br>FPROP | d512 (DSv4)<br>FPROP | d256<br>BPROPᵇ |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| **Data types** | | |  | | | |
| FP16 / BF16 | ⚠️ⁱ | ✅ | ✅ | ✅ | ✅ | ✅ᵇ |
| FP8 E4M3 / E5M2 (per-tensor) | ⚠️ⁱ | ✅ | ✅ | ✅ | ✅ | E4M3 ✅ᵇ · E5M2 ❌ |
| MXFP8 | ❌ | ✅ | ✅ˣ | ✅ | ⚠️ⁱᵛ | ✅ᵐˣ (E4M3 payloads, bf16 gradients, **no amax**) |
| O dtype ≠ QKV — **quantized graphs only** (fp16/bf16/fp8 out) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ᵇ (fp8 row: dQ/dK/dV e4m3, bf16 or fp16) · ✅ᵐˣ (mxfp8 row: bf16 only; fp16 declined, follow-up) |
| Block-scaled O (`sdpa_fp8` / `sdpa_mxfp8` + `sf_o`): FP4_E2M1 O + E4M3 scale per 16 d, or E4M3 O + UE8M0 scale per 32 d — dense/unsplit/unpacked only | ❌ | ✅ | ❌ | ❌ | ❌ | — |
| Head-dim envelope | none — runs the d128 kernelⁱ | f16 ×8 · fp8 ×16 | f16 ×8 · fp8 exact only (floor 128) | f16 ×8 · fp8 ×16 (floor 255) | f16 ×8 · fp8 ×16 (floor 256) | none — exact (256, 256) |
| **Layout** | | |  | | | |
| BSHD | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Arbitrary dense stride order (`dense_flex`) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| THD / ragged (packed varlen) | ✅ᶻ | ✅ᶻ | ✅ᶻ | ✅ᶻ | ✅ᶻ | ❌ |
| `cu_seq_len_q/kv` prefix sums (THD only) | ✅ᶻ | ✅ᶻ | ✅ᶻ | ✅ᶻ | ✅ᶻ | ❌ʲ |
| **Masks / features** | | |  | | | |
| Causal (top-left) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Causal bottom-right | ✅ | ✅ | ✅ | ✅ | ✅ | f16 ✅ · fp8 ✅ **`S_q % 128 == 0` only**ᵇ · mxfp8 ✅ **`S_q % 128 == 0` only**ᵐˣ |
| Causal right-band widening | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Sliding window (left) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Padding mask (`seq_len_kv`) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ graph form (carries `seq_len_q`); per-batch `seq_kv_lens` on the f16 row's standalone adapterᵇ |
| Padding mask + stats (per-batch LSE trim) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Dense padded-Q trim (O:=0, LSE:=−inf) | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| Attention sink (at `S_q == 1`: ❔ — see SM100 ˢ) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Base-2 stats (`stats_use_log2`) | ❔ | ❔ | ❔ | ❔ | ❔ | — |
| GQA / MQA (`H_q ≠ H_kv`) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| PackGQA | fp8 only | fp8 only | ❌ | ❌ | ❌ | — |
| Split-KV | fp8 only | fp8 only | ❌ᵛⁱⁱ | ❌ᵛⁱⁱ | ❌ᵛⁱⁱ | — |
| Fused epilogue gate (sdpa virtual `O_v` → `mul(O_v, sigmoid(G))`, `G = (B, H_q, S_q, D_v)`; graph tail + standalone `sample_gate`)ᵛⁱⁱⁱ | ❌ | ❌ | ❌ | f16/bf16 ✅ · fp8 ✅ (bf16 G) · mxfp8 ✅ (bf16 G; a gated e4m3 O is unscaled) | ❌ | — |
| Optional stats (LSE store compiled out) | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| Bias | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Ragged `S_kv` (non-multiple of 128) | ✅ⁱˣ | ✅ⁱˣ | ✅ⁱˣ | ✅ⁱˣ | ✅ⁱˣ | ✅ᵇ (any S_q / S_kv; padded to 128 / 256 — except bottom-right on the fp8 and mxfp8 rows, which need `S_q % 128 == 0`ᵇ ᵐˣ; the mxfp8 row also re-stages the scale-factor pads zero-filledᵐˣ) |
| FP16 softmax accumulate (`sdpa(softmax_precision=HALF)` op attribute) | ❔ⁱⁱⁱ | fp8 only (Rubin f16x2 arm) | fp8 only (same body as d128) | ❌ | ❌ | — |

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
(`dkv_reduce`, fixed order). Served: dense, top-left and bottom-right causal,
sliding window (left), MHA / GQA / MQA, **any** S_q / S_kv (a non-multiple of the
128-row q tile / 256-row kv block is padded through zero-filled staging copies,
`+inf` LSE rows, and the kernels' padded-mask arm at the uniform real S_kv), BSHD-
physical io, contiguous fp32 Stats. **One shape-conditioned decline, fp8 row only:
bottom-right causal needs `S_q % 128 == 0`.** The bottom-right diagonal is
`S_kv − S_q` in REAL rows; the f16 body takes the real lengths (`sq_real` /
`skv_real` on its `compile()`, `SQ_REAL` in its problem_size, adapter passes
`s_q_max`) while the fp8 body's ABI has no `seqlen_q_real` and derives the diagonal
and the q-tile trim from the PADDED compile extent (its kv term IS the runtime real
length), so a ragged S_q would shift both by `pad − S_q` rows — finite, wrong dQ/dK/dV
near the diagonal, no crash. `mismatch()` declines it at eligibility
(`Capabilities.bottom_right_s_q_multiple = 128`; the half row keeps 1) and the
adapter backstops (`SdpaBwdDslSm107Fp8._check_support_family`); a ragged S_kv under
bottom-right is served on both rows. Tests:
`test_sdpa_bwd_fp8_sm107.py::test_reject_bottom_right_with_ragged_s_q` (host, +
the half row admitting the same graph), `::test_fp8_adapter_backstop_refuses_bottom_right_with_ragged_s_q`,
`::test_causal_bottom_right_{rectangular,ragged_kv}` (Rubin accept),
`::test_unserved_bottom_right_ragged_s_q_graph_declines_as_not_supported` (Rubin,
typed error end to end); `test_sdpa_bwd_dsl_sm107.py::test_causal_bottom_right_ragged_s_q`
pins the f16 claim. Follow-up: thread `seqlen_q_real` through the fp8 body like the
f16 one, then set the field back to 1 (the reject test inverts by itself). The dS
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
scale, cast). The bf16-dS twin (`api_dsl_sm107.FP8_DS_DTYPE = DTYPE_BF16`: bf16 GEMMs
over exact E4M3 → bf16 upcasts of Q / K, three fold passes, `descale_dP` / `scale_dP`
bound and unused) is the A/B and oracle base. E5M2 payloads are declined (no body).
**Declined on both rows, each asserted by a test:** graph padding masks
(`seq_len_q/kv`: a padded `sdpa_backward` graph carries BOTH lengths — the frontend
requires them — and no body threads per-batch Q lengths, so the graph form is declined
rather than served while ignoring the q lengths; the f16 body does read per-batch kv
lengths, which the half row's STANDALONE adapter serves — `seq_kv_lens_present=True`
+ `execute(seq_kv_lens=)`, the same padded-mask arm, dead rows / batches exact zeros,
bottom-right keeps the dS zero-fill for its per-batch diagonal — ahead of every chunk, and
with a sliding window dropped from the stage-3 trim, so the GEMMs read the plain
bottom-right band there; the fp8 body takes one
uniform `seqlen_kv_real`, so its adapter declines them too; tests
`test_sdpa_bwd_dsl_sm107.py::test_adapter_per_batch_kv_lengths*`,
`::test_adapter_dead_kv_entry_is_exactly_zero`, `::test_padding_mask_graph_always_carries_seq_len_q`),
sink / dSink, bias / dBias, right-band widening, THD,
`dense_flex`, decode shapes, and `use_deterministic_algorithm` (the chain has no
atomics; the claim waits on the bring-up sweep). The bf16 d256 graph has a native
backend competitor (engine 17): pin the engine when validating or measuring. Both
rows are `opt_in`. Tests: `test_sdpa_bwd_dsl_sm107.py`, `test_sdpa_bwd_fp8_sm107.py`.
ᵐˣ **d=256 MXFP8 backward (`sdpa_bwd_sm107_mxfp8`, `sdpa_mxfp8_backward`;
`api_dsl_sm107.SdpaBwdDslSm107Mxfp8`, kernel `sm107/bprop_d256_mxfp8.py`, config
`bwd/config_sm107.py` `FAMILY_MXFP8`).**
ONE kernel plus scale-factor plumbing (no separate SF-repack or dequant kernels): the per-tensor fp8
body's pipeline with the F8_128x4 E8M0 scale factors dequantizing INSIDE every tcgen05
block-scale MMA (K / V / Q / dO / dO_T SF ride their operands' TMA barriers into TMEM
through UTCCP), P quantized to E4M3 with the fixed 2⁸ scale (byte 119, cuDNN's MXFP8
convention; `p_scale_log2` pinned to 8, any other value a typed decline), dV in TMEM
(bf16 TRUE-unit per-Q-head partials), dS = attn_scale · P ∘ (dP − delta) from the
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
causal at `S_q % 128 == 0` / sliding window (left) — WIDER than the SM100 MXFP8 row
(bottom-right, SWA). `descale_v` is the ROWWISE V scale in the backward (the C++ node's
own reference math dequantizes V like K); the adapter asserts that shape for `sf_v` and
byte counts only for the other six SF tensors (the node rewrites two SF strides before
lowering). **Declined, each asserted by a test:** `amax_dQ / dK / dV` requested as real
outputs — the backend's canonical MXFP8 backward graph declares them
(`test/python/sdpa/mxfp8.py`), so this is a documented parity gap (AGENTS Rule 9): the
row produces no amax; E5M2; fp16 gradients; bottom-right at a ragged
S_q; padding masks (the graph form, which carries `seq_len_q`, and per-batch kv lengths
on the standalone surface — this body takes one uniform `seqlen_kv_real`); sink / dSink;
bias / dBias; right-band widening; THD;
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
bytes exceed 128, elsewhere ≤ 3.9e-3). The host-side dequant (`_dequant_mxfp8_to_bf16`) decodes the E8M0 byte
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
ⁱ No native d=64 Rubin kernel, so a d=64 graph rides the d128 envelope (64 is a
multiple of 8 at f16 and of 16 at fp8) at ~2× the MMA cost.
ⁱⁱ `thd_d_shapes={(128,128)}` on the FP8 row is exact — d=64 THD is declined.
ⁱⁱⁱ **Accepted, not validated.** `softmax_precision=HALF` (requested as the
`sdpa()` op attribute — numerics-changing, so it is a graph fact gated by the
row's `softmax_precisions`, not a tuning knob) is gated on
`flavor == (128, 128)` (`fwd/api_dsl.py`), and a d=64 graph's *flavor* IS
(128,128), so the request passes the probe and the kernel runs. Untested is the
f16x2 exponent arm over the zero-padded 64 → 128 region.
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
ᵛⁱⁱ The f16 Rubin kernels wire no SplitHelpers.
ᵛⁱⁱⁱ **Fused epilogue gate `O := O * sigmoid(G)`** — a production feature of the
d256 f16/bf16, per-tensor FP8 and block-scale MXFP8 Rubin kernels
(`TemplateParams.epilogue_gate`; rows `sdpa_fwd_prefill_sm107`,
`sdpa_fwd_prefill_sm107_fp8` and `sdpa_fwd_prefill_sm107_mxfp8`,
`Capabilities.epilogue_gate_d_shapes = config_sm107.SM107_EPILOGUE_GATE_SHAPES`, the
same constant the standalone adapter's `check_support` twin reads). On the graph it
is the three-node tail `sdpa(virtual O_v) -> sigmoid(G) -> mul(O_v, s)`
(`cudnn._sdpa_tail.match_gate_tail`; either mul operand order), validated
python-natively and lowered with the mul output bound as the kernel's O — the
virtual `O_v` / `s` are never bound or materialised. Served **exactly** at
(256, 256) (not through the head-dim envelope), dense only (no THD), unsplit,
non-PackGQA, non-paged; the padding mask (`seq_len_kv`) composes. `O_v`, `G` and
the final O must be DECLARED (dim + stride, BSHD) — the classic C++
`pre_validate_node` needs both on the sdpa output and only user-assigned values
are pushed at lowering, so an undeclared `O_v` is a typed not-supported from
`validate()` rather than a bare `ValueError` out of planning. `G` is Q's dtype on
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

ᶻ **f16/bf16 THD is served on EVERY flavor** as of 2026-09-09 (d128, d192×d128,
d256, d512), and per-tensor FP8 THD at d128 and d192×d128. The f16 bodies were
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
verified to reproduce at `60cb6cda`. **Still declined: MXFP8 THD line-wide** —
those bodies keep the pre-upstream 7-arg setup call and their scale-factor
tensors have no packed-THD layout yet, so the row declines rather than
half-serving.

ʸ Per-tensor FP8 **THD** at d192×d128 came free with the DSv3 port and is
served as of 2026-09-09: that kernel *is* the shipped d128 FP8 body (only the
config factory differs), so its THD leg is the validated wiring — 5 THD cases
pass on `w2u1g-lc-0030`. Closing it needed both enforcement points widened
together, the row and the standalone adapter's gate; they now share one
constant (`config_sm107.SM107_FP8_THD_SHAPES`) so they cannot drift. d256 and
d512 still raise at `compile()` — their ported bodies call the setup kernel with
the pre-upstream 7-arg contract against a 14-arg helper, and their metadata
layout is 3B+2 where the helper builds 4B+4.

ˣ d192×d128 MXFP8 runs at **cga2 only** (SM100 serves the shape at cga1 and
cga2). See the paragraph above: at cga1 this flavor's scale-factor tiles cross
the 256 KiB version-0 tcgen05 descriptor window. Every other cell in that
column was measured on `w2u1g-lc-0030` (cc 10.7) by
`test_sdpa_fwd_{fp8,mxfp8}_sm100.py`, whose dense d192 cases stopped skipping on
Rubin in the same commit — 31 passed, and the 18 that still skip are the
PackGQA and THD families this line declines per-feature, not per-shape.
ⁱˣ Served through the padded path with synthesized full-length KV lengths
(`skv_tail_via_padding`). REQUIRED, not an optimization: with no mask the
kernel's KV loop bound is a floor division, so an un-synthesized ragged `S_kv`
would silently drop the tail tile.

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
`SCHED_LPT` at **(256, 256)** only. Validation: cos 1.0000 at n_kv 2/3/4/8,
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

Still declined, and why: d128/d512 f16 are **unvalidated** under LPT rather
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
`sdpa_bwd_sm120`. Head dims are a **continuum**, not per-model flavors: the
kernel picks Q/K and V head tiles independently (f16/bf16 head dims it would
tile at 256 on both sides run a dedicated template, `sm120/prefill_d256_f16.py`
with the same support).

D512 FP16/BF16 FPROP (`sm120/prefill_d512_f16.py`) supports **both D_QK and D_V
in (256, 512]**, in multiples of 8. It uses a 64 x 32 CTA and 512 x 512 head
tiles: (512, 512) is **native**; smaller shapes are **envelope-served** by
zero-padding, with the same MMA and on-chip storage cost.
FP8 FPROP (`sm120/prefill_d512_fp8.py`) adds the same independent band with
**multiples of 16** and a **64 x 64 CTA**. BPROP remains limited to 256.

| Feature | FPROP<br>d ≤ 256, any ×8;<br>both dims ∈ (256, 512], any ×8 | FPROP FP8<br>both ≤256 or both ∈ (256, 512], any ×16 | BPROP<br>d ≤ 256, any ×8 |
|---|:--:|:--:|:--:|
| **Data types** | | | |
| FP16 / BF16 | ✅ | — | ✅ |
| FP8 E4M3 / E5M2 (per-tensor) | — | ✅ | ❌ |
| MXFP8 | ❌ | ❌ | ❌ |
| O dtype ≠ QKV — **FP8 graphs only** (fp16/bf16/fp8 out) | ❌ | ✅ | — |
| Rectangular head dims (D_QK ≠ D_V) | ✅ (independent within each served band) | ✅ (independent) | ✅ (D_QK ≥ D_V) |
| Head-dim alignment (actual `D_QK`/`D_V`) | ×8, both ≤ 256ᵃ; or both ∈ (256, 512] | ×16, both ≤ 256ᵃ; or both ∈ (256, 512] | ×8, ≤ 256 |
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

Base-2 stats are wired into all three f16 templates and the FP8 template,
including the final Split-KV combine. Target-SM120 numerical validation remains
required; local SM80/SM100 results do not validate these kernels.

ᵃ **Head TILE granule and head-DIM alignment are different numbers — the column
headers quote the head-dim rule.** `GENERAL_HEAD_TILES` steps by 16 (f16), and
`FP8_GENERAL_HEAD_TILES` steps by 32; the wide FP8 flavor uses fixed 512-wide
head tiles. Those are the kernel's native tile sizes; an actual `D_QK`/`D_V`
only has to satisfy `d_pad_multiple` — **8** on the f16 row, **16** on the FP8
row (the TMA 16-byte global-stride rule at 2 and 1 bytes/elem) — and is
zero-padded up to the next tile. So a d=72 f16 graph is eligible and computes
on the 80-wide tile.

---

## SM80 (Ampere A100, cc 8.0 exactly)

Engines: `sdpa_fwd_prefill_sm80`, `sdpa_bwd_sm80`. Both use `mma.sync` (no
tcgen05) and assume the A100's 164 KiB opt-in SMEM — sm86/sm89 are declined.
The backward serves packed THD graphs (ᵏ); the forward does not yet.

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
| THD / ragged | ❌ / ✅ᵏ | ❌ / ✅ᵏ | ❌ / ✅ᵏ | ❌ / ✅ᵏ |
| `cu_seq_len_q/kv` | ❌ / ❌ʲ | ❌ / ❌ʲ | ❌ / ❌ʲ | ❌ / ❌ʲ |
| Strided / permuted Stats | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
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
The forward graph row still declines THD and RoPE, so graph eligibility is unchanged.

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
and become `cu_seqlens` on device in a one-warp setup launch. Like every FROST
THD row, the packed addressing is `prefix(lengths) × token stride`: the bound
ragged-offset VALUES are not read, so sequences must be adjacent (TE-style
padded THD with gaps between sequences, `cu_seqlens_padded != cu_seqlens`, is
not served and is runtime data that cannot be declined at plan time — issue
#737 tracks reading the offsets on device). **Declared
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
wrapper-only fusion no engine row admits. The forward row
still declines THD (the wrapper's `cu_seqlen` path serves it).

---

## Gaps at a glance

| Missing | Where |
|---|---|
| Backward pass entirely | SM90 |
| Backward outside d = 256 (f16/bf16, per-tensor FP8 E4M3 and block-scale MXFP8 E4M3): every other head dim; and on the d256 rows the dense padding mask, sink / dSink, bias / dBias, deterministic, THD, `dense_flex`, right-band widening, decode; on the MXFP8 row also fp16 gradients and the `amax_dQ/dK/dV` outputs | SM107 — the three d256 rows are the whole Rubin backward (see the SM107 table, ᵇ ᵐˣ) |
| Backward outside d ∈ (256, 512] (f16/bf16) or d = 256 (MXFP8) | SM100, SM103 — the two backward engines there serve exactly those bands |
| Backward per-batch padding mask (`seq_len_q/kv`) on a DENSE graph | SM100, SM103 — a UNIFORM non-tile-multiple length is served, and the THD path carries per-sequence lengths; a per-batch mask on a dense graph is not |
| Backward sink / dSink, bias / dBias | SM100, SM103 |
| Backward deterministic, decode | SM100, SM103 — served by the MXFP8 d=256 row only |
| MXFP8 backward: E5M2, bottom-right / band-widened / sliding-window masks, non-BSHD strides, `amax_*` outputs | SM100, SM103 |
| f16/bf16 forward split-KV | SM90, SM107 (Rubin), SM80 — none sets `split_kv_supported` |
| f16/bf16 forward PackGQA | SM107 (Rubin) — the row serves dense f16/bf16 at d128/d192×d128/d256/d512, THD on all of them as of 2026-09-09, and the dense padded-Q trim as of #1037; this is the machinery its kernels still lack (optional stats IS served — `lse_optional=True`) |
| d192×d128 quantized PackGQA / split-KV, and d192 MXFP8 THD | SM107 — the shape is served in FP8 and MXFP8 as of 2026-09-09, and per-tensor FP8 **THD** with it; PackGQA and split-KV stay wired in the d128 flavor only (`pack_gqa_d_shapes` / `split_d_shapes`), and the MXFP8 line declines THD row-wide |
| MXFP8 forward | SM90, SM120, SM80 (SM107 is served — see the SM107 table; d512 is ⚠️ⁱᵛ, correct but with no test module) |
| Per-tensor FP8 backward | every arch except SM107 d = 256 E4M3 (`sdpa_bwd_sm107_fp8`, ᵇ) |
| MXFP8 backward outside SM100/SM103 d = 256 and SM107 d = 256 (`sdpa_bwd_sm107_mxfp8`, ᵐˣ) | every arch |
| THD / ragged backward | SM120, SM107, and the SM100/SM103 MXFP8 row (the SM100/SM103 f16/bf16 row serves it — see ʰ; SM80 — see ᵏ) |
| THD forward | SM80 |
| **Native d=64 (GPT-OSS) forward kernel** | **SM107** — served via the d128 envelope at ~2× MMA cost. SM100/SM103 is native (⁷) for f16/bf16 (prefill, decode (ᵈ⁶⁴), paged and split-KV), per-tensor FP8 (prefill, paged, split-KV; cga1) and MXFP8 (dense / unsplit / unpaged; cga1) |
| Decode tile outside the d128 / d256 f16/bf16 flavors | SM100, SM103 — d192×128 / d512 decode and every fp8 / mxfp8 decode have no dedicated decode tile: each runs its flavor's prefill kernel at that flavor's own CGA width (f16 d512 and the quantized d128 flavors at `TILE_CGA_M=2`; per-tensor FP8 d256 and SM100 MXFP8 d256 / d512 are cga1 kernels; d192×128 selects 1 or 2 by shape). THD queries on the d128 f16/bf16 flavor keep its prefill pipeline (`TILE_CGA_M=2`) too (ᵈᵗ); d256 f16/bf16 graphs the adapter does not route onto the d256 decode tile (THD, or more packed Q rows than it routes, ᵈ) run the d256 prefill tile |
| **d192×d128 paged decode tile** | SM100, SM103 — paged (192, 128) is served (ᵖ) but at `S_q ≤ 8` runs the prefill tile. Measured on B200 (`S_q = 1`, `b = 32`, page 16, bf16, mixed `S_kv ≤ 4096`, default plan): 32/32 MHA **788.7 µs on the prefill tile vs 476.9 µs on the backend**; 32/8 GQA 275.8 vs 199.6 µs. Follow-up: a d192×d128 decode tile behind `TILE_CGA_M=1`, as ᵈᵗ is for d128 |
| d=64 quantized THD; d=64 MXFP8 paged / split-KV | SM100/SM103 (`thd_d_shapes` of both quantized rows, the MXFP8 row's `paged_d_shapes` / `split_d_shapes`, mirrored by `check_support`); every d=64 MXFP8 graph on SM107 (exact-shape gates) |
| Bias forward | SM90, SM100, SM107, SM120 |
| Dropout, ALiBi, `block_mask`, `score_mod` | every arch, both passes |
| Paged KV cache | every arch except SM100/SM103 forward on f16/bf16 d128 / d192×d128 / d256, per-tensor FP8 d128 and MXFP8 on every native flavor (see ᵖ); the f16/bf16 and FP8 d512 flavors, MXFP8 pools with unreordered SF or page_size < 128, THD queries over MXFP8 pools, packed (ragged-offset) block tables everywhere (THD queries over f16/bf16 pools ARE served — see ᵖ); THD queries and the attention sink over FP8 pools, a block-scaled O (`sf_o`) over FP8 and MXFP8 pools |
| Fused epilogue gate (`O * sigmoid(G)` tail) | every arch and flavor except SM107 d256 f16/bf16, per-tensor FP8 and MXFP8, exact (256, 256), dense / unsplit / non-PackGQA / non-paged (see the SM107 table) |
| PackGQA of a group sharing no factor with the 128-row tile (G = 3, 5, 7, …), and partial packing outside the SM100/SM103 f16/bf16 d128 / d256 kernels | every arch — such groups run unpacked (see ᵐ); the d192×d128 / d512 f16 and the fp8 / mxfp8 kernels pack the whole group only |
| Attention sink + split-KV (sink-aware `split_combine`) | every arch — a sink graph runs unsplit; at `S_q == 1` over a long KV that is one cluster per (batch, KV head) (see ˢ) |
| Attention sink at `S_q == 1` validated | every row except SM100/SM103 f16/bf16 (see ˢ) and SM90 f16/bf16: SM107 f16/bf16 and SM120 f16/bf16 accept it since the validator lift (f16/bf16 `sdpa()` graphs only) but are ❔; the FP8 / MXFP8 rows were never gated by that rule and stay ❔ as before. SM90 f16/bf16 is validated too (H200, `test_sdpa_fwd_dsl_sm90.py`: the #1095 accept graphs under a strict SM90 pin — dense packed / unpacked with a keyless batch, a THD envelope of one in both packed Stats layouts, base-2 Stats); paged KV and split-KV stay declined there by their own rows |
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

The two Rubin d=256 backward rows (`sdpa_bwd_sm107`, `sdpa_bwd_sm107_fp8`) use one
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
MXFP8 input paths. The declared SF_O atom geometry stays fixed; each call binds
current O, SF_O, optional Amax and device scales with observed storage checks.
FP4 O uses packed byte geometry, while V keeps its full logical head dimension.
SF_O offsets and prepared copy strides preserve Int64 addressing.
THD, paged, split-KV, PackGQA, gated output and shape-override combinations keep
their existing admission boundaries. Remaining conversion layouts use prepared
gather/scatter copies around a compact prepared plan. Standalone prepared calls
require caller-owned workspace.

### SM100 paged D128 THD split follow-up

Native FP16/BF16 D128/V128 paged THD prefill may use one-CTA split-KV with
optional GQA packing. It requires a matching native binder, bounded packed-Q
workspace, and packed NH/HN Stats; output Stats may use ln or log2. Q=1
retains its existing ragged decode path. Automatic selection is narrower:
SM100 BF16 B1, HND page16, GQA4 with Hq in {4,8,16}, ordinary bottom-right
causal attention, Q64–1024 and KV2048–16384, without shape overrides. Other
legal split records remain explicit tuning choices.
