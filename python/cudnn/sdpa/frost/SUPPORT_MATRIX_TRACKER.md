# FROST SDPA — support matrix

What the shipped FROST SDPA engines actually serve, one table per architecture.
Columns are the kernel **flavors** (native head-dim geometry, with the model
class it was tuned for in brackets) crossed with the pass; rows are features.

Source of truth is the `Capabilities` row of each engine
(`python/cudnn/sdpa/fwd/engines.py`, `python/cudnn/sdpa/bwd/engines.py`) — a
cell here is ✅ only when that row admits it. Anything not listed as a row
(dropout, ALiBi, paged KV, `block_mask`, `score_mod`, `rng_dump`,
`score_max`/`score_sum_exp`, tensor `attn_scale`, `unfuse_fma`, `Amax_S`) is
**declined by every FROST SDPA engine on every arch**.

All FROST engines are `opt_in=True`: set `CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1`
before `import cudnn` or the graph silently runs a cuDNN backend plan.

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

The backward is a **three-stage chain**, not one fused kernel: a fused d=512
backward needs 512 TMEM columns for dV and 512 more for dK against 512 per CTA,
so S and dS go to a GMEM workspace and the gradients are three batched GEMMs
over it (`do_dot` → `bprop_d512_f16_sm100` → `bprop_matmul_sm100`). Two
consequences a user can see: the workspace is `2·B·H_chunk·S_q·S_kv·2 B` (the
host loops over head chunks to hold it under 4 GiB; under THD it is
`2·H_chunk·(T_q + B·256)·pad(S_kv_max)·2 B` instead — see ʰ), and everything in the band
is **envelope-served** — the tiles are fixed at 512, so d=264 costs the same
MMA as d=512.

| Feature | d64 (GPT-OSS)<br>FPROP | d128 (Llama)<br>FPROP | d192×d128 (DSv3 MLA)<br>FPROP | d256 (Qwen)<br>FPROP | d512 (DSv4)<br>FPROP | d512 (DSv4)<br>BPROP<br>d ∈ (256, 512] |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| **Data types** | | | | | | |
| FP16 / BF16 | ⚠️⁷ | ✅ | ✅ | ✅ | ✅ | ✅ᵇ |
| FP8 E4M3 / E5M2 (per-tensor descale) | ⚠️⁷ | ✅ | ✅ | ❌ | ✅ | ❌ |
| MXFP8 (E4M3/E5M2 + per-32 E8M0 SF) | ❌⁸ | ✅ | ✅ | ❌ | ❌ | ✅ᵍ (E4M3 only, d=256) |
| O dtype ≠ QKV dtype — **quantized graphs only**¹ | ✅ | ✅ | ✅ | — | ✅ | ✅ᵍ (fp16/bf16 gradients) |
| Head-dim envelope (zero-padded below native) | **none — runs the d128 kernel**⁷ | f16 ×8 · fp8 ×16 · mxfp8 exact | f16 ×8 · **fp8 exact (192, 128) only**¹⁰ · mxfp8 exact | f16 ×8 · **fp8 exact 256 only**¹⁰ | f16 ×8 · fp8 ×16, floor 256² | f16 (256, 512] ×8ᵇ · mxfp8 exact 256ᵍ |
| **Layout** | | | | | | |
| BSHD | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ᵇ ᵍ |
| Arbitrary dense B/H/S stride order (`dense_flex`) | f16 only | f16 only | f16 only | ✅ | f16 only | ✅ᵇ ᶜ · ❌ᵍ |
| THD / ragged (packed varlen) | f16 only⁹ | ✅ | f16 only³ | ✅ | f16 + fp8³ | ✅ᵇ ʰ · ❌ᵍ |
| `cu_seq_len_q/kv` prefix sums (THD only) | f16 only⁹ | ✅ | ✅ | ✅ | ✅ | ❌ʲ |
| **Masks / features** | | | | | | |
| Causal (top-left) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ᵇ ᵈ ᵍ |
| Causal bottom-right | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ᵇ ᵈ · ❌ᵍ |
| Causal right-band widening | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ᵇ · ❌ᵍ |
| Sliding window (left) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ᵇ · ❌ᵍ |
| Padding mask (`seq_len_q/kv`) | ✅ | ✅ | ✅ | ✅ | ✅ | THD onlyᵇ ʰ · ❌ᵍ |
| THD + causal family (top-left / bottom-right / SWA / band) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ᵇ ʰ · ❌ᵍ |
| Padding mask + stats (per-batch LSE trim) | ✅ | f16/fp8 only⁴ | f16/fp8 only⁴ | ✅ | f16/fp8 only⁴ | ❌ |
| Dense padded-Q trim (O:=0, LSE:=−inf) | f16 only⁵ | f16 only⁵ | f16 only⁵ | ✅ | f16 only⁵ | ❌ |
| Attention sink | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| GQA / MQA (`H_q ≠ H_kv`) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ᵇ ᶠ ᵍ ʰ |
| Bias / dBias | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `use_deterministic_algorithm` | — | — | — | — | — | ❌ᵇ · ✅ᵍ |
| Ragged `S_kv` (non-multiple of 128) | ✅⁶ | ✅⁶ | ✅⁶ | ✅⁶ | ✅⁶ | ✅ᵇ ᵉ ᵍ |
| Decode-shaped (`S_q == 1`) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ᵇ · ✅ᵍ |

¹ **Reads as: on a quantized (fp8/mxfp8) graph in this column, O may be FP16,
BF16, E4M3 or E5M2.** It does NOT mean an f16/bf16 graph may convert O — the f16
row has no `out_dtypes` domain and `facts.uniform_dtype` requires O == Q there.
`—` marks a column with no quantized kernel at all.
² The d512 FP8 flavor serves head dims in (256, 512] on both axes; a smaller
graph is declined rather than routed onto it at >2× zero-padding cost.
³ The d192×d128 fp8/mxfp8 kernels are dense-only; d512 has no MXFP8 kernel.
⁴ MXFP8 lacks the `SEQ_Q_LENS_PRESENT` epilogue trim (`padded_stats=False`).
⁵ FP8 and MXFP8 rows are not plumbed for the dense padded-Q trim.
⁶ Served through the padded path with synthesized full-length KV lengths, or
natively when the causal band covers the KV tail.
⁷ **No d=64 kernel exists on SM100.** `_SM100_FLAVORS` is
`((128,128), (192,128), (256,256), (512,512))` (`fwd/api_dsl.py:56`), so
`_pick_flavor(64, 64)` returns `(128, 128)` and the graph runs the **d128 Llama
kernel** with the contraction zero-padded 64 → 128. Numerically exact (the pad
columns load as hard zeros, so S and the softmax are unchanged) but it burns
~2× the MMA work of a native d=64 kernel on both BMM1 and BMM2. GPT-OSS-class
d=64 is native on **SM80** (`gptoss` flavor) and on **SM120** (64 is a
supported head tile) — SM100 is the gap.
⁸ MXFP8 sets `d_pad_multiple=0` (exact native shapes only — the scale-factor
plumbing is not audited for envelope zero-padding), so a d=64 MXFP8 graph is
declined at plan time rather than padded.
ᵇ **`sdpa_bwd_sm100` — the d512 (DSv4) backward, and the ONLY FROST backward on
this arch: d ∈ (256, 512], multiples of 8, f16/bf16.** It is one engine over one
head-dim band, not a per-flavor column, so it does not follow the FPROP columns
to its left: below 257 the d256 flavors are the right kernel and the engine
declines, and every other `sdpa_backward()` shape on SM100 falls through to the
cuDNN backend. The whole band is **envelope-served** on 512-wide tiles, so d=264
pays d=512's MMA cost. Multiples of 8 rather than the forward's 16: the stage-3
epilogue narrows its store vector from 32 B to 16 B when d is not also a
multiple of 16.
ᶜ A non-BSHD io tensor is staged through the workspace (one copy in, and one
back out for a gradient); a BSHD-physical one is used in place. This is not
hypothetical — building dO as `torch.randn(o.shape)` instead of
`torch.empty_like(o)` loses o's memory format and yields a BHSD-contiguous dO.
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
quantized rows list `{(128,128), (512,512)}` (per-tensor) / `{(128,128)}`
(MXFP8), so d=64 **THD on FP8/MXFP8 is declined**. f16/bf16 THD rides the
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

Engines: `sdpa_fwd_prefill_sm107` (f16/bf16), `sdpa_fwd_prefill_sm107_fp8`
(per-tensor FP8) and `sdpa_fwd_prefill_sm107_mxfp8` (block-scale). **No
backward** on the Rubin line — those graphs fall through to the backend.

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

| Feature | d64 (GPT-OSS)<br>FPROP | d128 (Llama)<br>FPROP | d192×d128 (DSv3 MLA)<br>FPROP | d256 (Qwen)<br>FPROP | d512 (DSv4)<br>FPROP | BPROP<br>no engine |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| **Data types** | | |  | | | |
| FP16 / BF16 | ⚠️ⁱ | ✅ | ✅ | ✅ | ✅ | ❌ |
| FP8 E4M3 / E5M2 (per-tensor) | ⚠️ⁱ | ✅ | ✅ | ✅ | ✅ | ❌ |
| MXFP8 | ❌ | ✅ | ✅ˣ | ✅ | ⚠️ⁱᵛ | ❌ |
| O dtype ≠ QKV — **quantized graphs only** (fp16/bf16/fp8 out) | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| Head-dim envelope | none — runs the d128 kernelⁱ | f16 ×8 · fp8 ×16 | f16 ×8 · fp8 exact only (floor 128) | f16 ×8 · fp8 ×16 (floor 255) | f16 ×8 · fp8 ×16 (floor 256) | — |
| **Layout** | | |  | | | |
| BSHD | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Arbitrary dense stride order (`dense_flex`) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| THD / ragged (packed varlen) | ✅ᶻ | ✅ᶻ | ✅ᶻ | ✅ᶻ | ✅ᶻ | ❌ |
| `cu_seq_len_q/kv` prefix sums (THD only) | ✅ᶻ | ✅ᶻ | ✅ᶻ | ✅ᶻ | ✅ᶻ | ❌ |
| **Masks / features** | | |  | | | |
| Causal (top-left) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Causal bottom-right | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Causal right-band widening | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Sliding window (left) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Padding mask (`seq_len_kv`) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Padding mask + stats (per-batch LSE trim) | ✅ | fp8 onlyᵛⁱ | fp8 onlyᵛⁱ | ❌ᵛⁱ | ❌ᵛⁱ | ❌ |
| Dense padded-Q trim (O:=0, LSE:=−inf) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Attention sink | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| GQA / MQA (`H_q ≠ H_kv`) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| PackGQA | fp8 only | fp8 only | ❌ | ❌ | ❌ | ❌ |
| Split-KV | fp8 only | fp8 only | ❌ᵛⁱⁱ | ❌ᵛⁱⁱ | ❌ᵛⁱⁱ | ❌ |
| Optional stats (LSE store compiled out) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Bias | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Ragged `S_kv` (non-multiple of 128) | ✅ⁱˣ | ✅ⁱˣ | ✅ⁱˣ | ✅ⁱˣ | ✅ⁱˣ | ❌ |
| FP16 softmax accumulate (`softmax_precision=HALF`) | ❔ⁱⁱⁱ | fp8 only (Rubin f16x2 arm) | fp8 only (same body as d128) | ❌ | ❌ | — |

ⁱ No native d=64 Rubin kernel, so a d=64 graph rides the d128 envelope (64 is a
multiple of 8 at f16 and of 16 at fp8) at ~2× the MMA cost.
ⁱⁱ `thd_d_shapes={(128,128)}` on the FP8 row is exact — d=64 THD is declined.
ⁱⁱⁱ **Accepted, not validated.** `softmax_precision=HALF` is gated on
`flavor == (128, 128)` (`fwd/api_dsl.py`), and a d=64 graph's *flavor* IS
(128,128), so the knob passes the probe and the kernel runs. Untested is the
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
ᵛⁱ Needs the per-batch `seq_len_q` LSE trim, which the f16 Rubin kernels do not
carry (`padded_stats=False`). KV-side padding itself is served.
ᵛⁱⁱ The f16 Rubin kernels wire no SplitHelpers.

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
Two FP8 flavors stay out: **(128, 128)** is also bit-identical under LPT, but
its causal path reads 0.041–0.048 against the suite's 0.04 under NATURAL as
well, so it gets its own look before it is claimed; **(512, 512)** — the cga4×1
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

Still declined, and why: d128/d512 f16, d128 FP8 (tolerance, above) and every
MXFP8 flavor are **unvalidated** under LPT rather than known-incorrect; d512
(f16, FP8, MXFP8) **does not produce output** under LPT until the d512 kernels get the
`lpt_q_tiles_in_cga_units` argument and are re-validated (cga4×1 role-split, a
different scheduler shape — `prefill_d512_fp8.py:2533` notes the LPT range is
`q_clusters * CTA_MMA`). `SCHED_LPT_L2` is declined by **every** flavor —
its decode needs `qh_per_kh` and `seqlen_kv`, which the SM107 call sites do not
pass, so it raises rather than miscomputes. Both are follow-ups.

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

## SM120 / SM121 (Blackwell GeForce, cc 12.0–12.9)

Engines: `sdpa_fwd_prefill_sm120`, `sdpa_fwd_prefill_sm120_fp8`,
`sdpa_bwd_sm120`. Head dims are a **continuum**, not per-model flavors: the
kernel picks Q/K and V head tiles independently (f16/bf16 head dims it would
tile at 256 on both sides run a dedicated template, `sm120/prefill_d256_f16.py` with the same support; fp8 has no such flavor).

D512 FP16/BF16 FPROP (`sm120/prefill_d512_f16.py`) supports **both D_QK and D_V
in (256, 512]**, in multiples of 8. It uses a 64 x 32 CTA and 512 x 512 head
tiles: (512, 512) is **native**; smaller shapes are **envelope-served** by
zero-padding, with the same MMA and on-chip storage cost.
FP8 FPROP and BPROP remain limited to 256.

| Feature | FPROP<br>d ≤ 256, any ×8;<br>both dims ∈ (256, 512], any ×8 | FPROP FP8<br>d ≤ 256, any ×16 | BPROP<br>d ≤ 256, any ×8 |
|---|:--:|:--:|:--:|
| **Data types** | | | |
| FP16 / BF16 | ✅ | — | ✅ |
| FP8 E4M3 / E5M2 (per-tensor) | — | ✅ | ❌ |
| MXFP8 | ❌ | ❌ | ❌ |
| O dtype ≠ QKV — **FP8 graphs only** (fp16/bf16/fp8 out) | ❌ | ✅ | — |
| Rectangular head dims (D_QK ≠ D_V) | ✅ (independent within each served band) | ✅ (independent) | ✅ (D_QK ≥ D_V) |
| Head-dim alignment (actual `D_QK`/`D_V`) | ×8, both ≤ 256ᵃ; or both ∈ (256, 512] | ×16, ≤ 256ᵃ | ×8, ≤ 256 |
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
| Attention sink / dSink | ✅ | ✅ | ✅ / ✅ |
| Bias / dBias | ❌ | ❌ | ✅ / ✅ |
| GQA / MQA (`H_q ≠ H_kv`) | ✅ | ✅ | ✅ |
| Deterministic (`use_deterministic_algorithm`) | — | — | ✅ |
| Ragged `S_kv` (no tile rule) | ✅ | ✅ | ✅ |
| Decode-shaped (`S_q == 1`) | ✅ | ✅ | ✅ |

ᵃ **Head TILE granule and head-DIM alignment are different numbers — the column
headers quote the head-dim rule.** `GENERAL_HEAD_TILES` steps by 16 (f16), and
`SUPPORTED_HEAD_TILES_FP8` steps by 32. Those are the kernel's native tile sizes; an actual `D_QK`/`D_V`
only has to satisfy `d_pad_multiple` — **8** on the f16 row, **16** on the FP8
row (the TMA 16-byte global-stride rule at 2 and 1 bytes/elem) — and is
zero-padded up to the next tile. So a d=72 f16 graph is eligible and computes
on the 80-wide tile.

---

## SM80 (Ampere A100, cc 8.0 exactly)

Engines: `sdpa_fwd_prefill_sm80`, `sdpa_bwd_sm80`. Both use `mma.sync` (no
tcgen05) and assume the A100's 164 KiB opt-in SMEM — sm86/sm89 are declined.
Head dims below a flavor's native shape are zero-padded **host-side**, so there
is no alignment rule.

| Feature | d64 (GPT-OSS) | d128 (Llama) | d192×d128 (DSv3) | d256 (Qwen) |
|---|:--:|:--:|:--:|:--:|
| | F / B | F / B | F / B | F / B |
| **Data types** | | | | |
| FP16 / BF16 | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| FP8 / MXFP8 | ❌ / ❌ | ❌ / ❌ | ❌ / ❌ | ❌ / ❌ |
| **Layout** | | | | |
| BSHD / `dense_flex` | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| THD / ragged | ❌ / ❌ | ❌ / ❌ | ❌ / ❌ | ❌ / ❌ |
| `cu_seq_len_q/kv` | ❌ / ❌ | ❌ / ❌ | ❌ / ❌ | ❌ / ❌ |
| Strided / permuted Stats | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| **Masks / features** | | | | |
| Causal (top-left) | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| Causal bottom-right | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| Causal right-band widening | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| Sliding window (left) | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| Padding mask (+ stats, + padded-Q trim) | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| Attention sink / dSink | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| Bias / dBias | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| GQA / MQA | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| Deterministic | — / ✅ | — / ✅ | — / ✅ | — / ✅ |
| Ragged `S_kv` | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| Decode-shaped (`S_q == 1`) | ❌ / ❌ | ❌ / ❌ | ❌ / ❌ | ❌ / ❌ |

The SM80 backward additionally has a dedicated plain-dense **d=64 fast path**
(~2× on A100) that supports **no** features — it is selected only for a
feature-free d=64 graph.

---

## Gaps at a glance

| Missing | Where |
|---|---|
| Backward pass entirely | SM107 |
| Backward outside d ∈ (256, 512] (f16/bf16) or d = 256 (MXFP8) | SM100, SM103 — the two backward engines there serve exactly those bands |
| Backward per-batch padding mask (`seq_len_q/kv`) on a DENSE graph | SM100, SM103 — a UNIFORM non-tile-multiple length is served, and the THD path carries per-sequence lengths; a per-batch mask on a dense graph is not |
| Backward sink / dSink, bias / dBias | SM100, SM103 |
| Backward deterministic, decode | SM100, SM103 — served by the MXFP8 d=256 row only |
| MXFP8 backward: E5M2, bottom-right / band-widened / sliding-window masks, non-BSHD strides, `amax_*` outputs | SM100, SM103 |
| f16/bf16 forward split-KV, PackGQA, dense padded-Q trim | SM107 (Rubin) — the row serves dense f16/bf16 at d128/d192×d128/d256/d512, and THD on all of them as of 2026-09-09; these three are the machinery its kernels still lack (optional stats IS served — `lse_optional=True`) |
| d192×d128 quantized PackGQA / split-KV, and d192 MXFP8 THD | SM107 — the shape is served in FP8 and MXFP8 as of 2026-09-09, and per-tensor FP8 **THD** with it; PackGQA and split-KV stay wired in the d128 flavor only (`pack_gqa_d_shapes` / `split_d_shapes`), and the MXFP8 line declines THD row-wide |
| MXFP8 forward | SM120, SM80 (SM107 is served — see the SM107 table; d512 is ⚠️ⁱᵛ, correct but with no test module) |
| Per-tensor FP8 backward | every arch |
| MXFP8 backward outside SM100/SM103 d = 256 | every arch |
| THD / ragged backward | SM80, SM120, and the SM100/SM103 MXFP8 row (the SM100/SM103 f16/bf16 row serves it — see ʰ) |
| THD forward | SM80 |
| **Native d=64 (GPT-OSS) forward kernel** | **SM100, SM107** — served via the d128 envelope at ~2× MMA cost |
| d=64 MXFP8 / d=64 quantized THD | SM100, SM107 (exact-shape gates) |
| Bias forward | SM100, SM107, SM120 |
| Dropout, ALiBi, paged KV, `block_mask`, `score_mod` | every arch, both passes |
