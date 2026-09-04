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
| Head-dim envelope (zero-padded below native) | **none — runs the d128 kernel**⁷ | f16 ×8 · fp8 ×16 · mxfp8 exact | f16 ×8 · fp8 ×16 · mxfp8 exact | f16 ×8 | f16 ×8 · fp8 ×16, floor 256² | f16 (256, 512] ×8ᵇ · mxfp8 exact 256ᵍ |
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
| GQA / MQA (`H_q ≠ H_kv`) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ᵇ ᶠ ᵍ · dense only under THDʰ |
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
Its remaining conjunctions are declined, each with a reject test: **GQA** (the dK/dV
partials would have to be packed per Q head), a non-BSHD-physical layout (the packed
path has no staging copy), and a graph that does not declare
**`max_total_seq_len_q`/`_kv`** — those are REQUIRED here, because
`scratch_workspace_bytes()` is a build-time function and the blocked row count
comes from the packed totals before any buffer exists.
ʲ Not a gap in this engine: `cu_seq_len_q/kv` is a **forward-only** graph
attribute. `SDPA_backward_attributes` has no such input port and
`pygraph.sdpa_backward()` no such keyword, so no backward row can claim it and
none could be tested. Ragged backward lengths arrive as per-batch `seq_len_q/kv`.
⁹ `thd_d_shapes` is an **exact** membership test, not an envelope: the
quantized rows list `{(128,128), (512,512)}` (per-tensor) / `{(128,128)}`
(MXFP8), so d=64 **THD on FP8/MXFP8 is declined**. f16/bf16 THD rides the
envelope (`thd_d_shapes=None`) and works.
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

Engine: `sdpa_fwd_prefill_sm107_fp8` only. **No f16/bf16 and no MXFP8 forward,
no backward** on the Rubin line — those graphs fall through to the backend.

| Feature | d64 (GPT-OSS)<br>FPROP | d128 (Llama)<br>FPROP | BPROP<br>no engine |
|---|:--:|:--:|:--:|
| FP16 / BF16 | ❌ | ❌ | ❌ |
| FP8 E4M3 / E5M2 (per-tensor) | ⚠️ⁱ | ✅ | ❌ |
| MXFP8 | ❌ | ❌ | ❌ |
| O dtype ≠ QKV — **FP8 graphs only** (fp16/bf16/fp8 out) | ✅ | ✅ | — |
| Head-dim envelope | none — runs the d128 kernelⁱ | ×16 up to 128 | — |
| BSHD / `dense_flex` | ✅ / ❌ | ✅ / ❌ | ❌ |
| THD + `cu_seq_len` | ❌ⁱⁱ | ✅ | ❌ |
| Causal · bottom-right · right-band · SWA | ✅ | ✅ | ❌ |
| Padding mask (+ stats) | ✅ | ✅ | ❌ |
| Dense padded-Q trim | ❌ | ❌ | ❌ |
| Attention sink | ✅ | ✅ | ❌ |
| Bias | ❌ | ❌ | ❌ |
| FP16 softmax accumulate (`softmax_precision=HALF`) | ❔ⁱⁱⁱ | ✅ (Rubin-only f16x2 arm) | — |

ⁱ Same story as SM100 footnote ⁷: no native d=64 Rubin kernel, so a d=64 FP8
graph rides the d128 envelope (64 is a multiple of 16) at ~2× the MMA cost.
ⁱⁱ `thd_d_shapes={(128,128)}` is exact — d=64 THD is declined.
ⁱⁱⁱ **Accepted, not validated.** The `softmax_precision=HALF` domain is gated on
`flavor == (128, 128)` (`fwd/api_dsl.py:1100`), and a d=64 graph's *flavor* IS
(128,128), so the knob passes the probe and the kernel runs. What is untested is
the f16x2 exponent arm over the zero-padded 64 → 128 region. Validate before
relying on it; do not read ❔ as either a guarantee or a rejection.

---

## SM120 / SM121 (Blackwell GeForce, cc 12.0–12.9)

Engines: `sdpa_fwd_prefill_sm120`, `sdpa_fwd_prefill_sm120_fp8`,
`sdpa_bwd_sm120`. Head dims are a **continuum**, not per-model flavors: the
kernel picks Q/K and V head tiles independently.

| Feature | FPROP<br>d ≤ 256, any ×8 | FPROP FP8<br>d ≤ 256, any ×16 | BPROP<br>d ≤ 256, any ×8 |
|---|:--:|:--:|:--:|
| **Data types** | | | |
| FP16 / BF16 | ✅ | — | ✅ |
| FP8 E4M3 / E5M2 (per-tensor) | — | ✅ | ❌ |
| MXFP8 | ❌ | ❌ | ❌ |
| O dtype ≠ QKV — **FP8 graphs only** (fp16/bf16/fp8 out) | ❌ | ✅ | — |
| Rectangular head dims (D_QK ≠ D_V) | ✅ (independent) | ✅ (independent) | ✅ (D_QK ≥ D_V) |
| Head-dim alignment (actual `D_QK`/`D_V`) | ×8, ≤ 256ᵃ | ×16, ≤ 256ᵃ | ×8, ≤ 256 |
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
headers quote the head-dim rule.** `SUPPORTED_HEAD_TILES` steps by 16 (f16) /
32 (FP8), but those are the kernel's native tile sizes; an actual `D_QK`/`D_V`
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
| Backward THD + GQA | SM100, SM103 |
| Backward sink / dSink, bias / dBias | SM100, SM103 |
| Backward deterministic, decode | SM100, SM103 — served by the MXFP8 d=256 row only |
| MXFP8 backward: E5M2, bottom-right / band-widened / sliding-window masks, non-BSHD strides, `amax_*` outputs | SM100, SM103 |
| f16/bf16 forward | SM107 (Rubin) |
| MXFP8 forward | SM107, SM120, SM80 |
| Per-tensor FP8 backward | every arch |
| MXFP8 backward outside SM100/SM103 d = 256 | every arch |
| THD / ragged backward | SM80, SM120, and the SM100/SM103 MXFP8 row (the SM100/SM103 f16/bf16 row serves it — see ʰ) |
| THD forward | SM80 |
| **Native d=64 (GPT-OSS) forward kernel** | **SM100, SM107** — served via the d128 envelope at ~2× MMA cost |
| d=64 MXFP8 / d=64 quantized THD | SM100, SM107 (exact-shape gates) |
| Bias forward | SM100, SM107, SM120 |
| Dropout, ALiBi, paged KV, `block_mask`, `score_mod` | every arch, both passes |
