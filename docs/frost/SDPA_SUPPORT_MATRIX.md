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
> `python/cudnn/frost/README.md` § "The rules" #14. This table is maintained by
> hand; nothing else catches it going stale.

Legend: ✅ served natively · ⚠️ served, but **no native kernel for this head dim** —
the graph rides another flavor's envelope with TMA zero-padding (correct, but pays
the larger flavor's MMA cost) · ❌ declined at plan time · — not applicable · ⁿ footnote.

---

## SM100 / SM103 (Blackwell, cc 10.0–10.6)

Engines: `sdpa_fwd_prefill_sm100` (f16/bf16), `sdpa_fwd_prefill_sm100_fp8`,
`sdpa_fwd_prefill_sm100_mxfp8`. **No backward engine on SM100** — an
`sdpa_backward()` graph falls through to the cuDNN backend.

| Feature | d64 (GPT-OSS)<br>FPROP | d128 (Llama)<br>FPROP | d192×d128 (DSv3 MLA)<br>FPROP | d256 (Qwen)<br>FPROP | d512 (DSv4)<br>FPROP | BPROP |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| **Data types** |
| FP16 / BF16 | ⚠️⁷ | ✅ | ✅ | ✅ | ✅ | ❌ |
| FP8 E4M3 / E5M2 (per-tensor descale) | ⚠️⁷ | ✅ | ✅ | ❌ | ✅ | ❌ |
| MXFP8 (E4M3/E5M2 + per-32 E8M0 SF) | ❌⁸ | ✅ | ✅ | ❌ | ❌ | ❌ |
| O dtype ≠ QKV dtype (fp16/bf16/fp8 out) | ✅¹ | ✅¹ | ✅¹ | — | ✅¹ | — |
| Head-dim envelope (zero-padded below native) | **none — runs the d128 kernel**⁷ | f16 ×8 · fp8 ×16 · mxfp8 exact | f16 ×8 · fp8 ×16 · mxfp8 exact | f16 ×8 | f16 ×8 · fp8 ×16, floor 256² | — |
| **Layout** |
| BSHD | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Arbitrary dense B/H/S stride order (`dense_flex`) | f16 only | f16 only | f16 only | ✅ | f16 only | ❌ |
| THD / ragged (packed varlen) | f16 only⁹ | ✅ | f16 only³ | ✅ | f16 + fp8³ | ❌ |
| `cu_seq_len_q/kv` prefix sums (THD only) | f16 only⁹ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Masks / features** |
| Causal (top-left) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Causal bottom-right | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Causal right-band widening | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Sliding window (left) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Padding mask (`seq_len_q/kv`) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Padding mask + stats (per-batch LSE trim) | ✅ | f16/fp8 only⁴ | f16/fp8 only⁴ | ✅ | f16/fp8 only⁴ | ❌ |
| Dense padded-Q trim (O:=0, LSE:=−inf) | f16 only⁵ | f16 only⁵ | f16 only⁵ | ✅ | f16 only⁵ | ❌ |
| Attention sink | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Bias / dBias | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Ragged `S_kv` (non-multiple of 128) | ✅⁶ | ✅⁶ | ✅⁶ | ✅⁶ | ✅⁶ | ❌ |
| Decode-shaped (`S_q == 1`) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |

¹ Quantized rows only (fp8/mxfp8): O may be FP16, BF16, E4M3 or E5M2. On the f16
row O must equal Q.
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
⁹ `thd_d_shapes` is an **exact** membership test, not an envelope: the
quantized rows list `{(128,128), (512,512)}` (per-tensor) / `{(128,128)}`
(MXFP8), so d=64 **THD on FP8/MXFP8 is declined**. f16/bf16 THD rides the
envelope (`thd_d_shapes=None`) and works.

---

## SM107 (Rubin, cc 10.7–11.9)

Engine: `sdpa_fwd_prefill_sm107_fp8` only. **No f16/bf16 and no MXFP8 forward,
no backward** on the Rubin line — those graphs fall through to the backend.

| Feature | d64 (GPT-OSS)<br>FPROP | d128 (Llama)<br>FPROP | BPROP |
|---|:--:|:--:|:--:|
| FP16 / BF16 | ❌ | ❌ | ❌ |
| FP8 E4M3 / E5M2 (per-tensor) | ⚠️ⁱ | ✅ | ❌ |
| MXFP8 | ❌ | ❌ | ❌ |
| O dtype ≠ QKV (fp16/bf16/fp8 out) | ✅ | ✅ | — |
| Head-dim envelope | none — runs the d128 kernelⁱ | ×16 up to 128 | — |
| BSHD / `dense_flex` | ✅ / ❌ | ✅ / ❌ | ❌ |
| THD + `cu_seq_len` | ❌ⁱⁱ | ✅ | ❌ |
| Causal · bottom-right · right-band · SWA | ✅ | ✅ | ❌ |
| Padding mask (+ stats) | ✅ | ✅ | ❌ |
| Dense padded-Q trim | ❌ | ❌ | ❌ |
| Attention sink | ✅ | ✅ | ❌ |
| Bias | ❌ | ❌ | ❌ |
| FP16 softmax accumulate (`softmax_precision=HALF`) | ❌ⁱⁱⁱ | ✅ (Rubin-only f16x2 arm) | — |

ⁱ Same story as SM100 footnote ⁷: no native d=64 Rubin kernel, so a d=64 FP8
graph rides the d128 envelope (64 is a multiple of 16) at ~2× the MMA cost.
ⁱⁱ `thd_d_shapes={(128,128)}` is exact — d=64 THD is declined.
ⁱⁱⁱ The `softmax_precision=HALF` domain is gated on `flavor == (128, 128)`
(`fwd/api_dsl.py:1100`), and a d=64 graph's *flavor* is (128,128) — so the knob
is in fact accepted. Marked ❌ here only because the padded d=64 path has not
been validated with the f16x2 arm; treat as untested rather than rejected.

---

## SM120 / SM121 (Blackwell GeForce, cc 12.0–12.9)

Engines: `sdpa_fwd_prefill_sm120`, `sdpa_fwd_prefill_sm120_fp8`,
`sdpa_bwd_sm120`. Head dims are a **continuum**, not per-model flavors: the
kernel picks Q/K and V head tiles independently.

| Feature | FPROP<br>d ≤ 256, any ×16 | FPROP FP8<br>d ≤ 256, any ×32 tile | BPROP<br>d ≤ 256, any ×8 |
|---|:--:|:--:|:--:|
| **Data types** |
| FP16 / BF16 | ✅ | — | ✅ |
| FP8 E4M3 / E5M2 (per-tensor) | — | ✅ | ❌ |
| MXFP8 | ❌ | ❌ | ❌ |
| O dtype ≠ QKV (fp16/bf16/fp8 out) | ❌ | ✅ | — |
| Rectangular head dims (D_QK ≠ D_V) | ✅ (independent) | ✅ (independent) | ✅ (D_QK ≥ D_V) |
| Head-dim alignment | ×8, ≤ 256 | ×16, tile ×32, ≤ 256 | ×8, ≤ 256 |
| **Layout** |
| BSHD / `dense_flex` | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| THD / ragged | ✅ | ✅ | ❌ |
| `cu_seq_len_q/kv` | ✅ | ❌ | ❌ |
| Strided / permuted Stats | ✅ | ✅ | ✅ |
| **Masks / features** |
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

---

## SM80 (Ampere A100, cc 8.0 exactly)

Engines: `sdpa_fwd_prefill_sm80`, `sdpa_bwd_sm80`. Both use `mma.sync` (no
tcgen05) and assume the A100's 164 KiB opt-in SMEM — sm86/sm89 are declined.
Head dims below a flavor's native shape are zero-padded **host-side**, so there
is no alignment rule.

| Feature | d64 (GPT-OSS) | d128 (Llama) | d192×d128 (DSv3) | d256 (Qwen) |
|---|:--:|:--:|:--:|:--:|
| | F / B | F / B | F / B | F / B |
| **Data types** |
| FP16 / BF16 | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| FP8 / MXFP8 | ❌ / ❌ | ❌ / ❌ | ❌ / ❌ | ❌ / ❌ |
| **Layout** |
| BSHD / `dense_flex` | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| THD / ragged | ❌ / ❌ | ❌ / ❌ | ❌ / ❌ | ❌ / ❌ |
| `cu_seq_len_q/kv` | ❌ / ❌ | ❌ / ❌ | ❌ / ❌ | ❌ / ❌ |
| Strided / permuted Stats | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| **Masks / features** |
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
| Backward pass entirely | SM100, SM103, SM107 |
| f16/bf16 forward | SM107 (Rubin) |
| MXFP8 forward | SM107, SM120, SM80 |
| FP8 / MXFP8 backward | every arch |
| THD / ragged backward | every arch |
| THD forward | SM80 |
| **Native d=64 (GPT-OSS) forward kernel** | **SM100, SM107** — served via the d128 envelope at ~2× MMA cost |
| d=64 MXFP8 / d=64 quantized THD | SM100, SM107 (exact-shape gates) |
| Bias forward | SM100, SM107, SM120 |
| Dropout, ALiBi, paged KV, `block_mask`, `score_mod` | every arch, both passes |
