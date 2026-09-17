# FROST SDPA — support matrix

What the shipped FROST SDPA engines actually serve, one table per architecture.
Columns are the kernel **flavors** (native head-dim geometry, with the model
class it was tuned for in brackets) crossed with the pass; rows are features.

Source of truth is the `Capabilities` row of each engine
(`python/cudnn/sdpa/fwd/engines.py`, `python/cudnn/sdpa/bwd/engines.py`) — a
cell here is ✅ only when that row admits it. Anything not listed as a row
(dropout, ALiBi, `block_mask`, `score_mod`, `rng_dump`,
`score_max`/`score_sum_exp`, tensor `attn_scale`, `unfuse_fma`, `Amax_S`) is
**declined by every FROST SDPA engine on every arch**.

**Base-2 stats (`stats_use_log2`)** are served natively by every FROST forward
engine: the request is a plan-time epilogue specialization (natural-log LSE
scaled by log2(e) right before the store; -inf rows stay -inf). It is a
convention gate rather than a kernel feature — serving a base-2 request with
natural-log values would be correct O plus silently wrong Stats, so the
capability row must stay in step with the kernels. Under split-KV the per-split
partial LSEs stay natural (the combine merges them in that base) and only the
combine kernel's final LSE converts. Backward engines consume natural-log Stats
only (the graph attribute is forward-only).

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
| THD / ragged (packed varlen)ᵏ | f16 only⁹ | ✅ | f16 only³ | ✅ | f16 + fp8³ | ✅ᵇ ʰ · ❌ᵍ |
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
| Attention sink | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Base-2 stats (`stats_use_log2`) | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| GQA / MQA (`H_q ≠ H_kv`) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ᵇ ᶠ ᵍ ʰ |
| PackGQA (`PACK_GQA` knob: the GQA group packed into the Q tile)ᵐ | ✅ᵐ partial (d128 envelope) | ✅ᵐ partial | whole group onlyᵐ | ✅ᵐ partial | whole group onlyᵐ | — |
| Bias / dBias | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `use_deterministic_algorithm` | — | — | — | — | — | ❌ᵇ · ✅ᵍ |
| Ragged `S_kv` (non-multiple of 128) | ✅⁶ | ✅⁶ | ✅⁶ | ✅⁶ | ✅⁶ | ✅ᵇ ᵉ ᵍ |
| Decode-shaped (`S_q == 1`) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ᵇ · ✅ᵍ |
| **Decode tile** (`S_q · PACK_G ≤ 128`, decode + MTP; f16/bf16, dense or paged)ᵈᵗ | ✅ᵈᵗ (d128 envelope) | ✅ᵈᵗ **native** | ❌ (prefill tile) | ❌ (prefill tile) | ❌ (prefill tile) | — |
| Paged KV cache (`paged_attention_k/v_table` + padding mask)ᵖ | ✅ᵖ (d128 envelope) | ✅ᵖ | ❌ | ✅ᵖ | ✅ᵖ (native 512; (256, 512] envelope) | ❌ |
| Fused epilogue gate (sdpa virtual `O_v` → `mul(O_v, sigmoid(G))`; the SM107 rows serve it, see the SM107 table) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

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
bottom-right / SWA band, dense padded-Q trim, sink (dense; paged + sink stays declined
by ᵖ) — a keyless row with a sink (above the bottom-right diagonal, or `seq_len_kv[b] ==
0`) writes `O = 0`, `LSE = sink` whatever the sink's magnitude: the fold's operands are
selected for such a row, as the prefill kernels do since PR #1095 (`exp(sink − max)`
underflows in fp32 for sink ≤ −104, which used to give `O = NaN`, `LSE = −inf`;
`test_decode_kernel_keyless_rows_sink_magnitude` at −120 / −5 / +3, Stats on / base-2 /
off, and its graph-path twin pin it) — Stats incl. base-2, PackGQA incl. partial
packing (ᵐ: `HEADS_PER_TILE = PACK_G`, `G / PACK_G` packed heads per KV head), KV split
+ combine, NATURAL
/ LPT / LPT_L2. **Not on the decode tile: THD** (a ragged graph keeps `TILE_CGA_M=2`; a
pinned 1 declines), fp8 / mxfp8 (no quantized decode tile: their (128, 128) flavors keep
`cgas={2}`, their other flavors their own width), and the other f16 flavors (d192x128 /
d256 / d512 have no decode tile yet — their decode graphs run the prefill kernel as
before). d64 rides it through the d128 envelope. Measured on B200 (graph path,
CUDA-graph replay, b=32, d=128, S_kv=4096, page 16, bf16): 64/4 S_q=1 119.2 us on the
prefill tile → 48.9 us on the decode tile (the cuDNN backend's decode engine: 44 us);
64/4 MTP S_q=4 127.5 → 50.6 us; 64/8 S_q=1 234.6 → 96.4 us; 96/8 S_q=1 (G=12 packs 4,
partial PackGQAᵐ on both tiles) 615 → 225 us (the same shape unpacked on the decode tile:
875 us); d64 64/8 230.5 → 80.2 us. The kernel docstring carries the full table.

ᵖ **Paged KV (issue #920), f16/bf16 only, d128, d256 and d512 flavors** (`d_qk, d_v <=
512`; d=64 rides the d128 envelope, d=192/192 the d256 one, head dims in (256, 512] —
e.g. 384/384 — the d512 one zero-padded, at the d512 kernel's MMA cost; d=512/512 is
native. Mixed dims that would select d192x128 are declined; absorbed-MLA 576/512 has no
flavor envelope at all). The graph is cuDNN's own paged-cache contract: K/V are page
pools `[num_pages, H_kv, page_size, D]` — HND compact, or NHD (`[num_pages, page_size,
H_kv, D]` storage) declared through the strides — plus `(B, 1, max_pages, 1)` int32
block tables and `use_padding_mask` with `seq_len_q` / `seq_len_kv` (the per-batch KV
length is read on device; `paged_attention_max_seq_len_kv` defaults to `max_pages *
page_size`). `page_size` is a multiple of 8 that divides the 128-row KV tile or is a
multiple of it. Any `S_q` (decode or paged prefill), GQA (PackGQAᵐ — the whole
group, or its largest divisor of the tile), Stats out, and **THD queries**: ragged Q/O (ragged offsets +
`seq_len_q`) over the same pools — chunked prefill — with the THD scheduler walking
the Q units (no KV split there). KV split is proposed on dense-Q paged graphs by the
same wave-cost model as on dense graphs (they are padded by construction: the per-batch
lengths bound the walk on device and the split composes with them; it pays when
`B * H_kv` leaves SMs idle) and recombined by `split_combine_sm100`. The declared
`paged_attention_max_seq_len_kv` only sizes that cost model — a maximum that is not a
multiple of the 128-row KV tile (FlashInfer passes its true max verbatim, e.g. 4000)
does not withhold the split, unlike a mask-free dense `S_kv`, which rides synthesized
KV-tail padding the split cannot. Not yet: sink, fp8/mxfp8 pools,
packed (ragged-offset) block tables. Served by the `PAGED_KV` specialization of
`sm100/prefill_d128_f16.py`, `sm100/prefill_d256_f16.py` and `sm100/prefill_d512_f16.py`
(block-table indirection on the K/V TMA loads; boxes past a sequence's live pages are
TMA-OOB zero-filled) and, for decode / MTP shapes on the d128 flavor (`S_q * PACK_G <=
128`), of the d128 decode tile `sm100/decode_d128_f16.py` (ᵈᵗ). On the d512 flavor the
loader is role-split across the cga4 cluster: the sub-group 0 CTAs issue the K boxes and
the sub-group 1 CTAs the V boxes, each pair walking its own block table; decode-shaped
launches (`S_q` in [1, 8], MQA / GQA) and paged prefill (`S_q` up to 128 validated) both
run, with KV split as on d128.

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
`G / PACK_G` packed heads of one KV head); d64 rides the d128 envelope. The
d192×d128 and d512 f16 kernels and every fp8 / mxfp8 kernel pack
`HEADS_PER_TILE = G` and keep the **whole-group** contract (a non-divisor
group is declined there). Validated on B200: `test_sdpa_fwd_paged_sm100.py`
(graph + kernel, G = 12 / 6 / 5 / 3, S_q 1–8 bottom-right causal, HND / NHD),
`test_sdpa_fwd_dsl_sm100.py::test_dsl_sm100_pack_gqa_partial_group` (dense,
d128 / d256, Stats), and the `test_mhas_v2.py` paged partial-pack sweeps.

¹ **Reads as: on a quantized (fp8/mxfp8) graph in this column, O may be FP16,
BF16, E4M3 or E5M2.** It does NOT mean an f16/bf16 graph may convert O — the f16
row has no `out_dtypes` domain and `facts.uniform_dtype` requires O == Q there.
`—` marks a column with no quantized kernel at all.
² The d512 FP8 flavor serves head dims in (256, 512] on both axes; a smaller
graph is declined rather than routed onto it at >2× zero-padding cost.
³ The d192×d128 fp8/mxfp8 kernels are dense-only; d512 has no MXFP8 kernel.
⁴ Every SM100 / SM103 flavor carries the `SEQ_Q_LENS_PRESENT` epilogue trim (f16, per-tensor FP8 and MXFP8 alike, #1037).
⁵ Every forward kernel trims dense padded Q natively; there is no `dense_seq_q_trim` capability any more -- a graph with per-batch Q lengths compiles the trim specialization on every row.
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
| Padding mask + stats (per-batch LSE trim) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Dense padded-Q trim (O:=0, LSE:=−inf) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Attention sink | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Base-2 stats (`stats_use_log2`) | ❔ | ❔ | ❔ | ❔ | ❔ | — |
| GQA / MQA (`H_q ≠ H_kv`) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| PackGQA | fp8 only | fp8 only | ❌ | ❌ | ❌ | ❌ |
| Split-KV | fp8 only | fp8 only | ❌ᵛⁱⁱ | ❌ᵛⁱⁱ | ❌ᵛⁱⁱ | ❌ |
| Fused epilogue gate (sdpa virtual `O_v` → `mul(O_v, sigmoid(G))`, `G = (B, H_q, S_q, D_v)`; graph tail + standalone `sample_gate`)ᵛⁱⁱⁱ | ❌ | ❌ | ❌ | f16/bf16 ✅ · fp8 ✅ (bf16 G) · mxfp8 ✅ (bf16 G; a gated e4m3 O is unscaled) | ❌ | ❌ |
| Optional stats (LSE store compiled out) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Bias | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Ragged `S_kv` (non-multiple of 128) | ✅ⁱˣ | ✅ⁱˣ | ✅ⁱˣ | ✅ⁱˣ | ✅ⁱˣ | ❌ |
| FP16 softmax accumulate (`sdpa(softmax_precision=HALF)` op attribute) | ❔ⁱⁱⁱ | fp8 only (Rubin f16x2 arm) | fp8 only (same body as d128) | ❌ | ❌ | — |

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
| Attention sink / dSink | ✅ | ✅ | ✅ / ✅ |
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
Head dims below a flavor's native shape are zero-padded **host-side**, so there
is no alignment rule. The backward serves packed THD graphs (ᵏ); the forward
does not yet.

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

The SM80 backward additionally has a dedicated plain-dense **d=64 fast path**
(~2× on A100) that supports **no** features — it is selected only for a
feature-free d=64 graph.

ᵏ **THD / ragged backward** (`sdpa_bwd_sm80`). Q/K/V/O/dO and the gradients are
PACKED `[1, T, H, D]` **BSHD rows, each at its own token stride**: head stride
`D`, element stride 1, token stride `>= H*D` and a multiple of 8 elements
(16-byte rows for the `cp.async` loads). A compact port is the common case; a
K/V view into an interleaved `[T, 2, H, D]` record (token stride `2*H*D`, the
fused-KV slicing layout) is served at that stride, the gap columns never read or
written. The strides are plan-time (the compiled fakes carry them); a compact
port keeps the compact fake, byte-identical codegen.
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
| Backward pass entirely | SM107 |
| Backward outside d ∈ (256, 512] (f16/bf16) or d = 256 (MXFP8) | SM100, SM103 — the two backward engines there serve exactly those bands |
| Backward per-batch padding mask (`seq_len_q/kv`) on a DENSE graph | SM100, SM103 — a UNIFORM non-tile-multiple length is served, and the THD path carries per-sequence lengths; a per-batch mask on a dense graph is not |
| Backward sink / dSink, bias / dBias | SM100, SM103 |
| Backward deterministic, decode | SM100, SM103 — served by the MXFP8 d=256 row only |
| MXFP8 backward: E5M2, bottom-right / band-widened / sliding-window masks, non-BSHD strides, `amax_*` outputs | SM100, SM103 |
| f16/bf16 forward split-KV, PackGQA | SM107 (Rubin) — the row serves dense f16/bf16 at d128/d192×d128/d256/d512, THD on all of them as of 2026-09-09, and the dense padded-Q trim as of #1037; these two are the machinery its kernels still lack (optional stats IS served — `lse_optional=True`) |
| d192×d128 quantized PackGQA / split-KV, and d192 MXFP8 THD | SM107 — the shape is served in FP8 and MXFP8 as of 2026-09-09, and per-tensor FP8 **THD** with it; PackGQA and split-KV stay wired in the d128 flavor only (`pack_gqa_d_shapes` / `split_d_shapes`), and the MXFP8 line declines THD row-wide |
| MXFP8 forward | SM120, SM80 (SM107 is served — see the SM107 table; d512 is ⚠️ⁱᵛ, correct but with no test module) |
| Per-tensor FP8 backward | every arch |
| MXFP8 backward outside SM100/SM103 d = 256 | every arch |
| THD / ragged backward | SM120, and the SM100/SM103 MXFP8 row (the SM100/SM103 f16/bf16 row serves it — see ʰ; SM80 — see ᵏ) |
| THD forward | SM80 |
| **Native d=64 (GPT-OSS) forward kernel** | **SM100, SM107** — served via the d128 envelope at ~2× MMA cost (decode shapes ride the d128 decode tile, ᵈᵗ) |
| Decode tile outside the d128 f16/bf16 flavor | SM100, SM103 — d192×128 / d256 / d512 decode and every fp8 / mxfp8 decode have no dedicated decode tile: each runs its flavor's prefill kernel at that flavor's own CGA width (f16 d256 / d512 and the quantized d128 flavors at `TILE_CGA_M=2`; per-tensor FP8 d256 and SM100 MXFP8 d256 / d512 are cga1 kernels; d192×128 selects 1 or 2 by shape). THD queries on the d128 f16/bf16 flavor keep its prefill pipeline (`TILE_CGA_M=2`) too (ᵈᵗ) |
| d=64 MXFP8 / d=64 quantized THD | SM100, SM107 (exact-shape gates) |
| Bias forward | SM100, SM107, SM120 |
| Dropout, ALiBi, `block_mask`, `score_mod` | every arch, both passes |
| Paged KV cache | every arch except SM100/SM103 f16/bf16 forward on the d128, d256 and d512 flavors (see ᵖ; d192×d128 not wired); fp8/mxfp8 pools, sink, packed block tables everywhere |
| Fused epilogue gate (`O * sigmoid(G)` tail) | every arch and flavor except SM107 d256 f16/bf16, per-tensor FP8 and MXFP8, exact (256, 256), dense / unsplit / non-PackGQA / non-paged (see the SM107 table) |
| PackGQA of a group sharing no factor with the 128-row tile (G = 3, 5, 7, …), and partial packing outside the SM100/SM103 f16/bf16 d128 / d256 kernels | every arch — such groups run unpacked (see ᵐ); the d192×d128 / d512 f16 and the fp8 / mxfp8 kernels pack the whole group only |

ᵏ **THD / ragged forward layout.** Q/K/V/O must be BSHD-ordered over **(H, S, D)**
only — head dim innermost, then heads, then tokens (`graph_analyzer.packed_layout_ok`).
The batch stride is not gated: every sequence base comes from the ragged offsets and
the lowering binds the batch axis at extent 1, so its declared value is never read.
FlashInfer declares it equal to the token stride (`h * d`), which the previous
all-four-axes check refused at `b > 1`. Stats under THD are written in the caller's declared layout: packed `(T, H)` rows, head-major `(1, QH, head_stride)`, or -- a Stats tensor **without** ragged offsets -- the per-batch padded form -- the graph's logical `[b, h, s_max, 1]` Stats view over FlashInfer's physical, contiguous `(b, s_max, h)` `return_lse` buffer (declared strides `[s_max*h, 1, h, 1]`; the adapter rebuilds the view with `as_strided`, nothing is allocated in the logical order), stored per batch through the declared strides on every THD row (SM100 / SM107 / SM120, `Capabilities.thd_padded_stats`); the adapter seeds that buffer with `-inf` on the launch stream first, so the rows past a sequence's length read the backend's value. On SM100 / SM107 the THD templates compile with DYNAMIC batch and head extents (`compile(dynamic_bhk=True)`): one artifact per layout class (d, dtypes, masks, GQA ratio, packed vs declared strides), not per shape.
