# Gated Attention Block (SM107)

**This is an experimental API and subject to change.**

## Overview

The gated attention block is the first **model-level** FE-OSS API: a set of FROST CuTe-DSL kernels behind one
Python class, one workspace and one `execute()` call. It implements the gated attention sub-layer used by
Qwen3.5- and Qwen3.8-style models (the API is named by op geometry, the model is provenance only; the Qwen3.8
family's geometries, the Qwen3.5-397B tensor-parallel shards and the bound up to which the block is Qwen3.8-Flash-Next's
sparse layer are under [Qwen3.8 family / Qwen3.8-Flash-Next](#qwen38-family--qwen38-flash-next)):

```text
h [B, S, d_model]                                       (post input-layernorm)
 |
 |  (1) QKV + GATE projection      h @ W_qkvg^T          FROST GEMM
 +----------------------------------------------------------------------------+
 |  Q [B,S,H_q,D] | GATE [B,S,H_q,D] | K [B,S,H_kv,D] | V [B,S,H_kv,D]
 |
 |  (2)+(3) QK-RMSNorm per head (optional) + partial RoPE on the first rope_dim   ONE kernel (Q and K only)
 |
 |  (4) SDPA        O = softmax(Q K^T * scale + mask) V,  GQA H_q / H_kv        FROST SDPA (Rubin d256 rows)
 |
 |  (5) gate        O_gated = O * sigmoid(GATE)                                 elementwise, after the SDPA epilogue
 |
 |  (6) out projection   O_gated @ W_o^T                                        FROST GEMM
 v
out [B, S, d_model]
```

Every stage is a FROST kernel: the two projections drive the shipped FROST GEMM engine (pinned by name in the
graph's ranked plan list), stage (4) drives the shipped FROST SDPA through its standalone adapter, and stages
(2)+(3) and (5) are this block's own kernels. There are no cuBLAS or cuDNN-backend call-outs.

**Target: NVIDIA Rubin (SM107, compute capability 10.7) only.** Every other architecture is declined with a
typed error (`NotImplementedError`), never served slowly.

Three precisions share the signature, and two fp4 modes ride the MXFP8 one (both are fields on `MxQuantSpec`,
so they are unspellable on the bf16 and per-tensor FP8 pipelines rather than declined):

| mode | how it is selected | pipeline |
|---|---|---|
| bf16 / fp16 | `h`, weights and `cos`/`sin` in bf16 or fp16, `quant=None` | 5 stages, 5 launches (4 with `inplace_qkv`) |
| FP8, per-tensor static scales | e4m3 `h` / `W_qkvg` / `W_o` and a `QuantSpec` | unfused: FP8 projections with the descale folded into the epilogue, bf16 norm+RoPE, two quantize passes (Q/K/V, gated O), the Rubin per-tensor FP8 SDPA and an FP8 out projection (9 launches); fully fused: 3 launches |
| MXFP8, per-32-element E8M0 block scales | e4m3 `h` / `W_qkvg` codes, the two scale-factor blobs (`h_sf`, `w_qkvg_sf`) and an `MxQuantSpec` | unfused: block-scale projection GEMM, bf16 norm+RoPE, MXFP8 quantize (Q/K rowwise, V columnwise), the Rubin d256 MXFP8 SDPA, per-tensor quantize of O, FP8 out projection (9 stages); fully fused: 3 launches |
| MXFP8 with **MXFP4 weights** | `MxQuantSpec(w_qkvg_dtype=torch.float4_e2m1fn_x2)` and an e2m1 `W_qkvg [N, d_model // 2]`; `h`, `h_sf` and `w_qkvg_sf` unchanged | unfused: the MXFP8 pipeline with stage (1) on the catalog's mixed MXFP8 x MXFP4 block-scale row (E8M0 scales per 32 on both sides) -- the same 9 launches; fully fused: the same **3 launches**, the fused projection fork's e2m1-B arm reading the packed codes and the unchanged blob inside its norm+RoPE+quant epilogue (dense only, like the MXFP8 fused pipeline) |
| MXFP8 with **fp4 O** (`NVFP4` or `MXFP4`) | `MxQuantSpec(o_fp4=Fp4Format.NVFP4 \| Fp4Format.MXFP4)`, an e2m1 `W_o [d_model, H_q * D // 2]` of the SAME format and its scale blob (`sample_w_o_sf` / `w_o_sf`) | the per-tensor tail is replaced: one `quantize_fp4` launch writes the gated `O` as e2m1 codes plus the out projection's scale blob, and stage (6) becomes the fp4 x fp4 block-scale GEMM (no per-tensor scale on either side). Unfused: 9 launches (`quantize_fp4` takes the per-tensor quantize's place); fully fused: **4 launches** (the gated MXFP8 SDPA writes bf16 `O`, then `quantize_fp4`, then the fp4 out projection). Composes with the MXFP4 weights on both pipelines (fully fused: the e2m1-B projection fork + the fp4 tail, 4 launches) |

### Fusion knobs

Two optional forward fusions are constructor flags; each is a different compiled specialization behind the same
`execute()` signature, and both default to off. The backward has two knobs of its own, both default off and both
bitwise the unfused block: `fuse_gate_bwd` (the gate backward emits the SDPA backward's `delta`, one launch fewer) and
`fuse_wgrad_overlap` (the two weight-gradient GEMMs run on a block-owned side stream, forked from and joined back to
the launch stream through events, overlapping the SDPA backward chain) -- see [Backward](#backward).

- `fuse_norm_rope=True` folds stages (2)+(3) into stage (1)'s epilogue: Q/K tiles are normed and rotated on the
  fp32 accumulator and written once (needs `inplace_qkv`; inference only, no pre-norm Q/K is kept). Under FP8 /
  MXFP8 the same epilogue also quantizes: compact e4m3 `q8` / `k8` / `v8` (and the MXFP8 scale factors) come
  straight out of the GEMM, so the quantize passes disappear.
- `fuse_gate=True` folds stage (5) into the SDPA epilogue: `O *= sigmoid(GATE)` after the dead-row select, with
  the gate tile TMA-staged by the load warp (inference only: no pre-gate `O` for the backward). Dense and packed
  (`thd=True`): under THD the gate rides the packed d256 prefill's epilogue through the SDPA's THD gate descriptor
  (G mapped like Q / O, at the slab's gate column stride), so the packed block is one launch fewer too.
- `fuse_prepare=True` (serving, with `paged_kv_page_size > 0`) folds stages (2)+(3) and the paged-cache write-through (4w)
  into ONE launch: Q / K are normed and rotated in place and the post-RoPE K, the V and the raw indexer key land in the
  pools from the same registers -- every output bitwise the two-stage form, one launch fewer (two with the indexer band).
  In-place bf16 / f16 only, on the dense declaration with the write-through alone or on the block-sparse DECODE mode over the pools
  (`kv_source="paged"` on a `QsaSpec` geometry, 2026-10-21: the prepare kernel's paged arms -- the raw-key pool under caller lists, the
  indexer's query + compress arms under `index_source="indexer"`; `fuse_prepare=None` is the MEASURED default there: True on cc 10.7
  (2026-10-21: the fused step +9.6 .. +14.5 % at B &lt;= 4 and +4.4 .. +10.0 % at B 32 over 19 decode cells, bitwise), `False` on every other
  part until measured, and `False` out of place (`inplace_qkv=False`) on every part -- the AUTO reading is never True where an explicit
  True is declined); `fuse_norm_rope` together with it is a typed decline (pick one), as are `thd=True` and
  `kv_source="paged"` on a DENSE geometry (the prepare kernel's packed arm and the dense DECODE form's are follow-ups) and
  `kv_cache_dtype=torch.float8_e4m3fn` (the e4m3 cache's write-through quantizes in its own stage); a geometry with `rope_dim=0`
  (RMSNorm only) is a `ValueError` at `check_support` naming the zero-width RoPE table operands the fused launch would bind.
  `use_pdl=True` launches the fused stage with the programmatic-dependent-launch attribute, as it does the two stages it replaces.
  See "Serving: write-through".

With both on, the block is **three launches**: `proj(+norm+RoPE[+quant]) -> sdpa(+gate) -> out_proj`.

`geometry.qk_norm=False` runs RoPE-only Q/K on every path (norm weights are passed as `None`, no `rstd` is
produced, the backward drops the norm-weight gradients).

## API

```python
import cudnn
from cudnn.gated_attention_block import (
    GatedAttentionBlockFwd, GatedAttentionBlockGeometry, QuantSpec, MxQuantSpec, Fp4Format,
    build_fused_qkvg_weight, qkvg_from_hf, qkvg_from_vllm, GatedAttentionBlockBwd, RecomputePolicy, SavedForBackward,
)
```

### Geometry

`GatedAttentionBlockGeometry` (frozen dataclass; `validate()` raises `ValueError`):

| field | meaning |
|---|---|
| `d_model` | model width |
| `h_q`, `h_kv` | query heads and KV heads (GQA); `h_q % h_kv == 0` |
| `d_head` | head dim of Q, K, V and O (`d_qk == d_v`); selects the SDPA flavor (d256 today) |
| `rope_dim` | the LEADING dims of each head that rotate; `[rope_dim, d_head)` pass through |
| `qk_norm` (`True`) | apply QK-RMSNorm before RoPE; `False` = RoPE only, no norm weights, no `rstd` |
| `qk_norm_eps` (`1e-6`) | RMSNorm epsilon |
| `attn_scale` (`None`) | softmax scale; `None` = `d_head ** -0.5` |
| `is_causal` (`True`), `causal_bottom_right` (`False`) | causal mask and its diagonal alignment |
| `window_left`, `window_right` (`-1`) | sliding window bounds; `-1` = unbounded |

### Qwen3.8 family / Qwen3.8-Flash-Next

The block's math is also the gated attention sub-layer of the Qwen3.8 family (provenance only -- the API stays named by op
geometry). The five geometries below are pinned in the layout contract (`test_layout_contract.py`: the stage-(1) column map,
the GQA ratio, the tile plans, the norm kernel's fitted tile) and run end to end against the fp32 oracle on Rubin
(`test_block_end_to_end.py`, the `qwen38` cells: the unfused bf16 pipeline at all five and the fully fused pipeline at 24 / 2,
B = 2 for the Flash-Next geometries and B = 1 for the two large siblings, S in {512, 2051}, the weights loaded through
`qkvg_from_hf`):

| model | `d_model` | `h_q` / `h_kv` | `N` (= `n_qkvg`) | GQA group | stages (2)+(3) kernel |
|---|---|---|---|---|---|
| Qwen3.8-Flash-Next, TP 1 | 2560 | 24 / 2 | 13312 | 12 | TMA, 12-row tile |
| Qwen3.8-Flash-Next, TP 2 | 2560 | 12 / 1 | 6656 | 12 | TMA, 12-row tile |
| Qwen3.8-Flash-Next, TP 4 | 2560 | 6 / 1 | 3584 | 6 | LDG (no TMA tile fits 6 / 1) |
| Qwen3.8-27B | 5120 | 24 / 4 | 14336 | 6 | TMA, 12-row tile |
| Qwen3.8-2.4T-A95B | 8192 | 64 / 4 | 34816 | 16 | TMA, 16-row tile |

All five share `d_head = 256`, `rope_dim = 64`, QK-RMSNorm with zero-centered weights and the per-head `[q_h | gate_h]` split
of `q_proj` -- load them through `qkvg_from_hf` (next section). The 27B and 2.4T members are dense gated attention: the block
IS their attention layer at every sequence length it serves (a prefill block: `S = 1` is decode, out of the prefill bodies'
scope).

**Qwen3.5-397B tensor-parallel shards.** The 397B's full-attention layer (32 / 2 at TP 1, `d_model` 4096; the same `d_head` 256,
`rope_dim` 64, QK-RMSNorm and per-head `[q_h | gate_h]` split, loaded through `qkvg_from_hf` or `qkvg_from_vllm`) as the serving
stacks shard it -- the query heads divided across the ranks, the two KV heads divided at TP 2 and REPLICATED from TP 4 on (one KV
head per rank; the TP 4 shard is the serving deployment's) -- is pinned the same way (`test_layout_contract.py`, the `qwen35_shard`
cells: the N-axis map, the tile plans at 64 / 128 / 256, the fitted norm tile) and runs end to end on Rubin in all three forms, one
cell per shard within the dense suite's budget: dense (`test_block_end_to_end.py`: B = 1, S = 512, HF-form weights against both
oracles), EXTEND (`test_block_serving_extend.py`: two sequences, a 100-token step over a 60-row prefix and a 156-token step over a
100-row prefix) and DECODE (`test_block_serving_decode.py`: `S_q = 1` over `kv_lens` 100 / 384 / 500 and a padded entry), each shard
on the kernel block size its deployment allocates:

| shard | `d_model` | `h_q` / `h_kv` | `N` (= `n_qkvg`) | GQA group | stages (2)+(3) kernel | end-to-end cells (bf16) |
|---|---|---|---|---|---|---|
| Qwen3.5-397B, TP 1 | 4096 | 32 / 2 | 17408 | 16 | TMA, 16-row tile | the perf tables below (the kernel cell); the serving suites' 32 / 2 matrix at a shrunk width |
| Qwen3.5-397B, TP 2 | 4096 | 16 / 1 | 8704 | 16 | TMA, 16-row tile | dense; EXTEND and DECODE at page 16 |
| Qwen3.5-397B, TP 4 | 4096 | 8 / 1 | 4608 | 8 | TMA, 8-row tile | dense; EXTEND and DECODE at page 32 |
| Qwen3.5-397B, TP 8 | 4096 | 4 / 1 | 2560 | 4 | TMA, 4-row tile (one row per warp) | dense; EXTEND and DECODE at page 64 |

Every `N` is a whole number of 256-column tiles, so the forced 256-wide GEMM tile of the serving route holds at every shard and
`qkvg_tile_plan(64 / 128 / 256)` is well defined; the replicated single KV head lets every power-of-two head count fit a TMA norm
tile (the 3.8 family's 6 / 1 is the one pinned geometry that takes the LDG kernel). A shard's decode route is its GROUP's (the rule
reads `h_q / h_kv` and `d_head` only, never `d_model` or the KV-head count): 16 / 1 takes the 32 / 2 entries of the table under
"Serving: the decode form", 8 / 1 its own, 4 / 1 the group-4 entries -- every shard's group divides the prefill tile, so the KV is
streamed once per sequence and KV head at every `S_q <= 8` once the step fills the part (the decode tile in one unit, then the packed
prefill tile at `B x H_kv >= DECODE_PREFILL_ROUTE_MIN_CTAS` = 64 CTAs; below that fill the steps past the single-unit table ride two
units of the 64-column decode tile, split over the part); past a group's packed routes (32 / 2 from `S_q` 9) the step rides the decode
tile unpacked, one head per CTA, up to `S_q = 32`; a typed decline past 32 tokens.

**Flash-Next: exact for `<= 2051` visible tokens.** Flash-Next's attention layer is Qwen Sparse Attention (QSA): the same
gated core restricted, per query, to indexer-selected 4-token blocks under a 2048-token budget -- the top `min(512, n_blocks)`
complete blocks plus the open tail block, at most `2048 + 4 - 1 = 2051` keys per query. A query that sees `n` tokens has
`floor(n / 4)` complete blocks, and every one of them is selected when `floor(n / 4) <= 512`, i.e. `n <= 2051`: the selection
is the identity and causal attention over the visible tokens IS the QSA layer. So for every query with at most 2051 visible
tokens -- every sequence of up to 2051 tokens -- the dense block computes Flash-Next's attention layer exactly: the same
function (a sparse evaluation sums the same terms in another order, so the identity holds within the suite's budget, not
bitwise). Beyond 2051 visible tokens QSA drops the blocks its indexer did not select while this block attends densely -- a
different function: declare the geometry with `qsa=QsaSpec(...)` and hand the selection in as `execute(block_ids=)`
("Sparse attention (QSA)" below) and the block runs the sparse core as its attention stage -- or declare
`QsaSpec(index_source="indexer", index_band=True)` and the block derives the selection itself (the in-block indexer: the prefill form
over the prompt, and since 2026-10-21 its DECODE step over the paged compressed-key cache).

### Weights and tables

- `W_qkvg [N, d_model]` with `N = (2*H_q + 2*H_kv) * D`, column blocks `Q | GATE | K | V` (a fifth block `INDEX` under a
  block-sparse geometry's indexer band, below); the block's tile alignment is `QKVG_TILE_ALIGN = 64` columns. Build it ONCE
  at load time, never per call:
  - **from a HF Qwen checkpoint** (Qwen3-Next, Qwen3.5, the Qwen3.8 family / Flash-Next) with the documented entry
    point, which also prepares the two QK-norm weights:

    ```python
    w_qkvg, w_q_norm, w_k_norm = qkvg_from_hf(
        attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight, attn.q_norm.weight, attn.k_norm.weight, geometry,
        act_dtype=torch.bfloat16,   # the block's activation dtype; all three results come back in it
    )
    ```

    It applies the two conventions those checkpoints share: the double-width `q_proj` is split PER HEAD
    (`q_proj(x).view(..., H_q, 2*D).chunk(2, dim=-1)`, i.e. weight rows `[q_0 | gate_0 | q_1 | gate_1 | ...]`), and
    the QK-norm weights are zero-centered (`x_normed * (1 + w)`, `w` initialised to zeros), so the block -- which
    multiplies by the vector it is given -- receives `(1 + w)` in the activation dtype (rounding cost below).
  - **from three matrices with an explicit layout**: `build_fused_qkvg_weight(w_q_gate, w_k, w_v, geometry,
    q_gate_layout="per_head" | "flat")`. The layout is a property of the checkpoint and is not inferable from the
    tensor (`"flat"` = all Q heads, then all GATE heads); getting it wrong applies every gate to the wrong head with
    no error anywhere. Omitting `q_gate_layout` still means `"flat"` but emits a `FutureWarning` (the deprecation
    category Python shows under its default filters, so a loader in a library module sees it too) -- pass it explicitly.
  - **from a serving stack's merged `qkv_proj.weight` at a tensor-parallel rank** (vLLM's `QKVParallelLinear`, SGLang's
    likewise): `qkvg_from_vllm(qkv_proj_weight, q_norm_weight, k_norm_weight, geometry, act_dtype=torch.bfloat16)`. The
    merged weight's rows are `[q_proj | k_proj | v_proj]` at the RANK's head counts -- `2*H_q*D` rows of `q_proj` in the
    checkpoint's per-head `[q_h | gate_h]` order, then `H_kv*D` of `k_proj` and `H_kv*D` of `v_proj` -- and `geometry`
    describes the shard (`h_q`, `h_kv` = the rank's heads: 8 / 1 for a 32 / 2 model at TP 4, the KV head replicated across
    the ranks that share it). The twin cuts the three row ranges and hands them to `qkvg_from_hf`, so all three results are
    bitwise `qkvg_from_hf`'s on the same HF tensors: the per-head split is fixed (there is no layout argument), and the norm
    weights -- replicated on every rank -- take the `(1 + w)` form derived from the geometry. `w_qkvg` is a row permutation of
    the input with the same byte count, so a layer may replace its merged weight by it in place and keep its `o_proj.weight`
    as `W_o`. Under an indexer band the indexer projection arrives as `index_qk_proj_weight=`, or appended to the merged
    weight's rows as a stack that packs the (replicated) indexer projection into the same GEMM holds it -- one of the two,
    never both.
- `W_o [d_model, H_q * D]`.
- `cos`, `sin` `[B, S, rope_dim]` rotary tables in the activation dtype (rotate-half convention on the first
  `rope_dim` dims of every head).
- `w_q_norm`, `w_k_norm` `[D]` (both `None` iff `geometry.qk_norm` is `False`). The block multiplies the normalised
  row by this vector as given, so a zero-centered checkpoint weight is handed in as `(1 + w)` (`qkvg_from_hf` does
  this). Forming `1 + w` in fp32 and rounding it once into the activation dtype costs at most half an ulp at 1.0 per
  channel -- `2^-8 = 0.39 %` relative in bf16, `2^-11 = 0.049 %` in f16 -- a systematic per-channel scale error
  inside the block's accuracy budget (`cos >= 0.999` on `out`) but not bit-faithful to the checkpoint -- measured end to
  end at the five Qwen3.8 family geometries (the `qwen38` cells of `test_block_end_to_end.py`, Rubin, `w ~ N(0, 0.1)`): the
  block on the rounded weights reads `cos >= 0.99998` against an oracle fed the fp32 `(1 + w)` at every cell (and
  0.999995-0.999996 unfused / 0.999986-0.999987 fully fused against the oracle fed the same rounded weights). Per
  channel, for `w ~ N(0, sigma)` (2^18 samples, seed 0, the table `test_rounding_of_a_zero_centered_norm_weight_handed_in_as_one_plus_w` prints; `rel err = |rnd(1 + w) - (1 + w)| / (1 + w)`):

  | sigma of `w` | bf16 max rel err | bf16 mean rel err | bf16 channels rounded to exactly 1.0 | f16 max rel err | f16 mean rel err | f16 channels rounded to exactly 1.0 |
  |---|---|---|---|---|---|---|
  | 1e-3 | 0.389 % | 0.078 % | 97.5 % | 0.049 % | 0.018 % | 28.5 % |
  | 1e-2 | 0.389 % | 0.146 % | 22.9 % | 0.049 % | 0.018 % | 3.0 % |
  | 1e-1 | 0.389 % | 0.144 % | 2.4 % | 0.049 % | 0.018 % | 0.3 % |

  A trained `-2^-9 < w < 2^-8` is lost entirely in bf16 -- an asymmetric interval, because bf16's spacing is `2^-8`
  just below 1.0 and `2^-7` just above (f16 loses `-2^-12 < w < 2^-11`); it is the 97.5 % at `sigma = 1e-3` above.
  The faithful form -- the weight stored as `w`, the `1` added in fp32 inside the norm kernels behind a geometry
  field `norm_weight_offset = 1.0` -- is a follow-up; `qkvg_from_hf` derives which form to hand over from that same
  geometry field (absent or `0.0` today), so the two can never compose into `1 + (1 + w)`.
- MXFP8 only: `h_sf` and `w_qkvg_sf`, the E8M0 scale factors of `h` and `W_qkvg` in cuDNN's F8_128x4 order
  (`uint8` or `float8_e8m0fnu`; byte counts from `cudnn.gated_attention_block.kernels.proj_gemm.sf_blob_bytes`).
- fp4 weights are **packed e2m1**, dtype `torch.float4_e2m1fn_x2`, stored `[N, K // 2]` -- two codes per byte along
  the contraction axis, LOW nibble = even `k`. The block checks the STORAGE shape (`[N, d_model // 2]` for `W_qkvg`,
  `[d_model, H_q * D // 2]` for `W_o`); a logical `[N, K]` fp4 tensor, or `uint8` storage, is a typed `ValueError`
  (torch can `.view(torch.float4_e2m1fn_x2)` packed bytes but cannot cast to fp4).
  - MXFP4 `W_qkvg` (`MxQuantSpec.w_qkvg_dtype=torch.float4_e2m1fn_x2`): `w_qkvg_sf` is UNCHANGED -- the same E8M0 /
    32 F8_128x4 blob over `n_qkvg x d_model` as for e4m3 codes. The fused projection fork reads the same `[N, d_model // 2]`
    storage and the same blob through the packed sub-byte TMA format, whose rules the fused stage checks before any launch
    (typed `ValueError`): the fused pipeline's `d_model % 128 == 0`, a 32-byte-aligned weight base address and a row stride
    that is a multiple of 32 bytes (a contiguous `[N, d_model // 2]` weight satisfies both; a 16-byte-aligned slice of a
    larger buffer or a padded-row view does not), and -- on both arms -- a unit stride along K.
  - fp4 `W_o` (`MxQuantSpec.o_fp4`): its scale blob `w_o_sf` is in the SAME format as `O` -- `float8_e4m3fn` scales per
    16 for `Fp4Format.NVFP4`, `float8_e8m0fnu` per 32 for `Fp4Format.MXFP4` (either as `uint8` or as that dtype),
    `sf_blob_bytes(d_model, H_q * D, block)` bytes in F8_128x4 order (padded to whole 128-row x 4-block atoms; pad
    bytes, if any, `0x00`).

### Sparse attention (QSA) -- declaration, layout, loader and the sparse attention core

`QsaSpec` (frozen dataclass; attached as `GatedAttentionBlockGeometry(qsa=...)`; `validate()` raises `ValueError`) declares
block-sparse attention: every query attends to the keys of its SELECTED 4-token blocks and to the open tail block of its visible
range, under the causal mask (Qwen Sparse Attention: a 2048-token budget at block size 4 = the defaults `block_size=4,
top_k=512`). It is a DECLARATION ATTRIBUTE -- it changes the function -- never a knob.

| field | meaning |
|---|---|
| `block_size` (`4`) | tokens per selectable block; 4 only (one gather of four rows per block) |
| `top_k` (`512`) | blocks per query in the caller's list: a multiple of 4 in `[4, 512]` (it sizes the kernel's index staging) |
| `index_source` (`"caller"`) | `"caller"`: `execute(block_ids=)` carries the selection; `"indexer"`: the block runs the indexer over its own band and derives the selection (bf16, `index_band=True`; "The in-block indexer" below) |
| `index_band` (`False`) | `W_qkvg` carries a fifth band `ProjBlock.INDEX` of `(index_heads + index_kv_heads) * index_head_dim` columns below V |
| `index_cache_dtype` (`None`) | appended: the indexer cache's dtype under `index_source="indexer"` -- `None` / `torch.bfloat16` (the compressed keys and the queries as the compress kernel writes them) or `torch.float8_e4m3fn` (the fp8 indexer cache: both as plain e4m3 of the same values, no scales; on the prefill form scored through the scorer's MXFP8 arm with unit scale factors, `index_k_compressed` then an e4m3 buffer; AT DECODE over the pools (2026-10-21) the paged `index_kv_cache` is e4m3 -- the prepare / compress kernels store the e4m3 cast of the bf16 row directly and the paged scorer reads it on its fp8 `mma.sync` arm, no cast kernel). Numerics-changing (3 mantissa bits move the weakest selected blocks at the k-th boundary: 99.0 / 97.4 / 96.5 % set agreement with the bf16 selection at S = 4096 / 32768 / 131072 on synthetic activations, every flip a rank-boundary flip), so an attribute, never a knob; bf16 stays the default and the reference. Refused under `index_source="caller"` (nothing of the block reads or writes an indexer cache there); e5m2 named |
| `index_list_form` (`None`) | appended: the block-sparse DECODE mode's list form (`kv_source="paged"` on a dense `[B, S_q]` block) -- `"shared"`: ONE list per sequence, `[B, top_k]`, selected at the step-0 position `kv_lens[b] - S_q` and shared by the step's rows (the MTP form; `S_q <= 4`, the sparse core's shared-list cap); `"per_row"`: one list per row, `[B, S_q, top_k]` / `[T, top_k]`, at the row's own position (the serving stacks' per-row selection) at ANY `S_q`; `None`: `"shared"` at `S_q <= 4` and `"per_row"` past it under caller lists (the former derivation), `"per_row"` under the in-block indexer over the pools. A declaration attribute (numerics-changing: which keys the speculative rows see), never a knob. Typed declines: `"shared"` past four rows (`shared_list_rows`), `"shared"` under the indexer over the pools; `"shared"` on the block's own K / V (`kv_source="self"`) is refused by name. The envelope publishes `decode.index_list_forms` / `decode.index_list_form_default` |
| `index_heads`, `index_kv_heads`, `index_head_dim`, `index_norm_eps` (`4`, `1`, `128`, `1e-6`) | the indexer's geometry; they size the band (and `index_kv_heads x index_head_dim` the raw indexer key's write-through pool); under `index_source="indexer"` the scorer reads them (`index_heads` one of its packed head groups, `index_head_dim` its head dim) |

`QsaSpec.identity_bound` (`top_k * block_size + block_size - 1` = 2051 at the defaults) is the visible-token count up to which
every query's complete blocks fit the list, so a full list reproduces dense causal attention exactly. `is_causal=False` or a
sliding window together with `qsa` is a `ValueError` (the selection IS the sparsity).

Served today:

- the layout: `qkvg_blocks` / `qkvg_block_widths` / `qkvg_offsets` / `qkvg_heads` / `qkvg_head_dims` grow a fifth entry iff
  `index_band` (at `d_model=2560, h_q=24, h_kv=2`: `N = 13312 + 640 = 13952`, offsets `0, 6144, 12288, 12800, 13312`; a
  256-column tile plan is not defined there, a 64- or 128-column one is -- at decode row counts, `B x S_q <= 128`, the slab GEMM
  takes the small-M route's 128-row swap-AB tile, 109 CTAs, so the band declaration over the pools compiles with the opt-in flag
  unset and is capture-safe like the four-band one; past 128 rows the band slab keeps the graph route.  The route reads the
  part's SM count: the 128-row tile is its overflow branch, taken where the slab's 218 64-row tiles exceed one wave -- every
  served part (&lt;= 212 SMs); a part with 218 or more SMs would keep the band slab on the graph route at every row count); a
  geometry without the band keeps exactly its four-band tuples. Enumerate `geometry.qkvg_blocks`, never `ProjBlock` itself.
- the loader: `qkvg_from_hf(..., index_qk_proj_weight=attn.index_qk_proj.weight)` and `build_fused_qkvg_weight(...,
  index_qk_proj_weight=)` append the band (required iff declared, refused otherwise); `qkvg_from_vllm` takes it the same way, or
  appended to the merged weight's rows (a serving stack that packs the indexer projection into its QKV GEMM).
- the UNFUSED projection writes the wider slab; `index_k_raw_view(proj_slab, geometry, B, S)` and
  `GatedAttentionBlockFwd.index_k_raw(workspace)` expose the raw indexer key (the band's last `index_kv_heads * index_head_dim`
  columns, pre-norm, un-rotated -- what a serving cache keeps) as a zero-copy `[B, S, index_kv_heads, index_head_dim]` view.
- `execute(..., block_ids=, block_lens=)` (appended): `block_ids` `[T, top_k]` int32 (or `[B, S, top_k]`), per query the ids of its
  selected complete blocks (block `b` = tokens `[4b, 4b + 4)` of its own sequence), the valid prefix then `-1`; `block_lens` `[T]`
  int32 optional. Checked for form only (dtype, rank, shape, contiguity, device -- never read on the host; an int64 list is
  refused, not converted); refused on a block declared without `qsa`.
- the sparse attention core: stage (4) of a block declared with `qsa` is the index-list sparse d256 forward (the Rubin kernel
  behind the standalone adapter `cudnn.sdpa.fwd.sparse_gqa_sm107`), at the same operands and strides as the dense stage --
  the slab's Q / K / V column slices in place, or the compact buffers with `inplace_qkv=False` -- and the same launch count;
  `get_workspace_size()` is the dense block's (the core needs no scratch). What a query attends to, on device: the keys of
  its listed complete blocks that lie inside its causal range, plus the open tail block of its visible range (always), under
  the per-batch `seq_lens` when given. The list's count per query is `min(top_k, floor((pos + 1) / 4))`, clamped by
  `block_lens[t]` when given; a `-1` at an index below the count contributes no key (a masked block -- its bytes are still
  fetched, so short lists should pass `block_lens`); entries at or beyond the count are never read. A row with no visible
  key (`seq_lens[b] == 0`) is `out = 0`, `LSE = -inf` exactly. With the full list -- every complete block, which fits up to
  `identity_bound` visible tokens -- the block computes dense causal attention: the same function within the suite's
  budget, summed in another order (not bitwise). Served: bf16 / f16, `d_head == 256`, `h_q // h_kv <= 16`, dense
  `[B, S, d_model]` with or without `seq_lens`, `return_lse` on or off, `fuse_norm_rope` (without the band), `inplace_qkv`
  on or off, the indexer band on the unfused projection, `paged_kv_page_size` write-through; validated on Rubin (SM107) at
  the 24/2, 12/1, 6/1 and 32/2 geometries against the fp32 QSA oracle.
- packed sequences (`thd=True, num_sequences=B, max_seq_len=S_max, cu_seqlens=`, "Packed sequences (THD)" below): the sparse
  core's packed arm over the same `[1, T, H, D]` views. `execute(seq_lens=)` (REQUIRED: the `[B]` int32 lengths or the
  `[B+1]` int32 prefix sums) serves the Q and the KV side alike, and `block_ids` `[T, top_k]` carries each token's blocks
  numbered FROM ITS OWN SEQUENCE'S FIRST TOKEN (block `j` of sequence `b` = tokens `[4j, 4j + 4)` of that sequence; the count
  rule and the open tail are per sequence, positions restarting at every sequence like the RoPE tables). `get_workspace_size()`
  grows by the packed SDPA's small metadata scratch. A one-sequence packing is bitwise the dense `B = 1, S = T` sparse block;
  an empty sequence owns no rows. Validated on Rubin per sequence against the per-sequence fp32 QSA oracle (bf16 and f16).
  `fuse_gate=True` composes with `thd=True` (the sparse core's epilogue gate on the packed `[1, T, H_q, D]` GATE slice, read in
  place like O; validated on Rubin: the gated packed block within one rounding of the activation dtype of the unfused packed
  block on O, the LSE bitwise). Not under `thd`: the in-block indexer (the scorer reads dense `[B, S]` prompts) -- a typed
  decline naming the feature. The paged write-through composes with the packed sparse block (the stage addresses tokens; the
  packed write-through cell of the end-to-end module).
- the in-block indexer: a block declared `QsaSpec(index_source="indexer", index_band=True)` derives the selection itself from
  the band -- `execute(..., w_iq_norm=, w_ik_norm=)` instead of `block_ids=` -- and hands it back through the optional
  `block_ids_out` / `block_lens_out` / `index_k_compressed`; the contract is "The in-block indexer" in the serving section
  below -- and, over the pools (`kv_source="paged"`, the block-sparse DECODE mode), its DECODE step: "The in-block indexer's DECODE
  step over the pools" there (2026-10-21).  bf16 only (the scorer's dtype); validated on Rubin against the two-stage oracle (the selection under the margin
  rule, attention exact given the kernel's selection).

Every claim of the sparse core -- the dtypes, the head dim, the block size, the `top_k` range, the GQA group, the arms it does
not carry -- is read off the adapter's capabilities record (`cudnn.sdpa.fwd.sparse_gqa_sm107.SPARSE_CAPABILITIES`), which the
adapter's own `check_support` enforces; the block transcribes none of it. Typed declines at declaration, naming the feature:
`save_for_backward` (sparse training is out of scope; `GatedAttentionBlockBwd` refuses a `qsa` geometry), `quant`,
`fuse_norm_rope` together with `index_band` (the fused projection renders 256-column tiles; 640 is not a whole number of them --
the unfused projection serves the band), `causal_bottom_right`, an fp32 activation, `h_q // h_kv > 16`, `d_head != 256` (`thd`
together with `paged_kv_page_size` is served: the write-through addresses tokens, so the packed sparse block builds the same
stage); under
`index_source="indexer"` also an f16 activation (with or without the fp8 indexer cache), an `index_head_dim` other than the scorer's 128, an `index_heads` outside the
scorer's packed head groups (4, 8, 16, 32, 64) and `thd`; `index_cache_dtype` other than `None` / bf16 / e4m3 (e5m2 named), or e4m3 under
`index_source="caller"`, from `QsaSpec.validate()`; at execute an `index_k_compressed` of the other dtype (the message names the declared one); and at `check_support`, a CuTe DSL below the floor the Rubin target
needs (the public 4.8.0 wheel), named with the installed version, before the kernel module is imported. Under
`index_source="indexer"` the floor is checked at declaration as well -- the scorer's head groups are read off the DSA tree, whose
package imports its kernels, so a too-old DSL declines from `GatedAttentionBlockFwd(...)` before that import (and again from the
stage's `check_support`).  `fuse_gate=True` is served: the sparse core stages each query's GATE slice (the slab's GATE columns,
read in place) into its epilogue and writes `O * sigmoid(GATE)` in place of O after its dead-row select, so stage (5) disappears
exactly as on the dense block; the gated O differs from the unfused block's by at most one rounding of the activation dtype (the
unfused path rounds O to bf16 / f16 before the gate), the LSE is bitwise the same.

### Serving: index lists and paged KV

The contract a serving stack writes its shim against. It is ONE contract at the block's two entry points -- the block
itself (`GatedAttentionBlockFwd(..., geometry_with_qsa, paged_kv_page_size=P).execute(..., block_ids=, block_lens=,
k_cache=, v_cache=, slot_mapping=, ...)`: the whole sub-layer from `h` to `out`) and the standalone sparse attention core
behind its stage (4) (`cudnn.sdpa.fwd.sparse_gqa_sm107.SparseGqaFwdDslSm107(q=, k=, v=, o=, lse=, block_ids=, block_lens=,
seq_kv_lens=, top_k=)`: the attention call alone, BSHD operands at the strides its docstring admits) -- the block hands the
index tensors to the core unchanged. Everything below is checked for FORM on the host (dtype, rank, shape, contiguity,
device); every VALUE (an id, a count, a slot, a length) is device data the kernels consume, so a stack's tensors are handed
over as they are: never read back, never converted, never reallocated, graph-capturable.

| the stack hands in | form | meaning |
|---|---|---|
| `block_ids` | int32 `[T, top_k]` with `T = B x S_q` (row `b x S_q + s` is query `s` of sequence `b`), or `[B, S_q, top_k]`; contiguous; int32 ONLY (an int64 list is refused, never converted) | per query, the ids of its SELECTED COMPLETE 4-token blocks, RELATIVE to the query's own sequence: block `k` is that sequence's tokens `[4k, 4k + 4)` -- never a page, a slot or a global token index; the valid prefix then `-1`; order unspecified (a top-k output) |
| `block_lens` | int32 `[T]` / `[B, S_q]`, contiguous; optional | the valid-prefix length per query, consumed on device by the count rule below |
| `QsaSpec.index_list_form` (a declaration attribute) | `None` / `"shared"` / `"per_row"` | the decode mode's list form: one step-0 list per sequence (`[B, top_k]`, `S_q <= 4`) or one list per row (`[B, S_q, top_k]`, any `S_q`); `None` = shared up to four rows, per row past them (caller lists) and per row under the in-block indexer over the pools |
| `top_k` | `QsaSpec.top_k`: any multiple of 4 in `[4, 512]` (512 = the 2048-token budget over 4-token blocks, the default) | the list width; it sizes the kernel's per-query index staging, so a list is exactly `top_k` wide |
| `seq_lens` (block) / `seq_kv_lens` (core) | int32 `[B]`, optional and declared (`seq_lens_present=True`) | the KV length of every sequence; a query sees `min(pos + 1, seq_lens[b])` tokens; `0` = a dead sequence |
| `paged_kv_page_size` (`P`, a declaration attribute) | a positive multiple of 16 -- so 4 divides P and a 4-token block never straddles a page | the write-through: pools `k_cache` / `v_cache` `[num_pages, H_kv, P, D]` (HND compact or NHD by strides) and `slot_mapping [T]` int32 / int64 (flat slot `page x P + offset`; a negative slot writes nothing) -- the next section |
| `block_table`, `kv_lens` | int32 `(B, max_pages)`; int32 `[B]` = the logical KV length INCLUDING the step's new tokens | the paged-READ inputs (decode, and chunked prefill over a cached prefix): REQUIRED under `kv_source="paged"` -- the block-sparse DECODE / extend-with-prefix mode (below, and "Serving: the decode form") |
| `kv_cache_dtype` (a declaration attribute) | `None` (= the activation dtype: bf16 / f16 pools) or `torch.float8_e4m3fn` with per-tensor fp32 `k_scale` / `v_scale` at execute (1-element CUDA tensors, value = code x scale) | the pool's dtype: a bf16 Q over a bf16 pool, or a bf16 Q over an e4m3 pool with the attribute (numerics-changing, hence an attribute) -- the quantizing write-through on the prefill forms, the gathered cast + the d256 decode tile on the block-sparse decode mode at `S_q <= 4` ("Serving: the e4m3 KV cache" below); an fp8 Q is not a served form |
| `return_lse` -> `lse` | fp32 `[B, H_q, S_q]`, optional | `max + log(sum)` in natural log over the query's visible keys; `-inf` on a row with no visible key |

**The count rule -- which entries of a row are read.** For a query at position `pos` of its sequence (dense: the row
index `s`; decode mode: `kv_lens[b] - S_q + j`, below) the kernel derives `n_sel_default = min(top_k, floor((pos + 1) / 4))`
-- the complete blocks the query can see -- and reads `count = min(n_sel_default, clamp(block_lens[t], 0, n_sel_default))`
entries when `block_lens` is given, `n_sel_default` entries otherwise. A `-1` at an index BELOW the count is "no key"
(nothing is attended, the bytes of a zero-filled block still cross the fabric -- a stack whose lists are shorter than the
derived default SHOULD pass `block_lens`); an entry at or beyond the count is NEVER read, valid-looking or not. A negative
or oversized `block_lens` value is clamped, never faulted.

**The open tail block is the kernel's -- never list it.** The incomplete block that closes the query's visible range
`n_vis = min(pos + 1, seq_lens[b])` -- the `n_vis % 4` tokens from `4 x floor(n_vis / 4)` to `n_vis - 1`; for a query
inside its sequence, the `(pos + 1) % 4` tokens from `4 x floor((pos + 1) / 4)` to `pos` -- is always visible and
derived on device from the position and the KV length; a list carries complete blocks only. A listed block lying
entirely past the visible range (strictly past the open one) contributes nothing: its keys are masked one by one against
the position and the KV length, never a fault. The open block itself, listed inside the count, is the violation below --
attended twice.

**A list is a SET of distinct complete-block ids; what happens on a violation.** The contract: distinct ids, each a
complete block of the query's visible range (`4k + 3 <= pos`), the valid prefix then `-1`. The kernel does not check it
(the ids are device data; a check would be a read-back) and does not deduplicate: an id listed twice, or the open block
listed inside the count, is attended TWICE -- the softmax over the multiset, finite and bounded, a wrong number, never a
hang or a fault. The selections both serving stacks produce are top-k outputs over complete blocks, so no served list
violates it; the test tree's detector (`block_ids_contract_violations` in
`test/python/gated_attention_block/cutedsl/gated_block_qsa_reference.py`) rejects both forms, and the kernel suite pins the
twice-attended reading so a change of it is visible.

**Dead rows.** A query with no visible key (`seq_lens[b] == 0`; a decode row of an empty sequence) is `out = 0`,
`LSE = -inf` exactly -- a select on the emptiness predicate, never a scaled residue.

**The identity below 2051 visible tokens, and why there is no dense route under caller lists.** A query with `n <= 2051`
visible tokens has at most 512 complete blocks, so the FULL list makes the sparse core compute dense causal attention (the
same function, summed in another order). The block still ALWAYS runs the sparse core under `index_source="caller"`: it
never reads a list, so it cannot know that a list is the identity, and a dense route would ignore the list -- a different
function, never a knob. Under `index_source="indexer"` the block DOES know: at `S <= 2051` it launches no scorer and hands the
sparse core the identity list as a plan-time constant (below); routing that case to the dense SDPA instead of the sparse core
would be a legal performance knob there, and stays a follow-up.

**The decode mode and the speculative (MTP) rows -- served (2026-10-14): a dense `[B, S_q <= 4]` declaration on the `QsaSpec`
geometry with `kv_source="paged"` ("Serving: the decode form" below).** `S_q <= 4` query
rows per sequence (one new token, or `1 + drafts`), attention over the pools through `block_table` / `kv_lens`. ONE list per
sequence, `block_ids [B, top_k]` (+ `block_lens [B]`), selected at the STEP-0 position `pos_0 = kv_lens[b] - S_q` -- never
replicated per draft row. The visible set of row `j` at `pos_j = pos_0 + j` is DEFINED as (i) the listed blocks, complete at
step 0, inside its causal range, plus (ii) EVERY token from the step-0 tail start `4 x floor((pos_0 + 1) / 4)` to `pos_j`
-- so the block completed between `pos_0` and `pos_j`, which the step-0 list cannot hold, stays visible, the row's own token
included. vLLM's index reuse applies the step-0 list with each ROW's own tail (`tail_start = 4 x floor((pos_j + 1) / 4)`),
which at `pos_j = 4k + 3` has an empty tail and hides block `k` from that row: a strict subset of this set; both are
approximations of a per-row re-selection. The reference oracle takes the step-0 position explicitly (`pos0`); passing
`None` yields the per-row-tail reading, so a shim that needs that parity can pin it. Chunked prefill over a cached prefix
is the same paged-READ mode with one list per new row (`[T, top_k]`) at positions `prefix_len + s`: a dense `[B, S_q > 4]`
declaration on the `QsaSpec` geometry with `kv_source="paged"` takes `block_ids [B, S_q, top_k]` / `[T, top_k]` (the list form
is `QsaSpec.index_list_form`: `None` derives it from `S_q` at declaration -- one per sequence up to four rows, one per row past it --,
`"per_row"` declares one list per row at ANY `S_q`, the step's rows at 1 .. 4 included -- the serving stacks' per-row selection, and the
form the in-block indexer's decode step over the pools binds --, `"shared"` the step-0 list at `S_q <= 4` by name), at any `S_q`.

**The indexer fields.** `QsaSpec.index_heads`, `index_kv_heads`, `index_head_dim`, `index_norm_eps` size the fifth
band of `W_qkvg` -- and `index_kv_heads x index_head_dim` the raw indexer key's write-through pool `index_k_raw` (next
section); under `index_source="caller"` no kernel scores with them, under `index_source="indexer"` they are the scorer's.

**The in-block indexer (`QsaSpec(index_source="indexer", index_band=True)`, bf16).** The block derives the selection
from its own fifth band -- five launches at `T <= 4096` (two compress launches, the scorer GEMM, the radix top-k, the row
sort; one more sort window per further 4096 rows; two casts more under the fp8 indexer cache, below), no kernel of its own: (1) the band's indexer QUERIES, RMSNormed with
`w_iq_norm` (`index_norm_eps`) and partially rotated at the token's position -- the block-compress kernel at pool 1 over
the slab's query columns, bitwise the block's own norm + RoPE chain on those rows (one fp32 pass, one rounding); (2) the
COMPRESSED KEYS -- the band's raw key head mean-pooled in fp32 per complete 4-token block, RMSNormed with `w_ik_norm` and
rotated at the block START position `4j`, one rounding -- the same kernel over the slab's raw-key slice; (3) the
SELECTION -- the indexer scorer `sum_h relu(q_h . kbar_j) / sqrt(index_head_dim)` over the blocks below
`floor((pos + 1) / 4)` and the fused radix top-k (ties to the smaller id) -- `cudnn.ops.qsa_select_forward`, the
DSA compressed-logits path, so the dense `[T, n_blocks]` score tensor is never materialized; (4) the CANONICAL ORDER -- the
top-k's set is reproducible but its slot order is not, and the core accumulates in list order, so each row is sorted (the
selected ids descending, the `-1` padding last) before the core: two executes of the block are bitwise, and the list a caller
gets back is canonical. The list is `[T, top_k]` int32 with exactly the derived count `min(top_k, floor((pos + 1) / 4))` of
valid ids then `-1`, so the sparse core takes it without `block_lens`.

| the stack hands in / gets back | form | meaning |
|---|---|---|
| `w_iq_norm`, `w_ik_norm` | `[index_head_dim]` activation dtype, contiguous, 16-byte-aligned base; REQUIRED | the indexer's two RMSNorm weights in the pre-folded `(1 + w)` form the attention's `w_q_norm` / `w_k_norm` use (the loader's convention) |
| `block_ids` / `block_lens` | -- | REFUSED: the block derives the selection |
| `block_ids_out` | int32 `[T, top_k]` or `[B, S_q, top_k]`, contiguous, 16-byte-aligned base; optional | the block's OWN selection in canonical order (ids descending, then `-1`), written in place (the step-0 list a speculative-decoding caller reuses for the drafted rows) |
| `block_lens_out` | int32 `[T]` or `[B, S_q]`, contiguous; optional | the valid-prefix length of every row of that list, `min(top_k, floor((pos + 1) / 4))` -- the position rule, padding rows included |
| `index_k_compressed` | the indexer cache dtype (`QsaSpec.index_cache_dtype`: the activation dtype by default, `torch.float8_e4m3fn` under the fp8 indexer cache) `[B, >= floor(S / 4), index_head_dim]`, contiguous, 16-byte-aligned base; optional | the compressed-key cache: the compress writes rows `[0, floor(S_b / 4))` of every sequence (its complete blocks) and never touches the rest, so a cache sized for a longer context keeps its later rows |

A storage-offset view of the two norm weights, the cache or `block_ids_out` that starts a few bytes off a 16-byte boundary is refused by
name at `execute`, before any stage runs (the compress and cast kernels move those buffers in 16-byte vectors; the list is the sparse
core's TMA / bulk-copy base) -- the rule the caller-list form applies to `block_ids`.

**The RoPE tables are the attention's own.** The compress reads `cos` / `sin` at row `4j` for block `j` (every block start
is a position of the prefill) and the query norm at row `pos`: no appended block-start table and no position ids. A
decode step's per-token tables do not reach a block start; its compress (the step form of the same kernel) states its own
table-extent rule.

**The identity below 2051 visible tokens.** At `S <= QsaSpec.identity_bound` every query's complete blocks fit the list,
so the selection is the identity: the block launches NO scorer and NO query norm, hands the sparse core a plan-time constant
list (`ids[t, j] = j` below the row's count, `-1` past it; copied into `block_ids_out` on request) and reserves no indexer
workspace; the key compress runs only when `index_k_compressed` is given (a serving cache wants the prompt's compressed
keys whatever its length). Past the bound the block is BITWISE the caller-list block fed its own `block_ids_out`.

**Padding rows and `seq_lens`.** A row at or past its sequence's KV length is scored like every other row (the position
rule); the compressed-key rows past the sequence's complete blocks are zero-filled in the block's workspace per execute, so
such a row's extra candidates score 0 -- a positive visible block always wins, a zero-score tie goes to the smaller
(visible) id -- and the sparse core masks every key past the length anyway. The caller's `index_k_compressed` is not
zero-filled (its rows past a sequence's blocks are the caller's); a padding row's list may name blocks past its range,
which attention never reads.

**The fp8 indexer cache (`QsaSpec(index_source="indexer", index_band=True, index_cache_dtype=torch.float8_e4m3fn)`).** The
compressed keys and the indexer queries as plain e4m3 of the RMSNormed + rotated values (no scales -- the serving stacks' fp8 indexer
cache recipe), scored through `cudnn.ops.qsa_select_forward`'s e4m3 arm: the DSA MXFP8 unified scorer with all-ones E8M0
scale blobs, i.e. `sum_h relu(e4m3(q_h) . e4m3(kbar_j)) / sqrt(128)`. No new kernel: the two compress launches keep writing bf16
(the one-rounding chain) and the FP8 pipeline's per-tensor cast (`sat_e4m3(row x 1.0)` -- round-to-nearest-even, saturating at
448, never a NaN) writes the e4m3 operands into two workspace slots appended after the indexer's seven (every layout without the
attribute is byte-identical); the caller's `index_k_compressed` is then an **e4m3** buffer written by one more cast over the
`floor(S / 4)` rows of every sequence -- under `seq_lens` the rows `[floor(S_b / 4), floor(S / 4))` of a padded sequence receive
ZEROS (the bf16 form leaves them untouched); rows at or past `floor(S / 4)` stay the caller's in both forms. Below the identity
bound only the cache launches run (the bf16 compress into a staging slot, the cast into the cache: two launches where the bf16
form runs one); past it the census is the bf16 indexer's plus two casts (plus the cache cast). The selection is a different
FUNCTION of the inputs (3 mantissa bits: on synthetic activations the e4m3 selection agrees with the bf16 one on 99.0 % of the
blocks at S = 4096, 97.4 % at 32768 and 96.5 % at 131072, every flipped block a rank-boundary flip within ~5 % of the row's top
score; bf16 vs fp32 agrees at 99.8 %), so the dtype is a declaration attribute, never a knob, bf16 the default and the reference;
the model-quality question of serving it is the caller's. The e4m3 block's selection is pinned against the top-k of the fp64
scores of its own e4m3 operands (exact up to ties), against the fp32-operand oracle under the margin rule, and against the bf16
block on the same inputs (the agreement reported, every flip within 8 % of the row's top score); its first execute captures and
replays bitwise (`test_block_qsa_end_to_end.py`). The compress kernel's e4m3 store arm would fold the prefill form's casts away (a lever,
not built: the padded-row zero-fill contract decides).

**At decode over the pools (2026-10-21).** The in-block indexer's DECODE step ("The in-block indexer's DECODE step over the pools"
below) serves the attribute WITHOUT casts: the fused prepare's query + compress arms, the step form and the pool-1 compress STORE e4m3
directly -- each row is the bf16 form's packed words rounded once more by `cvt.rn.satfinite.e4m3x2.f32` (round-to-nearest-even,
saturating at +-448, never a NaN), i.e. **the e4m3 cache is the e4m3 cast of the bf16 row** -- plain e4m3 at unit scale, no scale field
anywhere; the ring stays bf16 (a bit copy). That is the serving stacks' indexer-cache chain and the prefill form's own convention, so a
paged cache a stack primes from the prefill form's e4m3 `index_k_compressed` rows and the step appends is ONE chain, bitwise (the
multi-step cell). The paged scorer reads the e4m3 pair on its fp8 `mma.sync.m16n8k32` arm (128-B rows, 4 k-steps, half the shared-memory
ring; every e4m3 x e4m3 product exact in fp32). `sample_index_kv_cache` / `index_kv_cache` are REQUIRED in the declared dtype (refused both
ways, at declaration and at execute); the `ix_q` workspace slot is carved at 1 B per element (every bf16 carve byte-identical); the launch
census is the bf16 form's (8 fused / 11 unfused; no cast kernel). The oracle is EXACT: the step's set == the fp64 top-k of the kernel's own
e4m3 `(ix_q, kbar)` up to ties, the cache rows bitwise the bf16 twin's cast, fused == unfused, two launches / the first-execute capture
bitwise, the multi-step uniform cache; the bf16-vs-e4m3 agreement is a statistic (every flipped block at the rank boundary). The envelope
publishes `decode.index_cache_dtypes == (bf16, e4m3)`; the e4m3 MAIN pool together with the indexer stays declined (`indexer_e4m3_pool`).
Performance (measured 2026-10-22/23: one exclusive Rubin cc 10.7 GPU, 212 SMs at the locked clock, every arm in one process per cell, A/B/A
CUDA-graph replay; speed-ups positive = faster): the lever of the e4m3 cache over the bf16 cache is the scorer's bytes -- +10.8 % (7.4 us)
of the decode step at B 32 x 128K x S_q 1 (the paged scorer 23.8 -> 16.8 us), +1.6 % geomean over 13 page-16 cells, +4.6 % over the B 32
cells, within +-1 % at B &lt;= 4 except the two B 1 x 8K cells (-1.1 / -1.0 %: the e4m3 path's fixed cost in the small prepare / score /
top-k kernels at a 0.26 MB cache, where halving the bytes buys nothing); the block over the e4m3 cache against the serving baseline's own
fp8 indexer path +7.5 % geomean over those 13 cells; the selection exact up to ties and the cache rows bitwise the bf16 twin's cast on
every measured cell. The e4m3 column is a different function in every table; bf16 stays the default and the reference.

**What it costs.** Past the bound the workspace grows by the indexer's queries (`T x index_heads x index_head_dim x 2` B),
the compressed keys (`B x S / 4 x index_head_dim x 2` B), the list, its sorted copy and its scores (`T x top_k x 12` B), the
row sort's index scratch and the scorer's compact-logits scratch, `~ B x S^2 / 8` fp32 (512 MiB per 32K-token sequence):
the price of one launch pair that never
materializes the dense score tensor; a row-windowed selection bounds it and is the lever for longer prompts.
`compile()` warms the scorer and the top-k at the real shape (their compile keys carry it), so `execute` compiles nothing
and allocates nothing of its own on either indexer-cache dtype (the e4m3 scorer's padded scale prefix is a plan constant);
the one allocator transient on the path is torch's: the row sort's 4 KB index template per 4096-row window (`torch.sort`
builds it even into preallocated outputs; an in-kernel row sort would remove it). The plan
constants (the query prefix, the scratch offsets, the counts, the identity list, the casts' scale word) are built on the host
at `compile()` and uploaded on the launch stream by the first `execute()` after it, never written at `compile()` on the ambient
stream; a CUDA graph that captures a block's FIRST execute records those uploads with their values (constants: a replay
re-writes the same ones), so warm the block up on the capture stream before capturing, as every capture of the block does.
`qsa_select_forward` itself creates its cached constants (the constant head weights, the packed-K row offsets, the e4m3 arm's unit scale
blobs) on the launch stream of the first call that needs them and orders a later call on another stream behind that fill through
an event recorded at the creation (not inside a capture, where no such event can be waited on -- the warm-up rule above covers it);
the head weights and the unit scale blobs are bucketed by power-of-two capacity (one flat allocation per bucket, the scorer handed a
view in the exact packed shape), so a serving process's distinct geometries keep O(log n) such constants resident, never one per shape.

**What the suite asserts** (`test_block_qsa_end_to_end.py`, the two-stage oracle): the selection as a SET against the
top-k of the scores recomputed in fp64 from the kernel's own bf16 queries and compressed keys up to ties; against the
fp32-operand oracle under the MARGIN RULE -- a block in exactly one of the two sets is accepted iff its oracle score lies
within `tol` of the row's k-th oracle score, `tol` the measured perturbation of the scores by the bf16 rounding of the
operands (not a tuned number); the same against the HF-rounding oracle (the pooled key cast to bf16 before its norm); the
queries and compressed keys within one bf16 ulp of the once-rounded fp32 chain; the count rule on every row; and attention
EXACT given the kernel's selection -- the block oracle on the kernel's list within the dense suites' budgets.

**The paged decode scorer (standalone).** The first piece of that decode-step indexer ships on its own:
`cudnn.ops.qsa_select_decode_paged_forward(qi, kbar, block_table, kv_lens, seq_q=1, top_k=512)`
scores a decode step's `seq_q` indexer query rows per sequence (`qi [B x seq_q, index_heads, 128]` bf16) against the compressed
keys of the complete 4-token blocks each row can see, read from the PAGED compressed-key cache the serving stacks keep on the
main block table (`kbar [num_pages, page_size / 4, 1, 128]` bf16: block `j` of sequence `b` is row `j % (page_size / 4)` of page
`block_table[b, j // (page_size / 4)]`; a power-of-two page size of at least 8), and selects the `top_k` per row with the
deterministic radix top-k. Row `s` of sequence `b` sits at position `kv_lens[b] - seq_q + s` and sees
`counts = clamp(floor((pos + 1) / 4), 0, n_blocks_max)` complete blocks; the scorer kernel writes `logits[t, 0:counts[t]]`
(`scale x sum_h relu(qi[t, h] . kbar[b, j])` in fp32: the ReLU-summed, head-reduced indexer score) and the counts on device -- no torch
prologue -- and the top-k returns `block_ids [T, top_k]` int32 in ASCENDING order with `-1` past `block_lens = min(top_k, counts)`
(ties to the smaller id), so two calls are bitwise. A `-1` table entry under a visible block reads as a zero key (logit 0);
a padding sequence (`kv_lens == 0`) writes nothing but its counts (0) and never reads the cache. The scorer kernel alone
(`kernels.qsa_score_paged.compile_qsa_score_paged` / `run_qsa_score_paged`: one warp-level `mma.sync.m16n8k16` per 8 keys x 16 D
over the `seq_q x index_heads` query rows padded to 16 -- the query tile staged once per CTA in shared memory behind one CTA
barrier, the keys streamed by `cp.async` through a per-warp ring into the B fragments -- with an L2 evict-first hint on those copies
where the installed DSL's bundled CUDA compiler is 13.4 or later, i.e. the cu13 toolchain of the 4.8.0 wheel, which the DSL binds only
under a CUDA 13 driver (`l2_hint`, a compile key of the kernel-level entry: `None` = that rule; the 13.3 PTX assembler of the public
4.7.0 cu13 wheel mis-encodes the hinted copy on sm_90 / sm_100 and the 12.9 assembler of the cu12 flavour -- what any wheel, 4.8.0
included, binds under a CUDA 12 driver -- emits the same encoding, so below 13.4 the plain copy is traced -- the same bytes, bitwise the
same logits; `True` forced on such a DSL is a typed decline naming the compiler it found, the wheel and the remedy, the cu13 flavour of
4.8.0 or later under a CUDA 13 driver; `False` the plain copy on any DSL); the programmatic-dependent-launch pair under `use_pdl`) runs
on any CuTe DSL device from sm_80; the
paired selector needs the radix top-k's cc 9.0. The e4m3 pair (`qi` and `kbar`
`float8_e4m3fn`: the fp8 indexer cache) is served since 2026-10-21 by the scorer's e4m3 READ arm -- `mma.sync.m16n8k32 .e4m3` on 128-B rows,
sm_89+, a different selection function chosen by the operands' dtype; a mixed pair is a form error; `units_per_warp` (1 / 2 / 4 / 8 units of 32 compressed rows per warp, default 1) and `use_pdl` are performance knobs of the
entry, `ring_depth` (2 / 4 / 8 n-tiles of keys in flight per warp, default 4) and `l2_hint` of the kernel-level compile -- each the same
function. The entry's module and its helpers import no CuTe DSL: the DSL floor, the `sm_107a` target on cc 10.7 and the radix top-k's
cc 9.0 are typed `NotImplementedError`s raised BEFORE the kernel module is imported, so the import path resolves on the public 4.6.2
wheel too;
`stream` takes a `torch.cuda.Stream`, a raw stream handle or `None` (torch's current stream), and every torch operation of the
call runs on it. The in-block binding of this scorer is the decode indexer's step ("The in-block indexer's DECODE step over the pools"
below, 2026-10-21); the dense-cache form of this selection is a follow-up.

**The CuTe DSL floor.** The Rubin sparse core needs the public `nvidia-cutlass-dsl` 4.8.0 wheel (the `sm_107a`
target); below it `check_support()` raises a typed `NotImplementedError` that names the installed version, BEFORE any
kernel module is imported -- never a `KeyError` from inside the DSL. An `index_source="indexer"` block raises the same
decline already at declaration (the declines paragraph above).

A prefill step with write-through into the stack's pools, the stack's own lists:

```python
import torch
from cudnn.gated_attention_block import GatedAttentionBlockFwd, GatedAttentionBlockGeometry, QsaSpec

geometry = GatedAttentionBlockGeometry(d_model=2560, h_q=24, h_kv=2, d_head=256, rope_dim=64, qsa=QsaSpec(top_k=512))
blk = GatedAttentionBlockFwd(h, w_qkvg, w_q_norm, w_k_norm, cos, sin, w_o, out, geometry,
                             return_lse=True, seq_lens_present=True, paged_kv_page_size=16)
blk.check_support()
blk.compile()
workspace = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device=h.device)
blk.execute(h, w_qkvg, w_q_norm, w_k_norm, cos, sin, w_o, out, workspace,
            seq_lens=seq_lens,                            # [B] int32: the KV length of every sequence (padding below S)
            lse=lse,                                      # [B, H_q, S] fp32, natural log
            block_ids=block_ids, block_lens=block_lens,   # [B * S, 512] int32 sequence-relative block ids, -1 padded; [B * S] int32
            k_cache=k_cache, v_cache=v_cache,             # [num_pages, 2, 16, 256] pools in h's dtype (HND; NHD by strides)
            slot_mapping=slot_mapping)                    # [B * S] int32 / int64 flat slots (page * 16 + offset); negative = no write
```

How a stack's vocabulary maps onto it (the shim is a renaming, not a conversion):

| the stack has | the block / core takes |
|---|---|
| the top-k output over compressed blocks -- vLLM's `block_indices [num_tokens, 512]` BEFORE its block-to-token expansion, SGLang's `topk_indices [rows, 512]` when its backend keeps block indices | `block_ids`, as is (int32, contiguous); no expansion to token ids |
| the stack's own indexer chain (the `index_qk_proj` GEMM, the compress, the scorer, the top-k) | `QsaSpec(index_source="indexer")`: the block's -- the band in the one projection GEMM, two compress launches, the scorer pair, the row sort; `block_ids_out` / `block_lens_out` hand the list back, `index_k_compressed` fills the stack's compressed-key cache |
| the per-row count -- vLLM's `min(visible_blocks, 512)` (the expand kernel's `complete_blocks`), SGLang's visible-block count of its row ranges | `block_lens` (optional; pass it whenever a list is shorter than the derived default) |
| vLLM's packed `[num_tokens, 2052]` buffer of token ids with the trailing count column | not a served form (the token-id arm is a follow-up) -- keep the block-id output of the top-k |
| `slot_mapping` / `out_cache_loc` | `slot_mapping` (flat slot = page x P + offset; negative = no write) |
| the page table at the stack's page granularity, `seq_lens` | `block_table`, `kv_lens` (the paged-READ mode: `kv_source="paged"`, the pools and the table as declaration samples -- the table CONTIGUOUS) |
| `k_scale` / `v_scale` (the per-layer DEscales of an fp8 KV cache; device tensors) | `kv_cache_dtype=torch.float8_e4m3fn` at declaration, `execute(k_scale=, v_scale=)` as 1-element fp32 CUDA tensors (read on device; a host float is refused) |
| the open tail, the causal mask, the per-row `visible_blocks` clamp | nothing -- the kernel derives all three from the position and the KV length |
| the output gate | nothing -- stage (5) is the block's (`sigmoid(gate)` after the dead-row select); a split merge is internal |

**The decode top-k kernel (standalone today).** `cudnn.gated_attention_block.qsa_topk.qsa_topk_decode(logits, counts, block_ids, *,
cluster_ctas=None, threads_per_cta=None, use_pdl=False, stream=None)` -- the DSL-free host entry: the kernel module
`kernels/qsa_topk_decode.py` behind it imports the CuTe DSL at module scope, so the entry gates the DSL version FIRST and imports the
kernel module lazily (a DSL below the floor reads as a typed `NotImplementedError` naming the version, never as an error from inside
the DSL; `qsa_topk_decode_kernel_module()` hands the gated module, with its plan-time / per-execute split, to an engine) -- selects,
per query row, the top-k block ids of the decode scorer's fp32 logits: `logits [T, n_blocks_max]` fp32 (row `t` scored in columns
`[0, counts[t])`; the columns past them never influence the result -- the kernel may read them, so they must be addressable, which
they are inside the tensor), `counts [T]` int32 (the VISIBLE-block count per row, clamped in-kernel to `[0, n_blocks_max]`; a row with more than
`top_k` visible blocks gets the score-based top-k, a row with at most `top_k` the identity list of every visible block -- so never pass
the list length `min(top_k, n_vis)` as the count: every row with more visible blocks than `top_k` would then read as exactly `top_k`,
take the identity path and return its oldest `top_k` blocks instead of a selection, with no error), `block_ids [T, top_k]` int32 -- the CALLER's buffer, nothing is allocated -- written in
CANONICAL order: the `min(top_k, counts[t])` selected ids strictly descending, then `-1`.  The set is the exact top-k of the fp32 values
(every selected logit at or above every unselected one; a tie at the k-th value keeps the smaller ids), so two launches are bitwise and
the list is the form the sparse core's list stage consumes without a sort.  One thread block per row (`threads_per_cta` 512 / 1024), or
the row split over a thread-block cluster of `cluster_ctas` 4 / 8 / 16 CTAs (1024 threads each; the histogram merged cluster-wide by
`st.async` + mbarrier, no release-ordered cluster barrier): the row's keys stay in registers, ONE 4096-bin histogram over the eight
octaves below the row maximum finds the k-th bin, its candidates are ranked exactly (an all-pairs rank up to 128 of them, narrowing
passes beyond), and a per-warp ballot compaction writes the canonical list.  Both knobs are
performance-only (every served value is the same function, pinned bitwise by the tests; no performance number is quoted until one is
measured on an exclusive GPU).  Typed declines: a CuTe DSL below the floor or without the device's target (the version / target named),
`use_pdl=True` below cc 9.0 (`griddepcontrol` exists from Hopper), a per-CTA shared-memory need above the device's opt-in limit (the
bytes named: `smem_bytes_for(cols_cap, cluster_ctas, threads_per_cta, top_k)` -- the 32 KB histogram plus a pool sized by the
fast path's staging regions (`threads / 32 x (top_k + 128)` words, at most 96 KB) or the fallback's list and ballot words, about
114 KB at 32768 columns x 1024 threads x `top_k` 512, above the 99 KB opt-in limit of some parts, which serve 512 threads); `cluster_ctas > 1` below cc 9.0 (clusters and DSMEM exist from Hopper) or on a public CuTe DSL wheel below 4.8.0 (the cluster
exchanges' vector `st.async` stores fail to compile on the released 4.7.0; the single-CTA form serves and the convenience entry picks it
there).  The declines are read in this order: the DSL floor, the geometry, the device's target, the device gates (clusters / DSMEM and
`griddepcontrol`, cc 9.0), the cluster form's DSL floor, the shared-memory limit -- a device gate before a form's DSL gate, so a part
below cc 9.0 is told what it lacks whatever wheel it runs; `ValueError` on any malformed form -- a cluster size outside
1 / 4 / 8 / 16, a thread count outside the domain (1024 for the cluster form), more than 64 keys per thread, a row block of more than 65535
rows.  The
in-block decode indexer binds it over the paged compressed-key cache ("The in-block indexer's DECODE step over the pools" below,
2026-10-21); the dense standalone selector keeps the radix top-k it binds today.


### Serving: write-through into a paged KV cache (`paged_kv_page_size`)

A serving stack keeps every token's post-norm, post-RoPE K and its V in a paged cache. `GatedAttentionBlockFwd(...,
paged_kv_page_size=P)` (appended; `0` = off, the default) declares that write-through: after norm + RoPE the block writes
each token's K and V row into the caller's pools at `execute(slot_mapping=)` -- ONE extra launch (two with the indexer band's
raw-key pool), or NONE with the knob `fuse_prepare=True`, which runs norm + RoPE and the write-through as a single launch that
is bitwise the two-stage form -- and then attends over its own K / V exactly as before, so `out` is bitwise the block's
without it and `get_workspace_size()` does not change. It is a declaration ATTRIBUTE (it changes the input contract), never a
knob (`fuse_prepare` is the knob: it changes no input and no output, only the launch count). Dense `[B, S, d_model]` and packed
(`thd=True`) blocks alike: the stage addresses tokens, so the packed block (`B = 1, S = T` inside) runs the same launch over the
same `[T, H_kv, D]` views.

| input (appended to `execute`) | shape / dtype | contract |
|---|---|---|
| `k_cache`, `v_cache` | `[num_pages, H_kv, P, D]` views in the activation dtype (bf16 / f16), `D` contiguous, the other three strides FREE: HND compact `(H*P*D, P*D, D, 1)`; NHD storage by strides `(P*H*D, D, H*D, 1)`; or a stack's UNIFIED content row `[num_pages, H_kv, P, 2D]` with K at `+0` and V at `+D` (token stride `2D`) -- `kv_cache[..., :D]` and `kv_cache[..., D:]` are the two views. Every non-innermost stride a whole number of 16-byte vectors and a 16-byte-aligned base (a typed `ValueError` otherwise: each row is written in 16-byte vectors) | REQUIRED under `paged_kv_page_size > 0`; post-RoPE K lands in `k_cache`, V in `v_cache`; nothing is repacked |
| `slot_mapping` | `[T]` int32 or int64, contiguous; the flat slot `page * P + offset` of every token (`T = B*S`, or the packed token count) | REQUIRED; a NEGATIVE slot (a padded token) writes nothing; a slot past the pool's capacity writes nothing either -- never a write past the pool; slot VALUES are device data, never read on the host |
| `index_k_raw` | `[num_pages, P, index_head_dim]` activation dtype, `D` contiguous: the raw indexer key's pool, written at the same slots | REQUIRED iff the geometry declares the indexer band (`QsaSpec.index_band`), refused otherwise |
| `block_table`, `kv_lens` | the paged-READ modes' page table and logical KV lengths ("Serving: the extend form" / "Serving: the decode form" below) | REQUIRED under `kv_source="paged"`; refused (`ValueError`) on a block declared `kv_source="self"`, which attends over its own K / V |

`P` is a positive multiple of 16 (the serving stacks' page granularity; a 4-token sparse block then never straddles a page);
a block declared without the attribute REFUSES every one of these inputs (`ValueError`) rather than ignoring a pool it would
never write. **`fuse_cache_write`** (appended; `None` = auto): on the bf16 / f16 fused norm + RoPE pipeline without an indexer band the write-through
is FOLDED into the fused projection's epilogue -- every K / V tile's post-norm, post-RoPE row is stored to the pools from the same registers the
slab store reads (the pool row is the slab row, bitwise the separate kernel; a negative slot or one past the pools writes nothing; no new
barrier, no new SMEM), one launch fewer per step (EXTEND 3 stages with `fuse_gate=True`, DECODE 4); `False` keeps the separate cache-write
launch (the control; on the DECODE form's unfused chain it also keeps the AUTO small-M fold of `fuse_cache_write_small_m=None` off -- an
explicit `fuse_cache_write_small_m` names that arm itself and is followed either way); `True` on the unfused chain, a quantized pipeline or an
indexer band is a typed `NotImplementedError`; `blk.cache_write_fused`
says what runs. The slot dtype (int32 / int64) is a compile key of the fused arm as of the separate kernel: both are compiled.
Served on the bf16 / f16 inference pipelines (`inplace_qkv` on or off, `fuse_norm_rope` on or off, `fuse_gate`
on or off, dense or packed); validated on Rubin at pages 16 / 32 / 64 / 128 on all three pool layouts, dense and packed, bf16
and f16 (the live pool rows bitwise the K / V rows attention consumed; padded tokens unwritten). `quant` (the pools of a
quantized cache arrive with a cache-dtype attribute and its cast kernel) and `save_for_backward` with `paged_kv_page_size` are
typed `NotImplementedError`s naming the feature; a CuTe DSL below FROST's floor is declined at `check_support`, named with
the installed version, before the kernel module is imported.

`fuse_prepare=True` needs the attribute (a `ValueError` without it), the in-place pipeline (`inplace_qkv=True`, a `ValueError`
otherwise) and not `fuse_norm_rope` (a typed `NotImplementedError`: the norm + RoPE already live in the projection epilogue
there); `thd=True`, `kv_source="paged"` on a DENSE geometry (the DECODE form) and `kv_cache_dtype=torch.float8_e4m3fn` together with it
are typed `NotImplementedError`s naming the knob (the prepare kernel's packed arm and the dense DECODE form's are follow-ups; the e4m3
cache's write-through quantizes in its own stage; on the block-sparse DECODE mode over the pools the knob is SERVED -- "The in-block
indexer's DECODE step over the pools" below); `rope_dim=0` with the knob is a `ValueError` at `check_support`, with or without `qk_norm`, naming the zero-width RoPE table
operands: the fused launch binds `cos` / `sin` as `[T, rope_dim]` operands and its artifact is traced with one-column tables, so a
`[T, 0]` table is not an operand it takes.  The fused stage takes `use_pdl` as the two stages it replaces do (the attribute and the
kernel's griddepcontrol pair).  Its
kernel (`kernels/qsa_prepare.py`) also carries the indexer's decode-step arms -- the queries' norm + RoPE, the raw-key ring and
the incremental compress of the compressed-key cache -- which the block binds when its decode mode lands.

### Forward

```python
blk = GatedAttentionBlockFwd(
    sample_h, sample_w_qkvg, sample_w_q_norm, sample_w_k_norm, sample_cos, sample_sin, sample_w_o, sample_out,
    geometry,
    return_lse=False,            # also write the softmax stats [B, H_q, S] fp32
    save_for_backward=False,     # keep what GatedAttentionBlockBwd needs (implies return_lse)
    seq_lens_present=False,      # per-batch KV lengths (padding) at execute time
    inplace_qkv=None,            # None -> not save_for_backward: norm/RoPE Q,K in place in the projection slab
    fuse_norm_rope=False,        # stages (2)+(3) inside the projection epilogue
    fuse_gate=False,             # stage (5) inside the SDPA epilogue
    fuse_cache_write=None,       # serving: where the paged write-through runs -- None = folded into the fused projection's epilogue where that arm serves, False = its own launch (the AUTO small-M fold off too), True = the fused arm (at decode row counts the request keeps the fork) or a typed decline -- "Serving: write-through into a paged KV cache"
    quant=None,                  # QuantSpec (FP8) | MxQuantSpec (MXFP8, + fp4 weights / fp4 O) | None (bf16 / fp16)
    sample_h_sf=None, sample_w_qkvg_sf=None,   # MXFP8 scale-factor blobs
    sample_w_o_sf=None,          # fp4 O only (MxQuantSpec.o_fp4): the e2m1 W_o's scale blob, in O's format
    saved_gate_copy=False,       # training save mode: True copies the GATE band into a compact saved.gate (see below)
    thd=False,                   # PACKED sequences: h / cos / sin / out are [T, .] (or [1, T, .]) token matrices -- see "Packed sequences (THD)"
    num_sequences=None,          # THD only, REQUIRED there: B, the number of sequences in the packing
    max_seq_len=None,            # THD only, REQUIRED there: S_max, the longest sequence the plan admits
    cu_seqlens=False,            # THD only: execute(seq_lens=) is [B+1] int32 prefix sums instead of [B] int32 lengths
    paged_kv_page_size=0,        # serving: write the post-RoPE K / V through into paged pools at execute(slot_mapping=) -- see "Serving: write-through"
    kv_source="self",            # serving: "paged" = stage (4) attends over the POOLS via execute(block_table=, kv_lens=): the EXTEND form (packed) or the DECODE form (dense) -- "Serving: the extend form" / "Serving: the decode form"
    token_capacity=None,         # serving, THD only: the packed rows are a CAPACITY (lengths may sum to T_live <= T); must equal T when given
    sample_k_cache=None, sample_v_cache=None, sample_block_table=None,   # kv_source="paged" only: the pools' and the table's shape / dtype / strides (compiled in; any table row stride)
    max_kv_len=None,             # the DECODE form: the logical KV bound kv_lens[b] <= max_kv_len; None = max_pages * P
    decode_split_kv=None,        # the DECODE form: the decode tile's split-KV; None = the decode split model (fp32 partials in the workspace + one combine)
    fuse_prepare=None,           # serving: stages (2)+(3) and that write-through as ONE launch (needs paged_kv_page_size > 0; same function); None = False on the dense forms and the caller-list band form, the MEASURED default on the indexer's DECODE step over the pools (True on cc 10.7 since 2026-10-21 where the fused form is admissible, False out of place and on other parts) -- "The in-block indexer's DECODE step over the pools"
    kv_cache_dtype=None,         # serving: None = the activation dtype; torch.float8_e4m3fn = the e4m3 KV cache (per-tensor k_scale / v_scale at execute) -- "Serving: the e4m3 KV cache"
    decode_route_policy=None,    # the DECODE form: None = "single_unit" (the decode tile in one unit, else the packed prefill tile where the step fills the part); "token_units" = the adapter's route -- "Serving: the decode form"
    decode_q_tile_max=None,      # the DECODE form: None = the decode tile's whole width table (16, 32, 64); 32 = the route before the 64-column width landed (a performance-only cap)
    use_pdl=None,                # programmatic dependent launch over the chain: None = True on the DECODE form's decode-tile routes on cc 10.7 (the measured default), False everywhere else; True / False explicit -- "Serving: the decode form"
    fuse_norm_rope_small_m=None, # the DECODE form: None = fuse_norm_rope=True LOWERS stage (1) to the unfused chain at B x S_q <= 128 (the measured faster step); True = the fused fork as declared
    fuse_cache_write_small_m=None, # the DECODE form's unfused chain: None = the paged write-through folded into the norm + RoPE launch at B x S_q < 128 rows or S_q = 1 (the measured rule), its own launch past that and under an explicit fuse_cache_write=False; True / False explicit
    sample_index_kv_cache=None, sample_index_state_cache=None, sample_index_state_block_table=None,   # the in-block indexer's DECODE step over the pools (QsaSpec(index_source="indexer") with kv_source="paged"), REQUIRED there, refused elsewhere: the paged compressed-key cache [n_pages, P / 4, 1, 128], the ring of raw indexer keys [blocks, cap >= S_q + 3, 1, 128], its [B, 1] int32 table -- "The in-block indexer's DECODE step over the pools"
    index_state_slot_mapping_present=False,   # that step: execute(index_state_slot_mapping=) names the ring's write rows (block x cap + pos % cap, -1 dead; slot_mapping's dtype) instead of the rows the kernels derive from the ring table
    index_commit_present=False,  # that step: a per-sequence commit count at execute (a follow-up: True is a typed NotImplementedError)
    rope_position_tables_only=False,   # that step under fuse_prepare=True: cos / sin REFUSED at execute; the fused launch rotates Q / K and the indexer queries from cos_pos / sin_pos at the device-derived position (the unfused twin with it is a typed decline)
)
workspace = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device=h.device)
blk.execute(h, w_qkvg, w_q_norm, w_k_norm, cos, sin, w_o, out, workspace,
            seq_lens=None,       # dense: the [B] int32 KV lengths iff seq_lens_present; THD: REQUIRED, the [B] lengths / [B+1] prefix sums; REFUSED on the DECODE form
            lse=None, saved=None, current_stream=None, h_sf=None, w_qkvg_sf=None,
            w_o_sf=None,         # required iff MxQuantSpec.o_fp4, refused otherwise
            k_cache=None, v_cache=None, slot_mapping=None,   # REQUIRED iff paged_kv_page_size > 0, refused otherwise -- "Serving: write-through"
            block_table=None, kv_lens=None,                  # REQUIRED iff kv_source="paged", refused otherwise -- "Serving: the extend form" / "Serving: the decode form"
            num_tokens_live=None,                            # the bounded packed form's LIVE token count, a host int (None = the capacity) -- "Serving: the extend form"
            k_scale=None, v_scale=None,                      # REQUIRED iff kv_cache_dtype=torch.float8_e4m3fn (1-element fp32 CUDA tensors), refused otherwise
            w_iq_norm=None, w_ik_norm=None,                  # REQUIRED iff QsaSpec(index_source="indexer"), refused otherwise -- "The in-block indexer"
            index_k_compressed=None, block_ids_out=None, block_lens_out=None,   # the indexer's optional outputs: the PREFILL form's compressed-key rows (refused at the decode step: the cache is paged), the list and its counts
            index_kv_cache=None, index_state_cache=None, index_state_block_table=None,   # REQUIRED iff the indexer's DECODE step over the pools, refused otherwise (16-byte bases; the declared forms) -- "The in-block indexer's DECODE step over the pools"
            index_state_slot_mapping=None,                   # REQUIRED iff index_state_slot_mapping_present (slot_mapping's dtype), refused otherwise
            cos_pos=None, sin_pos=None,                      # REQUIRED on that step: [1 or B, >= max_pages x P, rope_dim] position tables -- the table is the CEILING on positions; refused elsewhere
            n_commit=None)                                   # refused (index_commit_present=True is a follow-up)
```

Every appended argument (`sample_w_o_sf`, `w_o_sf`, `saved_gate_copy`, the four packing knobs `thd` / `num_sequences` /
`max_seq_len` / `cu_seqlens`, the serving attributes `paged_kv_page_size` / `kv_source` / `token_capacity` / `kv_cache_dtype` with their samples, the DECODE
form's `max_kv_len` / `decode_split_kv`, the knob `fuse_prepare`, the execute inputs, `num_tokens_live`, `k_scale` / `v_scale`, the indexer's five
execute inputs and -- last, 2026-10-21 -- the decode indexer's three samples + three flags and its seven execute inputs) sits at the end with a default,
so positional callers of the bf16, FP8 and MXFP8 pipelines are unchanged; `MxQuantSpec.o_fp4` and `sample_w_o_sf` must be given
together (a typed `ValueError` names the missing half), and `num_sequences` / `max_seq_len` / `cu_seqlens` are refused without
`thd=True`.

#### Training forward (`save_for_backward=True`)

The UNFUSED pipelines, out of place: bf16 / fp16, and the per-tensor FP8 (`QuantSpec`) and MXFP8 (`MxQuantSpec`) pipelines
(`inplace_qkv` defaults to `False` there; `fuse_norm_rope` / `fuse_gate` are typed declines; the MXFP8 pipeline's fp4 modes train too
and write the MXFP8 record byte for byte -- see "The fp4 weight modes' backward"). The block
**writes through** the caller-owned `SavedForBackward` record wherever the backward needs a tensor, with the same kernels as
inference (`out` is bitwise the inference block's): the projection GEMM writes `saved.proj_slab`, the SDPA writes the
**pre-gate** `saved.o` and `saved.lse`, norm+RoPE writes `saved.rstd_q` / `rstd_k`, and the sigmoid gate lands out of place in
the workspace. `execute()` still allocates nothing.

Under FP8 / MXFP8 the record is the SAME record the bf16 forward writes -- a bf16 `proj_slab` (the dequantized stage-(1)
product) whose Q/K bands are **pre-norm**, the bf16 pre-gate `o`, the exact fp32 `lse`, `rstd_*` -- with `h` the caller's e4m3
codes (`h_sf` is a forward input, never a record field). One routing differs from quantized inference: norm+RoPE writes the
normed Q/K out of place into two compact bf16 workspace slots (+17 KiB/token at the 397B geometry) instead of back over the
slab, and the quantize stages read those slots, so the slab keeps the pre-norm bands the backward differentiates; the launch
count (9) and the bytes moved are unchanged, and `out`, `o`, `lse` and the GATE / V bands are bitwise the quantized inference
forward's. `GatedAttentionBlockBwd` (bf16 / fp16) consumes such a record given the **dequantized** bf16 `h` and weights
(`dataclasses.replace(saved, h=h_dequantized)`; a record handed through with its e4m3 `h` is a typed `ValueError` naming that
contract and the `quant=QuantSpec` declaration); under `quant=QuantSpec` the backward differentiates the per-tensor fp8 record
natively, as written, and under `quant=MxQuantSpec` the MXFP8 record (see Backward).

What that backward computes over a quantized record -- the numerics contract. The record's `o` and `lse` are the quantized
SDPA's: computed over the e4m3 `q8` / `k8` / `v8` the forward quantized, with the kernel's e4m3 `P`. The bf16 backward
recomputes bf16 Q / K from the pre-norm slab bands (and reads the slab's bf16 V), differentiates the bf16 chain through
them, and recomputes `P = exp(S - lse)` from the bf16 scores against the quantized `lse`, so `P` no longer row-normalises
exactly. Its gradients are therefore the bf16 chain's gradients evaluated at the quantized forward's `o` / `lse` -- a
straight-through-style approximation whose distance from the exact gradient of the dequantized bf16 model is of the order
of the fp8 quantization error of Q / K / V / `P` -- not a bf16-accurate gradient of the dequantized model; `dW_o` inherits
the forward's own `o` error on top (it contracts `dy` with the gated `o` the model actually produced). The test holds the
result to the module's bf16 bounds against an fp64 oracle seeded with the record's `o` / `lse` (the exact function of the
record) and reports its cosine against the unquantized fp64 chain.

```python
from cudnn.gated_attention_block import SavedForBackward, saved_slab_views

B, S, T = h.shape[0], h.shape[1], h.shape[0] * h.shape[1]
g = geometry
blk = GatedAttentionBlockFwd(h, w_qkvg, w_q_norm, w_k_norm, cos, sin, w_o, out, g, save_for_backward=True,
                             seq_lens_present=seq_lens is not None)          # saved_gate_copy=False: the proj_slab save mode
blk.check_support(); blk.compile()
workspace = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device=h.device)  # no slab, no O: the record holds them

proj_slab = torch.empty(T, g.n_qkvg, dtype=h.dtype, device=h.device)         # [B*S, n_qkvg] (or [B, S, n_qkvg]), contiguous
q_pre, gate, k_pre, v = saved_slab_views(proj_slab, g, B, S)                 # zero-copy [B, S, heads, D] views of its bands
saved = SavedForBackward(
    h=h,                                                                     # the SAME tensor execute() runs on (verified)
    gate=gate, q_pre=q_pre, k_pre=k_pre,                                     # the views above, or None (the backward derives them)
    o=torch.empty(B, S, g.h_q, g.d_head, dtype=h.dtype, device=h.device),    # PRE-gate O, compact
    lse=torch.empty(B, g.h_q, S, dtype=torch.float32, device=h.device),      # natural-log LSE; `lse=` at execute is optional
    rstd_q=torch.empty(B, S, g.h_q, dtype=torch.float32, device=h.device),   # both None iff geometry.qk_norm is False
    rstd_k=torch.empty(B, S, g.h_kv, dtype=torch.float32, device=h.device),
    proj_slab=proj_slab,
    seq_lens=seq_lens,                                                       # the SAME tensor passed to execute(seq_lens=), or None
    # seq_lens_form=None: a dense record (seq_lens is a padding mask or None); "lengths" / "prefix" under thd=True (below)
)
blk.execute(h, w_qkvg, w_q_norm, w_k_norm, cos, sin, w_o, out, workspace, seq_lens=seq_lens, saved=saved)
```

A runnable version of this recipe, self-checked against torch on the record alone (the slab is `h @ W_qkvg^T`, `rstd_*` are the
RMSNorm statistics of the pre-norm bands, `out` is `(o * sigmoid(gate)) @ W_o^T`):
[`samples/frost/gated_attention_block/00_training_forward.py`](../../../samples/frost/gated_attention_block/00_training_forward.py).

Two **save modes**, chosen at declaration because the workspace carve differs (`execute` checks that the record agrees):

| mode | declaration | what is saved | forward cost |
|---|---|---|---|
| **proj_slab** (default) | `saved_gate_copy=False`; `saved.proj_slab` REQUIRED | the whole stage-(1) slab (34 KiB/token at the 397B geometry): `gate`, `q_pre`, `k_pre` and V are its column bands, nothing to recompute | none -- the GEMM writes the slab in place of the workspace one (6 launches, the inference chain's) |
| **gate-copy** | `saved_gate_copy=True`; `saved.proj_slab=None`, `saved.gate` a compact `[B, S, H_q, D]` buffer | the GATE band (16 KiB/token); `q_pre` / `k_pre` only if you pass buffers for them (else the backward recomputes them from `h`, `RecomputePolicy.RECOMPUTE_QK_PRE`) | one extra elementwise launch per copied band; the workspace keeps its slab |

Recommendation: `proj_slab` up to `S = 32K`, gate-copy beyond (the whole slab, 34 KiB/token at the 397B geometry, is about twice the GATE band alone).

Contracts verified before any launch, each a `ValueError` naming the field: `saved.h` is `h`'s storage; `saved.seq_lens`
**is** the `seq_lens` tensor passed to `execute` (or both `None` -- the backward declines padding at declaration from this
field, without a device read); `lse` (optional) is `saved.lse`'s storage; `saved.o` is compact `[B, S, H_q, D]` in the
activation dtype; `saved.rstd_q` / `rstd_k` are `[B, S, H]` fp32 compact (present iff `geometry.qk_norm`); every caller
buffer a kernel writes (`proj_slab`, `o`, `lse`, `rstd_*`, the gate-copy targets) is **16-byte aligned** -- they are TMA-store
targets, and a slice at an odd element offset is refused here rather than failing untyped after stage (1) launched; in the
proj_slab mode `saved.gate` / `q_pre` / `k_pre`, when given, alias `saved.proj_slab` exactly as `saved_slab_views` spells
them (each may be `None` there: the backward derives it from the slab). The refusal runs the other way too: `saved=` on a
block declared **without** `save_for_backward` is a `ValueError` naming the knob -- an inference forward writes none of the
record's tensors, so a silently ignored record would reach the backward uninitialised. A padded forward (`seq_lens`) is
served: a dead entry (`seq_lens[b] == 0`) leaves `saved.o[b] == 0`, `saved.lse[b] == -inf` and `out[b] == 0` exactly.

`execute()` allocates nothing, reads nothing back to the host and converts nothing: every intermediate is a
strided view of the caller's workspace, sized honestly by `get_workspace_size()`, so the call is CUDA-graph
friendly. All stages run on one launch stream (torch's current stream, or `current_stream`), on both the FROST
GEMM route and the DSL stages. Dtype, layout and shape mismatches, an unsupported architecture, or a feature the
selected specialization cannot serve raise typed errors at construction (`check_support`) rather than at launch.

Quantization specs:

- `QuantSpec(descale_h, descale_w_qkvg, descale_w_o, scale_q, scale_k, scale_v, scale_o, dtype=torch.float8_e4m3fn)`
  — static per-tensor scales for the FP8 pipeline.
- `MxQuantSpec(descale_w_o, scale_o=1.0, dtype=torch.float8_e4m3fn, block_size=32, w_qkvg_dtype=torch.float8_e4m3fn,
  o_fp4=None)` — the MXFP8 pipeline; the MXFP8 SDPA writes e4m3 `O` unscaled, so the fully fused path needs
  `scale_o == 1.0`. The two appended fields select the fp4 modes:
  - `w_qkvg_dtype`: `torch.float8_e4m3fn` (default, MXFP8 x MXFP8) or `torch.float4_e2m1fn_x2` (MXFP4 `W_qkvg`, the
    mixed row); any other dtype is a typed `NotImplementedError`.
  - `o_fp4`: `None` (default, per-tensor e4m3 `O`) or an `Fp4Format` member (anything else is a `TypeError`). Under
    `o_fp4` neither side of the out projection has a per-tensor scale -- the quantizer writes block scales only and
    `W_o` dequantizes through `w_o_sf` in the MMA -- so **`scale_o` and `descale_w_o` must both be `1.0`** (typed
    `ValueError` otherwise; a non-unit value would be silently dropped by the block-scale GEMM).
- `Fp4Format` (enum; one member = codes x scale dtype x block, so an illegal pairing cannot be spelled):

  | member | codes | scale dtype | scale block along K |
  |---|---|---|---|
  | `Fp4Format.NVFP4` | e2m1 | `torch.float8_e4m3fn` | 16 |
  | `Fp4Format.MXFP4` | e2m1 | `torch.float8_e8m0fnu` | 32 |

  Properties: `block_size`, `sf_torch_dtype`, `sf_cudnn_dtype` (and `fmt_name`, the kernel's format key). Neither
  format carries a global (per-tensor) scale: an NVFP4 block's e4m3 scale is `max(amax * fp32(1/6), 2^-9)` (the
  e4m3 min-subnormal floor keeps an all-zero block's scale NONZERO, so the encode `x / scale` stays finite), an
  MXFP4 block's E8M0 scale is the power of two at or above `amax * fp32(1/6)` -- `fp32(1/6)`, not an exact `/ 6`:
  the kernel and the reference multiply by the fp32 constant, and the two differ by one ulp; a dead (fully masked or
  zero-length) row quantizes to codes `0` exactly. The `O` codes are round-to-nearest-even on the e2m1 grid,
  saturating at 6.

**Per-step recalibration -- `update_quant_scales`.** The scales of a `QuantSpec` / `MxQuantSpec` are plan-time constants, and a
training step moves them: the weights change, so `descale_w_qkvg` / `descale_w_o` (and `descale_h`, the normed residual stream's)
are recomputed from the tensors quantized this step, and a per-tensor fp8 recipe recalibrates `scale_q` / `scale_k` / `scale_v` /
`scale_o` from a previous step's activation amax -- the block itself runs no amax pass, the caller computes them from the record it
already holds. `GatedAttentionBlockFwd.update_quant_scales(spec, *, current_stream=None)` re-points a COMPILED block at a new spec
without recompiling: an in-place write of the block's device scalars (the GEMM alphas, the quantize scales, the fp8 SDPA's descales,
and the fully fused fp8 fork's scale vector) on the launch stream (`current_stream`, else torch's current stream), then
`blk.quant = spec`; every consumer binds those tensors at `execute`, so the next execute on that stream runs at the new scales, and a
CUDA graph that captured an execute of a warmed-up block replays with the live values -- unless the captured execute was the block's
first: its one-time scalar write (one fill kernel per value; the first-execute write below) is captured with the capture-time values,
so every replay re-writes them and overrides a later eager `update_quant_scales`; warm up with one eager execute on the capture stream
before capturing (as every capture of the block does), or treat such a graph as pinned to the scales it captured and re-capture after
a recalibration. `GatedAttentionBlockBwd.update_quant_scales(spec)` is the
backward half: the plan-time constants the next execute's prologue launch stores from its kernel arguments are re-resolved on the
host (no device write; a CUDA graph captured before the call keeps the old constants -- re-capture after a recalibration). Typed
refusals, before any write: a block declared without `quant` (`ValueError`), a spec of the other class (`TypeError`), a differing
plan fact -- `dtype`, `block_size`, `w_qkvg_dtype`, `o_fp4` -- (`ValueError` naming it: those select kernels and the carve), the
spec class's own `validate` exactly as the declaration applied it (the fully fused MXFP8 path keeps `scale_o == 1.0`; a zero / inf /
NaN scale is its `ValueError`), and a block not yet compiled (`RuntimeError`); under `o_fp4` there is nothing to write and the call
validates and records it. The per-layer recipe of a training loop: one compiled block pair per dtype configuration,
`fwd.update_quant_scales(spec_l)` right before layer `l`'s forward and `bwd.update_quant_scales(spec_l)` right before its backward
(autograd runs the backward long after every layer's forward, so the step's spec travels with the layer), `spec_l`'s `descale_*`
from the tensors quantized this step and its activation scales from the previous step's record of that layer. `compile()` only
allocates those scalars; their values are written on the launch stream -- by the first `execute` of a compiled block and by every
`update_quant_scales` -- never on the stream ambient at compile time, and every such write first waits on an event `compile()`
records behind its allocations (under `torch.use_deterministic_algorithms(True)` an allocation itself fills with NaN on that
stream), so a block compiled on one stream and recalibrated or first executed on another reads what its execution stream wrote,
and nothing enqueued at compile time can land later and undo an update. Inside a CUDA-graph capture that wait is skipped: warm the
block up with one eager execute before capturing (the rule below).

#### Packed sequences (THD)

`thd=True` runs the block over ONE packed token matrix holding `B` sequences back to back -- the layout a varlen caller
already holds (TransformerEngine's and FlashAttention's "THD"): no padding, no per-batch axis, the per-sequence lengths as
an int32 tensor.

```python
lens = [300, 128, 200]                                   # B = 3 sequences, T = 628 packed tokens
h = torch.empty(sum(lens), d_model, device=dev, dtype=torch.bfloat16)                   # [T, d_model] or [1, T, d_model]; out likewise
cos, sin = ...                                           # [T, rope_dim] (or [1, T, rope_dim]): PER-TOKEN tables, positions restarting at every sequence
seq_lens = torch.tensor(lens, dtype=torch.int32, device=dev)                            # [B] lengths ...
cu_seqlens = torch.tensor([0, 300, 428, 628], dtype=torch.int32, device=dev)            # ... or [B+1] prefix sums (cu_seqlens=True; any base)
blk = GatedAttentionBlockFwd(h, w_qkvg, w_q_norm, w_k_norm, cos, sin, w_o, out, geometry,
                             thd=True, num_sequences=len(lens), max_seq_len=max(lens))  # + save_for_backward=True for training
blk.check_support(); blk.compile()
workspace = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device=dev)
blk.execute(h, w_qkvg, w_q_norm, w_k_norm, cos, sin, w_o, out, workspace, seq_lens=seq_lens)  # REQUIRED under thd
```

- **Shapes.** `sample_h` / `h`, `cos` / `sin`, `out` are `[T, .]` or `[1, T, .]`. Internally the block is a `B = 1, S = T`
  block: every token-wise stage (the two projections, norm + RoPE, the gate, the quantize passes) is the SAME launch as the
  dense block's -- at `B = 1` the packed block is bitwise the dense `B=1, S=T` one -- and only the SDPA runs its packed
  specialization (the FROST d256 THD arm, `SCHED_NATURAL`). The workspace carve is the dense `B=1, S=T` block's plus the
  packed SDPA's own scratch (metadata and per-sequence descriptors; `get_workspace_size()` reports it).
- **Lengths.** `execute(seq_lens=)` is REQUIRED: a contiguous 1-D int32 CUDA tensor on `h`'s device with exactly `B`
  entries (`cu_seqlens=False`, lengths) or `B+1` (`cu_seqlens=True`, prefix sums -- normalized to their first entry on the
  device, so a tensor sliced from a larger prefix works). It is handed to the SDPA as BOTH the Q-side and the KV-side
  lengths (self-attention) and is never read on the host, so the block stays CUDA-graph capturable: a replay may carry
  NEW lengths written through the captured tensor, as long as `B` is unchanged, every length is `<= max_seq_len` and the
  lengths still sum to `T` (rewrite `cos` / `sin` for the new packing too).
- **RoPE tables are per token.** Build `cos` / `sin` per sequence (positions `0 .. len_i - 1`) and concatenate along the
  token axis. A dense `[B, S, rope_dim]` table, or positions running across sequence boundaries, is a plausible-but-wrong
  RoPE after the first sequence.
- **Declaration bounds** (typed `ValueError`): `num_sequences >= 1`, `2 <= max_seq_len <= T` (`S = 1` is decode, out of
  the prefill bodies' scope; on cc 10.7 the standalone SDPA forward serves decode-shaped `d = 256` graphs on its
  decode tile -- `docs/operations/Attention.md`, "Paged attention" -- and the block serves dense paged declarations through
  the DECODE form described under "Serving: the decode form" below) and
  `num_sequences * max_seq_len >= T` -- a smaller product would silently cap the SDPA
  chain's packed capacity below `T` (the tokens past it are not processed, with no message downstream). `seq_lens_present`
  is mutually exclusive with `thd` (there is no per-batch KV padding mask under THD); the three packing knobs on a dense
  block are refused.
- **Caller contract on the lengths** (device data, never validated on the host): every length in `[0, max_seq_len]`,
  non-decreasing prefix sums, and **`sum(lengths) == T`**. Zero-length sequences are served (a caller whose `B` varies
  pads with empty sequences; an empty sequence owns no rows). `sum == T` is a contract, not a capacity: the SDPA writes
  nothing past the live total (rows `[cu[B], T)` of `saved.o` / `saved.lse` stay untouched), but the token-wise stages run
  over all `T` rows and the backward's weight-gradient GEMMs contract over all `T` rows, so lengths summing below `T`
  contaminate `dW_qkvg` / `dW_o` -- finite or NaN. Declare `T` as the live total.
- **Training record.** The record is the dense record at `(1, T)` -- `proj_slab [T, n_qkvg]`, `o [1, T, H_q, D]`,
  `lse [1, H_q, T]` (head-major, head stride exactly `T`: the ONE packed Stats layout the backward reads), `rstd_*
  [1, T, H]` -- plus `saved.seq_lens` (REQUIRED: the lengths tensor `execute` ran with, identity-verified) and
  `saved.seq_lens_form` (`"lengths"` / `"prefix"`, matching `cu_seqlens`; `None` is the padded dense record's value and is
  refused under `thd`). Both save modes serve.
- **Served / declined.** Served: bf16 / fp16 (inference and training, in place and out of place), the per-tensor FP8
  unfused pipeline (`QuantSpec`; inference and training -- a packed FP8 training forward writes the same bf16 record at
  `(1, T)` as the dense quantized training forward, with `saved.h` the e4m3 `h`; the packed per-tensor fp8 backward
  (`quant=QuantSpec`) differentiates it as written, the packed bf16 backward given the dequantized bf16 `h` and weights,
  exactly as on the dense side), the MXFP8 unfused
  pipeline (`MxQuantSpec`, the fp4 modes `w_qkvg_dtype` / `o_fp4` included; inference and training, the same bf16 record
  at `(1, T)`): its three `quantize_mxfp8` stages run their PACKED arm and write the Rubin d256 MXFP8 SDPA's
  per-sequence-tile-padded scale-factor layout -- per head the tiles of every sequence in `cu_seqlens` order, V's two
  D-planes adjacent inside a tile, the slot sized at the capacity `H * ((T + 127 * B) // 128) * 1024` bytes (the only
  carve difference to the dense `B=1, S=T` block; slack tiles and pad bytes `0x00`), each tile's sequence resolved on
  device from the lengths tensor (no cap on `B`; at `B = 1` the packed block is bitwise the dense one on its outputs, record and e4m3 payload, and V's scale-factor slot holds the dense atoms in plane-adjacent order); `fuse_norm_rope`
  (bf16 / fp16 inference in place: the projection fork norms and rotates per token with the per-token tables); `fuse_gate`
  (bf16 / fp16 inference: stage (5) folds into the packed SDPA's epilogue through the SDPA's THD gate descriptor -- the
  packed block at `B = 1` is bitwise the dense gated block, the fused gate within one output rounding of the separate
  stage, one launch fewer: `proj(+norm+RoPE) -> sdpa(+gate) -> out_proj` plus the packed SDPA's setup launch); the paged
  KV-cache write-through (`paged_kv_page_size`: `slot_mapping` is `[T]` over the packed tokens, a pad token's slot `-1` --
  "Serving: write-through into a paged KV cache" below) and, on top of it, the EXTEND form (`kv_source="paged"`: the SDPA reads
  every sequence's K / V from the pools; `token_capacity` makes `T` a capacity -- "Serving: the extend form" below; `fuse_gate`
  rides the paged body's epilogue there too, one launch fewer), block-sparse attention (`QsaSpec` with caller lists: the sparse
  core's packed arm, `block_ids` numbered per sequence, `fuse_gate` composing with it -- "Sparse attention (QSA)" above). Declined,
  typed: the fully fused quantized pipelines (the quantized THD SDPA bodies have no THD gate descriptor, and the fused
  MXFP8 projection fork decodes `(b, s_tile)` once per 128-row GEMM tile, which a packed tile may straddle), the MXFP8
  block BACKWARD over a packed record (its SDPA-layout MX quantize stages run the quantizer's dense arm only; the packed
  record and the packed delta exist, the packed MXFP8 backward is a follow-up) and the in-block indexer under `QsaSpec` (the
  paged write-through composes with the packed sparse block: the stage addresses tokens, so `slot_mapping` is `[T]` there too).
- **Declare `max_seq_len` tight.** The SDPA's unit grid is the plan-time envelope `B * ceil(max_seq_len / tile) * H_q`
  with dead units past the live total, and the backward's dS workspace scales with `ceil128(max_seq_len)`.

### Serving: the e4m3 KV cache (`kv_cache_dtype`)

**The e4m3 KV cache (`kv_cache_dtype=torch.float8_e4m3fn`, 2026-10-14; bf16 activations).** The serving stacks' `--kv-cache-dtype fp8`
recipe: the pools hold e4m3 codes under per-tensor fp32 DEscales `execute(k_scale=, v_scale=)` (value = code x scale; 1-element fp32
CUDA tensors read on device -- vLLM's per-layer `k_scale` / `v_scale`), a bf16 Q attends over the dequantized rows.  A declaration
ATTRIBUTE beside `paged_kv_page_size` (it changes the write-through's numerics), never a knob; `kv_cache_dtype=None` (or the
activation dtype) is the bf16 / f16 pool as before.  Two served forms, one attribute:

* **The quantizing WRITE-THROUGH** on every form that writes through over the block's own K / V (the dense prefill, QsaSpec or dense;
  the packed prefill is a typed decline until its accept cell lands): after norm + RoPE the post-RoPE K / V rows are quantized per
  tensor -- `code = sat_e4m3(row x (1 / scale))`, one fp32 product and one rounding, the FP8 pipeline's own cast kernel over the slab
  band -- into two compact e4m3 workspace slots and bit-copied into the e4m3 pools through their bf16 views (three launches where the
  bf16 write-through runs one; no new kernel).  The raw indexer key keeps its bf16 pool.  `out` is bitwise the bf16-pool block's (the
  write-through never changes attention); the pool rows are bitwise `(row.float() * (1 / scale)).clamp(-448, 448).to(torch.float8_e4m3fn)`.
* **The block-sparse DECODE mode over e4m3 pools** (`kv_source="paged"` on a dense `[B, S_q <= 4]` QsaSpec declaration -- the shared
  list): stage (4) is the gathered cast + the Rubin d256 DECODE tile -- `kernels/qsa_kv_upcast.py` reads, per (token, KV head) item,
  the sequence's listed 4-token blocks and the open tail over `[0, kv_lens[b])` through `block_table[b]`, dequantizes each e4m3 row
  (`bf16(fp32(code) x scale)`) and writes the visible rows COMPACTED in list order into the workspace scratch `[B x S_q x H_kv, 2, 2176,
  256]` bf16 plus a per-item length (0 = a dead item); the decode tile then runs over that scratch as a plain padded dense SDPA (no
  causal mask -- the gathered rows ARE the visible set -- the per-item length on the KV side only, the group packed, unsplit).  The
  sparse core is never taught fp8.  The mode norms + rotates OUT of place (the tile's per-item Q is a view of the compact normed Q alone:
  `inplace_qkv=None` reads as False there, `fuse_norm_rope` is declined) and gates in its own launch (`blk.gate_fused` False whatever
  `fuse_gate` says); `get_workspace_size()` grows by the scratch (2.2 MB per item) and holds it.

Typed declines (each pinned by `test_block_qsa_fp8_kv.py`): `torch.float8_e5m2` (by name) and any other dtype; the attribute without
`paged_kv_page_size`; `quant` / `save_for_backward` with it (the write-through's own rows); f16 activations; over the pools the EXTEND
form and the dense DECODE form (the paged THD body and the decode tile read bf16 / f16 pools: a dense e4m3 pool needs the tile's in-kernel
dequant arm -- a follow-up), `S_q > 4` (the per-token list form: 2.2 MB of scratch per item), `return_lse` (the tile over the scratch
writes item-major Stats), `decode_split_kv > 1` (unsplit over at most 17 KV tiles per item), `fuse_norm_rope` and an explicit
`inplace_qkv=True`; pools of the wrong dtype at declaration and at execute (both ways); `k_scale` / `v_scale` missing under the
attribute or given without it.  `serving_envelope(...).decode.kv_cache_dtypes` names the pool dtypes served at a geometry / dtype
(`(bf16, e4m3)` on a QsaSpec geometry with bf16 activations, the activation dtype alone otherwise).  Validated on Rubin against the fp32
oracle ON THE DEQUANTIZED POOL (the prefix rows quantized by torch with the per-tensor scales, the step rows the block's own write;
the gated O within atol 2e-2, `cos(out) > 0.999`, dead rows exact, the pools' step rows bitwise the torch cast, two launches bitwise;
the distance to the pre-quantization bf16 oracle reported, no gate): `S_q` 1 .. 4, B 1 / 3 / 4, `kv_lens` up to 20000 with dead and
shorter-than-the-step entries, pages 16 / 32 / 64 on HND / unified / NHD pools, `block_lens` given / absent, the indexer band's raw-key
write-through, the 24 / 2 Flash-Next geometry, the MTP residues, the write-through-only prefill form, the workspace sentinel, the
CUDA-graph replay over new scales / lengths / tables / lists / slots / inputs.  Performance: none claimed (functional cells only); the
unfused e4m3 decode step runs 9 launches against the bf16 mode's 5 -- the quantizing write inside the projection epilogue, the reciprocal
inside the write kernel and the in-kernel dequant are the levers a later change measures.

### Serving: the extend form (`kv_source="paged"`, `token_capacity`)

Every serving prefill of a sequence with a cached prefix -- a chunked prefill, a prefix-cache hit, a speculative verify -- is an
EXTEND step: the step's new tokens are written into the pools, and attention runs over each sequence's WHOLE logical KV range
(its prefix and the new tokens) from the pools, on the bottom-right diagonal. `kv_source="paged"` (appended; `"self"` = the block
as before) declares exactly that on a PACKED declaration:

```python
P, B_max, T_max, max_pages = 16, 128, 2048, 2048        # the kernel block size, the step's sequence / token capacities, the table width
geometry = GatedAttentionBlockGeometry(d_model=4096, h_q=32, h_kv=2, d_head=256, rope_dim=64, causal_bottom_right=True)
blk = GatedAttentionBlockFwd(h, w_qkvg, w_q_norm, w_k_norm, cos, sin, w_o, out, geometry,   # h / cos / sin / out: [T_max, .] token matrices
                             thd=True, num_sequences=B_max, max_seq_len=T_max, cu_seqlens=True, token_capacity=T_max,
                             paged_kv_page_size=P, kv_source="paged",
                             sample_k_cache=k_cache, sample_v_cache=v_cache,          # the pools' [num_pages, H_kv, P, D] shape / dtype / strides
                             sample_block_table=block_table)                           # [B_max, max_pages] int32 (a row slice of a wider table binds)
blk.check_support(); blk.compile()
workspace = torch.zeros(blk.get_workspace_size(), dtype=torch.uint8, device=h.device)  # ZEROED once (the bounded-token contract below)
blk.execute(h, w_qkvg, w_q_norm, w_k_norm, cos, sin, w_o, out, workspace,
            seq_lens=query_start_loc,                 # [B_max + 1] int32 prefix sums of the step's NEW tokens per sequence (zero-length pads)
            k_cache=k_cache, v_cache=v_cache,         # the pools the samples described, EXACTLY (shape and strides; the layout is compiled in)
            slot_mapping=slot_mapping,                # [T_max] int32 / int64: the new tokens' slots; -1 on every pad row
            block_table=block_table, kv_lens=kv_lens, # [B_max, >= max_pages] int32 page ids; [B_max] int32 LOGICAL lengths incl. the new tokens
            num_tokens_live=None)                     # or the step's live token count T_live: the token-indexed stages over T_live rows, no recompile
```

Five launches on ONE stream -- four with `fuse_gate=True`: `(1)+(2)+(3)` the projection with norm + RoPE over all `T_max` rows -- or
over the `T_live` rows of `num_tokens_live` -- (or `(1)` then `(2+3)`), `(4w)` the write-through of the new tokens' K / V at
`slot_mapping`, `(4)` the SDPA over the pools -- query row `i` of a sequence with `q_len` new tokens and logical length `kv_len`
attends keys `k <= i + kv_len - q_len` of `block_table[b]`'s pages; under `fuse_gate=True` its epilogue applies `O *= sigmoid(GATE)`
(the paged body carries the Rubin d256 gate seams, so stage `(5)` is not launched) -- then `(5)` the gate (unfused only) and `(6)`
the out projection. On the plain chain the write is ordered before the read by STREAM order alone: no kernel on the chain
launches as a programmatic dependent launch (pinned by `test_block_serving_extend.py`), so a kernel that adopts PDL after a pool
write must wait on its dependent grid before its first K / V load. The EXTEND form's `use_pdl` default is `False` (the measured default
is the DECODE form's, "Serving: the decode form"); under `use_pdl=True` the switch is PER STAGE: the write-through
(and the norm + RoPE pass, the separate gate) launch with the programmatic-stream-serialization attribute and their `griddepcontrol`
pair, while the paged prefill tile -- which carries no sites -- launches plain, so that link serializes and the read still follows the
write; `blk.pdl_stages` names the stages that took the attribute (never `"sdpa"` on this form), `blk.use_pdl_effective` is False,
and `serving_envelope().extend.pdl_stages` says so before anything is declared.

| appended | contract |
|---|---|
| `kv_source` | `"self"` (default) or `"paged"`; `"paged"` requires `paged_kv_page_size > 0`, `thd=True`, `geometry.is_causal=True` and `geometry.causal_bottom_right=True`, bf16 / f16 activations, inference (`save_for_backward=False`), `return_lse=False`, `fuse_gate` as declared (`True` folds stage (5) into the paged SDPA's epilogue), `window_left=-1`, `window_right` unbounded or 0, no `QsaSpec`, `d_head=256`; every other combination is a typed decline naming the feature (`GatedAttentionBlockFwd.serving_envelope` lists them) |
| `token_capacity` | THD only; `None` = the tight contract (lengths sum to `T`), an int = the BOUNDED form and must equal the samples' packed row count `T`: a step's lengths may sum to `T_live <= T`; rows `[T_live, T)` are pad rows -- finite `h` / `cos` / `sin`, slot `-1` (never written to the cache), never attended; their `out` rows are UNSPECIFIED but finite given finite inputs. Zero the workspace once and slice `out[:T_live]`: an unwritten `o` row then reads `0`, so pad rows of `out` are exactly `0` until a later, longer step overwrites those `o` rows (the `o` slot is aliased by no other stage's scratch -- pinned). Refused with `save_for_backward` (the backward's weight-gradient GEMMs contract over every row) |
| `sample_k_cache`, `sample_v_cache`, `sample_block_table` | REQUIRED under `kv_source="paged"`, refused otherwise: the pools' `[num_pages, H_kv, P, D]` shape / dtype / strides (the three stride patterns of the write-through, K and V of one HND / NHD kind, every non-innermost stride a 16-byte multiple) and the `[num_sequences, max_pages]` int32 block table (its row stride is free -- a `[:, :max_pages]` slice of a wider table -- and compiled in); the logical `S_kv` the tables may address is `max_pages * P` |
| `block_table` (execute) | `[num_sequences, >= max_pages]` int32 on `h`'s device, the page axis contiguous, the row stride as declared; row `b` lists sequence `b`'s pages in logical order (the serving stacks' NULL block `0` for a padded entry); read on device |
| `kv_lens` (execute) | `[num_sequences]` int32 contiguous: each sequence's LOGICAL KV length INCLUDING this step's new tokens (`prefix + q_len`); `0` = a padded entry (no keys: its `out` rows, if it has any, are exactly `0`); read on device |
| `k_cache`, `v_cache` (execute) | the pools, matching the samples EXACTLY (shape and strides; another page count or layout is another plan). Rows past `kv_len` inside a sequence's last live page may hold stale values but must be FINITE (a serving pool is zero-initialised); pages past the live page count are never read |
| `num_tokens_live` (execute, appended, the bounded form only) | the step's LIVE token count `T_live` as a HOST int; `None` (default) = the declared capacity, every stage over all `T_max` rows as before. With an int the projection (+ norm + RoPE), the cache write, the gate and the out projection run over `T_live <= T_max` rows with NO recompile (their kernels take the token count symbolically; the SDPA keeps the declared geometry and reads the live total from `seq_lens` on device, so the lengths must sum to `T_live`); `h` / `cos` / `sin` / `slot_mapping` / `out` may carry the declared `T_max` rows (the first `T_live` are read / written) or EXACTLY `T_live` rows -- a serving stack hands `hidden_states[rows]`, `output[rows]` and `slot_mapping[rows]` directly, with no staging copy -- and rows `[T_live, T_max)` of `out` are left UNTOUCHED (not written: slice nothing, or keep them as they were). Typed before any launch: a block declared without `token_capacity` refuses it by name, `T_live` outside `[1, T_max]` names the bound (`T_live > T_max` is the decline: more tokens are another declaration), a quantized pipeline and a dense projection on the graph route (a bounded block declared without `kv_source="paged"`) decline. A CUDA-graph capture bakes the host int into the captured launches: a replay serves the live count it was captured with, so a serving stack captures ONE graph per live-count bucket and pads a shorter step to the bucket (finite `h` / `cos` / `sin` pad rows, slot `-1`); the uncaptured EXTEND steps are where the count removes the staging copies |

`GatedAttentionBlockFwd.serving_envelope(geometry, dtype, device) -> ServingEnvelope` is the static envelope a serving stack's
init-time dispatch predicate reads instead of catching exceptions: the device / DSL facts, and per form (`extend` and `decode`,
described under "Serving: the decode form" below) whether it is served for that geometry and dtype (`served` / `reason`), the declaration it
requires, the validated page sizes and pool layouts, the dtypes, and `declines` -- every typed decline the form raises at
declaration with its exact message, read off the same table the constructor raises from, so no declaration-time decline escapes
the envelope.

Validated on Rubin (cc 10.7) at pages 16 / 32 / 64 / 128 on all three pool layouts, bf16 and f16, with vLLM-shaped buffers: padded
entries (`kv_len = 0`, the NULL block `0`, slot `-1`), prefixes ending mid-page, tail tiles, one / three / four / eight KV tiles, pads at the
end and in the middle of the packing, a row-sliced block table, the 2-chunk extend and the prefix-hit step against the single-shot oracle
within the dense suite's budget (`cos >= 0.999` on `out`, `atol 2e-2` on the gated O), live rows bitwise between a bounded and a tight
declaration, and -- with the projection GEMM driver's capture-safe route in the tree -- a CUDA-graph capture of the first execute replayed
over new lengths / tables / slots; `num_tokens_live` at T_live 412 and 300 of T_max 512: the live rows bitwise the capacity launch and the
rows past them untouched, the token tensors at exactly T_live rows bitwise too, a capture at T_live 412 replaying bitwise and serving the
padded next chunk; the three Qwen3.5-397B tensor-parallel shards (16 / 1, 8 / 1, 4 / 1 over `d_model` 4096, one replicated
KV head -- the table under "Qwen3.8 family" above) each with one EXTEND step at pages 16 / 32 / 64 and one DECODE step at the same
pages -- the rows of the support-matrix
tracker's `Rubin d256 gated attention block -- serving forms` section; the standalone adapter's own paged-prefill contract applies to the
pools (`docs/operations/Attention.md`, "Paged attention").

### Serving: the decode form (`kv_source="paged"` on a dense `[B, S_q]` block)

A decode step hands the attention sub-layer `B` sequences' NEW tokens -- one each, or `1 + num_spec` with speculative
decoding -- whose keys and values live in the paged cache. `GatedAttentionBlockFwd(sample_h [B, S_q, d_model], ...,
paged_kv_page_size=P, kv_source="paged", sample_k_cache=, sample_v_cache=, sample_block_table=)` declares that form: the
step's tokens are written through (4w) and stage (4) attends, per sequence, over the first `kv_lens[b]` keys of the pages
`block_table[b]` lists -- the prefix AND the step's own tokens -- on the bottom-right causal diagonal (the step's rows are
the LAST rows of a `kv_lens[b]`-key sequence; `geometry.is_causal=True` therefore needs `geometry.causal_bottom_right=True`;
`geometry.window_*` are not served). The batch is UNIFORM (`S_q` rows per sequence; a serving stack's padded entry is
`kv_lens = 0` + block table row `0` + slot `-1` and yields `out` rows exactly `0`, nothing written to block `0`). The
pipeline is `(1)+(2)+(3) -> (4w) -> (4) [+ combine] -> (5) -> (6)` -- past decode row counts. **At decode row counts (`B x S_q <= 128`,
the small-M projection route's bound) a `fuse_norm_rope=True` declaration LOWERS stage (1) to the unfused chain**, `(1) -> (2)+(3) -> (4w)
-> (4) [+ combine] -> (5) -> (6)`: the small-M route's slab GEMM and the norm + RoPE pass with the paged write-through folded in --
**`fuse_cache_write_small_m`** (appended; `None` = auto): the post-RoPE K / V rows go to the pools from the norm + RoPE kernel's own tiles
(the V band rides the launch, the pools bitwise the separate write) at `B x S_q < 128` rows or at `S_q = 1` (measured to B 128; past it
the rule is reachable only under an explicit `fuse_norm_rope=False`, the C0 lowering handing rows > 128 to the fused fork), the separate
cache-write launch at `B x S_q >= 128` with `S_q > 1` and wherever an explicit `fuse_cache_write=False` asks for it by name (the request
outranks the AUTO fold) -- six launches with the split combine (seven with the separate write) instead of
the 256-wide fused fork, which pads the step's handful of rows to one token tile and streams the slab weight from a quarter of the SMs (the
measured B 1 .. 32 decode step is faster on the unfused chain). The write-through rule is the measurement on a 212-SM cc 10.7 part: the arm
wins on every B 1 .. 4 cell (+1.1 .. +3.2 % per step on the plain chain, +2.4 .. +4.9 % with PDL on) and at B 32 / 128 x `S_q = 1`, and
loses 0.7-2.7 us per step at B 32 .. 128 x `S_q = 4` (128-512 rows); `True` / `False` pin either arm wherever the arm is served (`True` is a
typed decline on the fused fork, a quantized pipeline, the e4m3 cache, an indexer band, the EXTEND form), `blk.cache_write_in_norm_rope`
reads back what runs, `CACHE_WRITE_SMALL_M_DEFAULT_MAX_ROWS` is the bound. `fuse_norm_rope` stays what was declared,
`blk.fuse_norm_rope_effective` reads back what runs (`cache_write_fused` follows it: the write-through arm lives in the fork), and
`fuse_norm_rope_small_m=True` (a declaration knob, appended; performance-only, the same function either way within the fp32 roundings of two
kernels) keeps the fused fork as declared -- the A/B's control arm; so does an explicit `fuse_cache_write=True` (the write-through arm lives
in the fork, and the request outranks the AUTO lowering: `fuse_norm_rope_effective` reads True, `cache_write_fused` True); only the explicit
contradiction `fuse_norm_rope_small_m=False` + `fuse_cache_write=True` is a typed decline, naming both knobs. bf16 / f16, no indexer band,
16-bit pools, the dense DECODE form only: the EXTEND form, the block-sparse mode and the quantized pipelines
run `fuse_norm_rope` as declared.

**Programmatic dependent launch is the form's default on its decode-tile routes on cc 10.7** (`use_pdl=None` -> True;
`DECODE_PDL_DEFAULT_CCS` names the measured parts and every other part reads False until measured there; `use_pdl=False` is the plain
chain anywhere, `True` the switched chain where it is served): the norm + RoPE pass, the separate cache write, the separate gate, the
decode tile and its split combine launch with the programmatic-stream-serialization attribute and carry their `griddepcontrol` pair (a
wait before the first read of a predecessor's output, a release after the prologue), bitwise the plain chain eagerly and under capture;
the FROST GEMM stages -- the slab GEMM, the out projection and its split-K reducer -- launch with the attribute whatever the switch says
(their templates' own constant), so a `use_pdl=False` chain still holds three programmatic launches. `blk.use_pdl` is the resolved switch,
`blk.use_pdl_requested` the declaration (`None` = auto), `blk.pdl_stages` / `blk.use_pdl_effective` count the switched stages only, and
`serving_envelope().decode.pdl_default` says what auto resolves to on a part before anything is declared (`False` wherever the form is not
served -- another part, an unserved dtype; on the dense form the per-step reading is `decode.pdl_default and
decode.route(s_q, batch).pdl_capable`, a prefill-route step launching plain; under the block-sparse mode it is `decode.pdl_default` itself for a
served step -- the sparse core carries the sites at every served step width, and the dense route table does not apply there).  **The block-sparse decode mode takes the switch too**
(a `QsaSpec` geometry over bf16 / f16 pools): under `use_pdl=True` the sparse core launches with the attribute and carries its
`griddepcontrol` sites (every role's wait before its first read of a predecessor's output -- the slab, the pools, the block list, the
lengths, the table -- and no explicit release: the CTA exit triggers the dependent grid at the core's completion, the placement measured
on the cc 10.7 class, where a release at the core's prologue left the split combine's CTAs resident through the core's whole run and
cost 3.2-4.4 us per step at `B <= 4`) and its split combine takes the pair, bitwise the plain chain eagerly
and under capture (the first execute of a fresh switched declaration, captured with no warm-up, replays bitwise the eager plain chain);
`blk.pdl_stages` then names `sdpa_sparse` (the `sdpa` kind) beside the norm + RoPE pass, the write-through and the separate gate.  Its
AUTO reading is `True` on cc 10.7 (`SPARSE_DECODE_PDL_DEFAULT_CCS = ((10, 7),)` since 2026-10-27: the switched caller-list block measured
against the plain one in one process on a 212-SM cc 10.7 part -- CUDA-graph replay, the served body on each route, the lean tile at B 1 and
the tcgen05 core at B 4 and B 32 -- read +0.6 .. +2.5 % per step at B 1 x &#123;8K, 32K, 128K&#125; x S_q 1 / 4, +2.4 .. +3.5 % at B 4 and
+2.3 .. +2.7 % at B 32 x &#123;8K, 32K, 128K&#125; x S_q 1, +0.6 % / +3.8 % at B 4 / B 32 x 8K x S_q 4, bitwise on every cell and no cell slower than
its control band; `False` on every other part until measured there, never under the in-block indexer, and `use_pdl=False` is the plain chain
on cc 10.7), and `use_pdl=True` with `thd=True` on a `QsaSpec` geometry is a typed decline (the packed form reads the setup
launch's metadata above every role wait). Measured on a 212-SM cc 10.7
part: 1.1-1.5 us per decode step at B 1 (B 1 x KV 4K +5.0 %, 32K +4.0 %, 128K +2.5 %; +2.8 % geomean over the B 1 .. 4 cells), flat within
the control pair where the tile dominates; with the write-through folded in, B 1 x KV 4K 32.3 -> 29.3 us (+10.2 %) against the plain
seven-launch chain. The fused kernel the stage runs is a declaration fact:

The route is TABLE-DRIVEN (the default policy, `decode_route_policy="single_unit"`): ONE ordered table of the Rubin decode tile's
single-unit widths, `config_sm107.D256_DECODE_SINGLE_UNIT_Q_TILES = (16, 32, 64)` (the 64-column tile landed 2026-10-15: four
softmax groups, one unit of up to 64 packed rows, admitted for every packed group of this form -- paged under the bottom-right
causal mask), and ONE rule -- packed rows `S_q x group` within the widest width ride that width in ONE token unit, the KV streamed
once per (sequence, KV head); past the table the step rides the paged d256 PREFILL tile with the whole group packed (one Q tile per
sequence and KV head: the KV streamed once) wherever that tile packs the group -- a group dividing its 128 rows -- AND the step fills
the part: that tile is one UNSPLIT CTA per (sequence, KV head), so it needs `B x H_kv >= DECODE_PREFILL_ROUTE_MIN_CTAS` = 64 CTAs
(measured on the Rubin performance class at 32 / 2, `S_q = 8`, kv 32K: 330.6 us at B 1 against 58.7 on two 64-column units; 389.9 at
B 32 against the two units' 472.6); below the fill, and for a group it cannot pack (12 = 24 / 2), the step keeps the decode tile's
token-unit arm (up to two units of the 64-column tile, two streams, split over the part by the decode split model) and its unpacked
arm past that, the fewest streams any kernel offers a step that keeps the part busy (the prefill tile stays wherever no two-unit route
holds the step, whatever the batch). `DecodeRoute.kv_streams` names the KV streams per (sequence, KV head), `DecodeRoute.tile` /
`policy` / `q_tile_max` / `batch` which kernel ran under which policy, width cap and batch (`decode_route(geometry, S_q, batch=)`;
the envelope's `routes` are the reading without a batch, `serving_envelope().decode.route(S_q, batch=)` the one at a batch):

| `(group = H_q / H_kv, S_q)` | stage (4) | `decode_route(geometry, S_q)` |
|---|---|---|
| `S_q x group` within a width of the table `(16, 32, 64)` | the Rubin d256 **decode tile** (`sm107/decode_d256_f16.py`, the swap-AB tile at that width), the whole group packed, ONE token unit | `tile = "decode_d256_f16"`, `n_q` 16 / 32 / 64, `token_units = 1`, `kv_streams = 1`, `pack_gqa = True` |
| past the table, a group dividing 128 (16 = 32 / 2 and 16 / 1, 8 = 8 / 1, 4 = 4 / 1, 32 = 64 / 2, ...) with `S_q x group <= 128`, the step filling the part (`B x H_kv >= DECODE_PREFILL_ROUTE_MIN_CTAS` = 64) or no two-unit route holding it | the paged d256 **prefill tile** (the shared `sm100/prefill_d256_f16.py` body on its paged THD arm: the EXTEND form's binding at the dense step's `(B, S_q)` envelope, the whole group packed, unsplit, the gate in its epilogue): one Q tile per (sequence, KV head) | `tile = "prefill_d256_f16"`, `n_q = 0`, `token_units = 0`, `kv_streams = 1`, `pack_gqa = True`, `batch` = the declared batch |
| past the table, below that fill (32 / 2 at `S_q` 5 .. 8 under 32 sequences, 8 / 1 at 9 .. 16 and 4 / 1 at 17 .. 32 under 64), a group the prefill tile cannot pack (12), or a step past its 128 rows: up to two TOKEN UNITS of `64 // group` tokens (24 / 2 at `S_q` 6 .. 10) | the 64-column decode tile in token units, `ceil(S_q / (64 // group))` CTAs per (KV head, sequence, split), each streaming the KV range once, split over the part by the decode split model | `n_q = 64`, `token_units = 2`, `kv_streams = 2`, `pack_gqa = True` |
| ... and past two units, up to `S_q = DECODE_MAX_ROWS = 32` (32 / 2 from `S_q` 9, 24 / 2 from 11, 8 / 1 from 17, 24 / 4 from 21; 64 / 2 from 5, 128 / 2 from 3) | the decode tile UNPACKED (the SDPA adapter's own rule, the arm every group had before the table landed): one Q head per CTA, one token unit, `S_q` live rows of the 16-column tile (to 16 tokens) or the 32-column tile (17 .. 32) -- declared, unmeasured | `n_q = 16` / `32`, `token_units = 1`, `kv_streams = group`, `pack_gqa = False` |
| MHA (`group = 1`), `S_q <= 32` | the 16-column tile unpacked (the bit-exact unpacked fold; the 32-column tile from 17 tokens) | `n_q = 16` / `32`, `kv_streams = 1`, `pack_gqa = False` |
| `S_q > 32` (`DECODE_MAX_ROWS = DECODE_MAX_S_Q`) | NOT served -- a typed `NotImplementedError` at declaration (the `decode_rows` row): a longer dense step over the pools is the EXTEND form's PACKED (`thd=True`) business; `DECODE_PREFILL_TILE_ROWS = 128` bounds the prefill route only (past it the token-unit and unpacked arms, never a decline) | `tile = None`, `reason` set |
| `decode_q_tile_max=16` at 17 .. 32 tokens | NOT served -- the cap holds the table below the 32-column tile the unpacked step needs: honoured, never widened (a typed decline naming the cap; the envelope's `routes` predict it) | `tile = None`, `reason` names the cap |

Which `S_q` rides which tile at the served geometries (the default policy; `S_q` 1 .. 8; "prefill" = the prefill tile where the step
fills the part, `B x H_kv >= 64`, and two units of the 64-column tile below it):

| geometry (group) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| 32 / 2, 16 / 1 (16) | tile 16 | tile 32 | tile 64 | tile 64 | prefill / tile 64 x 2 units | prefill / tile 64 x 2 | prefill / tile 64 x 2 | prefill (128 rows) / tile 64 x 2 |
| 24 / 2 (12) | tile 16 | tile 32 | tile 64 | tile 64 | tile 64 | tile 64 x 2 units | tile 64 x 2 units | tile 64 x 2 units |
| 8 / 1 (8) | tile 16 | tile 16 | tile 32 | tile 32 | tile 64 | tile 64 | tile 64 | tile 64 |
| 24 / 4, 6 / 1 (6) | tile 16 | tile 16 | tile 32 | tile 32 | tile 32 | tile 64 | tile 64 | tile 64 |
| 4 / 1 (4) | tile 16 | tile 16 | tile 16 | tile 16 | tile 32 | tile 32 | tile 32 | tile 32 |
| 64 / 2 (32) | tile 32 | tile 64 | prefill / tile 64 x 2 | prefill (128 rows) / tile 64 x 2 | unpacked (160 rows, 32 streams) | unpacked | unpacked | unpacked |

Under `decode_q_tile_max=32` (the route before the 64-column tile landed, value for value -- the perf A/B's "develop behaviour"
arm): 32 / 2 rides the prefill tile from `S_q` 3 (below the fill: two units of the 32-column tile at `S_q` 3 / 4, the prefill tile at
5 .. 8 where two units hold no more than 4 tokens); 24 / 2 two units of the 32-column tile at `S_q` 3 / 4 and the 16-column tile
UNPACKED (12 streams) at `S_q` 5 .. 8; 8 / 1 the prefill tile from `S_q` 5 (below the fill two units of the 32-column tile to `S_q` 8);
24 / 4 two units of the 32-column tile at `S_q` 6 .. 8.

`decode_route_policy="token_units"` (a declaration knob, appended; `None` = `"single_unit"`) is the SDPA adapter's own route
verbatim -- the widest admitted width in up to two token units for every packed group past the table (32 / 2 at `S_q` 5 .. 8:
two units of the 64-column tile, two streams), then the unpacked tile, whatever the batch -- the other arm of the two routes' A/B
(below the prefill route's fill the two policies coincide; at the fill the default policy takes the prefill tile and
`token_units` keeps the two units); `decode_q_tile_max` (a declaration knob, appended; `None` = the whole table, `32` = the
measured widths) caps the table under either policy; `decode_route(geometry, S_q, policy=, max_q_tile=, batch=)` and
`serving_envelope(..., decode_route_policy=, decode_q_tile_max=)` read them. The CLAIMED envelope is `1 <= S_q <= DECODE_CLAIMED_MAX_S_Q = 4` on the PACKED decode tile at a
width up to `DECODE_CLAIMED_MAX_N_Q = 32` (`DecodeRoute.claimed`: the measured cells -- 24 / 2 and 32 / 2 at `S_q <= 2`, and at
`S_q` 3 / 4 under `decode_q_tile_max=32`); the served envelope runs to `S_q = DECODE_MAX_S_Q = DECODE_MAX_ROWS = 32` (the unpacked
arm's widest step; an MTP verify step of up to 8 tokens rides the one- and two-stream routes), and the 64-column tile's cells, the
prefill route, the token-unit cells past `S_q = 4` and the unpacked arm are routed and declared, unmeasured (the 64-column tile is
bitwise the two-unit 32-column form at the same split: the SDPA suite's cross-form pin, repeated at the block level). The tail
tile READS the pool rows past `kv_lens[b]` -- the last page's stale rows, and block `0` for the pages a short row does not list
-- and masks them to `P = 0`: they contribute nothing as long as they are FINITE, as a zero-initialized cache's are (a NaN
there is `0 x NaN`). On the decode tile the plan is a split-KV plan whenever the SDPA
heuristics' decode split model says so (`decode_split_kv=None`: an under-full launch splits until its K / V streams
saturate HBM; `decode_split_kv=<int>` pins it) -- fp32 partials in the block's workspace (`get_workspace_size()` holds
them) and ONE combine launch -- and `fuse_gate=True` then rides the split COMBINE: the tile has no gate seams, the combine
multiplies the merged fp32 value by `sigmoid(G)` before the single cast (FROST's one-rounding convention; the split is
floored at 2 under the gate). On an unsplit decode tile the gate is its own launch, stage (5):
`blk.gate_fused` says which runs; `fuse_gate` stays what was declared. On the prefill route the paged body runs UNSPLIT
(`decode_split_kv` must be `None` or 1: a pinned split there is a typed decline) and `fuse_gate=True` rides its epilogue, as on
the EXTEND form (`gate_fused`, no separate launch).

The decode split model behind `decode_split_kv=None` (`cudnn.sdpa.fwd.heuristics.choose_decode_tile_split_kv`, the same
function the SDPA graph API uses for a standalone decode graph; the block hands it the route's token units, the declared
`max_kv_len`, the part's SM count, the tile's width and the step's `(B, H_q, S_q)`) is fitted on the Rubin performance class: it
prices a CTA's KV stream per tile WIDTH (the 64-column tile streams a KV range in about three quarters of the time of two
32-column units, which is why the route takes it wherever two units would be needed), charges a launch of more CTAs than SMs
its tail WAVE in full (a CTA streams its chunk serially, so 256 CTAs on 212 SMs cost two waves, not 1.2) and therefore offers
the splits that fill one wave -- 3 / 5 / 6 / 7 beside the powers of two -- and prices the combine by the kernel the launch takes:
the decode-grid combine kernel (flat in the split count) on a small grid from split 8 on and on any grid inside its bound from
split 32, the lean kernel otherwise, the form pinned at compile. The block reads the model's eager-safe choice (the split path's
second host launch charged against its saving); `decode_split_kv` stays the one performance knob of the form -- the same function
under every value, measured or not -- so a replaying stack whose optimum differs pins it.

| input | shape / dtype | contract |
|---|---|---|
| `sample_k_cache`, `sample_v_cache` (declaration) | `[num_pages, H_kv, P, D]` views in the activation dtype: HND compact, NHD by strides, or the two halves of a unified `[num_pages, H_kv, P, 2D]` row (K at `+0`, V at `+D`), both of ONE layout kind | REQUIRED under `kv_source="paged"`: the strides are compiled into the SDPA's TMA descriptors; `execute`'s pools must be these views (shape AND strides) |
| `sample_block_table` (declaration) | `[B, max_pages]` int32, the page axis contiguous, ANY row stride -- a row slice of a stack's `[num_reqs, max_blocks_per_req]` table binds as a view | REQUIRED; `max_pages * P` bounds the logical KV length |
| `max_kv_len` (declaration) | int, `kv_lens[b] <= max_kv_len <= max_pages * P` | optional; the masks' clamp and the split model's extent; `None` = `max_pages * P` |
| `decode_split_kv` (declaration) | `None` or an int `>= 1` (`> 1` on the decode tile only; the prefill route runs unsplit) | the one performance knob of the form: the same function under every value |
| `decode_route_policy` (declaration) | `None` (= `"single_unit"`) or `"token_units"` | the route policy (the table above / the adapter's token-unit route): a performance-only knob, the same function under either; `DecodeRoute.policy` reads it back, `DecodeRoute.batch` the batch the default policy's prefill-route fill (`DECODE_PREFILL_ROUTE_MIN_CTAS`) was read at |
| `fuse_norm_rope_small_m` (declaration) | `None` (= the lowering at `B x S_q <= 128`, unless `fuse_cache_write=True` asks for the fork's write-through arm), `True` (the fused fork as declared) or `False` (the lowering, explicit; with `fuse_cache_write=True` a typed decline) | whether the fused norm + RoPE projection runs at decode row counts when `fuse_norm_rope=True`: a performance-only knob; `blk.fuse_norm_rope_effective` reads back what runs; refused with `fuse_norm_rope=False` and off the dense DECODE form |
| `decode_q_tile_max` (declaration) | `None` (= the whole table `(16, 32, 64)`) or a width of it (`32` = the route before the 64-column tile landed) | the width cap on the decode tile's single-unit table: a performance-only knob, the same function under every width; `DecodeRoute.q_tile_max` reads it back and the compiled width is pinned against the route at `compile()`; a cap held below the width an unpacked step needs is a typed decline naming it |
| `use_pdl` (declaration) | `None` (default = auto: `True` on the DECODE form's decode-tile routes for the bf16 / f16 block on the parts of `DECODE_PDL_DEFAULT_CCS` -- cc 10.7, the measured part -- and on the block-sparse decode mode's caller-list block on the parts of `SPARSE_DECODE_PDL_DEFAULT_CCS` -- cc 10.7 since 2026-10-27 --, `False` everywhere else), `True` or `False` (explicit, honoured on every part) | programmatic dependent launch over the chain: a performance-only knob (bitwise the plain chain, eagerly and under capture), bf16 / f16 only (an explicit `True` is a typed decline with `quant`, the e4m3 KV cache (`kv_cache_dtype=torch.float8_e4m3fn`), and with `thd=True` on a `QsaSpec` geometry; auto reads `False` there and never `True` where the explicit `True` is refused -- pools in the activation dtype, `kv_cache_dtype=None` or that dtype, are the bf16 / f16 block). PER STAGE: the norm + RoPE pass, the separate cache write, the separate sigmoid gate and the decode tile with its split combine take it -- or, on a `QsaSpec` geometry, the block-sparse core with its split combine (`sdpa_sparse`; served since 2026-10-21; auto `True` there on cc 10.7 since 2026-10-27 -- `SPARSE_DECODE_PDL_DEFAULT_CCS`, the measured part -- and `False` on every other part until measured; never under the in-block indexer, which declines the switch); the paged prefill tile of the prefill route carries no sites and launches plain (the link serializes, correct by PDL's semantics), so auto stays `False` on that route; the FROST GEMM stages launch with the attribute regardless. `blk.use_pdl` is the resolved switch, `blk.use_pdl_requested` the declaration, `blk.pdl_stages` / `blk.use_pdl_effective` read back what launched with the attribute, `DecodeRoute.pdl_capable` says it per route, `serving_envelope().decode.pdl_stages` / `.pdl_default` (the latter for a served form: `DECODE_PDL_DEFAULT_CCS` on the dense form, `SPARSE_DECODE_PDL_DEFAULT_CCS` -- cc 10.7, measured 2026-10-27 -- under the block-sparse mode; the per-step reading is `decode.pdl_default and decode.route(s_q, batch).pdl_capable` on the dense form and `decode.pdl_default` itself under the block-sparse mode, whose core carries the sites at every served step width) and `.extend.pdl_stages` per form |
| `fuse_cache_write_small_m` (declaration) | `None` (default = auto: the paged write-through folded into the norm + RoPE launch at `B x S_q < CACHE_WRITE_SMALL_M_DEFAULT_MAX_ROWS = 128` rows or at `S_q = 1`, the separate launch at 128+ rows with `S_q > 1` -- the measured rule -- and under an explicit `fuse_cache_write=False`, the separate launch asked for by name), `True` (folded in; a typed decline where the arm is not served; the strongest request, followed beside `fuse_cache_write=False` too) or `False` (the separate launch) | where the paged write-through runs on the DECODE form's UNFUSED chain (the lowered `fuse_norm_rope=True` declaration or `fuse_norm_rope=False`): bf16 / f16, no `quant`, no indexer band, 16-bit pools, the TMA norm + RoPE kernel; a performance-only knob (the pools and the slab bitwise either way, one launch fewer); `blk.cache_write_in_norm_rope` reads back what runs, the stage list loses `"cache_write"` under the arm |
| `sparse_tile_form` (declaration; a `QsaSpec` geometry only) | `"auto"` (default), `"lean"` or `"tcgen05"` | WHICH BODY of the index-list sparse core serves stage (4) on the block-sparse DECODE mode -- the tcgen05 core (`sparse_d256_f16`) or the LEAN decode tile (`sparse_lean_d256_f16`: `mma.sync` + `cp.async`, 128 threads per m16 tile, no TMEM / TMA / CLC): a performance-only knob (the same O / LSE under either body within the fixed-order-sum budget; never bitwise across the bodies, bitwise within each). `"auto"` consults the placement rule (cc 10.7, `B x H_kv <= 4` CTAs per chunk -- the lean grid's own unit: the step's `S_q` rows ride inside the CTA as m16 tiles, so `S_q` changes no grid and is not in the bound --, `top_k >= 8`, the lean body admissible) only on the parts of `heuristics.SPARSE_LEAN_TILE_DEFAULT_CCS` -- cc 10.7 since the end-to-end gate measured the lean route against the tcgen05 route there (2026-10-26, a 212-SM part at the clock lock, one process per cell, the control pair within 1 %: B 1 x 8K x S_q 1 34.8 -> 32.5 us per decode step (+6.8 %, 2.22 us; +5.9 % / 1.93 us symmetric with the arms built in the other order) and B 1 x 32K 36.6 -> 34.6 us (+5.7 %, 1.98 us; +5.6 % symmetric); `LEAN_SPARSE_MAX_UNITS` = 4 refit to the last class that wins inside its band; the unit is the CTAs per chunk since the same table's `S_q` 4 rows: B 1 x `S_q` 4 wins +6.4 / +6.6 / +6.8 % at 2 per chunk, B 4 x `S_q` 4 loses -8.8 .. -9.7 % at 8), so `"auto"` is the lean tile on cc 10.7 inside the bound and the tcgen05 core on every other part and past the bound --; `"lean"` is honoured where the body serves the declaration (the DECODE mode at `S_q <= 4` under the shared list -- one list per sequence, every row of the step in one CTA -- or the per-row list at one row per sequence, bf16 / f16, `G <= 16`, `top_k >= 8`; it ALWAYS splits: `decode_split_kv` None = the lean split of the default k, 65 at `top_k` 512, or one of `heuristics.sparse_lean_splits(top_k)` = 65 / 33 / 22 / 17 / 13 / 11 / 10 / 9, which SELECTS k) and a typed decline naming the arm elsewhere (per-row lists past one row -- the lean CTA gathers one list per sequence --, `S_q` past 4, THD, the prefill form, a split that is not a lean split, `top_k` below 8); a pinned `decode_split_kv` that is not a lean split keeps `"auto"` on the core (the pin is the stronger knob). `blk.sparse_tile_form_requested` keeps the declaration, `blk.sparse_tile_form` the resolved body, `blk.sparse_tile` its `kernel_template` (pinned against the compiled kernel); `serving_envelope().decode.sparse_tiles` / `.sparse_tile(s_q, batch, ...)` / `.sparse_tile_lean_default` say so before anything is declared. Over e4m3 pools (`kv_cache_dtype=torch.float8_e4m3fn`: the gathered cast + the dense decode tile, no body of the sparse core) an explicit body is a typed decline naming the arm and `blk.sparse_tile` reads `None` (`decode.sparse_tile(..., kv_cache_dtype=torch.float8_e4m3fn)` too). Refused (`ValueError`) off a `QsaSpec` geometry unless `"auto"` |
| `sparse_sub_tiles_per_cta` (declaration; with `sparse_tile_form="lean"` only) | `None` (default) or an int in `1 .. min(8, n_sub - 1)` | the lean tile's k -- its 32-key sub-tiles per CTA (the split is `ceil(n_sub / k)`, `n_sub = ceil((top_k + 2) x 4 / 32)` = 65 at `top_k` 512): `None` = the default (`heuristics.sparse_lean_sub_tiles_per_cta`: 1 today -- the k 2 / 3 / 4 renderings trace at 104-164 registers, a measured lever of the gate, not a default), an int pins it (with a pinned `decode_split_kv` the two must agree); a lean-only pin, refused by name (`ValueError`) under `"auto"` / `"tcgen05"` so it is never silently ignored. Performance-only: the same function under every k |
| `block_table` (execute) | `[B, max_pages]` int32 at the declared strides; `0` = the NULL block of a padded entry | REQUIRED; page ids are device data |
| `kv_lens` (execute) | `[B]` int32 contiguous: the LOGICAL KV length of every sequence INCLUDING the step's tokens; `0` = a padded entry | REQUIRED; bound to the SDPA's KV side ONLY (`seq_kv_lens = kv_lens`, `seq_q_lens = None`: every Q row of the step is live) |
| `seq_lens` (execute) | -- | REFUSED under `kv_source="paged"` (the dense block's self-attention length is a Q-side length; the KV length is `kv_lens`) |
| `k_cache`, `v_cache`, `slot_mapping` (execute) | as under "Serving: write-through" (`slot_mapping` is `[B * S_q]`) | REQUIRED (the write-through is part of the form) |

`GatedAttentionBlockFwd.serving_envelope(geometry, dtype, device) -> ServingEnvelope` is the STATIC record a serving stack
reads at construction instead of catching exceptions: the forms served (`forms`, drawn from `extend` / `decode`) and one record
per form (`extend`: `ExtendEnvelope`, `decode`: `DecodeEnvelope` -- `served`, the `reason` it would decline with, every typed
decline of the form in `declines`), `page_sizes` (16 / 32 / 64 / 128), `pool_layouts` (the three stride patterns), the decode
route per `(group, S_q)` (`decode.routes` for `S_q` 1 .. 33 -- the served envelope and its first decline --, `decode.route(s_q)`
for any step, `decode.max_s_q` = `decode.max_rows` = 32 / `decode.claimed_max_s_q` = 4, `decode.policy` / `decode.q_tile_max`; the
policy and the width cap the record reads are `serving_envelope(..., decode_route_policy=, decode_q_tile_max=)`'s), the pool dtypes (`dtypes`: bf16 / f16 -- an e4m3 pool is not served), the part and the CuTe DSL facts
(`device_cc`, `cutedsl_ok` / `cutedsl_reason`) -- derived from the same
helpers the declaration and `check_support` run, so no decline is reachable at `execute` that the record did not predict (the
execute-time refusals are the caller's malformed tensors, form-checked by name). It never raises for a device the block does
not serve. Typed declines of the form, at declaration: `kv_source="paged"` without `paged_kv_page_size`, a packed (`thd=True`)
declaration (the EXTEND form is its own binding), a quantized `quant`, `save_for_backward`, a head dim other than 256, a sparse
geometry on the PACKED form (a dense `QsaSpec` declaration is the block-sparse decode mode below), `is_causal` without
`causal_bottom_right`, a pool or table sample off its contract, `max_kv_len` past the table's reach, a step past
`DECODE_MAX_ROWS = DECODE_MAX_S_Q = 32` tokens (dense geometries; the `decode_rows` row), a `decode_q_tile_max` held below the width
an unpacked step needs, a pinned `decode_split_kv > 1` on the prefill route; `kv_source="self"` refuses every paged-read input rather
than ignoring it.

**On a `QsaSpec` geometry the same declaration is the block-sparse DECODE / extend-with-prefix mode (2026-10-14).**
`GatedAttentionBlockFwd(sample_h [B, S_q, d_model], ..., geometry_with_qsa (causal_bottom_right=True), paged_kv_page_size=P,
kv_source="paged", sample_k_cache=, sample_v_cache=, sample_block_table=)`: the step's tokens are written through (4w) and stage (4)
is the index-list sparse core READING THE POOLS -- per sequence the keys of the listed 4-token blocks and the open tail over the
first `kv_lens[b]` tokens of its `block_table` row, on the bottom-right diagonal (row `j` at `pos_j = kv_lens[b] - S_q + j`; a
negative position -- a sequence shorter than the step -- is a dead row, `kv_lens = 0` a dead entry: `out = 0`, `LSE = -inf`
exactly).  The list form is a DECLARATION fact -- `QsaSpec.index_list_form` (appended; `None` = derived from `S_q`): `"shared"` -- at
`S_q <= 4` ONE list per sequence, `execute(block_ids=)` `[B, top_k]` int32 (+ `block_lens [B]`), selected at the step-0 position
`kv_lens[b] - S_q` and shared by the step's rows (the visible set of "The decode mode and the speculative (MTP) rows" above: the listed
blocks plus every token from the step-0 tail start to the row's own position; the default under caller lists up to four rows, a typed
decline past them); `"per_row"` -- one list per row, `[B, S_q, top_k]` / `[B x S_q, top_k]` (+ `block_lens [B, S_q]` / `[B x S_q]`) at
positions `kv_lens[b] - S_q + j` -- the serving stacks' per-row selection, a chunked prefill over a cached prefix -- at ANY `S_q` (the
default past four rows and under the in-block indexer over the pools; the sparse core runs one work item per query row: no 32-row cap, no
decode-tile route).  What differs from the dense form:
`sample_block_table` must be CONTIGUOUS (the sparse core binds the whole `[B, max_pages]` table, and `execute(block_table=)` is
exactly that table -- a row slice of a wider table is a typed `ValueError`); `return_lse` is served (`[B, H_q, S_q]` fp32,
natural log); `fuse_gate` is served at every split (the core's epilogue unsplit, the split combine otherwise); `decode_split_kv`
is `None` (the fill rule: `ceil(SMs / (B x H_kv x S_q))` chunks, at most `min(16, ceil((top_k + 1) / 32))` -- the cap from the B 1 / B 4 x 32K x S_q 1 split sweeps on a 212-SM part with the pinned decode-grid combine; where that fill leaves a second wave of SMs (`B x H_kv x S_q x split > SMs`: a CTA streams its chunk serially, so 256 equal items on 212 SMs cost two item durations) the wave rule takes the split in `[1, that cap]` with the fewest `waves x (tiles per chunk + 1)`, ties to the smaller -- on a 212-SM part the T 128 step 3 chunks instead of none, B 32 x S_q 1 3 instead of 4, B 4 x S_q 4 6 instead of 7, every one-wave launch its fill-rule split) or an int in
`[1, ceil((top_k + 1) / 32)]` -- the one performance knob (fp32 partials inside `get_workspace_size()`, one combine launch), the
same function under every value; `max_kv_len` keeps its contract (at most the table's reach).  Typed declines: the packed
(`thd=True`) form (the serving table's `sparse` row: the sparse core serves packed sequences and paged pools one at a time), the
a split past the item's tile count, `causal_bottom_right` on the block's OWN K / V (`kv_source="self"`); the in-block indexer over the
pools (`index_source="indexer"`) and `fuse_prepare` on this mode are SERVED since 2026-10-21 (the next paragraph).  The envelope's
`decode.sparse_core` names the stage, `decode.requires` spells both list forms, `decode.routes` is empty (the dense tile's facts
do not apply).  **Two bodies, one function (`sparse_tile_form`, the knobs table above):** the sparse core is the tcgen05 tile
(`sparse_d256_f16`) or, on `sparse_tile_form="lean"`, the LEAN decode tile (`sparse_lean_d256_f16`: one `mma.sync` + `cp.async`
CTA of 128 x N_MT threads per (sequence, KV head, chunk of k 32-key sub-tiles; N_MT = ceil(S_q x G / 16) m16 tiles of four warps each), no TMEM / TMA / CLC, the SAME fp32 split-major partials into the workspace, the
same combine with the gate riding it) -- at `S_q <= 4` under the shared list (one list per sequence: every row of the step in one CTA,
`n_mt = ceil(S_q x G / 16)` m16 tiles over one gathered sub-tile; the per-row list at one row per sequence is the same list),
bf16 / f16, `G <= 16`, `top_k >= 8`, `decode_split_kv` None (the lean split 65 at the default k 1) or a lean split; `"auto"` takes the lean tile
on cc 10.7 inside the placement bound (`heuristics.SPARSE_LEAN_TILE_DEFAULT_CCS` = `((10, 7),)` since the gate measured the route end to end;
`LEAN_SPARSE_MAX_UNITS` = 4 CTAs per chunk, `B x H_kv`: B 1 and B 2 on two KV heads at every served `S_q` 1 .. 4) and the tcgen05 core on every other part and past the bound; `blk.sparse_tile` reads the body back.  Over e4m3 pools (`kv_cache_dtype=torch.float8_e4m3fn`) stage (4) is the gathered cast + the dense decode tile, no body of the
sparse core: an explicit body is a typed decline and `blk.sparse_tile` reads `None`.  On the standalone adapter an explicit `split_kv=1`
reads as unset (1 is its constructor default): under `tile_form="auto"` on a listed part it takes the lean split, so an unsplit launch there
is pinned with `tile_form="tcgen05"` (the block's `decode_split_kv` defaults to None and keeps the core for an explicit 1).  The two bodies
agree within the fixed-order-sum budget (the block suite: `cos(out) > 0.999`, the gated O and the LSE within
2e-2, the list bitwise), never bitwise across them.  Validated on Rubin against the fp32 oracle over the whole sequence (`test_block_qsa_decode.py`: `S_q` 1 .. 4
shared and 8 / 64 per row, `B` 1 / 2 / 4, `kv_lens` up to 20000 with dead and shorter-than-the-step entries, pages 16 / 32 / 64
on all three layouts, bf16 / f16, splits 1 / 2 / 4 / 17 and the rule, the gate fused and separate, the LSE, the indexer band's
raw-key write-through, the 24 / 2 Flash-Next geometry; the MTP residues on a short sequence; the block's form BITWISE the
standalone adapter on the same views; the workspace sentinel; CUDA-graph capture of the first execute and the replay over new
lengths / tables / lists / slots / inputs bitwise eager -- on the four-band geometry and on the indexer-band one at N 13952, whose
slab rides the small-M route's 128-row tile; the band chain's launch census one above the band-off chain's, the raw-key write).

**The in-block indexer's DECODE step over the pools (`QsaSpec(index_source="indexer", index_band=True)` with
`kv_source="paged"`; bf16; 2026-10-21).** The block-sparse DECODE mode derives the step's selection itself: declare the mode
with the indexer geometry and THREE more samples -- `sample_index_kv_cache [n_pages, P / 4, 1, 128]` (the PAGED compressed-key
cache in the activation dtype: compressed row `j` of sequence `b` lives on page `block_table[b, j // (P / 4)]`, row `j % (P / 4)`;
page 0 is the null page; a page holds its rows contiguously), `sample_index_state_cache [blocks, cap, 1, 128]` (the RING of raw
indexer keys, position-indexed -- the raw key of token position `p` of a sequence lives at row `p % cap` of its block; `cap >=
S_q + 3`, the compress kernel's rule `min_ring_rows`; block 0 is the null block) and `sample_index_state_block_table [B, 1]`
int32 (the ring block per sequence; `0` for a dead entry) -- and `execute(..., w_iq_norm=, w_ik_norm=, index_kv_cache=,
index_state_cache=, index_state_block_table=, cos_pos=, sin_pos=)`, the position tables `[1 or B, >= max_pages x P, rope_dim]`
over the table's whole reach (the table is the CEILING on positions: one row short is a `ValueError` naming the bound before any
stage runs).  `block_ids` / `block_lens`, `index_k_raw` (the raw key lives in the ring), `index_k_compressed` (the cache is paged)
and `n_commit` are refused by name; the optional `block_ids_out [B, S_q, top_k]` / `block_lens_out [B, S_q]` hand the step's
selection back (a serving stack's A/B against its own selector: the caller-list band block fed them is BITWISE this block).  The
list form is per row (`index_list_form` `None` or `"per_row"`; `"shared"` is the typed decline `indexer_shared_list`).  The step:
(1) the band's indexer QUERIES, RMSNormed with `w_iq_norm` and rotated at the token's position (bitwise the prefill form's pool-1
compress); (2) every committed row's raw key into the ring at `pos % cap`, and the row that completes a 4-token block pools the
block's four raw keys (the rows below the step from the ring, the rest from the step) through the compress chain -- RMSNorm with
`w_ik_norm`, partial RoPE at the block START position `cos_pos[4k]` -- into the block's cache row, BITWISE the prefill compress
of the same keys (so a prefill-written row and a decode-written row are one chain); (3) the paged scorer
(`cudnn_block_qsa_score_paged`) over each row's visible complete blocks `floor((pos + 1) / 4)` through the table --
`relu(q_h . kbar_j)` summed over the heads times `1 / sqrt(128)`; `counts` = the VISIBLE count, `block_lens_out = min(top_k,
counts)`; (4) the decode top-k (`cudnn_qsa_topk_decode`, the plan-time knobs from the row count and the table's reach; ids
strictly descending, `-1` past `min(top_k, counts)`) into `block_ids_out` or the workspace; (5) the sparse core over the pools on
the per-row list.  A row at a negative position (`kv_lens[b] < S_q`) and a dead entry (`kv_lens = 0`, ring block 0, page 0) write
nothing, score nothing (`counts 0`, every slot `-1`, `out = 0`, `LSE = -inf`).  Launches: under `fuse_prepare=True` the prepare
launch (`frost_qsa_prepare`) carries the norm + RoPE, the K / V write-through, the indexer-query norm and the ring / compress arm,
and the stage adds the scorer and the top-k -- eight kernels on the whole step with a split (the slab, the prepare, the scorer, the
top-k, the sparse tile, the combine, the out projection and its split-K reducer); under `False` the norm + RoPE pass, the pool-1
query compress, the step compress (`frost_qsa_compress_step`) and the K / V write-through run as their own launches -- eleven;
`None` is the MEASURED default (`serving_envelope(...).decode.fuse_prepare_default`: True on cc 10.7 -- the 2026-10-21 end-to-end
gate on an exclusive 212-SM part measured the fused step +9.6 .. +14.5 % faster at B &lt;= 4 and +4.4 .. +10.0 % at B 32 over 19 decode cells,
bitwise --, `False` on every other part until measured, and `False` out of place -- `inplace_qkv=False` -- on every part: the AUTO reading is
never True where an explicit True is declined); every output, the ring and the cache are BITWISE between the two readings.  `index_state_slot_mapping_present=True`
takes the ring's write rows from `execute(index_state_slot_mapping=)` (`block x cap + pos % cap`, `-1` on a dead row; int32 or
int64 = `slot_mapping`'s dtype) instead of deriving them from the ring table -- bitwise the derived form;
`rope_position_tables_only=True` (fused only) refuses `cos` / `sin` and rotates Q / K and the indexer queries from
`cos_pos` / `sin_pos` at the device-derived position -- bitwise the per-token form; `index_commit_present=True` is a follow-up
(a typed `NotImplementedError`).  No plan constant, allocation or host read on the execute path: the FIRST execute of a fresh
declaration captures into a CUDA graph without a warm-up, and the replay over a new step's tensors is bitwise eager.  The
workspace carve is the caller-list band block's plus five slots at the end (`ix_q`, the logits `[T, max_pages x P / 4]` fp32,
the counts, the ids, the compressed-slot report).  The envelope publishes the facts a serving stack reads (never re-spells):
`decode.index_sources` (`("caller", "indexer")` at bf16 x `index_head_dim` 128), `index_cache_dtypes` (`(bf16, e4m3)`: the e4m3
compressed-key cache served at decode since 2026-10-21, "At decode over the pools" above),
`index_head_dims` (`(128,)`), `index_kv_heads` (`(1,)`), `index_block_sizes` (`(4,)`), `index_top_k_range` (`(4, 512, 4)`),
`index_query_rows_max` (64: `S_q x next_pow2(index_heads)` rows per sequence), `index_state_rows_extra` (3: `cap >= S_q + 3`),
`index_page_sizes` (`(16, 32, 64, 128)`: `P / 4` a power of two), `fuse_prepare_default`; `decode.kv_cache_dtypes` is `(bf16,)`
under the indexer (the e4m3 pools wait for the sparse core's in-kernel dequant arm: `indexer_e4m3_pool`).  Typed declines by
row name (`decode.declines`, each the text the declaration raises): `indexer_query_rows`, `indexer_page_size`,
`indexer_e4m3_pool`, `indexer_shared_list`, `fuse_prepare_rope_tables_unfused`, `indexer_f16`, `indexer_head_dim`, `indexer_thd`
(the `indexer_e4m3_cache` row retired 2026-10-21: the e4m3 cache is served); `use_pdl=True` under the indexer keeps
a typed decline by name (the caller-list sparse chain serves the switch; the indexer's step threads the switch it is handed and is
handed `False` until its own switched chain is measured).  **Limitation:** there is no FROST writer
of the paged compressed-key cache at PREFILL yet -- the prefill form writes `index_k_compressed` densely (`[B, >= floor(S / 4),
128]`); a serving stack scatters those rows into the pages through its table and primes the ring with the prompt's last `cap`
raw keys (the band's `index_k_raw` rows or the slab's band), exactly as the multi-step cell of the end-to-end module does -- under the
e4m3 indexer cache the prefill form's e4m3 rows prime the paged e4m3 cache (one rounding convention, so the scatter is bitwise the step's
own rows).
Validated on Rubin against the two-stage oracle (the step's set equals the fp64 top-k of the block's own operands up to ties; the
attention of every row within the budget of the fp32 oracle fed the kernel's list; the ring rows and the completed cache rows
bitwise the compress kernel; dead rows exact): `S_q` 1 .. 4 x `B` 1 / 2 / 4 x `kv_lens` 2 .. 32768 with dead and
shorter-than-the-step entries, pages 16 / 32 / 64 on the three layouts, both knob readings bitwise with the launch census by
kernel name, the identity region bitwise the caller-list block on the full list, the first-execute capture, the ring-slot and
position-table forms, the MTP residues, the workspace sentinel, and eight carried decode steps after a prefill (a rejected draft
re-stepped; the cache rows bitwise a from-scratch compress of the accepted stream after every step).  Functional cells on a
development node; the fusion's default flipped to True on cc 10.7 after the exclusive-node end-to-end gate measured it (2026-10-21).

### Backward

`GatedAttentionBlockBwd(sample_dy, sample_saved, sample_w_qkvg, sample_w_q_norm, sample_w_k_norm, sample_cos,
sample_sin, sample_w_o, geometry, *, recompute=RecomputePolicy.RECOMPUTE_QK_PRE, need_dh=True,
need_dw_qkvg=True, need_dw_o=True, need_dw_norms=None, seq_lens_present=False, dw_norm_dtype=torch.float32,
fuse_gate_bwd=False, fuse_wgrad_overlap=False, thd=False, num_sequences=None, max_seq_len=None, cu_seqlens=False,
quant=None, grad_scaling="current", grad_scale_margin_log2=0)` is the
block backward: eight stages on ONE launch stream (the two
weight-gradient GEMMs on a block-owned side stream under `fuse_wgrad_overlap`, joined back before `execute` returns), no
allocation, against the forward's
`SavedForBackward(h, gate, o, lse, rstd_q, rstd_k, q_pre=None, k_pre=None, proj_slab=None, seq_lens=None)` record written
in the **proj_slab save mode** (`GatedAttentionBlockFwd(save_for_backward=True)`, the default `saved_gate_copy=False`;
`gate` / `q_pre` / `k_pre` may be `None` there -- they are bands of `proj_slab`). Recipe, the same lifecycle as the forward:

```python
from cudnn.gated_attention_block import GatedAttentionBlockBwd

bwd = GatedAttentionBlockBwd(dy, saved, w_qkvg, w_q_norm, w_k_norm, cos, sin, w_o, geometry)  # every need_* True
bwd.check_support()
bwd.compile()                                              # the artifacts first: get_workspace_size() needs them
workspace = torch.empty(bwd.get_workspace_size(), dtype=torch.uint8, device=dy.device)
dh, dw_qkvg, dw_o = torch.empty_like(saved.h), torch.empty_like(w_qkvg), torch.empty_like(w_o)
dw_q_norm, dw_k_norm = (torch.empty(geometry.d_head, dtype=torch.float32, device=dy.device) for _ in range(2))  # fp32
bwd.execute(dy, saved, w_qkvg, w_q_norm, w_k_norm, cos, sin, w_o, dh=dh, dw_qkvg=dw_qkvg, dw_o=dw_o,
            dw_q_norm=dw_q_norm, dw_k_norm=dw_k_norm, workspace=workspace)          # + current_stream=, seq_lens=None
```

`gated_attention_block_backward(dy, saved, w_qkvg, w_q_norm, w_k_norm, cos, sin, w_o, geometry, *, seq_lens=None,
recompute=..., current_stream=None, fuse_gate_bwd=False, fuse_wgrad_overlap=False, thd=False, max_seq_len=None, quant=None,
grad_scaling="current", scale_dp=None, scale_dy=None, scale_do=None, scale_dqkvg=None, h_t=None, h_t_sf=None, w_qkvg_t=None,
w_qkvg_t_sf=None, w_o_t=None, w_o_t_sf=None, grad_scale_margin_log2=0)` allocates the gradients and the workspace on the launch stream (`current_stream`,
else torch's current stream -- the caching allocator orders a buffer's reuse only against the stream it was allocated on),
caches the compiled block per declaration and returns `{"dh", "dw_qkvg", "dw_o", "dw_q_norm", "dw_k_norm"}`; which entries exist follows `requires_grad` on
`saved.h` / `w_qkvg` / `w_o` / `w_q_norm` / `w_k_norm` (the tensors are handed to the block detached). Under `thd=True` it
derives `num_sequences` and `cu_seqlens` from the record (`saved.seq_lens.numel()`, `saved.seq_lens_form`) and requires
`max_seq_len` (a `ValueError` naming it alone otherwise); both, with `max_seq_len`, are part of its cache key. `quant` /
`grad_scaling` (the quantized backward below) join the key too, and `scale_dp` / `scale_dy` / `scale_do` / `scale_dqkvg` and the
transposed artifacts `h_t` / `h_t_sf` / `w_qkvg_t` / `w_qkvg_t_sf` / `w_o_t` / `w_o_t_sf` pass through to `execute`; the gradients are allocated in `dy`'s dtype (bf16 under `quant`, where `saved.h` and the weights are e4m3 codes).

**Packed sequences (THD).** `GatedAttentionBlockBwd(..., thd=True, num_sequences=B, max_seq_len=S_max, cu_seqlens=False)` --
the forward's four knobs, appended last -- differentiates the packed training record of the section above: `dy` is
`[T, d_model]` or `[1, T, d_model]`, `cos` / `sin` the per-token tables, `dh` comes out in `saved.h`'s shape (the forward's `h`,
`[T, d_model]` or `[1, T, d_model]`; the two ranks may differ between `dy` and `h`). `saved.seq_lens` is
REQUIRED (the `[B]` / `[B+1]` int32 tensor the forward ran with; validated on the host: dtype, rank, element count per
`cu_seqlens`, device, contiguity -- never its values) and `saved.seq_lens_form` must match the block's form: a padded DENSE
record (`seq_lens_form=None`) is refused by a packed backward, a packed record by a dense one. `execute(seq_lens=)` is
optional and, when given, must be `saved.seq_lens` itself. Every token-wise stage is the dense `B=1, S=T` block's (the two weight-gradient GEMMs contract over `K = T`; the two
data-gradient GEMMs produce `T` rows and contract over `d_model` and `n_qkvg`); the SDPA backward runs the packed d=256 chain (`SdpaBwdDslSm107(thd=True)`: its own setup
launches, per-sequence descriptors, the kv-blocked dS workspace, the GQA fold bounded by the live total on the device).
`fuse_wgrad_overlap` and `fuse_gate_bwd` are both served, each bitwise the plain packed block: the gate backward's `delta`
at `B = 1, S = T` is byte for byte the packed head-major `[1, H_q, ceil128(T)]` delta the packed chain reads (tail zeroed),
so under `fuse_gate_bwd` the chain's own `dot_do_o` launch is gone and its `delta` region moves into the block's
(`test_thd_fused_gate_bwd_is_bitwise_the_unfused_packed_block`). The
declaration bounds and the lengths contract are the forward's. Workspace: the dense `B=1, S=T` carve plus the packed
chain's scratch -- `delta [1, H_q, ceil128(T)]` (unless `fuse_gate_bwd`), ONE head chunk of the kv-blocked dS, `[qh_chunk, ceil256(T + 256 B),
ceil128(max_seq_len)]` in the activation dtype, the metadata and descriptors, the GQA partials -- so declare `max_seq_len`
tight.

What runs (launch order, `T = B*S`): the out-projection dgrad `dO_gated = dY @ W_o`; the sigmoid-gate backward
(`dO`, `dG` into the GATE band of the `[T, N]` `dqkvg` slab, `O_gated` for the wgrad); the out-projection wgrad
`dW_o = dY^T @ O_gated`; the recompute of the post-norm / post-RoPE Q, K from the saved slab (the forward's norm+RoPE
kernel) and a compact copy of V; the Rubin d=256 SDPA backward (`SdpaBwdDslSm107`) into compact `dQ` / `dK` / `dV`;
the fused RoPE-adjoint + RMSNorm backward writing the Q / K / V bands of `dqkvg` plus fp32 `dW_norm` partials and their
fixed-order reduce; the projection wgrad `dW_qkvg = dQKVG^T @ h` and dgrad `dh = dQKVG @ W_qkvg`. `12 + c*(2+q)` kernel
launches (`c` = the SDPA backward's head chunks, `q` = its dQ GEMM launches per chunk: 1 under its single-launch dQ
rendering, the GQA ratio under the per-member twin; more when `S % 128 != 0`) -- 15 / 22 at the test geometry
(S=256 / S=1000), counted by CUPTI (`test_launch_count_is_honest`, on Rubin cc 10.7). Deterministic by
construction: no atomics anywhere, and the four GEMMs run the block's forced 256-wide N tile at one split-K slice
(`compile()` refuses a heuristic fallback typed; `check_support` declines a `d_model % 256 != 0` geometry) -- two runs
are bitwise equal, pinned on Rubin by `test_two_runs_are_bitwise` and `test_a_caller_stream_orders_every_stage`.
`need_*` fixes at build time which GEMMs exist (an output
given for a `need_*=False`, or missing for a `need_*=True`, is a typed error at `execute`); `need_dw_norms=None` follows
`geometry.qk_norm`, and asking for norm-weight gradients under `qk_norm=False` is a typed decline. `RecomputePolicy`
(`SAVE_ALL` / `RECOMPUTE_QK_PRE`) both read the slab's bands today; `RECOMPUTE_GATE` is reserved.

**Fusion knob -- `fuse_gate_bwd` (default `False`).** The SDPA backward's first launch is `delta = rowsum(dO * O)` over
the `dO` the gate backward just wrote and the `O` it just read. With the knob on, the gate-backward kernel emits `delta`
as a fourth output (the bf16 / fp16-rounded `dO` it stores, summed in the chain's own `dot_do_o` order, the pad rows
zeroed) into a block-owned fp32 `[B, H_q, S_pad]` region (`[1, H_q, ceil128(T)]` under `thd`: the dense arm at `B = 1, S = T`
is the packed chain's own head-major layout), and the SDPA backward adapter is built with
`external_delta=True` and reads that tensor: one launch and one read each of `O` and `dO` fewer (`11 + c*(2+q)` launches,
14 / 21 at the test geometry), the adapter's own `delta` region gone from its scratch (the block's region takes its
place, same bytes). Performance-only in the strict sense: the gradients are **bitwise** the unfused block's
(`test_fused_gate_bwd_is_bitwise_the_unfused_block`, bf16 and fp16, dense and causal, B=1 and B=3 under GQA; packed:
`test_thd_fused_gate_bwd_is_bitwise_the_unfused_packed_block`), and the
two-run and stream-order pins run under both knob values. Measured whole-backward effect: see the performance section.

**Scheduling knob -- `fuse_wgrad_overlap` (default `False`).** The two weight-gradient GEMMs are consumed by nothing
inside the backward: `dW_o = dY^T @ O_gated` is ready after the gate backward, `dW_qkvg = dQKVG^T @ h` after the
norm/RoPE backward. In order they sit on the launch stream between stages they do not feed, so their SM time is
serialized with the SDPA backward chain (below full SM occupancy at small S, with launch gaps between its kernels) and
with the `dh` dgrad. With the knob each of them is issued on a block-owned side stream instead: the side stream waits
an event recorded on the launch stream right after the GEMM's producer stage (fork), and the launch stream waits an
event recorded on the side stream after the GEMM (join) at the end of `execute` -- the latest legal point, because the
side GEMMs read only buffers no later stage writes (`dy`, `O_gated`, the finished `dqkvg` slab, `saved.h`) and get
their own appended scratch region (`gemm_scratch_side`, sized to the two wgrad plans; the forced tile at one split-K
slice carves no scratch, so it is 256 B at the 397B geometry) rather than sharing the in-order GEMMs' `gemm_scratch`
with the `dh` dgrad they now run next to. Every write the caller can observe is therefore still ordered on the launch
stream before `execute` returns: an ambient or an explicit launch stream sees the whole backward exactly as without the
knob (the stream-order probe `test_a_caller_stream_orders_every_stage` runs under it, both arms), the same kernels
launch (the CUPTI count is unchanged), and the gradients are **bitwise** the in-order block's
(`test_fuse_wgrad_overlap_is_bitwise_the_in_order_block`: bf16 and fp16, dense and causal, B=1 and B=3 under GQA, and
composed with `fuse_gate_bwd`) -- the GEMMs are deterministic, so moving them to another stream moves no bit. The side
stream is the block's own: one dedicated non-blocking stream at the device's lowest priority (a filler below any launch
stream), created through the driver at `compile()` together with the four events and released with the block --
nothing per execute, and never a torch pool stream, so never a caller's launch stream. One compiled block may be driven
from several host threads on different launch streams (the fork, the side GEMM's enqueue and the join record are
one locked section of host-side enqueues, and the join's check and wait take the same lock); the convenience
wrapper's per-call workspace, freed at return, is reused only behind the join. CUDA-graph
capture of `execute` works under the knob: the fork and the join are recorded as graph edges (the side stream joins
the capture and is joined back before it ends), and the replay is bitwise the eager run
(`test_cuda_graph_capture_replays_bitwise`, both knob values). The two do not compose on one block: a CUDA-graph
capture of `execute` must not overlap an `execute` of the same compiled block from another thread, and two
concurrent captures of one block are likewise unsupported -- the block has ONE side stream, which belongs to the
capture from the capture's first fork until its join, so an eager fork or join onto it in that window (or a second
capture's fork) is a typed `RuntimeError` naming the situation, raised before the capture is touched, instead of the
capture's invalidation or the eager stream silently joining the graph
(`test_fuse_wgrad_overlap_capture_and_eager_executes_do_not_overlap`); the capturing thread's own forks and joins
pass. A caller that mixes the two on one block (the convenience wrapper caches one block per declaration for the
process) serialises each capture against the block's eager executes. The knob is a typed `ValueError` at
`check_support` when neither `need_dw_o` nor `need_dw_qkvg` is set (nothing to overlap); the convenience wrapper,
whose needs follow `requires_grad`, runs the in-order block instead when the weights are frozen (a frozen-weights
training step must not fail over a scheduling knob). Measured whole-backward effect: see the performance section.

**Quantized backward (per-tensor fp8) -- `quant` (default `None`).** `GatedAttentionBlockBwd(..., quant=QuantSpec,
grad_scaling="current")`, with the per-tensor fp8 training forward's own `QuantSpec`, differentiates the quantized training
record **as written** -- e4m3 `saved.h`, e4m3 `w_qkvg` / `w_o`, the bf16 slab / `o` / `lse` / `rstd` -- from a bf16 `dy`, and
returns bf16 `dh` / `dw_qkvg` / `dw_o` and fp32 `dw_*_norm`. What runs (launch order): the fused PROLOGUE launch -- four
independent jobs dispatched by block range: the scalar-block init (every slot zeroed, `descale_dp`, and the `QuantSpec`'s plan-time
constants stored from the launch's kernel arguments), the amax of `dY` as per-CTA partials, the recompute of the
post-norm / post-RoPE Q, K with an e4m3 epilogue (`Q8` / `K8` at the forward's **static** `scale_q / scale_k` -- kernel arguments
of that launch, the values of their slots --, bit-exactly the
forward's own SDPA operands: the bf16 rounding first, then the cast, so no bf16 rebuild is written), and `V8` straight from the
slab band at `scale_v` (no compaction launch); the e4m3 quantize of `dY`, which reduces the partials and publishes `amax_dy`; the
e4m3 out-projection dgrad `dO_gated = dY8 @ W_o8 * alpha`; the gate backward's fp8 arm (`dO`, `dG`, the e4m3 `O_gated` for the
wgrad, `delta = rowsum(dO * O)` -- always, it is the fp8 SDPA backward's external delta --, and per-CTA partials of the stored
`dO`'s and `dG`'s `max |.|`, one plain store each from a persistent grid); the e4m3 quantize of `dO`, which reduces the `dO`
partials and publishes `amax_do`; the e4m3 out-projection wgrad `dW_o = dY8^T @ O_gated8 * alpha`; the Rubin d=256 per-tensor
fp8 SDPA backward (`SdpaBwdDslSm107Fp8`, external delta, `amax_dP` requested) into bf16 `dQ` / `dK` / `dV`; the fused
RoPE-adjoint + RMSNorm backward, storing per-CTA partials of the Q / K / V bands' `max |.|`; the fused EPILOGUE launch -- the
fixed-order `dW_norm` reduce (one column per block: the same fp32 chain as the standalone reduce) and the e4m3 quantize of
`dQKVG`, every block reducing `amax_dqkvg` from the `dG` and band partials and the first publishing it; the e4m3 projection
wgrad `dW_qkvg = dQKVG8^T @ h8 * alpha` and dgrad `dh = dQKVG8 @ W_qkvg8 * alpha`. Every gradient amax is a `max` over
per-CTA partials its producer stored (order-free, bitwise the standalone amax pass; the block's own kernels run no atomic --
the per-warp `atomicMax` into one slot the folds first shipped as serialised at the L2 and cost the gate backward 0.7 ms at
S = 32K), and the two fused launches are bitwise the standalone chain they replaced. Every e4m3 GEMM
runs the block's forced tile at its 64-byte MMA K form with a bf16 output and an fp32 `alpha = descale_A * descale_B` epilogue
read from a device slot. `12 + c*(2+q)` kernel launches -- 15 at the test geometry (Q/K RMSNorm on, GQA; 15 RoPE-only, the
epilogue launch stays for the quantize; 15 MHA: the SDPA backward's dK fold shares the dV fold's launch under GQA, and its
setup is one launch; +3 when `S % 128 != 0`, the q-side staging pads, and +2 when `S % 256 != 0`, the kv-side pads: 20 at a
padded causal S such as 992 or 1000, 17 at S = 384; 17 before the SDPA backward merged its setup and fold launches, 24 before
the launch fusion), counted by CUPTI in the quantized backward's own suite. Gradient scales (`grad_scaling`, a declaration attribute -- it moves
the e4m3 rounding points, so it is never a knob): `"current"` derives every gradient's per-tensor scale ON DEVICE from its own
amax pass in this step (`2**(floor(log2(448 / amax)) - FP8_GRAD_SCALE_MARGIN_LOG2)`, with `FP8_GRAD_SCALE_MARGIN_LOG2 = 0`); the margin
is the appended declaration attribute `grad_scale_margin_log2` (an int in `[0, 8]`, the module constant by default): a compile-time
constant of EVERY gradient quantize -- `dY`, `dO`, `dQKVG` here, `dY` alone under `MxQuantSpec` -- so a non-default value is a
different compiled block (never a knob, never a scalar slot), refused without `quant` and under `"delayed"` (the caller's scales
carry their own headroom there); `quant_scalars()` publishes the scales that ran;
`"delayed"` reads the previous step's `scale_dy` / `scale_do` / `scale_dqkvg` from `execute(...)` instead (each a 1-element fp32
CUDA tensor, required there and refused under `"current"`) while the amax passes still publish this step's amax. The softmax
scale `scale_s = 2**FP8_SCALE_S_LOG2` (`= 2**8`) is a module constant; `scale_dp` -- the fp8 SDPA backward's dP scale, cuDNN's
contract -- is the caller's 1-element fp32 CUDA tensor at `execute(scale_dp=)`, required under `quant` and refused without, and
its descale is derived from it on device. `bwd.quant_scalars(workspace)` returns zero-copy 1-element fp32 views of the block's
scalar region, named `amax_dy` / `amax_do` / `amax_dqkvg` / `amax_dp`, `scale_dy` / `descale_dy` / `scale_do` / `descale_do` /
`scale_dqkvg` / `descale_dqkvg`, `alpha_b1` / `alpha_b2` / `alpha_b7` / `alpha_b8`, `descale_dp`, and the `QuantSpec`'s plan-time
constants (`scale_q` / `scale_k` / `scale_v` / `scale_o`, `descale_q` / `descale_k` / `descale_v` / `descale_o`, `descale_w_o` /
`descale_h` / `descale_w_qkvg`, `scale_s` / `descale_s`, `scale_dqkv`) -- stored by the step's FIRST launch (the prologue's
scalar-init job) from its kernel arguments, so nothing is written to the device at `compile()` and an `execute` on any stream
reads only what that stream wrote (synchronise the stream after the step before reading them; they are rewritten by the next
`execute`). Workspace: the bf16 `O_gated`, recomputed Q / K and
compact V regions are replaced by the e4m3 `dY8` / `dO8` / `O_gated8` / `Q8` / `K8` / `V8` / `dQKVG8` plus the 256-B scalar block
and the fp32 `dY` amax partials (one word per CTA of the prologue's amax job, at most `SMs x 8`) -- about +12 KiB/token at the
397B geometry --, the delta region is always carved, and the SDPA scratch is the fp8 chain's (its e4m3 dS chunk is half the
bf16 one). Determinism: the block's own kernels run no atomic (every gradient amax is a `max` over per-CTA partials); the one
`atomicMax` left is the fp8 SDPA row's `amax_dP`, an int32 fold of non-negative fp32 bit patterns and therefore order-free -- so
two executes are bitwise equal under every knob set (pinned by `test_fp8_two_runs_are_bitwise`). `fuse_gate_bwd` is accepted and
inert under `quant`: the fused delta is mandatory there. `fuse_wgrad_overlap` is served (the side-stream GEMMs fork after the slots and operands they read are written). Packed sequences
(`thd=True`, the THD section below) are served over the packed per-tensor fp8 training record as written: every stage but the SDPA
row is token-wise at `B = 1, S = T`, and the fp8 SDPA row runs its THD chain with the gate backward's delta as its external one --
at `B = 1, S = T` that is the packed head-major `[1, H_q, ceil128(T)]` delta the packed chain reads -- so no `dot` pre-pass runs
over the e4m3 payloads (one rounding of dO, the dense arm's delta contract).
Declined (typed, naming the attribute): e5m2 codes, an fp16 `dy` (the quantized
backward is bf16), a bf16 `saved.h` or bf16 weights with a `QuantSpec` and e4m3 codes without one (both ways), and a geometry whose Q / K
rebuild only the LDG norm + RoPE kernel can tile: the fused prologue runs the TMA kernel, whose `tile_rows` must divide `h_q`,
be a multiple of `h_kv` and of 4 (nothing in 1..16 does for `h_q = 20` MHA or `h_q = 6` over `h_kv = 2`; the bf16 backward
serves such a geometry through the LDG rebuild). There is no `B*S` rule: the
two weight-gradient GEMMs contract over the token axis with MN-major e4m3 operands, and the TMA 16-byte rule binds an
operand's contiguous axis only, so a ragged token count (S = 1000 at B = 1) is served with its weight gradients. Every bf16
decline (padding, `window_left == 0`, `d_model % 256`, Rubin only, ...) is unchanged.

**Quantized backward (MXFP8) -- `quant=MxQuantSpec`.** `GatedAttentionBlockBwd(..., quant=MxQuantSpec, grad_scaling="current")`,
with the MXFP8 training forward's own `MxQuantSpec` (`descale_w_o`, `scale_o`: the per-tensor pair of the out projection, the one
per-tensor side of that pipeline), differentiates the MXFP8 training record **as written** -- e4m3 `saved.h` and weights, the bf16
slab / `o` / `lse` / `rstd` -- from a bf16 `dy`, and returns bf16 `dh` / `dw_qkvg` / `dw_o` and fp32 `dw_*_norm`. What runs (launch
order, one stream; the per-tensor fp8 chain's shape -- two fused small-kernel launches and one dual-axis quantize, every byte
bitwise the unfused chain's): the fused PROLOGUE -- the scalar-block init (every slot zeroed, the MxQuantSpec's plan-time constants
from the launch's kernel arguments; no `descale_dp` -- the MXFP8 SDPA backward has no dP scalar), the amax of `dY` as per-CTA partials,
the post-norm / post-RoPE rebuild of `Q` and `K` from the slab on a one-head x 32-token tile with the MXFP8 quantizes of both written
straight out of registers, rowwise AND columnwise (`q8` / `sf_q`, `q_T8` / `sf_q_T`, `k8` / `sf_k`, `k_T8` / `sf_k_T`; `q8` / `sf_q` and
`k8` / `sf_k` bitwise the forward's own; no bf16 rebuild buffer), and the rowwise quantize of `V` straight from the slab's V band (the
backward's V operand is rowwise, so the forward's columnwise `v8` cannot serve) --; the per-tensor e4m3 quantize of
`dY` -- the ONE per-tensor gradient of this pipeline, at the `grad_scaling` recipe's scale (`"current"` derived on device,
`"delayed"` the caller's `execute(scale_dy=)`); the e4m3 out-projection dgrad `dO_gated = dY8 @ W_o8 * alpha`; the gate backward's
fp8 arm (`dO`, `dG`, the e4m3 `O_gated` at `scale_o` for the wgrad, `delta = rowsum(dO * O)` -- always, it is the MXFP8 SDPA
backward's external delta); ONE dual-axis MXFP8 quantize of `dO` -- rowwise (the row's dP operand) and columnwise (its dV operand)
from one read, in the SDPA's own scale-factor layouts --; the e4m3 out-projection wgrad `dW_o = dY8^T @ O_gated8 * alpha`; the Rubin
d=256 MXFP8 SDPA backward (`SdpaBwdDslSm107Mxfp8`, external delta, its block-scaled dS chain) into bf16 `dQ` / `dK` / `dV`; the fused
RoPE-adjoint + RMSNorm backward; the fused EPILOGUE -- the `dW_norm` reduce and the dual-axis MXFP8 quantize of `dQKVG` from one read, in
the GEMMs' canonical F8_128x4 scale-factor order: rowwise `[T, N]` for the dgrad and TRANSPOSED (32-token blocks along `T`) as the
contiguous e4m3 `[N, T]` for the wgrad --; the two block-scale projection GEMMs over transposed
operands, the E8M0 dequant exact in the MMA (no alpha): `dW_qkvg = dQKVG8^T . h^T` against the CALLER's `h_t` -- `h` re-quantized along
tokens, e4m3 `[d_model, T]` contiguous -- with its blob `h_t_sf`, and `dh = dQKVG8 . W_qkvg^T` against the caller's `w_qkvg_t` --
`W_qkvg` re-quantized along N, e4m3 `[d_model, N]` -- with `w_qkvg_t_sf` (quantized once per weight update). The four artifacts are
`execute` keywords (`h_t` / `h_t_sf` required when `need_dw_qkvg`, `w_qkvg_t` / `w_qkvg_t_sf` when `need_dh`, each refused
otherwise), validated on the host before any launch: dtype, shape, the contiguous K-major storage (a `.t()` view of the un-transposed
codes is refused by name), 16-B alignment, the blobs' padded byte count (`kernels.proj_gemm.sf_blob_bytes(d_model, T)` /
`(d_model, N)`). The scale-factor blob of a transposed artifact is sized by `sf_blob_bytes(rows, k) = ceil128(rows) x ceil128(k) / 32`,
which is the same number for `(rows, k)` and `(k, rows)`: the byte count does not validate the blob's orientation. A blob built over
the un-transposed matrix (the forward's `h_sf` handed as `h_t_sf`) passes every host check and produces a wrong weight gradient;
build it over the transposed matrix exactly as the artifact it scales, and verify a new caller against the reference once. The
small launches are fused as on the per-tensor fp8 chain: 10 block launches with every gradient (20 before -- the fused PROLOGUE
replaces the scalar init, the dY amax pass, the bf16 Q / K rebuild and the five SDPA-operand quantizes; the dual-axis `dO` launch the two
`dO` quantizes; the fused EPILOGUE the `dW_norm` reduce and the two `dQKVG` quantizes; every payload, scale-factor blob, scalar slot and
gradient bitwise the unfused chain's), plus the SDPA row's `1 + c*(2+q) + (g > 1)` with `q = 1` -- the block-scale arm of the row
launches its dQ GEMM once per head chunk, like the plain renderings (its dQ record takes `b_head_group` = the GQA group: B and its scale
factors are indexed by `h // group`; bitwise the per-member launches it replaced) --: **15** launches at the test geometry (S = 512, B = 2,
GQA 8/2, `c = 1`, Q/K RMSNorm on), **15** RoPE-only (the epilogue stays for the cast), **14** MHA, **15** at the 397B geometry (B = 1,
S = 512, GQA 32/2, `c = 1`, `g = 16`: the suite's own 397B census cell) -- the per-tensor fp8 chain's count at the GQA cells --, and more
at a padded `S` (the row's staging pads: 30 at S = 992 or S = 1008 under GQA with the weight gradients, 29 at the dgrad-only S = 1000, 28
at S = 992 MHA, 22 at S = 384), every figure counted by CUPTI on Rubin (cc 10.7; identical on a 204-SM and a 212-SM part -- two datasets
of one tree, since torch's Philox draws follow the SM count, and the accept suite's docstring carries both datasets' margins) in the MXFP8
backward's own suite (`test_mxfp8_launch_count_is_honest`, ten census cells: the launch records against an expectation computed from the
block's rows and the adapter's facts, never typed; 0 memsets, 0 memcpys). The two changes arrived one at a time and each was counted the
same way: the unfused chain over the row's per-member dQ measured 28 / 27 / 24 / 40 and 43 / 41 / 38 / 35 on the same cells, the unfused
chain over the single-launch dQ 25 / 24 / 24 / 25 and 40 / 38 / 38 / 32, the fused chain over the per-member dQ 18 / 18 / 14 / 30 and
33 / 32 / 28 / 25. Measured on Rubin cc 10.7 (212 SMs, locked clocks, CUPTI device time, the 397B geometry at S = 8K) the PROLOGUE runs
in 0.099 ms against the eight launches it replaces at 0.184 (+85 %), the dual-axis `dO` launch in 0.042 against 0.116 (+178 %), the
EPILOGUE in 0.088 against 0.122 (+39 %); the PROLOGUE reads its own bytes at 3.7 TB/s against the per-tensor fp8 prologue's 6.3 TB/s
(its two-pass 32-token tile is resident 5-6 CTAs per SM against the shipped tile's 14), which is why the workspace carve is keyed on
the prologue's ARM (`mx_prologue_arm`): the alternative arm that keeps the bf16 TMA store and quantizes `q_T` / `k_T` from the bf16
buffers by a dual-axis launch carves the two bf16 rebuild regions again and changes nothing else. Under GQA the MXFP8 SDPA backward folds its per-Q-head dK partials in fp32
and rounds the sum once, like the reference, while its per-Q-head dV partials are bf16 (the kernel stores them from its epilogue;
fp32 ones do not fit its 327 KiB shared-memory budget), so dV carries one bf16 rounding per group member where a once-rounded
reference carries one in total (relative RMS about 3e-3 at a group of 4, the geometry the tests run, measured on the per-tensor fp8
row before it moved to fp32 partials); the modelled oracle folds dV the same way and the distance to a once-rounded fold is
reported per cell. `bwd.quant_scalars(workspace)` returns the same 29 views; eight are live -- `amax_dy` / `scale_dy` / `descale_dy` /
`alpha_b1` / `alpha_b2` (the dY point) and the constants `scale_o` / `descale_o` / `descale_w_o` -- and the other 21 read exactly 0.0
(no dO / dQKVG / dP scalar: those gradients are block-scaled; no per-tensor static scales of an fp8 record). `scale_dp` / `scale_do` /
`scale_dqkvg` are refused at `execute`. Workspace: the bf16 `O_gated` and compact V regions are replaced by the per-tensor `dY8` /
`O_gated8`, every block-scaled payload with its scale-factor blob (`dO8` rowwise and columnwise, `Q8` / `K8` rowwise and columnwise,
`V8` rowwise -- `D / 32` scale bytes per row --, `dQKVG8 [T, N]` and `dQKVG8^T [N, T]` with their padded canonical blobs, the
transposed pair only when the projection weight gradient is requested), the 256-B scalar block and the `dY` amax partials; the
bf16 recompute of Q / K is not carved either (the fused prologue quantizes it out of registers). Measured with `get_workspace_size()`
at default knobs (Rubin cc 10.7): **+64.6 KiB/token** at the 397B geometry, B = 1, S = 512 (122,912,000 B against the bf16 block's
89,031,168 B; +81.6 with the unfused chain's bf16 rebuild regions), of which the block's own carve is +47.75 KiB/token and the MXFP8
row's scratch +16.9 (+17.9 at S = 1024, +19.9 at S = 2048: the row's share grows with S, the carve's is flat; the GEMM scratch is 0 on
both); at the test geometry (B = 2, S = 512) +4.9 KiB/token (carve +12.65, row scratch +4.22, the bf16 GEMM plans' 12 MiB split-K scratch
gone on the MXFP8 K64 block-scale plans). The delta region is always carved, and the SDPA scratch is the MXFP8 row's (its block-scaled
dS: two e4m3 payloads plus their E8M0 atoms, `2 + 2/32` bytes per element; under GQA its bf16 `dV` and fp32 `dK` per-Q-head
partials). Determinism: no atomic anywhere on the MXFP8 chain -- the one amax (`dY`) is a max over per-CTA partials, the row's GQA
fold is a fixed-order reduce, the block-scale GEMMs are deterministic -- so two executes are bitwise equal under every knob set.
`fuse_gate_bwd` is accepted and inert (the fused delta is mandatory); `fuse_wgrad_overlap` is served (the side-stream GEMMs fork after
the operands and blobs they read are written). Declined (typed, naming the attribute) on top of the per-tensor fp8 arm's: `thd=True` with an MxQuantSpec (dense-only: the
backward's SDPA-layout MX quantize stages run the quantizer's dense arm only; the packed MXFP8 record the forward writes and the packed head-major delta both exist, the packed MXFP8 backward is a follow-up -- the per-tensor fp8 backward is served packed), `B*S % 32 != 0` when a projection weight gradient is requested -- the weight-gradient GEMM contracts over the
token axis through one E8M0 scale per 32-element K block, and the transposed quantize writes whole 32-token blocks -- (pass
`need_dw_qkvg=False`, pad or batch the sequence to a multiple of 32, or run the per-tensor fp8 backward, whose weight gradients take
no block scales; the data gradients `dh` / `dW_o` are served at any `T`), an artifact given without its need or a need without its
artifact, a `.t()`-view artifact, a wrong blob byte count or dtype. The block binds the MXFP8 row's dense plan with an external
delta: nothing here changes the row's capabilities.

**The fp4 weight modes' backward -- `quant=MxQuantSpec(w_qkvg_dtype=torch.float4_e2m1fn_x2)` and / or `MxQuantSpec(o_fp4=...)`.** The MXFP8
pipeline's two fp4 modes train on the unfused pipeline: their training forward (`save_for_backward=True`) writes the SAME record as the
MXFP8 training forward, byte for byte -- the fp4 tail replaces only the workspace's per-tensor `o8` by the e2m1 `o4` and its blob, and the
SDPA writes bf16 `o` -- and `GatedAttentionBlockBwd(quant=<that spec>)` is the MXFP8 backward with the two DATA-gradient GEMMs on the FROST
block-scale catalog's fp4 rows (the forward's own renderings at the dgrad's shapes) over the caller's TRANSPOSED e2m1 artifacts. The
weight gradients stay 8-bit (`h` is e4m3 in every fp4 mode: `dW_o` per-tensor e4m3, `dW_qkvg` the MXFP8 block-scale GEMM). An MXFP4 `W_qkvg`
puts `dh = dQKVG8 . W_qkvg^T` on the mixed e4m3 x e2m1 row, so `w_qkvg_t` is the packed e2m1 `[d_model, N // 2]` (`torch.float4_e2m1fn_x2`,
two codes per byte along N, low nibble = even n) with the UNCHANGED E8M0 / 32 `w_qkvg_t_sf` -- the same keyword, its dtype following
`w_qkvg_dtype`; nothing else changes (the MXFP8 launch census -- 15 at the test geometry, 15 RoPE-only, 14 MHA -- and the same carve). An fp4 `W_o` (`o_fp4`; `scale_o == descale_w_o == 1.0` by `MxQuantSpec`'s
own rule) puts `dO_gated = dY . W_o^T` on a block-scale row over `execute(w_o_t=, w_o_t_sf=)` (appended; required iff `o_fp4` whatever the
`need_*` set, since the gate backward needs `dO_gated`; refused otherwise) -- `W_o` re-quantized along `d_model` in `o_fp4`'s format, packed
e2m1 `[H_q * D, d_model // 2]` with its blob (`sf_blob_bytes(H_q * D, d_model, block)`: e4m3 scales per 16 for `Fp4Format.NVFP4`, E8M0 per 32
for `Fp4Format.MXFP4`; the other format's blob is a byte-count decline) -- and adds ONE launch, a block quantization of `dY` right after
the per-tensor one: under MXFP4 the MX-rowwise e4m3 `dY` with its canonical E8M0 blob (the mixed row again; the gradient stays 8-bit), under
NVFP4 `dY` itself cast to NVFP4 -- packed e2m1 with e4m3 scales per 16, the NVFP4 x NVFP4 row, the only catalog row for an e2m1 side with
e4m3 scales -- **as a two-level cast**: an NVFP4 block's e4m3 scale is `max(amax / 6, 2^-9)`, so a 16-element block of a raw output gradient
whose amax sits under the e2m1 midpoint `2^-11` would quantize to all zeros. The cast is therefore taken at the per-tensor power-of-two scale
of the `dY` quantization (`scale_dy`, read from its slot in-kernel: the tensor's amax lands in `[224, 448]` and a block is zeroed only when
its amax sits more than about 19 octaves below the tensor's) and undone in the gate backward, which multiplies the dgrad's output by
`descale_dy` before every use -- exact for a power of two, so a power-of-two scaling of `dY` leaves every gradient bitwise equivariant (the
suite's `2^-13` pin, which the single-level cast fails on its first assertion: at `2^-13` it keeps about 0.004-0.006 % of the codes).
Wherever the single-level scale byte was a normal e4m3 value the codes are identical (a power of two only shifts the exponent), so the
pre-scale is purely a floor remedy. Launches: the MXFP8 count under an MXFP4 `W_qkvg` alone, + 1 under an fp4 `W_o` (16 at the test geometry
with Q/K RMSNorm, 16 RoPE-only, 15 MHA, 31 at the padded GQA cell S = 992 with the weight gradients; the fp4 suite's census counts the
test-geometry, MHA and padded cells by CUPTI on Rubin against an expectation computed from the MXFP8 table plus the one dY block quantize,
never typed). Workspace: `dy_mx8` + its blob (MXFP4) or `dy4` + its blob (NVFP4) appended last; every MXFP8 region
is unchanged. The weight gradients are allocated at their logical shapes -- `(n_qkvg, d_model)` and `(d_model, H_q * D)` in `dy`'s dtype --
never with `empty_like(<weight>)`: a packed e2m1 weight's `.shape` is its storage `[rows, K // 2]` (the convenience wrapper sizes them from
the geometry; a packed e2m1 weight carries `requires_grad` like any other tensor, so its gradient is requested the same way). `bwd.quant_scalars()` reads the same eight live slots, `scale_o` / `descale_o` / `descale_w_o` at 1.0 and `alpha_b2` published
but read by no GEMM (the block-scale dgrad has no alpha). The oracle dequantizes the transposed e2m1 artifacts through their blobs -- two
fake-quants of one master weight along its two axes, the fp4 training recipe's straight-through estimator -- and takes the same two-level
`dY` point under NVFP4; the accept suite is `test_block_backward_fp4.py` (five configurations: MXFP4 `W_qkvg`; NVFP4 `W_o`; MXFP4 `W_o`; both
with either `W_o` format). Declined (typed, naming the attribute): an artifact in the wrong dtype for its weight's mode (an e4m3 `w_qkvg_t`
under an e2m1 `W_qkvg` and the reverse; uint8 bytes of packed codes get the `.view(torch.float4_e2m1fn_x2)` hint), a logical `[rows, K]` fp4
artifact (twice the packed data), a `.t()` view, `w_o_t` / `w_o_t_sf` without `o_fp4` or missing with it, the other format's `w_o_t_sf`, an
fp4 `h` (not served: `h` stays e4m3), the fp4 modes with a fused training forward (the forward's own declines), `B*S % 32 != 0` with a
projection weight gradient (inherited).

**Workspace** (`get_workspace_size()`, after `compile()`): the block's own regions -- `dO`, the `[T, N]` `dqkvg` slab,
`O_gated`, the recomputed Q / K / V, compact `dQ` / `dK` / `dV` -- `(6*H_q + 6*H_kv) * D * 2` bytes per token in
bf16 (102 KiB/token at the 397B geometry), plus the SDPA backward's scratch (`delta` -- the block's own region under
`fuse_gate_bwd` -- and the per-Q-head `dK` / `dV`
partials, ~32 KiB/token, and ONE dS chunk of `qh_chunk x S_q_pad x S_kv_pad x 2` bytes with `qh_chunk` a multiple of the
GQA group: **4.25 / 8.50 / 33.0 GiB at S = 8K / 16K / 32K** for the 397B geometry at B=1), plus the `dW_norm` partial
planes (`(n_ctas_q + n_ctas_k) x D x 4` bytes, at most `2 x SMs x 8 x 1 KiB`), plus the GEMMs' scratch
(`max(plan.workspace_bytes)`: 0 at 397B, 12 MiB at the test geometry -- the backend heuristic's split-K partials, never
launched on the forced tile; under `fuse_wgrad_overlap` a second such region, `gemm_scratch_side`, for the two
side-stream wgrad GEMMs, sized to their plans, appended last). At S=32K, B=1, 397B: ~36 GiB in total, dominated by the dS chunk.

## Requirements and limits

- `CUDNN_FRONTEND_ENABLE_FROST_ENGINES`: every FROST engine is opt-in, and the block drives them. The flag is read LIVE, when a
  GEMM plan's ranked list is built at `compile()` -- not once at `import cudnn` -- and a host module must never set it for
  the process (it flips every opt-in engine into every FE graph the process builds). The block does not need it: a dense
  bf16 / f16 projection whose `N` the forced 256-wide GEMM tile divides (`n_qkvg`, `d_model` and `h_q * d_head` all
  `% 256 == 0`, as at every served geometry), or whose shape the small-M route names a tile for (`B x S_q <= 128` on a dense
  bf16 / f16 projection whose N the route's tile grid divides -- the indexer band's 13952-column slab included: 109 x 128-row
  tiles), compiles with the flag UNSET on FROST's JIT-only route
  (`plan.route == "jit-only"`: no backend plan, no cuDNN handle at execute; `frost_gemm` never enters a ranked list), and the
  block-scale (MXFP8 / fp4-weight) projections log a warning and take the same route. With the flag set a plain declaration's
  backend admits the graph and pins the same JIT by name (`graph+jit`); a `capture_safe=True` declaration never enters the
  ranked list whatever the flag says. What still needs the flag: a dense shape WITHOUT a JIT tile (the graph route IS the plan
  there: the band slab past 128 rows, an N off both tile grids) and the per-tensor-fp8 projections -- a `RuntimeError` at
  `compile()` whose message names the flag (the GEMM plan is built there, not at `check_support()`). Nothing ever falls back
  to a cuDNN backend plan, so an unset flag is never a silently slower block. The SDPA stage binds its FROST class directly and
  does not consult the flag.
- CUDA-graph capture of a serving stack (the FIRST execute of a fresh declaration captured on a NEW stream, with no eager
  warm-up): on the forced-tile JIT route the captured first execute launches nothing and replays bitwise an eager run; the
  GRAPH GEMM route (no forced tile) creates a cuDNN handle at its first execute per `(device, stream)`, which cannot happen
  inside a capture -- a typed `RuntimeError` naming the route and the remedy (one eager execute on the capture stream first,
  or the forced-tile route), and no handle is created inside the capture; `compile()` under capture is refused the same
  way (the GEMM plan builds a cuDNN graph and JIT-compiles a kernel there). The GEMM driver's serving declaration,
  `build_proj_gemm(capture_safe=True)`, never enters the ranked list -- no backend plan, no FROST-engine compile, whether
  the flag is set or not -- so the plan is on the JIT route by construction: a refused forced compile is a typed
  `NotImplementedError` naming the tile and the reason, never the warning + graph-heuristic fallback the plain declaration
  takes. The block sets `capture_safe` on each dense projection stage of a paged declaration iff the driver's `auto` resolver
  names a JIT tile for the stage's own shape -- so the indexer band's slab (N 13952) is capture-safe at decode row counts
  through the small-M route (the band capture / replay cell of `test_block_qsa_decode.py`) and eager past 128 rows, where a
  capture is the driver's typed refusal. The quantized block's first-execute scale write is NOT refused under capture (the
  warm-up / re-capture contract above is the rule): `test_block_serving_capture.py`.
- Rubin (SM107) only; cuDNN 9.x, `nvidia-cutlass-dsl >= 4.8.0.dev0` (the Rubin arch names), torch.
- Backward: bf16 / fp16 (both against fp64 autograd on Rubin: `test_block_backward.py`) -- and per-tensor fp8 over the fp8
  training record (`quant=QuantSpec`: bf16 `dy` and gradients, dense or packed (`thd=True`), any `B*S`, head counts the TMA Q / K rebuild tiles),
  and MXFP8 over the MXFP8 training record (`quant=MxQuantSpec`: bf16 `dy` and gradients, dense only, `B*S % 32 == 0` when
  a projection weight gradient is requested, the caller's transposed artifacts `h_t` / `h_t_sf` / `w_qkvg_t` / `w_qkvg_t_sf`
  at `execute`; the fp4 weight modes on the same record -- a packed e2m1 `w_qkvg_t` under an MXFP4 `W_qkvg`, `w_o_t` / `w_o_t_sf`
  under `o_fp4`); **Rubin only -- the block
  binds ONE FROST engine class per declaration (`SdpaBwdDslSm107`, the Rubin d=256 SDPA backward; under `quant` its
  per-tensor fp8 row, `SdpaBwdDslSm107Fp8`, or its MXFP8 row, `SdpaBwdDslSm107Mxfp8`) and never falls back to the cuDNN
  backend's d=256 backward, exactly as the forward binds its FROST SDPA class (AGENTS.md Rule 9, a stated design
  decision: every other device is a typed decline)**; `d_head = 256`; `seq_len >= 2` (S = 1 is decode, out of the
  prefill bodies' scope); `d_model % 256 == 0` (the forced GEMM tile behind the determinism contract; `h_q * d_head`
  satisfies it through `d_head = 256`); `saved.proj_slab` required (the gate-copy save set, `saved_gate_copy=True`, is
  served by a later PR); no `seq_lens` yet (typed -- at declaration from `seq_lens_present=True` or a sample record
  whose `seq_lens` is a tensor, and at `execute` for the record handed there: a padded record contradicts a dense
  declaration and is refused before any launch; the `sdpa_bwd_sm107` row declines padding); `window_left > 0` only
  (or -1), `window_right` unbounded or 0 only; `dw_norm_dtype=torch.float32` only; `rope_dim > 0`;
  `get_workspace_size()` after `compile()`. A dense `S % 128 != 0` is differentiated like any other record: the
  forward's SDPA row masks its KV tail in-kernel (#1520); causal covers the tail.
- Packed sequences (`thd=True`), forward and backward: bf16 / fp16; the per-tensor FP8 and MXFP8 unfused forwards (the
  fp4 modes included), inference and training, and the per-tensor fp8 BACKWARD over its packed record as written
  (`quant=QuantSpec` with the packing knobs: the fp8 SDPA row's THD chain reading the gate backward's packed delta, every
  other stage token-wise at `B = 1, S = T`); the packed bf16 backward takes a packed quantized record with the dequantized
  `h` and weights (a record handed to it with its e4m3 `h` is the same typed decline as on the dense side, after the
  packed-length checks); `fuse_norm_rope` (bf16 / fp16 inference). `num_sequences >= 1`, `2 <= max_seq_len <= T`, `num_sequences * max_seq_len >= T`; the lengths
  tensor contiguous 1-D int32 on `h`'s device with `B` (`cu_seqlens=False`) or `B+1` (`cu_seqlens=True`) entries; every
  length `<= max_seq_len`, the lengths summing to `T` (the caller contract, not host-validated); the training record
  carries `saved.seq_lens` and `saved.seq_lens_form`. Declined (typed): `seq_lens_present` together with `thd`,
  `fuse_gate`, the fully fused quantized pipelines, the MXFP8 backward over a packed record, the packing knobs on a dense
  block.
- `d_head = 256` (the Rubin d256 SDPA flavor with the fused gate); `d_model % 128 == 0` under MXFP8.
- Paged KV-cache write-through (`paged_kv_page_size`, a positive multiple of 16): bf16 / f16 inference, dense `[B, S, d_model]`
  and packed (`thd=True`); the pools are the activation dtype, `[num_pages, H_kv, P, D]` views with `D` contiguous and every other
  stride a whole number of 16-byte vectors (HND compact, NHD by strides, or the two halves of a unified `[num_pages, H_kv, P, 2D]`
  row); `quant` and `save_for_backward` together with it are typed declines ("Serving: write-through into a paged KV cache" above).
- The EXTEND form (`kv_source="paged"`): bf16 / f16 inference on a PACKED declaration with `paged_kv_page_size > 0`,
  `geometry.causal_bottom_right=True` (`is_causal=True`), no window, no `QsaSpec`, `d_head=256`, `return_lse=False`,
  `fuse_gate=False`; the three samples at declaration, `block_table` / `kv_lens` at execute (refused under `kv_source="self"`); the
  pools match the samples exactly; `token_capacity` (THD only, equal to `T`, inference only) bounds the live tokens. Validated at
  pages 16 / 32 / 64 / 128 on all three pool layouts, bf16 and f16 ("Serving: the extend form" above); every other combination is a
  typed decline `serving_envelope` predicts.
- The serving DECODE form (`kv_source="paged"` on a dense `[B, S_q]` block over `paged_kv_page_size > 0`): bf16 / f16 inference,
  `d_head = 256`, `is_causal` with `causal_bottom_right` (no window), the batch uniform, `block_table` + `kv_lens` at `execute`
  (`seq_lens` refused), the pools and the table exactly the declared samples; the served envelope `S_q <= 32` (the claimed envelope
  `S_q <= 4` on the Rubin d256 decode tile; the MTP verify step to `S_q = 8` on the one- and two-stream routes; the prefill route, the
  token-unit cells past `S_q = 4` and the unpacked arm up to 32 tokens declared, unmeasured), a typed decline beyond; `fuse_gate` rides
  the decode tile's split combine or the paged prefill body's
  epilogue, else its own launch ("Serving: the decode form" above).  On a `QsaSpec` geometry the same
  declaration is the block-sparse DECODE / extend-with-prefix mode: stage (4) the sparse core reading the pools, one list per sequence
  at `S_q <= 4` (`block_ids [B, top_k]`) and one per row past it (any `S_q`), `decode_split_kv` in `[1, ceil((top_k + 1) / 32)]` or the
  fill rule, `fuse_gate` and `return_lse` served, `sample_block_table` contiguous, `fuse_prepare` served (`None` = the measured
  default), the in-block indexer's DECODE step over the pools served (`index_source="indexer"`: bf16, the per-row list, the paged
  compressed-key cache + the ring + its table as samples; "The in-block indexer's DECODE step over the pools" above); the packed form is
  a typed decline.
- The e4m3 KV cache (`kv_cache_dtype=torch.float8_e4m3fn`): bf16 activations with `paged_kv_page_size > 0`; the quantizing write-through
  on the dense prefill forms (QsaSpec or dense) and the block-sparse DECODE mode over e4m3 pools at `S_q <= 4` (the gathered cast + the
  d256 decode tile, out-of-place norm, the gate as its own launch, unsplit, no LSE); `execute(k_scale=, v_scale=)` 1-element fp32 CUDA
  tensors; the packed prefill, the EXTEND form, the dense DECODE form, `S_q > 4`, `return_lse`, a split, `fuse_norm_rope`, f16, e5m2 and
  `quant` with it are typed declines ("Serving: the e4m3 KV cache" above).
- FP8 / MXFP8: the UNFUSED pipelines train (`save_for_backward=True` writes the bf16 record described above), the fp4
  modes of the MXFP8 pipeline included; the fully fused quantized pipelines are inference only. The backward is bf16 / fp16
  -- and per-tensor fp8 / MXFP8 over the quantized training records (`quant=QuantSpec` / `quant=MxQuantSpec`, the record as
  written, the fp4 weight modes through their transposed e2m1 artifacts); the bf16 backward takes either record with the
  dequantized bf16 `h` and weights.
- FP8: a dense (no-mask) sequence length off a multiple of 128 is served; the SDPA masks the KV tail in-kernel
  (#1520); MXFP8: e4m3 codes only (e5m2 is a typed decline);
  the fully fused MXFP8 path needs `scale_o == 1.0` and, at `B > 1`, `S % 128 == 0` (a scale-factor atom is per
  sequence).
- `fuse_gate` and `fuse_norm_rope` are inference-only specializations (no pre-gate `O`, no pre-norm Q/K).
- Block-sparse attention (`QsaSpec`): bf16 / f16 inference, dense `[B, S, d_model]` (with or without `seq_lens`) or packed
  (`thd=True`, caller lists with sequence-relative ids), `d_head == 256`,
  `h_q // h_kv <= 16`, caller lists (`block_ids`, `top_k` a multiple of 4 in `[4, 512]`) or the in-block indexer
  (`index_source="indexer"`: bf16, the band, `index_head_dim` 128, `index_heads` in the scorer's packed groups; dense prompts, and the
  DECODE step over the pools); the indexer
  band on the UNFUSED projection; `fuse_gate` (the sparse core's epilogue gate) composes with it, dense or packed;
  `save_for_backward`, `quant`, `fuse_norm_rope` together with the band,
  `causal_bottom_right` on the block's own K / V (`kv_source="self"`; over the pools it is the block-sparse decode mode, where the
  in-block indexer's DECODE step is served since 2026-10-21), and `thd` together with the in-block indexer are typed declines ("Sparse attention (QSA)" above; `thd` together with the paged write-through is served -- the stage addresses tokens); what a serving stack hands in -- the list
  form, the count rule, the decode-mode list, the pools, the indexer's inputs and outputs -- is the contract page "Serving:
  index lists and paged KV" above.
- fp4 (`MxQuantSpec.w_qkvg_dtype` / `o_fp4`): MXFP8 pipeline only (unrepresentable on `QuantSpec` / bf16); trained on the
  unfused pipeline (`save_for_backward=True` writes the MXFP8 record; the fused forks stay inference-only) and differentiated
  by `GatedAttentionBlockBwd(quant=MxQuantSpec)` over the caller's transposed e2m1 artifacts; no global per-tensor scale in
  either fp4 format (`scale_o == descale_w_o == 1.0` under `o_fp4`); `d_head % (4 * block) == 0` under `o_fp4`
  (whole 4-block scale words per head: 64 for NVFP4, 128 for MXFP4; `d_head = 256` passes both); the MXFP4
  `W_qkvg` is served on both pipelines -- fully fused through the MXFP8 projection fork's e2m1-B arm
  (`NormRopeFusionParams.weight_fp4`, feature-detected: a checkout whose fork lacks the field declines typed) under the
  fused pipeline's own rules (inference only, dense only, `S % 128 == 0` at `B > 1`, `scale_o == 1.0`). `h` stays e4m3
  (an fp4 `h` is not served), and an fp4 `W_o` with an e4m3 `O` is not a served pairing.

## Performance

Whole block, B=1, `h_q=32 h_kv=2 d=256 d_model=5120` (the 397B geometry), Rubin perf node (212 SMs, SM clock
locked at 2376 MHz), speedup over the same bf16 torch chain (median of 5 launch-interleaved rounds x 30 launches;
the bf16 FROST control pair stayed within 0.6 %). The fp4 modes (MXFP4 weights, NVFP4 / MXFP4 `O`) are measured
separately below, at `d_model = 4096`, and are not columns of these `d_model = 5120` tables.

Causal:

| S | bf16 FROST | bf16 fully fused | FP8 unfused | FP8 fully fused | MXFP8 unfused | MXFP8 fully fused |
|---|---|---|---|---|---|---|
| 2048 | 2.96x | 3.16x | 4.22x | 4.90x | 3.84x | 4.30x |
| 4096 | 3.06x | 3.27x | 4.52x | 5.28x | 4.10x | 4.81x |
| 8192 | 2.79x | 2.95x | 4.33x | 4.90x | 4.06x | 4.68x |
| 16384 | 2.37x | 2.44x | 3.93x | 4.27x | 3.75x | 4.15x |
| 32768 | 1.90x | 1.96x | 3.41x | 3.61x | 3.22x | 3.44x |

Dense (no mask):

| S | bf16 FROST | bf16 fully fused | FP8 unfused | FP8 fully fused | MXFP8 unfused | MXFP8 fully fused |
|---|---|---|---|---|---|---|
| 2048 | 2.78x | 2.98x | 4.07x | 4.82x | 3.85x | 4.45x |
| 4096 | 2.74x | 2.93x | 4.24x | 4.87x | 4.00x | 4.58x |
| 8192 | 2.33x | 2.44x | 3.84x | 4.27x | 3.71x | 4.13x |
| 16384 | 1.85x | 1.90x | 3.37x | 3.63x | 3.30x | 3.51x |
| 32768 | 1.50x | 1.53x | 2.92x | 3.03x | 2.81x | 2.86x |

fp4 modes, `d_model = 4096` (not the 5120 of the tables above; `h_q=32 h_kv=2 d=256`, QK-norm on), B=1, Rubin perf node
(212 SMs, SM clock locked at 2376 MHz and SAMPLED per row: 2364 MHz at 4K, 2340 / 2364 at 8K, 2184 / 2100 at 16K and
2052 / 1968 at 32K causal / dense -- the lock power-caps at the long shapes), speedup over the same bf16 torch chain (median
of 5 launch-interleaved rounds x 30 launches, every arm of a row in one process; the MXFP8 FROST control pair within 0.6 % at
4K-16K and 2.0-2.5 % at dense 32K under the cap). S = 2048 is not quoted: in a ten-arm process that row is a sub-millisecond
window whose control pair read 30-44 %. The dense S = 32768 cells of BOTH fully fused NVFP4 `O` columns read 11-12 %
slower than their unfused twins -- a standing anomaly of the fused NVFP4 `O` pipeline, independent of the weight format.

Causal:

| S | MXFP8 unfused | MXFP8 fully fused | MXFP4 weights unfused | MXFP4 weights fully fused | NVFP4 O fully fused | MXFP4 weights + NVFP4 O fully fused |
|---|---|---|---|---|---|---|
| 4096 | 4.32x | 4.75x | 4.42x | 4.84x | 4.80x | 4.87x |
| 8192 | 4.24x | 4.71x | 4.35x | 4.76x | 4.71x | 4.76x |
| 16384 | 3.87x | 4.15x | 3.96x | 4.17x | 4.17x | 4.18x |
| 32768 | 3.24x | 3.39x | 3.32x | 3.41x | 3.47x | 3.47x |

Dense (no mask):

| S | MXFP8 unfused | MXFP8 fully fused | MXFP4 weights unfused | MXFP4 weights fully fused | NVFP4 O fully fused | MXFP4 weights + NVFP4 O fully fused |
|---|---|---|---|---|---|---|
| 4096 | 4.18x | 4.55x | 4.26x | 4.60x | 4.54x | 4.58x |
| 8192 | 3.83x | 4.12x | 3.91x | 4.16x | 4.11x | 4.14x |
| 16384 | 3.32x | 3.46x | 3.37x | 3.48x | 3.40x | 3.43x |
| 32768 | 2.82x | 2.88x | 2.89x | 2.91x | 2.57x | 2.56x |

Read across a row: the fully fused MXFP4-weight block is faster than the fully fused MXFP8 block by +1.9 / +1.0 / +0.6 /
+0.6 % (causal, 4K .. 32K) and +1.1 / +1.0 / +0.5 / +1.0 % (dense) -- the halved weight bytes of a projection that stays
MMA-bound (55-81 % of the 8-bit K32 MMA cap, causal); the 4K / 8K cells clear their control pair by more than 3x, the 16K /
32K cells of both masks sit under 2x theirs (causal -0.32 / -0.38 %, dense -0.59 / +2.53 %) and are reported, not claimed --
and faster than the unfused MXFP4-weight block by +9.6 / +9.5 / +5.3 / +2.6 % (causal) and +7.9 / +6.3 / +3.1 / +0.7 %
(dense), the fusion itself. With both fp4 modes the fully fused block sits within +1.4 / +1.0 / +0.1 / +0.1 % (causal) of the
fully fused NVFP4 `O` block, its 16K / 32K cells inside the control spread.

Backward, `fuse_gate_bwd` (the gate backward feeding the SDPA backward's delta): whole-backward wall time of the bf16 block
backward at the 397B geometry, B=1, causal, QK-norm on, Rubin perf node (212 SMs), knob off and on interleaved launch by launch
in one process with the knob-off arm timed twice as the control, 3 rounds x 20 launches per process, 3 fresh processes per S;
`+X % = unfused ms / fused ms - 1`, positive = the knob is faster. The gradients were bitwise equal between the arms in every
process.

| S | knob off (ms) | knob on (ms) | fuse_gate_bwd vs unfused | control pair |
|---|---|---|---|---|
| 2048 | 0.528 | 0.516 | +2.3 % | within 0.2 % |
| 8192 | 2.424 | 2.389 | +1.5 % | within 0.6 % |
| 32768 | 25.26 | 25.03 | +0.7 % (positive in 3 of 3 processes, but within 2x the control: a weak claim) | within 0.5 % |

The SM clock was locked at 2376 MHz but power-capped during the 8K and 32K runs (sampled 2150-2376 and 1550-1850 MHz), so
only the interleaved ratios are quoted; the absolute 8K / 32K milliseconds are capped-clock numbers. The knob removes one
launch and one read each of `O` and `dO` (32 KiB/token) per backward; it stays off by default.

Backward, `fuse_wgrad_overlap` (the two weight-gradient GEMMs on the block's side stream): the same protocol --
whole-backward wall time of the bf16 block backward at the 397B geometry, B=1, causal, QK-norm on, Rubin perf node
class (212 SMs), knob off and on interleaved launch by launch in one process (shuffled slot order) with the knob-off
arm timed twice as the control, 3 rounds x 20 launches per process, 3 fresh processes per S; `+X %` = the median over
processes of the per-process interleaved ratio `in-order / overlapped - 1` (the ms columns are medians of the
capped-clock absolute times and need not reproduce it), positive = the knob is faster. The gradients were bitwise
equal between the arms in every process and the kernel count of one backward was unchanged (the same launches on
another stream). Two settings of the other knob, since the two compose:

With `fuse_gate_bwd` off in both arms:

| S | knob off (ms) | knob on (ms) | fuse_wgrad_overlap vs in-order | control pair |
|---|---|---|---|---|
| 2048 | 0.514 | 0.499 | +2.8 % | within 0.2 % |
| 8192 | 2.326 | 2.305 | +0.5 % (positive in 3 of 3 processes, within 1.2x the control: a weak claim) | within 0.4 % |
| 32768 | 22.88 | 22.72 | +0.7 % | within 0.3 % |

With `fuse_gate_bwd` on in both arms (the fused configuration):

| S | knob off (ms) | knob on (ms) | fuse_wgrad_overlap vs in-order | control pair |
|---|---|---|---|---|
| 2048 | 0.508 | 0.495 | +2.7 % (positive in 3 of 3 processes, within 1.5x the control: a weak claim) | within 1.8 % |
| 8192 | 2.286 | 2.259 | +1.0 % | within 0.2 % |
| 32768 | 22.72 | 22.69 | +0.7 % (positive in 3 of 3 processes, within 1.8x the control: a weak claim) | within 0.4 % |

What overlaps, from the CUPTI timeline of one backward at S=8K: `dW_o` runs alongside the Q/K recompute, the V
compaction and the chain's first kernel, and `dW_qkvg` alongside the `dW_norm` reduce and the tail of the `dh` dgrad;
the persistent SDPA backward chain occupies every SM, so the side stream mostly removes the launch gaps around the two
GEMMs rather than hiding them (the CUPTI-measured time two kernels of one backward were in flight together rose from
about 6 us in order to about 65 us with the knob at S=8K, with `fuse_gate_bwd` off or on: `dW_o` co-resident for 32-34
us, `dW_qkvg` for 31-32 us). The gain is therefore largest at small S and shrinks as the chain's share grows. The clock
was locked but power-capped under the sustained 8K / 32K chain, so only the interleaved ratios are quoted. The knob
stays off by default.

Packed sequences (THD), forward -- the packed block against the dense block, Rubin (cc 10.7, 212 SMs, SM clock locked at
2376 MHz and sampled during every cell: 1872-2328 MHz, the lock power-caps under the long cells), both geometries 32/2
(`h_q=32 h_kv=2`) and 64/8 (`h_q=64 h_kv=8`) at `d_model = 4096` (not the 5120 of the forward tables above), `d_head = 256`,
RoPE 64, Q/K RMSNorm on, causal and dense (no mask), bf16, the unfused FP8 block and the unfused MXFP8 block (the three
quantized / unquantized pipelines over the SAME bf16 draws). Every cell is one process holding the six FROST blocks (each dtype's
packed block and its dense twin: identical FLOPs, identical kernels except the SDPA's packed specialization and, for MXFP8, the
packed scale-factor layout of its three `quantize_mxfp8` stages) and the bf16 torch chain, slots round-robin by launch with the slot
order shuffled per iteration, 3 rounds of 150 / 60 / 20 launches per slot (2K / 8K / 32K; varlen 60), median per slot per round then
the median of rounds (CUDA events around the whole block), the packed bf16 arm timed twice as the control pair (-1.97 .. +0.74 % over the cells: within 0.9 % in 15 of the 16, the 64/8
causal 32K x 1 cell at -1.97 % makes that cell's numbers a weak claim). `packed overhead = packed ms / dense ms - 1` on uniform packings (`B` sequences of `S` tokens each: the same FLOPs;
positive = the packed block is slower); the varlen cell packs `[2048, 4096, 6144, 8192]` (`B=4`, `max_seq_len = 8192`, 20480
tokens) and reports TFLOP/s on its exact per-sequence FLOPs -- the sum over the four sequences of the one FLOP model every row of
these tables uses, `analytic_flops` of `test/python/gated_attention_block/cutedsl/benchmark_baseline.py`: both attention BMMs over
`S_i^2`, halved under the causal mask, plus the dense `qkv_gate` and output projections over the `S_i` tokens, so a row's bf16, FP8
and MXFP8 arms share one FLOP count -- beside a FLOP-scaled estimate of the dense block's time (`dense ms at B x S_max x
FLOPs_varlen / FLOPs_dense`: the dense twin timed in the same process as the packed arm, not the uniform tables' 8K row, and the
same FLOP model at `B x S_max`; it assumes time scales linearly with work -- an estimate, not a measured equal-work dense run), a
dense torch run at `B x S_max` (which attends over the padding) and a
per-sequence torch loop (exact FLOPs, one dense call per sequence); speed-ups are positive numbers, `base ms / new ms - 1`. The
percentage of peak is against 8192 FLOP/clk/SM (bf16) x 212 SMs x the SM clock sampled during the cell; FP8 and MXFP8 against the
K32 cap of 16384 FLOP/clk/SM (the part's K64 peak is 32768, twice that, so halve those percentages for it). Every packed arm is
gated per sequence (the fp32 oracle for bf16 and the fake-quant oracles of the two quantized pipelines at `S <= 4096`, else the
dense block's own rows or the per-sequence torch chain); the FP8 and MXFP8 varlen rows have no per-sequence quantized reference
above `S_max = 4096` and are reported ungated. The 32K cells pack B=2 (32/2) and B=1 (64/8) sequences.

Forward, uniform packings (geomean packed overhead bf16 +2.3 %, FP8 -2.1 %, MXFP8 +0.8 %):

| heads Q/KV | mask | S | B (tokens) | dense bf16 ms | packed bf16 ms | packed bf16 overhead | packed bf16 TFLOP/s (% of peak) | dense FP8 ms | packed FP8 ms | packed FP8 overhead | packed FP8 TFLOP/s (% of the K32 cap) | dense MXFP8 ms | packed MXFP8 ms | packed MXFP8 overhead | packed MXFP8 TFLOP/s (% of the K32 cap) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32/2 | causal | 2048 | 4 (8192) | 0.664 | 0.702 | +5.7 % | 2838 (71 %) | 0.453 | 0.491 | +8.3 % | 4062 (50 %) | 0.449 | 0.461 | +2.7 % | 4321 (54 %) |
| 32/2 | causal | 8192 | 4 (32768) | 3.626 | 3.704 | +2.1 % | 3043 (81 %) | 2.194 | 2.310 | +5.3 % | 4879 (65 %) | 2.182 | 2.170 | -0.6 % | 5194 (69 %) |
| 32/2 | causal | 32768 | 2 (65536) | 15.914 | 15.898 | -0.1 % | 3078 (90 %) | 9.463 | 8.944 | -5.5 % | 5470 (80 %) | 8.830 | 8.928 | +1.1 % | 5480 (80 %) |
| 32/2 | dense | 2048 | 4 (8192) | 0.733 | 0.776 | +6.0 % | 2921 (73 %) | 0.489 | 0.530 | +8.4 % | 4282 (54 %) | 0.490 | 0.500 | +2.1 % | 4536 (57 %) |
| 32/2 | dense | 8192 | 4 (32768) | 5.043 | 5.049 | +0.1 % | 3103 (85 %) | 2.898 | 3.054 | +5.4 % | 5130 (70 %) | 2.882 | 2.891 | +0.3 % | 5420 (74 %) |
| 32/2 | dense | 32768 | 2 (65536) | 27.198 | 27.171 | -0.1 % | 3096 (90 %) | 17.437 | 14.315 | -17.9 % | 5876 (85 %) | 14.307 | 14.330 | +0.2 % | 5870 (85 %) |
| 64/8 | causal | 2048 | 4 (8192) | 1.375 | 1.430 | +4.1 % | 2882 (74 %) | 0.913 | 0.971 | +6.4 % | 4246 (54 %) | 0.895 | 0.907 | +1.3 % | 4545 (58 %) |
| 64/8 | causal | 8192 | 4 (32768) | 7.909 | 7.914 | +0.1 % | 2918 (83 %) | 5.158 | 4.920 | -4.6 % | 4693 (67 %) | 4.614 | 4.609 | -0.1 % | 5010 (72 %) |
| 64/8 | causal | 32768 | 1 (32768) | 16.293 | 16.830 | +3.3 % | 2940 (89 %) | 10.485 | 9.417 | -10.2 % | 5254 (79 %) | 9.297 | 9.571 | +2.9 % | 5170 (78 %) |
| 64/8 | dense | 2048 | 4 (8192) | 1.518 | 1.593 | +4.9 % | 2934 (77 %) | 0.991 | 1.061 | +7.1 % | 4405 (57 %) | 0.989 | 0.993 | +0.3 % | 4707 (61 %) |
| 64/8 | dense | 8192 | 4 (32768) | 10.871 | 10.882 | +0.1 % | 2930 (86 %) | 6.988 | 6.475 | -7.3 % | 4924 (72 %) | 6.228 | 6.236 | +0.1 % | 5114 (75 %) |
| 64/8 | dense | 32768 | 1 (32768) | 27.940 | 28.374 | +1.6 % | 2984 (92 %) | 17.941 | 15.143 | -15.6 % | 5591 (86 %) | 15.419 | 15.267 | -1.0 % | 5545 (85 %) |

Forward, varlen packing `[2048, 4096, 6144, 8192]` (the FLOP-scaled column is `packed ms / (dense ms at B x S_max x FLOPs_varlen /
FLOPs_dense) - 1`, an estimate that assumes time scales linearly with work; positive = the packed block is slower than that estimate):

| heads Q/KV | mask | dtype | packed ms | TFLOP/s (% of peak; FP8 / MXFP8: of the K32 cap) | vs the FLOP-scaled dense-time estimate | speed-up vs dense torch at B x S_max | speed-up vs the per-sequence torch loop |
|---|---|---|---|---|---|---|---|
| 32/2 | causal | bf16 | 2.018 | 3151 (78 %) | +5.9 % | +378.0 % | +204.4 % |
| 32/2 | causal | FP8 | 1.288 | 4934 (61 %) | +11.1 % | +648.7 % | +376.7 % |
| 32/2 | causal | MXFP8 | 1.210 | 5252 (65 %) | +6.6 % | +696.8 % | +407.4 % |
| 32/2 | dense | bf16 | 2.554 | 3297 (83 %) | +3.3 % | +324.3 % | +164.3 % |
| 32/2 | dense | FP8 | 1.580 | 5329 (67 %) | +10.8 % | +585.9 % | +327.3 % |
| 32/2 | dense | MXFP8 | 1.505 | 5594 (70 %) | +7.2 % | +620.0 % | +348.5 % |
| 64/8 | causal | bf16 | 4.173 | 3129 (81 %) | +3.6 % | +392.0 % | +207.3 % |
| 64/8 | causal | FP8 | 2.654 | 4920 (63 %) | +1.7 % | +673.6 % | +383.2 % |
| 64/8 | causal | MXFP8 | 2.508 | 5206 (67 %) | +6.6 % | +718.5 % | +411.2 % |
| 64/8 | dense | bf16 | 5.285 | 3251 (83 %) | +2.0 % | +335.3 % | +167.3 % |
| 64/8 | dense | FP8 | 3.287 | 5227 (67 %) | -3.5 % | +599.9 % | +329.8 % |
| 64/8 | dense | MXFP8 | 3.123 | 5501 (70 %) | +5.9 % | +636.7 % | +352.4 % |

Packed sequences (THD), backward -- the packed block backward against the dense one, Rubin (cc 10.7, 204 SMs, SM clock 2376 MHz,
sampled 2052-2352 MHz: the lock holds on the short cells, the long ones power-cap below it), the same two geometries at
`d_model = 4096`, bf16. Every cell is one process holding the packed block and its dense twin, slots round-robin by launch with the
slot order shuffled per iteration, 3 rounds of at least 60 ms per slot, median per slot per round then the median of rounds (CUDA
events around the whole block, plus the CUPTI device time of every launch over 30 iterations), the packed arm timed twice as the
control pair (within 0.9 % in every cell); the same overhead / varlen conventions as the forward tables, the percentage of peak against
8192 FLOP/clk/SM x 204 SMs x the sampled clock. Every packed arm is gated per sequence (fp32 / fp64 oracles at `S <= 4096`, else the
dense block's own rows or the per-sequence torch chain). The 32K backward cells pack B=1: the backward protocol keeps four blocks
resident (dense and packed, with and without `fuse_wgrad_overlap`), each with its dS head chunk (about 36 GiB at 32 heads x 32K),
which does not fit the device at B=2.

Backward (the block backward alone; the training-step column is the packed overhead of one training forward + backward; geomean packed overhead +1.0 %):

| heads Q/KV | mask | S | B (tokens) | dense ms | packed ms | packed overhead | with `fuse_wgrad_overlap` | training step (forward + backward) | packed TFLOP/s (% of peak) | packed speed-up vs torch |
|---|---|---|---|---|---|---|---|---|---|---|
| 32/2 | causal | 2048 | 4 (8192) | 1.716 | 1.739 | +1.3 % | +1.2 % | +2.5 % | 2371 (62 %) | +194.7 % |
| 32/2 | causal | 8192 | 4 (32768) | 10.103 | 10.202 | +1.0 % | +0.9 % | +1.4 % | 2425 (68 %) | +128.1 % |
| 32/2 | causal | 32768 | 1 (32768) | 23.332 | 23.558 | +1.0 % | +1.2 % | +1.1 % | 2450 (71 %) | +69.8 % |
| 32/2 | dense | 2048 | 4 (8192) | 1.955 | 1.972 | +0.9 % | +0.6 % | +2.1 % | 2439 (64 %) | +181.7 % |
| 32/2 | dense | 8192 | 4 (32768) | 14.019 | 14.029 | +0.1 % | -0.3 % | +0.1 % | 2547 (71 %) | +113.1 % |
| 32/2 | dense | 32768 | 1 (32768) | 39.638 | 39.669 | +0.1 % | -0.2 % | +0.1 % | 2564 (75 %) | +76.0 % |
| 64/8 | causal | 2048 | 4 (8192) | 3.454 | 3.471 | +0.5 % | +0.7 % | +1.8 % | 2455 (65 %) | +211.7 % |
| 64/8 | causal | 8192 | 4 (32768) | 20.562 | 21.243 | +3.3 % | +3.6 % | +2.6 % | 2381 (67 %) | +121.8 % |
| 64/8 | causal | 32768 | 1 (32768) | 47.256 | 47.738 | +1.0 % | +0.7 % | +0.6 % | 2441 (70 %) | +66.3 % |
| 64/8 | dense | 2048 | 4 (8192) | 3.932 | 3.951 | +0.5 % | +0.2 % | +2.1 % | 2504 (68 %) | +193.1 % |
| 64/8 | dense | 8192 | 4 (32768) | 28.491 | 29.182 | +2.4 % | +2.7 % | +1.8 % | 2487 (70 %) | +107.7 % |
| 64/8 | dense | 32768 | 1 (32768) | 80.663 | 80.618 | -0.1 % | +0.3 % | +0.2 % | 2537 (73 %) | +69.4 % |

Per stage, backward, causal, S=8192, B=4 (32768 tokens), dense vs packed (unfused):

| backward stage (CUPTI device ms per iteration) | 32/2 dense | 32/2 packed | delta | 64/8 dense | 64/8 packed | delta |
|---|---|---|---|---|---|---|
| B2 dO_gated dgrad GEMM | 0.604 | 0.603 | -0.1 % | 1.160 | 1.280 | +10.3 % |
| B3 sigmoid_gate_bwd | 0.349 | 0.351 | +0.4 % | 0.696 | 0.700 | +0.6 % |
| B1 dW_o wgrad GEMM | 0.700 | 0.702 | +0.2 % | 1.255 | 1.386 | +10.5 % |
| Q/K norm+RoPE recompute | 0.145 | 0.145 | -0.1 % | 0.315 | 0.321 | +1.8 % |
| compact V (elementwise) | 0.011 | 0.011 | +0.4 % | 0.034 | 0.035 | +3.5 % |
| fill / memset | 0.001 | -- | not launched (packed) | 0.001 | -- | not launched (packed) |
| SDPA dot_do_o | 0.169 | 0.169 | +0.3 % | 0.324 | 0.344 | +6.3 % |
| SDPA main kernel (dV + dS) | 2.208 | 2.283 | +3.4 % | 4.358 | 5.172 | +18.7 % |
| SDPA stage-3 dK GEMM | 1.339 | 1.342 | +0.2 % | 2.681 | 2.821 | +5.2 % |
| SDPA stage-3 dQ GEMM | 1.122 | 1.121 | -0.1 % | 2.254 | 2.396 | +6.3 % |
| SDPA dkv_reduce (GQA fold) | 0.170 | 0.174 | +2.3 % | 0.339 | 0.363 | +7.3 % |
| B5+B6 qk_norm_rope_bwd | 0.221 | 0.223 | +0.9 % | 0.486 | 0.502 | +3.1 % |
| dW_norm reduce | 0.005 | 0.005 | +1.8 % | 0.005 | 0.005 | +4.8 % |
| B7 dW_qkvg wgrad GEMM | 1.328 | 1.346 | +1.3 % | 2.772 | 2.916 | +5.2 % |
| B8 dh dgrad GEMM | 1.290 | 1.333 | +3.3 % | 2.894 | 3.023 | +4.4 % |
| B4 packed SDPA setup (metadata, per-sequence descriptors) | -- | 0.018 | packed only | -- | 0.060 | packed only |
| **all launches (median of the per-iteration sums)** | **9.588** | **9.811** | **+2.3 %** | **19.546** | **21.294** | **+8.9 %** |

Both come from the same CUPTI launch records (30 profiled iterations, the first 3 dropped): a stage row is the mean over the 27 kept
iterations of that stage's device time, the all-launches row the median of the same iterations' per-iteration sums, so the stage rows
add up to the mean total (9.662 / 9.826 ms at 32/2, 19.574 / 21.324 ms at 64/8), 0.1-0.8 % above the median.

Backward, varlen packing `[2048, 4096, 6144, 8192]` (unfused; `fuse_wgrad_overlap` moves the packed block by -0.8..+1.3 %, positive = faster;
the FLOP-scaled column is the same estimate as in the forward table):

| heads Q/KV | mask | view | packed ms | TFLOP/s (% of peak) | vs the FLOP-scaled dense-time estimate | speed-up vs dense torch at B x S_max | speed-up vs the per-sequence torch loop |
|---|---|---|---|---|---|---|---|
| 32/2 | causal | backward | 5.757 | 2387 (63 %) | +8.2 % | +287.4 % | +153.9 % |
| 32/2 | causal | training step | 7.880 | 2551 (67 %) | +7.4 % | +312.8 % | +166.6 % |
| 32/2 | dense | backward | 7.735 | 2443 (64 %) | +10.0 % | +267.3 % | +130.5 % |
| 32/2 | dense | training step | 10.465 | 2610 (69 %) | +8.3 % | +281.9 % | +137.6 % |
| 64/8 | causal | backward | 11.842 | 2379 (63 %) | +8.0 % | +286.6 % | +148.4 % |
| 64/8 | causal | training step | 16.241 | 2539 (67 %) | +5.9 % | +313.7 % | +162.7 % |
| 64/8 | dense | backward | 15.992 | 2406 (65 %) | +8.8 % | +265.7 % | +124.0 % |
| 64/8 | dense | training step | 21.757 | 2558 (69 %) | +7.0 % | +279.0 % | +132.8 % |

Scheduler policy: the packed SDPA pins `SCHED_NATURAL`. The same block built with `SCHED_LPT` on the packed SDPA
(bitwise-identical `O`, `LSE` and block output) changed the whole-block forward time by 32/2: 2K -0.2 %, 8K -2.4 %, 32K
-1.0 %; 64/8: 2K -1.7 %, 8K -2.8 %, 32K -1.4 % (2K / 8K / 32K causal, bf16; positive would be a gain; the FP8 packed
decode ignores the policy: 32/2 -0.0 %, 64/8 -0.3 %), so NATURAL stays.

## Related

- The SDPA epilogue gate is also reachable through the **graph API**: an `sdpa` node followed by `sigmoid` and
  `mul` pointwise nodes on `O` is served fused by the Rubin d256 FROST SDPA engines — see
  [Attention](/operations/Attention), "Fused epilogue gate".
- How the block is composed (workspace, streams, fusion knobs, typed declines) and how to build the next one:
  [Composing multi-kernel blocks in Python](/utilities/composing_kernel_blocks).
- The MLA sibling of the fused projection epilogue: [GEMM + RoPE + MXFP8 Projection](/operations/Matmul/gemm_fusions/gemm_proj_rope_mxfp8).
- Tests: `test/python/gated_attention_block/cutedsl/` (layout contract, reference oracle, end to end, FP8, MXFP8,
  fp4 weights / fp4 O (`test_block_fp4.py`, `test_proj_gemm_fp4.py`, `test_quantize_fp4.py`) and their backward
  (`test_block_backward_fp4.py`), packed sequences
  (`test_block_thd.py`, `test_block_thd_backward.py`, `test_block_thd_mxfp8.py`: the per-sequence oracles, the typed declines,
  the dense-vs-packed pins), per-stage kernels, stream ordering).
