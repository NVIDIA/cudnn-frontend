# Gated Attention Block (SM107)

**This is an experimental API and subject to change.**

## Overview

The gated attention block is the first **model-level** FE-OSS API: a set of FROST CuTe-DSL kernels behind one
Python class, one workspace and one `execute()` call. It implements the gated attention sub-layer used by
Qwen3.5- and Qwen3.8-style models (the API is named by op geometry, the model is provenance only; the Qwen3.8
family's geometries and the bound up to which the block is Qwen3.8-Flash-Next's sparse layer are under
[Qwen3.8 family / Qwen3.8-Flash-Next](#qwen38-family--qwen38-flash-next)):

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
  the gate tile TMA-staged by the load warp (inference only: no pre-gate `O` for the backward).

With both on, the block is **three launches**: `proj(+norm+RoPE[+quant]) -> sdpa(+gate) -> out_proj`.

`geometry.qk_norm=False` runs RoPE-only Q/K on every path (norm weights are passed as `None`, no `rstd` is
produced, the backward drops the norm-weight gradients).

## API

```python
import cudnn
from cudnn.gated_attention_block import (
    GatedAttentionBlockFwd, GatedAttentionBlockGeometry, QuantSpec, MxQuantSpec, Fp4Format,
    build_fused_qkvg_weight, qkvg_from_hf, GatedAttentionBlockBwd, RecomputePolicy, SavedForBackward,
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

**Flash-Next: exact for `<= 2051` visible tokens.** Flash-Next's attention layer is Qwen Sparse Attention (QSA): the same
gated core restricted, per query, to indexer-selected 4-token blocks under a 2048-token budget -- the top `min(512, n_blocks)`
complete blocks plus the open tail block, at most `2048 + 4 - 1 = 2051` keys per query. A query that sees `n` tokens has
`floor(n / 4)` complete blocks, and every one of them is selected when `floor(n / 4) <= 512`, i.e. `n <= 2051`: the selection
is the identity and causal attention over the visible tokens IS the QSA layer. So for every query with at most 2051 visible
tokens -- every sequence of up to 2051 tokens -- the dense block computes Flash-Next's attention layer exactly: the same
function (a sparse evaluation sums the same terms in another order, so the identity holds within the suite's budget, not
bitwise). Beyond 2051 visible tokens QSA drops the blocks its indexer did not select while this block attends densely -- a
different function; the indexer and the sparse core are not part of the block today.

### Weights and tables

- `W_qkvg [N, d_model]` with `N = (2*H_q + 2*H_kv) * D`, column blocks `Q | GATE | K | V`; the block's tile alignment
  is `QKVG_TILE_ALIGN = 64` columns. Build it ONCE at load time, never per call:
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
    quant=None,                  # QuantSpec (FP8) | MxQuantSpec (MXFP8, + fp4 weights / fp4 O) | None (bf16 / fp16)
    sample_h_sf=None, sample_w_qkvg_sf=None,   # MXFP8 scale-factor blobs
    sample_w_o_sf=None,          # fp4 O only (MxQuantSpec.o_fp4): the e2m1 W_o's scale blob, in O's format
    saved_gate_copy=False,       # training save mode: True copies the GATE band into a compact saved.gate (see below)
    thd=False,                   # PACKED sequences: h / cos / sin / out are [T, .] (or [1, T, .]) token matrices -- see "Packed sequences (THD)"
    num_sequences=None,          # THD only, REQUIRED there: B, the number of sequences in the packing
    max_seq_len=None,            # THD only, REQUIRED there: S_max, the longest sequence the plan admits
    cu_seqlens=False,            # THD only: execute(seq_lens=) is [B+1] int32 prefix sums instead of [B] int32 lengths
)
workspace = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device=h.device)
blk.execute(h, w_qkvg, w_q_norm, w_k_norm, cos, sin, w_o, out, workspace,
            seq_lens=None,       # dense: the [B] int32 KV lengths iff seq_lens_present; THD: REQUIRED, the [B] lengths / [B+1] prefix sums
            lse=None, saved=None, current_stream=None, h_sf=None, w_qkvg_sf=None,
            w_o_sf=None)         # required iff MxQuantSpec.o_fp4, refused otherwise
```

Every appended argument (`sample_w_o_sf`, `w_o_sf`, `saved_gate_copy`, and the four packing knobs `thd` / `num_sequences` /
`max_seq_len` / `cu_seqlens`) sits at the end with a default, so positional callers of the bf16, FP8 and MXFP8 pipelines are
unchanged; `MxQuantSpec.o_fp4` and `sample_w_o_sf` must be given together (a typed `ValueError` names the missing half), and
`num_sequences` / `max_seq_len` / `cu_seqlens` are refused without `thd=True`.

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
[`samples/frost/gated_attention_block/00_training_forward.py`](../../samples/frost/gated_attention_block/00_training_forward.py).

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
  the prefill bodies' scope) and `num_sequences * max_seq_len >= T` -- a smaller product would silently cap the SDPA
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
  `(1, T)` as the dense quantized training forward, with `saved.h` the e4m3 `h`, and the packed bf16 backward
  differentiates it given the dequantized bf16 `h` and weights, exactly as on the dense side), `fuse_norm_rope` (bf16 /
  fp16 inference in place: the projection fork norms and rotates per token with the per-token tables). Declined, typed:
  `fuse_gate` (the SDPA's epilogue gate has no THD gate descriptor;
  stage (5) runs as its own launch), MXFP8 and the fp4 modes (the MXFP8 SDPA row serves no THD, and the block-scale
  quantize writes one scale-factor atom per (sequence, head, 128-row tile) of a padded grid), the fully fused quantized
  pipelines.
- **Declare `max_seq_len` tight.** The SDPA's unit grid is the plan-time envelope `B * ceil(max_seq_len / tile) * H_q`
  with dead units past the live total, and the backward's dS workspace scales with `ceil128(max_seq_len)`.

### Backward

`GatedAttentionBlockBwd(sample_dy, sample_saved, sample_w_qkvg, sample_w_q_norm, sample_w_k_norm, sample_cos,
sample_sin, sample_w_o, geometry, *, recompute=RecomputePolicy.RECOMPUTE_QK_PRE, need_dh=True,
need_dw_qkvg=True, need_dw_o=True, need_dw_norms=None, seq_lens_present=False, dw_norm_dtype=torch.float32,
fuse_gate_bwd=False, fuse_wgrad_overlap=False, thd=False, num_sequences=None, max_seq_len=None, cu_seqlens=False,
quant=None, grad_scaling="current")` is the
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
grad_scaling="current", scale_dp=None, scale_dy=None, scale_do=None, scale_dqkvg=None)` allocates the gradients and the workspace on the launch stream (`current_stream`,
else torch's current stream -- the caching allocator orders a buffer's reuse only against the stream it was allocated on),
caches the compiled block per declaration and returns `{"dh", "dw_qkvg", "dw_o", "dw_q_norm", "dw_k_norm"}`; which entries exist follows `requires_grad` on
`saved.h` / `w_qkvg` / `w_o` / `w_q_norm` / `w_k_norm` (the tensors are handed to the block detached). Under `thd=True` it
derives `num_sequences` and `cu_seqlens` from the record (`saved.seq_lens.numel()`, `saved.seq_lens_form`) and requires
`max_seq_len` (a `ValueError` naming it alone otherwise); both, with `max_seq_len`, are part of its cache key. `quant` /
`grad_scaling` (the quantized backward below) join the key too, and `scale_dp` / `scale_dy` / `scale_do` / `scale_dqkvg` pass
through to `execute`; the gradients are allocated in `dy`'s dtype (bf16 under `quant`, where `saved.h` and the weights are e4m3 codes).

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
amax pass in this step (`2**(floor(log2(448 / amax)) - FP8_GRAD_SCALE_MARGIN_LOG2)`, with `FP8_GRAD_SCALE_MARGIN_LOG2 = 0`);
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
inert under `quant`: the fused delta is mandatory there. `fuse_wgrad_overlap` is served (the side-stream GEMMs fork after the slots and operands they read are written).
Declined (typed, naming the attribute): e5m2 codes, an fp16 `dy` (the quantized
backward is bf16), a bf16 `saved.h` or bf16 weights with a `QuantSpec` and e4m3 codes without one (both ways), `thd=True` with
`quant` -- dense-only for now, its packed arm a follow-up --, and a geometry whose Q / K
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
the operands and blobs they read are written). Declined (typed, naming the attribute) on top of the per-tensor fp8 arm's: `thd=True` with an MxQuantSpec (dense-only; no packed
MXFP8 record exists), `B*S % 32 != 0` when a projection weight gradient is requested -- the weight-gradient GEMM contracts over the
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

- `CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1` in the environment BEFORE `import cudnn`: every FROST engine is opt-in, and the
  block drives them. Without the flag the bf16 / f16 / per-tensor-fp8 projection GEMMs refuse to run -- a `RuntimeError`
  at `compile()` whose message names the flag (the GEMM plan is built there, not at `check_support()`) -- and the
  block-scale (MXFP8 / fp4-weight) projections log a warning and take FROST's JIT-only route instead. Neither ever falls
  back to a cuDNN backend plan, so an unset flag is never a silently slower block. The SDPA stage binds its FROST class
  directly and does not consult the flag.
- Rubin (SM107) only; cuDNN 9.x, `nvidia-cutlass-dsl >= 4.8.0.dev0` (the Rubin arch names), torch.
- Backward: bf16 / fp16 (both against fp64 autograd on Rubin: `test_block_backward.py`) -- and per-tensor fp8 over the fp8
  training record (`quant=QuantSpec`: bf16 `dy` and gradients, dense only, any `B*S`, head counts the TMA Q / K rebuild tiles),
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
  `get_workspace_size()` after `compile()`. A dense `S % 128 != 0` has no training record to differentiate: the
  forward's SDPA row declines it (its KV tail would be unmasked); causal covers the tail.
- Packed sequences (`thd=True`), forward and backward: bf16 / fp16; the per-tensor FP8 unfused forward, inference and
  training (its packed record goes through the packed bf16 backward with the dequantized `h` and weights; a record handed
  through with its e4m3 `h` is the same typed decline as on the dense side, after the packed-length checks);
  `fuse_norm_rope` (bf16 / fp16 inference). `num_sequences >= 1`, `2 <= max_seq_len <= T`, `num_sequences * max_seq_len >= T`; the lengths
  tensor contiguous 1-D int32 on `h`'s device with `B` (`cu_seqlens=False`) or `B+1` (`cu_seqlens=True`) entries; every
  length `<= max_seq_len`, the lengths summing to `T` (the caller contract, not host-validated); the training record
  carries `saved.seq_lens` and `saved.seq_lens_form`. Declined (typed): `seq_lens_present` together with `thd`,
  `fuse_gate`, MXFP8 / fp4, the fully fused quantized pipelines, the packing knobs on a dense block.
- `d_head = 256` (the Rubin d256 SDPA flavor with the fused gate); `d_model % 128 == 0` under MXFP8.
- FP8 / MXFP8: the UNFUSED pipelines train (`save_for_backward=True` writes the bf16 record described above), the fp4
  modes of the MXFP8 pipeline included; the fully fused quantized pipelines are inference only. The backward is bf16 / fp16
  -- and per-tensor fp8 / MXFP8 over the quantized training records (`quant=QuantSpec` / `quant=MxQuantSpec`, the record as
  written, the fp4 weight modes through their transposed e2m1 artifacts); the bf16 backward takes either record with the
  dequantized bf16 `h` and weights.
- FP8: a dense (no-mask) sequence length must be a multiple of 128 unless the causal mask or a padding mask
  covers the KV tail (the Rubin per-tensor FP8 SDPA contract); MXFP8: e4m3 codes only (e5m2 is a typed decline);
  the fully fused MXFP8 path needs `scale_o == 1.0` and, at `B > 1`, `S % 128 == 0` (a scale-factor atom is per
  sequence).
- `fuse_gate` and `fuse_norm_rope` are inference-only specializations (no pre-gate `O`, no pre-norm Q/K).
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

Packed sequences (THD) -- the packed block against the dense block, Rubin (cc 10.7, 204 SMs, SM clock 2376 MHz), both
geometries 32/2 (`h_q=32 h_kv=2`) and 64/8 (`h_q=64 h_kv=8`) at `d_model = 4096` (not the 5120 of the forward tables
above), `d_head = 256`, RoPE 64, Q/K RMSNorm on, causal and dense (no mask), bf16 (the forward also the unfused FP8
block). Every cell is one process holding the packed block and its dense twin (identical FLOPs, identical kernels except
the SDPA's packed specialization), slots round-robin by launch with the slot order shuffled per iteration, 3 rounds of
at least 60 ms per slot, median per slot per round then the median of rounds (CUDA events around the whole block; the
backward rows add the CUPTI device time of every launch over 30 iterations), the packed arm timed twice as the control
pair (within 0.9 % in every cell). `packed overhead = packed ms / dense ms - 1` on uniform packings (`B` sequences of
`S` tokens each: the same FLOPs; positive = the packed block is slower); the varlen cell packs `[2048, 4096, 6144,
8192]` (`B=4`, `max_seq_len = 8192`, 20480 tokens) and reports TFLOP/s on its exact per-sequence FLOPs (causal: the
exact masked pair count) beside a FLOP-scaled estimate of the dense block's time (`dense ms at B x S_max x FLOPs_varlen / FLOPs_dense`, which assumes time scales
linearly with work -- an estimate, not a measured equal-work dense run), a dense torch run at `B x S_max` (which attends
over the padding) and a per-sequence torch loop (exact FLOPs, one dense call per sequence); speed-ups are positive
numbers, `base ms / new ms - 1`. The percentage of peak is against 8192 FLOP/clk/SM (bf16) x 204 SMs x the SM clock
sampled during the cell (2052-2352 MHz: the lock holds on the short cells, the long ones power-cap below it); FP8
against the K32 cap of 16384 FLOP/clk/SM (the part's K64 peak is 32768, twice that, so halve the FP8 percentage for it).
Every packed arm is gated per sequence (fp32 / fp64 oracles at `S <= 4096`, else the dense block's own rows or the
per-sequence torch chain); the FP8 varlen rows have no per-sequence FP8 reference above `S_max = 4096` and are reported
ungated. The 32K forward cells pack B=2 (32/2) and B=1 (64/8) sequences, the 32K backward cells B=1: the backward
protocol keeps four blocks resident (dense and packed, with and without `fuse_wgrad_overlap`), each with its dS head
chunk (about 36 GiB at 32 heads x 32K), which does not fit the device at B=2.

Forward, uniform packings (geomean packed overhead bf16 +1.6 %, FP8 -1.9 %):

| heads Q/KV | mask | S | B (tokens) | dense bf16 ms | packed bf16 ms | packed bf16 overhead | packed bf16 TFLOP/s (% of peak) | dense FP8 ms | packed FP8 ms | packed FP8 overhead | packed FP8 TFLOP/s (% of the K32 cap) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 32/2 | causal | 2048 | 4 (8192) | 0.718 | 0.756 | +5.2 % | 2638 (68 %) | 0.485 | 0.523 | +7.8 % | 3813 (49 %) |
| 32/2 | causal | 8192 | 4 (32768) | 3.699 | 3.781 | +2.2 % | 2981 (80 %) | 2.285 | 2.372 | +3.8 % | 4751 (64 %) |
| 32/2 | causal | 32768 | 2 (65536) | 15.709 | 15.640 | -0.4 % | 3128 (91 %) | 9.467 | 9.073 | -4.2 % | 5393 (78 %) |
| 32/2 | dense | 2048 | 4 (8192) | 0.786 | 0.825 | +4.9 % | 2750 (71 %) | 0.518 | 0.556 | +7.3 % | 4077 (52 %) |
| 32/2 | dense | 8192 | 4 (32768) | 5.086 | 5.067 | -0.4 % | 3092 (85 %) | 2.971 | 3.150 | +6.0 % | 4974 (68 %) |
| 32/2 | dense | 32768 | 2 (65536) | 26.450 | 26.874 | +1.6 % | 3130 (90 %) | 16.914 | 14.177 | -16.2 % | 5933 (86 %) |
| 64/8 | causal | 2048 | 4 (8192) | 1.455 | 1.502 | +3.2 % | 2746 (71 %) | 0.960 | 1.018 | +6.0 % | 4050 (52 %) |
| 64/8 | causal | 8192 | 4 (32768) | 7.810 | 7.850 | +0.5 % | 2942 (81 %) | 5.110 | 4.952 | -3.1 % | 4663 (65 %) |
| 64/8 | causal | 32768 | 1 (32768) | 15.956 | 15.751 | -1.3 % | 3141 (89 %) | 10.115 | 9.065 | -10.4 % | 5458 (77 %) |
| 64/8 | dense | 2048 | 4 (8192) | 1.600 | 1.641 | +2.6 % | 2847 (74 %) | 1.042 | 1.088 | +4.4 % | 4295 (56 %) |
| 64/8 | dense | 8192 | 4 (32768) | 10.476 | 10.473 | -0.0 % | 3045 (84 %) | 6.823 | 6.421 | -5.9 % | 4966 (69 %) |
| 64/8 | dense | 32768 | 1 (32768) | 26.723 | 27.100 | +1.4 % | 3124 (90 %) | 16.973 | 14.548 | -14.3 % | 5820 (83 %) |

Forward, varlen packing `[2048, 4096, 6144, 8192]` (the FLOP-scaled column is `packed ms / (dense ms at B x S_max x FLOPs_varlen /
FLOPs_dense) - 1`, an estimate that assumes time scales linearly with work; positive = the packed block is slower than that estimate):

| heads Q/KV | mask | dtype | packed ms | TFLOP/s (% of peak; FP8: of the K32 cap) | vs the FLOP-scaled dense-time estimate | speed-up vs dense torch at B x S_max | speed-up vs the per-sequence torch loop |
|---|---|---|---|---|---|---|---|
| 32/2 | causal | bf16 | 2.093 | 3037 (77 %) | +5.7 % | +371.3 % | +200.2 % |
| 32/2 | causal | FP8 | 1.357 | 4686 (60 %) | +11.0 % | +627.2 % | +363.1 % |
| 32/2 | dense | bf16 | 2.638 | 3191 (82 %) | +2.9 % | +322.1 % | +162.0 % |
| 32/2 | dense | FP8 | 1.669 | 5044 (65 %) | +12.9 % | +567.0 % | +314.1 % |
| 64/8 | causal | bf16 | 4.275 | 3054 (81 %) | +2.6 % | +391.5 % | +210.6 % |
| 64/8 | causal | FP8 | 2.772 | 4710 (62 %) | +1.5 % | +658.1 % | +379.0 % |
| 64/8 | dense | bf16 | 5.383 | 3192 (83 %) | +0.7 % | +336.8 % | +171.9 % |
| 64/8 | dense | FP8 | 3.392 | 5065 (66 %) | -3.5 % | +593.2 % | +331.5 % |

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
  [Attention](../operations/Attention.md), "Fused epilogue gate".
- How the block is composed (workspace, streams, fusion knobs, typed declines) and how to build the next one:
  [Composing multi-kernel blocks in Python](../utilities/composing_kernel_blocks.md).
- The MLA sibling of the fused projection epilogue: [GEMM + RoPE + MXFP8 Projection](gemm_fusions/gemm_proj_rope_mxfp8.md).
- Tests: `test/python/gated_attention_block/cutedsl/` (layout contract, reference oracle, end to end, FP8, MXFP8,
  fp4 weights / fp4 O (`test_block_fp4.py`, `test_proj_gemm_fp4.py`, `test_quantize_fp4.py`) and their backward
  (`test_block_backward_fp4.py`), packed sequences
  (`test_block_thd.py`, `test_block_thd_backward.py`: the per-sequence oracle, the typed declines, the dense-vs-packed
  pins), per-stage kernels, stream ordering).
