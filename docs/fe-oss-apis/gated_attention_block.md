# Gated Attention Block (SM107)

**This is an experimental API and subject to change.**

## Overview

The gated attention block is the first **model-level** FE-OSS API: a set of FROST CuTe-DSL kernels behind one
Python class, one workspace and one `execute()` call. It implements the gated attention sub-layer used by
Qwen3.5-style models (the API is named by op geometry, the model is provenance only):

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
| MXFP8 with **MXFP4 weights** (unfused only) | `MxQuantSpec(w_qkvg_dtype=torch.float4_e2m1fn_x2)` and an e2m1 `W_qkvg [N, d_model // 2]`; `h`, `h_sf` and `w_qkvg_sf` unchanged | the MXFP8 unfused pipeline with stage (1) on the catalog's mixed MXFP8 x MXFP4 block-scale row (E8M0 scales per 32 on both sides) -- the same 9 launches; `fuse_norm_rope` is a typed decline (the fused projection fork is rendered for an e4m3 B) |
| MXFP8 with **fp4 O** (`NVFP4` or `MXFP4`) | `MxQuantSpec(o_fp4=Fp4Format.NVFP4 \| Fp4Format.MXFP4)`, an e2m1 `W_o [d_model, H_q * D // 2]` of the SAME format and its scale blob (`sample_w_o_sf` / `w_o_sf`) | the per-tensor tail is replaced: one `quantize_fp4` launch writes the gated `O` as e2m1 codes plus the out projection's scale blob, and stage (6) becomes the fp4 x fp4 block-scale GEMM (no per-tensor scale on either side). Unfused: 9 launches (`quantize_fp4` takes the per-tensor quantize's place); fully fused: **4 launches** (the gated MXFP8 SDPA writes bf16 `O`, then `quantize_fp4`, then the fp4 out projection). Composes with the MXFP4 weights on the unfused pipeline |

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
    build_fused_qkvg_weight, GatedAttentionBlockBwd, RecomputePolicy, SavedForBackward,
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

### Weights and tables

- `W_qkvg [N, d_model]` with `N = (2*H_q + 2*H_kv) * D`, column blocks `Q | GATE | K | V`. Build it from separate
  projection weights with `build_fused_qkvg_weight(w_q_gate, w_k, w_v, geometry, q_gate_layout="flat")`; the
  block's tile alignment is `QKVG_TILE_ALIGN = 64` columns.
- `W_o [d_model, H_q * D]`.
- `cos`, `sin` `[B, S, rope_dim]` rotary tables in the activation dtype (rotate-half convention on the first
  `rope_dim` dims of every head).
- `w_q_norm`, `w_k_norm` `[D]` (both `None` iff `geometry.qk_norm` is `False`).
- MXFP8 only: `h_sf` and `w_qkvg_sf`, the E8M0 scale factors of `h` and `W_qkvg` in cuDNN's F8_128x4 order
  (`uint8` or `float8_e8m0fnu`; byte counts from `cudnn.gated_attention_block.kernels.proj_gemm.sf_blob_bytes`).
- fp4 weights are **packed e2m1**, dtype `torch.float4_e2m1fn_x2`, stored `[N, K // 2]` -- two codes per byte along
  the contraction axis, LOW nibble = even `k`. The block checks the STORAGE shape (`[N, d_model // 2]` for `W_qkvg`,
  `[d_model, H_q * D // 2]` for `W_o`); a logical `[N, K]` fp4 tensor, or `uint8` storage, is a typed `ValueError`
  (torch can `.view(torch.float4_e2m1fn_x2)` packed bytes but cannot cast to fp4).
  - MXFP4 `W_qkvg` (`MxQuantSpec.w_qkvg_dtype=torch.float4_e2m1fn_x2`): `w_qkvg_sf` is UNCHANGED -- the same E8M0 /
    32 F8_128x4 blob over `n_qkvg x d_model` as for e4m3 codes.
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

bf16 / fp16 only, out of place (`inplace_qkv` defaults to `False` there; `fuse_norm_rope` / `fuse_gate` / `quant` are typed
declines). The block **writes through** the caller-owned `SavedForBackward` record wherever the backward needs a tensor,
with the same kernels as inference (`out` is bitwise the inference block's): the projection GEMM writes `saved.proj_slab`,
the SDPA writes the **pre-gate** `saved.o` and `saved.lse`, norm+RoPE writes `saved.rstd_q` / `rstd_k`, and the sigmoid gate
lands out of place in the workspace. `execute()` still allocates nothing.

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
  unfused pipeline (`QuantSpec`), `fuse_norm_rope` (bf16 / fp16 inference in place: the projection fork norms and rotates
  per token with the per-token tables). Declined, typed: `fuse_gate` (the SDPA's epilogue gate has no THD gate descriptor;
  stage (5) runs as its own launch), MXFP8 and the fp4 modes (the MXFP8 SDPA row serves no THD, and the block-scale
  quantize writes one scale-factor atom per (sequence, head, 128-row tile) of a padded grid), the fully fused quantized
  pipelines.
- **Declare `max_seq_len` tight.** The SDPA's unit grid is the plan-time envelope `B * ceil(max_seq_len / tile) * H_q`
  with dead units past the live total, and the backward's dS workspace scales with `ceil128(max_seq_len)`.

### Backward

`GatedAttentionBlockBwd(sample_dy, sample_saved, sample_w_qkvg, sample_w_q_norm, sample_w_k_norm, sample_cos,
sample_sin, sample_w_o, geometry, *, recompute=RecomputePolicy.RECOMPUTE_QK_PRE, need_dh=True,
need_dw_qkvg=True, need_dw_o=True, need_dw_norms=None, seq_lens_present=False, dw_norm_dtype=torch.float32,
fuse_gate_bwd=False, fuse_wgrad_overlap=False, thd=False, num_sequences=None, max_seq_len=None, cu_seqlens=False)` is the
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
recompute=..., current_stream=None, fuse_gate_bwd=False, fuse_wgrad_overlap=False, thd=False, max_seq_len=None)` allocates the gradients and the workspace on the launch stream (`current_stream`,
else torch's current stream -- the caching allocator orders a buffer's reuse only against the stream it was allocated on),
caches the compiled block per declaration and returns `{"dh", "dw_qkvg", "dw_o", "dw_q_norm", "dw_k_norm"}`; which entries exist follows `requires_grad` on
`saved.h` / `w_qkvg` / `w_o` / `w_q_norm` / `w_k_norm` (the tensors are handed to the block detached). Under `thd=True` it
derives `num_sequences` and `cu_seqlens` from the record (`saved.seq_lens.numel()`, `saved.seq_lens_form`) and requires
`max_seq_len` (a `ValueError` naming it alone otherwise); both, with `max_seq_len`, are part of its cache key.

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
`fuse_wgrad_overlap` is served (bitwise the in-order block); `fuse_gate_bwd` is a typed `NotImplementedError` under THD for
now (the packed chain computes its `delta` in the head-major packed layout and has no external producer yet). The
declaration bounds and the lengths contract are the forward's. Workspace: the dense `B=1, S=T` carve plus the packed
chain's scratch -- `delta [1, H_q, ceil128(T)]`, ONE head chunk of the kv-blocked dS, `[qh_chunk, ceil256(T + 256 B),
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
zeroed) into a block-owned fp32 `[B, H_q, S_pad]` region, and the SDPA backward adapter is built with
`external_delta=True` and reads that tensor: one launch and one read each of `O` and `dO` fewer (`11 + c*(2+q)` launches,
14 / 21 at the test geometry), the adapter's own `delta` region gone from its scratch (the block's region takes its
place, same bytes). Performance-only in the strict sense: the gradients are **bitwise** the unfused block's
(`test_fused_gate_bwd_is_bitwise_the_unfused_block`, bf16 and fp16, dense and causal, B=1 and B=3 under GQA), and the
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

- Rubin (SM107) only; cuDNN 9.x, `nvidia-cutlass-dsl >= 4.8.0.dev0` (the Rubin arch names), torch.
- Backward: bf16 / fp16 (both against fp64 autograd on Rubin: `test_block_backward.py`); **Rubin only -- the block
  binds ONE FROST engine class (`SdpaBwdDslSm107`, the Rubin d=256 SDPA backward) and never falls back to the cuDNN
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
- Packed sequences (`thd=True`), forward and backward: bf16 / fp16; the per-tensor FP8 unfused forward; `fuse_norm_rope`
  (bf16 / fp16 inference). `num_sequences >= 1`, `2 <= max_seq_len <= T`, `num_sequences * max_seq_len >= T`; the lengths
  tensor contiguous 1-D int32 on `h`'s device with `B` (`cu_seqlens=False`) or `B+1` (`cu_seqlens=True`) entries; every
  length `<= max_seq_len`, the lengths summing to `T` (the caller contract, not host-validated); the training record
  carries `saved.seq_lens` and `saved.seq_lens_form`. Declined (typed): `seq_lens_present` together with `thd`,
  `fuse_gate`, MXFP8 / fp4, the fully fused quantized pipelines, `fuse_gate_bwd`, the packing knobs on a dense block.
- `d_head = 256` (the Rubin d256 SDPA flavor with the fused gate); `d_model % 128 == 0` under MXFP8.
- FP8 / MXFP8 are inference only; the backward is bf16 / fp16.
- FP8: a dense (no-mask) sequence length must be a multiple of 128 unless the causal mask or a padding mask
  covers the KV tail (the Rubin per-tensor FP8 SDPA contract); MXFP8: e4m3 codes only (e5m2 is a typed decline);
  the fully fused MXFP8 path needs `scale_o == 1.0` and, at `B > 1`, `S % 128 == 0` (a scale-factor atom is per
  sequence).
- `fuse_gate` and `fuse_norm_rope` are inference-only specializations (no pre-gate `O`, no pre-norm Q/K).
- fp4 (`MxQuantSpec.w_qkvg_dtype` / `o_fp4`): MXFP8 pipeline only (unrepresentable on `QuantSpec` / bf16); inference
  only (`save_for_backward` is a typed decline, as for every quantized pipeline); no global per-tensor scale in
  either fp4 format (`scale_o == descale_w_o == 1.0` under `o_fp4`); `d_head % (4 * block) == 0` under `o_fp4`
  (whole 4-block scale words per head: 64 for NVFP4, 128 for MXFP4; `d_head = 256` passes both); the MXFP4
  `W_qkvg` runs on the unfused pipeline only -- `fuse_norm_rope` with an e2m1 `W_qkvg` is a typed
  `NotImplementedError` (the fused MXFP8 projection fork is rendered for an e4m3 B). `h` stays e4m3 (an fp4 `h` is
  not served), and an fp4 `W_o` with an e4m3 `O` is not a served pairing.

## Performance

Whole block, B=1, `h_q=32 h_kv=2 d=256 d_model=5120` (the 397B geometry), Rubin perf node (212 SMs, SM clock
locked at 2376 MHz), speedup over the same bf16 torch chain (median of 5 launch-interleaved rounds x 30 launches;
the bf16 FROST control pair stayed within 0.6 %). The fp4 modes (MXFP4 weights, NVFP4 / MXFP4 `O`) are not in these
tables: their perf-node measurement is pending, and no number is quoted until it exists.

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
  fp4 weights / fp4 O (`test_block_fp4.py`, `test_proj_gemm_fp4.py`, `test_quantize_fp4.py`), packed sequences
  (`test_block_thd.py`, `test_block_thd_backward.py`: the per-sequence oracle, the typed declines, the dense-vs-packed
  pins), per-stage kernels, stream ordering).
