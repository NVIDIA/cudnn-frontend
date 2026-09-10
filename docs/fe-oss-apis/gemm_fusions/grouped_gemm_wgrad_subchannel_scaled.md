# Grouped GEMM + Wgrad (Subchannel-Scaled NVFP4)

**This API is experimental and subject to change.**

Second-level-scaled ("subchannel-scaled") **weight-gradient** (2Dx2D) block-scaled
grouped GEMM for MoE workloads. The A operand (the transposed activation gradient)
is NVFP4 with **two** scale levels — per-(1, 16) FP8 (e4m3) first-level block scale
factors plus one FP32 second-level scale per `(hidden row x sgk tokens)` block
(`sgm = 1`, the rowwise-quant producer's `sf2` grid) — and B (the activation) is
single-level NVFP4. There is no second-level B scale. The reduction axis K is the
**ragged per-expert token axis**; M (`hidden`) and N (`intermediate`) are fixed
across experts.

## Operation

For each expert `e` with token range `[k0, k1)` split into `sgk`-token scale blocks
`b` (ascending order):

```
dW[e] = sum_b (A[:, b] @ B[b, :]) * SFA2[:, b_idx]        # f32: dW = partial * sfa2 + dW
dW[e] = (dW[e] * global_scale_a[e] * global_scale_b[e]).to(bf16)
```

`A = a_fp4 * sfa` and `B = b_fp4 * sfb` are the first-level dequantized operands.
The kernel forms one clean f32 partial accumulator per `sgk` block (tensor memory),
rescales it by the per-row SFA2 with a non-contractible `mul.rn` followed by a
separate add, and the output is verified **byte-exact** against a pure-torch
reference that replays those exact f32 ops in the same block order (SFA2 values are
arbitrary — they need not be powers of two). With `accumulate_on_output=True` the
bf16 tile is TMA-reduce-added into the caller's output instead of stored. Experts
with zero tokens produce zeros (or leave the accumulate target unchanged).

Source provenance: ported from the `bs_ggemm_harness` `wgrad` kernel
(`kernels/wgrad/kernel.py` / `kernel_sm107.py`). Reference:
`references/wgrad.py::wgrad_gemm_2nd_level`.

## Supported configurations

| Aspect | Supported |
|---|---|
| Architecture | SM100+ (Blackwell); transparent Rubin (SM107) kernel dispatch |
| A/B dtype | FP4 (`torch.float4_e2m1fn_x2`, or `torch.uint8` interpreted as packed fp4x2), K(token)-major |
| First-level scales | `torch.float8_e4m3fn`, `sf_vec_size = 16` (e5m3 reinterpretation on Rubin via `sf_fp8_dtype_override="e5m3"`) |
| Second-level scale | `torch.float32` SFA2 `(hidden, tokens_sum / sgk)`, **hidden-contiguous** (strides `(1, hidden)`); `sgk % 256 == 0` |
| Token counts | `tokens_sum % sgk == 0` and every expert's token count `% sgk == 0` (second-level blocks never straddle experts) |
| dW dtype | `torch.bfloat16` |
| Accumulator | `torch.float32` |
| MMA tiler | `(256, 128)` (default, 2-CTA) or `(128, 128)` |
| Cluster shape | `(1,1) (1,2) (2,1) (2,2) (1,4) (4,1) (2,4) (4,2) (4,4)`; default `(2, 1)` for the 2-CTA tiler, `(1, 1)` otherwise |
| Output modes | Dense `(expert_cnt, hidden, intermediate)` tensor, or discrete per-expert pointer arrays |
| Global scales | Optional `(expert_cnt,)` f32 pair, applied as `alpha = gsa[e] * gsb[e]` before the bf16 cast |
| Alignment | `hidden % 128 == 0`, `intermediate % 128 == 0` |
| Frameworks | torch only |

## Tensor layouts

| Tensor | Shape | Stride | Dtype |
|---|---|---|---|
| `a_tensor` | `(hidden, tokens_sum)` (logical) | `(tokens_sum, 1)` | fp4 |
| `b_tensor` | `(tokens_sum, intermediate)` (logical) | `(1, tokens_sum)` | fp4 |
| `sfa_tensor` | `(round_up(hidden, 128), round_up(tokens_sum/16, 4))` | assembled per-expert 128x4-atom layout (as `grouped_gemm_wgrad`) | `float8_e4m3fn` |
| `sfb_tensor` | `(round_up(intermediate, 128), round_up(tokens_sum/16, 4))` | assembled per-expert 128x4-atom layout | `float8_e4m3fn` |
| `sfa2_tensor` | `(hidden, tokens_sum / sgk)` | `(1, hidden)` | `float32` |
| `offsets_tensor` | `(expert_cnt,)` cumulative END token offsets | contiguous | `int32` |
| `wgrad_tensor` | `(expert_cnt, hidden, intermediate)` | contiguous | `bfloat16` |

The first-level scale tensors use the same per-expert assembled layout as the
existing `grouped_gemm_wgrad` API (each expert's `(mn, tokens_e / 16)` block scattered
into 128x4 atoms, experts concatenated). The compiled kernel treats the token axis as
dynamic: one compile serves every token distribution with the same `expert_cnt`,
`hidden`, `intermediate` and `sgk`.

## Wrapper API

```python
import cudnn, torch

result = cudnn.grouped_gemm_wgrad_subchannel_scaled_wrapper_sm100(
    a_tensor=a,            # (hidden, tokens_sum) fp4 dY^T, token-innermost
    b_tensor=b,            # (tokens_sum, intermediate) fp4 activations, token-innermost
    sfa_tensor=sfa,        # assembled e4m3 first-level A scales
    sfb_tensor=sfb,        # assembled e4m3 first-level B scales
    sfa2_tensor=sfa2,      # (hidden, tokens_sum/sgk) f32, strides (1, hidden)
    offsets_tensor=offs,   # (expert_cnt,) int32 cumulative end offsets
    sgk=512,
    output_mode="dense",   # or "discrete" (+ wgrad_ptrs / wgrad_tensor)
    global_scale_a=gsa,    # optional (expert_cnt,) f32
    global_scale_b=gsb,
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
)
dw = result["wgrad_tensor"]   # (expert_cnt, hidden, intermediate) bf16
```

When `wgrad_tensor`/`wgrad_ptrs` are omitted the wrapper allocates the output
(zero-initialized when `accumulate_on_output=True`). Class-API users
(`GroupedGemmWgradSubchannelScaledSm100` + `check_support()` / `compile()` /
`execute()`) pass their own output; `check_support()` validates the per-expert
`sgk` alignment on the sample offsets, and the same contract must hold for every
subsequent `execute()`.

## Notes

- `sf_fp8_dtype_override="e5m3"` reinterprets the e4m3-typed first-level scale bytes
  as UE5M3; it is accepted only on Rubin (SM107) and takes part in the compile cache
  key. The e5m3 path is validated byte-exact on Rubin (the tests re-encode the e4m3
  scales as UE5M3 and compare against the same reference) for dense/discrete output
  and `accumulate_on_output`.
- No JAX support (fp4 K-major operands and the hidden-contiguous SFA2 view are not
  expressible as row-major JAX arrays).
