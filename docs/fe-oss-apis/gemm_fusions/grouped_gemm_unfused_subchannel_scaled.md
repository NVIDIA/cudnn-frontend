# Grouped GEMM (Unfused, Subchannel-Scaled NVFP4)

**This API is experimental and subject to change.**

Second-level-scaled ("subchannel-scaled") **unfused** block-scaled grouped GEMM for
MoE workloads. Inputs are NVFP4 with **two** scale levels — per-(1, 16) FP8 (e4m3)
first-level block scale factors (SFA/SFB) plus FP32 second-level scales
(SFA2/SFB2) at `block2_shape = (sgm, sgn, sgk)` granularity — and the output D is
plain BF16 (no output quantization; this is the unfused GEMM-only counterpart of
the fused dGLU kernels; it computes the FC2 data-gradient GEMM `dY @ W`).

## Operation

For each expert `e` with token rows `rows_e`:

```
G[rows_e] = sum over sgk-tiles t of:
    (A[rows_e, t] @ B[e, :, t]^T) * SFA2[rows_e/sgm, t] * SFB2[e, :, t]
D[rows_e]  = G[rows_e] * alpha[e] * prob[rows_e]                    (no bias)
D[rows_e]  = G[rows_e] * alpha[e] + prob[rows_e] * bias[:, e]       (with bias)
```

where `A = a_fp4 * sfa` and `B = b_fp4 * sfb` are the first-level dequantized
operands (`value = data * sf * sf2`). SFA2 blocks cover `(sgm rows x sgk k)`;
SFB2 blocks cover `(sgn cols x sgk k)`. The per-token `prob` multiply is always
fused (pass ones for a plain GEMM). The default `vector_f32=True` configuration
is verified **byte-exact** against a pure-torch reference.

Source provenance: ported from the `bs_ggemm_harness` `dgrad_quant` kernel
(`kernels/dgrad_quant/kernel.py` / `kernel_sm107.py`), with the upstream
quantization epilogue removed. Reference:
`references/fc2_dgrad.py::fc2_dgrad_gemm`.

## Supported configurations

| Aspect | Supported |
|---|---|
| Architecture | SM100+ (Blackwell); transparent Rubin (SM107) kernel dispatch |
| A/B dtype | FP4 (`torch.float4_e2m1fn_x2`, or `torch.uint8` interpreted as packed fp4x2), k-major |
| First-level scales | `torch.float8_e4m3fn`, `sf_vec_size = 16` (e5m3 reinterpretation on Rubin via `sf_fp8_dtype_override="e5m3"`) |
| Second-level scales | `torch.float32`, `block2_shape` e.g. `(1, 256, 256)` or `(1, 512, 512)`; `k % sgk == 0`, `valid_m % sgm == 0` |
| D dtype | `torch.bfloat16`, n-major |
| Accumulator | `torch.float32` |
| MMA tiler | `(256, 128)` (default) or `(256, 256)` |
| Cluster shape | `(2, 1)` (default) or `(2, 2)` |
| Weight modes | Dense `(n, k, l)` tensors, or discrete per-expert pointer arrays |
| Bias | Optional BF16 `(n, l)`, stride `(1, n)`; `d = gemm * alpha + prob * bias` |
| prob | **Required** FP32 `(valid_m, 1, 1)` per-token multiplier |
| Alignment | `valid_m % 256 == 0` (every expert's rows 256-padded), `n % 64 == 0` |
| Frameworks | torch only (the MMA-tiled scale-factor views are not expressible as JAX arrays) |

## Tensor layouts

| Tensor | Shape | Stride | Dtype |
|---|---|---|---|
| `a_tensor` | `(valid_m, k, 1)` (logical) | `(k, 1, valid_m * k)` | fp4 |
| `sfa_tensor` | `(32, 4, ceil(valid_m/128), 4, ceil(ceil(k/16)/4), 1)` | MMA-tiled | `float8_e4m3fn` |
| `sfa2_tensor` | `(ceil(valid_m/sgm), ceil(k/sgk), 1)` | `(1, rows, rows * cols)` | `float32` |
| `b_tensor` (dense) | `(n, k, l)` (logical) | `(k, 1, n * k)` | fp4 |
| `sfb_tensor` (dense) | `(32, 4, ceil(n/128), 4, ceil(ceil(k/16)/4), l)` | MMA-tiled | `float8_e4m3fn` |
| `sfb2_tensor` (dense) | `(ceil(n/sgn), ceil(k/sgk), l)` | `(1, rows, rows * cols)` | `float32` |
| `padded_offsets` | `(l,)` cumulative END offsets after 256-padding | contiguous | `int32` |
| `alpha_tensor` | `(l,)` | contiguous | `float32` |
| `prob_tensor` | `(valid_m, 1, 1)` | `(1, 1, valid_m)` | `float32` |
| `bias_tensor` | `(n, l)` | `(1, n)` | `bfloat16` |
| `d_tensor` (out) | `(valid_m, n, 1)` | `(n, 1, valid_m * n)` | `bfloat16` |

The SFA2/SFB2 tensors are **rows-contiguous** (mode-0 major). A convenient way to
allocate them in torch:

```python
rows, cols = (valid_m + sgm - 1) // sgm, (k + sgk - 1) // sgk
sfa2 = torch.empty((1, cols, rows), dtype=torch.float32, device="cuda").permute(2, 1, 0)
```

Discrete mode replaces `b_tensor/sfb_tensor/sfb2_tensor` with 1-D `int64` device
tensors of per-expert base pointers (`b_ptrs/sfb_ptrs/sfb2_ptrs`) plus explicit
`n` and `b_dtype`. Every SFB2 per-expert base must be 16-byte aligned; with
back-to-back per-expert blocks this requires
`ceil(n/sgn) * ceil(k/sgk) * 4 % 16 == 0` when `num_experts > 1` (enforced by
`check_support`).

## Wrapper API

```python
import cudnn

result = cudnn.grouped_gemm_unfused_subchannel_scaled_wrapper_sm100(
    a_tensor=a,            # (valid_m, k, 1) fp4, k-major
    sfa_tensor=sfa,        # MMA-tiled first-level scales
    sfa2_tensor=sfa2,      # (ceil(valid_m/sgm), ceil(k/sgk), 1) f32
    padded_offsets=offs,   # (l,) int32 cumulative end offsets
    alpha_tensor=alpha,    # (l,) f32
    prob_tensor=prob,      # (valid_m, 1, 1) f32 -- required
    b_tensor=b,            # (n, k, l) fp4 (dense mode)
    sfb_tensor=sfb,
    sfb2_tensor=sfb2,      # (ceil(n/sgn), ceil(k/sgk), l) f32
    bias_tensor=None,      # optional (n, l) bf16
    block2_shape=(1, 256, 256),
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
)
d = result["d_tensor"]     # (valid_m, n, 1) bf16
```

The wrapper allocates `d_tensor` (or writes into a preallocated one passed via
`d_tensor=`), caches compiled kernels per configuration (dynamic in `valid_m`),
and returns a `TupleDict` with the single output `d_tensor`.

## Class API

```python
from cudnn import GroupedGemmUnfusedSubchannelScaledSm100

api = GroupedGemmUnfusedSubchannelScaledSm100(
    sample_a=a, sample_sfa=sfa, sample_sfa2=sfa2,
    sample_padded_offsets=offs, sample_alpha=alpha, sample_prob=prob,
    sample_d=d,
    sample_b=b, sample_sfb=sfb, sample_sfb2=sfb2,   # dense mode
    block2_shape=(1, 256, 256),
)
api.check_support()
api.compile()
api.execute(a_tensor=a, sfa_tensor=sfa, sfa2_tensor=sfa2,
            padded_offsets=offs, alpha_tensor=alpha, prob_tensor=prob,
            d_tensor=d, b_tensor=b, sfb_tensor=sfb, sfb2_tensor=sfb2)
```

Discrete mode: construct with `num_experts=l, b_shape=(n, k), b_dtype=...`
instead of the `sample_b/sample_sfb/sample_sfb2` trio, and pass
`b_ptrs/sfb_ptrs/sfb2_ptrs` to `execute()`.

## Architecture dispatch

On Rubin (SM107) devices the API transparently selects the Rubin-native kernel
module (`moe_blockscaled_grouped_gemm_unfused_subchannel_scaled_rubin.py`); the
public class and wrapper are unchanged. `sf_fp8_dtype_override="e5m3"`
(Rubin-only) reinterprets the FP8 first-level scale factors as E5M3 — the scale
tensors are still supplied as `torch.float8_e4m3fn` since torch has no e5m3
dtype. The e5m3 path is validated byte-exact on Rubin for both dense and discrete
weight modes (the tests re-encode the e4m3 scales as UE5M3 and compare against the
same reference).

## Limitations

- `prob_tensor` is mandatory (the kernel always fuses the per-token multiply).
- BF16 output only; no output quantization (use the fused dGLU/quant kernels for that).
- torch tensors only (no JAX).
- `m_aligned` must be 256 (`FIX_PAD_SIZE`); every expert's row group is 256-padded.
- Byte-exactness versus the reference holds for the default `vector_f32=True`.
