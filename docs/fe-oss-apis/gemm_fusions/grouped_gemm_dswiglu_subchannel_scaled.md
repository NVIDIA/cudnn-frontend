# Grouped GEMM + dSwiGLU (Subchannel-Scaled NVFP4)

**This API is experimental and subject to change.**

Second-level-scaled ("subchannel-scaled") **dSwiGLU-backward** block-scaled
grouped GEMM for MoE workloads (the FC2 data-gradient + dGLU backward fusion).
Inputs are NVFP4 with **two** scale levels — per-(1, 16) FP8 (e4m3) first-level
block scale factors (SFA/SFB) plus FP32 second-level scales (SFA2/SFB2) at
`block2_shape = (sgm, sgn, sgk)` granularity — and the fused epilogue consumes
the forward FC1 pre-activations `c`.

## Operation

For each expert `e` with token rows `rows_e` (`n` = GEMM/weight width; all
n-axis outputs cover `2n`, gate/up):

```
G[rows_e]        = alpha[e]^2 * (sum over sgk-tiles of (A @ B^T) * SFA2 * SFB2[e])
gate, up         = clamp(split_32col_bands(C[rows_e])) * beta[e]
sig              = sigmoid(gate);  swish = gate * sig
dy_gate          = G * prob * up * sig * (1 + gate * (1 - sig))
dy_up            = G * prob * swish
D[rows_e]        = merge(dy_gate, dy_up)            # bf16, or NVFP4-quantized
dprob[rows_e]    = rowsum(swish * up * G)            # always produced
dbias[e]         = colsum(D[rows_e])                 # optional
sfd2_gate/up     = per-(sgm x sgn) block max|dy| / (max(d_dtype) * max(sf_dtype))
```

`A = a_fp4 * sfa` and `B = b_fp4 * sfb` are the first-level dequantized
operands (`value = data * sf * sf2`). In the quantized-output mode, D is
NVFP4-encoded (`d_quant` packed e2m1 + `d_quant_sf` e4m3 row scales) with the
per-block `sfd2` descale folded into the row scales and `norm_const` as the
global encode scale. The default `vector_f32=True` configuration is verified
**byte-exact** against a pure-torch/DSL reference for `d`/`d_quant`/
`d_quant_sf`/`sfd2` (dprob and dbias are atomics-accumulated and compared with
tight tolerances).

Source provenance: ported from the `bs_ggemm_harness` `dgrad_dglu` kernel
(`kernels/dgrad_dglu/kernel.py`), with the deinterleaved-output capability
transplanted from `kernels/dgrad_dglu_rht_2` (`d_deinterleaved` constexpr) and
the dGeGLU/e5m3 paths not exposed. Reference:
`references/fc2_dgrad.py::fc2_dgrad` (dswiglu path).

## Deinterleaved output layout (`d_deinterleaved=True`, default)

The kernel's native storage interleaves gate/up in 32-column bands (matching
`c`). With `d_deinterleaved=True` the n-axis outputs are stored **deinterleaved
`[gate | up]`**: interleaved 32-column band `b` lands at deinterleaved position
`32*(b//2) + (b%2)*n`. This applies to `d_quant` (16-byte packed-band
granularity), `d_quant_sf` (2-byte band granularity), and `dbias` (32-column
granularity). The fused `sfd2` output is laid out as **concatenated halves**
`[gate blocks | up blocks]` — which makes `(d_quant, d_quant_sf, sfd2)`
directly consumable as a downstream GEMM's two-level NVFP4 input
(`a_data, a_scales, a_scales2` with `sgk = sgn`).

`d_deinterleaved=True` requires the quantized-D output (the harness-validated
combination); the BF16-D output is interleaved-only.

## Supported configurations

| Aspect | Supported |
|---|---|
| Architecture | SM100 (the kernel also compiles/runs as-is on sm103/sm107; no Rubin-native variant) |
| A/B dtype | FP4 (`torch.float4_e2m1fn_x2` or `torch.uint8` as packed fp4x2), k-major |
| First-level scales | `torch.float8_e4m3fn`, `sf_vec_size = 16` (no e8m0/e5m3) |
| Second-level scales | `torch.float32`; `sgn % 128 == 0` (MMA tile N), `k % sgk == 0`, `valid_m % sgm == 0` |
| C | `torch.bfloat16` `(valid_m, 2n, 1)`, n-major, gate/up interleaved in 32-col bands |
| D | `torch.bfloat16` `(valid_m, 2n, 1)`, or `torch.float4_e2m1fn_x2` (quantized mode, + `d_quant_sf` e4m3 + `norm_const`) |
| dprob | f32 `(valid_m, 1, 1)`, always produced, atomic-add (zero-init required) |
| dbias | Optional `(l, 2n, 1)` bf16 (f32 = testing-only path), atomics (zero-init required) |
| sfd2 | Fused f32 `(ceil(valid_m/sgm), 2*ceil(n/sgn), 1)`, atomic-max (zero-init required) |
| Activation | dSwiGLU only (`glu_clamp_max/min` = ±7.0 by default; both-or-neither) |
| MMA tiler / cluster | `(256, 128)`; cluster `(2, 1)`, `(2, 2)`, or `(2, 4)` |
| Weight modes | Dense `(n, k, l)` tensors, or discrete per-expert pointer arrays |
| alpha / beta | `(l,)` f32; alpha is applied **squared**; beta scales C inside dSwiGLU |
| Alignment | `valid_m % 256 == 0`, `n % 128 == 0`, `k % 64 == 0` |
| Frameworks | torch only |

## Wrapper API

```python
import cudnn, torch

result = cudnn.grouped_gemm_dswiglu_subchannel_scaled_wrapper_sm100(
    a_tensor=a,            # (valid_m, k, 1) fp4 dY, k-major
    sfa_tensor=sfa,        # MMA-tiled first-level scales
    sfa2_tensor=sfa2,      # (ceil(valid_m/sgm), ceil(k/sgk), 1) f32
    c_tensor=c,            # (valid_m, 2n, 1) bf16 pre-activations
    padded_offsets=offs,   # (l,) int32 cumulative end offsets
    alpha_tensor=alpha,    # (l,) f32 (applied squared)
    beta_tensor=beta,      # (l,) f32
    prob_tensor=prob,      # (valid_m, 1, 1) f32
    b_tensor=b,            # (n, k, l) fp4 (dense mode)
    sfb_tensor=sfb,
    sfb2_tensor=sfb2,      # (ceil(n/sgn), ceil(k/sgk), l) f32
    d_dtype=torch.float4_e2m1fn_x2,       # quantized-D mode (default: bf16)
    norm_const_tensor=torch.tensor([0.5], dtype=torch.float32, device="cuda"),
    dbias=True,
    block2_shape=(1, 256, 256),
    d_deinterleaved=True,
)
d_quant   = result["d_quant_tensor"]     # fp4x2, [gate | up] deinterleaved
d_qsf     = result["d_quant_sf_tensor"]  # e4m3 MMA-tiled row scales
dprob     = result["dprob_tensor"]
dbias_out = result["dbias_tensor"]
sfd2      = result["sfd2_tensor"]        # (rows, 2*nd, 1): [gate | up] halves
```

The wrapper owns the output allocations and their **zero-init contract**
(`dprob`/`dbias`/`sfd2` are atomically accumulated every launch). Class-API
users (`GroupedGemmDswigluSubchannelScaledSm100` + `check_support()` /
`compile()` / `execute()`) must zero those outputs themselves before each
`execute()`.

Discrete mode replaces `b_tensor/sfb_tensor/sfb2_tensor` with 1-D `int64`
per-expert pointer tensors (`b_ptrs/sfb_ptrs/sfb2_ptrs`) plus explicit `n` and
`b_dtype`; per-expert SFB2 blocks must be 16-byte aligned
(`ceil(n/sgn) * ceil(k/sgk) * 4 % 16 == 0` when `num_experts > 1`).

## DSMEM sfd2 reduction

With `dsmem_rowwise=True` (default) and an eligible geometry — quantized-D
output and `sgn / 128 == cluster_n > 1` (`sgn=256` with cluster `(2, 2)`,
`sgn=512` with `(2, 4)`) — the cross-tile rowwise sfd2 reduction runs through
DSMEM: each N-peer CTA `red.shared::cluster`-maxes its row descales into every
peer's smem parity slot and a per-parity `cluster_n`-arrival mbarrier replaces
the gmem atomic + counter-spin protocol. Outputs are **byte-exact** either way
(the identical bit-pattern u32 max; max is order-independent). Cluster shapes
`(2, 2)`/`(2, 4)` additionally require the N work-tile count `ceil(n/128)` to
be a `cluster_n` multiple. Non-eligible geometries keep the gmem protocol
unchanged.

## Notes

- With `sgn > 128` and the quantized-D output, one sfd2 block spans multiple N
  work tiles and the kernel runs a cross-tile arrival protocol; `execute()`
  asserts a conservative liveness bound
  (`(sgn/128 - 1) * min(m_cluster_tiles, n_tiles) < persistent grid clusters`)
  and sizes the sync-counter workspace per call (grow-only).
- `dprob` is bit-exact versus the reference only for `n <= 256` (two 128-wide
  N tiles); larger `n` agrees to ~1e-6 (f32 atomic arrival order).
- No JAX support (the MMA-tiled scale-factor views are not expressible as JAX
  arrays).
