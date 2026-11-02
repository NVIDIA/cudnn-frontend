# Linear Attention

Chunked linear-attention and state-space layers with a recurrent matrix state `S` per (sequence, head), served by the
open-source FROST kernels (CuTe DSL, `nvidia-cutlass-dsl`) through the Python frontend. There is no C++ API for these
operations. Every variant has a forward, a full backward, and a pair of summary operations for context parallelism.

| Variant | Recurrence (per token `t`) | Python op | Graph node types |
|---|---|---|---|
| GDN (Gated DeltaNet) | `S_t = alpha_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T`, scalar `alpha_t`, scalar `beta_t` | `gated_delta_net` | `GDN`, `GDN_BWD`, `GDN_SUMMARY`, `GDN_SUMMARY_BWD` |
| KDA (Kimi Delta Attention) | GDN with a per-key-channel decay `alpha_t in (0, 1]^K` | `kimi_delta_attention` | `KDA`, `KDA_BWD`, `KDA_SUMMARY`, `KDA_SUMMARY_BWD` |
| GDN-2 | KDA with a per-key erase gate `beta_t in R^K` and a per-value write gate `w_t in R^V` | `gated_delta_net_v2` | `GDN2`, `GDN2_BWD`, `GDN2_SUMMARY`, `GDN2_SUMMARY_BWD` |
| GDP (Gated DeltaProduct) | GDN with `n = num_householder` beta-gated Householder updates per token, one scalar decay per token | `gated_delta_product` | `GDP`, `GDP_BWD`, `GDP_SUMMARY`, `GDP_SUMMARY_BWD` |
| Mamba-2 (SSD) | `S_t = exp(A dt_t) S_{t-1} + dt_t B_t x_t^T`, `dt_t = clamp(softplus(dt_t + dt_bias), dt_min, dt_max)`, no erase term | `mamba2` | `MAMBA2`, `MAMBA2_BWD`, `MAMBA2_SUMMARY`, `MAMBA2_SUMMARY_BWD` |

The ops live in `cudnn.linear_attention.ops` and are PyTorch custom operators with autograd formulas; the summary ops
are `<op>_summary` and `<op>_summary_bwd`. The same node types are available on the Python graph API.

## Support Matrix

Requires an SM100 / SM103 / SM107 GPU (Blackwell data center) and the CuTe DSL runtime the package depends on
(`nvidia-cutlass-dsl`, a required dependency except on Windows ARM64; these kernels need 4.7.0 or newer and the engines
decline with an error that names the installed version otherwise). On Windows ARM64, the CuTe DSL and TVM-FFI runtime
dependencies are excluded from the base installation, so the base install does not enable these kernels there.
One linear-attention node per graph.

| | GDN | KDA | GDN-2 | GDP | Mamba-2 |
|---|---|---|---|---|---|
| io dtype (`q`, `k`, `v`, `O`, `dO`) | bf16, fp16 | bf16, fp16 | bf16, fp16 | bf16, fp16 | bf16, fp16 (`x`, `B`, `C`, `z`) |
| `q`/`k` head dim `K` | 64, 128 | 64, 128 | 64, 128 | 64, 128 | 128 (`N`, the state dim of `B` / `C`) |
| `v` head dim `V` | 64, 128 | 64, 128 | 64, 128 | 64, 128 | 64 (`P`, the head dim of `x`) |
| grouped query heads | `HK = H` or `HK = HV`; `O` and the states carry `HO = max(H, HV)` | same | same | same | `B`, `C` grouped over `G` with `H` a multiple of `G` |
| gate `g` | `[T, HO]` scalar; fp32, bf16, fp16 | `[T, HO, K]` per channel | `[T, HO, K]` per channel | `[T, HO]` scalar, on real tokens | `dt` `[T, H]` raw step logits; fp32, bf16, fp16 |
| write / erase gates | `beta` `[T, HO]`; fp32 or io dtype | `beta` `[T, HO]`; fp32 or io dtype | `beta` `[T, HO, K]`, `w` `[T, HO, V]`; io dtype | `beta` `[T * n, HO]`; fp32 or io dtype | none (`dt` is both the decay exponent and the write strength) |
| gate forms | `gate_domain` log or linear, `safe_gate` (`-exp(a_log) softplus(g + dt_bias)`), `use_beta_sigmoid_in_kernel`, `allow_neg_eigval` | same plus `gate_lower_bound` | same plus `beta_guard` (needs `use_qk_l2norm_in_kernel`) | same as GDN | `dt_softplus`, `dt_bias`, `dt_min`, `dt_max`, `A` `[H]`, skip `D` `[H]` or `[H, P]`, output gate `z` |
| `use_qk_l2norm_in_kernel` | yes | yes | yes | yes | n/a |
| recurrent state | `initial_state` / `final_state` `[N, HO, V, K]`, fp32, bf16 or fp16 | same | same | same | `[N, H, P, N]`, fp32, bf16 or fp16 |
| `state_checkpoints` (`checkpoint_every_n_tokens`) | multiple of 16 | multiple of 16 | multiple of 16 | multiple of 64 expanded sub-tokens | multiple of 128 |
| `state_indices` (paged state pool, forward only) | yes | yes | yes | yes | yes |
| `overwrite_initial_state` (in place, inference) | yes | yes | yes | yes | yes |
| `batch_invariant` | yes | yes | yes | yes | yes |
| backward outputs | `dQ, dK, dV, dG, dBeta, d_initial_state` (+ `d_a_log`, `d_dt_bias` under `safe_gate`) | same | same plus `dW` | same as GDN | `dQ (dC), dK (dB), dV (dx), dG (ddt), dA, d_initial_state, d_dt_bias, dD, dZ` |
| summaries (context parallelism) | `<op>_summary(k, v, g, beta, cu_seqlens[, initial_state])` gives `final_state` + `transition` `[N, HO, K, K]` fp32; `<op>_summary_bwd` gives `d_initial_state` + `transition` | same | same | same | `mamba2_summary(x, dt, A, B, cu_seqlens[, initial_state, dt_bias])`; `transition` `[N, H]` fp32, the log2 of the span's decay product |

For distributed KDA integrations, the explicit Kimi CP state merge
([docs](/operations/LinearAttention/kimi_cp_merge),
[GitHub](https://github.com/NVIDIA/cudnn-frontend/blob/develop/docs/operations/linear_attention/kimi_cp_merge.md))
combines caller-gathered FP32 affine summaries with IEEE FP32 arithmetic. It
leaves summary generation, communication, overlap, and autograd orchestration
to the downstream framework.

`T` is the total token count of the packed batch, `N` the number of sequences. Dtypes are kernel-native and strict:
callers convert, and every gradient comes back in its input's dtype. `cu_seqlens` is int32 or int64.

## Layout

All operands use the packed THD layout: tokens of every sequence concatenated along the first axis, with
`cu_seqlens` (`[N + 1]`) giving the sequence boundaries. A dense batch of `N` equal-length sequences is
`cu_seqlens = [0, T, 2T, ...]` over the flattened tokens; no copy is needed.

Operands may be strided views: the engines build their TMA descriptors from the buffer strides, and the only
requirement is a contiguous innermost (head-dim) axis. A column slice of a wider per-token buffer (for example the
`q`, `k`, `v` pieces of one fused projection) is accepted directly, forward and backward, without `.contiguous()`.
Tensors are rank 3 (`[T, heads, dim]`); a hidden-fused `[T, heads * dim]` slice needs a `.view` to rank 3, which is
free when `heads * dim` is contiguous.

GDP runs on two timelines: `q` and `g` have one row per real token, while `k`, `v` and `beta` have `n` rows per token.
They come either expanded (`[T * n, ...]`, update-major: the `n` updates of token `t` are rows `t * n` to `t * n + n - 1`)
or token-major (`k` `[T, n, HK, K]`, `v` `[T, n, HV, V]`, `beta` `[T, n, HO]`, with the caller's token stride), so the
per-token column slices of a joint `[V | K | Q]` projection (`[T, n * H * D]` viewed as `[T, n, H, D]`) are consumed in
place; `dk`, `dv` and `dbeta` come back in the input's shape. `num_householder == 1` is exactly GDN.

Mamba-2 uses the SSD vocabulary: `x` plays `v`, `B` plays `k`, `C` plays `q`, `dt` is the gate. The output is
`y = C^T S + D x`, gated by `z sigmoid(z)` when `z` is given.

## Semantics worth knowing

- **States and checkpoints.** `output_final_state=True` returns the per-sequence state after the last token.
  `checkpoint_every_n_tokens > 0` also returns the state series, one entry every `n` tokens strictly before each
  sequence end, as a non-differentiable dump that the backward reuses instead of recomputing. The rows are io dtype by
  default; every variant also takes `checkpoint_dtype=torch.float32` for fp32 rows (the backward reuses io rows at the
  chunk cadence and regenerates the dense series from any other rows). `state_indices`
  turns `initial_state` into a pool `[N_pool, ...]` addressed per sequence, written back in place; forward only
  and exclusive with checkpoints.
- **Summaries.** `<op>_summary` returns the end state of each sequence from a given start state together with the
  span's state-transition operator `transition`: for the delta-rule variants a `[N, HO, K, K]` fp32 matrix (the
  product of the span's decayed erase steps, in the backward's stored orientation), for Mamba-2 a `[N, H]` fp32 scalar
  per head (the log2 of the decay product), so that across spans `S_final = S_init * transition + S_H`.
  `<op>_summary_bwd` is the mirror for the state gradient. Together they implement context parallelism over
  sequence spans. Both are non-differentiable.
- **batch_invariant.** Each sequence's results are bitwise independent of the batch composition (whole-sequence
  scheduling; split-K load balancing is disabled).
- **Clamps and guards.** Mamba-2's `dt` clamp zeroes the gate gradient where active. GDN-2's `beta_guard` is
  recomputed, not differentiated, in the backward.
- **Engines.** The FROST engines are the only engines for these nodes and are pure pass-through: `check_support`
  verifies the kernel-native dtypes and head dims listed above and declines everything else. A plan can be pinned by
  name through `plan_name` (for example `gdn_frost`, `mamba2_frost`, `mamba2_summary_frost`).

## Example

```python
import torch
from cudnn.linear_attention.ops import gated_delta_net, mamba2

T, H, K, V = 4096, 16, 128, 128
cu_seqlens = torch.tensor([0, 1000, 2500, T], dtype=torch.int32, device="cuda")
q = torch.randn(T, H, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
k = torch.randn(T, H, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
v = torch.randn(T, H, V, dtype=torch.bfloat16, device="cuda", requires_grad=True)
g = -torch.nn.functional.softplus(torch.randn(T, H, device="cuda"))
beta = torch.sigmoid(torch.randn(T, H, device="cuda"))

o, final_state = gated_delta_net(q, k, v, g, beta, cu_seqlens, output_final_state=True, use_qk_l2norm_in_kernel=True)
o.sum().backward()

P, N, G = 64, 128, 2
x = torch.randn(T, H, P, dtype=torch.bfloat16, device="cuda")
dt = torch.randn(T, H, device="cuda")
A = -torch.exp(torch.randn(H, device="cuda"))
B = torch.randn(T, G, N, dtype=torch.bfloat16, device="cuda")
C = torch.randn(T, G, N, dtype=torch.bfloat16, device="cuda")
y, final_state = mamba2(x, dt, A, B, C, cu_seqlens, D=torch.ones(H, device="cuda"), output_final_state=True)
```

Benchmarks against the reference implementations are in `benchmark/linear_attention/`.
