# Linear Attention

Chunked prefill and backward linear attention kernels for Mamba-2, GDN, GDP, KDA, and GDN-2 served by the open-source FROST kernels through the Python frontend. Every variant has a forward, a full backward, and a pair of summary operations for context parallelism.

| Variant | Recurrence (per token `t`) | PyTorch op | Graph nodes |
|---|---|---|---|
| GDN (Gated DeltaNet) | `S_t = alpha_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T`, scalar `alpha_t`, scalar `beta_t` | `gated_delta_net` | `GDN`, `GDN_BWD`, `GDN_SUMMARY`, `GDN_SUMMARY_BWD` |
| KDA (Kimi Delta Attention) | GDN with a per-key-channel decay `alpha_t in (0, 1]^K` | `kimi_delta_attention` | `KDA`, `KDA_BWD`, `KDA_SUMMARY`, `KDA_SUMMARY_BWD` |
| GDN-2 | KDA with a per-key erase gate `beta_t in R^K` and a per-value write gate `w_t in R^V` | `gated_delta_net_v2` | `GDN2`, `GDN2_BWD`, `GDN2_SUMMARY`, `GDN2_SUMMARY_BWD` |
| GDP (Gated DeltaProduct) | GDN with `n = num_householder` beta-gated Householder updates per token, one scalar decay per token | `gated_delta_product` | `GDP`, `GDP_BWD`, `GDP_SUMMARY`, `GDP_SUMMARY_BWD` |
| Mamba-2 (SSD) | `S_t = exp(A dt_t) S_{t-1} + dt_t B_t x_t^T`, `dt_t = clamp(softplus(dt_t + dt_bias), dt_min, dt_max)`, no erase term | `mamba2` | `MAMBA2`, `MAMBA2_BWD`, `MAMBA2_SUMMARY`, `MAMBA2_SUMMARY_BWD` |

The ops live in `cudnn.linear_attention.ops` and are PyTorch custom operators with autograd formulas; the summary ops are `<op>_summary` and `<op>_summary_bwd`. The same graph nodes are available on the Python graph API.

## Support Matrix

Requires an SM100 / SM103 / SM107 GPU (Blackwell data center) and the CuTe DSL runtime the package depends on (`nvidia-cutlass-dsl`, a required dependency except on Windows ARM64; these kernels need 4.7.0 or newer and the engines decline with an error that names the installed version otherwise). On Windows ARM64, the CuTe DSL and TVM-FFI runtime dependencies are excluded from the base installation, so the base install does not enable these kernels there. One linear-attention node per graph.

| | GDN | KDA | GDN-2 | GDP | Mamba-2 |
|---|---|---|---|---|---|
| `q`/`k` head dim `K` | 64, 128 | 64, 128 | 64, 128 | 64, 128 | 128 (`N`, the state dim of `B` / `C`) |
| `v` head dim `V` | 64, 128 | 64, 128 | 64, 128 | 64, 128 | 64 (`P`, the head dim of `x`) |
| gate `g` | `[T, HO]` scalar | `[T, HO, K]` per channel | `[T, HO, K]` per channel | `[T, HO]` scalar, on real tokens | `dt [T, H]` raw step logits |
| write / erase gates | `beta [T, HO]` | `beta [T, HO]` | `beta [T, HO, K]`, `w [T, HO, V]` | `beta [T * n, HO]` | none (`dt` is both the decay exponent and the write strength) |
| gate forms | `gate_domain` log or linear, `safe_gate` (`-exp(a_log) softplus(g + dt_bias)`), `use_beta_sigmoid_in_kernel`, `allow_neg_eigval` | same plus `gate_lower_bound` | same plus `beta_guard` (needs `use_qk_l2norm_in_kernel`) | same as GDN | `dt_softplus`, `dt_bias`, `dt_min`, `dt_max`, `A [H]`, skip `D [H]` or `[H, P]`, output gate `z` |
| `use_qk_l2norm_in_kernel` | ✅ | ✅ | ✅ | ✅ | n/a |
| recurrent state | `initial_state` / `final_state`, `[N, HO, V, K]` | same | same | same | `[N, H, P, N]` |
| `state_checkpoints` (`checkpoint_every_n_tokens`) | multiple of 16 | multiple of 16 | multiple of 16 | multiple of 64 expanded sub-tokens | multiple of 128 |
| `state_indices` (paged state pool, forward only) | ✅ | ✅ | ✅ | ✅ | ✅ |
| `overwrite_initial_state` (in place, inference) | ✅ | ✅ | ✅ | ✅ | ✅ |
| `batch_invariant` | ✅ | ✅ | ✅ | ✅ | ✅ |

`T` is the total token count of the packed batch, `N` the number of sequences.

## Layout

All operands use the packed THD layout, where the tokens of every sequence are concatenated with `cu_seqlens` (`[N + 1]`) giving the sequence boundaries. For example, a dense batch of `N` equal-length sequences is `cu_seqlens = [0, T, 2T, ...]`.

Operands may be strided views, as the engines build their TMA descriptors from the buffer strides, and the only requirement is a contiguous innermost (head-dim) axis. A column slice of a wider per-token buffer (for example the `q`, `k`, `v` pieces of one fused projection) is accepted directly, forward and backward, without `.contiguous()`. Tensors are rank 3 (`[T, heads, dim]`).

## Additional Semantics

- **States and checkpoints.** `output_final_state=True` returns the per-sequence state after the last token. `checkpoint_every_n_tokens > 0` also returns the state series, one entry every `n` tokens strictly before each sequence end, as a non-differentiable checkpoint that the backward can reuse instead of recomputing. `state_indices` turns `initial_state` into a pool `[N_pool, ...]` addressed per sequence, written back in place.
- **batch_invariant.** Each sequence's results are bitwise independent of the batch composition (whole-sequence scheduling; split-K load balancing is disabled).
- **Clamps and guards.** Mamba-2's `dt` clamp zeroes the gate gradient where active. GDN-2's `beta_guard` is recomputed, not differentiated, in the backward.
- **Engines.** The FROST engines are the only engines for these nodes and are pure pass-through: `check_support` verifies the kernel-native dtypes and head dims listed above and declines everything else. A plan can be pinned by name through `plan_name` (for example `gdn_frost`, `mamba2_frost`, `mamba2_summary_frost`).

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

## Context Parallelism

The five families support chunkwise context parallelism. We leave the all-gather of the summary to the framework-level, but we compute the summaries and combine them.

Each rank summarizes one boundary sequence. `<op>_summary` for the last local sequence from a zero state for the forward, and `<op>_summary_bwd` on the first local sequence from a zero outgoing-state gradient for the backward. Each returns a state row and the span's transition operator. After the all-gather, the forward on rank `rank` takes `gathered[preceding_rank_start:rank]` and the backward `gathered[rank + 1:following_rank_stop]`, in ascending rank order. The fold is inside the ops. The forward takes the preceding ranks' slices as `preceding_state_update` and `preceding_transition`. The raw backward op `torch.ops.cudnn.<op>_bwd` takes the following ranks' slices as `following_state_grad` and `following_transition` and folds them into the gradient state the last local sequence starts from.

## Context Parallelism Example

Below, we show an example of using context parallelism in practice.

```python
# forward prepare: the LAST local sequence from a zero state
h, m = gated_delta_net_summary(k, v, g, beta, cu[-2:])  # fp32 [1, HO, V, K] and [1, HO, K, K]
# all-gather h[0] / m[0] into h_all / m_all
o, _, checkpoints, composed = gated_delta_net(
    q, k, v, g, beta, cu, scale=scale,
    preceding_state_update=h_all[pre_start:rank], preceding_transition=m_all[pre_start:rank]
)
# backward prepare: the FIRST local sequence from a zero outgoing-state gradient
g_, t = gated_delta_net_summary_bwd(q, k, g, beta, dO, cu[:2], scale=scale, output_transition=True)
# all-gather g_[0] / t[0] into g_all / t_all
dq, dk, dv, dg, dbeta, _, d_a_log, d_dt_bias = torch.ops.cudnn.gated_delta_net_bwd(
    dO, q, k, v, g, beta, cu, scale,
    composed_initial_state=composed if composed.numel() else None,
    state_checkpoints=checkpoints if checkpoints.numel() else None,
    following_state_grad=g_all[rank + 1 : fol_stop], following_transition=t_all[rank + 1 : fol_stop],
)
```
