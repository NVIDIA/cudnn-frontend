# Mixture of Experts with Expert Parallelism

The MoeEP operation fuses token routing, expert SwiGLU computation, and
expert-parallel communication. Global experts are sharded contiguously across
the ranks of an expert-parallel process group.

For token \(x_t\), selected expert \(e_{t,k}\), and routing weight \(p_{t,k}\),
the operation computes:

\[
y_t = \sum_{k=0}^{K-1} p_{t,k}
      \left(\operatorname{SiLU}(x_t W^{gate}_{e_{t,k}})
      \odot (x_t W^{up}_{e_{t,k}})\right)
      W^{down}_{e_{t,k}}
\]

The current implementation is exposed by the frontend-only Python
[`cudnn.moe_ep.MoeEp`](../fe-oss-apis/moe_ep.md) API. It is distinct from the
cuDNN graph [MoE Grouped Matmul](MoeGroupedMatmul.md) operation.

## Execution support

- NVIDIA Rubin SM107 GPUs (compute capability 10.7).
- CUDA and PyTorch execution.
- `nvidia-cutlass-dsl>=4.8.0` for the Rubin kernels. The package-wide
  `cutedsl` extra retains its 4.5.0 installation floor so other cuDNN Frontend
  operations remain usable with older compatible DSL versions.
- Fused SwiGLU with contiguous expert sharding.
- `apply_topk_in_fc1=True`.
- `hidden_size` divisible by 128.
- `intermediate_size` divisible by 256.
- `top_k <= min(32, num_experts)`.
- `num_experts` divisible by the expert-parallel group size.
- An explicit positive `max_tokens_per_rank`.

The fixed-resource CUDA Graph path has hardware acceptance through EP32 when
all ranks are in one direct-P2P MNNVL peer-access domain. The Python capability
layer does not impose an EP-size ceiling; cross-MNNVL execution is not part of
the validated support surface.

## Data formats

Inference activation and expert weights accept:

- BF16, FP16, or FP32 plain tensors, staged internally to MXFP8; or
- MXFP8 `BlockScaledTensor` values with logical block axis 1.

The current executable output format is BF16. The expert-combine path accepts
BF16 or MXFP8. NVFP4 types are represented by the public API but native NVFP4
operands, combine, and output are not executable by this backend.

Fixed-resource training narrows dynamic activation and gradient inputs to
contiguous BF16 or FP32 tensors. Training weights are contiguous MXFP8
block-scaled tensors.

## Tensor contracts

Let:

- \(T\) be the local token count;
- \(H\) be `hidden_size`;
- \(I\) be `intermediate_size`;
- \(K\) be `top_k`;
- \(E_{local}\) be the local expert count.

Inference uses:

- `activation`: `(T, H)`;
- `topk_idx`: `(T, K)`, Int32 or Int64, containing `-1` or a valid global
  expert ID;
- `topk_weights`: `(T, K)`, floating point;
- FC1 weights: `(E_local, H, 2I)`;
- FC2 weights: `(E_local, I, H)`;
- output: `(T, H)`, BF16.

Fixed-resource training additionally binds transposed backward weights with
shapes `(E_local, H, I)` and `(E_local, 2I, H)`. Dynamic tensors must share one
device and satisfy `T <= max_tokens_per_rank`.

## Expert-parallel communication

EP2+ execution requires:

- an initialized NCCL process group;
- `nvshmem4py` and usable NVSHMEM libraries;
- direct peer access among every pair of participating ranks; and
- consistent rank ordering, resource sizes, tuning, slot selection, and lane
  ordering across the group.

`max_recv_size_per_rank` bounds receive capacity. When omitted, it defaults to
the worst-case route count:

```text
ep_size * max_tokens_per_rank * top_k
```

Resources cannot grow during CUDA Graph replay. Capacity or storage changes
require resource preparation and graph capture again.

## API reference

See [MoE + Expert Parallel API](../fe-oss-apis/moe_ep.md) for installation,
constructor arguments, inference and training lifecycles, tuning, overflow
handling, and CUDA Graph usage.
