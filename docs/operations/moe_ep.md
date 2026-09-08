# Mixture of Experts with Expert Parallelism

The MoeEP operation fuses token routing, expert SwiGLU computation, and
expert-parallel communication. Global experts are sharded contiguously across
the ranks of an expert-parallel process group.

## Operation

Let $x_t \in \mathbb{R}^{H}$ be token $t$, $e_{t,k}$ its $k$-th
selected global expert, and $p_{t,k}$ the corresponding routing weight. For
each valid route, split the FC1 result into gate and up projections:

$$
\left[g_{t,k}, u_{t,k}\right]
    = x_t W^{\mathrm{fc1}}_{e_{t,k}},
\qquad
h_{t,k}
    = p_{t,k}\left(\mathrm{SiLU}(g_{t,k}) \odot u_{t,k}\right),
\qquad
z_{t,k}
    = h_{t,k} W^{\mathrm{fc2}}_{e_{t,k}}.
$$

The final token output is the sum over its selected experts:

$$
y_t = \sum_{\substack{0 \le k < K \\ e_{t,k} \ne -1}} z_{t,k}.
$$

When `gate_up_clamp=C`, the operation uses
$\min(g_{t,k}, C)$ for the gate and
$\mathrm{clip}(u_{t,k}, -C, C)$ for the up projection. A route whose
expert ID is `-1` contributes zero. Because the executable backend requires
`apply_topk_in_fc1=True`, it applies $p_{t,k}$ to the SwiGLU result before
FC2. The backend also stages plain inputs to MXFP8 and requantizes the routed
intermediate before FC2, so the equations describe the mathematical operation
rather than its finite-precision rounding.

With $E$ global experts and an expert-parallel group of size $P$, each rank
stores $E_{\mathrm{local}}=E/P$ consecutive experts. Global expert $e$ is
owned by group-relative rank

$$
\mathrm{owner}(e)
    = \left\lfloor \frac{e}{E_{\mathrm{local}}} \right\rfloor.
$$

## Python API

The operation is exposed by the frontend-only `cudnn.MoeEp` object API. Static
model, parallelism, capacity, and format choices are set in the constructor:

```python
from cudnn import MoeEp

op = MoeEp(
    num_experts=E,
    hidden_size=H,
    intermediate_size=I,
    top_k=K,
    ep_group=ep_group,                 # None for EP1
    max_tokens_per_rank=max_tokens,
    max_recv_size_per_rank=None,       # Defaults to P * max_tokens * K
    drop_on_overflow=False,
    output_format="bf16",
    combine_format="bf16",             # "bf16" or "mxfp8"
    apply_topk_in_fc1=True,
    weight_interleave_size=None,       # Or 32 for pre-interleaved MXFP8 W1
    gate_up_clamp=None,
)
```

`topk_idx` contains global expert IDs. Each rank passes its local tokens and
its contiguous shard of both expert-weight tensors:

`weight_interleave_size=32` declares that MXFP8 FC1 values already use
alternating 32-element gate/up strips. The default `None` uses conventional
gate-then-up order. Plain BF16/FP16/FP32 weights remain conventional and reject
the interleaved contract because they must be quantized and staged internally.

```python
output = op(
    activation,       # (T, H)
    fc1_weight,       # (E_local, H, 2I)
    fc2_weight,       # (E_local, I, H)
    topk_idx,         # (T, K), global expert IDs or -1
    topk_weights,     # (T, K)
)                     # (T, H), BF16
```

For inference CUDA Graph capture, call `op.warmup(...)` with the exact
bindings before capture. `MoeEp` supports `close()` and context-manager use.

Explicit sweep autotuning is available before capture:

```python
from cudnn import MoeEpTuningConfig

result = op.autotune(
    activation, fc1_weight, fc2_weight, topk_idx, topk_weights,
    candidates=[
        MoeEpTuningConfig(token_in_flag_batch=2),
        MoeEpTuningConfig(group_hint=256),
    ],
)
```

The current tuning is always included as the baseline. Candidates are
de-duplicated and limited to 32 including that baseline. The winner is applied
only to this operator instance.

Stateless training prepares only private execution lanes. Every invocation
receives independent native weights and caller-owned outputs:

```python
requirements = op.prepare_training(lane_count=1, device=device)
lane = op.training_lanes[0]

output = op.training_forward(
    lane, activation, topk_idx, topk_weights,
    weights=native_forward_weights,
    out=forward_outputs,
)
grad_activation, dprob, wgrad_operands = op.training_backward(
    lane, grad_output, topk_idx, topk_weights,
    weights=native_backward_weights,
    fc1_preact=forward_outputs.fc1_preact,
    fc1_a=forward_outputs.fc1_a,
    fc1_sfa=forward_outputs.fc1_sfa,
    valid_route_counts=forward_outputs.valid_route_counts,
    expert_offsets=forward_outputs.expert_offsets,
    out=backward_outputs,
)
```

The WGrad result is a fixed-capacity grouped-GEMM operand bundle, not dense
optimizer-ready weight gradients. See the detailed
[MoE + Expert Parallel API](../fe-oss-apis/moe_ep.md) reference for
installation, all constructor arguments, native layouts, buffer ownership,
overflow handling, and CUDA Graph requirements. MoeEP is
distinct from the cuDNN graph [MoE Grouped Matmul](MoeGroupedMatmul.md)
operation.

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

The stateless training CUDA Graph path has hardware acceptance through EP32 when
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

Training accepts contiguous BF16/FP32 or MXFP8 block-scaled activation and
gradient inputs. Execution weights use versioned kernel-native E4M3 payload
and Rubin-blocked E8M0 scale layouts.

## Tensor contracts

Let:

- $T$ be the local token count;
- $H$ be `hidden_size`;
- $I$ be `intermediate_size`;
- $K$ be `top_k`;
- $E_{\mathrm{local}}$ be the local expert count.

Inference uses:

- `activation`: `(T, H)`;
- `topk_idx`: `(T, K)`, Int32 or Int64, containing `-1` or a valid global
  expert ID;
- `topk_weights`: `(T, K)`, floating point;
- FC1 weights: `(E_local, H, 2I)`;
- FC2 weights: `(E_local, I, H)`;
- output: `(T, H)`, BF16.

All inference tensors must reside on one device, and the local token count must
satisfy `T <= max_tokens_per_rank`.

Stateless training uses:

- `activation` and `grad_output`: contiguous `(T, H)`, BF16, FP32, or MXFP8;
- `topk_idx`: contiguous `(T, K)`, Int32;
- `topk_weights`: contiguous `(T, K)`, FP32;
- independent forward and backward native weight packs with exact versioned
  `layout_id` values;
- required caller-owned forward output: `(T, H)`, BF16;
- required caller-owned `fc1_preact`, produced by training forward with
  `generate_c=True` and retained through matching backward;
- required caller-owned `grad_activation`: `(T, H)` view of a capacity buffer,
  FP32;
- required caller-owned `dprob`: source-order `(T, K)`, FP32;
- required caller-owned WGrad saved state and a fixed-capacity
  `MoeEpTrainingWgradOperands` bundle.

All dynamic training tensors must reside on one device and satisfy
`T <= max_tokens_per_rank`.

## Expert-parallel communication

EP2+ execution requires:

- an initialized NCCL process group;
- `nvshmem4py` and usable NVSHMEM libraries;
- direct peer access among every pair of participating ranks; and
- consistent rank ordering, buffer schemas, tuning, lane selection, and launch
  ordering across the group.

`max_recv_size_per_rank` bounds receive capacity. When omitted, it defaults to
the worst-case route count:

```text
ep_size * max_tokens_per_rank * top_k
```

Private lane resources cannot grow during CUDA Graph replay. Capacity changes
require a new operator preparation; caller-address changes require recapture.
