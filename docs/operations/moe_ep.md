# Mixture of Experts with Expert Parallelism

The MoeEP operation fuses token routing, expert SwiGLU computation, and
expert-parallel communication. Global experts are sharded contiguously across
the ranks of an expert-parallel process group.

Inference and training share one `MoeEp` object but use separate call
surfaces:

- `MoeEp.__call__` and `warmup` for inference;
- `prepare_training`, `training_forward`, and `training_backward` for training.

Training is stateless with respect to caller tensors. The operator retains
compiled kernels, runtime state, and per-lane NVSHMEM storage, but it does not
retain weights, saved forward state, or fallback weight staging. The symmetric
input and final-output views it does own are handed to the caller explicitly
by `training_symmetric_buffers`.

MoeEP is distinct from the cuDNN graph
[MoE Grouped Matmul](MoeGroupedMatmul.md) operation.

## Installation

```bash
pip install "nvidia-cudnn-frontend[cutedsl,comm]" torch torch-c-dlpack-ext
```

The `comm` extra supplies NVSHMEM and is only needed for EP2+. See
[Execution support](#execution-support) for the hardware and DSL floor, and
[Expert-parallel communication](#expert-parallel-communication) for what EP2+
requires at runtime.

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
y_t = \sum_{0 \le k < K} z_{t,k}.
$$

When `gate_up_clamp=C`, the operation uses
$\min(g_{t,k}, C)$ for the gate and
$\mathrm{clip}(u_{t,k}, -C, C)$ for the up projection. Every route must name a
global expert in `[0, E)`; negative IDs and dropped-route sentinels are not
supported. Because the executable backend requires
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

### Constructing the operator

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
    max_recv_size_per_rank=None,       # Physical pool size; see below for the default
    drop_on_overflow=False,
    output_format="bf16",
    combine_format="bf16",             # "bf16" or "mxfp8"
    apply_topk_in_fc1=True,
    weight_interleave_size=None,       # Or 32 for pre-interleaved MXFP8 W1
    gate_up_clamp=None,
    token_padding_size=128,            # Must be 128 for training WGrad operands
    sf_padding_size=128,               # Positive multiple of 128
    forward_tuning=None,               # Or an explicit MoeEpTuningConfig
    backward_tuning=None,              # Independent; epi_warps only
    validation_mode="strict",          # Or "trusted"
)
```

Every argument is keyword-only. `token_padding_size` and `sf_padding_size`
select the routed-token and scale-factor padding blocks; the training path
that exports WGrad operands requires both to be exactly 128 and rejects any
other value.

Forward and backward carry independent tuning. `forward_tuning` also governs
inference; `tuning` is a backward-compatible alias for it, and passing both
raises. `backward_tuning` defaults to a fresh `MoeEpTuningConfig` rather than
inheriting the forward one, and it accepts only `token_back_mode="epi_warps"`
with `reduce_topk_in_kernel=False`.

`weight_interleave_size=32` declares that MXFP8 FC1 values already use
alternating 32-element gate/up strips. The default `None` uses conventional
gate-then-up order. Plain BF16/FP16/FP32 weights remain conventional and reject
the interleaved contract because they must be quantized and staged internally.
Native training requires `weight_interleave_size=32`.

### Routing contract

Inference and training require dense routing: `topk_idx` has shape `(T, K)`
and every value must be a global expert ID in `[0, num_experts)`. Negative
IDs, including the conventional `-1` dropped-route sentinel, are not
supported. Private fixed-capacity staging may use `-1` after row `T`; callers
never pass or consume that tail.

`validation_mode="strict"` checks expert-ID values during eager execution and
warmup. `"trusted"` skips that device-value check and relies on the caller.
Structural shape, dtype, device, capacity, aliasing, and output checks stay
enabled in both modes. CUDA Graph capture and replay perform no semantic
routing validation, so routing contents updated at captured addresses must
continue to satisfy the same dense contract.

### Inference

`topk_idx` contains one valid global expert ID for every `(token, top-k slot)`.
Each rank passes its local tokens and its contiguous shard of both
expert-weight tensors:

```python
output = op(
    activation,       # (T, H)
    fc1_weight,       # (E_local, H, 2I)
    fc2_weight,       # (E_local, I, H)
    topk_idx,         # (T, K), global expert IDs in [0, E)
    topk_weights,     # (T, K)
)                     # (T, H), BF16
```

For inference CUDA Graph capture, call `op.warmup(...)` with the exact
bindings before capture. `MoeEp` supports `close()` and context-manager use.

### Explicit sweep autotuning

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

`MoeEp.autotune` measures inference forward. `MoeEp.autotune_training`
measures one training forward immediately followed by its matching backward:

```python
training_result = op.autotune_training(
    activation, grad_output, topk_idx, topk_weights,
    forward_weights=native_fw,
    backward_weights=native_bw,
    candidates=candidates,
    warmup_iters=3,
    timed_iters=10,
)
```

Both calls are collective over `ep_group`, must use the same ordered candidate
list on every rank, and must run outside CUDA Graph capture.
`autotune_training` must run before `prepare_training`. It accepts only native
weights; source packing and allocation are intentionally outside its measured
region.

The current `MoeEpTuningConfig` is prepended as a baseline, duplicate values
are removed, and the normalized list is limited to 32 candidates. Autotuning
keeps `reduce_topk_in_kernel` fixed because that flag changes where top-k
reduction is performed. Each timed iteration is reduced with rank MAX and the
candidate score is the median of those slow-rank samples. Equal scores select
the earlier candidate. `MoeEpAutotuneResult` reports `winner`, per-candidate
`latency_ms` and `samples_ms` as `MoeEpAutotuneCandidateResult` entries, and
`evaluated_candidates`.

The sweep is fail-fast. Any validation, allocation, compile, launch, timing,
synchronization, or teardown error ends the whole sweep. An error after
runtime/collective entry poisons the operator, and later execution is
rejected; close it and create a new instance. Compiled candidate kernels
remain in the process JIT cache. The production sweep does not compare
candidate outputs at runtime; supported candidates are covered by the separate
correctness suite.

Autotuning commits one active winner per instance, applied only to this
operator instance. A later inference or training sweep replaces it. Existing
CUDA Graph executables are invalid after the winner changes. Use these
sequences:

- inference: `autotune` → eager winner launch (performed by `autotune`) →
  capture;
- training: `autotune_training` → `prepare_training` → allocate outputs →
  eager forward/backward → rank synchronization → capture.

### Stateless training preparation

Preparation allocates only private execution lanes. It is collective over
`ep_group` and must run outside CUDA Graph capture:

```python
requirements = op.prepare_training(
    lane_count=1,
    device=None,  # current CUDA device; pass an explicit device for multi-GPU hosts
    native_weight_storage_mode="contiguous",  # Or "discrete"
)
lane = op.training_lanes[0]
symmetric = op.training_symmetric_buffers(lane)
```

`prepare_training` does not accept or bind weights. It returns a plain mapping
whose values are:

```text
(shape, stride, dtype, alignment_bytes)
```

The requirements mapping contains `output`, `fc1_preact`, `fc1_a`, `fc1_sfa`,
`valid_route_counts`, `expert_offsets`, `grad_activation`, `dprob`, `fc1_b`,
`fc1_sfb`, `fc2_a`, `fc2_sfa`, `fc2_b`, and `fc2_sfb`.

Buffers come from two places. `training_symmetric_buffers(lane)` returns the
cuDNN-allocated `forward_input`, `forward_input_scale`, `backward_input`,
`backward_input_scale`, `output`, `grad_activation`, and `dprob` tensors for
that lane; quantize directly into the input pairs and bind the returned output
tensors in the public output bundles. The caller allocates the remaining
entries of the requirements mapping. cuDNN validates exact shape, stride,
dtype, alignment, device, and non-aliasing before launch.

`device=None` binds the current CUDA device. An explicit CUDA device takes
precedence. Every later training tensor must use the bound device. Each entry
in `op.training_lanes` is a `MoeEpExecutionLane`.

### Native weight ABI

Forward and backward receive independent packs:

```python
from cudnn import (
    MoeEpNativeForwardWeights,
    MoeEpNativeBackwardWeights,
    MoeEpNativeWeight,
    MoeEpNativeWeightLayout,
)
```

Each `MoeEpNativeWeight` contains:

- `payload`: kernel-native E4M3 data;
- `scale`: contiguous Rubin-blocked E8M0 scales;
- `layout_id`: the exact versioned payload-and-scale layout.

Execution validates the `layout_id` and passes payload and scale pointers to
the kernel without transformation or retention. Eager calls may use different
weight addresses. CUDA Graph capture pins every referenced address until the
graph executable is destroyed.

Let `B(R, C) = round_up(R, 128) * round_up(C, 4)`. The native V1 contracts
are:

- forward FC1: payload `(E_local, H, 2I)`, stride `(2HI, 1, H)`, with
  gate/up 32-column strips; scale `(E_local, B(2I, H/32))`;
- forward FC2: payload `(E_local, I, H)`, stride `(IH, 1, I)`; scale
  `(E_local, B(H, I/32))`;
- backward W2-transpose: contiguous payload `(E_local, H, I)`; scale
  `(E_local, B(I, H/32))`;
- backward W1-transpose: contiguous payload `(E_local, 2I, H)`, with
  gate/up 32-row strips; scale `(E_local, B(H, 2I/32))`.

Every native scale tensor is contiguous E8M0. The corresponding
`MoeEpNativeWeightLayout` enum value is required; a compact or differently
swizzled scale tensor is rejected even when its element count matches.

When upstream does not already produce native weights, use caller-owned
staging:

```python
native_fw = op.pack_forward_weights(source_fw, out=forward_staging)
native_bw = op.pack_backward_weights(source_bw, out=backward_staging)
```

The equivalent standalone `pack_forward_weights` and `pack_backward_weights`
functions are also exported. Packing allocates nothing: every transformed
payload or scale is written to the supplied `MoeEpForwardWeightStaging` /
`MoeEpBackwardWeightStaging` bundle. These fallback packers consume logical
gate-then-up `MoeEpForwardWeights` / `MoeEpBackwardWeights` with compact
axis-1 scales; already interleaved, blocked producers should construct the
native packs directly instead of packing them again.

#### Discrete expert allocations

Training also accepts weights whose experts live in independent allocations.
Bind this compile-time ABI explicitly:

```python
from cudnn import (
    MoeEpNativeDiscreteBackwardWeights,
    MoeEpNativeDiscreteForwardWeights,
    MoeEpNativeDiscreteWeight,
)

requirements = op.prepare_training(
    device=device,
    native_weight_storage_mode="discrete",
)

native_fw = MoeEpNativeDiscreteForwardWeights(
    fc1=MoeEpNativeDiscreteWeight(
        fc1_payload_ptrs, fc1_scale_ptrs,
        MoeEpNativeWeightLayout.FORWARD_FC1_GATE_UP_INTERLEAVED_32_V1,
    ),
    fc2=MoeEpNativeDiscreteWeight(
        fc2_payload_ptrs, fc2_scale_ptrs,
        MoeEpNativeWeightLayout.FORWARD_FC2_K_MAJOR_V1,
    ),
)
```

Build the backward bundle analogously with
`MoeEpNativeDiscreteBackwardWeights` and the discrete-only
`BACKWARD_W2_DGRAD_NK_ROW_MAJOR_V1` and
`BACKWARD_W1_DGRAD_GATE_UP_INTERLEAVED_32_NK_ROW_MAJOR_V1` layout IDs. These
IDs are intentionally distinct from the contiguous transpose IDs: the
upstream pointer-table kernel has no stride argument and interprets each
payload physically as row-major GEMM `(N, K)`.

The per-expert discrete payload contracts are:

- forward FC1: physical `(2I, H)` row-major, equivalent to the V1
  `(H, 2I)` K-major view; gate/up uses 32-column strips;
- forward FC2: physical `(H, I)` row-major, equivalent to the V1
  `(I, H)` K-major view;
- backward W2 dgrad: physical `(I, H)` row-major;
- backward W1 dgrad: physical `(H, 2I)` row-major, with gate/up 32-column
  strips along GEMM K.

The scale pointee contracts remain the matching atom-blocked V1 layouts
listed above. Each `payload_ptrs` and `scale_ptrs` value must be a contiguous CUDA
`torch.int64[E_local]` table. The table address must be 8-byte aligned; every
non-null pointee must be 256-byte aligned. MoeEP passes all four tables
directly to the kernel and never allocates, copies, or refreshes them.

Each pointee must describe one expert with the corresponding physical layout,
dtype, and complete storage extent. Pointer tables
cannot encode or prove those properties or pointee non-aliasing, so they are a
caller contract.
`validation_mode="strict"` checks non-null/alignment values eagerly and
requires the same live table object, address, and PyTorch version to be
validated before capture. Inference-mode tables without version counters are
revalidated on every eager call and rejected during strict capture.
`"trusted"` performs only structural table checks.

The caller owns the tables and all pointees through kernel completion. For CUDA Graph,
their addresses and allocations must remain live until the graph executable is
destroyed. In-place optimizer updates are allowed. Any weight reallocation
requires updating the table outside capture and warming up/capturing again;
do not modify a table concurrently with a launch on another stream.

`autotune_training` accepts the same `native_weight_storage_mode` argument.
Its mode is part of the winning specialization; the subsequent
`prepare_training` call must use the same mode. Every EP rank must select the
same mode.

### Forward

```python
from cudnn import MoeEpTrainingForwardOutputs

y = op.training_forward(
    lane,
    activation,
    topk_idx,
    topk_weights,
    weights=native_fw,
    out=MoeEpTrainingForwardOutputs(
        output=y_out,
        fc1_preact=fc1_preact,
        fc1_a=fc1_a,
        fc1_sfa=fc1_sfa,
        valid_route_counts=valid_route_counts,
        expert_offsets=expert_offsets,
    ),
)
```

`activation` may be contiguous BF16/FP32 or an axis-1 MXFP8
`BlockScaledTensor`. To avoid input staging entirely, quantize directly into
the lane's `forward_input` and `forward_input_scale` views and build the MXFP8
input from `forward_input[:T]` and
`forward_input_scale[:T, :ceil_div(hidden_size, 32)]`. That logical scale view
has unit column stride and may keep the lane buffer's padded row stride;
training accepts it without copying. Routing metadata is still staged
privately.

The training forward always emits the FC1 preactivation, so `fc1_preact` is
required: the caller provides its destination and retains it through the
matching backward. `fc1_a`, `fc1_sfa`, `valid_route_counts`, and
`expert_offsets` are also required caller-owned destinations after
`prepare_training()`.

`output` is required and must be the lane's symmetric `output` buffer, which
the forward kernel writes directly. The return is a logical `(T, H)` view of
it.

### Backward and WGrad

```python
from cudnn import MoeEpTrainingBackwardOutputs

dx, dprob, operands = op.training_backward(
    lane,
    grad_output,
    topk_idx,
    topk_weights,
    weights=native_bw,
    fc1_preact=fc1_preact,
    fc1_a=fc1_a,
    fc1_sfa=fc1_sfa,
    valid_route_counts=valid_route_counts,
    expert_offsets=expert_offsets,
    out=MoeEpTrainingBackwardOutputs(
        grad_activation=dx_out,
        dprob=dprob_out,
        fc1_b=fc1_b,
        fc1_sfb=fc1_sfb,
        fc2_a=fc2_a,
        fc2_sfa=fc2_sfa,
        fc2_b=fc2_b,
        fc2_sfb=fc2_sfb,
    ),
)
```

`grad_output` has the same BF16/FP32/MXFP8 input choices as forward. To avoid
staging, quantize it directly into the lane's `backward_input` and
`backward_input_scale` buffers and build the same logical prefix views
described for `forward_input`. `fc1_preact` and the four forward WGrad values
are required and passed explicitly because cuDNN does not retain the forward
output bundle.

`grad_activation` and `dprob` are required destinations, and both must be the
corresponding buffers from `training_symmetric_buffers(lane)`; the backward
kernel writes them directly. `grad_activation` is BF16 and `dprob` is FP32.

All six backward WGrad fields are required. `operands` is always a
`MoeEpTrainingWgradOperands` containing non-owning views of the exact caller
buffers. The WGrad result is a fixed-capacity operand bundle, not dense
optimizer-ready weight gradients. Here `K_pool` is the fixed routed-token pool
capacity, not the model's top-k value:

- `fc1_b` remains gate/up-interleaved with shape `(K_pool, 2I)` and stride
  `(2I, 1)`;
- `fc1_a` is contiguous `(K_pool, H)` and `fc2_a` is contiguous
  `(K_pool, I)`, so both activations are directly usable in `dY.T @ X`;
- all four scale tensors are written in the final grouped-WGrad 128x4
  interleaved layout;
- no public compact scale, deinterleave copy, physical transpose, slot export,
  or scale-expansion kernel is used.

The fixed tensor extent is storage capacity, not a promise that every row is
defined. For local expert `e`, let `begin` be zero when `e == 0` and
`expert_offsets[e - 1]` otherwise. Only
`[begin, begin + valid_route_counts[e])` is valid. The remainder of that
expert's padded segment, and the capacity tail after `expert_offsets[-1]`,
hold unspecified data and scale values. Consumers must use both metadata
tensors: `expert_offsets` locates each physical segment and
`valid_route_counts` limits the rows read from it.

The existing grouped WGrad API derives its GEMM K extent from adjacent
`expert_offsets` and therefore reads complete padded segments. Under this
validity contract MoeEP operands are **not** guaranteed to be directly
consumable by that API.

### Ownership and lifetime

- The caller owns all native weights, saved forward state, WGrad operands, and
  optional pack staging.
- cuDNN borrows caller-allocated tensors for one call and does not cache their
  Python objects or pointers. The per-lane buffers from
  `training_symmetric_buffers` stay cuDNN-owned for the lane's lifetime.
- The caller must provide `fc1_preact` to forward and keep it live through the
  matching backward; cuDNN has no private preactivation fallback or workspace
  alias.
- Forward WGrad outputs, segment metadata, and backward WGrad outputs remain
  live until the independent grouped WGrad consumer completes.
- cuDNN owns per-lane local and NVSHMEM symmetric storage; only the documented
  input and final-output views are exposed to the caller.
- One lane may be active on only one stream at a time.
- All EP ranks must submit distributed forward/backward calls in identical
  order.
- Unordered concurrent replay of distributed MoeEP graphs on independent CUDA
  streams is unsupported and must not be used. Multiple streams must be
  serialized into the same total device-execution order on every EP rank, by
  stream FIFO or explicit CUDA event dependencies; matching host submission
  order alone is insufficient.
- The caller owns forward/backward weight-version consistency.
- `MoeEp.close()` releases only private runtime resources and never clears or
  frees caller memory.

### Overflow

Overflow is private per-launch state. Each forward and backward applies the
`drop_on_overflow` policy before returning; there is no public overflow tensor
and no separate finalize step.

`drop_on_overflow=True` discards routes beyond the pool capacity. The default
`False` instead fires a device-side assertion, which surfaces asynchronously
and requires `torch._assert_async`. EP2+ reduces the overflow flag with a
scalar MAX across the group before applying the policy, so every rank reaches
the same decision.

Results are numerically usable only when no overflow occurs. A launch that
overflows may raise or drop work according to `drop_on_overflow`, but its
returned values fall outside the supported correctness contract.

Private pre-reduction data and scale planes persist across launches and are
not cleared. Under dense, non-overflow routing every active
`(token, top-k slot)` is completely overwritten before reduction. Rows in
`[T, max_tokens_per_rank)` stay unspecified and must not be returned or
consumed.

### CUDA Graph capture

1. Collectively call `prepare_training`.
2. Allocate every capture binding from the returned requirements.
3. Materialize or provide native weights at stable addresses.
4. Run ordinary forward/backward warmups for every captured specialization.
5. Capture calls using every caller-owned destination returned by
   `prepare_training()`, including primary outputs, saved forward state, and
   forward/backward WGrad tensors.
6. Keep all captured input, output, saved-state, staging, and native-pack
   addresses stable until every referencing graph executable is destroyed.

The local token count `T` is fixed by the input shapes used during capture.
Every replay of that graph must use the same `T`, shapes, and addresses.
Tensor contents, routing, `valid_route_counts`, and `expert_offsets` may
change at those fixed addresses on each replay, but all routing IDs must stay
valid and dense because replay performs no value check. Eager invocations may
use a different `T` and replace addresses between calls, subject to the
configured capacity.

Private lane resources cannot grow during replay. Capacity changes require a
new operator preparation; caller-address changes require recapture.

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
the validated support surface. This acceptance requires one identical total
device-execution order across all EP ranks. Unordered concurrent replay of
distributed MoeEP graphs on independent CUDA streams is unsupported and must
not be used. Serialize multiple streams with stream FIFO or explicit CUDA
event dependencies; identical host submission order does not establish device
order.

## Data formats

`output_format` and `combine_format` take a `MoeFormat` enum value or its
string name: `"bf16"`, `"mxfp8"`, or `"nvfp4"`. Anywhere this page says a
tensor may be plain or block-scaled, the accepted Python type is
`MoeTensor`, which is `Union[torch.Tensor, BlockScaledTensor]`.

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
- `topk_idx`: `(T, K)`, Int32 or Int64, with every value in `[0, E)`;
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
- required forward `output`, the lane's symmetric buffer: `(T, H)`, BF16;
- required caller-owned `fc1_preact`, written by training forward and retained
  through the matching backward;
- required `grad_activation`, the lane's symmetric buffer: `(T, H)` view of a
  capacity buffer, BF16;
- required `dprob`, the lane's symmetric buffer: source-order `(T, K)`, FP32;
- required caller-owned WGrad saved state and a fixed-capacity
  `MoeEpTrainingWgradOperands` bundle.

All dynamic training tensors must reside on one device and satisfy
`T <= max_tokens_per_rank`.

The backend may use `-1` internally to mask private capacity rows after `T`;
that sentinel is not part of the public input. Persistent pre-reduction data
and scale planes are not cleared between launches. Correctness relies on every
non-overflow active `(token, k)` route completely overwriting its plane before
top-k reduction. Physical rows in `[T, max_tokens_per_rank)` are unspecified
and must not be returned or consumed.

## Expert-parallel communication

EP2+ execution requires:

- an initialized NCCL process group;
- `nvshmem4py` and usable NVSHMEM libraries;
- direct peer access among every pair of participating ranks; and
- consistent rank ordering, buffer schemas, tuning, lane selection, and launch
  ordering across the group.

`max_recv_size_per_rank` is the physical receive-pool capacity in token rows,
including per-expert padding.

When omitted it defaults to the worst case, in which every active expert owns
its own padded segment:

```text
raw      = ep_size * max_tokens_per_rank * top_k
active   = min(experts_per_rank, raw)
capacity = (active + (raw - active) // token_padding_size) * token_padding_size
```

`raw` above is the unbounded raw-route limit; the padded capacity is
deliberately not the same as rounding that route count once.

An explicit capacity `P` must satisfy `P % 128 == 0`. The frontend reverse-maps
`P` to the largest logical route limit whose worst-case padded capacity is
exactly `P`, and rejects any `P` that cannot be represented exactly. Because
that logical limit is conservative, a favorable expert distribution that would
in fact fit in `P` can still be refused; this early overflow is what keeps any
distribution the kernel accepts from exceeding the prescribed pool. On
overflow the launch may raise or drop work according to `drop_on_overflow`,
and its numerical outputs are not guaranteed usable.
