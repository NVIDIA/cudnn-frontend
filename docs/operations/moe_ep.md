# Mixture of Experts with Expert Parallelism

The MoeEP operation fuses token routing, expert SwiGLU computation, and
expert-parallel communication. Global experts are sharded contiguously across
the ranks of an expert-parallel process group.

Inference and training share one `MoeEp` object but use separate call
surfaces:

- `MoeEp.__call__` and `warmup` for inference;
- `prepare_training`, `training_forward`, and `training_backward` for training.

Training is stateless with respect to caller tensors. The operator retains
compiled kernels, runtime state, and one instance-owned NVSHMEM workspace, but it does not
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

The operation is exposed by the frontend-only `cudnn.MoeEp` object API. It
accepts one immutable, nested configuration:

```python
from cudnn import (
    MoeEp,
    MoeEpConfig,
    MoeEpDataPathConfig,
    MoeEpFc1WeightLayout,
    MoeEpModelConfig,
    MoeEpNativeWeightStorageMode,
    MoeEpParallelConfig,
    MoeEpTuningConfig,
    MoeFormat,
)

config = MoeEpConfig(
    model=MoeEpModelConfig(
        num_experts=E,
        hidden_size=H,
        intermediate_size=I,
        top_k=K,
    ),
    parallel=MoeEpParallelConfig(
        ep_group=ep_group,
        max_tokens_per_rank=max_tokens,
        physical_recv_pool_rows=None,
        drop_on_overflow=False,
        token_padding_size=128,
        sf_padding_size=128,
    ),
    data_path=MoeEpDataPathConfig(
        output_format=MoeFormat.BF16,
        combine_format=MoeFormat.BF16,
        apply_topk_in_fc1=True,
        fc1_weight_layout=MoeEpFc1WeightLayout.GATE_THEN_UP,
        gate_up_clamp=None,
    ),
    inference_tuning=MoeEpTuningConfig(),
    training_forward_tuning=MoeEpTuningConfig(),
    training_backward_tuning=MoeEpTuningConfig(),
    training_weight_storage_mode=MoeEpNativeWeightStorageMode.CONTIGUOUS,
    validation_mode="strict",
)
op = MoeEp(config)
```

The component configs and tuning configs are frozen. Corresponding `MoeEp`
properties are read-only; construct a new config/operator for manual
reconfiguration. There is no scalar-kwargs constructor, `from_config`,
property setter, or general `reconfigure` method. `dataclasses.replace` can
build a modified config value before constructing a new operator.

For migration from the experimental scalar constructor:

- model geometry moves to `MoeEpModelConfig`;
- `ep_group`, capacity, overflow policy, and padding move to
  `MoeEpParallelConfig`;
- output/combine formats, top-k placement, FC1 layout, and clamp move to
  `MoeEpDataPathConfig`;
- the old `forward_tuning` is split into `inference_tuning` and
  `training_forward_tuning`, while `backward_tuning` becomes
  `training_backward_tuning`;
- `weight_interleave_size=None` maps to `GATE_THEN_UP`, and
  `weight_interleave_size=32` maps to `GATE_UP_INTERLEAVED_32`.

The old constructor accepted negative `gate_up_clamp` values and used their
absolute magnitude. The config-only API rejects negatives; callers must pass
the intended non-negative magnitude explicitly.

The three phase tuning fields are independent. Training backward accepts only
`token_back_mode="epi_warps"` with
`reduce_topk_in_kernel=False`. `dgrad_optimization` is a backward-only tuning
choice; inference and training forward require its default value,
`"baseline"`.

### Dgrad optimization profiles

`training_backward_tuning.dgrad_optimization` selects one of three upstream
dgrad configurations:

- `"baseline"` preserves the grouped schedule and all existing defaults;
- `"rolling"` selects the upstream rolling/phase-interleaved schedule;
- `"ds3_ep4_pattern"` selects the upstream preset named for its original DS3
  EP4 workload.

For example:

```python
from dataclasses import replace

from cudnn import MoeEpTuningConfig

config = replace(
    config,
    training_backward_tuning=MoeEpTuningConfig(
        dgrad_optimization="rolling",
    ),
)
```

The name identifies the preset's origin, not an EP4-only runtime restriction.
Additional topologies and problem shapes are supported, subject to the generic
kernel resource, layout, output, full-route-capacity, and MXFP8 constraints.
Both contiguous and discrete native backward weights are supported.

DS3 owns its scheduling and transport preset fields. It requires
`token_back_mode="epi_warps"`, `token_in_flag_batch=1`, `group_hint=None`, and
`reduce_topk_in_kernel=False`. The default `epi_flag_batch=(1, 1)` is treated
as handing that field to the preset and is canonicalized to `(4, 2)`;
explicit `(4, 2)` is equivalent. Other values are rejected instead of
creating a custom DS3 variant.

`MoeEpFc1WeightLayout.GATE_THEN_UP` denotes conventional source ordering.
`GATE_UP_INTERLEAVED_32` denotes alternating 32-row gate/up strips. Inference
supports both layouts and normalizes them to the same kernel-native layout.
Training preparation, instance packing, native weights, and standalone pack
functions support only `GATE_UP_INTERLEAVED_32`.

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

result = op.autotune_inference(
    activation, fc1_weight, fc2_weight, topk_idx, topk_weights,
    candidates=[
        MoeEpTuningConfig(token_in_flag_batch=2),
        MoeEpTuningConfig(group_hint=256),
    ],
)
```

Training forward and backward are tuned independently:

```python
forward_result = op.autotune_training_forward(
    activation, topk_idx, topk_weights,
    forward_weights=native_fw,
    candidates=candidates,
)
backward_result = op.autotune_training_backward(
    activation, grad_output, topk_idx, topk_weights,
    forward_weights=native_fw,
    backward_weights=native_bw,
    candidates=candidates,
)
```

All three methods are collective over `ep_group`, require the same ordered
candidate list on every rank, and run outside CUDA Graph capture. Training
autotune must run before `prepare_training` and accepts only native weights.
Forward autotune times only training forward. Backward autotune runs the
matching forward outside timing to create saved tensors, then times only
backward.

The current `MoeEpTuningConfig` is prepended as a baseline, duplicate values
are removed, and the normalized list is limited to 32 candidates. Autotuning
keeps `reduce_topk_in_kernel` fixed because that flag changes where top-k
reduction is performed. Each timed iteration is reduced with rank MAX and the
candidate score is the median of those slow-rank samples. Equal scores select
the earlier candidate. `MoeEpAutotuneResult` reports `winner`, per-candidate
`latency_ms` and `samples_ms` as `MoeEpAutotuneCandidateResult` entries, and
`evaluated_candidates`.

Backward candidates may select different dgrad profiles. Candidate failure
remains fail-fast: if a selected profile violates a generic kernel resource,
layout, or capacity constraint, the sweep aborts instead of silently skipping
that candidate. The reported winner contains the canonical public tuning;
`group_hint=None` still resolves to the resident cluster count of the runtime
device.

The sweep is fail-fast. Any validation, allocation, compile, launch, timing,
synchronization, or teardown error ends the whole sweep. An existing active
backend remains live while temporary candidates are searched, so candidate or
winner-validation failure leaves the operator's config/backend state
unchanged. This can temporarily require memory for the active backend and one
candidate backend. Failure while replacing/closing the active backend after a
winner has passed final validation is unrecoverable and poisons the operator.
Compiled candidate kernels remain in the process JIT cache. The production
sweep does not compare candidate outputs at runtime; supported candidates are
covered by the separate correctness suite.

Each method applies its own winner before returning and leaves the other two
tuning fields unchanged. Inference retains its warmed winner backend. Training
autotune closes candidate backends and leaves normal training preparation to
the caller. The compilation cache can avoid recompilation of the winner, but
does not remove prepare, workspace allocation, first launch, or warmup costs.
Both training methods and `prepare_training` use the immutable
`config.training_weight_storage_mode`; they do not accept a per-call storage
mode. Existing CUDA Graph executables are invalid after any winner change.

### Stateless training preparation

Preparation allocates one set of private instance resources. It is collective over
`ep_group` and must run outside CUDA Graph capture:

```python
requirements = op.prepare_training(
    device=None,  # current CUDA device; pass an explicit device for multi-GPU hosts
)
symmetric = op.training_symmetric_buffers()
```

`prepare_training` does not accept or bind weights. It returns a plain mapping
whose values are:

```text
(shape, stride, dtype, alignment_bytes)
```

The requirements mapping contains `output`, `fc1_preact`, `fc1_a`, `fc1_sfa`,
`valid_route_counts`, `expert_offsets`, `grad_activation`, `dprob`, `fc1_b`,
`fc1_sfb`, `fc2_a`, `fc2_sfa`, `fc2_b`, and `fc2_sfb`.

Buffers come from two places. `training_symmetric_buffers()` returns the
cuDNN-allocated `forward_input`, `forward_input_scale`, `backward_input`,
`backward_input_scale`, `output`, `grad_activation`, and `dprob` tensors for
the instance; quantize directly into the input pairs and bind the returned output
tensors in the public output bundles. The caller allocates the remaining
entries of the requirements mapping. cuDNN validates exact shape, stride,
dtype, alignment, device, and non-aliasing before launch.

`device=None` binds the current CUDA device. An explicit CUDA device takes
precedence. Every later training tensor must use the bound device.

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
    MoeEpNativeWeightStorageMode,
)

# Construct this op with
# training_weight_storage_mode=MoeEpNativeWeightStorageMode.DISCRETE.
requirements = op.prepare_training(device=device)

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

Set `MoeEpConfig.training_weight_storage_mode` before constructing the
operator. Both training autotune methods and `prepare_training` consume that
single value, and every EP rank must configure the same mode.

### Forward

```python
from cudnn import MoeEpTrainingForwardOutputs

y = op.training_forward(
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
the instance's `forward_input` and `forward_input_scale` views and build the MXFP8
input from `forward_input[:T]` and
`forward_input_scale[:T, :ceil_div(hidden_size, 32)]`. That logical scale view
has unit column stride and may keep the instance buffer's padded row stride;
training accepts it without copying. Routing metadata is still staged
privately.

The training forward always emits the FC1 preactivation, so `fc1_preact` is
required: the caller provides its destination and retains it through the
matching backward. `fc1_a`, `fc1_sfa`, `valid_route_counts`, and
`expert_offsets` are also required caller-owned destinations after
`prepare_training()`.

`output` is required and must be the instance's symmetric `output` buffer, which
the forward kernel writes directly. The return is a logical `(T, H)` view of
it.

### Backward and WGrad

```python
from cudnn import MoeEpTrainingBackwardOutputs

dx, dprob, operands = op.training_backward(
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
staging, quantize it directly into the instance's `backward_input` and
`backward_input_scale` buffers and build the same logical prefix views
described for `forward_input`. `fc1_preact` and the four forward WGrad values
are required and passed explicitly because cuDNN does not retain the forward
output bundle.

`grad_activation` and `dprob` are required destinations, and both must be the
corresponding buffers from `training_symmetric_buffers()`; the backward
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
  Python objects or pointers. Buffers from `training_symmetric_buffers` stay
  cuDNN-owned for the `MoeEp` instance's lifetime.
- The caller must provide `fc1_preact` to forward and keep it live through the
  matching backward; cuDNN has no private preactivation fallback or workspace
  alias.
- Forward WGrad outputs, segment metadata, and backward WGrad outputs remain
  live until the independent grouped WGrad consumer completes.
- cuDNN owns one set of local and NVSHMEM symmetric instance storage; only the documented
  input and final-output views are exposed to the caller.
- One `MoeEp` instance does not support overlapping GPU work. Prepare, warmup,
  eager calls, graph capture, and graph replay for that instance must use one
  CUDA stream and execute sequentially.
- The implementation does not bind or validate the stream. Python locking
  serializes host submission only, and graph replay bypasses Python; violating
  this contract may silently corrupt shared state or deadlock.
- Stable addresses do not imply result retention. A later call may overwrite
  the instance-owned symmetric output, grad-activation, dprob, routing, and
  finalizer storage. Consume or copy results before the next call.
- Applications requiring parallel resource isolation must create multiple
  `MoeEp` instances. The caller must still establish the same instance/graph
  launch order and required CUDA-event dependencies on every EP rank.
- All EP ranks must submit distributed forward/backward calls in identical
  order.
- Concurrent replay of multiple graphs from one instance is unsupported,
  including replay on independent CUDA streams. For multiple instances on
  different streams, the caller must establish the same total device-execution
  order on every EP rank with explicit CUDA event dependencies; matching host
  submission order alone is insufficient.
- The caller owns forward/backward weight-version consistency.
- `MoeEp.close()` releases only private runtime resources and never clears or
  frees caller memory.

### Overflow

Overflow is private per-launch state. Each forward and backward applies the
`drop_on_overflow` policy before returning; there is no public overflow tensor
and no separate finalize step.

The transport kernel always discards routes beyond the receiving rank's pool
capacity and completes its ready/tail protocol. Each rank derives the same
group-wide overflow bit from the per-expert route totals already exchanged by
the router; no additional scalar collective is launched for this policy.

`drop_on_overflow=True` returns after that safe truncation. The default
`False` instead applies `torch._assert_async` after the communication protocol
has completed, so every EP rank reaches the same fatal decision without
leaving peers in a ready-counter wait. A real error-mode overflow can leave
every rank's CUDA context in a sticky error state; restart the whole EP worker
group rather than continuing or restarting only the receiving rank.

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

If one instance is used to capture multiple training graphs, capture them
sequentially on the same CUDA stream and replay them sequentially on that same
stream. Concurrent replay is unsupported. All graphs share the instance-owned
symmetric buffers, so replaying a later graph may overwrite the earlier
graph's output, grad-activation, and dprob values.

The local token count `T` is fixed by the input shapes used during capture.
Every replay of that graph must use the same `T`, shapes, and addresses.
Tensor contents, routing, `valid_route_counts`, and `expert_offsets` may
change at those fixed addresses on each replay, but all routing IDs must stay
valid and dense because replay performs no value check. Eager invocations may
use a different `T` and replace addresses between calls, subject to the
configured capacity.

Private instance resources cannot grow during replay. Capacity changes require a
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
graphs from one instance is unsupported and must not be used. When separate
instances use separate streams, establish identical CUDA-event ordering on all
EP ranks; identical host submission order does not establish device order.

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
- required forward `output`, the instance's symmetric buffer: `(T, H)`, BF16;
- required caller-owned `fc1_preact`, written by training forward and retained
  through the matching backward;
- required `grad_activation`, the instance's symmetric buffer: `(T, H)` view of a
  capacity buffer, BF16;
- required `dprob`, the instance's symmetric buffer: source-order `(T, K)`, FP32;
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
- consistent rank ordering, buffer schemas, tuning, instance selection, and launch
  ordering across the group.

`physical_recv_pool_rows` is the exact physical receive-pool capacity in token rows,
including per-expert padding.

When omitted it resolves once to a 128-row-aligned canonical capacity covering
the worst-case inference and training padding policies. For one padding block,
the required rows are:

```text
raw      = ep_size * max_tokens_per_rank * top_k
active   = min(experts_per_rank, raw)
capacity = (active + (raw - active) // token_padding_size) * token_padding_size
```

`raw` above is the unbounded raw-route limit; the padded capacity is
deliberately not the same as rounding that route count once.

An explicit capacity `P` must satisfy `P % 128 == 0`. For each operation
padding policy, the frontend resolves the largest logical route limit `L`
bounded by `raw` whose worst-case padded requirement is at most `P`. The
kernel receives `L` and exact `P` independently, so overprovisioned tail rows
remain part of the strict tensor/workspace ABI without becoming work items.
Because `L` is conservative, a favorable expert distribution that would in
fact fit in `P` can still be refused. On
overflow the launch may raise or drop work according to `drop_on_overflow`,
and its numerical outputs are not guaranteed usable.

For the qualified DS3 workload, the backward kernel requires the exact
logical route limit

```text
logical = ep_size * max_tokens_per_rank * top_k
        = 4 * 4096 * 8
        = 131072
```

Per-expert padding expands that limit to a physical receive pool of `131968`
rows. Therefore public `physical_recv_pool_rows` must be omitted (automatic)
or set to a 128-row-aligned value of at least `131968`; `131072` is the
kernel's logical limit and is not a valid public physical-pool spelling for
DS3. Larger physical pools are accepted as exact ABI capacity. Training
forward and backward both use logical `L=131072`.
