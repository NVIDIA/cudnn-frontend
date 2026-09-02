# Mixture of Experts with Expert Parallelism

`cudnn.moe_ep` provides Rubin SM107 fused SwiGLU MoE execution with optional
expert parallelism. Inference and training share one `MoeEp` object but use
separate call surfaces:

- `MoeEp.__call__` and `warmup` for inference;
- `prepare_training`, `training_forward`, and `training_backward` for training.

Training is stateless with respect to caller tensors. The operator retains
compiled kernels, runtime state, and private per-lane NVSHMEM scratch, but it
does not retain weights, forward state, output buffers, or fallback weight
staging.

## Installation

```bash
pip install "nvidia-cudnn-frontend[cutedsl,comm]" torch torch-c-dlpack-ext
```

Rubin MegaMoE requires `nvidia-cutlass-dsl>=4.8.0`. EP2+ also requires an
initialized NCCL process group and an NVSHMEM topology in which all
participating ranks are directly peer-addressable.

## Constructing the operator

```python
from cudnn import MoeEp

op = MoeEp(
    num_experts=num_experts,
    hidden_size=hidden,
    intermediate_size=intermediate,
    top_k=top_k,
    ep_group=ep_group,
    max_tokens_per_rank=max_tokens,
    max_recv_size_per_rank=recv_capacity,
    drop_on_overflow=False,
    output_format="bf16",
    combine_format="bf16",
    apply_topk_in_fc1=True,
    weight_interleave_size=32,
)
```

Native training requires `weight_interleave_size=32`. FC1 payloads then use
alternating 32-element gate/up strips.

## Explicit sweep autotuning

`MoeEp.autotune` measures inference forward. `MoeEp.autotune_training`
measures one training forward immediately followed by its matching backward:

```python
result = op.autotune(
    activation, fc1_weight, fc2_weight, topk_idx, topk_weights,
    candidates=candidates,
    warmup_iters=3,
    timed_iters=10,
)

training_result = op.autotune_training(
    activation, grad_output, topk_idx, topk_weights,
    forward_weights=native_fw,
    backward_weights=native_bw,
    candidates=candidates,
)
```

Both calls are collective over `ep_group`, must use the same ordered candidate
list on every rank, and must run outside CUDA Graph capture.
`autotune_training` must run before `prepare_training`. It accepts only native
weights; source packing and allocation are intentionally outside its measured
region.

The current `MoeEpTuningConfig` is prepended as a baseline, duplicate values are
removed, and the normalized list is limited to 32 candidates. Autotuning keeps
`reduce_topk_in_kernel` fixed because that flag changes where top-k reduction
is performed. Each timed iteration is reduced with rank MAX and the candidate
score is the median of those slow-rank samples. Equal scores select the earlier
candidate. `MoeEpAutotuneResult` reports `winner`, per-candidate `latency_ms`
and `samples_ms`, and `evaluated_candidates`.

The sweep is fail-fast. Any validation, allocation, compile, launch, timing,
synchronization, or teardown error ends the whole sweep. An error after
runtime/collective entry poisons the operator, and later execution is rejected;
close it and create a new instance. Compiled candidate kernels remain in the
process JIT cache. The production sweep does not compare candidate outputs at
runtime; supported candidates are covered by the separate correctness suite.

Autotuning commits one active winner per instance. A later inference or
training sweep replaces it. Existing CUDA Graph executables are invalid after
the winner changes. Use these sequences:

- inference: `autotune` → eager winner launch (performed by `autotune`) →
  capture;
- training: `autotune_training` → `prepare_training` → allocate outputs →
  eager forward/backward → rank synchronization → capture.

## Stateless training preparation

Preparation is collective over `ep_group` and must run outside CUDA Graph
capture:

```python
requirements = op.prepare_training(
    lane_count=1,
    device=None,  # current CUDA device; pass an explicit device for multi-GPU hosts
)
lane = op.training_lanes[0]
```

`prepare_training` does not accept or bind weights. It returns a plain mapping
whose values are:

```text
(shape, stride, dtype, alignment_bytes)
```

The mapping contains `output`, `fc1_preact`, `fc1_a`, `fc1_sfa`,
`valid_route_counts`, `expert_offsets`, `grad_activation`, `dprob`, `fc1_b`,
`fc1_sfb`, `fc2_a`, `fc2_sfa`, `fc2_b`, and `fc2_sfb`. TE allocates these
buffers and passes them to each invocation. cuDNN validates exact shape,
stride, dtype, alignment, device, and non-aliasing before launch.

`device=None` binds the current CUDA device. An explicit CUDA device takes
precedence. Every later training tensor must use the bound device.

## Native weight ABI

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
payload or scale is written to the supplied staging bundle. These fallback
packers consume logical gate-then-up `MoeEpForwardWeights` /
`MoeEpBackwardWeights` with compact axis-1 scales; already interleaved,
blocked producers should construct the native packs directly instead of
packing them again.

## Forward

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
`BlockScaledTensor`. MXFP8 input bypasses the BF16-to-MXFP8 quantization
stager. Routing still copies data into private symmetric memory because remote
ranks address that memory directly.

`fc1_preact` is required because the training forward kernel always runs with
`generate_c=True`; TE must provide its destination and retain it through the
matching backward. `fc1_a`, `fc1_sfa`, `valid_route_counts`, and
`expert_offsets` are also required caller-owned destinations after
`prepare_training()`.

`output` is required. The return is a logical `(T, H)` view of that
caller-owned capacity buffer.

## Backward and WGrad

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

`grad_output` has the same BF16/FP32/MXFP8 input choices as forward.
`fc1_preact` and the four forward WGrad values are required and passed
explicitly because cuDNN does not retain the forward output bundle.

`grad_activation` and `dprob` are required caller-owned destinations.

All six backward WGrad fields are required. `operands` is always a
`MoeEpTrainingWgradOperands` containing non-owning views of the exact caller
buffers.

The producer-native ABI is directly consumable by the separately invoked
grouped WGrad kernel. Here `K_pool` is the fixed routed-token pool capacity,
not the model's top-k value:

- `fc1_b` remains gate/up-interleaved with shape `(K_pool, 2I)` and stride
  `(2I, 1)`;
- `fc1_a` and `fc2_a` use the advertised transpose-view layouts;
- all four scale tensors are written in the final grouped-WGrad 128x4
  interleaved layout;
- no public compact scale, deinterleave copy, physical transpose, slot export,
  or scale-expansion kernel is used.

## Ownership and lifetime

- TE owns all native weights, output bundles, saved forward state, WGrad
  operands, and optional pack staging.
- cuDNN borrows these tensors for one call and does not cache their Python
  objects or pointers.
- TE must provide `fc1_preact` to forward and keep it live through the matching
  backward; cuDNN has no private preactivation fallback or workspace alias.
- Forward WGrad outputs, segment metadata, and backward WGrad outputs remain
  live until the independent grouped WGrad consumer completes.
- cuDNN owns private per-lane local and NVSHMEM symmetric scratch.
- One lane may be active on only one stream at a time.
- All EP ranks must submit distributed forward/backward calls in identical
  order.
- The caller owns forward/backward weight-version consistency.
- `MoeEp.close()` releases only private runtime resources and never clears or
  frees caller memory.

## Overflow

Overflow is private per-launch state. Each forward and backward applies the
configured policy before returning; there is no public overflow tensor or
`finalize_overflow` method. EP2+ retains the scalar MAX reduction required to
make the policy rank-consistent.

## CUDA Graph capture

1. Collectively call `prepare_training`.
2. Allocate every capture binding from the returned requirements.
3. Materialize or provide native weights at stable addresses.
4. Run ordinary forward/backward warmups for every captured specialization.
5. Capture calls using every caller-owned destination returned by
   `prepare_training()`, including primary outputs, saved forward state, and
   forward/backward WGrad tensors.
6. Keep all captured input, output, saved-state, staging, and native-pack
   addresses stable until every referencing graph executable is destroyed.

Dynamic contents may change at fixed addresses. Eager invocations may replace
addresses between calls.

## Breaking migration

Removed:

- `MoeEpTrainingResources`
- `MoeEpTrainingSlot`
- `MoeEpTrainingWeights`
- `prepare_training_resources`
- `refresh_weights`
- `finalize_overflow`

The old resource-owned forward/backward state is replaced by explicit
per-invocation native weight packs and caller-owned output buffers. No
compatibility shim is retained.
