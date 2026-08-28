# Mixture of Experts with Expert Parallelism

`cudnn.moe_ep` provides a fused SwiGLU MoE implementation for Rubin SM107.
Experts are sharded contiguously across an optional expert-parallel process
group.

## Supported configuration

- CUDA execution on Rubin SM107 (compute capability 10.7)
- fused SwiGLU with contiguous expert sharding across `ep_group`
- BF16 output, including when the combine path uses MXFP8
- BF16 or MXFP8 combine
- plain BF16, FP16, or FP32 inference operands, or MXFP8
  `BlockScaledTensor` operands
- `apply_topk_in_fc1=True`
- `hidden_size` divisible by 128
- `intermediate_size` divisible by 256
- `top_k <= min(32, num_experts)`
- `num_experts` divisible by the EP group size
- explicit positive `max_tokens_per_rank`

`output_format` is currently executable only as `"bf16"`. NVFP4 is represented
by the public format types but NVFP4 operands, combine, and output are not
executable by the current MegaMoE backend.

The fixed-resource CUDA Graph path has hardware acceptance through EP32 within
one direct-P2P MNNVL peer-access domain. The Python capability layer does not
impose an EP-size ceiling; this statement describes validated hardware scope,
not support for cross-MNNVL execution.

## Installation

Install the dedicated optional dependencies:

```bash
pip install nvidia-cudnn-frontend[moe_ep]
```

The extra supplies the CuTeDSL and NVSHMEM Python dependencies. PyTorch with
CUDA support is also required. EP2+ additionally requires an initialized NCCL
process group and a usable NVSHMEM peer topology.

## Public API and constructor

The public surface exports `MoeEp`, `MoeEpTrainingWeights`,
`MoeEpTrainingResources`, `MoeEpTrainingSlot`, `MoeEpExecutionLane`,
`MoeEpTrainingWgradOperands`, `BlockScaledTensor`, `MoeFormat`, and
`MoeEpTuningConfig`.

The `MoeEp` constructor accepts:

| Parameter | Current contract |
| --- | --- |
| `num_experts` | Positive global expert count; divisible by EP size |
| `hidden_size` | Positive and divisible by 128 |
| `intermediate_size` | Positive and divisible by 256 |
| `top_k` | Positive and no larger than 32 or `num_experts` |
| `ep_group` | Optional initialized `torch.distributed.ProcessGroup`; `None` selects EP1 |
| `max_tokens_per_rank` | Required by the executable backend and must be positive |
| `max_recv_size_per_rank` | Optional positive receive-pool capacity |
| `drop_on_overflow` | `False` by default; selects fatal-assert versus reporting/drop policy |
| `output_format` | `"bf16"` only for current execution |
| `combine_format` | `"bf16"` or `"mxfp8"` |
| `apply_topk_in_fc1` | Must be `True` |
| `gate_up_clamp` | Optional finite clamp magnitude |
| `token_padding_size` | Positive; training fixed resources use 128 internally |
| `sf_padding_size` | Positive multiple of 128; training fixed resources use 128 internally |
| `tuning` | Optional `MoeEpTuningConfig`; must match on every EP rank |

When `max_recv_size_per_rank` is omitted, the backend allocates for the
worst-case receive count:

```text
ep_size * max_tokens_per_rank * top_k
```

An explicit value is capped at that same worst-case count.

## Breaking training API migration

This release removes the legacy dynamic compact training API:

- `MoeEp.backward(...)`
- constructor arguments `generate_c` and `backward_wgrad_mode`
- forward returns containing compact `fc1_c` and `route_metadata`
- `MoeEpWgradForwardStash` and `MoeEpWgradOperands`

Old:

```python
output, fc1_c, route_metadata = op(
    activation, w1, w2, topk_idx, topk_weights
)
dx, dprob = op.backward(
    grad_output, w1, w2, topk_idx, topk_weights, fc1_c, route_metadata
)
```

New:

```python
resources = op.prepare_training_resources(
    training_weights,
    slot_count=2,
    lane_count=1,
)
slot = resources.slots[0]
lane = resources.lanes[0]
resources.refresh_weights()
output = resources.forward(
    slot, lane, activation, topk_idx, topk_weights
)
dx, dprob, operands = resources.backward(slot, lane, grad_output)
overflow = resources.finalize_overflow((slot,), lane)
```

`dprob` now follows the MXFP8-staged kernel numerical contract and relaxed
atomic accumulation order. Dynamic inputs use a trusted-caller contract.
Distributed graph support requires one direct-P2P MNNVL domain, and
distributed lanes must be ordered consistently across ranks with captured
events. The fixed-capacity WGrad result is a producer ABI; no specific grouped
WGrad consumer is guaranteed in this release.

## Inference forward

`MoeEp.__call__` is the inference-forward surface:

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
)

output = op(
    activation,
    fc1_weight,
    fc2_weight,
    topk_idx,
    topk_weights,
)
```

`activation`, `fc1_weight`, and `fc2_weight` may independently be plain BF16,
FP16, or FP32 tensors, or MXFP8 `BlockScaledTensor` values. The logical shapes
are:

- `activation`: `(T, H)`
- `fc1_weight`: `(E_local, H, 2I)`
- `fc2_weight`: `(E_local, I, H)`
- `topk_idx` and `topk_weights`: `(T, K)`

MXFP8 operands are block-scaled along logical axis 1. The output always has
shape `(T, H)` and dtype `torch.bfloat16`. This surface does not return compact
FC1 or route metadata stashes and does not provide backward.

### Inference CUDA Graph capture

Call `warmup` with the exact tensors that will be captured:

```python
import torch

op.warmup(activation, fc1_weight, fc2_weight, topk_idx, topk_weights)
if ep_group is not None:
    torch.distributed.barrier(group=ep_group)

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    graph_output = op(
        activation, fc1_weight, fc2_weight, topk_idx, topk_weights
    )
```

`warmup` completes runtime bootstrap, symmetric allocation, weight staging,
JIT compilation, and one real launch. It is collective by contract for EP2+
but intentionally does not issue a process-group barrier. Captured inference
weights must expose usable PyTorch version counters and must match the warmed
weight cache. Replay may update captured tensor contents in place but may not
replace their storage.

`MoeEp` supports explicit `close()` and context-manager use. One instance is
bound to one CUDA device. Do not close an operator while a stream is capturing
or while graph work using its resources remains outstanding.

## Fixed-resource training

Training uses `prepare_training_resources`:

```python
from cudnn import MoeEpTrainingWeights

weights = MoeEpTrainingWeights(
    forward_fc1=forward_fc1_mxfp8,
    forward_fc2=forward_fc2_mxfp8,
    backward_w2_transpose=backward_w2t_mxfp8,
    backward_w1_transpose=backward_w1t_mxfp8,
)

resources = op.prepare_training_resources(
    weights,
    slot_count=2,
    lane_count=1,
)

slot0, slot1 = resources.slots
lane0 = resources.lanes[0]

# Required after each in-place source-weight update and before the first
# forward/backward that consumes that version.
resources.refresh_weights()

y0 = resources.forward(
    slot0,
    lane0,
    activation0,
    topk_idx0,
    topk_weights0,
)
dx0, dprob0, operands0 = resources.backward(slot0, lane0, grad_output0)
overflow = resources.finalize_overflow((slot0,), lane0)
```

`prepare_training_resources` is collective across `ep_group` and must execute
outside CUDA Graph capture. All ranks must use matching static configuration,
slot/lane counts, and tuning. The training backend internally enables FC1
preactivation generation and fixed-capacity WGrad operands, and fixes token and
scale-factor padding to 128.

`forward`, `backward`, and `finalize_overflow` enqueue the same device
operations in ordinary execution and in a caller-owned CUDA Graph. The caller
owns capture, replay, stream/event dependencies, slot reuse, and resource
lifetime.

A `MoeEp` instance can own only one training-resource set. Closing that set is
terminal for the operator: replacing source-weight storage requires a new
operator, new resources, and new graph captures.

### Slots and lanes

A persistent slot owns state that survives from matching forward to backward:

- routing indices and weights
- pool-native FC1 preactivation
- expert counts and padded offsets
- kernel dprob
- backward auxiliaries and outputs
- fixed-capacity WGrad operands
- per-slot overflow flags

An execution lane owns mutable router, protocol, and kernel scratch. Multiple
active streams require distinct lanes. Distributed MegaMoE communication
kernels must be ordered identically on every rank with captured CUDA events;
the kernels cannot be launched in unordered concurrent lanes.

All peer-visible regions are constructed in deterministic order. Their sizes
are validated and normalized across ranks before symmetric allocation so every
named region has the same peer offset.

## Tensor contracts

For inference through `MoeEp.__call__`, `activation` has:

- shape `(T, H)`
- BF16, FP16, FP32, or MXFP8 block-scaled input

Inference `fc1_weight` and `fc2_weight` accept the same plain or MXFP8 operand
families. MXFP8 activation and weights must be represented by
`BlockScaledTensor` with logical block axis 1.

The inference `topk_idx` contract is:

- shape `(T, K)`
- Int32 or Int64
- each element is `-1` or a valid global expert ID

The inference `topk_weights` contract is:

- shape `(T, K)`
- floating point

Fixed-resource training uses a narrower, graph-stable staging ABI:

- contiguous BF16 or FP32 `activation` and `grad_output`, each shaped `(T, H)`;
- contiguous Int32 `topk_idx` shaped `(T, K)`;
- contiguous FP32 `topk_weights` shaped `(T, K)`;
- all tensors on one device and `T <= max_tokens_per_rank`.

Expert IDs and finite dynamic values remain a trusted-caller replay contract;
they are not revalidated by host code after graph capture.

`MoeEpTrainingWeights` contains four contiguous MXFP8 block-scaled tensors:

- `forward_fc1`: `(E_local, H, 2I)`
- `forward_fc2`: `(E_local, I, H)`
- `backward_w2_transpose`: `(E_local, H, I)`
- `backward_w1_transpose`: `(E_local, 2I, H)`

Each data and scale tensor must be contiguous, reside on one device, and use
logical block axis 1. Plain FP16 operands are accepted by inference staging,
but fixed-resource training accepts only BF16 or FP32 `activation` and
`grad_output`.

Replacing weight storage requires preparing resources and capturing again.
Callers must establish stream/event ordering for in-place weight updates.

### Explicit weight refresh contract

`MoeEpTrainingWeights` uses a public contiguous MXFP8 layout, while the Rubin
kernels consume fixed-address K-major, gate/up-interleaved, and blocked-scale
layouts. `resources.refresh_weights()` enqueues the required device-only copies
and layout transforms into the internal kernel bindings.

The caller must obey all of the following:

- update both the data and scale tensors in place; their storage addresses,
  shape, stride, dtype, device, and capacity must remain unchanged;
- call `resources.refresh_weights()` after every source-weight update and
  before any forward or backward that consumes the new version;
- establish stream ordering from the weight update to the refresh and from the
  refresh to the first consumer, using the same stream or CUDA events;
- do not refresh between a matching forward and backward; both operations must
  observe the same four-tensor weight version;
- do not overlap a refresh with any forward/backward that reads the shared
  internal weight bindings, including operations using another slot or lane;
- replace any source storage only by closing the existing operator, creating a
  new `MoeEp` instance and resources, and recapturing every graph that
  references them. Closed resources cannot be reopened or replaced on the same
  operator.

For CUDA Graph execution, capture the refresh at the appropriate update
boundary:

```python
with torch.cuda.graph(graph, stream=stream):
    # An optimizer or external producer must complete its in-place updates
    # before this node.
    resources.refresh_weights()
    y = resources.forward(slot, lane, x, topk_idx, topk_weights)
    dx, dprob, operands = resources.backward(slot, lane, grad_output)
    overflow = resources.finalize_overflow((slot,), lane)
```

Replay then executes the captured device refresh; Python is not called during
replay. Activation `x` has the same storage rule under CUDA Graph capture:
its contents may change in place, but replacing its captured storage requires
recapture.

## Backward outputs

Fixed-resource backward returns:

- `grad_activation`: fixed-slot `(T, H)` FP32 view
- `dprob`: source-order `(T, K)` kernel dprob
- `MoeEpTrainingWgradOperands`

Kernel dprob follows the MXFP8-staged backward numerical contract and relaxed
atomic accumulation order. Bitwise determinism is not guaranteed.

`MoeEpTrainingWgradOperands` contains:

- FC1 operands: `fc1_a`, `fc1_sfa`, `fc1_b`, `fc1_sfb`
- FC2 operands: `fc2_a`, `fc2_sfa`, `fc2_b`, `fc2_sfb`
- `expert_offsets` and `valid_route_counts`

These tensors have fixed addresses and fixed capacity. Device
`expert_offsets`/`valid_route_counts` describe the live expert segments and
padding rows are zero. This release guarantees the producer ABI only; a
specific grouped-WGrad consumer is future integration work. The result is not
a pair of dense gradients that can be passed directly to an optimizer.

## Tuning

`MoeEpTuningConfig` exposes semantic-preserving performance controls:

- `token_back_mode`: `"epi_warps"`, `"standalone_warps"`, or
  `"reuse_dispatch_warps"`
- `epi_flag_batch`: one of the validated `(M, N)` flag-batch pairs
- `token_in_flag_batch`: `1`, `2`, `4`, `8`, or `16`
- `group_hint`: `None`, `64`, `128`, `256`, `512`, `768`, or `1024`
- `reduce_topk_in_kernel`: Boolean

Every rank in an EP group must use the same tuning configuration.
`reduce_topk_in_kernel=True` requires BF16 combine/output,
`apply_topk_in_fc1=True`, and `token_back_mode="epi_warps"`.

## Capacity and overflow

`max_recv_size_per_rank` defines bounded receive capacity. Resources cannot
grow during capture. A capacity change requires preparation and recapture.

Inference checks its per-call overflow result after the fused launch. The
fixed-resource training transport deterministically truncates overflow so all
ranks complete the communication protocol. `finalize_overflow` combines the
forward and backward flags for all selected slots and performs one captured
scalar MAX all-reduce for EP2+.

With `drop_on_overflow=True`, `finalize_overflow` returns a one-element Int32
CUDA tensor: zero means no overflow and nonzero means truncation occurred.
Dropped routes contribute zero. With `drop_on_overflow=False`, overflow is a
fatal device assertion; this mode requires `torch._assert_async`, and EP2+
requires an NCCL process group. The training transport still truncates first
to let every rank finish the protocol before the public policy is applied at
graph tail.

## CUDA Graph execution

For inference:

1. all ranks call `MoeEp.warmup` with the exact capture bindings;
2. the caller aligns ranks after warmup;
3. each rank captures its forward graph;
4. graph execs are replayed in the same cross-rank order.

For fixed-resource training:

1. all ranks collectively call `prepare_training_resources`;
2. all ranks perform an ordinary
   `refresh_weights -> forward -> backward -> finalize_overflow` warmup so
   every staging, MegaMoE, and WGrad-export kernel is compiled;
3. each rank captures its outer graph;
4. ranks align after capture;
5. graph execs are submitted in lockstep without host synchronization inside
   a replay burst;
6. all stream work completes before resources are closed.

For EP2+:

Distributed MegaMoE launches must have the same order on every rank.
Independent lane storage does not permit unordered collective-kernel overlap.
Use distinct lanes for simultaneously active streams and captured CUDA events
to impose the same cross-stream order on every rank.

Each graph binds fixed tensor shapes, addresses, slots, lanes, and token
extent. Dynamic routing values may change in place; dynamic shapes may not.
Different token extents may share prepared resources, but each extent must be
warmed and captured as its own graph specialization.

## NVSHMEM topology

MegaMoE kernels use direct symmetric peer pointers. With
`NVSHMEM_REMOTE_TRANSPORT=none`, every EP rank must appear in the P2P connected
list and `NVSHMEM_TEAM_SHARED` must span the complete EP world.

Selecting `ibrc` does not by itself make a non-P2P peer directly addressable.
Cross-MNNVL execution is not part of the current support matrix.

Global expert `e` belongs to group-relative EP rank
`e // experts_per_rank`. Noncontiguous global-rank process groups are accepted
when their group-relative topology and direct peer access are valid. When
creating multiple subgroups, every world rank must create them in the same
order.

## Validation coverage

The following configurations are exercised by the current test and probe
suite. This list describes validation coverage and does not broaden the
topology contract beyond one direct-P2P MNNVL domain:

- EP1 inference, fixed-resource training, overflow, and CUDA Graph replay
- single-node EP2/EP3/EP4 inference
- single-node EP2/EP4 training
- noncontiguous EP2 inference and training subgroups
- multi-node forward acceptance for EP4/EP6/EP12/EP16
- multi-node backward acceptance for EP8/EP16/EP32
- fixed-resource CUDA Graph launchers for EP8/EP16/EP32

## Validation

Run host-side and local tests:

```bash
python -m pytest \
  test/python/moe_ep/test_moe_ep_forward.py \
  test/python/moe_ep/test_moe_ep_backward.py \
  -m L0
```

Run SM107 single-node distributed tests from an exclusive GPU allocation:

```bash
python -m pytest \
  test/python/moe_ep/test_moe_ep_forward.py \
  test/python/moe_ep/test_moe_ep_backward.py \
  -m L1
```

`test/python/moe_ep/test_moe_ep_multinode.py` is torchrun-native and requires
the standard `LOCAL_RANK`, `LOCAL_WORLD_SIZE`, `RANK`, and `WORLD_SIZE`
environment. The multi-node fixture initializes NCCL and defaults
`NVIDIA_IMEX_CHANNELS=0`.

Run distributed fixed-resource probes from an existing Slurm allocation:

```bash
NVSHMEM_REMOTE_TRANSPORT=none \
data/script/run_moe_ep_forward_multinode_slurm.sh backward-ep8

NVSHMEM_REMOTE_TRANSPORT=none \
data/script/run_moe_ep_forward_multinode_slurm.sh graph-ep8
```

The launcher provides `backward-ep16`, `backward-ep32`, `graph-ep16`, and
`graph-ep32` for the larger EP configurations. The fatal captured-overflow
assertion is a separate `graph-ep8-error` expected-failure task.

The graph tasks invoke `test/python/moe_ep/probe_moe_ep_training_graph.py`,
which covers collective warmup, capture alignment, lockstep replay bursts,
dynamic routing/overflow recovery, ordered multi-lane execution, and
collective teardown.
