# MegaMoE backend

The private MegaMoE backend provides Rubin SM107 MXFP8 execution for
`cudnn.moe_ep`.

## Executable capability

- CUDA Rubin SM107 (compute capability 10.7)
- BF16 output with BF16 or MXFP8 combine
- `hidden_size % 128 == 0`
- `intermediate_size % 256 == 0`
- `top_k <= min(32, num_experts)`
- explicit positive `max_tokens_per_rank`
- `apply_topk_in_fc1=True`

Inference accepts plain BF16/FP16/FP32 or MXFP8 operands and stages plain
operands to MXFP8. Fixed-resource training accepts contiguous BF16/FP32
activation and grad-output tensors, contiguous Int32 routing indices,
contiguous FP32 routing weights, and four contiguous MXFP8 training-weight
packs. NVFP4 operands, non-BF16 output, and `apply_topk_in_fc1=False` are not
executable.

## Public execution paths

`MoeEp.__call__` is the inference-forward surface. It returns only the fused
BF16 `(T, H)` output and does not expose a compact training stash or backward.
Inference CUDA Graph capture requires `MoeEp.warmup` with the exact capture
bindings before capture. EP ranks must align after warmup and replay in the
same cross-rank order.

Training uses fixed resources:

```python
resources = op.prepare_training_resources(
    weights,
    slot_count=2,
    lane_count=1,
)
slot0, slot1 = resources.slots
lane0 = resources.lanes[0]

resources.refresh_weights()
y0 = resources.forward(slot0, lane0, x0, topk_idx0, topk_weights0)
dx0, dprob0, wgrad0 = resources.backward(slot0, lane0, grad0)
overflow = resources.finalize_overflow((slot0,), lane0)
```

`prepare_training_resources` is collective over the EP group, executes outside
capture, and fixes the training kernel to FC1-preactivation generation,
fixed-capacity WGrad operands, and token/scale-factor padding 128.

The same methods execute ordinarily during warmup and enqueue identical nodes
inside a caller-owned outer CUDA Graph. MoeEP does not own or wrap graph
replay. The ordinary warmup must cover
`refresh_weights -> forward -> backward -> finalize_overflow` so staging,
forward, backward, and WGrad-export kernels are compiled before capture.

## Fixed resource model

- A persistent slot owns one microbatch's routing snapshot, pool-native FC1
  preactivation, kernel dprob, outputs, backward auxiliaries, overflow flags,
  and fixed-capacity WGrad operands.
- An execution lane owns mutable router, barrier, and kernel scratch.
- Every symmetric region is built in deterministic order and its size is
  normalized by name across EP ranks before allocation.
- Multiple streams require distinct lanes. Distributed MegaMoE kernels must be
  ordered consistently on every rank with captured CUDA events; independent
  lane storage does not permit unordered communication overlap.
- `max_recv_size_per_rank` bounds allocation. Capacity never grows during
  capture; changing it requires new resources and graph capture.

## Weights and WGrad outputs

`MoeEpTrainingWeights` contains four address-stable MXFP8 block-scaled tensors:
forward W1/W2 and independently quantized backward W2-transpose/W1-transpose.
When forward weights use compact K-major storage and backward transpose weights
use standard contiguous storage, the kernels alias weight data directly. In
this layout, W1 gate/up values are interleaved in 32-element strips and only
scales require kernel-native staging. When forward weights use standard
contiguous storage, weight data and scales are copied and reordered into
persistent kernel buffers. After every in-place data+scale update, the caller
must enqueue `resources.refresh_weights()` before the first consumer,
with explicit stream/event ordering. A matching forward/backward pair must use
one version; refresh cannot overlap any consumer on another slot/lane. Replacing
source storage requires closing the old operator, creating a new `MoeEp`
instance and resources, and capturing a new graph. Closed resources are
terminal and cannot be replaced on the same operator. Capturing the refresh
turns these transforms into fixed-address graph nodes, so replay does not call
Python.

Backward returns kernel dprob directly. It follows the MXFP8-staged numerical
contract and relaxed atomic accumulation order.

`MoeEpTrainingWgradOperands` is a fixed-capacity producer ABI. Device
`expert_offsets` and `valid_route_counts` describe the current valid K extent;
padding is zeroed. No specific downstream grouped-WGrad consumer is guaranteed
by this milestone.

## Overflow policy

`max_recv_size_per_rank` bounds the fixed receive pool. When omitted, it uses
the worst-case `ep_size * max_tokens_per_rank * top_k`; an explicit value is
capped at that count.

The fixed-resource transport truncates deterministically so every rank
completes its communication protocol. `finalize_overflow` aggregates the
selected slots and performs a scalar MAX all-reduce for EP2+. With
`drop_on_overflow=True`, it returns a one-element Int32 status tensor and
dropped routes contribute zero. With `drop_on_overflow=False`, the graph tail
uses `torch._assert_async`; EP2+ error mode requires NCCL.

## Distributed support

Hardware acceptance covers EP1, EP2/4, EP8, EP16, and EP32 on one MNNVL
peer-access domain. The Python capability layer has no hard EP-size ceiling;
the listed sizes are validated scope rather than cross-MNNVL support.

Current tests additionally cover single-node EP3 inference, noncontiguous EP2
subgroups, multi-node EP4/6/12/16 forward, multi-node EP8/16/32 backward, and
EP8/16/32 fixed-resource graph launchers. EP2+ probes perform collective
warmup, independent capture, capture alignment, diagnostic replay, lockstep
production-like replay bursts, overflow/recovery, ordered multi-lane
execution, and collective teardown.

The kernels use direct peer pointers obtained from NVSHMEM symmetric tensors.
`NVSHMEM_REMOTE_TRANSPORT=none` is valid only when every EP rank is directly
P2P-accessible (`NVSHMEM_TEAM_SHARED` spans the EP world). IBRC initialization
alone does not make non-P2P peers directly addressable by these kernels.
