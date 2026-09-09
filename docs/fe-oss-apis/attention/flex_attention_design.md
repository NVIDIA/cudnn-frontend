# Flex Attention Mask Plan Design

This document describes the interval-mask planner and kernel protocols used by
`cudnn.flex_attention`.
For public APIs and examples, see [Flex Attention](flex_attention.md).

Implementation entry points in this repository:

- [Plan construction](../../../python/cudnn/flex_attention/plan/builder.py)
- [Plan ownership and signatures](../../../python/cudnn/flex_attention/plan/mask_plan.py)
- [Planner kernels](../../../python/cudnn/flex_attention/plan/kernels/)
- [Attention kernels](../../../python/cudnn/flex_attention/kernels/)

## 1. Purpose and Scope

FlexAttention represents a static arbitrary attention mask as interval endpoints. A planner
converts those endpoints into a reusable `MaskPlan`, and the attention kernels consume the plan
without evaluating a Python mask function or rebuilding block visibility.

This document specifies:

- the public mask representation and coordinate system;
- the planner stages and their intermediate data;
- the partial/full block topology;
- the consumer-specific packed predicate payload;
- the forward and backward consumption protocols; and
- plan ownership, compatibility, and correctness requirements.

General Q/K/V pipelines, scheduler policy, head-dimension tuning, benchmarks, and optimization
history are outside the scope of this document.

## 2. Public Mask Representation

The public `mask_func` argument is a contiguous CUDA `int32` tensor:

```text
mask_func[Hmask, nfunc, total_q]
```

`mask_func` is retained as the API name, but the value is a tensor rather than a callable.

- `Hmask` is `1` or `Hq`. `Hmask=1` shares one mask across all Q heads.
- `nfunc` is a positive odd integer.
- `total_q` is `B * Sq` for fixed-length BSHD input and the total Q-token count for Varlen THD
  input.
- Every endpoint is a K position local to the sample that owns the Q row.

For one Q row, endpoints `F0, F1, ..., F(nfunc-1)` define a union of half-open intervals:

```text
visible(q) = [0, F0) U [F1, F2) U [F3, F4) U ...
```

For example:

```text
endpoints = [32, 64, 96]
visible K = [0, 32) U [64, 96)
```

The endpoints of each row must be ordered and bounded by the sample's K length:

```text
0 <= F0 <= F1 <= ... <= F(nfunc-1) <= sample_k_len
```

Equal adjacent endpoints represent an empty interval. The public interface does not require the
caller to construct a block mask or a packed bit tensor.

## 3. Coordinate System

The design separates logical coordinates from physical storage addresses.

### 3.1 Fixed Length

Fixed-length BSHD input uses one common `Sq` and `Sk` for every sample. Q rows are flattened in
batch-major order in `mask_func`, while endpoint values remain sample-local K positions.

### 3.2 Varlen

Varlen THD input uses `cu_seqlens_q` and `cu_seqlens_k` to identify sample boundaries. The public
endpoint for a Q token is still local to that token's sample. It is never an offset into the
flattened K tensor.

Host-side validation checks that both prefix tensors:

- are rank-1, contiguous CUDA `int32` tensors on the Q device;
- have the same shape.

The planner validates their values on the GPU: the prefixes must begin at zero, end at
`total_q`/`total_k`, be monotonically nondecreasing, and respect `max_seqlen_q`/
`max_seqlen_k`. The result is returned through the exact-allocation header described in Section
4.5, so invalid input still raises a synchronous public exception without a separate full-prefix
D2H copy.

The public builder clones the prefix tensors. The resulting `MaskPlan` owns those clones, so a
later in-place mutation of the caller's tensors cannot change the sample partition of an existing
plan.

### 3.3 Planner and Consumer Coordinates

Planner kernels consume the public sample-local endpoints directly. The row's sequence descriptor
provides `q_offset`, `k_offset`, `q_len`, and `k_len`; endpoint values are clamped and validated
against that row's local `k_len`. The planner does not add a physical K offset to the endpoints and
does not build a padded endpoint tensor.

The compact topology stored in the final plan uses sample-local block indices. A consumer forms a
physical address from:

```text
sample physical offset + sample-local block offset + in-block coordinate
```

Consequently, a plan row cannot address Q/K/V data in another sample.

## 4. Planner Overview

The planner resolves the exact consumer before materializing a payload because block geometry and
packed-bit ownership depend on the target kernel.

```text
sample-local interval endpoints
              |
              v
structural input validation
              |
              v
resolve consumer topology and compact Varlen row prefixes
              |
              v
Q2K classify -> visible_bits, full_bits, partial/full counts
              |
              +-------------------------------+
              | build_backward=True           |
              v                               |
K2Q count from backward Q2K bitsets           |
              |                               |
              v                               |
scan counts and allocate exact compact outputs|
              |                               |
              v                               |
materialize Q2K/K2Q CSR and packed payloads   |
              |                               |
              v                               |
materialize architecture-neutral FWD schedule|
              |                               |
              +-------------------------------+
              v
MaskPlan
```

### 4.1 Input Validation and Consumer Resolution

The host first validates tensor rank, dtype, device, layout, Q/K/V geometry, head counts, endpoint
shape, and Fixed/Varlen arguments. Endpoint values and Varlen prefix values are validated by the
classifier and allocation-header path, avoiding separate elementwise validation launches and D2H
waits. The builder then resolves a consumer configuration for the current architecture and
direction.

The resolved configuration provides the planner with:

- logical Q and K block sizes;
- `q_stage` and CTA-group topology;
- PackGQA mapping;
- MMA/register ownership;
- number of physical payload subtiles;
- number of payload groups; and
- padded `uint32` words per payload group.

This separation is important: the sparse block relation is architecture-neutral, but the packed
predicate payload is not.

### 4.2 Compact Row Prefixes

Fixed-length plans have a rectangular upper bound for Q and K plan rows. Varlen plans instead
compute the number of valid Q and K block rows for each sample, followed by prefix sums such as:

```text
cu_total_q_plan_rows[B + 1]
cu_total_k_plan_rows[B + 1]
```

These prefixes compact away nonexistent rows. They also map a compact outer row back to its
sample, so the classifier and consumer do not have to scan `cu_seqlens` to rediscover ownership.
Forward and backward may use different prefixes because their tile geometries can differ.

One architecture-neutral `VarlenGeometry` kernel builds all required FWD and optional BWD block
prefixes from `cu_seqlens_q/k`. The same kernel validates prefix start/end values, ordering, and
maximum lengths and writes a device error flag. SM90, SM100, and SM103 use the same algorithm;
only consumer tile sizes and the compilation target differ.

### 4.3 Q2K Classification

The Q2K classifier examines every valid pair of one Q plan row and one sample-local K block. It
assigns exactly one of three states:

- `empty`: no valid score element is visible;
- `full`: the complete Q/K tile is in bounds and every score element is visible; or
- `partial`: at least one element is visible, but the tile is not fully visible or requires Q/K
  tail protection.

The classifier writes four temporary outputs:

```text
visible_bits[Hmask, upper_q_rows, words_for_k_blocks]
full_bits[Hmask, upper_q_rows, words_for_k_blocks]
partial_counts_tmp[Hmask, upper_q_rows]
full_counts_tmp[Hmask, upper_q_rows]
```

`visible_bits` records all non-empty Q/K block pairs. `full_bits` is a subset that records the
fully visible pairs. Within the valid K-block domain:

```text
partial = visible_bits & ~full_bits
empty   = ~visible_bits
```

The bitsets provide a compact intermediate representation that can be read in either Q-major or
K-major order. They are planner workspace and are not retained by `MaskPlan`.

The classifier validates every endpoint against the owning sample's local K range and checks the
complete endpoint sequence for monotonic order. It writes any interval or Varlen-prefix failure to
the same device error word used by exact allocation. The host reads that error once and raises
before materialization, so malformed metadata cannot produce a plan.

### 4.4 Backward K2Q Counting

When `build_backward=True`, the planner classifies the mask again with the backward consumer's
tile geometry. The resulting Q-major bitsets describe the same mathematical mask at the backward
block granularity.

The K2Q count kernel reads those bitsets by K row and counts partial and full Q contributors for
each K block. This is the counting phase of a sparse transpose; it does not rescan all endpoint
intervals and does not yet write the final K2Q indices.

### 4.5 Count, Scan, and Exact Allocation

Classification uses upper-bound workspace because the exact number of active blocks is unknown
before the GPU kernels run. The planner reduces the count arrays to a small header containing:

```text
forward Q rows, forward partial nnz, forward full nnz,
backward Q rows, backward K rows, backward partial nnz, backward full nnz,
error flags
```

The host reads this eight-value header once. The readback establishes exact allocation sizes and
turns device-side metadata errors into public exceptions.

Fixed and Varlen plans use architecture-neutral scan/header kernels on SM90, SM100, and SM103.
FWD and BWD classification remain independent, and both modes retain one allocation-header D2H.

For fixed length, `FixedScanHeader` scans the FWD and optional K2Q/dedicated-dQ count arrays. The
temporary counts already have their retained shape, so the plan reuses them directly while the
kernel writes exact exclusive CSR offsets and the allocation header.

For Varlen, `VarlenScanHeader` scans the upper-bound count arrays, stores their inclusive scans,
and writes the same allocation header. After the header establishes the exact row and NNZ sizes,
one `VarlenCompactMetadata` kernel copies every valid per-head count prefix and converts the
inclusive scans into the retained exact offsets:

```text
partial_offset[0] = 0
partial_offset[i + 1] = partial_offset[i] + partial_count[i]

full_offset[0] = 0
full_offset[i + 1] = full_offset[i] + full_count[i]
```

The compact materializer handles FWD, optional K2Q, and optional dedicated-dQ metadata in one
launch. It does not pad the retained CSR and does not allocate capacity-based outputs. Partial and
full outputs remain independent, and empty blocks require no compact storage.

### 4.6 Q2K Materialization

The Q2K materializer revisits set bits in the classification workspace and writes two independent
CSR structures:

```text
partial CSR                         full CSR
mask_block_cnt                      full_block_cnt
mask_block_offset                   full_block_offset
mask_block_idx                      full_block_idx
mask_block_masks
```

`mask_block_idx` and `full_block_idx` contain sample-local K block indices. A partial entry also
receives one consumer-specific packed predicate payload. A full entry does not have a payload.

The design does not assume that either index list is contiguous. It does not create full-block
runs and does not merge partial and full blocks into one index array.

### 4.7 K2Q Materialization

The K2Q materializer performs the write phase of the sparse transpose. For every K-major row, it
writes the sample-local Q block indices that were counted in Section 4.4, again with independent
partial and full CSR arrays.

The K2Q payload is generated for the backward consumer's accumulator layout. Generic backward
may also materialize `dq_write_order` and `dq_write_order_full`. Each entry stores only the dQ
write rank and remains parallel to the corresponding block-index array; the Q block index is read
from `mask_block_idx` or `full_block_idx`. This metadata defines a legal order for parallel dQ
accumulation and does not change mask visibility.

If a dedicated dQ kernel requires its own Q-major layout, the plan may additionally contain a
separate Q2K view materialized for that consumer.

### 4.8 Architecture-Neutral Forward Schedule Materialization

SM90, SM100, and SM103 forward share one plan-owned task schedule. The planner produces one task
for every valid Q plan row and scheduled head:

```text
num_forward_tasks = valid Q plan rows * scheduled heads
```

`fwd_work_desc` is present for every supported forward plan. `sequence_desc` is present for
Varlen and for consumer families that require an explicit sequence descriptor:

| Descriptor | Contents | Purpose |
|---|---|---|
| `sequence_desc[B, 8]` | `q_offset`, `k_offset`, `q_len`, `k_len`, Q-plan-row begin/count, valid K-block count, reserved field | Maps logical rows to one sample and defines address/tail bounds. |
| `fwd_work_desc[num_forward_tasks, 4]` | `m_block`, scheduled head, `batch_idx`, `q_valid_rows` | Gives each scheduled task its complete Q-tile identity. |

The planner computes task cost as:

```text
task_cost = partial_block_count + full_block_count
```

Partial and full blocks deliberately have equal scheduling cost. Positive-cost tasks are stably
ordered by L2 section, descending task cost, and the existing head/Q-block tie order; zero-cost
tasks follow in batch/head/Q-block order. The planner implements this order with bounded counting:
one kernel builds a `(section, task_cost)` histogram, one CTA computes stable section rank and
bucket offsets, and one warp per section scatters 32-task chunks in tie order. The implementation
therefore preserves the exact lexicographic order without a chain of general-purpose GPU sorts or
data-dependent host reads. The kernel consumes the resulting prepared work queue; it does not
enumerate all batch Q tiles or reconstruct task ownership from `cu_seqlens`.

The descriptor layout and ordering are architecture-neutral; only the queue backend differs:

- SM90 uses `PlanDynamicPersistentTileSchedulerSm90`, backed by a call-local software atomic
  counter. Fixed and Varlen forward use the same queue mechanism, and every task is obtained from
  the dynamic queue.
- SM100/SM103 use `PlanClcPersistentTileSchedulerSm100`, backed by the hardware CLC queue.

The SM90 counter is runtime workspace rather than plan storage. This lets one immutable plan be
used concurrently on different CUDA streams without sharing mutable scheduler state.

`q_len`, `k_len`, and `q_valid_rows` remain necessary after scheduling. They protect physical
loads and stores at sequence tails; they are not used to rediscover which tasks exist.

## 5. Packed Predicate Payload

Only partial blocks carry `mask_block_masks`. Its logical shape is:

```text
mask_block_masks[
    partial_nnz,
    physical_subtile,
    payload_group,
    uint32_word,
]
```

This tensor is not a row-major bitmap of the score tile. The materializer follows the target
consumer's MMA accumulator ownership and places each score predicate in the order used by the
final register fragment:

```text
bit = 1  -> keep the score
bit = 0  -> replace the score with -inf
```

The attention kernel therefore does not reevaluate interval endpoints or execute a generic mask
modifier. It loads the `uint32` words for its payload group and applies compile-time-unrolled bit
selection to the score accumulator.

### 5.1 R2P Semantics

R2P names the data flow from a register fragment and a packed predicate to predicated register
values. It does not require a hardware intrinsic literally named `R2P`.

- SM100/SM103 loads the score fragment from TMEM and applies constexpr bit selection.
- SM90 applies the same predicate semantics to a WGMMA register fragment.

The final payload word can contain padding bits. The consumer emits accesses only for compile-time
elements that exist in its accumulator fragment, so padding cannot cause an out-of-bounds register
access.

## 6. Kernel Consumption

### 6.1 Forward

Forward consumes the partial and full CSR arrays separately:

```text
for block in partial_range:
    load K/V
    compute score
    load packed predicate
    softmax_step(apply_mask=True)

for block in full_range:
    load K/V
    compute score
    softmax_step(apply_mask=False)
```

An empty block is absent from both ranges. A full block never loads a packed predicate. A plan row
with no active blocks writes `O=0` and `LSE=-inf`.

### 6.2 Two-Stage SMEM Mask Pipeline

SM90 forward and the generic SM100/SM103 qstage1 + 2CTA forward kernel stage partial payloads
through a two-stage SMEM packed-mask pipeline. Each stage owns one CTA-native payload slot and its
barrier state. For an M128xN128 non-PackGQA consumer, the pipeline uses 4 KB of mask SMEM per CTA:

```text
SM90:             256 payload groups * 2 uint32 words * 4 bytes = 2 KB per stage
SM100/SM103 2CTA: 128 payload groups * 4 uint32 words * 4 bytes = 2 KB per stage
```

For a partial block, the load warp bulk-copies the CTA-native payload plane from GMEM to the
selected SMEM stage. The MMA/softmax consumers wait for that stage, copy their payload words to
registers, release the stage, and apply the packed predicate. PackGQA can reduce the number of
payload groups while preserving the same protocol.

Pipeline state advances exactly once for each partial block assigned to a stage. Full and empty
blocks neither load a payload nor advance the mask pipeline. The same protocol applies to Fixed,
Varlen, PackGQA, and non-PackGQA generic shapes.

SM90 D256 uses an M128xN64 tile. The smaller N dimension leaves enough shared memory for the
two-stage mask pipeline; its non-PackGQA payload contains one `uint32` word per consumer thread and
uses 2 KB across both stages. The generic SM100/SM103 qstage1 + 1CTA and qstage2 + 1CTA variants,
and the dedicated SM100/SM103 D256 forward kernel, retain direct GMEM-to-register payload loads.
Delivery method does not change the payload ABI or the partial/full topology.

### 6.3 Backward

Each backward consumer uses the traversal that matches its output:

- Q-major consumers use a Q2K view;
- K-major dK/dV consumers use a K2Q view;
- partial entries load their packed predicate; and
- full entries use `full_block_idx`, do not load a packed predicate, and read write-order metadata
  only when the consumer requires it.

The concrete kernel may choose its loop order, but it must preserve the independent partial/full
semantics. There is no raw-index bypass for singleton-heavy masks.

## 7. Architecture and Plan Compatibility

Endpoint semantics and the partial/full CSR organization are common to SM90, SM100, and SM103. A
materialized topology and payload, however, belong to one concrete consumer because block
geometry and accumulator ownership differ across WGMMA, tcgen05/TMEM, direction, and CTA
topology.

Every materialized view carries an `ArbitraryPlanSignature`. It records the architecture family,
direction, kernel family, tile geometry, `q_stage`, CTA-group size, PackGQA configuration, MMA
layout, payload layout, scheduler layout, and dQ-order format.

Dispatch compares the signature field by field before launch. A mismatch raises an explicit error;
the runtime never interprets one consumer's payload as another consumer's payload.

The same public mask can therefore be planned separately for different architectures or kernel
topologies. One `MaskPlan` can own forward, backward, and dedicated dQ views, but each view remains
consumer-specific.

## 8. Plan Ownership and Reuse

`MaskPlan` is an opaque, read-only owner of compact topology, packed payloads, runtime geometry,
and optional schedule metadata.

- Q/K/V tensor identity is not part of the binding; new values can reuse a plan when geometry is
  unchanged.
- A fixed-length plan binds batch size, `Sq`, `Sk`, head counts, head dimensions, dtype, device,
  architecture, and consumer topology.
- A Varlen plan additionally owns cloned Q/K prefix tensors and their versioned runtime binding.
- A training call requires a plan built with backward views.
- `debug_snapshot()` returns tensor clones and does not expose mutable internal storage.

## 9. Design Invariants

1. Public endpoints and final CSR indices are sample-local.
2. Every valid Q/K block pair belongs to exactly one of `empty`, `partial`, and `full`.
3. Partial and full blocks use independent CSR arrays and do not require contiguous indices.
4. Only partial blocks carry packed predicate payloads.
5. Payload bit order exactly matches the target consumer's accumulator ownership.
6. Full blocks do not load a payload, apply a bit-select, or advance the mask pipeline.
7. Q2K and K2Q views describe the same mathematical block relation at their consumer geometry.
8. Varlen never uses fixed-length addressing.
9. A plan-signature mismatch is an error, not a fallback condition.
10. Direct and SMEM-staged payload delivery preserve the same payload ABI and mask semantics.
