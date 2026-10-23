# Block Sparse Attention (BSA)

**This is an experimental API and subject to change.**

## Overview

Block Sparse Attention computes non-causal scaled dot-product attention over a
block-level sparse pattern. For query token `i`, let `m = floor(i / block_size)`
be its query-block index and let `K_m` be the union of the key/value blocks
listed for that query block. The operation is

$$
O_i = \text{softmax}_{j \in K_m}
\left(\frac{Q_i K_j^T}{\sqrt{D}}\right)V_j.
$$

The softmax is normalized once over all valid tokens in the selected blocks.
`block_sizes` can shorten individual key/value blocks so that padded tokens do
not participate in the softmax.

Unlike [NSA Selection Attention](nsa.md#1-selection-attention), BSA supplies one
list of key/value blocks per **query block**. NSA Selection supplies routing
metadata at query-token granularity and is one component of the larger NSA
pipeline.

BSA is implemented with Python CuTe DSL/JIT kernels.

## Installation

The CuTe DSL runtime is a required dependency, so the base install is enough:

```bash
pip install nvidia-cudnn-frontend
```

The package-wide `nvidia-cutlass-dsl[cu13]>=4.6.2` dependency floor applies to
BSA. Sage FP8 relies on functionality introduced in CuTe DSL 4.6.1, which the
package floor already satisfies. A defensive runtime check remains for
environments that force an older DSL.

## Forward

```python
import torch
from cudnn.ops import (
    block_sparse_attention_forward,
    block_sparse_attention_backward,
    block_sparse_attention_fp8_forward,
)

q = torch.randn(1, 8, 1024, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 8, 2048, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn_like(k)

# For the SM90 blk64 path, this example has 16 Q blocks and 32 KV blocks.
# Every Q block attends to the first four KV blocks.
q2k_block_index = torch.arange(4, device="cuda", dtype=torch.int32)
q2k_block_index = q2k_block_index.view(1, 1, 1, 4).expand(1, 8, 16, 4).contiguous()
block_sizes = torch.full((32,), 64, device="cuda", dtype=torch.int32)

result = block_sparse_attention_forward(
    q,
    k,
    v,
    q2k_block_index,
    block_sparse_num=4,
    block_sizes=block_sizes,
    sparse_block_size=64,
)
o, lse = result
# The same values are available as result["o_tensor"] and result["lse_tensor"].
```

The default layout is `BHSD`. Pass `layout="bshd"` for tensors in `BSHD`
layout. The output follows the input layout, while `lse` is always the FP32
natural-log log-sum-exp with shape `(B, H_q, S_q)`.

For explicit Blackwell blk64 calls, TMA-aligned BSHD inputs are read through
strided views. Both unsplit outputs and split-KV combined outputs are written
directly in the requested layout. Misaligned inputs retain their staging fallback.

### Sparse metadata

| Argument | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `q2k_block_index` | `(B, H_q, N_q, K_max)` | `int32` | KV-block IDs; only the valid prefix of each last dimension is read |
| `block_sparse_num` | scalar | Python `int` | Fixed valid-prefix length for every query block |
| `q2k_block_nums` | `(B, H_q, N_q)` | `int32` | Optional per-query-block valid-prefix lengths; overrides `block_sparse_num` |
| `block_sizes` | `(N_kv,)` or backend-supported batched form | `int32` | Number of valid tokens in each KV block |

Here `N_q = ceil(S_q / sparse_block_size)` and
`N_kv = ceil(S_kv / sparse_block_size)`. Except for packed GQA described below,
metadata is per query head. The metadata tensors must reside on the same CUDA
device as `q`, `k`, and `v`. The value ranges below are a hard caller contract.

#### Value ranges (caller contract)

Let `K_max = q2k_block_index.shape[-1]`. Only the active prefix of each
`q2k_block_index` row — the first `q2k_block_nums[b, h, m]` entries with
variable counts, or the first `block_sparse_num` entries with a fixed count —
is consumed. Values in the inactive suffix are ignored.

- Within each row, active `q2k_block_index` values must be unique integers in
  `[0, N_kv)`.
- For forward with variable counts, every `q2k_block_nums` value must be in
  `[0, K_max]` when `allow_empty_block_nums=True`, and in `[1, K_max]`
  otherwise. Backward variable counts may be in `[0, K_max]`.
- With fixed counts, `block_sparse_num` must be in `[1, K_max]`. The
  FP16/BF16 SM100/SM103 blk128 path additionally requires an even value, i.e. an even
  `block_sparse_num` in `[2, K_max]`.
- The `block_sizes` entry for every physical KV block referenced by an active
  `q2k_block_index` value must be in `[1, sparse_block_size]`. Entries for
  unreferenced physical KV blocks are ignored. A zero-sized referenced block is
  not supported by regular FP16/BF16 attention; use `q2k_block_nums` (or the
  sparse index prefix) to drop the block instead. FP8 SM100/SM103 blk128 also
  accepts zero-sized blocks, as specified in its architecture contract below.

Tensor value ranges and per-row uniqueness are not validated at runtime.
Violating the contract is unsupported and may produce invalid results or
invalid device memory accesses.

On the SM100/SM103 blk128 path, `pack_gqa=None` automatically packs GQA when
the GQA ratio `r = H_q / H_kv` divides 128. Packed metadata has shape
`(B, H_kv, ceil(S_q * r / 128), K_max)`. Pass `pack_gqa=False` to use the
unpacked `(B, H_q, ceil(S_q / 128), K_max)` contract on every architecture.
Explicit `pack_gqa=True` requires `r` to divide 128.

When `block_sizes=None`, each referenced physical KV block is treated as full.
Provide `block_sizes` whenever a referenced final block is only partially
valid.

`sparse_block_size=None` chooses blk64 on SM90/SM120 and blk128 on
SM100/SM103. On SM90 or SM120, passing `sparse_block_size=128` selects a native
KV128 kernel that consumes blk128 metadata directly; it does not
expand the metadata or invoke the blk64 kernel. Passing `sparse_block_size=64`
explicitly selects the SM100/SM103 blk64 CuTe DSL path, whose shape support is
narrower. `kv_splits` is available on SM90 and the explicit Blackwell blk64
path; `use_clc` applies only to the explicit Blackwell blk64 path.

For SM120 blk128 with fixed `block_sparse_num`, full physical KV blocks
(`block_sizes=None`), and a KV sequence length divisible by 128, the dispatcher
uses an FA4-style native specialization with a dedicated K/V load warp,
register-resident Q, and a four-fold-unrolled sparse loop. Most CTAs process
128 Q rows; an underfilled final scheduling wave may use 64 Q rows per CTA
while still loading full KV128 blocks and using the parent Q block's original
metadata. This is Q-work scheduling, not KV128-to-KV64 lowering. Variable
per-row block counts, explicit block sizes, and partial final KV blocks use
the general native blk128 kernel. The FA4-style specialization requires
`nvidia-cutlass-dsl >= 4.7.0`; an older public DSL version is rejected with
a version-specific error before the specialized kernel is imported. The
package-wide downstream dependency floor is unchanged.

`kv_splits=2..256` computes FP32 partial outputs and combines them, with
workspace growing linearly in the split count. SM90 accepts an explicit integer
split count or `kv_splits="auto"` (the default stays 1). On SM90 the split
count is chosen so one split's K/V region of a head fits in about 3/4 of L2:
once a head's K/V exceeds L2 the unsplit kernel turns DRAM-bound, and a split
grid orders every Q tile of one KV range before the next. The BF16/FP16 MHA
policy applies at one head per wave (`ceil(S_q / 64)` at least `3 * SMs`), at most 25% density and fixed
counts, and declines (one split) otherwise; it was measured on H100 with
independent random indices (64 MiB K/V per head: 2 splits, +20%; 128 MiB: 4,
+70%; 256 MiB: 8, +94%; dense lists lose 12%). A caller-provided output
also resolves to one split. The SM100/SM103 blk64 path also accepts `kv_splits="auto"`; CLC is
compatible with split execution. Pass `use_clc=True` to use persistent CLC
scheduling with `kv_splits>1`, or `use_clc=False` to use one tile per CTA. The
default `use_clc=None` keeps CLC disabled for split execution because the
automatic scheduler policy has not been tuned for that combination.

The BF16 automatic policy is tuned assuming approximately uniform random KV
indices with little overlap between the neighborhoods of different Q blocks.
It uses host-visible tensor metadata and device properties, without reading
indices or counts back to the host. Highly overlapping, shared or local-window
indices can favor a different split count even at the same shape and density;
benchmark explicit `kv_splits` values for those masks. This assumption affects
performance only, not attention semantics or supported mask patterns.

For fewer than 512 total Q tiles (`B * H * ceil(S_q / 64)`), the existing policy
uses the fixed active block count, or index capacity for variable counts:
256--449 blocks select two splits, 450--899 select four, and 900 or more select
eight. Larger Q grids keep one split except in the measured long-Q region:
BF16 MHA with Q/K/V head dimensions 128, `S_q >= 262144`, a fixed count of at
least 512 active KV blocks, and density at most 1/8 (active blocks divided by
`ceil(S_k / 64)`). In that region, K+V bytes **per batch and KV head** above twice
the device L2 size select eight splits for at least 900 active blocks and four
or more heads; a single head requires at least 1024 active blocks. Otherwise,
four or more heads select four splits when per-head K+V bytes exceed L2.
All other cases, including an unavailable L2 size, keep one split. Variable
per-row counts also keep one split on larger Q grids because index capacity
does not describe actual row work. The Sage FP8 policy is separate.

Automatic selection
falls back to a smaller split count when the estimated live workspace does not
fit the available CUDA allocator budget; an explicit split count that exceeds
that budget raises `RuntimeError`.

## Sage FP8 forward

Sage FP8 is forward-only and supports blk64 on SM90, SM100/SM103, and SM120,
plus native blk128 on SM90, SM100/SM103, and SM120. Its public wrapper accepts contiguous
BF16 Q, K, and V tensors in `BHSD` layout and performs FP8 quantization internally:

```python
fp8_result = block_sparse_attention_fp8_forward(
    q,
    k,
    v,
    q2k_block_index,
    block_sparse_num=4,
    sparse_block_size=64,
)
o_fp8 = fp8_result["o_tensor"]
```

Q, K, and V must be BF16 MHA tensors in contiguous BHSD layout, have matching
batch and head counts, and use `D=128`. The wrapper accepts the same sparse
index metadata used by the regular forward API for the selected block size.
It lazily loads the quantizer, creates the E4M3 tensors and FP32 scales needed
by the kernel, and does not expose those implementation details as public
inputs or outputs.

`sparse_block_size` defaults to 64. On SM120, pass `sparse_block_size=128` with
`q2k_block_index` shaped `(B, H, ceil(S_q / 128), K_max)` and, when supplied,
`q2k_block_nums` shaped `(B, H, ceil(S_q / 128))`. Active indices address
physical 128-token KV blocks, so they must be in `[0, ceil(S_kv / 128))`.
For `block_sizes`, use `N_kv = ceil(S_kv / 128)` and valid-token counts in
`[1, 128]` for referenced blocks. The native kernel consumes this metadata
directly. SM100/SM103 also support this blk128 metadata layout, with the
additional empty-block and split-KV semantics below.

On SM90, pass `sparse_block_size=128` with indices shaped
`(B, H, S_q // 128, capacity)` and KV block IDs measured in units of 128 tokens.

The result is a one-key `TupleDict` containing `o_tensor`, a contiguous BF16
tensor of shape `(B, H, S_q, 128)`. This API does not return LSE and has no
backward implementation.

V quantization keeps the reciprocal and multiplication in FP32 until the
final E4M3 conversion, including the internal BHDS and blocked V layouts.
Q retains the Sage BF16 intermediate-rounding recipe; K uses FP32 centering.
FP32 rounding near an FP8 midpoint can select an adjacent E4M3 value across
implementations.

The SM100/SM103 blk128 kernel retains the upstream rounded-probability
normalization: P uses a factor of 128, and its denominator is accumulated from
the same rounded FP8 probabilities by the widened PV Tensor Core operation.
Other FP8 kernels use their existing normalization. Outputs are approximate
and are not expected to be bitwise equal across block sizes or architectures.

The architecture-specific FP8 contracts are:

- SM90 accepts any positive batch and head counts and requires both sequence
  lengths to be multiples of the selected `sparse_block_size` (64 or 128).
  It uses fixed `block_sparse_num` with full KV blocks of that size.
  Native blk128 shares each K/V tile between two compute warpgroups.
  Sparse indices may be non-contiguous (including prefix
  slices, stepped slices, permuted views, and broadcasts); their strides are
  consumed directly without an index copy. `q2k_block_nums` and `block_sizes`
  are not supported. It does not use split-KV. V quantization
  writes sequence-contiguous BHDS storage internally, so the attention kernel
  directly loads V for Hopper FP8 WGMMA without an in-kernel transpose.
  Both block sizes accumulate PV directly into FP32 output registers using
  FP8 WGMMA, accepting its reduced internal accumulation precision instead
  of promoting each KV tile with a separate FP32 addition.
- SM100/SM103 blk64 accepts any positive batch and head counts,
  and requires both sequence lengths to be multiples of 64. It uses fixed
  `block_sparse_num` with full 64-token KV blocks; `q2k_block_nums` and
  `block_sizes` are not supported. Split-KV is selected internally, and the
  public FP8 API does not expose `kv_splits` or `use_clc`.
- SM100/SM103 native blk128 accepts positive arbitrary Q/KV lengths, including
  physical tails, with MHA and D=128. Fixed counts may be odd; variable
  `q2k_block_nums` may be zero, producing zero output. `block_sizes` accepts
  rank 1, 2, or 3 and values in `[0, 128]`; zero masks a whole block, and the
  physical final-block length always limits the supplied extent. Inactive
  index capacity is ignored. Sparse metadata views are materialized as needed.
  The optional `kv_splits=1..256` partitions the live KV list using FP32 partials
  and the existing combine kernel; the default is one. Other FP8 paths require
  the new keyword to remain at its default. `softmax_scale`, when supplied,
  must be a positive finite host scalar. This implementation is limited to
  compute capabilities 10.0 and 10.3 and requires CuTe DSL >= 4.6.2.
- SM120 supports blk64 and blk128, accepts any positive batch and head counts,
  non-aligned Q/KV sequence tails, fixed or per-query-block counts, and
  `block_sizes` shaped `(N_kv,)`, `(B, N_kv)`, or `(B, H, N_kv)`. It does not
  use split-KV.

`block_sparse_attention_fp8_forward` relies on functionality introduced in
CuTe DSL 4.6.1; the SM90 blk128 path requires 4.6.2. Package-supported
installations provide CuTe DSL 4.6.2 or newer.
Its internal Q/K/V quantizer is also implemented in CuTe DSL and is
loaded lazily, so importing `cudnn` does not eagerly import it.

## Backward

Backward is an explicit API rather than a registered PyTorch autograd
operation. It recomputes probabilities from the forward output and LSE:

```python
dout = torch.randn_like(o)
grads = block_sparse_attention_backward(
    dout,
    q,
    k,
    v,
    o,
    lse,
    q2k_block_index,
    block_sparse_num=4,
    block_sizes=block_sizes,
    sparse_block_size=64,
)
dq, dk, dv = grads
```

The result keys are `dq_tensor`, `dk_tensor`, and `dv_tensor`. Optional
preallocated `dq_tensor`, `dk_tensor`, and `dv_tensor` arguments are supported.
The backward implementation builds a bucketed K-to-Q CSR task layout on the
GPU. `bucket_size_blocks` is an optional tuning override; leaving it unset uses
the backend default.

For SM100/SM103 blk128, long-query automatic buckets retain at least one SM
wave of logical `(batch, head, KV block, query group)` tasks, or the previous
default's task count when it is smaller. Otherwise, the previous bucket
default is used. This shape-based guard does not inspect device-side mask
contents; explicit `bucket_size_blocks` overrides remain authoritative.

For blk64, with contiguous per-row count metadata, at least two KV blocks and at least
1024 (batch, head, KV block) tasks, backward pairs KV neighborhoods to reuse
common Q/dO loads and dQ accumulation; smaller problems do not recover the
per-call pairing cost and use regular tasks.
Pairing metadata is limited to 64 MiB; fixed counts or unsupported metadata
layouts use regular tasks. Duplicate sparse edges retain their multiplicity.
Paired tasks retain FP32 dK/dV accumulation even with a single query bucket;
regular tasks can use the direct dK/dV epilogue when its alignment requirements hold.
The initial eight-KV matching is refined within 32-KV windows, with refinement
skipped above 1024 bitset words. Metadata construction is included in the
wrapper's work and is rebuilt for each call.
Backward interleaves its tasks across attention heads only for queries shorter
than 98,304 tokens, where it was measured to help; longer queries keep the
head-major task order.

These optimizations keep the public block size at 64 and preserve the existing
wrapper-owned workspace model. They do not add an allocation-free graph engine.
Use `benchmark/bsa/benchmark_sm100_blk64.py` for a synthetic complete-API
comparison; it records the imported source, input hashes and individual CUDA
Event samples. Numerical tests, sanitizer checks and measurements on each
target GPU are required before drawing performance conclusions.

Backward defaults to blk64 on SM90 and blk128 on SM100/SM103. Pass the same
explicit `sparse_block_size` used by forward when selecting the Blackwell blk64
path. SM100/SM103 blk128 backward accepts one-dimensional `block_sizes`
(`int32`, shape `(N_kv,)`) and masks each KV block's padded suffix consistently
with forward. Valid tokens occupy the prefix of each physical block; callers
zero `dout` for padded queries and discard their outputs. SM110 blk128 backward
still requires `block_sizes=None`.

## Current support

### Forward

| Architecture | Sparse block | Public input / kernel dtype | QK / V dimensions | Attention |
| --- | ---: | --- | --- | --- |
| SM90 | 64 or 128 (explicit) | FP16, BF16 | each of 64, 96, 128 | MHA, GQA, MQA |
| SM90 | 64 | BF16 / FP8 E4M3 | QK=128, V=128 | MHA |
| SM90 | 128 (explicit) | BF16 / FP8 E4M3 | QK=128, V=128 | MHA |
| SM100/SM103 | 128 | FP16, BF16 | QK=V=64, 96, or 128 | MHA, GQA, MQA |
| SM100/SM103 | 64 (explicit) | BF16 | QK=128, V=128 | MHA |
| SM100/SM103 | 64 | BF16 / FP8 E4M3 | QK=128, V=128 | MHA |
| SM100/SM103 | 128 (explicit) | BF16 / FP8 E4M3 | QK=128, V=128 | MHA |
| SM120 | 64 | FP16, BF16 | QK=128, V=128 | MHA, GQA, MQA |
| SM120 | 128 (explicit) | FP16, BF16 | QK=128, V=128 | MHA, GQA, MQA |
| SM120 | 64 | BF16 / FP8 E4M3 | QK=128, V=128 | MHA |
| SM120 | 128 (explicit) | BF16 / FP8 E4M3 | QK=128, V=128 | MHA |

For FP16/BF16, SM90 blk64 requires `S_q` to be a multiple of 64; native blk128 supports
positive arbitrary Q/KV lengths, including partial final blocks. Its fixed count may be
any positive value. The SM100/SM103 blk128 fixed count must be even and at
least two; SM120 and the explicit Blackwell blk64 path accept any positive
fixed count. Variable counts use `q2k_block_nums`. `allow_empty_block_nums`
defaults to `False`; when it is `True`, empty rows (`q2k_block_nums == 0`)
produce `O = 0` and `LSE = -inf`. SM90 blk64 selects the empty-row handling as a
compile-time specialization, so the default non-empty configuration keeps its
branch-free fast path. Its split-KV execution excludes empty rows. Native SM90
blk128 supports empty rows and empty splits with `kv_splits=1..256`.
Sage FP8 uses the architecture-specific contracts above. SM100/SM103 native
blk128 supports empty rows and explicit split-KV; its default remains unsplit.

### Backward

| Architecture | Sparse block | Dtype | Head dimension | Attention |
| --- | ---: | --- | ---: | --- |
| SM90 | 64 or 128 (explicit) | BF16 | 128 | MHA |
| SM100/SM103 | 64 | BF16 | 128 | MHA |
| SM100/SM103 | 128 | BF16 | 64 or 128 | MHA |

Backward is not implemented for SM120. It requires equal QK/V dimensions and
does not currently support GQA/MQA.

SM90 native blk128 requires CuTe DSL >= 4.6.2. Q/K/V, O/dO and gradient
buffers must have a contiguous head dimension, 16-byte aligned base pointers,
and row strides divisible by 16 bytes. BHSD/BSHD layouts and aligned noncompact
rows are addressed directly. Sparse metadata can be strided and remains on the
GPU; backward makes noncontiguous sparse indices and counts contiguous before
building CSR, as in blk64. Forward supports `block_sizes` of rank 1, 2, or 3; backward supports ranks
1 and 2. Both paths predicate sequence tails even without `block_sizes`.
Forward uses Q128 tiles and shares each KV128 load across two compute
warpgroups. Backward uses Q64xKV128 CTAs: both compute warpgroups execute all
five GEMMs, splitting KV rows for QK/dP/dK/dV and head dimensions for dQ.
Each native Q128/KV128 CSR edge streams two Q64 subtiles without expanding
the metadata. Backward accumulates gradients with FP32 atomics
and is not deterministic. Benchmark the explicit path for your shapes; blk64
remains the default, and native blk128 is not faster for every workload.

## Limitations

The current sparse kernels do not implement causal or local masking, dropout,
`mask_mod`, `score_mod`, paged KV cache, softcap, or variable-length packed
sequences. Regular forward/backward inputs must be rank four, use FP16/BF16 as
allowed above, and have a contiguous last (head-dimension) axis. Sage FP8 uses
the stricter BF16 BHSD contract described above and quantizes to E4M3
internally. Regular forward outputs are contiguous in the requested BHSD or
BSHD layout, including split-KV execution; FP8 output is contiguous BF16 BHSD.

Compilation is lazy. The first call for a new static configuration JIT-compiles
the relevant kernel; subsequent calls reuse an in-process cache.

The Torch public surface consists of allocating function wrappers under
`cudnn.ops`, also available under `cudnn` and `cudnn.BSA`; there is no separate `APIBase` class or explicit `compile()`
lifecycle for BSA.

Correctness tests and FP32 references are under
`test/python/block_sparse_attention/cutedsl`.

## SM100/SM103 FP8 blk128 source

The native producer and its private transfer/epilogue helpers are adapted from
[NVIDIA-JerryChen/Block-Sparse-Attention, f953680](https://github.com/NVIDIA-JerryChen/Block-Sparse-Attention/tree/f953680ff93a2ef2321ad7dc4291451eea4a9c3c/csrc/fwd/sm100_blk128):
`bsa_fwd_sm100_fp8_fused_pv.py` and `bsa_fwd_sm100_fp8.py` map to the same
filenames under `python/cudnn/block_sparse_attention/csrc/fwd/sm100_blk128/`.
The 16-token TMEM row-max helper is incorporated into the private blk128 helper
module. The integration uses cuDNN's existing quantizer, allocating Torch API,
device-specific JIT cache, current-stream execution, and split combine.
Standalone upstream package entry points and native AOT deployment are not
part of this API.

## Acknowledgements

We would like to express our gratitude to [huangyitong.hyt@alibaba-inc.com](mailto:huangyitong.hyt@alibaba-inc.com) and
[wenting.swt@alibaba-inc.com](mailto:wenting.swt@alibaba-inc.com) for providing testing and optimization feedback
throughout the deployment process, which has continuously advanced the BSA kernel
toward Speed of Light.

## Experimental JAX support

`cudnn.ops.block_sparse_attention_forward_jax` and
`cudnn.ops.block_sparse_attention_backward_jax` provide explicit forward/backward;
`cudnn.jax.block_sparse_attention` adds first-order reverse-mode differentiation.
All three accept JAX arrays, eagerly or under `jax.jit`, and require no PyTorch.
`cudnn.jax.block_sparse_attention_forward` and `cudnn.jax.block_sparse_attention_backward`
have been removed; use the `cudnn.ops` names.

The JAX implementation reuses the SM100 blk128 forward, bucketed CSR, backward preprocess,
backward, and gradient conversion kernels. Install `jax[cuda13]` alongside the
frontend (CuTeDSL >=4.7, JAX >=0.9.1). The runtime imports no PyTorch; torch parity tests are separate.

The JAX signatures omit Torch-specific launch options and caller-provided
output buffers. XLA owns the outputs and workspaces.

### Initial contract

| Property | Supported |
| --- | --- |
| Device | All arrays on one CUDA GPU of exactly compute capability 10.0 (SM100); other GPUs may be visible. Single process |
| Data | BF16 Q/K/V/O/dO/dQ/dK/dV; FP32 LSE and accumulation |
| Dimensions | MHA with equal head counts, D=64 or 128; positive sequence lengths divisible by 128 |
| Layout | Compact BHSD or BSHD, independently specialized |
| Sparsity | 128-token blocks, int32 indices `[B,H,Sq/128,C]`, no `block_sizes` |
| Counts | Fixed even `block_sparse_num` in `[2,C]`, or runtime int32 `q2k_block_nums[B,H,Sq/128]` |
| Differentiation | First-order reverse-mode Q/K/V gradients through `block_sparse_attention` |

Variable counts must be in `[1,C]`, or `[0,C]` with
`allow_empty_block_nums=True`. Only each row's active prefix is read. Active
indices must be unique and in `[0,Sk/128)`. An empty row returns zero output,
negative-infinity LSE, and zero query gradient. Unselected K/V receive zero
contributions. Metadata **values are the caller's responsibility**: the runtime
validates shapes, dtypes, and static options, without copying metadata to the
host or synchronizing to inspect it. Invalid values are outside the contract.

SM103/SM110/SM120, blk64, FP16/FP8, GQA, different V dimensions, partial blocks,
variable sequence lengths, causal masks, dropout, sharding, `vmap`, JVP, and
higher derivatives are unsupported. KDA is outside this change. Static options
(including scale and fixed count) specialize compilation; sparse arrays remain
runtime operands. Do not mutate saved forward inputs or metadata before backward.

### Usage

```python
import jax
import jax.numpy as jnp
from functools import partial
from cudnn.jax import block_sparse_attention as attention
from cudnn.ops import (
    block_sparse_attention_forward_jax as forward,
    block_sparse_attention_backward_jax as backward,
)

q = jnp.ones((1, 2, 256, 64), dtype=jnp.bfloat16)
k = jnp.ones((1, 2, 512, 64), dtype=jnp.bfloat16)
v = jnp.ones_like(k)
indices = jnp.broadcast_to(jnp.array([0, 2], jnp.int32), (1, 2, 2, 2))

o, lse = forward(q, k, v, indices, 2)  # eager; asynchronous GPU execution
run = jax.jit(partial(forward, block_sparse_num=2))
result = run(q, k, v, indices)
o, lse = result["o_tensor"], result["lse_tensor"]

dq, dk, dv = backward(jnp.ones_like(o), q, k, v, o, lse, indices, 2)

def loss(q, k, v, indices):
    return attention(q, k, v, indices, 2).astype(jnp.float32).sum()

grad = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))
dq, dk, dv = grad(q, k, v, indices)
dq.block_until_ready()  # needed for host timing, not between GPU operations
```

Explicit forward returns a JAX pytree with `o_tensor`, `lse_tensor`; backward returns `dq_tensor`,
`dk_tensor`, `dv_tensor`. Both support tuple unpacking and dictionary access. Explicit helpers
do not register autodiff rules; use `attention` for `jax.grad`. LSE and metadata
are not differentiable public outputs. Backward must receive the exact matching
forward Q/K/V/O/LSE, sparse pattern, scale, layout, and empty-row option.

### Execution and ownership

Every launch receives XLA's CUDA stream. Forward is one custom call; backward
composes CSR construction, preprocessing, attention backward, and gradient
conversion inside one custom call. XLA owns all outputs and workspaces. CSR
counts and multi-bucket dK/dV accumulators are initialized for each invocation;
preprocessing clears dQ accumulation. No mutable global scratch or public input
buffer is donated. Atomic accumulation does not promise bitwise determinism.

`TensorSpec.mode` presents compact BHSD/BSHD memory in the order expected by each
kernel. It does not rewrite DLPack capsules or insert a Python transpose. XLA may
still insert layout conversions when surrounding operations use incompatible
physical layouts; this is not a universal zero-copy guarantee.

Two cached call builders retain static configuration and kernel objects, never
input arrays or pointers. XLA handles compilation and execution; there is no
separate plan lifecycle. One `custom_vjp` registration connects forward and
backward. `bucket_size_blocks` optionally controls backward query buckets;
the default reuses the torch path's heuristic. Multi-device placement and cache
portability across architectures require further qualification.

The JAX tests live in `test/python/block_sparse_attention/jax` and are collected by the OSS
suite. Torch-free import and gradient checks run in fresh subprocesses. To run
the suite in a torch-free container, pass `--confcutdir=test/python/block_sparse_attention/jax`
so pytest does not load the torch-based parent conftest.
