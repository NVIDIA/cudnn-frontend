# Causal Conv1d Decode Update (SM100)

**This FE-OSS API is experimental and subject to change.**

`CausalConv1dUpdateSm100` advances a four-token depthwise causal-convolution
cache in place and emits the fused-SiLU output for one decode step. It targets
the no-bias, width-four short convolution used by GDN/KDA-style linear
attention blocks. It is a standalone native primitive; it does not yet add a
serving-framework integration or make cuDNN the default route. The optional
`cudnn.fla.accelerate_fla(targets="short_conv")` adapter preserves FLA 0.5.2's
decode-update interface and routes only this exact supported subset to the
native primitive; all other configurations retain FLA's original path.

For row `n`, channel `d`, and selected cache slot `s`, the operation is:

```text
updated_state[s, d, :] = [state[s, d, 1], state[s, d, 2],
                          state[s, d, 3], x[n, d]]
output[n, d] = SiLU(sum_j updated_state[s, d, j] * weight[d, j])
```

When `state_indices` is omitted, row `n` selects slot `n`. When it is present,
`s = state_indices[n]`. Indexed slots must be in range and unique within the
decode batch because state is mutated. The kernel checks these properties on
device and traps on violation. The resulting CUDA error is asynchronous and
the failed update is not transactional.

## Supported contract

- GPU: exactly SM100 (compute capability 10.0)
- `x`: contiguous BF16 `[N, D]`
- `weight`: contiguous BF16 `[D, 4]`
- `state`: contiguous BF16 `[S, D, 4]`, updated in place
- `state_indices`: optional contiguous CUDA int32 `[N]`
- output: contiguous BF16 `[N, D]`
- activation: SiLU, always fused
- bias and autograd: unsupported
- pointer alignment: 16 bytes for BF16 tensors and 4 bytes for indices

This narrow contract does not cover a bias term, a returned intermediate state
for speculative decoding, arbitrary convolution width, prefill, training, or a
general Mamba causal-convolution interface. The indexed path is provided for
functional paged-state support; its duplicate-index validation has not been
performance-characterized as a fast path.

For the two measured Qwen3.5 decode signatures `N=128, D=2048` and
`N=128, D=4096`, an unindexed call compiles a two-row CTA specialization. It
loads each channel's four-tap weight once and reuses it across the two rows,
while issuing both rows' state/input loads before arithmetic. Indexed calls and
every other shape retain the original one-row kernel, including its device-side
index validation and channel-tail handling. This route is an internal schedule
choice; it does not change the public API or cache key.

## High-level wrapper

The standard FE-OSS wrapper allocates the output and returns a `TupleDict`:

```python
import torch
from cudnn import causal_conv1d_update_wrapper_sm100

x = torch.randn(8, 2048, device="cuda", dtype=torch.bfloat16)
weight = torch.randn(2048, 4, device="cuda", dtype=torch.bfloat16)
state = torch.randn(8, 2048, 4, device="cuda", dtype=torch.bfloat16)

result = causal_conv1d_update_wrapper_sm100(x, weight, state)
output = result["output_tensor"]
```

`cudnn.causal_conv1d_update(...)` and
`cudnn.ops.causal_conv1d_update(...)` expose the same mutation but return the
output Tensor directly. They are convenient for model-integration shims and
are not drop-in replacements for APIs with different argument order, bias, or
return-state conventions.

Both helpers cache compiled kernels by device, shape, and indexed/non-indexed
signature. The cache is bounded. The first call for a signature performs JIT
compilation, so warm the exact signature before latency measurement or CUDA
Graph capture.

## Class API

Use the class API for explicit compilation, output ownership, and repeated
execution:

```python
import torch
from cudnn import CausalConv1dUpdateSm100

x = torch.empty(8, 2048, device="cuda", dtype=torch.bfloat16)
weight = torch.empty(2048, 4, device="cuda", dtype=torch.bfloat16)
state = torch.empty(8, 2048, 4, device="cuda", dtype=torch.bfloat16)
output = torch.empty_like(x)

op = CausalConv1dUpdateSm100(x, weight, state, output)
op.check_support()
op.compile()

with torch.no_grad():
    op.execute(x, weight, state, output)
```

`execute(..., current_stream=...)` accepts a CUDA driver stream handle. The
high-level helpers allocate their output on that stream as well; they require a
concrete stream handle rather than the `CU_STREAM_PER_THREAD` sentinel. Callers
remain responsible for cross-stream dependencies and synchronization.

## Semantic provenance and benchmarking

Behavioral references are FLA 0.5.2 `ShortConvolution.step` (MIT) and the
public `causal_conv1d_update` contract from Dao-AILab/causal-conv1d at revision
`cd81f0413cad2fc1e6f17e785ac39f59aae690cd` (BSD-3-Clause). No source from
either project is included. The implementation uses CUTLASS/CuTe DSL, inline
PTX, and in-tree NVIDIA FROST primitives.

The two-row scheduling idea was selected from audited internal Kernel Factory
candidate `73d90c7f...`, but that standalone source was not copied into this
module. The FE specialization is independently implemented on top of the
existing native inline-PTX path and keeps FE's architecture, alignment, alias,
index, and cache contracts.

On ComputeLab job `3999943` (B200 SM100, driver `610.57.04`, CUDA `13.0`,
PyTorch `2.13.0+cu130`, CUTLASS DSL `4.7.0`), an interleaved same-process CUPTI
run measured the FE specialization against the preceding FE implementation at
`2.304` versus `2.464` microseconds for `N=128, D=2048`, and `2.848` versus
`3.328` microseconds for `N=128, D=4096` (201 samples per arm and shape). Paired
median speedups were `1.134x` and `1.170x`; the `D=2048:D=4096` 2:1 weighted
geometric mean was `1.146x`. The raw result SHA-256 is
`fa0d3e7d5fc2d7a0ad2f53362ddf21932dfec109cde119d9dadf4e7fdb9e4cbd`.
These are direct kernel-active measurements, not single-node CUDA-graph replay
or end-to-end model latency.

Use `benchmark/causal_conv1d_update_sm100.py` on the target GPU for route-proof,
correctness, and interleaved comparison against FLA. Performance measurements
must include the exact GPU, software environment, shapes, and raw artifacts;
the indexed path should be reported separately from the no-index decode path.
