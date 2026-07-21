# Grouped GEMM + WGrad (Unified)

`GroupedGemmWgradSm100` and `grouped_gemm_wgrad_wrapper_sm100` are experimental
SM100+ APIs for grouped MoE weight gradients. The same public surface dispatches
BF16 inputs to the BF16 kernel and preserves the legacy FP4/FP8 block-scaled
backend.

Install the optional CuTe DSL dependencies before importing either API:

```bash
pip install nvidia-cudnn-frontend[cutedsl]
```

## Operation

For expert `e`, let `begin = 0` for the first expert and
`begin = offsets_tensor[e - 1]` otherwise, and let
`end = offsets_tensor[e]`. The API computes

```text
Wgrad[e] = A[:, begin:end] @ B[begin:end, :]
```

When `accumulate_on_output=True`, that result is accumulated into the existing
output. The caller must therefore initialize every output allocation. When it
is false, the kernel overwrites the output; an empty expert produces zero.

## BF16 contract

The BF16 backend accepts:

| Argument | Shape | Supported stride/major | Dtype |
| --- | --- | --- | --- |
| `a_tensor` | `(hidden, tokens_sum)` | `(tokens_sum, 1)` K-major or `(1, hidden)` M-major | `torch.bfloat16` |
| `b_tensor` | `(tokens_sum, intermediate)` | `(1, tokens_sum)` K-major or `(intermediate, 1)` N-major | `torch.bfloat16` |
| `offsets_tensor` | `(num_experts,)` | contiguous `(1,)` | `torch.int32` |
| dense `wgrad_tensor` | `(num_experts, hidden, intermediate)` | `(hidden * intermediate, intermediate, 1)` | BF16, FP16, or FP32 |
| one discrete output | `(hidden, intermediate)` | `(intermediate, 1)` | BF16, FP16, or FP32 |
| `wgrad_ptrs` | `(num_experts,)` | contiguous `(1,)` | `torch.int64` |

`offsets_tensor` is a non-decreasing cumulative sum. Every expert token count
(`offsets[e] - offsets[e - 1]`) must be a multiple of 256, and the final offset
must equal `tokens_sum`. Inputs, metadata, and outputs must reside on the same
CUDA device and satisfy the API's alignment checks.

BF16 uses FP32 accumulation and requires `sf_vec_size=16`. Pass `None` for
`sfa_tensor`, `sfb_tensor`, `global_scale_a`, and `global_scale_b`. BF16 rejects
every non-`None` scale or global-scale control with `ValueError`; it never falls
through to another backend. Only a supported FP4/FP8 operand pair selects the
legacy block-scaled backend, which continues to support its existing scale
tensors and global scales.

`input_order` describes how the token dimension is stored:

- `"tensor2d"` (default) uses one global 2-D tensor and its declared strides.
- `"tensor_ragged"` uses per-expert K-contiguous blocks concatenated in memory.
  In this mode only each input's unit-stride axis is meaningful; non-unit host
  strides are ignored when per-expert TMA descriptors are built.

### Output modes

With `output_mode="dense"`, provide or let the wrapper allocate the contiguous
stacked `wgrad_tensor`. `wgrad_ptrs` is forbidden.

With `output_mode="discrete"`, either:

- omit both output arguments and let the wrapper allocate a stacked tensor and
  construct an internal pointer array; or
- provide a CUDA `torch.int64` `wgrad_ptrs` array containing one non-null,
  16-byte-aligned output address per expert.

For explicit pointer-only output, `result["wgrad_tensor"]` is `None`. The caller
owns all pointed-to output allocations and must keep both those allocations and
the pointer tensor alive until work on `current_stream` completes. The API
records the pointer tensor on the launch stream, but it cannot manage the
lifetime of allocations represented only by integer addresses.

The wrapper always returns `TupleDict(wgrad_tensor=...)`; it contains exactly
one item and supports either keyed access or tuple unpacking.

## Block-scaled contract

The legacy block-scaled backend is selected only by a supported matching FP4/FP8
operand pair. It preserves the pre-existing scale-factor contract: provide
`sfa_tensor` and `sfb_tensor`, and provide `global_scale_a` and
`global_scale_b` where the selected low-precision format requires them. BF16
does not reinterpret these controls; it rejects them instead.

## API usage

### BF16

#### Wrapper

Dense BF16 output:

```python
import cudnn
import torch

result = cudnn.grouped_gemm_wgrad_wrapper_sm100(
    a_tensor=a_tensor,
    b_tensor=b_tensor,
    sfa_tensor=None,
    sfb_tensor=None,
    offsets_tensor=offsets_tensor,
    output_mode="dense",
    wgrad_dtype=torch.bfloat16,
    input_order="tensor2d",
)
wgrad_tensor = result["wgrad_tensor"]
```

Discrete BF16 outputs owned by the caller:

```python
expert_outputs = [
    torch.empty(
        (hidden, intermediate), dtype=torch.bfloat16, device="cuda"
    )
    for _ in range(offsets_tensor.numel())
]
wgrad_ptrs = torch.tensor(
    [output.data_ptr() for output in expert_outputs],
    dtype=torch.int64,
    device="cuda",
)

result = cudnn.grouped_gemm_wgrad_wrapper_sm100(
    a_tensor=a_tensor,
    b_tensor=b_tensor,
    sfa_tensor=None,
    sfb_tensor=None,
    offsets_tensor=offsets_tensor,
    output_mode="discrete",
    wgrad_ptrs=wgrad_ptrs,
    wgrad_dtype=torch.bfloat16,
    input_order="tensor_ragged",
)
assert result["wgrad_tensor"] is None
```

#### Reusable class lifecycle

The class API requires output descriptors at construction and output storage at
execution. This dense BF16 example compiles once and accepts later calls with a
different `tokens_sum` when static dimensions, dtypes, majors, and configuration
remain compatible:

```python
op = cudnn.GroupedGemmWgradSm100(
    sample_a=a_tensor,
    sample_b=b_tensor,
    sample_sfa=None,
    sample_sfb=None,
    sample_offsets=offsets_tensor,
    sample_wgrad=wgrad_tensor,
    acc_dtype=torch.float32,
    input_order="tensor2d",
)
op.check_support()
op.compile()
op.execute(
    a_tensor=a_tensor,
    b_tensor=b_tensor,
    sfa_tensor=None,
    sfb_tensor=None,
    offsets_tensor=offsets_tensor,
    wgrad_tensor=wgrad_tensor,
)
```

For a discrete class instance, replace `sample_wgrad` with
`sample_wgrad_expert=expert_outputs[0]`, `num_experts`, `wgrad_shape`, and
`wgrad_dtype`, then pass `wgrad_ptrs` to `execute`.

### Block-scaled

#### Wrapper

```python
result = cudnn.grouped_gemm_wgrad_wrapper_sm100(
    a_tensor=a_tensor,
    b_tensor=b_tensor,
    sfa_tensor=sfa_tensor,
    sfb_tensor=sfb_tensor,
    offsets_tensor=offsets_tensor,
    output_mode="dense",
    wgrad_dtype=torch.bfloat16,
    input_order="tensor_ragged",
)
```

#### Reusable class lifecycle

```python
op = cudnn.GroupedGemmWgradSm100(
    sample_a=a_tensor,
    sample_b=b_tensor,
    sample_sfa=sfa_tensor,
    sample_sfb=sfb_tensor,
    sample_offsets=offsets_tensor,
    sample_wgrad=sample_wgrad_tensor,
    acc_dtype=torch.float32,
)
assert op.check_support()
op.compile()
op.execute(
    a_tensor=a_tensor,
    b_tensor=b_tensor,
    sfa_tensor=sfa_tensor,
    sfb_tensor=sfb_tensor,
    offsets_tensor=offsets_tensor,
    wgrad_tensor=wgrad_tensor,
)
```

## Scheduling, cache, and errors

The BF16 kernel uses dynamic persistent scheduling. The token dimension is
compiled dynamically, and the wrapper cache abstracts the token-sized axes of A
and B while retaining static dimensions, layouts, dtypes, output descriptors,
tiling, cluster shape, input order, and accumulation mode in its key. A changed
static contract creates a different cached operator or fails validation.

The APIs reject unsupported dtypes or layouts, malformed/unaligned offsets or
pointers, mixed devices, forbidden BF16 scale controls, unsupported tiling, use
before `compile()`, unavailable CUDA, and devices below SM100. Support and
validation errors are reported as `ValueError` or `RuntimeError`; callers should
not rely on this experimental API remaining source-compatible across releases.
