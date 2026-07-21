# Grouped GEMM (SM100 BF16)

**This is an experimental API and subject to change.** It requires an NVIDIA
SM100-or-newer GPU and the optional CuTe DSL dependencies:

```bash
pip install nvidia-cudnn-frontend[cutedsl]
```

`GroupedGemmSm100` and `grouped_gemm_wrapper_sm100` implement the neutral,
unfused BF16 MoE grouped GEMM. They support dense stacked expert weights or a
device pointer per expert, optional bias, optional materialization of the
intermediate `C`, and static or dynamic tile scheduling.

## Operation

Let `padded_offsets[g]` be the exclusive end of expert `g`'s contiguous row
range and let `begin_g` be zero for the first expert or
`padded_offsets[g - 1]` otherwise. For rows in `[begin_g, padded_offsets[g])`:

```text
G_g = alpha[g] * A_g @ B_g.T
C_g = G_g + prob_g * bias[:, g]     # when bias is present
D_g = C_g
```

Without bias, `C_g = G_g` and `D_g = prob_g * G_g`. Accumulation is FP32;
`C` and `D` may independently use BF16, FP16, or FP32.

## Tensors and layouts

For total padded rows `M`, inner dimension `K`, output dimension `N`, and `L`
experts:

| Tensor | Shape | Required stride / dtype |
| --- | --- | --- |
| `A` | `(M, K, 1)` | `(K, 1, M*K)`, BF16 |
| dense `B` | `(N, K, L)` | `(K, 1, N*K)`, BF16 |
| discrete `b_ptrs` | `(L,)` | contiguous CUDA int64 pointers to `(N, K)` BF16 matrices |
| `padded_offsets` | `(L,)` | `(1,)`, CUDA int32 cumulative ends |
| `alpha` | `(L,)` | `(1,)`, CUDA FP32 |
| `prob` | `(M, 1, 1)` | `(1, 1, 1)`, CUDA FP32 |
| optional `bias` | `(N, L)` | `(1, N)`, BF16/FP16/FP32 |
| `C`, `D` | `(M, N, 1)` | `(N, 1, M*N)`, BF16/FP16/FP32 |

`M` and every cumulative offset must be 256-aligned. Dense weights are
K-major. Discrete mode uses `b_major="k"`, `n=N`, and
`b_dtype=torch.bfloat16`.

The pointer-array tensor must be contiguous, non-null, eight-byte aligned, and
on the same device as `A`; each target pointer must satisfy the kernel's
alignment contract. The API records the pointer-array tensor on the launch
stream. The caller must keep every pointed-to expert allocation alive and must
not modify or free it until that stream completes.

## Wrapper API

Dense mode:

```python
import cudnn
import torch

result = cudnn.grouped_gemm_wrapper_sm100(
    a_tensor=a,
    padded_offsets=padded_offsets,
    alpha_tensor=alpha,
    b_tensor=b,
    bias_tensor=bias,
    prob_tensor=prob,
    c_dtype=torch.float32,
    d_dtype=torch.bfloat16,
    generate_c=True,
    use_dynamic_sched=True,
)
d, c = result
assert d is result["d_tensor"]
assert c is result["c_tensor"]
```

Discrete mode changes only the weight arguments:

```python
result = cudnn.grouped_gemm_wrapper_sm100(
    a_tensor=a,
    padded_offsets=padded_offsets,
    alpha_tensor=alpha,
    b_ptrs=b_ptrs,
    n=N,
    b_dtype=torch.bfloat16,
    b_major="k",
    bias_tensor=bias,
    prob_tensor=prob,
)
```

The `TupleDict` order is exactly `d_tensor`, then `c_tensor`. `c_tensor` is
`None` unless `generate_c=True`; the kernel still uses an internal C buffer
when it is needed for execution.

## Class API

The class API takes representative sample tensors, then follows
`check_support()` -> `compile()` -> `execute()`:

```python
op = cudnn.GroupedGemmSm100(
    sample_a=a,
    sample_c=c,
    sample_d=d,
    sample_padded_offsets=padded_offsets,
    sample_alpha=alpha,
    sample_b=b,
    sample_bias=bias,
    sample_prob=prob,
    acc_dtype=torch.float32,
    generate_c=True,
    use_dynamic_sched=False,
)
assert op.check_support()
op.compile()
op.execute(
    a_tensor=a,
    c_tensor=c,
    d_tensor=d,
    padded_offsets=padded_offsets,
    alpha_tensor=alpha,
    b_tensor=b,
    bias_tensor=bias,
    prob_tensor=prob,
)
```

For a discrete class instance, replace `sample_b` with
`num_experts=L`, `b_shape=(N, K)`, `b_dtype=torch.bfloat16`, and pass
`b_ptrs` to `execute()`.

## Scheduling, caching, and errors

- `use_dynamic_sched=False` uses the static scheduler.
- `use_dynamic_sched=True` compiles a dynamic-M callable and reuses it for
  compatible M values; discrete mode also allocates the per-expert tensor-map
  workspace. Wrapper cache keys retain dtype, layout, expert count, optional
  features, scheduler choice, output policy, tile/cluster shape, and overlap
  margin.
- Dense and discrete weight arguments are mutually exclusive. Invalid shapes,
  strides, dtypes, devices, alignment, offsets, pointer entries, output
  descriptors, tiles/clusters, or a target below SM100 raise `ValueError` or
  `RuntimeError` before launch.
- Fused GLU, dGLU, and WGrad APIs select BF16 from BF16 operands while keeping
  their existing FP4/FP8 block-scaled backends. For BF16 on those fused APIs,
  scale-factor controls are `None`; see their operation pages for the exact
  dispatch and return contracts.
