# JAX SDPA Python frontend route (experimental)

**Experimental APIs:** `cudnn.ops.sdpa_forward_jax` and
`cudnn.ops.sdpa_backward_jax` are subject to changes in API and feature support.

These separate, imperative calls build attention graphs with the Python frontend
and execute serialized cuDNN backend plans through XLA FFI. Both support
`jax.jit`; neither supplies automatic differentiation. The companion JAX
integration supplies a custom VJP that calls forward and backward separately.
Compiled execution does not call Torch or Python callbacks.

JAX is optional. Installing the frontend wheel does not install JAX, and
importing `cudnn` or using the Torch SDPA API does not load the JAX bridge.
Only access to the JAX op functions requires JAX.

The companion JAX change selects these functions for
`jax.nn.dot_product_attention(..., implementation="cudnn")` when
`JAX_CUDNN_SDPA_USE_PYTHON_FRONTEND=1`. Set the variable before tracing.
`0` or unset keeps the existing XLA/cuDNN implementation. Already compiled
executables retain their route. The `xla` and default implementation choices
are unchanged.

The initial native executor supports dense FP16/BF16 attention, MHA/GQA/MQA,
unbatched inputs, scale, causal masking, local windows, additive bias, boolean
masks, padding lengths, and LSE. Each configuration remains subject to cuDNN
backend support checks.

Forward takes Q/K/V in `[B, T, N, H]` / `[B, S, K, H]` layout and returns
`(output, stats)`. Output matches the query shape and dtype. Stats are FP32
log-sum-exp values with shape `[B, T, N]`. Unbatched inputs omit `B`.

Backward takes `(query, key, value, output, grad_output, stats)` and returns
`(dq, dk, dv, dbias)`. Pass the original inputs, the same attention options,
and the output and unmodified FP32 stats saved by forward. `grad_output` must
match output's shape and dtype. Backward does not rerun forward. Gradients
match the original input shapes and dtypes, including reduction of broadcast
bias gradients. `dbias` is `None` when bias is absent. Masks, lengths and stats
have no gradient.

Both calls accept `bias`, `mask`, `scale`, `is_causal`, `query_seq_lengths`,
`key_value_seq_lengths` and `local_window_size`. Bias and boolean masks must
broadcast to `[B, N, T, S]`; lengths are int32 arrays of shape `[B]`.
Backward also accepts `deterministic=True` to select deterministic backend
backward plans.

```python
import cudnn.ops
import jax

options = dict(scale=0.125, is_causal=True)
forward = jax.jit(lambda q, k, v: cudnn.ops.sdpa_forward_jax(q, k, v, **options))
backward = jax.jit(
    lambda q, k, v, o, do, stats: cudnn.ops.sdpa_backward_jax(
        q, k, v, o, do, stats, deterministic=True, **options
    )
)
output, stats = forward(query, key, value)
dq, dk, dv, dbias = backward(query, key, value, output, grad_output, stats)
```

For `jax.grad`, use the companion `jax.nn.dot_product_attention` integration
with the environment setting above. Its optional public LSE keeps JAX's
output dtype and stop-gradient behavior; its backward rule saves the FP32
stats separately. The unreleased `cudnn.sdpa_jax` and `cudnn.ops.sdpa_jax`
aliases are replaced by the two explicit operations.

The native bridge is compiled on first use using a C++17 compiler, CUDA toolkit,
installed frontend headers, cuDNN headers/libraries and XLA FFI headers from
`jaxlib`. A missing `xla/ffi/api/ffi.h` raises `ImportError` before compilation.
Wheel builds do not compile this bridge. Set `CUDNN_PATH` for a nonstandard
backend installation. Build products are cached under the frontend kernel cache.
Input/output layouts are fixed at compilation. XLA allocates workspace and passes
its CUDA stream. Native plans retain their host descriptors for the process
lifetime; do not use this prototype with unbounded shape churn. Process-specific
FFI targets are not portable exports or persistent compilation-cache artifacts.

This is an initial integration. Frontend kernel-plan export, packed THD, sinks,
dropout, quantization, paged KV, distributed sharding, JVP, and higher-order
AD are not implemented here. Native backend plans are explicitly selected;
this must not be reported as coverage of the frontend's CuTe DSL engines.

Run the paired tests with the companion JAX checkout on `PYTHONPATH`:

```bash
cd test/python
python -m pytest sdpa/jax/test_sdpa_jax.py -m L0 -q
```

Each numerical case compares the separate forward/backward calls with
`jax.nn` using identical inputs and upstream gradients with the setting unset,
`0`, and `1`, plus an XLA reference. Numerical parity and deterministic replay
are separate tests.

The paired suite is not the full JAX regression suite. From the companion JAX
checkout, run the upstream SDPA tests with each setting:

```bash
JAX_NUM_GENERATED_CASES=1000000 JAX_CUDNN_SDPA_USE_PYTHON_FRONTEND=0 \
  python -m pytest tests/nn_test.py tests/fused_attention_stablehlo_test.py \
  tests/fused_attention_stablehlo_multigpu_test.py -k DotProductAttention -q -ra
```

Repeat with `1` and with the route variable unset. The large generated-case
limit covers the full parameter grid; JAX otherwise samples these cases.
Public `jax.nn` tests exercise route selection. Tests that call the internal
cuDNN primitive directly remain legacy regression checks under either setting.
Report hardware and runtime skips separately from passes.
