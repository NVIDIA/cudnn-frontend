# JAX SDPA Python frontend route (experimental)

**Experimental API:** `cudnn.sdpa_jax` and its opt-in JAX integration are subject
to changes in API and feature support.

`cudnn.sdpa_jax` builds attention graphs with the Python frontend and executes
serialized cuDNN backend plans through XLA FFI. It does not call Torch or execute
Python callbacks during a compiled invocation.

JAX is optional. Installing the frontend wheel does not install JAX, and
importing `cudnn` or using the Torch SDPA API does not load the JAX bridge.
Only access to `cudnn.sdpa_jax` requires JAX.

The companion JAX change selects this function for
`jax.nn.dot_product_attention(..., implementation="cudnn")` when
`JAX_CUDNN_SDPA_USE_PYTHON_FRONTEND=1`. Set the variable before tracing.
`0` or unset keeps the existing XLA/cuDNN implementation. Already compiled
executables retain their route. The `xla` and default implementation choices
are unchanged.

The initial native executor supports dense FP16/BF16 attention, MHA/GQA/MQA,
unbatched inputs, scale, causal masking, local windows, additive bias, boolean
masks, padding lengths, and optional LSE. `jax.jit` and reverse-mode AD use
separate forward/backward graphs. Backward returns Q/K/V and bias gradients.
The public LSE is stop-gradient; backward saves FP32 LSE even when it is not
returned. Each configuration remains subject to cuDNN backend support checks.

The native bridge is compiled once using a C++17 compiler, CUDA toolkit, installed
frontend headers, and cuDNN headers/libraries. Set `CUDNN_PATH` for a nonstandard
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

Each numerical case reuses its inputs and upstream gradient with the setting
unset, `0`, and `1`. Numerical parity and deterministic replay are separate tests.

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
