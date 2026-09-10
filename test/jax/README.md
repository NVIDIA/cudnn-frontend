# Isolated JAX tests

Run with one visible SM100 GPU and the edited frontend installed:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python -m pytest test/jax -q
```

Requires CuTeDSL >=4.7 and JAX >=0.9.1 with CUDA support. The suite does not load
`test/python/conftest.py`. Fresh-process tests block every torch import, including
when torch and TVM-FFI are installed. Keep torch parity tests under
`test/python/fe_api/bsa/test_BSA_jax_parity.py`.

The suite covers output/gradient numerics, both physical layouts, variable and
empty rows, multiple backward buckets, asynchronous metadata changes, input
immutability, result pytrees, unsupported inputs, and initialized-output argument
ordering in the shared bridge. Compiled forward HLO checks apply to compact
standalone inputs; they do not promise arbitrary surrounding JAX graphs avoid
layout conversions.

Run this directory separately from torch tests in GPU CI. The repository's public
GitHub workflows currently provide no GPU test job; adding a qualified runner is
still needed for automatic JAX qualification.

For JAX/torch/GPU timing comparisons and known overhead findings, see the
[BSA benchmark](../../benchmark/bsa/README.md). Run it separately: it imports torch.
