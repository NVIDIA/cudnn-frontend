# cuDNN-FE CuteDSL kernels from JAX (MVP / draft)

**Purpose of this MVP:** show the *actual per-kernel cost* of exposing a CuteDSL
kernel to JAX as a **real primitive** (composes with `@jax.jit`, no host callback),
so the FE / TE-JAX folks can feel the shape before we commit to it. `gemm_amax`
is the first kernel because its output shapes come entirely from input shapes —
no data-dependent shape, so it isolates the pure bridge boilerplate.

> Status: **unverified end-to-end.** Needs SM100 + `jax` + `jax-tvm-ffi`, plus the
> two GOTCHAs below resolved on hardware. This is a direction preview (draft MR).

## Architecture — FE owns a stable API, tvm-ffi is internal transport

```
JAX user / TE-JAX
   |  cudnn.experimental.jax.gemm_amax(...)      <-- STABLE, FE-owned contract
   v
_bridge.py  (register_ffi_target + version guard)   <-- written ONCE
   v
jax-tvm-ffi  ->  XLA custom call                     <-- swappable transport
   v
CuteDSL kernel compiled with --enable-tvm-ffi        <-- SAME binary torch calls
```

Why this split (not "hand TE the raw kernel"):
- The kernel's tvm-ffi ABI is **not** a stable contract — it drifts. Keeping it
  behind an FE-owned op means a drift is a one-file pin bump here, invisible to
  callers. If TE built on the raw kernel, every drift would break TE directly and
  FE couldn't see it.
- The transport is replaceable: the public op signature is the contract, so
  `jax-tvm-ffi` could later be swapped for a hand-written XLA FFI handler with
  zero caller impact.

## What is / isn't in scope here

- **In:** the JAX bridge boilerplate — abstract-eval (static output shapes),
  ffi_call, output/stream/init handling, the reusable registration helper.
- **Out (orthogonal):** de-torching the kernels (the separate "remove torch"
  effort). Until that lands, the op takes a `_sample_torch` shim to drive the
  kernel's existing `check_support()/compile()`. That collapses to plain
  shape/dtype descriptors once de-torching is done.
- **Out:** autograd. Forward-only, but a complete forward boundary.

## Per-kernel boilerplate (the number people asked for)

Everything framework-neutral is in `_bridge.py` (~90 lines, written once). Adding
a kernel = one file like `gemm_amax.py`: a wrapper that reorders `(rets, args)`
into the kernel's parameter order (~3 lines), one `register_once`, and a JAX-facing
op with shape inference + `ffi_call` (~30 lines). For a dense, shape-from-inputs
kernel that is **~40–60 lines and no C++**. Kernels with data-dependent output
shapes (grouped / MoE / sparse) additionally need a padded/offsets abstract-eval —
a real per-kernel design item, not boilerplate.

## Open GOTCHAs (hardware, not boilerplate — the PoC must nail these)

1. **stream** — the kernel compiles with `use_tvm_ffi_env_stream=False`; the JAX
   path needs `True` so the CUDA stream comes from the tvm-ffi env stream that
   jax-tvm-ffi sets from XLA. One flag in the kernel's `compile()` — the only
   kernel-side edit this MVP needs.
2. **amax init** — XLA output buffers are uninitialized; the kernel atomicMax-es
   into amax and needs it pre-filled with `-inf`. Handled here via a donated input
   `input_output_alias`ed onto the output.
3. **layout** — the compiled kernel is specialized to the sample's strides; XLA
   hands row-major buffers by default. Declare operand/result layouts on
   `ffi_call`, or match input layout. Left for the PoC.

## Run

```bash
pip install 'jax[cuda13]' jax-tvm-ffi                 # apache-tvm-ffi pinned in _bridge.py
export PYTHONPATH=build
pytest test/python/experimental/jax/test_gemm_amax_jax.py -q   # skips if no SM100 / no jax
```

<sub>note to self: claude design session — "cudnn-FE CuteDSL kernels -> JAX (tvm-ffi) MVP".
cwd /home/scratch.yanxu_libs/cudnn_frontend</sub>
