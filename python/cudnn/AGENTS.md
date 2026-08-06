# python/cudnn — Agent Guide

The `cudnn` Python package: pybind11-backed graph API plus pure-Python **frontend-only OSS kernels** (CuTeDSL). See `README.md` in this directory for the package inventory and [../../AGENTS.md](../../AGENTS.md) for build/test commands.

## Import-time rules (the most common way to break this package)

- `import cudnn` must work **without** torch/cutlass/cuda-python installed. Everything that needs them is exported lazily via `_LAZY_OPTIONAL_IMPORTS` in `__init__.py` — a module-level `__getattr__` imports the submodule on first attribute access and re-raises failures as `ImportError` pointing at `pip install nvidia-cudnn-frontend[cutedsl]`.
- Never add an eager `import torch` / `import cutlass` to `__init__.py` or anything it imports transitively. `api_base.py` itself imports them at top level, which is why kernel classes must only be reachable through the lazy table.
- Reuse the existing `[cutedsl]` extra (`pyproject.toml` optional-dependencies) unless a kernel truly needs a new package.

## Frontend-only kernel package layout

```
python/cudnn/<operation>/            # or sdpa/<direction>/, gemm/cutedsl/<layout>/<fusion>/
├── __init__.py                      # exports API class + wrapper via __all__
├── api.py                           # APIBase subclass + <operation>_wrapper() function
└── <kernel_module>.py               # CuTeDSL kernel implementation(s); some families use csrc/ per-arch trees
```

All GEMM fusions live under `gemm/`, grouped by how the operands are laid out:

```
python/cudnn/gemm/
├── cutedsl/
│   ├── dense/<fusion>/              # amax, dsrelu, proj_rope_mxfp8, srelu, swiglu
│   ├── grouped/<fusion>/            # dglu, dsrelu, dswiglu, glu, glu_hadamard,
│   │                                #   quant, srelu, swiglu, unfused, wgrad
│   └── discrete_grouped/<fusion>/   # dswiglu, swiglu (per-expert weight pointers)
├── ops/                             # backend-independent torch custom-op contracts
└── reference/                       # pure-PyTorch MATMUL/POINTWISE correctness engine
```

Shared helpers (schedulers, metadata utils, e.g. `gemm/cutedsl/grouped/moe_*.py`) stay internal to the family package — never exported through `cudnn`.

## The APIBase contract (`api_base.py`)

Every OSS kernel API extends `APIBase` and implements:

- `check_support() -> bool` — validate dtype/shape/stride/arch/config via the `_check_tensor_*` / `_value_error_if` helpers; must set `self._is_supported`. Works on `TensorDesc` (metadata-only tensors), so it runs without GPU storage.
- `compile()` — calls `self._ensure_support_checked()`, builds and `cute.compile`s the kernel, caches in `self._compiled_kernel`.
- `execute(..., current_stream=None)` — runs the cached kernel.

`__call__` = compile-if-needed + execute. High-level wrappers (`<op>_wrapper_sm100(...)`) allocate outputs and return a **`TupleDict`** (dict that also unpacks as a tuple) with stable, documented key order. FP4x2 packing: use `_tensor_shape`/`_tensor_stride`, which double the innermost dim when `interpret_uint8_as_fp4x2` is set.

## Adding a new frontend-only API — required checklist

1. Kernel package under the closest existing family (layout above).
2. `APIBase` subclass + wrapper in `api.py`.
3. Exports: family `__init__.py` `__all__` **and** `_LAZY_OPTIONAL_IMPORTS` in `python/cudnn/__init__.py`; register any new package dir in `pyproject.toml` packages list.
4. Docs: page under `docs/fe-oss-apis/` (family subdir) + link it from `docs/fe-oss-apis/overview.md`.
5. Tests: `test/python/fe_api/<family>/test_<op>.py` (+ `_utils.py`/reference), covering check_support pass/fail and numerical reference comparison.

The `cutedsl-kernel-integration` skill (`skills/cutedsl-kernel-integration/`) documents this workflow in detail, including how to classify a kernel into a family — follow it for any kernel integration.

## Other notes

- `wrapper.py` `Graph` context manager (the pythonic graph builder) requires cuDNN backend ≥ 9.12 (`backend_version() >= 91200`) and builds plans on `__exit__`.
- Torch custom ops live in `experimental/ops/` (pattern doc: `docs/adding_torch_custom_ops.md`); they cache built graphs per config and use stable `_UIDs` enums.
- dtype conversions go through `datatypes.py`, which probes torch/cutlass availability lazily — keep it that way.
- Formatting: black, line length 160.
