# python/cudnn — Agent Guide

The `cudnn` Python package: pybind11-backed graph API plus pure-Python **frontend-only OSS kernels** (CuTeDSL). See `README.md` in this directory for the package inventory and [../../AGENTS.md](../../AGENTS.md) for build/test commands.

## Import-time rules (the most common way to break this package)

- `import cudnn` must work **without** torch/cutlass/cuda-python installed. Everything that needs them is exported lazily via `_LAZY_OPTIONAL_IMPORTS` in `__init__.py` — a module-level `__getattr__` imports the submodule on first attribute access and re-raises failures as `ImportError` pointing at `pip install nvidia-cudnn-frontend[cutedsl]`.
- Never add an eager `import torch` / `import cutlass` to `__init__.py` or anything it imports transitively. `api_base.py` itself imports them at top level, which is why kernel classes must only be reachable through the lazy table.
- Reuse the existing `[cutedsl]` extra (`pyproject.toml` optional-dependencies) unless a kernel truly needs a new package.

## Hard rules

Numbered so reviews can cite them; the list grows — append, never renumber.

**Rule 1 — `execute()` is a zero-surprise hot path: validate, never convert, never allocate.**

- **No implicit conversions.** Never `.to(dtype)`, and never a `reshape()` that can
  copy, on an execute argument: both silently allocate and launch a kernel per
  call, and the fresh pointer breaks CUDA-graph capture. Worse, for an *output*
  tensor a reshape copy swallows the kernel's write. Validate dtype / shape /
  contiguity and bind a true view (`.view()` or a checked `reshape`), raising
  `ValueError` otherwise — see `_checked_lse_view` / `_checked_sinks_1d` /
  `_checked_seq_lens` in `sdpa/fwd/api_dsl.py`.
- **No per-execute allocations.** No `torch.empty`/`torch.zeros` inside
  `execute()`: scratch is carved from the caller's workspace
  (`scratch_workspace_bytes()` contract), and a dead ABI slot may use a
  one-time cached dummy (`_dummy`) at most. Prefer compiling the unused
  operand out entirely (CuTeDSL specializes on `None` via
  `cutlass.const_expr` — see the SM120 SDPA kernel's optional lse/sinks).
- **Init-time flags are compile-time specializations; `execute()` must match
  them exactly, in both directions.** A required-but-missing tensor must
  raise, never fall back to a zeros dummy (zeros sinks change the softmax
  denominator; zeros seq lens mask every row — silently wrong output). A
  provided-but-uncompiled tensor must also raise, never be silently ignored.
- **No degenerate-path fixups.** Runtime-degenerate inputs (e.g. all-zero
  THD ``seq_kv_lens``) go through the kernel's own dead-row path — never
  re-implemented adapter-side with `fill_`/`copy_` writes (surprise kernel
  launches, and a second copy of the semantics that can drift). If a packed
  extent would be zero, bind a never-dereferenced dummy view over storage
  the contract already guarantees.

**Rule 2 — `execute()` launches exactly the kernels the plan promised:
serve the declared layout natively, or decline — never adapt.**

Rule 1 bans implicit conversions and allocations; this rule bans the loophole
that survives its letter: "helpful" adapter-side work that makes an
unsupported input runnable.

- **No hidden kernel launches.** A gather/scatter "normalization" copy, a
  `.contiguous()`, a layout repack, a scatter-back after the launch — each is
  an extra kernel that silently changes the measured perf profile per
  configuration. **Carving the copy's scratch from the caller's workspace
  does NOT make it acceptable**: Rule 1's workspace-carve exemption covers
  metadata buffers and dead-slot dummies, never data-tensor copies.
- **Can't address the declared layout natively? Decline in
  `check_support()`** (`NotImplementedError` naming the offending tensor and
  its strides) so the Router picks an engine that honors the declaration.
  Silent wrong results are the worst failure mode; a silent slow path is the
  second worst — both hide behind a green test. See
  `_thd_check_strides_native` in `sdpa/fwd/api_dsl.py`.
- **Precedent is not a license.** The SM100 dense path's compact-BSHD
  normalization (`dense_layout_ok`: "one gather/scatter copy otherwise")
  predates this rule and is grandfathered — do not cite it to justify a new
  copy path, and treat migrating it to serve-or-decline as open cleanup.
- The flip side of declining: whatever `check_support()` ACCEPTS, the kernel
  must address natively (layout-driven offset math, strides encoded in TMA
  descriptors) — acceptance is a promise about the execute path, not about
  what the adapter can patch up.

**Rule 3 — `execute()` never reads device memory to the host.**

Rules 1 and 2 both cite CUDA-graph capture as the reason for what they ban, but
neither names the thing that breaks it most directly: a device-to-host read.

- **No `.item()` / `.tolist()` / `.cpu()` / `.to("cpu")` / `.numpy()` /
  `float(tensor)` / `int(tensor)` / `torch.is_nonzero`**, and no branch or
  f-string that forces one, on an execute argument or anything derived from one.
  A D2H read makes `execute()` synchronous — the whole point of an async launch
  API is gone. **Nor may it block**: no `torch.cuda.synchronize()`, no
  stream/event `synchronize()`. A sync reads nothing but costs the same.
- **It is a functional gap, not a slow path.** A blocking D2H during stream
  capture is illegal, so a path that does one **cannot be CUDA-graph captured
  at all** — which is how every inference stack runs decode.
- **Its cost is the queue, not the transfer.** Measured on SM100: one
  `.tolist()` costs 11 µs against a drained queue, 2.6 ms behind 16 queued
  matmuls. Any figure you measure in a microbenchmark is the floor.
- **If a device value must shape the launch**, pass its pointer and dereference
  in-kernel, or compile on an envelope and let the kernel read the real extent
  from device metadata (the f16 prefill kernels already do this for head dims).
- **A validation that needs a device read is not a validation.** Decline the
  declaration in `check_support()` — per Rule 2, the graph says what it will
  hand you — or assert in-kernel. Reading lengths back to decide whether to
  raise buys nothing: the Router had to choose an engine before any buffer
  existed.

Known violations, all pre-existing and each needing a kernel-side change, so
none is precedent:

- The FP8/MXFP8 `seq_len_q` guard in `sdpa/fwd/engines.py`. This one cannot be
  lifted to `check_support()`: `use_padding_mask=True` requires a `seq_len_q`
  tensor even when only KV is padded, so no static rule separates "declares
  per-batch Q lengths" from "the lengths are actually short" — declining the
  declaration would drop the KV-only-padding population these kernels serve
  correctly. It goes away when the FP8 kernels get the epilogue trim; until
  then the read is what keeps a short length from being silently ignored.
- `cu_seqlens_{q,k}.to(dtype=..., device="cpu")` in the SM80 packed-THD backward
  (`sdpa/bwd/kernels/bprop_f16_sm80.py`). Reachable only through the standalone
  wrapper: the registered `sdpa_bwd_sm80` spec declares `thd=False`, so
  `graph.execute()` does not route here. Still a violation, and it is the one to
  fix first if that spec ever gains THD.

When auditing this list, grep for the ARGUMENT, not the call shape:
`device="cpu"` finds `to(dtype=..., device="cpu")`, which `to(device="cpu")`
misses.

**Rule 4 — compile keys are PLAN-TIME-ONLY: never key a kernel compile on
runtime data values.**

`cute.compile` takes seconds. Anything an execute path feeds into a
compile-cache key (an `lru_cache`d `compile()` wrapper, a template parameter,
a fake-tensor extent) must be derivable from the graph declaration alone —
tensor dtypes, declared strides, head counts, head dims, flags. Values read
out of runtime tensors (THD packed token totals, max sequence lengths, batch
contents) change every step under continuous batching, so a key that includes
them degenerates into a fresh multi-second compile per `execute()` — a
pathology that no correctness test catches (issue #552 is the case study:
`sq=t_q, skv=t_kv` in the THD compile key). Rule 3 bans the read that feeds
such a key; this rule bans the key itself — a runtime value that arrives
legally (a caller-passed host scalar, an `int(tensor.shape[...])`) still must
not become a compile key.

- **Runtime extents compile DYNAMIC.** Use `cute.sym_int()` in the fake
  tensors (one symbol per ragged group) so one compiled artifact re-binds any
  total; runtime scalars the launch needs (grid extents like THD `max_sq`)
  are `cutlass.Int32` call arguments, never compile parameters.
- **Derived values count.** A stride tuple whose batch stride is
  `t_q * token_stride` smuggles the runtime total into the key just as
  surely as `sq=t_q` — normalize it out (zero the never-stepped batch
  stride, rebuild it symbolically kernel-side).
- **Compile at plan time, re-bind at execute.** With a plan-time-only key
  there is no reason to defer: `compile()` builds the artifact once and the
  execute path's cached call must be a guaranteed hit. Guard it with a
  cache-miss regression test (see
  `test_dsl_sm100_thd_compile_key_plan_time_only`), not by inspection.
- **Known open cleanup (issue #604)**: the SM80 engines' `_compile_cached`
  (#493) still keys `SQ`/`SKV` under `THD_VARLEN` — migrate it to dynamic
  token extents like the SM100/SM120 THD compiles rather than copying its
  pattern.

**Rule 5 — every torch operation on the execute path is ordered on the
LAUNCH stream, never implicitly on torch's current stream.**

The kernel launches on the stream carried by the execute-time handle
(`ExecutionContext.stream`), but torch enqueues work — H2D metadata uploads,
buffer resets (`zero_()`), post-kernel reductions (`div_()`, `copy_()`),
and the caching allocator's stream-tagging of fresh blocks — on
`torch.cuda.current_stream()`. When the two differ, the prep and the kernel
race (PR #543 is the case study: the THD `[seq_kv | cu_q | cu_k]` upload vs
the kernel that reads it).

- **Resolve the launch stream FIRST**, before any torch work in the execute
  path, and run every torch op (including allocator calls: workspace-less
  fallback allocations, cached-dummy first use) inside
  `_torch_stream_context(current_stream, device)` — see the fp8/mxfp8 amax
  resets and both `_execute_thd` paths in `sdpa/fwd/api_dsl.py`.
- **Consumers too, not just producers**: anything reading what the kernel
  wrote (`amax_o.div_()`, an O scratch copy-back) belongs on the launch
  stream for the same reason.
- The PyTorch-integration path launches on torch's current stream, where the
  context is a no-op — the race only bites direct graph-API users with an
  explicit handle stream, which is exactly why tests miss it. Order the work
  by construction rather than relying on the common case.


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
