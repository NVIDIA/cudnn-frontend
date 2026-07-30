# Remove PyTorch from the CuTeDSL Dependency Extra

## Goals and Public Behavior

- Remove both `torch` and `torch-c-dlpack-ext` from the `cutedsl` optional dependency in `pyproject.toml`; do not add a replacement PyTorch extra. Both must be removed because `torch-c-dlpack-ext` itself declares `torch` as a dependency.
- Keep `cudnn` and every module under `python/cudnn` importable without PyTorch when their other dependencies are installed. Missing PyTorch must be reported only when a PyTorch-dependent entry point is invoked.
- Add and publicly export `cudnn.TorchNotAvailableError`, subclassing `ImportError`, so callers can catch the missing-framework case without catching unrelated runtime failures. All PyTorch-dependent entry points must raise this error with the message: `"<feature> requires PyTorch, but PyTorch is not installed. Install a compatible PyTorch distribution separately; nvidia-cudnn-frontend[cutedsl] does not install it."`
- Preserve Python-visible results and behavior when PyTorch is installed, except that the optional AOT DLPack accelerator is no longer installed automatically.
- Keep JAX support out of scope, while structuring dependency detection and dtype conversion so adding another framework is additive.

## Implementation Changes

### Framework dependency boundary

- Add an empty `python/cudnn/_deps/__init__.py` and a per-framework dependency module at `python/cudnn/_deps/torch_dep.py`. Keep the package initializer empty so the dependency layer's import order and no-cycle guarantee remain explicit.
- Make `torch_dep.py` the sole location for a runtime `import torch`. It attempts that import once, on the first call to `is_available()` or `require()`, and caches either the imported module or the original import failure; importing `torch_dep` itself must not trigger the attempt, and repeated probes and requirements must never retry the import. `is_available()` returns a Boolean without raising. `require(feature)` returns the cached Torch module on success, or raises `TorchNotAvailableError` with the canonical feature-specific message, chained from the captured import failure, when Torch is unavailable. The `torch_dep` name avoids ambiguity with the framework module it wraps, and the helper must have no dependency on top-level `cudnn`, so importing it during `cudnn` initialization cannot create a cycle.
- Define `TorchNotAvailableError` in the dependency layer and import/re-export it from `cudnn` before defining the lazy optional-symbol loader; do not expose other dependency-helper internals.
- Replace unconditional Torch imports throughout `python/cudnn` with this helper. Use postponed annotations and type-checking-only imports where annotations do not need runtime evaluation.
- Call `require(feature)` before the first Torch-specific allocation, dtype lookup, stream operation, custom-op dispatch, or validation. Availability probes and genuinely framework-neutral paths must use `is_available()` and remain non-throwing.
- Add `except TorchNotAvailableError: raise` immediately before `_load_optional_symbol`'s existing `except Exception` arm so the public error passes through unchanged. Continue using the existing generic `ImportError` wrapper for missing non-framework dependencies such as CUTLASS.

### Import-time Torch usage and API defaults

- Give `python/cudnn/api_base.py` explicit treatment because it is imported by all CuTeDSL APIs: defer `torch.device` and Torch memory-format access until use, and make constructing Torch-specific descriptors fail with `TorchNotAvailableError` rather than breaking module import.
- The current-source audit found no distinct `None` branch for parameters whose defaults are Torch values (`memory_format`, `dtype`, and the `*_dtype` parameters). Convert those defaults uniformly to `None`; before landing, recheck documentation and call sites so a concurrent semantic change is not silently overwritten.
- Update each converted annotation to postponed `Optional[torch.dtype]`, resolve the current effective Torch value immediately after `require(feature)`, and state the fixed effective default explicitly in the docstring—for example, “`None` means `torch.float32`.” Keep this wording distinct from pre-existing dynamic `None` semantics, which must continue to say what is inferred or omitted—for example, “`None` means match `q.dtype`” or “`None` means not provided and is invalid in discrete mode.” Audit both kinds wherever they appear together so users cannot mistake every optional dtype for a fixed default. Do not use opaque sentinel objects that leak through `inspect.signature()`, `help()`, or IDE tooltips.
- Build Torch-to-cuDNN and Torch-to-CUTLASS dtype maps lazily after availability is confirmed. Replace framework-independent values such as the signed int32 maximum with Python constants.
- Include standalone helpers such as `yarn.py`, stream utilities, reference engines, and internal kernel modules in the same import-safety sweep; no module-level Torch attribute access may remain outside the explicitly guarded custom-op implementation blocks.
- Keep `wrapper.Graph` explicitly PyTorch-only in this change: importing `cudnn.wrapper` remains safe, but constructing `Graph` without PyTorch deliberately fails fast with `TorchNotAvailableError`. This is stricter than waiting for allocation or execution, but matches the wrapper's current contract—automatic PyTorch output/workspace allocation and default Torch-stream management—and avoids creating a partially usable object. The lower-level `cudnn.pygraph` plus integer/DLPack buffer paths remain the framework-neutral route; supporting `wrapper.Graph` with caller-provided non-Torch buffers and streams is future work.

### Torch custom-operator registration

- Treat the custom-op work as a dedicated refactor for:
  - `python/cudnn/ops/causal_conv1d.py`
  - `python/cudnn/experimental/ops/sdpa.py`
  - `python/cudnn/experimental/ops/moe_grouped_matmul.py`
- Keep each existing module as an import-safe facade that preserves its public functions, starts each call with `require(feature)`, and only then lazily imports its sibling Torch implementation module.
- Retain the strict requirement that implementation modules also import successfully without Torch because the agreed acceptance criterion covers every module under `python/cudnn`. This intentionally costs more than exempting private implementation modules: each implementation must bind the Torch module through `torch_dep.require(feature)` and place that binding, dtype maps, annotated definitions, `torch.library.Library`, `custom_op`, `register_fake`, `register_autograd`, and `torch.compiler` work inside one module-level `if torch_dep.is_available():` block. Do not add another direct `import torch`.
- With Torch installed, the facade's lazy import executes that guarded block once through Python's module cache, registering every schema, fake implementation, compiler hook, and autograd formula exactly once while preserving current operator names and callable signatures.

### Datatype conversion

- Retain `cudnn.datatypes.is_torch_available()` as a compatibility Boolean probe backed by `cudnn._deps.torch_dep.is_available()`; it must not raise.
- Make `is_cutlass_available()` check CUTLASS independently of Torch. Move `_torch_to_cutlass_data_type_dict` construction behind its own Torch-availability guard so CUTLASS-native `cutlass.Numeric` types continue to work when CUTLASS is present and Torch is absent.
- Keep Torch-specific converters as optional entries in `_library_type`'s converter sequence, leaving a clear insertion point for a future JAX converter.
- When Torch is absent, Torch-specific converters return no match. Framework-neutral conversion of an unsupported, non-CUTLASS dtype continues to raise `ValueError("Unsupported tensor data type.")`; PyTorch-only public APIs must call `require(feature)` before reaching that generic path.

## Packaging and Documentation

- Remove `torch` and `torch-c-dlpack-ext` from `[project.optional-dependencies].cutedsl` together.
- Update the validated installation-instruction inventory in the same release as the metadata change:
  - `docs/fe-oss-apis/{overview,bsa,nsa,dsa}.md` and `docs/fe-oss-apis/gemm_fusions/gemm_proj_rope_mxfp8.md`
  - `python/cudnn/native_sparse_attention/sparse_attention.md`, `benchmark/dsa/README.md`, and the install-error messages in `benchmark/dsa/benchmark_dsa_sparse_attention_backward.py`
  - `benchmark/cutedsl_fusion_kernels/README.md` and `python/cudnn/__init__.py`'s optional-dependency hint
- Finish with a repository-wide search for `[cutedsl]` so newly added or differently worded installation guidance is not missed.
- State the installation order clearly:
  1. Install the PyTorch build appropriate for the user's CUDA environment.
  2. Install `nvidia-cudnn-frontend[cutedsl]`.
  3. Optionally install `torch-c-dlpack-ext` afterward to restore its AOT Torch-to-DLPack fast path and avoid TVM-FFI JIT/fallback overhead.
- Explain that omitting `torch-c-dlpack-ext` can affect conversion startup/performance but not numerical API behavior. Installing it after the chosen PyTorch distribution prevents its hard dependency from selecting an unintended default Torch build.
- Leave existing PyTorch examples and API descriptions unchanged except for effective-default documentation and installation prerequisites.

## Validation

- Do not add or modify test files. Before merge, open a GitHub issue in `NVIDIA/cudnn-frontend` titled “Add permanent no-Torch import CI coverage,” link it from the implementation PR and this plan, and treat it as the durable tracking artifact for the explicitly deferred regression test.
- Run a one-off AST audit over `python/cudnn` that rejects every runtime Torch import outside `cudnn/_deps/torch_dep.py`: `import torch`, aliased imports such as `import torch as _torch`, `from torch import ...`, imports from Torch submodules such as `from torch.x import ...`, and literal dynamic imports through `importlib.import_module(...)` or `__import__(...)` when the target is `torch` or starts with `torch.`. Type-checking-only imports remain permitted. The audit must also fail on Torch-valued defaults and unguarded module-scope Torch attribute/decorator/registration access. For the three custom-op implementation modules, permit Torch syntax only beneath the single top-level `if torch_dep.is_available():` guard and reject it elsewhere. Also run `python -m compileall python/cudnn`.
- Implement the no-Torch subprocess with a `MetaPathFinder` that raises `ModuleNotFoundError("No module named 'torch'", name=fullname)` for exactly `torch` and `torch.*`; do not use `sys.modules["torch"] = None` or a different exception type. Before importing project code, verify `import cutlass` and `import tvm_ffi` succeed under this blocker so their optional-Torch probes cannot cause false failures.
- With the blocker active and the remaining CuTeDSL dependencies installed, import `cudnn`, every public namespace, and every module under `python/cudnn`, including the guarded custom-op implementation modules. Confirm `is_torch_available()` is false, `cudnn.pygraph` remains usable, and representative CuTeDSL, wrapper, custom-op, and reference-engine calls raise `cudnn.TorchNotAvailableError` with the canonical message.
- Build a wheel, create a clean temporary virtual environment, install the wheel's `[cutedsl]` extra, and assert that neither distribution appears in `pip list`, `importlib.util.find_spec("torch")` returns `None`, and `importlib.metadata.distribution("torch-c-dlpack-ext")` raises `PackageNotFoundError`. This is the acceptance check for direct and transitive dependency removal.
- In an environment with PyTorch but without `torch-c-dlpack-ext`, exercise a representative `enable_tvm_ffi=True` conversion path to confirm the supported JIT/fallback path remains functional.
- With PyTorch installed, run the existing graph/router, wrapper, causal-convolution, SDPA, MoE, API-base, Yarn, and supported FE-API test suites without modifying them. Confirm custom operators register once and current effective dtype defaults remain unchanged.

## Assumptions and Deferred Work

- All non-Torch dependencies required by the imported feature modules are installed during the no-Torch import smoke test.
- Consumers needing PyTorch manage its installation and CUDA compatibility themselves; `torch-c-dlpack-ext` is an optional post-install performance accelerator.
- Public `memory_format`, `dtype`, and `*_dtype` signatures with Torch-valued defaults change to `None`; the current-source audit found no conflicting `None` semantics, and documented/runtime effective defaults remain unchanged.
- `wrapper.Graph` remains PyTorch-only; a framework-neutral high-level allocator/stream abstraction is deferred.
- No JAX adapter, new tensor protocol, permanent test coverage, or CI policy is introduced in this change; the required GitHub follow-up issue tracks permanent coverage separately.
