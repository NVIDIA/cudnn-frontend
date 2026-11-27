# python/cudnn — Agent Guide

The `cudnn` Python package: nanobind-backed graph API plus pure-Python **frontend-only OSS kernels** (CuTeDSL). Package inventory: `README.md` in this directory. Build/test commands: [../../AGENTS.md](../../AGENTS.md). SDPA rules (S1, S2, ...): [sdpa/AGENTS.md](sdpa/AGENTS.md), read before touching `python/cudnn/sdpa/`.

## Import-time rules

- `import cudnn` must work **without** torch/cutlass/cuda-python. Everything needing them is exported lazily via `_LAZY_OPTIONAL_IMPORTS` in `__init__.py`; its module `__getattr__` re-raises failures as `ImportError` naming the missing framework module (torch, jax, cuda-python) plus the `pip install nvidia-cudnn-frontend` hint, or, for a CuTe DSL below `CUTEDSL_MIN_VERSION`, the DSL upgrade (Rule 7).
- Never add an eager `import torch` / `import cutlass` to `__init__.py` or anything it imports transitively. Kernel modules import them at top level, so they must be reachable only through the lazy table.
- A function may share a name with its implementation module; Python installs an imported submodule on its parent, bypassing module `__getattr__`. Check direct-submodule, sibling-symbol and public-symbol import orders in fresh interpreters. Detector: `test_ops_callable_exports_survive_import_order` in `test/python/core/test_import_boundaries.py`.
- FROST GEMM shared compiler/codegen helpers publish explicit export mappings onto the selected family module. Shared executors use bounded generic runtime interfaces; family subclasses bind `TileConfig` and `GemmBinding`. Keep concrete type imports in family modules; no wildcard family imports back into shared modules. Check facade-first and direct-family-first imports in fresh processes under both `CUDNN_FRONTEND_GEMM_ARCH_FAMILY=sm100|sm120`.
- Reuse the required CuTeDSL dependencies (`pyproject.toml` `[project] dependencies`) unless a kernel needs a new package. The `[cutedsl]` extra holds only `cuda-python`.

## Hard rules

Numbered so reviews can cite them. Append, never renumber.

**Rule 1 — `execute()` is a hot path: validate, never convert, never allocate.**

- **No implicit conversions.** Never `.to(dtype)` or a copying `reshape()` on an execute argument: each allocates, launches a kernel and breaks CUDA-graph capture; on an output, a reshape copy swallows the kernel's write. Validate dtype/shape/contiguity, bind a true view, else raise `ValueError` (`_checked_lse_view` / `_checked_sinks_1d` / `_checked_seq_lens` in `sdpa/fwd/api_dsl.py`).
- **No per-execute allocations.** No `torch.empty`/`torch.zeros` in `execute()`. Carve scratch from the caller's workspace (`scratch_workspace_bytes()`). A dead ABI slot may use a one-time cached `_dummy` at most (grandfathered, closed for new code: Rule 8). Prefer compiling the operand out (`cutlass.const_expr` on `None`, as in the SM120 SDPA optional lse/sinks).
- **Init-time flags are compile-time specializations; `execute()` must match them in both directions.** A required-but-missing tensor raises (never a zeros dummy: zeros sinks change the softmax denominator, zeros seq lens mask every row). A provided-but-uncompiled tensor raises too.
- **No degenerate-path fixups.** Runtime-degenerate inputs (all-zero THD `seq_kv_lens`) use the kernel's own dead-row path, never adapter-side `fill_`/`copy_`. For a zero packed extent, bind a never-dereferenced dummy view over contract-guaranteed storage.
- **Validate overlapping optional declarations as a set.** When ragged offsets, `cu_seqlen` and `seq_len` tensors can declare the same thing, each combination is defined or explicitly rejected (#266).
- **An execute-time shape/stride override reaches the executor or raises before launch.** A raw uid-map plan cannot consume it; never drop the override triple. Filter override-enabled graphs per plan binding capability, keeping prepared plans. Detectors: `test_uid_map_plan_rejects_runtime_overrides_before_execute`, `test_override_filter_preserves_compatible_split_candidates`.
- **Shape overrides do not enlarge the producer's storage.** Validate metadata inputs against observed span and effective shape; check pointer alignment for the element type. Detector: `test_dense_metadata_rejects_short_observed_storage_and_misalignment`.
- **A runtime SF expert stride needs a per-plane contract.** FROST MoE `F8_128x4` weight scales require E outermost, dense bytes within each expert plane, enough blocked capacity per plane, and a non-overlapping 16-byte aligned expert pitch. Whole-blob density is insufficient (an interleaved view passes it and selects wrong scales). Detectors: `test_moe_weight_sf_plane_binding`, `test_moe_weight_sf_runtime_stride` in `gemm/frost/test_moe_grouped_block_scale_matmul_fwd.py`.
- **Native geometry caches retain no storage observations.** A hit still checks current dtype, device, address alignment and observed byte span. Mark uninitialized entries explicitly (an empty shape is an input to reject). Derive binder ABI tests from the real host signature including optional slots, then run the real graph launch (`test_sdpa_native_dense_binding.py`).
- **A shared native binder keeps each caller's validation contract.** Standalone SDPA length tensors require exactly the declared batch; graph binding may accept a larger carrier. Keep the stricter adapter check before common binding (standalone native decode test in `test_sdpa_native_dense_binding.py`).
- **Check workspace placement for Python-only producers.** A producer may expose `.device` without `__dlpack_device__`; validate it before a prepared launch. Reject malformed or partial declarations; a bare integer address has no metadata. Use a launch recorder for invalid-device controls so an unfixed guard cannot poison the CUDA context. Detector: `test_python_workspace_device_is_checked_before_launch`.
- **Grouped GEMMs validate every operand's logical K before launch**, accounting for each side's packed FP4 storage. Detector: `gemm/frost/test_moe_grouped_matmul_fwd.py::test_moe_rejects_mismatched_k_before_launch`.
- **Validate workspace before initializing reduction outputs**; a rejected call preserves caller buffers. Detectors in `gemm/frost/`: `test_moe_workspace_failure_preserves_reduction_outputs`, `test_split_k_missing_workspace_preserves_reduction_outputs`.

**Rule 2 — `execute()` launches exactly the kernels the plan promised: serve the declared layout natively, or decline.**

- **No hidden kernel launches**: no gather/scatter normalization copy, `.contiguous()`, layout repack or scatter-back. Carving the copy's scratch from the workspace does not make it acceptable; Rule 1's workspace exemption covers metadata buffers and dead-slot dummies only.
- **Cannot address the declared layout natively? Decline in `check_support()`** with `NotImplementedError` naming the tensor and its strides, so the Router picks another engine (`_thd_check_strides_native` in `sdpa/fwd/api_dsl.py`).
- Whatever `check_support()` accepts, the kernel addresses natively (layout-driven offsets, strides in TMA descriptors).
- **Dense SDPA forward serves declared strides or declines.** SM100/SM107/SM120 bind aligned covering B/H/S permutations directly; graph and standalone admission share `config_sm100.dense_bind_strides`, each port with its own storage element width. Unaligned outer strides decline before compilation. SM90 uses `config_sm90.dense_bind_strides` (`dense_flex`, a unit axis pinned to D). No compact-BSHD normalization exception remains; split partials and quantization scalars are legitimate algorithmic scratch.
- **Architecture admission needs a native compile-and-execute detector**, not only a mocked capability check. Qualify a new op through a scoped compile target; do not widen a shared compiler architecture gate. Detector: `test_DSA_indexer_top_k_sm120_native_boundary` (changed-input replay at the shared-memory limits).

**Rule 3 — `execute()` never reads device memory to the host.**

- No `.item()` / `.tolist()` / `.cpu()` / `.to("cpu")` / `.numpy()` / `float(tensor)` / `int(tensor)` / `torch.is_nonzero`, and no branch or f-string forcing one, on an execute argument or anything derived from it. No `torch.cuda.synchronize()` or stream/event `synchronize()` either.
- A blocking D2H is illegal during stream capture, so such a path **cannot be CUDA-graph captured**: a functional gap, since inference stacks capture decode. Its cost is the queued work ahead of it, so microbenchmarks show only the floor.
- If a device value must shape the launch, pass its pointer and dereference in-kernel, or compile on an envelope and read the real extent from device metadata.
- A validation that needs a device read is invalid: decline in `check_support()` (Rule 2) or assert in-kernel.
- **Prove it with a test; grep is not a detector** (`int(cu[i])` on a CUDA tensor escapes a `.item()` search):

  ```python
  torch.cuda.set_sync_debug_mode("error")   # any blocking D2H now raises
  try:
      out.backward(grad)                    # or graph.execute(...)
  finally:
      torch.cuda.set_sync_debug_mode("default")
  ```

  Example: `test_varlen_backward_does_not_sync`. Confirm the test is RED against the old code.
- When auditing, grep for the argument (`device="cpu"`), not the call shape.
- **Suspect duplicated logic first.** Violations are usually a second copy of a conversion already done device-side elsewhere (packed-to-padded LSE repad: device-side `searchsorted` in `sdpa/fwd/torch_op.py` vs a host loop in `torch/sdpa_provider.py`). Extract the device-side one and call it from both.
- SM80 packed backward wrapper: without `max_s_kv`, the packed capacity bounds the grid and B+1 prefixes stay on device. Capacity, Stats head pitch, launch bounds and deterministic-counter size are distinct; a caller may reserve more than `B * max_sequence_length`. Cover poisoned slack and changed device prefixes under replay. Detector: `test_wrapper_without_length_hints_does_not_sync`.

**Rule 4 — compile keys are PLAN-TIME-ONLY: never key a kernel compile on runtime data values.**

`cute.compile` takes seconds. Every compile-cache key input (an `lru_cache`d `compile()`, a template parameter, a fake-tensor extent) must derive from the graph declaration alone: dtypes, declared strides, head counts, head dims, flags. Runtime values (THD packed totals, max sequence lengths, batch contents) change every step and turn each `execute()` into a recompile (#552). This holds even for a legally obtained host value such as `int(tensor.shape[...])`.

- **Runtime extents compile dynamic.** Use `cute.sym_int()` in fake tensors (one symbol per ragged group). Launch scalars (THD `max_sq`) are `cutlass.Int32` call arguments.
- **Derived values count.** A batch stride of `t_q * token_stride` leaks the runtime total; zero the never-stepped batch stride and rebuild it kernel-side.
- **Cold-empty compilation cannot infer compact strides from empty samples.** Packed tensors may keep arbitrary outer strides and Torch's empty `dim_order()` can ignore them. Build symbolic token extents from logical descriptors; verify empty-to-nonempty reuse with compilation forbidden. Detector: `gemm/cutedsl/test_grouped_gemm_wgrad_empty.py`.
- **Compile at plan time, re-bind at execute.** The execute path's cached call must be a guaranteed hit. Guard with a cache-miss test (`test_dsl_sm100_thd_compile_key_plan_time_only`).
- **Prove artifact reuse across plans.** `compile_cached`'s serializer declines runtime dataclass arguments. A `Constexpr` dataclass is exportable only when the DSL executor shows it is absent from runtime arguments, and its configuration must be in the specialization key; an immutable template module can read `PARAMS` inside the host. Build a second plan with `cute.compile` forbidden, assert a real cache hit, check outputs and replay (`test_replan_reloads_prepared_artifact`, SM80). Set `CUDNN_FRONTEND_COMPILED_CACHE_INPROCESS_MEMO=0` for that process first, else the in-process memo serves the object (`stats()["memo_hits"]` grows, `hits` does not).
- **Cache-key decorators preserve `None`.** `template_key()` returns `None` for an unidentifiable specialization; appending a suffix must keep it uncacheable (string formatting makes unrelated kernels share an object). Detector: `test_uncacheable_fp8_hosts_do_not_reuse_another_kernel` (SDPA API contract suite).
- **Compiled CuTe launchables belong to their CUDA context.** Pin the plan's device when compiling storage-free declarations; compile and execute under it; reject differently placed bindings before lowering pointers. An outer argument-only `functools.lru_cache` bypasses `compiled_cache`'s device/context key; delegate reuse to the device-aware cache. Standalone wrapper caches holding a launchable or workspace need the input device in their key. Detectors: `test_compilation_and_launch_preserve_the_declared_or_resolved_device`, `test_compilation_bypasses_argument_only_caches_across_devices` (sparse SDPA); `test_block_scaled_wrapper_cache_separates_devices` (grouped GLU; use an explicit per-device capture stream, since `graph.default_capture_stream` can keep the first device's stream); `test_quant_cache_keeps_each_devices_workspace`.
- **SM80 THD (issue #604, closed) is the pattern to copy:** symbolic packed extents; the backward host takes Int64 capacities and launch bounds at runtime (Stats head pitch, deterministic-counter size); tensor compiles key on `b = 1, sq = skv = 0` plus plan-time sequence count. Detectors: `test_sm80_bwd_thd_compile_key_plan_time_only`, `test_graph_thd_compile_key_is_plan_time_only`.
- **Key on exactly the contract-relevant set.** Under-keying (shape only while `check_support()` validates weight/RoPE/scale descriptors) returns a wrong artifact; over-keying (`alpha`, `m`/`n`/`k` on a shape-generic kernel) recompiles spuriously (#553). The key is what `check_support()` validates plus what the kernel specializes on.
- **Shared scratch needs an owner.** When cached plans share an allocation, invalidate cached geometry when another plan writes it. Test A→B→A with the same pointer (`test_te_workspace_plan_switch`).

**Rule 5 — every torch operation on the execute path is ordered on the LAUNCH stream.**

The kernel runs on `ExecutionContext.stream`; torch enqueues H2D uploads, `zero_()`, `div_()`/`copy_()` and allocator stream-tagging on `torch.cuda.current_stream()`. If they differ, prep and kernel race (#543). PyTorch-integration callers launch on the current stream, so tests miss this; only direct graph-API users with an explicit stream hit it.

- **Resolve the launch stream first** and run every torch op, including allocator calls and cached-dummy first use, inside `_torch_stream_context(current_stream, device)` (fp8/mxfp8 amax resets and `_execute_thd` in `sdpa/fwd/api_dsl.py`). Consumers of kernel output (`amax_o.div_()`, copy-back) belong there too.
- **A raw stream handle never goes straight into `torch.cuda.ExternalStream`.** Torch before pytorch/pytorch#183258 (v2.13.0; NGC 26.06 and torch <= 2.12 lack it) returns a fresh non-blocking stream for `ExternalStream(0)`, unordered with a launch on `CUstream(0)` (#1165). Use `cudnn._torch_stream` (`as_torch_stream`, `stream_context`); no engine writes its own wrapper. Recipe R1. Detector: monkeypatch `torch.cuda.ExternalStream` to raise and drive execute with handle 0 (`test_hopper_marshal_stream.py`).
- **TVM-FFI can override an explicit stream from the producer's DLPack exchange API.** `OperandBuffer.current_work_stream` reports the default stream. For Cake launches pass DLPack-only `DeviceView` wrappers over validated buffers and enter `tvm_ffi.use_raw_stream` with the execution stream. Test capture and changed-input replay on an explicit nondefault stream (an empty graph exposes the failure).
- **Native Cake bindings need explicit DLTensor strides, even contiguous.** TMA setup reads them. XLA bridges supply strides and use XLA-owned scratch on the XLA stream. Test full forward/backward chains (LM-head and mHC JAX suites, changed-input replay).
- **Stream order does not serialize host calls.** Cached scratch needs a lock through the final queued read (including reserve/growth), or per-call storage. Detector: `simplified_sparse_attention/cutedsl/test_simplified_sparse_attention.py::test_host_threads_share_stream`.
- **The device is implicit state like the stream.** Wrap execute-path allocations in the device context. Validate a raw pointer argument's device residency and dtype (a CUDA int64 tensor) before passing its address (#517).
- **Restore the caller's device after launching on another device**, including after errors. Scope `ensure_current_context` and the native call with `device_context(x.device)` and select `current_stream(x.device)` explicitly. Test with operands on one GPU and another current: check the binding stream and device/stream restoration after success and injected native exceptions. Detector: `native_forward_fp8_cases.test_native_default_stream_with_another_device_current`.
- **An entry-owned buffer is initialized on the stream current at allocation.** A direct-call entry allocating its own output/scratch must order later launches behind that initialization through `cudnn.frost.init_order` (`pending_of`, `record`, `order_behind`); no entry adds its own event bookkeeping (`python/cudnn/frost/README.md` rule 16 lists adopters). Detector: warm launch on another stream parked behind a delay kernel (`test/python/init_order_schedules.py`).

**Rule 6 — every Frost-generated kernel has a cuDNN-attributable symbol.**

- Right after every `@cute.kernel` definition, including auxiliary and generated-template kernels:

  ```python
  kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)
  ```

- Use the function's real name and keep default `keep_mangled_name=True`; no compiler flags or symbol rewriting.
- Verify: `(cd test/python && pytest -q test_frost_kernel_name_prefix.py)`.
- The call runs at import and may need a DSL newer than the `pyproject.toml` floor; it is legal only because Rule 7's gate runs before the kernel module is imported.

**Rule 7 — gate the CuTe DSL version at runtime; never assume the installed DSL satisfies your kernel.**

- The DSL floor must not exceed the lowest version any downstream pins exactly or caps: vLLM, SGLang, FlashInfer, TransformerEngine, Megatron-LM, and PyTorch's `torch/_native/cutedsl_utils.py` allowlist (4.8.0 as of 2026-11, set by vLLM/SGLang `==4.8.0`). `pyproject.toml` states it (`>=4.8.0`). The runtime floor `frost.buffers.CUTEDSL_MIN_VERSION` is still `(4, 7, 0)` until a follow-up raises it, so a kernel using a 4.8 API passes `floor=(4, 8, 0)` explicitly. Re-check the downstreams before raising either. Kernel-specific requirements still need runtime checks.
- Before a path imports a version-specific DSL API, check `cudnn.frost.buffers.cutedsl_state()` / `cutedsl_too_old()` (floor `CUTEDSL_MIN_VERSION`; a higher kernel floor is passed as `floor=`, never a second version parse) and **decline, or raise an error naming the version** via `cutedsl_requirement_error(what, floor=...)`. The failure must never surface as `AttributeError`/`TypeError`/`ModuleNotFoundError` from the DSL, nor as a `pip install` hint.
- The gate sits at the caller's entry, before the kernel module import: the op's route check (`_can_route_causal_conv1d_bulk` in `ops/causal_conv1d.py`, `_validated_native_update` in `ops/_causal_conv1d_update.py`), an engine's `check_support`, or the family `__init__` lazy import. Kernel-file module scope may assume the floor only because of that gate (#799).
- Pass an explicit kernel floor when qualification exceeds the helper floor. Detector: `test_old_dsl_fails_before_kernel_import` (simplified sparse attention, 4.8 floor). Check PyTorch conditional-graph capability before allocating filtered repair state (`test_filtered_runtime_declines_before_allocation`).
- CUTLASS can extend `cutlass.__path__` with operator-only packages. Compare concrete `cutlass`, `cutlass.cute`, `cutlass.base_dsl` origins; reject mixed or missing core origins (`moe_ep/cutemok/test_cutemok.py`).
- **A version floor does not imply target support.** Public 4.7.0 lacks `sm_107a`; check target capability before admitting SM107 plans; test graph and standalone declines with the real wheel and a controlled missing-target probe.
- **A DSL that knows a target can bind a compiler that lacks it.** `cutedsl_arch_requirement_error` checks the DSL `Arch` only. DSL 4.8.0 refuses `sm_107a` on a compiler up to CUDA 13.3 from inside the trace, and a default install binds the CUDA 12.9 flavor even under a CUDA 13 driver. A kernel admitting cc 10.7 also gates on `CUDA_VERSION` before JIT and tests the decline (`fp8_projection_decode.DSL_COMPILER_FLOORS`, `dsl_compiler_error`).
- Check the DSL compiler CUDA version separately when a kernel depends on it: `from cutlass.base_dsl.version_info import CUDA_VERSION`. `torch.version.cuda` and `CUDA_HOME` do not identify it. Example: both QSA backward entries decline the CUDA 12 compiler for every one-KV-head geometry (12Q/1KV, 6Q/1KV) before JIT, with tests on the decline.
- **Qualify both compiler flavors for new/changed kernels and DSL upgrades.** Run cold compilation and numerical GPU tests with cu12 and cu13 in separate environments/processes and empty, separate caches, at the dependency floor and the newer version being qualified, on every claimed architecture: forward/backward, supported dtypes, boundary/tail cases, finite outputs, existing accuracy contract. Import- or compile-only success is not correctness. Unsupported compiler/arch combinations decline before JIT, with a test. Report unavailable lanes; never loosen tolerances or skip failing supported lanes.
- Record source revision, installed DSL distributions, loaded native compiler path and `CUDA_VERSION.major`/`.minor`, GPU/driver, framework, backend/NVRTC, pass/fail/skip counts. With DSL 4.8 the default wheel installs cu12; `[cu13]` adds another flavor the loader may pick, so verify the loaded flavor. A cu12 DSL with a CUDA 13 framework isolates DSL compatibility only. Workflow: [integration skill](../../skills/cutedsl-kernel-integration/SKILL.md).
- **Known floors** (extend when you depend on a newer API; say so in the PR if a user-facing op's floor rises):
  - `cutlass.experimental.*` (primitives, `cuda.tensor_map`; all of `cudnn/frost/tile_dsl`) → 4.7.0.
  - `cudnn._jax` → 4.7.1 (JAX dependency group and runtime guard enforce it).
  - `cutlass.experimental.cuda.create_tensor_map_im2col` (`conv/cutedsl/conv3d_postops`) → 4.9.
  - `cp.async.cg.shared.global.L2::cache_hint` with a `createpolicy` register (`gated_attention_block/kernels/qsa_score_paged.py`) → a DSL compiling with CUDA 13.4+ (cu13 flavor of 4.8.0 under a CUDA 13 driver). Older assemblers mis-encode it on sm_90/sm_100. The kernel reads `CUDA_VERSION` (`L2_HINT_MIN_COMPILER`), traces the plain copy below the floor, and declines a forced `l2_hint=True` by name, stating the observed defect and prescribing the cu13 flavor under a CUDA 13 driver (not a wheel number).
  - `cutlass.cute.experimental` layer (`@cute.experimental.kernel`/`.jit`/`.compile`, IR-level `allocate`/`tma_load`/`dot`/`partition_and_copy`; `gemm/frost/projection_decode`) → 4.8.0. The layer is deprecated, so the entry probes its names before importing the kernel and declines by name (`fp8_projection_decode.DSL_LAYER`, `dsl_layer_error`). No version ceiling.
- **Bind DSL names that moved between releases through `cudnn._cutlass_compat`**, never their old home (old homes warn and will raise `AttributeError` inside the trace). Objects deprecated in place warn only when used (e.g. `cutlass.utils.StaticPersistentTileScheduler`, `PersistentTileSchedulerParams`); use the vendored `cudnn._cutlass_helpers.static_persistent_tile_scheduler`. Detectors: `test_kernel_module_reads_no_deprecated_name_off_a_dsl_module` and `test_trace_raises_no_deprecation_warning_from_the_kernel_package` (`test/python/gemm/frost/projection_decode/test_fp8_projection_decode.py`); `test_rendering_pins_the_convention` and the trace test in `test/python/gemm/frost/swiglu_decode/test_swiglu_fp4_decode.py`.
- **A `cutlass.cute.experimental` kernel reads two env namespaces.** Cross-target render needs both `CUTE_DSL_ARCH` and `CUTE_EXPERIMENTAL_DSL_ARCH`. Dumps: `CUTE_EXPERIMENTAL_DSL_KEEP=cubin,ptx` with `CUTE_EXPERIMENTAL_DSL_DUMP_DIR` (`CUTE_DSL_KEEP` dumps nothing), plus `CUTE_EXPERIMENTAL_DSL_NO_CACHE=1` for a real compile. On 4.8.0 the unified `is_experimental=True` spelling does not preprocess such kernels, so keep the layer's own decorators (`test_frost_kernel_name_prefix.py` recognizes both).
- **PTX compiler output depends on the kernel symbol name.** Rule 6's prefix changes the symbol, so compare "the same kernel" on PTX; do SASS comparisons and instruction-order audits on one rendering with its final symbol.
- Tests that import a kernel module directly `pytest.skip` on a too-old DSL. CI `oss:` lanes span supported DSL versions (`ci/stages/python_tests/cutedsl.yml`, internal CI); a lane below your floor must show skips, not errors.
- Keep declaration and execute-preflight validators out of modules that import newer DSL APIs. Exercise the public constructor and preflight in a fresh process on the floor wheel. Detector: `test_gated_block_declaration_and_execute_preflight_do_not_import_cache_kernel`.
- **MLIR argument widths are part of a DSL API migration.** `FastDivmodDivisorV2` carries two MLIR values; static coordinates can carry zero. Custom `__new_from_mlir_values__` slices by lengths recorded at extraction (`test_cutlass_schedulers.py`).
- **A co-resident library may patch the DSL preprocessor.** quack-kernels (>= 0.6.2, pinned by vLLM/SGLang, in the CI image) replaces `DSLPreprocessor.visit_If` on `import quack`, making a name assigned in the `else` body of `if const_expr(S) and D:` function-scoped; a later dynamic loop reassigning it fails as `TYPE_UNSTABLE_JOIN`. Give such loop-local names their own spelling. Detector: rerun the kernel's tests with quack imported first (a `-p` plugin).
- **Packed FP4 crosses the launch as bytes.** Below 4.8, `from_dlpack(..., enable_tvm_ffi=True)` binds `torch.float4_e2m1fn_x2` with packed shape/strides and the kernel silently reads wrong bytes. Launch a `torch.uint8` view and recast in the `@cute.jit` entry with `cute.recast_tensor(t, cutlass.Float4E2M1FN)` (`kernel_qk_view`, `IndexerScoreUnifiedSm100Mxfp4._qk_tensor` in `deepseek_sparse_attention/score_recompute/`). Detector: numerics tests on a 4.7 lane.

**Rule 8 — the graph API owns no device memory and never blocks the host, at build and at execute.**

Rules 1 and 3 cover `execute()` in torch terms; this rule adds plan build and driver-API spellings.

- **No plan- or engine-owned device allocation.** No `cuMemAlloc`/`cudaMalloc`, `torch.empty`/`torch.zeros`, `frost.buffers.DeviceBuffer`, at build or execute, cached or not. Execute scratch (per-batch metadata, on-device TMA descriptors, an unrequested output the kernel always writes) is carved from the caller's workspace and declared via `get_workspace_size()` (`prepared.py`: `meta_ptr = workspace_ptr`, `o_desc_ptr = workspace_ptr + off_o_desc`). A path without a workspace contract gets one. Owned memory has GC-timed release; a collection inside another `torch.cuda.graph` window invalidates the capture (#1151). The relaxed-mode guard on remaining finalizers (#1152) is defence in depth only.
- **A dead ABI slot is compiled out or bound to `0` (R3 is canonical); never allocated, never another operand's address.** `0` is legal: `cuTensorMapEncodeTiled` accepts NULL, the DSL rejects only negative addresses (`cute.runtime.nullptr`), tvm-ffi passes `cute.Pointer` as ints, so a slot that turns out live faults loudly. A borrowed operand address (`sinks_ptr = q.ptr`) turns that fault into a silent read of Q. Rule 1's cached `_dummy` is grandfathered for `sdpa/fwd/api_dsl.py` only.
- **No host-blocking call at build or execute**: `cuStreamSynchronize`, `cuCtxSynchronize`, `cudaDeviceSynchronize`, `cuEventSynchronize`, synchronous `cuMemcpy*`/`cuMemsetD*`, `torch.cuda.synchronize()`, `.item()`. Build is lazy and runs inside a stream capture in serving stacks, so it meets the execute standard: `cuMemcpyHtoDAsync` on the launch stream, `cuMemsetD32Async`, constants baked into the kernel or written by a setup kernel.
- **An engine that cannot be async declines.** CAKE KDA plans on the host, so it checks `cuStreamIsCapturing` and raises under capture (`linear_attention/cake/compiler.py::check_not_capturing`). The one documented host wait outside engines: a direct-call entry whose captured launch writes an entry-owned buffer with pending initialization waits on that event (`cudnn.frost.init_order`); allocate and finish setup before capture to avoid it.
- **Detectors.** `test_execute_allocates_nothing_and_never_synchronizes` (`set_sync_debug_mode("error")` plus allocator accounting; run the same around BUILD); `test_collect_unrelated_resources_during_capture` (GC in a global-mode capture window, then replay and a native launch); a capture test whose first execute of a shape is inside `torch.cuda.graph`.

### Rule 8 / Rule 5 recipes

One canonical answer per situation; reviewers cite the recipe name. If none fits, extend the recipe here in the PR; never write a local variant.

**R1 — raw stream handle to torch stream.**
```python
from cudnn._torch_stream import as_torch_stream, stream_context

with stream_context(ctx.stream, device):          # torch work on the launch stream
    buf.copy_(src)
tensor.record_stream(as_torch_stream(ctx.stream, device))
```
- Never call `torch.cuda.ExternalStream` / `get_stream_from_external` directly. `stream_context(None)` and a handle equal to the current stream are no-ops; `0`/`1` and torch's default stream map to `torch.cuda.default_stream(device)`.
- `CUstream(2)` is per-thread, not legacy default. Reject it (also before the raw-current-stream shortcut); use a concrete `torch.cuda.Stream`. Detector: `core/cutedsl/test_torch_stream_sentinels.py`.
- When staging a caller tensor, record its storage on the consuming stream even when `stream=None`. Use `contiguous_on_stream` / `copy_into_on_stream` (`core/cutedsl/test_torch_stream_staging.py`).

**R2 — execute needs scratch** (metadata, on-device descriptors, an unrequested always-written output, staging for a dead-but-required slot). Declare, carve, never allocate:
```python
def get_workspace_size(self) -> int:                 # BaseEngine / CompiledPlan
    return ws_align(meta_bytes) + ws_align(desc_bytes) + ...
def execute(self, graph, variant_pack, ctx):
    ws = Workspace.over(variant_pack, self.get_workspace_size(), type(self).__name__)  # frost/workspace.py
    meta = ws.take(4 * b + 4, "int32"); desc = ws.view(off, "int64", (slots * 16,))
```
- Adapters: `scratch_workspace_bytes()` + `WorkspaceCarver(workspace, bytes, label).take(numel, dtype)` (`sdpa/fwd/api_dsl.py`).
- `Workspace(None, ...)` raises the canonical missing-workspace error; reuse it, never fall back to `torch.empty`/`DeviceBuffer`.
- A wrapper that allocates per call on the caller's behalf does so under `stream_context(<launch stream>)`: the caching allocator orders reuse only against the allocating stream.
- Eager JAX wrappers: `block_until_ready` does not extend storage lifetime through a foreign CUDA consumer. Pair stream-ordered allocate/free around the launch (`grouped.backend_utils.wrapper_workspace`) or use an XLA custom call owning scratch. Test multiple pending consumers on independent streams with an intercepted bounded byte copy; never run a real kernel on deliberately recycled scratch.
- Direct API workspace: validate CUDA device type and the operand's ordinal before launch (`Workspace(..., device=...)`).
- DLPack export selects the producer's `__dlpack_device__()` and restores the caller's context on errors. For CAI-only workspace use runtime pointer attributes (older cuda-python returns zero for `CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL`). Detector: `core/cutedsl/test_workspace_device.py` (two GPUs).

**R3 — dead ABI slot.** In order: (1) compile it out: `Optional`/`None` parameter read only under `cutlass.const_expr(flag)`, `flag` in the compile key, `None` at both compile and launch (DSA sm90 `mTopkIdx`, `mTopkLength`); (2) pointer ABI: bind `0` (`prepared.py` `sinks_ptr`, dense `o_desc_ptr`/`meta_ptr`); (3) required `cute.Tensor` parameter: borrow bytes from the workspace via R2 (`sdpa/bwd/api_dsl.py` dense `seq_kv`/`desc_words`). Never a cached `torch.zeros` dummy; never `q.ptr` borrowing (a live read silently reads Q).

**R4 — a live per-batch table the caller did not supply** (lengths, offsets, scale scalars). R2 carve + one async fill on the launch stream (`cuMemsetD32Async`, `cuMemcpyHtoDAsync`, `buffers.memset_zero_async`), or a scalar kernel argument. Never `torch.tensor(values, device=...)` per execute; never a plan-owned tensor built at `compile()`.

**R5 — input not in the kernel's layout/dtype.** Decline in `check_support()` with `NotImplementedError` naming tensor and strides/dtype (`_thd_check_strides_native`). No `.contiguous()`, `.to(dtype)` or repack, even into the workspace (Rule 2).

**R6 — something must block the host.** The engine is non-capturable: check `cuStreamIsCapturing` and raise first (`linear_attention/cake/compiler.py::check_not_capturing`); document it in the docstring.

**R7 — need a cuDNN handle, caller gave none.** Graph lowering: `_pygraph._backend_handle_for_lowering` (per thread and device, destroyed at exit). A handle-less execute runs every engine on the caller's current stream: `pygraph._resolve_stream` resolves it once into `ExecutionContext.stream` and `_backend_exec_handle` re-streams the default handle to it. A launch takes `engines.base.require_stream(ctx.stream, ...)`; a direct (non-graph) entry resolves None itself (`_torch_stream.caller_stream`). Python plans build and plan inside `frost.device.build_device(<handle ordinal>)` (`_compile_python_plan`, `_planning_scope`). Detectors: `core/graph/test_no_handle_stream.py`, `core/frost/test_frost_no_handle_stream.py`, `core/frost/test_frost_build_device_scope.py`. Torch-op layers: per-device cached handle re-streamed to the current stream before each call (`linear_attention/ops/common.py::get_handle`, `ops/norm/_common.py`). Never `cudnn.create_handle()` inside a plan, engine or C++ graph object.

**R8 — owning a CUDA resource with GC-timed release** (tests and C++ PyGraph only). Release inside `cuThreadExchangeStreamCaptureMode(RELAXED)`, restore in `finally` (`frost/buffers.py DeviceBuffer.__del__`, `pygraph.h CaptureModeGuard`). Production code reaching this skipped R2/R7.

**R9 — proving it.** Around warm `build()` + `execute()`: `torch.cuda.set_sync_debug_mode("error")` and `torch.cuda.memory_stats()["allocation.all.allocated"]` unchanged across three executes (`test_sdpa_prepared_thd.py::test_execute_allocates_nothing_and_never_synchronizes`, `test_sdpa_bwd_thd_sm80.py::test_graph_thd_execute_does_not_allocate`). Capture safety: `test_cuda_capture_lifetime.py`. R1: `test_torch_stream.py` with `ExternalStream` monkeypatched to raise.

Host-overhead migrations: measure warm enqueue and captured GPU replay separately, and check changed-input outputs after timing. A tiny async memset can add cross-engine waits between graph kernels; if replay regresses, inspect inter-node timestamps (the prepared FP8 amax reset uses an SM kernel for this reason). Keep the empty-input reduction identity (`test_prepared_fp8_empty_thd_resets_amax_without_attention`).

**Rule 9 — backend and FROST share one FE Python graph contract; no special treatment at the caller boundary.**

- **Same API and semantics.** For workloads both engines support, preparation and execution accept the same arguments with the same meaning for bindings, overrides, outputs, workspace, streams, capture/replay, ownership and invalidation. Callers (including FlashInfer) never branch on the selected engine. A private method consumed outside FE is a caller boundary.
- **Preparation belongs inside FE**, during build/compile. Geometry first known from execute-time overrides is prepared and reused internally, with no extra caller-visible step. Keep per-call buffer observations and checks. An optional lower-level interface needs benefit beyond internal caching and must serve both engines.
- **Optimize the shared path.** Reuse provider-independent preparation, UID ordering, validation, binding and caching so both engines benefit. A backend-only prepared entry beside the old FROST path is not unified. Engine-specific launch lowering stays internal; kernel pipeline templates need not be shared.
- **Capabilities may differ; the contract may not.** Express differences through support checks and plan selection. Never silently ignore an argument, weaken validation, or require a caller workaround for one engine; internal fallbacks keep execution and lifetime semantics.
- **Verify both routes** with the same caller code: fresh bindings, overrides, replan/invalidation, workspace/stream changes, capture/replay. Check outputs and invalid-input behavior, not ranking or plan order. Measure shared host optimizations per route.
- **Native context checks preserve the thread's execution policy.** A real stream determines its context; default-stream sentinels use the handle's device. Retain primary contexts once per device. The common execution boundary restores the caller's original context on success and failure, including an empty worker context. Detectors: `core/graph/test_ensure_current_context.py`, `core/graph/test_execute_device_context.py` (two GPUs for foreign-device cases).
- **A guard keeping the backend planner off a graph covers every path that creates a backend plan.** Example: the cc 10.7 single-query MXFP8 guard (`manifest.EngineFamily.backend_guard`, `sdpa/fwd/backend_guard.py`; cuDNN 9.26/9.27 planner SIGSEGVs). Besides `_finalize_backend_layout` and `backend_plan_entries`, the out-of-sequence creators `_append_backend_plan` (explicit pin, replayed entry) and `_lower_backend_plan` go through `_pygraph._refuse_guarded_backend` and raise the typed decline; `key()`/`serialize()` only lower. Detector: `sdpa/frost/test_sdpa_backend_guard.py::test_explicit_backend_pin_on_a_guarded_graph_is_a_typed_decline`. A version-bounded guard also needs its re-measuring detector (test/AGENTS.md).
- The first version where a defect was measured is not its introduction. A lower bound needs evidence from an unaffected earlier release; otherwise exclude the engine until the verified fixed version. Test earlier supported versions in default planning and explicit replay without version-skipping the numerical detector (`test_sdpa_bwd_backend_guard.py`, SM90 ragged backward).
- **Configuration-specific backend defects need a plan guard** that keeps the engine's valid knobs: filter enumeration records, check explicit replay after the backend resolves missing knobs, and stop delegators/custom ranking from restoring the rejected config. Check at planning/build, not execute; do not query knobs for unguarded families. Detector: `sdpa/frost/test_sdpa_fp8_backend_split_guard.py` (raw-backend controls, constant-V output oracle; correct Stats alone are insufficient).
- **Provider support is not graph invalidity.** Keep the graph facts of a layout FROST cannot run and let the provider's matcher decline; an empty `invalid` fact record can bypass backend guards. Detector: mixed dense/ragged Q/K/V checks in `test_sdpa_bwd_backend_guard.py`.
- **Names FlashInfer reads are pinned**: `cudnn.sdpa_capabilities()`, `cudnn.frost.cutedsl_runtime_available(device_cc=None)`, `cudnn.datatypes.torch_to_cudnn_data_type`, `pygraph.selected_engine` / `.backend_plan_entries()`, and private `_SdpaThdBinder.supports_*`, `datatypes._torch_to_cudnn_data_type`. Detector: `test/python/sdpa/frost/test_flashinfer_consumed_surface.py` (host-only, signatures, cc 10.7 + 4.7.x DSL → False). Renaming one requires updating FlashInfer's fallbacks first. The runtime probe is device-aware because the DSL floor is 4.7.0 generally and 4.8.0 on cc 10.7.

**Rule 10 — a kwarg that makes a graph ineligible for the python engines is read by the family's analyzer under the spelling of EVERY op that takes it.**

- The analyzer record is the node's kwargs verbatim (`sdpa/graph_analyzer._record_from_node`); rows gate on derived facts. A capability no row claims still needs its fact: a False fact silently admits the graph. Spellings differ: `sdpa()`/`sdpa_backward()` use `score_mod`/`score_mod_bprop`, `sdpa_fp8()` uses `fn`; reading only `fn` dropped TE flex-attention modifiers.
- When an op gains a kwarg a kernel cannot honor, add the spelling to the analyzer and a detector cell in the same change. One-line check: `graph.selected_engine` on a graph with the kwarg.
- Detectors: `test/python/sdpa/frost/test_sdpa_score_mod_decline.py` (host-only; every spelling, every fwd/bwd row declines by name), `test/python/sdpa/graph/test_flexible_sdpa_te_shape.py` (TE shape served by backend with the modifier).

**Rule 11 — consume what a thread loaded from a shared-memory stage before anything releases that stage, and prove the order in the compiled kernel.**

- **Hazard.** A consumer reading a stage with ordinary loads (`ld.shared`, `ldmatrix`) and then releasing it can receive the next fill in any load whose destination registers are unread at the release. No fault, no hang; wrong values, at rates from rare to every launch depending on shape, part and thread blocks per SM. Functional tests can pass.
- **A release** is anything after which the stage may be rewritten: every `mbarrier` arrive form (plain, counted, `expect_tx`, remote, cluster), `tcgen05.commit` to a barrier, named/cluster-barrier arrive, a peer send completing the peer's barrier (`st.async`), and a CTA barrier behind which the stage is refilled (`cp.async`, bulk copy, TMA load, another warp's stores). Loading and releasing lanes may differ.
- **Source order is not consumption.** The compiler moves arithmetic, compares, votes, shuffles and register-operand `mma` behind an independent release; only the generated code proves order.
- **Not an ordering:** `bar.warp.sync`, a CTA barrier, `fence.acq_rel`/`fence.sc`, the arrive's release semantics, a cross-proxy fence before the loads, a warp-collective alone, a delay, a polling wait.
- **Do this:** release through `cudnn.frost.tile_dsl.consume`: `consume_then_arrive` (local; elected via `pred=`, counted via `count=`), `consume_then_arrive_on_peer`, `consume_then_read_tile_id_arrive` (scheduler slot credit). Each sums every loaded word, masks with the kernel's release-zero argument and adds it to the barrier address. Pass every value the lanes loaded since acquiring the stage, each once, never a derived value (differences can cancel). Every loading warp releases through the helper itself. Contract and cost: [frost/README.md](frost/README.md).
- **Where the helper does not fit** (no address operand, e.g. a CTA barrier before a refill or a peer send; or another warp/block releases): put the release behind a store of a result computed from the values and show it in the disassembly; or, for an async-proxy refill (TMA load, bulk copy, cluster launch-control response), put `fence.proxy.async.shared::cta` between the last load and the release. Name the choice in the kernel's shared-memory buffer table and keep the site under warm stress.
- **Never let the kernel know the mask.** The release-zero argument has exactly one use, `release_zero(arg)`. Any compare/branch/assert/assume on it (even an entry guard `if arg == 0:`) lets the optimizer constant-fold it; an inline-asm zero (`opaque_i32_zero`) is folded by the assembler. The host passes literal 0; non-zero is undefined behaviour.
- **Tensor memory** (precaution): keep `tcgen05.ld`'s `tcgen05.wait::ld` and the first read of its registers before the release that frees the slot. No exemption for tensor memory.
- **Scope:** new and changed release sites. Existing sites keep their release arrive, fence or named barrier until moved onto the helper.
- **Detectors.** Proof: in the disassembly of every shipped rendering, each load of a released stage has a read of every destination register before the release on every path. `warm_stress.assert_release_dependence` pins the helper's chain in PTX; `sdpa/frost/test_tile_dsl_consume_then_arrive.py` applies it to every form against an unsafe twin. Rate bound: `warm_stress.assert_warm_stress` ([test/AGENTS.md](../../test/AGENTS.md)) on a shape that refills the stage within one launch, with at least two thread blocks per SM where possible. Cold functional tests and launch-1-equals-launch-2 checks do not test this.

**Rule 12 — split a long tensor-core reduction across at least two accumulator chains, and validate against FP64 on the smallest-magnitude outputs.**

- **Hazard.** On SM100 the `tcgen05.mma` FP32 accumulate step truncates toward zero instead of rounding to nearest, so one accumulator chain over the whole reduction collects a same-signed error proportional to the running sum's magnitude, growing faster than linearly in the issue count. At a reduction length near a million pixels (tens of thousands of MMA issues) the error at small-magnitude outputs reaches `atol`, and the maximum absolute error does not show it: BF16 output quantization dominates there.
- **Do this.** Feed alternate MMA issues to separate tensor-memory accumulators and fold them into one IEEE FP32 value in the epilogue (`conv/frost/templates/sm100_conv_wgrad.py` uses `_NUM_ACC_STAGES` slots for this, each seeded by an `enable_input_d`-cleared issue). Two chains are the floor; every extra chain divides the error.
- **Detector.** Compare a long-reduction kernel against an FP64 contraction restricted to the outputs with `|exact| < atol` (rms and max of `kernel - exact`). A single chain lands orders of magnitude above an IEEE FP32 sequential sum of the same products, and no max-absolute-error check can see it.

## Frontend-only kernel package layout

Organize by operation, then backend (`frost`, `cutedsl`), then architecture (`sm100`, `sm120`) where needed. Shared public functions go in the op's `api.py`; backend-independent contracts in `ops.py` (or `ops/`). Backend-specific adapters and layout/fusion subpackages stay inside their backend.

```text
python/cudnn/<op>/
├── __init__.py      # lazy public exports
├── api.py           # shared public functions
├── ops.py           # backend-independent operation contracts
└── <backend>/<sm_x>/
```

Example: `gemm/cutedsl/{dense,grouped,discrete_grouped}/<fusion>/` (grouped by operand layout); `gemm/frost/` has `sm100/`, `sm120/`, facades `compiler.py` / `epilogue_codegen.py`, shared `kernel_templates/`, `ln_prologue/`, and direct-call entries `projection_decode/`, `swiglu_decode/` beside the engine; `gemm/ops/` holds torch custom-op contracts.

Keep helpers in the narrowest package shared by their users (`gemm/cutedsl/grouped/moe_*.py`, `gemm/frost/kernel_templates/`). Do not export them through `cudnn`.

## CuTeDSL kernel bodies

These break at `compile()`, not import: `python -c "import ..."` and `pytest --collect-only` stay green. After refactoring a kernel body, run that flavor's own tests.

Control flow and helpers:
- **Do not factor `@cute.kernel` body code into a plain Python helper.** The DSL AST-transforms only the decorated function (`for` → `ir_loop`, `if` → `scf` region); a helper's ops can land outside the enclosing region (`Error building ...` through `ir_loop` → `scf_execute_dynamic`). Duplicate across flavors, or mark the helper `@cute.jit`.
- `for x in (a, b)` over a Python tuple becomes a dynamic `ir_loop` and cannot iterate heterogeneous objects (`GridConstant[TensorMap]`). Unroll or use `cutlass.range_constexpr`.
- Never reuse a persistent loop index in a nested unrolled loop. Detectors: GDN-2 `test_prep_fwd_matches_uncut_and_reference`, `test_prep_checkpoint_series` in `linear_attention/test_la.py`.
- Loaded integer vectors can infer `Int8` without signedness; set the declared dtype before widening to INT32. CuTeDSL 4.8.0 unary integer vector negation can crash; subtract from a typed zero vector. Keep each node's compute dtype. Detector: `gemm/frost/test_matmul_mainloop_fusion.py::test_mainloop_int8_identity_preserves_signed_arithmetic`.

Indexing and memory:
- **Array slices need wide coordinates for wide strides.** In 4.8, `cutlass.Array` slicing (`a[batch, row, head, :]`) multiplies each coordinate by a stride cast to the coordinate's width; an Int64 layout does not prevent overflow. Widen each dynamic sliced coordinate first; full scalar indexing is a different path. Detector: `TestNativeHalfStrides::test_physical_wide_stride` (SM100/SM107 forward). Rerun spill checks per admitted form/dtype after widening (SM107 D256 decode probe at the 96-register limit).
- Widen operands before triangular workspace products in producer and consumer; derive tile decoder bounds from the input extent (`test_triangular_offsets.py`).
- An unused name can hold a required shared-memory allocation; keep the allocation when a later operand aliases it, remove only the binding (Hq16 sparse-attention teacher kernel, tests in `simplified_sparse_attention/cutedsl/`).
- **Exact workspace reuse** needs an ownership proof and compiler-visible load-before-store order: keep the aliased load/store in one inline PTX block and hold loaded values in named temporaries until every input is consumed. Partial overlaps and cross-lane readers stay unsafe. Detector: KDA `test_state_chain_summary_alias_capture`; run it from a parent source tree and check the nested kernel module path (the linear-attention conftest overlays its sibling tree, so `cudnn.__file__` alone is insufficient).
- JAX-shared kernels: match scratch flag storage to the native ABI (byte flags need `uint8`, not JAX Boolean). Detector: long-key cases in `simplified_sparse_attention/cutedsl/test_filtered_jax.py`.
- **Every TMA scatter4 source needs 128-byte SMEM alignment**; a 16-byte staging row advances 64 bytes per issue. Select STG for such pitches unless padding is in layout and resource accounting. Detector: `gemm/frost/test_moe_grouped_block_scale_matmul_fwd.py::test_moe_block_scale_narrow_n_routing` (`scatter-False-24`).
- **Immutable tensor-map reuse is local to a consumer and a launch.** A persistent warp skips acquisition only while reusing a map it acquired; track sequence/expert per consuming warp, reacquire on switch, reset at kernel entry. Never cache acquisition in a host plan or infer it from another CTA's fence. Detectors: `sdpa/frost/test_sdpa_thd_tensormap_acquire.py`, `gemm/cutedsl/test_wgrad_tensormap_acquire.py`.
- Resident block-scale SF words use only TMEM left after accumulator staging; account for every operand, M block and word; keep SF-ready/AB-ready waits and producer-stage release. Detectors: `test_block_scale_resident_sf_tmem_budget`, `test_block_scale_resident_sf_exact_fit`, one-sided dequant numerics in `gemm/frost/test_block_scale_matmul.py`.

Barriers and synchronization:
- **A named barrier reached from a warp-role branch is a subset barrier.** `pipeline.NamedBarrier` defaults to the aligned-CTA form; lower `barrier_cta_arrive` and `barrier_cta_sync` with `aligned=False` (split-phase protocols, TMEM alloc/dealloc gates). Do not call `TmemAllocator.wait_for_alloc()` there (it rebuilds an aligned barrier). In inline PTX, `bar.sync` is `barrier.sync.aligned`; use `barrier.sync`. Spell subset barriers via `tile_dsl.barrier.named_barrier_sync_unaligned`. Run `compute-sanitizer --tool synccheck` and inspect PTX. Detector: `test_rendering_pins_the_convention` in `test/python/gemm/frost/swiglu_decode/test_swiglu_fp4_decode.py` (five `barrier.sync`, three `bar.sync`).
- A nonblocking barrier arrival must complete before the same warp reuses that barrier; use a separate barrier for tile completion when consumers can outrun producers. Detector: `test_canonical_jax_glu_hadamard_quant_delayed_act`.
- An `empty` mailbox acknowledgement must order prior reads before the producer overwrites (write-after-read, even for read-only consumers); `bar.warp.sync` is insufficient. Keep the MoE scheduler's release arrives. Detector: `test_moe_block_scale_mailbox_acknowledgements_order_prior_reads` in `gemm/frost/test_template_epilogue_parity.py`. Rule 11 is the general form.
- **TMA-store epilogue buffer reuse.** Every `PipelineTmaStore.producer_commit()` needs a `producer_acquire()` in the issuing thread before the buffer is rewritten (after commit before the barrier, or before the next write). `producer_tail()` waits only at exit; another store's acquire can mask a missing one, so check every configuration that drops it (#2350). Detectors: `gemm/cutedsl/test_tma_store_buffer_reuse.py` (host-only), `gemm/cutedsl/test_grouped_gemm_glu.py::test_rubin_mxfp8_glu_without_c_matches_with_c`.
- **`griddepcontrol.launch_dependents` behind warp-role branches fires at the first role reaching it.** Name the role in "released at the end" and document the consumer's obligation to issue its own wait. Detector: `test_dependent_that_waits_reads_the_eager_bytes` (SwiGLU FP4 decode); rendering pin counts one `launch_dependents`.

Instruction-level traps:
- **The assembler folds `rcp.approx.ftz.f32` of an immediate.** State such factors as bits (`tile_dsl.pointwise.opaque_fp4_max_rcp()`, `0x3E2AAAAB`). Detectors: `test_rendering_of_every_helper` (`test/python/sdpa/frost/test_tile_dsl_quant_helpers.py`), SwiGLU FP4 rendering pin (two `rcp.approx.ftz.f32` per block-scale chain).
- **`cute.arch.rcp_approx` as an intrinsic can be speculated across a branch.** Where a reference instruction stream matters, use `tile_dsl.pointwise.rcp_approx_ftz`. Detector: SwiGLU FP4 rendering pin requires predicated reciprocals.
- **Block-reduction pairing order is part of the kernel's bytes.** Use `tile_dsl.rowwise.block_reduce` (its `order` has no default) and name the order in the entry's text. Detector: `test_block_sum_is_the_documented_tree` in `test/python/sdpa/frost/test_tile_dsl_rowwise.py`.
- **`redux.sync.max.f32` assembles for cc 10.x only** (rejected on `sm_90a`, `sm_110a`, `sm_120a`, `sm_121a`). Take warp max from `tile_dsl.rowwise.block_reduce` (`kind="max"`, `"max_index"`), which uses a positive target list. Detectors: `test_rendering_of_every_primitive[sm_120a]`, `test_shuffle_arm_is_the_redux_arm`.
- **`cute.arch.shuffle_sync_up` defaults to clamp 31 before DSL 4.7.0**, so a warp prefix sum returns its input. Pass `mask_and_clamp=0`. Detector: H128 D512 conflict-rank cases of `deepseek_sparse_attention/cutedsl/test_DSA_sparse_attention_backward.py` on 4.6.2.
- **A `setmaxnreg` split needs `.launch(..., min_blocks_per_mp=1)`.** With `.reqntid` alone ptxas emits C7508 (never surfaced by the DSL) and drops every `setmaxnreg`. Once live, the producer budget is hard: keep per-lane producer state small and accept only warp grids the split fits (`Sm120KernelTemplate.warp_layout_reject`). Detector: compile-only SASS pin, one `USETMAXREG.DEALLOC` and one `TRY_ALLOC` at the template's counts plus a spill bound (pattern: `sdpa/frost/test_sdpa_bwd_d256_sm100.py`).
- **A wgmma accumulator read not dominated by `wgmma.wait_group` serializes every wgmma** (C7517, C7514, `WARPGROUP.DEPBAR` after every HGMMA). Issue the wait unconditionally (a wait with nothing pending is a no-op). Detector: compile with `CUTE_DSL_COMPILER_OPT="remarks{ptx}"` and search for `C7514`/`C7517`, or compare `WARPGROUP.DEPBAR` count to `wait_group` calls. No test pins this yet.

Performance claims:
- For fractional-microsecond A/B, repeat with shared addresses, reversed construction order and an independently compiled unchanged-source control.
- Check that an optimization's admission condition is hit by repository benchmark shapes or a documented target workload before adding a compile-cache specialization.

## FROST GEMM arch trees

- `gemm/frost/sm100/` and `sm120/` each own a `compiler.py`, `epilogue_codegen.py` and `kernel_templates/`. `cudnn.gemm.frost.compiler` / `.epilogue_codegen` are facades that become the active family's module on first import (`arch_family.active_family()`: once per process from the current GPU; pin with `CUDNN_FRONTEND_GEMM_ARCH_FAMILY=sm100|sm120`). Import the facade path; `monkeypatch.setattr(compiler, ...)` reaches the real module.
- Each tree renders only its own pipelines: `_check_own_family` declines the other family's template after the semantic gates (`is served by the sm<N> arch tree, but this process runs the sm<M> tree`; the frost test conftest reports it as a skip). The sm120 pipeline on an SM 10.x GPU needs `CUDNN_FRONTEND_GEMM_ARCH_FAMILY=sm120`.
- A fix that applies to both trees lands in both copies. A test of one tree's renderer imports that tree by name (`cudnn.gemm.frost.sm120.compiler`).
- Locate a template with `kernel_registry.template_path(file)` / `KernelTemplate.path`, never `Path(compiler.__file__)`. Code both trees import stays in `gemm/frost/kernel_templates/` ([frost/README.md](frost/README.md), owner-names-the-directory rule).

## FROST GEMM planning invariants

- **Small-M block-scale planning** uses `GemmFacts.shapes_fixed`. Only fixed shapes select token widths 8/16/32 after swapAB; overrides keep width 64 and at most 48 weight tiles; dynamic shapes bypass strategy selection; explicit knob replay bypasses automatic choices. Keep the rule in `gemm/frost/planning.py`, probe proposed and retained baseline plans, leave fitted cost coefficients unchanged; automatic candidates use workspace reduction. The rule leads on SM100; other GPUs keep their first choice and add the geometry as an autotune candidate.
- On a CC 12.x part with no cost profile (`planning._calibrated`), two profile-free rules lead with the auto pick second: block-scaled fp4×fp4 up to 32 tokens plans the 32-row tile with tokens on M; dense fp8×fp8 up to 16 tokens with K >= 2048 plans a synthesized 16-row 1x8-warp tile; both use one-wave split-K; the fp8 rule declines when weight tiles alone fill more than half the SMs. A cost profile retires both. Detector: `gemm/frost/test_tile_select_analytic.py`.
- Dense SM100, mainloop-fusion, block-scale and SM103 hosts resolve L2 swizzle width once per call and pass an Int32 kernel argument; never compute it per warp. Keep the mixed-CGA pinned identity mapping and the runtime width-one bypass. MoE resolves per-group raster on device.
- Narrow block-scale tiles extract arbitrary SF sub-rows (across 128-row atoms) and have no kernel-N <= 32 requirement; explicit narrow replay on override/dynamic graphs is legal with per-call SF capacity and output checks. Do not import the restricted narrow implementation's shape gates.
- **`TileConfig.cluster_k_reduce`** selects split-K mode: False (and old knob records) = workspace + separate reducer; True = all slices in one cluster, zero partial workspace. K cluster dimension derives from this flag and `split_k_slices`, never a separate axis. Reject unsupported templates, unlaunchable clusters and insufficient SMEM; never change a requested mode during build/execute. Automatic planning keeps False. Keep the flag in canonical names and knob round-trips. Epilogue codegen takes `workspace_split_k` directly; no config rewriting or source-regex template wrapping. Cluster device code only in templates implementing the full protocol (none in mainloop fusion or SM103); the epilogue parity test compares their base drain. Derive the cluster CTA barrier's participant count and fragment span from the epilogue warp count.
- SM100 dense/block-scale cluster split-K: every epilogue lane (including masked) stages before an async-proxy fence and CTA rendezvous; elected bulk copies register byte counts at the root before issue; root waits for transactions, reads local slots and gathers readers before a cluster-scope release/acquire ack. Budget S fragment slots, not S tiles. Keep the cluster exit rendezvous (`test_cluster_splitk_persistent_mailbox_reuse`). Gate TMA compute and stores in one SSA region. Mixed-width block-scale CTA-K uses the wider operand's bit width (`test_cluster_splitk_mixed_width_k_tiles_after_swap`). Both compiler families call `recipe.check_split_k_shape` in the launch guard and `explain()` before seeds, workspace writes or launch.

## The public function contract

Public functions (`cudnn.ops.<op>_<direction>(...)` / `..._jax(...)`, see `docs/operations/api_conventions.md`) allocate outputs and return them in a stable documented order. PyTorch functions return a **`TupleDict`**; JAX functions return a plain tuple or dict (`jax.jit` rejects `TupleDict`).

JAX: tracers do not reveal buffer placement. A default device may supply tuning properties only if launch and scratch sizes stay valid on every supported execution device. Pin the compiler architecture and query shared-memory limits for the build device; unrelated GPUs must not affect eligibility. Cache launchers by full configuration including device. Tests use `jax.default_device` without hiding other GPUs and reuse jitted functions on non-default devices with changed inputs and gradients (`test/python/norm/jax/test_norm_jax.py`).

## Adding a new frontend-only API — required checklist

1. Kernel package under the closest existing family (layout above).
2. Public functions in `api.py`.
3. Exports: `_LAZY_EXPORTS` in `python/cudnn/ops/__init__.py` (or `python/cudnn/ops/<family>/__init__.py`). New package directories need no registration (`pyproject.toml` `include = ["cudnn*", ...]`, namespace packages on).
4. Docs: page under `docs/operations/<family>/`, listed in `docs/fern/versions/latest.yml`, linked from `docs/operations/overview.md`.
5. Tests: `test/python/<op>/cutedsl/test_<op>.py` (+ `_utils.py`/reference): supported and unsupported configs, numerical reference comparison.
6. DSL version gate (Rule 7): decline with a version-naming error below `CUTEDSL_MIN_VERSION`; tests skip there.

Follow the `cutedsl-kernel-integration` skill (`skills/cutedsl-kernel-integration/`) for any kernel integration, including family classification.

## Other notes

- `wrapper.py` `Graph` context manager requires cuDNN backend ≥ 9.12 (`backend_version() >= 91200`) and builds plans on `__exit__`.
- Torch custom ops live with their operation family, may be re-exported from `experimental/ops/` (pattern: `docs/utilities/adding_torch_custom_ops.md`), cache built graphs per config and use stable `_UIDs` enums.
- dtype conversions go through `datatypes.py`, which probes torch/cutlass lazily; keep it lazy.
- Formatting: black, line length 160.
- CUDA-owning objects, GC-timed release and stream capture: Rule 8.

## DSA training composition

Training may compact valid active indices only if forward and backward share that exact metadata. Score targets preserve original slots, mask invalid/inactive entries, and normalize over retained slots after summing heads; never expose compacted indices as target order. Detector: `deepseek_sparse_attention/cutedsl/test_DSA_training.py::test_native_training_and_original_score_slots`.

## Linear-attention decay factorisation range

Splitting decay `2^(g[t] - g[s])` into `exp2(+g[t])` and `rcp(exp2(g[s]))` on separate MMA operands breaks inside the documented gate range `alpha = exp(g)` in (0, 1]: at an in-tile log2 prefix of -126 with bf16 operands (NaN) and about -16 with fp16 (overflow). Tests drawing `alpha` from `uniform(0.5, 1)` miss this.

- Route such tiles through `linear_attention/frost/common/steep_decay.py` (keeps `g[t] - g[s] <= 0`; its docstring holds dispatch and sync invariants). Keep the exact path out of the fast kernel.
- Bound the inverse operand by key magnitude as well as gate prefix (unnormalized `k=5` overflows fp16 at -13.75).
- Multiply per-token factors `exp2(g)`; never subtract inclusive fp32 prefixes. Reduce corrections in a fixed order.
- Use `encode_gate_exchange` (log2 prefixes only) for scan classification and exchange encoding; keep stage selection and barriers per schedule. GDN-2 schedules share stages in `common/gdn2_operands.py`; KDA keeps inline operand code and calls `common/steep_decay.py` with `beta_ptr=None`. Read exchange scales before releasing a raw stage. Forward kernels size steep scratch from cfg cosizes; backward kernels carve it at cfg byte offsets and assert capacity in `build_cfg`.
- Keep steep state gradients and bounded keys in FP32 through projections (`common/gdn2_state_grad.py`); workspace sized by `dstate_scratch_cosize`. Barriers: `mb_dstate_input_ready`, `mb_k_decay_inv_ready` / `mb_decay_done`, `mb_u_smem_ready` (scalar readers acquire directly). The full-backward CG2 release protects workspace and U until the restored-key projection finishes, including before the next persistent work item.
- Range tests vary state-gradient magnitude independently of gate magnitude (large FP32 seed, large FP16-derived carry). Test mild-path state-operand range separately (FP16, `g=0`, zero inputs, final-state gradient 70000).
- Replay one CUDA graph and workspace through mild, steep and mild gates; check finiteness before bitwise equality.
- An exact relaunch must not read what the fast launch wrote: seed overwrite-capable plans from `common/state_snapshot.py`, chosen from the graph contract, never pointer equality at first call (a graph can bind separate buffers at warmup and alias them later). A forward graph without a final-state output cannot overwrite its seeds, so its indexed input pools need no snapshot. Snapshots copy raw bytes and keep source dtype; test every state dtype with aliasing, padded storage and changed-state replay.

Detectors (steep section of `linear_attention/test_la.py`, GDN-2 and KDA): `test_steep_channel_decay_fwd_bwd`, `test_steep_channel_decay_fp16`, `test_steep_fp16_inverse_range_without_normalization`, `test_steep_multiple_channels_repeatable`, `test_steep_forward_state_alias_after_graph_warmup`, `test_steep_channel_decay_overwrite_initial_state`, `test_steep_channel_decay_backward_overwrite_initial_state`, `test_steep_read_only_state_pool`, `test_steep_strided_state_overwrite`, `test_steep_summary_matches_reference`.

## Sparse query pairing

With a nonfinite key, paired sparse queries need per-query masking before dQ arithmetic (zero dS times a nonfinite key is NaN); a union mask is insufficient, including across unaligned packed sequence boundaries. Keep any nonfinite fallback on device, re-evaluated each replay. Detectors: `QSA/cutedsl/test_qsa_backward_sm100.py::test_sparse_pair_isolates_a_nonfinite_key_row`, `QSA/cutedsl/test_qsa.py::test_paired_backward_nonfinite_isolation`.
