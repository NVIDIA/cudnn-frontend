# python/cudnn — Agent Guide

The `cudnn` Python package: nanobind-backed graph API plus pure-Python **frontend-only OSS kernels** (CuTeDSL). See `README.md` in this directory for the package inventory and [../../AGENTS.md](../../AGENTS.md) for build/test commands.

## Import-time rules (the most common way to break this package)

- `import cudnn` must work **without** torch/cutlass/cuda-python installed. Everything that needs them is exported lazily via `_LAZY_OPTIONAL_IMPORTS` in `__init__.py` — a module-level `__getattr__` imports the submodule on first attribute access and re-raises failures as `ImportError` that names the missing framework module (torch, jax, cuda-python) alongside the base `pip install nvidia-cudnn-frontend` hint — or, for a CuTe DSL below `CUTEDSL_MIN_VERSION`, points at the DSL upgrade (Rule 7).
- A function and its implementation module can share a name. Python installs an imported
  submodule directly on its parent, bypassing module `__getattr__`. Check direct-submodule,
  sibling-symbol, and public-symbol import orders in fresh interpreters; the detector is
  `test_ops_callable_exports_survive_import_order` in `test/python/core/test_import_boundaries.py`.
- Never add an eager `import torch` / `import cutlass` to `__init__.py` or anything it imports transitively. Kernel modules import them at top level, which is why they must only be reachable through the lazy table.
- FROST GEMM's shared compiler/codegen helpers publish explicit export mappings
  onto the selected family module. Shared executors use bounded generic runtime
  interfaces; family subclasses bind `TileConfig` and `GemmBinding`. Keep those
  concrete type imports in the family modules to avoid reverse dependencies.
  Avoid wildcard family imports back into shared modules. Check facade-first
  and direct-family-first imports in fresh processes
  under both `CUDNN_FRONTEND_GEMM_ARCH_FAMILY=sm100|sm120` settings.
- Reuse the existing required CuTeDSL dependencies (`pyproject.toml` `[project] dependencies`) unless a kernel truly needs a new package. The `[cutedsl]` extra now holds only `cuda-python`.

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
- **Overlapping optional declarations are validated as a set, not one by
  one.** When two mechanisms can declare the same thing (ragged offsets vs
  `cu_seqlen` vs plain `seq_len` tensors), each combination is either
  defined or explicitly rejected — an unhandled overlap is an untested code
  path with unspecified semantics, and "both supplied" is exactly the case
  no per-argument check catches (raised in review on PR #266).
- **An execute-time shape/stride override must reach the executor or raise
  before launch.** A raw uid-map plan cannot consume it; do not silently drop
  the override triple. Filter override-enabled graphs per plan's binding capability,
  preserving prepared plans while declining tensor-only ones. Detectors:
  `test_uid_map_plan_rejects_runtime_overrides_before_execute` and
  `test_override_filter_preserves_compatible_split_candidates`.
- **Shape overrides do not enlarge the producer's storage.** Validate metadata
  inputs against their observed span as well as their effective shape, and check
  pointer alignment for the element type. The host-only detector is
  `test_dense_metadata_rejects_short_observed_storage_and_misalignment`.
- **A runtime SF expert stride needs a per-plane contract.** FROST MoE weight
  scales in `F8_128x4` require E outermost, dense byte storage within each expert
  plane, sufficient blocked capacity per plane, and a non-overlapping 16-byte
  aligned expert pitch. Whole-blob density does not establish this: an
  interleaved expert view can pass it and silently select the wrong scales.
  Detector: `gemm/frost/test_moe_grouped_block_scale_matmul_fwd.py`'s
  `test_moe_weight_sf_plane_binding` and `test_moe_weight_sf_runtime_stride`.
- **Native geometry caches retain no storage observations.** A geometry hit must
  still check the current dtype, device, address alignment and observed byte span.
  Mark an uninitialized cache entry explicitly: an empty shape is an input to
  reject, not an initialized entry. Derive binder ABI tests from the actual host
  signature, including optional slots, then exercise the real graph launch.
  `test_sdpa_native_dense_binding.py` covers these boundaries and changed-input replay.
- **A shared native binder must preserve each caller's validation contract.**
  Standalone SDPA length tensors require exactly the declared batch size; graph
  binding may accept a larger carrier for an effective batch. Keep the adapter's
  stricter check before common binding. The standalone native decode test in
  `test_sdpa_native_dense_binding.py` rejects oversized length carriers after
  warmup and checks that per-call scale changes do not modify the plan.
- **Workspace placement checks also apply to Python-only producers.** A
  supported producer can expose `.device` without `__dlpack_device__`; validate
  that observable placement before a prepared launch. Reject present malformed
  or partial declarations rather than treating them as absent. A bare integer
  address has no such metadata. Use a launch recorder for invalid-device regression
  controls so an unfixed guard cannot poison the CUDA context. Detector:
  `test_python_workspace_device_is_checked_before_launch`.
- **Grouped GEMMs validate every operand's logical K before launch.** Account
  for each side's packed FP4 storage independently. Detector:
  `gemm/frost/test_moe_grouped_matmul_fwd.py::test_moe_rejects_mismatched_k_before_launch`.
- **Validate workspace before initializing reduction outputs.** A rejected call
  must preserve caller buffers. Detectors: `test_moe_workspace_failure_preserves_reduction_outputs`
  and `test_split_k_missing_workspace_preserves_reduction_outputs` in `gemm/frost/`.

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
- **Dense SDPA forward serves declared strides or declines.** SM100/SM107/SM120
  bind aligned covering B/H/S permutations directly. Graph and standalone
  admission share `config_sm100.dense_bind_strides`; each port uses its own
  storage element width. Unaligned outer strides decline before compilation.
  The prior compact-BSHD normalization exception has been removed; split
  partials and quantization scalars remain legitimate algorithmic scratch.
- The flip side of declining: whatever `check_support()` ACCEPTS, the kernel
  must address natively (layout-driven offset math, strides encoded in TMA
  descriptors) — acceptance is a promise about the execute path, not about
  what the adapter can patch up.
- Architecture admission needs a native compile-and-execute detector, not
  only a mocked capability check. Shared compiler architecture maps may also
  gate unrelated kernels; qualify a new operation through a scoped compile
  target instead of widening that shared support gate. The detector is
  `test_DSA_indexer_top_k_sm120_native_boundary`, including changed-input replay
  at the documented shared-memory limits.

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
- **Prove it; do not grep for it.** The list above is a reminder, not a
  detector — the spellings are many (`int(cu[i])` on a CUDA tensor is a
  blocking copy that a search for `.item()` will not find) and a reviewer who
  greps a subset concludes "clean". Assert the property instead:

  ```python
  torch.cuda.set_sync_debug_mode("error")   # any blocking D2H now raises
  try:
      out.backward(grad)                    # or graph.execute(...)
  finally:
      torch.cuda.set_sync_debug_mode("default")
  ```

  Put that in a test (see `test_varlen_backward_does_not_sync`), and check the
  test is RED against the old code before trusting it — a sync test that was
  never seen to fail is asserting nothing.
- **Suspect duplicated logic first.** Every violation found so far has been a
  *second* copy of a conversion that was already device-side somewhere else:
  the packed-to-padded LSE repad existed in both `sdpa/fwd/torch_op.py` (with
  `searchsorted`, device-side) and `torch/sdpa_provider.py` (a `for i in
  range(B): int(cu[i])` loop). Extract the correct one and call it from both
  rather than writing the obvious loop again.

The SM80 packed backward wrapper now shares the prepared graph chain. When
`max_s_kv` is absent, the packed capacity bounds the grid; B+1 prefixes stay
on device. `test_wrapper_without_length_hints_does_not_sync` detects the old
CPU copy after warmup (verified RED on the preceding wrapper). Capacity,
Stats head pitch, launch bounds and deterministic-counter size are distinct:
a caller may reserve more storage than `B * max_sequence_length`. Preserve
that capacity while honoring the caller's valid launch bounds; cover poisoned
slack and changed device prefixes under replay.

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
- **Prove artifact reuse across plans.** A stable key does not guarantee that
  `compile_cached` exported an artifact: its wrapper serializer declines
  runtime dataclass arguments. A `Constexpr` dataclass is exportable only when
  the DSL executor establishes that it is absent from the runtime arguments;
  its baked configuration must still be represented in the specialization key.
  An immutable template module can instead read its `PARAMS` inside the host.
  Build a second plan with `cute.compile` forbidden, assert
  a real cache hit, and check the reloaded artifact's outputs and graph replay;
  `test_replan_reloads_prepared_artifact` is the SM80 detector. Such a test
  sets `CUDNN_FRONTEND_COMPILED_CACHE_INPROCESS_MEMO=0` for its own process
  first: `compile_cached` keeps an in-process memo in front of the files, so
  the second plan would otherwise get the first plan's object without a disk
  hit (`stats()["memo_hits"]` grows, `hits` does not) and a key compiled by
  an earlier test in the same process would turn the first build into a memo
  hit too, leaving nothing on disk to reload.
- **Compiled CuTe launchables belong to their CUDA context.** Pin a plan's device
  when compiling storage-free declarations; compile and execute under that device,
  and reject differently placed bindings before lowering pointers. An outer
  argument-only `functools.lru_cache` can bypass `compiled_cache`'s device/context
  key and return another GPU's launchable: delegate reuse to the device-aware
  cache. Detector: intercept device contexts and compilation, build identical
  plans for two device ordinals, and verify each reaches compilation in its own
  context; no invalid cross-device GPU launch is needed. See the sparse SDPA
  tests `test_compilation_and_launch_preserve_the_declared_or_resolved_device`
  and `test_compilation_bypasses_argument_only_caches_across_devices`.
  Standalone wrapper caches also need the input device in their key when an
  object retains a launchable or workspace. A device-aware outer memo cannot
  protect an inner cache that omits it. The grouped GLU detector
  `test_block_scaled_wrapper_cache_separates_devices` alternates two devices
  without clearing caches and replays captures from each after changing inputs.
  Use an explicit per-device capture stream in that detector: Torch's implicit
  `graph.default_capture_stream` can retain the first device's stream.
  `test_quant_cache_keeps_each_devices_workspace` checks the grouped quant
  cache before a foreign workspace can reach a CUDA launch.
- **Issue #604 is closed**: SM80 THD compiles use symbolic packed extents.
  The prepared backward host takes Int64 capacities and launch bounds at
  runtime, including the compact Stats head pitch and deterministic-counter
  size; the retained tensor compiles use `cute.sym_int` and key on
  `b = 1, sq = skv = 0` plus the plan-time sequence count. The regression tests are
  `test_sm80_bwd_thd_compile_key_plan_time_only` (wrapper) and
  `test_graph_thd_compile_key_is_plan_time_only` (graph path). Copy that
  pattern, not a shape-keyed one.
- **Key on exactly the contract-relevant set — no more, no less.** Both
  failure modes shipped on PR #553 and were caught in review: *under-keying*
  (the cache keyed only `x.shape`/`w.shape` while `check_support()`
  validated weight, RoPE, and scale descriptors — a hit can return an
  artifact compiled for a different contract, i.e. wrong results) and
  *over-keying* (`alpha` passed at launch, `m`/`n`/`k` on a shape-generic
  kernel — every miss is a spurious multi-second recompile). Enumerate what
  `check_support()` validates and what the kernel specializes on; the key
  is that set.

- **Scratch state needs an owner, not just a pointer.** When cached plans share
  an allocation, invalidate cached geometry whenever another plan writes it.
  Test A→B→A reuse with the same pointer; see
  `test_te_workspace_plan_switch` for the compact GQA regression.

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
- **The device is implicit state exactly like the stream.** A `torch.empty`
  (or any allocator call) without a device context silently allocates on the
  *current* GPU, not the input tensor's — wrap execute-path allocations in
  the right device context as well as the stream context. And a raw pointer
  argument is a contract: validate device-residency and dtype (a CUDA int64
  tensor, not a host tensor) before handing its address to a kernel — both
  flagged in review on PR #517.
- **Restore the caller's device after a launch on another device.** Binding
  a stream's CUDA context can change the ambient device even when no torch
  operation runs. Scope that binding and launch with `device_context`; test
  with a side stream on Q's device while another device is current, then
  assert the caller's device is unchanged. Detector:
  `native_forward_fp8_cases.test_native_default_stream_with_another_device_current`.
- **A raw stream handle never goes straight into `torch.cuda.ExternalStream`.**
  Every eager caller on torch's default stream hands us a default-stream
  legacy sentinel (`0`, `cudaStreamLegacy` = 1), and torch
  before PR pytorch/pytorch#183258 (in v2.13.0; NGC 26.06 and torch <= 2.12
  lack it) returns a fresh NON-BLOCKING pool stream for `ExternalStream(0)`.
  Torch work issued in that context is unordered with a kernel launched on
  `CUstream(0)`: on an idle GPU the copies win the race and every isolated
  test passes; under xdist load the kernel reads stale conversion buffers and
  a staged output is copied back before it is written (the qa sm90
  `hopper_cuda` reds, PR #1165 — the same trap FROST SDPA hit in #682/#717/#860).
  Map the legacy sentinels and torch's own default stream to
  `torch.cuda.default_stream(device)`, the current stream to itself, and only a
  genuine side stream to `ExternalStream(handle, device=device)`. The one
  implementation is `cudnn._torch_stream` (`as_torch_stream`, `stream_context`,
  with the raw-handle fast path); every engine calls it, none writes its own
  wrapper. Detector:
  monkeypatch `torch.cuda.ExternalStream` to raise and drive the execute path
  with handle 0 (`test_hopper_marshal_stream.py`).

SDPA-specific hard rules (cited as Rule S1, S2, ...) live in
[sdpa/AGENTS.md](sdpa/AGENTS.md) — read it before touching anything under
`python/cudnn/sdpa/`.

**Rule 6 — every Frost-generated kernel has a cuDNN-attributable symbol.**

- Immediately after every ``@cute.kernel`` definition, including auxiliary and
  generated-template kernels, call the public naming API:

  ```python
  kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)
  ```

- Use the decorated function's actual name, keep the default
  ``keep_mangled_name=True``, and do not use compiler flags or symbol rewriting
  instead.
- Verify with ``(cd test/python && pytest -q test_frost_kernel_name_prefix.py)``.
- This call runs at module import, and DSL APIs used this way can be newer than
  the `pyproject.toml` floor admits. It is legal only because Rule 7's gate runs
  before the kernel module is imported — do not add an import path that skips
  it.

**Rule 7 — gate the CuTe DSL version at runtime; never assume the installed
DSL satisfies your kernel.**

- The `pyproject.toml` floor on `nvidia-cutlass-dsl` is `>=4.8.0`, matching
  vLLM v0.32.0 and SGLang v0.5.22, which both pin 4.8.0. Kernel-specific
  requirements still need runtime checks: environments can bypass dependency
  resolution, and newer kernels can require APIs or targets above the package
  floor. Keep the existing version gates and actionable errors.
- Before a path imports a DSL-version-specific API, check the installed version
  with `cudnn.frost.buffers.cutedsl_state()` / `cutedsl_too_old()` (floor:
  `CUTEDSL_MIN_VERSION`) and **decline, or raise an error that names the
  version** — `cutedsl_requirement_error(what)` builds it. Never let the failure
  surface as an `AttributeError` / `TypeError` / `ModuleNotFoundError` from
  inside the DSL, and never let it read as a missing-dependency install hint:
  the package is installed, and that `pip install` changes nothing.
- The gate lives at the entry the caller hits, before the kernel module is
  imported: the semantic op's route check (`_can_route_causal_conv1d_bulk` in
  `ops/causal_conv1d.py`, `_validated_native_update` in
  `ops/_causal_conv1d_update.py`), an engine's `check_support`, or the family
  `__init__`'s lazy import. Module-scope code in kernel files may assume the
  floor only because that gate ran first.
- A version floor does not guarantee target-architecture support. For example,
  public 4.7.0 lacks `sm_107a`; check the actual target capability before
  admitting SM107 plans. Test graph and standalone declines with the real
  unsupported wheel as well as a controlled missing-target probe.
- Check the DSL compiler's CUDA version separately when a kernel depends on
  its toolkit flavor: `from cutlass.base_dsl.version_info import CUDA_VERSION`.
  `torch.version.cuda` and `CUDA_HOME` do not identify the compiler selected by
  the DSL wheel. QSA's 12Q/1KV two-pass key kernel, for example, compiles with
  the CUDA 13 flavor but crashes the CUDA 12 compiler in DSL 4.7/4.8; its
  `check_support` rejects that compiler before JIT compilation, and its tests
  exercise the rejection rather than crashing or silently skipping the gate.
- Known floors — extend this list when you take a dependency on a newer API,
  and say so in the PR body if it raises the floor of a user-facing op:
  `cutlass.experimental.*` (primitives, `cuda.tensor_map`; everything under
  `cudnn/frost/tile_dsl` inherits it) → 4.7.0.
  Native im2col tensor-map creation used by `conv/cutedsl/conv3d_postops`
  (`cutlass.experimental.cuda.create_tensor_map_im2col`) → 4.9.
  `cp.async.cg.shared.global.L2::cache_hint` with a `createpolicy` register (the paged decode
  scorer's key copies, `gated_attention_block/kernels/qsa_score_paged.py`) → a DSL compiling
  with CUDA 13.4 or later, i.e. the cu13 toolchain of 4.8.0 (the DSL binds its cu13 flavour
  only under a CUDA 13 driver): the 13.3 PTX assembler of the public 4.7.0 cu13 wheel
  mis-encodes the hinted copy's uniform operands on sm_90 / sm_100 (an illegal-instruction
  trap at the first copy; the PTX is identical), and the cu12 flavour of both public wheels,
  4.7.0 and 4.8.0 -- what the DSL binds under a CUDA 12 driver -- reports CUDA 12.9, whose
  assembler emits the same encoding (compile-only dumps), so the kernel reads `CUDA_VERSION`
  (`L2_HINT_MIN_COMPILER`), traces the plain copy below the floor and declines a forced
  `l2_hint=True` by name -- the decline states the defect as observed (never as a property of
  the compiler it found) and prescribes the cu13 flavour under a CUDA 13 driver, not a wheel
  number.
- Tests that import a kernel module directly `pytest.skip` on a too-old DSL —
  they do not fail. CI runs the `oss:` lanes across the supported DSL versions
  (`ci/stages/python_tests/cutedsl.yml` in internal CI); a lane below your floor
  must show skips, not errors.
- Keep declaration and execute-preflight metadata validators outside modules
  that import newer DSL kernel APIs. A correctly gated stage cannot protect
  an earlier constructor import of its validator. Exercise the public
  constructor and preflight in a fresh process on the real dependency-floor
  wheel; `test_gated_block_declaration_and_execute_preflight_do_not_import_cache_kernel`
  is the detector for the gated block's paged-cache validation.
- **MLIR argument widths are part of a DSL API migration.**
  `FastDivmodDivisorV2` carries two MLIR values (encoded divisor and scalar),
  while its predecessor carried one. Custom `__new_from_mlir_values__` methods
  must slice by the lengths recorded during extraction, rather than assuming
  one value per field. Static coordinates can contribute zero values, too.
  `test_cutlass_schedulers.py` checks these round trips on real MLIR values.
- **The installed DSL's preprocessor may be patched by a co-resident library.**
  quack-kernels (>= 0.6.2, pinned by vLLM and SGLang, and present in the CI
  test image) replaces `DSLPreprocessor.visit_If` process-wide on
  `import quack`. It rewrites `if const_expr(S) and D: ... else: body` into a
  constexpr `if` that holds a copy of `body`, so a name assigned in `body`
  becomes function-scoped. If a later dynamic loop reassigns that name, the
  loop carries it as maybe-unset (`TYPE_UNSTABLE_JOIN ... None on one path`)
  whenever `S` is true. Give such loop-local names their own spelling.
  Detector: rerun the kernel's tests with quack imported first (for example,
  a `-p` plugin that does `import quack`).
- **Packed FP4 crosses the launch as bytes.** Below 4.8,
  `from_dlpack(..., enable_tvm_ffi=True)` binds a real
  `torch.float4_e2m1fn_x2` tensor with its packed shape and strides (two
  values per unit), while the kernel is compiled for the E2M1 ones. The kernel
  then reads the wrong bytes along the packed dim and returns wrong numbers
  without an error. Launch a `torch.uint8` view and recast it inside the
  `@cute.jit` entry with `cute.recast_tensor(t, cutlass.Float4E2M1FN)`
  (`kernel_qk_view` and `IndexerScoreUnifiedSm100Mxfp4._qk_tensor` in
  `deepseek_sparse_attention/score_recompute/`). Detector: the kernel's
  numerics tests on a 4.7 lane.
- Why: PR #799's `causal_conv1d_update` imported `frost.tile_dsl` from a route
  with no version check and broke the 4.6.2 lane — the version vLLM and SGLang
  shipped at the time — with a bare `ModuleNotFoundError: cutlass.experimental`;
  the bulk route next to it had the check and declined cleanly. Earlier, PR #854's
  module-scope `set_name_prefix(..., remove_cutlass_symbol=True)` failed the
  same way on a since-dropped 4.5.x lane, reported as "install optional
  dependencies".

**Rule 8 — the graph API owns no device memory and never blocks the host: at
build AND at execute.**

Rules 1 and 3 say this for `execute()` in torch vocabulary. This rule closes
the two gaps that produced the #1151 sm103 red: plan build, and the driver-API
spellings the torch vocabulary does not name.

- **No plan- or engine-owned device allocation, ever.** Not `cuMemAlloc` /
  `cudaMalloc`, not `torch.empty` / `torch.zeros`, not `frost.buffers.DeviceBuffer`,
  at build or at execute, cached or not. Execute scratch — per-batch metadata,
  on-device TMA descriptors, an output the kernel always writes but the graph
  did not request — is carved from the caller's workspace and declared through
  `get_workspace_size()` (`prepared.py`: `meta_ptr = workspace_ptr`,
  `o_desc_ptr = workspace_ptr + off_o_desc`). A path with no workspace contract
  (the direct `jit_from_cudnn_graph` MoE call) gets one; it does not get an
  allocation. Owned device memory has a GC-timed release, and a cyclic
  collection inside someone else's `torch.cuda.graph` window turned three
  per-plan dummies into `cuMemFree -> 900` and an invalidated capture (#1151 on
  sm103). #1152's relaxed-mode guard on the two remaining finalizers is defence
  in depth, not a licence.
- **A dead ABI slot is compiled out, or bound to `0`, or borrowed — never
  allocated.** Prefer `cutlass.const_expr` on `None` so the operand does not
  exist. Otherwise `0` is a legitimate address for a slot the kernel never
  dereferences: `cuTensorMapEncodeTiled` accepts a NULL global address (only
  misalignment is rejected), the DSL rejects only negative pointer addresses
  and ships `cute.runtime.nullptr`, and the tvm-ffi positional entry carries
  `cute.Pointer` parameters as plain integers — 12 prepared THD launches with
  `sinks_ptr = 0` pass bit-exact. A dead slot that turns out to be live then
  faults loudly instead of reading garbage. Borrowing an aligned address the
  contract already guarantees (`sinks_ptr = q.ptr` when `HAS_SINK` is off,
  descriptor-only THD rows aliasing live Q/O storage) is the other acceptable
  form. A cached `torch.zeros` dummy is not: Rule 1's `_dummy` exemption is
  grandfathered for `sdpa/fwd/api_dsl.py` and closed for new code.
- **No host-blocking call, build or execute.** `cuStreamSynchronize`,
  `cuCtxSynchronize`, `cudaDeviceSynchronize`, `cuEventSynchronize`, the
  synchronous `cuMemcpy*` / `cuMemsetD*` forms, `torch.cuda.synchronize()`,
  `.item()`. Build is lazy — it runs on the first execute of a shape, which in
  a serving stack is inside a stream capture (a FlashInfer graph-cache miss) —
  so build is held to the execute standard: uploads via `cuMemcpyHtoDAsync` on
  the launch stream, fills via `cuMemsetD32Async`, constants baked into the
  kernel image or written by a setup kernel. `cuMemAlloc` + `cuMemsetD32` +
  `cuStreamSynchronize(0)` at build was the #1151 anti-pattern.
- **An engine that cannot be async declines; it does not sync.** The CAKE KDA
  route plans its work items on the host (`cuMemcpyDtoHAsync` +
  `cuStreamSynchronize`) and therefore checks `cuStreamIsCapturing` and raises
  under capture (`linear_attention/cake/compiler.py::check_not_capturing`).
  That is the only legal shape of an exception: declared in the engine, loud,
  never on a captured stream.
- **Detectors.** `test_execute_allocates_nothing_and_never_synchronizes`
  (`torch.cuda.set_sync_debug_mode("error")` plus allocator accounting around
  a prepared execute) — run the same assertion around the BUILD;
  `test_collect_unrelated_resources_during_capture` (GC inside a global-mode
  capture window, then replay and a native cuDNN launch); and a capture test
  whose first execute of a shape happens inside `torch.cuda.graph`, so the lazy
  build runs under capture.


### Rule 8 / Rule 5 recipes — do exactly this, do not reinvent

One canonical answer per situation. Every engine copies the recipe; a reviewer
cites the recipe name. If a recipe does not fit, say so in the PR and extend the
recipe here — do not write a local variant (the audit behind #1165/#1167/this
section found 18 hand-rolled stream wrappers, 5 of them wrong).

**R1 — a raw stream handle becomes a torch stream.**
```python
from cudnn._torch_stream import as_torch_stream, stream_context

with stream_context(ctx.stream, device):          # torch work on the launch stream
    buf.copy_(src)
tensor.record_stream(as_torch_stream(ctx.stream, device))
```
Never call `torch.cuda.ExternalStream` / `get_stream_from_external` directly.
`stream_context(None)` is a no-op; a handle equal to torch's current stream is
a no-op via the raw-handle fast path; `0`/`1` and torch's default stream
resolve to `torch.cuda.default_stream(device)`. `CUstream(2)` is a per-thread
stream, NOT the legacy default. Torch interop rejects this sentinel: a cache
key or allocator stream record based on `(device, 2)` aliases distinct host
threads, and mapping it to the legacy default misorders scratch lifetimes.
Use a concrete `torch.cuda.Stream` or its raw handle. The rejection must happen
before the raw-current-stream shortcut too. Detector:
`core/cutedsl/test_torch_stream_sentinels.py`. When staging a caller tensor,
record its storage on the consuming stream even when `stream=None`: the caller
may have entered a side-stream context after allocating it elsewhere. Use
`contiguous_on_stream` / `copy_into_on_stream`; `core/cutedsl/test_torch_stream_staging.py`
checks both explicit and implicit current streams with allocator reuse controls.

**R2 — execute needs scratch (metadata, on-device descriptors, an output the
kernel always writes but the graph did not request, staging for a dead-but-
required tensor slot).** Declare it, carve it, never allocate it:
```python
def get_workspace_size(self) -> int:                 # BaseEngine / CompiledPlan
    return ws_align(meta_bytes) + ws_align(desc_bytes) + ...
def execute(self, graph, variant_pack, ctx):
    ws = Workspace.over(variant_pack, self.get_workspace_size(), type(self).__name__)  # frost/workspace.py
    meta = ws.take(4 * b + 4, "int32"); desc = ws.view(off, "int64", (slots * 16,))
```
(Adapters: `scratch_workspace_bytes()` + `WorkspaceCarver(workspace, bytes, label).take(numel, dtype)`
in `sdpa/fwd/api_dsl.py`.) `Workspace(None, ...)` already raises
`"<owner> requires a N-byte workspace but execute() received none; allocate
graph.get_workspace_size() bytes and pass the buffer to execute()"` — reuse
that error, never fall back to `torch.empty` / `DeviceBuffer` when the caller
passed nothing. A path with no workspace contract gets one. Where a wrapper
allocates per call on the caller's behalf, it allocates under
`stream_context(<launch stream>)` (R1): the caching allocator orders a block's
reuse only against the stream it was allocated on, so scratch allocated on
torch's ambient stream for a handle re-streamed to a side stream is a
use-after-free waiting for load. For eager JAX wrappers, allocation readiness
(`block_until_ready`) does not extend storage lifetime through a foreign CUDA
consumer. Pair caller-layer stream-ordered allocation/free around the launch
(`grouped.backend_utils.wrapper_workspace`), or use an XLA custom call that owns
scratch. A cached plan's latest reference cannot cover overlapping calls. Test
multiple pending consumers on independent streams with an intercepted bounded
byte copy; never run a real kernel on an intentionally recycled scratch pointer.
For direct API workspace, validate CUDA device type and the operand's ordinal
before launch (`Workspace(..., device=...)`); byte size/alignment alone also
accept host memory.

DLPack metadata export must select the producer's `__dlpack_device__()` and
restore the caller's context even on errors; `stream=-1` avoids synchronization
but does not bypass a producer's current-device check. For CAI-only workspace,
use runtime pointer attributes: older cuda-python driver bindings return zero
for `CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL` on every GPU. The foreign-device,
export-failure and capture checks in `core/cutedsl/test_workspace_device.py`
cover these boundaries with two visible GPUs.

**R3 — a dead ABI slot (the compiled kernel never dereferences it).** In order
of preference: (1) compile it out — an `Optional`/`None`-typed kernel parameter
read only under `cutlass.const_expr(flag)`, with `flag` in the compile key, and
`None` passed at BOTH compile and launch (DSA sm90 `mTopkIdx`, `mTopkLength`);
(2) pointer ABI: bind `0` (`prepared.py` `sinks_ptr`, dense `o_desc_ptr` /
`meta_ptr`); (3) tensor ABI with a required `cute.Tensor` parameter: borrow the
bytes from the workspace via R2 (`sdpa/bwd/api_dsl.py` dense `seq_kv` /
`desc_words`). Never a cached `torch.zeros` dummy, never `q.ptr` borrowing
(a live read then reads Q bytes silently instead of faulting). `0` is legal at
every layer: `cuTensorMapEncodeTiled` accepts NULL, the DSL rejects only negative
addresses, the tvm-ffi positional entry carries pointers as ints.

**R4 — a live per-batch table the caller did not give you (lengths, offsets,
scale scalars).** R2 (carve) + fill on the launch stream with one async op
(`cuMemsetD32Async`, `cuMemcpyHtoDAsync`, `buffers.memset_zero_async`), or make
it a scalar kernel argument. Never `torch.tensor(values, device=...)` per execute
(pageable H2D + implicit sync), never build it at `compile()` into a plan-owned
tensor.

**R5 — the input is not in the layout/dtype the kernel takes.** Decline in
`check_support()` with a `NotImplementedError` naming the tensor and its
strides/dtype (`_thd_check_strides_native` in `sdpa/fwd/api_dsl.py`), so the
Router picks another engine. Not `.contiguous()`, not `.to(dtype)`, not a
repack/copy-back — even into the workspace (Rule 2). If the engine is meant to
serve that input, the kernel reads it natively.

**R6 — something must block the host (host-planned work items, a D2H read
of lengths).** The engine is non-capturable: check `cuStreamIsCapturing` and
raise before doing it (`linear_attention/cake/compiler.py::check_not_capturing`),
and say so in its docstring. Never a silent `cuStreamSynchronize` /
`torch.cuda.synchronize()` / `.item()` on a build or execute path.

**R7 — you need a cuDNN handle and the caller gave none.** Graph API lowering:
`_pygraph._backend_handle_for_lowering` (process default, one per thread and
device, stream 0, destroyed at exit) — the graph never owns one. torch-op
layers: the per-device cached handle re-streamed to torch's current stream
before every call (`linear_attention/ops/common.py::get_handle`,
`ops/norm/_common.py`). Never `cudnn.create_handle()` inside a plan, engine or
C++ graph object.

**R8 — you own a CUDA resource whose release can be GC-timed (only tests and
the C++ PyGraph may).** Release inside `cuThreadExchangeStreamCaptureMode(RELAXED)`
and restore in `finally` (`frost/buffers.py DeviceBuffer.__del__`,
`pygraph.h CaptureModeGuard`). Production plans and engines reach this recipe
only if R2/R7 were skipped — fix that instead.

**R9 — proving it.** Around a warm `build()` + `execute()`:
`torch.cuda.set_sync_debug_mode("error")` (no sync), and
`torch.cuda.memory_stats()["allocation.all.allocated"]` unchanged across three
executes (no allocation) — `test_sdpa_prepared_thd.py::test_execute_allocates_nothing_and_never_synchronizes`,
`test_sdpa_bwd_thd_sm80.py::test_graph_thd_execute_does_not_allocate`. For a
capture-safety claim, `test_cuda_capture_lifetime.py` (GC inside a global-mode
window, then replay and a native launch). For R1, monkeypatch
`torch.cuda.ExternalStream` to raise and drive the path with handle 0
(`test_torch_stream.py`).

For a host-overhead migration, measure warm enqueue and captured GPU replay
separately, and check changed-input outputs after timing. A tiny async memset
can introduce cross-engine waits between graph kernels even when their kernel
durations improve. If replay regresses, inspect profiler timestamps between
nodes as well as individual kernel durations; the prepared FP8 scalar amax
reset uses an SM kernel to avoid that wait. Keep the empty-input reduction
identity correct when no attention host is launched
(`test_prepared_fp8_empty_thd_resets_amax_without_attention`).


**Rule 9 — backend and FROST share one FE Python graph contract; no special
treatment at the caller boundary.**

- **Same API and semantics.** For workloads supported by both engines, graph
  preparation and execution must accept the same arguments and preserve the
  same meaning of bindings, shape/stride overrides, outputs, workspace, streams,
  capture/replay, ownership, and invalidation. Users and integrations such as
  FlashInfer must not branch on the selected engine to invoke a different API.
  A private method consumed outside FE is still a caller boundary for this rule.
- **Preparation belongs inside FE.** Prepare plan-derived binding state during
  build/compile. Geometry first supplied through execute-time overrides can be
  prepared and reused internally when it becomes known; it does not require an
  extra caller-visible preparation step. Keep current buffer observations and
  required checks per call. Any optional lower-level interface needs a demonstrated
  benefit beyond internal caching and must serve both engines under one contract.
- **Optimize the shared path.** Reuse provider-independent preparation,
  UID ordering, metadata validation, binding, and caching work so host-overhead
  improvements benefit both engines. A backend-only prepared entry plus the old
  FROST path is not a unified optimization; neither is merely renaming that entry.
  Engine-specific launch lowering stays internal, without forcing either engine
  through a slower common implementation. This does not require factoring kernel
  pipeline templates into a common implementation.
- **Capabilities may differ; the contract must not.** Express real differences
  through support checks and plan selection. Decline unsupported requests under
  the common contract; never silently ignore an argument, weaken validation, or
  require a caller-side workaround for one engine. Any internal fallback must
  preserve the same execution and lifetime semantics.
- **Verify both routes explicitly.** Changes to this shared contract need tests
  using supported plans from both engines with the same caller code, covering
  fresh bindings, overrides, replan/invalidation, workspace/stream changes, and
  capture/replay as applicable. Check outputs and invalid-input behavior, not
  heuristic ranking or plan-list order. For shared host optimizations, measure
  both routes and report their scope separately; one engine's result is not proof
  for the other.
- **Native context checks preserve the thread's execution policy.** A real
  stream determines its context; default-stream sentinels use the handle's
  device. Checking only whether any context is bound misses a caller on the
  wrong GPU. Retain primary contexts once per device, including repeated cold
  calls. `core/graph/test_ensure_current_context.py` covers cold threads,
  foreign contexts, first native use, lazy import and retain count; its
  foreign-device cases require two visible GPUs.
- **A guard that keeps the backend's planner away from a graph covers EVERY
  path that creates a backend plan, not only the heuristics query.** The cc 10.7
  single-query MXFP8 guard (`manifest.EngineFamily.backend_guard`,
  `sdpa/fwd/backend_guard.py`: the cuDNN 9.26 / 9.27 planner SIGSEGVs there)
  first intercepted the planning sequence only -- `_finalize_backend_layout`
  and `backend_plan_entries`. `create_execution_plan(<backend engine id>,
  knobs)`, a replayed autotune record, still reached the C++
  `Graph::create_execution_plan` and took the process down the same way (rc 139
  on 9.26.0.51, no exception to catch), so "explicit diagnostic selection stays
  available" was a crash for exactly the selection it named. The two
  out-of-sequence creators, `_append_backend_plan` (the explicit pin and the
  materialization of a replayed entry) and `_lower_backend_plan` (a late
  heuristics query for a classic call), go through
  `_pygraph._refuse_guarded_backend` and raise the typed decline with the guard's
  reason; `key()` / `serialize()` only lower, which completes on that domain.
  Detector: `sdpa/frost/test_sdpa_backend_guard.py::test_explicit_backend_pin_on_a_guarded_graph_is_a_typed_decline`
  (host-only, cc 10.7 modelled, a `_lower_backend_graph` tripwire), seen RED on
  the first guard. A version-bounded guard also needs the detector that
  re-measures its bound (test/AGENTS.md).
- **The names FlashInfer reads are pinned.** `cudnn.sdpa_capabilities()`,
  `cudnn.frost.cutedsl_runtime_available(device_cc=None)`,
  `cudnn.datatypes.torch_to_cudnn_data_type`, `pygraph.selected_engine` /
  `.backend_plan_entries()` and the private spellings FlashInfer still uses
  (`_SdpaThdBinder.supports_*`, `datatypes._torch_to_cudnn_data_type`) are a
  caller boundary: `test/python/sdpa/frost/test_flashinfer_consumed_surface.py`
  (host-only) pins each with its signature, including the cc 10.7 + 4.7.x DSL
  -> False cell of the arch-aware probe. Renaming one means updating
  FlashInfer's fallbacks first; that test failing is the reminder. The runtime
  probe takes the part because the DSL floor is 4.7.0 everywhere and 4.8.0 on
  cc 10.7 (Rule 7): a device-blind "available" is the blind spot that routes a
  cc 10.7 graph to the backend engine.

**Rule 10 — a kwarg that makes a graph ineligible for the python engines is
read by the family's analyzer under the spelling of EVERY op that takes it.**

- The analyzer's record is the node's kwargs verbatim
  (`sdpa/graph_analyzer._record_from_node`), and the rows gate on FACTS derived
  from it; a capability no row claims still needs its fact, because the decline
  by name is what keeps the graph on the backend -- a False fact is a silent
  admission. `sdpa()` / `sdpa_backward()` spell the score modifier `score_mod` /
  `score_mod_bprop`, `sdpa_fp8()` spells it `fn`; `has_score_mod` read only `fn`,
  so to the rows no f16 graph ever had a modifier. The moment 1.31 gave the
  SM100 prefill row a d64 flavor it served TransformerEngine's flex-attention
  graphs (b2 h4 s16 d64) as plain attention and the modifier was dropped
  (post-scale bias: 38 % of O outside 5e-2; softcap: 33 / 8192 elements). It
  surfaced as an apparent cuDNN 9.27 -> 9.28 regression in TransformerEngine's
  flex-attention tests; the change that mattered was the FE python package
  1.30 -> 1.31 in the same test-container refresh.
- When an op gains a kwarg a kernel cannot honor, add the spelling to the
  analyzer and a cell to the detector in the same change; `graph.selected_engine`
  on a graph with the kwarg is the one-line check.
- Detectors: `test/python/sdpa/frost/test_sdpa_score_mod_decline.py` (host-only:
  the fact on every spelling, every fwd / bwd row declines, the row admitting
  the control graph declines by name) and
  `test/python/sdpa/graph/test_flexible_sdpa_te_shape.py` (the TE shape is
  served by the backend and O carries the modifier).

## Frontend-only kernel package layout

```
python/cudnn/<operation>/            # or sdpa/<direction>/, gemm/cutedsl/<layout>/<fusion>/
├── __init__.py                      # exports the public functions via __all__
├── api.py                           # <op>_<direction>() / <op>_<direction>_jax() functions
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
├── frost/                           # the FROST GEMM engine (JIT fused matmul chains from cuDNN graphs)
│   ├── sm100/, sm120/               #   one tree per arch family: compiler.py + epilogue_codegen.py + kernel_templates/
│   ├── compiler.py, epilogue_codegen.py  # facades: become the active family's module (arch_family.py)
│   ├── kernel_templates/            #   template code SHARED by both trees (split-K reduction)
│   └── ln_prologue/                 #   the LayerNorm-prologue gated dual GEMM substrate: host (__init__.py) + ONE RENDERED body per arch (two arms: pairing=columns = TriMul K1, pairing=accumulators = TriMul K3)
│                                    #   (sm107_dual_gemm.py; a rendering for one arch is not a registry template, so not under kernel_templates/)
├── ops/                             # backend-independent torch custom-op contracts
└── reference/                       # pure-PyTorch MATMUL/POINTWISE correctness engine
```

Shared helpers (schedulers, metadata utils, e.g. `gemm/cutedsl/grouped/moe_*.py`) stay internal to the family package — never exported through `cudnn`.

## CuTeDSL kernel bodies

Loaded integer vectors can infer `Int8` while retaining no signedness metadata.
Set the declared integer dtype on the vector wrapper before widening to INT32.
CuTeDSL 4.8.0's unary integer vector negation can also crash when building its
zero constant; subtract from a typed zero vector. Keep each graph node's compute dtype.
`gemm/frost/test_matmul_mainloop_fusion.py::test_mainloop_int8_identity_preserves_signed_arithmetic`
checks both operands, compute dtypes and B binding orders with exact comparisons.

Named barriers inside warp-specific branches need unaligned sync and arrive
instructions. CuTeDSL 4.8's `NamedBarrier.arrive_and_wait()` emits an aligned
barrier, including through `TmemAllocator.wait_for_alloc()`. Check the launch
with Compute Sanitizer `--tool synccheck`; ordinary parity tests can pass despite
divergent-barrier errors. The SM100 GLU/Hadamard/quant JAX parity tests cover this.

A nonblocking barrier arrival must complete before the same warp uses that
barrier again. Use a separate barrier for tile completion when consumers can
outrun producers after a subtile arrival. Synccheck alone can miss this hang;
`test_canonical_jax_glu_hadamard_quant_delayed_act` delays ACT warps to expose it.

**Do not factor code out of a `@cute.kernel` body into a plain Python helper.**
The DSL AST-transforms only the decorated function's own source: `for` becomes
an `ir_loop`, `if` becomes an `scf` region. A helper called from the kernel is
not transformed, so the ops it emits can land outside the enclosing region.

Hoisting an 11-line block that ran correctly inline into a
`write_clamped_kv_descs(...)` helper — called from inside
`if nvvm.elect_sync() and tidx < 32:` — turned 212 passing forward tests into
31 failures (`Error building ...`, traceback through `ir_loop` →
`scf_execute_dynamic`). Unrolling the helper's own loop did not help; the
helper *call* was the problem. Duplicating the block across flavors is the
correct trade here. Factor only host-side code, or code you can mark
`@cute.jit`.

Never reuse a persistent loop index in a nested unrolled loop. In GDN-2 prep,
reusing `j` for operand packing changed the next raw-stage refill index.
The GDN-2 `test_prep_fwd_matches_uncut_and_reference` and
`test_prep_checkpoint_series` cases in `linear_attention/test_la.py` exercise
multiple work items per CTA and detect this regression.

Related: inside a kernel body, `for x in (a, b)` over a Python tuple is
rewritten into a dynamic `ir_loop` and cannot iterate heterogeneous objects
(e.g. `GridConstant[TensorMap]`). Unroll it, or use `cutlass.range_constexpr`.

**A named barrier reached from a warp-role branch is a subset barrier, even
when its declared participant count is correct.** `pipeline.NamedBarrier`
lowers its operations with the aligned-CTA promise by default; using it from
only the producer, MMA, epilogue, or reduction warps is undefined. Lower both
`barrier_cta_arrive` and `barrier_cta_sync` with `aligned=False`, including
split-phase arrive/wait protocols and TMEM allocation/deallocation gates.
Do not call `TmemAllocator.wait_for_alloc()` for such a gate: it rebuilds an
aligned named barrier, so call the original subset barrier directly. Run the
affected kernel under `compute-sanitizer --tool synccheck`; ordinary numerical
tests do not diagnose this contract violation.

The same rule applies to inline PTX: `bar.sync` is an alias for
`barrier.sync.aligned`, even with an explicit participant count. Use
`barrier.sync` for a warp-role subset and inspect the emitted PTX; successful
replay and synccheck runs do not establish that the aligned-CTA promise is
valid. D192 dense and THD backward share this Stats-publication barrier.

**Detector.** These break at `compile()`, not at import — `python -c "import ..."`
and `pytest --collect-only` both stay green. After any refactor of a kernel
body, run that flavor's own tests.

**A `setmaxnreg` register split needs `min_blocks_per_mp=1` on the launch.**
With `.reqntid` alone (what `block=` emits) ptxas cannot bound the register
count at entry: it prints C7508 and drops every `setmaxnreg`. The SASS then has
no `USETMAXREG`, the consumer warps never get their raised budget, and ptxas may
size every warp for several CTAs per SM — the sm120 dual MoE GEMM fell to 40
registers with 11 KB of spills and ran 4-6x slower. `.launch(...,
min_blocks_per_mp=1)` emits `.minnctapersm 1` and makes the split live. The DSL
never surfaces C7508 (not even under `CUTE_DSL_COMPILER_OPT=warnings`), so the
numerics tests cannot see a lost split. Once the split is live, the producer
budget is a hard limit: keep per-lane producer state small (one elected lane
holding a tile's worth of GATHER rows spilled at 24 registers and made sm120 MoE
GATHER up to 1.45x slower; each TMA-warp lane now issues its own GATHER4s), and
accept only warp grids the split fits (`Sm120KernelTemplate.warp_layout_reject`).
**Detector:** a compile-only SASS pin — one `USETMAXREG.DEALLOC` and one
`TRY_ALLOC` at the template's counts, plus a spill bound on producer-heavy cases
(the SDPA pins in `sdpa/frost/test_sdpa_bwd_d256_sm100.py` are the pattern).

**Immutable tensor-map reuse is local to a consumer and a launch.** A persistent
loader or store warp may skip acquisition only while it keeps using an unchanged
map it already acquired. Track the sequence/expert per consuming warp, reacquire
on a switch, and reset that state at kernel entry. Never cache acquisition in a
host plan or infer it from another CTA's fence. Verify fresh pointer bindings and
changed device-side boundaries during CUDA-graph replay against an independent
reference; `sdpa/frost/test_sdpa_thd_tensormap_acquire.py` and
`gemm/cutedsl/test_wgrad_tensormap_acquire.py` exercise those lifetimes. Fences
surrounding descriptor replacement, data visibility, and pipeline synchronization
have different contracts and remain required.

**Every TMA scatter4 source needs 128-byte SMEM address alignment.** An aligned
ring base does not align each four-row issue: a 16-byte staging row advances
only 64 bytes, so every second issue faults. Select STG for such staging pitches
unless padding is included in both the layout and resource accounting. Detector:
`gemm/frost/test_moe_grouped_block_scale_matmul_fwd.py::test_moe_block_scale_narrow_n_routing`
(`scatter-False-24`, including capture/replay), verified RED before the store gate.

For fractional-microsecond performance changes, repeat A/B with shared addresses,
reversed construction order, and an independently compiled unchanged-source
control. An A/A that shares one compiled kernel does not bound the variation
introduced by separate compilation and code placement.
Also check that an optimization's admission condition is exercised by the
repository benchmark shapes or a documented target workload. A favorable
synthetic shape establishes a local effect, not representative benefit; it does
not by itself justify another compile-cache specialization.

Resident block-scale SF words use only TMEM left after the accumulator staging
decision. Account for every distinct operand, M block and word; fake-dequant
sides continue to read their initialized slot zero. Keep the SF-ready/AB-ready
waits and producer-stage release intact when moving the copies. Detectors:
`test_block_scale_resident_sf_tmem_budget`, `test_block_scale_resident_sf_exact_fit`
and the existing one-sided-dequant numerics in `gemm/frost/test_block_scale_matmul.py`.

An `empty` mailbox acknowledgement must order prior reads before a producer
overwrites the slot. Read-only consumers still need this write-after-read
ordering; `bar.warp.sync` alone does not establish it with a producer outside
that warp. Keep the MoE scheduler's release arrives unless a separate ordering
proof covers every slot load. Numeric/racecheck success alone is not that proof.
`test_moe_block_scale_mailbox_acknowledgements_order_prior_reads` in the existing
`gemm/frost/test_template_epilogue_parity.py` pins this until such a proof exists.

Small-M block-scale planning uses `GemmFacts.shapes_fixed` to distinguish fixed
graphs from override-enabled graphs. Only fixed shapes select token widths
8/16/32 after swapAB; overrides keep width 64 and at most 48 weight tiles.
Dynamic shapes bypass strategy selection, and explicit knob replay bypasses
automatic choices. Keep this bounded rule in shared `gemm/frost/planning.py`,
probe both its proposed plan and any retained baseline, and leave fitted cost
coefficients unchanged. Automatic candidates still use workspace reduction.
The B200-targeted rule leads on SM100; other GPUs retain the existing first
choice and include the new geometry as an autotuning candidate. A four-shape
SM107 comparison found both wins and regressions from unconditional preference.

The dense SM100, mainloop-fusion, block-scale and SM103 hosts resolve the L2
swizzle width once per call from the current logical M/N/K, then pass an Int32
kernel argument. Do not compute this shape-only division chain in each warp.
Keep the mixed-CGA pinned identity mapping and the shared helper's runtime
width-one division bypass. MoE still resolves its per-group raster on device
because the group extents are device data.

Narrow block-scale tiles here extract arbitrary SF sub-rows, including across
128-row atoms. They have no kernel-N <= 32 requirement: explicit narrow replay
on override/dynamic graphs remains legal, with SF capacity and output checks on
every call. Do not import the restricted narrow implementation's shape gates.
Automatic small-M planning still uses the fixed/override/dynamic rules above.

`TileConfig.cluster_k_reduce` explicitly selects the split-K reduction mode.
False (including old knob records) uses workspace plus a separate reducer;
True puts all slices in one cluster and reports zero partial workspace.
The K cluster dimension is derived from this flag and `split_k_slices`, never
an independent tuning axis. Reject unsupported templates, unlaunchable clusters
and insufficient SMEM; never change a requested mode during build or execute.
Automatic planning retains False. Keep the flag in canonical names and public
knob round-trips. Epilogue codegen takes `workspace_split_k` directly; do not
rewrite the config's split count or wrap template regions through source regexes.
Keep cluster device code only in templates that implement the full protocol;
mainloop fusion and SM103 must contain no incomplete cluster branches. The
epilogue parity test compares their shared base drain with the supported
templates after removing only cluster reduction and its root-write predicate.
Other bounds, loads and stores remain part of that comparison. Name the cluster
CTA barrier and derive both its participant count and fragment span from the
caller's epilogue warp count.
For the SM100 dense and block-scale cluster implementations, every
epilogue lane, including masked output lanes, stages its fragment before an
async-proxy fence and CTA rendezvous. Elected bulk copies register their byte
counts at the root before issue. The root waits for transaction completion,
reads local receive slots and gathers all readers before a release/acquire
acknowledgement at cluster scope. Budget S fragment slots (S-1 receive plus one
local source), not S full accumulator tiles. Keep the cluster exit
rendezvous: `test_cluster_splitk_persistent_mailbox_reuse` exposed both slot
reuse hazards with an elected-only handoff and premature DSMEM lifetime under
racecheck. Gate TMA compute and stores together in one SSA region; separate
guards leave the store's vectors undefined. For mixed-width block-scale inputs,
CTA-K elements use the wider operand's bit width after either orientation;
`test_cluster_splitk_mixed_width_k_tiles_after_swap` covers shallow legal K.
Both compiler families use `recipe.check_split_k_shape` in the launch guard
and `explain()`: every split must have a K tile, and workspace reduction must
retain its compiled N chunk. Check before seeds, workspace writes or launch;
reject without changing the selected reduction mode.

## The public function contract

Public functions (`cudnn.ops.<op>_<direction>(...)` / `cudnn.ops.<op>_<direction>_jax(...)`, see `docs/operations/api_conventions.md`) allocate outputs and return them in a stable, documented order. PyTorch functions return a **`TupleDict`** (dict that also unpacks as a tuple). JAX functions return a plain tuple or dict: `jax.jit` rejects `TupleDict`, which is not a registered pytree.

## Adding a new frontend-only API — required checklist

1. Kernel package under the closest existing family (layout above).
2. Public functions in `api.py`.
3. Exports: `_LAZY_EXPORTS` in `python/cudnn/ops/__init__.py` (or `python/cudnn/ops/<family>/__init__.py` for an op in a family); register any new package dir in `pyproject.toml` packages list.
4. Docs: page under `docs/operations/<family>/`, listed in `docs/fern/versions/latest.yml` and linked from `docs/operations/overview.md`.
5. Tests: `test/python/<op>/cutedsl/test_<op>.py` (+ `_utils.py`/reference), covering supported and unsupported configurations and numerical reference comparison.
6. DSL version gate (Rule 7): the route declines with a version-naming error below `CUTEDSL_MIN_VERSION`, and the tests skip there instead of failing.

The `cutedsl-kernel-integration` skill (`skills/cutedsl-kernel-integration/`) documents this workflow in detail, including how to classify a kernel into a family — follow it for any kernel integration.

## Other notes

- `wrapper.py` `Graph` context manager (the pythonic graph builder) requires cuDNN backend ≥ 9.12 (`backend_version() >= 91200`) and builds plans on `__exit__`.
- Torch custom-op implementations live with their owning operation family and may be re-exported from `experimental/ops/` while maturing (pattern doc: `docs/utilities/adding_torch_custom_ops.md`); they cache built graphs per config and use stable `_UIDs` enums.
- dtype conversions go through `datatypes.py`, which probes torch/cutlass availability lazily — keep it that way.
- Formatting: black, line length 160.

CUDA-owning objects, GC-timed release and stream capture: Rule 8.


## DSA training composition

Metadata for backward and targets has two different ordering contracts. Training
may compact valid active indices only if forward and backward share that exact
metadata. Score targets must preserve original slots, mask invalid/inactive
entries, and normalize over retained slots after summing heads. Never reuse
compacted training indices as the caller-visible target order. Detector:
`deepseek_sparse_attention/cutedsl/test_DSA_training.py::test_native_training_and_original_score_slots`
checks holes, duplicates, bounded lengths, all-masked rows and target alignment.

## Linear-attention decay factorisation range

A chunk kernel that splits a decay `2^(g[t] - g[s])` into `exp2(+g[t])` on one MMA
operand and `rcp(exp2(g[s]))` on the other is exact only while each factor stays
in the operand's range: an in-tile log2 prefix of -126 with bf16 operands (`exp2`
goes subnormal, `rcp(..., ftz=True)` returns inf, `0 * inf` is NaN), and about -16
with fp16 operands (`k * 2^-g` overflows). Both lie inside the documented gate range,
`alpha = exp(g)` in (0, 1], and a test drawing `alpha` from `uniform(0.5, 1)` misses them.

- Route such tiles through `linear_attention/frost/common/steep_decay.py`, which keeps every
  exponent at `g[t] - g[s] <= 0`. Its module docstring holds the dispatch and
  synchronisation invariants.
- Keep the exact path out of the fast kernel. Compiled in behind a never-taken branch,
  it cost the backward 13%.
- Bound the inverse operand with the key magnitude as well as the gate prefix:
  unnormalized `k=5` overflows fp16 at a prefix of -13.75.
- Sum individual token logs over the interval. Never subtract inclusive fp32 prefixes:
  a -1e8 gate erases the following -1 gates from both.
- Reduce corrections in a fixed order, including when several channels are steep.
- Use `encode_gate_exchange` for scan classification and exchange encoding. Keep stage selection and
  barriers in each schedule. All six chunk schedules share gate conditioning, scans, row normalization
  and operand packing through the `@cute.jit` stages in `common/gdn2_operands.py`. The rolled FP32
  backward scan retains its local load/store schedule. Read exchange scales before releasing a raw
  stage; register-only packing may follow that release. `encode_gate_exchange` accepts log2 prefixes
  only, and returns exponentials or exact token logs according to the channel ballot. The forward kernels
  declare the steep scratch arrays from their cfg cosizes; the backward kernels carve it from the bank fill
  at cfg byte offsets and assert the capacity in `build_cfg`.
- Range tests must vary state-gradient magnitude independently of gate magnitude. A large finite
  FP32 seed, or a carry accumulated from finite FP16 output gradients, can overflow an undecayed
  FP16 state operand even when the decayed carry and all returned gradients fit. The large-seed
  and large-carry regressions in the steep section of `linear_attention/test_la.py` cover both backward
  implementations.
- Keep steep state gradients and bounded key operands in FP32 through their projections
  (`common/gdn2_state_grad.py`). Narrowing a bounded key first can erase a term that becomes
  representable after multiplication by the state gradient. The caller-owned per-CTA workspace
  is sized by the kernel cfg's `dstate_scratch_cosize` (an fp32 state plus the bounded-key ring); state
  publication uses `mb_dstate_input_ready`, and the
  key ring uses `mb_k_decay_inv_ready` / `mb_decay_done`. Scalar readers acquire these
  thread-produced barriers directly, including `mb_u_smem_ready` for restored dK; tensor-core
  completion only orders the async tensor-core operations. Full backward's
  CG2 release protects both that workspace and U until the restored-key projection finishes,
  including the last chunk before the persistent CTA takes another work item.
- Check mild state-operand range independently: FP16 inputs, `g=0`, zero Q/K/V/initial
  state and an FP32 final-state gradient of 70000 isolate overflow in the mild path
  at `e4896ef2`, with finite FP64 reference gradients. Steep-gate coverage does not
  establish arbitrary FP32 state-gradient support for that path.
- Replay the same CUDA graph and workspace through mild, steep and mild gates. Replaying fixed
  inputs does not check whether the steep flag and scheduler tickets reset. Check finiteness before
  bitwise equality; matching NaN bit patterns are not a successful numerical replay.
- An exact relaunch must not read what the fast launch wrote. Seed overwrite-capable plans
  from a state snapshot (`common/state_snapshot.py`), chosen from the graph contract, never
  from first-call pointer equality: a graph can bind separate state buffers at warmup and
  alias them later. A forward graph without a final-state output cannot overwrite its seeds;
  indexed input pools in such graphs need no snapshot.

Detectors, all in the steep section of `linear_attention/test_la.py`: `test_steep_channel_decay_fwd_bwd`
(and `test_steep_channel_decay_fp16`), `test_steep_fp16_inverse_range_without_normalization`,
`test_steep_multiple_channels_repeatable`, `test_steep_forward_state_alias_after_graph_warmup`,
`test_steep_channel_decay_overwrite_initial_state`, `test_steep_channel_decay_backward_overwrite_initial_state`,
`test_steep_read_only_state_pool`, `test_steep_strided_state_overwrite` and
`test_steep_summary_matches_reference`. Scratch bounds are asserted in each kernel's `build_cfg`.

State snapshots preserve both the source dtype and its element width. The snapshot
kernel copies raw bytes and does not convert FP16 to FP32. Exercise every accepted
state dtype with forward and backward aliasing, padded storage, and changed-state
CUDA-graph replay. The overwrite and replay cases in the steep section of
`linear_attention/test_la.py` cover these paths.

Pairing sparse queries requires per-query masking before dQ arithmetic when a
key is nonfinite: a key valid for one query can be masked for its neighbor,
and zero dS multiplied by that key still produces NaN. A union mask alone is
insufficient, including across an unaligned packed sequence boundary. Keep any
nonfinite fallback entirely on device and re-evaluate it on each graph replay.
The detector is `QSA/cutedsl/test_qsa.py::test_paired_backward_nonfinite_isolation`,
which changes finite/nonfinite inputs under one captured graph for both shared
and separate packed sequences.
