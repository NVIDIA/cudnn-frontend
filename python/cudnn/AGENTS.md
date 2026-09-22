# python/cudnn — Agent Guide

The `cudnn` Python package: pybind11-backed graph API plus pure-Python **frontend-only OSS kernels** (CuTeDSL). See `README.md` in this directory for the package inventory and [../../AGENTS.md](../../AGENTS.md) for build/test commands.

## Import-time rules (the most common way to break this package)

- `import cudnn` must work **without** torch/cutlass/cuda-python installed. Everything that needs them is exported lazily via `_LAZY_OPTIONAL_IMPORTS` in `__init__.py` — a module-level `__getattr__` imports the submodule on first attribute access and re-raises failures as `ImportError` that names the missing framework module (torch, jax, cuda-python) alongside the base `pip install nvidia-cudnn-frontend` hint — or, for a CuTe DSL below `CUTEDSL_MIN_VERSION`, points at the DSL upgrade (Rule 7).
- Never add an eager `import torch` / `import cutlass` to `__init__.py` or anything it imports transitively. `api_base.py` itself imports them at top level, which is why kernel classes must only be reachable through the lazy table.
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
  (`scratch_workspace_bytes()` contract; a missing workspace raises, never
  allocates). No new `_dummy` callers: the remaining sites are enumerated by
  the ratchet test (`test_sdpa_fwd_rule8_ratchet.py`) and retired per R3.
  Prefer compiling the unused operand out entirely (CuTeDSL specializes on
  `None` via `cutlass.const_expr` — see the SM120 SDPA kernel's optional
  lse/sinks).
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

Known violations, all pre-existing and each needing a kernel-side change, so
none is precedent:

- `cu_k.to(dtype=..., device="cpu")` in the SM80 packed-THD WRAPPER path
  (`_sm80_thd_backward` in `sdpa/bwd/api_dsl.py`), taken only when the caller
  passes no `max_s_kv` hint. Reachable only through the standalone wrapper: the
  `sdpa_bwd_sm80` engine path bounds its kv-tile grid and relay counter from
  the graph's envelope `S_max` and turns the per-batch lengths into
  `cu_seqlens` on device, so `graph.execute()` never reads a length. Still a
  violation on the wrapper surface (a caller contract, documented there).

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
- **Issue #604 is closed**: the SM80 THD compiles (forward and backward) take
  the packed token extents as `cute.sym_int` and key on `b = 1, sq = skv = 0`
  plus the plan-time sequence count; the regression tests are
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
- **A raw stream handle never goes straight into `torch.cuda.ExternalStream`.**
  Every eager caller on torch's default stream hands us a default-stream
  sentinel (`0`, `cudaStreamLegacy` = 1, `cudaStreamPerThread` = 2), and torch
  before PR pytorch/pytorch#183258 (in v2.13.0; NGC 26.06 and torch <= 2.12
  lack it) returns a fresh NON-BLOCKING pool stream for `ExternalStream(0)`.
  Torch work issued in that context is unordered with a kernel launched on
  `CUstream(0)`: on an idle GPU the copies win the race and every isolated
  test passes; under xdist load the kernel reads stale conversion buffers and
  a staged output is copied back before it is written (the qa sm90
  `hopper_cuda` reds, PR #1165 — the same trap FROST SDPA hit in #682/#717/#860).
  Map the sentinels and torch's own default stream to
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

- The `pyproject.toml` floor on `nvidia-cutlass-dsl` (`>=4.6.2`) is the
  **downstream** floor, not ours: vLLM and SGLang inherit quack-kernels'
  `==4.6.2`, and a higher floor would make this package uninstallable next to
  them. The FROST-derived kernels need more (`CUTEDSL_MIN_VERSION`, 4.7.0). So
  an installed DSL that satisfies pip can still be below what a kernel needs,
  and every backend/kernel must cope with that at runtime.
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
- Known floors — extend this list when you take a dependency on a newer API,
  and say so in the PR body if it raises the floor of a user-facing op:
  `cutlass.experimental.*` (primitives, `cuda.tensor_map`; everything under
  `cudnn/frost/tile_dsl` inherits it) → 4.7.0.
- Tests that import a kernel module directly `pytest.skip` on a too-old DSL —
  they do not fail. CI runs the `oss:` lanes across the supported DSL versions
  (`ci/stages/oss_tests/jobs.yml` in internal CI); a lane below your floor
  must show skips, not errors.
- Why: PR #799's `causal_conv1d_update` imported `frost.tile_dsl` from a route
  with no version check and broke the 4.6.2 lane — the version vLLM and SGLang
  ship — with a bare `ModuleNotFoundError: cutlass.experimental`; the bulk
  route next to it had the check and declined cleanly. Earlier, PR #854's
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
a no-op via the raw-handle fast path; `0`/`1`/`2` and torch's default stream
resolve to `torch.cuda.default_stream(device)`.

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
(APIBase adapters: `scratch_workspace_bytes()` + `WorkspaceCarver(workspace, bytes, label).take(numel, dtype)`
in `api_base.py`, re-exported by `sdpa/fwd/api_dsl.py`.) `Workspace(None, ...)` already raises
`"<owner> requires a N-byte workspace but execute() received none; allocate
graph.get_workspace_size() bytes and pass the buffer to execute()"` — reuse
that error, never fall back to `torch.empty` / `DeviceBuffer` when the caller
passed nothing. A path with no workspace contract gets one. A direct adapter
caller (the `<op>_wrapper`, a test, downstream code constructing the class) is a
caller: the wrapper allocates `scratch_workspace_bytes()` once per call and
passes it; an engine never allocates because the wrapper forgot. Where a
wrapper allocates per call on the caller's behalf, it allocates under
`stream_context(<launch stream>)` (R1): the caching allocator orders a block's
reuse only against the stream it was allocated on, so scratch allocated on
torch's ambient stream for a handle re-streamed to a side stream is a
use-after-free waiting for load.
An APIBase that
wraps a cuDNN backend `pygraph` forwards `graph.get_workspace_size()` as
`get_workspace_size()` and takes `workspace=` at execute, validated with
`WorkspaceCarver(workspace, bytes, label)` (construct only, no `take`);
`pygraph.execute(workspace=None)` lowers to a null pointer, so None is legal
only when the size is 0 (`native_sparse_attention/sliding_window_attention`).
A staged output the kernel accumulates in a wider dtype than the caller's buffer
(a bf16 `d_index_k` behind an fp32 atomic epilogue) is R2 scratch, not an
allocation: carve the wide view, zero it with R4 (skip when the kernel's own
prologue clears it — one clear, in-kernel), launch, `copy_` into the caller's
buffer on the launch stream; the fp32-output path writes the caller's buffer
directly. The real fix is a native-dtype epilogue store
(`deepseek_sparse_attention/indexer_backward/`). A CuTe-DSL adapter that
launches on the TVM-FFI environment stream still carves from the caller's
workspace and `record_stream`s EVERY tensor the launch touches, the workspace
included, on `as_torch_stream(current_stream, device)`: the launch is invisible
to the caching allocator (`indexer_top_k/api.py`); when the kernel's fake
declares `assumed_align` above the carver's 16 B, `execute()` checks the carved
chunk's `data_ptr()` against it and raises. Descriptors over a carve (CuTe
tensors, `DeviceView`s) are host objects and may be memoized per plan keyed on
the workspace BASE pointer — the `OperandBuffer` capsule owns its own
shape/stride, so the memo holds no device memory; a caller rotating workspaces
pays the conversions again and nothing else (`hopper/kda_engine.py`; detector
`test_kda_sm90_cuda.py::test_alternating_workspaces_recarve`).
When an adapter's execute has several required outputs the kernel always
writes (DSA `SparseAttentionBackward`: `dq`, `dkv`, `d_sink`), make ALL of them
positional-required on every backend and raise `ValueError("<name> must be
preallocated; use <op>_wrapper for automatic output allocation")` on `None`;
the wrapper allocates the full set under `stream_context(launch)`. Never let one
backend require an output the others allocate — the "d_sink only for D576" split
hid a per-execute `torch.zeros_like` on every other route.
A direct adapter caller (a standalone `<op>_wrapper`, a test, downstream code
constructing `SdpaFwdDsl*` itself) is a caller: it allocates
`scratch_workspace_bytes()` and passes it (`sdpa_fwd_wrapper_sm80` does, under
`stream_context(<launch stream>)`). An adapter never keeps a `workspace is
None` allocation branch, not even for "standalone use" —
`SdpaFwdDsl._scratch_base` and `WorkspaceCarver` raise the same contract error.
Test suites get the buffer from an autouse shim
(`test/python/gemm/frost/conftest.py::_moe_plan_workspace`,
`test/python/sdpa/frost/conftest.py::_sdpa_adapter_workspace`);
`@pytest.mark.no_workspace_shim` turns it off to test the error itself.
A CuTeDSL APIBase that serves torch and JAX carves with the framework-neutral
`cudnn.frost.workspace.Workspace(workspace, nbytes, type(self).__name__).take(nbytes,
"uint8")` (`buffers.probe` reads torch via `__cuda_array_interface__`, anything
else via `__dlpack__`); a Pointer-typed kernel parameter gets `view.data_ptr()`
(tvm-ffi carries `cute.Pointer` as int), a Tensor-typed one gets the
`DeviceView`. `scratch_workspace_bytes()` is `max(align_up(kernel.get_workspace_bytes(),
128), 128)` from a kernel instance memoised at plan time and reused by
`compile()`, so the workspace is always required and a mode that needs 0 bytes
has no NULL case; legal because the helper kernel rewrites every launch's
descriptor slots and zeroes its own scheduler counter, so one buffer serves
non-overlapping executions and carries no cross-launch state. The
`*_wrapper_sm100` functions allocate via
`backend_utils.allocate_wrapper_workspace(framework, op.scratch_workspace_bytes(),
device, current_stream)` exactly where they allocate outputs, memo and cold path
alike; an eager-JAX workspace has no `record_stream`, so the API holds it until
the next execute the way it holds `_live_ptrs`
(`gemm/cutedsl/grouped/**`, `discrete_grouped/**`; detectors
`fe_api/test_grouped_gemm_rule8.py`, `fe_api/grouped_gemm/test_grouped_gemm_rule8_fusions.py`).
An APIBase whose kernel needs per-execute metadata built on device (HSTU
block-sparse CSR from `func`) plus a wide-dtype accumulator (fp32 dQ) declares
both in `scratch_workspace_bytes()` from plan-time geometry
(`hstu_{q2k,k2q,d256_bwd}_block_sparse_workspace_bytes`: pure ints, 128-B
chunks in carve order) and carves them at execute with one `WorkspaceCarver`:
`take()` the accumulator, zero it on the launch stream, hand
`carver.remaining()` to the nested builder. `compile()` has no workspace: a
builder in compile-only mode compiles from fakes (R11) and returns None, so
`check_support()` computes and caches the byte count before `compile()` runs
and the wrapper allocates from that cached value (`hstu/hstu_attention/api.py`,
`_interface.py`, `_kernels/block_sparse_builder.py`).

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
An Optional operand is Optional through the WHOLE chain: the jit `__call__`
parameter, every `@cute.kernel` parameter and every `@cute.jit` helper that
forwards it carry `cute.Tensor | None`, and every re-layout of it — host-side
(`cute.make_tensor(m.iterator, ...)`) and in-kernel
(`seqlen.offset_batch_Q(m, ...)`) — sits under `cutlass.const_expr(m is not
None)`; guarding only the final writes leaves a `None.iterator` trace error one
refactor away. Put a compile-time tripwire in the `else` branch (`assert not
self.compute_lse`) so `None` handed to a live specialization fails at
`cute.compile`, never as a silently skipped write; the flag that decides `None`
is in the compile key. Detector: allocation delta 0 across three warm executes
with the flag off, plus the compile key carrying it (`mDenom` in
`score_recompute/dense_score_recompute_sm100.py` ->
`indexer_score_unified_sm100(_mxfp8).py`; `mTopkLength` in
`sparse_score_recompute_sm100.py`;
`test_DSA_indexer_forward.py::test_denom_slot_compiled_out`,
`test_DSA_sparse_score_recompute.py::..._sm100_topk_length_none_compiles_out`).
Grandfathered `_dummy` sites in `sdpa/fwd/api_dsl.py` (23; ratchet
`test_sdpa_fwd_rule8_ratchet.py::test_no_new_dummy_callers`) are retired by
their family's kernel-side change — the fp8/mxfp8 SM100/SM107 sites by the fp8
pointer-ABI migration (dead slots bind 0, absent amax/scales compile out), the
SM120 and SM80 sites by `Optional`-typing the kernel slot. Do not convert one to
a workspace borrow when that flips a dense plan's `get_workspace_size()` from 0
(tests pin 0 for dense fp8/mxfp8 and SM100 f16); borrow only where the path
already carves (THD: the SM100 sinks slot, the SM120 V stub), and never fill a
dead slot.
`cute.compile` placeholders never allocate: a `cute.Pointer` parameter's
compile stand-in is `cute.runtime.make_ptr(dtype, <aligned dummy address>,
cute.AddressSpace.gmem, assumed_align=...)` — `make_ptr(cutlass.Int64, 16, gmem,
assumed_align=8)` for a pointer table, `make_ptr(cutlass.Uint8, 128, gmem,
assumed_align=128)` for `workspace_ptr` — never
`from_dlpack(torch.empty(...)).iterator` (the 8·E-byte `_compile_*_ptrs` device
buffers existed only for `.iterator`). A `cute.Tensor` workspace parameter's fake
is `_make_fake_cute_tensor(cutlass.Uint8, (scratch_workspace_bytes(),), (1,),
assumed_align=128)` with the SAME static extent the execute-time `DeviceView`
carries (tvm-ffi checks static dims); when the API's `_interpret_uint8_as_fp4x2`
default is True, pass `interpret_uint8_as_fp4x2=False` or Uint8 silently becomes
FP4x2 (`wgrad/_blockscaled_api.py`).

**R4 — a live per-batch table the caller did not give you (lengths, offsets,
scale scalars).** R2 (carve) + fill on the launch stream with one async op
(`cuMemsetD32Async`, `cuMemcpyHtoDAsync`, `buffers.memset_zero_async`), or make
it a scalar kernel argument. Never `torch.tensor(values, device=...)` per execute
(pageable H2D + implicit sync), never build it at `compile()` into a plan-owned
tensor. A pointer table over a uniformly strided output is CALLER-layer work:
the APIBase REQUIRES the table (`ValueError` naming the helper), a public helper
next to the wrapper derives it once per buffer on the launch stream
(`gemm/cutedsl/grouped/wgrad/api.py::wgrad_expert_ptrs`: `torch.arange(base,
base + E*stride_bytes, stride_bytes, dtype=torch.int64)` under
`_torch_stream_context`, `torch.full` for E <= 1; JAX eager: packed uint8), and
the wrapper calls it when the caller passed none. Never derive it per execute
inside `execute()` (the API cannot memoize it and it hides an R1
`record_stream`), never a host list through `torch.tensor(..., device=)`; or the
kernel takes `(base, stride)` as scalars. The fe_api sync detector (R9) flags
the host list on the first call, so the failure is loud, not a silent
serialisation (`fe_api/test_grouped_gemm_rule8.py::test_wgrad_discrete_requires_wgrad_ptrs`).
A device-side scheduler counter / ticket / semaphore the kernel self-resets is
still caller scratch: carve it (R2) and `memset_zero_async` it on the launch
stream EVERY execute, not once — the buffer is shared scratch another engine may
have used, and a launch killed mid-way leaves it dirty. The contract this buys is
"executions sharing one workspace buffer must not overlap", never "one plan per
device/stream"; a wrapper that allocates per call needs no stream/device key for
state ownership (`indexer_backward_v2_sm100`; detector
`test_DSA_indexer_backward.py::test_v2_counter_lives_in_caller_workspace`
poisons the workspace with 0xFF before each execute). The same rule zeroes an
absent optional state port (the Hopper KDA seed) on every execute, so capture
records a memset node that replays. An output the kernel accumulates atomically
(DSA `d_sink` via `sum_dSink`, SM90 `dkv_accum`) is zeroed the same way, after
`resolve_stream()` and before the launch; an output whose finalizer overwrites
every element (the dKV `convert` kernels) gets no reset at all — verify the
finalizer's row/column coverage by reading it, and say so in the comment
(`sparse_attention_backward/_interface_sm100.py`).
Plan-time scalar constants a kernel reads from memory (FP8 `alpha` / `scale` /
`descale`) are R4 too: declare ONE slot LAST in the workspace layout
(`gated_attention_block/api.py` `_Intermediates.quant`, `_QUANT_WORDS` order,
`_align_up(4 * n)`), fill each present word at execute with
`fill_word_async(ws.data_ptr() + off + 4*i, 1, init_word("fp32", v), stream)`
before the first stage that reads it, and hand kernels
`_view(ws, off, (n,), float32)[i:i+1]` slices (a `[0:4]` slice serves a kernel
that takes a short vector). Never `torch.tensor(...)` / `torch.full(...,
device=)` at `compile()` — the plan-owned tensor is the #1151 class — and never a
per-execute host list. Detectors:
`test_block_fp8.py::test_fill_quant_slot_writes_the_nine_words_on_the_given_stream`
(0xFF-poisoned workspace, side stream, sync-debug armed, allocation delta 0,
nothing outside the slot written), `::test_quant_slot_is_the_last_workspace_slot`,
`test_block_mxfp8.py::test_mxfp8_quant_slot_holds_only_the_out_projection_pair`.

**R5 — the input is not in the layout/dtype the kernel takes.** Decline in
`check_support()` with a `NotImplementedError` naming the tensor and its
strides/dtype (`_thd_check_strides_native` in `sdpa/fwd/api_dsl.py`), so the
Router picks another engine. Not `.contiguous()`, not `.to(dtype)`, not a
repack/copy-back — even into the workspace (Rule 2). If the engine is meant to
serve that input, the kernel reads it natively. When a kernel hard-codes an
operand's stride (the NSA Top-K LSE, read through a fixed `(1, s_q)` layout),
`check_support()` validates the FULL stride tuple of the normalised view, not
just `stride[-1] == 1`, and `execute()` re-checks the live tensor and raises
`ValueError` (never `.contiguous()`).
A size-1 innermost dim has no observable stride: `stride[-1] == 1` checks exempt
`shape[-1] == 1` (the SM90 `(B, 1, H).transpose(1, 2)` singleton view is
contiguous with stride `(H, 1, H)`), or a correct layout is declined (12 SM90
reds in the score-recompute batch). Contiguity-only kernels: `check_support()`
loops every descriptor with `_not_implemented_error_if(not desc.is_contiguous(),
f"<Op> addresses only contiguous {name}; got shape {desc.shape} strides
{desc.stride}")`, and the interface re-checks `tensor.is_contiguous()` at
execute with `ValueError` (`sparse_attention_backward/api.py`,
`_interface_sm90.py`, `_interface_sm100.py`); a wrapper's plan cache keys on
the contiguity so a strided call builds its own plan and is declined at
`check_support()` instead of hitting a cached contiguous plan at execute.
An output the kernel needs padded or aligned is never staged into scratch and
copied back (two hidden launches plus a per-call allocation): require the
caller's tensor to already be the view with the padded stride (`out[..., :S_k]`
over `(..., ceil4(S_k))`), validate the FULL stride tuple plus base alignment at
execute (`ValueError` naming the allocation to make), let the wrapper allocate
the padded buffer on the launch stream (R1) and return the view, and keep the
APIBase descriptor on the padded allocation with `is_contiguous()` in
`check_support()` (`indexer_forward/_interface.py::_validate_out_view`; detector
`test_DSA_indexer_forward.py::test_padded_out_is_bound_directly`).
Wrapper-layer convenience (`.contiguous()`, `.to(int32)`, padded output
allocation) lives only in `<op>_wrapper`, never in an `_interface*.py` the
engine shares; it runs BEFORE the memo key and plan build (so the plan sees the
tensors it will execute) and under `stream_context(<launch stream>)`, and it
never copies an OUTPUT (a `.contiguous()` on `out` returns a temp the caller
never sees — the old SM90 score-recompute path did exactly that).
R5 interim clause (named and dated, 2026-09): when a production contract forces
a dtype/layout the kernel cannot read natively and declining would leave the
arch with no engine (sm90 KDA: FlashInfer's `kda()` mandates bf16 g/beta/state
and int64 cu under capture; the kernels read `const float*` / `const int*`),
the conversion may stay ONLY as a BUILD-declared staging carve: sized in
`__init__` from the graph's declared dtypes and strides
(`linear_attention/hopper/marshal.py::staging_ports`), never from the runtime
view; the copy runs on the launch stream inside `stream_context`; a runtime
buffer that does not match the declaration raises naming the port; the module
docstring records it as a Rule 2 exception with the kernel change that retires
it. It is listed under GRANDFATHERED in the audit, not cited as precedent.
Express the native layout as a host predicate over strides and reuse it on
both sides: `_supports_bwd_original_qkv_layout` (unit-stride D, token/head
strides % 8, non-overlapping, 16-B base) / `_supports_bwd_direct_grad_layout`
(+ non-zero strides) gate `check_support()` (`NotImplementedError` naming
tensor, shape and strides) and are re-checked in `_interface` (`ValueError`).
Route-dependent requirements (the D=256 two-kernel TMA path: all seven tensors
contiguous) key on the dispatch ROUTE, not the head dim, so a sibling kernel
that serves the layout natively (qlen=1 direct at D=256) is not declined
(`HSTUBwdSm100.check_support`). Deleting a `.clone(memory_format=...)` /
`permute().clone()` / `empty_strided` + `copy_` staging pair is the expected
shape of an R5 patch.

**R6 — something must block the host (host-planned work items, a D2H read
of lengths).** The engine is non-capturable: check `cuStreamIsCapturing` and
raise before doing it (`linear_attention/cake/compiler.py::check_not_capturing`),
and say so in its docstring. Never a silent `cuStreamSynchronize` /
`torch.cuda.synchronize()` / `.item()` on a build or execute path. A value the
host needs that the caller already knows is a required host argument, never
inferred by a device read (R10); a device-data table the kernel consumes
(offsets, pointer arrays) is a documented contract validated from host metadata
(dtype, shape, alignment, device), not read back to check its values.
A host check of device VALUES (an offsets table, a pointer array) never earns a
sync: the default path trusts the documented device-data contract, and the
check lives behind one env var read once at import, runs `cuStreamIsCapturing`
FIRST and raises under capture, and is never memoized
(`gemm/cutedsl/grouped/backend_utils.py::debug_validate_offsets`,
`CUDNN_FE_GROUPED_GEMM_VALIDATE_DEVICE_VALUES`).

**R7 — you need a cuDNN handle and the caller gave none.** Graph API lowering:
`_pygraph._backend_handle_for_lowering` (process default, one per thread and
device, stream 0, destroyed at exit) — the graph never owns one. torch-op
layers: the per-device cached handle re-streamed to torch's current stream
before every call (`linear_attention/ops/common.py::get_handle`,
`ops/norm/_common.py`). Never `cudnn.create_handle()` inside a plan, engine or
C++ graph object.
A torch-facing block that drives a backend `pygraph` keeps ONE process-lifetime
handle per device (`gated_attention_block/kernels/proj_gemm.py::graph_handle`:
`with torch.cuda.device(idx): cudnn.create_handle()`, memoised by device index,
never destroyed), obtains it at `compile()` so `cudnnCreate` never runs inside a
captured execute, and re-streams it with `cudnn.set_stream(handle, stream)`
immediately before every `graph.execute(vp, ws, handle)` (`cudnnSetStream` is
host-only handle state, capture-legal; `_pygraph.execute` reads the stream off
the handle). Never one handle per (device, stream) created lazily at execute.
Re-streaming is single-threaded: concurrent threads on one device pass their own
`handle=`. Detector:
`test_block_end_to_end.py::test_graph_route_handle_is_per_device_and_restreamed`
(monkeypatch `cudnn.create_handle` to fail, execute on two streams, same handle,
stream follows).

**R8 — you own a CUDA resource whose release can be GC-timed (only tests and
the C++ PyGraph may).** Release inside `cuThreadExchangeStreamCaptureMode(RELAXED)`
and restore in `finally` (`frost/buffers.py DeviceBuffer.__del__`,
`pygraph.h CaptureModeGuard`). Production plans and engines reach this recipe
only if R2/R7 were skipped — fix that instead.

**R9 — proving it.** Around a warm `check_support()`/`compile()`/`build()` + `execute()`:
`torch.cuda.set_sync_debug_mode("error")` (no sync), and
`torch.cuda.memory_stats()["allocation.all.allocated"]` unchanged across three
executes (no allocation) — `test_sdpa_prepared_thd.py::test_execute_allocates_nothing_and_never_synchronizes`,
`test_sdpa_bwd_thd_sm80.py::test_graph_thd_execute_does_not_allocate`. For a
capture-safety claim, `test_cuda_capture_lifetime.py` (GC inside a global-mode
window, then replay and a native launch). For R1, monkeypatch
`torch.cuda.ExternalStream` to raise and drive the path with handle 0
(`test_torch_stream.py`). Every `fe_api` test already runs each `APIBase`
`execute()` under `set_sync_debug_mode("error")` (`test/python/fe_api/conftest.py`);
an R6 engine's tests carry `@pytest.mark.allow_host_sync` and say why. Rule 8
holds build to the execute standard, so a family's own tests also wrap
`__init__` + `check_support()` + `compile()` in `set_sync_debug_mode("error")`
(`test_NSA_topk_reduction.py`, `test_NSA_swa.py`, `test_NSA_compression_attention.py`).
The `compile_allocates_nothing` fixture in the fe_api conftest is the R11 detector.
Feed the sync detector tensors the plan has never
seen (a fresh `.clone()` of the offsets / pointer table per execute): an
id-keyed validation memo hid a per-tensor D2H from every warm test
(`fe_api/test_grouped_gemm_rule8.py::test_execute_never_synchronizes`).
Run the allocation detector over the CONVERTED-dtype graphs too (bf16 gate /
int64 cu / bf16 state), not just the native one — a staging allocation only
shows up there
(`test_kda_sm90_cuda.py::test_execute_allocates_nothing_and_never_synchronizes[...bf16_gate_int64_cu_bf16_state]`);
and over the output-dtype axis: an fp32 output that needs 0 bytes must run with
`workspace=None`, a bf16 one must raise `requires a \d+-byte workspace` before
any in-place stage mutates anything (`test_DSA_indexer_backward.py`).
A multi-stage block runs the allocation assertion around BUILD as well
(`compile_allocates_nothing` covers every sub-engine's compile) and proves the
"first execute of a shape inside `torch.cuda.graph`, replay bit-identical" claim
directly (`test_block_fp8.py::test_execute_allocates_nothing_and_never_synchronizes`,
`::test_first_execute_inside_capture`, `::test_quant_slot_is_filled_on_launch_stream_and_matches_spec`).
For a capture-time allocation claim read `allocation.all.allocated` INSIDE the
`torch.cuda.graph` window: on torch 2.13 `capture_begin` itself adds two
allocations, so a before/after around the `with` is red on innocent code
(`_capture_allocating_nothing` in `test_hstu_attention.py` /
`test_hstu_block_sparse.py`). HSTU detectors:
`test_hstu_attention.py::test_fwd_bwd_execute_allocates_nothing_and_never_synchronizes`,
`::test_workspace_contract`, `::test_check_support_declines_unaligned_layouts`,
`test_hstu_block_sparse.py::test_builder_workspace_bytes_match_carve` (the
carver's final offset equals the bytes function for every geometry).
For a grouped / pointer-table API: compile eagerly, run the FIRST execute of
the plan over an offsets tensor and pointer table it has never seen inside
`torch.cuda.graph` with a torch workspace, read `allocation.all.allocated`
inside the window (must be 0), replay twice and compare bit-exact with an eager
execute (atomically accumulated outputs such as dGLU `dprob` get a tolerance);
feed the allocation/sync detector fresh tables per execute; check the workspace
contract with None / undersized / a 64-byte-offset slice; a metadata-only
`TensorDesc` for the sample offsets proves the build reads no values
(`fe_api/test_grouped_gemm_rule8.py`, `fe_api/grouped_gemm/test_grouped_gemm_rule8_fusions.py`).
A wrapper cache-smoke test that stubs `check_support` / `compile` / `execute`
must stub `scratch_workspace_bytes` too, since the wrapper now sizes from it.

**R10 — a launch envelope the caller already knows (max sequence length, packed
total, batch count, top-k width).** It is a required host int at plan time (an
`__init__` argument or a graph attribute), validated against the sample shapes
that already encode it (`q.shape[0] >= cum[-1]`, `sample_out.shape[1] ==
max_seqlen_k`); never `cu_seqlens.diff().max().item()` or `.cpu().tolist()` at
build or execute (`TopKReduction`, `CompressionAttention` take the packed total
from `sample_q.shape[0]`; `Dense*ScoreRecompute` takes `max_seqlen_q/k`). When
absent, raise `ValueError` naming the argument; when the kernel cannot serve a
layout without reading it back (a host-driven per-batch loop), decline in
`check_support()` (R5) until a native kernel exists. A WRAPPER may keep a
documented `.item()` fallback on its own surface only, must key its memo on the
resulting value, and must say in its docstring that omitting the hint costs one
blocking read per call and is not capturable (`csa_compressor_forward_wrapper(total_comp=None)`).

**R11 — compile() needs an operand the kernel only sees at execute (scratch,
semaphore, scheduler counter, an output the caller passes later).** Never
`torch.empty`/`torch.zeros` a stand-in and never `from_dlpack` a plan-owned
tensor at build. Build the ABI from metadata: tensor slot ->
`cute.runtime.make_fake_tensor(dtype, sym-shape, stride=..., assumed_align=...)`
or `make_fake_compact_tensor(dtype, sym-shape, stride_order=..., assumed_align=...)`
(APIBase: `_make_fake_cute_tensor_from_desc(desc)` when a TensorDesc exists —
`causal_conv1d_update_sm100/api.py`, `hstu/hstu_lmsd/api.py` workspace descs;
module-level: `flex_attention/dispatch.py::_make_fake_bwd_aux_tensors`,
`_make_fake_fp32_scratch`, `_make_fake_semaphore`, `_make_fake_scheduler_counter`);
pointer slot -> `cute.runtime.make_ptr(dtype, 0, cute.AddressSpace.gmem, assumed_align)`
(`_FakeTensor.iterator` is rejected by the DSL). The fake must reproduce the
layout the launch-time conversion yields — dynamic extents, static leading
stride 1, the compact path's divisibility, and a static stride 0 on a size-1
mode where `from_dlpack` canonicalizes it (flex semaphores with stage 1) — prove
it with a CPU `from_dlpack` parity test next to the builder
(`test_flex_attention_contracts.py::test_compile_fakes_match_dlpack_layouts`).
Wrap `cute.compile` in `with torch.cuda.device(desc.device)`: fakes carry no
device. Prove the build with R9 around `check_support()` + `compile()`
(`test_flex_attention.py::test_explicit_api_compile_allocates_nothing_and_never_synchronizes`).
A `compile()` that only "primes" a lazily-compiling launcher (the real
`cute.compile` runs at the first `execute()` against live tensors) is a Rule 4
miss as well as an R11 blind spot: split the launcher into a plan-time
`compile_<op>_kernel(dtype, shape envelope, flags)` (module cache, fake
operands, under `torch.cuda.device(desc.device)`) and an execute-time
`launch_<op>_kernel(compiled, <caller tensors>, <carved scratch>)`, so
`compile_allocates_nothing` has something to measure and `execute()` is a
guaranteed cache hit (`indexer_top_k/indexer_top_k_decode_varlen.py`). Still
lazy, tracked: `IndexerBackward`, `DenseIndexerBackward`, `IndexerForward`.
`from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=ndim-1)` keeps
size-1 modes dynamic (`(?,?,?):(?{i64},?{i64},1)` for `(B,1,n)` and `(1,n,L)`),
so `make_fake_tensor(dtype, (sym_int(),)*rank, (sym_int64(),)*(rank-1)+(1,),
assumed_align=16)` reproduces it exactly
(`block_sparse_builder._fake_dynamic_tensor`); a divisibility the launch-time
view guarantees goes on the fake too (`sym_int64(divisibility=8)` for the HSTU
fp32 accumulator).


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
├── frost/                           # the FROST GEMM engine (JIT fused matmul chains from cuDNN graphs)
│   ├── sm100/, sm120/               #   one tree per arch family: compiler.py + epilogue_codegen.py + kernel_templates/
│   ├── compiler.py, epilogue_codegen.py  # facades: become the active family's module (arch_family.py)
│   └── kernel_templates/            #   template code SHARED by both trees (split-K reduction)
├── ops/                             # backend-independent torch custom-op contracts
└── reference/                       # pure-PyTorch MATMUL/POINTWISE correctness engine
```

Shared helpers (schedulers, metadata utils, e.g. `gemm/cutedsl/grouped/moe_*.py`) stay internal to the family package — never exported through `cudnn`.

## CuTeDSL kernel bodies

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

Related: inside a kernel body, `for x in (a, b)` over a Python tuple is
rewritten into a dynamic `ir_loop` and cannot iterate heterogeneous objects
(e.g. `GridConstant[TensorMap]`). Unroll it, or use `cutlass.range_constexpr`.

**Detector.** These break at `compile()`, not at import — `python -c "import ..."`
and `pytest --collect-only` both stay green. After any refactor of a kernel
body, run that flavor's own tests.

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
6. DSL version gate (Rule 7): the route/`check_support` declines with a version-naming error below `CUTEDSL_MIN_VERSION`, and the tests skip there instead of failing.

The `cutedsl-kernel-integration` skill (`skills/cutedsl-kernel-integration/`) documents this workflow in detail, including how to classify a kernel into a family — follow it for any kernel integration.

## Other notes

- `wrapper.py` `Graph` context manager (the pythonic graph builder) requires cuDNN backend ≥ 9.12 (`backend_version() >= 91200`) and builds plans on `__exit__`.
- Torch custom-op implementations live with their owning operation family and may be re-exported from `experimental/ops/` while maturing (pattern doc: `docs/utilities/adding_torch_custom_ops.md`); they cache built graphs per config and use stable `_UIDs` enums.
- dtype conversions go through `datatypes.py`, which probes torch/cutlass availability lazily — keep it that way.
- Formatting: black, line length 160.

CUDA-owning objects, GC-timed release and stream capture: Rule 8.
