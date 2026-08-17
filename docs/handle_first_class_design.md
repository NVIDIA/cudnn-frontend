# First-class `cudnn.Handle` — design

The backend `cudnnHandle_t` carries a device binding (compute capability, SM
count, ...) and the current stream. On the FE side the handle is a **bare int**
(`create_handle()` returns `reinterpret_cast<intptr_t>(handle)`), so it has no
place to hang per-handle state. That state has instead accreted as side tables
and per-engine queries:

- stream: the module-global `_handle_to_stream` dict (PR #611);
- device: **three** parallel stacks — the backend handle; pygraph's separate
  `sm_count`/`sm_version`/`device_property` args (deviceless AOT); and frost's
  own `current_device()` + driver introspection (`frost/device.py`), which every
  python engine re-queries because nothing hands it the device.

This makes `create_handle()` return a first-class `Handle` object that owns
`{backend_handle, device, stream}`, collapsing those into one concept. The
naming anticipates the front end BEING "cudnn" and today's cuDNN becoming "cudnn
backend": this object is the `handle`; the wrapped `cudnnHandle_t` is the
`backend_handle`. It is **optional** by design — python engines (frost, cutedsl,
linear-attention) need device+stream, not a `cudnnHandle_t`, so the backend
becomes one consumer rather than the anchor.

## Hard constraints (from a full call-site inventory)

1. **The C++ boundary needs no change; the handoff is EXPLICIT in Python.** Every
   handle-consuming binding takes `std::intptr_t` / `std::optional`. The backend
   handle is extracted **explicitly** in our Python code — `to_backend_handle(h)`
   at each handoff (`_execute*`, `backend_graph`, and the workspace / cuda-graph
   methods, whose signatures name `handle` rather than a `*args` passthrough).
   `deserialize` is the one genuinely ambiguous classic overload (`(data)` vs
   `(handle, data, ...)`), so it stays a passthrough and unwraps just its first
   positional. A reader can grep `backend_handle` and trace the plumbing
   top-to-bottom without an IDE. A full inventory confirmed **every** handle→C++
   handoff is in `_pygraph`/`__init__` (the `__getattr__` delegation carries no
   handle), so the set is closed. `Handle` deliberately has **no `__index__`**:
   the only path to the backend is those explicit calls, and a Handle that
   reaches a binding unconverted fails loudly instead of being silently coerced.
   C++ never reads device off the handle. **The C++ handle ABI is unchanged**;
   the only `.cpp` change is renaming the `set_stream`/`destroy_handle` bindings
   to `_raw_*` (`python/properties.cpp`) so the Python wrappers own those names.
2. **The `create_handle` wrapper must be defined AFTER the `__init__.py` symbol-
   copy loop**, or the raw pybind symbol shadows it.
3. **Dunder minimalism.** `Handle` defines **no** int-coercing dunders and leaves
   `__eq__`/`__hash__`/`__bool__` at the object defaults (identity eq, identity
   hash, always-truthy). This satisfies all three Python pressure points at once:
   - `_handle_to_stream` uses the handle as a dict key -> needs it hashable
     (identity hash is fine);
   - `wrapper.py` compares the stored handle against the string sentinel
     `'auto'` and `None` -> a value-based `__eq__` that calls `int(other)` would
     raise on a str; identity `__eq__` returns `False` cleanly;
   - `if handle:` / `handle or 0` -> a live Handle must stay truthy.
   Giving Handle a value `__eq__` without `__hash__` would make it unhashable
   (TypeError on the stream dict) — so we give it neither.
4. **The Python handle APIs take a `cudnn.Handle` only; a raw backend int is
   rejected.** `cudnn.create_handle()` is the only way to make a handle in the
   Python API, so every real caller already holds a Handle (verified across
   flashinfer / sglang / the FE's own code; torch uses the C++ frontend, not this
   module). A bare int would silently opt out of the Handle's device/stream
   tracking and device-scoped build, so `to_backend_handle` / `set_stream` /
   `get_stream` / `destroy_handle` / `execute(handle=)` raise on a non-Handle. A
   framework holding a foreign `cudnnHandle_t` wraps it once —
   `cudnn.Handle(backend_handle, ordinal, stream)` — so it becomes first-class
   (gaining the same device/stream/scoping) rather than a second-class bare int.
5. **Deviceless AOT must not eager-query the driver.** `test_deviceless_aot_
   compilation.py` builds with `device_property=` and no live handle, targeting
   an SM that differs from any local GPU. `Handle.device` populated from a
   deserialized `DeviceProperties` JSON (`deviceVer`, `multiProcessorCount`) must
   stay authoritative when set; the live-driver path is used only when no
   override/descriptor is present.
6. **Device caches key by ordinal, not by Handle.** The existing per-device
   `lru_cache`s (frost.device.\*, occupancy map, tile budgets) assume a stable
   ordinal; a process driving two GPUs must not cross-contaminate.

## `Handle`

```
class Handle:
    backend_handle: int | None  # the wrapped cudnnHandle_t; None = pure-python (future)
    _ordinal: int | None        # CUDA device ordinal this handle is bound to
    stream: int | None          # authoritative; absorbs _handle_to_stream
    device -> DeviceInfo        # lazy, cached by ordinal
    # no __index__/__int__; __eq__/__hash__/__bool__ at object defaults (identity, truthy)

# the explicit handoff (grep `backend_handle` to trace it):
to_backend_handle(h)          # a Handle's .backend_handle, a foreign int, or None
```

`create_handle()` (Python, after the copy loop): `Handle(backend_handle=
_pybind_module.create_handle(), ordinal=<current device>)`.

### `DeviceInfo` (the union `Handle.device` exposes)

Lives in `cudnn/_device.py` — the FE's **single owner** of a GPU's facts. Each
fact is a `@cached_property` that queries the driver once and caches **on the
instance**, and there is one instance per ordinal (`device_info(ordinal)`,
lru-cached), so a GPU's facts are asked for once and shared. `Handle.device` is
that object.

This inverts the previous direction: the driver queries used to live in
`frost/device.py` and `DeviceInfo` delegated down to them. Now the common layer
owns the queries, and `frost/device.py`'s fact functions (`compute_capability`,
`multiprocessor_count`, ...) are **thin shims** onto `device_info(ordinal)` — so
frost consumes the same object rather than running a parallel introspection
stack, and its ~24-file / 65-site call surface is unchanged. (A later step can
repoint those sites at `handle.device.*` directly where a handle is in scope; the
ownership move here is the enabling half.) Fields (superset the inventory proved
is consumed):

| field | form | consumers |
|---|---|---|
| `ordinal` | int | frost build/guard, operand views |
| `compute_capability` | (major, minor) | frost arch gate, api_base sm107, tensor_adapter |
| `sm_version` | packed `major*10+minor` | Context, kernel_registry ranges, DSA gates |
| `sm_count` | int (user-overridable) | tile scorer, heuristics `SM_COUNT_TARGET` |
| `shared_memory_per_block_optin` | bytes | tile SMEM budget |
| `oversized_shared_memory_per_block` | bytes (0 if unsupported — load-bearing) | tile SMEM budget |
| `l2_cache_bytes` | bytes | L2 swizzle budget |
| `device_name` | str | diagnostics |

`sm_version` is a derived property of `compute_capability` so the two forms
cannot drift. `Handle.device` also owns the serializable backend
`DeviceProperties` (deviceless AOT); when built from a descriptor/override the
fields come from its JSON, not the driver.

### Environment facts (versions) — `cudnn/_env.py`

Device facts are per-ordinal; **version** facts (CUDA driver, CUDA runtime) are
process-global — one per process regardless of which GPU a handle is bound to.
Putting them on `DeviceInfo` would duplicate them per ordinal, and on `Handle`
per handle, so they get their own owner `cudnn/_env.py` (`driver_version()`,
`runtime_version()`). This mirrors the backend, which exposes its own versions as
argument-less globals (`cudnnGetVersion`, `cudnnGetCudartVersion`), never off a
handle or the `DEVICEPROP` descriptor — cuDNN's own version stays there,
`cudnn.backend_version()`. The CUDA queries had accreted as re-reads in each
engine (the `DeviceInfo` oversized-SMEM gate, the cutile GDN/KDA `check_support`);
`_env` collects them. ~100 ns and off the execute hot path — the cache is a single
owner returning a constant, not a speed play.

## Stream model

`Handle.stream` is the single source of truth. Today the stream lives in two
disconnected places: `set_stream` **writes** `_handle_to_stream` (PR #611) but
`_resolve_stream` **reads** live via `cudnn.get_stream` -> `cudnnGetStream` on
every python-engine execute. Handle unifies them:

- `set_stream(h, s)`: Handle -> compare/write `h.stream`, call `_raw_set_stream`
  only on change (and only if `h.backend_handle is not None`); foreign int ->
  today's `_handle_to_stream` path.
- `get_stream(h)`: Handle -> return `h.stream` (no round-trip); foreign int ->
  raw binding.
- `_resolve_stream` / `_build_context`: read `h.stream` for a Handle. This kills
  the per-execute `cudnnGetStream` that #611 did not remove.
- Preserve the **raise-not-fallback** contract: a failed query on a *supplied*
  handle raises (asserted by `test_dispatch.py:585`), never silently falls back
  to the torch current stream.

## Device-consumer migration (frost / linear-attention / sdpa)

`to_backend_handle` does **not** help here — these want a device **ordinal**, not
the backend handle int. Every device-derived frost build constant (`_current_arch`,
`_plan_device`, `_grid_num_clusters`, `_sm_count`, the SMEM/L2 budgets) already
funnels through `frost.device.current_device()`/`resolve_device(None)`, so rather
than thread an ordinal through ~20 signatures, a **scoped build-device override**
covers them all at once:

- `frost/device.py` — `build_device(ordinal)`, a context manager that scopes an
  override into `current_device()` (like `torch.cuda.device()`). `None` = no
  override (classic current-device). Grep `build_device` / `_build_device` to
  trace it: the context manager + `current_device()`'s read + the one hinge.
- **Frost GEMM (done):** `FrostGemmEngine.build_plan(graph, plan, ctx)` wraps
  `build_gemm_plan(graph)` in `with build_device(ctx.handle.device.ordinal)`, so
  the whole build bakes for the handle's GPU. `tile_config._sm_count()` is
  re-routed off `torch.cuda.current_device` onto `frost.device` so it honours the
  scope too (it was the one query that bypassed `current_device()`).
- **Frost GEMM compile target (done):** the scope also had to reach the *cute
  compile arch*, which the earlier constants did not. cutedsl derives the compile
  target from the ambient CUDA device (`torch.cuda.get_device_capability`), so a
  build for handle-GPU-A while GPU-B is current baked A's constants into a
  B-targeted kernel. `_frost_compile_options()` (in `gemm/frost/compiler.py`) now
  pins `--gpu-arch sm_<scope>` into the `cute.compile()` options string, so the
  compile target follows the scope. The arch is part of the baked, content-hashed
  source, so a cross-arch kernel can no longer collide in the JIT cache with a
  same-source same-machine one. **Limitation:** the pin is honoured on the public
  `nvidia-cutlass-dsl >= 4.7` (frost's `CUTEDSL_MIN_VERSION`) and on internal RCs;
  only a public wheel below the floor never threads `--gpu-arch`, and frost already
  declines those as too-old (`buffers.cutedsl_too_old`, which the support check
  reuses so an internal RC's own `0.x` numbering is judged new, not old). On such a
  wheel a handle-scoped build **fails loud** — it cannot pin the target and cutedsl
  resolves it from an arch captured at *import* time, which we can neither set nor
  reliably read (a live-device comparison would miss an import-on-B / build-on-A
  process), so `_frost_compile_options` refuses any `build_device`-scoped build
  rather than bake scope constants into a possibly-mis-targeted kernel. An unscoped
  build makes no cross-device promise and is unchanged.
- **Not yet scope-following (documented holes):** `check_support`/kernel-selection
  gates read the ambient arch (`buffers.current_sm()`), and the linear-attention
  kernels lazy-compile at first *execute* — after the build scope has closed — so
  their compile target is the execute-time device. Same-GPU (scope == ambient, the
  normal case) all of these agree; a handle-scoped build on a sub-floor wheel is
  the only case that diverges, and it is the fail-loud path above.
- `_check_plan_device` **stays unchanged** and correct: it is the EXECUTE-time
  launch guard and must check the *live* current device (where the launch is
  going) against the baked device — the override is a build-scope only, unset at
  execute, so the guard keeps reading the live device.
- `current_device()` is otherwise unchanged — it is the fallback whenever no
  build scope is active (no-handle, render-only, execute).
- **Linear-attention (done):** `gdn/gdn2/kda_engine.build_plan` wrap their build
  in `with build_device(ctx.handle.device.ordinal)` too, and their one device-baked
  constant — `num_sm = multiprocessor_count(current_device_id())` — is re-routed
  onto `frost.device.current_device()` so it follows the scope (it read the
  `buffers` probe, which bypassed it). `test_la.py` 359 passed / 0 failed on SM100.
- **sdpa frost engines:** nothing to scope at the engine level — their build bakes
  no device constant from `current_device`; arch gating is in `check_support`, and
  the one `torch.cuda.current_device()` (`sdpa/bwd/engines.py`) tags a TensorDesc's
  operand device, which is correctly the live device (as `VariantPack.device`).

`workspace` stays a per-execute argument — it is not handle state and does not
belong on the Handle (today it is conflated into `ExecutionContext`).

`VariantPack.device` (operand DLPack views) stays on `current_device()`: it is
read at EXECUTE, where the operands live on the current device, so the live device
is correct (and the build scope is not active then).

## Validation

- Single-GPU L0 (matmul/conv/rope/norm) — regression. Handle core: 41 passed.
- Frost GEMM no-regression with the build-device adoption: `test_public_execute_
  flavors` + `test_stream_respect` 32 passed (SM100).
- **Cross-device redirect** (`test/python/gemm/frost/test_build_device.py`): scope
  a build to a *different-arch* real GPU (L40S sm89 / H100 sm90 / A100 sm80) and
  assert every frost build constant (`_current_arch`, `_plan_device`, the
  re-routed `_sm_count`) reports THAT device — the multi-GPU behaviour a
  single-GPU run cannot otherwise exercise, proven on parley's SM80..SM100 range
  without needing two Blackwells (device queries only, no kernel launch).
  `_check_plan_device` remains the execute-time launch guard against baking on one
  GPU and launching on another.
- **Compile-target follows scope** (SM100, cutedsl 4.7): the `--gpu-arch` pin is
  non-regressive on the matching-arch path — `test_matmul.py` bf16 sweep 677
  passed / 337 skipped end-to-end with `--gpu-arch sm_100a` baked in. That the pin
  actually moves the target is shown by compiling one graph three ways: `sm_100a`
  (machine) and `sm_103a` (a different Blackwell sibling) both compile, while
  `sm_90a` fails in the arch-specific NVVM backend — impossible if the option were
  ignored (all three would target sm_100 and pass), so on 4.7 the option reaches
  the compiler. The sub-floor-wheel fail-loud is unit-checked by forcing the
  support probe false and asserting a `build_device`-scoped build raises while an
  unscoped one passes through.
- **Forced through flashinfer's GEMM fuzzer** (SM100): this build dropped into
  flashinfer's `.venv` (via a `PYTHONPATH` shim) and forced onto the `cudnn`
  backend across the full unified GEMM fuzz cross-product (bf16 / fp8 / nvfp4 /
  mxfp4 / mxfp8, mm + bmm) — 731 passed / 0 failed / 151 xfailed (the xfails are
  flashinfer's pre-tracked, backend-agnostic findings). The first-class `Handle`,
  `set_stream` idempotency and `destroy_handle` clear are exercised on every one.

## Relationship to PR #611

This PR is the full first-class-Handle superset of #611 (it reimplements the
set_stream idempotency as `Handle.stream` and keeps the discarded-context fix).
#611 stays open as the minimal, low-risk fallback. Exactly one merges.
