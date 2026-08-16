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

`note to self: claude::11323ca1-07bc-4fc4-8ec7-ba95d8f061d8 — first-class Handle. cwd /home/scratch.yanxu_libs/cudnn_frontend`

## Hard constraints (from a full call-site inventory)

1. **The C++ boundary needs no change; the handoff is EXPLICIT in Python.** Every
   handle-consuming binding takes `std::intptr_t` / `std::optional`. The backend
   handle is extracted **explicitly** in our Python code — `to_backend_handle(h)`
   at the named handoffs (`_execute*`, `backend_graph`) and `unwrap_handles(args,
   kwargs)` at the opaque passthroughs (`get_workspace_size`, cuda-graph,
   `deserialize`) — so a reader can grep `backend_handle` and trace the plumbing
   top-to-bottom without an IDE. A full inventory confirmed **every** handle→C++
   handoff is in `_pygraph`/`__init__` (the `__getattr__` delegation carries no
   handle), so the set is closed. `Handle` deliberately has **no `__index__`**:
   the only path to the backend is those explicit calls, and a Handle that
   reaches a binding unconverted fails loudly instead of being silently coerced.
   C++ never reads device off the handle. **No `.cpp` changes.**
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
4. **Foreign raw-int handles keep working.** A framework may create a
   `cudnnHandle_t` via the C API and pass the bare int. Those have no Handle
   object, so stream/device fall back to the `_handle_to_stream` registry (keyed
   by `int(handle)`) + a live `cudnnGetStream`. All wrappers branch
   `isinstance(handle, Handle)`.
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

# the explicit handoffs (grep `backend_handle` to trace them):
to_backend_handle(h)          # a Handle's .backend_handle, a foreign int, or None
unwrap_handles(args, kwargs)  # same, for any Handle in a passthrough call
```

`create_handle()` (Python, after the copy loop): `Handle(backend_handle=
_pybind_module.create_handle(), ordinal=<current device>)`.

### `DeviceInfo` (the union `Handle.device` exposes)

Sourced from `frost/device.py`'s driver introspector (the richest, framework-
neutral, already `lru_cache`d per ordinal). Reuse those functions; do not add a
fourth query stack. Fields (superset the inventory proved is consumed):

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
- `_check_plan_device` **stays unchanged** and correct: it is the EXECUTE-time
  launch guard and must check the *live* current device (where the launch is
  going) against the baked device — the override is a build-scope only, unset at
  execute, so the guard keeps reading the live device.
- `current_device()` is otherwise unchanged — it is the fallback whenever no
  build scope is active (no-handle, render-only, execute).
- **Pending:** the same hinge wrap for `linear_attention/frost/*_engine.py`
  (gdn/gdn2/kda) and the sdpa frost engines; those also read `buffers.current_sm()`
  / `current_device_id()` at build, which must be routed through the scope too.

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

## Relationship to PR #611

This PR is the full first-class-Handle superset of #611 (it reimplements the
set_stream idempotency as `Handle.stream` and keeps the discarded-context fix).
#611 stays open as the minimal, low-risk fallback. Exactly one merges.
