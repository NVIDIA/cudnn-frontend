# frost_gemm's execute path: one gate table, evaluated twice

What the per-execute path should look like if it is designed from what actually
varies per call, and what that deletes from what is there now.

Nothing here is implemented. This is the brief for doing it.

## The question this answers

`check_support` and the per-execute gate are the same predicate evaluated at two
times — once against the shapes the graph declared, once against the shapes the
call is running. Today they are two bodies of code computing the same facts, so
they can drift, and the execute side rebuilds facts that were settled when the
kernel compiled.

Dynamic shape is why the checks must exist at execute at all. It is NOT why they
are written twice. `_tma_alignment_reject` is the proof: it is the one predicate
factored as a pure function of `(dtypes, majors, M, N, K)`, and it is called from
both sites — `compiler.py:2427` with the graph's `mm.M/N/K` and `compiler.py:1861`
with the runtime ones. Its docstring says so: *"One rule ... for both the
graph-time and runtime dims."* The other three gates were never factored that
way, so each fact lives twice.

## What varies per call

**Baked when the kernel compiles — a runtime value cannot change it**

- `a_major` / `b_major`: which dim is contiguous, compiled into the TMA
  descriptor and the MMA operand descriptor
- every operand's dtype, and the fp4 two-elements-per-byte packing factor
- the epilogue store / aux load vector width, hence each output's and aux's
  required alignment (`_output_align_reqs`, `_aux_align_reqs`)
- the modulus in the TMA 16-byte rule (`128 // bits`)
- rank 3, the operand roles and their count, which are SFA/SFB, the block-scale
  block size, `is_multi_gemm`, which outputs are `norm2` reductions

**Free per call**

- M, N, K, batch
- the pointers
- **the outer strides.** `_contiguous_dim` only asks which dim has stride 1;
  nothing constrains the others, and the max-allocation override case depends on
  that freedom (a `[m, k]` corner of an `[mb, kb]` buffer has row stride `kb`).

**Therefore re-checked per call — the intersection**

| gate | why it re-runs | the baked half |
|---|---|---|
| TMA 16-byte alignment | the contiguous extent IS M/N/K | the modulus, and which of M/N/K is contiguous |
| operand shape agreement | every operand's extents move with M/N/K | A is `(b, M, K/kpack)`, B is `(b, N, K/kpack)` |
| pointer alignment | new call, new buffers | required bytes per role |
| output `tensor_alignment` | it is `min(ptr, stride, shape)`, and stride/shape move | the required vector width |
| layout / major | the buffer is new; its contiguous dim could differ | the wanted major |

**Recomputed per call although nothing in it can change:** `_output_align_reqs`,
`_aux_align_reqs`, `k_factor`, the `_MAJOR_CONTIGUOUS_DIM[major]` lookup, and
`_finalize_reductions`' `startswith("reduction_")` / `rsplit` walk over
`chain.outputs`.

## Found while classifying: the TMA gate checks the wrong quantity

`_tma_alignment_reject` checks `extent * bits % 128 == 0`, where `extent` is K
(k-major) or M/N. Its own docstring says TMA encodes *"the contiguous input
dimension's stride in 16-byte units"* — stride, not extent. The two are the same
number only when rows are dense.

Since outer strides are free (above), a caller can hand in a `[m, k]` corner of
an `[mb, kb]` allocation whose row stride `kb` is not 16-byte aligned while `k`
is. `k=64, kb=72` at bf16: extent `64*16=1024` passes, row stride
`72*2=144` bytes is not a multiple of 16. The gate accepts it and TMA
mis-strides every row past the first — silently wrong numbers, the failure mode
the gate exists to prevent.

The existing override test does not discriminate: it uses `kb=128`, where both
quantities are aligned.

**Not yet reproduced on hardware.** Confirm before fixing: build the override
case with a deliberately unaligned outer stride and compare against the backend.
If confirmed, the gate should take the row stride, which it has (the buffer's
`stride()`), rather than inferring it from the extent.

## The shape to build

One table, built where the analyzer already knows these things, evaluated by
both sites:

```python
# built once, when the plan compiles
gate = GateTable(
    operands=[OperandGate(role, axis_of_m_or_n, axis_of_k, major, kpack,
                          ptr_align, mode, vector_bytes), ...],
    tma_modulus=[(role, 128 // bits, which_extent), ...],
)

# check_support, at the declared shapes
reason = gate.reject(declared_mnk, declared_strides=..., pointers=None)

# execute, at the runtime shapes
reason = gate.reject(runtime_mnk, slots)
```

`reject` is one pass over `operands`. There is no second formulation to drift,
and the baked half is computed once.

The per-execute path then reads:

```python
slots = pack.views(self._indices)          # one crossing, already there
M, N, K = self._extents(slots)             # build-known axes, 3 index reads
if (reason := self._gate.reject((M, N, K), slots)) is not None:
    raise ValueError(reason)
self._launch(slots, (M, N, K), stream)
```

`self._extents` reading build-recorded axes also closes the class of bug the
override work hit: `shape[1]` / `shape[2]` hard-codes the caller's axis
convention, which is not the graph's.

## What that deletes

1. `resolve_variant_pack` and `run_resolved`'s `{id(t): buf}` indirection — the
   engine knows the operand order at build, so one `views()` result sliced by
   build-time ranges replaces `pull()` and its two dict lookups per operand.
2. The five intermediate lists (`_operands`, the layout comprehension, `_named`,
   the SF comprehension, `pairs`) and the four walks over them → one pass.
3. `_output_align_reqs` / `_aux_align_reqs` per execute.
4. `k_factor` and the fp4 branches, evaluated in three places per call.
5. `_finalize_reductions`' string parsing → a build-time index list.
6. The `shape[1]` / `shape[2]` convention assumption.

## Is pure python good enough?

For gemm, plausibly yes, and this is the case worth trying it on: **gemm is one
launch**, not GDN's eight.

Today 43.1 us. The floor is 1 launch (1.85) + one DSL crossing (~2) ≈ 4.
Above that: `graph.execute` entry ~6, `_normalize` 2.0, views ~1, and ~29 of
engine + compiler python. The table design targets that 29; the rest of the
python is already thin.

If the gate collapses to one pass and the entry shrinks, mid-teens is the
plausible landing zone — the same order as flashinfer's 14.5, which also has no
graph API to pay for. **This is an estimate from subtraction, not a measurement.**
Confirm first: time `run_resolved` minus `_call_positional` to size the gate
machinery alone. If the gate is not most of the 29, this design is aimed at the
wrong thing and the brief should be rewritten before any code moves.

## Traps

- **Measure from a drained queue and sweep the burst size.** GDN's device time is
  ~52 us against a ~50 us host, so back-to-back timing reads back the device
  rate. `proto_exchange_api/fe_floor.py` reported 53 us for a 20 us stage this
  way; `fe_floor2.py` is the corrected form.
- **Do not read launch cost out of an nsys trace.** CUPTI adds ~2.2 us per traced
  API call: `cudaLaunchKernelEx` is 1.85 us untraced and 4.06 traced.
- **Never overwrite the venv's `.so` while a test run is in flight.** It
  segfaults workers, twice observed, and looks exactly like a concurrency bug in
  the code under test.
- **Do not trust grep for what a kernel layer calls on a buffer.** `reshape`,
  `permute` and `stride(dim)` were each found only by running the whole suite.
- **`cd` to the worktree explicitly.** The shell's cwd resets between commands,
  and building from the wrong one produces a `.so` that imports but is missing
  symbols.

## Where to do it

On this branch. The design leans on the variant pack, `VariantPackSlot` and
`pack.views()`, which are all here; starting elsewhere would mean rebuilding
them or re-deriving the numbers above. Keep it as its own commit so it can be
dropped without touching the migration.

A fresh session is fine to execute it — that is what this file is for. Read it
and `HANDOFF_variant_pack.md` first; between them nothing above needs
re-deriving.

```
note to self: claude::774e8e99-23ad-4a94-be0d-53ed5ee4def9 — "cuDNN FE variant-pack normalization"
cwd /home/scratch.yanxu_libs/cudnn_frontend · workspace /home/scratch.yanxu_gpu/fe_pr1
```
