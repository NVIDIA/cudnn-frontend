# A FROST gemm launch, down to the driver

`graph.execute()` spends about 20 us of host time on a bf16 matmul whose kernel takes
three addresses and six integers. This documents where that goes, what a caller has to
promise to get rid of it, and what codegen now emits so that anyone can.

Run it:

```bash
CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1 \
  python benchmark/gemm/frost/benchmark_bare_launch.py
```

Requires SM100. Every result is checked bit-identical against `graph.execute()`, and the
script exits non-zero at the first mismatch — so any timing it prints was preceded by a
passing check, and a fast number that is also wrong cannot reach you.

## The measurement

Host time per call, minimum over 25 bursts of 64, `M=N=256 K=128` on one SM100 part.
The kernel is asynchronous, so these are CPU times — which is exactly what a
launch-bound serving loop pays.

| | us |
|---|--:|
| `graph.execute(dict)` today | 19.5 |
| ... with only the engine's launch closure replaced | 10.7 |
| `plan.execute(ptrs)`, this demo | **2.3** |
| `cuLaunchKernelEx` alone | 2.05 |

The last two lines are the claim: once the caller hands over an array instead of a dict,
essentially all remaining host time is the driver.

The middle line is worth keeping separately. It is the same public entry point with the
same per-call bookkeeping above the engine, and only the launcher underneath swapped —
so it separates what the tables buy (**8.8 us**) from what the entry point's dict-to-pack
conversion costs (**8.4 us**). They are independent; either can be done first.

## The contract

Between `build_plans()` and any launch, for every bound tensor:

* dtype, rank, extents and strides are what the graph declared,
* the innermost (unit-stride) axis stays innermost,
* the base address keeps the alignment the kernel was compiled for,
* the buffer lives on the plan's device.

**Only the addresses may change.** None of this is verified — verifying it is most of the
20 us. Violate it and the result is silent corruption or an illegal access.

This is a deliberate trade, not an oversight, and it is one an inference server keeps
without effort: fixed weights, fixed head dims, a pool of same-shaped activations. The
`--vary-m` flag shows the next rung up, where the token count moves too and the caller
re-supplies `problem_size`.

## What codegen emits

Two module-level tables, appended to the generated kernel. Both are read off the
signature that was just rendered rather than rebuilt alongside it, so they cannot drift
from the kernel they describe.

### `SLOT_TABLE` — what is in each kernel parameter

Verbatim, for the demo's default shape:

```python
SLOT_TABLE = (
    ('m',              'scalar',   8, ('problem_size', 0)),
    ('n',              'scalar',   8, ('problem_size', 1)),
    ('k',              'scalar',   8, ('problem_size', 2)),
    ('tma_a_desc_0',   'tma',    128, ('a', 0)),
    ('tma_b_desc_0',   'tma',    128, ('b', 0)),
    ('out_stride_m_0', 'scalar',   8, ('problem_size', 10)),
    ('out_stride_n_0', 'scalar',   8, ('problem_size', 11)),
    ('out_stride_l_0', 'scalar',   8, ('problem_size', 12)),
    ('tma_c_desc_0',   'tma',    128, ('c', 0)),
)
```

Enough to build the parameter block from scratch — the demo does, and diffs it against
what a real launch passed, byte for byte.

A parameter codegen cannot classify is emitted as `kind == 'unknown'` with a null source,
and a consumer must refuse the kernel. **A refusal is the correct outcome; a guess is
not.** This is what makes the mechanism safe to extend one flavor at a time.

### `PATCH_GROUPS` — what a changed quantity reaches

The transpose, plus two things a per-slot view cannot carry: the descriptors a quantity
was built into, and the grid axis it sizes.

```python
PATCH_GROUPS = {
    'm':        (((0, 0, 8),),  (3, 8),  'x'),   # writes, maps to re-encode, grid axis
    'n':        (((1, 0, 8),),  (4, 8),  'y'),
    'k':        (((2, 0, 8),),  (3, 4),  None),
    'addr_a_0': (((3, 0, 8),),  (),      None),
}
```

An address group names no descriptor: a CUtensorMap keeps its global address in the first
eight bytes, so rebinding a TMA operand is a store, never a re-encode.

`PROBLEM_FIELDS` names each `problem_size` position, so a key is self-describing.

**The patch set is a function of the contract, not of the kernel.** One built block, three
regimes:

| the caller may vary | per call |
|---|---|
| addresses | one store per operand |
| addresses and M | + `m`'s store, its descriptors re-encoded, `gridDimX` |
| anything | rebuild |

How many descriptors M costs depends on the epilogue, and the table is what says so rather
than a rule of thumb: under a TMA store the output descriptor spans M as well, so
`--vary-m` re-encodes A **and** C and leaves B alone; under an STG epilogue the output is a
bare address and only A is re-encoded. Either way B never moves, because N and K did not.

## Where the rest comes from

Nothing in the demo is harvested from a captured launch.

* **The cubin** is `compiled._launchable.__cubin__`, which the DSL keeps when
  `CUTE_DSL_KEEP` asked for it — set before the first compile. It is not in `ir_module`:
  by the time that is visible it has been lowered to LLVM and the `gpu.binary` op is gone.
* **The kernel** is identified by being the only function in the module
  (`cuModuleGetFunctionCount` / `cuModuleEnumerateFunctions` — note the count comes
  first). Its name is then cross-checked against `kernel_info`, which is keyed by the
  mangled symbol. **No name is ever reconstructed.**
* **The geometry** is the generated module's own closed form:
  `grid = (ceil(m/cgrp_tile_m) * cluster_m, ceil(n/cgrp_tile_n) * cluster_n, batch)`,
  `block = (threads_per_cta, 1, 1)`, `cluster = cluster_shape_mnk`.
* **Dynamic shared memory** is the one field without a first-class source. All of it is
  dynamic, so `CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES` reads 0, and it is not a module
  constant — the DSL sums the kernel's `cutlass.Array` declarations at compile time. It
  reaches the device as exactly one constant in the compiled host module, which is what
  the demo reads. Recomputing it here would duplicate a layout that lives in the kernel
  body and would drift the first time a buffer is added, so the durable fix is upstream:
  a property next to `__cubin__`.

## Scope, and how it extends

The demo is one narrow case on purpose — a dense bf16 matmul with a TMA-store epilogue —
so that its claims can be checked in a minute.

What is classified today, measured by compiling each flavor and comparing the table's
widths against `cuFuncGetParamInfo` on the resulting cubin:

| flavor | outcome |
|---|---|
| plain, relu epilogue, 1024³, nvfp4 block scale | every parameter classified |
| aux bias, two dense outputs, amax reduction, multi-gemm | refused: 1-2 `unknown` |
| MoE | no table emitted at all |

The refusals are all the same parameter. An STG epilogue passes its output as
`mC_tap_i: cute.Tensor`, whose fake this does not model, so it comes out `unknown` and a
consumer must refuse the kernel. **That is the mechanism working, not a gap in it** — and
it is the obvious next increment, since codegen writes that fake's shape and stride
expressions a few lines away from where the table is built.

Three kinds cover everything classified so far:

| kind | width | rebinding |
|---|---|---|
| `scalar` | 4 or 8 | a store |
| `tma` | 128 | a store into the first 8 bytes |
| `tensor` | 8 + 8 per symbol its fake carries | a store, plus the tail if the shape moved |

(`ptr` exists too, for a kernel that takes a bare address; nothing emits one today.)

MoE gets no table because its `problem_size` carries `num_experts` and `num_groups` where
a dense one carries `batch`, and its operands are unclassified — there is nothing to
describe truthfully, so it describes nothing. A per-call stream is also not done; that is
one more store into the launch configuration.

Linear attention needs much less of this than gemm does. Its per-sequence boundaries live
in a device array, so a ragged shape change never touches the parameter block; each
operand carries its address plus one extent, and the TMA descriptors are built by device
kernels into a workspace rather than encoded on the host. The per-call cost there scales
with the number of launches, not with shape complexity. (Measured separately, on the GDN
forward path; not reproducible from this file.)
