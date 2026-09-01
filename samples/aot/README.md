# AOT kernel export/import — one file per step

Three ways a kernel reaches a caller, in increasing order of commitment. Flow 1
is what ships today. Flows 2 and 3 are the AOT track: they share one build path
and one call machinery, and differ only in whether the compiled kernel becomes a
file or stays resident in the process.

Each flow is split into the steps a real deployment splits into, so a reader can
see exactly which half needs a compiler and which half does not.

**The kernel is a TMA tile add** — `c = a + b` over 2-D `fp32` tiles, operands
loaded through TMA, written in CuTeDSL. Small enough that the samples stay about
the flows, but not a toy in the way a flat vector add is: it builds TMA
descriptors on every call, which is what a real kernel does and what dominates
per-call host cost once a kernel has many operands. All three flows run the same
kernel, so they differ in how it is reached and in nothing else.

A production kernel is exported the same way: [`bench_sdpa_export.py`](bench_sdpa_export.py)
puts a FROST SDPA forward graph through the identical lifecycle. It needs
`CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1`, which is how the FROST engines opt in.

## Flow 1 — imperative (today, experimental)

| file | step |
|---|---|
| [`flow1_imperative.py`](flow1_imperative.py) | build *and* execute, same process, every process |

Compile the kernel in this process and call it. No graph, no artifact, no ABI —
which is what the shipped per-family entry points do underneath. It JITs on
first use and caches in a module-level dict that dies with the process; the
sample prints the cold and warm call times, and the gap between them is the cost
flows 2 and 3 exist to remove.

## Flow 2 — AOT, export to file

| file | step | needs a compiler? |
|---|---|---|
| [`flow2_export.py`](flow2_export.py) | build box: compile the set, write one container | yes |
| [`flow2_import_execute.py`](flow2_import_execute.py) | deploy box, Python: open it, execute by name | **no** |
| [`flow2_import_execute.cpp`](flow2_import_execute.cpp) | deploy box, C++: same, no Python in the process | **no** |

The export step is the only one that needs a GPU toolchain or cutlass. The two
consumers are interchangeable — same container, same uid order, same results.
`flow2_import_execute.py` asserts that no kernel toolchain was loaded in its
process, which is the property the whole flow exists for.

## Flow 3 — AOT, register to memory

| file | step |
|---|---|
| [`flow3_register_global.py`](flow3_register_global.py) | compile and publish under a name — **nothing is written to disk** |
| [`flow3_global_execute.py`](flow3_global_execute.py) | fetch by name and execute, from Python |
| [`flow3_global_execute_nanobind.cpp`](flow3_global_execute_nanobind.cpp) | fetch by name and execute, from a C++ extension in the same process |
| [`flow3_orchestrator.py`](flow3_orchestrator.py) | drives both consumers against one registration |

`flow3_orchestrator.py` is the interesting one:

```
A)  register_global (Python)  ->  execute (Python)
B)  register_global (Python)  ->  execute (C++, nanobind extension)
```

Case B is the half a pure-Python sample cannot show. The kernel is compiled by
Python and never serialised; the C++ executor is a *separate shared object* in
the same process and reaches the same graph by name. It prints what the
extension sees in the registry, which is how you can tell it shares the real one
rather than holding a private copy — the failure mode that
`kernel_library.h`'s default-visibility annotation exists to prevent.

## Shared

| file | what |
|---|---|
| [`common.py`](common.py) | the graph every sample builds, and the per-call pointer gather |
| [`build.sh`](build.sh) | compiles the two C++ pieces |
| [`bench_cpu_costs.cpp`](bench_cpu_costs.cpp) / [`.py`](bench_cpu_costs.py) | every way of reaching one kernel, priced against each other |
| [`bench_native_add.cu`](bench_native_add.cu) | the hand-written kernel the two floor rows launch |
| [`bench_sdpa_export.py`](bench_sdpa_export.py) / [`bench_sdpa_cpu_costs.cpp`](bench_sdpa_cpu_costs.cpp) | the same pricing on a FROST SDPA forward graph |
| [`tile_add_primitives.py`](tile_add_primitives.py) | the same kernel one level down, in `cutlass.experimental.primitives` |

`bench_cpu_costs` is the one to run for a table. Every arm ends in the *same*
CuTeDSL kernel out of the *same* container — flow 2, flow 3, a direct tvm-ffi
call, the tvm-ffi global table, and (as a floor) a hand-written kernel through
both the runtime and driver APIs. Because the arms differ only in the call path,
`fe_* − ffi_*` is the frontend's front door over an OSS-engine plan, which is
otherwise easy to overstate by differencing two measurements that ran different
dispatch stacks over different kernels.

### Pricing TMA

`--plain` exports the same arithmetic with a flat 1-D load and no descriptors.
It is not a flow — it exists so the two can be differenced:

```bash
python flow2_export.py --bench            # the TMA tile add
python flow2_export.py --plain --bench    # the same add, no TMA
./bench_cpu_costs mykernels.cudnn
./bench_cpu_costs plainkernels.cudnn
```

| call path | plain add | TMA tile add | delta |
|---|---|---|---|
| `ffi_module` (C++, TVM FFI) | 1.86 | 2.04 | +0.18 |
| `fe_container` (FE `graph.execute`) | 2.09 | 2.35 | +0.26 |

Two descriptors per call, so roughly 0.09 µs each — an order of magnitude below
FROST's ~1.5 µs per operand. TMA descriptors are cheap; FROST's `_build_descs`
is the outlier.

The front door does not move with TMA: +0.22 µs on the plain add, +0.31 µs on
the TMA one, against a ±0.02 µs noise floor. It is a property of the caller, not
of what the kernel does.

## Running

```bash
export CUDA_PATH=/path/to/cuda CUDNN_PATH=/path/to/cudnn

python flow1_imperative.py                  # flow 1

python flow2_export.py                      # flow 2, build box
python flow2_import_execute.py              #         deploy box, Python
./build.sh && ./flow2_import_execute        #         deploy box, C++

python flow3_register_global.py             # flow 3, publish
python flow3_global_execute.py              #         fetch + execute, Python
./build.sh && python flow3_orchestrator.py  #         both consumers, one registration
```

`build.sh` also builds `bench_cpu_costs`, which needs the extra artifacts from
`python flow2_export.py --bench`.

## Requirements

A Blackwell GPU, CUDA 13.x, and cuDNN frontend built with the AOT entry points —
which compile in only when `apache-tvm-ffi` is importable at CMake configure
time:

```bash
pip install -e ".[cutedsl]" --no-build-isolation
pip install nanobind          # flow 3's C++ consumer only
```

`--no-build-isolation` matters: under PEP 517 isolation `tvm_ffi` is not
importable when CMake configures, so the AOT entry points are left out and
report that cleanly at runtime. `nanobind` is needed only to build
`flow3_global_execute_nanobind`; without it `flow3_orchestrator.py` skips case B
and still runs case A.

## Scope

The engine behind flows 2 and 3 is `CuteDslTmaAddEngine`, written as the
vehicle, so these samples exercise no workspace slots, no pass-by-value scalars
and no replacement slots. `bench_sdpa_export.py` covers a kernel that does.
The cuDNN backend engine and the cuda-python / CUDA C++ engines report "not
implemented" from the AOT entry points; retrofitting them is real work and is
not done here.

[`tile_add_primitives.py`](tile_add_primitives.py) is the same kernel written
against the raw primitives instead of `cpasync.make_tiled_tma_atom` — the
descriptor built by hand, the PTX bulk-tensor copy issued by hand, the mbarrier
driven by hand. It is not a fourth flow and nothing imports it; it is there
because that is the level FROST works at, so it is the level you end up
debugging at. Its own four gotchas are listed in its docstring, the sharpest
being that the descriptor parameter must be `cutlass.GridConstant[TensorMap]`.
The canonical form for these is the DKG tutorial at
`examples/CuTeDSL/experimental/primitives/tutorial/04_tma_load.py`.

Writing the TMA kernel turned up four things that compile clean and fail at
runtime, recorded in `cutedsl_tma_add_engine.py` next to the code that avoids
them: `cute.copy` must not be inside `elect_one` (the kernel deadlocks in
`mbarrier_wait`, which retries forever rather than erroring); `tma_partition`
wants the gmem tensor already tiled to the CTA; a layout built on the host
cannot cross into the kernel region; and `make_fake_compact_tensor` defaults to
a stride order that is not row-major, so the artifact compiles and then rejects
a row-major caller at the ABI boundary.
