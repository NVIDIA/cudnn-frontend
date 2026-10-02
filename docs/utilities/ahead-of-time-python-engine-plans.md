# Ahead-of-time plans for Python engines

`graph.serialize()` works when the selected plan belongs to a Python (CuTeDSL) engine. The blob carries the compiled kernels and the launch sequence the plan runs. `deserialize()`, in Python or in C++, gives back a graph that executes it with no Python engine, no JIT and no cuDNN handle.

This is the same serialize/deserialize API that cuDNN backend plans already use. A graph whose selected plan is a backend plan serializes exactly as before.

## Usage

Build on a machine with the GPU you will deploy on:

```python
graph.validate()
graph.build_operation_graph()
graph.create_execution_plans([cudnn.heur_mode.A])
graph.check_support()
graph.build_plans()             # the selected plan may be a FROST engine's
blob = bytes(graph.serialize())  # raises if that plan cannot be exported
```

Run it in any process, Python:

```python
graph = cudnn.pygraph()
graph.deserialize(blob)          # no cutlass import, no compile
graph.execute({uid: buffer, ...}, workspace)
```

or C++ (samples/aot/run_plan.cpp):

```cpp
cudnn_frontend::graph::Graph graph;
graph.deserialize(blob);                   // no handle, no device properties
graph.get_workspace_size(workspace_size);
graph.execute(handle_or_nullptr, uid_to_ptr, workspace);  // stream: the handle's, else the default stream
```

`samples/aot/` has both sides end to end.

## What can be exported

| Plan | Exported |
|---|---|
| cuDNN backend plans | Yes, as before (needs libcudnn where it runs) |
| FROST SDPA forward, SM100 / SM107, f16 / bf16, prepared launch: dense, decode, split-KV, THD (packed or padded Stats), paged KV | Yes |
| FROST SDPA forward on the tensor ABI (fp8 / mxfp8, SM120, SM80, layouts the prepared launch declines) | No |
| FROST SDPA backward, FROST GEMM, FROST conv, linear attention, C++ OSS RMSNorm+SiLU | No |

A plan that cannot be exported makes `serialize()` raise `cudnnGraphNotSupportedError` with the reason. It never writes a different plan than the one that runs.

## What a deserialized plan is

- **The declared shapes.** The artifact is the plan for the graph as declared, like a backend plan. Execute-time shape overrides are refused. The kernels are shape-generic, so graphs of different shapes carry the same kernel. The loader maps each distinct kernel once per process.
- **The GPU it was built for.** Compute capability and SM count must match exactly. The cubins are architecture-specific SASS, and FROST kernels bake the SM count into their schedules. A mismatch is an error at `deserialize()`.
- **Linux, with two runtime libraries.** The kernels link against `libtvm_ffi.so` (apache-tvm-ffi) and `libcute_dsl_runtime.so` (nvidia-cutlass-dsl), resolved by SONAME. The Python package preloads both from its installed wheels. A C++ process needs them on `LD_LIBRARY_PATH` or its rpath.
- **Native code.** The blob contains shared objects that are loaded and run. Load only blobs you trust, as you would a shared library.
- **The handle's stream.** Execute launches on the stream of the handle it is given, or the default stream with none, like any deserialized backend plan. Pass a handle whose stream is the one you are capturing or ordering against. `populate_cuda_graph()` is not available; capture `execute()` on a stream instead.
- **Capture- and thread-safe.** Execute allocates nothing, never synchronizes, and may run from many threads. First calls of a kernel are serialized internally, because the CuTeDSL runtime's one-time module init can deadlock when two threads race it.

## How it works

Exporting needs a C compiler driver (`$CC`, `cc`, `gcc` or `clang`) to link each kernel's exported object into a shared object.

A plan is exportable when it implements `CompiledPlan.launches(graph, variant_pack, ctx)`. That method returns exactly what `execute()` issues, in order, without issuing it. Both are built on one binding function, so the two cannot drift apart. A launch is either a compiled kernel's positional tvm-ffi entry with its arguments, or a stream-ordered 32-bit fill from `cudnn.frost.buffers`.

Export calls `launches()` twice, over a variant pack describing every operand exactly as declared. Each operand and the workspace sit at a distinct placeholder address, and the addresses differ between the two calls. Export then classifies each argument:

- **Constant.** Equal in both calls, so its value is bound into the artifact.
- **Address.** It moved with exactly one operand, or with the workspace. It is stored as that buffer plus a byte offset.
- **Anything else.** Export refuses the plan and names the argument.

Nothing is launched, allocated or read from the device during export.

The payload sits under the `"aot"` key of the usual serialized graph (`include/cudnn_frontend/experimental/aot_engine.h` reads it):

```text
format, abi ("tvm-ffi"), target {compute_capability, sm_count}, producer {versions},
workspace_size, modules [shared objects], steps [
  {op: "call", module, symbol, args: [{int|float|bool|none|stream|array|tensor+offset|workspace}]},
  {op: "fill32" | "fill32_2d", dst, count | pitch/width/height, word}]
```

## Adding an engine

Implement `launches()` on the same binding `execute()` uses (`cudnn/sdpa/fwd/prepared.py` is the reference). Arguments must be integers, floats, `None`, tuples of integers, buffer addresses, or the launch stream. Kernels must be compiled with `--enable-tvm-ffi` (through `compiled_cache.compile_cached` or in-process). Per-call host work the artifact must reproduce has to be a fill; anything else, such as a torch op, keeps the plan unexportable.
