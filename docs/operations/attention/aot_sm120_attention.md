# Ahead-of-time attention for SM120 (experimental)

`libcudnn_frontend_aot_sm120.so` serves attention forward on compute capability 12.0 GPUs (RTX 50xx, RTX PRO Blackwell) to
C and C++ applications. It has no Python, cuDNN, NVRTC or runtime compiler. Every kernel is compiled ahead of time. A plan is
built once per *shape class*, and `execute()` takes the run-time extents, strides and pointers and only launches. The
motivating consumer is llama.cpp/ggml, where building a cuDNN graph cost about 30 ms per new shape.

**Experimental:** the C ABI may still change before the first release that ships it.

## Contract

- **Nothing compiles at run time.** The library holds SASS for `sm_120a` only (no PTX), so it loads on compute capability
  12.0 and nowhere else. `ldd` lists libstdc++, libm, libgcc_s and libc; the CUDA runtime is linked statically and the CUDA
  driver is the only other run-time dependency.
- **Plan build loads code; execute() launches.** `plan_create` loads every kernel module the plan can launch (a few ms for
  the first plan of a class in a process). `execute()` validates its arguments, looks up the kernel and launches it: no
  allocation, no synchronization, no module loading.
- **Caller-owned workspace.** Split-KV launches need a 256-byte aligned device workspace (`workspace_size()`); other launches
  need none.
- **Streams and CUDA graphs.** A plan is immutable once built. One plan may be executed concurrently from several threads and
  streams, each call with its own workspace, and `execute()` can be captured in a CUDA graph.

## Two backends behind one API

| KV dtype | kernels | Q / O | masks | head dim |
|---|---|---|---|---|
| f16, bf16 | FROST SM120 prefill templates (`python/cudnn/sdpa/fwd/kernels/sm120/`), exported ahead of time with run-time batch, sequence lengths, head counts and strides; decode-shaped calls run them split-KV and PackGQA | same dtype as KV | none, causal (bottom-right), causal + 128-token window with sinks | 64, 128, 256 |
| q8_0, q4_0 (ggml blocks) | flash-decode over the quantized cache (`aot/sm120/src/decode_q8_q4.cu`) | f32 | none or an additive f16 `[b, s_q, s_kv]` mask | 128, 256 |

The plan's `kv_dtype` picks the backend. A combination outside this table fails `plan_create` with
`CUDNN_FE_AOT_STATUS_NOT_SUPPORTED` and a message naming what is missing. Run-time extents outside a backend's envelope fail
`execute()` the same way. Neither ever falls back silently.

## C++ API

`include/cudnn_frontend/experimental/sm120_aot_attention.h` is header-only. It dlopens the library and needs only
`cuda_runtime.h`. The library is found through `Library::load(path)`, then `$CUDNN_FRONTEND_AOT_SM120_LIB`, then
`libcudnn_frontend_aot_sm120.so` on the loader path.

```c++
#include "cudnn_frontend/experimental/sm120_aot_attention.h"
namespace aot = cudnn_frontend::experimental::sm120_aot;

// Once per shape class (e.g. at model load).
aot::SdpaDesc desc = aot::make_desc();
desc.head_dim  = 128;
desc.q_dtype   = desc.kv_dtype = desc.o_dtype = CUDNN_FE_AOT_DTYPE_F16;
desc.mask      = CUDNN_FE_AOT_MASK_CAUSAL;        // bottom-right: query i sees keys j <= i + s_kv - s_q
desc.gqa_ratio = 4;                               // every call has h_q == 4 * h_kv (0: any ratio)
aot::SdpaPlan plan;
if (plan.build(desc) != CUDNN_FE_AOT_STATUS_SUCCESS) { /* plan.last_error() */ }

// Every call: extents, BYTE strides {batch, seq, head}, pointers. The head dim is contiguous.
aot::SdpaArgs a = aot::make_args();
a.b = n_seq; a.h_q = 32; a.h_kv = 8; a.s_q = n_tokens; a.s_kv = n_kv;
a.q = q; a.k = k_cache; a.v = v_cache; a.o = o;
a.q_stride[0] = ...; a.q_stride[1] = ...; a.q_stride[2] = ...;   // likewise k_stride, v_stride, o_stride
size_t ws_bytes = 0;
plan.workspace_size(a, &ws_bytes);               // depends on extents only
plan.execute(a, workspace, ws_bytes, stream);    // launches only
```

Strides are in bytes, so a ggml view passes straight through: for a tensor with `ne = [d, s, h, b]`, the strides are
`{nb[3], nb[1], nb[2]}`. FROST operands need 16-byte aligned pointers and strides. A KV cache whose sequence stride spans
every KV head, and a K/V slice of a larger cache, both work.

`plan.query(args, &knobs, &kernel_name)` reports what `execute()` would launch.

## C ABI

`include/cudnn_frontend/experimental/sm120_aot_attention_abi.h` declares the flat C ABI that the library exports (11
functions). Its rules:

- **Growable structs.** Every struct starts with `struct_size`, and structs only grow at the end. The library reads only the
  fields the caller's `struct_size` covers and treats the rest as zero.
- **ABI version.** `cudnn_fe_aot_sm120_abi_version()` lets a loader refuse an older library; the C++ header does this.
- **Status codes.** Every call returns a `cudnn_fe_aot_status_t`, and `cudnn_fe_aot_sm120_last_error()` gives this thread's
  reason for the last failure.

## How a kernel is selected

The selection is decided offline. `aot/sm120/build.py` runs FROST's own SM120 heuristic
(`cudnn.sdpa.fwd.heuristics.recommend`, the one the Python engine uses) over a grid:
- `b * h_kv` from 1 to 128, `s_q` from 1 to 16384 and `s_kv` from 1 to 131072, in power-of-two buckets;
- G = h_q / h_kv from 1 to 8, plus one table for G > 8;
- SM counts 170 (RTX 5090) and 188 (RTX PRO 6000).

At each grid cell it keeps the first proposed knob set that is compiled; if none is, it snaps the first proposal into the
compiled domain. On the default matrix, 73.2% of cells use the heuristic's first choice unchanged. The result is emitted as
`uint8` tables per (class, SM count, G) and compiled into the library.

At run time, `execute()` buckets the extents by `ceil(log2(x))`, clamps past the last bucket, and reads the table. The plan
picks the table whose SM count is nearest the device's.

Selection only affects performance. Every compiled kernel of a class computes the same function for any extents; a PackGQA
kernel is picked only for calls whose G equals its baked ratio.

To tune, `aot::knob_sets(desc, ...)` lists the prebuilt knob sets of a class. Pinning one with `plan.build(desc, &knobs)`
bypasses the table, and a set that is not prebuilt is `NOT_SUPPORTED`.

The quantized-KV backend picks its own configuration from (head dim, KV type, n_q * G), plus the split count from the SM
count and occupancy.

## The default matrix

`aot/sm120/variants.py` is the only place the matrix is declared:

- **Dtypes:** f16 and bf16.
- **Classes:** d 64 / 128 / 256 × {no mask, causal}, plus d 64 causal with sinks and d 64 causal + 128-token window with
  sinks (gpt-oss).
- **PackGQA ratios:** G = 2, 4, 6 and 8, and only G = 8 for the sink classes. G = 6 uses PackGQA with `tile_q % G != 0`.
- **Splits:** 1, 2, 4, 8, 16 and 32. Splits combine with sinks: the combine folds each head's sink once.
- **Tiles:** the class's KV tile (128, or 64 at d = 256).
- **Extra knob sets:** unsplit sets at tile_m 64 and 128 for tuning.

That is 444 prefill kernels and 4 combine kernels, 47.5 MiB in total. Compiling takes about 25 s and the heuristic pass about
15 s on a 96-core build host; no GPU is needed.

To add a variant, edit `CLASSES`, `PACK_GQA`, `SPLITS`, `EXTRA_KNOBS` or `DTYPES` in `variants.py` and rebuild. The tables are
regenerated from the heuristic, and `test_sdpa_sm120` enumerates and runs every prebuilt knob set, so a new variant is tested
without editing the test.

## Build

```bash
pip install -e .        # this checkout + the CuTeDSL wheel (nvidia-cutlass-dsl >= 4.8)
cmake -B build -DCUDNN_FRONTEND_BUILD_AOT_SM120=ON -DPython3_EXECUTABLE=$(which python) \
      -DCUDNN_FRONTEND_BUILD_SAMPLES=OFF -DCUDNN_FRONTEND_BUILD_TESTS=OFF
cmake --build build -j  # -> build/aot/sm120/lib/libcudnn_frontend_aot_sm120.so, build/aot/{test,bench}_sdpa_sm120
```

The option is OFF by default, and `include/` stays header-only. `python aot/sm120/build.py --out <dir>` builds the library
without CMake.

The steps are:
1. the heuristic pass;
2. one `cute.compile(..., options="--gpu-arch sm_120a")` plus `export_to_c` per kernel, through
   `python/cudnn/sdpa/fwd/kernels/sm120/aot_host.py`;
3. the generated dispatch registry;
4. nvcc for the decode kernels;
5. one link.

### The CuTeDSL runtime is not linked

An `export_to_c` object embeds its cubin and builds its TMA descriptors itself. Its only imports are ten thin wrappers
(`_cudaLaunchKernelEx`, `_cudaLibraryLoadData`, `_cuTensorMapEncodeTiled`, ...) that `libcute_dsl_runtime.so` (42 MB)
provides. `aot/sm120/src/runtime_shim.cpp` provides them instead:
- the `cuda*` ones forward to cudart;
- the `cu*` ones resolve the driver function through `cudaGetDriverEntryPointByVersion`;
- `_cuTensorMapEncodeTiled` applies the CuTeDSL runtime's post-encode fix for drivers up to CUDA 13.1.

Each wrapper was read from the CuTeDSL 4.8.0 runtime's disassembly. `build.py --runtime cute-dsl` links the original runtime
instead, and `test_sdpa_sm120 --compare-lib` checks that both libraries produce bit-identical outputs.

## Tests and benchmark

- `test_sdpa_sm120 [--lib <so>]` checks prefill (f16/bf16) against an fp32 reference, in two ways:
  - through the selection table, over seven shapes and G in {1, 2, 3, 4, 6, 8, 12};
  - with every prebuilt knob set pinned.

  It also checks:
  - quantized decode against a double-precision reference (84 directed and 40 random cases);
  - CUDA-graph capture and replay (bit-identical);
  - two concurrent streams (bit-identical to single-stream runs);
  - rejection of unsupported classes and bad arguments.

  It exits 0 with `SKIPPED` when there is no compute capability 12.0 device or no library.
- `aot/sm120/test/crosscheck_frost.py --lib <so>` runs the AOT library and the Python FROST engine with the same knob set on
  the same inputs and requires bit-identical outputs.
- `bench_sdpa_sm120 [--lib <so>]` reports, per case:
  - plan build time;
  - the host time of the first `execute()` after the build;
  - the steady-state host time per `execute()`;
  - the kernel time.

## Limits

- **Architecture:** `sm_120a` only; sm_121 (DGX Spark) is not built.
- **Masks:** the f16/bf16 kernels mask by index. There is no mask tensor, ALiBi or softcap, and a sliding window is built
  only with sinks at d = 64. A caller with a causal mask tensor (llama.cpp's `KQ_mask`) has to recognize it as causal.
- **Outputs and sequences:** there is no LSE output. Batches are dense: every sequence has the same `s_q` and `s_kv`.
- **Quantized KV:** n_q (= `s_q`) up to 8, K and V of the same type, no sinks, no window.
- **Platforms:** Linux only. The C++ loader has a Windows code path, but it is untested and the build is Linux-only.
