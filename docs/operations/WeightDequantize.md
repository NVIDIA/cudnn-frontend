# Custom weight dequantization (experimental)

This PR implements **milestone (a): decoder-owned global loads**. The customer
CUDA C++ function owns all packed-weight, embedded-scale and auxiliary-table
loads. Decoder staging/scratch uses zero shared memory; the tensor-core GEMM
still uses its ordinary shared-memory A/B tiles. Nonzero scratch declarations
are rejected. Managed TMA/vector loads and multi-row/iterative block decoding
are deferred to milestone (b), whose block lifetime/alignment ABI remains open.
The earlier managed-load prototype is preserved on
`scottyokim:syokim/custom-weight-dequantization-managed-loads` at `e1dfa7e9`.


`weight_dequantize` lets an application define a narrow weight **numerical data
format** in CUDA C++ and feed its dequantized values directly to a GEMM. The
program interprets each code (for example signed INT8/INT4/INT2, a custom floating
point representation, or a codebook entry), applies the chosen scaling scheme,
and produces FP16/BF16 values. It can apply block scales, a global scale, or both.
A new numerical format does not require a new frontend or backend datatype enum.

This is a prototype requiring a backend build that implements the experimental
weight-decode program and operation descriptors described below.
The public C++ API remains declared when built with ordinary cuDNN headers, but
lowering returns `GRAPH_NOT_SUPPORTED` without `CUDNN_WEIGHT_DECODE_ABI_VERSION`.
Patched headers with an unpatched runtime are also rejected at descriptor
creation. A matching version number (including 9.28) alone is insufficient.
Do not treat this as a released or stable cuDNN ABI.

## C++ graph API

```cpp
std::shared_ptr<Tensor_attributes> Graph::weight_dequantize(
    std::shared_ptr<Tensor_attributes> weights,
    std::vector<std::shared_ptr<Tensor_attributes>> auxiliaries,
    Weight_dequantize_attributes attributes);
```

`Weight_dequantize_attributes::set_program(Weight_dequantize_program const&)`
copies the source and metadata. `set_name` names the operation. The program has
these fields and corresponding fluent `set_<field>` setters:

| Field | Type | Default / contract |
| --- | --- | --- |
| `source` | `std::string` | Required CUDA C++ source; 1–1048575 bytes, no embedded NUL |
| `entry` | `std::string` | Required unqualified ASCII identifier, at most 127 bytes |
| `abi_version` | `int64_t` | 1 |
| `tile_shape` | `std::vector<int64_t>` | `[32,64]`; the current engine supports only this tile |
| `cta_smem_bytes` | `int64_t` | 0; reserved, nonzero rejected |
| `stage_smem_bytes` | `int64_t` | 0; reserved, nonzero rejected |
| `input_alignment` | `int64_t` | 16; power of two, at most 256, promised for every physical input |
| `constants` | `std::vector<int64_t>` | Empty; at most 64 compile-time values, in supplied order |

`weights` describes physical INT8/UINT8 byte storage, including for formats whose
codes are not integers. Supply zero to eight ordered physical auxiliary tensors
for scales, zero points, codebooks, indices, or other format metadata. They are
normal runtime tensors bound in the variant pack; changing their contents does
not require recompilation. Their order determines `auxiliary[i]` in the program.

The result is a **virtual logical B tensor**. Set its `[1,K,N]` dimensions
explicitly: they cannot be inferred from the number of encoded bytes. Set its
datatype to HALF or BFLOAT16, or inherit that type from the graph's intermediate
datatype. Its default stride is row major. Do not mark it as a graph output or
bind an allocation for it. For example, with `source` defined below:

```cpp
namespace fe = cudnn_frontend;
fe::graph::Graph graph;
graph.set_io_data_type(fe::DataType_t::HALF)
     .set_intermediate_data_type(fe::DataType_t::HALF)
     .set_compute_data_type(fe::DataType_t::FLOAT);
int64_t M = 37, K = 83, N = 75, lda = 88;
int64_t bytes = (K * N * 4 + 7) / 8, scale_count = ((K + 23) / 24) * ((N + 19) / 20);
auto W = graph.tensor(fe::graph::Tensor_attributes().set_dim({1,1,bytes})
    .set_stride({bytes,bytes,1}).set_data_type(fe::DataType_t::UINT8));
auto S = graph.tensor(fe::graph::Tensor_attributes().set_dim({1,1,scale_count})
    .set_stride({scale_count,scale_count,1}).set_data_type(fe::DataType_t::FLOAT));
auto G = graph.tensor(fe::graph::Tensor_attributes().set_dim({1,1,1})
    .set_stride({1,1,1}).set_data_type(fe::DataType_t::FLOAT));
auto program = fe::graph::Weight_dequantize_program()
    .set_source(source).set_entry("decode").set_constants({4, 3, 24, 20});
auto B = graph.weight_dequantize(W, {S, G},
    fe::graph::Weight_dequantize_attributes().set_name("dequant").set_program(program));
B->set_dim({1,K,N}).set_data_type(fe::DataType_t::HALF);
auto A = graph.tensor(fe::graph::Tensor_attributes().set_dim({1,M,K})
    .set_stride({M*lda,lda,1}));
auto C = graph.matmul(A, B, fe::graph::Matmul_attributes());
C->set_output(true).set_data_type(fe::DataType_t::FLOAT);
```

Then use the ordinary validate/build/plan/execute sequence, checking each returned
status. The current prototype engine is native GEMM engine 10 with default
configuration; tests select it with `create_execution_plan(10, {})`. Bind A,
W, S, G and C at execution. FORT inserts the decoder into the GEMM's shared-memory
producer; there is no separate dequantization kernel or dense B global workspace.
After building the plan, always query and allocate execution workspace. Small
grids can select split-K and require up to 64 MiB for FP32 partial outputs,
followed by a deterministic reduction launch. This is separate from decoder
scratch, which remains zero. Using the sample's RAII `Surface` helper:

```cpp
int64_t workspace_bytes = 0;
// Check the returned status, as for the other graph calls.
graph.get_workspace_size(workspace_bytes);
Surface<uint8_t> workspace(std::max<int64_t>(1, workspace_bytes), 0);
graph.execute(handle, bindings, workspace_bytes ? workspace.devPtr : nullptr);
```

The workspace allocation must remain alive until execution completes, be
16-byte aligned, and not overlap any bound tensor. Use independent workspace
and outputs for concurrent executions. It needs no initialization.
See the executable [C++ sample](../../samples/cpp/matmul/custom_weight_dequantize.cpp)
for allocations, support probes, CPU references, serialization and plan reuse.

## Python graph API

The Python graph API exposes the same program fields as keyword arguments:

```python
B = graph.weight_dequantize(
    W, source=source, entry="decode", auxiliaries=[S, G],
    abi_version=1, tile_shape=[32, 64],
    cta_smem_bytes=0, stage_smem_bytes=0, input_alignment=16,
    constants=[4, 3, 24, 20], out_dims=[1, K, N], name="dequant",
)
B.set_data_type(cudnn.data_type.HALF)
C = graph.matmul(A, B)
C.set_output(True).set_data_type(cudnn.data_type.FLOAT)
graph.validate()
graph.build_operation_graph()
graph.create_execution_plan(10, {})
graph.check_support()
graph.build_plans()
workspace_bytes = graph.get_workspace_size()
workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=a_gpu.device) if workspace_bytes else None
graph.execute({A: a_gpu, W: storage_gpu, S: block_scales_gpu,
               G: global_scale_gpu, C: c_gpu}, workspace)
```

Create physical graph tensors with the same dimensions, strides and datatypes as
in the C++ example. `out_dims` or `B.set_dim([1,K,N])` sets the logical dimensions.
`source` is a CUDA C++ string, **not a Python callable**. The Python IR, pybind
binding, and C++ API all lower to the same backend program and operation
descriptors. Execution only binds pointers; it does not compile source, convert
weights on the host, allocate buffers, or synchronize the CUDA stream.

## Device program and numerical meaning

The backend prepends ABI v1 types and the conversion helper, then compiles source
in `namespace fort_user_weight_decoder` with NVRTC. Supply an ordinary `__device__`
function with this signature; do not redeclare the ABI types:

```cpp
using FortWeightDecodeValue = unsigned short;  // FP16/BF16 bits, supplied by FORT
struct FortWeightDecodeTileV1 {
    int abi_version, thread_id, thread_count, auxiliary_count;
    long long batch, k_begin, n_begin, full_k, full_n;
    int tile_k, tile_n, valid_k, valid_n, output_stride;
    int output_bfloat16;
    const long long* constants;
    int constant_count;
};
__device__ FortWeightDecodeValue fort_weight_decode_from_float(float);
```

This example interprets signed two's-complement 8/4/2-bit codes and optionally
applies block and global scales. Each row is part of one contiguous bit stream;
`constants = [bits, scaling_flags, block_k, block_n]`, with flags 1 = global,
2 = block, 3 = block followed by global. The auxiliary order is block scales,
then global scale. A custom floating-point format would replace the signed
integer conversion with its exponent/mantissa interpretation.

```cpp
__device__ void decode(const FortWeightDecodeTileV1& t, const void* storage,
    const void* const* auxiliary, void*, void*, FortWeightDecodeValue* output) {
    const int bits=int(t.constants[0]), scaling=int(t.constants[1]);
    const long long block_k=t.constants[2], block_n=t.constants[3];
    const long long blocks_n=(t.full_n+block_n-1)/block_n;
    const auto* block_scales=static_cast<const float*>(auxiliary[0]);
    const auto* global_scale=static_cast<const float*>(auxiliary[1]);
    for (int i=t.thread_id; i<t.tile_k*t.tile_n; i+=t.thread_count) {
        const int row=i/t.tile_n, col=i%t.tile_n;
        if (row>=t.valid_k || col>=t.valid_n) continue;
        const long long k=t.k_begin+row, n=t.n_begin+col;
        const long long bit_index=(k*t.full_n+n)*bits;
        const unsigned code=(static_cast<const unsigned char*>(storage)[bit_index/8]
            >> (bit_index%8)) & ((1u<<bits)-1);
        // Give the code its signed numerical meaning, then dequantize it.
        const int value=int(code)-((code & (1u<<(bits-1))) ? (1<<bits) : 0);
        float weight=float(value);
        if (scaling & 2) weight *= block_scales[(k/block_k)*blocks_n+n/block_n];
        if (scaling & 1) weight *= global_scale[0];
        // Both scale operations precede the single rounding to the MMA type.
        output[row*t.output_stride+col]=fort_weight_decode_from_float(weight);
    }
}
```

The two scale multiplies occur before a single rounding to the chosen MMA type.
Scale blocks are located using full logical coordinates; they can cross producer
tile boundaries, as the 24x20 blocks above do. Two numerical scaling steps do not
require two decoder calls or two extra pipeline stages.

## Resource and support contract

All 256 CTA threads enter the callback uniformly. Distribute work with
`thread_id` / `thread_count`; write every valid output element using
`output_stride`. Guard every input/metadata read at K/N tails. FORT zeroes invalid
output elements after the callback and owns stage publication/reuse barriers.
CTA barriers inside the callback must be uniform across all threads.

Both scratch arguments are always null. Their reserved positions retain the
existing ABI 1 signature; programs must not declare private shared-memory
buffers. Only the output tile may be written. Complete synchronous loads before
returning; do not retain pointers, launch kernels, synchronize across CTAs or
use FORT's `cp.async` commit/wait groups. Source is trusted CUDA C++ code; support
checks cannot prove customer indexing or synchronization correct.

The GEMM stage budget excludes decoder staging/scratch:

```text
M_tile    = 16, 32, 64 or 128, depending on shape, scheduling and the SMEM budget
per_stage = 2 * (M_tile * 40 + 32 * 72)  # padded internal MMA operands
stages    = 2
```

For M divisible by 256, N and K divisible by 64, K >= 64, a grid of at least
one CTA per device SM and a budget of at least 50,176 bytes, the backend can
instead use M=256. It groups two unchanged K=32 callbacks into a K=64 mainloop
step, sharing engine barriers across the two tiles. This still allocates only
dense GEMM operands, with no decoder scratch or packed staging. The callback
ABI and all-thread participation remain unchanged.

The original padded choices require 14,336 / 19,456 / 29,696 bytes. A smaller budget
selects a smaller M tile, with an unpadded M=32 fallback at 12,288 bytes.
For small nongrouped grids, N >= 128 divisible by 64 and K >= 512 divisible by
32, the backend may partition K among 2-128 CTAs per output tile. M <= 16 then
uses an M=16 tile with 11,776 bytes of operand SMEM. The minimum accepted budget
remains 12,288 bytes. Selection uses the actual GPU's SM count, minimum work per
slice and a 64 MiB global-workspace cap. Each CTA writes FP32 partials, then a
second launch sums them in fixed order before the final output conversion.
No atomics or workspace initialization are required. Both launches participate
in capture/replay. Query the plan's workspace; do not assume it is zero.
The callback still sees a 32x64 output at stride 64 and all 256 CTA threads;
FORT performs the internal layout conversion afterward. Existing ABI 1
programs and legal CTA barriers remain valid. At least 12,288 bytes must fit. Scratch is rejected during frontend
validation, backend descriptor finalization, and native support planning.

The current engine accepts exact SM120 and NVRTC >=12.8, one decode feeding B of
one static unbatched row-major GEMM, FP16/BF16 A and B, FP32 compute and same-type
or FP32 C. A's leading dimension and base must be 16-byte aligned. Physical
storage and auxiliary tensors must be contiguous `[1,1,length]` vectors; their
promised alignment must satisfy the program and their natural datatype
alignment. Output must be virtual. Batched/dynamic/override shapes, decode-A,
additional fusions, alternate callback tile shapes and user split-K/Stream-K
knobs are rejected. Automatic internal split-K requires no new frontend option.
The backend remains authoritative for graph, resource, and launch-pointer checks.

The full source, entry, constants, resource declarations and ordered tensor
connections participate in structural serialization and graph identity; changing
them builds a different program. Runtime scale **values** are not part of that
identity. Plan serialization also retains the program. Treat serialized plans
containing customer device source with the same trust as the original source.
Serialized split-K kernels retain their original M tile, slice count and
workspace requirement; existing unsplit plans remain valid with zero workspace.

This frontend operation adds setup/lowering work only. It uses the existing
prototype's synchronous custom B producer and tensor-core GEMM; it does not
promise performance parity with built-in format-specific dequantization paths.
Measure each decoder and workload, including register use and occupancy.

The ggml sample maps adjacent warp lanes along physical K and accumulates eight
adjacent output columns for a vector store. This avoids scattered per-byte
loads across columns and the bank conflicts from scalar K-fastest output stores.
It keeps the original byte layout and unaligned-input support. The backend
mainloop now reuses each decoded tile across up to 256 M rows, uses padded
`ldmatrix` loads and groups two K slices when the shape and budget permit.
On an RTX PRO 6000, M=512/N=12288/K=4096 Q4_K initially improved from approximately 15.4 ms to 0.61 ms. The grouped-K upgrade
now measures about 0.45 ms through the public backend. A matched same-GPU test
measured llama.cpp MMQ at 0.18 ms, including activation quantization; the two
paths use different arithmetic contracts and cuDNN still has performance
headroom. Smaller shapes and constrained budgets retain the earlier kernel.
See the backend `docs/fort-native-weight-decode.md` for the reproducible
`nativeWeightDecodePrototype --benchmark` command and remaining limitations.

## Tests

The [C++ unit tests](../../test/cpp/weight_dequantize.cpp) cover validation,
program ownership, auxiliary ordering, graph identity and structural round trips.
The C++ sample and [Python tests](../../test/python/test_weight_dequantize.py) check
8/4/2-bit signed values with global/block/both scales and FP16/BF16, tail tiles,
scale blocks crossing tiles, and independent runtime scale updates. Python also
checks plan serialization and CUDA graph capture. They require the prototype
backend for execution and skip when its headers/runtime or hardware are absent.

```bash
./build/bin/tests '[weight_dequantize]'
./build/bin/samples '[weight_dequantize]'
cd test/python
pytest test_weight_dequantize.py
```

## Q4_K, Q6_K and IQ1_S without repacking

The C++ sample `samples/cpp/matmul/custom_weight_dequantize.cpp` also runs original
ggml Q4_K (144 bytes), Q6_K (210 bytes) and IQ1_S (50 bytes) blocks, each encoding
256 weights. Its customer CUDA C++ source is
[`weight_dequantize/ggml_decode.h`](../../samples/cpp/matmul/weight_dequantize/ggml_decode.h).
The same source string can be passed to Python's `graph.weight_dequantize`.

Physical row n contains ceil(K/256) complete blocks along K and produces logical
B[:,n]. Bytewise metadata loads avoid extra padding for the 210/50-byte block
strides. Q4_K reads its 16-byte metadata prefix and eight-byte code runs with
vector loads when the addresses permit, and reuses the local scale/minimum for
eight weights. Unaligned addresses use word or byte fallbacks without stronger
alignment requirements or reads past a complete physical block. A warp register
transpose turns the decoded values into adjacent N values for vector shared
stores. Tail lanes participate in every shuffle and carry zero values.
For very wide Q6_K matrices (N >= 65536), lane-independent block and bit-plane
calculations improve the measured output-head cases. Smaller Q6_K matrices
retain their original addressing, which performed better in the broader sweep.
IQ1_S also decodes eight values per lane, using the grid loads described below.
These optimizations are customer example code; the engine has no knowledge
of ggml formats.
The sample sets both the program input_alignment and the weight tensor
alignment to 1 and tests a deliberately unaligned weight base. This does not relax the natural alignment of auxiliary FLOAT tensors.

Q4_K applies the embedded superblock delta/minimum and 6-bit local scale/minimum
fields. Q6_K assembles signed six-bit values from separate bit planes, then
applies signed int8 local scales and the FP16 superblock scale. IQ1_S uses the
canonical 2048x8 ternary grid, an odd scale multiplier and a +/-1/8 offset;
it is not plain binary 1-bit dequantization. Its grid is a FLOAT auxiliary tensor
read directly from global memory. Each IQ1_S lane reuses one scale/offset across
eight values and loads the grid entry with two 16-byte loads when aligned.
Four-byte-aligned grid bases use scalar loads; no new alignment promise or
codebook representation is required. Warp shuffles transpose the lane-local
values for a vector shared-memory store. Q4_K and Q6_K require no auxiliaries.

Expected weights are computed from original codes/scales before packing.
Identity GEMMs verify every rounded output weight exactly (all IQ1_S indices),
and separate tail GEMMs exercise multiple blocks. M=16/N=256/K=544 cases exercise
automatic split-K and queried workspace with all three formats and both operand
types. A further M=1/K=128 case with N=512 times the actual GPU SM count covers the
wide-grid grouped path and aligned Q4 word loads for all formats and types.
IQ1_S tail and split-K cases also use grid bases offset by one or two FLOATs
to exercise the scalar fallback through the C++ frontend.
The C++ samples allocate the plan's workspace for every execution.
Python numerical tests similarly cover split-K for signed INT8/INT4/INT2 with
global/block/combined scaling, including plan reload and CUDA graph replay.
Both FP16 and BF16 are tested.
The grid's MIT license and pinned ggml source revision accompany the sample.
These are correctness examples, not measured performance comparisons to ggml.

## Wide-grid K128 mainloop update

The backend can group four unchanged 32x64 callback tiles when 1 <= M <= 32,
N is divisible by 64, K >= 128 is divisible by 128, and N/64 is at least eight
times the current GPU's SM count. M <= 16 uses a 16x64x128 engine tile with
23,552 bytes of operand shared memory; M <= 32 uses 32x64x128 with 28,672 bytes.
Each allocation holds one group. Smaller budgets or ineligible shapes keep
the previous selection. This reduces publication/layout-conversion barriers
for wide GEMMs such as the output head; it adds no decoder scratch, packed
staging or workspace. Other shapes can still require split-K workspace.

No frontend signature or device ABI change is needed. Existing saved plans
keep their original geometry; rebuild plans to obtain new scheduling. Supply
the updated example source and rebuild its plan to obtain the decoder load
changes. Model-shape timings and validation are recorded in backend MR !4559;
these examples remain correctness tests, not end-to-end model benchmarks.

## K128 grouping and packed-word example update (2026-10-09)

This update retains ABI 1 and decoder-owned loads. Wide small-M GEMMs can publish
four K32 tiles together, reducing engine barriers. Selection requires 1 <= M <= 32,
N divisible by 64, K >= 128 divisible by 128, at least eight CTAs per actual GPU
SM, and sufficient operand SMEM. M16/M32 use 23,552/28,672 bytes. The callback
still sees 32x64, stride 64, all 256 threads, and null scratch. No workspace is
needed on this grouped path. New serialized names preserve the exact M/K
geometry; existing plans keep their old geometry. Rebuild plans to retune.

The example Q4_K decoder loads eight code bytes per lane with aligned words
(or byte fallback), reuses scales/minima across eight values, and transposes
registers before its vector shared store. It requires only complete original
blocks, not padded/repacked weights. Wide Q6_K uses uniform block/plane
coordinates; the specialization stays inside the Q6 branch. IQ1_S and smaller
Q6_K address expressions remain unchanged. These format decisions exist only
in customer example source, not in the FORT engine. Supply the updated source
and rebuild the plan to obtain these sample-side gains.

Compared with backend `ab3765866` and frontend `94039fa6`, all 198 supplied
spreadsheet shapes pass the final numerical harness on ultra's RTX PRO 6000
(SM120, 188 SMs, CUDA/NVRTC 13.4). Six output-head shapes select K128; the
previous 143 split-K and 29 grouped-K64 selections remain. The improvements
are concentrated in Q4_K and the small-M output head; many other cases are
unchanged. The following controls use medians of three independent processes
per implementation, with before/after order alternated on the same GPU:

| Format / shape | M | N | K | Previous us | Updated us | llama.cpp us | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen head q6 | 1 | 248320 | 2048 | 764.8 | 535.1 | 276.3 | 1.43x |
| Qwen head q6 | 16 | 248320 | 2048 | 812.7 | 596.4 | 310.2 | 1.36x |
| Qwen head q6 | 32 | 248320 | 2048 | 837.4 | 582.7 | 333.7 | 1.44x |
| Qwen head q6 | 512 | 248320 | 2048 | 2941.5 | 2902.2 | 2430.1 | 1.01x |
| Llama FFN q4 | 1 | 4096 | 14336 | 103.7 | 85.6 | 10.0 | 1.21x |
| Llama FFN q4 | 1 | 14336 | 4096 | 105.3 | 85.9 | 11.9 | 1.23x |
| Llama FFN q4 | 512 | 4096 | 14336 | 660.4 | 480.7 | 212.7 | 1.37x |
| Llama FFN q4 | 512 | 14336 | 4096 | 454.8 | 450.4 | 209.8 | 1.01x |
| Llama FFN iq1 | 2 | 14336 | 4096 | 96.2 | 96.2 | 9.7 | 1.00x |
| Tail regression iq1 | 37 | 64 | 256 | 9.9 | 9.9 | 6.9 | 1.00x |

Full-sweep rows slower by more than 10% versus the prior local sweep: [].
The remaining worst model-shape cuDNN/llama.cpp ratio is 10.22x, so this is
not performance parity. The small-M IQ1_S cases remain a major gap.

Each process uses 11 alternating cuDNN/llama.cpp rounds of 20 dependent CUDA
graph operations, with three warmup replays per timed side. Allocation,
compilation and reference calculation are excluded; all cuDNN phases and
llama.cpp activation quantization/fixup are included. Inputs are reused, caches
are not flushed, and clocks are not locked. llama.cpp remains pinned to
`08246a28f6000100433d297c4e037c02e9d2d464`. These are synthetic fixtures at model
dimensions, not end-to-end models, tokens/s, or routed/grouped MoE.

Every packed weight is checked against the ggml CPU decoder; full outputs are
checked for finiteness and NRMSE against FP64 unrounded-weight GEMM, plus 256
CPU-double samples against FP16-rounded weights at atol=rtol=0.005. cuDNN uses
FP16 activations/decoded weights and FP32 accumulation; MMQ uses Q8_1 activation
quantization and INT8 MMA. The errors are retained beside timings; these tests
do not establish model-level accuracy or elementwise tolerance on every large
output. The tolerance was not relaxed.

Full backend integration and host planner tests pass. New tests cover exact
K128 SMEM/grid boundaries, partial M, single/multiple groups, CTA barriers,
FP16/BF16 output, plan and kernel-cache reload, capture/rebind, and Q4 packed
base offsets 1/2/3 with physical tail blocks. Frontend C++ samples pass
1,734,318 assertions in two cases, including all three formats and both operand
types on the aligned wide-grid path; Python passes 45 tests, zero skipped.
Compute Sanitizer: ggml/pipeline memcheck zero errors, pipeline racecheck zero
hazards/warnings, and ggml shuffle synccheck zero errors.

Actual backend CUBIN resources (including the runtime's dynamic-SMEM opt-in):

| Kernel / example | Registers/thread | Dynamic operand SMEM | Theoretical CTAs/SM |
| --- | ---: | ---: | ---: |
| K128 M16 Q6_K | 77 | 23,552 B | 3 |
| K128 M32 Q6_K | 77 | 28,672 B | 3 |
| Split-K M16 Q4_K | 40 | 11,776 B | 6 |
| K64 M256 Q4_K | 124 | 50,176 B | 2 |

All four inspected kernels have zero stack/local bytes and no LDL/STL/CALL
instructions. CUDA reserves an additional 1 KiB/CTA. These counts depend on
customer code and do not guarantee spill-free arbitrary decoders. Hardware
performance counters are unavailable to this account (`ERR_NVGPUCTRPERM`), so
these results use controlled timings and binary inspection, not counter-based
bandwidth or stall attribution.

The report, source snapshots, CSV, annotated workbook, controls and logs are in
`/tmp/cudnn-dequant-upgrade4` on ultra. See `README.txt`, `matrix/comparison.csv`,
`matrix/2026-10-08-cudnn-dequant-all-k128-words.xlsx` and `controls/summary.json`.
For example: `bash /tmp/cudnn-dequant-upgrade4/run-compare.sh 1 248320 2048 q6`.
Broad uniform-address and callback-interface experiments were discarded after
regression checks; no new callback ABI is exposed. Decoder/MMA overlap and
engine-managed packed staging remain future work. Other GPUs and Windows have
not been measured in this update. This remains a draft prototype.

## IQ1_S vector loads and guarded medium-M scheduling (2026-10-09)

This increment improves the customer IQ1_S decoder and the format-independent
split-K planner. It follows backend `1f00cd62a6` and frontend `3fae893f70`.

Each IQ1_S lane now decodes eight weights from one codebook index. It reuses
the block scale and offset, reads the existing FLOAT grid entry with two
16-byte global loads, and transposes registers before a 16-byte shared store.
The vector path checks the actual codebook pointer. Bases aligned to only
four bytes use scalar loads, including offsets of 4, 8 and 12 bytes. Packed
weights retain their original 50-byte blocks and byte-alignment support.
No codebook compression, repacking, scratch allocation or API change is needed.
Q4_K and Q6_K retain their existing decoder paths.

For 17 <= M <= 128, an unsplit grid with at least one but fewer than two CTAs
per actual GPU SM now targets four waves instead of two. Smaller grids retain
the old target: a broader four-wave policy regressed some cases by increasing
partial-output/reduction traffic and was discarded. Existing K divisibility,
minimum work per slice, shared-memory, shape and 64 MiB workspace checks still
apply. M <= 16 retains its eight-wave target; larger M retains two waves.
The engine never inspects the format constant or the decoder's source to tune.

ABI 1 remains 256 threads, a 32x64 stride-64 callback tile, and null scratch.
The decoder owns all loads. Synchronous B decoding and the existing MMA
pipeline remain; this update does not implement producer/MMA overlap or packed
staging. Rebuild plans to obtain new scheduling and supply the updated example
source for IQ1_S gains. Saved binaries retain their recorded geometry. Always
query execution workspace again after rebuilding a plan: more splits can
increase it even though decoder scratch remains zero.

All 198 supplied spreadsheet cases pass on ultra's RTX PRO 6000 Blackwell
Server GPUs (SM120, 188 SMs, CUDA/NVRTC 13.4). No final sweep case regressed
by more than 10% against the previous local sweep. Selected controls below
use the median of three independent processes per version, with old/new
order alternated on the same GPU. Timings are microseconds:

| Format / shape | M | N | K | Previous us | Updated us | llama.cpp us | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama iq1 | 2 | 14336 | 4096 | 96.1 | 78.4 | 9.6 | 1.23x |
| Llama iq1 | 1 | 4096 | 14336 | 92.2 | 76.7 | 9.0 | 1.20x |
| Llama iq1 | 32 | 14336 | 4096 | 151.1 | 90.3 | 28.5 | 1.67x |
| Llama q4 | 32 | 14336 | 4096 | 117.5 | 97.4 | 27.7 | 1.21x |
| Llama q6 | 128 | 14336 | 4096 | 154.7 | 135.1 | 81.0 | 1.15x |
| Llama iq1 | 512 | 4096 | 14336 | 587.6 | 457.2 | 215.6 | 1.29x |
| Llama iq1 | 2048 | 1024 | 4096 | 199.2 | 149.5 | 79.1 | 1.33x |
| Tail iq1 | 37 | 75 | 512 | 42.0 | 15.6 | 18.0 | 2.70x |
| Qwen head q6 | 1 | 248320 | 2048 | 535.4 | 535.5 | 276.3 | 1.00x |
| Llama q4 | 64 | 4096 | 14336 | 115.1 | 115.1 | 40.2 | 1.00x |

The remaining worst model-shape ratio in the sweep is 8.51x
cuDNN/llama.cpp (q4_K, M=1, N=4096, K=14336).
These improvements do not establish performance parity. They are synthetic
fixtures at Llama3.1-8B and Qwen3.6-35B-A3B dimensions, not end-to-end models,
real routed MoE, or tokens/s measurements.

The comparison protocol remains 11 alternating cuDNN/llama.cpp rounds of
20 dependent CUDA-graph operations, with three warmup replays per timed side.
Both cuDNN split-K phases and llama.cpp activation quantization/fixup are
included; compilation, allocation and reference calculations are excluded.
Caches are warm/reused, clocks unlocked, and no other timed job shares each GPU.
llama.cpp remains pinned to `08246a28f6000100433d297c4e037c02e9d2d464`.
cuDNN uses FP16 decoded weights/activations and FP32 accumulation; llama.cpp
MMQ uses Q8_1 activation quantization and INT8 MMA. Every packed weight is
checked against ggml's CPU decoder. Full outputs pass finiteness and NRMSE
checks against FP64 unrounded-weight GEMM; 256 sampled outputs additionally
pass the FP16-weight contract at atol=rtol=0.005. Numerical metrics accompany
every timing. These checks do not measure model-level accuracy.

Host planner boundaries, full backend integration, frontend C++ samples and
45 Python tests pass. New backend tests cover IQ1_S auxiliary offsets 4/8/12,
unaligned packed bases and partial grids in FP16/BF16. C++ frontend samples
also bind offset codebooks. Existing identity cases check every decoded
weight and all 2048 grid indices. Compute Sanitizer reports zero memcheck
errors for ggml/pipeline cases, zero pipeline race hazards/warnings, and zero
ggml synchronization errors.

Actual compiled backend kernels, with the runtime's dynamic-SMEM opt-in:

| Kernel / example | Registers/thread | Operand SMEM | Theoretical CTAs/SM |
| --- | ---: | ---: | ---: |
| Split M16 IQ1_S | 40 | 11,776 B | 6 |
| Split M32 IQ1_S | 40 | 14,336 B | 6 |
| Grouped M256 IQ1_S | 122 | 50,176 B | 2 |
| Split M32 Q4_K | 40 | 14,336 B | 6 |
| Split M128 Q6_K | 64 | 29,696 B | 3 |

All five inspected kernels have zero local/stack storage and no LDL, STL or
CALL instructions. The IQ1_S change adds no shared-memory allocation. CUDA
also reserves 1 KiB per CTA. Resource use depends on customer code; this is
not a spill-free guarantee for arbitrary decoders. Hardware performance
counters remain unavailable (`ERR_NVGPUCTRPERM`), so stall/bandwidth attribution
is limited to controlled experiments and binary inspection.

Reproduction and complete evidence: `/tmp/cudnn-dequant-upgrade5` on ultra,
including `README.txt`, `matrix/comparison.csv`,
`matrix/2026-10-08-cudnn-dequant-all-iq1-vector.xlsx`, `controls/summary.json`,
source snapshots and sanitizer/resource logs. For example:
`bash /tmp/cudnn-dequant-upgrade5/run-compare.sh 2 14336 4096 iq1`.
Other GPUs and Windows were not measured in this increment. The reviews
remain draft prototypes; engine-owned staging is still milestone (b).

## Q4_K vector metadata and packed transpose (2026-10-09)

This increment optimizes the supplied customer Q4_K decoder. The previous
pushed baseline is backend `561d0b26e` / frontend `eead6319`. The FORT mainloop,
planner, public API and ABI 1 contract are unchanged. Use the revised example
source when building a new plan; existing saved binaries keep their old code.

For a 16-byte-aligned Q4_K block, one 16-byte global load reads both FP16
superblock coefficients and all twelve local scale/minimum bytes. The decoder
extracts the selected group's six-bit scale and minimum from those registers.
An aligned eight-byte code run uses one 64-bit load; four-byte-aligned runs use
two word loads, and other addresses use byte loads. Metadata has its original
byte-load fallback. All reads stay inside the original complete 144-byte block,
including logical K tails; the program still promises only byte alignment.

The Q4_K register transpose now keeps two decoded halfwords per 32-bit register.
Four byte permutations and eight warp shuffles replace twelve shuffles plus
halfword selections. This rearranges FP16/BF16 bit patterns without changing
numerical values. Every warp lane participates, including invalid tail lanes.
IQ1_S keeps its previous transpose: applying this change there regressed one
large-M case in the broader sweep and was discarded. Q6_K is unchanged.
No additional shared memory, decoder scratch or execution workspace is added.

All 198 supplied spreadsheet cases pass on ultra's RTX PRO 6000 Blackwell
Server GPUs (SM120, 188 SMs, CUDA/NVRTC 13.4). No final sweep case regressed by
more than 10% against the previous local sweep. Controls below are medians of
three independent processes per version, alternating old/new order on the
same GPU. Times are microseconds:

| Format / shape | M | N | K | Previous us | Updated us | llama.cpp us | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama q4 | 1 | 4096 | 14336 | 85.6 | 54.4 | 10.0 | 1.57x |
| Llama q4 | 1 | 14336 | 4096 | 85.8 | 56.0 | 11.9 | 1.53x |
| Llama q4 | 32 | 14336 | 4096 | 97.4 | 67.6 | 27.7 | 1.44x |
| Llama q4 | 512 | 14336 | 4096 | 450.4 | 385.5 | 210.0 | 1.17x |
| Llama q4 | 512 | 4096 | 14336 | 480.8 | 390.4 | 212.9 | 1.23x |
| Llama q4 | 2048 | 1024 | 4096 | 160.1 | 122.2 | 78.6 | 1.31x |
| Tail q4 | 37 | 75 | 512 | 14.4 | 14.3 | 17.8 | 1.01x |
| Llama iq1 | 2 | 14336 | 4096 | 78.5 | 78.5 | 9.6 | 1.00x |
| Llama iq1 | 512 | 14336 | 4096 | 435.4 | 435.4 | 213.9 | 1.00x |
| Qwen head q6 | 1 | 248320 | 2048 | 535.4 | 535.4 | 276.2 | 1.00x |

The worst remaining model-shape ratio is 8.50x cuDNN/llama.cpp
(iq1_s, M=1, N=4096, K=14336). The gains are specific
to the updated example source, and performance parity is still not reached.
These are synthetic fixtures at Llama3.1-8B and Qwen3.6-35B-A3B dimensions,
not end-to-end model, real routed-MoE or tokens/s measurements.

The comparison uses 11 alternating cuDNN/llama.cpp rounds of 20 dependent
CUDA-graph operations, with three warmup replays per timed side. Both cuDNN
phases and llama.cpp quantization/fixup are included; compilation/allocation/
reference work is excluded. Inputs/caches are reused and clocks are unlocked.
Timing jobs do not overlap on a GPU. llama.cpp is pinned to
`08246a28f6000100433d297c4e037c02e9d2d464`.

cuDNN timings use FP16 decoded weights and activations with FP32 accumulation;
BF16 is also supported and correctness-tested. The production llama.cpp graph
selects MMVQ (`mul_mat_vec_q`) for the measured M=1 cases, with Q8_1 activation
quantization and integer dot products (DP4A for Q4_K). Larger cases select MMQ
with INT8 MMA. The baseline therefore is not exclusively an MMQ comparison.
Kernel names for every case are retained in the logs. No activation
quantization or integer arithmetic has been added to the cuDNN contract.

Every packed weight matches ggml's CPU decoder. Full outputs pass finiteness
and NRMSE checks against FP64 unrounded-weight GEMM; 256 sampled outputs also
pass the FP16-rounded-weight contract at atol=rtol=0.005. These checks do not
establish model-level accuracy. Error metrics accompany every timing.

Full backend integration, frontend C++ samples (1,734,318 assertions in two
cases) and all 45 Python tests pass. Existing identity GEMMs check every
rounded Q4_K weight and all eight metadata groups, in FP16 and BF16. New
packed-base offsets 0/1/2/3/4/8/12/15/16 exercise every vector/word/byte branch
and exact allocation ends with a partial logical block. Compute Sanitizer
reports zero ggml/pipeline memcheck errors, zero pipeline race hazards or
warnings, and zero ggml synchronization errors.

Actual backend CUBIN resources, with the runtime's dynamic-SMEM opt-in:

| Kernel / example | Registers/thread | Operand SMEM | Theoretical CTAs/SM |
| --- | ---: | ---: | ---: |
| Split M16 Q4_K | 38 | 11,776 B | 6 |
| Split M32 Q4_K | 40 | 14,336 B | 6 |
| Grouped M256 Q4_K | 124 | 50,176 B | 2 |
| Split M16 IQ1_S | 40 | 11,776 B | 6 |
| Grouped M256 IQ1_S | 122 | 50,176 B | 2 |

All five inspected kernels have zero local/stack bytes and no LDL, STL or CALL
instructions. CUDA also reserves 1 KiB per CTA. Arbitrary customer decoders can
have different resource use. Hardware counters remain unavailable
(`ERR_NVGPUCTRPERM`); diagnosis uses controlled experiments and binary inspection.

Additional investigation, kept outside the implementation: for Q4_K
M=1/N=4096/K=14336, the same mainloop with a zero-output decoder takes about
19 us, while a transposing FP16-passthrough decoder takes about 75 us. These
are diagnostic workloads with different memory traffic, not additive stall
measurements. Grouping more K tiles did not provide a broad win because the
larger operand allocation reduced residency.

A scratch GEMV using a decoder that returns thread-local contiguous values
reached about 23-25 us on the two large M=1 Q4_K FFN shapes, using 32 values
per thread and FP32 FMA. It rounds decoded weights to FP16 and does not
quantize activations. Longer vectors reuse metadata and avoid the shared-tile
exchange; at M=8 and M=16 scalar FMA loses its advantage. This is an exploratory
standalone result, not the reviewed cuDNN implementation: it requires a
separate opt-in callback contract and has only selected-shape FP16/cuBLAS
validation. It has not passed the full backend/frontend/BF16/lifecycle suite
or the three-process llama.cpp comparison. No register-output ABI is exposed.
The evidence supports a dedicated very-small-M path as the next investigation.

Complete evidence and reproduction are in `/tmp/cudnn-dequant-upgrade6` on ultra:
`README.txt`, `matrix/comparison.csv`,
`matrix/2026-10-08-cudnn-dequant-all-q4-vector.xlsx`, `controls/summary.json`,
final source snapshots, CUBINs and sanitizer logs. For example:
`bash /tmp/cudnn-dequant-upgrade6/run-compare.sh 1 4096 14336 q4`.
Other GPUs and Windows were not measured. Engine-owned packed staging remains
milestone (b), and both reviews remain draft prototypes.
