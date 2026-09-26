# Custom weight dequantization (experimental)

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
| `cta_smem_bytes` | `int64_t` | 0; fixed scratch bytes per CTA |
| `stage_smem_bytes` | `int64_t` | 0; scratch bytes per pipeline stage |
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
graph.execute({A: a_gpu, W: storage_gpu, S: block_scales_gpu,
               G: global_scale_gpu, C: c_gpu}, None)
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

CTA scratch is 256-byte aligned, initialized to zero once per CTA, and persists
for the CTA. Stage scratch is independently aligned, initially unspecified, and
lives until its stage is consumed; initialize it before reading it. Zero-sized
scratch pointers are null. The callback may write only its output and declared
scratch. It must complete synchronous memory operations before returning and
must not retain pointers, launch kernels, synchronize across CTAs, or use FORT's
`cp.async` commit/wait groups. Source is trusted device code; resource checks do
not prove its indexing or synchronization correct.

Scratch declarations are independently capped at 1 MiB by descriptor validation.
The engine then computes stages from the effective shared-memory budget:

```text
fixed     = align_up(cta_smem_bytes, 256)
per_stage = 2048 bytes A + 4096 bytes B + align_up(stage_smem_bytes, 256)
stages    = min(8, floor((available_shared_memory - fixed) / per_stage))
```

At least two stages must fit. Simple numerical conversions need no extra scratch.
More scratch can reduce the number of pipeline stages and therefore performance.

The current engine accepts exact SM120 and NVRTC >=12.8, one decode feeding B of
one static unbatched row-major GEMM, FP16/BF16 A and B, FP32 compute and same-type
or FP32 C. A's leading dimension and base must be 16-byte aligned. Physical
storage and auxiliary tensors must be contiguous `[1,1,length]` vectors; their
promised alignment must satisfy the program and their natural datatype
alignment. Output must be virtual. Batched/dynamic/override shapes, decode-A,
additional fusions, alternate tile shapes, split-K and Stream-K are rejected.
The backend remains authoritative for graph, resource, and launch-pointer checks.

The full source, entry, constants, resource declarations and ordered tensor
connections participate in structural serialization and graph identity; changing
them builds a different program. Runtime scale **values** are not part of that
identity. Plan serialization also retains the program. Treat serialized plans
containing customer device source with the same trust as the original source.

This frontend operation adds setup/lowering work only. It uses the existing
prototype's synchronous custom B producer and tensor-core GEMM; it does not
promise performance parity with built-in format-specific dequantization paths.
Measure each decoder and workload, including register use and occupancy.

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
