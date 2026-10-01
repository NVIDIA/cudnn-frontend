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
| `abi_version` | `int64_t` | 1; use 2 for managed loads |
| `tile_shape` | `std::vector<int64_t>` | `[32,64]`; the current engine supports only this tile |
| `cta_smem_bytes` | `int64_t` | 0; fixed scratch bytes per CTA |
| `stage_smem_bytes` | `int64_t` | 0; scratch bytes per pipeline stage |
| `input_alignment` | `int64_t` | 16; power of two, at most 256, promised for every physical input |
| `constants` | `std::vector<int64_t>` | Empty; at most 64 compile-time values, in supplied order |
| `load_mode` | `int64_t` | 0 (DECODER); ABI 2 supports 1 (TMA_BULK), 2 (VECTOR_256) |
| `storage_bits` | `int64_t` | 0; ABI 2 requires 2, 4, or 8 |
| `row_stride_bytes` | `int64_t` | 0; ABI 2 requires padded rows as specified below |

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
For ABI 1, the engine computes stages from the effective shared-memory budget
as follows; ABI 2 additionally reserves managed staging/barriers described below:

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

## Optional engine-managed packed loads (ABI 2)

Physical transport and numerical interpretation are separate contracts. A regular
row of codes can represent signed integers, a custom floating-point type, or
indices into a nonlinear codebook, with arbitrary runtime scaling in each case.
The engine only moves bits; the customer's CUDA C++ still performs dequantization.
ABI 1 remains the default for irregular layouts, bit planes, non-byte-aligned
rows, or metadata-dependent addressing.

| `load_mode` | ABI | Input passed to customer entry | Engine operation |
| --- | --- | --- | --- |
| `0` / `DECODER` | 1 | Original global storage pointer | Customer owns weight loads |
| `1` / `TMA_BULK` | 2 | Ready immutable shared tile | `cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes` per valid K row |
| `2` / `VECTOR_256` | 2 | Per-thread, by-value eight-word fragment | Naturally aligned `ld.global.v8.b32` |

The TMA mode uses linear bulk copies, not a multidimensional tensor-map descriptor
or a swizzled tile. This keeps pointers as ordinary variant-pack bindings and
requires no host/device descriptor allocation or extra setup kernel. Both managed
modes retain the existing SM120-only support envelope. VECTOR_256 additionally
requires NVRTC 12.9 or newer (PTX 8.8); TMA_BULK retains the NVRTC 12.8 minimum.
These are explicit choices, not an autotuning policy or a guaranteed speedup.

### Physical layout and rejection checks

Set `abi_version=2`, `load_mode=1` or `2`, `storage_bits` to 2, 4, or 8,
`row_stride_bytes` to the physical row stride, and `input_alignment>=32`.
All three new fields are INT64 backend attributes:
`CUDNN_ATTR_WEIGHT_DECODE_LOAD_MODE`, `CUDNN_ATTR_WEIGHT_DECODE_STORAGE_BITS`,
and `CUDNN_ATTR_WEIGHT_DECODE_ROW_STRIDE_BYTES`. C++ uses the corresponding fluent
setters on `Weight_dequantize_program`; Python uses matching keyword arguments.
The C++ program also exposes `DECODER`, `TMA_BULK`, and `VECTOR_256` constants.

Logical weight `(k,n)` starts at bit
`8*k*row_stride_bytes + n*storage_bits` in the physical byte vector. Codes are
low-bit-first, contiguous along N, and each K row starts at a byte boundary.
No bit offset, transpose, interleave, swizzle, pointer indirection, or repacking is
implicit. For logical `[1,K,N]`, the contract requires:

- Physical storage is a nonvirtual contiguous INT8/UINT8 `[1,1,length]` vector.
- `row_stride_bytes` is positive, at most INT32_MAX, and a multiple of 32.
- `row_stride_bytes >= round_up(ceil(N/64) * (64*storage_bits/8), 32)`.
- `length >= K*row_stride_bytes`, including the final row's padding.
- `input_alignment` is a power of two in `[32,256]` and promises alignment of
  **every physical input**, including auxiliary tensors. The frontend propagates
  this promise into tensor descriptors; the backend verifies actual addresses.
- ABI 1 requires all three managed-load fields to be zero. ABI 2 requires a
  supported managed mode. Unknown modes/widths and incompatible ABIs are rejected.

Padding bytes must be allocated and readable, but need not encode numerical zero.
FORT masks decoded K/N tails to zero. A managed callback must still respect valid
logical coordinates and its assigned fragment when looking up scales or writing
output. FORT validates declared extents and pointer alignment; the caller remains
responsible for providing allocations of those extents and valid auxiliary data.
The engine declines unsupported layouts instead of silently allocating/repacking.

For example, N=75 with four-bit codes needs 64 physical bytes per K row in this
contract. ABI 1's tightly packed stream needs only 38 bytes for the first 75 codes
(and can continue at a half-byte boundary); it is a different physical layout.
Customers must select a contract matching the storage they actually supply.

### Device ABI and ownership

The backend supplies these types; do not redeclare them in customer source:

```cpp
struct FortWeightDecodeTileV2 {
    int abi_version, thread_id, thread_count, auxiliary_count;
    long long batch, k_begin, n_begin, full_k, full_n;
    int tile_k, tile_n, valid_k, valid_n, output_stride;
    int output_bfloat16;
    const long long* constants;
    int constant_count;
    int storage_bits;
};
struct FortWeightDecodeSharedTileV2 {
    const unsigned char* data;
    int row_stride_bytes;
};
struct FortWeightDecodeFragmentV2 {
    unsigned words[8];
    int k, n, code_offset, value_count;
};

// TMA_BULK entry:
__device__ void decode(const FortWeightDecodeTileV2& tile,
    FortWeightDecodeSharedTileV2 input, const void* const* auxiliary,
    void* cta_scratch, void* stage_scratch, FortWeightDecodeValue* output);
// VECTOR_256 entry (a separate signature; an overload is also permitted):
__device__ void decode(const FortWeightDecodeTileV2& tile,
    FortWeightDecodeFragmentV2 input, const void* const* auxiliary,
    void* cta_scratch, void* stage_scratch, FortWeightDecodeValue* output);
```

Common context fields retain their ABI 1 meanings, with `abi_version=2`.
All 256 CTA threads call the entry uniformly for each nonempty K tile, so uniform
CTA barriers inside the decoder are permitted. An all-out-of-range K tile skips
the callback and produces zero B. All pointers, including context/constants,
auxiliary arrays, shared input, scratch, and output, are borrowed for this call;
do not retain them. Ordered auxiliary **global base pointers** and compile-time
constants are unchanged. Neither managed entry receives the original packed
weight base pointer.

For TMA_BULK, `data` points to a 32x64-code tile, without swizzling. Its local row
stride is `64*storage_bits/8` bytes, independently of the global row stride.
Missing K rows are zero-filled; the N tail contains allocated physical padding.
FORT has completed the bulk transfers and the acquire/proxy synchronization
before invoking the callback. The input is read-only and remains valid until the
callback returns. The engine owns barrier initialization, expected transaction
counts, completion waits, invalidation, and stage reuse. Each barrier is initialized
once per CTA, alternates phases when its stage is reused, and is invalidated only
after the final use.

For VECTOR_256, `(input.k,input.n)` gives the first owned logical position **within
this 32x64 tile**. `value_count` is the number of valid consecutive N values owned
by this thread. Each owned code begins at bit
`(input.code_offset+i)*tile.storage_bits` of `words`, for `0<=i<value_count`.
The words are in increasing byte-address order. An inactive thread receives a
zero count (and zero words); its coordinates must not be dereferenced. Ownership
is disjoint: each valid output value belongs to exactly one thread.

Eight-bit fragments own at most 32 values; four-bit fragments own at most 64.
A two-bit 256-bit load spans 128 codes, so a 64-column tile owns either the first
or second half (`code_offset=0` or `64`). Adjacent tiles may read the same 32-byte
payload; this deliberate tradeoff preserves natural alignment without forcing a
shared-memory round trip. The fixed-size input is passed by value and is available
for compiler inlining/register scalarization; arbitrary decoder code may still
cause register pressure or spills. Customer code writes its decoded FP16/BF16
values to the supplied dense shared-memory output, as in ABI 1.

### Resource budget, synchronization, and performance limits

Let `align256(x)` round a byte count up to 256. Resource planning uses:

```
fixed = align256(cta_smem_bytes)
packed = TMA_BULK ? 32*64*storage_bits/8 : 0
barrier = TMA_BULK ? 256 : 0
stage = 6144 + packed + barrier + align256(stage_smem_bytes)
stages = min(8, floor((effective_shared_budget - fixed) / stage))
```

At least two stages must fit. The 6144 bytes hold dense FP16/BF16 A and B tiles.
Each TMA stage reserves its packed tile and a separately aligned region containing
an eight-byte mbarrier, ahead of user stage scratch. User reservations retain
their exact meaning and cannot overlap engine state. VECTOR_256 reserves neither
packed shared storage nor a TMA barrier. CTA scratch is initialized once; user
stage scratch remains unspecified until the callback initializes it.

FORT owns asynchronous A-copy groups, TMA barriers, publication of B, and stage
reuse. Customer callbacks must finish their work synchronously and must not issue
asynchronous operations against those groups, modify engine-owned input/barriers,
or preserve borrowed pointers. The dense B tail is masked **after numerical
conversion**, because a zero code may represent a nonzero weight.

This prototype waits for managed input before calling the cooperative decoder.
It does not add independent producer warps or pipeline TMA conversion concurrently
with MMA. TMA setup/wait overhead and extra shared memory can reduce performance
or stage count; register fragments reduce load instruction count but concentrate
decode work in fewer threads and can increase register pressure. The two-bit
vector path can reread payloads across neighboring tiles. Benchmark a customer's
actual format and shape before selecting a mode; no universal performance ranking
is claimed.

Source is still compiled with the GEMM in one NVRTC translation unit. These are
versioned **source/device-call contracts**, not a promise of a stable binary ABI
for arbitrary separately compiled objects. Runtime linking/LTO remains future
work and would need a separately distributed, versioned ABI header and toolchain
compatibility rules.

### Frontend use and persistence

```cpp
using Program = cudnn_frontend::graph::Weight_dequantize_program;
auto program = Program().set_source(managed_source).set_entry("decode")
    .set_abi_version(2).set_load_mode(Program::TMA_BULK)
    .set_storage_bits(4).set_row_stride_bytes(row_stride_bytes)
    .set_input_alignment(32).set_constants({4, 3, 24, 20});
```

```python
B = graph.weight_dequantize(
    W, source=managed_source, entry="decode", auxiliaries=[S, G],
    abi_version=2, load_mode=1, storage_bits=4,
    row_stride_bytes=row_stride_bytes, input_alignment=32,
    constants=[4, 3, 24, 20], out_dims=[1, K, N],
)
```

Use mode 2 and the fragment entry for VECTOR_256. The numerical/scaling logic can
be shared between both entry overloads. The executable samples show both.

The header feature marker is now `CUDNN_WEIGHT_DECODE_ABI_VERSION=2` (maximum
supported ABI); it does not change the default ABI. New frontend arguments are
appended, preserving positional calls. All load metadata is owned, serialized,
and included in graph/plan/cache identity. Old JSON without these fields defaults
to decoder-owned ABI 1. New headers do not send new attributes for ABI 1, allowing
older prototype backends to continue supporting their original contract. ABI 2
lowering requires matching backend capabilities and never fabricates enum values.
Plan reload, variant-pack rebinding, and CUDA graph capture use the ordinary
kernel launch path with no extra allocation, host synchronization, or kernel.

### Managed-load test coverage

The executable C++ sample tests 54 combinations: three load modes, three code
widths, three scaling schemes, and two operand types. Python adds invalid physical
layouts, ABI mismatches, actual pointer-alignment rejection, plan serialization,
and CUDA graph capture for each load mode. C++ contract tests check the new
metadata's graph identity, copied ownership, old-JSON defaults, and unsupported
header behavior. The backend integration suite additionally tests stage reuse,
nonlinear codebooks, poisoned padding, and scratch/barrier separation under
Compute Sanitizer. These establish correctness for the examples, not performance
for arbitrary customer formats.
