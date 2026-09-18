# MoE Grouped Matmul

## Overview

The MoE Grouped Matmul operation computes a grouped matrix multiplication across experts, as used in Mixture-of-Experts (MoE) layers. Each expert has its own weight matrix, and tokens are routed to experts via `first_token_offset`.

Three routing modes are supported:

**None mode** (tokens already routed per expert):

$$\text{Output}[1,\ S \times \text{topK},\ N] = \text{Token}[1,\ S \times \text{topK},\ K]\ \times\ \text{Weight}[E,\ K,\ N]$$

**Gather mode** (gather tokens from unrouted layout before matmul):

$$\text{Output}[1,\ S \times \text{topK},\ N] = \text{Token}[1,\ S,\ K]\ \times\ \text{Weight}[E,\ K,\ N]$$

**Scatter mode** (scatter output back to token order after matmul):

$$\text{Output}[1,\ S \times \text{topK},\ N] = \text{Token}[1,\ S \times \text{topK},\ K]\ \times\ \text{Weight}[E,\ K,\ N]$$

where $E$ = number of experts, $S$ = number of tokens, $K$ = hidden size, $N$ = output (weight) size.

### Tensor Roles by Mode

| Tensor | Shape | Modes |
|---|---|---|
| `Token` | `[1, S*topK, K]` (None/Scatter) or `[1, S, K]` (Gather) | All |
| `Weight` | `[E, K, N]` | All |
| `FirstTokenOffset` | `[B*E, 1, 1]` (B represents batch size), INT32 | All |
| `TokenIndex` | `[1, S*topK, 1]`, INT32 | Gather, Scatter |
| `TokenKs` | `[1, S*topK, 1]`, INT32 | Scatter only |
| `TopK` | scalar int32 | Scatter only |

## Support Matrix

The support matrix is based on the latest cuDNN backend.

| Operation | Minimum cuDNN | Minimum cublasLt | Datatype (I/O) | Compute type | Fusion pattern |
|---|---|---|---|---|---|
| MoE Grouped Matmul (forward) | 9.18.0 | — | int8, fp8, fp16, bf16, fp32, nvfp4/mxfp8 (9.21.0) | FLOAT | 9.21.0: SwiGLU, AMAX, etc. |
| MoE Grouped Matmul Bwd | 9.22.0 | 13.5 | fp16, bf16 | FLOAT | - |

### Important Notes

1. `FirstTokenOffset` contains `B * E` values with the total token count implicit from the token tensor dimension.
2. In **Scatter** mode, both `TokenIndex` and `TokenKs` are required, and `top_k` must be explicitly provided.
3. In **Gather** mode, `TokenIndex` is required.

---

## MoE Grouped Matmul Forward

### C++ API

```cpp
std::shared_ptr<Tensor_attributes>
moe_grouped_matmul(std::shared_ptr<Tensor_attributes> token,
                   std::shared_ptr<Tensor_attributes> weight,
                   std::shared_ptr<Tensor_attributes> first_token_offset,
                   std::shared_ptr<Tensor_attributes> token_index,   // optional, pass nullptr for None mode
                   std::shared_ptr<Tensor_attributes> token_ks,      // optional, pass nullptr unless Scatter mode
                   Moe_grouped_matmul_attributes options);
```

`Moe_grouped_matmul_attributes` is a lightweight structure with setters:

```cpp
Moe_grouped_matmul& set_name(std::string const&);

// Required: selects the routing mode
Moe_grouped_matmul& set_mode(MoeGroupedMatmulMode_t mode);
// MoeGroupedMatmulMode_t::NONE    — tokens already routed
// MoeGroupedMatmulMode_t::GATHER  — gather before matmul
// MoeGroupedMatmulMode_t::SCATTER — scatter after matmul

// Required for SCATTER mode
Moe_grouped_matmul& set_top_k(int32_t top_k_value);

Moe_grouped_matmul& set_compute_data_type(DataType_t value);
```

### Python API

#### Low-level graph API (`cudnn.pygraph`)

```python
output = graph.moe_grouped_matmul(
    token,                                # Token tensor
    weight,                               # Weight tensor
    first_token_offset,                   # Expert routing offsets
    token_index=None,                     # Required for Gather/Scatter modes
    token_ks=None,                        # Required for Scatter mode
    mode=cudnn.moe_grouped_matmul_mode.NONE,  # NONE, GATHER, or SCATTER
    top_k=1,                              # Top-k value; required for Scatter mode
    compute_data_type=cudnn.data_type.NOT_SET,
    name=None,
)
```

**Args:**
- `token` (cudnn_tensor): Token data.
  - None/Scatter mode: shape `(1, S*topK, K)`
  - Gather mode: shape `(1, S, K)`
- `weight` (cudnn_tensor): Expert weight data with shape `(E, K, N)`.
- `first_token_offset` (cudnn_tensor): INT32 tensor of shape `(B*E, 1, 1)`. The $i$-th entry is the index of the first token assigned to expert $i$.
- `token_index` (Optional[cudnn_tensor]): INT32 tensor of shape `(1, S*topK, 1)`. Maps each routed slot to a source token index. Required for Gather and Scatter modes.
- `token_ks` (Optional[cudnn_tensor]): INT32 tensor of shape `(1, S*topK, 1)`. The expert index for each routed token. Required for Scatter mode.
- `mode` (cudnn.moe_grouped_matmul_mode): Routing mode — `NONE`, `GATHER`, or `SCATTER`.
- `top_k` (int): Top-k routing value. Must be provided for Scatter mode.
- `compute_data_type` (Optional[cudnn.data_type]): Data type for internal computation. Defaults to FLOAT.
- `name` (Optional[str]): Name for the operation.

**Returns:**
- `output` (cudnn_tensor): Output tensor of shape `(1, M_out, N)`, where `M_out = token_index.shape[1]` for Gather mode, otherwise `token.shape[1]`.

#### High-level experimental API

```python
from cudnn.experimental.ops import moe_grouped_matmul

output = moe_grouped_matmul(
    token,              # (1, M, K) torch.Tensor, fp16 or bf16
    weight,             # (E, K, N) torch.Tensor, column-major inner dims
    first_token_offset, # (B*E, 1, 1) torch.Tensor, INT32
    token_index=None,   # (1, S*topK, 1) INT32; required for gather/scatter
    token_ks=None,      # (1, S*topK, 1) INT32; required for scatter
    mode="none",        # "none", "gather", or "scatter"
    top_k=1,            # required for scatter mode
)
```

The high-level API handles cuDNN handle management and graph caching automatically. cuDNN graphs are built once per unique (shape, dtype, mode, top_k) configuration and reused across subsequent calls.

### Configurable Options

- **Mode** (`mode`): Controls how tokens are routed to and from expert weight matrices.
  - `NONE`: Tokens are already ordered by expert (pre-routed). Direct grouped matmul with no reordering.
  - `GATHER`: Tokens are in original (un-routed) order. `TokenIndex` specifies which source token each expert slot reads from.
  - `SCATTER`: Tokens are pre-routed, but the output is scattered back to the original token order. Requires both `TokenIndex` and `TokenKs`.

- **TopK** (`top_k`): The number of experts each token is routed to. Required in Scatter mode for the scatter-back computation.

- **Compute data type** (`compute_data_type`): Sets the precision for internal accumulation. Defaults to FLOAT for fp16/bf16 I/O.

### Example (Python)

```python
import torch
import cudnn

num_experts = 8
token_num   = 1024   # S
hidden_size = 512    # K
weight_size = 256    # N

graph = cudnn.pygraph(
    intermediate_data_type=cudnn.data_type.FLOAT,
    compute_data_type=cudnn.data_type.FLOAT,
)

# Token: [1, S, K], bf16, row-major
token_t = graph.tensor(
    name="token",
    dim=[1, token_num, hidden_size],
    stride=[token_num * hidden_size, hidden_size, 1],
    data_type=cudnn.data_type.BFLOAT16,
)

# Weight: [E, K, N], bf16, column-major inner dims (stride[1] == 1)
weight_t = graph.tensor(
    name="weight",
    dim=[num_experts, hidden_size, weight_size],
    stride=[hidden_size * weight_size, 1, hidden_size],
    data_type=cudnn.data_type.BFLOAT16,
)

# FirstTokenOffset: [E, 1, 1], INT32
fto_t = graph.tensor(
    name="first_token_offset",
    dim=[num_experts, 1, 1],
    stride=[1, 1, 1],
    data_type=cudnn.data_type.INT32,
)

output_t = graph.moe_grouped_matmul(
    token_t, weight_t, fto_t,
    mode=cudnn.moe_grouped_matmul_mode.NONE,
    compute_data_type=cudnn.data_type.FLOAT,
    name="moe_fwd",
)
output_t.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)

graph.validate()
graph.build_operation_graph()
graph.create_execution_plans([cudnn.heur_mode.A])
graph.check_support()
graph.build_plans()
```

---

### Frost forward scheduling

The experimental open-source Frost GEMM engine exposes a separate performance
choice for grouped forward plans using `mode=NONE`. SM100 supports ordinary
(non-block-scaled) plans; SM120 supports ordinary and block-scaled plans:

- Omitted `cudnn.knob_type.SCHED_POLICY`, or value `0`: dynamically claimed
  cluster tickets, the unchanged default.
- Value `1`: static strided cluster tickets. Each cluster derives its next
  tile without a global scheduler counter or its reset launch.

Enable the Frost engine before constructing the graph, for example with
`CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1`. Starting from a supported Frost plan's
complete tile record, create a separate plan for the alternate policy:

```python
engine_id, tile_knobs = graph.get_engine_and_knobs_at_index(frost_plan_index)
static_knobs = dict(tile_knobs)
static_knobs[cudnn.knob_type.SCHED_POLICY] = 1
graph.create_execution_plan(engine_id, static_knobs)
```

Run the normal support/build steps before execution, and benchmark both policies
on the caller's shapes and routing distributions. The policy does not change tile
geometry or numerical semantics, and the default proposal does not automatically
select static scheduling. The public engine/knob record replays the chosen
policy. Static policy requests for non-MoE graphs, or SM100 block-scaled graphs,
are declined. SM120 uses one CTA per cluster. This policy is not passed to
closed-source cuDNN engines.

### Frost paired SwiGLU for small SM100 groups

The experimental `frost_moe_swiglu_pair` engine (20401) serves two grouped
BF16 projections followed by FP32 SwiGLU and a BF16 output. Declare the
weights as two `graph.slice` views of one `[E, K, 2*N]` parent, with the gate
half first and the up half second. The parent is the external variant-pack
input; the slices are virtual graph values. No weight repack is performed
by plan execution.

This specialization supports SM100, 1 through 513 total routed rows, and
positive `N` and `K` divisible by 64. Tokens and output must be compact;
weight row/expert strides must be positive and 16-byte aligned. Expert starts
are contiguous INT32 metadata with the existing monotone in-range MoE contract.
Only standard unit-beta SwiGLU is supported: no auxiliary outputs, reductions,
quantization, dynamic shapes, or additional pointwise operations.

The engine exposes the existing public GEMM knob vocabulary. Its implemented
record has `TILE_M=128`, `TILE_N=8`, `TILEK=128` bytes,
`MMA_TILE_M=128`, `MMA_TILE_N=8`, `MMA_TILE_K=32` bytes,
one CTA, `SWAP_AB=1` and `SCHED_POLICY=1`. The physical M tile includes both
projections and produces 64 output features. Other knob combinations decline.
The ordinary Frost engine (20400) remains available. Heuristic enumeration
is not a performance ranking; tune the eligible complete MoE configurations.

Build the execution plan before capture and provide its queried workspace.
Execution binds the current parent pointer and caller stream without GPU
allocation, synchronization, device reads, or compilation. Retained CUDA
Graphs may bind distinct parent allocations to the same compiled plan.

The implementation incorporates Kernel Factory's small-token candidate and
an independently validated early-PDL change. It builds on Yanqin Zhai's
[SM100 swap-AB implementation](https://github.com/NVIDIA/cudnn-frontend/pull/1090),
NVIDIA CUTLASS example 113 layout concepts, and canonical rank-5 weight pairing.
TRT-LLM gated-row interleaving informed the exploration; no TRT-LLM kernel body
is copied. The row-range extension reuses the generic persistent kernel. For R<=8,
the plan selects the validated smaller scheduler path; R9–513 retains the
generic persistent scheduler.

### Frost SM120 shared-input fusion

For ordinary forward grouped matmul, the experimental Frost SM120 engine can
fuse a pair of BF16 or FP16 GEMMs that share the token tensor and use separate
K-major weights, followed by a pointwise epilogue such as gated activation.
The pair uses one token load per tile, separate FP32 accumulators, and the
graph's existing epilogue expression. Both scheduler policies above apply.

The weights may be views into one packed allocation with a larger expert
stride; execution uses the declared strides without repacking. All input
pointer and TMA stride alignment requirements still apply. The selected tile
must fit both weight tiles and epilogue staging in shared memory. Graphs with
more GEMMs, separate token operands, block scales or cross-row reductions and
quantization are declined by this fusion path. Supported single-GEMM paths
retain their existing contracts.

### Paired BF16 output projection on SM100

The opt-in `frost_moe_fc2_pair` engine (`20402`) serves a single unfused
`moe_grouped_matmul` with BF16 inputs/output and FP32 accumulation. It accepts
1–513 routed rows, output widths divisible by 128, and reduction dimensions
divisible by 64. Expert weights use the ordinary `(E,K,N)` graph declaration,
with contiguous K and nonoverlapping row/expert strides aligned to 16 bytes;
no pre-shuffle or weight repacking is required. Tokens/output must be compact,
and device-resident int32 offsets describe one group per expert. Dynamic graph
dimensions, operand transforms, and epilogues are declined.

Use the same public M128/N8/K32-byte, single-CTA, swap-AB, static-scheduler knobs
as engine `20401`. Engine `20402` writes both channel halves directly, without
SwiGLU; the existing engine `20401` retains its SwiGLU contract. These plans
are tuning candidates, not a performance ranking. Compilation occurs during
plan build; execution consumes caller-owned workspace and the supplied stream,
without allocation, conversion, or synchronization.

The row-range extension reuses the existing persistent kernel, including
multiple token tiles per expert and multiple waves of work. Coverage includes
empty and skewed groups, expert indices beyond the first warp, pitched weights,
retained graph captures, and changing inputs/weights/offsets. It changes engine
eligibility, not the kernel mathematics or the meaning of the public knobs.
At BF16 E128/top8/H2048/I768 on a 1000W B200, the measured complete FlashInfer
MoE benefit is about11% at T8 and under0.5% at T64; tune the complete eligible
engine combinations for the intended workload. See `FROST_MOE_HANDOFF.md` for
exact measurements and validation boundaries.

## MoE Grouped Matmul Backward

The backward operation computes the weight gradient $d\text{Weight}$ given the upstream gradient $d\text{Output}$ and the forward token activations:

$$d\text{Weight}[E,\ K,\ N] = \text{Token}^T[1,\ S,\ K]\ \times\ d\text{Output}[1,\ S,\ N]$$

per expert, where the per-expert token slices are determined by `FirstTokenOffset`.

### C++ API

```cpp
std::shared_ptr<Tensor_attributes>
moe_grouped_matmul_bwd(std::shared_ptr<Tensor_attributes> doutput,
                       std::shared_ptr<Tensor_attributes> token,
                       std::shared_ptr<Tensor_attributes> first_token_offset,
                       Moe_grouped_matmul_bwd_attributes options);
```

`Moe_grouped_matmul_bwd_attributes` is a lightweight structure with setters:

```cpp
Moe_grouped_matmul_bwd& set_name(std::string const&);

Moe_grouped_matmul_bwd& set_compute_data_type(DataType_t value);
```

### Python API

#### Low-level graph API (`cudnn.pygraph`)

```python
dweight = graph.moe_grouped_matmul_bwd(
    doutput,                              # Upstream gradient tensor
    token,                                # Forward token activations
    first_token_offset,                   # Expert routing offsets
    compute_data_type=cudnn.data_type.NOT_SET,
    name=None,
)
```

**Args:**
- `doutput` (cudnn_tensor): Upstream gradient with shape `(1, S, N)`, same layout as the forward output.
- `token` (cudnn_tensor): Forward token activations with shape `(1, S, K)`.
- `first_token_offset` (cudnn_tensor): INT32 tensor of shape `(B*E, 1, 1)`, same as used in the forward pass.
- `compute_data_type` (Optional[cudnn.data_type]): Data type for internal accumulation.
- `name` (Optional[str]): Name for the operation.

**Returns:**
- `dweight` (cudnn_tensor): Weight gradient with shape `(E, K, N)`.

### Example (Python)

```python
import torch
import cudnn

num_experts = 8
token_num   = 1024
hidden_size = 512
weight_size = 256

graph = cudnn.pygraph(
    intermediate_data_type=cudnn.data_type.FLOAT,
    compute_data_type=cudnn.data_type.FLOAT,
)

# dOutput: [1, S, N], bf16, row-major
doutput_t = graph.tensor(
    name="doutput",
    dim=[1, token_num, weight_size],
    stride=[token_num * weight_size, weight_size, 1],
    data_type=cudnn.data_type.BFLOAT16,
)

# Token: [1, S, K], bf16, row-major
token_t = graph.tensor(
    name="token",
    dim=[1, token_num, hidden_size],
    stride=[token_num * hidden_size, hidden_size, 1],
    data_type=cudnn.data_type.BFLOAT16,
)

# FirstTokenOffset: [E, 1, 1], INT32
fto_t = graph.tensor(
    name="first_token_offset",
    dim=[num_experts, 1, 1],
    stride=[1, 1, 1],
    data_type=cudnn.data_type.INT32,
)

dweight_t = graph.moe_grouped_matmul_bwd(
    doutput_t, token_t, fto_t,
    compute_data_type=cudnn.data_type.FLOAT,
    name="moe_bwd",
)
# dweight: [E, K, N]
dweight_t.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)

graph.validate()
graph.build_operation_graph()
graph.create_execution_plans([cudnn.heur_mode.A])
graph.check_support()
graph.build_plans()
```


## Frost packed FP8 weights (draft)

The optional `weight_layout="blocked_128x128_v1"` attribute declares physical
`[E,N/128,K/128,128,128]` E4M3 weights. The inner strides are
`[K*128,16384,128,1]`; each expert pitch is at least `N*K` bytes and 16-byte
aligned. Omitting the attribute preserves the ordinary rank-3 weight contract.
Packing happens during preparation, and TMA addresses the declared layout and
expert pitch directly during execution.

This layout requires SM100, N/K divisible by 128, eligible one-CTA N64/128 K128
tactics, and power-of-two cluster M/N with product at most 16. Unsupported
declarations decline. The layout belongs to the graph; it is not a performance
knob. Replaying `(engine, knobs)` requires reconstructing the same graph layout.
Classic backend graph key/serialization rejects this Python-only declaration.
See [the integration handoff](../../FROST_MOE_HANDOFF.md) for validation boundaries
and the matching FlashInfer draft.
