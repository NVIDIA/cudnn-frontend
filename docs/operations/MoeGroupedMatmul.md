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
