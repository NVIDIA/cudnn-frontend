# Native Sparse Attention (NSA)

**This is an experimental API and subject to change.**

## Overview

The Native Sparse Attention (NSA) module implements the sparse attention mechanism described in [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/pdf/2502.11089). NSA provides high-performance sparse attention kernels optimized for Blackwell (SM100+) GPUs. The Selection, Compression, and Top-K components are implemented with CUTLASS/CUTE. Sliding Window Attention currently utilizes cuDNN backend.

NSA combines multiple attention strategies to efficiently process long sequences:

1. **Selection Attention**: Attends to dynamically selected important blocks across the full context
2. **Compression Attention**: Attends to compressed key-value representations for global context
3. **Sliding Window Attention**: Attends to a local sliding window for fine-grained local context
4. **Top-K Reduction**: Identifies the most important key-value blocks for selection attention

### Architecture

```
                                     Query (Q)
                                         |
            +----------------------------+---------------+
            |                            |               |
            v                            v               v
     +-------------+   +-------+   +-----------+   +-----------+
     | Compression |-->| Top-K |-->| Selection |   |  Sliding  |
     |  Attention  |   |       |   | Attention |   |  Window   |
     +-------------+   +-------+   +-----------+   +-----------+
            |                            |               |
            |  [O_comp]                  |  [O_sel]      |  [O_swa]
            |                            |               |
            +----------------------------+---------------+
                                |
                                v
                         Combine Outputs
                                |
                                v
                         Final Output (O)
```

Each component can be used independently or combined for the full NSA pipeline.

---

## Installation

Install the cuDNN Frontend package with the CuteDSL optional dependencies:

```bash
pip install nvidia-cudnn-frontend[cutedsl]
```

---

## API Usage

### NSA Namespace

All NSA components are accessible through the `NSA` namespace:

```python
from cudnn import NSA

# Access individual components
NSA.SelectionAttention         # Class API
NSA.selection_attention_wrapper  # High-level wrapper

NSA.CompressionAttention
NSA.compression_attention_wrapper

NSA.SlidingWindowAttention
NSA.sliding_window_attention_wrapper

NSA.TopKReduction
NSA.topk_reduction_wrapper
```

---

## Components

### 1. Selection Attention

Selection Attention performs attention on dynamically selected key-value blocks. Given pre-computed block indices (typically from Top-K Reduction), it efficiently attends only to the most relevant parts of the context.

#### Shapes

- **Inputs**
  - `Q` (Query): `(T, H_q, D)` where `T` is total sequence length, `H_q` is number of query heads, `D` is head dimension
  - `K` (Key): `(T, H_kv, D)` where `H_kv` is number of key-value heads
  - `V` (Value): `(T, H_kv, D_v)` where `D_v` is value dimension
  - `block_indices`: `(T, H_kv, K)` – indices of selected blocks for each query position
  - `block_counts`: `(T, H_kv)` – number of valid blocks per query position
  - `cum_seqlen_q`: `(batch_size + 1,)` – cumulative sequence lengths for queries
  - `cum_seqlen_k`: `(batch_size + 1,)` – cumulative sequence lengths for keys (must equal `cum_seqlen_q`)

- **Outputs**
  - `O` (Output): `(T, H_q, D_v)`
  - `L` (LogSumExp): `(T, H_q)`
  - `M` (Max): `(T, H_q)`

#### Equation

For each query position $q$ attending to selected blocks $\mathcal{B}_q$:

$$
O[q] = \sum_{b \in \mathcal{B}_q} \sum_{k \in \text{block}_b} \text{softmax}\left(\frac{Q[q] \cdot K[k]^T}{\sqrt{D}}\right) V[k]
$$

#### High-level Wrapper

```python
from cudnn import NSA

result = NSA.selection_attention_wrapper(
    q_tensor=q,
    k_tensor=k,
    v_tensor=v,
    block_indices_tensor=block_indices,
    block_counts_tensor=block_counts,
    cum_seqlen_q_tensor=cum_seqlen_q,
    cum_seqlen_k_tensor=cum_seqlen_k,
    block_size=64,
    scale_softmax=None,  # Defaults to 1/sqrt(head_dim)
    o_dtype=torch.bfloat16,
    acc_dtype=torch.float32,
    max_s_q=1024,
    max_s_k=1024,
    stream=None,
)
o, l, m = result
# Key access: result["o_tensor"], result["l_tensor"], result["m_tensor"]
```

#### Class API

```python
from cudnn import NSA
from cuda.bindings import driver as cuda

selection_attention = NSA.SelectionAttention(
    sample_q=q,
    sample_k=k,
    sample_v=v,
    sample_o=o,
    sample_l=l,
    sample_m=m,
    sample_block_indices=block_indices,
    sample_block_counts=block_counts,
    sample_cum_seqlen_q=cum_seqlen_q,
    sample_cum_seqlen_k=cum_seqlen_k,
    max_s_q=1024,
    max_s_k=1024,
    acc_dtype=torch.float32,
    block_size=64,
    scale_softmax=None,
)
assert selection_attention.check_support()
selection_attention.compile()
selection_attention.execute(
    q_tensor=q,
    k_tensor=k,
    v_tensor=v,
    o_tensor=o,
    l_tensor=l,
    m_tensor=m,
    block_indices_tensor=block_indices,
    block_counts_tensor=block_counts,
    cum_seqlen_q_tensor=cum_seqlen_q,
    cum_seqlen_k_tensor=cum_seqlen_k,
    scale_softmax=None,
    current_stream=stream,
)
```

#### Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `block_size` | `int` | Size of each attention block. Must be one of `{16, 32, 64}` | `64` |
| `scale_softmax` | `float \| None` | Softmax scaling factor | `1/sqrt(head_dim)` |
| `acc_dtype` | `torch.dtype` | Accumulator dtype. Must be `torch.float32` | `torch.float32` |
| `max_s_q` | `int` | Maximum sequence length for queries | Required for T,H,D |
| `max_s_k` | `int` | Maximum sequence length for keys | Required for T,H,D |

#### Constraints

- Input dtype must be `float16` or `bfloat16`
- `H_q` must be divisible by `H_kv` (supports GQA/MQA)
- Currently only supports `T,H,D` layout (variable-length batched sequences)
- `cum_seqlen_q` and `cum_seqlen_k` must be identical
- Requires SM90+ (Hopper or newer)

---

### 2. Compression Attention

Compression Attention performs attention over compressed key-value sequences. This is useful for maintaining a global view of the context with reduced memory and computation.

#### Shapes

- **Inputs**
  - `Q` (Query): `(B, H_q, S_q, D)` or `(T, H_q, D)`
  - `K` (Key): `(B, H_kv, S_kv, D)` or `(T_kv, H_kv, D)` – compressed KV sequence
  - `V` (Value): `(B, H_kv, S_kv, D_v)` or `(T_kv, H_kv, D_v)`
  - `cum_seqlen_q`: `(batch_size + 1,)` – cumulative sequence lengths for queries (T,H,D layout only)
  - `cum_seqlen_k`: `(batch_size + 1,)` – cumulative sequence lengths for compressed keys (T,H,D layout only)

- **Outputs**
  - `O` (Output): Same shape as `Q`
  - `LSE` (LogSumExp, optional): `(B, H_q, S_q)` or `(T, H_q)`

#### Equation

Standard scaled dot-product attention with compressed causal masking:

$$
O = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{D}} \cdot \alpha_q \cdot \alpha_k + \text{mask}\right) \cdot V \cdot \alpha_v \cdot \alpha_o^{-1}
$$

where $\alpha_q$, $\alpha_k$, $\alpha_v$, $\alpha_o$ are optional scaling factors for quantized inputs.

#### High-level Wrapper

```python
from cudnn import NSA

result = NSA.compression_attention_wrapper(
    q_tensor=q,
    k_tensor=k,
    v_tensor=v,
    cum_seqlen_q_tensor=cum_seqlen_q,  # Required for T,H,D layout
    cum_seqlen_k_tensor=cum_seqlen_k,  # Required for T,H,D layout
    enable_lse=False,
    o_dtype=torch.bfloat16,
    qk_acc_dtype=torch.float32,
    pv_acc_dtype=torch.float32,
    mma_tiler_mn=(128, 128),
    is_persistent=False,
    scale_q=1.0,
    scale_k=1.0,
    scale_v=1.0,
    inv_scale_o=1.0,
    scale_softmax=None,  # Defaults to 1/sqrt(head_dim)
    stream=None,
)
o, lse = result
# Key access: result["o_tensor"], result["lse_tensor"]
```

#### Class API

```python
from cudnn import NSA

comp_attn = NSA.CompressionAttention(
    sample_q=q,
    sample_k=k,
    sample_v=v,
    sample_o=o,
    sample_lse=lse,  # Optional, set to None for inference
    sample_cum_seqlen_q=cum_seqlen_q,
    sample_cum_seqlen_k=cum_seqlen_k,
    qk_acc_dtype=torch.float32,
    pv_acc_dtype=torch.float32,
    mma_tiler_mn=(128, 128),
    is_persistent=False,
    scale_q=1.0,
    scale_k=1.0,
    scale_v=1.0,
    inv_scale_o=1.0,
    scale_softmax=None,
)
assert comp_attn.check_support()
comp_attn.compile()
comp_attn.execute(
    q_tensor=q,
    k_tensor=k,
    v_tensor=v,
    o_tensor=o,
    lse_tensor=lse,
    cum_seqlen_q_tensor=cum_seqlen_q,
    cum_seqlen_k_tensor=cum_seqlen_k,
    current_stream=stream,
)
```

#### Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `mma_tiler_mn` | `Tuple[int, int]` | Kernel tile size `(TILE_M, TILE_N)` | `(128, 128)` |
| `is_persistent` | `bool` | Enable persistent kernel mode | `False` |
| `scale_q` | `float` | Q tensor scale factor (for FP8 inputs) | `1.0` |
| `scale_k` | `float` | K tensor scale factor (for FP8 inputs) | `1.0` |
| `scale_v` | `float` | V tensor scale factor (for FP8 inputs) | `1.0` |
| `inv_scale_o` | `float` | Output inverse scale factor (for FP8 outputs) | `1.0` |
| `scale_softmax` | `float \| None` | Softmax scaling factor | `1/sqrt(head_dim)` |
| `qk_acc_dtype` | `torch.dtype` | QK accumulator dtype | `torch.float32` |
| `pv_acc_dtype` | `torch.dtype` | PV accumulator dtype | `torch.float32` |
| `enable_lse` | `bool` | Enable LogSumExp output (wrapper only) | `False` |

#### Constraints

- Input dtype must be `float16`, `bfloat16`, or `float8_e4m3fn`
- Output dtype must be `float16`, `bfloat16`, or `float8_e4m3fn`
- Head dimension `D` must be one of `{32, 64, 128}`
- `H_q` must be divisible by `H_kv` (supports GQA/MQA)
- Requires SM100+ (Blackwell or newer)

---

### 3. Sliding Window Attention

Sliding Window Attention performs attention within a local window around each query position. This captures fine-grained local dependencies efficiently.
This implementation is a wrapper around cudnn backend (and is not strictly open source).

#### Shapes

- **Inputs**
  - `Q` (Query): `(B, H_q, S_q, D)` or `(T, H_q, D)`
  - `K` (Key): `(B, H_kv, S_kv, D)` or `(T, H_kv, D)`
  - `V` (Value): `(B, H_kv, S_kv, D_v)` or `(T, H_kv, D_v)`
  - `seq_len_q`: `(B, 1, 1, 1)` – sequence lengths for queries (T,H,D layout only)
  - `seq_len_kv`: `(B, 1, 1, 1)` – sequence lengths for keys/values (T,H,D layout only)

- **Outputs**
  - `O` (Output): Same shape as `Q`
  - `Stats` (optional): `(B, H_q, S_q, 1)` or `(T, H_q, 1)` – softmax statistics for training

#### Equation

For each query position $q$, attention is restricted to key positions within the window:

$$
O[q] = \sum_{k : q - L \leq k \leq q + R} \text{softmax}\left(\frac{Q[q] \cdot K[k]^T}{\sqrt{D}}\right) V[k]
$$

where $L$ is `left_bound` and $R$ is `right_bound`.

#### High-level Wrapper

```python
from cudnn import NSA

result = NSA.sliding_window_attention_wrapper(
    q_tensor=q,
    k_tensor=k,
    v_tensor=v,
    seq_len_q_tensor=seq_len_q,      # Required for T,H,D layout
    seq_len_kv_tensor=seq_len_kv,    # Required for T,H,D layout
    left_bound=512,
    right_bound=0,  # Causal: no looking ahead
    is_infer=True,  # Set False to output stats for training
    attn_scale=None,  # Defaults to 1/sqrt(head_dim)
    o_dtype=torch.bfloat16,
    intermediate_data_type=torch.float32,
    compute_data_type=torch.float32,
    cudnn_handle=handle,  # Recommended to reuse handle
    stream=None,
)
o, stats = result
# Key access: result["o_tensor"], result["stats_tensor"]
```

#### Class API

```python
from cudnn import NSA
import cudnn

handle = cudnn.create_handle()  # Reuse across calls

swa = NSA.SlidingWindowAttention(
    sample_q=q,
    sample_k=k,
    sample_v=v,
    sample_o=o,
    sample_stats=stats,  # Set to None for inference
    left_bound=512,
    right_bound=0,
    sample_seq_len_q=seq_len_q,
    sample_seq_len_kv=seq_len_kv,
    max_seq_len_q=1024,
    max_seq_len_kv=1024,
    attn_scale=None,
    intermediate_data_type=torch.float32,
    compute_data_type=torch.float32,
    cudnn_handle=handle,
)
assert swa.check_support()
swa.compile()
swa.execute(
    q_tensor=q,
    k_tensor=k,
    v_tensor=v,
    o_tensor=o,
    stats_tensor=stats,
    seq_len_q_tensor=seq_len_q,
    seq_len_kv_tensor=seq_len_kv,
    current_stream=stream,
)
```

#### Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `left_bound` | `int` | Number of positions to look back (left of diagonal) | `0` |
| `right_bound` | `int` | Number of positions to look ahead (right of diagonal). Set to 0 for causal | `0` |
| `attn_scale` | `float \| None` | Attention scaling factor | `1/sqrt(head_dim)` |
| `is_infer` | `bool` | Inference mode (no stats output) | `False` |
| `intermediate_data_type` | `torch.dtype` | Intermediate computation dtype | `torch.float32` |
| `compute_data_type` | `torch.dtype` | Compute dtype | `torch.float32` |
| `max_seq_len_q` | `int` | Maximum sequence length for queries | Required for T,H,D |
| `max_seq_len_kv` | `int` | Maximum sequence length for keys/values | Required for T,H,D |
| `cudnn_handle` | `cudnn.handle \| None` | cuDNN handle (recommended to reuse) | Creates new handle |

#### Constraints

- Supports both `B,H,S,D` (batched) and `T,H,D` (variable-length) layouts
- For `T,H,D` layout, requires `seq_len_q` and `seq_len_kv` (and optionally ragged offset tensors, otherwise fully packed layout is assumed)
- `cudnn_handle` should be reused across calls for performance

---

### 4. Top-K Reduction

Top-K Reduction identifies the most important key-value blocks for each query position based on attention scores. This is used to generate block indices for Selection Attention.

#### Shapes

- **Inputs**
  - `Q` (Query): `(B, H_q, S_q, D)` or `(T, H_q, D)`
  - `K` (Key): `(B, H_kv, S_kv, D)` or `(T, H_kv, D)`
  - `LSE` (LogSumExp): `(B, H_q, S_q)` or `(T, H_q)` – from a prior attention pass
  - `cum_seqlen_q`: `(batch_size + 1,)` – cumulative sequence lengths (T,H,D layout only)
  - `cum_seqlen_k`: `(batch_size + 1,)` – cumulative sequence lengths (T,H,D layout only)

- **Outputs**
  - `topk_scores`: `(B, H_kv, S_q, K)` or `(T, H_kv, K)` – top-K attention scores
  - `topk_indices`: `(B, H_kv, S_q, K)` or `(T, H_kv, K)` – indices of top-K blocks

#### Equation

For each query position, compute block-level attention scores and select the top $K$ blocks:

$$
\text{block\_score}[q, b] = \sum_{k \in \text{block}_b} \exp\left(\frac{Q[q] \cdot K[k]^T}{\sqrt{D}} - \text{LSE}[q]\right)
$$

$$
\text{topk\_indices}[q] = \text{argtop}_K(\text{block\_score}[q, :])
$$

#### High-level Wrapper

```python
from cudnn import NSA

result = NSA.topk_reduction_wrapper(
    q_tensor=q,
    k_tensor=k,
    lse_tensor=lse,
    cum_seqlen_q_tensor=cum_seqlen_q,  # For T,H,D layout
    cum_seqlen_k_tensor=cum_seqlen_k,  # For T,H,D layout
    max_s_q=1024,  # Optional for T,H,D (inferred from cum_seqlen_q if omitted)
    max_s_k=1024,  # Optional for T,H,D (inferred from cum_seqlen_k if omitted)
    acc_dtype=torch.float32,
    k_value=16,  # Number of blocks to select
    selection_block_size=64,
    compress_stride=32,
    is_causal=True,
    mma_tiler_mn=(128, 128),
    scale_softmax=None,
    current_stream=stream,
)
topk_scores, topk_indices = result
# Key access: result["topk_scores_tensor"], result["topk_indices_tensor"]
```

#### Class API

```python
from cudnn import NSA

topk = NSA.TopKReduction(
    sample_q=q,
    sample_k=k,
    sample_lse=lse,
    sample_topk_scores=topk_scores,
    sample_topk_indices=topk_indices,
    sample_cum_seqlen_q=cum_seqlen_q,
    sample_cum_seqlen_k=cum_seqlen_k,
    max_s_q=1024,
    max_s_k=1024,
    acc_dtype=torch.float32,
    k_value=16,
    selection_block_size=64,
    compress_stride=32,
    is_causal=True,
    mma_tiler_mn=(128, 128),
    scale_softmax=None,
)
assert topk.check_support()
topk.compile()
topk.execute(
    q_tensor=q,
    k_tensor=k,
    lse_tensor=lse,
    topk_scores_tensor=topk_scores,
    topk_indices_tensor=topk_indices,
    cumulative_s_q_tensor=cum_seqlen_q,
    cumulative_s_k_tensor=cum_seqlen_k,
    current_stream=stream,
)
```

#### Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `k_value` | `int` | Number of top blocks to select | `16` |
| `selection_block_size` | `int` | Size of blocks for selection | `64` |
| `compress_stride` | `int` | Stride for compression blocks | `32` |
| `is_causal` | `bool` | Apply causal masking | `True` |
| `mma_tiler_mn` | `Tuple[int, int]` | Kernel tile size | `(128, 128)` |
| `scale_softmax` | `float \| None` | Softmax scaling factor | `1/sqrt(head_dim)` |
| `acc_dtype` | `torch.dtype` | Accumulator dtype | `torch.float32` |
| `max_s_q` | `int \| None` | Maximum query sequence length (optional for `T,H,D`, inferred from `cum_seqlen_q`) | `None` |
| `max_s_k` | `int \| None` | Maximum key sequence length (optional for `T,H,D`, inferred from `cum_seqlen_k`) | `None` |

#### Constraints

- Input dtype for Q/K must match
- LSE dtype must match `acc_dtype`
- `topk_indices` must be `int32`
- Requires SM100+ (Blackwell or newer)

**Note**: The returned values exclude the first block and neighboring blocks from the reduction. Rows with all `-inf` scores and `-1` indices are expected for positions near the beginning of sequences.

---

## Tensor Formats

### Supported Layouts

| Component | `B,H,S,D` | `T,H,D` |
|-----------|-----------|---------|
| Selection Attention | ❌ | ✅ |
| Compression Attention | ✅ | ✅ |
| Sliding Window Attention | ✅ | ✅ |
| Top-K Reduction | ✅ | ✅ |

### T,H,D Format (Variable-Length Batched)

Used for sequences of varying lengths packed into a single tensor:

- **Q/K/V**: `(T, H, D)` where `T = sum(seq_lengths)`
- **cum_seqlen**: `(batch_size + 1,)` – cumulative sequence lengths, e.g., `[0, 128, 320, 512]` for 3 sequences of lengths 128, 192, 192

### B,H,S,D Format (Fixed-Length Batched)

Traditional batched format with padding:

- **Q/K/V**: `(B, H, S, D)` where `B` is batch size, `S` is padded sequence length

---

## Data Types

### Supported Input/Output Types

| Dtype | Selection | Compression | Sliding Window | Top-K |
|-------|-----------|-------------|----------------|-------|
| `float16` | ✅ | ✅ | ✅ | ✅ |
| `bfloat16` | ✅ | ✅ | ✅ | ✅ |
| `float8_e4m3fn` | ❌ | ✅ | ❌ | ❌ |

### Accumulator Types

All components require `float32` accumulator dtype for numerical stability.

---

## Hardware Requirements

| Component | Minimum SM | Notes |
|-----------|------------|-------|
| Selection Attention | SM90 (Hopper) | - |
| Compression Attention | SM100 (Blackwell) | - |
| Sliding Window Attention | SM80+ | Uses cuDNN backend |
| Top-K Reduction | SM100 (Blackwell) | - |

---

## Usage Examples

For complete usage examples and tests, see:

- `test/python/fe_api/nsa/test_NSA_selection_attention.py`
- `test/python/fe_api/nsa/test_NSA_compression_attention.py`
- `test/python/fe_api/nsa/test_NSA_swa.py`
- `test/python/fe_api/nsa/test_NSA_topk_reduction.py`

### Example: Full NSA Pipeline

```python
import torch
from cudnn import NSA
import math

# Configuration
batch_size = 2
seq_len = 2048
h_q, h_kv = 32, 8  # GQA with 4:1 ratio
head_dim = 128
k_value = 16
block_size = 64
compress_stride = 32
window_size = 512

device = "cuda"
dtype = torch.bfloat16

# Allocate input tensors (T,H,D format)
seq_lengths = torch.tensor([seq_len] * batch_size, dtype=torch.int32, device=device)
cum_seqlen = torch.tensor([0, seq_len, 2 * seq_len], dtype=torch.int32, device=device)
T = cum_seqlen[-1].item()

q = torch.randn(T, h_q, head_dim, dtype=dtype, device=device)
k = torch.randn(T, h_kv, head_dim, dtype=dtype, device=device)
v = torch.randn(T, h_kv, head_dim, dtype=dtype, device=device)

# Step 1: Compression Attention
o_cmp, lse_cmp = NSA.compression_attention_wrapper(
    q_tensor=q,
    k_tensor=k,
    v_tensor=v,
    cum_seqlen_q_tensor=cum_seqlen,
    cum_seqlen_k_tensor=cum_seqlen,
    enable_lse=True,
    o_dtype=torch.bfloat16,
)

# Step 2: Top-K Reduction to find important blocks (uses LSE from compression)
topk_scores, topk_indices = NSA.topk_reduction_wrapper(
    q_tensor=q,
    k_tensor=k,
    lse_tensor=lse_cmp,
    cum_seqlen_q_tensor=cum_seqlen,
    cum_seqlen_k_tensor=cum_seqlen,
    k_value=k_value,
    selection_block_size=block_size,
    compress_stride=compress_stride,
    is_causal=True,
)

# Step 3: Selection Attention for important blocks
block_counts = torch.full((T, h_kv), k_value, dtype=torch.int32, device=device)

o_sel, l_sel, m_sel = NSA.selection_attention_wrapper(
    q_tensor=q,
    k_tensor=k,
    v_tensor=v,
    block_indices_tensor=topk_indices,
    block_counts_tensor=block_counts,
    cum_seqlen_q_tensor=cum_seqlen,
    cum_seqlen_k_tensor=cum_seqlen,
    block_size=block_size,
)

# Step 4: Sliding Window Attention for local context
o_swa, stats_swa = NSA.sliding_window_attention_wrapper(
    q_tensor=q,
    k_tensor=k,
    v_tensor=v,
    seq_len_q_tensor=seq_lengths,
    seq_len_kv_tensor=seq_lengths,
    left_bound=window_size,
    right_bound=0,
    is_infer=False,
    attn_scale=1.0 / math.sqrt(head_dim),
)

# Step 5: Combine outputs (application-specific weighted combination)
# This is a simplified example; actual combination uses learned gating
final_output = o_cmp + o_sel + o_swa  # Placeholder combination
```

---

## References

- [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (arXiv:2502.11089)](https://arxiv.org/pdf/2502.11089)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [cuDNN](https://developer.nvidia.com/cudnn)
