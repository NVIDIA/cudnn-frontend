# Native Sparse Attention (NSA) Module

The Native Sparse Attention (NSA) module implements Native Sparse attention as described in the [Native Sparse Attention: Hardware-Aligned and Natively
Trainable Sparse Attention](https://arxiv.org/pdf/2502.11089).

## File Structure

Currently, only the selection component of NSA is implemented.

```
python/cudnn/native_sparse_attention/
├── __init__.py                          # Main module initialization and NSA namespace
├── sparse_attention.md                  # This documentation file  
└── selection/                           # Selection attention implementation
    ├── __init__.py                      # Selection module exports (SelectionAttention and SelectionAttentionWrapper)
    ├── api.py                           # High-level API class and wrapper function
    └── NSA_select_attn_fwd_hmma.py      # CuteDSL kernel implementation
├── compression/                         # Compression attention implementation
    ├── __init__.py                      # Compression module exports (CompressionAttention and CompressionAttentionWrapper)
    ├── api.py                           # High-level API class and wrapper function
    └── fmha.py                          # CuteDSL kernel implementation
├── sliding_window/                      # Sliding window attention implementation
    ├── __init__.py                      # Sliding window module exports (SlidingWindowAttention and SlidingWindowAttentionWrapper)
    ├── api.py                           # High-level API class and wrapper function
    └── NSA_swa_fwd_hmma.py              # CuteDSL kernel implementation
└── top-k/                               # TODO, not implemented yet
```

## Installation

Install the optional cudnn dependences required for NSA:
```bash
pip install nvidia-cudnn-frontend[cutedsl]
```

## Usage

Sample usage and tests can be found in the (test/python) folder:
- [test_NSA_selection_attention.py](test/python/test_NSA_selection_attention.py), `pytest test/python/test_NSA_selection_attention.py`
- [test_NSA_compression_attention.py](test/python/test_NSA_compression_attention.py), `pytest test/python/test_NSA_compression_attention.py`
- [test_NSA_swa.py](test/python/test_NSA_swa.py), `pytest test/python/test_NSA_swa.py`

Once all components are implemented, we will offer a central NSA API that will do the full NSA computation end-to-end. We will also offer the individual components as standalone APIs, as demonstrated below.

### Basic Usage with NSA Namespace

```python
import torch
from cudnn import NSA

# Prepare input tensors
q = torch.randn((T, H_q, D), dtype=torch.bfloat16, device='cuda')
k = torch.randn((T, H_kv, D), dtype=torch.bfloat16, device='cuda')
v = torch.randn((T, H_kv, D_v), dtype=torch.bfloat16, device='cuda')

# Define sparse attention pattern
block_indices = torch.tensor(..., device='cuda')  # Selected block indices
block_counts = torch.tensor(..., device='cuda')   # Number of blocks per sequence
seq_offsets = torch.tensor(..., device='cuda')    # Sequence boundaries

# Execute sparse attention
o, l, m = NSA.SelectionAttentionWrapper(
    q, k, v, block_indices, block_counts, seq_offsets,
    block_size=64,
    scale_softmax=None,  # Defaults to 1/sqrt(head_dim)
    acc_dtype=torch.float32
)
```

### Advanced Usage with Compiled Kernels

```python
import torch
from cudnn import NSA

# Prepare input and output tensors
q = torch.randn((T, H_q, D), dtype=torch.bfloat16, device='cuda')
k = torch.randn((T, H_kv, D), dtype=torch.bfloat16, device='cuda')
v = torch.randn((T, H_kv, D_v), dtype=torch.bfloat16, device='cuda')
o = torch.zeros((T, H_q, D_v), dtype=torch.bfloat16, device='cuda')
l = torch.zeros((T, H_q), dtype=torch.float32, device='cuda')
m = torch.zeros((T, H_q), dtype=torch.float32, device='cuda')

# Define sparse attention pattern
block_indices = torch.tensor(..., device='cuda')  # Selected block indices
block_counts = torch.tensor(..., device='cuda')   # Number of blocks per sequence
seq_offsets = torch.tensor(..., device='cuda')    # Sequence boundaries
max_s = 8192

# Create and configure selection attention instance. The sample input/output tensors are used for compilation, and must have the same dtype, shape, and strides as the actual input/output tensors.
selection_attention = SelectionAttention(
    sample_q=q,
    sample_k=k, 
    sample_v=v,
    sample_o=o,
    sample_l=l,
    sample_m=m,
    sample_block_indices=block_indices,
    sample_block_counts=block_counts,
    sample_seq_offsets=seq_offsets,
    acc_dtype=torch.float32,
    max_s=max_s,
    block_size=64,
    scale_softmax=None,
)

# Check hardware and configuration support
assert selection_attention.check_support()

# Compile kernel
selection_attention.compile()

# Execute on actual data
selection_attention.execute(
    q_tensor=q,
    k_tensor=k,
    v_tensor=v,
    o_tensor=o,
    l_tensor=l,
    m_tensor=m,
    block_indices_tensor=block_indices,
    block_counts_tensor=block_counts,
    seq_offsets_tensor=seq_offsets,
    scale_softmax=None,
)
```

## Tensor Formats

### Input Tensors

Selection Attention currently only supports T,H,D input format. B,H,S,D is not yet supported.

#### T,H,D Format:
- **Q (Query)**: `(T, H_q, D)`
- **K (Key)**: `(T, H_kv, D)`  
- **V (Value)**: `(T, H_kv, D_v)`
- **block_indices**: `(num_blocks,)`
- **block_counts**: `(batch_size,)`
- **seq_offsets**: `(batch_size + 1,)`

Compression Attention and SWA support both T,H,D and B,H,S,D input formats.
#### B,H,S,D Format:
- **Q (Query)**: `(B, H_q, S_q, D)`
- **K (Key)**: `(B, H_kv, S_kv, D)`
- **V (Value)**: `(B, H_kv, S_kv, D_v)`
- **O (Output)**: `(B, H_q, S_q, D_v)`

### Output Tensors

#### T,H,D Format:
- **O (Output)**: `(T, H_q, D_v)`
- **L (LogSumExp)**: `(T, H_q)`
- **M (Max)**: `(T, H_q)`

#### B,H,S,D Format:
- **O (Output)**: `(B, H_q, S_q, D_v)`
- **L (LogSumExp)**: `(B, H_q, S_q)`
- **M (Max)**: `(B, H_q, S_q)`
