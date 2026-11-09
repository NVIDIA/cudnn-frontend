# Simplified sparse attention

Experimental PyTorch and JAX attention with a trainable, single-head indexer. The
kernels ship in the `nvidia-cudnn-frontend` wheel under Apache-2.0; no separate
attention wheel is required. Install PyTorch separately, and install JAX with
CUDA support to use the JAX API. CuTe DSL 4.8.0 or newer is required.
The PyTorch path for long rectangular inputs (at least 131,072 keys) needs CUDA conditional
graph support, tested with PyTorch 2.13. An unavailable conditional-graph API
is rejected before filtered-state allocation or compilation.

The indexer scores each permitted key with `q_idx @ k_idx.T / sqrt(128)` and
selects the largest `topk` scores. Attention computes softmax over those keys
with scale `1/16`, then multiplies by V. The auxiliary objective is the KL
divergence from the mean attention distribution across query heads to the
indexer's selected-key distribution. The teacher distribution is detached.

## API

```python
import torch
from cudnn.ops import SimplifiedSparseAttentionMetadata, simplified_sparse_attention

# Q: [Tq, Hq, 256]; K/V: [Tk, 1, 256]; indexer Q/K: [Tq, 128]/[Tk, 128].
metadata = SimplifiedSparseAttentionMetadata(
    key_starts=key_starts,          # Inclusive key bounds, int32 [Tq].
    key_ends=key_ends,              # Exclusive key bounds, int32 [Tq].
    query_valid=query_valid,        # bool [Tq]; False for padding.
    loss_denominator=denominator,   # FP32 [1], positive global query count.
    triangular_scores=False,
    mask_invalid_rows=True,
)
result = simplified_sparse_attention(q, k, v, q_idx, k_idx, metadata, topk=512)
output, indexer_loss = result       # Also result["output"], result["indexer_loss"].
loss = output.float().square().sum() + indexer_loss
loss.backward()
```

`output` has Q's shape and BF16 dtype. `indexer_loss` is an FP32 scalar that
already includes `loss_coeff` (default `0.1`) and division by
`loss_denominator`. Do not apply the coefficient again. Zero disables the
indexer training signal. Autograd returns gradients for Q, K, V, indexer Q,
and indexer K. The index selection and teacher distribution are not
differentiated. Higher-order derivatives are unsupported.

| Property | Supported values |
| --- | --- |
| GPU | SM100, SM103 |
| Input dtype | BF16 |
| Query heads | 16, 32, 96 |
| KV heads / head dimension | 1 / 256 |
| Indexer dimension | 128 |
| `topk` | 512, 1024, 2048 |
| Token layout | Packed THD; rectangular Q/K extents allowed |
| Q/K/V storage | Compact head/channel dimensions; 16-byte-aligned pointers and token strides |
| Empty streams | Unsupported; padding rows are supported |

All tensors must reside on one CUDA device. Supply contiguous metadata in the
types above. Each valid row must have `0 <= key_start < key_end <= Tk`;
invalid rows must be marked in `query_valid`. Bounds describe causal and
document masks in physical key coordinates. `triangular_scores=True` requires
`Tq == Tk` and no bounds above the causal diagonal. Use
`all_query_rows_valid=True` only when every row is valid;
`mask_invalid_rows=False` likewise requires no padding. Indexer inputs must
be finite. The optional `SSA_DEBUG_CHECKS=1` validates finiteness and selected
bounds, but synchronizes the host and is for debugging only.

The caller owns projections, packed layout construction, and distributed
communication. Supply all query heads when forming the teacher distribution.
The optional `kv_ready_event` is joined after index selection and before
attention reads K/V. Q, indexer Q/K, and metadata must already be ready on
the input device's current stream. The caller must keep K/V storage alive
on both producer and consumer streams.

## Workspace and CUDA graphs

The convenience API allocates outputs, saved autograd state, and scratch.
The long-sequence filtered indexer also keeps repair storage per device and
stream. To reserve its maximum rectangular extents before training:

```python
from cudnn.ops import simplified_sparse_attention_reserve
simplified_sparse_attention_reserve(max_query_len, max_key_len, topk=2048)
```

Reserve on the stream that will run attention. Calls above the reservation
can grow storage and rebuild the internal repair graph. Short-key paths do
not need persistent repair storage.

**External CUDA graph capture is unsupported**, including after reservation
or warmup. The public API rejects capture before launching work. Internal
conditional CUDA graphs repair filtered rows; they do not provide capture
support for the public training call. There is no cuDNN Graph API node.

## JAX

`cudnn.ops.simplified_sparse_attention_jax` returns a plain
`(output, indexer_loss)` tuple. It supports `jax.jit`, `jax.grad`, and
`jax.vjp` for the five activation inputs. Pass metadata as array arguments:

```python
import jax
from cudnn.ops import simplified_sparse_attention_jax

@jax.jit
def training_loss(q, k, v, q_idx, k_idx, starts, ends, valid, denominator):
    output, indexer_loss = simplified_sparse_attention_jax(
        q, k, v, q_idx, k_idx, starts, ends, valid, denominator,
        topk=512, loss_coeff=0.1,
    )
    return (output.astype("float32") ** 2).sum() + indexer_loss

gradients = jax.grad(training_loss, argnums=(0, 1, 2, 3, 4))
```

Shapes, dtypes, key bounds, padding, and loss scaling match the PyTorch API.
`topk`, `loss_coeff`, and `triangular_scores` are static options. Metadata,
index selection, and teacher targets are not differentiated. Use one process,
one GPU per call, and only one supported GPU architecture visible to that
process. Distributed execution, `vmap`, and higher-order derivatives are
unsupported. PyTorch remains an import dependency of the shared kernel modules;
execution uses native JAX FFI calls, without PyTorch tensor conversion or a
Python callback.

XLA owns scratch and schedules kernels on its stream. There is no JAX reserve
API or K/V producer event. Long rectangular inputs use the shared filtered
selector and an XLA loop that repairs failed rows in groups of 128. They do
not need PyTorch's conditional-graph API. Square triangular input currently
uses a dense score workspace in the JAX path.

External CUDA graph capture is unsupported for JAX as well. These FFI calls
opt out of XLA CUDA command-buffer capture; JIT remains supported.

Neither API has a deterministic mode. FP32 atomic gradient reductions can vary
with scheduling. Moving packed rows can also change scalar-loss rounding.
See the [determinism support page](/operations/determinism)
([GitHub](https://github.com/NVIDIA/cudnn-frontend/blob/develop/docs/operations/determinism.md))
for measured limits.

## Validation and benchmarks

The suite covers numerical forward/backward results, auxiliary loss scaling,
invalid inputs, upstream-gradient ownership, top-k edge cases, capture
rejection, bytewise replay, packed batch changes, and golden hashes.

```sh
cd test/python
pytest simplified_sparse_attention/cutedsl -m L0
# Longer filtered-indexer and repair checks:
pytest simplified_sparse_attention/cutedsl -m L1
```

The [benchmark guide](https://github.com/NVIDIA/cudnn-frontend/blob/develop/benchmark/simplified_sparse_attention/README.md)
measures PyTorch eager and JAX JIT forward/training latency after warmup.
