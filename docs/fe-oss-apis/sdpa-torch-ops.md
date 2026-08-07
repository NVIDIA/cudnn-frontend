# SDPA torch custom ops: `cudnn::sdpa_fwd` / `cudnn::sdpa_bwd`

PyTorch custom ops (`torch.library`) exposing the full cuDNN SDPA feature
surface — the features `torch.nn.functional.scaled_dot_product_attention`'s
aten contract cannot express:

- **attention sinks** — per-Q-head logits folded into the softmax denominator
- **sliding window** — `window_left` (cuDNN convention: visible tokens
  *including* self; FA2's `(w, 0)` maps to `window_left = w + 1`)
- **bottom-right causal alignment** — inference-style diagonals
- **padded batches** — per-batch actual lengths via `seq_len_q` / `seq_len_kv`
- **THD / varlen packing** — FlashAttention-style `(T, H, D)` + `cu_seqlens`

The ops build cuDNN pygraph `sdpa` / `sdpa_backward` nodes; the engine Router
picks the best serving plan (FROST OSS kernels or cuDNN-backend engines) per
configuration. Graphs are cached per configuration (bounded, thread-safe;
cuDNN handles are thread-local).

## Usage

```python
import torch
import cudnn

_ = cudnn.sdpa_torch  # lazy public export: importing registers cudnn::sdpa_fwd / cudnn::sdpa_bwd

# Dense BHSD with sinks + sliding window
o, lse = torch.ops.cudnn.sdpa_fwd(q, k, v, scale, is_causal=True,
                                  window_left=128, sinks=sinks, return_lse=True)

# THD / varlen (FA-style packed (T, H, D) + cu_seqlens), differentiable:
q, k, v = (t.requires_grad_(True) for t in (q_thd, k_thd, v_thd))
o, lse = torch.ops.cudnn.sdpa_fwd(q, k, v, scale, is_causal=True,
                                  cu_seqlens_q=cu, cu_seqlens_kv=cu,
                                  max_seqlen_q=mx, max_seqlen_kv=mx,
                                  return_lse=True)
o.backward(grad)  # routes through cudnn::sdpa_bwd via register_autograd

# Or through the python wrapper (same op underneath):
o = cudnn.sdpa_torch(q, k, v, is_causal=True, cu_seqlens_q=cu, cu_seqlens_kv=cu,
                     max_seqlen_q=mx, max_seqlen_kv=mx)
```

## Contracts and limits

- Dense tensors are BHSD `(B, H, S, D)` (any strides; the graph declares the
  actual layout). Varlen tensors are packed `(T, H, D)`; non-contiguous views
  (e.g. K/V slices of a fused `(T, 2, H, D)` KV projection) are declared with
  their true strides. On the varlen path, a non-dense innermost dim or a
  misaligned base pointer is repaired by one copy (warned as slow path); the
  dense path declares the given strides as-is.
- One io dtype per call (`fp16` or `bf16`); mixed-dtype inputs are rejected.
- `sdpa_bwd` serves the **THD/varlen** path. Dense backward and sink backward
  (dSink) are follow-ups and raise `NotImplementedError`. It consumes a
  **padded** `(B, H, max_seqlen_q, 1)` fp32 LSE (backend restriction: bprop
  THD rejects ragged LSE on SM8X/SM12X).
- Autograd (`register_autograd`) requires `return_lse=True` on the forward;
  the glue converts the packed TH1 stats to the padded layout device-side.
- Both ops ship `register_fake` meta kernels. `cudnn::sdpa_fwd` passes
  `torch.library.opcheck` on the dense and varlen paths, including
  dynamic-shape AOT dispatch (`torch.compile`-ready); the opcheck autograd
  case exercises `cudnn::sdpa_bwd` through the registered backward.

## Requirements

- `nvidia-cudnn-frontend[cutedsl]`, cuDNN backend ≥ 9.6 (THD token-major
  stats), sm80+.

Tests: `test/python/test_cudnn_sdpa_torch_ops.py`.
