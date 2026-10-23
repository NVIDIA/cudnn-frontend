# OSS native sparse-attention contract

Use `cudnn.NSA` to compose Native Sparse Attention from compression, top-k selection, sparse selection attention, and sliding-window attention. Each component owns a separate specialization and stream contract.

## Before you begin

Install the `cutedsl` extra, select one component, and prepare a dense reference
plus every routing or sequence-length intermediate it consumes. Record the
layout family and all specialization-defining dimensions.

## NSA namespace

`cudnn.NSA` lazily exports the four component classes and their wrappers. Namespace import validates optional Python dependencies only; it does not establish that a particular shape is supported.

## compression_attention_wrapper

Compression attention accepts Q, K, and V plus optional cumulative sequence lengths for THD layout. It supports FP32 accumulation controls, an MMA tile, persistent scheduling, input and output scales, and an optional softmax scale. Set `enable_lse=True` when a later stage needs natural-log LSE. The result mapping contains `o_tensor` and, when enabled, `lse_tensor`.

Signature: `compression_attention_wrapper(q_tensor, k_tensor, v_tensor, cum_seqlen_q_tensor=None, cum_seqlen_k_tensor=None, enable_lse=False, o_dtype=None, qk_acc_dtype=torch.float32, pv_acc_dtype=torch.float32, mma_tiler_mn=(128, 128), is_persistent=False, scale_q=1.0, scale_k=1.0, scale_v=1.0, inv_scale_o=1.0, scale_softmax=None, stream=None)`.

This CUTLASS DSL path requires SM100+. Fixed layout uses rank-4 BHSD tensors. THD uses rank-3 tensors and requires both cumulative-length tensors of the same length; they may be INT32 or INT64. Q, K, and V share FP16, BF16, or E4M3 dtype; output may use any of those types; both accumulators are FP32. QK and V dimensions must match and be 32, 64, or 128, and Q heads must be divisible by KV heads. Scaling parameters used to construct a class specialization may be overridden at execution only where the class contract permits it.

## topk_reduction_wrapper

Top-k reduction consumes compressed Q, K, and LSE, then produces `topk_scores` and INT32 `topk_indices`. Configure `k_value`, the selection block size, accumulation dtype, and optional THD sequence metadata. The logical KV length, not padded storage, determines valid candidates.

Signature: `topk_reduction_wrapper(q_tensor, k_tensor, lse_tensor, cum_seqlen_q_tensor=None, cum_seqlen_k_tensor=None, max_s_q=None, max_s_k=None, acc_dtype=torch.float32, k_value=16, selection_block_size=64, compress_stride=32, is_causal=True, mma_tiler_mn=(128, 128), scale_softmax=None, current_stream=None)`. It requires SM100+, matching FP16 or BF16 Q/K, FP32 LSE and accumulation, and INT32 cumulative lengths in THD mode. Supply both cumulative arrays or neither; THD maximum lengths are specialization inputs.

## selection_attention_wrapper

Selection attention consumes Q, K, V, block indices, and block counts. `block_size` defines how selected blocks map to physical KV tokens. The output mapping contains the attention output and the internal `l` and `m` statistics used by the selected kernel. Keep block indices within the logical KV range and block counts within the last dimension of the index tensor.

Signature: `selection_attention_wrapper(q_tensor, k_tensor, v_tensor, block_indices_tensor, block_counts_tensor, cum_seqlen_q_tensor=None, cum_seqlen_k_tensor=None, block_size=64, scale_softmax=None, o_dtype=None, acc_dtype=torch.float32, max_s_q=None, max_s_k=None, stream=None)`. The current implementation requires SM90+ and THD rank-3 Q/K/V; rank-4 dense input raises `NotImplementedError`. Supply cumulative Q lengths and `max_s_q`; `max_s_k`, when supplied, must equal `max_s_q`. Block indices are INT32 `[T, H_kv, K]`, counts are INT32 `[T, H_kv]`, and block size is 16, 32, or 64. Q/K/V share FP16 or BF16 dtype, Q heads are a multiple of KV heads, and accumulation is FP32.

## sliding_window_attention_wrapper

Sliding-window attention is a cuDNN Graph-backed wrapper. It accepts dense or ragged Q, K, and V; left and right diagonal bounds; optional per-batch sequence lengths and ragged offsets; and `is_infer` to control statistics. `is_infer=False` returns statistics required by training workflows. Bounds use the same diagonal convention as the Graph SDPA API.

Signature: `sliding_window_attention_wrapper(q_tensor, k_tensor, v_tensor, seq_len_q_tensor=None, seq_len_kv_tensor=None, q_ragged_offset_tensor=None, k_ragged_offset_tensor=None, v_ragged_offset_tensor=None, o_ragged_offset_tensor=None, stats_ragged_offset_tensor=None, left_bound=0, right_bound=0, is_infer=False, attn_scale=None, o_dtype=None, intermediate_data_type=torch.float32, compute_data_type=torch.float32, cudnn_handle=None, stream=None)`. Dense rank-4 mode rejects sequence-length and ragged-offset inputs. THD rank-3 mode requires both sequence-length tensors and maximum lengths derived by the wrapper; provide all applicable ragged offsets together or leave all absent so the wrapper computes fully packed offsets.

This wrapper owns or accepts a cuDNN handle. It updates the handle stream to `stream` before execution. Do not concurrently reuse one handle on unrelated streams without external coordination.

## Reuse class specializations

The corresponding classes are `CompressionAttention`, `TopKReduction`, `SelectionAttention`, and `SlidingWindowAttention`. Constructors use representative tensors to compile or build a specialization. Every later `execute` call must preserve device, dtype, rank, layout family, and specialization-defining dimensions.

Current exports: [NSA namespace](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/native_sparse_attention/__init__.py#L35).

## Expected result

Each component matches its independently computed stage and passes compatible
metadata to the next component. This page does not publish an end-to-end NSA
sample; successful namespace import alone does not establish kernel support or
cross-stage correctness.
