# Configure attention recipes

Add one attention feature at a time to a verified dense SDPA baseline. Every recipe changes the support surface and cache key.

## Before you begin

Complete the [attention tutorial](/tutorials/attention) and keep its dense
forward and backward checks as a baseline. Confirm the version, architecture,
dtype, layout, and backward support of every feature you add.

## Training and inference

Set `generate_stats=True` for a forward result that will feed `sdpa_backward`. Inference can omit statistics. Keep forward output and statistics alive until backward completes.

## Masks

- Diagonal bands express causal and sliding-window limits with explicit top-left or bottom-right alignment.
- Padding masks require query and key/value sequence-length tensors.
- Block masks use a packed tile representation and require an implementation that supports them. On SM10x, they require cuDNN 9.26 or newer.
- A score modifier function inserts a graph callback before softmax and cannot be combined with every built-in mask path.

## Layouts and state

Dense BSHD/BHSD layouts use explicit strides. THD packs variable sequences and uses ragged offsets. Paged attention treats K/V as cache containers and uses integer page tables plus a logical maximum sequence length. Sink tokens add persistent score state.

## Dropout

Use either probability plus Philox seed/offset tensors, or an explicit mask plus scale. The mask applies after softmax. Preserve seed/offset progression across executions and expose RNG dump only when needed for backward or diagnostics.

## Low precision

FP8 and MXFP8 paths add descale, scale, amax, reorder, dimension, and architecture constraints. Validate against a higher-precision baseline before tuning.

Published recipes: [causal BF16](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/sdpa_bf16_causal.py), [dropout](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/sdpa_dropout.py), [paged KV cache](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/sdpa_paged_kv_cache.py), [C++ block mask](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/sdpa_block_mask.cc), and [sliding window](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/sdpa_sliding_window_chunked_prefill.py).

## Expected result

Each linked sample executes one focused variation. Your derived graph should
continue to match the dense baseline wherever the feature is inactive and
should fail during support checking, rather than execution, when the selected
backend cannot implement the requested combination.
