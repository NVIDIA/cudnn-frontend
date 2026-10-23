# OSS SDPA D=256 contract

Use the optional D=256 wrappers for Blackwell workloads that need the CUTLASS DSL implementation rather than the general cuDNN Graph SDPA operation.

## Before you begin

Install the `cutedsl` extra, use SM100 or newer, and prepare dense or THD BF16
or FP16 tensors with head dimension 256. Keep a framework SDPA forward and
backward reference with the same mask, window, and softmax scale.

## sdpa_fwd_wrapper_sm100_d256

`cudnn.sdpa_fwd_wrapper_sm100_d256(q_tensor, k_tensor, v_tensor, cum_seqlen_q_tensor=None, cum_seqlen_k_tensor=None, max_s_q=None, max_s_k=None, qk_acc_dtype=torch.float32, pv_acc_dtype=torch.float32, mma_tiler_mn=(128, 128), is_causal=False, window_size=(-1, -1), scale_softmax=None, scale_output=1.0, current_stream=None)` returns `o_tensor` and FP32 `lse_tensor`.

Q, K, and V must be CUDA FP16 or BF16 tensors on the same SM100+ device, with matching dtypes and head dimension 256. Dense inputs use BHSD rank-4 tensors with contiguous D and stride order D, H, S, B. Variable-length inputs use THD rank-3 tensors, or rank-4 tensors with a leading dimension of one, plus both INT32 cumulative-sequence tensors. `max_s_q` and `max_s_k` must cover the maximum differences in those cumulative arrays.

Both accumulator types must be FP32 and `mma_tiler_mn` must be `(128, 128)`. Noncausal mode currently requires the full `(-1, -1)` window. Causal mode sets the right window to zero; the left bound must remain smaller than the maximum KV length minus one. An omitted or zero `scale_softmax` uses `1 / sqrt(256)`. `scale_output` scales O but not LSE.

## sdpa_bwd_wrapper_sm100_d256

`cudnn.sdpa_bwd_wrapper_sm100_d256(q_tensor, k_tensor, v_tensor, o_tensor, do_tensor, lse_tensor, cum_seqlen_q_tensor=None, cum_seqlen_k_tensor=None, max_s_q=None, max_s_k=None, acc_dtype=torch.float32, mma_tiler_mn=(128, 128), dkdv_mma_tiler_mn=(128, 64), is_causal=False, window_size=(-1, -1), scale_softmax=None, current_stream=None)` returns `dq_tensor`, `dk_tensor`, and `dv_tensor`.

Backward requires the exact forward Q, K, V, output, natural-log LSE, layout mode, cumulative-sequence metadata, causal or window mode, and softmax scale. All data tensors and allocated gradients use the input FP16 or BF16 dtype; LSE and accumulation are FP32. The current implementation fixes the dQ tile at `(128, 128)` and the dK/dV tile at `(128, 64)`. Do not reuse LSE from the general Graph SDPA implementation unless its representation is explicitly proven compatible.

Backward has no `scale_output` argument. For a training pair, keep forward `scale_output=1.0` unless the application has independently accounted for the output scale in its gradient definition.

## Reuse compiled classes

### SdpafwdSm100D256

`cudnn.SdpafwdSm100D256` accepts representative forward tensors and specialization options in its constructor. Its `execute` method writes caller-provided output and optional LSE on the supplied current stream.

### SdpabwdSm100D256

`cudnn.SdpabwdSm100D256` accepts representative backward tensors and matching forward metadata. Its `execute` method writes caller-provided dQ, dK, and dV. Wrapper calls allocate outputs and manage the cache; class calls give the application direct ownership of output and compilation lifetime.

## Keep fallback explicit

These direct wrappers do not automatically fall back to Graph SDPA. Catch only documented unsupported conditions, then build a separate Graph SDPA path whose layouts and output semantics are independently validated.

Current sources: [forward](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/sdpa/fwd/api.py#L22) and [backward](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/sdpa/bwd/api.py#L21).

## Expected result

Forward output and natural-log LSE match the reference, and backward reproduces
Q, K, and V gradients using the same forward metadata. No executable
documentation sample is published for this wrapper, so both checks are
required before replacing the general Graph SDPA path.
