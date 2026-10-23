# OSS normalization fusion contracts

This page covers two current OSS normalization paths: the direct SM100 RMSNorm randomized-Hadamard wrapper and the Graph-detected RMSNorm + SiLU fusion.

## Before you begin

Install the optional kernel dependencies, use SM100 for the direct RHT wrapper,
and prepare a PyTorch normalization reference. For the Graph fusion, preserve
the exact inference RMSNorm-to-SiLU pattern required by the OSS heuristic.

## rmsnorm_rht_amax_wrapper_sm100

`cudnn.rmsnorm_rht_amax_wrapper_sm100(x_tensor, w_tensor, eps=1e-5, num_threads=None, rows_per_cta=None, current_stream=None)` returns `o_tensor` and `amax_tensor`.

X is a CUDA activation tensor whose last dimension is normalized. W is the CUDA RMSNorm weight for that dimension. The tensors must use the dtype, contiguity, and aligned feature extent accepted by the current SM100 kernel. `eps` is part of the numerical contract. `num_threads` and `rows_per_cta` tune the compiled specialization; leave them unset unless a measured configuration is known to support the shape.

The output has X's logical shape. Amax is a per-CTA intermediate, not one global scalar. Downstream quantization must reduce or interpret it according to its own scale contract.

## RmsNormRhtAmaxSm100

Construct `cudnn.RmsNormRhtAmaxSm100` with representative X, W, output, and amax tensors. The constructor compiles a specialization. Call `execute(x_tensor, w_tensor, o_tensor, amax_tensor, current_stream=None)` with compatible caller-owned buffers.

The class does not own input or output memory. Keep all buffers alive through stream completion and do not reuse one output buffer concurrently across streams.

Current source: [RMSNorm RHT amax API](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/rmsnorm_rht_amax/api.py#L52).

## Run the RMSNorm + SiLU Graph fusion

The current checkout recognizes the Graph pattern `rmsnorm(INFERENCE) -> swish(beta=1.0)` when plans are created with `cudnn.heur_mode.OPENSOURCE`. There is no top-level `cudnn.rmsnorm_silu` wrapper or class.

The current engine requires BF16 X and scale tensors, an epsilon tensor, and at least rank-2 X. It flattens all dimensions after `dim[0]` into the normalized width C and treats `dim[0]` as the token count. The SiLU output can be BF16 on SM80+, E4M3 FP8 on SM89+, or NVFP4 on SM100+. Inference is the only supported RMSNorm phase; the RMSNorm result must feed the immediately following Swish node, and an explicitly non-unit `swish_beta` prevents pattern matching.

SM100 has tuned configurations for C in `{64, 128, 160, 256, 320, 512, 640, 1024}` and token counts in `{1560, 6240, 24960, 99840, 399360}`. Other positive shapes use a conservative vectorization heuristic and can still be rejected when C cannot be divided into a legal load configuration. NVRTC needs the CUDA numeric headers at runtime; set `CUDA_HOME` or `CUDA_PATH` when the toolkit is not available under `/usr/local/cuda`.

The canonical sample builds the recognized graph, requests the OSS plan, and compares BF16 output with a PyTorch reference:

<Code
  src="../samples/python/oss_rmsnorm_silu.py"
  language="python"
  maxLines={0}
/>

Current source: [RMSNorm + SiLU engine](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/experimental/sm100_rms_norm_silu_engine.h#L118).

## Expected result

The published Graph sample selects the OSS RMSNorm-plus-SiLU plan and compares
BF16 output with PyTorch. The direct randomized-Hadamard wrapper has no
documentation-owned executable sample and needs separate output and amax
validation on the target.
