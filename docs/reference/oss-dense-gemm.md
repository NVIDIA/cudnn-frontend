# OSS dense GEMM fusion contracts

Use the optional dense GEMM wrappers for Blackwell block-scaled matrix products whose epilogue exactly matches one of the exported fusions. These wrappers call CUTLASS DSL kernels directly and do not inherit cuDNN Graph fallback behavior.

## Before you begin

Install the `cutedsl` extra, use SM100 or newer, and establish a dense
higher-precision reference. Derive the scale-factor layout with the selected
wrapper's utilities instead of treating it as an ordinary matrix.

## Prepare block-scaled operands

All five families require an SM100+ CUDA device, but they do not share one dtype contract. Amax, sReLU, and dsReLU are block-scaled paths. SwiGLU selects a dense path when SFA/SFB are absent and a block-scaled path when both are present. Projection+RoPE accepts fixed BF16 inputs and produces MXFP8 outputs. For block-scaled calls, scale tensors must use the layout computed for `sf_vec_size`, the MMA tile, and operand dimensions; they are not ordinary dense `[M, K]` tensors.

Treat `mma_tiler_mn`, `cluster_shape_mn`, scale vector size, major-order flags, output dtype, and optional two-CTA scheduling as compilation inputs. The common cluster rule is positive dimensions no greater than four with total cluster size no greater than 16; individual kernels can impose stricter combinations.

## gemm_amax_wrapper_sm100

`cudnn.gemm_amax_wrapper_sm100` computes a block-scaled GEMM and returns the matrix output plus its absolute maximum. Supply A, B, SFA, and SFB with compatible shapes. Choose output dtype and layout through the wrapper options. The amax result is FP32 and must not be interpreted as a per-row scale.

Signature: `gemm_amax_wrapper_sm100(a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_major="n", c_dtype=torch.float32, acc_dtype=torch.float32, mma_tiler_mn=(128, 128), cluster_shape_mn=(1, 1), sf_vec_size=32, stream=None)`. A and B are rank-3 `[M, K, L]` and `[N, K, L]` with the same FP4 or FP8 dtype. `sf_vec_size` is 16 or 32 subject to the input/scale dtype combination. Accumulation is FP32; C can be FP32, FP16, BF16, FP8, or compatible FP4. Cluster dimensions must be powers of two in `{1, 2, 4}`. The current validator rejects an M tile of 256 and several additional launch combinations.

### GemmAmaxSm100

Use `cudnn.GemmAmaxSm100` when outputs are preallocated. Its constructor compiles from representative inputs and its `execute` method writes C and amax on the selected stream.

## gemm_srelu_wrapper_sm100

`cudnn.gemm_srelu_wrapper_sm100` applies ReLU after the GEMM and may quantize the activated result. `prob_tensor` is a required FP32 `[M, 1, L]` input; `norm_const_tensor` is optional. The result mapping contains `c_tensor`, `d_tensor`, and optional `sfd_tensor` or `amax_tensor`.

Signature: `gemm_srelu_wrapper_sm100(a_tensor, b_tensor, sfa_tensor, sfb_tensor, prob_tensor, alpha=1.0, c_major="n", c_dtype=torch.bfloat16, d_dtype=torch.bfloat16, acc_dtype=torch.float32, mma_tiler_mn=(256, 256), cluster_shape_mn=None, norm_const_tensor=None, sf_vec_size=16, vector_f32=False, stream=None)`. A and B use the same FP4 or FP8 dtype; SFA and SFB use the same E8M0 or E4M3 scale dtype. FP8 D requires both an allocated SFD and `norm_const_tensor`; the wrapper allocates SFD. FP4 input with floating D produces FP32 amax. C and D share the selected major order.

### GemmSreluSm100

`cudnn.GemmSreluSm100` is the caller-allocation class form.

## gemm_dsrelu_wrapper_sm100

`cudnn.gemm_dsrelu_wrapper_sm100` implements the backward derivative for the ReLU fusion. It consumes required A, B, forward C, SFA, SFB, and FP32 probability tensors. It returns `d_tensor`, FP32 `dprob_tensor`, and optional `sfd_tensor` or `amax_tensor`.

Signature: `gemm_dsrelu_wrapper_sm100(a_tensor, b_tensor, c_tensor, sfa_tensor, sfb_tensor, prob_tensor, alpha=1.0, d_major="n", d_dtype=torch.bfloat16, acc_dtype=torch.float32, mma_tiler_mn=(256, 256), cluster_shape_mn=None, norm_const_tensor=None, sf_vec_size=16, vector_f32=False, stream=None)`. C must be floating FP16, BF16, or FP32. D can additionally be FP8, which requires the wrapper-created SFD and an explicit normalization constant. Use the same shape, scale-vector size, alpha, and probability data as forward.

### GemmDsreluSm100

`cudnn.GemmDsreluSm100` is not interchangeable with the forward class: its input roles, output gradients, and epilogue validation are different.

## gemm_swiglu_wrapper_sm100

`cudnn.gemm_swiglu_wrapper_sm100` splits the GEMM output into gate and value halves and applies a SwiGLU epilogue. The GEMM N dimension must therefore be compatible with an even split. It returns full-width `ab12_tensor`, half-width `c_tensor`, and optional `sfc_tensor` or `amax_tensor`.

Signature: `gemm_swiglu_wrapper_sm100(a_tensor, b_tensor, alpha=1.0, c_major="n", ab12_dtype=torch.float32, c_dtype=torch.float16, acc_dtype=torch.float32, mma_tiler_mn=(128, 128), cluster_shape_mn=None, sfa_tensor=None, sfb_tensor=None, norm_const_tensor=None, sf_vec_size=16, vector_f32=False, ab12_stages=4, stream=None)`. Omit both scale tensors for the dense path, which accepts matching FP16, BF16, FP32, or FP8 A/B. Supply both for the block-scaled FP4/FP8 path. FP8 C requires SFC and `norm_const_tensor`; FP4 input with BF16 C produces amax. Layouts require 16-byte alignment, and the block-scaled path aligns M and N to its MMA tile.

### GemmSwigluSm100

`cudnn.GemmSwigluSm100` exposes explicit compile and execute phases with caller-provided C, D, scale, and amax buffers.

## gemm_proj_rope_mxfp8_wrapper_sm100

`cudnn.gemm_proj_rope_mxfp8_wrapper_sm100` fuses projection GEMM, per-head RoPE, and dual-direction MXFP8 quantization. It requires projection and frequency metadata that agree on token count, head count, head dimension, and rotated dimension. Its result contains the row- and column-oriented quantized Q representations and corresponding scale factors.

Signature: `gemm_proj_rope_mxfp8_wrapper_sm100(x, w, cos, sin, w_out_in=False, stream=None)`. X is BF16 `[tokens, Q_LORA]`. W is BF16 `[Q_LORA, heads * 192]`, or its transpose when `w_out_in=True`. Cos and sin are BF16 `[tokens, 64]`. Tokens must be a multiple of 128. Outputs are E4M3 `[tokens, heads, 192]` data plus E8M0 scales in rowwise `[tokens, heads, 6]` and columnwise `[tokens / 32, heads, 192]` layouts.

This fusion has a narrower contract than calling Graph matmul, RoPE, and block scaling separately. Use the composed Graph path when the model needs a different RoPE layout, dtype, or materialized intermediate.

### GemmProjRopeMxfp8Sm100

`cudnn.GemmProjRopeMxfp8Sm100` owns the compiled specialization and accepts caller-provided projection, frequency, quantized output, and scale buffers at execution.

## Select wrapper or class form

Wrappers allocate outputs and return a `TupleDict` that supports named and positional access. Class constructors accept sample tensors and compile a specialization; `execute` requires caller-owned outputs with identical metadata. Do not share one class instance across devices or incompatible streams unless its source contract explicitly permits it.

Current sources: [`gemm_amax`](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/gemm_amax/api.py#L18), [`gemm_srelu`](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/gemm_srelu/api.py#L32), [`gemm_dsrelu`](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/gemm_dsrelu/api.py#L32), [`gemm_swiglu`](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/gemm_swiglu/api.py#L49), and [`gemm_proj_rope_mxfp8`](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/gemm_proj_rope_mxfp8/api.py#L22).

## Expected result

A supported wrapper matches the dense reference within the tolerance selected
for its quantization and epilogue, and class-form reuse preserves all
specialization metadata. This page has no executable documentation sample, so
run those checks before replacing a Graph fallback.
