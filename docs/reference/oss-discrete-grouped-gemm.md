# OSS discrete-weight grouped GEMM contracts

Use discrete grouped GEMM when expert weights and scale factors are supplied as device pointer arrays rather than one contiguous expert tensor. The current checkout exports forward GLU and backward dGLU families for SM100+.

## Before you begin

Install the `cutedsl` extra, use SM100 or newer, and build device pointer arrays
whose pointed-to expert buffers remain alive through stream completion. Keep a
per-expert dense reference and validated padded offsets.

## Build pointer arrays

`b_ptrs` and `sfb_ptrs` are CUDA tensors containing device addresses for each expert's B and SFB allocation. The pointed-to buffers must remain alive until the asynchronous kernel completes. Every expert must use the declared N, K, dtype, major order, and scale layout. Pointer-array storage itself must have the dtype and alignment expected by the wrapper.

Use monotonic `padded_offsets` to map contiguous A rows to experts. `alpha` provides per-expert scaling. Cache identity includes the declared weight shape and dtype, but pointer values may change between compatible executions.

## discrete_grouped_gemm_swiglu_wrapper_sm100

`cudnn.discrete_grouped_gemm_swiglu_wrapper_sm100` computes pointer-based grouped GEMM followed by the selected GLU activation. Required inputs are A, B pointer array, SFA, SFB pointer array, padded offsets, alpha, N, and B dtype. Optional bias, normalization constant, probability, activation parameters, and scheduling affect the epilogue.

The result includes the pre-activation C and GLU output D. Floating outputs may include amax; quantized outputs include row and column scale factors. The pre-activation N dimension must be compatible with the two GLU halves.

### DiscreteGroupedGemmSwigluSm100

The class form is `cudnn.DiscreteGroupedGemmSwigluSm100`.

## discrete_grouped_gemm_dswiglu_wrapper_sm100

`cudnn.discrete_grouped_gemm_dswiglu_wrapper_sm100` computes the pointer-based GLU backward fusion. It consumes A, B pointers, C, SFA, SFB pointers, offsets, alpha, beta, routing probability, and its gradient. Set `generate_dbias=True` only when the selected activation and layout support bias-gradient generation.

Use the same `act_func`, linear offset, GeGLU alpha, clamp bounds, N, weight dtype, weight major order, and scale layout as forward. Results include row- and column-oriented gradient data plus optional scale, amax, probability-gradient, and bias-gradient tensors according to dtype and flags.

### DiscreteGroupedGemmDswigluSm100

The class form is `cudnn.DiscreteGroupedGemmDswigluSm100`.

## Validate before launch

Both wrappers require SM100+, FP32 accumulation, legal low-precision operand types, aligned dimensions, compatible scale-factor shapes, and a supported MMA tile and cluster. Device pointer arrays create no ownership: freeing one expert allocation before stream completion is a use-after-free.

Current package exports: [discrete grouped GEMM namespace](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/discrete_grouped_gemm/__init__.py#L1).

## Expected result

Forward and backward outputs match the same expert boundaries and activation
parameters as the dense reference, and changing compatible pointer values does
not change specialization identity. No executable documentation sample is
published, so pointer lifetime and numerical checks remain deployment gates.
