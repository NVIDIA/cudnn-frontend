# OSS grouped GEMM fusion contracts

Use grouped GEMM wrappers for contiguous Mixture-of-Experts activations with per-expert weights and expert boundary metadata. These SM100+ kernels use a different tensor contract from the cuDNN Graph `moe_grouped_matmul` operation.

## Before you begin

Install the `cutedsl` extra, use SM100 or newer, and validate expert offsets,
aligned row counts, scale layouts, and per-expert dense references. Do not reuse
the Graph `moe_grouped_matmul` tensor contract for these direct wrappers.

## Describe expert groups

`padded_offsets` contains cumulative, aligned M boundaries for each expert. Its final entry defines the padded activation extent used by the kernel, while the logical routed-token count determines valid rows. Offsets must be monotonic and compatible with `m_aligned`. `alpha` contains per-expert scaling values.

Contiguous-weight forms use B and SFB tensors with an expert dimension. Unified forms may instead use device pointer arrays `b_ptrs` and `sfb_ptrs` together with explicit N, dtype, and major order. Do not pass both representations unless the individual wrapper documents a precedence rule.

The common scale-factor layout follows the selected MMA atom: blocks of 128 matrix elements are subdivided into 32-by-4 scale tiles, with the K groups determined by `sf_vec_size`. Use the source validator rather than deriving strides from a dense logical shape alone.

## grouped_gemm_wrapper_sm100

The neutral BF16 wrapper computes an unfused grouped GEMM with optional bias, routing-probability scaling, and materialized intermediate C. It accepts dense stacked expert weights or device-pointer arrays, supports static or dynamic tile scheduling, accumulates in FP32, and returns D followed by optional C.

Rows and cumulative expert offsets are 256-aligned. A is BF16 K-major; C and D may be BF16, FP16, or FP32. Dense B is BF16 K-major. Pointer-array mode accepts K-major or N-major expert weights and makes the caller responsible for the lifetime and alignment of every referenced allocation.

### GroupedGemmSm100

The class form is `cudnn.GroupedGemmSm100`. Construct it with representative tensors, call `check_support()` and `compile()` once, then reuse it only with tensors that match the compiled dtype, layout, expert-count, scheduler, and output policy.

## grouped_gemm_swiglu_wrapper_sm100

Computes grouped block-scaled GEMM followed by SwiGLU. A is the routed activation matrix, B is the expert weight tensor, and SFA/SFB carry block scales. The pre-activation N dimension is split in half. Optional `prob` applies per-row routing probability. Floating outputs use amax; quantized outputs use row and column scale factors.

### GroupedGemmSwigluSm100

The class form is `cudnn.GroupedGemmSwigluSm100`.

## grouped_gemm_dswiglu_wrapper_sm100

Computes the grouped SwiGLU backward fusion. It consumes the forward activation data and upstream gradient needed to produce the derivative with respect to the GEMM result and any enabled auxiliary outputs. Preserve expert offsets, activation mode, scale layout, and probability metadata from forward.

### GroupedGemmDswigluSm100

The class form is `cudnn.GroupedGemmDswigluSm100`.

## grouped_gemm_srelu_wrapper_sm100

Computes grouped GEMM followed by ReLU, optional bias, optional probability scaling, and optional output quantization. The wrapper accepts contiguous or pointer-array weights in the current unified implementation.

### GroupedGemmSreluSm100

The class form is `cudnn.GroupedGemmSreluSm100`.

## grouped_gemm_dsrelu_wrapper_sm100

Computes the corresponding grouped ReLU backward fusion. Forward and backward output scale tensors are separate data products; do not reuse a forward quantization buffer as a gradient scale buffer.

### GroupedGemmDsreluSm100

The class form is `cudnn.GroupedGemmDsreluSm100`.

## grouped_gemm_glu_wrapper_sm100

Computes the generalized GLU forward fusion. `act_func` selects the supported GLU variant, including SwiGLU-compatible behavior. Optional linear offset, GeGLU alpha, clamp range, bias, normalization constant, probability, and dynamic scheduling all change behavior or compilation identity.

### GroupedGemmGluSm100

The class form is `cudnn.GroupedGemmGluSm100`.

## grouped_gemm_dglu_wrapper_sm100

Computes generalized GLU backward. Use the same activation function, linear offset, GeGLU coefficient, and clamp bounds as forward. The result may include dA-like data, probability gradients, and bias gradients according to enabled inputs.

### GroupedGemmDgluSm100

The class form is `cudnn.GroupedGemmDgluSm100`.

## grouped_gemm_glu_hadamard_wrapper_sm100

Adds a randomized Hadamard transform after the GLU fusion and reports pre- and post-transform amax values where requested. The Hadamard tensor and transform size must match the output feature dimension.

### GroupedGemmGluHadamardSm100

The class form is `cudnn.GroupedGemmGluHadamardSm100`.

## grouped_gemm_glu_hadamard_quant_wrapper_sm100

Adds optional rowwise or colwise randomized Hadamard output and NVFP4 quantization to the grouped GLU/SReLU fusion. FP4 A and B use packed E2M1 values with E4M3 or E8M0 block scales. D and rowwise RHT may be BF16 or packed NVFP4; colwise RHT is NVFP4-only and uses flat per-expert transposed storage.

This path requires SM100+, `m_aligned=256`, `N` divisible by 64, FP4 K alignment, and `N_out` divisible by 32. NVFP4 output additionally requires `N_out` divisible by 128 and is not supported with SReLU. At most one RHT direction may be selected. Dense and pointer-array weight modes are mutually exclusive.

### GroupedGemmGluHadamardQuantSm100

The class form is `cudnn.GroupedGemmGluHadamardQuantSm100`. JAX supports the dense canonical FP4 path only; bias, dynamic scheduling, pointer-array weights, and the class API remain unsupported for JAX.

## grouped_gemm_quant_wrapper_sm100

Runs grouped block-scaled GEMM with output quantization but without a fixed activation family. It accepts contiguous or pointer-array weights, optional bias, probability, normalization constant, row scale, and caller-provided D. `discrete_col_sfd` controls the column-scale representation and must match downstream consumption.

Alternative weight representations on this wrapper provide the unified quantized behavior; there is no separately exported `grouped_gemm_quant_unified` API in the current checkout.

### GroupedGemmQuantSm100

The class form is `cudnn.GroupedGemmQuantSm100`.

## grouped_gemm_wgrad_wrapper_sm100

Computes per-expert weight gradients from grouped A and B-like operands. `output_mode="dense"` returns one dense expert tensor; pointer output mode writes through `wgrad_ptrs`. `input_order`, scale vector size, global scales, output dtype, and `accumulate_on_output` must match the caller's accumulation strategy.

### GroupedGemmWgradSm100

The class form is `cudnn.GroupedGemmWgradSm100`.

## Validate common constraints

All wrappers require SM100+, FP32 accumulation, valid CUTLASS DSL dtypes, compatible block-scale tensors, aligned K and N extents, and a legal MMA tile and cluster. The current forward, activation-backward, and quantization families require `m_aligned=256`; weight-gradient APIs instead use their own input and output offset contract. Optional FP4, FP8, BF16, FP16, and FP32 outputs are not supported in every combination. Always treat the individual wrapper validator as authoritative and record the chosen options in the graph or kernel cache key.

Current package exports: [grouped GEMM namespace](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/gemm/cutedsl/grouped/__init__.py#L1).

## Expected result

Every expert slice and requested epilogue matches its dense reference, including
routing probability and gradient outputs where present. This page records the
source contract but publishes no executable documentation sample, so support
and accuracy must be verified for the selected wrapper and target.
