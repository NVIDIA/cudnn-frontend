# Optional OSS engine contracts

Use optional open-source kernels only when their exported wrapper, dependencies, architecture, layout, and fallback contract match the workload.

## Before you begin

Install the optional packages from the same frontend checkout and identify the
target GPU architecture. Establish a cuDNN Graph or framework reference before
replacing it with an OSS wrapper.

## Install optional dependencies

Install the `cutedsl` extra to obtain the pinned CUTLASS DSL package, CUDA Python, PyTorch, TVM FFI, and DLPack extension declared by this checkout. Keep the frontend, optional packages, CUDA runtime, and GPU architecture compatible.

## Choose an entry point

- `heur_mode.OPENSOURCE` lets the Graph backend consider registered OSS slots.
- Top-level optional exports provide fused SM100 GEMM, grouped GEMM, SDPA D=256, and RMSNorm kernels.
- `cudnn.BSA`, `cudnn.DSA`, and `cudnn.NSA` expose sparse-attention families.
- `cudnn.experimental.ops` provides PyTorch convenience operators with explicit experimental status.

The legacy OSS guide set covered these distinct families. Keep each item in the
inventory even when several wrappers share an implementation or dependency:

- Sparse attention: [BSA](/optional-kernels/oss-block-sparse-attention), [DSA](/optional-kernels/oss-deepseek-sparse-attention), and [NSA](/optional-kernels/oss-native-sparse-attention).
- Attention: [SDPA forward and backward with head dimension 256](/optional-kernels/oss-sdpa-d256).
- [Dense fused GEMM](/optional-kernels/oss-dense-gemm): amax, dReLU, projection plus RoPE MXFP8, ReLU, and SwiGLU variants.
- [Grouped GEMM](/optional-kernels/oss-grouped-gemm): dGLU, dReLU, dSwiGLU, GLU, GLU plus Hadamard, quantized, ReLU, SwiGLU, and weight-gradient variants.
- [Discrete grouped GEMM](/optional-kernels/oss-discrete-grouped-gemm): SwiGLU and dSwiGLU variants.
- [Normalization](/optional-kernels/oss-normalization): RMSNorm plus randomized Hadamard transform and amax. The former RMSNorm+SiLU package is absent from the current checkout.

## Validate support before execution

Inspect the package `__all__`, wrapper validation, test shapes, and architecture checks. Optional imports are lazy and raise an installation hint when dependencies are absent. Do not document implementation helpers that are not exported, and do not assume every exported wrapper supports the same stream or allocation contract.

## Preserve fallback boundaries

For Graph planning, keep a cuDNN backend heuristic entry when fallback is desired. Direct wrapper calls have no automatic Graph fallback unless their source contract implements one.

## Expected result

An imported wrapper accepts the exact deployed shape, dtype, layout, and
architecture and matches the reference result. Unsupported optional imports or
kernel contracts leave the ordinary Graph path available as an explicit
fallback; direct wrapper calls do not acquire one automatically.
