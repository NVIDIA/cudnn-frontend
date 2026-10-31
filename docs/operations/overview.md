# Operations Overview

The API reference groups graph operations and experimental Python kernels by
operation. Use the sidebar to find an operation and its related APIs.

- [Attention](/operations/Attention)
- [Sparse Attention](/operations/sparse_attention)
- [Linear Attention](/operations/LinearAttention)
- [Matmul and GEMM Fusions](/operations/Matmul)
- [Convolutions](/operations/Convolutions) and [Causal Conv1d](/operations/CausalConv1d)
- [Normalizations](/operations/Normalizations)
- [Block Scaling](/operations/BlockScaling)
- [RoPE](/operations/RoPE)
- [Pointwise and Reduction](/operations/Pointwise)

## Experimental Python APIs

**FE-OSS APIs are experimental and subject to change.**

The GEMM CuTeDSL APIs are type-erased and torch-lazy: torch is imported only when torch tensors are passed. JAX arrays are additionally accepted wherever the kernel's tensor layouts are expressible as row-major arrays (each API's page has a "JAX support" section with its exact contract):

- **Dense fusions** (amax, swiglu, srelu, dsrelu): full JAX eager support, plus `jax.jit`-compatible XLA custom-call entry points for all four (built on `cudnn.jax.call` / CuTeDSL's native `cutlass.jax` bridge; see `gemm_amax.md` "Using JAX arrays").
- **Grouped / discrete-grouped**: JAX eager support in discrete (pointer-array) weight modes — unfused grouped GEMM, glu/dglu (BF16), dsrelu (FP8), wgrad (BF16), and discrete-grouped swiglu/dswiglu (FP8) — plus a `jax.jit`-compatible `*_jax_sm100` entry point for each of those same families (built on `cudnn.jax.call`; each API page documents its exact jit contract). Contiguous grouped MXFP8 SwiGLU/dSwiGLU wrappers accept Torch tensors and canonical JAX arrays, including under `jax.jit`, with explicit FP8 output dtype and `sf_vec_size=32` for JAX. The unified GLU wrapper likewise dispatches canonical MXFP8 JAX inputs to `grouped_gemm_glu_jax_sm100` (SwiGLU/GeGLU), which runs the Rubin GLU kernel on SM107. The unified dGLU, wgrad and quant wrappers likewise dispatch canonical JAX inputs to `grouped_gemm_dglu_jax_sm100` (dSwiGLU/dGeGLU), `grouped_gemm_wgrad_jax_sm100` and `grouped_gemm_quant_jax_forward`, which run the Rubin kernels on SM107. The grouped SReLU, dSReLU, GLU+Hadamard and GLU+Hadamard+quant wrappers dispatch to `grouped_gemm_srelu_jax_forward`, `grouped_gemm_dsrelu_jax_sm100`, `grouped_gemm_glu_hadamard_jax_forward` and `grouped_gemm_glu_hadamard_quant_jax_forward` (SM100+, as in Torch; GLU+Hadamard+quant runs its Rubin kernel on SM107). These canonical entries take dense weights in MXFP8 or packed FP4 (as `uint8`, with E4M3 scales per 16 for NVFP4), and GLU, dGLU and wgrad also take BF16; each page lists its formats. Existing Torch behavior and wrapper defaults are unchanged. Discrete block-scaled weights, column-major bias layouts and the op classes reject JAX with clear errors.
- **proj_rope_mxfp8**: JAX eager support on both input paths with `w_out_in=True` (the transposed [in, out] weight view is torch-only), plus the `jax.jit`-compatible `gemm_proj_rope_mxfp8_jax_sm100` entry point.

New APIs follow the [Python API Naming Conventions](/operations/api_conventions).
The experimental API catalog includes:
- [Causal Conv1d](/operations/CausalConv1d/causal_conv1d) and [Decode Update](/operations/CausalConv1d/causal_conv1d_update)
- [FLA Integration Shims](/operations/LinearAttention/fla)
- [Kimi Delta Attention in JAX](/operations/LinearAttention/kda_jax)
- [GEMM + Amax](/operations/Matmul/gemm_fusions/gemm_amax)
- [GEMM + RoPE + MXFP8 Projection](/operations/Matmul/gemm_fusions/gemm_proj_rope_mxfp8)
- [Gated Attention Block (SM107)](/operations/Attention/gated_attention_block) — projection, QK-norm + RoPE, SDPA, sigmoid gate, out projection as one FROST block (bf16 / FP8 / MXFP8, optional MXFP4 weights and NVFP4 / MXFP4 output); index-list sparse attention (`QsaSpec`) with the in-block indexer's decode and extend steps over the paged side caches, and write-through into a paged KV cache
- [Triangle Multiplicative Update (SM100 / SM103 / SM107)](/operations/Matmul/triangle_multiplicative_update) — the AlphaFold-class outgoing / incoming triangle multiplication as three FROST launches behind cuEquivariance's `triangle_multiplicative_update` signature (bf16)
- [Triangle Attention (SM107)](/operations/Attention/triangle_attention) — the AlphaFold-class attention with a pair bias and a key mask (right-padded lengths or a dense boolean mask) on the FROST SDPA engine behind cuEquivariance's `triangle_attention` signature (bf16 / fp16, D &lt;= 64)
- [Tail RoPE + Microscaled QDQ](/operations/RoPE/rope-qdq)
- [Prepared BF16 Tail RoPE](/operations/RoPE/rope-tail)
- [GEMM + SwiGLU](/operations/Matmul/gemm_fusions/gemm_swiglu)
- [GEMM + sReLU](/operations/Matmul/gemm_fusions/gemm_srelu)
- [GEMM + dsReLU](/operations/Matmul/gemm_fusions/gemm_dsrelu)
- [Grouped GEMM (BF16)](/operations/Matmul/gemm_fusions/grouped_gemm)
- [Grouped GEMM + GLU (Unified)](/operations/Matmul/gemm_fusions/grouped_gemm_glu)
- [Grouped GEMM + GLU + Hadamard](/operations/Matmul/gemm_fusions/grouped_gemm_glu_hadamard)
- [Grouped GEMM + GLU + Hadamard + Quant](/operations/Matmul/gemm_fusions/grouped_gemm_glu_hadamard_quant)
- [Grouped GEMM + dGLU (Unified)](/operations/Matmul/gemm_fusions/grouped_gemm_dglu)
- [Grouped GEMM + SwiGLU (Legacy, Contiguous-only)](/operations/Matmul/gemm_fusions/grouped_gemm_swiglu)
- [Grouped GEMM + dSwiGLU (Legacy, Contiguous-only)](/operations/Matmul/gemm_fusions/grouped_gemm_dswiglu)
- [Grouped GEMM + sReLU (Unified)](/operations/Matmul/gemm_fusions/grouped_gemm_srelu) — optionally tanh
  soft-clamped via `tanh_clamp_scale`
- [Grouped GEMM + dsReLU (Unified)](/operations/Matmul/gemm_fusions/grouped_gemm_dsrelu)
- [Discrete Grouped GEMM + SwiGLU](/operations/Matmul/gemm_fusions/discrete_grouped_gemm_swiglu)
- [Discrete Grouped GEMM + dSwiGLU](/operations/Matmul/gemm_fusions/discrete_grouped_gemm_dswiglu)
- [Grouped GEMM + Quant (Legacy, Dense-only)](/operations/Matmul/gemm_fusions/grouped_gemm_quant)
- [Grouped GEMM + Quant (Unified)](/operations/Matmul/gemm_fusions/grouped_gemm_quant_unified)
- [Grouped GEMM + Wgrad](/operations/Matmul/gemm_fusions/grouped_gemm_wgrad)
- [Block Sparse Attention (BSA)](/operations/sparse_attention/bsa)
- [Query Sparse Attention (QSA): attention and indexer entry points](/operations/sparse_attention/qsa)
- [QSA training from existing token indices](/operations/sparse_attention/qsa/qsa_indexed)
- [QSA two-pass backend implementation notes](/operations/sparse_attention/qsa/qsa_two_pass)
- [DeepSeek Sparse Attention (DSA)](/operations/sparse_attention/dsa)
- [Flex Attention](/operations/sparse_attention/flex_attention) and [mask plan design](/operations/sparse_attention/flex_attention/flex_attention_design)
- [HSTU Attention (Blackwell SM100/SM103)](/operations/sparse_attention/hstu_attention)
- [HSTU LayerNorm-Multiply-SiLU-Dropout (LMSD)](/operations/sparse_attention/hstu_lmsd)
- [Native Sparse Attention (NSA)](/operations/sparse_attention/nsa)
- [CSA Fused Compressor](/operations/sparse_attention/csa)
- [Aligned HCA Backward](/operations/sparse_attention/csa/aligned_hca_backward)
- [DSv4.1 Vision RoPE Backward](/operations/RoPE/vision_rope_backward)
- [Engram Saved-State Gate](/operations/Pointwise/engram_saved_gate)
- [RMSNorm + RHT + Amax](/operations/Normalizations/rmsnorm_rht_amax)
- [SDPA Backward, d = 128, dense and THD MHA/GQA/MQA (SM100 / SM103)](/operations/Attention/sdpa_bwd_sm100_d128)
- [SDPA Backward (SM120)](/operations/Attention/sdpa_bwd_sm120)
- [SDPA Backward, d = 256 and d in (256, 512] (SM107 / Rubin; SM100 / SM103 via the 2x2-datapath body)](/operations/Attention/sdpa_bwd_sm107)
- [Compact GQA Backward (SM107)](/operations/Attention/compact_gqa_backward)
- [NVFP4 Attention QAT Backward](/operations/Attention/nvfp4_attention_qat_backward)
- [RMSNorm + SiLU](/operations/Normalizations/rmsnorm_silu)
- [Fused Conv3D + Bias + Residual + RMSNorm + SiLU](/operations/Convolutions/conv3d_postops)
- [NVFP4 Block-Scale Conversion](/operations/BlockScaling/nvfp4_block_scale_conversion)
- [DSv4.1 mHC projection/RMS backward](/operations/Matmul/gemm_fusions/mhc_projection_bwd)

## Installation and setup

All Frontend OSS APIs come installed with the `nvidia-cudnn-frontend` package. Except on Windows ARM64, `nvidia-cutlass-dsl[cu13]>=4.8.0` and `apache-tvm-ffi` are required dependencies, so the CuTeDSL runtime comes with the base install (and the DSL pulls `cuda-python` in transitively):
```bash
pip install nvidia-cudnn-frontend
```
On Windows ARM64, these two runtime dependencies are excluded from the base installation, so the base install does not enable the CuTeDSL APIs there.

(`pip install nvidia-cudnn-frontend[cutedsl]` still works; the `cutedsl` extra now names only `cuda-python`. A few APIs still want extras of their own — the cuTile linear-attention engines need `[cutile]`.)

The Triton NVFP4 attention QAT backward API additionally requires the
`triton` extra. Its API page documents the framework and GPU requirements.

Those required dependencies are framework-neutral. Install your tensor framework separately — from a checkout, the PEP 735 dependency groups pin the right companion packages:
```bash
pip install --group torch   # torch + torch-c-dlpack-ext
pip install --group jax     # jax >= 0.5 (XLA entry points via cutlass.jax, shipped with nvidia-cutlass-dsl)
```
(For the published wheel, `pip install torch torch-c-dlpack-ext` or `pip install "jax>=0.5"` directly.)

After installation, you can import the APIs directly from the `cudnn` package, i.e. `from cudnn import {your_operation}`

## API Usage

The causal-convolution family exposes semantic PyTorch operations only through
`cudnn.ops.causal_conv1d` and `cudnn.ops.causal_conv1d_update`; they return
ordinary tensors and keep prepared kernel objects private. The wrapper and
class conventions below apply to prepared frontend-only kernel APIs.

Most compiler-style operations expose the following two APIs. Functional
autograd integrations such as BSA and Flex Attention instead document their
own wrapper and reusable-plan lifecycle on their operation pages.

### 1. High-level wrapper

- Single pythonic function call
- Allocates and returns output tensors
- Returns outputs as a `TupleDict` (supports both dictionary-style key access and tuple unpacking)
- No explicit compilation step – internally caches compiled kernels via a simple dictionary lookup
- When to use:
  - Fast prototyping and common cases
  - You want automatic allocation and minimal boilerplate
  - You are okay with the library managing the compiled-kernel cache

```python
from cudnn import {your_operation}_wrapper
result = {your_operation}_wrapper(
    inputs,
    ...,
    config_options,
    ...,
    stream=None,
)

# Dictionary-style access (recommended)
primary_output = result["output_tensor_name"]

# Tuple unpacking (order follows documented wrapper output keys)
out0, out1 = result
```

### 2. Class API

- Explicit lifecycle with compile and execute steps
- Reusable object with underlying compiled kernel for multiple executions
- Requires preallocated output tensors
- When to use:
  - You need to reuse a compiled kernel across many calls
  - You want explicit control over compilation and lifecycle management
  
```python
from cudnn import {your_operation}

op = {your_operation}(
    sample_inputs,
    ...,
    sample_outputs,
    ...,
    config_options,
    ...
)
op.compile()
op.execute(
    inputs,
    ...
    outputs,
    ...
    current_stream=None,
)
```
Methods:
- `check_support()` – validates target problem configuration (i.e. tensor shapes, tensor strides, dtypes, tiling/cluster/kernel configurations, environment, etc.)
- `compile()` – compiles the kernel with the provided sample tensors and parameters.
- `execute(inputs, ..., outputs, ..., current_stream)` – runs the kernel with the provided inputs and outputs.
  
## Common Parameters and Conventions

- CUDA stream (`current_stream` in class API, `stream` in wrapper)
  - The cuda stream to use for operation kernel execution.
  - Default: None (uses default stream)

- Determinism
  - Several kernels reduce with cross-CTA atomics, so those outputs are not bit-exact run to
    run. Kernels react to `torch.use_deterministic_algorithms(True)`: they switch to a
    deterministic path where one exists, and otherwise raise (or warn, under
    `warn_only`) rather than silently return a non-reproducible result.
  - Where a deterministic path exists it can also be selected per call with
    `deterministic=True`, independent of the torch setting. See
    [Grouped GEMM + dsReLU](/operations/Matmul/gemm_fusions/grouped_gemm_dsrelu#deterministic-dprob).


## File structure and examples

- All FE OSS APIs are implemented in the `python/cudnn` directory.
- Correctness tests/samples are implemented under `test/python/<operation>/cutedsl` (JAX bindings under `test/python/<operation>/jax`).
