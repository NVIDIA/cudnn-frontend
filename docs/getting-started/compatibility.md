# Compatibility and feature availability

This matrix centralizes comparison data. Each API reference page repeats all constraints that apply to that API.

| Feature | Status | cuDNN | CUDA | GPU | Platforms | Backends | Languages | Dependencies | Verification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Convolution forward and gradients | Stable | See selected engine support | See installed cuDNN release | Engine-dependent | Linux (executed), Windows (not executed) | cuDNN backend | Python, C++ | cuDNN, CUDA | Forward, data-gradient, and weight-gradient samples executed in Python and C++ on SM100 |
| Core Graph operations and lifecycle | Stable | 8.5+ for the C++ frontend; Python wrapper paths require 9.12+ | A CUDA toolkit supported by the installed cuDNN release | Architecture supported by the selected cuDNN engine | Linux (executed), Windows (source-supported, not executed) | cuDNN backend | Python, C++ | cuDNN, CUDA | Python and C++ representative samples executed with CUDA 13, cuDNN 9.15.1, and SM100 |
| FP8, MXFP8, NVFP4, and block scaling | Stable or experimental by operation | Operation-dependent | CUDA 12 or later; CUDA 13 used for this checkout | Hopper for FP8; Blackwell for MXFP8/NVFP4 and SM100 kernels | Linux (partially executed), Windows (not executed) | cuDNN backend | Python, C++ | cuDNN, CUDA | C++ FP8 and MXFP8 attention samples executed on SM100; unsupported block-scaling paths are reported locally |
| Legacy C++ builder API | Legacy | See individual descriptor and operation | See installed cuDNN release | Operation-dependent | Linux (header inventory), Windows (not executed) | cuDNN backend | C++ | cuDNN, CUDA | Header inventory only |
| Normalization families | Stable | Operation-dependent | See installed cuDNN release | Engine-dependent | Linux (executed), Windows (not executed) | cuDNN backend | Python, C++ | cuDNN, CUDA | LayerNorm and RMSNorm samples executed in Python and C++; additional C++ normalization samples executed on SM100 |
| Open-source CUTLASS DSL engines | Experimental | Package-specific | CUDA 13 for the current optional dependency set | SM90-SM120 depending on the package; most GEMM families require SM100+ | Linux (imports and one normalization kernel executed; other kernels source-verified), Windows (not claimed) | CUTLASS DSL OSS engines, cuDNN graph integration | Python, C++ engine selection | nvidia-cutlass-dsl[cu13], PyTorch, cuda-python, apache-tvm-ffi, torch-c-dlpack-ext | All current optional exports imported with installed CUTLASS DSL 4.5.2 on SM100, although the project extra pins 4.5.0; the BF16 RMSNorm+SiLU Graph fusion executed and matched a PyTorch reference; other kernels remain source-verified unless an owning page records a numerical run |
| Python IR routing and custom engines | Unreleased | 9.12+ when falling back to the backend | CUDA 13 tested for this checkout | Depends on the selected engine | Linux (executed), Windows (not executed) | Python engine, cuDNN fallback | Python | Python engine implementation | Automatic manifest discovery, ranked selection, and backend fallback are covered by the engine-selection and core lifecycle samples |
| SDPA and RoPE | Stable with option-specific constraints | Option and implementation dependent | See installed cuDNN release | SM80+ with narrower limits by implementation | Linux (executed), Windows (not executed) | cuDNN backend | Python, C++ | cuDNN, CUDA | SDPA, masking, cache, dropout, and selected backward samples executed on SM100; standalone RoPE samples were compiled or syntax-checked because they require cuDNN 9.24+ |

## Read the labels

**Stable** identifies a supported public surface. **Experimental** may change without the normal compatibility guarantees. **Unreleased** exists in this checkout but is not claimed for an earlier package release. **Legacy** remains for migration and compatibility.

## Verification limits

- Windows behavior and architecture-specific OSS ranges are source-verified but were not executed in the recorded Linux Blackwell environment.
- The recorded environment used CUTLASS DSL 4.5.2 while `pyproject.toml` pins 4.5.0. Successful imports verify the observed environment, not the pinned extra.
