# Fused RMSNorm + SiLU

**This is an experimental API and subject to change.**

## Overview

The Fused RMSNorm + SiLU engine implements a single-kernel fusion of RMS normalization followed by SiLU (Swish) activation. It is designed and optimized specifically for the WAN VAE decoder's L2Norm + SiLU pattern on B200, but supports arbitrary problem sizes on SM80 to SM103 GPUs.

The engine uses a persistent RMSNorm kernel compiled at runtime via NVRTC. For SM100 (Blackwell) with known VAE problem sizes, sweep-tuned optimal knob configurations are used. For other architectures or problem sizes, a conservative fallback heuristic selects valid kernel parameters.

### Fusion Pattern

```text
Input (X) ──→ RMSNorm(X, scale, ε) ──→ SiLU(Y) ──→ Output (Z)
```

The cuDNN graph API detects this pattern automatically when an `rmsnorm` node (inference phase) feeds directly into a `swish` node.

---

## Hardware Requirements

| GPU Architecture | SM | bf16 Output | FP8 E4M3 Output | NVFP4 E2M1 Output |
|------------------|-----|-------------|------------------|--------------------|
| < SM80           | — | ❌ (not supported) | ❌ | ❌ |
| Ampere (A100)    | SM80 | ✅           | ❌                | ❌                  |
| Ada (L40S, 4090) | SM89 | ✅           | ✅                | ❌                  |
| Hopper (H100)    | SM90 | ✅           | ✅                | ❌                  |
| Blackwell (B200)  | SM100 | ✅           | ✅                | ✅                  |
| Blackwell (B300) | SM103 | ✅           | ✅                | ✅                  |
| > SM103           | — | ❌ (not yet validated) | ❌ | ❌ |

---

## Data Types

### Supported Input/Output Types

| Dtype | Input | Scale (gamma) | Output | Minimum SM |
|-------|-------|---------------|--------|------------|
| `bfloat16` | ✅ | ✅ | ✅ | SM80 |
| `float8_e4m3fn` | ❌ | ❌ | ✅ | SM89 |
| `fp4_e2m1` (NVFP4) | ❌ | ❌ | ✅ | SM100 |

### Compute Type

All internal computation uses `float32` for numerical stability.

---

## Environment Variables

The engine uses NVRTC to compile the kernel at runtime, which requires access to CUDA Toolkit headers.

| Variable | Purpose | Default |
|----------|---------|---------|
| `CUDA_HOME` | Path to CUDA Toolkit installation (e.g., `/usr/local/cuda`) | — |
| `CUDA_PATH` | Alternative to `CUDA_HOME` (checked if `CUDA_HOME` is not set) | — |

If neither is set, the engine defaults to `/usr/local/cuda/include` for header resolution.

The NVRTC compiler needs these headers at runtime:
- `cuda_bf16.h`, `cuda_fp8.h`, `cuda_fp4.h` — numeric type definitions
- `cuda_fp16.h` — half-precision support

Both `x86_64` and `aarch64` target include paths are added automatically (non-existent paths are silently ignored by NVRTC).

---

## Problem Size Support

### Optimized Sizes (SM100 LUT)

On SM100 (Blackwell), the following VAE problem sizes use sweep-tuned knob configurations for optimal performance:

- **Hidden dimensions (C):** 64, 128, 160, 256, 320, 512, 640, 1024
- **Token counts:** 1560, 6240, 24960, 99840, 399360
- **Output dtypes:** bf16, FP8 E4M3, NVFP4 E2M1

Total: 120 optimized configurations (8 × 5 × 3).

### Fallback Heuristic (All Architectures)

For problem sizes not in the LUT (including all non-SM100 GPUs), the engine uses a conservative fallback heuristic:

- **Supported C:** Any C ≥ 32 where C is divisible by `BYTES_PER_LDG / sizeof(bfloat16) * 32` for some valid `BYTES_PER_LDG ∈ {2, 4, 8, 16}`
- **Supported token counts:** Any positive integer
- **Examples of supported C values:** 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 4096, 5120, 8192, ...
- **Examples of unsupported C values:** 1, 7, 16, 33, 48 (fail vectorization divisibility constraints)

### L2Norm Equivalence

The WAN VAE uses L2 normalization, which is equivalent to RMSNorm with an adjusted epsilon:

```text
ε_cudnn = ε_l2norm / C
```

where `C` is the hidden dimension. This adjustment is exact (verified to 0 mismatches across all problem sizes).

---

## API Usage

The engine is accessed through the cuDNN graph API with `heur_mode.OPENSOURCE`:

```python
import cudnn
import torch

C, num_tokens = 512, 24960
eps = 1e-5

graph = cudnn.pygraph(
    intermediate_data_type=cudnn.data_type.FLOAT,
    compute_data_type=cudnn.data_type.FLOAT,
)

X = graph.tensor(
    dim=[num_tokens, C, 1, 1],
    stride=[C, 1, 1, 1],
    data_type=cudnn.data_type.BFLOAT16,
)
scale = graph.tensor(
    dim=[1, C, 1, 1],
    stride=[C, 1, 1, 1],
    data_type=cudnn.data_type.BFLOAT16,
)
epsilon = graph.tensor(
    dim=[1, 1, 1, 1],
    stride=[1, 1, 1, 1],
    data_type=cudnn.data_type.FLOAT,
    is_pass_by_value=True,
)

# Build the RMSNorm → SiLU fusion pattern
Y = graph.rmsnorm(
    norm_forward_phase=cudnn.norm_forward_phase.INFERENCE,
    input=X, scale=scale, epsilon=epsilon,
)[0]
Y.set_dim([num_tokens, C, 1, 1]).set_stride([C, 1, 1, 1])
Y.set_data_type(cudnn.data_type.BFLOAT16)

Z = graph.swish(input=Y, swish_beta=1.0)
Z.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)

# Build with OPENSOURCE heuristic mode
graph.build([cudnn.heur_mode.OPENSOURCE])

# Execute
x = torch.randn(num_tokens, C, dtype=torch.bfloat16, device="cuda")
w = torch.ones(1, C, 1, 1, dtype=torch.bfloat16, device="cuda")
eps_val = torch.full((1, 1, 1, 1), eps, dtype=torch.float32)
out = torch.empty(num_tokens, C, dtype=torch.bfloat16, device="cuda")
workspace = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device="cuda")

graph.execute(
    {X: x.view(num_tokens, C, 1, 1), scale: w, epsilon: eps_val,
     Z: out.view(num_tokens, C, 1, 1)},
    workspace,
)
```

---

## Tests

- **`test/python/test_sm100_rms_norm_silu_graph_api.py`** — Full 120-config sweep (bf16 + FP8 + NVFP4) of the optimized problem shapes for VAE on B200