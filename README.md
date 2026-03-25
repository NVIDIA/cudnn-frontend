
# cuDNN FrontEnd(FE)

**cuDNN FE** is the modern, open-source entry point to the NVIDIA cuDNN library and high performance open-source kernels. It provides a C++ header-only library and a Python interface to access the powerful cuDNN Graph API and open-source kernels.

## 🚀 Latest news:

We will begin open-sourcing kernels based on customer needs, with the goal to educate developers and enable them to customize as needed.

We are now shipping **OSS kernels**, allowing you to inspect, modify, and contribute to the core logic. Check out our latest implementations:

*   **[GEMM + Amax](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/gemm_amax):** Optimized FP8 matrix multiplication with absolute maximum calculation.
*   **[GEMM + SwiGLU](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/gemm_swiglu):** High-performance implementation of the SwiGLU activation fused with GEMM.
*   **[Grouped GEMM + GLU](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/grouped_gemm/grouped_gemm_glu):** Unified grouped GEMM GLU API supporting dense and discrete MoE weight layouts.
*   **[Grouped GEMM + dGLU](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/grouped_gemm/grouped_gemm_dglu):** Unified grouped GEMM dGLU backward API supporting dense and discrete MoE weight layouts.
*   **[Grouped GEMM + SwiGLU](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/grouped_gemm/grouped_gemm_swiglu):** SwiGLU activation fused with Grouped GEMM.
*   **[Grouped GEMM + dSwiglu](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/gemm_dswiglu):** dSwiglu activation fused with Grouped GEMM.
*   **[Discrete Grouped GEMM + SwiGLU](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/discrete_grouped_gemm/discrete_grouped_gemm_swiglu):** Per-expert-pointer SwiGLU grouped GEMM for MoE workloads without weight packing.
*   **[Discrete Grouped GEMM + dSwiGLU](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/discrete_grouped_gemm/discrete_grouped_gemm_dswiglu):** Per-expert-pointer dSwiGLU backward grouped GEMM for MoE workloads without weight packing.
*   **[Grouped GEMM + Quant](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/grouped_gemm/grouped_gemm_quant):** Legacy dense-only grouped GEMM quant API for MoE FC2/dFC1 workloads.
*   **[Grouped GEMM + Quant (Unified)](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/grouped_gemm/grouped_gemm_quant):** Unified grouped GEMM quant API with per-row gating for MoE FC2/dFC1 workloads.
*   **[NSA](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/native_sparse_attention/):** Native Sparse attention as described in the Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention.
*   **[cudnn SDPA Fprop](https://github.com/NVIDIA/cudnn-frontend/tree/main/include/cudnn_frontend/generated/sdpa):** Open sourcing the Hopper and Blackwell fprop kernels with stats.
*   **[Fused RMSNorm + SiLU](https://github.com/NVIDIA/cudnn-frontend/tree/main/include/cudnn_frontend/generated/rms_norm_silu):** Implementation of a fused kernel of RMS normalization followed by SiLU (Swish) activation. 

## 🔥🔥🔥  SOTA Attention Kernels from cudnn backend

#### Llama 3.1 style Forward and Bprop with causal masking
<p align="center">
  <img src="benchmark/sdpa_benchmark_training/results/gb300_919_only_cudnn/llama3.1_top_left.png" alt="Llama 3.1 SDPA Benchmark on GB300 (only cuDNN)" width="600"/>
</p>

#### Deepseek v3 style Forward and Bprop with causal masking

<p align="center">
  <img src="benchmark/sdpa_benchmark_training/results/gb300_919_only_cudnn/dsv3_top_left.png" alt="DSv3 SDPA Benchmark on GB300 (only cuDNN)" width="600"/>
</p>


## Key Features

*   **Unified Graph API:** Create reusable, persistent `cudnn_frontend::graph::Graph` objects to describe complex subgraphs.
*   **Ease of Use:** Simplified C++ and Python bindings (via `pybind11`) that abstract away the boilerplate of the backend API.
*   **Performance:** Built-in autotuning and support for the latest NVIDIA GPU architectures.

## Installation

### 🐍 Python

The easiest way to get started is via pip:

```bash
pip install nvidia_cudnn_frontend
```

**Requirements:**
*   Python 3.8+
*   NVIDIA driver and CUDA Toolkit

### ⚙️ C++ (Header Only)

Since the C++ API is header-only, integration is seamless. Simply include the header in your compilation unit:

```cpp
#include <cudnn_frontend.h>
```

Ensure your include path points to the `include/` directory of this repository.

## Building from Source

If you want to build the Python bindings from source or run the C++ samples:

**1. Dependencies**
*   `python-dev` (e.g., `apt-get install python-dev`)
*   Dependencies listed in `requirements.txt` (`pip install -r requirements.txt`)

**2. Python Source Build**
```bash
pip install -v git+https://github.com/NVIDIA/cudnn-frontend.git
```
*Environment variables `CUDAToolkit_ROOT` and `CUDNN_PATH` can be used to override default paths.*

**3. C++ Samples Build**
```bash
mkdir build && cd build
cmake -DCUDNN_PATH=/path/to/cudnn -DCUDAToolkit_ROOT=/path/to/cuda ../
cmake --build . -j16
./bin/samples
```

## Documentation & Examples

*   **Developer Guide:** [Official NVIDIA Documentation](https://docs.nvidia.com/deeplearning/cudnn/frontend/v1.9.0/developer/overview.html)
*   **C++ Samples:** See `samples/cpp` for comprehensive usage examples.
*   **Python Samples:** See `samples/python` for pythonic implementations.

## 🤝 Contributing

We strictly welcome contributions! Whether you are fixing a bug, improving documentation, or optimizing one of our new OSS kernels, your help makes cuDNN better for everyone.

1.  Check the [Contribution Guide](CONTRIBUTING.md) for details.
2.  Fork the repo and create your branch.
3.  Submit a Pull Request.

## Debugging

To view the execution flow and debug issues, you can enable logging via environment variables:

```bash
# Log to stdout
export CUDNN_FRONTEND_LOG_INFO=1
export CUDNN_FRONTEND_LOG_FILE=stdout

# Log to a file
export CUDNN_FRONTEND_LOG_INFO=1
export CUDNN_FRONTEND_LOG_FILE=execution_log.txt
```

**Logging Levels:**
- `CUDNN_FRONTEND_LOG_INFO=0`: No logging
- `CUDNN_FRONTEND_LOG_INFO=1`: Full logging with tensor dumps
- `CUDNN_FRONTEND_LOG_INFO=10`: Basic logging (safe for CUDA graph capture)

Alternatively, you can control logging programmatically via `cudnn_frontend::isLoggingEnabled()`.

## License

This project is licensed under the [MIT License](LICENSE).
