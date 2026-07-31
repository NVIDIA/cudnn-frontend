
# cuDNN Frontend (FE)

[![PyPI version](https://img.shields.io/pypi/v/nvidia-cudnn-frontend.svg)](https://pypi.org/project/nvidia-cudnn-frontend/)
[![PyPI downloads](https://img.shields.io/pypi/dm/nvidia-cudnn-frontend.svg)](https://pypi.org/project/nvidia-cudnn-frontend/)
[![Python versions](https://img.shields.io/pypi/pyversions/nvidia-cudnn-frontend.svg)](https://pypi.org/project/nvidia-cudnn-frontend/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE.txt)
[![Docs](https://img.shields.io/badge/docs-nvidia.github.io-blue.svg)](https://nvidia.github.io/cudnn-frontend/)

**cuDNN Frontend** is NVIDIA's modern, open-source entry point to the cuDNN library and a growing collection of high-performance open-source kernels — scaled dot-product attention (**SDPA / Flash Attention**), grouped GEMM fusions for **Mixture-of-Experts (MoE)** training, fused normalization + activation, and more.

It provides a **header-only C++ API** and a **Python interface** (with native PyTorch integration) to the cuDNN Graph API, targeting NVIDIA **Hopper** (H100/H200) and **Blackwell** (B200/GB200/GB300) GPUs across FP16, BF16, FP8, and **MXFP8** precision.

**Links:** [Documentation](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/) · [Blog & Deep Dives](https://nvidia.github.io/cudnn-frontend/) · [PyPI](https://pypi.org/project/nvidia-cudnn-frontend/) · [Release Notes](https://github.com/NVIDIA/cudnn-frontend/releases) · [Samples](samples/)

## 🚀 Latest news:

We will begin open-sourcing kernels based on customer needs, with the goal to educate developers and enable them to customize as needed.

We are now shipping **OSS kernels**, allowing you to inspect, modify, and contribute to the core logic. Check out our latest implementations:

*   **[GEMM + Amax](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/gemm/cutedsl/dense/amax):** Optimized FP8 matrix multiplication with absolute maximum calculation.
*   **[GEMM + SwiGLU](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/gemm/cutedsl/dense/swiglu):** High-performance implementation of the SwiGLU activation fused with GEMM.
*   **[GEMM + sReLU](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/gemm/cutedsl/dense/srelu):** High-performance implementation of squared-ReLU fused with GEMM.
*   **[GEMM + dsReLU](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/gemm/cutedsl/dense/dsrelu):** High-performance implementation of dsquared-ReLU fused with GEMM.
*   **[Grouped GEMM (BF16)](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/gemm/cutedsl/grouped/unfused):** Unfused BF16 grouped GEMM with dense and discrete MoE weight layouts.
*   **[Grouped GEMM + GLU](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/gemm/cutedsl/grouped/glu):** Unified BF16 and legacy block-scaled grouped GEMM GLU API supporting dense and discrete MoE weight layouts.
*   **[Grouped GEMM + GLU + Hadamard](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/gemm/cutedsl/grouped/glu_hadamard):** Dense grouped GEMM GLU forward fusion with a fused Hadamard transform and per-expert AMAX reduction.
*   **[Grouped GEMM + dGLU](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/gemm/cutedsl/grouped/dglu):** Unified BF16 and legacy block-scaled grouped GEMM dGLU backward API supporting dense and discrete MoE weight layouts.
*   **[Grouped GEMM + SwiGLU](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/gemm/cutedsl/grouped/swiglu):** SwiGLU activation fused with Grouped GEMM.
*   **[Grouped GEMM + dSwiglu](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/gemm/cutedsl/grouped/dswiglu):** dSwiglu activation fused with Grouped GEMM.
*   **[Grouped GEMM + sReLU](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/gemm/cutedsl/grouped/srelu):** Contiguous grouped squared-ReLU GEMM for MoE workloads.
*   **[Grouped GEMM + dsReLU](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/gemm/cutedsl/grouped/dsrelu):** Contiguous grouped dsquared-ReLU GEMM for MoE workloads.
*   **[Discrete Grouped GEMM + SwiGLU](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/gemm/cutedsl/discrete_grouped/swiglu):** Per-expert-pointer SwiGLU grouped GEMM for MoE workloads without weight packing.
*   **[Discrete Grouped GEMM + dSwiGLU](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/gemm/cutedsl/discrete_grouped/dswiglu):** Per-expert-pointer dSwiGLU backward grouped GEMM for MoE workloads without weight packing.
*   **[Grouped GEMM + Quant](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/gemm/cutedsl/grouped/quant):** Legacy dense-only grouped GEMM quant API for MoE FC2/dFC1 workloads.
*   **[Grouped GEMM + Quant (Unified)](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/gemm/cutedsl/grouped/quant):** Unified grouped GEMM quant API with per-row gating for MoE FC2/dFC1 workloads.
*   **[Grouped GEMM + Wgrad](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/gemm/cutedsl/grouped/wgrad):** Unified BF16 and legacy block-scaled grouped GEMM weight-gradient API supporting dense and discrete output layouts for MoE workloads.
*   **[BSA](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/block_sparse_attention/):** Block-sparse attention forward and backward CuTe DSL kernels for block-level routing metadata.
*   **[NSA](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/native_sparse_attention/):** Native Sparse attention as described in the Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention.
*   **[SDPA Backward: SM100, D=256](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/sdpa):** SDPA Backward pass for D=256 on SM100.
*   **[cudnn SDPA Fprop](https://github.com/NVIDIA/cudnn-frontend/tree/main/include/cudnn_frontend/generated/sdpa):** Open sourcing the Hopper and Blackwell fprop kernels with stats.
*   **[Fused RMSNorm + SiLU](https://github.com/NVIDIA/cudnn-frontend/tree/main/include/cudnn_frontend/generated/rms_norm_silu):** Implementation of a fused kernel of RMS normalization followed by SiLU (Swish) activation.
*   **[SDPA PyTorch Op](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/experimental/ops):** PyTorch custom operator for cuDNN-accelerated Scaled Dot-Product Attention with autograd and `torch.compile` support.
*   **[DSA](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/deepseek_sparse_attention):** DSA/CSA kernels for DSv4 and DSv3.2 for fprop and bprop.

Contributor credits for these OSS CuTe DSL kernels are listed in [Acknowledgements](ACKNOWLEDGEMENTS.md).

## Tech talks

* See our latest talk on GPU-Mode
  
   ▶ [Watch on YouTube](https://www.youtube.com/watch?v=kxP-vp1dgFY)

## 🔥🔥🔥  SOTA Attention Kernels from cudnn backend

#### Llama 3.1 style Forward and Bprop with causal masking (GB300)
<p align="center">
  <img src="https://github.com/NVIDIA/cudnn-frontend/blob/main/benchmark/sdpa_benchmark_training/results/llama3.1/gb300/llama3.1_top_left.png" alt="Llama 3.1 SDPA Benchmark on GB300 (only cuDNN)" width="600"/>
</p>

#### Deepseek v3 style Forward and Bprop with causal masking (GB300)

<p align="center">
  <img src="https://github.com/NVIDIA/cudnn-frontend/blob/main/benchmark/sdpa_benchmark_training/results/dsv3/gb300/dsv3_top_left.png" alt="DSv3 SDPA Benchmark on GB300 (only cuDNN)" width="600"/>
</p>

## Key Features

*   **Unified Graph API:** Create reusable, persistent `cudnn_frontend::graph::Graph` objects to describe complex subgraphs.
*   **Ease of Use:** Simplified C++ and Python bindings (via `pybind11`) that abstract away the boilerplate of the backend API.
*   **Performance:** Built-in autotuning and support for the latest NVIDIA GPU architectures.

## Installation

### 🐍 Python

The easiest way to get started is via pip:

```bash
pip install nvidia-cudnn-frontend
```

**Requirements:**
*   Python 3.9+
*   NVIDIA driver and CUDA Toolkit
*   NVIDIA cuDNN (minimum 8.5.0)

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

*   **Developer Guide:** [Official NVIDIA Documentation (latest)](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/)
*   **Blog & Deep Dives:** [nvidia.github.io/cudnn-frontend](https://nvidia.github.io/cudnn-frontend/) — release notes, installation guides, and technical deep-dives (MXFP8 attention, FP8 scale layouts, etc.)
*   **C++ Samples:** See [`samples/cpp`](samples/cpp) for end-to-end examples covering convolution, matmul, SDPA / Flash Attention, normalization, and more.
*   **Python Samples:** See [`samples/python`](samples/python) for Jupyter notebooks and PyTorch integration patterns.
*   **OSS Kernels:** See [`python/cudnn/`](python/cudnn/) for source of SDPA, grouped GEMM + SwiGLU/GLU, RMSNorm + SiLU, Native Sparse Attention, and other open-sourced kernels.
*   **PyTorch Custom Ops:** See [`python/cudnn/experimental/ops`](python/cudnn/experimental/ops) for `torch.compile`-compatible wrappers around cuDNN kernels.

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

### Environment report

When filing a bug, include the output of the environment collector — it reports the frontend/backend versions, GPU/driver properties, and every cuDNN/CUDA library copy on the system (loaded vs on disk):

```bash
python -m cudnn.collect_env
```

If `import cudnn` itself fails, download [collect_env.py](python/cudnn/collect_env.py) and run it standalone with any Python.

### Overriding the CUDA runtime library

When the frontend is built with dynamic loading enabled, it locates the CUDA runtime
(`libcudart.so.*`) at runtime by searching for the supported major versions. In some
environments (for example, containers such as GKE where the TCPXO NCCL plugin mounts a
different `libcudart` major version from the host) multiple versions of `libcudart` may be
visible on the library search path, and the automatic detection aborts with a
`Multiple libcudart libraries found` error.

To resolve this, set the `CUDNN_FRONTEND_CUDART_LIB_NAME` environment variable to the
library name (or full path) that should be loaded. This bypasses the automatic detection:

```bash
export CUDNN_FRONTEND_CUDART_LIB_NAME=libcudart.so.13
# or an absolute path
export CUDNN_FRONTEND_CUDART_LIB_NAME=/usr/local/cuda/lib64/libcudart.so.13
```

## License

This project is distributed primarily under the [Apache License 2.0](LICENSE.txt).
A subset of files remain under the [MIT License](LICENSE-MIT.txt); each source
file declares its license with an SPDX `SPDX-License-Identifier:` tag. See
[LICENSING.md](LICENSING.md) for the full list of MIT-licensed files and the
rationale, and [THIRD_PARTY_LICENSES.txt](THIRD_PARTY_LICENSES.txt) for
third-party attributions.
