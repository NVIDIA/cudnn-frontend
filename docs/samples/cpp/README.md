# C++ documentation samples

These standalone files are the canonical sources included in the documentation.

| Sample | Documentation | Purpose | Verification |
| --- | --- | --- | --- |
| `matmul.cc` | Getting Started / Quick Start | Build, plan, execute, and numerically check BF16 matmul | Compile and end-to-end run on Blackwell |

Compile and run from the repository root:

```bash
nvcc -std=c++17 -Iinclude docs/samples/cpp/matmul.cc -lcudnn -lnvrtc -o /tmp/cudnn_fe_matmul
/tmp/cudnn_fe_matmul
```

Requirements: CUDA toolkit (including NVRTC), cuDNN development libraries, this repository's headers, a C++17-capable host compiler, and a supported NVIDIA GPU. Success prints the output shape and workspace size.

See the generated [sample index](/verified-samples/catalog) for the complete inventory and the
runner command for each sample.
