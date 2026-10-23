# cuDNN Frontend

cuDNN Frontend lets applications describe tensor computations as graphs and execute them on NVIDIA GPUs. You can use a deep learning framework, such as PyTorch, if it already provides the features and performance you need. Use cuDNN Frontend when you need finer control over the computation or features that the framework does not yet support.

The repository provides a header-only C++ API, Python graph APIs, PyTorch integration, and optional open-source CUTLASS DSL kernels for selected attention, normalization, and GEMM workloads.

## Choose a learning path

### Python and PyTorch

1. [Install the Python package](/getting-started/installation#python).
2. [Run the Python quick start](/getting-started/quickstart).
3. [Build your first `cudnn.Graph`](/tutorials/first-graph).
4. [Integrate a reusable graph with PyTorch](/tutorials/pytorch-integration).
5. Move to the [explicit `cudnn.pygraph` lifecycle](/how-to/explicit-lifecycle) only when you need routing, inspection, or plan control.

### C++ application integration

1. [Install or consume the header target](/getting-started/installation#c).
2. [Run the C++ quick start](/getting-started/quickstart).
3. Learn [runtime ownership and asynchronous execution](/concepts/runtime-resources).
4. Use the [operation tutorials](/tutorials) and [deployment guides](/how-to).

### Advanced kernels and deployment

1. Learn [API layers and graph lifecycle](/concepts/api-layers).
2. Choose [plans and autotune](/how-to/plan-selection).
3. Add [dynamic shapes](/how-to/dynamic-shapes), [serialization](/how-to/deployment), or [CUDA Graphs](/how-to/cuda-graphs).
4. Evaluate [low-precision formats](/how-to/low-precision) and [open-source engines](/optional-kernels/oss-engines) against the [compatibility matrix](/getting-started/compatibility).

## Interfaces

| Interface | Use it for |
| --- | --- |
| `cudnn.Graph` | Python-first PyTorch integration with automatic lifecycle, output allocation, and workspace ownership |
| `cudnn.pygraph` | Explicit Python IR, inspection, custom routing, and lifecycle control |
| `cudnn_frontend::graph::Graph` | Header-only C++ integration with explicit resources and complete Graph API control |
| `cudnn.graph`, `cudnn.jit`, `graph_cache` | Shape-specialized Python graph construction and caching |
| `cudnn.ops` | PyTorch custom operators supplied by this package |
| OSS and experimental packages | Architecture-specific fused kernels and convenience operators with narrower stability guarantees |

The documentation targets commit `0a07d4ee28f8ae2f2106bb766b1878cc732ee80c`. Feature availability still depends on the installed cuDNN backend, CUDA runtime, GPU architecture, shape, layout, data type, and engine.

## Find exact contracts

- [Tutorials](/tutorials) teach complete workflows.
- [How-to guides](/how-to) solve focused integration tasks.
- [Concepts](/concepts) explain the model and constraints.
- [API reference](/api) is the alphabetical contract dictionary.
- [Sample catalog](/verified-samples/catalog) publishes every executable example and its verification status.
- [Verification record](/verification) identifies the environment, results, and limits behind documentation claims.
