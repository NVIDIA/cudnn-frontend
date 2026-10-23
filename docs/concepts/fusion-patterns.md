# Fusion patterns and support semantics

Use this page to answer two different questions:

1. **Which fusion-like patterns does this frontend name explicitly?** Use the inventories below.
2. **Will one concrete graph run on one concrete system?** Build an execution plan with the graph's real tensor metadata. There is no static, universal list for this question.

cuDNN Frontend does not enumerate every graph that a cuDNN backend can fuse. A Graph exposes fusion opportunities; the selected execution plan decides how to implement them. Support can change with the cuDNN version, GPU, tensor dimensions, strides, data types, alignment, workspace limit, and numerical requirements. Backend engines are therefore [discovered per graph at plan time](https://github.com/NVIDIA/cudnn-frontend/blob/d3c03b5881cdaccc4c0e07dd9ab4c7c6bc105715/python/cudnn/engines/router.py#L27-L36).

This page covers the public patterns in this checkout. Source links are pinned to a repository commit so that API claims remain auditable. A listed composite, matcher, direct kernel, or working example does **not** guarantee that an arbitrary configuration is supported or that execution uses one physical kernel.

## Choose the relevant support category

| Category | What is statically enumerable | Authoritative support test |
| --- | --- | --- |
| Runtime-discovered backend graph | Graph operations and their semantics, but not every supported composition | The complete graph successfully reaches `build_plans()` with the application's metadata and policy |
| Graph OSS matcher | The subgraph shape recognized by `heur_mode.OPENSOURCE` | The matcher accepts the exact graph and an OSS plan successfully builds |
| Frontend composite operation | The public operation and its primitive expansion | The expanded graph successfully builds a plan |
| Direct or specialized kernel | Exported class or wrapper names and their fixed operation composition | The API's validator accepts the inputs and the specialization compiles |
| Repository example | One tested composition and its recorded metadata | Evidence only for that example; re-run planning for other metadata or environments |

Do not use a successful `validate()` call as a support test. It proves graph structure and tensor metadata are valid, not that an execution engine or compiled plan exists.

## Graph OSS matcher inventory

`OPENSOURCE` planning has two registered matcher slots in this checkout. Selecting only `OPENSOURCE` provides no general fallback when neither slot matches. Combining it with ordinary heuristic modes also queries cuDNN backend candidates.

### RMSNorm followed by SiLU

- Pattern: `rmsnorm(INFERENCE) -> swish(beta=1)`.
- Python: `graph.rmsnorm(...)`, then `graph.swish(...)`.
- C++: `cudnn_frontend::graph::Graph::rmsnorm(...)`, then `Graph::pointwise(..., PointwiseMode_t::SWISH_FWD)`.
- Matcher contract: the nodes are adjacent and connected by the same tensor UID; X and scale are BF16; epsilon is required; X has rank 2 or greater. Output may be BF16, E4M3 FP8, or NVFP4 subject to architecture gates.
- Details: [normalization guide](/optional-kernels/oss-normalization) and [matcher source](https://github.com/NVIDIA/cudnn-frontend/blob/d3c03b5881cdaccc4c0e07dd9ab4c7c6bc105715/include/cudnn_frontend/graph_interface.h#L460-L599).

### One SDPA node

- Pattern: one composite or unified SDPA node. A user-authored `matmul -> softmax -> matmul` chain is not this matcher.
- Python: `graph.sdpa(...)`.
- C++: `cudnn_frontend::graph::Graph::sdpa(...)`.
- Matcher contract: current NVRTC prefill lookup accepts head dimension 64 or 128 on SM90 or the SM100 family. Registration alone does not validate every data type or optional SDPA feature; successful plan creation is authoritative.
- Details: [matcher source](https://github.com/NVIDIA/cudnn-frontend/blob/d3c03b5881cdaccc4c0e07dd9ab4c7c6bc105715/include/cudnn_frontend/graph_interface.h#L350-L458), [SM90 lookup](https://github.com/NVIDIA/cudnn-frontend/blob/d3c03b5881cdaccc4c0e07dd9ab4c7c6bc105715/include/cudnn_frontend/experimental/sm90_sdpa_prefill_engine.h#L45-L82), and [SM100 lookup](https://github.com/NVIDIA/cudnn-frontend/blob/d3c03b5881cdaccc4c0e07dd9ab4c7c6bc105715/include/cudnn_frontend/experimental/sm100_sdpa_prefill_engine.h#L54-L91).

These are the only OSS matcher slots registered by [`Graph::create_execution_plans()`](https://github.com/NVIDIA/cudnn-frontend/blob/d3c03b5881cdaccc4c0e07dd9ab4c7c6bc105715/include/cudnn_frontend/graph_interface.h#L2815-L2842).

## Frontend composite operation inventory

A composite operation expands fixed semantics into primitive Graph nodes. The backend can fuse some or all of those primitives, execute them separately, or reject the graph.

### SDPA forward composites

| Variant | Python identifier | C++ identifier | Expansion and key limits |
| --- | --- | --- | --- |
| Standard SDPA | `graph.sdpa` | `cudnn_frontend::graph::Graph::sdpa` | `Q @ K^T`, optional scale, score modification, mask or bias, softmax, optional dropout, then `P @ V` |
| FP8 SDPA | `graph.sdpa_fp8` | `cudnn_frontend::graph::Graph::sdpa_fp8` | Standard dataflow plus per-tensor dequantization, scaling, quantization, and amax operations |
| MXFP8 SDPA | `graph.sdpa_mxfp8` | MXFP8 overload of `cudnn_frontend::graph::Graph::sdpa_fp8` | Standard dataflow plus block-scale dequantization, scaling, quantization, and amax operations |

The composite forward path rejects `block_mask`, cumulative sequence-length inputs, and ragged-offset multipliers. Consult the [composite expansion](https://github.com/NVIDIA/cudnn-frontend/blob/d3c03b5881cdaccc4c0e07dd9ab4c7c6bc105715/include/cudnn_frontend/node/scaled_dot_product_flash_attention.h#L604-L1073), [support checks](https://github.com/NVIDIA/cudnn-frontend/blob/d3c03b5881cdaccc4c0e07dd9ab4c7c6bc105715/include/cudnn_frontend/node/sdpa_support_surface.h#L389-L546), and [Python SDPA reference](/api/python/sdpa) before allocating tensors.

### SDPA backward composites

| Variant | Python identifier | C++ identifier | Expansion and key limits |
| --- | --- | --- | --- |
| Standard SDPA backward | `graph.sdpa_backward` | `cudnn_frontend::graph::Graph::sdpa_backward` | Recomputes probabilities and dropout, then forms dV, dP/dS, dK, and dQ; GQA and optional dBias add reductions |
| FP8 SDPA backward | `graph.sdpa_fp8_backward` | `cudnn_frontend::graph::Graph::sdpa_fp8_backward` | Backward attention plus per-tensor dequantization, scaling, quantization, and amax reductions |
| MXFP8 SDPA backward | `graph.sdpa_mxfp8_backward` | MXFP8 overload of `cudnn_frontend::graph::Graph::sdpa_fp8_backward` | Backward attention plus block-scale dequantization, scaling, quantization, and amax reductions |

Standard dense inputs are rank 4 and require unit embedding stride. Head dimensions, masks, ragged layouts, dropout, determinism, and dBias are architecture- and version-dependent. Regular FP8 backward starts at cuDNN 9.1 on Hopper. See the [standard backward expansion](https://github.com/NVIDIA/cudnn-frontend/blob/d3c03b5881cdaccc4c0e07dd9ab4c7c6bc105715/include/cudnn_frontend/node/scaled_dot_product_flash_attention.h#L1075-L2057), [FP8 backward source](https://github.com/NVIDIA/cudnn-frontend/blob/d3c03b5881cdaccc4c0e07dd9ab4c7c6bc105715/include/cudnn_frontend/node/sdpa_fp8_bwd.h#L17-L1120), and the [Python backward reference](/api/python/sdpa-backward).

### Internal softmax fallback

- Expansion: max reduction, subtraction, exponentiation, sum reduction, and division; optional sink and statistics add branches.
- Public API: internal to composites before cuDNN 9.21; there is no standalone Python Graph operation.
- Output modes: statistics only, max plus sum-exp, or none of those outputs.
- Source: [softmax composite](https://github.com/NVIDIA/cudnn-frontend/blob/d3c03b5881cdaccc4c0e07dd9ab4c7c6bc105715/include/cudnn_frontend/node/softmax.h#L86-L217).

### Diagonal-band mask fallback

- Expansion: generate row and column indices, adjust bounds, compare, then binary-select scores or negative infinity.
- Python: exposed through SDPA mask attributes.
- C++: `cudnn_frontend::graph::Graph::diagonal_band_mask`.
- Limit: before cuDNN 9.21, the composite rejects cumulative sequence lengths.
- Details: [C++ reference](/api/cpp/diagonal-band-mask) and [mask composite](https://github.com/NVIDIA/cudnn-frontend/blob/d3c03b5881cdaccc4c0e07dd9ab4c7c6bc105715/include/cudnn_frontend/node/diagonal_band_mask.h#L121-L273).

## Direct and specialized kernel inventory

These APIs dispatch fixed fused or specialized kernels; they do not match arbitrary public Graph subgraphs. Their validators define support, and direct calls do not inherit Graph fallback. The optional Python OSS packages require their optional dependencies and compatible hardware.

### Dense GEMM direct kernels

| Fixed pattern | Class API | Allocating wrapper API |
| --- | --- | --- |
| GEMM + amax | `cudnn.GemmAmaxSm100` | `cudnn.gemm_amax_wrapper_sm100` |
| GEMM + squared ReLU | `cudnn.GemmSreluSm100` | `cudnn.gemm_srelu_wrapper_sm100` |
| GEMM + derivative of squared ReLU | `cudnn.GemmDsreluSm100` | `cudnn.gemm_dsrelu_wrapper_sm100` |
| GEMM + SwiGLU | `cudnn.GemmSwigluSm100` | `cudnn.gemm_swiglu_wrapper_sm100` |
| Projection GEMM + RoPE + dual-layout MXFP8 quantization | `cudnn.GemmProjRopeMxfp8Sm100` | `cudnn.gemm_proj_rope_mxfp8_wrapper_sm100` |

All five families require SM100+, but data type, layout, scale-factor, alignment, tile, cluster, and output contracts differ. Use the [dense GEMM guide](/optional-kernels/oss-dense-gemm) and the package [export sources](https://github.com/NVIDIA/cudnn-frontend/blob/d3c03b5881cdaccc4c0e07dd9ab4c7c6bc105715/python/cudnn/__init__.py#L286-L295).

### Grouped GEMM direct kernels

| Fixed pattern | Class API | Allocating wrapper API |
| --- | --- | --- |
| Grouped GEMM + SwiGLU | `cudnn.GroupedGemmSwigluSm100` | `cudnn.grouped_gemm_swiglu_wrapper_sm100` |
| Grouped GEMM + dSwiGLU | `cudnn.GroupedGemmDswigluSm100` | `cudnn.grouped_gemm_dswiglu_wrapper_sm100` |
| Grouped GEMM + squared ReLU | `cudnn.GroupedGemmSreluSm100` | `cudnn.grouped_gemm_srelu_wrapper_sm100` |
| Grouped GEMM + derivative of squared ReLU | `cudnn.GroupedGemmDsreluSm100` | `cudnn.grouped_gemm_dsrelu_wrapper_sm100` |
| Grouped GEMM + configurable GLU | `cudnn.GroupedGemmGluSm100` | `cudnn.grouped_gemm_glu_wrapper_sm100` |
| Grouped GEMM + configurable GLU + Hadamard | `cudnn.GroupedGemmGluHadamardSm100` | `cudnn.grouped_gemm_glu_hadamard_wrapper_sm100` |
| Grouped GEMM + configurable dGLU | `cudnn.GroupedGemmDgluSm100` | `cudnn.grouped_gemm_dglu_wrapper_sm100` |
| Grouped GEMM + quantization | `cudnn.GroupedGemmQuantSm100` | `cudnn.grouped_gemm_quant_wrapper_sm100` |
| Grouped GEMM weight gradient | `cudnn.GroupedGemmWgradSm100` | `cudnn.grouped_gemm_wgrad_wrapper_sm100` |

These are contiguous Mixture-of-Experts kernels with per-expert weights and boundary metadata. Exact offset, scale-factor, output, and pointer-array alternatives are documented in the [grouped GEMM guide](/optional-kernels/oss-grouped-gemm) and [package exports](https://github.com/NVIDIA/cudnn-frontend/blob/d3c03b5881cdaccc4c0e07dd9ab4c7c6bc105715/python/cudnn/grouped_gemm/__init__.py#L49-L67).

### Discrete grouped GEMM direct kernels

| Fixed pattern | Class API | Allocating wrapper API |
| --- | --- | --- |
| Pointer-array grouped GEMM + configurable GLU | `cudnn.DiscreteGroupedGemmSwigluSm100` | `cudnn.discrete_grouped_gemm_swiglu_wrapper_sm100` |
| Pointer-array grouped GEMM + configurable dGLU | `cudnn.DiscreteGroupedGemmDswigluSm100` | `cudnn.discrete_grouped_gemm_dswiglu_wrapper_sm100` |

These APIs require device pointer arrays whose pointed-to buffers remain alive until asynchronous work completes. See the [discrete grouped GEMM guide](/optional-kernels/oss-discrete-grouped-gemm) and [package exports](https://github.com/NVIDIA/cudnn-frontend/blob/d3c03b5881cdaccc4c0e07dd9ab4c7c6bc105715/python/cudnn/discrete_grouped_gemm/__init__.py#L21-L26).

### Normalization direct kernel

- Pattern: RMSNorm + randomized Hadamard transform + amax.
- Class: `cudnn.RmsNormRhtAmaxSm100`.
- Wrapper: `cudnn.rmsnorm_rht_amax_wrapper_sm100`.
- Details: [normalization guide](/optional-kernels/oss-normalization) and [package exports](https://github.com/NVIDIA/cudnn-frontend/blob/d3c03b5881cdaccc4c0e07dd9ab4c7c6bc105715/python/cudnn/rmsnorm_rht_amax/__init__.py#L11-L16).

RMSNorm + SiLU is the Graph OSS matcher described earlier, not another current direct wrapper.

### SDPA head-dimension-256 direct kernels

| Fixed pattern | Class API | Allocating wrapper API |
| --- | --- | --- |
| SDPA forward + LSE | `cudnn.SdpafwdSm100D256` | `cudnn.sdpa_fwd_wrapper_sm100_d256` |
| SDPA backward | `cudnn.SdpabwdSm100D256` | `cudnn.sdpa_bwd_wrapper_sm100_d256` |

These APIs require SM100+, head dimension 256, and their documented dense or variable-length layouts. They do not automatically fall back to Graph SDPA. See the [D=256 guide](/optional-kernels/oss-sdpa-d256) and [package exports](https://github.com/NVIDIA/cudnn-frontend/blob/d3c03b5881cdaccc4c0e07dd9ab4c7c6bc105715/python/cudnn/sdpa/__init__.py#L7-L12).

### Sparse-attention specialized kernels

| Family | Public Python identifiers | Fixed scope |
| --- | --- | --- |
| Block Sparse Attention | `cudnn.BSA`, `cudnn.block_sparse_attention_forward`, `cudnn.block_sparse_attention_backward` | Block-sparse attention forward and backward |
| DeepSeek Sparse Attention | `cudnn.DSA` | DSA stages |
| Native Sparse Attention | `cudnn.NSA` | Compression, selection, and sliding-window stages |

These are specialized attention families, not matchers over public Graph nodes. Follow the family links in the [OSS engine inventory](/optional-kernels/oss-engines#choose-an-entry-point).

### Causal-convolution fixed kernels

- Python convenience APIs: `cudnn.ops.causal_conv1d`, `cudnn.ops.causal_conv1d_nwh`, and `cudnn.ops.b2b_causal_conv1d`.
- C/C++ entry points: `cudnnCausalConv1dForward`, `cudnnCausalConv1dBackward`, `cudnnCausalConv1dNwhForward`, `cudnnCausalConv1dNwhBackward`, `cudnnB2BCausalConv1dForward`, and `cudnnB2BCausalConv1dBackward`.
- Patterns: causal depthwise Conv1d with optional bias or SiLU; and back-to-back projection convolution, Q/K gate, mixer or skip, and V gate, with backward.
- Details: [causal-convolution guide](/how-to/causal-convolution) and [C++ samples](https://github.com/NVIDIA/cudnn-frontend/tree/d3c03b5881cdaccc4c0e07dd9ab4c7c6bc105715/samples/cpp/causal_conv1d).

The optional Python OSS families above do not have equivalent public C++ wrappers in this checkout. Causal convolution instead exposes public cuDNN C entry points. `cudnn.experimental.ops.scaled_dot_product_attention` and `cudnn.experimental.ops.moe_grouped_matmul` wrap existing Graph operation families and do not add fusion patterns to this inventory.

## Runtime-discovered backend graph examples

Arbitrary Graph chains have no static support table. Frontend [expands composites and submits an operation graph](https://github.com/NVIDIA/cudnn-frontend/blob/d3c03b5881cdaccc4c0e07dd9ab4c7c6bc105715/include/cudnn_frontend/graph_interface.h#L947-L986); [heuristics and plan building](https://github.com/NVIDIA/cudnn-frontend/blob/d3c03b5881cdaccc4c0e07dd9ab4c7c6bc105715/include/cudnn_frontend/graph_interface.h#L2797-L2859) determine whether an implementation exists.

| Composition | Repository evidence | Qualification |
| --- | --- | --- |
| `matmul -> bias/add -> ReLU` | [First-graph tutorial](/tutorials/first-graph#construct-a-multi-operation-graph) | Python and C++ examples execute and compare results |
| `conv_fprop -> bias -> ReLU` or leaky ReLU | [Convolution tests](https://github.com/NVIDIA/cudnn-frontend/blob/d3c03b5881cdaccc4c0e07dd9ab4c7c6bc105715/test/python/test_conv_bias.py#L28-L116) | Concrete 2-D and 3-D cases only |
| `slice -> matmul -> ReLU` | [C++ sample](https://github.com/NVIDIA/cudnn-frontend/blob/d3c03b5881cdaccc4c0e07dd9ab4c7c6bc105715/samples/cpp/misc/slice.cpp#L28-L104) | Builds, serializes, deserializes, and executes one graph |
| Logical `reshape -> ReLU` or `transpose -> add` | [Memory-bound sample](https://github.com/NVIDIA/cudnn-frontend/blob/d3c03b5881cdaccc4c0e07dd9ab4c7c6bc105715/samples/cpp/membound/membound_fusion.cpp#L28-L144) | Requires CUDA 13.1, cuDNN 9.22, and data-center Blackwell |
| `compare-greater -> logical-and` | [Boolean sample](https://github.com/NVIDIA/cudnn-frontend/blob/d3c03b5881cdaccc4c0e07dd9ab4c7c6bc105715/samples/cpp/membound/boolean_fusion.cpp#L28-L119) | Same cuDNN and architecture gates as the memory-bound sample |

A sample proves only its own tensor metadata and environment. Materializing an intermediate can also change the operation graph presented to planning.

## Verify support for a concrete Graph

Build and test the graph with the application's actual metadata. For Python, the support probe after graph construction is:

```python
graph.validate()
graph.build_operation_graph()
graph.create_execution_plans(
    [
        cudnn.heur_mode.OPENSOURCE,
        cudnn.heur_mode.A,
        cudnn.heur_mode.FALLBACK,
    ]
)
graph.check_support()
graph.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE)
```

Use `[cudnn.heur_mode.OPENSOURCE]` when the application requires an OSS matcher and must fail rather than fall back. Omit `OPENSOURCE` when only cuDNN backend plans should be considered. The equivalent C++ modes are `cudnn_frontend::HeurMode_t::OPENSOURCE`, `A`, and `FALLBACK`; check every returned `cudnn_frontend::error_t` before continuing.

### Interpret lifecycle failures

| Last successful stage | What has been established | What has not been established |
| --- | --- | --- |
| `validate()` | Graph structure and tensor metadata are valid | Engine support |
| `build_operation_graph()` | Backend operation descriptors were created | A candidate or compiled plan exists |
| `create_execution_plans()` | Candidate discovery completed | A candidate supports every graph detail |
| `check_support()` | At least one candidate plan supports the graph | Kernel or plan compilation succeeds |
| `build_plans()` | A plan compiled for this graph and environment | Numerical correctness or one-kernel execution |
| `execute()` plus reference comparison | This input case meets the tested numerical tolerance | Other shapes, types, layouts, GPUs, or versions are supported |

After `build_plans()`, allocate `graph.get_workspace_size()` bytes, execute representative inputs, and compare outputs and applicable gradients with an independent reference. Repeat plan creation and numerical verification whenever the cuDNN frontend or backend version, GPU, dimensions, strides, data types, alignment, workspace policy, or numerical policy changes.

For custom Python routing, assert the selected engine ID. For the cuDNN backend path, verify that a plan builds for the complete graph. Do not infer a one-kernel fusion from successful execution: backend engine names and kernel grouping are not stable public contracts. See [routing verification guidance](/concepts/python-ir-routing#prove-the-selected-path) and the [explicit lifecycle guide](/how-to/explicit-lifecycle).
