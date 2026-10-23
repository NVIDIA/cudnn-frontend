# OSS engines and stability

The repository combines cuDNN backend plans, Python execution engines, and open-source CUTLASS DSL kernels. They share graph goals but not identical packaging or compatibility promises.

## Built-in backend

The cuDNN backend is selected through heuristics and Graph plans. Its support surface depends on the installed backend, GPU, graph metadata, and plan policy.

## Python engines

`BaseEngine`, `PlanConfig`, `ExecutionContext`, the static engine manifest, and family heuristic hooks define an unreleased in-tree extension path. Engines own exact support checks, compilation, workspace reporting, and execution on caller context. Applications cannot register engines at runtime.

## OSS packages

Optional packages export fused SDPA, sparse attention, normalization, dense/grouped GEMM, and projection/RoPE kernels. They require the `cutedsl` dependency set and have explicit architecture, layout, shape, and dtype ranges. Package `__all__` is the public boundary; implementation modules are not automatically public.

## Experimental operators

`cudnn.experimental.ops` exposes PyTorch-facing functions that can change more quickly. An experimental label must appear in its reference page, local compatibility constraints, import path, and sample catalog entry.

## Fallback

Graph OSS heuristic selection can retain backend candidates. Direct package wrappers may not fall back. Applications should treat unsupported signals differently from programming or execution errors.
