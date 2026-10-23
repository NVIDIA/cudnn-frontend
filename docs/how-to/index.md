# How-to guides

Use these guides when you already understand the graph model and need to complete a concrete integration task.

## Construct and integrate graphs

- [Use the Python `cudnn.Graph` wrapper](/how-to/python-graph) — Let PyTorch tensors drive graph construction, allocation, and execution.
- [Control the explicit `cudnn.pygraph` lifecycle](/how-to/explicit-lifecycle) — Inspect and control validation, planning, building, and bindings.
- [Build and cache decorated graphs](/how-to/graph-cache) — Define complete metadata keys for reusable decorated graphs.
- [Configure matmul layouts and projections](/how-to/matmul-layouts) — Express batched matrices, broadcasts, transposes, and fused projections.
- [Choose a normalization family](/how-to/normalization) — Select axes, statistics, state, and backward dataflow.
- [Apply rotary position embeddings](/how-to/rope) — Build RoPE forward and backward and integrate scaling with attention.
- [Resample and pool tensors](/how-to/resampling) — Derive output shapes and retain training indices where required.
- [Reshape, transpose, slice, and concatenate](/how-to/shape-operations) — Preserve explicit shape and stride semantics through structural operations.
- [Build convolution variants](/how-to/convolution-variants) — Choose between general Graph convolution and specialized operators.
- [Run Causal Conv1d operators](/how-to/causal-convolution) — Use the optional PyTorch causal-convolution bindings and test their integration paths.
- [Configure attention recipes](/how-to/attention-recipes) — Add masks, layouts, state, dropout, and low precision to a verified baseline.

## Plan, execute, and deploy

- [Diagnose graph failures](/how-to/troubleshoot) — Isolate description, support, compilation, and runtime failures by lifecycle stage.
- [Select, filter, and autotune plans](/how-to/plan-selection) — Apply application policy and benchmark supported candidates.
- [Manage handles, streams, workspace, and variant packs](/how-to/runtime-resources) — Make asynchronous runtime ownership and lifetimes explicit.
- [Run dynamic shapes with a kernel cache](/how-to/dynamic-shapes) — Bind runtime overrides and size workspace for each specialization.
- [Serialize, precompile, and deploy graphs](/how-to/deployment) — Version graph artifacts with their target properties and compatibility.
- [Capture and update a CUDA Graph](/how-to/cuda-graphs) — Replay supported cuDNN work with stable resources.

## Extend and migrate

- [Register a PyTorch custom operator](/how-to/pytorch-custom-operator) — Connect one schema to eager execution, fake tensors, autograd, and compilation.
- [Inspect automatic Python engine selection](/how-to/custom-engine) — Inspect manifest-discovered engines, combined plan ranking, strict selection, and backend fallback.
- [Pin a C++ engine and knobs](/how-to/engine-knobs) — Discover graph-local knobs and retain a checked heuristic fallback.
- [Use scalar and constant tensors](/how-to/scalars-and-constants) — Choose graph-identity, pass-by-value, or device-backed scalar storage.
- [Migrate from the legacy C++ builder API](/how-to/legacy-migration) — Replace descriptor construction one verified graph at a time.
- [Look up optional kernel contracts](/optional-kernels) — Check exported OSS wrapper signatures, specialization constraints, and fallback boundaries.
- [Convert graphs to low-precision formats](/how-to/low-precision) — Treat quantization, scaling, computation, and validation as one pipeline.
