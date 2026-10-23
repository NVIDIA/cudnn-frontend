# Concepts

Concept pages explain the mental model behind the task guides and API contracts.

- [Glossary](/concepts/glossary) — Resolve the terms used across graph construction, planning, and execution.
- [API layers and graph lifecycle](/concepts/api-layers) — Compare the Python, C++, decorator, operator, and legacy surfaces and their lifecycle ownership.
- [Python IR routing invariants](/concepts/python-ir-routing) — Understand manifest discovery, plan spaces, UID ownership, and freeze boundaries.
- [Graphs, tensors, virtual values, and fusion](/concepts/graphs-and-tensors) — Learn how tensor identity and virtual intermediates define a graph.
- [Fusion patterns and support semantics](/concepts/fusion-patterns) — Distinguish explicit fused APIs from backend-discovered graph fusion.
- [Shapes, strides, broadcasting, and data types](/concepts/layouts) — Separate logical shape, physical storage, broadcasting, and compute precision.
- [Operations and training dataflow](/concepts/operations-training) — Connect forward outputs and saved state to backward graphs.
- [Engines, heuristics, plans, routing, and autotuning](/concepts/execution-plans) — Follow candidate discovery through policy filtering and plan selection.
- [Runtime resources and asynchronous execution](/concepts/runtime-resources) — Understand handle, stream, buffer, and workspace lifetimes.
- [Compilation, reuse, caching, and deployment](/concepts/compilation-caching) — Define graph identity and compatible reuse across processes and devices.
- [Attention layouts, masks, and state](/concepts/attention) — Interpret BSHD, BHSD, THD, masks, dropout state, and KV caches.
- [Attention support history and benchmarks](/concepts/attention-support-history) — Place versioned features and performance evidence in context.
- [Numerical precision and block scaling](/concepts/numerical-precision) — Choose storage, compute, scale, and accumulation types deliberately.
- [OSS engines and stability](/concepts/oss-stability) — Understand optional dependency, specialization, and fallback boundaries.
