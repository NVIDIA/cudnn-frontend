# Graphs, tensors, virtual values, and fusion

A frontend graph is a directed acyclic dataflow graph. Operation nodes consume tensor descriptors and produce new descriptors. Runtime storage remains separate from those descriptors until execution.

## Tensor identity

Dimensions describe logical extents; strides describe storage. Data type controls representation, while compute type controls operation arithmetic. Names support readable lookup and wrapper IO contracts. Numeric UIDs are the stable backend identity used by variant packs and serialization.

Names and UIDs must be unique in a graph. A ragged-offset tensor adds variable-length storage metadata. A pass-by-value tensor represents a scalar supplied through the backend scalar contract. Compile-time constants become part of graph identity.

## Virtual values

Operation outputs are virtual by default. A virtual value can connect nodes without a caller-owned device allocation. `set_output(True)` makes it non-virtual so execution must bind storage. Inputs, final outputs, saved statistics crossing graph boundaries, and externally inspected intermediates need non-virtual storage.

## Fusion

Graph composition exposes a fusion opportunity, but the selected engine decides implementation. Shape, stride, type, alignment, backend version, architecture, workspace, and numerical policy all influence available plans. The contract is the graph result and documented resource behavior, not a guaranteed kernel count. See [Fusion patterns and support semantics](/concepts/fusion-patterns) for the patterns that the frontend and fixed OSS kernels own explicitly, plus examples of backend-discovered compositions.

## Ownership

Python IR owns `Tensor` and `Node` objects; C++ Graph uses shared tensor attributes. Callers own runtime buffers, handles unless wrapped, and explicit workspace. Keep all resources alive until asynchronous execution completes.
