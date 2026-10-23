# Manage handles, streams, workspace, and variant packs

Keep graph metadata reusable while making runtime ownership explicit.

## Before you begin

Start with a built graph that already passes a numerical check. Decide which
component owns the cuDNN handle, CUDA stream, workspace, runtime tensors, and
pass-by-value storage for the full asynchronous execution lifetime.

## Own the handle

A cuDNN handle carries the CUDA stream and backend context. In Python, create and destroy it with `cudnn.create_handle()` and `destroy_handle()`, or let `cudnn.Graph(handle="auto")` manage a cached handle. In C++, use `cudnnCreate` and `cudnnDestroy` with RAII in application code.

## Bind the stream

Set or query the stream with `cudnn.set_stream()` and `get_stream()`, or `cudnnSetStream` and `cudnnGetStream`. Execution is asynchronous. Keep inputs, outputs, pass-by-value storage, workspace, graph, plan, and handle valid until dependent work completes.

## Allocate workspace

Call `get_workspace_size()` after the selected plan is built. Allocate at least that many device bytes with suitable alignment. Reuse workspace only when concurrent executions cannot overlap on the same storage.

## Build the variant pack

Bind each non-virtual tensor UID to a compatible device pointer. C++ maps `shared_ptr<Tensor_attributes>` to pointers; Python accepts tensor objects, UIDs, or supported names and extracts pointers from integers, PyTorch tensors, or DLPack providers. Use sorted UID/template helpers only with the graph and plan that produced them.

The [C++ quick start](/getting-started/quickstart) uses `cudaMallocManaged()` for inspectable data and a separate device workspace.

## Expected result

Repeated executions use the intended stream, bind every non-virtual tensor,
and keep all pointers and workspace valid until work completes. Concurrent
executions use nonoverlapping workspace unless the allocator provides an
equivalent stream-safe ownership guarantee.
