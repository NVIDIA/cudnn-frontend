# Capture and update a CUDA Graph

Populate cuDNN work into a CUDA Graph, instantiate it, replay it with stable resources, and update supported bindings without rebuilding the frontend graph.

## Before you begin

Start with a built C++ graph and a supported CUDA Graph runtime. Keep all
captured allocations alive, bind the intended stream to the cuDNN handle, and
verify ordinary graph execution before attempting capture. The high-level
Python Graph wrapper does not expose this lifecycle.

## Prepare stable resources

Build the cuDNN execution plan first. Allocate input, output, pass-by-value, and workspace addresses that remain valid for capture and replay. Bind the intended stream to the cuDNN handle.

## Populate and instantiate

Use `populate_cuda_graph` with the frontend graph, handle, runtime bindings, workspace, and CUDA Graph object. Instantiate the CUDA Graph executable and launch it on a compatible stream. Synchronize only where the application requires completion.

## Update bindings

Use `update_cuda_graph` only for the updates documented by the backend. Stable-address requirements still apply to resources captured by address. When shapes, plan, node topology, or unsupported bindings change, repopulate and reinstantiate instead.

References: [`populate_cuda_graph`](/api/cpp/populate-cuda-graph) and [`update_cuda_graph`](/api/cpp/update-cuda-graph).

## Verification limits

On CUDA 13.1, cuDNN 9.15.1, and SM100, native population returned
`CUDNN_STATUS_NOT_SUPPORTED_CUDA_GRAPH_NATIVE_API`; the minimum supported
backend, driver, and runtime combination was not established. The high-level
Python wrapper also does not expose direct CUDA Graph population or update
methods. No executable populate, replay, and update sample is published for
that recorded environment.

## Expected result

On a supported backend, population creates a CUDA Graph that can be
instantiated and replayed with stable resources; documented binding updates
take effect without rebuilding the frontend graph. The recorded environment
did not reach that result, so deploy this path only after an end-to-end replay
and update test on the target stack.
