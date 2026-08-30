# Dynamic Shapes and Kernel Cache

## Dynamic Shapes

Causes other APIs (such as the kernel cache) to treat the graph as a dynamic shape graph.

The API to achieve the above is:
```cpp
graph.set_dynamic_shape_enabled(true)
```

## Kernel Cache
The kernel cache significantly reduces plan build time by re-using a previously compiled kernel for a given execution plan. Kernel caching is enabled only for dynamic shape graphs.

If a graph's kernel cache attribute is set, the kernel cache will store the kernel which was compiled for the graph's execution plan. 
On future same-topology operation graphs, the kernel cache may bind the previously compiled kernel to the execution plan to avoid recompilation.

The API to create a kernel cache is:
```cpp
auto kernel_cache = std::make_shared<cudnn_frontend::KernelCache>();
```

The API to set a dynamic shape graph's kernel cache is:
```cpp
graph.set_kernel_cache(kernel_cache)
```

## Thread-safety contract for KernelCache

The `KernelCache` API (`build`, `from_json`, `to_json`, `is_finalized`) is **internally synchronized** with a plain `std::mutex`. Multiple threads may call these methods concurrently on the same `KernelCache` instance without external synchronization.

`build()` is idempotent under the lock: the first caller to acquire the mutex initializes and finalizes the descriptor; subsequent callers return `OK` immediately without re-initializing it. This makes it safe for N graph threads sharing one `KernelCache` to each call `build()` concurrently — exactly one backend descriptor is created and finalized.

`to_json()` holds the lock across the full backend JSON render. A caller thread entering `build_operation_graph` concurrently may block briefly behind an ongoing serialize. The three calls Myelin makes in `flush()` — `is_finalized`, `build(nullptr)`, `to_json` — are individually atomic but not jointly; another thread can interleave between them. This is safe: the design intent is that `flush()` is called after the pool finishes.

`from_json()` is synchronized the same way as `build()`. Call it before sharing the `KernelCache` with any thread that will call `build()`.

**Scope of the thread-safety claim.** The lock protects the `KernelCache` API members and the inherited `desc`/`status` fields. The inherited raw-descriptor operations (`initialize`, `finalize`, `get_ptr`) remain **unsynchronized** — callers that use them directly must provide their own synchronization. The backend cuDNN descriptor pointed to by `get_ptr()` is owned by the `KernelCache` and destroyed only in the destructor; it is safe to read the pointer value (and dereference it via cuDNN API calls) from any thread after `build()` or `from_json()` completes, without holding the lock. For callers that must read the pointer while `build()` may still be in flight on another thread, use `get_ptr_locked()` (returns by value, lock held for the load only).

## Override Shape

Override shape allows supplying **at execution time** tensor shapes that differ from the shapes used when building the graph. A single execution plan can thus support multiple dynamic shapes without rebuilding the graph for each shape.

Typical usage: build the graph and execution plan once with a "cache shape", then on each `execute` call pass the actual shapes for that run via `override_uids`, `override_shapes`, and `override_strides`.

API to enable override shape:
```cpp
graph.set_override_shape_enabled(true)
```

Call this before building the graph (together with other options such as `set_dynamic_shape_enabled` or `set_kernel_cache`). It supports chaining; when the return type is `Error`, use `.is_good()` to check success.

Execution API with overrides:
```cpp
graph->execute(handle, variant_pack, workspace_ptr, override_uids, override_shapes, override_strides)
```

When using override shape, query the workspace for the actual runtime shape immediately before allocating the workspace:
```cpp
int64_t workspace_size = 0;
graph->get_workspace_size(handle, workspace_size, override_uids, override_shapes, override_strides);
Surface<int8_t> workspace(workspace_size);

graph->execute(handle, variant_pack, workspace.devPtr, override_uids, override_shapes, override_strides);
```

This runtime workspace query may return a different size than the workspace queried for the cache shape used to build the graph. Allocate the workspace after this query for each set of override shapes.

Where:
- `override_uids`: list of tensor UIDs whose shapes are being overridden
- `override_shapes`: new shape for each tensor in `override_uids` (each element is a `std::vector<int64_t>`)
- `override_strides`: new stride for each tensor in `override_uids`

The three vectors must have the same length. Tensors not listed in `override_uids` keep the shapes defined at graph build time.
