# Build and cache with `cudnn.graph`, `cudnn.jit`, and `graph_cache`

Use the decorators in `cudnn.graph` to construct a `cudnn.Graph` lazily from PyTorch calls and cache it by the metadata that controls graph compatibility.

## Before you begin

You need CUDA-enabled PyTorch and a cuDNN Graph backend supported by the
decorator. Complete [Build and extend a graph](/tutorials/first-graph) and
identify every piece of metadata that changes the graph contract.

## Choose the decorator

- `@cudnn.graph(...)` wraps one function and uses the default graph cache unless another cache is supplied.
- `@cudnn.jit(...)` accepts the same graph-building options and is the concise JIT-style spelling.
- `cudnn.graph_cache(key_fn, maxsize=256)` creates a bounded cache decorator with a caller-defined key.

## Define a complete key

Include device identity, dtype, shape, stride, training mode, scalar attributes, and any option that changes the graph. The built-in default key covers positional tensor metadata and keyword values but application containers or hidden global state may require an explicit `key_fn`.

## Handle eviction and failures

The cache is least-recently-used through `functools.lru_cache`. Eviction releases the Python wrapper when no other references remain. A failed graph build is not a valid cache result; correct the metadata or support condition rather than reusing a partially built wrapper.

Source contracts: [`python/cudnn/graph.py`](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/graph.py#L1).

The complete cache examples are available in
[Python](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/pytorch_cached_graph.py) and
[C++](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/graph_cache.cc).

## Expected result

Both examples build one graph entry, execute it with two compatible sets of
buffers, and compare the results with an independent calculation. A changed
shape, stride, dtype, device, or behavior-controlling option produces a
different cache entry.
