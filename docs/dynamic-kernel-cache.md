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
