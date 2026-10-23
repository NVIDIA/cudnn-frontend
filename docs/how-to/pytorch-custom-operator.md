# Register a PyTorch custom operator

Expose a cuDNN-backed operation to eager PyTorch, autograd, fake tensor propagation, and `torch.compile` through one explicit dispatcher contract.

## Before you begin

Use CUDA-enabled PyTorch with `torch.library.custom_op`, complete
[Build and extend a graph](/tutorials/first-graph), and define the public
operator schema plus every input property that changes graph compatibility.

## Define the schema

Use `torch.library.custom_op` or the lower-level `torch.library.Library.define/impl` API to declare tensor, scalar, optional, list, mutation, and aliasing behavior precisely. The current repository uses both styles. Choose one registration owner for a qualified operator name and keep its defaults consistent with the public Python wrapper.

The canonical sample uses `custom_op` because it provides concise fake and autograd registration on the installed PyTorch version:

<Markdown src="/snippets/generated/python/pytorch_custom_operator/registration.mdx" />

Do not assume that `custom_op` adds a fixed amount of dispatch overhead. The current source uses it for Causal Conv1d; benchmark the chosen registration form with the deployed PyTorch version.

## Register CUDA and fake implementations

The CUDA kernel builds or retrieves a metadata-keyed cuDNN graph and executes it on the current stream. The fake implementation computes only output shapes, strides, dtypes, devices, and aliasing. It must not allocate CUDA storage or call cuDNN.

Include device, dtype, shape, stride, and every scalar or enum that changes graph structure in the cache key. A shape-only key can incorrectly reuse a plan for a transposed or differently aligned tensor. The sample builds one `cudnn.Graph` per complete input metadata key:

<Markdown src="/snippets/generated/python/pytorch_custom_operator/cache.mdx" />

For explicit `cudnn.pygraph` integrations, derive the UID order once with `_get_variant_pack_uids_sorted()` only when using that internal fast path, build pointers in exactly that order, and call `_execute_with_ptrs`. Those underscore methods are not stable public APIs. Prefer the public `execute` or `cudnn.Graph.__call__` path unless measured Python overhead requires the internal path.

Allocate workspace per concurrent execution. Caching the required byte count is safe for one compiled metadata signature; caching one mutable workspace tensor is not safe when streams can overlap. A `cudnn.Graph` wrapper manages workspace internally, while explicit `pygraph` code should allocate or pool storage with stream-aware ownership.

When using explicit handles, cache at most one handle per device and call `cudnn.set_stream(handle, torch.cuda.current_stream(device).cuda_stream)` before every launch whose stream may differ. The wrapper's `handle="auto"` path performs current-stream binding for the sample.

## Register autograd

Save exactly the forward tensors and statistics needed by backward. Return gradients in schema order and `None` for non-differentiable arguments. Test repeated backward, non-contiguous layouts, and mixed requires-grad inputs.

<Markdown src="/snippets/generated/python/pytorch_custom_operator/autograd.mdx" />

An explicit UID enum is useful when a low-level graph uses stable integer UIDs and pointer-order execution. Do not use it as a substitute for a metadata cache key: UID identity and runtime tensor compatibility are separate contracts.

## Compile and test

Test direct calls, `torch.library.opcheck`, autograd, fake tensors, and `torch.compile` with cache hits and distinct metadata signatures. The current `cudnn.ops.causal_conv1d` and experimental SDPA operators are source examples of registrations, but their optional compiled bindings can have separate build requirements.

Complete executable: [PyTorch custom operator](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/pytorch_custom_operator.py). It checks eager output, gradients, fake registration through compilation, and a cache hit on CUDA.

There is no C++ counterpart because schema, fake-tensor, autograd, and
`torch.compile` registration are Python dispatcher contracts. A LibTorch
extension needs its own registration and verification path.

## Expected result

The sample checks eager output and gradients, runs fake-tensor propagation
through `torch.compile`, and proves a compatible call hits the graph cache.
Calls with distinct metadata must build or retrieve a distinct cache entry.
