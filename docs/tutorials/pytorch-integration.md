# Integrate with PyTorch

Wrap a shape-specialized cuDNN graph in PyTorch code, cache it by metadata, and preserve autograd behavior. Use `cudnn.Graph` for direct eager integration and a registered custom operator when `torch.compile` needs an explicit schema and fake kernel.

The graph-cache example has Python and C++ versions. Autograd and
`torch.compile` use language-specific PyTorch integration surfaces, which are
called out separately below.

## Before you begin

Complete [Build and extend a graph](/tutorials/first-graph). The Python cache and
autograd examples require CUDA-enabled PyTorch and cuDNN backend 9.12 or newer.
The paired C++ cache example requires the C++17/CUDA toolchain. The
`torch.autograd.Function` section assumes familiarity with PyTorch's custom
autograd contract.

## Choose a cache key

Include every value that changes graph metadata or behavior: device, dtype, shape, stride, operation attributes, training mode, and relevant backend choices. Do not key only by shape when layouts or dtypes vary.

## Build once

Construct and exit the `cudnn.Graph` context on the cache miss. The wrapper owns the workspace and reuses its built plan. Keep the graph alive as long as cached entries may execute.

<Tabs>
<Tab title="Python" language="python">


<Code
  src="../samples/python/pytorch_cached_graph.py"
  language="python"
  maxLines={0}
/>

The wrapper owns output allocation and workspace, so the cache stores the
built `cudnn.Graph` object.

</Tab>
<Tab title="C++" language="cpp">


<Code
  src="../samples/cpp/graph_cache.cc"
  language="cpp"
  maxLines={0}
/>

C++ caches an application-owned graph entry containing the tensor
descriptors, cuDNN handle, built plan, and workspace. Runtime buffers are
rebound on each execution. The cache key includes the CUDA device, input
dimensions and strides, and the relevant data types.

</Tab>
</Tabs>
The sample keys the cache by device, dtype, shapes, and strides, executes two
different buffer sets with the same metadata, and asserts that only one graph
was built. Add every operation attribute and training-mode choice that changes
the graph contract to the key; do not reuse the entry after a metadata change.

## Add autograd

For a composite training operator, implement `torch.autograd.Function`: save the tensors and forward statistics needed by backward, execute a separately cached backward graph, and return one gradient for each differentiable input. Test both numerical values and `torch.autograd.gradcheck` where the chosen dtype permits it.

<Code
  src="../samples/python/pytorch_autograd.py"
  language="python"
  maxLines={0}
/>

This sample wraps cuDNN ReLU forward and backward graphs in a
`torch.autograd.Function`, caches each graph by the tensor metadata that defines
its contract, and compares both the output and input gradient with PyTorch.
There is no C++ tab for this example because `torch.autograd.Function` has no
equivalent cuDNN frontend graph operation. LibTorch provides a separate C++
autograd API, but this repository's standalone C++ documentation harness does
not configure or link LibTorch; a LibTorch wrapper needs its own integration
and build contract rather than a translation of this Python sample.

## Support compilation

Use [Register a PyTorch custom operator](/how-to/pytorch-custom-operator) when tracing or `torch.compile` must reason about the operation. Register a schema, CUDA implementation, fake implementation, and autograd formula; do not hide shape-dependent allocation from the fake kernel.

The `torch.autograd.Function` example verifies eager forward and backward.
`torch.compile` coverage is provided separately by the custom-operator guide.

## Expected result

The cache examples execute two compatible buffer sets while building one graph
entry. The autograd example compares ReLU output and input gradients with
PyTorch and confirms one cached forward graph and one cached backward graph.
For `torch.compile`, continue to the custom-operator guide and verify its fake
kernel and dispatcher registration separately.
