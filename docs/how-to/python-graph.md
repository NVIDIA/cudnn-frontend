# Use the Python `cudnn.Graph` wrapper

Use `cudnn.Graph` when PyTorch owns runtime tensors and you want the frontend to automate lifecycle stages, output allocation, workspace, handle reuse, and current-stream binding.

`cudnn.Graph` requires cuDNN backend 9.12 or newer. The repository's general
frontend minimum of 8.5 does not apply to this wrapper; see
[installation requirements](/getting-started/installation#verification-limits).

## Before you begin

Install CUDA-enabled PyTorch and cuDNN backend 9.12 or newer, then complete
[Build and extend a graph](/tutorials/first-graph). Use this wrapper when
PyTorch tensors should define and supply the graph contract.

## Declare the call contract

Pass `inputs` and `outputs` as ordered names, graph tensors, or representative PyTorch tensors. Names must resolve to tensors captured during construction. Positional calls bind in that declared order; dictionary-oriented execution uses tensor names or descriptors through the underlying graph.

<Code
  src="../samples/python/matmul.py"
  language="python"
  maxLines={0}
/>

## Build on context exit

Inside the context, operation methods are forwarded to an owned `cudnn.pygraph`. On successful exit the wrapper validates, creates the graph representation, queries heuristic plans, checks support, builds, allocates workspace when enabled, and resolves the IO tuples. An exception discards the incomplete graph.

## Control resources

`handle="auto"` uses a cached handle and updates its CUDA stream to PyTorch's current stream at execution. Supply a handle when the application owns it. Set `workspace_alloc=False` only when you will provide a compatible workspace through a lower-level path; the wrapper cannot execute without enough storage.

## Reuse safely

Reuse the wrapper with buffers whose shape, stride, dtype, and device satisfy the built graph. Build a distinct cache entry for a new metadata signature. `set_io_tuples()` can change the ordered wrapper contract before execution but cannot make incompatible tensor metadata legal.

See [`cudnn.Graph`](/api/python#g) and [Your first graph](/tutorials/first-graph).

This guide is specific to the Python wrapper. For explicit C++ construction,
use the [C++ quick start](/getting-started/quickstart).

## Expected result

The sample builds a matrix-multiplication graph on context exit, executes it
with PyTorch tensors, and matches `torch.matmul`. Reusing the wrapper with
compatible metadata reuses its built graph; incompatible metadata requires a
distinct graph or cache entry.
