# Quick start

Below is a quick start example that builds and executes a batched BF16 matrix multiplication using cuDNN. Tensor `A` has shape `[2, 4, 8]`, tensor `B` has shape `[1, 8, 6]`, and cuDNN broadcasts `B` across the batch dimension to produce `C` with shape `[2, 4, 6]`.

The Python and C++ examples implement the same graph. Python uses the high-level `cudnn.Graph` wrapper to manage the graph lifecycle. C++ exposes each lifecycle operation so that resource ownership and runtime bindings are explicit.

The Python `cudnn.Graph` wrapper requires cuDNN backend 9.12 or newer; the
repository's general frontend minimum of 8.5 does not apply to this wrapper.

<Tabs>
<Tab title="Python" language="python">


The Python wrapper accepts PyTorch CUDA tensors while the graph is being described. It records their metadata and later accepts the actual runtime buffers through a function-like call.

<Code
  src="../samples/python/matmul.py"
  language="python"
  maxLines={0}
/>

Run the sample from the repository root:

<Markdown src="/snippets/generated/shell/run_quickstarts/python.mdx" />

The example performs these steps:

1. **Allocate runtime inputs.** `a_data` and `b_data` are BF16 PyTorch tensors on the GPU. Their shapes define a batched matmul, and the singleton leading dimension of `b_data` makes it broadcast across both batches of `a_data`. The random values are runtime data, but their shapes, strides, and dtypes also provide the metadata needed to describe the graph.

2. **Open a `cudnn.Graph` construction context.** `io_data_type` sets the default type of values entering and leaving the graph. `compute_data_type=torch.float32` requests FP32 matmul accumulation, which is more accurate than BF16 accumulation. `inputs` and `outputs` define the positional call contract: the first argument to `graph(...)` binds to `matmul::A`, the second binds to `matmul::B`, and the tensor named `C` is returned. `handle="auto"` makes the wrapper reuse a default cuDNN handle and bind it to PyTorch's current CUDA stream at execution time. Note that in this example, the data type of the input tensors are inferred from the PyTorch tensors and that of the output tensor is set explicitly. Hence setting `io_data_type` is redundant.

3. **Add the matmul operation.** `graph.matmul(...)` captures the PyTorch tensors through their DLPack-compatible interface and creates cuDNN tensor descriptors from their metadata. It does not consume any tensor values while constructing the graph. Naming the node `matmul` gives its inputs the stable canonical names `matmul::A` and `matmul::B` used by the graph's input list.

4. **Declare the external output.** `set_output(True)` makes the result non-virtual, so cuDNN writes it to caller-visible storage instead of treating it as an internal temporary. `set_name("C")` connects the descriptor to `outputs=["C"]`, and `set_data_type(torch.bfloat16)` requests a BF16 result after FP32 accumulation.

5. **Build the graph when the context closes.** Exiting the `with` block validates the graph, builds the backend operation graph, creates execution plans from the default heuristic modes, checks hardware support, builds the viable plans, and allocates workspace for the selected plan. This host-side preparation is intended to happen once for a compatible set of tensor metadata.

6. **Execute with runtime buffers.** `graph(a_data, b_data)` maps the positional arguments to the declared inputs, allocates a PyTorch output tensor with the inferred shape and stride, supplies the wrapper-managed workspace, and enqueues the selected plan on PyTorch's current CUDA stream. Because there is one declared output, the call returns one tensor rather than a tuple.

7. **Verify the result.** The PyTorch reference converts the inputs to FP32 before multiplication to match cuDNN's requested compute type, then converts the result to BF16 to match `C`. `torch.testing.assert_close` checks the complete result using tolerances appropriate for BF16 output.

Keep `graph` alive and call it repeatedly to amortize graph construction overhead. Rebuild it when the input shape, stride, dtype, operation attributes, or other graph metadata changes.

</Tab>
<Tab title="C++" language="cpp">


The C++ frontend describes the same operation as the Python counterpart, but the application creates the cuDNN handle, allocates output and workspace storage, binds device pointers, checks status values, and releases resources itself.

<Code
  src="../samples/cpp/matmul.cc"
  language="cpp"
  maxLines={0}
/>

Compile and run from the repository root:

<Markdown src="/snippets/generated/shell/run_quickstarts/cpp.mdx" />

The example performs these steps:

1. **Fix the problem dimensions.** The constants describe `A[batch, m, k]`, `B[1, k, n]`, and `C[batch, m, n]`. `B` has one batch and is broadcast across `A`; the inner `k` dimensions match, which is required by matrix multiplication.

2. **Allocate unified storage.** `cudaMallocManaged` reserves buffers for both BF16 inputs and the FP32 output. Managed allocations have one address that is accessible from the CPU and GPU, so this first example does not need separate host vectors or explicit host-to-device copies. CUDA migrates or maps the memory as each processor accesses it, and the application remains responsible for releasing it with `cudaFree`.

3. **Initialize deterministic inputs.** The host writes `1` to `A` and `2` to `B` through their managed pointers. Each output is therefore the sum of eight products `1 * 2`, making `16` an exact, easy-to-check expected value. These allocations belong to the application; graph tensor objects created later only describe how cuDNN interprets their memory.

4. **Create a cuDNN handle.** The handle carries the cuDNN execution context, including the CUDA stream on which work is enqueued. This sample uses the handle's default stream. An application with its own stream should call `cudnnSetStream` before building or executing the graph.

5. **Set graph-wide data types.** `set_io_data_type(BFLOAT16)` establishes the default external type. FP32 intermediate and compute types request higher-precision accumulation. Individual tensor descriptors can override these defaults, as the output does.

6. **Describe tensor layouts.** Each `Tensor_attributes` object supplies a name, dimensions, row-major strides, and data type. For example, `A` uses strides `{m * k, k, 1}`, so advancing its batch, row, or column index moves the expected number of elements. `B` uses a leading dimension of one, which expresses batch broadcasting.

7. **Add the operation and expose its result.** `graph.matmul(a, b, ...)` adds a matmul node and returns its output descriptor. `set_output(true)` tells cuDNN that `C` requires application-provided storage. Setting `C` to FP32 keeps the accumulated result in FP32 rather than converting it back to BF16.

8. **Build a reusable execution plan.** `graph.build(...)` validates the graph, queries the `A` and `FALLBACK` heuristic modes for candidate engine configurations, checks whether they are supported by the current cuDNN backend and GPU, and builds a plan. A failed status includes the diagnostic returned by the frontend.

9. **Allocate plan workspace.** `get_workspace_size()` reports temporary GPU memory required by the selected plan. The application allocates it only when the size is nonzero and can reuse that allocation across executions of this graph.

10. **Create the variant pack.** The map binds each graph tensor descriptor to its runtime device pointer. This separates the reusable graph and plan from a particular set of input and output buffers; a later execution can supply different pointers with compatible layouts and types.

11. **Execute the plan.** `graph.execute(...)` enqueues work using the handle, variant pack, and workspace. Because execution is asynchronous with respect to the host, the example calls `cudaDeviceSynchronize()` before the CPU reads the managed output buffer.

12. **Validate and release resources.** After synchronization, the loop reads `c_dev` directly from the host and checks every output against `16.0f`. Small C++17 RAII owners call `cudaFree` for workspace and managed tensors and call `cudnnDestroy` for the handle on both success and early error returns. The application still owns those resources; RAII only makes that ownership exception-safe.

As in Python, build once and execute repeatedly while tensor metadata and operation attributes remain compatible. Rebind the variant pack when only the device addresses change.

</Tab>
</Tabs>
Graph construction runs on the host. Execution consumes the device pointers supplied for non-virtual inputs and outputs. Reuse the built graph when shapes, strides, data types, and supported dynamic-shape overrides remain compatible.
