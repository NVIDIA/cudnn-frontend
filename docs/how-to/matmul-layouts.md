# Configure matmul layouts and projections

Describe matrix multiplication with logical dimensions and explicit strides. The final two dimensions are `M x K` for `A` and `K x N` for `B`; leading dimensions are broadcast-compatible batches.

## Before you begin

Complete [Build and extend a graph](/tutorials/first-graph) and review
[logical shapes and physical strides](/concepts/layouts). Write down the
logical `M`, `N`, `K`, batch dimensions, and element strides of both inputs.

## Choose a layout

- Row-major `A[M,K]` uses strides `[K,1]`.
- A transposed logical view can use dimensions `[M,K]` with strides `[1,M]` without moving data, when the selected engine supports it.
- Batched weights can use a leading extent of `1` to broadcast, or a zero leading stride only where the backend contract explicitly permits it.
- Column-major projections and sliced tensors must retain accurate element strides and alignment.

<Tabs>
<Tab title="Python" language="python">


<Code
  src="../samples/python/matmul_basic.py"
  language="python"
  maxLines={0}
/>

</Tab>
<Tab title="C++" language="cpp">


<Code
  src="../samples/cpp/matmul_basic.cc"
  language="cpp"
  maxLines={0}
/>

</Tab>
</Tabs>
## Set precision deliberately

Set the input storage type on each tensor, `compute_data_type=torch.float32` in Python or `compute_data_type=fe::DataType_t::FLOAT` in C++ for FP16/BF16 accumulation where desired, and the output storage type on the returned tensor. Compute type does not automatically choose output type.

## Add projections

Compose bias, activation, or another matmul through virtual intermediates. Verify that the entire graph has a supported plan; support for each isolated operation does not guarantee support for the composition.

See [`matmul`](/api/python/matmul) and the [C++ counterpart](/api/cpp/matmul).

The published sample covers row-major batched broadcasting. Column-major and
transposed-weight layouts are not exercised by that sample.

## Expected result

Both samples execute row-major batched matrix multiplication, including a
broadcast weight batch, and compare the output with an independent result.
Treat column-major, transposed, and sliced layouts as separate cases that need
their own support and numerical checks.
