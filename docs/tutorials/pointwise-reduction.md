# Pointwise and Reduction

This tutorial builds pointwise addition expressions with matching and broadcast shapes, then sums one axis of a tensor with a reduction. This teaches the shape rules shared by arithmetic, activation, comparison, selection, and statistics graphs.

## Before you begin

Complete [Build and extend a graph](/tutorials/first-graph). You need a CUDA-capable GPU
and either CUDA-enabled PyTorch or the C++17/CUDA toolchain. The examples use
FP32 so that their reference checks can use tight tolerances.

## Principles

For a pointwise operation with multiple inputs, every input tensor must have the same rank (number of dimensions). Corresponding dimensions then align: an extent of `1` broadcasts to the other input's extent, while conflicting non-unit extents fail validation. For example, `[3, 4, 5, 6]` and `[3, 4, 1, 6]` align and broadcast happens on the second-last dimension, but `[3, 4, 5, 6]` does not align with `[3, 4, 2, 6]` or `[4, 5, 6]`.

A reduction preserves rank in the Graph API and sets reduced extents to `1`, so reducing `[N, C, H, W]` over `C` produces `[N, 1, H, W]`.

## Pointwise addition

### Matching shapes

The first example shows how to add two tensors that have the same shape.

<Tabs>
<Tab title="Python" language="python">


<Code
  src="../samples/python/add_basic.py"
  language="python"
  maxLines={0}
/>

You can find the detail of the Python API in [`add`](/api/python/add).

</Tab>
<Tab title="C++" language="cpp">


<Code
  src="../samples/cpp/add_basic.cc"
  language="cpp"
  maxLines={0}
/>

The C++ example uses `Pointwise_attributes`; see [`add`](/api/cpp/add) for details.

</Tab>
</Tabs>
### Broadcast one input

In the second example, `A` has shape `[2, 3, 4]` and `B` has shape `[2, 3, 1]`. The last dimension of `B` expands across all four columns, producing an output with shape `[2, 3, 4]`.

<Tabs>
<Tab title="Python" language="python">


<Code
  src="../samples/python/add_broadcast.py"
  language="python"
  maxLines={0}
/>

</Tab>
<Tab title="C++" language="cpp">


<Code
  src="../samples/cpp/add_broadcast.cc"
  language="cpp"
  maxLines={0}
/>

</Tab>
</Tabs>
## Reduction

The third example models a common convolutional-network reduction. The tensor has logical NCHW dimensions `[10, 8, 4, 4]` but uses NHWC (channels-last) memory layout. Summing over the logical `C` dimension produces `[10, 1, 4, 4]`; the memory layout changes the strides, not the dimension labels.

<Tabs>
<Tab title="Python" language="python">


<Code
  src="../samples/python/reduction_basic.py"
  language="python"
  maxLines={0}
/>

You can find the detail of the Python API in [`reduction`](/api/python/reduction).

</Tab>
<Tab title="C++" language="cpp">


<Code
  src="../samples/cpp/reduction_basic.cc"
  language="cpp"
  maxLines={0}
/>

The C++ example uses `Reduction_attributes`; see [`reduction`](/api/cpp/reduction) for details.

</Tab>
</Tabs>
The operation result remains virtual until `set_output(True)`. Comparison operations produce boolean-compatible graph values; `binary_select` uses such a condition to choose broadcast-compatible values. `genstats` is a specialized reduction that returns sum and squared sum for normalization workflows.

Note that reduction on other dimensions may not be supported. You would see an error of "no valid engine configs" when cuDNN compiles the graph.

## Expected result

The matching-shape and broadcast examples produce the same values as ordinary
tensor addition. The reduction example returns shape `[10, 1, 4, 4]` and
matches a channel-wise sum even though the input uses channels-last storage.
