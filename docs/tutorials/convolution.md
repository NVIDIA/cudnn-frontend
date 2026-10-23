# Train a Convolution

Connect convolution forward, data-gradient, and weight-gradient graphs and understand which tensor metadata is shared across training passes.

## Before you begin

Complete [Build and extend a graph](/tutorials/first-graph) and review
[shapes and strides](/concepts/layouts). You need a CUDA-capable GPU and
either CUDA-enabled PyTorch or the C++17/CUDA toolchain. The examples use 2-D
NCHW tensors with contiguous storage, unit stride and dilation, and one element
of padding on each spatial side.

## Define layouts and attributes

Convolution dimensions are logical, while strides define storage. Set padding, filter stride, and dilation explicitly. The spatial rank determines the length of each attribute vector. Forward consumes `X` and `W`; data-gradient consumes `dY` and `W`; weight-gradient consumes `X` and `dY`.

## Build forward

Use the Python [`conv_fprop`](/api/python/conv-fprop) or C++
[`Graph::conv_fprop`](/api/cpp/conv-fprop) API.

<Tabs>
<Tab title="Python" language="python">


<Code
  src="../samples/python/conv_fprop_basic.py"
  language="python"
  maxLines={0}
/>

</Tab>
<Tab title="C++" language="cpp">


<Code
  src="../samples/cpp/conv_fprop_basic.cc"
  language="cpp"
  maxLines={0}
/>

</Tab>
</Tabs>
The output spatial extent follows the padding, dilation, filter, and stride relationship. Mark only the pass result as non-virtual unless a later operation consumes it in the same graph.

## Add gradients

Use the Python [`conv_dgrad`](/api/python/conv-dgrad) or C++
[`Graph::conv_dgrad`](/api/cpp/conv-dgrad) API to produce `dX` with the
original input shape.

<Tabs>
<Tab title="Python" language="python">


<Code
  src="../samples/python/conv_dgrad_basic.py"
  language="python"
  maxLines={0}
/>

</Tab>
<Tab title="C++" language="cpp">


<Code
  src="../samples/cpp/conv_dgrad_basic.cc"
  language="cpp"
  maxLines={0}
/>

</Tab>
</Tabs>
Use the Python [`conv_wgrad`](/api/python/conv-wgrad) or C++
[`Graph::conv_wgrad`](/api/cpp/conv-wgrad) API to produce `dW` with the
filter shape.

<Tabs>
<Tab title="Python" language="python">


<Code
  src="../samples/python/conv_wgrad_basic.py"
  language="python"
  maxLines={0}
/>

</Tab>
<Tab title="C++" language="cpp">


<Code
  src="../samples/cpp/conv_wgrad_basic.cc"
  language="cpp"
  maxLines={0}
/>

</Tab>
</Tabs>
## Verify training dataflow

Compare every pass independently with a framework reference before composing bias, activation, statistics, or reduction nodes. A forward plan being supported does not imply that the two gradient graphs support identical layouts.

## Expected result

The forward, data-gradient, and weight-gradient samples each build and execute
one graph. They preserve the matching tensor shapes across passes and compare
the produced values with the corresponding dense convolution calculation.
