# Train with normalization

Build a normalization forward pass, retain the statistics needed by backward, and connect upstream gradients to input and parameter gradients.

## Before you begin

Complete [Build and extend a graph](/tutorials/first-graph) and review
[training dataflow](/concepts/operations-training). You need a CUDA-capable
GPU and either CUDA-enabled PyTorch or the C++17/CUDA toolchain. The examples
use LayerNorm because its saved mean and inverse variance make the
forward/backward dependency explicit.

## Choose axes and state

LayerNorm and RMSNorm reduce trailing feature axes. BatchNorm and InstanceNorm reduce different batch and spatial axes and can update running state. Training forward APIs return saved mean and inverse variance when their backward operation needs them; inference paths consume established running statistics instead.

## Build forward

<Tabs>
<Tab title="Python" language="python">


<Code
  src="../samples/python/layernorm_basic.py"
  language="python"
  maxLines={0}
/>

</Tab>
<Tab title="C++" language="cpp">


<Code
  src="../samples/cpp/layernorm_basic.cc"
  language="cpp"
  maxLines={0}
/>

</Tab>
</Tabs>
Keep epsilon as a pass-by-value scalar or supported graph tensor. Mark the normalized result as output. Also materialize saved statistics when they cross graph or iteration boundaries.

## Build backward

<Tabs>
<Tab title="Python" language="python">


<Code
  src="../samples/python/layernorm_backward_basic.py"
  language="python"
  maxLines={0}
/>

</Tab>
<Tab title="C++" language="cpp">


<Code
  src="../samples/cpp/layernorm_backward_basic.cc"
  language="cpp"
  maxLines={0}
/>

</Tab>
</Tabs>
Backward consumes the original input, scale, saved statistics, and upstream gradient. It produces input gradient plus scale and bias gradients where the family defines those parameters.

## Extend to other families

Use [Choose a normalization family](/how-to/normalization) for BatchNorm, InstanceNorm, RMSNorm, adaptive LayerNorm, `genstats`, and running-state decisions. Exact contracts are listed in the [normalization API entries](/api/python#l).

## Expected result

The forward samples produce the normalized output plus saved mean and inverse
variance. The backward samples consume those statistics and verify input,
scale, and bias gradients against the corresponding LayerNorm calculation.
