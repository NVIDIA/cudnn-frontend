# Scaled Dot-Product Attention

In this tutorial, we will build a causal grouped-query attention graph, decide whether training statistics are required, and tie the output to a PyTorch reference. Read [Attention layouts, masks, and state](/concepts/attention) first if you are unfamiliar with the terms BSHD, BHSD, [GQA](https://en.wikipedia.org/wiki/Transformer_%28deep_learning%29#Multi-Query_Attention), or KV-cache.

## Before you begin

Complete [Build and extend a graph](/tutorials/first-graph) first. You need a
CUDA-capable GPU, a cuDNN backend that supports BF16 SDPA forward and backward,
and either CUDA-enabled PyTorch or the C++17/CUDA toolchain. This tutorial uses
small dense tensors; paged, ragged, and reduced-precision attention have
additional requirements.

## Describe Q, K, and V

The standard dense logical shape is `(B, H, S, D)`. Query heads may be a
multiple of key/value heads for GQA. This example uses two query heads and one
key/value head, so both query heads share the same key/value head.

The tensors have logical BHSD dimensions and BSHD physical storage. Their
strides place the sequence axis before the head axis in memory. The upstream
output gradient uses contiguous BHSD storage to demonstrate that supported
inputs to one graph do not all need the same physical layout.

Head dimensions, last-dimension strides, maximum sizes, and supported IO types
depend on the selected implementation and architecture. The Python
[`sdpa`](/api/python/sdpa) and [`sdpa_backward`](/api/python/sdpa-backward)
references and the C++ [`sdpa`](/api/cpp/sdpa) and
[`sdpa_backward`](/api/cpp/sdpa-backward) references state the complete
local constraints.

## Build forward and backward graphs

Training uses two graphs. The forward graph produces the attention output and
FP32 log-sum-exp statistics. The backward graph consumes those values together
with `Q`, `K`, `V`, and an upstream output gradient, then produces `dQ`, `dK`,
and `dV`.

<Tabs>
<Tab title="Python" language="python">


<Code
  src="../samples/python/sdpa_bf16_causal.py"
  language="python"
  maxLines={0}
/>

Run the sample from the repository root after activating an environment
with the frontend and PyTorch installed:

```bash
python docs/samples/python/sdpa_bf16_causal.py --strict
```

The Python wrapper owns the cuDNN handle, output allocation, and workspace.
The sample creates BSHD PyTorch tensors, transposes them into logical BHSD
views, and passes their dimensions and strides to both graphs.

</Tab>
<Tab title="C++" language="cpp">


<Code
  src="../samples/cpp/sdpa_bf16_causal.cc"
  language="cpp"
  maxLines={0}
/>

Compile and run the sample from the repository root:

```bash
nvcc -std=c++17 -Iinclude docs/samples/cpp/sdpa_bf16_causal.cc -lcudnn -lnvrtc -o /tmp/cudnn_fe_sdpa_training
/tmp/cudnn_fe_sdpa_training
```

C++ explicitly owns the cuDNN handle, runtime buffers, variant packs, and
workspace. The forward and backward functions construct separate graph
tensor descriptors that bind to the same saved output and statistics
buffers.

</Tab>
</Tabs>
## Match the Forward and Backward Contracts

You need to pass the scale explicitly even when you want the usual `1 / sqrt(D)`. For a causal mask with top-left alignment, prefer the diagonal-band parameters described on the API page. Set `generate_stats=True` for training because backward consumes log-sum-exp statistics; omit stats for inference-only use cases.

The backward graph must repeat the forward attention scale and mask. Its `Q`,
`K`, `V`, output, and statistics descriptors must also match the metadata used
to produce the saved runtime buffers. Only the upstream gradient uses a
different stride in this example.

The Python sample runs PyTorch scaled dot-product attention with
`enable_gqa=True`, applies the same upstream gradient, and compares the forward
output plus all three gradients. The C++ sample computes the same causal
softmax and its gradients with direct CPU loops, then compares the output,
`dQ`, `dK`, and `dV`.

## Recipes for Variations of Attention

cuDNN supports variations of attention by passing the corresponding arguments
to the `sdpa()` call. The [Configure attention
recipes](/how-to/attention-recipes) page provides examples for causal
masking, dropout, paged KV caches, block masks, and sliding windows. Additional
features in the SDPA API family include sink tokens, score modifier functions,
ragged tensors, FP8, and MXFP8. Ragged tensors use tensor metadata. For reduced
precision, Python provides `sdpa_fp8()` and `sdpa_mxfp8()`, while C++ provides
the corresponding `sdpa_fp8()` overloads.

The executable tutorial verifies the dense BF16 causal path. Each advanced
recipe has its own version, architecture, and feature constraints.

## Expected result

Both samples build and execute causal GQA forward and backward graphs, then
compare the output and all three input gradients with an independent numerical
reference.
