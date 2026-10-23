# Advanced: block-scaled matmul

Quantize values by fixed-size blocks, execute a low-precision matmul, and track scale tensors on Blackwell. This tutorial assumes familiarity with [numerical precision and block scaling](/concepts/numerical-precision).

## Before you begin

You need an SM100 through SM109 GPU and cuDNN 9.14 or newer. The Python sample
also requires CUDA-enabled PyTorch with `torch.float8_e8m0fnu`. Complete
[Build and extend a graph](/tutorials/first-graph) before working through the packed
scale layouts used here.

## Quantize blocks

The complete example dequantizes two MXFP8 inputs, multiplies them with FP32
accumulation, and block-quantizes the result. It uses zero-valued FP8 inputs so
the final quantized output has an exact zero reference.

<Tabs>
<Tab title="Python" language="python">


<Code
  src="../samples/python/block_scaled_matmul.py"
  language="python"
  maxLines={0}
/>

</Tab>
<Tab title="C++" language="cpp">


<Code
  src="../samples/cpp/block_scaled_matmul.cc"
  language="cpp"
  maxLines={0}
/>

</Tab>
</Tabs>
This MXFP8 path requires cuDNN 9.14 or newer and an SM100 through SM109 GPU.
The Python version also requires a PyTorch build that exposes
`torch.float8_e8m0fnu`.

`block_scale_quantize` emits quantized data and scale factors in the required packed layout. The block size, axis, reorder type, and scale dtype must match the consuming operation. Scales are data, not graph metadata; bind them at execution.

## Dequantize or consume

Use [`block_scale_dequantize`](/api/python/block-scale-dequantize) to reconstruct a higher-precision graph value, or supply the quantized tensor and scales directly to a supported low-precision matmul path. Keep FP32 accumulation when the API permits it and compare against an unquantized reference.

The scale tensors use the `F8_128x4` reordered layout. For input A, a scale
covers 32 values along K; for input B, it covers 32 values along K in the
transposed physical layout. Output quantization uses `axis=2`, so each scale
covers 32 adjacent output columns. Scales are data, not graph metadata; bind
them at execution.

## Measure error

Validate absolute output error and the emitted scale values. Report saturation, underflow, and scale granularity rather than relying on a single tolerance.

The executable example covers MXFP8 on SM100. NVFP4 and dynamic-shape block
scaling have separate support and verification requirements.

## Expected result

Both samples execute dequantize, FP32-accumulating matmul, and block-quantize in
one graph. They verify that zero FP8 inputs produce an exactly zero quantized
output and materialized output scale factors.
