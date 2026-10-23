# Advanced: convert graphs to FP8, MXFP8, or NVFP4

Introduce low precision as an explicit quantize, scale, compute, and validate pipeline rather than only changing an enum.

## Before you begin

Establish a BF16 or FP32 baseline first. Check the required GPU architecture,
cuDNN and CUDA versions, scale layout, and dimension multiples for the format;
then complete the [block-scaling tutorial](/tutorials/block-scaling) for
MXFP8 or NVFP4 work.

## Choose the format

FP8 is available on Hopper and later for supported operations. MXFP8 and NVFP4 use block scaling and target Blackwell paths in this checkout. Each path defines scale granularity, packed layout, reorder type, dimension multiples, and output/amax behavior.

## Build the dataflow

1. Compute or receive an amax value in a supported precision.
2. Derive scale and reciprocal scale using the documented block granularity.
3. Quantize and reorder values into the operation's required representation.
4. Bind scale or descale tensors as runtime data.
5. Execute with a supported compute type and materialize output/amax state.
6. Dequantize or compare in higher precision.

The [Python block-scaled matmul sample](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/block_scaled_matmul.py)
and its [C++ counterpart](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/block_scaled_matmul.cc) execute an
MXFP8 dequantize, matmul, and block-scale quantize graph. Use them as the
starting point for checking scale layout and output metadata.

## Validate accuracy and support

Measure saturation, underflow, relative error, and task-level tolerance. Keep a BF16 or FP32 reference. Record the exact SM, cuDNN, CUDA, shapes, layout, and format because support is not interchangeable across low-precision names.

## Verification limits

Python and C++ MXFP8 block-scaled matmul samples and C++ FP8 and MXFP8
attention samples executed on SM100. An NVFP4 end-to-end path and a standalone
block-scale round trip were not verified.

## Expected result

The selected low-precision path remains within the application's measured
error and saturation limits and passes on every deployed architecture. The
published block-scaled matmul and C++ attention samples establish their
specific MXFP8 and FP8 paths on SM100; they do not establish NVFP4 or
arbitrary-operation support.
