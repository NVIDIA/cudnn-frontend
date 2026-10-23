# Build convolution variants

Start with the core Graph convolution APIs, then select a convenience operator only when its layout, activation, dependency, and architecture constraints match the model.

## Before you begin

Run the [convolution tutorial](/tutorials/convolution) and record the
logical dimensions, storage strides, padding, filter stride, dilation, dtype,
and training requirements of the operation you need to replace.

## Core Graph variants

Use `conv_fprop`, `conv_dgrad`, and `conv_wgrad` for N-D convolution with explicit padding, filter stride, dilation, and layout. Add bias and activation as graph nodes when a supported composed plan exists.

## Causal Conv1d

`cudnn.ops.causal_conv1d` exposes a PyTorch custom operator for causal depthwise-style 1D convolution. The package also contains NWH and back-to-back projection/mixer variants when their compiled bindings are available. These convenience paths own PyTorch autograd registration and use stricter shapes and layouts than general Graph convolution.

See [Run Causal Conv1d operators](/how-to/causal-convolution) for the tensor contracts, high-level variants, low-level pointer bindings, and verification procedure.

## Select one path

Prefer the Graph API when you need general layouts or composition. Prefer `cudnn.ops` when its exact PyTorch contract matches the model and you need its registered fake/autograd behavior. Do not mix their cache or output-allocation assumptions.

Core sample: [convolution forward](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/conv_fprop_basic.py).

## Expected result

The selected path matches the framework reference for forward and every
required gradient. A Graph implementation remains reusable and composable;
the convenience operator additionally provides the documented PyTorch fake,
autograd, and compilation behavior.
