# Resample and pool tensors

Configure pooling or resampling windows with explicit window dimensions, strides, pre-padding, post-padding, and padding mode.

## Before you begin

Use the C++ Graph API and write down the input dimensions, window, stride,
pre-padding, post-padding, rounding rule, and padding mode. Verify the output
shape independently before constructing the graph.

## Determine output shape

For each spatial dimension, derive the output from the input extent, effective window, padding, and stride using the operation's rounding rule. Do not assume symmetric padding when pre and post values differ.

The C++ Graph API provides `resample` with `Resample_attributes`; the old Python binding surface does not expose a matching public `cudnn.pygraph.resample` method in this checkout.

For the max-pooling example below, each spatial input extent is 4, the window
and stride are 2, and padding is 0. The output extent is therefore
`1 + (4 - 2) / 2 = 2`, so `[N, C, H, W] = [1, 1, 4, 4]` becomes
`[1, 1, 2, 2]`.

## Build and execute max pooling

The example fills the input with values 1 through 16, executes non-overlapping
2-by-2 max pooling, and compares all four outputs with an index-based
reference.

<Code
  src="../samples/cpp/resample_basic.cc"
  language="cpp"
  maxLines={0}
/>

Run it from the repository root:

```bash
python tools/doc/run_doc_samples.py --id cpp-resample-basic --require-execution
```

## Preserve training state

Max pooling backward may require generated indices from forward. Materialize and retain that auxiliary tensor with the exact index type and layout expected by backward. Average-pooling padding semantics affect the divisor; choose the documented mode rather than inferring it from a framework default.

See the [C++ `resample` reference](/api/cpp/resample) for attribute and
output-index contracts.

## Expected result

The sample produces shape `[1, 1, 2, 2]` and values `[6, 8, 14, 16]`. When
backward needs forward indices, the application retains and binds that exact
auxiliary tensor. No matching public Python Graph operation exists in this
checkout.
