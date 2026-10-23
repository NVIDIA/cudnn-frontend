# Operations and training dataflow

Operation families differ in what forward must preserve for backward and which state crosses iteration boundaries.

## Forward and backward pairs

Convolution has forward, data-gradient, and weight-gradient operations. Pointwise activations often have a backward form that consumes the upstream gradient and either the forward input or output. RoPE backward uses the same frequency convention. SDPA backward needs the forward inputs, output, upstream gradient, and softmax statistics.

## Saved statistics

LayerNorm, BatchNorm, InstanceNorm, RMSNorm, and SDPA can emit statistics used later by backward. Keep them virtual only when backward is in the same graph. Materialize them when another graph or iteration consumes them.

## Running state

BatchNorm training can update running mean and variance. These are application state: bind current values, capture the next values, and define the update order. Do not confuse saved per-batch statistics with long-lived running state.

## Auxiliary outputs

Pooling indices, RNG dumps, amax values, scale tensors, routing metadata, and page-table state may be necessary for backward or the next operation. Their type, shape, and lifetime are part of the operation contract even when they are not the primary result.

## Gradient verification

Verify forward first, then each gradient independently. Match compute and output types, masks, layouts, epsilon, dropout state, and reduction axes. Use finite differences only where dtype and scale make them meaningful.
