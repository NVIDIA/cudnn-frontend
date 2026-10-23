# Numerical precision and block scaling

Storage precision, compute precision, scaling, and output conversion are separate decisions.

## Conventional floating point

FP16 and BF16 commonly use FP32 accumulation. The output can remain FP32 or convert to an IO type. Reduction order and fused implementation can still produce small differences from a framework reference.

## FP8

FP8 extends range or precision through a scale. Operations may consume descale tensors and produce amax values used to update later scales. Hopper supports selected FP8 paths; individual APIs define layout and dimension constraints.

## Block scaling

MXFP8 and NVFP4 associate scales with fixed-size blocks rather than whole tensors. Scale layout, exponent representation, block axis, reorder type, and vectorization must match the consuming kernel. Blackwell provides the target architecture for these paths in the current checkout.

## Error model

Quantization can saturate large values, underflow small values, and add block-dependent relative error. Validate scale computation, amax, quantized distribution, dequantized values, operation output, and task-level quality.

## Compatibility

Never infer support from a data-type enum alone. A format can exist in the API while a particular operation, shape, architecture, or backend plan remains unsupported.
