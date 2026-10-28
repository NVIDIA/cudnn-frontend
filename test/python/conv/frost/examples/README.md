# C96 causal Conv3D with FROST

[`03_causal_c96.py`](03_causal_c96.py) runs a nine-frame sequence in chunks of
one, four, and four frames through `cudnn.pygraph.conv_fprop`. It checks the
stitched result against an FP32 convolution rounded to BF16 before bias, and
checks the saved history against the final two input frames.

```bash
CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1 \
  python test/python/conv/frost/examples/03_causal_c96.py
```

The C96 path requires SM100, BF16 input/weights/output, FP32 accumulation,
C=96, K=96 or 192, a 3×3×3 filter, one group, unit stride/dilation, and no fused pointwise epilogue.
The dense FROST runtime gate requires CuTe DSL 4.9 or a compatible internal
build. Tensor dimensions are logical NCDHW/KCTRS with compact channels-last
storage and the engine's existing pointer-alignment requirements.

For `h` available history frames, concatenate `[history, current]` and use
pre-padding `(2-h, 1, 1)` and post-padding `(0, 1, 1)`, with `h` in 0, 1, or 2.
The convolution then emits one frame per current input frame. Reset history
between unrelated sequences. The caller owns history and layout conversions;
the graph executes on the supplied tensors. Bias remains a separate BF16
addition after the convolution's BF16 output conversion.

K96 uses three full 32-channel reduction tiles. K192 uses two 64-channel tiles;
TMA zero-fills the final 32 lanes at every filter position. Both routes read
compact C96 storage. Weight conversion can be reused across calls. History
assembly, input/output conversion, bias, and cache copies are per-call costs.

The example explicitly selects `frost_conv` to demonstrate coverage. C96 plans
follow ordinary cuDNN plans in the default ordering; explicit selection and
`OPENSOURCE` queries still expose FROST. Compare complete-operation latency
before selecting a route for an application. Coverage does not imply a speedup.
