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

For multi-frame K96 outputs, the two-CTA 128-row tile shares each weight tile
across four adjacent output tiles. Separate accumulators preserve the reduction
order while reducing repeated weight loads. This uses no additional caller
workspace.

Initial single-frame K96 calls use a weight scan followed by the convolution.
The scan preserves the NaN/Inf effect of the two fully padded temporal planes,
so the convolution can omit their zero products. These plans declare 384 bytes
of caller workspace through `get_workspace_size()` and refresh the flags on
every execution. Concurrent executions need separate workspace buffers.

The example explicitly selects `frost_conv` to demonstrate coverage. C96 plans
follow ordinary cuDNN plans in the default ordering; explicit selection and
`OPENSOURCE` queries still expose FROST. Compare complete-operation latency
before selecting a route for an application. Coverage does not imply a speedup.

## Optional tiled caller conversions

`--tiled-layout` uses the optional Triton dependency to assemble history and
transpose the input in one kernel, then transpose the output and add bias in a
second kernel. The helpers in `causal_c96_layout.py` are caller code shared by
both providers. `graph.execute()` still receives prepared tensors and caller
workspace. The helpers allocate their returned tensors; compilation must be
warmed up before CUDA graph capture.

```bash
# Install the existing frontend triton extra, or use an environment with Triton.
CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1 \
  python test/python/conv/frost/examples/03_causal_c96.py --tiled-layout
CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1 \
  python test/python/conv/frost/examples/03_causal_c96.py --provider cudnn --tiled-layout
```

The conversion example accepts nonempty CUDA BF16 input with positive strides,
C96, and zero to two history frames. Its output conversion takes compact
channels-last K96/K192 tensors and a contiguous channel bias. Tensor offsets
must fit in signed 32 bits, and batch size must be at most 65535. Bias is added after BF16 convolution rounding.
The ordinary example does not import or require Triton.

The public CuTe DSL 4.8.1 cu12 and cu13 wheels lack the native im2col API used
by dense FROST. Both must decline C96 before importing its kernel template;
`test_frost_conv_c96_dependency_gate.py` checks the installed dependency rather
than substituting a version string. A skipped C96 execution test on these
wheels is not evidence that C96 compiled or ran with CUDA 12. The cuDNN provider
can still use the same tiled caller.
