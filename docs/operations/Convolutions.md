
# Convolutions

## Convolution Fprop

Convolution fprop computes:

$$ response = image * filter $$

### C++ API


```
std::shared_ptr<Tensor_attributes> conv_fprop(std::shared_ptr<Tensor_attributes> image,
                                                  std::shared_ptr<Tensor_attributes> filter,
                                                  Conv_fprop_attributes);
```

Conv_fprop_attributes is a lightweight structure with setters:  
```
Conv_fprop_attributes&
set_padding(std::vector<int64_t>)

Conv_fprop_attributes&
set_stride(std::vector<int64_t>)

Conv_fprop_attributes&
set_dilation(std::vector<int64_t>)

Conv_fprop_attributes&
set_name(std::string const&)

Conv_fprop_attributes&
set_compute_data_type(DataType_t value)

Conv_fprop_attributes&
set_convolution_mode(ConvolutionMode_t mode_)
```

### Python API

- conv_fprop
    - image
    - weight
    - pre_padding
    - post_padding
    - stride
    - dilation
    - convolution_mode
    - compute_data_type
    - name

The symmetric-padding overload accepts `padding` in place of `pre_padding` and `post_padding`.

## CuTe DSL Fused Conv3D + Post-Operations

The experimental direct Python APIs `Conv3dRmsNormSiluSm100` and
`Conv3dRmsNormSiluPadSm100` fuse Conv3D with bias, optional residual addition,
channel RMS normalization, affine scale, and SiLU. Related variants provide
raw Conv3D, bias/residual/spatial padding, and causal input packing plus Conv3D.
These APIs are not selected automatically by `cudnn.pygraph`.

### Current Limitations

- SM100 or SM103 GPUs, `nvidia-cutlass-dsl >= 4.9`, and BF16 or FP32/TF32 inference;
  no backward implementation.
- Fixed `3x3x3` convolution, unit stride and dilation, and one group. The
  valid-convolution APIs consume already-padded input when padding is needed.
- Selected Wan 2.1 and Wan 2.2 encoder/decoder channel pairs. Integrated normalization
  supports Co=96/160/192/256/320; wider outputs compose raw convolution with standalone
  normalization. The causal variant supports `12->160`. See the per-variant table below.
- FP32 convolution requires explicit TF32 permission and input-rounding workspace.
  Strict FP32 convolution stays with the caller, composed with FP32 standalone post-ops.
- Contiguous NTHWC activations, except `CausalConv3dWithCacheSm100`, which accepts
  strided NCTHW input. Weights require variant-specific packing before execution.
- Normalization uses an L2-norm clamp followed by `sqrt(C)` scaling, with fixed
  storage-dtype rounding points; it is not arbitrary-epsilon RMSNorm.
- Padding/cache layouts are fixed for streaming convolution. The fused
  normalization-and-padding variant writes current frames and zeros but leaves
  existing-history interiors for the caller to fill.

See the [fused Conv3D + post-operations API reference](/operations/Convolutions/conv3d_postops)
for the complete per-variant channel table, tensor shapes, normalization math,
weight formats, and history/stream contracts.

## Convolution Dgrad

Convolution dgrad computes data gradient during backpropagation.

### C++ API

```
std::shared_ptr<Tensor_attributes> conv_dgrad(std::shared_ptr<Tensor_attributes> dy,
                                                  std::shared_ptr<Tensor_attributes> w,
                                                  Conv_dgrad_attributes);
```

Conv_dgrad_attributes is a lightweight structure with setters:  
```
Conv_dgrad_attributes&
set_padding(std::vector<int64_t>)

Conv_dgrad_attributes&
set_stride(std::vector<int64_t>)

Conv_dgrad_attributes&
set_dilation(std::vector<int64_t>)

Conv_dgrad_attributes&
set_name(std::string const&)

Conv_dgrad_attributes&
set_compute_data_type(DataType_t value)

Conv_dgrad_attributes&
set_convolution_mode(ConvolutionMode_t mode_)
```

### Python API

- conv_dgrad
    - loss
    - filter
    - pre_padding
    - post_padding
    - stride
    - dilation
    - convolution_mode
    - compute_data_type
    - name

`conv_dgrad` infers the dims of the returned dX from `loss`, `filter` and the padding/stride/dilation
attributes; per spatial axis the inferred extent is
`(S_loss - 1) * stride + (R - 1) * dilation + 1 - (pre_padding + post_padding)` (minus `2 * padding` for
the symmetric-padding overload). With `stride > 1` the forward extent division loses the remainder, so
the inferred dX can be smaller than the forward input even without padding: a 32x32 input with a 3x3
filter infers a 31x31 dX at stride 2 and padding 0, while stride 1 with padding 1 infers 32x32. Set the
dims and strides explicitly to get a dX shaped like the forward input:

```python
dX = graph.conv_dgrad(loss=dY, filter=W, padding=[1, 1], stride=[2, 2], dilation=[1, 1])
dX.set_output(True).set_dim(list(X.get_dim())).set_stride(list(X.get_stride()))
```

The symmetric-padding overload accepts `padding` in place of `pre_padding` and `post_padding`.

## Convolution Wgrad

Convolution wgrad computes weight gradient during backpropagation.

### C++ API

```
std::shared_ptr<Tensor_attributes> conv_wgrad(std::shared_ptr<Tensor_attributes> dy,
                                                  std::shared_ptr<Tensor_attributes> x,
                                                  Conv_wgrad_attributes);
```

Conv_wgrad_attributes is a lightweight structure with setters:  
```
Conv_wgrad_attributes&
set_padding(std::vector<int64_t>)

Conv_wgrad_attributes&
set_stride(std::vector<int64_t>)

Conv_wgrad_attributes&
set_dilation(std::vector<int64_t>)

Conv_wgrad_attributes&
set_name(std::string const&)

Conv_wgrad_attributes&
set_compute_data_type(DataType_t value)

Conv_wgrad_attributes&
set_convolution_mode(ConvolutionMode_t mode_)
```

### Python API

- conv_wgrad
    - image
    - loss
    - pre_padding
    - post_padding
    - stride
    - dilation
    - convolution_mode
    - compute_data_type
    - name

`conv_wgrad` cannot infer the dims of the returned dW from the graph inputs; set them explicitly
(the filter dims/strides) before marking it an output, otherwise graph construction fails with `dims not set`:

```python
dW = graph.conv_wgrad(image=X, loss=dY, padding=[1, 1], stride=[2, 2], dilation=[1, 1])
dW.set_output(True).set_dim(list(W.get_dim())).set_stride(list(W.get_stride()))
```

The symmetric-padding overload accepts `padding` in place of `pre_padding` and `post_padding`.

### Experimental FROST weight gradient

`frost_conv_wgrad` executes a single `graph.conv_wgrad(image=X, loss=dY, ...)`
through the existing graph planning and execution API. Enable experimental
engines with `CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1`; this engine remains behind
backend plans in automatic ranking. Select its plan explicitly when qualifying
it. Forward convolution and weight gradients have separate engine families;
`frost_conv_wgrad` owns slot 0 of the weight-gradient family (engine ID 21200).
Public engine/knob records can be replayed with `create_execution_plan`.

The functional implementation accepts:

- SM100 or SM103 and CuTeDSL 4.9 or newer. SM100 has execution evidence;
  SM103 and other compiler versions/flavors remain unverified.
- Matching X/dY storage from FP16, BF16, FP32 (TF32 MMA), FP8 E4M3, or
  FP8 E5M2, independently combined with any of those five dW storage types.
  Accumulation is FP32. FP8 is unscaled; scale tensors are not accepted.
- Positive rank-5 compact channels-last tensors: logical NCDHW for X,
  NKZPQ for dY, and KCTRS for dW. Declare dW dimensions and strides explicitly.
  Ungrouped cross-correlation only.
- Independent pre/post padding, spatial strides from 1 through 8, and positive
  dilation. Each axis must satisfy `(filter_extent - 1) * dilation <= 31`.
  The rank-5 im2col corners are
  `lower = -padding_before` and
  `upper = padding_after - (filter_extent - 1) * dilation`, both in `[-16, 15]`.
  Padding is nonnegative and output dimensions must be positive.
- Partial channel and reduction tiles. Every tensor address and channel-row
  pitch must be 16-byte aligned: K must be a multiple of `16 / input_bytes`,
  and C a multiple of both `16 / input_bytes` and `16 / output_bytes`.
  `L = N * Z * P * Q` is the reduction length, distinct from output-channel K;
  L need not fill a reduction stage. Descriptor and signed 32-bit scheduler
  bounds are checked before compilation.
- One optional unary epilogue from the existing FROST convolution vocabulary,
  including ReLU slope/clips, leaky-ReLU slope, and swish beta. Its virtual
  intermediate and computation must be FP32. The epilogue runs once after
  the entire reduction, before conversion to dW storage. Extra graph nodes
  and materialized intermediates are declined.

The conservative default remains `TILE_M=128`, `TILE_N=128`, `TILEK=128`
(bytes), `CTA_GROUP=1`, and `STAGES=6`. Public knob replay permits M=64/128,
N=32 through 256 in steps of 32, reduction stages of 64/128/256 bytes,
one-/two-CTA MMA groups, and `STAGES` from 2 through 6 (clamped to the
device's per-block opt-in shared-memory capacity). M is per CTA; N is the
output width of the MMA (the paired CTAs stage half of operand X's N
channels each). Each work item folds two accumulation chains in TMEM (two
inter-tile accumulator stages when `TILE_N <= 128`, one when `TILE_N > 128`),
with two output-store stages.

The implemented TMEM load/store paths impose these epilogue restrictions:

| Per-CTA M | CTA group | dW dtype | Legal TILE_N |
| --- | --- | --- | --- |
| 64 | 1 | FP8 E4M3/E5M2 | Multiples of 64 |
| 64 | 1 | FP16/BF16/FP32 | Multiples of 32 |
| 64 | 2 | All five | Multiples of 64 |
| 128 | 1 or 2 | All five | Multiples of 32 |

Every work item completes the unsplit reduction for one `(K, C)` output tile
and one filter position. The persistent kernel uses tiled dY TMA loads,
im2col X TMA loads, MN-major MMA operands, and tiled dW TMA stores. TF32 uses
128-byte swizzling with 32-byte atomicity. Rank-5 im2col filter offsets are
five bits; the engine rejects larger offsets before kernel compilation. Execution
launches one kernel on the caller's stream, needs zero workspace bytes, and
supports CUDA-graph capture.

Tensor dimensions and convolution geometry are runtime values. Supported shapes
share one compiled artifact per device/target, input/output dtype, pipeline
configuration, and epilogue (including scalar attributes). Unsupported
configurations decline before JIT compilation. There is no performance-based
configuration selector in this milestone.

Replay determinism and batch-reduction invariance are separate: dW reduces the
batch axis, so splitting or reordering that reduction can change FP32 rounding.
See the [determinism support page](/operations/determinism) ([GitHub](https://github.com/NVIDIA/cudnn-frontend/blob/develop/docs/operations/determinism.md))
and [qualification report](https://github.com/NVIDIA/cudnn-frontend/blob/develop/test/python/determinism_goldens/README.md#frost-convolution-weight-gradient).
