# Fused Conv3D + Bias + Residual + RMSNorm + SiLU

**This is an experimental API and subject to change.**

## Overview

Direct Python APIs for BF16 and FP32/TF32 video-VAE inference, covering selected
Wan 2.1 and Wan 2.2 encoder/decoder channel families. The fused normalization variants keep intermediate results on chip:

```text
Conv3D -> bias -> optional residual -> L2 normalization -> sqrt(C) scaling
       -> gamma -> SiLU -> contiguous output or padding + two-frame cache
```

For C384/C640/C1024 output, normalization is separate: `Conv3dRawSm100` followed by
`RmsNormSiluPadSm100`. C512 is supported by standalone post-operations only.
These APIs are not selected automatically by `cudnn.pygraph`.

## Operation Contract

Let `B(x)` denote conversion to the storage dtype (BF16 or FP32). Convolution accumulates in FP32; both fused
normalization variants then compute the following, preserving each rounding point:

```text
conv = B(Conv3D(input, packed_weight))
main = B(float(conv) + float(conv_bias))

if residual is present:
    skip = residual
    if residual_bias is present:
        skip = B(float(skip) + float(residual_bias))
    value = B(float(main) + float(skip))
    residual_output = value
else:
    value = main

denominator = max(sqrt(sum(float(value[c])**2 for c in channels)), 1e-12)
normalized = B(float(value) / denominator)
scaled = B(float(normalized) * sqrt(output_channels))
affine = B(float(scaled) * float(gamma))
activated = B(float(affine) / (1 + exp(-float(affine))))
```

Normalization spans the full channel row at each `[n, t, h, w]`. It is L2
normalization followed by `sqrt(C)` scaling, not additive-epsilon RMSNorm.
The bias/residual/padding variant stops at `value`, requires a residual, and
adds a zero bottom row and right column. BF16 preserves the shown intermediate conversions.

## Requirements

- SM100 or SM103 GPU, `nvidia-cutlass-dsl >= 4.9`, BF16 or FP32 inference
- Contiguous NTHWC activation layout, except the causal convolution's NCTHW input
- `3x3x3` convolution with unit stride, unit dilation, and one group
- Valid-convolution inputs must have T/H/W >= 3 (causal convolution pads internally)
- Tensor pointers must be 16-byte aligned; causal `previous` needs 8-byte alignment
  and its strided input needs element alignment. Output/scratch buffers must not
  overlap operands or each other.
- No autograd: disable grad mode when passing tensors with `requires_grad=True`

Supported convolution pairs are `96->96`, `96->192`, `192->192`, `192->384`,
`384->384`, `16->384`, `160->160`, `160->320`, `320->320`, `320->640`,
`640->640`, `512->256`, `256->256`, and `48->1024`.

| API | Supported channels | Operation |
| --- | --- | --- |
| `Conv3dRawSm100` | All pairs above | Valid convolution without bias |
| `Conv3dRmsNormSiluSm100` | Pairs above with Co in 96/160/192/256/320 | Fused post-operations, contiguous output |
| `Conv3dRmsNormSiluPadSm100` | Same as contiguous normalization | Output padding and current-frame cache writes |
| `Conv3dBiasResidualPadSm100` | All pairs above | Bias, residual, bottom/right padding |
| `CausalConv3dWithCacheSm100` | `12->160` | Input/history packing with padding, then convolution |
| `RmsNormSiluPadSm100` | 96/160/192/256/320/384/512/640/1024 | Standalone post-operations, padding, history/cache copies |

Wrappers and classes support the same configurations. Callers select an alternative
implementation for other configurations.

For preallocated buffers, construct the corresponding class with sample tensors,
call `check_support()` and `compile()` once, then `execute()` with compatible
tensors. Preparation validates sample metadata, alignment, non-overlap, and
inference-only use. `TensorDesc` samples validate metadata only; callers supply
storage satisfying the requirements above.

`execute()` binds and launches without repeating validation. Every call must
preserve the prepared shapes, strides, dtypes, devices, optional-tensor presence,
alignment, non-overlap, and inference-only contract. New compatible storage is
allowed. Execution does not allocate and supports CUDA Graph capture with stable
tensor addresses. Execution records tensor stream lifetimes.

Wrappers allocate outputs and validate when creating a cached plan. Cache hits
use the same unchecked execution contract; pointer alignment and autograd state
are not part of the metadata cache key. Use explicit prepared plans to avoid
wrapper allocation and cache-lookup overhead. All APIs accept a CUDA driver
stream through `current_stream`.

Plans bind concrete tensor metadata. Executables specialize on device, channels,
dtype, fusion options, and index width; N/T/H/W, history length, and causal input
strides are runtime values. Compatible plans reuse the shared cuDNN compiled-plan cache, including disk
artifacts keyed by architecture, software environment, and source.
Wrappers additionally retain a 128-plan FIFO cache. Keep explicit plans when
managing a larger working set or preallocated buffers.

## FP32 and TF32

FP32 convolution requires explicit `allow_tf32=True` on both weight packing and
plan/wrapper construction. It uses one TF32 product per operand pair, with FP32
accumulation and periodic FP32 promotion. Input and weight significands round to
nearest/even TF32. Storage and post-operations use FP32.

For valid convolution, a prepared plan takes `sample_workspace` at construction
and `workspace` at execution. This contiguous FP32 buffer has the input's shape
and device. Execution launches **input rounding, then convolution**. The workspace
cost is four bytes per input element. Wrappers allocate it automatically; it must
not overlap other operands. Weight rounding/packing is a one-time preparation step.

```python
import torch
from cudnn import Conv3dRawSm100, pack_conv3d_weight_sm100

packed_weight = pack_conv3d_weight_sm100(weight_fp32, allow_tf32=True)
workspace = torch.empty_like(input_fp32)
plan = Conv3dRawSm100(
    input_fp32, packed_weight, output_fp32,
    allow_tf32=True, sample_workspace=workspace,
)
plan.check_support()
plan.compile()
plan.execute(input_fp32, packed_weight, output_fp32, workspace=workspace)
```

The causal variant rounds during its existing packing kernel and uses
`padded_input` as workspace. Its history/cache retains the original FP32 values.
Standalone `RmsNormSiluPadSm100` supports FP32 without a TF32 flag or extra workspace.
For strict FP32 convolution, the caller runs native cuDNN convolution and composes
these standalone post-operations. Callers select the convolution implementation
and pass precision permissions explicitly through the API.

## Weight Packing

Pack OITRS checkpoint weights once into OTRSI, with each input-channel row
zero-padded to a multiple of 64 for BF16 or 32 for TF32:

```python
from cudnn import pack_conv3d_weight_sm100

packed_weight = pack_conv3d_weight_sm100(checkpoint_weight)
```

Reuse the packed weights; repack after changing the source weights.

## Causal Conv3D with Cache

`CausalConv3dWithCacheSm100` runs streaming causal convolution and returns raw input
history for the next chunk. It executes input packing followed by `12->160` convolution.
It accepts strided NCTHW chunks, inserts up to two raw history frames, adds
causal/spatial zeros, and pads C12 to C16. History is contiguous C12 NTHWC;
output uses the input dtype in NTHWC without bias.

```python
from cudnn import causal_conv3d_with_cache_wrapper_sm100, pack_causal_conv3d_weight_sm100

# Pack once, outside inference. Checkpoint weights have shape [160,12,3,3,3].
packed_weight = pack_causal_conv3d_weight_sm100(checkpoint_weight)
outputs = causal_conv3d_with_cache_wrapper_sm100(chunk, packed_weight, previous=history)
raw_output = outputs["output"]
history = outputs["cache_output"]
```

For BF16, the packed weight is `[160,448]`: each of the 27 filter positions has 16
channels, with four zero channels, followed by one all-zero filter position.
For TF32 it is `[160,3,3,3,32]`, with each C12 filter row rounded and zero-padded to C32.

For input `[N,12,T,H,W]`, the preallocated class takes `padded_input` of shape
`[N,T+2,H+2,W+2,16]`, `output` of shape `[N,T,H,W,160]`, and `cache_output`
of shape `[N,min(2,T+history_frames),H,W,12]`. These output/workspace buffers
must not overlap any operand. Packed weights and output/workspace buffers
must be 16-byte aligned; previous history must be 8-byte aligned.

## Raw Conv3D Variant

```python
from cudnn import conv3d_raw_wrapper_sm100

raw_output = conv3d_raw_wrapper_sm100(input, packed_weight)
```

`Conv3dRawSm100` returns `[N,T-2,H-2,W-2,Co]` in the input dtype for input
`[N,T,H,W,Ci]`, with FP32 accumulation and no bias or post-operations.

## Contiguous Normalization Variant

```python
from cudnn import conv3d_rmsnorm_silu_wrapper_sm100

outputs = conv3d_rmsnorm_silu_wrapper_sm100(
    input, packed_weight, bias, gamma,
    residual=residual,             # optional
    residual_bias=residual_bias,   # optional; requires residual
)
```

For input `[N, T, H, W, Ci]`, `outputs["output"]` has shape
`[N, T-2, H-2, W-2, Co]`. When residual fusion is requested,
`residual_output` has the same shape and preserves the pre-normalization
sum; otherwise it is `None`. The preallocated plan is `Conv3dRmsNormSiluSm100`.

## Normalization and Padding Variant

`Conv3dRmsNormSiluPadSm100` writes current activations into
`padded_output[:, 2:, 1:-1, 1:-1, :]`, zeros spatial borders and missing-history
planes, and writes the last `min(2, current_frames)` activations into the cache tail.
With `history_frames=P` (0-2), existing-history interiors remain untouched.
The wrapper leaves them uninitialized: the caller must fill them before use.
History copies and kernel writes are disjoint and can run in either order on
the same stream. History is already activated and is not normalized again.

```python
from cudnn import conv3d_rmsnorm_silu_pad_wrapper_sm100

outputs = conv3d_rmsnorm_silu_pad_wrapper_sm100(
    input, packed_weight, bias, gamma,
    history_frames=0 if previous_cache is None else previous_cache.shape[1],
    residual=residual,             # optional
    residual_bias=residual_bias,   # optional; requires residual
)

padded = outputs["padded_output"]
cache = outputs["cache_output"]
residual_output = outputs["residual_output"]

# Caller-owned history: these interiors are uninitialized by the wrapper.
if previous_cache is not None:
    history_frames = previous_cache.shape[1]
    padded[:, 2 - history_frames : 2, 1:-1, 1:-1, :].copy_(previous_cache)
    current_frames = input.shape[1] - 2
    old_frames = max(cache.shape[1] - current_frames, 0)
    if old_frames:
        cache[:, :old_frames].copy_(previous_cache[:, -old_frames:])
```

For input `[N, T, H, W, Ci]` and `Co` output channels, the unpadded convolution
shape is `[N, T-2, H-2, W-2, Co]`. The outputs are:

- `padded_output`: `[N, T, H, W, Co]`, with two leading temporal planes and a
  one-pixel spatial border;
- `cache_output`: space for the last `min(2, T-2+P)` activated frames, where
  `P=history_frames`; the kernel writes current frames and the caller writes
  any older frames;
- `residual_output`: the pre-normalization sum when residual fusion is
  requested, otherwise `None`.

The class takes `history_frames` at construction, not a history tensor.
Prefilled history interiors are preserved exactly.

## Standalone Normalization and Padding Variant

`RmsNormSiluPadSm100` applies the same post-operations to an existing BF16 or FP32
tensor, without convolution. Unlike the fused Conv3D variant, it copies history.

```python
from cudnn import rmsnorm_silu_pad_wrapper_sm100

outputs = rmsnorm_silu_pad_wrapper_sm100(
    raw_output, gamma,
    input_bias=conv_bias,          # optional deferred Conv3D bias
    previous=previous_cache,       # optional: one or two activated frames
    residual=residual,             # optional
    residual_bias=residual_bias,   # optional; requires residual
    save_input=False,              # set True to save the value even without residual
)
```

For input `[N, T, H, W, C]` and `P` prior frames (0, 1, or 2), it produces:

- `padded_output`: `[N, T+2, H+2, W+2, C]`;
- `cache_output`: the last `min(2, T+P)` activated frames;
- `residual_output`: the pre-normalization value when a residual is
  supplied or `save_input=True`, otherwise `None`.

Omitting `input_bias` skips bias addition. `save_input=True` saves the
pre-normalization value (including any bias) even without a residual; for the
class API, supply `sample_residual_output` to enable this output.

## Spatial Padding Variant

```python
from cudnn import conv3d_bias_residual_pad_wrapper_sm100

output = conv3d_bias_residual_pad_wrapper_sm100(
    input, packed_weight, bias, residual, residual_bias=residual_bias,
)["padded_output"]
```

`Conv3dBiasResidualPadSm100` returns `[N,T-2,H-1,W-1,Co]`, including the zero
bottom row and right column. No normalization or activation is applied.

## Benchmarks

### Kernel Microbenchmark

`benchmark/conv/cutedsl/benchmark_conv3d_postops.py` compares all variants except
`CausalConv3dWithCacheSm100` against matching `torch.compile` operations using synthetic
BF16 or FP32 tensors and CUDA Graph timing. Setup, allocation, and correctness checks are
excluded, as are caller-owned history copies for fused Conv3D padding. Both paths
prepare weights once outside timing: channels-last-3D for Torch, packed OTRSC for
the custom convolution.

Use `--production` for a compact encoder matrix at 480x832 video resolution:
Wan 2.1 uses batch 1; Wan 2.2 uses batch 32 with spatial sizes after patchification.
Each stage uses its steady-state frame count, with selected T=1 padded-fusion
cases at C96/C192/C160/C320. Raw convolution and one contiguous fused-output case
per fused width provide component comparisons. C384 includes only standalone
preparation, matching the e2e demo's use of Torch/cuDNN convolution for BF16.
The demo uses custom C384 convolution for TF32; benchmark it with `--shape`.
Use `--shape` for other shapes or variants. `--residual` and `--history` apply
uniformly to the selected cases, including T=1. Use `--history 0` for empty history.
Use `--dtype fp32 --allow-tf32` for TF32 convolution; both implementations
receive the same permission and custom timing includes input rounding. Strict FP32
is available with `--dtype fp32 --variant rmsnorm_silu_pad`. The microbenchmark checks
FP32 outputs against the matching Torch implementation with NRMSE <= 1e-3.
Standalone normalization rows show `-` for the unused Co.

### End-to-End Wan 2.1/2.2 VAE Encoding

`benchmark/e2e/Wan2-VAE/run_model.py` checks both paths against eager
`AutoencoderKLWan` with matching arithmetic permissions, then times the compiled
Diffusers encoder or decoder versus the custom implementation. Both compile the
selected component as a full graph. Timing includes history copies and
intermediate allocations. Setup and warmup run before timing. The runner reports
median latency from consecutive samples after warming each path, plus first-call
time and peak additional memory. Use `--profile-path` for separate profiling.

Defaults use the Wan 2.2 production architecture, seeded random weights, and
synthetic video. Use `--wan-version 2.1` for synthetic Wan 2.1; local checkpoints
supply both architecture and weights:

```bash
python benchmark/e2e/Wan2-VAE/run_model.py
python benchmark/e2e/Wan2-VAE/run_model.py --wan-version 2.1 --batch-size 1
python benchmark/e2e/Wan2-VAE/run_model.py \
    --model /path/to/Wan-AI/Wan2.2-TI2V-5B-Diffusers
```

See the [e2e README](../../../benchmark/e2e/Wan2-VAE/README.md) and
[dependencies](../../../benchmark/e2e/Wan2-VAE/requirements.txt).
Use `--help` for defaults or `--check-only` for correctness without timing.
