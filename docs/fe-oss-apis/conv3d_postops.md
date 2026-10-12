# Fused Conv3D + Bias + Residual + RMSNorm + SiLU

**This is an experimental API and subject to change.**

## Overview

Direct Python APIs for BF16 video-VAE inference, tuned for WAN's `base_dim=160`
channel family. The fused normalization variants keep intermediate results on chip:

```text
Conv3D -> bias -> optional residual -> L2 normalization -> sqrt(C) scaling
       -> gamma -> SiLU -> contiguous output or padding + two-frame cache
```

For C640 output, normalization is separate: `Conv3dRawSm100` followed by
`RmsNormSiluPadSm100`. These APIs are not selected automatically by `cudnn.pygraph`.

## Operation Contract

Let `B(x)` denote BF16 conversion. Convolution accumulates in FP32; both fused
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
adds a zero bottom row and right column. Do not reassociate the BF16 additions.

## Requirements

- SM100 or SM103 GPU, `nvidia-cutlass-dsl >= 4.9`, BF16 inference only
- Contiguous NTHWC activation layout, except the causal convolution's NCTHW input
- `3x3x3` convolution with unit stride, unit dilation, and one group
- Valid-convolution inputs must have T/H/W >= 3 (causal convolution pads internally)
- Tensor pointers must be 16-byte aligned; causal `previous` needs 8-byte alignment
  and its strided input needs only BF16 alignment. Output/scratch buffers must not
  overlap operands or each other.
- No autograd: disable grad mode when passing tensors with `requires_grad=True`

| API | Supported channels | Operation |
| --- | --- | --- |
| `Conv3dRawSm100` | `160->160`, `160->320`, `320->320`, `320->640`, `640->640` | Valid convolution without bias |
| `Conv3dRmsNormSiluSm100` | `160->160`, `160->320`, `320->320` | Valid convolution and fused post-operations, contiguous output |
| `Conv3dRmsNormSiluPadSm100` | `160->160`, `160->320`, `320->320` | Same fusion with output padding and current-frame cache writes |
| `Conv3dBiasResidualPadSm100` | `160->160`, `160->320`, `320->320`, `320->640`, `640->640` | Valid convolution, bias, residual, bottom/right padding |
| `CausalConv3dWithCacheSm100` | `12->160` | Two kernels: input/history packing with padding, then convolution without bias |
| `RmsNormSiluPadSm100` | `160`, `320`, `640` | Standalone post-operations, padding, and history/cache copies; no convolution |

Only the listed channels are supported; unsupported channel configurations,
including 128/256/512, have no fallback. Wrappers and classes share these limits.

Wrappers allocate outputs and cache compiled plans. For preallocated buffers,
construct the corresponding class with sample tensors, call `check_support()`
and `compile()` once, then `execute()` with matching metadata. Class execution
does not allocate and supports CUDA Graph capture with stable tensor addresses.
All APIs accept a CUDA driver stream through `current_stream`.

Compilation specializes on tensor shapes, strides, and enabled post-operations;
the first call for a new configuration includes JIT compilation. Wrappers share
a 128-plan FIFO cache. Evicted configurations require plan compilation again;
retain explicit class instances when managing a larger working set.

## Weight Packing

Pack OITRS checkpoint weights once into OTRSI, with each input-channel row
zero-padded to a multiple of 64:

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
output is BF16 NTHWC without bias.

```python
from cudnn import causal_conv3d_with_cache_wrapper_sm100, pack_causal_conv3d_weight_sm100

# Pack once, outside inference. Checkpoint weights have shape [160,12,3,3,3].
packed_weight = pack_causal_conv3d_weight_sm100(checkpoint_weight)
outputs = causal_conv3d_with_cache_wrapper_sm100(chunk, packed_weight, previous=history)
raw_output = outputs["output"]
history = outputs["cache_output"]
```

The packed weight is `[160,448]`: each of the 27 filter positions has 16
channels, with four zero channels, followed by one all-zero filter position.

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

`Conv3dRawSm100` returns BF16 `[N,T-2,H-2,W-2,Co]` for input
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
`residual_output` has the same shape and preserves the pre-normalization BF16
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
- `residual_output`: the pre-normalization BF16 sum when residual fusion is
  requested, otherwise `None`.

The class takes `history_frames` at construction, not a history tensor.
Prefilled history interiors are preserved exactly.

## Standalone Normalization and Padding Variant

`RmsNormSiluPadSm100` applies the same post-operations to an existing BF16
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
- `residual_output`: the pre-normalization BF16 value when a residual is
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
BF16 tensors and CUDA Graph timing. Setup, allocation, and correctness checks are
excluded, as are caller-owned history copies for fused Conv3D padding.

Use `--production` for relevant WAN cases (plus generic `conv3d_rmsnorm_silu`),
or `--shape` for a specific case. `--residual` and `--history` select one setting,
not a sweep. Standalone normalization rows show `-` for the unused Co.

### End-to-End WAN 2.2 VAE Encoding

`benchmark/e2e/Wan2.2-VAE/run_model.py` checks both optimized paths against eager
`AutoencoderKLWan`, then times the compiled Diffusers encoder versus the custom
encoder using these APIs with its remaining Torch operations compiled. Timing
includes history copies and intermediate allocations, but excludes setup and warmup.
The runner reports alternating-order median A/B latency, then uses `_perfshare`
to profile kernel shares separately, outside the speedup measurement.

Defaults use the production architecture, seeded random weights, and synthetic
video. An optional local checkpoint replaces the weights; nothing is downloaded:

```bash
python benchmark/e2e/Wan2.2-VAE/run_model.py
python benchmark/e2e/Wan2.2-VAE/run_model.py \
    --model /path/to/Wan-AI/Wan2.2-TI2V-5B-Diffusers
```

See the [e2e README](../../benchmark/e2e/Wan2.2-VAE/README.md) and
[dependencies](../../benchmark/e2e/Wan2.2-VAE/requirements.txt).
Use `--help` for defaults or `--check-only` for correctness without timing.
