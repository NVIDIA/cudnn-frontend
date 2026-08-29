# DeepSeek-V4 Indexer Preprocessing

`cudnn.ops.fwht_mxfp4_qdq` is an experimental, inference-only CuTe DSL
operation for the precision recipe used by the DeepSeek-V4 sparse indexer. It
fuses:

1. a seven-stage Sylvester-order H128 transform accumulated in FP32;
2. normalization by `1 / sqrt(128)` followed by the recipe's BF16 rounding
   boundary; and
3. independent group-32 finite-E2M1 quantize/dequantize with power-of-two
   UE8M0 scales.

The output is BF16 because the pinned DeepSeek-V4 inference implementation
simulates FP4 Q/K values before its BF16 indexer score contraction.

## Implementation and semantic source

The operator semantics were checked against
[`deepseek-ai/DeepSeek-V4-Pro@b5968e9`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/b5968e9190ef611bbf34a7229255be88a0e937c1),
specifically `inference/model.py::rotate_activation`, the `Indexer` Q path,
the compressor K path, and `inference/kernel.py::fp4_quant_kernel`.

The self-contained CuTe DSL device kernel uses 64 threads and four threads per
row. Each lane loads two aligned
32-byte runs, carries two FP32 values per 64-bit register through the FWHT,
reduces two group maxima in packed BF16, and uses native UE8M0-scaled E2M1
decode.

The FE integration owns its semantic API, checked alignment and empty-input
contract, out-of-place tail-safe launch, current-device stream selection, and
per-device/compute-capability cache. No external optimized kernel is invoked at
runtime.

## Usage

Install the optional CuTe DSL and PyTorch dependencies, then call the semantic
operation directly:

```bash
pip install nvidia-cudnn-frontend[cutedsl]
pip install torch torch-c-dlpack-ext
```

```python
import cudnn

# q: contiguous CUDA BF16 [B, S, H, 128]
q_simulated = cudnn.ops.fwht_mxfp4_qdq(q)

# compressed_k: contiguous CUDA BF16 [B, S // 4, 128]
k_simulated = cudnn.ops.fwht_mxfp4_qdq(compressed_k)
```

The result has the same shape, dtype, and device as the input. The operation is
out-of-place; an integration may replace its Q/K reference with the returned
tensor without mutating the original projection.

Internally, the kernel materializes the recipe's normalized BF16 values before
the group amax, reduces magnitudes directly in packed BF16x2 registers, and uses
packed power-of-two multiply plus native E2M1 conversion for QDQ. This preserves
the official BF16 boundary while shortening register lifetimes; it does not
change the public operator semantics.

## Current contract

- input and output logical shape: `[..., 128]`
- input and output dtype: BF16
- input layout: contiguous and 32-byte aligned
- input values: finite model activations whose FP32 FWHT intermediates remain
  finite
- target characterized so far: B200
- autograd: unsupported
- framework or layout fallback: none

The model-facing API deliberately contains no `Wrapper`, architecture suffix,
tile configuration, or explicit compile lifecycle. Its internal compiled
artifact is cached per CUDA device and compute capability and reused across
dynamic row counts.
Call the operation once before CUDA graph capture so compilation occurs outside
the captured region.

## B200 validation

The focused GPU suite passed 30 tests on an NVIDIA B200. The baseline and
optimized schedules were
bitwise equal at every checked shape, including 15/16/17-row CTA boundaries
and the six DeepSeek model-shaped Q/K tensors. A same-process interleaved
CUDA-event device-interval A/B measured the optimized versus baseline schedule
at 1.046x, 1.049x, and 1.064x for Q row counts 524,288, 1,048,576, and
2,097,152. Compressed-K row counts 2,048 and 4,096 were effectively unchanged;
8,192 rows improved 1.333x.

In a separate same-run semantic-call comparison, the optimized operation was
6.762x, 8.002x, and
8.881x faster than the pinned official fast-Hadamard plus TileLang FP4
composition for the arithmetic Q+K preprocessing pairs at sequence lengths
8,192, 16,384, and 32,768. That measurement excludes the indexer projections,
score GEMM, top-k, sparse attention, assembled indexer layer, and full model.
