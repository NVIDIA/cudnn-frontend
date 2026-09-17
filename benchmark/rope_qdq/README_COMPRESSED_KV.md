# Compressed-KV RoPE + FP4 QDQ benchmark

This benchmark measures `cudnn.RopeQDQInplace` with
`quantization="fp4", group_size=16, scale_format="e4m3", backend="frost"`.
It uses the same prepared FE API as callers, including runtime validation.
The output is BF16; the measured stage rotates the last 64 channels and
quantizes/dequantizes every group of 16 with E4M3 block scales.

The model contract follows
[DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277/inference).
Supply its `inference` directory with the pinned `model.py`, `kernel.py`,
and `config.json`. The benchmark verifies file hashes and obtains RoPE
frequencies from that source. Activations are synthetic. It runs eight
prefill shapes (batch 1/4, source length 4096/16384, compression ratio 1/2)
and four decode shapes (batch 1/4, ratio 1/2, one emitted latent).
Every latent has 512 channels. Compressor/projection/norm, the indexer,
cache publication, attention, backward, and model throughput are excluded.

Use SM100, Triton >=3.7.0, CUDA-enabled PyTorch with `torch.compile`,
FlashInfer with NVFP4 quantization and native KV dequantization, and the
dependencies required by the pinned DeepSeek source. The benchmark uses
PyTorch's `emulate_precision_casts` compiler option. A toolkit able to compile
SM100 kernels is required; missing controls fail the run.

```bash
python benchmark/rope_qdq/bench_compkv.py \
  --deepseek-source /path/to/DeepSeek-V4.1-Flash/inference \
  --output /path/to/new-results/compkv.json
python benchmark/rope_qdq/audit_compkv_result.py \
  /path/to/new-results/compkv.json
```

Eighteen controls cover the source implementation, FlashInfer RoPE plus
source quantization, eager and compiled Torch math, and FlashInfer's CUDA
and CuTe NVFP4 implementations followed by GPU dequantization. The fused
Torch controls include default and tuned compilation. Their normalization
uses an FP16 rounding step to preserve the source's FP4 midpoint behavior;
the original reciprocal-based variants remain additional controls.
FlashInfer NVFP4 controls use linear block scales and global scale one.
These controls are attributed to their providers; the benchmark's Torch
math, glue, and fused Frost kernel are authored here.

Correctness compares BF16 bits, including signed zero, before timing.
Seven input distributions, changed Graph inputs, guarded storage, pointer
alignment variants, and negative controls are retained. No tolerance is
relaxed for performance comparisons. A control that fails a stress input
still competes if it is exact on both timed distributions; Frost must pass
all seven distributions. Failure tensors and generated artifacts are saved.

Each Graph invokes a provider on three disjoint inputs. Inputs and scratch
are restored outside timing before every replay. GPU events and host wall
time are reported separately for hot and evicted caches, with eight blocks
that alternate provider order. Speedups use the fastest valid native control
for each shape and regime. The auditor independently checks coverage,
artifacts, Graph routes, and timing arithmetic. GPU isolation, thermal state,
and an independent repeat must be verified separately before publishing a
performance claim.
