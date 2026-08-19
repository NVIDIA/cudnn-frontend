# Sparse Attention Inference Benchmark

Forward-only microbenchmark of block-sparse attention (BSA), driven through
the public `cudnn.block_sparse_attention_forward` API, with an optional
FA4-lineage CuTe DSL arm (`flash_attn.cute`) on identical block masks.

## What is measured

The workload models video-diffusion sparse attention (e.g. VSA): batch 1,
head_dim 128, bf16, non-causal, and a data-dependent boolean block mask per
head. Masks are seeded top-k selections of KV blocks per query block, with
the diagonal block always kept. Both arms attend exactly the same token set
in every cell.

Three sparse-block granularities are swept — **64, 128, and 256** tokens —
at each requested sparsity, plus one **dense** run per case (every block
selected) that shows the kernel's peak as an upper reference for the sparse
bars.

Reported TFLOPS count only the selected blocks:

```
FLOPs = 4 * H * D * S * keep_blocks_per_row * block_tokens
```

Granularity handling per arm (kept lossless in terms of attended tokens):

- **cuDNN**: 64 and 128 are native block sizes; 256-token masks are
  re-expressed on 128-token blocks.
- **FA4** (`flash_attn.cute`): the SM100 kernel selects at 256-token Q
  granularity with a 128-token KV tile cap. Masks finer than 256 are
  aggregated on the Q side, so rows in a 256-token group attend the union of
  their blocks — real extra work the shared FLOP count does not credit. Its
  TFLOPS are therefore "work done per second on the requested mask", which is
  the deployment-relevant number when the workload's mask is finer than the
  kernel's granularity floor.

## Default cases

Wan2.1 text-to-video shapes at deployed latent sizes (padded to the VSA tile
grid), head_dim 128:

| case | heads | seqlen |
|---|---|---|
| `wan1.3b-480p` | 12 | 39936 |
| `wan14b-480p` | 40 | 39936 |
| `wan14b-720p` | 40 | 92160 |

Custom shapes: `--cases 12x65536` (heads x seqlen). The default sparsity is
0.9, the value video sparse-attention deployments typically target.

## Requirements

- Blackwell (SM100) GPU
- PyTorch with CUDA support
- `pip install nvidia-cudnn-frontend[cutedsl]` (or a development install of
  this repository's `python/` package with the `cutedsl` extra)
- Optional, for the FA4 arm: a flash-attention build providing
  `flash_attn.cute` with block-sparsity support, plus `nvidia-cutlass-dsl`
  and `quack-kernels`. The arm is skipped with a notice when not importable.

## How to run

```bash
python benchmark_sparse_attention_inference.py                    # full default grid
python benchmark_sparse_attention_inference.py --cases wan14b-480p --sparsities 0.8,0.9
python benchmark_sparse_attention_inference.py --csv results.csv --plot results.png
python benchmark_sparse_attention_inference.py --check            # fp32 parity checks
```

Timing uses CUDA events with an adaptive iteration count (~0.5 s per cell
after warmup). `--check` validates every arm and granularity against an fp32
masked-softmax reference at a small shape before trusting the numbers.
