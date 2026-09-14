# SM120 native blk128 BSA benchmark

See [the optimization screening log](SM120_OPTIMIZATION.md) for ongoing
same-device comparisons against the current kernel and the additional 5% goal.

This benchmark covers the native SM120 BF16 block-sparse attention forward
kernel with the target workload:

- Q/K/V: BHSD `[1, 8, 142720, 128]`
- sparse block size: 128
- density: 15% and 20%
- patterns: strided and local
- dense baseline: PyTorch cuDNN SDPA

Run from the repository root after installing the checkout:

```bash
python -u benchmark/bsa/benchmark_sm120_blk128.py --fail-below-target
```

The first calls JIT-compile the CuTe DSL kernel during warmup. CUDA-event
timings therefore report steady-state execution. The benchmark validates the
first and last representative output rows against an FP32 reference and exits
nonzero when the slowest 20% density case is below 4.5x.

One SM120 run produced the following medians. Treat these as a reference, not
as guaranteed numbers; clocks, thermals, and competing work affect results.

| Density | Pattern | Sparse median | Speedup vs dense | Density conversion |
| ---: | --- | ---: | ---: | ---: |
| 14.9776% | strided | 32.5618 ms | 6.8960x | 103.2852% |
| 14.9776% | local | 32.5812 ms | 6.8919x | 103.2234% |
| 20.0000% | strided | 43.4039 ms | 5.1734x | 103.4679% |
| 20.0000% | local | 43.4490 ms | 5.1680x | 103.3605% |

The dense median in that run was 224.5454 ms. The kernel consumes logical
blk128 metadata directly; the benchmark does not expand it to blk64 metadata.

The fixed-top-k, full-KV-block case now selects an FA4-style SM120
specialization. It keeps Q in registers, assigns K/V TMA traffic to a dedicated
load warp, and aliases the K/V shared-memory backing with the O epilogue. A
paired run against the original native blk128 kernel produced:

| Implementation | 20% strided | 20% local |
| --- | ---: | ---: |
| Original native blk128 | 44.8605 ms | 44.6901 ms |
| FA4-style native blk128 | 43.1693 ms | 43.0053 ms |
| Latency reduction | 3.77% | 3.77% |

The specialization is used when `q2k_block_nums=None`, `block_sizes=None`, and
the KV sequence length is divisible by 128. Other blk128 inputs continue to use
the general native kernel. Both paths consume blk128 metadata directly.

The fixed-top-k main loop uses two-fold dynamic-loop unrolling. An interleaved
61-repeat A/B run compared it with the otherwise identical non-unrolled loop;
O and LSE were bit-exact between the two kernels.

| Density | Pattern | Non-unrolled loop | Two-fold unroll | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 14.9776% | strided | 32.5327 ms | 32.3441 ms | 1.0058x |
| 14.9776% | local | 32.6073 ms | 32.4250 ms | 1.0056x |
| 20.0000% | strided | 43.8519 ms | 43.5766 ms | 1.0063x |
| 20.0000% | local | 43.9766 ms | 43.7476 ms | 1.0052x |

The same screening rejected output TMA stores (about 0.1%, below the adoption
threshold), L2 K/V prefetching, fixed-bound specialization, two-Q-tile
persistence, delayed V waits, warp-voted identity rescaling, and source-level
QK loop reordering. Those alternatives were either neutral or slower on the
target workload.
