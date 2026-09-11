# SM120 native blk128 BSA benchmark

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
| 14.9776% | strided | 33.5868 ms | 6.6767x | 100.0005% |
| 14.9776% | local | 33.4844 ms | 6.6971x | 100.3064% |
| 20.0000% | strided | 44.8006 ms | 5.0055x | 100.1096% |
| 20.0000% | local | 44.6532 ms | 5.0220x | 100.4401% |

The dense median in that run was 224.2484 ms. The kernel consumes logical
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
