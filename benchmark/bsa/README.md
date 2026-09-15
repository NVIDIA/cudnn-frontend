# SM120 native blk128 BSA benchmark

Start with [the reviewer summary](SM120_REVIEW_SUMMARY.md) for the adopted
optimizations, current paired results, limitations, and reproduction commands.

The [same-device three-path comparison](SM120_THREE_PATHS.md) measures current
native KV128 against the original native KV64 kernel and the archived PR #1010
adapter, with identical selected tokens. Two independent runs measured about
1.12x-1.13x against those paths; this is a different baseline from the additional
1.05x goal below. The comparison includes summary tables, a portable harness,
source checks, and archive-preparation instructions. Raw timing JSON is archived
locally rather than tracked in the repository; the harness generates fresh
reports for reproduction.

See [the optimization screening log](SM120_OPTIMIZATION.md) for ongoing
same-device comparisons against the current kernel and the additional 5% goal.

For a kernel-to-kernel A/B comparison, save the baseline version of
`python/cudnn/block_sparse_attention/csrc/fwd/sm120_blk128/bsa_fwd_sm120_fa4.py`
as `baseline.py` under your local ignored agent workspace, then run:

```bash
python benchmark/bsa/benchmark_sm120_blk128_pair.py \
  --baseline-source /path/to/agent/agent_space/baseline.py \
  --repeats 101 --min-speedup 1.05 --fail-below-target \
  --json /path/to/agent/agent_benchmark/paired.json
```

The saved source and installed candidate use the same installed dependencies.
This isolates the kernel-file change; it does not reproduce another checkout's
entire dependency environment. The script compiles both variants, alternates
A/B and B/A launches, checks representative rows against FP32 before and after
timing, and exits nonzero if any requested density/pattern misses the target.
Repeat the run in separate processes before accepting a small gain. A 1.05x
speedup means a 4.76% latency reduction; use `--min-speedup 1.052631579` if the
requirement is a full 5% latency reduction. JSON includes raw event samples and
source hashes, but no host name, IP address, GPU UUID, or source paths.

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

The earlier fixed-top-k main loop introduced two-fold dynamic-loop unrolling.
An interleaved 61-repeat A/B run compared it with the otherwise identical non-unrolled loop;
O and LSE were bit-exact between the two kernels.

| Density | Pattern | Non-unrolled loop | Two-fold unroll | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 14.9776% | strided | 32.5327 ms | 32.3441 ms | 1.0058x |
| 14.9776% | local | 32.6073 ms | 32.4250 ms | 1.0056x |
| 20.0000% | strided | 43.8519 ms | 43.5766 ms | 1.0063x |
| 20.0000% | local | 43.9766 ms | 43.7476 ms | 1.0052x |

The current loop uses four-fold unrolling. Its initial 101-pair run against the
two-fold `9869b9b6` baseline on a Server Edition GPU measured 1.0034x–1.0041x
across both densities and address patterns. This small incremental gain is
**not** the additional 1.05x target. Full O/LSE tensors were bit-exact in a
separate four-case check; FP16/BF16 tests cover the four-way loop and its tails.
See [the screening log](SM120_OPTIMIZATION.md#four-fold-unroll-checkpoint) for
the paired table and acceptance limits.

The latest checkpoint additionally balances an underfilled final scheduling
wave. It only divides those final Q tiles into 64-row work units; every work
unit still loads native KV128 blocks and consumes the original blk128 sparse
metadata. There is no blk64-kernel call, metadata expansion, quantization,
extra scratch allocation, or second kernel launch. SM-count planning happens
at JIT tracing, not on cache hits. The latest 101-pair run against `9869b9b6`
measured 1.0079x–1.0114x across the four cases. This includes the earlier
unrolling gain and **still does not meet 1.05x**. See the
[final-wave checkpoint](SM120_OPTIMIZATION.md#final-wave-scheduling-checkpoint)
for timings, accuracy coverage, and the distinction from internal KV tiling.

The same screening rejected output TMA stores (about 0.1%, below the adoption
threshold), L2 K/V prefetching, fixed-bound specialization, two-Q-tile
persistence, delayed V waits, warp-voted identity rescaling, and source-level
QK loop reordering. Those alternatives were either neutral or slower on the
target workload.
