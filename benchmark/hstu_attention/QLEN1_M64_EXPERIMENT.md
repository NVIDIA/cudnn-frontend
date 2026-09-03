# HSTU qlen=1 M64 forward experiment

This branch evaluates the smallest one-CTA SM100-family MMA M dimension for
the causal BF16 HSTU workload with `seqlen_q=1`, four heads, head dimension
128, and variable KV lengths averaging about 2048.

## Implemented variants

- `tc-m64*`: M64/N128 QK and PV tiles, with unsplit, split2, and split4
  schedules.
- `tc-m64-n64*`: the same M64 instruction with the KV tile reduced from N128
  to N64.
- `tc-epi1*`: the existing M128 tile with only the warp owning output row zero
  participating in the qlen=1 epilogue.
- `tc-m64-16dp-tail-kv5*`: native 16-data-path TMEM conversion, tail-only
  validity checks, and a five-stage KV pipeline.

M64 uses the native 32-data-path TMEM load/store mapping for its SiLU stage.
Reinterpreting the existing M128 software fragment layout for M64 is not
correct. The boundary runner covers KV lengths 1, 2, 63, 64, 65, 127, 128,
129, 255, 256, 257, 2048, 2049, and 3072; unsplit M64 is bitwise identical to
M128 for those lengths. Split2 and split4 also pass the float32 oracle on B300
and Rubin with maximum absolute error below `1.6e-5`.

## Initial performance result

The interleaved runner rotates implementation order within one process so
node-frequency drift affects each candidate comparably. The stable Rubin
measurements were:

| Batch | Selected M128 (ms) | Best M64/N128 (ms) | M64 delta | Best M64/N64 (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 0.0385 (`split4`) | 0.0411 (`split2`) | +6.7% | 0.0434 (`split4`) |
| 512 | 0.2169 (`split2`) | 0.2221 (`split2`) | +2.4% | 0.2548 (`split4`) |
| 1024 | 0.4222 (`split2`) | 0.4313 (`split2`) | +2.2% | 0.4978 (`split4`) |

On B300, M128+split4 remains about 4% faster than the best M64 schedule at
BS=64. The BS=512 comparisons favor M128 slightly. BS=1024 measurements move
by more than the M64/M128 difference as the shared node changes frequency;
interleaved forward/reverse runs changed which tile won, so there is no
reproducible speedup. Restricting the output epilogue to one warp is correct
but does not materially change either architecture's latency.

Reducing N to 64 is consistently slower because every KV token is useful in
this workload: it doubles the roughly 2K-long KV loop and its TMA/barrier
overhead without eliminating redundant math.

## Final five-stage result

The initial M64 path above inherited a three-stage KV ring. A later pass uses
the shared-memory saved by M64 for five KV stages and moves column validity
checks to the final partial KV block. In the interleaved 36-case sweep this
final M64/N128 implementation is 1.075x faster than M128 on B300 and 1.159x
faster on Rubin in geometric mean. Fixed split2 is not retained because it
regresses short-KV and high-grid cases. The automatic supported path is now
M64/N128, five KV stages, and no split on both architectures.

Run the controlled comparison with:

```bash
python benchmark/hstu_attention/_m64_perf_runner.py \
  --impls tc tc-split2 tc-split4 tc-m64 tc-m64-split2 tc-m64-split4 \
  --batch-sizes 64 512 1024
```
