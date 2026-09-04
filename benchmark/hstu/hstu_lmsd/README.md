# HSTU LMSD Benchmark

This directory benchmarks the explicit HSTU LayerNorm-Multiply-SiLU-Dropout
(LMSD) CuTe DSL operation.

## Files

- `model_shapes.py` defines named smoke and model workloads.
- `executor.py` owns inputs, outputs, workspaces, and precompiled kernels.
- `harness.py` provides the micro and end-to-end benchmark CLI.

## Workloads

| Name | Rows (`N`) | Hidden (`D`) | `u` row stride | Purpose |
|---|---:|---:|---:|---|
| `smoke` | 8,192 | 512 | 2,048 | Fast functional benchmark check |
| `hstu_256k` | 256,000 | 512 | 2,048 | Reduced HSTU workload |
| `hstu_production` | 2,739,421 | 512 | 2,048 | Production HSTU workload |

The sliced `u` layout is intentional: it is created from `(N, 2048)` storage
and viewed as `(N, 512)`, matching the production row stride and large-address
behavior.

## Run

Run from the repository root after installing the CuTe DSL and Torch optional
dependencies. The current kernels require an SM10x GPU and BF16 operands.

```bash
# Forward and backward micro benchmarks plus the op-level E2E benchmark.
python -m benchmark.hstu.hstu_lmsd.harness --shape smoke --mode all

# Production forward micro benchmark.
python -m benchmark.hstu.hstu_lmsd.harness \
    --shape hstu_production --mode forward

# Production forward-to-backward dataflow and a JSON result artifact.
python -m benchmark.hstu.hstu_lmsd.harness \
    --shape hstu_production --mode e2e \
    --json benchmark/results/hstu_lmsd_production.json
```

Use `--mode forward` or `--mode backward` for an individual micro benchmark,
`--mode e2e` for one forward followed by one backward, and `--mode all` for all
three measurements. `--warmup` and `--repeats` control sampling; their defaults
are 25 and 100.

## Measurement Contract

- Tensor allocation, random initialization, support checks, and JIT compilation
  happen before timing.
- Timed execution uses caller-owned outputs and persistent backward reduction
  workspaces, so the micro benchmark measures kernel execution rather than the
  allocating convenience wrappers.
- CUDA events measure device elapsed time. Results report minimum, mean, p50,
  and p95 latency.
- `logical GB/s` counts public tensor reads and writes. It intentionally excludes
  internal dWeight/dBias workspace traffic and cache-line effects, making the
  byte definition stable across implementation changes.
- The E2E case is operator-level training dataflow (`forward -> backward`), not
  a full recommendation model benchmark.

## Kernel Configuration Rationale

The forward kernel has one fixed implementation path. The choices below are not
user-selectable runtime or compile-time knobs.

| Choice | Rationale |
|---|---|
| 32 threads per row, 4 rows per CTA | Gives each warp one independent row while preserving the established LayerNorm reduction order. |
| One-row copy tile | Derives row ownership directly from the warp and avoids multi-row tile address work. |
| Per-warp asynchronous `u` staging | Starts one 1024-byte global-to-shared bulk copy, reduces `x` while it is in flight, then reads `u` with 128-bit shared-memory loads. |
| One current-row stage | Avoids the register, shared-memory, and barrier state required by cross-row prefetch. |
| Scalar FP32 pointwise tail | Preserves the validated instruction schedule and bitwise output behavior. |
| Reread `x` after the row reduction | Shortens the register live range; the second read is expected to hit cache. |
| Streamed full 32-bit Philox samples | Matches the reference dropout probability and consumes each four-word result immediately to shorten its register live range. |
| Load `weight` and `bias` after Philox | Keeps parameter fragments out of the integer-heavy random-number loop. |
| 64-bit row pointer rebasing | Keeps large production layouts in one launch without overflowing row byte offsets. |

Same-job validation on Rubin SM107a at `N=2,739,421`, `D=512`, BF16, a sliced
`u` with row stride 2,048, and a locked 4752 MHz memory clock measured the
previous forward at 2.2224 ms and this runtime-`N` implementation at 2.1029 ms.
That is a 5.38% latency reduction (1.0568x speedup). Each value is the median
of five measurements with 10 warmups and 50 repetitions per measurement.

The retained staging is deliberately narrow: only the current `u` row uses an
asynchronous bulk copy. Full `x`/`u` staging, cross-row prefetch, deeper
pipelines, shared-memory output staging, register capping, and hand-selected
Philox integer instructions were measured and rejected. They added more
staging, synchronization, or live state than the overlap recovered. The
detailed measurements remain available in the pull-request discussion; they
are not part of the executable kernel contract.
