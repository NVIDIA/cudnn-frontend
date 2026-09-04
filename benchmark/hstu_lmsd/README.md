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
python -m benchmark.hstu_lmsd.harness --shape smoke --mode all

# Production forward micro benchmark.
python -m benchmark.hstu_lmsd.harness \
    --shape hstu_production --mode forward

# Production forward-to-backward dataflow and a JSON result artifact.
python -m benchmark.hstu_lmsd.harness \
    --shape hstu_production --mode e2e \
    --json benchmark/results/hstu_lmsd_production.json
```

Use `--mode forward` or `--mode backward` for an individual micro benchmark,
`--mode e2e` for one forward followed by one backward, and `--mode all` for all
three measurements. `--warmup` and `--repeats` control sampling.
Their defaults are 25 and 100, matching the benchmark that was previously
embedded in the operation test.

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

The forward kernel exposes only the shipping path. The choices below are fixed,
not user-selectable runtime or compile-time knobs, so the implementation being
reviewed is unambiguous.

| Choice | Rationale |
|---|---|
| 32 threads per row, 2 rows per CTA | Preserves the established LayerNorm reduction order and bitwise output behavior. |
| One-row copy tile | Derives row ownership directly from the warp and avoids multi-row tile address work. |
| Packed FP32x2 pointwise tail | Evaluates independent element pairs while retaining FP32 arithmetic. |
| Reread `x` after the row reduction | Shortens the register live range; the second read is expected to hit cache. |
| Full 32-bit Philox samples | Matches the reference dropout probability instead of quantizing it to 16 bits. |
| 64-bit row pointer rebasing | Keeps large production layouts in one launch without overflowing row byte offsets. |

Historical same-job tuning on Rubin SM107a at `N=2,739,421`, `D=512`, BF16,
and a locked 4752 MHz memory clock measured this configuration at 0.933x the
latency of the preceding CuTe DSL forward kernel (1.824 ms baseline). A
16-thread-per-row variant was faster but changed the LayerNorm accumulation
order, so it is not part of the shipping implementation.

The following alternatives were measured and rejected: TMA/shared-memory
staging, shared-memory output staging, multi-warp row reductions, register
capping, hand-selected Philox integer instructions, and deferred input or
output partitioning. They either added staging/synchronization work, reduced
resident warps, or reproduced instructions that the compiler already emitted.
The detailed measurements remain available in the pull-request discussion;
they are not part of the executable kernel contract.
