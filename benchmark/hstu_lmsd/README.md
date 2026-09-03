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
