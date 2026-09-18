# Mamba-2 SSD integration measurement, 2026-09-17

B200 (148 SMs), B=2, L=2048, H=64, P=N=64, G=1, logical chunk=32,
BF16 I/O, FP32 A/D/dt_bias/final state. Forward checkpoints are recomputed in
backward. The default API uses FP32 intermediate storage; BF16 is explicit.

Device times in microseconds, CUDA Graph replay. The pair is timed as an actual
forward followed by backward, rather than the sum of separate measurements.

| Intermediate storage | Gate | FE forward | FE backward | FE pair | Triton pair | Pair speedup |
|---|---|---:|---:|---:|---:|---:|
| BF16 | None | 54.50 | 301.01 | 364.46 | 812.95 | 2.23x |
| BF16 | SiLU | 90.31 | 331.94 | 423.12 | 878.41 | 2.08x |
| FP32 | None | 54.49 | 362.78 | 426.05 | 813.13 | 1.91x |
| FP32 | SiLU | 90.20 | 391.15 | 482.51 | 876.17 | 1.82x |

The JSON files include all raw timing samples, error metrics, source hashes,
software versions and the clean upstream Mamba commit. Both implementations
run in the same process. Complete SSD includes timestep preprocessing, skip,
optional SiLU, state propagation, backward recomputation and reductions. These
are component timings; GatedRMSNorm, convolution, projections, model execution,
compilation, allocation and Python dispatch are outside the timed scope.

## Validation

- 17 Mamba2 pytest cases: independent FP64 output/state and all-gradient
  comparisons, partial chunks (lengths 1/31/32/33/65/128/129), grouped B/C,
  optional state/gate/skip/bias, FP32/BF16 storage, checkpoint reuse, output-only
  and state-only losses, graph plan identity, dependency/architecture/layout
  declines, allocation-free execute, pointer rebinding, nondefault/concurrent
  streams, CUDA Graph training replay and torch.compile training.
- 47 additional targeted regressions: import boundaries, persistent-cache
  infrastructure, API signature parity, knob vocabulary, FROST device contract,
  and existing GDN/KDA/GDN2/GDP routing/default-scale behavior. One import probe
  needed CUDA's lib64 directory on LD_LIBRARY_PATH for its framework-free
  subprocess; it passed after fixing the validation environment.
- Whole target-shape gated and ungated forward/backward plus CUDA Graph replay:
  Compute Sanitizer memcheck reports 0 errors; racecheck reports 0 hazards.
  Both use `--report-api-errors explicit`, without kernel filters or device-error
  suppressions. This excludes a driver-internal module-recompilation warning,
  previously reproduced with a PyTorch-only allocation.
- Cross-process artifact reload: 6 cache hits, 0 misses, with `cute.compile`
  replaced by a function that raises; forward and backward both complete.
- Required Black and SPDX pre-commit hooks passed. The matching Python extension
  was rebuilt from this checkout against cuDNN 9.26 headers.

Numerical gates compare relative RMS error < 1% and max absolute error divided
by the reference peak < 1.5%, with denominator floor 1e-10. These are finite
tensor comparisons, not a claim about training convergence. Only SM100 was
validated; the engine declines other architectures.

Reproduce from the repository root:

```bash
python benchmark/linear_attention/benchmark_mamba2.py \
  --mamba-repo /path/to/mamba --intermediate-dtype bfloat16 --output bf16.json
python benchmark/linear_attention/benchmark_mamba2.py \
  --mamba-repo /path/to/mamba --intermediate-dtype float32 --output fp32.json
cd test/python
pytest -s -q linear_attention/test_mamba2.py
```
