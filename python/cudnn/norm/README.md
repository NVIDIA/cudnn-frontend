# cudnn.norm — FROST norm backend (CUTLASS CuTe-DSL)

JIT normalization kernels written directly in CUTLASS **CuTe DSL** primitives,
mirroring the structure of `cudnn.gemm.frost`. Supports **fprop** and **bprop**
for five variants — **LayerNorm, RMSNorm, GroupNorm, BatchNorm, InstanceNorm** —
with **bf16 / fp16 / fp32** I/O (statistics and parameter grads are fp32). Lower
precisions (fp8/fp4/mx) are intentionally deferred.

## Layout

```
norm/
  frost/                     shared framework
    api.py                   norm_forward / norm_backward (torch-tensor dispatch)
    config.py                NormVariant enum + RowwiseSpec/BatchNormSpec derivation
    dtypes.py                bf16/fp16/fp32 <-> cutlass/torch tables
    reductions.py            block_reduce_sum / block_reduce_sum2 (warp-shuffle + smem)
    engine.py                optional cuDNN-graph engine registration (frost_norm_eng0)
  fprop/
    frost/
      rowwise.py             forward: LayerNorm/RMSNorm/InstanceNorm/GroupNorm (shared kernel)
      batchnorm.py           forward: BatchNorm
  bprop/
    frost/
      rowwise.py             backward: LayerNorm/RMSNorm/InstanceNorm/GroupNorm
      batchnorm.py           backward: BatchNorm
  tests/
    test_norms.py            correctness vs PyTorch (fprop + bprop, all dtypes)
```

## Public API

```python
from cudnn.norm.frost import NormVariant, norm_forward, norm_backward

# LayerNorm over the last dim
y, mean, rstd = norm_forward(NormVariant.LAYER_NORM, x, gamma, beta,
                             normalized_shape=[D], eps=1e-5)
dx, dgamma, dbeta = norm_backward(NormVariant.LAYER_NORM, dy, x, gamma, mean, rstd,
                                  normalized_shape=[D])

# GroupNorm / InstanceNorm on [N, C, *spatial]
y, mean, rstd = norm_forward(NormVariant.GROUP_NORM, x, gamma, beta, num_groups=G)
y, mean, rstd = norm_forward(NormVariant.INSTANCE_NORM, x, gamma, beta)

# RMSNorm (no centering; beta optional)
y, _, rstd = norm_forward(NormVariant.RMS_NORM, x, gamma, normalized_shape=[D])

# BatchNorm (training updates running stats in place if provided)
y, sm, sr = norm_forward(NormVariant.BATCH_NORM, x, gamma, beta,
                         training=True, momentum=0.1,
                         running_mean=rm, running_var=rv)
```

## Kernel variants (`impl=`)

`norm_forward`/`norm_backward` take an `impl` selector for the per-sample norms:

| `impl`      | what it does                                                        | status |
|-------------|--------------------------------------------------------------------|--------|
| `"scalar"`  | one element per thread, two-pass (baseline)                        | tested |
| `"vec"`     | 128-bit vectorized load/store via register fragments (`autovec_copy`) | tested |
| `"cpasync"` | stages each row into shared memory once with `cp.async` (reads X once, not twice); forward only, backward falls back to scalar | tested (sm_80) |
| `"tma"`     | TMA bulk-tensor staging (`cp.async.bulk.tensor` + mbarrier)        | sm_90+ only; written, **not runtime-validated** (see below) |
| `"auto"`    | vectorized when the shape allows, else scalar (default)           | tested |

```python
y, mean, rstd = norm_forward(NormVariant.LAYER_NORM, x, g, b, normalized_shape=[D], impl="cpasync")
```

`cpasync`/`tma` require the row to fit in static shared memory (<=48 KB on sm_80),
`M % V == 0`, and `(M // V) % 32 == 0`; otherwise the dispatcher falls back.

**Perf note (A100, forward):** all variants are memory-bound; on this A100 they
currently reach ~40-90% of `torch.nn.functional`'s tuned kernels (see
`tests/bench_norms.py`). `cp.async` staging helps at mid sizes by cutting global
X traffic from 2x to 1x. Beating torch needs more tuning (warp-per-row for small
D, multi-row CTAs, register-resident X, block-size autotune) — tracked in TODO.

**TMA status:** `impl="tma"` (`fprop/frost/rowwise_tma.py`) is written against the
public CuTe-DSL TMA API (`make_tiled_tma_atom`, `tma_partition`, mbarrier,
`cute.copy(..., tma_bar_ptr=)`) and guarded to raise on pre-sm_90 GPUs. It cannot
run on this A100 (no TMA); a compile for `CUTE_DSL_ARCH=sm_90a` still trips a
CuTe-DSL region-isolation check at the tile-selection step. Finish + validate on
Hopper/Blackwell. Use `cpasync` on sm_80.

## Design

The four **per-sample** norms share one kernel. Viewing the (contiguous) input
as `[R, M]` (R = number of normalization groups, M = reduction length), the
affine parameter index for element `(r, j)` is
`(r % groups_per_sample) * channels_per_group + (j // gamma_inner_span)`, which
specializes to each variant (see `config.py`). One CTA owns one group and does a
two-pass reduce → normalize. **BatchNorm** reduces across the batch per channel
and has its own kernel (one CTA per channel, so `dgamma`/`dbeta` need no
atomics; the per-sample backward uses fp32 atomics for the cross-group `dgamma`
/`dbeta` reduction).

Kernels are compiled with `cute.compile` and cached per
`(dtype, structural flags, block_threads)` via `functools.lru_cache`; shapes
(`R`, `M`, group sizes, `eps`) are dynamic kernel arguments, so a shape change
does not force a recompile.

## Environment

The kernels require the CUTLASS CuTe DSL. On the current dev box (A100 / sm_80,
CUDA driver 12.4) use a **cu12** DSL flavor — the `nvidia-cutlass-dsl-internal`
package ships cu13 only, which needs a newer driver. A working venv:

```bash
python3 -m venv norm_venv12 && source norm_venv12/bin/activate
pip install nvidia-cutlass-dsl                 # cu12 default (public)
pip install "cuda-python>=12.8,<13"            # pin bindings to 12.x for the 12.4 driver
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

On a Blackwell (sm_100) box with a cu13-capable driver, install
`nvidia-cutlass-dsl-internal` instead (matches `cudnn.gemm.frost`).

## Tests

```bash
python cudnn/norm/tests/test_norms.py
```

Validates every variant's fprop + bprop against PyTorch (`F.layer_norm`,
`F.rms_norm`, `F.group_norm`, `F.instance_norm`, `F.batch_norm`) and autograd,
for fp32/fp16/bf16. The test registers a lightweight stub `cudnn` package so the
pure-Python `cudnn.norm` subtree imports without the compiled cuDNN extension.

## Status / TODO

- [x] fprop + bprop kernels for all 5 variants, bf16/fp16/fp32, validated vs PyTorch.
- [ ] cuDNN-graph engine wiring (`engine.py` is a scaffold): a `graph_analyzer`
      that reads `graph.nodes` for `NORM_FWD`/`NORM_BWD` and a `probe`/`build`
      registered via `cudnn.frost.register_engine`, mirroring `cudnn.gemm.frost`.
- [x] Vectorized (128-bit) load/store kernels (`impl="vec"`).
- [x] cp.async smem-staged forward (`impl="cpasync"`, reads X once).
- [~] TMA staged forward (`impl="tma"`) — written, needs sm_90+ to finish/validate.
- [ ] cp.async/TMA **backward** (currently backward falls back to scalar).
- [ ] Perf tuning to beat torch: warp-per-row for small D, multi-row CTAs,
      register-resident X reuse, block-size autotune, gamma-in-smem.
- [ ] Welford (numerically stable) reduction for very large M.
- [ ] BatchNorm vectorized/cp.async variants (currently scalar only).
- [ ] Blackwell (sm_100) path using the internal DSL + TMA, matching gemm/frost.
```
