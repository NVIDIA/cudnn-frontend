# cudnn.norm — sm_100 norm backend (CUTLASS primitives)

JIT normalization kernels for **Blackwell (sm_100)**, written on **CUTLASS
primitives** (`cutlass.experimental.primitives` / `nvvm` cp.async staging, `SmemAllocator`
reductions) and the shared `cudnn.frost.tile_dsl` library — mirroring the
structure of `cudnn.sdpa` and `cudnn.gemm.frost`. Supports **fprop** and
**bprop** for five variants — **LayerNorm, RMSNorm, GroupNorm, InstanceNorm,
BatchNorm** — with **bf16 / fp16 / fp32** I/O (statistics and parameter grads are
fp32). Lower precisions (fp8/fp4/mx) are deferred.

This replaces the previous plain-cute-dsl A100/sm_80 implementation (scalar / vec
/ cp.async / TMA `impl=` variants). That code used only `cute.arch` intrinsics
and warp shuffles; this backend uses CUTLASS primitives as the FROST kernels do.

## Layout

```
norm/
  config_sm100.py      NormVariant + shape derivation (RowwiseSpec/BatchNormSpec)
                       + TemplateParams (frozen/hashable cache key) + Cfg/make_cfg
  dtypes.py            bf16/fp16/fp32 <-> cutlass/torch tables
  utils.py             DLPack -> cute.Tensor helpers
  _common_sm100.py     device helpers: cp.async stage_row + block_reduce_sum2
  graph_analyzer.py    NormGraphFacts + analyze(graph) + variant-pack resolution
  fprop/
    api.py             norm_fprop (torch-tensor dispatch by NormVariant)
    engines.py         cuDNN-graph engines (Capabilities/mismatch/probe/build/lower)
    kernels/
      layernorm_sm100.py  rmsnorm_sm100.py  groupnorm_sm100.py
      instancenorm_sm100.py  batchnorm_sm100.py
  bprop/
    api.py             norm_bprop
    engines.py         cuDNN-graph engines (backward)
    kernels/           layernorm / rmsnorm / groupnorm / instancenorm / batchnorm
  tests/
    test_norms.py      correctness vs PyTorch autograd (fprop + bprop, all dtypes)
    test_engines.py    graph-analyzer facts + engine probe accept/reject
    bench_norms.py     LayerNorm bandwidth vs torch
```

Each flavor has its own kernel entry module. LayerNorm/RMSNorm share the
reduce-last-dim kernel (they differ only in centering); InstanceNorm reuses the
GroupNorm kernel (InstanceNorm = GroupNorm with `num_groups = C`); BatchNorm is a
distinct across-batch kernel (one CTA per channel, no atomics). The shared
cp.async staging and block reduction live in `_common_sm100.py`.

## Public API (direct, non-graph)

```python
from cudnn.norm import NormVariant, norm_fprop, norm_bprop

# LayerNorm over the last dim
y, mean, rstd = norm_fprop(NormVariant.LAYER_NORM, x, gamma, beta,
                           normalized_shape=[D], eps=1e-5)
dx, dgamma, dbeta = norm_bprop(NormVariant.LAYER_NORM, dy, x, gamma, mean, rstd,
                               normalized_shape=[D])

# GroupNorm / InstanceNorm on [N, C, *spatial]
y, mean, rstd = norm_fprop(NormVariant.GROUP_NORM, x, gamma, beta, num_groups=G)
y, mean, rstd = norm_fprop(NormVariant.INSTANCE_NORM, x, gamma, beta)

# RMSNorm (no centering; beta optional)
y, _, rstd = norm_fprop(NormVariant.RMS_NORM, x, gamma, normalized_shape=[D])

# BatchNorm (training updates running stats in place if provided)
y, sm, sr = norm_fprop(NormVariant.BATCH_NORM, x, gamma, beta, training=True,
                       momentum=0.1, running_mean=rm, running_var=rv)
```

## How the kernels use CUTLASS primitives

Norms are memory-bound (no tensor cores), so the primitives that matter are
staging and reductions, not MMA. The row-wise kernels (LN/RMS/GN/IN) take two
compile-time knobs, resolved per shape by `make_cfg`:

**`stage_mode`** — how the row reaches shared memory (X read once, not twice):
- `STAGE_BULK` (**default** when the row is 128-bit aligned and fits 48 KB smem):
  one **`cp.async.bulk`** (`nvvm.cp_async_bulk_shared_cluster_global`, TMA-family)
  stages the whole row, completion signaled by an **mbarrier**. Single
  instruction, no `M % (bt·V)` constraint.
- `STAGE_CPASYNC`: per-thread `cp.async` (`tile_dsl.tma.load_tile` →
  `nvvm.cp_async_shared_global`); needs `M % (bt·V) == 0`. Kept available.
- `STAGE_NONE`: row too big for smem → read X from global directly.

**`vec`** — 128-bit register-fragment load/store (`autovec_copy`) in the compute
loops. Correct and available (all `stage_mode × vec` paths are tested), but **OFF
by default**: measured *slower* than scalar stores for this one-CTA-per-row
pattern (the fragment round-trip through smem adds bank conflicts; scalar global
stores are already coalesced).

Shared primitives: **`SmemAllocator`** (`cutlass.memory`) for staging + reduction
scratch and the mbarrier; **`block_reduce_sum2`** (warp-shuffle + smem combine)
for the two partials (`sum(x)`,`sum(x²)` fwd; `sum(dxhat)`,`sum(dxhat·xhat)` bwd);
`import cutlass.experimental.primitives as nvvm` is the internal NVVM layer, under the name the public
`nvidia-cutlass-dsl` wheels ship (builds that also export `cutlass.primitives` alias it
to the same module). BatchNorm reads
global directly (strided cross-batch access — no contiguous row to stage).

## cuDNN graph engines

`fprop/engines.py` and `bprop/engines.py` register **two** engines total —
`norm_fprop_sm100` and `norm_bprop_sm100` — with the shared `cudnn.frost` engine
framework (one `EngineFamily` per phase in `cudnn/engines/manifest.py`). The variants are not
distinct kernel geometries (unlike SDPA's d256/d512), so one engine per phase
serves *all* variants: its `Capabilities.variants` advertises the set and its
`lower` dispatches to the right per-flavor kernel by `facts.variant` (the
`gemm.frost` single-engine model, not SDPA's engine-per-geometry). The shared
`graph_analyzer.analyze` parses a single `LAYERNORM`/`RMSNORM`/`INSTANCENORM`/
`BATCHNORM` (`_BWD`) node into `NormGraphFacts`; `Capabilities`/`mismatch` does
the judging (facts never judge — same contract as `cudnn.sdpa`). Enable and pin:

```python
import cudnn  # engines self-register lazily when FROST is enabled
# NV_CUDNN_FE_ENABLE_FROST_ENGINES=1
g.select_engines(["norm_fprop_sm100"])   # serves whichever norm variant the graph has
```

**Status of the graph path:** the analyzer/engine logic is unit-tested
(`tests/test_engines.py`) against synthetic nodes matching the real
`cudnn._pygraph` norm schema. End-to-end graph execution (`select_engines` →
`execute`) needs a *built* cuDNN frontend; validate there. Two follow-ups for the
graph path: (1) if `epsilon`/`momentum` arrive as runtime scalar tensors rather
than node params, read them from the variant pack in `lower_*`; (2) cuDNN has no
native GroupNorm node, so the GroupNorm engine is reachable only via the direct
API (the kernel is still used for InstanceNorm graphs).

## Adaptive and zero-centered affine parameters in graphs

The graph engines support adaptive LayerNorm, RMSNorm, and InstanceNorm,
forward and backward. Pass `adaptive=True` to any of these operations:

```python
graph.layernorm(..., adaptive=True)
graph.layernorm_backward(..., adaptive=True)
graph.rmsnorm(..., adaptive=True)
graph.rmsnorm_backward(..., adaptive=True)
graph.instancenorm(..., adaptive=True)
graph.instancenorm_backward(..., adaptive=True)
```

The default `adaptive=False` preserves ordinary normalization. The caller
supplies per-sample scale and optional bias; generating those parameters from
conditioning and applying a residual gate are separate graph operations.

| Normalization | Dense input | Adaptive scale/bias | Reduction axes |
|---|---|---|---|
| LayerNorm / RMSNorm | `[B,S,D]` | `[B,1,D]` | `D` |
| LayerNorm / RMSNorm | `[S,B,D]` | `[1,B,D]` | `D` |
| InstanceNorm | `[N,C,L]`, `[N,C,H,W]`, `[N,C,D,H,W]` | `[N,C,1]`, `[N,C,1,1]`, `[N,C,1,1,1]` | Spatial axes |

Parameter gradients match scale/bias shape and reduce only over their broadcast
axes, retaining the sample dimension. LN/RMSNorm statistics have shape
`X.shape[:-1] + [1]`; InstanceNorm statistics match its adaptive affine shape.
InstanceNorm currently requires channels-first contiguous storage; channels-last
and other strided layouts are declined without repacking.

The existing `graph.adalayernorm(...)` and `graph.adalayernorm_backward(...)`
spellings remain supported. `layernorm(adaptive=True)` and its backward
counterpart create those same AdaLN nodes and retain backend lowering. The
new spelling infers per-token statistics for the rank-3 layouts above; the legacy
spelling keeps its existing inference defaults, so sequence-first callers should
specify the statistics dimensions explicitly.
Adaptive RMSNorm and InstanceNorm are handled by Python engines and have no
backend lowering. Explicit `adaptive=False` keeps backend eligibility.

Zero-centered gamma uses the same graph expression for ordinary/adaptive
LayerNorm and RMSNorm, and adaptive InstanceNorm: pass the virtual result of `gamma + 1` as the scale. For example, with
`X` and `gamma` already declared:

```python
# gamma is [B,1,D] for X[B,S,D]. Scalars are host pass-by-value inputs.
one = graph.tensor_like(torch.ones((1, 1, 1), dtype=torch.float32))
epsilon = graph.tensor_like(torch.full((1, 1, 1), 1e-5, dtype=torch.float32))
effective_gamma = graph.add(gamma, one)
# Declare the virtual scale layout before inferring adaptive statistics.
effective_gamma.set_dim(gamma.get_dim()).set_stride(gamma.get_stride())
effective_gamma.set_data_type(cudnn.data_type.FLOAT)
y, rstd = graph.rmsnorm(
    input=X, scale=effective_gamma, epsilon=epsilon,
    norm_forward_phase=cudnn.norm_forward_phase.TRAINING, adaptive=True,
)
# Bind one to a host tensor containing 1 and epsilon to the desired epsilon.
# Mark and bind y/rstd as outputs through the usual graph API.
```

The addition is fused into gamma loads, including the `DX` calculation in
backward. It rounds to `effective_gamma`'s declared dtype before normalization;
choosing FP32 preserves the FP32 sum, while FP16/BF16 reproduces a rounded
intermediate. `DScale` is also the gradient of the original gamma because the
addition has derivative one. The intermediate scale must stay virtual, and
the host offset must equal one. Either operand order is accepted.

These graph specializations support FP16, BF16, or FP32 inputs and affine
parameters, including mixed input/parameter dtypes, with FP32 computation,
statistics, and parameter gradients. Storage must be dense, row-major, and
16-byte aligned. They use the shared row kernels, compile during plan build,
write caller-provided buffers directly, and require no workspace. Training
and inference forward plans, and backward plans with parameter gradients,
support CUDA graph capture and execution on the handle's stream. LN/RMSNorm backward
parameter reductions use FP32 atomics, so their summation order can vary. AdaIN
reduces spatial gradients within one CTA per sample/channel and writes them
directly, without atomics or a separate gradient-clearing launch.

The ordinary norm schedules remain in use for graphs without these features.
Pytest compares outputs, saved statistics, and gradients with PyTorch functional
normalization and autograd references, including mixed dtypes, zero-centered
gamma, legacy AdaLN equivalence, backend routing, and capture/replay. The
new specializations have correctness and capture coverage; performance
tuning remains separate work. Run their graph tests with:

```bash
pytest test/python/norm/graph/test_frost_norm_affine.py
```

## Environment

Requires Blackwell (SM100 or SM103; Hopper SM90 through the opt-in `norm_*_sm90` rows), a CUDA 13-capable driver, and a CUTLASS DSL build
>= 4.7.0 that exports `cutlass.experimental.primitives` (every public wheel does). The norm entry points
reject builds without this module. Check the active environment before running:

```bash
python -c "import cutlass.experimental.primitives"
pip install torch --index-url https://download.pytorch.org/whl/cu128   # Blackwell
```

## Tests

```bash
pytest test/python/norm/cutedsl/test_frost_norm_kernels.py   # correctness vs PyTorch
pytest test/python/norm/graph/test_frost_norm_engines.py     # engine facts/probe logic
python benchmark/norms/bench_norms.py                        # LayerNorm bandwidth
```

Both correctness suites register a stub `cudnn` package (with a dummy `pygraph`
so the `cudnn.frost` lifecycle patch installs) so the pure-Python `cudnn.norm`
subtree and `cudnn.frost.tile_dsl` import without the compiled cuDNN frontend.

## Performance (LayerNorm forward, sm_100)

One CTA per row, TMA bulk-async staged + scalar stores. Beats
`torch.nn.functional` at large rows, trails at small D where a single CTA
underfills the row:

| N×D | fp32 | fp16 | bf16 |
|-----|------|------|------|
| 16384×8192 | 1.42× | 1.53× | 1.55× |
| 8192×4096  | 0.90× | 0.64× | 0.61× |
| 4096×1024  | 0.16× | 0.15× | 0.15× |

(ratio vs torch; >1 = faster.) Small-D is the main tuning target (warp-per-row).

Measured trade-offs on this GPU (16384×8192): TMA **bulk** vs per-thread
**cp.async** staging is ~a wash (bulk wins fp32, cp.async wins fp16 by ~15%),
both with scalar stores; **vectorized** stores are ~2× *slower* than scalar here,
so `vec` is off by default. TMA-bulk’s real advantage is single-instruction issue
+ no `M % (bt·V)` constraint, not raw bandwidth — as expected, tensor-map TMA
shines for tiled/tensor-core loads, less so for flat reduction rows.

## Status / TODO

- [x] fprop + bprop kernels for all 5 variants, bf16/fp16/fp32, validated vs PyTorch autograd.
- [x] Staging on CUTLASS primitives: TMA **bulk-async** (default) + per-thread **cp.async**, fwd + bwd (reads X once).
- [x] Vectorized 128-bit load/store paths (all `stage_mode × vec` combos tested) — kept but off by default (slower here).
- [x] cuDNN-graph engine layer: 2 engines (`norm_fprop_sm100` / `norm_bprop_sm100`, variant-dispatched) + probe/facts tests.
- [ ] Graph-path e2e validation on a built cuDNN frontend; runtime epsilon/momentum plumbing.
- [ ] Perf: warp-per-row / multi-row CTAs for small D (main gap); block-size autotune.
- [ ] BatchNorm staging/vectorization (currently scalar global, strided cross-batch).
- [ ] Welford (numerically stable) reduction for very large M; lower precisions (fp8/fp4/mx).
```
