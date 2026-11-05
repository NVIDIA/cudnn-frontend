# Attention determinism

These five suites check the public cuDNN Frontend APIs. Run from `test/python`:

```bash
pytest linear_attention/frost/test_*_determinism.py QSA/cutedsl/test_qsa_determinism.py -m L0
pytest core/test_determinism_utils.py
```

| Suite | API | Forward results | Backward results |
| --- | --- | --- | --- |
| `TestKDA` | `kimi_delta_attention` | output, final state | dQ, dK, dV, dG, dBeta, dInitialState |
| `TestGDP` | `gated_delta_product`, 1/2/3 Householder steps | output, final state | dQ, dK, dV, dG, dBeta, dInitialState |
| `TestGDN` | `gated_delta_net` | output, final state | dQ, dK, dV, dG, dBeta, dInitialState |
| `TestGDN2` | `gated_delta_net_v2` | output, final state | dQ, dK, dV, dG, dBeta, dW, dInitialState |
| `TestQSA` | `qsa_forward`, `qsa_backward` | BF16 output, FP32 output, LSE | dQ, dK, dV |

The four linear-attention suites live in `linear_attention/frost/test_*_determinism.py`.
They pin FROST and set `batch_invariant=True`. They cover FP16 and BF16 with
nonzero initial state and fixed output/final-state gradients. QSA lives in
`QSA/cutedsl/test_qsa_determinism.py`; its public API has no determinism switch.
It covers BF16 with 12:1, 24:2, and 6:1 query/KV head counts.

Replay checks use fresh inputs and side-stream contention. Batch
checks reuse exact sequence data and upstream gradients while removing,
reordering, and inserting neighbors. They include nonaligned lengths, partial
tiles, and empty linear-attention sequences. QSA retains each sequence's mask
while shifting its packed token and block offsets. Stress cases cross the
8192-token linear-attention span boundary and QSA's dense/sparse forward
boundary, or create more sequence/head work items than GPU SMs.

Comparisons check tensor names, shapes, dtypes, and raw bytes. Signed zero
differences fail. Non-finite results fail, except QSA's specified negative
infinity LSE for an empty attention row. Snapshots own their storage. The
helper tests include signed-zero, NaN-payload, missing-gradient, and golden
mutation controls.

## How to read the test cases

A **suite** is one operation's test class. A **case** is one collected pytest
node after parameter expansion. For example,
`TestKDA::test_replay[bf16-bprop]` is one case: BF16 backward, eight executions.
Those eight executions are not eight reported tests. Backward cases also check
the forward outputs and saved state used to compute gradients.

The expanded set has **156 cases**: 151 operation cases and five harness
controls. This adds 79 cases to the first 77-case set. All 156 cases are L0,
including stress and boundary cases, so the default marker selects the full set.
List the exact node IDs without executing the kernels:

```bash
pytest core/test_determinism_utils.py linear_attention/frost/test_*_determinism.py \
  QSA/cutedsl/test_qsa_determinism.py --collect-only -q
```

| Check | What stays fixed | What changes | Pass condition |
| --- | --- | --- | --- |
| Replay | Input bytes, state, masks, upstream gradients | Fresh buffers and concurrent GPU work | Every result byte matches the first execution |
| Batch invariance | Each selected sequence and its upstream gradients | Batch size, neighbors, order, packed offsets | Its results match the corresponding baseline slice |
| Neighbor isolation | Target sequence, mask, state, upstream gradients | Other sequences' data, state, gradients, and QSA masks | Target results do not change |
| Golden regression | Input hashes and complete runtime key | Frontend source revision | All result hashes match the committed baseline |
| Numerical reference | Mathematical inputs and gradient sources | GPU kernel versus independent FP64 implementation | Existing operation tolerances pass |

The golden cases check replay and the numerical reference before reading the
saved baseline. The new head-layout, GDP step-count, isolated-loss, and QSA
mask-replay cases also check an independent reference. A reference pass alone
does not establish bitwise determinism.

### Linear attention: 28 cases per family, plus four GDP cases

The shared implementation is `linear_attention/determinism.py`. Each FROST
wrapper pins one operation. Except where listed, inputs use BF16, two Q/K/V
heads, K=V=64, and a nonzero FP32 initial state.

| Test method | Level | Parameters and purpose | Cases per family |
| --- | --- | --- | ---: |
| `test_replay` | L0 | FP16/BF16 × fprop/bprop; lengths 65, 0, 17, 1; eight executions; input mutation check | 4 |
| `test_batch_invariance` | L0 | FP16/BF16 × fprop/bprop; removal/reorder; empty state and state-gradient passthrough | 4 |
| `test_golden` | L0 | fprop/bprop; lengths 17, 3; three executions, reference, saved hashes | 2 |
| `test_long_batch_invariance` | L0 | fprop/bprop; lengths 17001, 17; four executions then standalone comparison | 2 |
| `test_many_work_items_replay` | L0 | fprop/bprop; four sequences × enough heads to exceed the SM count; eight executions | 2 |
| `test_tile_boundaries` | L0 | FP16/BF16 × fprop/bprop; lengths 1, 15, 16, 17, 63, 64, 65 | 4 |
| `test_head_layouts` | L0 | fprop/bprop × three head/dimension layouts below; checks shared-head gradient reductions | 6 |
| `test_neighbor_isolation` | L0 | fprop/bprop; fixed 17-token target, changed 65- and 3-token neighbors | 2 |
| `test_gradient_source` | L0 | bprop with output loss only or final-state loss only | 2 |
| GDP `test_householder_steps` | L0 | fprop/bprop × 1/2 steps; the other GDP tests use 3 steps | +4 for GDP |

The three head layouts are `(Q/K heads, V heads, K dimension, V dimension)`:
`(1, 4, 64, 128)`, `(4, 1, 128, 64)`, and `(2, 2, 128, 128)`.
Tile, head-layout, isolated-loss, and GDP step-count cases each run four
executions, then compare every sequence alone, the reversed batch, and a
duplicated first sequence. GDP slices K/V/Beta gradients on the expanded
Householder axis. State tensors use `[sequence, output heads, V, K]`.

### QSA: 35 cases

QSA uses BF16 Q/K/V with head dimension 256. The saved FP32 output and LSE
must match even when the BF16 output rounds to the same value.

| Test method | Level | Parameters and purpose | Cases |
| --- | --- | --- | ---: |
| `test_replay` | L0 | fprop/bprop × Q:KV heads 12:1, 24:2, 6:1; eight executions; input mutation check | 6 |
| `test_batch_invariance` | L0 | Same six combinations; lengths 65, 3, 17; removal/reorder with masks preserved | 6 |
| `test_golden` | L0 | fprop/bprop; lengths 17, 3; three executions, reference, saved hashes | 2 |
| `test_sparse_batch_invariance` | L0 | fprop/bprop; lengths 2057, 3; four executions then standalone comparison | 2 |
| `test_dispatch_boundary` | L0 | fprop/bprop × total tokens 512/513; a fixed 3-token target before/after a neighbor | 4 |
| `test_mask_replay` | L0 | fprop/bprop × all selected, every seventh row empty, all masked; reference and eight executions | 6 |
| `test_mask_batch_invariance` | L0 | Same six combinations; each sequence alone, reversed batch, duplicate sequence | 6 |
| `test_neighbor_isolation` | L0 | fprop/bprop; change neighboring values, masks, and upstream gradients | 2 |
| `test_zero_upstream_gradient_bprop` | L0 | Eight executions with zero dO; all gradients must be zero | 1 |

Head parameters have explicit IDs such as `q24-kv2`. A failure such as
`test_dispatch_boundary[total513-fprop]` identifies the routing boundary and
direction without a separate lookup table. A test stops at its first mismatch;
later checks in that case are not claimed as passes.

## Golden checks

Each golden records SHA-256 hashes of all input and result bytes, with their
shapes and dtypes. This retains full precision without large tensor files.
The runtime key includes the GPU, SM count, Torch, CUDA, driver API version,
cuDNN, CuTe DSL, and TVM FFI.
The frontend revision is deliberately not part of the key: the check must
detect changes across frontend revisions on the same runtime.

Golden cases first pass replay and an independent numerical reference.
Linear attention uses the existing FP64 recurrent references on the CPU.
QSA uses its existing FP64 attention and gradient reference. Reference
tolerances test mathematical correctness; golden and replay checks use no
tolerance. A first execution is only a replay baseline, never a saved golden.

A normal run reads this directory. An unrecorded runtime skips only the golden
comparison; replay and batch checks still run. For CI, require a complete
runtime baseline with an explicit directory. Missing goldens then fail:

```bash
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  pytest linear_attention/frost/test_*_determinism.py QSA/cutedsl/test_qsa_determinism.py -m L0
```

Record candidates into a **new** directory on a known-good revision:

```bash
CUDNN_DETERMINISM_RECORD=/tmp/attention-goldens-candidate \
  pytest linear_attention/frost/test_*_determinism.py QSA/cutedsl/test_qsa_determinism.py -m L0 -k golden
```

Recording refuses to overwrite an existing file. Check all ten cases, inspect
the input/output hashes, and explain any numerical change before replacing
committed files. Never enable recording in a PR check. A new runtime needs its
own reviewed baseline; matching results across architectures is not assumed.

## Agent and review rules

Follow [the root agent rules](../../../AGENTS.md#kernel-determinism-and-batch-invariance)
and [the test guide](../../AGENTS.md). The PR template requires an accompanying
determinism test change for kernel changes, or an explicit unsupported-path
explanation. Extend this table and the suites when adding an operation or a
new numerical path. Keep skips limited to unsupported hardware/dependencies;
a missing FROST plan on a supported runtime is a failure.

The initial scope is the FROST training path and packed public QSA training API
on SM100, SM103, and SM107. It does not establish a contract for cuTile, CAKE,
Hopper, QSA indexed/two-pass APIs, checkpoint/summary/decode paths, shared gate
parameters, or distributed training recipes. Existing tests cover some of
these paths separately. Full recipe CI and an automatic source-to-test PR
gate remain separate integration work; the checklist is a review safeguard.

## Validation: 2026-11-04

Base commit: `da88207aa5b9197b0c84514a7321d5bc3d2683b4` (`develop`).
GPU: SM100, 148 SMs. Runtime: Torch `2.15.0.dev20260820+cu134`, CUDA `13.4`,
cuDNN `92500`, CuTe DSL `4.8.0.dev0`, TVM FFI `0.1.12`.
The complete runtime record is in [the goldens](5f8a825586430e85/kda-fprop.json).
The test banner confirmed imports from this checkout.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest core/test_determinism_utils.py \
  linear_attention/frost/test_*_determinism.py \
  QSA/cutedsl/test_qsa_determinism.py -m L0 -q
```

The GPU run used these same 156 cases before their markers were unified as L0,
with `-m "L0 or L1"`. Test bodies are unchanged. Default L0 collection was
checked after the marker change.

Result: **142 passed, 14 failed, 0 skipped** in 474.36 seconds.
Of the 79 added cases, 73 passed and six failed. All six new failures are QSA.

| Suite | fprop | bprop |
| --- | --- | --- |
| KDA | 13 passed | 15 passed |
| GDP | 15 passed | 17 passed |
| GDN | 13 passed | 15 passed |
| GDN-2 | 13 passed | 15 passed |
| QSA | 15 passed, 2 failed | 6 passed, 12 failed |

All five harness controls passed. All ten golden cases passed both independent
reference checks and saved-byte comparison in the fresh test process.
Black, Python syntax, license headers, and `git diff --check` passed.
The four runtime warnings were existing CuTe DSL allocator deprecations.

All 14 failures are in `QSA/cutedsl/test_qsa_determinism.py::TestQSA`.
The table gives the first mismatch in each test; atomic-race counts can vary.

| Test | First mismatch |
| --- | --- |
| `test_replay[bprop-q12-kv1]` | replay 1 dK: 2 / 43,520 bytes |
| `test_replay[bprop-q24-kv2]` | replay 1 dK: 2 / 87,040 bytes |
| `test_replay[bprop-q6-kv1]` | replay 1 dK: 1 / 43,520 bytes |
| `test_batch_invariance[bprop-q12-kv1]` | standalone sequence 0 dK: 4 / 33,280 bytes |
| `test_batch_invariance[bprop-q24-kv2]` | standalone sequence 0 dK: 1 / 66,560 bytes |
| `test_batch_invariance[bprop-q6-kv1]` | reordered batch `(1, 0)`, sequence 0 dQ: 1 / 199,680 bytes |
| `test_sparse_batch_invariance[fprop]` | saved FP32 output: 237 / 36,864 bytes |
| `test_sparse_batch_invariance[bprop]` | replay 1 dK: 137 / 1,054,720 bytes |
| `test_dispatch_boundary[total513-fprop]` | target at position 0, saved FP32 output: 196 / 36,864 bytes |
| `test_dispatch_boundary[total513-bprop]` | target at position 0, saved FP32 output: 196 / 36,864 bytes; gradient comparison not reached |
| `test_mask_replay[all-selected-bprop]` | replay 1 dK: 1 / 43,520 bytes |
| `test_mask_replay[empty-rows-bprop]` | replay 4 dK: 1 / 43,520 bytes |
| `test_mask_batch_invariance[all-selected-bprop]` | standalone sequence 0 dK: 1 / 33,280 bytes |
| `test_mask_batch_invariance[empty-rows-bprop]` | duplicated sequence 0 dK: 1 / 33,280 bytes |

QSA backward [documents floating-point atomic reductions](../../../docs/operations/sparse_attention/qsa.md).
The dK replay drift is consistent with that reduction order changing. The dQ
batch mismatch needs separate kernel analysis. The backward stress test fails
replay before it reaches the batch comparison.

Both 512-token boundary cases passed. At 513 tokens, fprop and bprop cases
first failed on saved FP32 output; BF16 output matched. This isolates the
batch-dependent forward routing threshold. All-masked rows, changed-neighbor
isolation, and zero-upstream-gradient QSA cases passed in this run. The two
failing mask-replay cases passed the numerical reference before failing exact
byte comparison.

An earlier forward probe confirmed that a 3-token sequence uses the gated
forward path when packed after 2057 tokens, and the ungated path alone.
Both results passed the independent reference. BF16 output was byte-identical,
but saved FP32 output differed by at most `2.384185791015625e-7`; LSE also
differed in 6 / 144 bytes, by at most `5.960464477539063e-8`.
These are batch-invariance failures even though numerical tolerances pass.
The failing tests remain active, with no xfail or relaxed tolerance.

## Method

- [Megatron module tests](https://github.com/NVIDIA/Megatron-LM/tree/main/tests/unit_tests/determinism): fixed inputs, restored state, outputs and gradients.
- [Megatron kernel replay PR](https://github.com/NVIDIA/Megatron-LM/pull/7148): raw-byte comparison, side-stream contention, negative controls, and coverage review.
- [Nemotron functional tests](https://github.com/NVIDIA/Megatron-LM/tree/main/tests/functional_tests/test_cases/nemotron): deterministic execution and saved regression values.
- [Internal recipe CI](https://gitlab-master.nvidia.com/dl/JoC/nemo-ci): follow-up integration target; its contents were not accessible during this change.
