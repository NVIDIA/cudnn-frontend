# Determinism test results

This report covers every operation API tested by
[PR #2266](https://github.com/NVIDIA/cudnn-frontend/pull/2266): KDA, GDP, GDN,
GDN-2, QSA, DSA sparse attention, and DSA indexer forward/backward.

Measured on 2026-11-04 across **285 L0 cases per device**:

- **SM100: 243 passed, 25 failed, 17 skipped.**
- **SM103: 244 passed, 24 failed, 17 skipped.**
- **SM107: 244 passed, 24 failed, 17 skipped.**
- **SM90 Hopper: 23 passed, 0 failed, 262 skipped** (17 FP8 indexer cases and six harness controls passed).

These counts describe the tested shapes and runtime, not a guarantee for every
shape or device. Failures stay active; no xfails or relaxed byte checks are used.

## API and datatype results by architecture

Each count is one pytest case after parameter expansion, not one kernel launch.
Backward cases also check any forward tensors they produce. A failed case stops
at its first mismatch; later comparisons in that case are not passes.
The table assigns each case once, to its direction.

The four linear-attention APIs are in `cudnn.linear_attention.ops`; their bprop
rows use PyTorch autograd through the same API. QSA APIs are in `cudnn.ops`.
`DSA` is imported from `cudnn`.

**SM100, SM103, SM90, and SM107** were measured.
A dash means no measurement. Architecture skips are not passes on another GPU.
SM103/SM107 gates also require a supported CuTe DSL, including the `sm_107a`
target on SM107.

### SM100/103

Each count below is **SM100 / SM103**. These are separate measured runs.
Their software runtimes differ; the results are not a hardware-only comparison.

| API | Direction | Datatype under test | Passed | Failed | Skipped |
| --- | --- | --- | ---: | ---: | ---: |
| `kimi_delta_attention` | fprop | BF16 | 10 / 10 | 0 / 0 | 0 / 0 |
| `kimi_delta_attention` | fprop | FP16 | 3 / 3 | 0 / 0 | 0 / 0 |
| `kimi_delta_attention` | bprop | BF16 | 12 / 12 | 0 / 0 | 0 / 0 |
| `kimi_delta_attention` | bprop | FP16 | 3 / 3 | 0 / 0 | 0 / 0 |
| `gated_delta_product` | fprop | BF16 | 12 / 12 | 0 / 0 | 0 / 0 |
| `gated_delta_product` | fprop | FP16 | 3 / 3 | 0 / 0 | 0 / 0 |
| `gated_delta_product` | bprop | BF16 | 14 / 14 | 0 / 0 | 0 / 0 |
| `gated_delta_product` | bprop | FP16 | 3 / 3 | 0 / 0 | 0 / 0 |
| `gated_delta_net` | fprop | BF16 | 10 / 10 | 0 / 0 | 0 / 0 |
| `gated_delta_net` | fprop | FP16 | 3 / 3 | 0 / 0 | 0 / 0 |
| `gated_delta_net` | bprop | BF16 | 12 / 12 | 0 / 0 | 0 / 0 |
| `gated_delta_net` | bprop | FP16 | 3 / 3 | 0 / 0 | 0 / 0 |
| `gated_delta_net_v2` | fprop | BF16 | 10 / 10 | 0 / 0 | 0 / 0 |
| `gated_delta_net_v2` | fprop | FP16 | 3 / 3 | 0 / 0 | 0 / 0 |
| `gated_delta_net_v2` | bprop | BF16 | 12 / 12 | 0 / 0 | 0 / 0 |
| `gated_delta_net_v2` | bprop | FP16 | 3 / 3 | 0 / 0 | 0 / 0 |
| `qsa_forward` | fprop | BF16 | 15 / 15 | 2 / 2 | 0 / 0 |
| `qsa_backward` | bprop | BF16 | 6 / 7 | 12 / 11 | 0 / 0 |
| `DSA.sparse_attention_forward_wrapper` | fprop | BF16 | 18 / 18 | 0 / 0 | 0 / 0 |
| `DSA.sparse_attention_forward_wrapper` | fprop | FP16 | 6 / 6 | 0 / 0 | 0 / 0 |
| `DSA.sparse_attention_backward_wrapper` | bprop | BF16 | 13 / 13 | 5 / 5 | 0 / 0 |
| `DSA.sparse_attention_backward_wrapper` | bprop | FP16 | 6 / 6 | 0 / 0 | 0 / 0 |
| `DSA.indexer_forward_wrapper` | fprop | BF16 Q/K; BF16 weights | 15 / 15 | 0 / 0 | 0 / 0 |
| `DSA.indexer_forward_wrapper` | fprop | BF16 Q/K; FP32 weights | 9 / 9 | 0 / 0 | 0 / 0 |
| `DSA.indexer_backward_wrapper` | bprop | BF16 Q/K/W; BF16 dK | 4 / 4 | 0 / 0 | 0 / 0 |
| `DSA.indexer_backward_wrapper` | bprop | BF16 Q/K/W; FP32 dK | 9 / 9 | 2 / 2 | 0 / 0 |
| `DSA.dense_indexer_backward_wrapper` | bprop | BF16 Q/K/W; BF16 dK | 1 / 1 | 1 / 1 | 0 / 0 |
| `DSA.dense_indexer_backward_wrapper` | bprop | BF16 Q/K/W; FP32 dK | 2 / 2 | 3 / 3 | 0 / 0 |
| `DSA.indexer_forward_wrapper` | fprop | MXFP8 E4M3 Q/K; E8M0 scales; BF16 weights | 17 / 17 | 0 / 0 | 0 / 0 |
| `DSA.indexer_forward_wrapper` | fprop | SM90 FP8 E4M3 Q/K; FP32 scales; BF16 weights | 0 / 0 | 0 / 0 | 17 / 17 |

### SM90 Hopper

All **17 FP8 indexer cases passed** on SM90, including scores, LSE, replay,
batch changes, and the saved golden. The earlier attention and BF16 indexer
suites still gate on SM100/SM103/SM107; their SM90 rows below are coverage
gaps, not claims that the underlying API is unsupported. The full L0 selection
had 262 architecture skips and six passing harness controls.

| API | Direction | Datatype under test | Passed | Failed | Status |
| --- | --- | --- | ---: | ---: | --- |
| `kimi_delta_attention` | fprop | BF16 | — | — | Not covered by these suites on SM90 |
| `kimi_delta_attention` | fprop | FP16 | — | — | Not covered by these suites on SM90 |
| `kimi_delta_attention` | bprop | BF16 | — | — | Not covered by these suites on SM90 |
| `kimi_delta_attention` | bprop | FP16 | — | — | Not covered by these suites on SM90 |
| `gated_delta_product` | fprop | BF16 | — | — | Not covered by these suites on SM90 |
| `gated_delta_product` | fprop | FP16 | — | — | Not covered by these suites on SM90 |
| `gated_delta_product` | bprop | BF16 | — | — | Not covered by these suites on SM90 |
| `gated_delta_product` | bprop | FP16 | — | — | Not covered by these suites on SM90 |
| `gated_delta_net` | fprop | BF16 | — | — | Not covered by these suites on SM90 |
| `gated_delta_net` | fprop | FP16 | — | — | Not covered by these suites on SM90 |
| `gated_delta_net` | bprop | BF16 | — | — | Not covered by these suites on SM90 |
| `gated_delta_net` | bprop | FP16 | — | — | Not covered by these suites on SM90 |
| `gated_delta_net_v2` | fprop | BF16 | — | — | Not covered by these suites on SM90 |
| `gated_delta_net_v2` | fprop | FP16 | — | — | Not covered by these suites on SM90 |
| `gated_delta_net_v2` | bprop | BF16 | — | — | Not covered by these suites on SM90 |
| `gated_delta_net_v2` | bprop | FP16 | — | — | Not covered by these suites on SM90 |
| `qsa_forward` | fprop | BF16 | — | — | Not covered by these suites on SM90 |
| `qsa_backward` | bprop | BF16 | — | — | Not covered by these suites on SM90 |
| `DSA.sparse_attention_forward_wrapper` | fprop | BF16 | — | — | Not covered by these suites on SM90 |
| `DSA.sparse_attention_forward_wrapper` | fprop | FP16 | — | — | Not covered by these suites on SM90 |
| `DSA.sparse_attention_backward_wrapper` | bprop | BF16 | — | — | Not covered by these suites on SM90 |
| `DSA.sparse_attention_backward_wrapper` | bprop | FP16 | — | — | Not covered by these suites on SM90 |
| `DSA.indexer_forward_wrapper` | fprop | BF16 Q/K; BF16 weights | — | — | Not covered by these suites on SM90 |
| `DSA.indexer_forward_wrapper` | fprop | BF16 Q/K; FP32 weights | — | — | Not covered by these suites on SM90 |
| `DSA.indexer_backward_wrapper` | bprop | BF16 Q/K/W; BF16 dK | — | — | Not covered by these suites on SM90 |
| `DSA.indexer_backward_wrapper` | bprop | BF16 Q/K/W; FP32 dK | — | — | Not covered by these suites on SM90 |
| `DSA.dense_indexer_backward_wrapper` | bprop | BF16 Q/K/W; BF16 dK | — | — | Not covered by these suites on SM90 |
| `DSA.dense_indexer_backward_wrapper` | bprop | BF16 Q/K/W; FP32 dK | — | — | Not covered by these suites on SM90 |
| `DSA.indexer_forward_wrapper` | fprop | MXFP8 E4M3 Q/K; E8M0 scales; BF16 weights | — | — | Unsupported precision on SM90 |
| `DSA.indexer_forward_wrapper` | fprop | SM90 FP8 E4M3 Q/K; FP32 scales; BF16 weights | 17 | 0 | Passed |

### SM107 (Rubin)

All 285 L0 cases ran on SM107: **244 passed, 24 failed, 17 skipped**.
All 17 MXFP8 indexer cases passed. The 17 skips are the SM90-only FP8 cases.
The table covers 279 operation cases; six harness controls also passed.

| API | Direction | Datatype under test | Passed | Failed | Skipped |
| --- | --- | --- | ---: | ---: | ---: |
| `kimi_delta_attention` | fprop | BF16 | 10 | 0 | 0 |
| `kimi_delta_attention` | fprop | FP16 | 3 | 0 | 0 |
| `kimi_delta_attention` | bprop | BF16 | 12 | 0 | 0 |
| `kimi_delta_attention` | bprop | FP16 | 3 | 0 | 0 |
| `gated_delta_product` | fprop | BF16 | 12 | 0 | 0 |
| `gated_delta_product` | fprop | FP16 | 3 | 0 | 0 |
| `gated_delta_product` | bprop | BF16 | 14 | 0 | 0 |
| `gated_delta_product` | bprop | FP16 | 3 | 0 | 0 |
| `gated_delta_net` | fprop | BF16 | 10 | 0 | 0 |
| `gated_delta_net` | fprop | FP16 | 3 | 0 | 0 |
| `gated_delta_net` | bprop | BF16 | 12 | 0 | 0 |
| `gated_delta_net` | bprop | FP16 | 3 | 0 | 0 |
| `gated_delta_net_v2` | fprop | BF16 | 10 | 0 | 0 |
| `gated_delta_net_v2` | fprop | FP16 | 3 | 0 | 0 |
| `gated_delta_net_v2` | bprop | BF16 | 12 | 0 | 0 |
| `gated_delta_net_v2` | bprop | FP16 | 3 | 0 | 0 |
| `qsa_forward` | fprop | BF16 | 15 | 2 | 0 |
| `qsa_backward` | bprop | BF16 | 6 | 12 | 0 |
| `DSA.sparse_attention_forward_wrapper` | fprop | BF16 | 18 | 0 | 0 |
| `DSA.sparse_attention_forward_wrapper` | fprop | FP16 | 6 | 0 | 0 |
| `DSA.sparse_attention_backward_wrapper` | bprop | BF16 | 14 | 4 | 0 |
| `DSA.sparse_attention_backward_wrapper` | bprop | FP16 | 6 | 0 | 0 |
| `DSA.indexer_forward_wrapper` | fprop | BF16 Q/K; BF16 weights | 15 | 0 | 0 |
| `DSA.indexer_forward_wrapper` | fprop | BF16 Q/K; FP32 weights | 9 | 0 | 0 |
| `DSA.indexer_backward_wrapper` | bprop | BF16 Q/K/W; BF16 dK | 4 | 0 | 0 |
| `DSA.indexer_backward_wrapper` | bprop | BF16 Q/K/W; FP32 dK | 9 | 2 | 0 |
| `DSA.dense_indexer_backward_wrapper` | bprop | BF16 Q/K/W; BF16 dK | 1 | 1 | 0 |
| `DSA.dense_indexer_backward_wrapper` | bprop | BF16 Q/K/W; FP32 dK | 2 | 3 | 0 |
| `DSA.indexer_forward_wrapper` | fprop | MXFP8 E4M3 Q/K; E8M0 scales; BF16 weights | 17 | 0 | 0 |
| `DSA.indexer_forward_wrapper` | fprop | SM90 FP8 E4M3 Q/K; FP32 scales; BF16 weights | 0 | 0 | 17 |

The SM100/103 table covers **279 operation cases per GPU**: SM100 has
237 passed, 25 failed, and 17 skips; SM103 has 238 passed, 24 failed, and
17 skips. Six harness controls passed on each GPU, giving 285 cases per run.
All **16 operation goldens on each of SM100, SM103, and SM107** passed
replay, an independent reference, and the saved-byte comparison. The SM90 FP8
golden also passed on Hopper and has its own runtime record. Other GPU/runtime
combinations need their own reference-checked records.

Datatype details:

- Linear attention uses FP16/BF16 Q/K/V, with FP32 initial/final state and state
  gradients. Golden, head-layout, long-sequence, and isolated-loss cases use BF16.
- QSA uses BF16 Q/K/V and gradients; saved FP32 output and LSE are also compared.
- DSA sparse attention uses FP16/BF16 Q/KV and gradients, FP32 sink/statistics
  and dSink, and INT32 sparse indices/lengths. Replay and batch checks cover both
  input dtypes. Selection, boundary, neighbor, and golden cases use BF16.
- Indexer forward uses BF16 Q/K, BF16 or FP32 weights, and FP32 scores. Both
  weight dtypes cover replay, batch changes, and ragged packing; other forward
  cases use BF16 weights.
- MXFP8 forward uses E4M3 Q/K and E8M0 block scales, one scale per 32 values.
  Hopper FP8 uses E4M3 Q/K and FP32 token/head descales. Both new test paths
  use BF16 weights and FP32 scores; Hopper also compares FP32 LSE.
- Both indexer backward APIs use BF16 Q/K/weights and dQ/dWeights. Score inputs,
  normalizers, mutated score buffers, and loss gradients use FP32. dK uses BF16
  or FP32. Replay and basic batch checks cover both dK dtypes; the other backward
  cases use FP32 dK. Sparse indices use INT32.

FP8/MXFP8 support limits are separate from hardware test status:

| API | SM100/SM103 | SM90 Hopper | SM107 |
| --- | --- | --- | --- |
| Sparse attention fprop/bprop with FP8/MXFP8 inputs | Unsupported; FP16/BF16 only | Unsupported; FP16/BF16 only | Unsupported; FP16/BF16 only |
| Indexer forward | MXFP8 supported | FP8 token/head scales; MXFP8 unsupported | MXFP8 supported |
| Sparse/dense indexer backward with FP8/MXFP8 inputs | Unsupported; BF16 Q/K/W required | Unsupported; BF16 Q/K/W required | Unsupported; BF16 Q/K/W required |

## What is checked

Replay restores inputs, uses fresh buffers, adds concurrent GPU work, and
compares raw result bytes. Batch tests compare a fixed sequence alone, with
neighbors, after reorder, and after duplication. Neighbor tests change other
sequences while keeping the target fixed. Numerical references are separate
from byte comparisons. Goldens store input/output hashes for a fixed runtime.

The four linear-attention APIs select FROST with `batch_invariant=True`.
DSA sparse attention and sparse indexer backward use `deterministic=True`.
QSA and dense indexer backward expose no determinism switch.

Indexer forward covers H32/H64, head dimension 128, causal ratios 1/4,
127/128/129-row and key boundaries, masked rows, and ragged query/key lengths.
Sparse backward covers H32/H64, local/global Top-K IDs, negative padding,
Top-K 128/896/2048, and 193 query rows over 96 deterministic producer planes.
Dense backward covers H64, head dimension 128, and causal ratio 1.

The new quantized forward cases cover H32/H64, ratios 1/4, 127/128/129
boundaries, empty rows, ragged batches, changed neighbors, and scale-only
changes. Each sequence is quantized once from CPU-seeded data. Batch changes
preserve its quantized payload and logical scale bytes; MXFP8 packing rebuilds
only the physical scale layout and THD padded prefixes. Comparisons include
scores, Hopper LSE, and input/scale mutation checks. References use dequantized
inputs and FP64 score computation. Golden hashes include the FP8 payload and
physical scale buffers. No backward FP8 kernel is claimed by these tests.

Indexer backward restores each API's mutable score buffers before every call.
It compares dQ/dWeights/dK and the mutated scores. Sparse scratch starts at
`0xA5` and gradient outputs start at NaN. `loss_coeff = B * S_q` keeps the
normalization fixed when batch size changes; nonzero `grad_loss` stays at 0.75.
Zero-loss cases require all gradients to be exactly zero.

See the [full coverage and attention failure report](https://github.com/Anerudhan/cudnn-frontend/blob/feat/attention-determinism-tests/test/python/determinism_goldens/README.md)
for each test method, shape, and case count.

## Failures

SM90 had no failed cases.

### SM100

- **QSA: 14 failed cases.** Failures include backward replay/batch comparisons,
  sparse batch checks, the 513-token boundary, and backward mask cases. First
  mismatches are dK/dV or saved FP32 output; both 512-token controls pass.
- **DSA sparse attention: 5 failed cases.** All fail backward batch invariance
  in dKV. Forward and dQ match before that failure; later dSink comparisons are
  not reached. All 24 fprop cases and same-shape replay pass.
- **DSA BF16 indexer: 40 passed, 6 failed, 0 skipped.** The first mismatch in each failed case is dK.
  Sparse large-Top-K cases pass replay and reference, then fail after batch
  reorder. Dense backward also has replay and neighbor differences. Sparse dK
  reduction assigns query rows to fixed producers; changing batch position can
  change the grouping even when replay is stable. Dense backward uses atomic
  dK accumulation. These facts explain why a replay pass alone cannot establish
  batch invariance. Exact mismatch counts may vary between runs.

| Indexer case | First mismatch |
| --- | --- |
| `TestDSAIndexerBackward::test_replay[dk-fp32-dense-h64]` | replay 1 d_index_k: 2166/262144 bytes differ |
| `TestDSAIndexerBackward::test_batch_invariance[dk-bf16-dense-h64]` | batch (0, 1), sequence 0 d_index_k: 1/65536 bytes differ |
| `TestDSAIndexerBackward::test_batch_invariance[dk-fp32-dense-h64]` | batch (0, 1), sequence 0 d_index_k: 793/131072 bytes differ |
| `TestDSAIndexerBackward::test_sparse_persistent_rows[topk896]` | batch (1, 0), sequence 0 d_index_k: 132237/1048576 bytes differ |
| `TestDSAIndexerBackward::test_sparse_persistent_rows[topk2048]` | batch (1, 0), sequence 0 d_index_k: 202412/1048576 bytes differ |
| `TestDSAIndexerBackward::test_neighbor_isolation[dense]` | changed neighbor d_index_k: 1095/131072 bytes differ |

The backward golden cases use one query. Their passes do not establish
bitwise determinism for the multi-query reductions that fail above.

### SM103

SM103 had **24 failures**: 13 QSA, five DSA sparse-attention backward, and
six BF16 indexer backward cases. All 17 MXFP8 indexer cases passed.
The same five DSA attention and two sparse-indexer reorder cases failed on
SM100 and SM103. Dense indexer first mismatches were all in dK.

Two cases that failed in the SM100 run passed in this SM103 run:
`TestQSA::test_replay[bprop-q6-kv1]` and
`TestDSAIndexerBackward::test_batch_invariance[dk-bf16-dense-h64]`.
SM103 instead failed `TestDSAIndexerBackward::test_replay[dk-bf16-dense-h64]`
at replay 5, with one of 131072 dK bytes different. These are observed run
outcomes, not proof that the underlying nondeterminism is fixed.

The [coverage report](https://github.com/Anerudhan/cudnn-frontend/blob/feat/attention-determinism-tests/test/python/determinism_goldens/README.md)
lists all 24 SM103 failed case IDs and their first mismatch.

### SM107

SM107 had **24 failures**: 14 QSA, four DSA sparse-attention backward, and
six BF16 indexer backward cases. All 17 MXFP8 indexer cases passed. The DSA
attention failures are the query-boundary cases at H64 Q129/Q257 and H128
Q74/Q75; each first fails in dKV. The H128 basic batch case passed in this run.
Indexer failures use the same case IDs as SM103, with dK as the first mismatch.
The coverage report lists all 24 failed case IDs and byte counts.

## Runtime and reproduction

- GPU: SM100, 148 SMs; device name `NVIDIA Graphics Device`.
- Torch `2.15.0.dev20260820+cu134`, CUDA `13.4`, cuDNN `92500`,
  CuTe DSL `4.8.0.dev0`, TVM FFI `0.1.12`.
- Kernel base: `da88207aa5b9197b0c84514a7321d5bc3d2683b4`. This PR changes
  tests and documentation; kernel code is unchanged.
- Golden runtime key: `5f8a825586430e85`. Recording was disabled for validation.
- Imports resolved to the checkout under test. Validation combines three disjoint
  runs: the four linear-attention suites plus QSA (137 passed, 14 failed),
  DSA/BF16 indexer plus six controls (89 passed, 11 failed), and the new FP8
  module (17 passed, 17 skipped in 20.17 s). The first two runs are retained
  from the prior commit; their code is unchanged. Node IDs match default L0
  collection: 285 cases in 0.29 s.

SM90 validation used 124 SMs, Torch `2.14.0a0+b2c75dd062.nv26.09`, CUDA
`13.4` (driver API `13040`), cuDNN `92700`, CuTe DSL `4.8.0`, and TVM FFI
`0.1.14.post1`. Golden runtime key: `ffe6ff35e0022211`. A source snapshot
of commit `35cf3082e5f3e505b066097f29bf232a848ea269` was built in an isolated
container; imports resolved to that snapshot. Its test and kernel source are
unchanged by this report. The 285-case L0 run took 13.71 s with golden
recording disabled and the reviewed Hopper record selected.

SM103 validation used `NVIDIA B300 SXM6 AC`, 148 SMs, and the same Torch,
CUDA, cuDNN, CuTe DSL, and TVM FFI versions listed for SM90 above. Runtime key:
`7a727a3958771fbc`. Source snapshot: `8671c79c2a1e7073bf1efe0cce1939f093a0d840`.
The 285-case L0 run took 482.93 s with recording disabled. Its 16 new golden
records passed replay and independent references before comparison; existing
records were not overwritten. Linear-attention gate input hashes differ from
the SM100 runtime, so no cross-runtime byte equality is claimed.

SM107 validation used `NVIDIA VR NVL72`, 212 SMs, on an ARM64 host. Torch,
CUDA, cuDNN, CuTe DSL, and TVM FFI versions match the SM90/SM103 runs; the CUDA
driver API is `13050`. Runtime key: `72e27159dc117014`. Source snapshot:
`35cf3082e5f3e505b066097f29bf232a848ea269`; imports resolved to that snapshot.
The 285-case L0 run took 399.55 s with recording disabled. All 16 new golden
records passed replay and independent references. Some CPU-generated inputs
differ from the x86 runs, so cross-device byte equality is not inferred.
The test and kernel source are unchanged across these source snapshots.

Run from `test/python`, with imports resolving to this checkout:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4
export CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens"
python -m pytest core/test_determinism_utils.py \
  linear_attention/frost/test_*_determinism.py \
  QSA/cutedsl/test_qsa_determinism.py \
  deepseek_sparse_attention/cutedsl/test_DSA*determinism.py -q
```

Add `--collect-only` to list the 285 L0 cases. `CUDNN_DETERMINISM_GOLDENS`
makes missing runtime baselines fail instead of skipping the golden check.

## Scope limits

SM100, SM103, SM90, and SM107 execution are measured. The FP8 suite runs on
SM90, and the MXFP8 suite admits SM100/SM103/SM107; earlier suites gate on
SM100/SM103/SM107. Unsupported backward FP8 APIs have no positive execution
cases. FP16 indexer inputs, MXFP4 indexer paths, fused/separate Top-K generation,
paged indexer forward, THD dense indexer backward, score recompute, pooled DSA,
other linear-attention backends, and distributed recipes remain outside this
set. The forward indexer tests check scores, not Top-K generation or its derivative.
