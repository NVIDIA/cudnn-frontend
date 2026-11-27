# Attention determinism

For API support by GPU and datatype, see the
[determinism status page](../../../docs/operations/determinism.md).
This report holds test coverage, measured results, and failure diagnostics.

### Quantized batch-invariant matmul (2026-11-24)

The [batch-invariant suite](../gemm/batch_invariant/README.md) passed on SM100 (P2021, 68 SMs) and SM107
(208 SMs) with cuDNN 9.29.0 and CuTe DSL 4.8.1. On both devices the forward
comparison covered per-tensor E4M3/E5M2, groupwise E4M3, MXFP8, NVFP4, MXFP4
and MXFP8 × MXFP4, with BF16 and FP16 outputs. A single FlashInfer rollout row
and reordered or repeated rows matched direct training outputs and the original
FlashInfer kernels. The tests exercised N=128 with K=512 and K=8192 (including
split-K), the two-block MMA path at M=257 and N=264, and native cuDNN matmuls for
dX and dW of the effective floating-point inputs. The tests fix the already
quantized payload and logical scales before changing the batch.

At M=1024, N=4096, K=8192, six refilled-pipeline cases per GPU completed 3,000
warm and 3,000 shared-memory-poisoned launches each. Each result matched an
independently decoded FP64 reference rounded to BF16. The unsafe control
reproduced on both devices. Fourteen small quantized forward cases per GPU
also passed strict [golden comparison](../gemm/batch_invariant/goldens/). SM103 hardware execution
remains unverified.

### BF16 batch-invariant matmul (2026-11-23)

On 2026-11-24, all 27 cases in the [batch-invariant test suite](../gemm/batch_invariant/README.md) passed on
SM100, including 6 plan-selection cases and 14 FlashInfer integration cases.
On SM100 (68 SMs), the [batch-invariant matmul tests](../gemm/batch_invariant/test_matmul_determinism.py) compared output
bytes across repeated launches, row counts, row permutations and duplicates,
and compact B layouts. It covered K16 tail padding and fixed split-K with a
column bias. The two [batch-invariant baseline files](../gemm/batch_invariant/goldens/aa6120c0182e69f6) also match an
independent FP64 reference rounded to BF16 on inputs with exact dyadic values.
The backward compatibility case uses the batch-invariant forward output to form `dY`. On
SM100, native cuDNN matmul plans computed `dX` and `dW` (`eng0_k24=0`) and
a native ADD reduction computed `dBias` (`eng3_k2=0`). All three gradients
passed their FP64 reference checks, including with FROST engines enabled.

Run from `test/python` with the SM100 device selected:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDNN_DETERMINISM_GOLDENS="$PWD/gemm/batch_invariant/goldens" \
    python -m pytest -n0 -m L0 -q gemm/batch_invariant/test_matmul_determinism.py
```

The seven cases passed on the batch-invariant changes based on `2b98a25d`, with cuDNN
9.29.0, CuTe DSL 4.8.1, PyTorch 2.15.0+cu132 and TVM-FFI 0.1.15.post2.
The 2048 × 2048 × 2048 pipeline case ran in 256 blocks with four blocks per
cluster and three refills per stage. All 3,000 warm launches and 3,000
shared-memory-poisoned launches matched the independent reference bitwise.
The unsafe control produced a mismatch in all 200 launches. SM103 and SM107
remain unverified.

### MoeEp.training_wgrad: replay, expert invariance and warm stress (2026-11-23)

`MoeEp.training_wgrad` runs the vendored Rubin WGrad kernel. Each output tile is reduced by one CTA over its expert's
whole padded segment, with no split-K and no atomics. Its L0 determinism cells are in
`test/python/moe_ep/test_moe_ep_wgrad.py`. Synthetic operands are placed in the producer layout, and every check
compares raw bytes.

Two shapes are used:

| Shape | Experts | dW1 / dW2 per expert | Token rows per expert | What it reaches |
|---|---|---|---|---|
| Small | 4 | 512 × 512 / 256 × 512 | 300, 0, 128, 129 | Partial and empty segments |
| Multi-wave | 8 | 2048 × 2048 / 1024 × 2048 | 2048, 1500, 1024, 0, 1, 1793, 640, 2047 | Up to 16 K tiles of 128 rows. Each GEMM launches more 2-CTA clusters than one wave holds (512 thread blocks on 200 SMs) |

The cells:

| Cell | Shape | What it checks |
|---|---|---|
| `test_training_wgrad_replays_bitwise` (BF16, FP32) | small | Eight replays with fresh outputs under concurrent work give identical bytes. A separate tolerance check against an independent PyTorch MXFP8 reference |
| `test_training_wgrad_replays_bitwise_across_waves` (BF16, FP32) | multi-wave | The same, plus that the operands are unchanged after the calls |
| `test_training_wgrad_expert_results_do_not_depend_on_neighbors` | small | An expert's bytes are unchanged with new neighbors, after reordering, and when duplicated |
| `test_training_wgrad_expert_results_do_not_depend_on_neighbors_across_waves` (BF16) | multi-wave | The same: new neighbors that move the expert's segment, reversed order, and duplication into every slot |
| `test_training_wgrad_discrete_and_accumulate_replay_bitwise` | small | Replay for discrete destinations (BF16, overwrite), dense FP32 with `accumulate=True`, and discrete FP32 with `accumulate=True`. Every replay starts from the same destination value |
| `test_training_wgrad_layouts_store_the_same_accumulator` | small | Every FP32 layout and storage form stores identical bytes |
| `test_training_wgrad_warm_stress` | multi-wave, dense BF16 | `warm_stress.assert_warm_stress` against FE grouped WGrad (`cudnn.grouped_gemm_wgrad_wrapper_sm100`, 128 × 128 tiles), an independent kernel with byte-identical results. The A/B pipeline stages refill at least 1296 times per launch, with 2.56 thread blocks per SM |

Results:

| Item | Result |
|---|---|
| GPU | NVIDIA VR NVL72, cc 10.7, 200 SMs, driver 610.47.04 |
| Container | `nvcr.io/nvidia/pytorch:26.05-py3`: PyTorch 2.12.0a0 (nv26.05), CUDA 13.2 |
| CuTe DSL | public `nvidia-cutlass-dsl` 4.8.0 and 4.8.1 |
| pytest | **44 passed**, 0 skipped, on each DSL. This is all of `test_moe_ep_wgrad.py` at L0 and L1 |
| warm stress | On each DSL: the cold launches agree and equal the independent reference; warm 0 of 16384 launches wrong, poisoned 0 of 16384; per-launch rate below 0.0183 % (95 %) |
| Positive control | The unsafe ring twin failed 200 of 200 launches (600 thread blocks, cluster 2), so the stress is validated on this GPU |

No golden files are recorded. On SM90, SM100/103 and SM120 the WGrad path is declined by the capability check.

Command, run from the cudnn-frontend checkout's `test/python`:

```bash
pytest moe_ep/test_moe_ep_wgrad.py -m "L0 or L1" -q -rs
```

### MoeEp forward and backward: no determinism suite (2026-11-23)

`MoeEp.__call__`, `MoeEp.training_forward` and `MoeEp.training_backward`
(the Rubin SM107 MXFP8 MegaMoE kernels behind `cudnn.moe_ep`) have no
determinism suite. The status page lists them as unverified on SM107. The gap
covers these kernel paths:

- Forward: `scheduling="static"` (the default), `"atomic"` and
  `"phase_interleave"`; `fc2_bulk_store` off, `"auto"` and explicit stage
  counts; `mixed_cga`. Each combines with the `epi_warps`, `standalone_warps`
  and `reuse_dispatch_warps` token-back modes.
- Backward: the `dgrad_optimization` profiles `baseline`, `rolling` (M16
  traversal, scaled FP8 conversion) and `ds3_ep4_pattern` (rank input dedup,
  asynchronous token-in flags).

Known order-dependent operations: training backward adds per-tile dprob
partials with FP32 atomics, and inference does not fix the order of token
communication.

The functionality tests do some raw-byte checks, but none of them is a
determinism cell. `test_moe_ep_forward.py::test_forward_kernel_tuning_inference_matches_reference_and_replays`
and `test_moe_ep_backward.py::test_training_forward_kernel_tuning_does_not_accumulate_across_calls`
check each tuned forward output against an independent reference, and against
the default tuning's output bytes over repeated calls. These checks do not
cover replay under concurrent work, batch invariance, warm stress or goldens,
and no golden files are recorded.

### GH200 block-sparse attention and causal convolution goldens (2026-11-19)

The SM90 blk64 block-sparse attention suite and the FP16 causal convolution
decode suite had no goldens for GH200, so their four and six `test_golden`
cases skipped there. This adds them; no kernel or test code changes.

Runtime: NVIDIA GH200 480GB (SM90, 132 SMs, aarch64 host), PyTorch
2.14.0+cu130, CUDA 13.0, driver API 13020, cuDNN 92700, CuTe DSL 4.8.1,
tvm-ffi 0.1.15.post2, at commit `86b74686`. The goldens are in
`c079e9b62638eb64/`.

```
F="block_sparse_attention/cutedsl/test_BSA_determinism.py causal_conv1d/cutedsl/test_causal_conv1d_update_determinism.py"
CUDNN_DETERMINISM_RECORD=<new directory> pytest -m L0 $F   # 42 passed
CUDNN_DETERMINISM_GOLDENS=<new directory> pytest -m L0 $F  # 42 passed
pytest -m L0 $F                                           # twice, files copied in: 42 passed, 0 skipped
```

Run from `test/python`. The 42 cases are 30 block-sparse attention and 12
causal convolution. Each `test_golden` case checks its problem against the
FP32 reference and replays it bitwise before the hashes are recorded.

The six causal convolution records equal the GeForce RTX 4090 records in
`c74756f44f441362/` byte for byte, inputs and outputs. The block-sparse
attention records do not compare that way with the H100 and B300 records in
`f2590ab8d14a5923/` and `d6ab5c84009f8fa5/`. Their integer metadata hashes
match, but the BF16 and FP16 q, k, v and dO drawn from the suite's CPU
generator do not. On this aarch64 host, PyTorch 2.12.0 through 2.16.0.dev give
the same operand bytes, so the split follows the host CPU's `randn` path, not
the PyTorch version. Outputs from these four files are only comparable with
records drawn on an aarch64 host.

### SM100 D256 swap-AB in-launch reduction goldens for a 152-SM part (2026-11-20)

The two golden cells of `TestClusterReduce`
(`test_replay_capture_and_golden[bf16-packed-split8-kwargs0]` and
`[f16-unpacked-split4-kwargs1]`) had no golden for a GB200: with
`CUDNN_DETERMINISM_GOLDENS` set they failed there at the comparison, their
last statement (the
[record below](#sm100-d256-swap-ab-in-launch-reduction-on-a-152-sm-part-2026-11-19)).
This adds the two files; no kernel or test code changes.

Runtime: NVIDIA GB200 (compute capability 10.0, 152 SMs, aarch64 host),
PyTorch 2.14.0+cu130, CUDA 13.0, driver API 13020, cuDNN 92700, CuTe DSL
4.8.0, tvm-ffi 0.1.14.post0, at commit `edbcff2db`, the commit in front of the
one that adds the files. The goldens are in `52ecb6a4cddc4ab8/`. The key
carries the cuDNN version and not its build: the libraries were the
pre-release debug build 9.27.0.21, and a released 9.27 reads 92700 as well.

```
S=sdpa/frost/test_sdpa_fwd_decode_d256_sm100.py
G="TestClusterReduce and replay_capture_and_golden"
CUDNN_DETERMINISM_RECORD=<new directory> pytest $S -k "$G"            # 2 passed
CUDNN_DETERMINISM_GOLDENS=<new directory> pytest $S -k "$G"           # second process: 2 passed
CUDNN_DETERMINISM_GOLDENS=<new directory> pytest $S -k TestClusterReduce  # 68 passed, 0 skipped
```

Run from `test/python`. Before its hashes are recorded each cell checks both
forms against the float64 reference, replays each form eight times with fresh
buffers and concurrent GPU work, compares the two forms byte for byte and
compares a retained CUDA graph of the in-launch form with eager execution.
Both forms reproduce the one record of their path.

The two files equal, byte for byte, the candidates recorded the day before on
another device of the class at an earlier head of
[PR #2780](https://github.com/NVIDIA/cudnn-frontend/pull/2780), before the
152-SM rows existed; no rendering of this engine's kernels differs between the
two heads.
Against the records of the same two cases for the 148-SM part (x86-64 host) in
`c84dd9abaee769b4/` the shapes and dtypes are equal and all ten hashes differ,
those of the inputs included: the aarch64 host draws other operand bytes, as
in the GH200 record above.

#### The suite on that part at the same commit

The same device, runtime and commit, before the two files were added:

| Selection | Result |
| --- | --- |
| `test_sdpa_fwd_decode_cluster_reduce.py` (no device needed; 131 cells) | **131 passed** |
| `TestClusterReduce` with `CUDNN_DETERMINISM_GOLDENS` set to this directory (68 cells) | **66 passed, 2 failed**: the two golden cells, at the comparison |
| `TestClusterReduce` against the newly recorded directory | **68 passed** |
| `test_two_chunks_follow_the_split_order` in 30 fresh processes | **30 of 30 passed** |
| `test_sdpa_fwd_decode_d256_sm100.py` without the class | **74 passed** |
| `test_sdpa_fwd_split_kv_sm100.py` and `test_sdpa_split_combine_sm100.py` | **335 passed** |
| `test_sdpa_fwd_paged_sm100.py` | **351 passed** |

No cell skipped in any of them. In each of the 30 processes the two-chunk cell
printed `393216 bfloat16 O elements compared and equal in split order; the
reverse order moves 6 of them`; its seed is fixed, so the 30 processes repeat
one draw. The nine default-plan cells passed again.

```text
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" python -m pytest $S -q -rs -k TestClusterReduce
python -m pytest $S -q -rs -s -k "TestClusterReduce and two_chunks"   # once per process
python -m pytest $S -q -rs -k "not TestClusterReduce"
python -m pytest sdpa/frost/test_sdpa_fwd_decode_cluster_reduce.py -q -rs
# the last two with the compiled-plan cache on
python -m pytest sdpa/frost/test_sdpa_fwd_split_kv_sm100.py sdpa/frost/test_sdpa_split_combine_sm100.py -q -rs
python -m pytest sdpa/frost/test_sdpa_fwd_paged_sm100.py -q -rs
```

A default graph at the row of 2 chunks was read outside the suite, with 12,288
declared keys and one and two tokens. With 24 query heads on 4 KV heads and 19
sequences, and with 8 on 1 and 76, the first plan is the swap-AB tile split 2
ways with `CLUSTER_K_REDUCE`: 76 clusters, and one kernel per execution. With
8 on 1 and 77 sequences the plan carries no knob and an execution launches two
kernels. All six graphs matched the float64 reference. Through explicit
records at that split, 64, 72, 76 and 77 clusters with 4,096, 8,192 and 12,288
declared keys stored equal O and Stats bytes in both forms (98,304 to 233,472
BF16 O elements per configuration).

### SM100 D256 swap-AB in-launch reduction on a 152-SM part (2026-11-19)

The suite of the [record below](#sm100-d256-swap-ab-in-launch-reduction-2026-11-18)
ran on one GB200 (compute capability 10.0, 152 SMs; aarch64 host; CuTe DSL
4.8.0, cuDNN 9.27 debug libraries, PyTorch 2.14.0+cu130), exclusive to the
run, after `heuristics.FUSED_CLUSTER_BOUND` gained the rows of that part (76
clusters of 2 chunks, 36 of 4, 15 of 8). The
[record above](#sm100-d256-swap-ab-in-launch-reduction-goldens-for-a-152-sm-part-2026-11-20)
repeats it at a later commit: there the class reads 66 passed and 2 failed
without the part's golden files and 68 passed with them, and the policy file
131 passed.

The graph-plan group has nine cells from here on. Each builds the default
plan of a two-token step with no knob record, reads from the compiled plan's
record which form ran -- the in-launch form where the plan's clusters are
within the part's row for its split, the two-launch form past the row or on a
part without one -- and rebuilds both forms from public records with the bytes
of the default plan. On this part the default plan ran in the launch at 4 and
8 clusters of 8 chunks, 4 and 36 clusters of 4, 40 clusters of 2, and 15
clusters of 8 with 8/1 heads; it ran two launches at 16 clusters of 8 (24/4
and 8/1 heads) and at 37 clusters of 4.

| Selection | Result of this run |
| --- | --- |
| `TestClusterReduce` with `CUDNN_DETERMINISM_GOLDENS` set (68 cells) | **65 passed, 3 failed** |
| `test_sdpa_fwd_decode_d256_sm100.py` without the class | **74 passed** |
| `test_sdpa_fwd_decode_cluster_reduce.py` (no device needed) | **116 passed** |

The three failures are the same three cells that failed on this device before
the rows were added:

- Both golden cells: this runtime had no recorded golden for
  `sdpa-decode-d256-swap-ab-bf16-packed-split8` and
  `sdpa-decode-d256-swap-ab-f16-unpacked-split4`. Everything in front of the
  golden comparison passed in both cells (eight replays per form, the two
  forms byte for byte, the retained CUDA graph, changed inputs). Without the
  checking mode the comparison was skipped: a coverage gap for this runtime,
  which the files of the record above close.
- `test_two_chunks_commute`: with the inputs this host draws, the separate
  pass reducing two chunks in the reverse order differs from the forward order
  in a few BF16 elements, by one to three units of the last place. The same
  difference appears on the
  two-launch plan's own slab, so it is a property of that pass's two-chunk
  arithmetic and of the draw, not of the in-launch form; the two forms are
  byte-equal for that configuration. The cell's premise does not hold for
  every input; the cell is replaced, see below.

The element-budget, stale-partial, refilled-workspace and batch-invariance
cells passed, as did the replay and retained-graph steps in front of the two
golden comparisons, so the statements of the status page hold on this part.
The warm stress passed behind its control (the unsafe ring failed 200 of 200
launches at cluster size two); the stressed launch had 72 thread blocks on
152 SMs, below the control's density, as on the 148-SM part.

```text
suite=sdpa/frost/test_sdpa_fwd_decode_d256_sm100.py
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" python -m pytest "$suite" -q -rs -k TestClusterReduce
python -m pytest "$suite" -q -rs -k "not TestClusterReduce"
python -m pytest sdpa/frost/test_sdpa_fwd_decode_cluster_reduce.py -q -rs
```

The policy file had 116 cells at this run. It has 131 now: five more
rule-table cells at and past the row of 2 chunks, and ten cells that pin, for
the nine default-plan steps of the device cell, the form the rules give on a
148-SM and on a 152-SM part. They need no device; all 131 passed on the 152-SM
part in the run of the record above.

#### The two-chunk cell after this run

`test_two_chunks_commute` required the separate pass, run over the in-launch
slab in the reverse split order, to reproduce the in-launch O bytes. That pass
accumulates `acc + w * v` from zero as a fused multiply-add: the first chunk's
product is rounded before the second is added, the second is not. Two chunks
in the other order therefore give the same Stats (their denominator is one
FP32 addition of the two weights) and, now and then, another stored O element.

The comparison was repeated on the 148-SM part (x86-64 host) for 385 seeds of
the cell's runner (4 sequences, 49,152 BF16 O elements per seed) under both
paths of the host's `torch.randn`: the default vectorised one and the scalar
one (`ATEN_CPU_CAPABILITY=default`). The scalar path draws the bytes of the
aarch64 host (Q and K of the twelve seeds compared are equal), and under it
the 148-SM part reproduced the differing elements of the 152-SM run for those
twelve seeds: the same count per seed, and the same position and bits in the
13 of 14 elements compared, seed 7 included.

| `randn` path | Seeds with a differing O element | Differing elements | Largest difference |
| --- | --- | --- | --- |
| vectorised (x86-64 default) | 244 of 385 | 369 of 18,923,520 | 15 units of the last place |
| scalar (the aarch64 host's bytes) | 224 of 385 | 321 of 18,923,520 | 15 units of the last place |

Seed 7, the cell's seed, has no such element on the vectorised path and one
on the scalar path, which is why the cell passed on the x86-64 host and failed
on the aarch64 one. In all 770 runs the two forms stored equal O and Stats
bytes, their partial slabs were equal, the separate pass in split order
reproduced the in-launch O and Stats, and the reverse order reproduced the
Stats.

The cell is now `test_two_chunks_follow_the_split_order`. It compares both
forms and the separate pass in split order over 32 full-cache sequences
(393,216 BF16 O elements), requires the Stats of the reverse order byte for
byte, and prints how many O elements the reverse order moves; it makes no
assertion on those. On the 148-SM part it passed under both `randn` paths, and
the reverse order moved 6 elements under each. On the 152-SM part (aarch64
host) it passed in the run of the record above, where the reverse order moved
6 elements as well. The row "Two chunks commute" of the table in the record
below describes the cell it replaces.

### SM100 D256 swap-AB in-launch reduction (2026-11-18)

`sdpa/frost/sdpa_decode_cluster_reduce_sm100.py` (collected through
`test_sdpa_fwd_decode_d256_sm100.py::TestClusterReduce`) covers the SM100 D256
swap-AB split decode tile with its partial results reduced by the separate
pass and, with `CLUSTER_K_REDUCE=1`, inside the launch. It is forward-only and
uses the standalone adapter; four cells go through `pygraph.sdpa` and the
public knob record.

Every cell runs both forms of one split count on the same inputs. Each form is
compared with a float64 reference under the declaration's mask and lengths
(the tolerances of the existing decode suite), and the two forms are then
compared byte for byte: O, FP32 Stats, the FP32 partial slabs, and the output
of the separate reduction entry run over the slab the in-launch form left.
Final outputs sit in NaN-filled storage one token row larger than declared;
the workspace is filled with a large finite value, so a partial that no chunk
wrote would dominate the reduction and fail the reference.

| Group | Cells | What varies |
| --- | --- | --- |
| Numerics and two-form bytes | 36 | every split count from 2 to 8; head ratios 24/4, 8/1, 16/4, 32/8 and 32/2 (6, 8, 12 and 16 packed rows in one tile); head widths 256, 224 and 192; packed 1 to 4 tokens, unpacked 2, 3, 4, 6 and 8 tokens; FP16/BF16; pages 16, 32, 64 with per-batch lengths and a dense cache without length operands; padding, bottom-right causal, top-left causal and sliding-window masks; a paged plan without the Q length operand; one user with a full cache and with 128 and 1,024 live keys under a 2,048-key declaration; lengths 0, 1, 64, 127, 128, 129, 1,000 and full; more chunks than declared tiles; 1 to 5 tiles per chunk; padded query rows; output storage that is not 8-byte addressable; no Stats output |
| Element budget | 3 | 64 users x 2 packed tokens at 4, 8 and 8 chunks (2,359,296 BF16 O elements), 24 users x 6 unpacked tokens at 4 and 8 chunks (1,769,472 BF16) and 24 users x 2 packed tokens at 8 chunks (294,912 FP16), full caches; a reduction in the reverse split order must not reproduce the bytes |
| Two chunks commute | 1 | the reverse-order control reproduces the bytes at 2 chunks, so those cells carry no weight in the budget |
| Stale partials | 2 | a full-length launch, then one with live lengths 1 / 100 / 0 on the same workspace without refilling it; both forms |
| Partials of this launch | 1 | 384 in-launch launches with uneven chunks, the workspace refilled with the finite value before each one: a reduction that read a partial before its chunk stored it would move the bytes |
| Batch invariance | 4 | one sequence alone, among 8, among 64, reordered, duplicated and with other neighbours, at a fixed split; packed 2 tokens x 8 chunks and unpacked 6 tokens x 4 chunks; both forms |
| Replay, capture and golden | 2 | eight fresh-buffer replays with concurrent GPU work per form; a retained CUDA graph against eager execution; changed lengths and queries under the retained graph; one golden per path that both forms reproduce |
| Warm refill stress | 1 | clusters of two, ten key tiles per chunk, 256 warm and 256 poisoned launches behind the cluster-of-two control |
| Graph plan and public record | 4 | the default plan of a two-token step carries the knob and its record; the compiled plan's record names the form that ran; a record with and without the knob rebuilds the same bytes |
| Declines | 9 | explicit records and standalone requests outside the form (a head width of 200 among them); an unaligned workspace |

A right-band mask and the NHD pool layout are admitted by the form and have no
cell here.

Measured on one B200-class development GPU (compute capability 10.0, 148 SMs;
CuTe DSL 4.8.0, cuDNN 9.27 debug libraries, PyTorch 2.13.0+cu130), shared with
other users: **63 passed** with `CUDNN_DETERMINISM_GOLDENS` set, so the two
golden comparisons ran. The element-budget cells printed
`2359296 bfloat16 O elements compared and equal`,
`1769472 bfloat16 O elements compared and equal` and
`294912 float16 O elements compared and equal`.

The warm stress passed behind its control (the unsafe ring failed 200 of 200
launches at cluster size two). This kernel holds one thread block per SM, so
the stressed launch had 72 thread blocks on 148 SMs; the report marks such a
shape as below the control's three thread blocks per SM, and a zero there
bounds the failure rate weakly. It is finite evidence for the tested shape.

The two golden cases (`sdpa-decode-d256-swap-ab-bf16-packed-split8`,
`sdpa-decode-d256-swap-ab-f16-unpacked-split4`) are recorded for that runtime;
the records of a 152-SM part were added on 2026-11-20 (first record of this
suite above). SM103 was not run: the knob is declined there, and the status
page says unverified. `test_sdpa_fwd_decode_cluster_reduce.py` holds the cells that need
no device (76 passed): the knob record, the heuristics rule and its tables,
the recorded cluster bound, the declines, the source pin of the cluster
barrier and the launch-frame parity of the native and Python binders.

```text
suite=sdpa/frost/test_sdpa_fwd_decode_d256_sm100.py -k TestClusterReduce
python -m pytest "$suite" -m L0 -q -rs
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" python -m pytest "$suite" -m L0 -q -rs -k replay
```

### SM107 D256 decode register pressure (2026-11-17)

The host compile check failed after
[PR #2675](https://github.com/NVIDIA/cudnn-frontend/pull/2675), commit
`f2810d269`. The added Int64 output slicing fixes wide-stride overflow but
raises register pressure in the 64-column tile. With the same CuTe DSL 4.8.0
compiler, the parent body has no spills; the changed body has 5 STL, 5 LDL,
and 16 stack bytes. Four-column output groups restore zero spills while
retaining the wide coordinates. Other tile widths keep their group size.

An isolated Linux x86-64 container with cuDNN 9.25.1.1, CUDA 13.4.0,
CuTe DSL 4.8.0, and CUDA `nvdisasm` compiled all six
spill-probe forms in BF16 and FP16: **12 passed in 95.74 seconds**, zero skips,
zero STL/LDL/stack bytes, 94–96 registers. This used source `c1a1caf2a` plus
the fix. The isolated pytest harness retained the test's assertions and
subprocess probe, loading the production kernel through `load_template`
instead of the adapter import. No GPU kernels were launched. The SM100/SM107
source-body comparison also passed with host-only imports.

The L0 `test_decode_kernel_four_column_groups` now has eight cases: two head
groups, split counts 1/2, and FP16/BF16. Each retains the independent reference
and adds raw-byte O/Stats comparisons with fresh allocations, concurrent work,
sequence removal/reordering/duplication, changed neighbors, and input-mutation
checks. These runtime cases, the retained wide-physical-stride tests, and Rubin
performance remain **unverified** because no SM107 GPU was available. Backward
uses other kernels. No golden files were recorded or replaced.

Compiler check command inside the isolated container:
`python -m pytest /work/test_sm107_spill_probe.py -q -s`.
Native qualification command from `test/python`:
`pytest sdpa/frost/test_sdpa_fwd_decode_d256_sm107.py -m L0 -k "four_column_groups or wide_tile_renderings"`.

### FP16 causal convolution decode update

`causal_conv1d/cutedsl/test_causal_conv1d_update_determinism.py` exercises the
public width-four inference API with identity and SiLU, compact three- and
four-element state, and channel-last three-element state. It checks the output
and in-place state against an independent FP32-compute reference and compares
their bytes over eight launches with fresh state and output allocations and
concurrent CUDA work. The stress cases use 33 rows and 1025 channels with padded
input strides, an indexed padding row, and state slots outside the live batch.
Batch checks replay one sequence alone, reorder the batch, replace neighbors,
and duplicate the sequence into a distinct state slot. The six small golden
cases use two rows and 17 channels, recording both outputs after the replay and
reference checks. The operation is inference-only and has no backward.

Replay, isolated batch checks, and all six goldens passed on GeForce RTX 4090
(SM89), with PyTorch 2.12.1+cu130 and CuTe DSL 4.8.0. The goldens are in
`c74756f44f441362/`.

### Conv3D post-operations

`conv/cutedsl/test_conv3d_postops_determinism.py` adds 27 L0 cases:

| Checks | Cases | Coverage |
| --- | ---: | --- |
| Replay and batch invariance | 12 | One case per API family and dtype; N96/N160/N192/N256 tiles, C320 normalization and C1024 standalone reduction distributed across cases |
| Small runtime-specific goldens | 12 | One case per API family and dtype, after replay and an independent numerical reference |
| Persistent warm stress | 3 | Raw C192 TF32, prepared C192 BF16, and the separate BF16 C12 causal body |

Replay uses fresh output allocations and concurrent GPU work. Batch checks keep
inputs fixed while splitting, reordering, duplicating, inserting/removing
sequences, and changing neighbors. T1/H7/W9 exercises partial M tiles and
history/cache boundaries; the golden geometry uses T3/H5/W7. Stress uses
T3/H97/W121 to reuse persistent workers and refill the shared-memory ring,
with the shared helper's validated control and warm/poisoned launches.
Output, padded output, cache, residual, and exposed stress workspaces are
compared as raw bytes; inputs are checked for mutation. Caller-owned activated
history is populated before comparing full fused outputs. Independent Torch
references check numerical accuracy. These are forward-only APIs with batch
size N >= 1.

Measured on **2026-11-17, NVIDIA B200 (SM100, 148 SMs)** at source revision
`3916c70f` plus this test/doc addition: Torch 2.13.0+cu132, CUDA driver 13040,
cuDNN backend 9.20.0, CUTLASS DSL 4.9.0.dev1 (CUDA 13.4 compiler), and
TVM-FFI 0.1.12.
The representative selection passed **27/27 in 12.57 s**, including exact checks
against all 12 records under `e236f71ed93bdd1c/`. The Conv3D L0/L1 correctness
suite passed **92/92 in 12.69 s**, with one Torch synchronization-debug warning.
Both runs had zero failures or skips. Correctness cases sample tile widths,
K padding, reduction layouts, and dtype-specific paths; they also exercise
dynamic geometry/history, graph replay, disk-cache reload, and stream lifetimes.

The original selection on this runtime recorded **45/45 in 83.52 s**, checked
**45/45 in 21.10 s** in a separate process, and passed correctness **158/158 in
152.29 s**. That correctness run reported seven CUTLASS static-loop compile
warnings and one Torch synchronization-debug warning. The golden inputs and
records are unchanged; each passed an independent numerical reference before
recording.

Each of the three stress cases completed 3,000 warm and 3,000 poisoned launches
with zero byte mismatches. The one-CTA and two-CTA positive
controls each detected errors in 200/200 launches. The production persistent
launch uses 148 thread blocks (one per SM); the controls use three blocks
per SM. The prepared BF16 case refills 74 times and raw TF32 refills 156 times,
and the causal BF16 body refills twice. Stress compares against cold-launch
bytes; the independent numerical reference is checked separately before stress.

Commands from `test/python` (record into a fresh directory, then check in a
separate process):

```bash
python -c 'import cudnn; print(cudnn.__file__)'
python -m pytest conv/cutedsl/test_conv3d_postops_determinism.py -m L0 --collect-only -q
export CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDNN_TEST_NO_ISOLATION=1 OMP_NUM_THREADS=8
CUDNN_DETERMINISM_RECORD=/tmp/conv3d-new-goldens python -m pytest \
  conv/cutedsl/test_conv3d_postops_determinism.py -m L0 -q -rs -x
CUDNN_DETERMINISM_GOLDENS=/tmp/conv3d-new-goldens python -m pytest \
  conv/cutedsl/test_conv3d_postops_determinism.py -m L0 -q -rs -x
# After reviewing and adding the records, check the repository baseline:
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" python -m pytest \
  conv/cutedsl/test_conv3d_postops_determinism.py -m L0 -q -rs
python -m pytest conv/cutedsl/test_conv3d_postops.py -m 'L0 or L1' -q -rs -x
```

See the [golden workflow](#golden-checks) for recording and reviewing new baselines.

### D256 nonpaged split physical origins (2026-11-16)

`sdpa/frost/test_sdpa_fwd_d256_split_origins_determinism.py` exercises fixed
forward owners 20519 and 20570, FP16/BF16, token/head Stats and splits 2/5.
It retains the independent FP64 reference and changed-pointer/pitch/length
CUDA Graph checks, and adds byte comparisons with fresh allocations,
concurrent work, reordered/duplicated/removed sequences, changed neighbors,
and a separate B=1 graph carrying the same first sequence. This is forward
binding coverage; the backward algorithm and its status are unchanged.

On SM100, SM103 and SM107, actual Torch 2.11.0+cu130, cuDNN 9.24.0.43,
CuTe DSL 4.8 and CUDA 13.4 compiler qualification completed at source
`50535efb4` (production unchanged from `171bcd81a`). Imports and the matching
native binding were checked on each target. Each target passed 366 native
binder checks, two independent-origin reference/rebind/capture cases and 16
compact split controls without failures or skips.

| Target | SM count | New suite, recording | Separate-process strict check | Runtime fingerprint |
|---|---|---|---|---|
| SM100 | 148 | 11 passed, 0 skipped | 11 passed, 0 skipped | `b6cd858eb78e109b` |
| SM103 | 148 | 11 passed, 0 skipped | 11 passed, 0 skipped | `104472d5f9694663` |
| SM107 | 216 | 11 passed, 0 skipped | 11 passed, 0 skipped | `d42d57b738f5eead` |

The selected SM107 CI entry also exposes these same checks through
`test_sdpa_fp8_sm107.py::TestD256HalfSplitOrigins`, preserving their parameter
marks without duplicating the test bodies. After integrating the mixed-KV
owner fix, source `7849117b6` passed 11 recording checks and 11 independent
strict checks through this entry on the 216-SM runtime above. All eight golden
files were byte-identical to the existing records. Its integrated owner and
graph-analyzer checks passed all 701 cases on native SM107.

The new suite contains three host extension-gate checks and eight GPU cases.
Each GPU case completes the reference, replay and batch checks before recording.
Twenty-four new golden files were inspected and added; no existing golden was
replaced. The four token-Stats hashes per SM103/SM107 target are byte-identical
to the earlier successful run before the HN-only fixture correction.
These results cover B=1/4, splits 2/5, and the stated origin/stride variations;
they do not change the engines' public behavior notes.

Reproduce from `test/python`, using a fresh recording directory and then a
separate process for the strict comparison:

```bash
suite=sdpa/frost/test_sdpa_fwd_d256_split_origins_determinism.py
CUDNN_DETERMINISM_RECORD=/tmp/new-d256-origin-goldens pytest "$suite" -m L0 -q
CUDNN_DETERMINISM_GOLDENS=/tmp/new-d256-origin-goldens pytest "$suite" -m L0 -q
```

The pre-fix B300 detector produced NaN output for both dtypes when independent
physical offsets were not forwarded. The historical SM107 strict expected
failures stopped at the missing-metadata assertion; bypassing only that assertion
exposed the numerical error. The old native extension also fails all six new
nonpaged D256 frame cases while passing the six paged controls. Intermediate
new-fixture failures exposed omitted total-Q capacity and undersized B=1 HN
storage, both corrected in tests without another production change.

### Fixed SM107 quantized backward implementations (2026-11-15)

Per-tensor FP8 engine 20605 and MXFP8 engine 20606 retain their records,
domains and diagnostic numerical policies. Their bodies, fixed hosts,
contracts and configuration helpers now live in `sm107/d256_fp8` and
`sm107/d256_mxfp8`. The 59 moved device definitions have identical ASTs;
the shared parser, workspace and prepared binding chain are preserved.

On native SM107 with CUDA 13.4 compilation, CuTe DSL 4.8, cuDNN 9.30 and
Torch 2.14.1+cu130, frozen source `916d60187` produced the following L0
results. Every frozen source hash and the actual import path were checked.

| Suite under `test/python/sdpa/frost/` | Result |
|---|---|
| `test_sdpa_bwd_sm107_quant_implementations.py` | 8 passed |
| `test_sdpa_bwd_fp8_sm107.py` | 247 passed, 8 skipped, 1 stale classic-admission fixture failed |
| `test_sdpa_bwd_mxfp8_sm107.py` | 163 passed, 29 skipped |
| `test_sdpa_bwd_thd_fp8_sm107.py` | 152 passed |
| `test_sdpa_bwd_thd_mxfp8_sm107.py` | 148 passed |

The 37 skips were 30 unavailable nvdisasm checks, six unavailable historical
reference dumps and one superseded bring-up placeholder. The admission
failure was reproduced with the identical assertion on frozen parent
`42508c7e4` in the same environment. Disabling opt-in no longer removes
default Python engines, so that test did not reach the classic C++ path it
intended to exercise. Source `ae13f0299` explicitly withholds Python
candidates in that test; the native admission test and the existing eight
head/architecture boundary cases then all passed. The original C++ admission
assertion remains. This correction changes no production behavior or tolerance.

The eight new cases cover causal D256, S=257, MHA 4/4 and GQA 4/2 with
BF16 gradients. Both default and diagnostic dS policies are exercised for
each family. They check reference gradients, requested FP8 gradient/dP amax,
actual body/profile identity, exported-record rebuild, fresh port bindings,
input preservation and four poisoned CUDA Graph replays after changing dO.
All eight passed new golden recording and separate-process strict checking
under runtime key `22fb9766a4ab5fdc`; no existing golden was replaced.

The strict/control job's final summary command used an unavailable bare
`python` executable after all pytest processes had finished. Its preserved
XML confirmed eight strict passes and matching parent/candidate admission
failures; the summary was recovered without rerunning those GPU tests.

These results qualify the tested fixed-plan repeatability and rebinding
cases. General batch invariance, other GPU architectures, absent SASS checks
and cross-runtime byte identity remain unverified. Deterministic-mode
declarations are unchanged.

### Fixed D256 and D192 half backward implementations (2026-11-15)

`test_sdpa_bwd_d256_chunk_grid.py` covers the shared host's final GEMM
batch/head launch limit for half, FP8 and MXFP8 declarations. Its native
SM100 half cases check all three gradients against FP64, retained capture
with changed upstream gradients, THD rebatching that removes empty neighbors,
and dense batch chunks that preserve a whole GQA group. The byte checks do
not extend the general batch-invariance or cross-architecture guarantees below.

The SM107 native D256 and shared SM100/SM107 two-by-two bodies now have
explicit implementation directories and hosts. Engine 20632 fixes the native
SM107 body; opt-in 20633 fixes the dense Rubin interleaved profile. SM100
20607 and SM107 D192/V128 20610 keep their records and shared body. Legacy
20604 delegates and requires replanning. The 57 moved device definitions
retain identical ASTs; workspace and reduction order are unchanged.

On CUDA 13.4 compilation, CuTe DSL 4.8, cuDNN 9.30 and Torch 2.14.1+cu130,
frozen source `42508c7e4` passed the following `-m L0` selections. All source
hashes and the actual import path were checked before execution.

| Native GPU | Suites under `test/python/sdpa/frost/` | Result |
|---|---|---|
| SM107 | `test_sdpa_bwd_dsl_sm107.py`, `test_sdpa_bwd_thd_sm107.py`, `test_sdpa_bwd_d192_sm107.py`, `test_sdpa_bwd_sm107_d256_implementations.py`, `test_sdpa_bwd_sm107_d256_contract.py` | 490 passed, 88 skipped |
| SM100 B200 | `test_sdpa_bwd_d256_sm100.py`, `test_sdpa_bwd_d192_sm100.py`, `test_sdpa_bwd_thd_d256_sm100.py`, `test_sdpa_bwd_thd_d192_sm100.py`, `test_sdpa_bwd_config_d256_2x2.py` | 405 passed, 26 skipped |

The SM107 skips were 34 fingerprints pinned to DSL 4.7, six unavailable
historical reference dumps, 46 unavailable nvdisasm checks and two documented
graph-padding declines. The SM100 skips were 26 nvdisasm checks. The 12 new
native record tests exercised FP16/BF16, dense GQA and causal MHA/GQA, actual
body/profile identity, public-record rebuild despite legacy half switches,
fresh bindings and changed-input CUDA Graph replay.

`test_sdpa_bwd_d256_implementation_determinism.py` at source `d96a3583e`
adds 20 bounded cases: 12 on SM107 (20632, 20633 and D192/V128 20610), eight
on SM100 (D256/D256 and D192/V128 20607). Every case uses FP16 or BF16,
causal S=257 and MHA 4/4 or GQA 4/2. It checks all gradients against an
FP64 reference, finite results, unmodified inputs, fresh poisoned buffers,
eight executions with concurrent work, batch size one versus three, reordered
and duplicated sequences, and changed neighbor gradients. An initial test
incorrectly bound FP64 oracle Stats; the corrected test binds FP32 Stats
without changing production or tolerance.

All 20 cases passed through new golden recording, then all passed in a
separate process with recording disabled and strict checking enabled. The
new records use runtime keys `c8aefc970ef3af15` (SM100) and `22fb9766a4ab5fdc` (SM107);
no existing golden was replaced. Other-architecture parameter cases are
explicitly skipped in each lane. These are tested-configuration results:
general batch invariance, SM103 and the existing L1 stress cases are not
newly qualified. Existing graph deterministic-mode declarations are unchanged.

An additional CPU-only differential run selected L0 cases across every
`test/python/sdpa` file, excluding PTX/SASS/render/compile/cubin test names.
It used no CUDA device and `--noconftest` to inspect host consumers across
the tree. Candidate/parent passed 11,479/11,470 cases and had identical
14,779 call/setup failure IDs and 16 collection-error IDs; no candidate-only
failure was introduced. The 65 differing traces contained the same
CUDA-unavailable error messages, with changed source lines or relocated
adapter paths. The additional nine passes are the new fixed-owner host
contracts; 32 additional native cases were skipped without a GPU. This
differential audit is not a green full-suite runtime result.

### Fixed SM107 D512 backward implementation (2026-11-15)

Engine 20608 retains its public record, the D264–D512 aligned envelope,
workspace and reduction order. Its body, host, contract and fixed ring
configuration now live together in `sm107/d512_f16_2x2`; the old adapter
module re-exports the same class. All 31 moved device function/class ASTs
are unchanged. The graph still declines deterministic mode and packed THD.

On native cc 10.7 with cuDNN 9.30, CuTe DSL 4.8, CUDA 13.4 compilation and
Torch 2.14.1+cu130, frozen source `8abfcd5b8` passed the following selections.
The exact import path, compiler target and all frozen source hashes were
checked before pytest. Run these files from `test/python` with `-m L0`:

| Suite in `sdpa/frost/` | Result |
|---|---|
| `test_sdpa_bwd_d512_sm107.py` | 104 passed; 2 SASS checks skipped because nvdisasm was unavailable |
| `test_sdpa_bwd_sm107_d512_implementation.py` | 6 passed: FP16/BF16, D384 GQA and D512 MHA/GQA, public record rebuild, fresh bindings and changed-input CUDA Graph replay |
| `test_sdpa_bwd_config_sm100_2x2.py`, `test_sdpa_bwd_explicit_engines.py`, `test_sdpa_bwd_heuristics.py` | 269 host-contract checks passed on the GPU worker |

After integrating the D128 pair parent, source `ecbdf5e8b` added
`test_sdpa_bwd_sm107_d512_determinism.py`. Its four FP16/BF16 cases use
D384 GQA (4/2 heads) and D512 MHA (4/4), causal S=257. Each checks all three
gradients against a separate reference, finite results, unmodified inputs,
fresh poisoned storage and eight replays with concurrent work. It compares
the same sequence alone, in a batch of three, reordered and duplicated,
and after changing only a neighbor's upstream gradient.

All four cases passed before recording new goldens. A separate process then
passed all four again with `CUDNN_DETERMINISM_RECORD` unset and
`CUDNN_DETERMINISM_GOLDENS` pointing at the new records. The reviewed files
are under runtime key `22fb9766a4ab5fdc`; no existing golden was replaced.
The production implementation is identical between these two snapshots.

These checks establish repeatability and batch isolation for the tested
configurations, not general batch invariance across the engine's domain.
Other GPU targets, the two SASS checks and the existing L1 time-slicing
stress were not qualified by these runs. The existing L0 delayed-empty
observer detector passed without changing its assertions.

### Explicit SM100 D512 half backward owners (2026-11-15)

The L0 `sdpa/frost/test_sdpa_bwd_d512_half_grid.py` boundary suite also
exercises the final GEMM's batch/head product. Both half dtypes and fixed
bodies compare all three gradients with an independent reference, replay
retained captures with changed upstream gradients, and remove empty sequence
neighbors while preserving packed input bytes. MQA crosses head-chunk
boundaries. The modeled SM103/SM107 admission checks are not native execution
qualification, and these bounded cases do not broaden the general batch
invariance status below. The device arithmetic and existing goldens are unchanged.

The role-split and 2x2 bodies moved without executable AST changes. Fixed
hosts name the body, parameter record and artifact independently; both retain
the shared native binder's overlap checks. The legacy graph record declines,
and the ordinary default remains role split. No numerical golden changed.

On SM100, cuDNN 9.30, CuTe DSL 4.8 and CUDA 13.4 compilation, a frozen source
snapshot passed the following selections. The loaded source was checked in
each process. Run from `test/python` with `-m L0`; paths below are under
`sdpa/frost/`. Counts include host contracts and must not be presented as
counts of native numerical cases.

| Selection | Result |
|---|---|
| `test_sdpa_bwd_config_sm100_2x2.py`; `test_sdpa_bwd_d512_explicit_engines.py`; `test_sdpa_bwd_heuristics.py`; `test_sdpa_bwd_explicit_engines.py` | 659 passed, including four native public-record rebuild/refusal checks |
| `test_sdpa_bwd_dsl_sm100.py`: `test_dense`, `test_causal_dtypes`, `test_head_dim_band`, `test_gqa_mqa`, `test_deterministic_request_is_bitwise_across_launches`, `test_prepared_sm100_rebind_stream_and_replay`, both `test_prepared_backward_artifact_reloads_in_fresh_process` variants, `test_stage2_2x2_is_bitwise_the_role_split` | 62 passed |
| `test_sdpa_bwd_thd_sm100.py`: `test_graph_thd_self_attention`, `test_graph_thd_cross_attention`, `test_graph_thd_stats_packings`, `test_graph_thd_zero_length_sequence`, `test_graph_thd_gqa`, `test_graph_thd_causal`, `test_graph_thd_causal_bottom_right`; `test_sdpa_bwd_thd_origins_sm100.py` | 45 passed |
| `test_band_capabilities.py`; `test_sdpa_graph_analyzer.py`; `test_sdpa_bwd_native_layouts_sm100.py` | 692 passed, 8 skipped because only one CUDA device was allocated |
| `test_sdpa_bwd_dsl_sm100.py`: `test_stage2_2x2_debug_clk_dump_accounts_the_waits`, `test_stage2_4x1_debug_clk_dump_accounts_the_waits`, `test_stage2_2x2_waits_for_delayed_empty_observer`, `test_stage2_2x2_sass_pins` | 8 passed; the SASS checks include cross-target compilation, not SM107 execution |

The two pytest summaries were **766 passed, 74 deselected** and
**700 passed, 8 skipped**. The deselections retain the suites' L1 cases.
Both public owner records rebuild the same body with opposite legacy-switch
values and compare all three gradient outputs as raw bits. Existing dense
and packed suites retain independent references, poisoned storage, fresh
bindings, capture/replay, GQA and physical-origin checks. Standalone fixed
hosts had separately passed 16 dense/replay and D384/Stats-packing cases.

This establishes repeatability for the tested records. General batch
invariance, SM103 execution and multi-device execution remain unverified by
this run; existing support claims elsewhere keep their original scope.

### Forward SDPA body retirement (2026-11-15)

The retirement candidate redirects standalone and retained envelope routes
onto the existing isolated bodies, removes 17 replaced source files and 18
unreachable private helpers, and transfers SM120 envelopes to explicit owners.
All retained device function/class ASTs are unchanged. Backward and the
existing numerical support statuses are unchanged; no golden was replaced.

The bounded runs used cuDNN 9.30, CuTe DSL 4.8 and CUDA 13.4 compilation,
with source imports checked in each frozen checkout. Source `6b61ce82c`
contains the tested production files; earlier snapshots differ only in test
path/owner fixes described below. Run the named suites/selectors from
`test/python` with `-m "L0 or L1"`:

| Target | Checks | Result and limits |
|---|---|---|
| SM100 B200 | `test_sdpa_fwd_small_half_explicit_engines.py`, `test_sdpa_fwd_small_fp8_explicit_engines.py`; decode quadrant/D64 envelope, paged mixed-head/D384 envelope, MXFP8 D128/D192/D256 Stats selectors | 98 passed; fixed records, rebind, changed-input capture and independent references included |
| SM107 | `test_sdpa_fwd_sm107_half_explicit_engines.py`; small-head zero-padding identity and served widths; pair and streamed-bias oracle selectors; adapter contract controls | 106 passed, 2 existing XFAIL: nonpaged D256 split lowering omits independent physical origins. These were not newly skipped or fixed by retirement |
| SM90 H100 | `test_sdpa_native_d64_sm90.py` standalone feature/envelope selectors and D64/D512 compile probes; dedicated D128 adapter/graph API; `test_owned_three_wg_split_replay_batch_and_golden` | 43 passed, 4 skipped. The skipped three-WG cases reached the missing runtime-golden check; they do not count as complete golden passes. Initial filename assertions were repaired to check the owner directory and `kernel.py` |
| SM120 RTX PRO 6000 | Half and quantized general/wide explicit-engine suites; envelope ownership, obsolete-record rejection, rebind and capture | 88 passed initially, with one stale MXFP8 helper pin. After fixing the helper, 16 targeted cases passed, including that case and additional FP8 envelope/THD and half THD/padded-Stats controls |

The four missing H100 goldens are for split3/unpacked and split17/packed,
FP16/BF16, under Torch 2.14.1+cu130 and the runtime above. Existing goldens
and support claims from earlier measured runtimes remain intact. SM103 and
SM121 were not rerun for this cleanup; neither inherits these new results.
The SM120 envelope tests establish identity, reference correctness and the
replay exercised by their suites, not general batch invariance.

### SM107 small-head FP8 implementation identity (2026-11-15)

Exact D64/D128 per-tensor FP8 uses engine 20563,
`sdpa_fwd_sm107_d128_fp8_multi_q`. Explicit dense D128 swap-AB uses 20564,
`sdpa_fwd_sm107_d128_fp8_swap_ab`, reusing the existing SM100 device body
and configuration. Native device definitions are AST-identical to the
parent, and no new copy of the swap-AB body is introduced.

`TestSmallFp8Implementation` in `sdpa/frost/test_sdpa_fp8_sm107.py` checks
public record rebuild without candidate selection, fresh tensors/workspace,
changed scales and retained capture for both native widths, both CGAs,
E4M3/E5M2 inputs, BF16/FP8 outputs and fixed splits. The same entry collects
independent D128 THD origins, existing block-output artifact/rebind/capture,
swap-AB and CGA1-split replay/batch checks, sink splits, D64 narrow tiles and
the native service-register PTX detector. Run those selected groups and
`sdpa/frost/test_sdpa_fwd_sm107_small_fp8_contract.py` from `test/python`
with `-m "L0 or L1"`.

Frozen snapshots corresponding to source `903faab1d` ran on native cc 10.7,
with cuDNN 9.30.0, CuTe DSL 4.8.0, compiler/NVRTC 13.4 and
Torch 2.14.1+cu130; the stress worker had 208 SMs. Import and architecture
were asserted. Taking the latest result for each selected case gives
**249 passes, 10 missing-golden skips and no unresolved failures**.
This includes CPU contract cases on the GPU worker and existing MXFP8
block-output controls. The final sink/narrow/PTX/contract batch passed
98 cases without skips or failures.

The intermediate runs preserve two fixture failures in their raw results:
block-output tests pinned retired engine 20514, and sink tests listened to
the old router hook instead of inspecting the selected compiled artifact.
Both were migrated and their complete affected groups passed. No numerical
tolerance or device arithmetic changed. Strict historical THD, swap-AB and
CGA1-split comparisons remain unavailable for fingerprint `aa97de087283f297`
(driver API 13050); no replacement goldens were recorded.

Native D64 and D128 refill stress each use 1,024 CTAs and seven ring refills
per work item. Independent numerical checks precede the stress loop. Each
passed 256 warm and 256 shared-memory-poisoned launches with zero byte
mismatches; the unsafe-ring control failed 200/200 launches. Existing shared
swap-AB and CGA1-split refill tests also passed. This is finite evidence,
not a proof of race freedom or a comparison between different plans.

Fifteen frozen parent/head/head/parent cases cover native D64/D128 with CGA1
and CGA2, dense, split and HALF-softmax, D128 THD, and two shared swap-AB
plans. Full configuration, parameters, TMA metadata, normalized public
knobs, workspace, inputs, outputs and changed-scale capture digests match
across all four processes. The integrated `f643ba345` host check passed
1,165 cases with eight skips; 15,360 graph/mode declarations preserve all
5,760 ordered candidates and placements against the frozen parent. The
legacy D64 unspecified CGA is normalized to its actual one-CTA default.
No other architecture or backward path is newly qualified here.

After integrating actual SM100 MXFP8 parent `b24979cd4` and F36
`a2773e404`, clean source `8b41963a0` retained 1,165 host passes/eight skips,
15,360 small-head graph/modes with 5,760 ordered records and 9,216 mid-width
graph/modes with 4,096 ordered records, all unchanged. A generic-consumer
follow-up passed **58 selected cases with zero skips or failures** on the
native SM107 worker. It covered scalar KV-length overrides and changed-input
capture, padded Stats strides/rebind/replay, standalone split domains,
compiler narrow-V rejection, envelope/HALF attributes and paging controls;
the count includes host contracts and negative controls.

The first follow-up sweep had 55 passes and three failures: two extra private
probes requested unsupported mask-free scalar-tail splits at D64 and D192,
and one paging assertion expected the legacy rejection text. Parent and new
owners reject the two split combinations identically; the final sweep checks
those explicit rejections. Paging controls first verify the corresponding
dense graph is accepted by the current owner. The support page records the
scalar-tail split limit. Arithmetic, eligibility and tolerances are unchanged;
these results extend the named selected groups, not whole-tree qualification.

The shared-consumer follow-up on source `0cafa2907` passed **1,550 checks
with nine pre-Rubin-only skips** on native SM107. It covers the complete
analyzer, heuristic-geometry, band and frontend-integration files, D512
owner contracts and the shared graph case `test693`. The initial run had
1,517 passes and 12 band-fixture failures before stopping: the SM107 swap-AB
row was sampled outside its eight-query domain, and its implementation
declines were missing from the fixture oracle. Legal short-query samples
and the current owner contracts fix those assertions without excluding
any row. All 26 host band checks and 21 host routing controls pass.

Three additional unchanged shared graph cases pass numerics and an exact
owner tally: `test1350` uses D128 multi-Q, `test522` uses D192/V128 multi-Q,
and `test670` uses D256 single-Q, all with HALF softmax. Run these cases from
`sdpa/graph/test_mhas_v2.py::test_sdpa_fp8_fwd_cc107_L0` at the default
`MHAS_CC107_MULT=4`; the test seed and CUDA data generator are unchanged.
This follow-up changes test consumers only; it does not claim a complete
cc107 graph sweep or relax any numerical tolerance.

### SM107 D192/D256 FP8 implementation identity (2026-11-15)

Exact D192/V128 and D256 per-tensor FP8 now use engines 20561 and 20562,
`sdpa_fwd_sm107_d192_fp8_multi_q` and `sdpa_fwd_sm107_d256_fp8_single_q`.
Their device definitions are AST-identical to the parent bodies. Public
record reconstruction, fresh tensors/workspace, changed scales and retained
CUDA Graph execution collect through `TestMidFp8Implementation` in
`sdpa/frost/test_sdpa_fp8_sm107.py`. The same entry retains independent THD
origins, packing/split, padded-head and fused D256 gate coverage.

Frozen source snapshots corresponding to `6a41dfff3` ran on native cc 10.7
with 208 SMs, cuDNN 9.30.0, CuTe DSL 4.8.0, compiler/NVRTC 13.4,
Torch 2.14.1+cu130 and driver API 13040. Imports and target were asserted.
The first completed native batch passed 282 cases and skipped 16: 12 lacked
historical THD goldens and four target another architecture. The second
batch passed 204 cases without skips or failures: D192 CGA1 public records,
D192/D256 dense head envelopes, both service-register PTX checks, metadata
and D192 HALF-softmax/keyless-row reference cases. These batches include
host contract tests run on the GPU worker. Historical comparisons remain
unverified for fingerprint `bc17a6445417eec9`; no new goldens were recorded.

Both refill stress cases launch 1,024 CTAs with multiple K/V ring refills.
After independent numerical checks, each passed 256 warm and 256
shared-memory-poisoned launches with zero byte mismatches. The unsafe-ring
control failed 200/200 launches. Finite stress is not a proof of race freedom.

A frozen parent/head/head/parent comparison covered ten cases: each width's
dense E4M3-to-BF16, packed split E5M2-to-E4M3, HALF-softmax and nonpaged THD,
plus D192 causal single-CTA E4M3 output and D256 BF16 sigmoid gate. Full
configuration, template parameters, TMA metadata, normalized public knobs,
workspace, inputs, O/Stats/Amax and changed-scale capture digests were equal
in all four processes. This evidence applies to those fixed plans only.

After integrating actual develop `e555c3955`, head `6f7bc694c` retained 1,126
host passes and eight skips. Its runtime delta from the frozen snapshots is
formatting and parent integration; the new implementation bodies are unchanged.
No other architecture or backward path is newly qualified here.

### SM107 D512 FP8 implementation identity (2026-11-15)

The exact D512 per-tensor FP8 graph now uses engine 20549,
`sdpa_fwd_sm107_d512_fp8_role_split`. Its role-split body and arithmetic are
unchanged. `TestD512Fp8Implementation` in `sdpa/frost/test_sdpa_fp8_sm107.py`
checks explicit public records without candidate generation, reconstruction,
fresh bindings and retained capture for E4M3/E5M2 inputs, BF16/FP8 outputs
and fixed split counts. Independent THD origins cover token/head Stats and
present/absent total hints, including replay and batch invariance through the
shared numerical reference. The existing D512 packing, split and paging
suites also collect through that entry point.

The refill stress uses eight KV tiles per work item, a three-slot K/V ring
and four-slot transfer ring, with 1,024 CTAs. It checks the numerical reference
before byte comparisons and runs 256 warm plus 256 shared-memory-poisoned
launches alongside the unsafe-ring control. This finite test is not a proof
of race freedom. Historical THD golden keys retain their old `cga2` MMA-group
suffix while public records now state the physical cluster width `cga4`;
this preserves the original baselines instead of recording replacements.

Qualification used isolated frozen snapshots culminating in `c501e487d`,
on native cc 10.7 with cuDNN 9.30.0, CuTe DSL 4.8.0, compiler/NVRTC 13.4,
Torch 2.14.1+cu130 and driver API 13050. The source import and target
architecture were asserted before pytest. Run the selected groups from
`test/python` with `-m L0`:

```text
sdpa/frost/test_sdpa_fp8_sm107.py::TestD512Fp8Implementation
sdpa/frost/test_sdpa_fp8_sm107.py::TestNativeD512Fp8Pack
sdpa/frost/test_sdpa_fp8_sm107.py::TestNativeD512Fp8Split
sdpa/frost/test_sdpa_fp8_sm107.py::TestNativeD512Fp8Paged
sdpa/frost/test_sdpa_fp8_sm107.py::TestWideFp8PhysicalOrigins
sdpa/frost/test_sdpa_fp8_sm107.py::TestServiceRegisterRelease::test_service_register_release[prefill_d512_fp8]
sdpa/frost/test_sdpa_fwd_sm107_d512_fp8_contract.py
```

The first functional run had 218 passes, four missing-golden skips and two
failures in a stale route observer: selected implementations load directly,
so the old router hook was not called. The numerical/capture checks completed
before that observer failed. The test now inspects the actual compiled
artifact, owner and physical/MMA geometry, including cache hits. The final
paged group plus larger stress and PTX detector passed 27 cases with six
missing-golden skips. The separate final-owner/origins/metadata run passed
52 with eight missing-golden skips; it also included the three affected
packing/scheduler/gate metadata tests and the running-max source inventory.
Taking the latest result for each selected case gives **266 passes,
14 missing-golden skips and no unresolved failures**; this total includes
host contract checks collected on the GPU worker.

The 1,024-CTA stress had zero byte mismatches in both 256-launch phases;
the unsafe-ring control failed all 200 launches. A frozen parent/head/head/parent
comparison covered dense E4M3-to-BF16, packed split E5M2-to-E4M3, HALF softmax
E4M3-to-BF16 and nonpaged THD E5M2-to-BF16. Complete configurations, template
parameters, TMA metadata, workspace, normalized public records, inputs,
outputs and changed-scale capture results matched exactly across all four
processes. This is bounded equivalence evidence, not a cross-runtime promise.

Historical paging and THD goldens are unavailable for runtime fingerprint
`aa97de087283f297`. Strict historical-golden qualification therefore remains
unverified for this runtime; no replacement baselines were recorded. No
other architecture or backward path is newly qualified by this change.

After integrating the shared admission changes from `a766c397a`, head
`bf16a8a5d` passed 43 additional native record/origin/paged-capture and owner
contract checks with zero failures or skips (one unchanged warm case was
deselected because the larger stress above was already qualified). The
shared CPU suites retained 1,089 passes and eight skips. Repeating the
1,152 graph/mode comparison against that new parent preserved all 1,760
ordered candidate records and placement decisions.

A follow-up caller audit migrated the generic envelope assertion and scalar
KV-tail helper to the D512 owner and physical CGA4. The helper passed native
E4M3-to-BF16 cases with split counts 1 and 4 at D512, checking six execute-time
KV lengths, fresh binding and changed-input CUDA Graph replay. The envelope
contract also passed (three cases total, zero skips). Separate host entrypoint
checks passed 43 cases, with 34 GPU skips and nine runtime cases deselected.
These are additional selected fixture checks; device/runtime code is unchanged.

The shared-consumer review follow-up on source `34350c64e` passed
1,550 selected checks on the native SM107 worker, with nine pre-Rubin-only
frontend integration skips. This covers the complete graph-analyzer,
heuristic-geometry, band-capability and frontend-integration files, the
D512 owner contracts, and shared graph `test_sdpa_fp8_fwd_cc107_L0[test693]`.
The count is primarily host contracts; the shared graph reproducer completed
its numerical checks and exact tally through engine 20549. The two analyzer
failures are fixed with current-owner semantic checks and separate retired
record rejection; 12 host tally controls retain exact counts and reject
backend/wrong-owner fallback. Runtime code and numerical tolerances did not
change, and the wider cc107 graph sweep is not claimed as qualified.

### Dense D128 FP8 swap-AB qualification (2026-11-13)

`sdpa/frost/native_fp8_dense_swap_ab.py` is exposed as `TestDenseFp8SwapAB`
in the existing SM100/SM103 and SM107 FP8 CI entry files. It pins the public
`SWAP_AB=1` plan, the actual shared decode module, and the five-stage,
eight-live-column geometry. Forward only; no backward path is added.
Packing and split count are fixed during byte comparisons. Fixed CPU-quantized
E4M3/E5M2 payloads feed FP8 and half outputs, with O/Stats batch invariance,
whole-batch Amax replay, fresh bindings, scheduling pressure, duplicated and
changed neighbors, independent numerical references and per-path goldens.
The functional matrix covers masks, sink, unequal logical head widths,
row gaps, optional Stats/Amax, empty actual lengths and current-scale capture.
D128 padded dense splits keep their existing typed decline; unsplit padding
and unpadded split execution are qualified separately. Warm stress includes
multiple ring refills and the shared unsafe-ring control.

The local SM100 engineering GPU (148 SMs) passed all 28 functional/domain/byte
cases in 58.38 s, including four newly recorded golden paths. A separate
process passed the same 28 cases in strict golden-check mode in 56.30 s:
0 failures, 0 skips; the two warm cases were run separately. Both E4M3 and E5M2
warm cases passed in 13.51 s, each with 0 wrong outputs in 256 warm and 256
shared-memory-poisoned launches. The unsafe-ring control failed 200/200 launches.
The workload has 2,048 CTAs and 13 ring refills per launch. Its numerical
reference is checked before stress; the stress reference itself is the
byte-stable cold launch. This is a finite rate check, not a proof of race freedom.

Runtime: cuDNN 9.30.0.4, CuTe DSL 4.8.0 with compiler and NVRTC 13.4,
Torch 2.14.1+cu130, driver API 13020. The Torch CUDA build label is distinct
from the compiler/runtime used for the kernels. The strict run used
`CUDNN_DETERMINISM_GOLDENS` with recording unset; the four new JSON files are
committed without replacing earlier goldens. Source import was asserted to
the isolated checkout. The final warm report names distinguish both dtypes.
Native SM107 also passed 28 functional/domain/byte cases, four independent
strict golden comparisons, and both warm cases, with no failures or skips.
Both dtypes had 0 mismatches in 256 warm and 256 poisoned launches; the
unsafe-ring control failed 200/200 launches. The frozen candidate was imported
from its isolated package, using cuDNN 9.30.0.4, CuTe DSL 4.8.0,
compiler/NVRTC 13.4 and Torch 2.14.0+cu130. Four new SM107 goldens are committed.
B300 (SM103) passed the same 28 cases, four independent strict golden
comparisons and two warm cases, with no failures or skips. Each dtype had
0 mismatches in 256 warm and 256 poisoned launches, with the unsafe-ring
control reproducing 200/200 failures. It used cuDNN 9.30.0.4, CuTe DSL 4.8.0,
compiler/NVRTC 13.4 and Torch 2.14.1+cu130; four SM103 goldens are committed.
The parent rejects the public dense `SWAP_AB=1` plan at build time; the
candidate passes the same public-entry test. Run from `test/python`:

```bash
pytest sdpa/frost/test_sdpa_fwd_fp8_sm100.py::TestDenseFp8SwapAB -m L0 -q -rs
pytest sdpa/frost/test_sdpa_fp8_sm107.py::TestDenseFp8SwapAB -m L0 -q -rs
```

The local strict command selected `-k 'not warm_refill'`; the separate warm
command selected `-k warm_refill -s`. Independent collection of all 30 cases
from the SM107 entry passed before its native execution above. The SM107
recording phase selected `-k 'not warm_refill'`; its separate strict phase
selected `-k golden` with recording unset.

On SM100, four graph boundary cases (split9 with empty partials, E5M2 window,
unequal-width strided sink, and empty Q/KV) passed each of unfiltered
memcheck, racecheck and synccheck using Compute Sanitizer 13.5: 0 errors,
0 race hazards and 0 warnings. Eleven existing D64 paged/configuration/public
knob regressions also passed. No production numerical tolerance was changed.

A final test-only extraction moved the reference and handle helpers to
`fp8_test_reference.py` to remove reverse imports from architecture entry
modules. All eight moved definitions retain identical ASTs, and all four
production files remain byte-identical to the three GPU qualifications.
The final SM100 run passed all 30 dense cases with strict goldens plus the
existing known-E5M2-edge scope regression (31 passed, no failures or skips).

Review follow-up on the same B300 runtime passed all 30 cases with recording
unset and the four existing goldens unchanged. Graph cases now poison outputs
and initialize workspace to both zero and `0xff` for eager execution and
changed/restored-input replay. Every declared Amax is also checked with
`sdpa.fp8.assert_amax_o` against the unscaled reference and stored output in
pre-scale units; the existing output check is retained. The 26 CPU-tensor
Amax checker regressions also pass. A 12-cell host precision-domain check
reproduced six incorrect HALF/BFLOAT16 admissions before the fix and passes
after aligning the public SWAP_AB domain with the lowering's None/FLOAT limit.
This changes no kernel arithmetic or default selection.

Record new goldens in a fresh directory only after numerical and replay checks
pass, then compare them in a separate process with `CUDNN_DETERMINISM_GOLDENS`.

These suites check six attention families and the DSA indexer APIs. Run from `test/python`:

```bash
pytest linear_attention/frost/test_*_determinism.py QSA/cutedsl/test_qsa_determinism.py \
  deepseek_sparse_attention/cutedsl/test_DSA*determinism.py -m L0
pytest compressed_sparse_attention/cutedsl/test_CSA_compressor.py -m L0 \
  -k 'batch_invariance or replay_determinism or determinism_golden'
pytest block_sparse_attention/cutedsl/test_BSA_determinism.py -m L0
pytest core/test_determinism_utils.py
```

| Suite | API | Forward results | Backward results |
| --- | --- | --- | --- |
| `conv/cutedsl/test_conv3d_postops_determinism.py` | Conv3D raw, norm/SiLU, prepared norm/SiLU, residual/spatial pad, C12 causal input, standalone norm/SiLU preparation | Output, cache, residual, padding and exposed workspace bytes | Unsupported (inference only) |
| `gemm/cutedsl/test_grouped_gemm_wgrad_empty.py` | `grouped_gemm_wgrad_wrapper_sm100`, cold-empty compilation | — | exact-zero overwrite and unchanged accumulated dW bytes; nonempty reuse, mixed empty experts and CUDA Graph replay |
| `gemm/cutedsl/test_grouped_gemm_dglu_activation.py` | `grouped_gemm_dglu_wrapper_sm100` and `GroupedGemmDgluSm100`, `activation_tensor` with `dgeglu` and `dswiglu` | — | activation and D_row: replay under concurrent work, expert reorder, duplicate and neighbor isolation; dSwiGLU activation byte-equal to the BF16 GLU forward D; dprob numerical only |
| `gemm/cutedsl/test_grouped_gemm_glu_branch_layout.py` | `grouped_gemm_glu_wrapper_sm100`, `glu_layout="gate_then_up"` | saved C, D in both directions, both scale planes | — |
| `gemm/cutedsl/test_grouped_gemm_dglu_gate_then_up.py` | `grouped_gemm_dglu_wrapper_sm100`, `glu_layout="gate_then_up"` | — | D in both directions, both scale planes, dprob; dbias numerical reference |
| `gemm/cutedsl/test_grouped_gemm_glu_hadamard_layout.py` | `grouped_gemm_glu_hadamard_wrapper_sm100`, `glu_layout="gate_then_up"` | C, D, pre/post-RHT amax | — |
| `gemm/cutedsl/test_grouped_gemm_glu_hadamard_quant_gate_then_up.py` | `grouped_gemm_glu_hadamard_quant_wrapper_sm100` | C, D/RHT payloads and scales, including standard GEMM-blocked column scales | — |
| `gemm/cutedsl/test_grouped_gemm_glu_hadamard_quant_canonical.py` | `GroupedGemmGluHadamardQuantSm100` and its wrapper, natural tensor ranks with either GLU branch order | C, D/RHT payloads and scales; legacy-rank equality, replay, batch invariance and graph replay | — |
| `gemm/cutedsl/test_grouped_gemm_glu_hadamard_quant_determinism.py` | `grouped_gemm_glu_hadamard_quant_wrapper_sm100` | C, D, columnwise RHT data and GEMM-blocked scales | Unsupported (forward-only fusion) |
| `gemm/cutedsl/test_grouped_gemm_dglu_bf16_determinism.py` | `grouped_gemm_dglu_wrapper_sm100`, BF16 kernel with `deterministic=True` | — | dprob and D_row: replay under concurrent work (both activations, dense/discrete, static/dynamic scheduler, ragged and empty experts), expert reorder, duplicate and neighbor isolation in both GLU orders; launch-stream ordering; D_row equal to the atomic path |
| `TestKDA` | `kimi_delta_attention` | output, final state | dQ, dK, dV, dG, dBeta, dInitialState |
| `causal_conv1d/cutedsl/test_causal_conv1d_update_determinism.py` | `cudnn.ops.causal_conv1d_update`, FP16 width-four decode | FP16 output and mutated state | Unsupported (inference only) |
| `TestGDP` | `gated_delta_product`, 1/2/3 Householder steps | output, final state | dQ, dK, dV, dG, dBeta, dInitialState |
| `TestGDN` | `gated_delta_net` | output, final state | dQ, dK, dV, dG, dBeta, dInitialState |
| `TestGDN2` | `gated_delta_net_v2` | output, final state | dQ, dK, dV, dG, dBeta, dW, dInitialState |
| `TestQSA` | `qsa_forward`, `qsa_backward` | BF16 output, FP32 output, LSE | dQ, dK, dV |
| `TestQSABackwardEngine` | `qsa_backward`, `qsa_indexed_backward` | — | dQ, dK, dV |
| CSA compressor focused cases | `CSA.csa_compressor_forward_wrapper`, `CSA.csa_compressor_backward_wrapper` | BF16 pooled output, including exact-zero capacity rows | BF16 dKV/dScore, FP32 dAPE |
| `TestDSA` | `sparse_attention_forward_wrapper`, `sparse_attention_backward_wrapper` | output, max logits, LSE, optional indexer LSE | dQ, dKV, dSink |
| `TestDSADeterministicDKV` | `sparse_attention_backward_wrapper` dKV layouts (`deterministic_dkv_mode`, SM100/SM103) | output, max logits, LSE (H128 D576: reference inputs) | dQ, dKV, dSink |
| `TestDSAIndexerForward` | `indexer_forward_wrapper` | dense scores | — |
| `TestDSAIndexerBackward` | `indexer_backward_wrapper`, `dense_indexer_backward_wrapper` | — | dIndexerQ, dWeights, dIndexerK, mutated score buffers |
| `TestDSAIndexerFP8` | `indexer_forward_wrapper` | FP32 scores; SM90 FP32 LSE | — (no backward API for FP8/MXFP8/NVFP4 inputs) |
| `moe/frost/test_shared_expert_decode_determinism.py` | `SharedExpertDecode` (the shared-expert decode kernel pair, `Geometry(HIDDEN, INTERMEDIATE)`) | BF16 output `y`, the inter-launch BF16 `act` and FP32 `gate_part` scratch rows | — (inference-only step kernels; no backward) |
| `attention_gate/frost/test_gate_quant_decode_determinism.py` | `AttentionGateQuantDecode` (the attention gate + quantize decode step, rows 1 .. 160) | FP8 E4M3 output `out` | — (inference-only step kernel; no backward) |
| `gemm/frost/projection_decode/test_fp8_projection_decode_determinism.py` | `Fp8ProjectionDecode` (the FP8 split-K projection for decode rows, `ProjectionGeometry(OUT_FEATURES, IN_FEATURES)` at a fixed `split_k`) | BF16 output `out` (the entry's only result) | — (inference-only step kernel; no backward) |
| `norm/frost/test_add_norm_quant_decode_determinism.py` | `AddNormQuantDecode` (the add + RMSNorm + quantize decode step, hidden 5120, rows 1 .. 1024) | the updated BF16 residual, BF16 output `out`, FP8 E4M3 or packed E2M1 `codes`, the NVFP4 E4M3 block `scales` | — (inference-only step kernel; no backward) |
| `attention_prepare/frost/test_attention_prepare_decode_determinism.py` | `AttentionPrepareDecode` (the attention preparation decode step, 24 query and 4 key/value heads of width 256, tokens 1 .. 1024) | BF16 `q_out` and `gate_out`, the K and V rows of the paged cache | — (inference-only step kernel; no backward) |
| `row_quantize/frost/test_row_quantize_fp4_decode_determinism.py` | `RowQuantizeFp4Decode` (the FP4 row quantization decode step, width 5120, rows 1 .. 1024) | packed E2M1 `codes`, the E4M3 block `scales` in the 128 x 4 layout | — (inference-only step kernel; no backward) |
| `row_quantize/frost/test_row_quantize_fp8_decode_determinism.py` | `RowQuantizeFp8Decode` (the FP8 row quantization decode step, width 17408, rows 1 .. 1024) | E4M3 `codes`, one FP32 scale per row (`scales`) | — (inference-only step kernel; no backward) |
| `row_argmax/frost/test_row_argmax_decode_determinism.py` | `RowArgmaxDecode` (the row argmax decode step, width 65536, rows 1 .. 1024, without a table or through an int32 / int64 id table) | the INT32 or INT64 result vector `out` (the entry's only result) | — (inference-only step kernel; no backward) |
| `test_gdn_decode_fused.py` / `test_gdn_verify_fused.py` (the `rank_geometry` cells) | `GdnDecodeFused`, `GdnVerifyFused` (the fused GDN decode / MTP-verify step kernels, per-rank `Geometry`) | FP8 output, BF16 output, SSM state slots, conv-state pool, reader tickets | — (inference-only step kernels; no backward) |
| `test_custom_reduction_determinism.py` | `pygraph.custom_reduction`: finite Torch-rule argmax/argmin and direct ADD body | INT32 index projection for FP16/BF16/FP32 values; FP32 sum output | — (no custom-reduction backward entry point) |
| `block_sparse_attention/cutedsl/test_BSA_determinism.py` (`TestBlk64Forward`, `TestBlk64Backward`) | `block_sparse_attention_forward`, `block_sparse_attention_backward`, `block_sparse_attention_backward_metadata` with `sparse_block_size=64` (SM90 blk64 kernels; the SM100/SM103 blk64 path through the same API) | O, LSE (BF16, FP16; SM100/SM103: BF16) | dQ, dK, dV (BF16; bytes measured on every route, asserted only for the prepared regular route's dK/dV; order-free golden patterns) |
| `sdpa/frost/test_sdpa_fwd_d128_sm90_determinism.py` | the SM90 D128 template (`SdpaFwdDslSm90`, the 128 x 128 tile pinned) | O, Stats | none (forward-only template; no backward) |
| `sdpa/frost/test_sdpa_fwd_sm90_owned_determinism.py` | fixed SM90 dedicated D128 and three-WG D512 bodies, FP16/BF16 | O, Stats; claim counter in stress | none (forward-only qualification) |
| `sdpa/frost/test_sdpa_mxfp8_own_block_determinism.py` | `pygraph.sdpa_mxfp8(causal_exclude_own_block=32)` (SM100/SM103 MXFP8 D128 FROST kernel) | BF16 O, FP32 Stats | — (the option is forward-only) |
| `sdpa/frost/test_sdpa_bwd_mxfp8_sm100_d128.py` | `sdpa_mxfp8_backward` on the `sdpa_bwd_sm100_d128_mxfp8` row (SM100 / SM103, D128) | — (backward only) | BF16 dQ, dK, dV; FP32 amax_dQ / dK / dV when requested |
| `sdpa/frost/test_sdpa_bwd_sm90.py` | `pygraph.sdpa_backward` through the opt-in `sdpa_bwd_sm90` row (the Hopper dense D64 / D128 key-owner chain with the ordered dQ relay) | — (backward-only engine row) | dQ, dK, dV |

### SM90 D128 template forward (issue #378): validation, 2026-11-13 (UTC)

`sdpa/frost/test_sdpa_fwd_d128_sm90_determinism.py` adds nine L0 forward-only
cases for the dedicated SM90 D128 template (`sm90/prefill_d128_f16.py`, the SM90
row's 128 x 128 tile). Every case builds the direct adapter with the tile pinned,
so a declaration the template does not serve fails instead of running the D512
body, and checks the compiled template before it compares a byte.

- Four replay and batch-invariance cells
  (`test_d128_forward_replay_and_batch_invariance`): no mask, MHA, FP16, S 500
  (partial Q and KV tiles); causal top-left, GQA 4, BF16, S_q 1280, S_kv 1500;
  causal bottom-right, MHA, FP16, S_q 448, S_kv 1000; causal bottom-right, GQA 4,
  BF16, S 256. Each plans the sequence alone and inside a batch of 40, 8, 16 or
  256 whose plan is another record of the template on an H100 (the persistent
  loop with several items per CTA, the paired causal walk, the NATURAL stride and
  the claim counter). Fresh storage and a poisoned workspace on every replay,
  under concurrent work; first, reordered, duplicated and changed-neighbor
  arrangements; an FP64 O and Stats reference; input bytes unchanged. The record
  each arrangement ran is recorded through `frost_routing.measured`, not asserted.
- Four goldens (`test_d128_forward_golden`), one per cell on the sequence alone,
  named `sdpa-sm90-dense-d128-t128-<mask>-<layout>-<dtype>` so they cannot
  collide with the D512 body's d128 records.
- One warm stress cell (`gpu_exclusive`): the K/V stages of the persistent loop
  (16 KV tiles per item, items wrapping each CTA's ring), 1000 warm and 1000
  poisoned launches. The claim counter's id ring has its warm stress in
  `test_sdpa_fwd_d128_sm90.py` (`test_d128_claim_ring_warm_stress`).

Results: NVIDIA H100 80GB HBM3 (132 SMs), CUDA driver API 13.0, CUDA 13.0, Torch
2.14.0+cu130, cuDNN 9.27.0, CuTe DSL 4.8.0; goldens recorded under runtime key
`d704b9829f609ace` (4 files). Strict run: 9 passed, 0 skipped,
0 failed. The record each arrangement ran was not captured by this run. Warm stress: the K/V stages passed; the claim counter's ring (test_sdpa_fwd_d128_sm90.py) passed.

```bash
cd test/python
CUDNN_DETERMINISM_RECORD=/tmp/d128 python -m pytest -m L0 -q -rs sdpa/frost/test_sdpa_fwd_d128_sm90_determinism.py -k golden
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest -m L0 -q -rs sdpa/frost/test_sdpa_fwd_d128_sm90_determinism.py
```

The branch also edits code the D512 body shares (`sm90/_common_hopper.py`,
`frost/tile_dsl/wgmma.py`, `config_sm90.py`). The D512 body's SM90 golden suites were recorded at develop into a new directory (127 tests) and
checked strictly against this branch: 127 passed, 0 failed, bytes identical.

Forward only: the template has no backward. THD, paged KV, split-KV, PackGQA,
sinks, windows, per-batch lengths and `s_q == 1` are declined by the template and
not covered here.


### Grid/walk split-combine zero weights — 2026-11-13

`TestSplitCombineZeroWeight` is registered in both `test_sdpa_fwd_dsl_sm100.py`
and `test_sdpa_fwd_dsl_sm107.py`. The helper module has no independent test
collection. The suite covers FP32 partial O/LSE and FP16/BF16 O in both
grid and lean-walk forms: finite-LSE underflow with NaN, positive infinity
and negative infinity payloads at 16, 32 and 129 splits. It also checks empty
rows, repeated launches, reordered/duplicated/removed sequences, changed
neighbors, multiple CTA waves and changed/restored inputs under CUDA Graph
replay. Numerical references and exact O/LSE byte comparisons are separate.
No backward or attention-producer qualification is implied.

The final numerical bodies pass all 44 cases on SM100, SM103 and SM107 with
CuTeDSL 4.8.0, compiler/NVRTC 13.4 and cuDNN backend 9.30.0. Each GPU has four
new runtime-specific golden records; existing records are unchanged. The
final predicate change was checked against these already-recorded outputs
with recording disabled. SM103 and SM107 additionally pass 182 existing
finite-input grid/walk regressions, including dtype/Stats, strided stores and
the gated grid entry. Finite-input component A/B/A checks retain exact O/LSE
bytes; they are not a serving-performance or default-selection claim.

SM100 racecheck and synccheck each pass ten BF16 underflow/replay cases with
zero reported hazards/errors. Memcheck executes the same ten passing cases
and reports no device-memory fault, but exits 99 with 40 retained host API
version-probe diagnostics from `cuGetProcAddress_v2` (requested 13030/13041,
driver 13020). This is not an all-clean CUDA API run. Native SM103/SM107
sanitizer execution is not claimed.

```bash
cd test/python
env -u CUDNN_DETERMINISM_RECORD CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest sdpa/frost/test_sdpa_fwd_dsl_sm100.py::TestSplitCombineZeroWeight \
  -o addopts= -m L0 -q -rs --tb=short
# Use test_sdpa_fwd_dsl_sm107.py on SM107.
```

The regression control predating the fix fails all 18 original BF16
underflow cells numerically. The initial fix corrected those cases but
delayed grid partial-O loads; retaining the original live-split predicate
alongside the exponential-zero check restored their early issue. The
reduction order, launch selection, workspace and pointer ABI are unchanged.

### SM100 D64 FP8 swap-AB decode — 2026-11-13

The existing `TestNativeFp8Decode` and `TestNativeFp8DecodeDeterminism` entries
now cover explicit `SWAP_AB=1` D64 decode, in addition to D128's existing
lowering. `SWAP_AB=0` and an unset knob preserve D64 prefill. The public knob
record round-trips through integer keys; unsupported geometries decline.
No existing engine eligibility or heuristic ranking changes.

Qualification on a local SM100 development GPU (NVML name `NVIDIA Graphics
Device`,148SM), cuDNN9.30.0.4, CuTeDSL4.8/compiler13.4, loaded NVRTC13.4,
Torch2.14.1+cu130: **77 cases passed** in one invocation, including six strict
goldens, both FP8 input types, half/FP8 output, independent strided K/V page
tables, partial/empty sequences, unequal logical widths, window/sink, and
changed-input retained replay. This is not qualification on a named B200/GB200
product or SM103. The two existing warnings concern xdist marker availability
and PyTorch's synchronization-debug feature.

Four existing strict records are compared against the unchanged parent in the
same runtime and retain their bytes. Only two new D64 records are committed;
no existing record or tolerance is changed. Each D64 warm-refill case uses
1776CTAs/148SM, five stage refills,256 warm+256 poisoned launches, with zero
mismatches. The unsafe-ring negative control fails200/200. D128 controls pass
unchanged. Fixed-plan O/Stats batch invariance covers insertion, reordering,
duplication and B513; whole-batch Amax is checked separately.

Before adding explicit knob plumbing, four D64 MTP/window/sink cases each
passed CUDA13.5.25 memcheck, synccheck and racecheck with zero reported errors.
The routed kernel bytes are unchanged by that plumbing. Racecheck filters
`kns=cudnn_kernel_`; CUDAAPI error reporting is disabled. These are not
unfiltered CUDAAPI or SM103 sanitizer results.

Run the existing CI entries from `test/python`, with recording unset:

```bash
pytest sdpa/frost/test_sdpa_fwd_fp8_sm100.py \
  -k 'TestNativeFp8Decode or test_native_fp8_decode_architecture_route' -m L0 -q
```

For a runtime without an existing golden environment, use the standard
parent-record/strict-compare protocol. Never record the old paths from the
candidate merely to remove a missing-golden failure.

### Hopper CUDA KDA optional recurrent states

On H200, BF16 KDA forward and backward matched bit for bit for all four
combinations of supplying an initial state and requesting a final state. The
control supplied the initial state explicitly (zeros for an absent seed) and
requested the final state (with zero upstream gradient for an unused result).
The pack had lengths 0, 65 and 63. The check covers the output, all five input
gradients, and each requested state or state gradient.

With both states omitted, two CUDA Graph replays with different query values
also matched eager output and dQ bit for bit. On H200 with CUDA 13.4,
cuDNN 9.26.0 and PyTorch 2.14.0a0, the following command passed all 24 tests.
A separate comparison with `0e6176e1` produced identical output and gradient
hashes for all four state combinations.

```bash
cd test/python
python -m pytest linear_attention/hopper/test_kda_sm90_cuda.py
```

### KDA compact batch-invariant workspace — B300, 2026-11-13–14

The FROST KDA length rule preserves each document's pieces while reserving
`min(B * P, B + total_tokens // 8192)` slots instead of `B * P`.
Empty documents retain an identity slot; checkpoint-cadence rounding can only
reduce the filled count. This bound depends only on declared shapes, so it
remains valid when device lengths change during captured replay. Non-invariant
plans and other linear-attention families keep their previous capacity.

The performance table and final CP8 precision matrix use upstream
`a9b1e43ec4071dc0ce08574b8b11d8c5a571aab8`, after #2560 was merged.
Initial qualification used `eb99ae8c6b5d4ec95b30733e46e778731a341581`.
Rebase preserves the new upstream non-invariant SM100 ragged schedule;
1,152 host-choice comparisons confirm the B300 choices are unchanged.
The additional workspace regression is measured on B300 only; SM100 and SM107
have not been rerun for this change. Hopper code is unchanged, and no new H20
measurement is claimed. No numerical threshold is relaxed.

Real CP8 measurements use 1,048,576 packed tokens, KDA H32/K128/V128,
BF16 Q/K/V and FP32 gates/beta/CP merge, with `batch_invariant=True` in both
arms. The downstream benchmark contains one KDA layer and one complete CSA
layer, both forward/backward, including ratio-4 pooling, Top-K512 selection,
sparse attention, teacher KL, gradients, layout conversion and communication.
It is not a whole-model or 7+1-layer measurement. One-time CPU planning and
JIT compilation are excluded; execution is eager.

Each arm/case runs in a fresh process with five warmups and 21 samples, then
repeats with reversed baseline/candidate order. Time is the per-iteration
slowest-rank median followed by the median of the two rounds; memory is the
maximum across ranks, measured iterations and rounds. KDA forward/backward
times are summed on each rank before selecting the maximum.

| Global THD documents | Peak allocated GiB, old → new | Saved | KDA fwd+bwd ms, old → new | KDA change | Complete benchmark ms, old → new | Time change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 mixed, two 256K long | 142.284 → 72.812 | 48.8% | 29.614 → 29.726 | +0.38% | 417.534 → 416.818 | -0.17% |
| 1,000 mixed, one 512K long | 114.554 → 72.882 | 36.4% | 30.769 → 30.222 | -1.78% | 470.033 → 469.371 | -0.14% |
| 1,024 × 1K | 86.813 → 66.350 | 23.6% | 14.127 → 12.928 | -8.49% | 120.450 → 119.212 | -1.03% |
| 20 × 1K + two 502K | 74.432 → 71.218 | 4.3% | 32.795 → 32.881 | +0.26% | 529.003 → 529.186 | +0.03% |
| One 1M | 72.956 → 72.956 | 0.0% | 29.613 → 29.673 | +0.20% | 672.316 → 672.651 | +0.05% |

K means 1,024 tokens. Peak reserved memory for these rows is respectively
155.535→87.309, 140.527→85.332, 118.389→68.707, 88.143→81.746, 75.803→75.803 GiB. The single-document allocation is unchanged.
These are real allocation reductions; arithmetic and dtypes are unchanged.

The largest aggregate changes are **+0.38% KDA** and
**+0.05% complete benchmark**. Every individual round also meets
the 5% gate: maximum changes are **+0.54% KDA** and
**+0.098% complete benchmark**. Small timing changes are
noise-level observations, not a claim that the unchanged CSA kernel became faster.

Initial qualification uses Torch 2.12.1+cu130 and cuDNN 9.20.0. CuTeDSL 4.8.0 and
4.8.1 each pass **39 non-stress cases** under both native compiler flavors,
CUDA 12.9 and 13.4. CUDA 12 compiler testing here is not a full CUDA 12
framework deployment; Torch remains the cu130 build. CUDA 13 installations
use `nvidia-cutlass-dsl[cu13]`.

After rebase, **46 workspace/piece-selection cases**, **28 determinism
cases**, both isolated 16-case native comparisons, and the complete two-round
CP8 performance/precision matrix pass again. The 92 kernel/tile-DSL files
are byte-identical to the earlier qualification, so the compiler, sanitizer,
warm-stress and PTX/SASS evidence below remains attached to those exact bodies.
The current native comparisons verify 86 baseline and 89 candidate module
paths/hashes; both also reproduce the original upstream output bytes.

The regression covers all four graph phases (forward, backward, summary,
backward summary), BF16/FP16 with log/safe gates, K/V widths 64/128, grouped
heads, FP32 caller checkpoints at cadence 16/64, empty and ragged documents,
workspace canaries and changed-length/input CUDA Graph replay. A single
symbolic prologue compilation handles varying batch/token/capacity extents,
including 1,028 documents. Allocation/compile tripwires and sync-debug mode
protect the prepared execute path. The capacity regression fails on the
original full-allocation behavior.

**28 existing KDA determinism cases** and **43 existing long/ragged/steep-gate
compatibility cases** pass. Two goldens under `e9b64a40b0600a6d` are new
runtime records made with isolated original upstream production sources and
an independent CPU FP64 reference; the candidate reproduces both unchanged.
No existing golden is replaced. The initial isolated original-versus-candidate comparison
also gives raw-byte equality for **16/16 cases**, covering all output,
gradient, state and summary-transition results, with all 86 imported source
module paths/hashes verified. The baseline uses its own test tree: the
linear-attention conftest otherwise prepends the candidate checkout to
`cudnn.__path__` even when the root package appears to come from the baseline.

Six CP8 full-versus-compact comparisons (the five distributions above plus
one weak-gate single-document case) preserve every byte of KDA O, dQ, dK,
dV, dG and dBeta on all eight ranks. This is measured coverage, not a
cross-runtime bitwise guarantee. The unchanged CSA atomic backward is
checked with its numerical tolerance, not claimed bitwise.

Source review removed the prologue's duplicate GMEM metadata writers by
using one CTA. It also corrected generic shared-memory barrier credits in
prefill, recompute and summary, retained the generic-to-async proxy fences
and warp-granular TMEM/TMA completion contracts, and ordered ticket loads
before scheduler-slot release with `consume_then_arrive`. Beta release
follows the dependent inverse store; the FP32 gate/prefix alias has a
pre-store warp synchronization. No arithmetic or precision mode is changed.

With Compute Sanitizer 13.4.92, Memcheck, Synccheck and Racecheck each pass
**18 cases with zero errors/hazards**. The genuinely isolated original
upstream Racecheck control reports 22 hazards (8 errors, 14 warnings).
Four warm-stress cases pass 3,000 warm and 3,000 shared-memory-poisoned
native graph executions each (**24,000 total**, zero mismatches); the unsafe
control fails 200/200. Their deliberately two-CTA grid forces persistent
scheduler wrapping and is not full-occupancy evidence. The actual PTX/SASS
for 32 production kernel bodies retains all 160 scheduler release
read-before-arrive dependencies. GMEM writer safety additionally relies on
the single-CTA ownership proof; Racecheck alone does not establish it.

Reproduce the in-repository checks from `test/python` with recording disabled:

```bash
export CUDNN_TEST_NO_ISOLATION=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m pytest linear_attention/test_la.py -q -s -k 'shared_stage_reuse or changed_length_replay or batch_invariance_long_ragged or piece_capacity'
python -m pytest linear_attention/frost/test_kda_determinism.py -q -s
python -m pytest linear_attention/test_la.py -q -s -k 'kda and (long_fwd_batch_invariant_bitwise or long_bwd_batch_invariant_bitwise or long_odd_chunk_count_batch_invariant_bitwise or summary_chain_batch_invariant_length_rule_bitwise or summary_bwd_chain_batch_invariant_length_rule_bitwise or long_zero_length_sequences_pass_through or context_parallel_context_geometry or steep_summary_matches_reference or steep_fp16_inverse_range_without_normalization or steep_cuda_graph_gate_transitions or steep_channel_decay_initial_state or steep_narrow_gate_ragged_tail)'
for tool in memcheck synccheck racecheck; do
  compute-sanitizer --tool "$tool" --error-exitcode 99 --report-api-errors no \
    python -m pytest linear_attention/test_la.py -q -s \
    -k 'shared_stage_reuse'
done
```

The measured sanitizer environment places a task-local `ldconfig` symlink to
`/sbin/ldconfig.real` ahead of the system shell wrapper. This avoids a child
process injection startup failure before GPU APIs, produces identical linker
cache output and changes no kernel filtering or validation. Initial failed
startup logs are retained separately from the successful qualification.

Compatibility, docstring, compile-cache, CUDA Graph and fence audits pass.
Changed-file pre-commit and syntax checks pass; Fern validation has zero
errors (the unauthenticated redirect check is skipped).

### KDA independent-head workspace reuse — B300, 2026-11-14

This follow-up starts from `03ed4530337c3e9bd01565b493d99d66a1c6924d`,
the compact-piece change described above. It targets the KDA layer's complete
input/output/gradient/transport/workspace peak. The earlier complete KDA+CSA
table is a different measurement and is not used for the below-20-GB claim.
The published change is rebased onto develop
`3fa2cfb38f6b278e1ab2de2eb26c40f2bfa3f560`, whose linear-attention sources
are byte-identical to that measured baseline.

A large backward plan partitions independent heads into two or four waves.
Each wave keeps the original sequence pieces, BT16 recurrence and checkpoint
cadence. Internal state/checkpoint head coordinates are compact; caller states,
CP summaries, gradients and saved contexts retain their original layouts.
Grouped-head and gate-parameter reductions keep their order after all waves.
No dtype, precision default, numerical threshold or golden is changed.

The H/G region is shared across their non-overlapping lifetimes. Once both
state chains finish, the combined H/G+M arena holds the checkpoint series.
X/dX remain disjoint and live until backward consumes them. Document-dense
ranks can partition all owned states as well as checkpoints. This second mode
is disabled with an external context. Caller/coarse checkpoints retain their
full-head ABI. The original row tables remain immutable: wave counts use a
separate two-element buffer because the original counts alias terminal row
entries. Every regenerated queue row has exactly one writer. The main queues place
pieces with at least 512 tokens (BT16) before shorter pieces, reducing the tail
of the persistent schedule. A CTA prefix scan assigns disjoint destinations
from the two ends of the queue. Partial rounds participate with zero counts;
two blocking CTA barriers publish and retire the eight warp counts. Queue
ordering changes no task identity, chunk bounds or floating-point operation.

The shape-only policy has a 5 GiB checkpoint/state capacity preference and
an occupancy guard. It is not a universal memory limit for every head size,
external state or unbounded document count. The added compile-key combinations
are (checkpoint divisor, state divisor) = (1,1), (2,1), (4,1), (2,2), (4,4).
The compile-count detector requires exactly four new artifacts across three
ragged capacities for the four nontrivial combinations. Warm prepared execution
retains the existing stream/capture contract and adds no allocation or length
readback.

Qualification on B300, driver 580.173.02, Torch 2.12.1+cu130 and cuDNN 9.20.0:

- **60 new non-stress tests pass**: 48 BF16/FP16 log/safe-gate cases across
  equal/GQA/GVA heads, both divisors and both storage modes; two policy/cache
  tests; two rectangular K/V cases, two external-context geometries, two
  non-invariant chain cases, two queue identity/partition cases and two
  standalone default state-chain ABI cases with CPU FP64 references.
  They compare all input/state/parameter gradients, canaries, poisoned native
  graph replay and changed input/length data at stable addresses.
- **71 existing tests pass**, including 28 determinism/golden/reference cases,
  the compact-piece regressions, and four GDN/GDN2 long-sequence checks
  of the shared chain default. The standalone default tests also pass on the
  unchanged parent. Existing numerical tolerances and goldens are unchanged.
- **16 equal-head cases pass in each additional compiler lane**:
  CuTeDSL 4.8.0 with CUDA compiler 12.9/13.4, and 4.8.1 with compiler 12.9.
  The main 60-case lane uses 4.8.1/compiler 13.4. CUDA 13 dependencies use
  `nvidia-cutlass-dsl[cu13]`. All four lanes use the final v7 kernel sources,
  including long-first queue scheduling; the extra lanes explicitly force both
  divisors. CUDA 12 compiler testing does not qualify a fully CUDA 12 framework
  deployment.
- **6,000 warm/poisoned native graph executions pass** (3,000 of each), with
  the unsafe control failing 200/200. The deliberately two-CTA grid forces
  scheduler wrapping; it is not full-occupancy stress evidence.
- **Memcheck, Synccheck and Racecheck each pass three cases** (one full-state
  wave and both direct queue-partition divisors),
  including changed-metadata capture replay, with zero errors/hazards. No
  kernel or API-error filtering is used. This sanitizer lane sets
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`: the ordinary expandable
  allocator reports a recoverable `cuMemCreate` permission failure under
  instrumentation. An independent Torch-only program reproduces the two API
  errors with expandable allocation and reports zero with regular allocation.
  Numerical/performance runs retain their original allocator settings.
- Actual generated code covers 13 queue/chain prerequisite waits and retains
  100 scheduler release dependencies in 20 summary/recompute bodies across
  five rendered hosts. PTX data-flow checks and SASS mask/address/arrival
  checks pass. Backward's existing SMEM handoffs are unchanged; this review's
  new ownership edges are the inter-kernel queues, phase-arena reuse and
  generic shared-memory prefix counts. The three emitted queue kernels retain
  shared stores → `barrier.sync` → shared loads → `barrier.sync` before reuse.
  A follow-up SASS operand audit additionally verifies that all eight words
  from the two `LDS.128` instructions are consumed before the retirement
  barrier, including the eighth count used only in the total. The consumption
  precedes the valid-row branch, so tail lanes are covered. Barrier placement
  alone would not prove this obligation; no extra fence was added to the
  unchanged numerical kernels.

The 2026-11-14 line-by-line review adds **six boundary tests**, all passing on
the same B300 / DSL 4.8.1 / compiler 13.4 environment in 306.15 s (51 deselected,
no failures or skips). The seven production files are unchanged from the
original qualification; the earlier test totals above are retained evidence,
not a new rerun of those suites.

- Four runtime-head tests compare all backward outputs with the unsplit D=1
  reference for 16 shape/configuration pairs, including 6 and 12 heads. Both
  storage modes and divisors are covered. Each case retains exactly two
  compiled launch objects (D=1 and its wave configuration) as head count changes;
  all comparisons are byte-identical and workspace canaries remain intact.
- Two independent-stream tests use separate preallocated workspaces, explicit
  handles/streams and captured graphs sharing compiled code. Each stream runs
  64 poisoned replays per storage mode: 256 replays total. Both streams are
  submitted before waiting, with comparisons every eight replays. All gradient
  bytes match the D=1 reference, input ports remain unchanged, and all canaries
  pass. This covers concurrent submissions; it does not claim a measured
  percentage of hardware overlap or permit shared workspace across streams.

Run only these additions with:

```bash
python -m pytest linear_attention/frost/test_kda_checkpoint_waves.py -q -x -rs -k 'runtime_heads or independent_streams'
```

Reproduce the new checks from `test/python`:

```bash
export CUDNN_TEST_NO_ISOLATION=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m pytest linear_attention/frost/test_kda_checkpoint_waves.py -q -rs -k 'not warm_stress'
python -m pytest linear_attention/frost/test_kda_wave_boundaries.py -q -rs
python -m pytest linear_attention/frost/test_kda_wave_queues.py -q -rs
python -m pytest linear_attention/frost/test_kda_checkpoint_waves.py -q -s -k warm_stress
for tool in memcheck synccheck racecheck; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
    compute-sanitizer --tool "$tool" --error-exitcode 99 \
    python -m pytest linear_attention/frost/test_kda_checkpoint_waves.py \
    linear_attention/frost/test_kda_wave_queues.py -q -rs \
    -k 'test_wave_queue_identity or (all-states and equal and bf16-log-d64 and 2)'
done
```

The first non-invariant fixture failed its routing assertion because its
oversubscribed SM103 shape selected the uncut path. Reducing its nominal
sequence/head grid selects the ordinary chain and both cases pass; no production
routing or numerical threshold was changed. Earlier failed/interrupted runs
are retained separately and are not counted as passes.

The additional memory/replay measurements are B300-only. No new H20, B200 or
SM107 result is claimed, and Hopper code is unchanged.

Final real CP8 measurements use eight B300 devices, 1,048,576 total tokens,
131,072 tokens per rank, H32/K128/V128, BF16 Q/K/V/dO and FP32 gates, beta,
their gradients and CP merge. `batch_invariant=True` in both arms. Each case
runs the parent and candidate in separate fresh processes, three warmups and
31 measured iterations per arm. Time is the median of the per-iteration
maximum-rank forward+backward time; memory is the maximum over ranks and
measured iterations. The region includes KDA and boundary communication;
one-time CPU planning/JIT, validation and hashing are excluded. This is eager
execution, not a CUDA Graph performance measurement.

All inputs, outputs, gradients, CP transport tensors and workspace contribute
to the allocated peak. Peak counters are reset before releasing the preceding
iteration's gradients, so the iteration transition is included. The reference
already includes #2574's compact-piece optimization. Both arms use the same
existing cuLA integration snapshot, including its pre-existing worktree changes;
no pristine cuLA commit or new cuLA change is claimed. The CP harness is separate
from this repository; the native regression commands above are self-contained.

| Global THD token lengths | Allocated GiB, before → after | Reserved GiB, before → after | F+B ms, before → after | Change |
| --- | ---: | ---: | ---: | ---: |
| 2,048 × 512 | 22.998 → 17.271 | 23.596 → 17.869 | 13.520 → 13.562 | +0.31% |
| 1,024 × (480, 544) | 22.998 → 17.271 | 23.596 → 17.869 | 13.610 → 13.550 | -0.44% |
| 2,047 × 256 + 524,544 | 25.754 → 17.372 | 26.850 → 18.467 | 29.551 → 29.639 | +0.30% |
| 1,536 × 64, then 512 × 1,856 | 36.953 → 17.256 | 42.158 → 17.270 | 22.071 → 22.813 | +3.36% |
| 256 × 1,856; 1,536 × 64; 256 × 1,856 | 28.990 → 18.108 | 29.850 → 18.969 | 18.002 → 18.381 | +2.11% |
| 4,096 × 256 | 25.749 → 17.366 | 26.848 → 18.465 | 14.851 → 15.109 | +1.74% |
| 8,192 × 128 | 31.252 → 18.619 | 33.352 → 20.719 | 17.500 → 17.902 | +2.29% |
| one 1M, weak gate | 20.285 → 16.215 | 21.219 → 17.148 | 29.511 → 29.766 | +0.86% |

The maximum allocated footprint is **19.992 decimal GB**
(18.619 GiB), below 20,000,000,000 bytes for all eight measured cases.
The maximum allocator reservation is **22.247 decimal GB**
(20.719 GiB), which is **not below 20 GB**. These Torch counters do
not include driver/context allocations or every NCCL allocation. The claim is
about the live KDA tensor/workspace footprint, not a process-wide reservation
ceiling, a whole-model footprint or a universal limit for arbitrary THD layouts.

The sensitive middle-skew case also passes a second round with candidate-first
order: 17.955 → 18.352 ms
(+2.21%). The largest measured latency regression across
the table and this repeat is **3.36%**. Full-tensor SHA256 records match
for all six inputs and O/dQ/dK/dV/dG/dBeta on all eight ranks, in every paired
case and the reversed-order repeat. Hashing is outside the timed region.
The weak-gate control uses one 1M document with gates scaled by 0.01.

The expanded low-document matrix on the same unchanged production code exposes
a performance exception: global **256 × 4K** at real CP8 takes
12.663 → 13.557 ms (**+7.06%**), reproduced with candidate-first arm order at
12.656 → 13.563 ms (**+7.16%**). All output and input-gradient byte hashes still
match on every rank. Disabling head waves diagnostically recovers latency to
12.659 ms, but raises candidate allocated memory from 16.285 to 20.309 GiB.
This is not a production fix. The initial eight-case table remains valid, but
the <=5% criterion is **not met over the expanded coverage**; PR #2577 remains
Draft pending a scheduling-policy improvement. The full 13-case matrix and
Nsight Systems attribution are retained in the PR description.

Compatibility, docstring, compile-cache, CUDA Graph and fence audits pass.
Changed-file pre-commit and syntax checks pass; Fern reports zero errors and
one warning because unauthenticated redirect checking is unavailable.


### H96 KDA workspace qualification — B300 CP8, 2026-11-14

This follow-up retains the preceding baseline/candidate production sources and
measurement method, increasing equal Q/K/V heads from 32 to **96 per rank**.
Global THD length is 1,048,576, local length 131,072, K=V=128, BF16 Q/K/V/dO,
FP32 gates/beta/CP merge, and batch invariance enabled. Every arm is a fresh
process with three warmups and 31 measured forward+backward iterations, including
CP boundary communication. Arm order alternates across the ten cases. This is
KDA only; input construction, planning/JIT and full-tensor hashing are outside
the timed region.

| Global THD lengths | F+B ms, before → after | Change | Allocated GiB, before → after | Candidate reserved GiB |
| --- | ---: | ---: | ---: | ---: |
| 256 × 4,096 | 37.116 → 39.419 | +6.21% | 61.746 → 42.797 | 42.801 |
| one 1M | 81.786 → 84.159 | +2.90% | 60.832 → 42.609 | 45.406 |
| 20 × 1,024 + 2 × 514,048 | 90.681 → 94.813 | +4.56% | 61.409 → 42.678 | 45.406 |
| 2,048 × 512 | 39.621 → 40.103 | +1.22% | 68.967 → 43.198 | 44.986 |
| 1,024 × (480, 544) | 39.803 → 40.006 | +0.51% | 68.967 → 43.198 | 44.986 |
| 2,047 × 256 + 524,544 | 80.328 → 83.474 | +3.92% | 77.238 → 44.155 | 47.434 |
| 1,536 × 64; 512 × 1,856 | 63.007 → 64.648 | +2.60% | 110.828 → 51.764 | 51.801 |
| 256 × 1,856; 1,536 × 64; 256 × 1,856 | 51.539 → 52.565 | +1.99% | 86.941 → 45.261 | 56.895 |
| 4,096 × 256 | 43.442 → 43.956 | +1.18% | 77.221 → 44.138 | 47.428 |
| 8,192 × 128 | 50.920 → 51.918 | +1.96% | 93.725 → 46.018 | 52.312 |

All ten pairs have identical full-tensor hashes for the six inputs and
O/dQ/dK/dV/dG/dBeta on every rank, with finite outputs/gradients. All seven
loaded production-module hashes match the qualified source manifests.
Allocated memory decreases **29.96–53.29%**. Nine cases stay within 5% latency
regression in this round; **256 × 4K regresses 6.21%**. This exception retains
the PR's Draft status rather than widening the acceptance threshold.

Both cases with at least 4.5% first-round regression were repeated in reverse
arm order (candidate first), again with three warmups and 31 samples:

| Global THD lengths | F+B ms, baseline → candidate | Change | Byte equality |
| --- | ---: | ---: | --- |
| 256 × 4,096 | 37.122 → 39.374 | +6.07% | all ranks, outputs and input gradients |
| 20 × 1,024 + 2 × 514,048 | 90.712 → 94.801 | +4.51% | all ranks, outputs and input gradients |

The memory peaks are unchanged. The first case remains above the 5% target;
this H96 follow-up also leaves the earlier H32 7.06% / 7.16% exception unresolved.

The candidate uses four checkpoint waves throughout. Document-dense ranks also
use four state waves; long-document ranks retain full-head states. Mixed
layouts can therefore select (4,4) and (4,1) on different ranks. The table
reports the median of per-iteration maximum-rank F+B and the memory maximum
across ranks/iterations, not an average that hides imbalance.

H96 inputs, outputs and gradients alone occupy approximately **36.094 GiB /
38.755 decimal GB per rank**, before transport or workspace. The H32 below-20-GB
result does not extend to this H96 configuration. Candidate allocated peaks
range from 42.609 to 51.764 GiB; the largest reservation is 56.895 GiB.
These are Torch counters, not process-wide VRAM caps. No new H20/SM90, SM100
or SM107 result is implied.

The native test follow-up uses BF16/log-gate **D64**, with FP32 gates and
initial/final states; the CP8 performance matrix above uses **K=V=128**.
The same runtime as the preceding qualification is retained: Torch 2.12.1+cu130,
cuDNN 9.20.0, CuTeDSL 4.8.1/compiler 13.4 and driver 580.173.02. Run from
`test/python` with an explicit handle bound to each capture stream:

```bash
CUDNN_TEST_NO_ISOLATION=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -m pytest linear_attention/frost/test_kda_checkpoint_waves.py -q -x -rs -k 'runtime_heads or (independent_streams and h96)'
```

**6 passed, 53 deselected, 2 warnings in 315.64 s**, with no failures or skips.
The warnings are Torch's `torch.jit.script` deprecation. Four runtime-head
cases now cover 20 shape/configuration pairs, including H96 for both divisors
and both storage modes, while reusing the same two compiled launch objects
per case. Two H96 independent-stream cases run 64 poisoned CUDA Graph replays
per stream and storage mode: **256 total**, comparing every eight replays.
All backward outputs match the unsplit D=1 path byte for byte; input immutability
and workspace canaries pass. Concurrent submission does not measure hardware
overlap or permit sharing a workspace across streams.

All seven loaded production-module hashes and the executed test-source hash
were checked against the local source. This follow-up changes tests and
documentation only; production kernels remain byte-identical to `a19c476d`.
No numerical tolerance or golden changed. Earlier sanitizer and warm-stress
results remain retained evidence, not H96 reruns. Compatibility audit: passed
for the added head-count/storage-mode checks. Docstring audit: passed for the
unchanged production surface.

### Additional H96 workspace reduction — B300 CP8, 2026-11-14

This follow-up compares against the preceding qualified memory-optimized
parent `e74a4ced` (production `a19c476d`). It retains the H96-per-rank,
K=V=128, real CP8, 1M global THD configuration and all-resident measurement
method above (warmed eager F+B; CUDA Graph is tested separately). Forward reuses private FP32 H/M/X in at most two head waves.
Backward may use eight waves only when the memory trigger is met and each
wave has at least four planned piece/head tasks per SM; few long documents
retain more head parallelism. This shape-only estimate is neither an
occupancy guarantee nor a hard memory cap.

| Global THD lengths | F+B ms, parent → new | Change | Allocated GiB, parent → new | New reserved GiB |
| --- | ---: | ---: | ---: | ---: |
| 256 × 4,096 | 39.419 → 39.369 | -0.13% | 42.797 → 42.797 | 42.801 |
| one 1M | 84.159 → 84.114 | -0.05% | 42.609 → 42.609 | 45.406 |
| 20 × 1,024 + 2 × 514,048 | 94.813 → 94.791 | -0.02% | 42.678 → 42.678 | 45.406 |
| 2,048 × 512 | 40.103 → 40.791 | +1.72% | 43.198 → 39.700 | 41.488 |
| 1,024 × (480, 544) | 40.006 → 40.702 | +1.74% | 43.198 → 39.700 | 41.488 |
| 2,047 × 256 + 524,544 | 83.474 → 83.471 | -0.00% | 44.155 → 42.609 | 45.406 |
| 1,536 × 64; 512 × 1,856 | 64.648 → 66.416 | +2.73% | 51.764 → 42.107 | 49.818 |
| 256 × 1,856; 1,536 × 64; 256 × 1,856 | 52.565 → 53.782 | +2.31% | 45.261 → 40.743 | 46.572 |
| 4,096 × 256 | 43.956 → 44.813 | +1.95% | 44.138 → 40.171 | 41.820 |
| 8,192 × 128 | 51.918 → 53.363 | +2.78% | 46.018 → 41.114 | 49.195 |

All ten cases preserve the six input hashes and full O/dQ/dK/dV/dG/dBeta
hashes on all eight ranks. Every output/gradient is finite, and the ten
loaded production-module hashes match the measured snapshot. The allocated
range becomes **39.700–42.797 GiB**, versus **42.609–51.764 GiB** for the
parent; maximum additional latency is **2.78%** in this matrix. The worst
skewed case saves **9.657 GiB / 18.66%**. The unchanged all-resident tensors
alone still require about **36.094 GiB** before transport or workspace.
Reservations are Torch allocator high-water marks, not a cuDNN-owned pool
or a process-wide memory cap.

An unguarded eight-wave prototype was rejected: one long document, mixed22
and the 2,048-document long-tail case regressed 11.48%, 8.98% and 12.25%
relative to the parent. The final policy protects these long-document ranks.
Against the *original unoptimized* baseline, the selected implementation
still has +6.07% at 256 × 4K and +5.41% at the first skewed case. Thus this
follow-up does not meet the original all-case 5% PR acceptance criterion;
the prior H32 exception is not resolved or rerun.

The final source additionally moves kernel-name registration into the shared
queue module, with a prefix regression, so forward import order does not
depend on a backward caller. All function/class ASTs remain identical to
the ten-case measured implementation after that naming-only correction.
No arithmetic, dtype, reduction order, public tensor layout, numerical
tolerance or golden changed. Runtime head/piece extents remain symbolic;
the new compile axis is the bounded divisor.

Final-source native runs use the same Torch 2.12.1+cu130, cuDNN 9.20.0,
CuTeDSL 4.8.1/compiler 13.4 runtime as above. Commands from `test/python`:

```bash
python -m pytest -q -x -rs linear_attention/frost/test_kda_forward_waves.py \
  core/frost/test_frost_kernel_name_prefix.py -k 'forward or kda_head_wave'
python -m pytest -q -x -rs linear_attention/frost/test_kda_checkpoint_waves.py \
  -k '(bitwise_and_capture and (equal-8 or gqa-8 or gva-8)) or capacity_policy or (runtime_heads and 8)'
```

With `CUDNN_TEST_NO_ISOLATION=1` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`, the
forward/prefix selection reports **18 passed, 4 deselected, 2 warnings in
413.10 s**; backward reports **27 passed, 58 deselected, 2 warnings in
449.21 s**, with no failures or skips. Warnings are the existing Torch JIT
deprecation. Forward covers BF16/FP16, log/safe gates, equal/grouped heads,
rectangular states, empty/ragged sequences, poisoned capture replay,
changed prefixes and public checkpoints plus all autograd gradients.
Runtime-head checks reuse the same two compiled launch objects across head
counts, including H96. Backward covers eight-wave checkpoint-only and
owned-state modes, safe/log gates, FP16/BF16, equal/grouped heads, changed
prefixes and H96 runtime-head reuse.

The forward warm stress completes 3,000 warm and 3,000 SMEM-poisoned launches
with zero wrong bytes; its unsafe control fails 200/200. The target forces
**two total CTAs** to exercise repeated work, below the control's three CTAs
per SM. This is a limited race-rate check, not proof of safety for arbitrary
scheduling. Input immutability and workspace canaries also pass.

The first backward selection aborted because a newly expanded fixture forced
eight waves on only four heads. The fixture now sizes equal/grouped/safe-gate
operands consistently using `max(4, divisor)`; production selection already
requires divisibility. Its exit-134 log is retained and not counted as a pass.
The corrected run uses a fresh compiled-cache directory. Superseded V1/V2
incomplete native runs likewise are not counted.

Compute Sanitizer 13.4 runs the BF16/log D64 forward two-wave capture case
and the backward eight-wave owned-state capture case, including changed
prefixes. Each tool reports **2 passed**, no skips: Memcheck **0 errors**
(79.33 s), Synccheck **0 errors** (86.60 s), Racecheck **0 hazards, 0 errors,
0 warnings** (1818.23 s). No kernel/API filtering is used. The instrumentation
lane uses `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` for the
previously isolated allocator/tool incompatibility; ordinary measurements
retain their original allocator settings. These are D64 checks, not a
sanitized CP8 1M run.

Fresh-process paired repeats use the final production source, three warmups
and 31 timed iterations per arm. Arm order is parent→candidate for mixed22
and uniform8192, candidate→parent for skew2048. These are timing-only repeats:
finite, geometry and all loaded source hashes are checked; full-tensor byte
parity is supplied by the ten-case matrix and native tests above.

| Layout | F+B ms, parent → final | Change | Allocated GiB, parent → final |
| --- | ---: | ---: | ---: |
| low-mixed22 | 94.637 → 94.812 | +0.18% | 42.678 → 42.678 |
| skew2048 | 64.607 → 66.318 | +2.65% | 51.764 → 42.107 |
| uniform8192 | 51.921 → 53.371 | +2.79% | 46.018 → 41.114 |

The memory peaks are unchanged from the primary matrix.

The final forward fixture additionally changes V and G in place alongside
the THD prefixes, verifying that replay rebuilds both H and M from current
values. The complete forward/prefix selection was rerun: **18 passed,
4 deselected, 2 warnings in 177.69 s**, no failures or skips. It again completes
3,000 warm and 3,000 poisoned launches with zero wrong bytes and an unsafe
control of 200/200; the same two-CTA coverage limitation applies. Production
source is unchanged by this additional test. All eight changed source/test
files were matched against the final remote snapshot.

Changed-file pre-commit and Python compilation checks pass. Fern reports
zero errors and one warning because the unauthenticated redirects check is
skipped. No public API or dependency requirement changed.

No new H20/SM90, SM100, SM107, CSA or full-model result is implied. Earlier
independent-stream two/four-wave backward evidence remains retained, not an
eight-wave or forward concurrency rerun. Compile-cache and CUDA Graph audits:
passed for the exercised runtime-head, prefix and replay configurations;
other architectures remain unverified for this addition.

### H96 exact state reuse and dense-THD workspace — SM100 CP4, 2026-11-15

This experiment branches from `a1de8974`. The budget is now **workspace only**:
external Q/K/V/G/Beta/dO, output and input gradients are excluded. Earlier
all-resident H96 figures above must not be interpreted as workspace figures.
The target is decimal 20 GB (18.626 GiB).

Forward H and X use exactly the same FP32 region. A row-owning lane loads its
H vector before storing X, in one inline PTX block with named temporary
registers. M remains separate; no other lane reads that H vector. This saves
one of three forward state arrays for square K/V without extra launches.
Normal shapes retain one/two forward waves. Above 16 GiB of live state per
wave, four/eight waves are allowed only with at least four planned tasks per
SM. Backward can additionally use sixteen waves above a conservative 128 GiB
full-arena estimate, with the same task guard. These are bounded shape
policies, not occupancy guarantees or a universal workspace cap.

Per-head recurrence, arithmetic order, chunking, FP32 states and CP merge,
public tensor layouts, coarse checkpoints and numerical tolerances are
unchanged. No golden is updated. The existing compile-key divisor domain is
bounded; token counts, sequence prefixes, pointer values and runtime head
counts do not become new specialization axes.

#### Measurement

The specified four-GPU environment runs real CP4, global 1,048,576 tokens,
262,144 tokens per rank, H96 per rank, K=V=128, BF16 Q/K/V/dO, FP32 gates and
states, and `batch_invariant=True`. The same cuLA working-package snapshot
is used by both arms. It includes pre-existing contiguous CP summary and
backward context-range fixes; no cuLA source is changed in this branch.

Each case starts in a fresh process: three warmups and fifteen eager F+B
iterations, including CP boundary communication when needed. Latency is the
median of the per-iteration slowest rank. Planning, compilation, random input
creation and hashing are excluded. Full input and O/dQ/dK/dV/dG/dBeta hashes
are compared per rank. Source-module hashes and loaded runtime are recorded.

Native workspace is the exact requested per-graph buffer size. A separate,
untimed audit samples active internal allocations at every cuDNN execution,
subtracting deduplicated external I/O storage live at that phase; it includes
summary/context and Torch-owned CP transport. This is an execution-boundary
sample, not an allocation-event trace or a whole-device/NCCL allocation cap.
Allocator reservations are reported separately. CUDA Graph is checked for
correctness, not used for these performance measurements.

All memory values below are decimal GB. Layouts are in packed document order;
`n × length` denotes n documents. Every row has full per-rank byte parity.

| Global THD layout | Native workspace GB, parent → candidate | Sampled internal GB, parent → candidate | F+B ms, parent → candidate | Latency change |
| --- | ---: | ---: | ---: | ---: |
| 1 × 1,048,576 | 13.117 → 13.117 | 13.388 → 13.388 | 174.246 → 174.570 | +0.19% |
| 20 × 1,024 + 2 × 514,048 | 13.331 → 13.331 | 13.401 → 13.401 | 209.410 → 209.891 | +0.23% |
| 256 × 4,096 | 7.701 → 7.701 | 7.752 → 7.752 | 80.165 → 80.124 | -0.05% |
| 2,048 × 512 | 7.530 → 7.530 | 7.581 → 7.581 | 82.711 → 82.714 | +0.00% |
| 1,024 × (480, 544) | 7.530 → 7.530 | 7.581 → 7.581 | 82.651 → 82.676 | +0.03% |
| 2,047 × 256 + 524,544 | 13.117 → 13.117 | 13.388 → 13.388 | 172.667 → 173.322 | +0.38% |
| 1,536 × 64; 512 × 1,856 | 15.652 → 10.440 | 15.715 → 10.503 | 106.016 → 106.027 | +0.01% |
| 256 × 1,856; 1,536 × 64; 256 × 1,856 | 8.643 → 8.270 | 8.712 → 8.333 | 92.881 → 92.802 | -0.08% |
| 4,096 × 256 | 9.975 → 8.542 | 10.025 → 8.593 | 91.462 → 91.444 | -0.02% |
| 8,192 × 128 | 19.648 → 13.105 | 19.699 → 13.155 | 109.656 → 109.629 | -0.02% |
| 16,384 × 64 | 38.993 → 14.616 | 39.044 → 14.666 | 151.345 → 151.547 | +0.13% |
| 32,768 × 32 | 77.684 → 13.037 | 77.735 → 13.087 | 234.933 → 237.908 | +1.27% |

All twelve layouts are below 20 GB: the maximum native workspace is
**14.616 GB**, and maximum sampled internal storage is **14.666 GB**.
The largest measured latency regression is **1.27%**, on 32,768 documents.
The parent already meets the workspace target for ten of twelve layouts;
the two dense extremes are reduced from 39.044/77.735 GB sampled internal
storage to 14.666/13.087 GB. Some forward savings do not lower the overall
maximum because backward was already the larger phase.

Measurement provenance: V2 measured all twelve layouts. Final V3 adds only
the guarded backward-sixteen-wave selection; one-document, skew2048 and
32,768-document rows were rerun on V3. The other nine rows retain their V2
measurements. Captured geometry replay through both selectors confirms
identical decisions for those nine; constructor/allocation/execution ASTs
outside the two policy helpers and constant are identical, and the state-chain
source has the same SHA256. They are not presented as exact-V3 reruns.

The comparison is against the already-optimized parent `a1de8974`, not the
original pre-wave baseline in the earlier B300 report. Earlier performance
exceptions in that report are not resolved by this comparison.

#### Native and safety validation

Runtime: Torch 2.14.1+cu130, CUDA 13.0, cuDNN 9.24.0, CuTeDSL 4.8.1 with
CUDA 13.4 compiler. Both arms use the same extension rebuilt from the parent
source in an isolated venv. Commands run from `test/python`, with
`CUDNN_TEST_NO_ISOLATION=1` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

```bash
python -m pytest -q -x -rs linear_attention/frost/test_kda_state_reuse.py \
  linear_attention/frost/test_kda_forward_waves.py \
  linear_attention/frost/test_kda_wave_boundaries.py -k 'not warm_stress'
python -m pytest -q -x -rs linear_attention/frost/test_kda_checkpoint_waves.py \
  -k '(bitwise_and_capture and (equal-16 or gqa-16 or gva-16)) or capacity_policy or (runtime_heads and 16)'
```

- V1: **35 passed, 6 deselected in 482.99 s**. Warm stress completed
  3,000 warm and 3,000 poisoned launches with zero wrong bytes; the unsafe
  control failed 200/200. The tested shape has only two total CTAs, below
  the control's three CTAs/SM, so this is limited schedule coverage rather
  than a universal race-free guarantee.
- V2: **41 passed, 1 deselected in 764.14 s**, no skips or failures. The
  deselected warm-stress case ran in V1 with identical state-chain source.
  Includes forward four/eight-wave coverage, cache reuse and independent
  FP64 state-chain references.
- V3 backward sixteen-wave selection: **27 passed, 84 deselected in
  569.96 s**, no skips or failures. The 24 parity/capture combinations cover
  two storage modes, three head patterns and four dtype/gate cases; two
  additional runtime-head checks include H96, plus the policy boundary test.
- V3 forward policy: **1 passed, 16 deselected in 2.39 s**.

These selections overlap; their counts are not a count of unique tests.
Compile-cache audit: passed for the exercised forward/backward head waves;
runtime head changes reuse the same two baseline/wave compiled objects.
CUDA Graph audit: passed for the exercised preallocated native calls,
poisoned workspaces, changed-input replay and prefix changes. This is not a
whole distributed CP4 captured-graph pool benchmark. No C++ port is involved.

Standalone alias coverage includes K/V64/128 square and rectangular states,
forward/reverse, one/three pieces, tail and transition product, changed values
at stable addresses, graph replay and a trailing canary. Shared chain entry
checks also compare with an independent FP64 recurrence. Forward wave tests
cover grouped heads, log/safe gates, FP16/BF16, public checkpoints and all
autograd gradients; backward sixteen-wave tests cover grouped heads,
checkpoint-only/owned-state paths, changed lengths and H96 runtime reuse.

Compute Sanitizer 13.4.92, scoped to the sixteen standalone alias/capture
cases: Memcheck **16 passed, 1 deselected, 0 errors (15.22 s)**;
Synccheck **16 passed, 1 deselected, 0 errors (11.78 s)**; Racecheck
**16 passed, 1 deselected, 0 hazards, 0 errors, 0 warnings (133.80 s)**.
No device kernel filter or memory/race/synchronization suppression is used.
These are small native tests, not a sanitized distributed CP4 1M run.

This is a **qualified, not unfiltered, sanitizer pass**. Unfiltered memcheck
on the candidate reports 41 `cuGetProcAddress_v2` / error-1 API queries;
the unchanged parent independently reports the same 41 queries while both
of its default-chain FP64-reference tests pass. The newer tool's default
extended mode additionally prints 41 driver diagnostics: requested API
versions 13010–13041 exceed driver version 13000. The saved parent XML
attributes all 41 explicit errors to `cuda.bindings._internal.driver`.
Neither unfiltered run reports a device memory violation, but both exit 99
and are retained as failures, not counted as clean tool passes.

The qualified runs retain explicit API error checking and suppress only
`kind=Api`, `api=cuGetProcAddress_v2`, `result=1`; all other explicit API
results and device checks remain enabled. A parent control with this exact
suppression reports **2 passed, 6 deselected, 0 errors (3.37 s)**.
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` is used only in this
instrumentation lane. Original logs, the failed narrower stack-filter
attempt, XML and the final suppression are retained in the experiment
artifacts. No CUDA library, driver or production dependency is changed.

```xml
<ComputeSanitizerOutput>
  <record>
    <kind>Api</kind>
    <what><api>cuGetProcAddress_v2</api><result>1</result></what>
  </record>
</ComputeSanitizerOutput>
```

The invocation is `compute-sanitizer --tool <memcheck|synccheck|racecheck>
--report-api-errors explicit --suppressions <file.xml> --error-exitcode 99`,
followed by `python -m pytest -q -x -rs
linear_attention/frost/test_kda_state_reuse.py -k summary_alias`.
The API-reporting modes and XML suppression semantics follow the
[Compute Sanitizer documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#error-suppression).
Racecheck detects shared-memory hazards; it does not prove global alias
ownership. H/X safety additionally relies on the documented single-lane
ownership, same-block load-before-store PTX and exact-view contract.

The negative control runs the new alias test from an isolated parent source
and test tree, verifying the loaded state-chain path and SHA256 before each
case. With caches disabled, the unchanged parent fails the first case
(`1-False-64-64`): **177,678/196,608 tail bytes differ**, pytest exit 1,
**1 failed, 1 deselected in 1.57 s**, stopped at first failure. Saved PTX
shows the parent's store before its load, while the candidate PTX has the
single load-before-store block. This validates the detector for the newly
supported alias contract; it is not a failure of the parent's supported
separate-storage contract.

Earlier attempts launched parent/mutant controls from the candidate test tree;
its conftest overlaid candidate kernel sources despite the parent top-level
`cudnn.__file__`. Those controls are invalid and excluded. Native candidate
suites and parent API controls use matching source/test trees. The CP4
benchmark is a standalone harness with per-module hashes, unaffected by this
pytest overlay. The corrected control's wrapper also expected a `state`
mismatch specifically, so it exits 1 on the actual `tail` mismatch; the saved
pytest traceback and source check, not that wrapper status, establish the
expected negative result.

The initial pilot used an older cuLA snapshot and failed the contiguous-summary
contract before producing a measurement. Its log is retained and excluded.
An initial extension import lacked its runtime library path; subsequent
imports and all reported tests use the corrected launch environment. No
partial or failed run is counted as a pass.

Changed-file pre-commit and Python compilation checks pass. All four
production/test hashes match the measured V3 snapshot. Fern reports zero
errors and one warning for the skipped unauthenticated redirects check.

Other GPUs, independent-stream concurrency for this new alias, safe-gate
CP4 1M performance, and full-model/CSA measurements are unverified here.
Existing earlier-architecture evidence is retained, not represented as a
rerun of this change.

### PR integration and strict review — B200, 2026-11-15

Integration commit `2be2c82d31eb33fa611a867e095a01699b03adc9` merges upstream
`develop` through `b24979cd4640385b7b6791664c975388ac167250`. All linear-attention
implementation and test files remain byte-identical to `edeea242` after the
merge. The native extension was rebuilt from the merged tree. The subsequent
review changes three internal docstrings and adds zero-initial-state alias
coverage; stripping docstrings gives identical production Python ASTs. The
performance tables above are retained measurements, not post-merge reruns.

The full production diff against current upstream was reviewed, including
compact/full-head indices, rectangular state arenas, empty/one-piece paths,
summary coverage of every multi-piece slot, caller context/checkpoint ownership,
all queue writers and counters, phase-arena lifetimes, PDL prerequisites,
TMA/TMEM completion, reduction order and compile-key bounds. No new confirmed
race, bounds or numerical defect was found. `run_state_chain`,
`build_chain_backward` and `CompiledKdaBwd.run_chain` now document destructive
exact H/X reuse, compact versus public head layouts, and independent writable
storage for concurrent streams.

The alias load/store executes in one side-effectful inline PTX block with
named temporaries. The CuTeDSL 4.8.0 and 4.8.1 wrappers were independently read
from their released wheels and have identical `inline_ptx` implementations;
the NVVM op documents side-effectful lowering. Same-lane ownership handles the
H/X edge, blocking CTA barriers publish and retire queue shared counts, and
`griddepcontrol.wait` handles inter-kernel consumers and next-wave reuse. The
[PTX dependency instruction](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-griddepcontrol)
orders prerequisite completion and memory visibility. Existing numerical
kernel async-proxy/TMA/TMEM protocols are unchanged; no speculative global
fence was added. Earlier sanitizer and generated-code evidence above is
retained with its stated qualifications, not counted as new runs.

Merged-source native tests use Torch 2.14.1+cu130, cuDNN 9.24.0 and the same
B200 environment as the CP4 experiment. Each compiler lane uses its own empty
compiler/artifact caches and separate process; loaded nested module paths and
SHA256 values are verified, not just the top-level package path.

| Selection | DSL / actual compiler | Result |
|---|---|---|
| State alias, forward waves, boundaries, queues, forward warm/poison stress | 4.8.1 / CUDA 13.4 | 44 passed, 903.64 s |
| Backward bitwise/capture, policy, cache, H96 runtime and independent streams | 4.8.1 / CUDA 13.4 | 104 passed, 7 deselected, 1691.33 s |
| Alias, default FP64-reference chains, forward waves, sixteen-wave backward and policies | 4.8.0 / CUDA 12.9 | 35 passed, 118 deselected, 533.68 s |
| Same selection | 4.8.0 / CUDA 13.4 | 35 passed, 118 deselected, 535.16 s |
| Same selection | 4.8.1 / CUDA 12.9 | 35 passed, 118 deselected, 522.68 s |
| Final alias file, including zero initial state | 4.8.0 / CUDA 12.9 | 33 passed, 18.75 s |
| Same final alias file | 4.8.0 / CUDA 13.4 | 33 passed, 19.55 s |
| Same final alias file | 4.8.1 / CUDA 12.9 | 33 passed, 19.11 s |
| Same final alias file | 4.8.1 / CUDA 13.4 | 33 passed, 19.12 s |

The final alias file contains 32 capture/byte-parity cases (four rectangular
or square K/V pairs, both directions, one/three pieces, caller seed or implicit
zero seed) plus one policy test. The explicit zero-seed cases exercise the
constant FP32 operand in the load-before-store PTX. Existing seeded case
behavior is unchanged. These selections overlap; their totals are not unique
coverage counts. No numerical tolerance or golden changed. The cu12 lanes
qualify that DSL compiler on the CUDA 13 framework, not a complete CUDA 12
framework/backend deployment. No new SM103/SM107 or Hopper result is claimed.

From `test/python`, the main native selections are:

```bash
python -m pytest -q -x -rs linear_attention/frost/test_kda_state_reuse.py \
  linear_attention/frost/test_kda_forward_waves.py \
  linear_attention/frost/test_kda_wave_boundaries.py \
  linear_attention/frost/test_kda_wave_queues.py
python -m pytest -q -x -rs linear_attention/frost/test_kda_checkpoint_waves.py \
  -k '16 or capacity_policy or compile_count or independent_streams'
```

The second selector also matches `bf16`/`fp16` IDs, so it exercises the full
backward dtype/head-pattern parity matrix across divisors 2/4/8/16, not just
sixteen waves. Extra compiler lanes select:

```text
summary_alias or arena_policy or capacity_policy or default_state_chain
or (forward_waves_bitwise_and_capture and 8)
or (checkpoint_waves_bitwise_and_capture and equal-16)
```

The `8` selection also includes forward rectangular `k128`/`v128` IDs at other
divisors. The final zero-seed qualification runs the complete
`test_kda_state_reuse.py` in all four compiler lanes. Tests use matching source
and test trees, `CUDNN_TEST_NO_ISOLATION=1` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`; the exact loaded compiler paths and source
hashes are retained in the qualification manifests.

The fresh forward stress again reports 0/3,000 warm and 0/3,000 poisoned
failures, with the unsafe control failing 200/200. The target still has only
two CTAs, so this is limited scheduling coverage. Compatibility, docstring,
compile-cache, CUDA Graph and fence audits pass for the exercised scope.
Changed-file pre-commit/syntax checks pass. Fern reports zero errors and one
warning (unauthenticated redirect validation); the repository's pinned CLI
was used after checking that no project-local executable override was present.

This refresh updates the existing Draft PR. The original-baseline >5%
performance exceptions above remain open; the CP4 incremental comparison
against `a1de8974` does not resolve them. Old upstream CI on `a19c476` does
not validate the refreshed head.

## SM100 D128 per-tensor FP8 decode — qualification, 2026-11-12 (UTC)


Review follow-up on `6536a9d4226ced931c8c41886e72c6c2ac9cd5cc`, B200 with
that same runtime: score-slot credit now consumes all eight TMEM score words
through the release-zero dependency; the existing wait/fence and arrival counts
stay in place. Capture cleanup covers errors, and sync-debug mode wraps execute
rather than PyTorch capture setup. Four production PTX renderings pin the full
eight-word dependency. The detector's full core suite passes 29 checks.

Jobs 5887863 and 5887949 together pass 30 native functional cases, eight existing
dual-stream cases, four PTX checks, four replay/batch cases, two warm/poisoned
stress cases and six strict **unchanged** pre-review goldens. All three sanitizer
tools pass four cases each with zero errors/hazards, using the same unfiltered
commands below. CUDA API error reporting remains disabled. No numerical
threshold or golden changed for the synchronization fix. New runtime evidence
is SM100-only; the unchanged SM103 route keeps its prior qualification.

The first runner passed 29 core and 38 functional checks but four PTX checks
could not create its nested pytest temporary directory. The second runner
creates the missing parent and runs only those four checks plus the remaining
phases; source bytes are identical. This setup failure remains separate from
the passing results. Its run root is
`/tmp/yanxu-fp8-native-review-b200-v2-5887949`; strict/sanitizer phases set
`CUDNN_DETERMINISM_GOLDENS=<run_root>/source/test/python/determinism_goldens`
and leave recording unset. The earlier source/record history below is retained.

`TestNativeFp8Decode` and `TestNativeFp8DecodeDeterminism` in
`sdpa/frost/test_sdpa_fwd_fp8_sm100.py` exercise the native five-stage KV ring
and eight-live-column swap-AB lowering through the actual architecture CI
entry. It serves dense Q with paged KV, CGA1, per-tensor E4M3/E5M2 and at most
eight packed query rows. FP16/BF16 and FP8 O, optional Amax/Stats, final log2
Stats, independent K/V maps, padded head dimensions, and strided Stats are
covered. There is no separate determinism switch and no new backward path.

The four replay/batch cases fix packing and split count, quantize each payload
once, and preserve scales across isolation, insertion, reordered/duplicated
sequences and B513. Fresh allocations, concurrent GPU work, retained-graph V
sign changes and exact restoration are checked. O/Stats are byte invariant;
whole-batch Amax is checked numerically after rebatching and bitwise on replay.
CPU FP64 numerical thresholds are unchanged: O 0.04 (or three times the FP8
quantization floor), Amax 0.03, Stats 0.0006. Padding canaries and input bytes
are checked independently.

B200 (SM100, 148 SMs), CUDA compiler and loaded NVRTC 13.4, CuTeDSL 4.8,
cuDNN 9.30.0.4, Torch 2.14.1+cu130:

- 30 functional cases pass across the incremental runs: 10 graph cases each
  make six reference/replay checks, plus feature, route, config, optional-output
  and direct-template Stats-base cases. No functional skips.
- Four new replay/batch cases and both affected older D128/CGA1 replay/batch
  cases pass. Two warm-refill cases each use 592 CTAs, five KV tiles and five
  ring refills; 256 warm and 256 poisoned launches have zero mismatches. The
  unsafe-ring control fails 200/200. The warm harness uses actual compact
  output allocations; separate replay tests retain padded/strided outputs.
- Four new golden records and two updated old D128/CGA1 records pass in a fresh
  strict process. The two old records retain identical input/runtime hashes;
  O, Stats and Amax hashes change with the new swap-AB reduction and FP8 P
  conversion. Independent numerical and batch checks pass before these updates.
  No other existing golden is changed.
- Unfiltered CUDA 13.4 memcheck, synccheck and racecheck each pass all four new
  goldens. Memcheck/synccheck report zero errors; racecheck reports zero hazards,
  errors and warnings. CUDA API error reporting is disabled.

Final architecture routing checks pass on both B200 and B300. Each also passes
the two affected D128/CGA1 strict goldens: updated SM100 records and unchanged
SM103 parent records. This verifies the final SM100-only selection after the
candidate's two-GPU qualification.

The earlier warm-stress run stopped at the harness's contiguous-output
precondition after all four replay/batch cases passed. The correction allocates
compact output storage before prepare; it does not copy a kernel result or
change the kernel. A separate old-template control with log2 split-partial
Stats fails both E4M3/E5M2 numerical checks, and the corrected natural-log
partial guard passes both. These expected RED failures are separate from the
passing qualification counts.

SM103 candidate qualification also passed the 30 functional cases, four new
replay/batch cases, two affected older replay/batch cases, both refill stress
cases and six fresh-process goldens. All six candidate input/output hashes
match SM100. However, its measured performance did not justify replacing the
existing SM103 schedule, so the final route selects this body only on SM100.
The SM103 candidate goldens are not committed; SM103's existing records stay
unchanged. No new SM103 sanitizer result is claimed.

Source: base `5c99b2c8b2e05b6b3f414cf9fbeb782954faa3cd` with the native FP8
changes in this PR. The frozen source/library manifests verify the actual
import root and native-extension inputs. The B200 V2 qualification ran from
`/tmp/yanxu-fp8-native-qual-b200-v2-5886831`. Final edits limit FP8 selection to
SM100 and correct comments; the qualified kernel's executable AST is unchanged.
The previous functional/batch checks used identical routed kernel arithmetic;
the only earlier AST correction guards final log2 Stats with `SPLIT_KV == 1`
(the normal adapter already cleared that flag on split partials).

Run from `test/python`, with `CUDNN_DETERMINISM_RECORD` unset:

```bash
python -m pytest sdpa/frost/test_sdpa_fwd_fp8_sm100.py \
  -k 'TestNativeFp8Decode or test_native_fp8_decode_architecture_route' -m L0 -q -rs
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" python -m pytest \
  sdpa/frost/test_sdpa_fp8_d512_paged_determinism.py \
  -k 'd128 and cga1' -m L0 -q -rs
```

The sanitizer commands below use the CUDA 13.4 executable, qualification
Python, and frozen run root named above. There is no kernel-name or launch
filter. Each tool runs a separate process:

```bash
"$sanitizer134" --tool memcheck --error-exitcode 91 --report-api-errors no \
  "$python312" -u "$run_root/probe.py" memcheck
"$sanitizer134" --tool synccheck --error-exitcode 91 --report-api-errors no \
  "$python312" -u "$run_root/probe.py" synccheck
"$sanitizer134" --tool racecheck --error-exitcode 91 --report-api-errors no \
  "$python312" -u "$run_root/probe.py" racecheck
```

The wrapper verifies source/library hashes and runtime, then calls `pytest.main`
with a result-recording plugin and these arguments:

```text
sdpa/frost/test_sdpa_fwd_fp8_sm100.py::TestNativeFp8DecodeDeterminism::test_golden
-o addopts= -m L0 -q -rs --tb=short --junitxml=<run_root>/results/<tool>/pytest.xml
```

`CUDNN_DETERMINISM_RECORD` is unset; `CUDNN_DETERMINISM_GOLDENS=<run_root>/goldens`
selects the preceding record phase. The environment sets
`CUDNN_TEST_NO_ISOLATION=1`, `CUDNN_TEST_TIMEOUT=240`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1`,
`CUDNN_FRONTEND_DISABLE_COMPILED_CACHE=1`, preloads CUDA 13.4 NVRTC, and uses
the cuDNN 9.30.0.4 library directory with isolated source/cache paths.

### SM100 / SM103 D128 MXFP8 backward — qualification, 2026-11-13 (UTC)

`sdpa/frost/test_sdpa_bwd_mxfp8_sm100_d128.py` covers the cc 10.0 / 10.3 CuTeDSL
chain (prep, dK/dV, persistent dQ) behind `sdpa_mxfp8_backward`. The only atomics
are the dQ kernel's work counter and the amax maximum. The counter decides which
cluster computes a unit, not the order the unit sums in, and a maximum is exact in
any order. Every plan fixes the dK/dV walk split, so repeated calls are bitwise
identical. Under `use_deterministic_algorithm` the walk
split is decided from one batch element, so results are also batch invariant.
Without that request a large batch may keep the walk unsplit: a different order,
still repeatable ([status page](../../../docs/operations/determinism.md)).

The checks:
- Repeats into NaN-poisoned outputs, eager and CUDA-graph replay, compare raw
  bytes (dense, tail, causal, and causal + tail + amax).
- Batch invariance: B = 2 against two B = 1 problems on the same data, and, where
  the default plan differs (dense, tail, causal), the deterministic B = 8 plan.
- Concurrent load and fresh allocations; a warm execute that compiles, allocates
  and synchronizes nothing; a first build inside a capture.
- A warm stress of the dQ kernel's `ld.shared`-released stages.
- An L1 case past 2^32-byte offsets: the last batch element of B = 34 is bitwise
  the B = 1 problem on its data.
- Accuracy against the repo's MXFP8 reference, and error against FP32 matched to
  the cuDNN backend's.

`test_golden` holds one golden per distinct numerical path of the chain:
- the walk-split dK/dV plan (B = 1, S = 1024, deterministic request);
- the unsplit plan (B = 8, default request);
- causal with an odd block count (S = 1152);
- a sequence tail with amax (S = 1000, all three amax outputs).

Each case passes the MXFP8 reference check and three replays before its bytes are
hashed. The four records were written into a new directory under
`CUDNN_DETERMINISM_RECORD` and passed a strict `CUDNN_DETERMINISM_GOLDENS` check in a
fresh process (4 passed) before they were copied to `76371bd3c67774a8/`. That runtime
is B300, CUDA driver API 13.4, Torch 2.11.0+cu130, cuDNN 9.28, CuTe DSL 4.8.1 and
TVM FFI 0.1.15.post2. Other runtimes skip the comparison until a reviewed baseline is
recorded for them.

Qualification on 2026-11-14: B300 SXM6 AC (148 SMs), driver 615.71.10, Torch
2.11.0+cu130, cuDNN 9.28.0, CuTe DSL 4.8.1 (compiling with CUDA 13.4). The row
module passes **55 tests** at L0 + L1 with zero skips, the four goldens included. The
GPU-free eligibility module passes **42**. The warm stress has 0 mismatches in 3000
warm and 3000 poisoned launches; its unsafe control fails 200 / 200 on the same GPU.
From `test/python`:

```bash
CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1 CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest -q -m "L0 or L1" \
  sdpa/frost/test_sdpa_bwd_mxfp8_sm100_d128.py sdpa/frost/test_sdpa_bwd_mxfp8_sm100_d128_eligibility.py
```

On 2026-11-16, after the row was extended to cc 10.0, the same B300 runtime passed the row module (55, the
goldens included), the eligibility module (43, with the new cc 10.0 admission case) and `test_band_capabilities.py`
(26) with zero skips.

B200 qualification on 2026-11-16 (UTC): B200 (148 SMs), driver 615.71.10, otherwise the
runtime above. The four cells were recorded into a new directory under
`CUDNN_DETERMINISM_RECORD`, passed a strict `CUDNN_DETERMINISM_GOLDENS` check in a fresh
process, and were copied to `ea37b1c6cd40d408/`. Their input and output hashes equal the
B300 records: the `sm_100a` and `sm_103a` builds write the same bytes for these cells. The
command above passes the row module (55 tests, the goldens included) and the eligibility
module (43) with zero skips. The warm stress has 0 mismatches in 3000 warm and 3000
poisoned launches; its unsafe control fails 200 / 200.

Compute Sanitizer has not run on this row: the 12.9 sanitizer (2025.2.1) reports the B300 as
an unsupported device, and a CUDA 13 build was not available on the test node.

### D128 paged live-column decode — qualification, 2026-11-12 (UTC)

`sdpa/frost/test_sdpa_fwd_decode_live_columns_determinism.py` covers FP16/BF16
O and FP32 Stats for the eight-live-row swap-AB lowering, with fixed packing
and split counts. Cases include Q1/G8, Q2/G4, split1/4 and D80/V120 inside
the D128 envelope. Replay uses fresh buffers and concurrent work; batch
changes include isolation, insertion, reordering, duplication and more than
one device wave. Retained graphs change V, query/KV lengths and independent
K/V page maps, then restore and compare bytes. Numerical checks use CPU FP64.
The refill stress uses four blocks per SM, warm/poisoned runs and the unsafe
ring control. Backward uses other kernels and is outside this suite.

B200 native qualification (CUDA compiler/NVRTC 13.4, CuTeDSL 4.8,
cuDNN 9.30.0.4, Torch 2.14.1+cu130): four replay/batch cases and two warm
stress cases pass. Four new golden records were created after the numerical
and replay checks, then all four pass in a separate strict process. Existing
goldens are unchanged; the new records baseline the swap-AB reduction order.
The stress uses 592 blocks (4 per SM), 256 warm and 256 poisoned launches per
dtype, with no mismatches; the unsafe control fails 200/200. No skipped cases.

Unfiltered Compute Sanitizer memcheck, synccheck and racecheck each pass the
four golden cases. Memcheck/synccheck report 0 errors; racecheck reports 0
hazards, errors or warnings. CUDA API error reporting is disabled. Under
instrumentation FE reports cuDNN 93000 and Torch 92400; the explicit standalone
FROST path executes no backend attention. No new SM103 sanitizer run is claimed.
The separate public graph suite passes 28 cases, including legacy D256
wide-tile controls. The source/import and runtime library hashes are verified.

The B200 run used base revision `a736a06c6d8dfde1b4322eee0430f93bae78178c`
with the frozen D128 overlays subsequently committed in `fd911361d886b60a083464b6b3e790ba302c7733`.
The three production modules have identical executable ASTs to PR revision
`73bcd50c6166a574fab360136a76c90f6f3ea012`; the determinism test is byte-identical.
The base revision alone does not contain this lowering. The frozen source
manifest also verifies 134 native-extension build inputs.

The actual sanitizer invocations below ran in separate processes. Here
`sanitizer134` denotes the installed CUDA 13.4 `compute-sanitizer` executable,
`python312` the qualification environment's Python, and `run_root` the frozen
run directory (`/tmp/yanxu-d128-determinism-5882703`). No kernel-name filter,
launch limit, or launch-dump option was used:

```bash
"$sanitizer134" --tool memcheck --error-exitcode 91 --report-api-errors no \
  "$python312" -u "$run_root/probe.py" memcheck
"$sanitizer134" --tool synccheck --error-exitcode 91 --report-api-errors no \
  "$python312" -u "$run_root/probe.py" synccheck
"$sanitizer134" --tool racecheck --error-exitcode 91 --report-api-errors no \
  "$python312" -u "$run_root/probe.py" racecheck
```

The wrapper verifies source/library hashes, the import path, GPU and runtime,
then calls `pytest.main` from the frozen `source/test/python` directory with
the following arguments (plus a result-recording plugin):

```text
sdpa/frost/test_sdpa_fwd_decode_live_columns_determinism.py::test_live_columns_golden
-m L0 -q -rs --tb=short --junitxml=<run_root>/results/<tool>/pytest.xml
```

For all three tools, `CUDNN_DETERMINISM_RECORD` was unset and
`CUDNN_DETERMINISM_GOLDENS=<run_root>/goldens` selected the four records from
the preceding record phase. The wrapper used `CUDNN_TEST_NO_ISOLATION=1`,
`CUDNN_TEST_TIMEOUT=240`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
`CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1` and
`CUDNN_FRONTEND_DISABLE_COMPILED_CACHE=1`, with CUDA 13.4 NVRTC preloaded,
the cuDNN 9.30.0.4 library directory, and isolated source/cache paths.

Native B300 SXM6 AC (cc10.3, 148 SMs, same compiler/runtime versions) passes 40
checks through `TestDecodeLiveColumns` in the actual SM100/SM103 CI entry,
including the older sink, routing, wide-tile and Int64 page-table-stride cases.
Its four new golden records pass a separate strict process. Both dtype stress
cases report 0 mismatches in 256 warm and 256 poisoned launches at 592 CTAs; the
unsafe-ring control fails 200/200. No skips. The B300 job's subsequent
performance phase failed during FlashInfer import before any measurements;
that harness failure does not change these completed qualification results.

Run from `test/python` with `CUDNN_DETERMINISM_RECORD` unset:

```bash
python -m pytest -m L0 -q sdpa/frost/test_sdpa_fwd_decode_live_columns_determinism.py
python -m pytest -m L0 -q sdpa/frost/test_sdpa_fwd_decode_live_columns_sm100.py
```

### SM90 dense D64 / D128 backward chain — validation, 2026-11-12 (UTC)

`sdpa/frost/test_sdpa_bwd_sm90.py` covers the opt-in `sdpa_bwd_sm90` row
(`pygraph.sdpa_backward`; dense compact-BSHD BF16 / FP16 at D64 and D128;
MHA, GQA and MQA; unmasked or top-left causal). The FP64 reference gate runs
eleven shape cells per dtype, including tail lengths and a one-query graph.
Repeatability: repeated executions with a poisoned workspace are compared as
bytes on two multi-wave grids (256 and 144 CTAs, beyond one wave of the 132
SMs), and identical-input CUDA Graph replays with a poisoned workspace are
compared as bytes for causal GQA at D128 under both the deterministic and the
default request; a replay with a changed dO is then checked against a fresh
FP64 reference. Batch invariance: reordered, duplicated and uniform batches,
and the sequence declared alone through a one-batch graph, are compared as
bytes in four cells: BF16 D128 causal GQA, FP16 D64 unmasked MHA, BF16 D64
causal MHA and FP16 D128 unmasked GQA. Adjacent cases pin the
three launches without allocation or synchronization, the declines by name
and the batch / query-head grid limits. No runtime golden is recorded: the
FP64 reference is the numerical arm and the byte comparisons are the
repeatability and batch-invariance arms. SM100/103 and SM107 are outside this
row's range.

Measured at source revision 8c1ddeb66 on an SM90 validation host (cc 9.0,
132 SMs) with Torch 2.12.1+cu130, CUDA 13.0 and `nvidia-cutlass-dsl` 4.8.0:
the file passed 43 collected cases with no skips in 46.4 s from a cold
compiled-plan cache (34 compiled entries). At revision a78f6ce4 (identical
kernel and chain),
compute-sanitizer racecheck on `_case(torch.bfloat16, 64, True, 1, 3, 1, 65, 257)`
and `_case(torch.float16, 128, True, 1, 3, 1, 65, 257)` reported zero hazards;
every shared-memory writer and every ring reader arrives its mbarrier directly,
where the earlier elected-lane publication reported 20 hazard sites per case.
The commands, run from `test/python/sdpa/frost`:

```bash
CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1 pytest -q -rA test_sdpa_bwd_sm90.py
CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1 compute-sanitizer --tool racecheck --kernel-name kns=bprop_sm90 \
  python -c "import torch; from test_sdpa_bwd_sm90 import _case; _case(torch.bfloat16, 64, True, 1, 3, 1, 65, 257)"
CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1 compute-sanitizer --tool racecheck --kernel-name kns=bprop_sm90 \
  python -c "import torch; from test_sdpa_bwd_sm90 import _case; _case(torch.float16, 128, True, 1, 3, 1, 65, 257)"
```

### SM90 bottom-right causal backward — validation, 2026-11-18 (UTC)

On GH200 (cc 9.0), the opt-in `sdpa_bwd_sm90` chain passed the 57-case suite
with cuDNN 9.27.0, PyTorch 2.14.0+cu130 and CuTe DSL 4.8.1. The tested
kernel source has Git blob `f4a85ee168ee`. Run from `test/python` with
`CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m pytest -q sdpa/frost/test_sdpa_bwd_sm90.py`.

The FP64 gradient checks pin this engine at both FP16 and BF16. Cases include
a square control, offsets 63 and 65 on opposite sides of the 64-row query
tile boundary, and one query attending to 513 keys. The same rectangular
geometry also runs with top-left, bottom-right and unmasked graphs in one
process. The 144-CTA rectangular GQA case checks repeated gradients bytewise.
D128 GQA covers poisoned workspace, capture, replay and a changed upstream
gradient under both request modes. FP16 D64 and BF16 D128 GQA compare dQ, dK
and dV bytes across reordered and duplicated batches and the sequence alone.

### Native SDPA service register release — validation, 2026-11-12 (UTC)

Twenty-six SM100/SM107 forward templates now release the unchanged service
register budget at one common warpgroup instruction before role dispatch.
`TestServiceRegisterRelease` in the SM100 DSL and SM107 FP8 CI entry files
compiles every changed template and checks the actual PTX, including both
entries of D512 MXFP8 and the separate correction-group release. Three old
template controls fail this detector; all 26 candidate templates pass without
skips. SM107 results in this compile run are cross-compilation only.

B200 (SM100, 148 SMs), Torch 2.14.1+cu130, cuDNN 9.30.0.4 and CuTe DSL 4.8.0
using actual CUDA 13.4 and NVRTC 13.4: 127 native cases pass. They cover 72
dense half/FP8/MXFP8 physical-stride permutations across D128/D192/D256/D512,
24 half storage/rebind cases, four D192 cases, five D512 half datapath cases,
one FP8 D512 multi-tile case, two MXFP8 D512 multi-wave cases, and 19 existing
paged FP8 determinism cases. The latter include nine strict unchanged-parent
goldens, nine repeatability/batch-invariance cases and one warm case. No
tracked golden or numerical tolerance changes. The warm case has only four
CTAs and is not a several-waves stress qualification.

Racecheck, memcheck and synccheck each pass 12 dense stride/replay cases
(three input families and four widths), with zero errors or hazards. Only
racecheck filters instrumentation to `kns=cudnn_kernel_`; memcheck and
synccheck are unfiltered. CUDA API error reporting is disabled, so these
results qualify the selected memory/synchronization tools only. Earlier
private harness failures are retained: simultaneous golden record/check
settings, counting two L1 cases before pytest marker deselection, and a
rejected sanitizer filter argument. None changed production bytes or relaxed
an assertion; completed numerical cases were retained, and only missing
phases were rerun. Native SM103 and performance are not measured here.
This forward-only change makes no new backward determinism claim.

Native SM107, Torch 2.14.0+cu130, cuDNN 9.30.0.4 and CuTe DSL 4.8.0 with
CUDA 13.4/NVRTC 13.4: 104 candidate cases pass without skips or failures.
They reuse the 72 dense stride permutations, 12 split1 half storage/rebind
cases, four D512 FP8 strict goldens and four repeatability/batch-invariance
cases, eight D576/V512 half nonpaged cases and four half determinism/golden
cases. Eight parent goldens are recorded in a separate process and checked
without recording by the candidate. The common half fixture's twelve
pre-existing SM107 split exclusions are deselected before execution; this
run does not qualify those split configurations. Native SM107 sanitizers
remain unmeasured. The initial private runner used a CPython 3.12 extension
in a 3.10 container and failed before GPU tests; the corrected run uses a
matching extension whose native build inputs equal the frozen source.

From `test/python`, the generated-instruction detector is:

```bash
python -m pytest -m L0 sdpa/frost/test_sdpa_fwd_dsl_sm100.py::TestServiceRegisterRelease \
  sdpa/frost/test_sdpa_fp8_sm107.py::TestServiceRegisterRelease -q -rs
```

The existing strict paged suite uses the normal golden workflow below;
record its nine golden cases from the unchanged parent into a separate
directory, then run the candidate with recording disabled. Dense sanitizer
selection is `test_sdpa_fwd_dsl_sm100.py::TestNativeDenseStrides -m L0 -k order0`.
The additional multi-wave selection is
`test_sdpa_fwd_mxfp8_sm100.py::test_mxfp8_d512_dense_multi_wave_alias -m L1`.
The frozen validation base is `7c6ea67c633a8509ba976cbf3b0bb0c32473b606`;
source and library hashes and the actual import path were checked.

### SM120 cooperative paged prefill — validation, 2026-11-12 (UTC)

The paged half-MMA prefill loader partitions the same vector copies across
all four existing service warps. Half K/V retains per-thread async-copy
completion before publication; FP8 K/V retains ordinary release publication
after widening stores. Full-barrier arrivals and named-barrier participation
include all 128 producer threads. CTA size, register budgets, shared-memory
layout and attention arithmetic are unchanged. Dense, decode and wide-head
producer counts retain their existing values.

The initial candidate on parent `b2c5270751eeab2213d85eb3103e0ef507937901`
passed 78 existing cases with zero failures/skips, including eight unchanged
strict parent O/Stats goldens, explicit-plan rebind/capture, reordered and
duplicated batches, changed neighbors, width/mixed-KV controls and half refill
stress. Racecheck, memcheck and synccheck each passed three selected cases
with zero errors: one affected small-page half prefill case and two unchanged
decode-ring controls. These runs used Compute Sanitizer 2026.4 (CUDA 13.5),
with API error reporting disabled; the zero-error result applies to the
selected memory/synchronization tools.

`test_dsl_sm120_paged_vector_loop_warm` additionally covers FP16/E4M3 and
BF16/E5M2 K/V, retaining both half cases. The M128/N128 shape has 512 live
CTAs plus 512 empty-sequence controls, rather than 1,024 active CTAs. Each
live CTA reads three KV tiles and refills twice. Independent numerical checks
precede byte stress. The initial half checks pass 256 warm and 256 poisoned
launches per dtype, with zero differing output-storage bytes; the unsafe
control fails 200/200 launches. This corrects the interpretation of the
total launch grid in the earlier records without changing those measurements.

The supplemental run passes all four half/FP8 warm cases with the same
256 warm and 256 poisoned launches and the 200/200 unsafe control, plus the
two FP8 small-page strict-golden cases under each sanitizer, with zero
errors. Together the runs cover 80 distinct cases and five cases per
sanitizer; the repeated half controls are not counted as new coverage.
An initial supplemental allocation exited during GPU inventory before
testing. The successful replacement preserves the one-GPU/architecture/idle
checks. Generated PTX confirms that service warps 8 through 11 reach one
common register-release instruction, with 128 full-barrier arrivals and
384 participants in the tested M128 tile's K/V reuse barriers.

Measured on RTX PRO 6000 Blackwell Server Edition, 188 SMs, Torch
2.14.1+cu130, cuDNN 9.30.0.4, CuTe DSL 4.8.0 with actual CUDA 13.4 and
NVRTC 13.4. Source/import locations and runtime library hashes were checked.
No golden, tolerance or skip changed. This is forward-only, fixed-runtime
qualification; SM121 and cross-runtime identity remain unverified.

From `test/python`, with recording disabled:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest -n0 -m L0 sdpa/frost/test_sdpa_fwd_dsl_sm120.py \
  -k 'small_page_cache_determinism or paged_vector_loop' -q -rs
```

### SM120 warp-cached page tables — validation, 2026-11-12 (UTC)

The common paged loader now loads a tile's live page IDs across a producer
warp and selects them with shuffles before the vector live predicate. Tiles
whose worst-case page span exceeds 32 retain the scalar lookup. The rolled
copy loop, payload operations, physical Int64 offsets, waits, arrivals and
arithmetic order are unchanged. This changes only forward kernels.

`test_dsl_sm120_small_page_cache_determinism` adds four L0 cases to the
selected architecture entry: FP16/BF16 Q/O with half K/V or E4M3/E5M2 K/V.
D128 M128/N128 packed plans use eight-token pages and KV lengths
`[257, 129, 0, 247, 1]`: sixteen-page full tiles, one-row tails, empty
neighbors and independent K/V page order. The existing helper checks CPU-built
fixed inputs, O/FP32 Stats bytes, numerical references, source/canary integrity,
fresh buffers, isolated/reordered/duplicated/changed-neighbor batches,
129-sequence scheduling stress and CUDA Graph replay.

The unchanged parent loader recorded eight goldens after reference/replay
checks; all eight passed strict comparison in a fresh candidate process.
Four existing records were byte-identical to the committed files, and only
four new `page8-blocks35` records are added under `c36345d9c3841d69`.
The targeted candidate selection passed **78 cases, zero skips/failures**:
the existing 14 vector-loop cases, four new small-page cases, eight native
rebind cases, 24 width cases, 20 mixed-KV cases, six partial-GQA decode cases
and two decode-ring cases. Memcheck and synccheck each passed three selected
page8/reused-ring cases with zero errors. Racecheck passed the page8 case
unfiltered and both ring cases with `--kernel-name kns=cudnn_kernel_`, with
zero hazards. The original unfiltered attempt exceeded its 32 GiB host-memory
limit; a 128 GiB retry passed page8 but hit the 180-second test watchdog in
the first ring case's Torch numerical comparison. Both ring cases then passed
with FROST-only instrumentation and unchanged numerical references (7.30s
and 4.81s). Those interrupted attempts are not counted as passing runs.
CUDA API error reporting was disabled for these sanitizer runs, so their
zero-error counts describe the selected memory/synchronization tools only.

The two warm cases retain 1,024 CTAs (5.45 per SM), two refills per launch,
256 warm and 256 shared-memory-poisoned launches per dtype with zero differing
output-storage bytes. Their unsafe control failed 200/200 launches. Each
256-launch block bounds per-launch failure below 1.16% at 95% confidence;
this is not the helper's 3,000-launch 0.1% bound.

Measured on RTX PRO 6000 Blackwell Server Edition, 188 SMs, Torch
2.14.1+cu130, cuDNN 9.30.0.4 and CuTe DSL 4.8.0 compiled with CUDA 13.4;
NVRTC 13.4 was loaded explicitly. Actual imports, source and library hashes
were checked. The frozen parent SM120 runtime source matches develop
`9be74f4d7f748ebbf14febe6a9c624f61e9355c4`. SM121 is unverified; this makes
no cross-runtime or backward determinism claim.

From `test/python`, with recording disabled:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest -m L0 sdpa/frost/test_sdpa_fwd_dsl_sm120.py \
  -k 'small_page_cache_determinism or paged_vector_loop' -q -rs
```

### SM120 paged vector-copy loop — validation, 2026-11-12 (UTC)

The selected architecture entry `sdpa/frost/test_sdpa_fwd_dsl_sm120.py`
adds 14 L0 cases under `paged_vector_loop`: eight explicit-plan rebind/capture
cases for half D128 prefill, D512, D128 decode and half-Q/FP8-KV prefill; four
D128 M128/N128 repeatability/batch-invariance cases for FP16/BF16 with packed
or unpacked GQA; and two sustained refill checks. The production change rolls
the page-copy vector loop without changing its register budget, synchronization
or arithmetic. It changes no backward path.

The four byte cases preserve O and FP32 Stats across replay, fresh buffers,
isolated/reordered/duplicated batches, 129-sequence scheduling stress and
changed neighbors under capture. Inputs are built on CPU and reused across
batch variants; numerical reference and canary checks are separate. Four
new runtime records are committed under `c36345d9c3841d69`; no existing golden
is replaced. Recording followed successful reference and byte checks, then a
fresh process passed all four with recording disabled. Parent/candidate A/B
measurements separately preserve identical FROST output bytes on all three
measured BF16 shapes; no numerical change is intended.

RTX PRO 6000 Blackwell Server Edition (SM120, 188 SMs), Torch 2.14.1+cu130,
cuDNN 9.30.0.4, CuTe DSL 4.8.0 using CUDA 13.4 and NVRTC 13.4:
**14 passed, 364 deselected, one warning in 72.47s**; fresh strict goldens:
**4 passed, 374 deselected, one warning in 24.26s**. The warning is Torch's
prototype synchronization-debug mode. The frozen source derives from
`b1034fa93283627d0619685bf7ef93fdaa0d09cd`; source/test hashes and the actual
import location were checked. SM121 is unverified. GPU-exclusive scheduling
markers were added after this serial qualification; test bodies are unchanged.

The warm cases use 1,024 CTAs (5.45 per SM), three KV tiles and two refills per
launch. Each dtype passed 256 warm and 256 shared-memory-poisoned launches
with zero differing output-storage bytes. The unsafe control failed 200/200
launches on this GPU. Each block bounds a per-launch failure rate below 1.16%
at 95% confidence; this is not the helper's 3,000-launch 0.1% bound. The stress
helper compares a cold byte reference, while an independent numerical check
runs before stress. This addition has no new sanitizer measurement.

From `test/python`, with recording disabled:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest -n0 -m L0 sdpa/frost/test_sdpa_fwd_dsl_sm120.py \
  -k paged_vector_loop -q -x
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest -n0 -m L0 sdpa/frost/test_sdpa_fwd_dsl_sm120.py \
  -k paged_vector_loop_determinism -q -x
```

### SM120 wide partial PackGQA — validation, 2026-11-11 (UTC)

The new L0 cases qualify explicit native forward plans for D512/V512 and
D576/V512 with GQA ratios 96 and 128, two KV heads, and FP16/BF16 queries.
There are 16 dense and 16 paged half cases across split counts 1 and 3,
plus 16 paged mixed-KV cases across E4M3/E5M2 storage at split count 3.
They preserve O and FP32 Stats bytes across repeat, isolated/reordered/
duplicated/changed-neighbor batches and CUDA Graph replay. The stress batch
has 33 sequences; numerical references, empty/tail/keyless rows, source
immutability and output canaries remain independent checks. KV quantization
is fixed across batch variants. This is forward-only coverage; no backward
API or arithmetic is changed.

Measured on RTX PRO 6000 Blackwell Server Edition (SM120, 188 SMs), driver
595.84.01, Torch 2.14.0+cu130, cuDNN backend 9.24.0.43, CuTe DSL 4.8.0 with
its CUDA 13.4 compiler, and the original image's cuda-bindings 13.0.3.
The frozen source derives from `3c6a485cd3b9`; all 2,547 source/binary files
per arm were hash-checked and the imported `cudnn` path was verified before
each run. The compatible extension's binding/build inputs are unchanged.
SM121 and other runtimes are unverified.

The immutable parent recorded 14 existing wide dense/THD/paged controls;
the candidate passed all 14 against those unchanged hashes. The 48 new
subgroup cases passed numerical/batch/replay checks before recording, then
all 48 passed a separate strict comparison with recording disabled. These
62 records are added under runtime fingerprint `812f98809d9324c5`; no existing
record is replaced. An additional 18 new functional cases and one admission
case pass. Memcheck, racecheck and synccheck each pass the same 24-case subset:
all 18 functional cases plus six representative D576 half/mixed subgroup
cases. Memcheck reports zero errors, racecheck zero errors/warnings, and
synccheck zero errors. Racecheck/synccheck filter to `cudnn_kernel_` symbols;
memcheck is unfiltered and CUDA API errors remain enabled. All allocations
were released.

From `test/python`, the strict new-golden selection is:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  pytest -v -m L0 sdpa/frost/test_sdpa_fwd_dsl_sm120.py \
  sdpa/frost/test_sdpa_fwd_d576_sm120.py sdpa/frost/test_sdpa_fwd_paged_sm120.py \
  -k wide_subgroup
```

The old controls select `test_d576_dense_thd_determinism` and
`test_d576_paged_determinism` in `test_sdpa_fwd_d576_sm120.py`. The functional
selection is `wide_partial_pack or pack_gqa_admits_any_ratio_up_to_the_tile`.
The sanitizer selection on the same three files is
`wide_partial_pack or (wide_subgroup and 3-96-dims1-dtype1) or (wide_subgroup_mixed and 96-dims1)`.
Host graph-analysis checks pass 629 cases, and four variants compile offline;
neither substitutes for the target results above.

Earlier attempts are retained as failures: the first runner stopped before
collection because xdist was absent; the second recorded 18 failures before
interruption (10 ordinary-generator division-by-zero errors, eight reference
cache permission errors) and 15 passes. After setting a writable Triton cache
and using explicit legal plans, the remaining 53 cases passed, followed by
the strict 48 and the three sanitizer passes. The ordinary candidate-generator
failure is a separate integration blocker; these explicit-plan results do
not claim it is fixed. Existing measurements below are retained, not rerun
as part of this addition.

### GLU + Hadamard + quant column-scale packing — SM100, 2026-11-11

The shared columnwise scale store now writes standard GEMM 128x4 scale atoms.
The existing reference decoder independently gathers those physical bytes
before checking scales and dequantized RHT values. A negative-control run of
the updated column-scale test against pristine `683f294b` failed: 3626 of 4096
scale elements mismatched, with maximum absolute error 93. The legacy decoder
was self-consistent with the old producer; this fixes its GEMM consumer contract.

Qualification used the edited `683f294b` checkout on NVIDIA GB200 (SM100,
152 SMs), driver 580.126.20, PyTorch 2.13.0a0+8145d630e8.nv26.06 / CUDA 13.3,
cuDNN backend 92600, and CuTe DSL 4.8.0 compiling with CUDA 13.4. Import-path
and source-hash checks confirmed the tested checkout. Results:

- Existing Torch quantization suite: **17 passed, 6 Rubin-only skips**, 17.24 s.
- New determinism suite: **5 passed**, 11.93 s. The default 32-channel interleaved
  input uses expert M=[256,0,512], N=256/2048, K=512. Separate independent
  GEMM/SwiGLU/H16/dequantization checks accompany raw-byte comparisons of every
  output across fresh allocations, concurrent work, changed-weight/scale graph
  replay, and expert isolation/reordering/duplication/removal/changed neighbors.
- JAX selection: **16 passed, 196 deselected**, 45.53 s, using isolated JAX 0.11.1
  and NumPy 2.2.6. Columnwise cases use an independent input dequantization and
  GEMM/SwiGLU/H16 reference as well as Torch byte parity and compiled replay.
  The fixture's RHT maximum 25984 exceeds the 10752 range at encode scale 0.25;
  columnwise RHT therefore uses 1/64 (range 172032), with finite/per-block scale
  representability assertions. Inputs and numerical tolerances are unchanged.

The warm-stress case uses M=[1024,0,2048], N=2048, K=4096, crossing the staged
input ring and reusing epilogue buffers. The unsafe control failed 200/200
launches. The actual kernel had **0 mismatches in 3000 warm and 3000 poisoned
launches**. These are cold-reference stability checks; the independent numerical
reference runs separately, not as an exact-byte stress reference. The kernel
launches 152 blocks (one per SM), below the control's three blocks per SM.
Each 3000-launch arm bounds the observed per-launch failure rate below 0.0998%
at 95% confidence for this tested configuration; it does not establish race
freedom or transfer the control's geometry to this kernel.

SM103, SM107, CUDA 12 compilation and JAX batch invariance remain unverified.
This is a forward-only fusion; no backward API is claimed. No runtime golden
or numerical tolerance was changed. From `test/python`, with the edited source
on the import path:

```bash
python -m pytest gemm/cutedsl/test_grouped_gemm_glu_hadamard_quant.py -m L0 -q -rs
python -m pytest gemm/cutedsl/test_grouped_gemm_glu_hadamard_quant_determinism.py -m L0 -q -rs -s
python -m pytest gemm/jax/test_grouped_gemm_canonical_jax.py -k glu_hadamard_quant -m L0 -q -rs
```

### Grouped wgrad empty inputs — validation, 2026-11-11

`gemm/cutedsl/test_grouped_gemm_wgrad_empty.py` covers NVFP4 E2M1/E4M3
and MXFP8 E4M3/E8M0 with BF16/FP32 outputs, dense/discrete output storage,
matrix/flat scale buffers, 1CTA/2CTA tiles, and overwrite/accumulate modes.
Each of its 64 cases starts with an empty wrapper cache. It checks exact-zero
overwrite or byte-preserving accumulation (including negative zero), reuses
the compiled artifact for nonempty and mixed-empty experts with compilation
forbidden, rebinds fresh empty inputs, and replays a CUDA Graph twice.
The independent reference uses exact unit operands/scales and per-expert token
counts; empty outputs are compared directly to zero or the original bytes.
No runtime-specific golden is required for these exact boundary identities.
This qualifies the empty-expert contract, not general nonempty determinism.

The cold NVFP4 case failed before the fix in
`from_dlpack(...).mark_compact_shape_dynamic(...)`. Empty FP4 samples retain
strides that do not describe a nonempty compact tensor, and Torch's empty
`dim_order()` may ignore those strides. Compiling their logical descriptors
with symbolic token extents avoids inferring a layout from the empty sample.
The kernel's empty accumulate path also skips the output reduction: adding
positive zero changed negative zero in the original output.

Measured on GB200 SM100 with Torch `2.13.0a0+8145d630e8.nv26.06`, framework
CUDA 13.3, cuDNN 92600, and CuTe DSL 4.8.0. Separate cold-cache CUDA 13.4
and CUDA 12.9 DSL compiler runs each passed **131 cases**, with **45 existing
skips** and **2 cases deselected** by the default selection. All 64 new cases passed in each lane.
The runs took 101.48 s and 99.28 s respectively. SM103 and SM107 are unverified.
From `test/python`, both compiler lanes ran:

```bash
python -m pytest -n0 -v --tb=short \
  gemm/cutedsl/test_grouped_gemm_wgrad_empty.py \
  gemm/cutedsl/test_grouped_gemm_wgrad.py \
  gemm/cutedsl/test_wgrad_tensormap_acquire.py
```

### Prepared SM90 Indexer Forward + Top-K plan — validation, 2026-11-10

`test_DSA_indexer_compressed_sm90_determinism.py` now covers the public
`DSA.prepare_indexer_forward_top_k` / `IndexerForwardTopKPlan.execute` path for
BF16 and per-token E4M3 FP8. The direct plan cases compare indices, logits,
selected-logit softmax and LSE as raw bytes across fixed-input replays with
queued side-stream work, and compare each preserved sequence alone, packed
with empty/ragged neighbors, reordered and duplicated. They also run the
existing independent FP32 reference. Exact ties, unsupported specializations,
zero-candidate rows and caller-buffer sentinels are covered by adjacent L0
cases. The CUDA Graph case mutates packed prefixes, causal offsets and
candidate offsets at stable addresses, poisons outputs and workspace, and
compares repeated replay bytes with an independently invoked eager wrapper.

The plan adds no new arithmetic body: it retains the already-qualified SM90
compact scorer and deterministic radix Top-K launchables. Therefore no new
runtime golden was recorded; the direct raw-byte equality to the existing
public wrapper and its independent reference is the numerical A/B arm. This
is forward-only API coverage; there is no prepared backward entry point.
SM100/103 and SM107 are rejected by the plan contract.

Measured on an SM90 validation host with Torch 2.12.1+cu130, CUDA 13.0,
cuDNN backend 92000 and `nvidia-cutlass-dsl[cu13]` 4.8.1 (CUDA 13.4 DSL
compiler). Imports resolved to the synchronized validation checkout. The
focused SM90 run passed all prepared-plan, combined-SM90 and API-export cases;
the post-rebase run passed 23 collected cases with 134 deselected and no
failures in 13.59 s. Three of those pytest cases capture CUDA Graphs. They
contain six explicit `graph.replay()` call sites and issue seven replay
invocations in the focused run because the Megatron-like helper replays both
its initial and changed payloads. Fixed-input replay loops inside the other
cases are kernel invocations, not additional pytest cases. On the separate
SM100 validation host, the complete existing
compressed-indexer L0 file passed 96 cases with 16 marker-deselected cases;
the architecture-admission/API-export focus passed 2 cases. These SM100
results are compatibility evidence for the shared stage-2 host refactor, not
support for the SM90-only plan. The final SM90 command, run from
`test/python`, was:

```bash
CUDNN_TEST_TIMEOUT=1800 CUBLAS_WORKSPACE_CONFIG=:4096:8 pytest -q -s \
  deepseek_sparse_attention/cutedsl/test_DSA_indexer_compressed_sm90_determinism.py \
  deepseek_sparse_attention/cutedsl/test_DSA_indexer_forward.py \
  -k "prepared_plan or sm90_compressed or api_is_split_from_top_k"
```

### Grouped dGLU recomputed activation for dSwiGLU — SM100, 2026-11-23 UTC

`test_grouped_gemm_dglu_activation.py` now runs every existing recomputed
activation case for both `dgeglu` and `dswiglu`. New L0 cases check the
following:

- The dSwiGLU activation is byte-equal to the BF16 GLU forward `D` when the
  saved C holds the forward's FP32 pre-activation. One power-of-two entry per
  A row and power-of-two alpha and beta make that exact. The check covers
  both `vector_f32` modes, both `glu_layout` values, dense and discrete
  weights, an empty expert and capacity past the final offset.
- The activation and `D_row` replay byte for byte under concurrent work. The
  shape is 128 tiles, enough to cross the persistent schedule.
- Those outputs are unchanged by expert reorder, duplication and a changed
  neighbor expert, and the inputs are not mutated.
- JAX inputs are rejected for the class, the dense wrapper and the discrete
  wrapper.

Each case keeps an independent Torch reference. dprob accumulates with float
atomics across N tiles, so it is checked numerically and not byte-compared.
No golden files were added. The forward byte equality is a cross-kernel
reference that does not depend on the runtime.

Run from `test/python` with the modified checkout imported:

```bash
python -m pytest gemm/cutedsl/test_grouped_gemm_dglu_activation.py -m L0 -q -rs
```

Measured on NVIDIA GB200 SM100 (148 SMs), driver 595.71.05, Torch
`2.14.1+cu130`, CuTe DSL 4.8.0 with its CUDA 13.4 compiler, TVM FFI 0.1.16
and JAX 0.9.1 (CPU, for the host-only rejections). The frontend 1.34.0
source was imported over a stubbed compiled extension. All 66 cases ran:
64 passed and 2 failed. The two failures are both
`test_dglu_activation_caller_workspace_and_first_capture` cases. A
`DeviceView` workspace needs the compiled extension's DLPack export, which
the stub does not provide, and the same two cases fail identically on the
unmodified source. A mutated kernel that multiplies `up * prob` before the
SiLU product fails the forward byte-equality cases. SM103 and SM107 remain
unverified; SM90 is unsupported.

### Canonical grouped GLU branch storage — SM100, 2026-11-10 UTC

The new BF16, NVFP4 and MXFP8 suites exercise conventional `[gate | up]`
storage in the existing SM100/103 GLU and dGLU kernels. They distinguish branch ordering
from the APIs' existing natural-rank tensor support. Comparisons against the
legacy interleaved path use independently permuted test data; independent
numerical references separately check the GEMM/activation results. Quantized
payloads and scale bytes remain fixed for replay and batch comparisons.
Packed and discrete expert weights are covered, including empty experts.

MXFP8/NVFP4 backward uses `deterministic=True` for dprob byte comparisons.
BF16 dprob byte qualification is limited to the single-N-tile F=128 case. dbias
retains the existing atomic reduction, so its check is numerical rather than
a bitwise-determinism claim. The option does not remove the existing 256-row
expert-token alignment or change the default interleaved layout.

Run from `test/python` with the modified checkout imported:

```bash
python -m pytest gemm/cutedsl/test_grouped_gemm_glu_branch_layout.py \
  gemm/cutedsl/test_grouped_gemm_dglu_gate_then_up.py -m L0 -q -rs
```

Measured on NVIDIA GB200 SM100 (152 SMs), Torch
`2.13.0a0+8145d630e8.nv26.06`, CUDA 13.3, driver API 13030,
cuDNN 92600, CuTe DSL 4.8.0 with its CUDA 13.4 compiler, and TVM FFI 0.1.11.
The modified frontend 1.33.0 source and matching rebuilt Python extension
were imported. The initial qualification passed 66 forward and 35 backward
cases without skips, including runtime-hash comparisons. Those snapshots are
kept as external validation artifacts rather than committed fixtures. The
corresponding tests retain their reference, replay and batch-invariance checks
without requiring a runtime-specific baseline. Existing runtime records are unchanged.
SM103 remains unverified; the option is unavailable on SM90 and SM107.

The final source also passes ten allocator-poison CUDA Graph regressions:
six forward cases (three precisions, one- and two-CTA tiles), and four
block-scaled backward cases (NVFP4/MXFP8, both branch layouts). These exercise runtime TMA
descriptor reuse after scale-buffer allocations change; the shared descriptor
helper now publishes and acquires descriptors through the tensormap proxy.
The original interleaved path reproduced the failure before this fix.

The NVFP4 GLU+Hadamard sibling passed 13 cases. Its checks include
dense/discrete weights, SMEM RHT, dense TMEM RHT, independent numerical
references, replay, batch invariance and explicit rejection of
canonical discrete weights with `use_tmem_post_rht_amax=True` (a compiler
limitation; callers can select the existing SMEM path).

GLU+Hadamard+quant covers BF16 row RHT and NVFP4 row/column RHT for
both GLU input layouts. Columnwise scales use the shared standard GEMM-blocked
output contract documented above. The 28-case branch-layout suite checks exact
payload/scale bytes, independent GEMM/activation/Hadamard/dequantization
references, changed-input graph replay and expert reorder/duplication.

### GLU+Hadamard+quant natural tensor ranks — SM100, 2026-11-11

The 52-case `test_grouped_gemm_glu_hadamard_quant_canonical.py` L0 suite
qualifies the natural-rank adapter independently of GLU branch order. It covers
`gate_then_up` and `interleaved_32`, dense/discrete NVFP4 E2M1 weights with E4M3
MMA-packed scales, BF16/NVFP4 D and rowwise RHT, and standard GEMM-blocked
colwise scales. Primary GEMMs use logical K=512, N=256/2048, and
expert M=[256, 0, 512]; batch and dynamic-M cases vary those token counts.

All output payload and scale bytes must equal the existing kernel-facing API.
Independent GEMM/SwiGLU/Hadamard/dequantization references retain their previous
tolerances. The suite separately checks three fresh-allocation/noise-stream
replays, expert-alone/reorder/duplication/removal/changed-neighbor invariance,
and changed A/B/scale CUDA Graph replay. These replay invocations are internal
to the 52 pytest cases, not additional collected cases. Caller-owned flat and
contiguous rank-six scale outputs, prepared execute without framework tensor
view operations, malformed packed-scale spans, empty experts, entirely empty
batches and no-RHT calls are also covered. The adapter does not change kernel
arithmetic, and no runtime-specific golden was created or rewritten.

The qualification of the edited `e0f75a6e` checkout with the column-scale
correction passed **97 cases with 6 Rubin-only skips**, in 150.63 seconds:
17 passes / 6 skips in the existing quantization suite, 28 passes in the
branch-layout suite, and 52 passes in the natural-rank suite. Runtime was NVIDIA
GB200 SM100, driver 580.126.20, Torch 2.13.0a0+8145d630e8.nv26.06 / CUDA 13.3,
cuDNN backend 92600, and CuTe DSL 4.8.0 compiling with CUDA 13.4.
The independent default-output qualification above records the later isolated
scale fix separately. SM103, SM107 and CUDA 12 compilation remain unverified
for this revised feature suite; earlier alternate-format counts do not qualify
its current scale contract. From `test/python`:

```bash
python -m pytest gemm/cutedsl/test_grouped_gemm_glu_hadamard_quant.py \
  gemm/cutedsl/test_grouped_gemm_glu_hadamard_quant_gate_then_up.py \
  gemm/cutedsl/test_grouped_gemm_glu_hadamard_quant_canonical.py -m L0 -q -rs
```

### Rebased grouped GLU stack — SM100, 2026-11-11

The feature was rebuilt on develop `683f294b` with the default NVFP4 column-scale
packing fix as a separate prerequisite. The tested stack `c6e49e62` passed
**102 Torch cases with 6 Rubin-only skips**: 97 from the three quantization,
branch-layout and natural-rank suites above (149.01 s), plus all 5 default-layout
determinism cases (11.93 s). The JAX selection passed **16 cases** (45.26 s).
Subsequent history cleanup moved API documentation into the prerequisite fix
and clarified documentation; the tested feature source and tests are unchanged.
The runtime was the same GB200 / CUDA 13.4 compiler environment documented above.

Separate processes imported the standalone fix and feature stack, generated
identical inputs, and compared raw-byte hashes of C, D, columnwise RHT and its
scales. All outputs matched for the default `interleaved_32` path at expert
M=[256,0,512], N=256/2048, K=512. Each process also checked an independent
numerical reference. The 15-case TE integration selection passed (48.64 s)
using TE Python source `21d47d4d` and its previously built native libraries;
this was not a fresh full native rebuild. SM103/SM107 remain unverified.

### Rebased grouped GLU stack — SM100, 2026-11-16

The prerequisite column-scale fix and grouped GLU feature were rebased on
develop `5304d19d`. On four GB200 GPUs (SM100), the standalone fix passed
**24 Torch and 16 JAX cases**, with 6 SM107-only scale-format skips. The feature
selection, including the current upstream Hadamard suite, passed **371 unique
Torch cases and 16 JAX cases**, with the same 6 SM107-only skips. The two-device
cache-isolation case was rerun with two visible GPUs and passed.

The feature-layout reference now follows the upstream Hadamard contract:
pre-RHT amax is the maximum of stored BF16 D, and post-RHT amax is the maximum
after rounding the transformed values to BF16. Independent GEMM/activation
checks remain in place; their tolerances were not relaxed. Separate processes
again compared all default `interleaved_32` output bytes between the standalone
fix and feature at expert M=[256,0,512], N=256/2048, K=512. Both comparisons
matched, and each process checked its independent numerical reference.

The FE binding and TE native libraries were freshly built against this stack.
The TE integration selection on upstream main `f96e7b24` passed **115 unique
cases**, with 6 existing bias-support skips. Four-rank Megatron integration on
main `7c0fac79` passed **81 cases**: GTP 12, FSDP V2 11, FSDP V1 3, adapter/config
51, CUDA graph 1, and canonical grouped MLP determinism replay 3. Counts are
per test case, not multiplied by ranks. Tests that trace dispatch now include
the Hadamard forward variants selected by TE.

The runtime used Torch `2.13.0a0+8145d630e8.nv26.06` (CUDA 13.3), FE backend
92600, CuTe DSL 4.8.0 and the CUDA 13.4 compiler. The optional NCCL EP extension
was disabled in the fresh TE build. These are targeted numerical and replay
checks, not a complete upstream CI run or a new performance measurement.
SM103/SM107 remain unverified.

### Custom reduction — SM100 validation, 2026-11-09 (UTC)

The new `test_custom_reduction_determinism.py` L0 suite has **seven cases**:
argmax and argmin index projections for FP16, BF16 and FP32, plus a direct
FP32 ADD body. These measurements are new; the earlier attention-suite
measurements and 330-case collection record below are retained separately
and were not rerun for this addition.

Each case checks its independent Torch CPU reference, device input bytes,
fresh-buffer replay with queued side-stream work, and a target row alone,
packed among 32 neighbors, reordered, duplicated, and with changed neighbor
data. Each also replays a large batch of `16 * SM_count + 1` rows: **2369 rows**
on the measured 148-SM device, with the target at row 1184. The reduction
extent is 257, exercising the nominal 256-element boundary. These are workload
shapes, not measurements of the backend's tile size or physical CTA waves.
There are 10 graph calls per case (70 per complete suite run), not 70 pytest
cases. Actual kernel overlap is not measured.

The suite orders the producing Torch stream into the actual cuDNN handle
stream, then orders outputs back before comparisons. It preserves the handle
stream and uses no device-wide prelaunch barrier to drain the queued noise.
Snapshots own their storage; result comparisons check names, shapes, dtypes,
finite values, and raw bytes without tolerance. Index inputs remain local to
each row. No packed offsets, upstream gradients, mutable state, or custom-
reduction backward entry point exist in these tested graphs.

The index-projection graphs consume only the INT32 index. The helper's value
stream remains internal and is not claimed as a tested output. The finite
integer-valued inputs include repeated extrema: the target argmax is **5**
and argmin is **7** under the Torch tie rule. The direct ADD body uses the
same bounded data divided by eight, making all partial sums exactly
representable in FP32; the target sum is **-41/8 (-5.125)**. This establishes
no blanket guarantee for arbitrary user callbacks, arbitrary floating-point
sum data, NaN/infinity policies, or unused tuple outputs. Each graph strictly
selects a publicly enumerated native cuDNN backend plan; absence fails rather
than falling back to another provider. No OSS CuTe DSL kernel path is qualified
by this run.

Source qualification: public parent `8202053a0eac9f224a9961c3f3150765ab3be98f`,
frozen test SHA256
`5e4d7d169a55829a7a4ba43e24c0c34c5f9aa991bc72c2b8be3ff930932db644`.
The test was overlaid onto an isolated validation artifact freshly built at
`a49f23d07c068ec8442da6241bbd755c51460624`; the compiled sources are equivalent
to the public parent. Import resolution was asserted against that artifact's
`/artifacts/python/cudnn`, and both the compiled-header and loaded-backend
versions were asserted to be **92900** before execution. This is native
artifact validation with the frozen source test, not a rebuilt module from
the later documentation/golden addition.

Runtime: GPU name reported as `NVIDIA Graphics Device`, SM100 (cc 10.0),
148 SMs; Torch `2.14.0a0+4fdf77b940.nv26.08`, CUDA 13.4, driver API 13030,
cuDNN 92900, nvidia-cutlass-dsl 4.7.1 and apache-tvm-ffi 0.1.13.post3.
CuTe DSL is runtime-key metadata here; the selected numerical path is native
cuDNN. The golden runtime key is `5592daabbefc7ee5`.

| Run | Passed | Failed | Errors | Skipped | Deselected | Pytest time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| New baseline recording | 7 | 0 | 0 | 0 | 0 | 11.64 s |
| Strict check, goldens mounted read-only and recording unset | 7 | 0 | 0 | 0 | 0 | 11.89 s |
| Negative control: swap valid argmax/argmin factories | 0 | 6 intended reference failures | 0 | 0 | 1 | 2.38 s |
| Negative control: replace direct ADD with legal MUL, identity 1, and a distinct body key | 0 | 1 intended reference failure | 0 | 0 | 6 | 0.81 s |

Both normal runs collected all seven L0 cases, with two Torch JIT deprecation
warnings each. The negative controls reached native execution and failed at
the independent reference assertions, rather than compilation, support checks,
missing goldens, or harness exceptions. The mutations were confined to isolated
validation copies; the frozen source above was unchanged.

Executed pytest commands inside the validation container, from `/tmp/tests`
with the asserted artifact import active:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDNN_TEST_NO_ISOLATION=1
export CUDNN_DETERMINISM_RECORD=/results/goldens
unset CUDNN_DETERMINISM_GOLDENS
python3 -m pytest -m L0 -o addopts='--tb=short --no-header' \
  -o cache_dir=/tmp/pytest_cache --junitxml=/results/result.xml \
  test_custom_reduction_determinism.py

# Separate strict run; /goldens is the reviewed baseline, mounted read-only.
unset CUDNN_DETERMINISM_RECORD
export CUDNN_DETERMINISM_GOLDENS=/goldens
python3 -m pytest -m L0 -o addopts='--tb=short --no-header' \
  -o cache_dir=/tmp/pytest_cache --junitxml=/results/result.xml \
  test_custom_reduction_determinism.py
```

The swapped-factory control used the same strict environment and pytest command
with `-p det_mutation -k arg_index`; its plugin interchanged the two public
modifier factories. The legal-MUL control used `-k direct_add` after replacing
only the copied body's ADD with MUL, its identity with 1, and its body key with
`test.python.custom_reduction.determinism.direct_multiply.v1`.

Seven JSON files under `5592daabbefc7ee5/` are new baselines for this previously
uncovered numerical API; no prior golden was replaced and no numerical change
is intended. Recording followed replay and references. Independent integer
arithmetic and IEEE byte encoding reconstructed all input, iota, and expected
output hashes (5, 7, and -41/8), and the seven complete file SHA256 values
remained unchanged after the strict run and when copied into this directory.
A normal checkout check uses:

```bash
unset CUDNN_DETERMINISM_RECORD
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest test_custom_reduction_determinism.py -m L0 -q -rs
```

The two custom-reduction support rows now cover the measured SM100
configurations only. SM103 remains explicitly unverified in the existing
combined SM100/103 column; SM90, SM107, other hardware, other data profiles,
and arbitrary reducers remain unverified.

The four linear-attention suites live in `linear_attention/frost/test_*_determinism.py`.
They pin FROST and set `batch_invariant=True`. They cover FP16 and BF16 with
nonzero initial state and fixed output/final-state gradients. QSA lives in
`QSA/cutedsl/test_qsa_determinism.py` and enables `deterministic=True` on
backward; the forward has no switch.
It covers BF16 with 12:1, 24:2, and 6:1 query/KV head counts.
DSA uses FP16/BF16 and enables `deterministic=True` on backward;
`TestDSADeterministicDKV` also pins each SM100/SM103 dKV layout.

CSA compressor coverage currently lives in
`compressed_sparse_attention/cutedsl/test_CSA_compressor.py`. It checks repeated
forward/dKV/dScore bytes, `deterministic=True` dAPE, strict torch deterministic mode,
poisoned caller workspace/output CUDA Graph replay, ragged and empty segments, and
exact-zero static-capacity rows for the generic and ratio-128 forward families. The
focused determinism set has 29 L0 cases: five replay cases; eighteen forward/deterministic-
backward batch cases spanning ratios 2, 4, and 128, both `coff` values, scalar and
wide generic layouts, and both supported ratio-128 head dimensions; and six
replay-plus-reference goldens for generic ratio-2, generic ratio-4 overlap, and the
dedicated ratio-128 path. Batch cases compare a fixed sequence alone, before and after
different-size neighbors, duplicated, and beside changed neighbor data. The ratio-128
case crosses its small/default forward schedule boundary. The ratio-128 `coff=2`,
head-dimension-512 forward boundary remains a non-strict xfail with its raw-byte
assertion intact because CI observed a batch-size-dependent byte; all other cases remain
ordinary assertions. Backward isolates the fixed sequence's upstream gradient and checks
BF16 dKV/dScore plus FP32 dAPE bytes.

At revision `dcb7bddaa358`, the new SM100 run (CUDA 13.0, cuDNN 9.24, PyTorch
2.14.1, CUTLASS DSL 4.8.1) reported 18/18 batch cases, 5/5 replay cases, and 6/6
golden/reference cases passed. The SM103 run (CUDA 13.0, cuDNN 9.20, PyTorch
2.12.1, CUTLASS DSL 4.7.1) reported the same 29 passes. The two committed runtime
baselines use keys `842e0ee8788d6de6` and `4e05d04e8e72ed1b`; identical inputs
produced identical output hashes across the two recorded runtimes. Current-head CI pipeline
80937667 later observed one of 1024 BF16 output bytes differ in
`r128-c2-d512-small-to-default-fprop` when the target moved from the standalone small
schedule into a default-schedule batch. A fresh local rerun passed, so the failure is
environment-sensitive but still disproves the broader forward batch-invariance claim; the
raw-byte assertion is retained as a non-strict xfail. Forward is Replay only in the combined
SM100/103 column. All nine deterministic-backward batch cases passed on both recorded
runtimes, so opt-in deterministic backward remains Supported. The earlier capacity-zero and
poisoned replay cases passed on SM90, but SM90 batch invariance and goldens remain
unverified; SM107 is also unverified. The default atomic dAPE path remains Not guaranteed
by contract.

At revision `bca6dcd50b81`, the requested `@cudnn-ci-bot run python/misc` pipeline
reported 147 passed / 71 skipped / 0 failed CSA cases on SM90, 218 passed / 0 skipped /
0 failed on SM100, and 218 passed / 0 skipped / 0 failed on SM107. The separate SM100
ratio-128 contract gate passed all 21 cases and every shipped forward/backward schedule.

Replay checks use fresh inputs and side-stream contention. Batch
checks reuse exact sequence data and upstream gradients while removing,
reordering, and inserting neighbors. They include nonaligned lengths, partial
tiles, and empty linear-attention sequences. QSA retains each sequence's mask
while shifting its packed token and block offsets. Stress cases cross the
8192-token linear-attention span boundary and QSA's dense/sparse forward
boundary, or create more sequence/head work items than GPU SMs.

Comparisons check tensor names, shapes, dtypes, and raw bytes. Signed zero
differences fail. Non-finite results fail unless the operation declares an
infinity sentinel: QSA uses negative-infinity LSE; DSA uses positive-infinity
LSE and negative-infinity max logits for empty rows. DSA also checks the exact
empty-row locations. Snapshots own their storage. Helper controls reject
changed signed zeros, NaN payloads, missing gradients, wrong infinity signs,
and golden mutations.

## Block-sparse attention blk64 — SM90 and SM103 validation, 2026-11-10 (UTC)

Change under test: the SM90 blk64 forward and backward kernels of the
`xshang/bsa-sm90-blk64-perf` branch (12 kernel commits on `develop`; the suite
and this record are the branch's following commit; the runs below used the
branch on `develop` `a1e86b04`, and the branch was then rebased onto
`56c9609a`, `393b1699` and `7ab3ffd7`; the first two leave the BSA package
unchanged and the third touches only its JAX wrapper `jax_api.py`, not the kernels) and
the new `block_sparse_attention/cutedsl/test_BSA_determinism.py` suite
([30 cases](#block-sparse-attention-blk64-30-cases)), which drives the blk64
API on SM90 and, through the same API, the SM100/SM103 blk64 path. The other
determinism suites were not rerun. Both GPUs ran the pure-Python kernels from a
source checkout through a Slurm wrapper that points `import cudnn` at
`<src>/python` and stubs the nanobind module (`cudnn._compiled_module`, which
the BSA kernels never call), so the runtime records report cuDNN backend
`92600` from that stub; the test banners confirmed imports from the checkout
under test.

- H100: `NVIDIA H100 80GB HBM3`, SM90, 132 SMs, driver 615.71.10 (driver API
  13040); Torch `2.15.0+cu132`, CUDA 13.2, CuTe DSL `4.8.1` (CUDA 13.4
  compiler), TVM FFI `0.1.15.post1`, Python 3.12.5. Runtime key
  `f2590ab8d14a5923` ([goldens](f2590ab8d14a5923), four records).
- B300: `NVIDIA B300 SXM6 AC`, SM103, 148 SMs, the same driver and software.
  Runtime key `d6ab5c84009f8fa5` ([goldens](d6ab5c84009f8fa5), three records;
  the FP16 golden skips by contract). B200 (SM100) and SM107 were unavailable
  and stay unverified.

Source under test: the final runs executed the committed test file byte for
byte (SHA256
`b1a3a413d46c8ce5a45ca1d54a5d077f9d298f9c99b576ad517caee931c09553`, git blob
`ee63b87a`, which `git rev-parse HEAD:test/python/block_sparse_attention/cutedsl/test_BSA_determinism.py`
prints on the PR head) and read the committed goldens; their checkout was a
working-tree snapshot of the PR head whose only difference from the committed
tree is the text of this record, filled in from their logs afterwards. The
earlier runs listed below executed unpublished working-tree revisions of the
suite that are not on the branch (their job logs name them). Those revisions
differ from the committed file as follows: none had the FP16 `kv_splits=2`
and FP16 KV-tail cases, the batch-invariance cases of the `kv_splits=2` and
FP16 fixed-count configurations, the API-rejection check in front of the
contract skips, or the `measured` channel (all added after review); in
addition, revision 1 monkeypatched the TF32 flag and passed `block_sizes=None`
in the KV-tail case, and revision 2 still passed `block_sizes=None` there.
Their backward cases are the committed ones.

Run from `test/python` (the wrapper adds `-rA --durations=20`):

```bash
python -c 'import cudnn; print(cudnn.__file__)'
python -m pytest -m L0 --collect-only -q block_sparse_attention/cutedsl/test_BSA_determinism.py
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest -m L0 block_sparse_attention/cutedsl/test_BSA_determinism.py
```

The goldens were recorded first into a new directory
(`CUDNN_DETERMINISM_RECORD=<new dir>` with `-k golden` and
`CUDNN_DETERMINISM_GOLDENS` unset), after their cases passed three executions
and the FP32 reference; the hashes were inspected (every input hash is
byte-identical between the two GPUs, every output hash differs between them,
as different kernels require) and the files placed. The golden cases and
their recorded tensors are the same in every revision, so the records were
not regenerated; the strict runs read the placed records with recording
unset.

| GPU | Run | Result |
| --- | --- | --- |
| H100 | committed test file, `--collect-only -q` | 30 node IDs (job 5779013) |
| H100 | committed test file, strict check | 30 passed, 0 skipped, 0 failed in 36.83 s (job 5779012); the run's `measured` summary printed the seven backward records |
| B300 | committed test file, strict check | 17 passed, 13 skipped (contract), 0 failed in 206.98 s (job 5779014; 167 s of it was the first case's one-time DSL initialization on that node) and, repeated on a second B300 node, 17 passed, 13 skipped, 0 failed in 62.06 s (job 5779135); both runs' `measured` summaries printed the five backward records |
| H100 | revision 1, first full run, no golden baseline | 20 passed, 4 skipped (goldens without a baseline), 24 errors in 33.99 s (job 5775580; 24 cases). Every error was the `api` fixture's teardown: pytest 9's `monkeypatch` records `torch.backends.cuda.matmul.allow_tf32` as unset and cannot delete it. Fixed with an explicit save/restore; no check changed |
| H100 | revision 1, golden recording (`-k golden`) | 4 passed, 20 deselected in 17.46 s (job 5775581) |
| H100 | revision 2, strict check | 24 passed, 0 skipped in 33.52 s (job 5775738) |
| H100 | revision 3, strict check | 24 passed, 0 skipped, 0 failed in 30.88 s (job 5776744). A first attempt (job 5776567) landed on a node whose GPU reported itself busy or unavailable and errored every case at CUDA stream creation in the conftest; it was repeated on another node |
| B300 | revision 2, golden recording (`-k golden`) | 3 passed, 1 skipped (FP16 golden: contract), 20 deselected in 47.67 s (job 5775739) |
| B300 | revision 2, full run, no golden baseline | 12 passed, 11 skipped, **1 failed** in 60.40 s (job 5775740): `TestBlk64Forward::test_replay[fprop-bf16-variable-kvtail-512tiles]` passed eight bitwise replays, then failed the FP32 reference (5881 of 4194304 output elements, max difference 0.21); see below |
| B300 | revision 3, strict check | 16 passed, 8 skipped (contract), 0 failed in 233.47 s (job 5776786; 174 s of it was the first case's one-time DSL initialization on that node, the same suite ran in 60 s on another B300). A first attempt (job 5776568) hung at the Python import step on another node for ten minutes and was cancelled |

On the committed suite B300 skips thirteen cases by contract: the nine FP16
forward cases (four replay, four batch, the golden; the SM100/SM103 blk64
forward serves BF16 only) and the four prepared-route cases
(`block_sparse_attention_backward_metadata` serves SM90 only), each after the
API rejected a tiny instance of the configuration (`skip_by_contract`,
`NotImplementedError` naming BF16 or the SM90-only clause). The earlier
revisions had three FP16 forward cases and no rejection check, so they
skipped eight; revision 2's full run without a baseline also skipped its
three golden comparisons.

### The B300 reference failure: a partial final KV block without `block_sizes`

That case ran 512 query tiles over a 2543-token KV length (a 47-row final
block) with `block_sizes=None`. A diagnostic on the same inputs (job 5775845;
identical results on `develop` `a1e86b04`, job 5775846) confined the mismatch
to the two query tiles whose active list is exactly the final block (counts 1),
with the LSE of every tile attending that block off by up to 0.27. It is
independent of CLC scheduling and of the poison value in inactive list slots,
isolating one tile reproduces it, the same shape without a KV tail is clean,
and explicit `block_sizes` with a 47-row final block removes it entirely (0
differing rows). Code reading agrees: the SM100/SM103 blk64 forward only
switches to an exact KV layout for a partial tail and does not mask the tail
rows, so TMA's zero-filled rows join the softmax with logit 0, while the SM90
kernel masks the tail from the KV length. This is inside the documented caller
contract ("Provide `block_sizes` whenever a referenced final block is only
partially valid", `docs/operations/sparse_attention/bsa.md`) and pre-exists
this change. The suite's KV-tail case now passes `block_sizes` with the
shortened final block on every GPU; the `block_sizes=None` tail form stays
covered numerically on SM90 by `test_BSA_forward.py`
(`test_sm90_blk64_partial_kv_blocks`, the short-last-tile case). It is a
numerical-contract observation, not a determinism result: the case replayed
bitwise before failing the reference.

### Measured backward bytes

Every backward case passed its FP32 reference, its route assertions and its
empty-row checks on both GPUs; the regular route's dK/dV over prepared
metadata replayed bitwise (asserted; SM90 only). The other gradients are
measured, not asserted. Each cell gives the range of differing bytes over the
seven replays (fresh buffers, side-stream work) or over the five batch
comparisons (packed, reversed, duplicated twice, changed neighbors), against
the first execution or the batch element alone. The replay comparisons cover
the packed batch (1572864 bytes per gradient), the batch comparisons the
target's slice (524288 bytes per gradient). A zero means no byte differed in
that run; it is not a guarantee. The table keeps the strict run of revision 3
(same backward cases) beside the final runs as an earlier measurement: the
counts move from run to run, the picture does not.

| GPU | Run | Route | Replay dQ | Replay dK | Replay dV | Batch dQ | Batch dK | Batch dV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H100 | job 5779012 (final) | `regular` | 3-7 | 12-37 | 2-21 | 2-5 | 12-18 | 3-9 |
| H100 | job 5779012 (final) | `grouped` | 3-11 | 0-19 | 0-11 | — | — | — |
| H100 | job 5779012 (final) | `paired` | 0-1 | 1-3 | 1-3 | 0-1 | 0-2 | 0 |
| H100 | job 5779012 (final) | `prepared-regular` | 2-6 | 0 (asserted) | 0 (asserted) | — | — | — |
| H100 | job 5779012 (final) | `prepared-paired` | 0-6 | 0-1 | 1-3 | — | — | — |
| H100 | job 5776744 (revision 3) | `regular` | 6-10 | 28-38 | 15-23 | 3-5 | 10-15 | 3-6 |
| H100 | job 5776744 (revision 3) | `grouped` | 6-9 | 10-23 | 8-12 | — | — | — |
| H100 | job 5776744 (revision 3) | `paired` | 0-1 | 1-3 | 1 | 0-1 | 0-1 | 0 |
| H100 | job 5776744 (revision 3) | `prepared-regular` | 5-7 | 0 (asserted) | 0 (asserted) | — | — | — |
| H100 | job 5776744 (revision 3) | `prepared-paired` | 0-1 | 0-1 | 0-2 | — | — | — |
| B300 | job 5779014 (final) | `regular` | 6-9 | 11-23 | 5-9 | 0-2 | 13-20 | 2-9 |
| B300 | job 5779014 (final) | `grouped` | 7-13 | 9-18 | 2-10 | — | — | — |
| B300 | job 5779014 (final) | `paired` | 0-2 | 0-5 | 0-4 | 0 | 0-2 | 0 |
| B300 | job 5779135 (final, second node) | `regular` | 10-15 | 34-41 | 23-27 | 0-2 | 7-23 | 1-8 |
| B300 | job 5779135 (final, second node) | `grouped` | 4-16 | 1-14 | 0-13 | — | — | — |
| B300 | job 5779135 (final, second node) | `paired` | 0-2 | 2-5 | 1-3 | 0 | 0-4 | 0 |
| B300 | job 5776786 (revision 3) | `regular` | 7-9 | 11-18 | 8-15 | 0-4 | 11-19 | 5-8 |
| B300 | job 5776786 (revision 3) | `grouped` | 5-11 | 2-17 | 0-10 | — | — | — |
| B300 | job 5776786 (revision 3) | `paired` | 1-4 | 0-4 | 0-3 | 0 | 0-3 | 0 |

The default regular route rebuilds the K-to-Q CSR on every call, and its
scatter kernel appends each query block behind an atomic cursor, so a KV
block's query list is accumulated in a call-dependent order; the prepared
route reuses one CSR, and its direct dK/dV stores then replay bitwise while
dQ (reduce-added across the KV-block CTAs) still differs. The paired kernel's
first arriver stores dK/dV and later arrivers reduce-add; the grouped route
reduce-adds dK/dV through the FP32 workspace.

## D512 FP8 paged forward — SM100 validation, 2026-11-09 (UTC)

This addition covers `pygraph.sdpa_fp8` through the FROST SM100/SM103 D512
forward body, with dense Q and native paged K/V. It adds no backward path.
Numerics are unchanged from the existing D512 FP8 arithmetic; the new
eligibility and page addressing had no earlier golden. No existing golden
was replaced.

The eight L0 determinism cases in
`sdpa/frost/test_sdpa_fp8_d512_paged_determinism.py` comprise four replay/batch
cases and four corresponding goldens: E4M3→BF16, E4M3→E4M3, E5M2→FP16 and
E5M2→E5M2, with split counts 1/4, NHD/HND pools and PackGQA. Each target's
quantized Q/K/V and FP32 scale bytes stay fixed across batches of 1, 3, 5
and 129 requests, reordered/duplicated targets and changed neighbors. Every
call binds fresh buffers and workspace. Repeated calls include queued work
on a separate stream; actual overlap is not measured. O and Stats are compared
per sequence; Amax is compared on repeated identical batches and checked
against the whole-batch numerical reference otherwise. Input bytes must remain
unchanged. The independent reference dequantizes the operands and computes
attention in Torch; it is separate from raw-byte comparisons.

The adjacent paging tests cover page sizes 8/16/32/64/128/256, query lengths
1/2/16/257, empty and partial KV pages, Q trimming, causal/window masks,
sinks with and without split, separate K/V tables and independently selected
pool layouts, D384/V320 head padding, runtime stride rebinding and changed
page/scalar CUDA Graph replay. Dense/THD controls exercise the existing
unpaged D512 implementation.

Measured on an SM100 development GPU with 148 SMs, driver API 13020,
Torch 2.14.1+cu130, cuDNN 9.28.0.38, CuTe DSL 4.8.0 (CUDA 12.9 compiler),
TVM FFI 0.1.14.post1, Python 3.10.12, and FE `868b3ca` plus this patch.
Imports resolved to the isolated checkout's `python/cudnn`. SM103 was
unavailable and remains unverified. SM90/SM107 do not admit this D512 FP8
paged path. Source qualification does not establish cross-GPU byte equality.

| Run | Passed | Failed | Skipped |
| --- | --- | --- | --- |
| Prepared D128/D256/D512 replay, four D512 batch cases, admission, comparator controls | 37 | 0 | 0 |
| New goldens recorded into an empty candidate directory after reference/replay | 4 | 0 | 0 |
| Strict golden check plus final page-boundary/envelope cases, recording unset | 16 | 0 | 0 |
| Existing dense/THD D512 capture and graph split controls | 40 | 0 | 0 |
| Compute Sanitizer memcheck, final page-boundary/envelope cases | 8 | 0 | 0 |
| Compute Sanitizer synccheck, same cases | 8 | 0 | 0 |
| Compute Sanitizer racecheck, page8 single-Q and page128 persistent/window cases | 2 | 0 | 0 |

These rows describe separate runs; shared cases are not additional coverage.
The first D512 prepared paging case fails on the unchanged parent because
there is no FROST candidate. During test development, shrinking a declared
split-plan batch was rejected by its existing contract; the batch tests now
build a plan for each batch size. Torch's absent FP8 `isfinite` implementation
also required the shared snapshot helper to promote only its finiteness
predicate, while preserving original FP8 snapshots and byte comparisons.
Two comparator controls cover payload preservation, signed zero and NaN
rejection. Neither issue was waived with a skip or weakened byte comparison.

Run from `test/python`, with `PYTHONPATH` pointing to the checkout:

```bash
python -m pytest sdpa/frost/test_sdpa_prepared_fp8_paged.py::test_prepared_fp8_paged_strides_and_capture \
  sdpa/frost/test_sdpa_fp8_d512_paged_determinism.py::test_fp8_d512_paged_replay_and_batch_invariance \
  sdpa/frost/test_sdpa_fwd_paged_sm100.py::test_paged_graph_fp8_declines_off_contract \
  core/test_determinism_utils.py -x -q
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" python -m pytest \
  sdpa/frost/test_sdpa_fp8_d512_paged_determinism.py \
  sdpa/frost/test_sdpa_fwd_paged_sm100.py::test_paged_graph_fp8_d512_boundaries \
  sdpa/frost/test_sdpa_fwd_paged_sm100.py::test_paged_graph_fp8_d512_head_envelope -x -q
python -m pytest sdpa/frost/test_sdpa_prepared_fp8.py::test_prepared_fp8_capture_replay_reads_current_scales \
  sdpa/frost/test_sdpa_fp8_d512_graph_split.py -k 512 -x -q
```

Memcheck and synccheck each report **0 errors**; racecheck reports **0 hazards**
(0 errors, 0 warnings) for its two selected boundary cases. Their process uses isolated
`cuda-bindings==13.2.0`, matching driver API 13020, with all API diagnostics
enabled. An initial memcheck with the environment's 13.4.1 bindings reported
40 API-resolution errors (13.3/13.4 requests to the 13.2 driver), while its
8 numerical tests passed; it was not counted as a clean sanitizer run.
No kernel memory error was reported. Reproduce with the runtime above:

```bash
compute-sanitizer --tool memcheck --error-exitcode 99 python -m pytest \
  sdpa/frost/test_sdpa_fwd_paged_sm100.py::test_paged_graph_fp8_d512_boundaries \
  sdpa/frost/test_sdpa_fwd_paged_sm100.py::test_paged_graph_fp8_d512_head_envelope -x -q
# Repeat with --tool synccheck. For racecheck, add -k '8-1-1-True or 128-257'.
```

The paged kernel launch explicitly retains its register split. The emitted
SM100 SASS contains `USETMAXREG.TRY_ALLOC` with 240 registers and
`USETMAXREG.DEALLOC` with 40; the original launch without the occupancy hint
had no register-split instructions. Dense launch geometry is unchanged.

The four new records are under runtime key `ef27de8cf1fd96d2`. Recording used
`CUDNN_DETERMINISM_RECORD` with a new temporary directory and `-k golden`;
the separate strict run used the committed records with recording unset.

## How to read the test cases

A **suite** is one operation's test class. A **case** is one collected pytest
node after parameter expansion. For example,
`TestKDA::test_replay[bf16-bprop]` is one case: BF16 backward, eight executions.
Those eight executions are not eight reported tests. Backward cases also check
the forward outputs and saved state used to compute gradients.

The expanded set has **330 cases**: 324 operation cases and six harness
controls. All 330 cases are L0,
including stress and boundary cases, so the default marker selects the full set.
List the exact node IDs without executing the kernels:

```bash
pytest core/test_determinism_utils.py linear_attention/frost/test_*_determinism.py \
  QSA/cutedsl/test_qsa_determinism.py deepseek_sparse_attention/cutedsl/test_DSA*determinism.py --collect-only -q
```

| Check | What stays fixed | What changes | Pass condition |
| --- | --- | --- | --- |
| Replay | Input bytes, state, masks, upstream gradients | Fresh buffers and concurrent GPU work | Every result byte matches the first execution |
| Batch invariance | Each selected sequence and its upstream gradients | Batch size, neighbors, order, packed offsets | Its results match the corresponding baseline slice |
| Neighbor isolation | Target sequence, mask, state, upstream gradients | Other sequences' data, state, gradients, and QSA masks | Target results do not change |
| Golden regression | Input hashes and complete runtime key | Frontend source revision | All result hashes match the committed baseline |
| Numerical reference | Mathematical inputs and gradient sources | GPU kernel versus independent reference | Existing operation tolerances pass |

The golden cases check replay and the numerical reference before reading the
saved baseline. The new head-layout, GDP step-count, isolated-loss, and QSA
mask-replay cases also check an independent reference. A reference pass alone
does not establish bitwise determinism.

### Linear attention: 28 cases per family, plus eight GDP cases

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
| GDP `test_qk_l2norm` | L0 | fprop/bprop × the fused q/k normalization at the default epsilon and at `qk_l2norm_eps=1e-10`; lengths 65, 17, 3; rows scaled around both epsilons; reference, four executions, partitions | +4 for GDP |

The three head layouts are `(Q/K heads, V heads, K dimension, V dimension)`:
`(1, 4, 64, 128)`, `(4, 1, 128, 64)`, and `(2, 2, 128, 128)`.
Tile, head-layout, isolated-loss, and GDP step-count cases each run four
executions, then compare every sequence alone, the reversed batch, and a
duplicated first sequence. GDP slices K/V/Beta gradients on the expanded
Householder axis. State tensors use `[sequence, output heads, V, K]`.

### Fused GDN step kernels: per-geometry replay and batch invariance, no goldens

The fused decode / MTP-verify step entries (`cudnn.linear_attention.frost.gdn_decode_fused` / `gdn_verify_fused`) carry their determinism
cells inside their own modules (`linear_attention/frost/test_gdn_decode_fused.py`, `test_gdn_verify_fused.py`; the FlashInfer plan-API twin in
`test_flashinfer_gdn_backend.py`), at the default geometry (16 K / 48 V heads, hidden 5120) and at five per-rank geometries ((1, 8, 8192),
(2, 16, 8192), (4, 32, 8192), (4, 16, 4096) and, since 2026-11-06, (2, 8, 4096) -- 8-way tensor parallelism of a hidden-4096 model; the FlashInfer
plan-API twin covers the first four).  Every comparison is raw bytes: the FP8 and BF16 outputs, every FP32 state slot, the conv-state
pool and the reader tickets.

| Check | Cells | What it holds |
|---|---|---|
| Replay | `*_matches_the_torch_model_and_replays_bitwise` / `*_matches_the_torch_reference_and_replays_bitwise`, `test_deterministic_*`, the graph-replay cells | Two fresh identical operand sets (and a CUDA-graph replay over distinct layers) are byte-identical; the independent fp64 reference is a separate check at the operations' tolerances |
| Batch invariance | `test_rank_geometry_rows_are_batch_invariant` (decode), `test_rank_geometry_requests_are_batch_invariant` (verify) | A row / request alone writes the bytes it writes inside a 4-row / 4-request step (its outputs, its state slot(s), its conv slot).  The verify step holds this under ONE gated-norm association: the one-request pick takes the reference kernel's two-CTAs-per-head association (`ctas2`, 4-row chains), several requests `ctas1` (8-row chains) -- `norm_form_for(requests)`, a by-design dependence of the normed output's last bits on the request count, so the lone request runs under `ctas1` in the cell |
| Cross-form | `*_knob_arms_are_bitwise_*`, `test_pair_and_one_cta_forms_are_bitwise_at_one_request` | Every form family (cluster pair, register-capped, streamed, thread mappings, issue orders, ring depths) writes the bytes of the one-CTA form |
| Reference-kernel bytes | `test_bitwise_with_the_reference_verify_kernel_outputs[*]` (with `CUDNN_FROST_GDN_VERIFY_REFERENCE_DIR`), the FlashInfer `frost` bitwise cells | The default geometry replays the saved reference-kernel output sets byte for byte; no committed goldens (the saved sets are the oracle) |

GPU coverage: cc 10.0 (a GB200; the per-rank cells 2026-11-05), cc 10.3 (verified on a GB300, 2026-11-05: the same cells at the default and the four per-rank geometries, the saved reference-kernel sets byte for byte, the three register-capped default-geometry forms rendering the same `sm_103a` cubin as before the geometry change) and cc 12.0 (an RTX 5090, 99 KB per-CTA opt-in, 2026-11-05: the default geometry's cells and the per-rank cells -- the replay, batch-invariance and cross-form cells there compare against the landing-slab one-CTA form, the one this part holds; the arms the part does not hold skip by their typed decline); cc 10.7 in the record of 2026-11-10 below.  The fifth geometry's cells, (2, 8, 4096), were verified
on cc 10.3 (a GB300, 152 SMs, 2026-11-06, inside a vLLM container image: torch 2.14.0+cu130, nvidia-cutlass-dsl 4.8.0, cuDNN backend 9.24.0;
`pytest linear_attention/frost/test_gdn_decode_fused.py -k k2v8h4096` 30 passed / 244 deselected, `pytest linear_attention/frost/test_gdn_verify_fused.py
-k k2v8h4096` 37 passed / 306 deselected) and on cc 10.0 (a GB200, 152 SMs, in the repository's CI: torch 2.14.0a0+b2c75dd062.nv26.09, CUDA 13.4,
driver API 13040, cuDNN 92700, nvidia-cutlass-dsl 4.8.0; 16 pytest workers; the CI run of `test/python/linear_attention` on 2026-11-06 (UTC)
at the head of the pull request that added the cells (#2319), and the scheduled run of 2026-11-09 on `develop` -- the CI log prints its selection, not a
pytest command line, so the counts are read from the junit report of each run and are the same in both: the `k2v8h4096` cells of
`test_gdn_decode_fused.py` 30 passed, of `test_gdn_verify_fused.py` 37 passed, none skipped or failed); on cc 12.0 those cells are unverified.

cc 10.7 record (2026-11-10 UTC): a VR200-class part (cc 10.7, 212 SMs; aarch64 host, Python 3.12) with torch 2.14.0+cu130, CUDA 13.0, driver API
13050, cuDNN 92700, nvidia-cutlass-dsl 4.8.0, tvm-ffi 0.1.14.post0; one in-process pytest run (`CUDNN_TEST_NO_ISOLATION=1`) on one visible device,
with `CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1`, `CUDNN_FRONTEND_DISABLE_COMPILED_CACHE=1` and `CUDNN_FROST_GDN_VERIFY_REFERENCE_DIR` set to the 16
saved reference-kernel sets, from a clean tree at `868b3cae4` (#2369); no file under `python/cudnn/linear_attention` or
`test/python/linear_attention` differs between that commit and `8e31a5725`, the `develop` head this record was written on.  `cd test/python &&
pytest linear_attention/frost/test_gdn_decode_fused.py linear_attention/frost/test_gdn_verify_fused.py
linear_attention/frost/test_gdn_support_matrix_docs.py -v -ra --durations=15` -> 616 passed, 4 skipped, 0 failed in 590 s:
`test_gdn_decode_fused.py` 273 passed / 1 skipped, `test_gdn_verify_fused.py` 340 passed / 3 skipped, `test_gdn_support_matrix_docs.py` 3 passed.
Each of the five per-rank geometries: 30 passed (decode) and 37 passed (verify), none skipped.  Replay:
`test_rank_geometry_matches_the_torch_model_and_replays_bitwise` 25, `test_rank_geometry_matches_the_torch_reference_and_replays_bitwise` 50,
and at the default geometry `test_deterministic_and_matches_the_torch_model`, `test_graph_replay_over_distinct_layers_matches_eager` and
`test_deterministic_and_graph_replay_over_distinct_layers_matches_eager`, 1 each.  Batch invariance:
`test_rank_geometry_rows_are_batch_invariant` 5, `test_rank_geometry_requests_are_batch_invariant` 5 (the verify cells under one gated-norm
association, as on the other parts).  Cross-form: `test_knob_arms_the_part_holds_are_bitwise_with_the_default_pick_at_b4` 55 and
`test_rank_geometry_knob_arms_are_bitwise_with_the_one_cta_form` 95 (decode); `test_knob_arms_the_part_holds_are_bitwise_with_the_default_pick`
43, `test_rank_geometry_knob_arms_are_bitwise_with_the_one_cta_form` 115 and `test_pair_and_one_cta_forms_are_bitwise_at_one_request` 2
(verify).  Reference-kernel bytes: `test_bitwise_with_the_reference_verify_kernel_outputs` 14 passed, 2 skipped.  The 4 skips:
`test_operands_on_another_cuda_device_are_rejected_before_any_launch` in both modules ("needs two CUDA devices") -- rerun in the same runtime
with two devices visible and no reference directory, `pytest linear_attention/frost/test_gdn_decode_fused.py
linear_attention/frost/test_gdn_verify_fused.py -v -ra -k another_cuda_device` -> 2 passed, 600 deselected; and the saved sets `r16t6_pad` and
`r8t6_padskip` ("the reference kernel declined this set": their saved reference files hold a decline instead of outputs, so the two cells skip
wherever these sets are used; the torch reference is their oracle).  Not run on cc 10.7: the FlashInfer plan-API twin
(`test_flashinfer_gdn_backend.py`).

### Shared-expert decode pair: replay, batch invariance and goldens per geometry and form

`moe/frost/test_shared_expert_decode_determinism.py` pins `cudnn.SharedExpertDecode` (the bf16 shared expert next to a routed MoE,
1 .. 24 tokens, two launches) at both validated geometries (hidden 8192 with intermediate 256 and 128).  Every comparison is raw bytes
over the pair's output `y` and the live rows of its inter-launch scratch (`act` bf16, `gate_part` fp32); the independent fp64 reference of
the bf16 operands is a separate check at the harness band (rel-L2 <= 2e-2; `test_shared_expert_decode.py`).

| Test method | Level | Parameters and purpose | Cases |
| --- | --- | --- | ---: |
| `test_replay` | L0 | geometry x T 1 / 6 / 24; eight executions on fresh buffers under side-stream contention; reference check | 6 |
| `test_replay_on_a_fresh_entry_and_fresh_operands` | L0 | two entries, two fresh identical operand sets | 2 |
| `test_batch_invariance` | L0 | geometry x inside T 6 / 24: every token alone (T 1, one m-tile, row 0) vs inside, and inside the reversed batch -- the T 24 cell crosses the two-m-tile form boundary and the fragment row position | 4 |
| `test_duplicated_token_and_neighbour_isolation` | L0 | a token duplicated (both copies = alone), neighbours redrawn around it in the two-tile form, a row-strided x view vs its contiguous copy | 2 |
| `test_row_strided_views_write_the_contiguous_bytes_and_leave_the_inputs_unmodified` | L0 | geometry x T 5 / 20; `x` and `out` as column slices of a slab 64 columns wider than the hidden size (a row stride that is not the hidden size): the live rows' bytes equal the contiguous run's, the slab's columns past the hidden size keep their sentinel, every input holds the bytes it held before the launches | 4 |
| `test_twenty_four_to_twenty_five_is_the_typed_decline` | L0 | the T ceiling is a `NotImplementedError`, never a provider switch | 1 |
| `test_golden` | L0 | geometry x form (T 3 / 19); three executions, the reference, the saved hashes (`shared-expert-decode-<geometry>-tiles<n>-ksplit8`: the default form -- eight K slices, x resident by bulk row copies, the gate leaves folded on the cluster leader) | 4 |
| `test_v1_arm_reproduces_the_v1_goldens` | L0 | geometry x form on the SAME operands, the first release's form (`v1_cfg`: K_SPLIT 2 / 4, the x ring, the launch-2 gate fold) against the `ksplit2` / `ksplit4` sets recorded with that release -- the shared code paths moved no first-release addition | 4 |

Golden sets: the `ksplit2` (I 256) / `ksplit4` (I 128) sets were recorded with the first release (2026-11-06, cc 10.3) and are kept as the
v1 arm's; the `ksplit8` sets are the one-wave default's -- a NEW rounding class (reason: the pinned `K_SPLIT` moved from 2 / 4 to 8, the eight
slices are added in cluster-rank order, and the gate logit's leaves are added per row-pair group on the cluster leader before launch 2's
lane fold), recorded on the same part after the replay, batch-invariance and reference cells passed.  The knob arms (`PDL`, `PDL_PAIR`,
`PDL_RELEASE`, `N_GROUPS`, `X_BULK_ROWS`, `X_STAGES`, `W_PHASES`, `ROWS_PER_CTA`, `KRED_SPLIT`) are bitwise with the default in
`test_shared_expert_decode.py` (`test_bit_neutral_knob_arms_write_the_default_bytes`, `test_kred_split_arm_is_bitwise_at_the_one_group_geometry`);
the `K_SPLIT` / `GATE_CLUSTER_FOLD` / v1 rounding-order arms are within the band there and each is pinned bitwise-replay (three executions
under side-stream contention) and batch-invariant (every token of a T 6 step alone vs inside) in
`test_order_arms_are_rounding_order_arms_within_the_band`; goldens exist for the default and the first release's form only; the
graph-capture replay is bitwise with eager there too.

GPU coverage: cc 10.3 and cc 10.7 (replay, batch invariance and goldens) and cc 10.0 (replay and batch invariance; the golden cells are not
covered there).  Run of record on cc 10.3 (2026-11-06): a GB300 (cc 10.3, 152 SMs) with torch 2.14.0+cu130, CUDA 13.0, driver 13030,
cuDNN 92400, nvidia-cutlass-dsl 4.8.0 (the goldens' `runtime` block); `cd test/python && pytest
moe/frost/test_shared_expert_decode_determinism.py` -> 27 passed, 0 skipped, and `pytest moe/frost/test_shared_expert_decode.py` ->
191 passed, both from a clean tree at the commit that carries the `ksplit8` goldens.  Another runtime stack (torch / CUDA / cuDNN / DSL
version) hashes to another goldens directory, so the golden cells SKIP there -- a coverage gap, not a pass -- until a set is recorded for
it with a stated reason.

cc 10.0 record (2026-11-07 UTC, repeated 2026-11-09): a GB200 (cc 10.0, 152 SMs) in the repository's CI, with torch 2.14.0a0+b2c75dd062.nv26.09,
CUDA 13.4, driver API 13040, cuDNN 92700, nvidia-cutlass-dsl 4.8.0, tvm-ffi 0.1.14.post1; 16 pytest workers.  Two runs of
`test/python/moe/frost`: at the head of the pull request that added the suite (#2340), and the scheduled run of 2026-11-09 on `develop` (the CI log
prints its selection, not a pytest command line; the counts are read from the junit report of each run and are the same in both).
`test_shared_expert_decode.py`: 197 passed, 0 skipped.  `test_shared_expert_decode_determinism.py`: 19 passed, 8 skipped.  Passed:
`test_replay` 6, `test_replay_on_a_fresh_entry_and_fresh_operands` 2, `test_batch_invariance` 4,
`test_duplicated_token_and_neighbour_isolation` 2, `test_row_strided_views_write_the_contiguous_bytes_and_leave_the_inputs_unmodified` 4,
`test_twenty_four_to_twenty_five_is_the_typed_decline` 1.  Skipped: every case of `test_golden` (4) and of
`test_v1_arm_reproduces_the_v1_goldens` (4), each with "no golden for ..." -- no set is recorded for that runtime, so on cc 10.0 the golden
comparison is a coverage gap, not a pass.  A cc 10.0 part with 148 SMs (the host cells pin the same forms for 148, 152 and 212 SMs) has one
device run (2026-11-10 UTC; x86 host, Python 3.10, torch 2.13.0+cu130, CUDA 13.0, driver API 13010, cuDNN 92700, nvidia-cutlass-dsl 4.8.0,
tvm-ffi 0.1.13.post3; in-process with `CUDNN_TEST_NO_ISOLATION=1` and `CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1`; on the tree of the pull request
that added the cc 10.7 record below: the sources of `8e31a5725` with that pull request's comment, status-cell pin and documentation edits):
`cd test/python && pytest
moe/frost/test_shared_expert_decode.py moe/frost/test_shared_expert_decode_determinism.py linear_attention/frost/test_gdn_support_matrix_docs.py
-v -ra --durations=10` -> `test_shared_expert_decode.py` 197 passed, 0 skipped; `test_shared_expert_decode_determinism.py` 19 passed, 8
skipped, with the per-method counts of the 152-SM part (the 8 golden cases skip there too: no set for that runtime);
`test_gdn_support_matrix_docs.py` 3 passed.  cc 12.x declined by type.

cc 10.7 record (2026-11-10 UTC): a VR200-class part (cc 10.7, 212 SMs; aarch64 host, Python 3.12) with torch 2.14.0+cu130, CUDA 13.0, driver API
13050, cuDNN 92400, nvidia-cutlass-dsl 4.8.0, tvm-ffi 0.1.15.post1 (the `runtime` block of the goldens in `73840df1b366af2c/`); in-process
pytest runs (`CUDNN_TEST_NO_ISOLATION=1`), one fresh process per command, with `CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1` and
`CUDNN_FRONTEND_DISABLE_COMPILED_CACHE=1`, from a clean tree at `868b3cae4` (#2369); between that commit and `8e31a5725`, the `develop` head
this record was written on, `python/cudnn/moe` and `test/python/moe` differ only in the comment and the status-cell pin of #2392.  From
`test/python`: `pytest moe/frost/test_shared_expert_decode.py -v -ra --durations=15` -> 197 passed, 0 skipped.  `pytest
moe/frost/test_shared_expert_decode_determinism.py -v -ra --durations=15` against the goldens committed at that commit -> 19 passed, 8 skipped:
`test_replay` 6, `test_replay_on_a_fresh_entry_and_fresh_operands` 2, `test_batch_invariance` 4,
`test_duplicated_token_and_neighbour_isolation` 2, `test_row_strided_views_write_the_contiguous_bytes_and_leave_the_inputs_unmodified` 4 and
`test_twenty_four_to_twenty_five_is_the_typed_decline` 1 passed; every case of `test_golden` (4) and of
`test_v1_arm_reproduces_the_v1_goldens` (4) skipped with "no golden for ..." (no set for this runtime key).  With the replay, batch-invariance
and reference cases passed, the set was recorded into a new directory, `CUDNN_DETERMINISM_RECORD=<new directory> pytest
moe/frost/test_shared_expert_decode_determinism.py -v -ra -k golden` -> 8 passed, 19 deselected, and checked strictly,
`CUDNN_DETERMINISM_GOLDENS=<that directory> pytest moe/frost/test_shared_expert_decode_determinism.py -v -ra` -> 27 passed, 0 skipped.  The 8
files are committed as `73840df1b366af2c/`.  Reason for the new set: a new runtime key (GPU name, compute capability, SM count and driver API
version differ from the cc 10.3 set's), not a numerical change -- the input and output hashes of all 8 records (`y`, `act`, `gate_part`;
`ksplit2` / `ksplit4` / `ksplit8`, one and two m-tiles, both geometries) equal those of the cc 10.3 set `6451c9d94a2d4210/`.  That equality is
an observation on these two runtimes, not a contract across architectures.  A cc 10.7 part that reports another GPU name, SM count or software
version hashes to another key, and the golden cells skip there.

### Attention gate + quantize decode step: replay, batch invariance and goldens

`attention_gate/frost/test_gate_quant_decode_determinism.py` pins `cudnn.AttentionGateQuantDecode` (the sigmoid output gate of an attention
block fused with the static E4M3 quantization of the following projection's input: bf16 `attn` / `gate` `[rows, width]`, one fp32 scale,
E4M3 `out`; rows 1 .. 160, one launch).  Every comparison is raw bytes over the one output `out`; the step has no backward, no state and no
scratch.  The independent reference is a separate check (`gate_quant_utils.assert_reference`: every byte is the code of one of the two bf16
values that bracket the fp64 gated value, and of the nearest one outside a 2.5-fp32-unit margin around the bracket's midpoint; then the
torch model of the rounding chain, bitwise) and gates the goldens.  The kernel has no reduction, so there is no split or association to
hold fixed across batch layouts; the cells build one operand set and slice or permute it.

| Test method | Level | Parameters and purpose | Cases |
| --- | --- | --- | ---: |
| `test_replay` | L0 | rows 1 / 17 / 160 x width 2048 / 6144, and 160 x 32768 (2,560 CTAs of 256 threads: several waves); eight executions into fresh buffers under side-stream contention; reference check | 7 |
| `test_replay_on_a_fresh_entry_and_fresh_operands` | L0 | width 2048 / 1032; two entries, two fresh identical operand sets | 2 |
| `test_launch_on_a_side_stream_writes_the_same_bytes` | L0 | the same call under a side stream (ordered after the operands' producer stream) vs the current stream | 1 |
| `test_batch_invariance` | L0 | width 2048 / 6144 / 1032 / 1536 x `THREADS` 128 / 256; one 160-row operand set: every row alone (a one-row call) vs inside the 160-row call, vs the first 2 / 3 / 8 / 16 / 17 / 24 / 32 / 64 rows as one call, vs the reversed batch.  At 2048 / 6144 a row is whole CTAs; at 1536 rows share CTAs, at 1032 also warps | 8 |
| `test_insertion_removal_and_reordering` | L0 | width 2048 / 1032 x `THREADS`; a 24-row call, then a row inserted at position 5, rows 3 / 11 / 23 removed, a permutation: every row keeps its bytes | 4 |
| `test_duplicated_row_and_neighbour_isolation` | L0 | width 2048 / 6144 / 1032 / 1536 x `THREADS`; a row twice in one call (both copies = alone); the row among 63 redrawn neighbours | 8 |
| `test_pick_function_moves_no_byte` | L0 | every served thread count vs the default pick; then a pick table that depends on the row count (128 threads up to 8 rows, 256 above): both configs are compiled at plan time, and every row alone writes the bytes it writes inside 2 .. 160 rows on either side of the pick boundary | 1 |
| `test_inputs_are_unmodified` | L0 | `attn`, `gate`, `scale` bytes before and after twelve launches (caller-owned and entry-owned output, the release arm included) | 1 |
| `test_one_sixty_to_one_sixty_one_and_the_empty_call_are_typed_declines` | L0 | 161 rows is a `NotImplementedError`, zero rows a `ValueError`; never a launch, never another provider | 1 |
| `test_golden` | L0 | width 2048 x 3 rows (scale 0.01) and width 6144 x 19 rows (scale 1.0): three executions, the reference, the three knob arms on the same operands, the saved hashes | 2 |

Golden sets: `attention-gate-quant-decode-w2048-rows3` and `attention-gate-quant-decode-w6144-rows19`, the first goldens of this API,
recorded after the replay, batch-invariance and reference cells passed -- in `6451c9d94a2d4210/` for the cc 10.3 runtime below and in
`d2a97fcefba4b867/` for the cc 10.7 runtime below.  The knob arms (`THREADS` 128, `PDL_RELEASE`, both) write the default's bytes inside
`test_golden` and, on more shapes, in `test_gate_quant_decode.py` (`test_bit_neutral_knob_arms_write_the_default_bytes`), so the one set
holds for every served config.  The release arm is compared with no dependent grid on the stream; a chain with a dependent launched under
the programmatic-stream-serialization attribute is not covered here.  The graph-capture replay (bitwise with eager, operands changed in
place) is in `test_gate_quant_decode.py`.  The input and output hashes of the two records are equal between the cc 10.3 and the cc 10.7
set: an observation on these two runtimes, not a contract across architectures.

GPU coverage: cc 10.3 and cc 10.7 (replay, batch invariance and goldens); cc 10.0 has not been run on a device (unverified); cc 9.0 and
cc 12.x are declined by type.  Every run below is an in-process pytest run (`CUDNN_TEST_NO_ISOLATION=1`), one fresh process per command
with empty CuTe DSL and driver compile caches, from `test/python` of a clean tree at the commit that carries the goldens.

cc 10.3 record (2026-11-11 UTC): a GB300 (cc 10.3, 152 SMs; aarch64 host, Python 3.12) with torch 2.14.0+cu130, CUDA 13.0, driver API 13030,
cuDNN 92400, nvidia-cutlass-dsl 4.8.0 (compiling with CUDA 13.4), tvm-ffi 0.1.15.post1 (the `runtime` block of the goldens in
`6451c9d94a2d4210/`).  `pytest attention_gate/frost/test_gate_quant_decode.py core/frost/test_frost_kernel_name_prefix.py -q -ra` -> 147
passed, 0 skipped.  `CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" pytest attention_gate/frost/test_gate_quant_decode_determinism.py
-q -ra` -> 35 passed, 0 skipped (the case counts of the table).  `pytest -m L2 attention_gate/frost/test_gate_quant_decode_census.py -q -s`
-> 1 passed.  Before the set existed the same determinism command without the goldens variable read 33 passed, 2 skipped (both
`test_golden` cases, "no golden for ..."); the set was then recorded into a new directory, `CUDNN_DETERMINISM_RECORD=<new directory> pytest
attention_gate/frost/test_gate_quant_decode_determinism.py -q -ra -k golden` -> 2 passed, 33 deselected.

CUDA 12.9 compiler flavour on the same part and runtime (2026-11-11 UTC; the CuTe DSL 4.8.0 install without its CUDA 13 toolchain
libraries, so the DSL binds its CUDA 12 flavour: `CUDA_VERSION` 12.9; torch and the driver stay CUDA 13 -- this isolates DSL compatibility
and is not a deployment that uses CUDA 12 throughout): the same three commands -> 146 passed, 1 skipped (`test_rendering_pins_the_convention`
for `sm_107a`: that compiler rejects the target, and the cell skips with the entry's own decline); 35 passed, 0 skipped -- the golden cases
pass against the SAME set, whose runtime key does not name the compiler flavour, so the two flavours write the same bytes for those cases;
1 passed.

CuTe DSL 4.8.1 on the same part (2026-11-11 UTC; the 4.8.1 wheels put first on the path of the same runtime, compiling with CUDA 13.4):
the functional command -> 147 passed, 0 skipped; the determinism command WITHOUT the goldens variable -> 33 passed, 2 skipped (both
`test_golden` cases: the DSL version is part of the runtime key and no set is committed for that key, so the golden comparison is a
coverage gap there, not a pass); the census -> 1 passed.  A record into a new directory under that runtime (2 passed, 33 deselected) holds
the input and output hashes of the committed 4.8.0 set; it is not committed.  With only the CUDA 12 toolchain libraries of 4.8.1
(`CUDA_VERSION` 12.9): 146 passed, 1 skipped (the `sm_107a` rendering cell, as under 4.8.0); 33 passed, 2 skipped (the golden cases, same
reason); 1 passed; its record holds the same hashes (that run's tree differed from the commit in this report's text only).  Cold compile on an x86 host (fake tensors, empty caches): under 4.8.0 and under
4.8.1 the kernel's four arms render byte-identical cubins for `sm_100a`, `sm_103a` and `sm_107a` with the CUDA 13.4 compiler and for
`sm_100a` / `sm_103a` with the CUDA 12.9 compiler; `sm_107a` does not compile under CUDA 12.9 in either version (the entry's typed
decline).  Not run on a device: 4.8.1 on cc 10.7.

cc 10.7 record (2026-11-11 UTC): a VR200-class part (cc 10.7, 212 SMs; aarch64 host, Python 3.12) with torch 2.14.0+cu130, CUDA 13.0, driver
API 13050, cuDNN 92400, nvidia-cutlass-dsl 4.8.0 (compiling with CUDA 13.4), tvm-ffi 0.1.15.post1 (the `runtime` block of the goldens in
`d2a97fcefba4b867/`).  The same three commands -> 147 passed, 0 skipped; 35 passed, 0 skipped; 1 passed.  Before its set existed the
determinism command read 33 passed, 2 skipped, and the set was recorded the same way (2 passed, 33 deselected).  Reason for the second set:
a new runtime key (GPU name, compute capability, SM count and driver API version differ), not a numerical change.  A part that reports
another GPU name, SM count or software version hashes to another key, and the golden cases skip there -- a coverage gap, not a pass.

Host-entry change, re-run record (2026-11-11 UTC).  A later commit binds the entry's slab operands with a runtime extent and scopes its
compile and its launch to the entry's device.  The kernel source is unchanged; the compiled kernel's parameter block changes (a pointer
and an extent per slab), and from that commit on the kernel the entry compiles on a device is the rendering the functional module pins,
byte for byte, whatever the first call's row count is.  The functional module gains three cases (147): the declared extents of the
entry's views, the entry's own launch arguments compiled against the pinned rendering, and an entry on a device that is not the caller's
current one (needs two CUDA devices).  The determinism suite and both golden sets are unchanged.  Re-run from a clean tree at that
commit with the three commands above, in the runtimes above:

- cc 10.3 (the GB300 runtime, one visible device), CuTe DSL 4.8.0 compiling with CUDA 13.4: 149 passed, 1 skipped (the two-device case:
  "needs two CUDA devices" -- not covered on cc 10.3); 35 passed, 0 skipped against the unchanged set in `6451c9d94a2d4210/`; 1 passed.
- the same part with the CUDA 12.9 compiler flavour of 4.8.0: 148 passed, 2 skipped (the `sm_107a` rendering case, as before, and the
  two-device case); 35 passed, 0 skipped against the same set; 1 passed.
- cc 10.7 (the VR200-class runtime) with TWO devices visible, CuTe DSL 4.8.0 compiling with CUDA 13.4: 150 passed, 0 skipped -- the
  two-device case ran there: first call, warm calls with both output owners and a side stream of the entry's device, the bytes of the
  same call made with that device current, the caller's device unchanged; 35 passed, 0 skipped against the unchanged set in
  `d2a97fcefba4b867/`; 1 passed.
- CuTe DSL 4.8.1 on the cc 10.3 part (the 4.8.1 wheels put first on the path of the same runtime, as in the 4.8.1 record above; these
  two runs are one commit later, which changes this report's text only).  Compiling with CUDA 13.4: 149 passed, 1 skipped (the
  two-device case); the determinism command WITHOUT the goldens variable 33 passed, 2 skipped (the golden cases: no set is committed for
  that runtime key -- a coverage gap, as before); 1 passed.  With only the CUDA 12 toolchain libraries (CUDA 12.9): 148 passed, 2 skipped
  (the `sm_107a` rendering case and the two-device case); 33 passed, 2 skipped; 1 passed.  A record into a new directory under each
  flavour (2 passed, 33 deselected) holds the input and output hashes of the committed 4.8.0 set; it is not committed.  Not run on a
  device: 4.8.1 on cc 10.7.
- Cold compile on an x86 host at that commit (a cc 8.0 device: every device case skips), CuTe DSL 4.8.0 and 4.8.1: 65 passed with CUDA
  13.4, 64 passed and the `sm_107a` rendering case skipped with CUDA 12.9, under either version; the three rendering cases and the
  launch-argument case are among them.
### FP8 projection for decode rows: replay and batch invariance at every row count, goldens per byte class and token tile

`gemm/frost/projection_decode/test_fp8_projection_decode_determinism.py` pins `cudnn.Fp8ProjectionDecode` (an FP8 E4M3 weight times FP8
E4M3 activations to bf16, 1 .. 32 rows, one launch) at its seven served byte classes: `ProjectionGeometry(4608, 8192)` at `split_k` 4 and
at `split_k` 2, `ProjectionGeometry(8192, 2048)` at `split_k` 2 and -- since 2026-11-13 -- `ProjectionGeometry(5120, 6144)` at `split_k`
4 and at `split_k` 2, `ProjectionGeometry(16384, 5120)` and `ProjectionGeometry(14336, 5120)` at `split_k` 1 (one K slice per weight
tile: no cluster reduction).  The records of 2026-11-11 below are of the first three classes, 227 cases at the time.  Forward only: the
entry is an inference step with no backward entry point, so there is no backward case.  Every comparison is raw bytes over the entry's
one result (`out`, bf16); the independent fp64 reference of the E4M3 operands is a separate check under a per-element bound (half a bf16
unit plus the fp32 accumulation term; the utilities module beside the suite), and operands whose partial sums are exact in fp32 are
compared byte for byte with the host in `test_fp8_projection_decode.py`.  The kernel is specialised on the row count -- one rendering per
row count -- so replay and batch invariance have one case per served row count, not a sampled set.  Operands come from a CPU generator,
so an operand set has the same bytes on every device.

| Test method | Level | Parameters and purpose | Cases |
| --- | --- | --- | ---: |
| `test_replay` | L0 | class x rows 1 .. 32; eight executions on fresh output buffers under side-stream contention; then the fp64 reference under its bound | 224 |
| `test_replay_on_a_fresh_entry_and_fresh_operands` | L0 | class x rows 3 / 12 / 20 (one per token tile); two entries, two identical operand sets drawn afresh | 21 |
| `test_batch_invariance` | L0 | class x R 2 .. 32; the R-row call replayed three times; every row equals the same row launched alone (one row: the 8-row token tile, accumulator column 0); the reversed batch and a shuffled batch | 217 |
| `test_inserting_or_removing_a_row_keeps_the_other_rows_bytes` | L0 | class x 8 / 16 / 31 rows; one new row inserted at the head, in the middle and at the tail (8 -> 9 and 16 -> 17 change the token tile, 31 -> 32 reaches the ceiling): the original rows keep their bytes, the inserted row writes its alone bytes; read from the larger batch it is the removal | 21 |
| `test_duplicated_row_and_redrawn_neighbours` | L0 | class; a row twice in one batch (both copies = alone); the other rows redrawn around it at the head, middle and tail of 8-, 16- and 32-row batches | 7 |
| `test_pick_arms_cover_other_ring_depths_at_every_class` | L0 | host only: the arms of the next case hold at least two ring depths other than the pick's per class, none names a split, each validates on the reference part; a class that runs its table row as written has its early-release twin, both weight tiles and both fixed depths at every row count | 1 |
| `test_pick_is_byte_neutral` | L0 | class; for every row count 1 .. 32, this part's pick against the form the pick returns on modelled parts (148, 212 and 64 SMs; 100 KB of shared memory per SM) and against every band of the pick's table, each run as the entry's `form`; a class that runs its table row as written also against its fixed arms (rings of two and of four stages, the other weight tile) and, through the band of its one row, against its early-release twin | 7 |
| `test_inputs_are_not_modified` | L0 | class; every operand holds its bytes after six launches (the row-alone form and both sides of each token-tile boundary) | 7 |
| `test_thirty_two_to_thirty_three_rows_is_the_typed_decline` | L0 | the row ceiling is a `NotImplementedError` at the launch and at the plan-time compile, never another provider | 1 |
| `test_golden` | L0 | class x rows 3 / 12 / 20; three executions on a fresh entry, the reference, the saved hashes (`fp8-projection-decode-n<OUT>-k<IN>-split<S>-tile<T>`) | 21 |

Collected: 527 cases (526 on a device, 1 host-only).  Replay executions: `test_replay` 224 x 8 = 1,792; `test_batch_invariance` 217 x 3 of
the batch, 217 x 2 of the permuted batches and 7 x 32 of a row alone; `test_golden` 21 x 3.  At the three classes of the 2026-11-11
records: 227 cases (226 on a device, 1 host-only) -- 96, 9, 93, 9, 3, 1, 3, 3, 1 and 9 in the order of the table; `test_replay` 96 x 8 =
768; `test_batch_invariance` 93 x 3 of the batch, 93 x 2 of the permuted batches and 3 x 32 of a row alone; `test_golden` 9 x 3.

The remaining performance knobs (the 128-row weight tile, ring depths from two stages up, 64 / 128 tensor-memory columns, a token tile
wider than the rows need, the weight loader waiting for the predecessor, the late tensor-memory release, the dependent-launch attribute
and the release of the successor) are bitwise with the default in `test_fp8_projection_decode.py`
(`test_performance_arms_write_the_default_bytes`, 105 cases at 8 row counts each: 15 arms per class; 45 at the three classes of the
2026-11-11 records), which also holds the CUDA-graph replay against eager, the warm-launch stress with byte references (2,016 launches
per case, plainly and behind another kernel), the warm stress of the four classes that run their table row as written behind the GPU's
positive control (`test_warm_stress_of_the_classes_run_as_written`: 16 cases of 1,024 back-to-back launches and 1,024 launches behind a
launch that overwrites every SM's shared memory, every output byte compared after every launch) and the cell that prints, without
asserting, how many elements differ between split 4 and split 2 -- two byte classes, each replayed bitwise.

Golden sets, one file per byte class and token tile, each recorded after the replay, batch-invariance and reference cases passed:
9 files of the first three classes, recorded on cc 10.3 into the directory of the runtime that already holds the shared-expert decode
pair's cc 10.3 set (`6451c9d94a2d4210/`), and 21 files of all seven classes, recorded on cc 10.0 on 2026-11-13 into a new runtime
directory (`c84dd9abaee769b4/`).  A new set is owed with a stated reason when the split, the K tile (128 elements, four K-32
tensor-core instructions) or the cast changes.  The cc 10.3 runtime of `6451c9d94a2d4210/` therefore has no file for the 12
`test_golden` cases of the four classes added on 2026-11-13 (4 classes x rows 3 / 12 / 20): there they skip by default and fail under
`CUDNN_DETERMINISM_GOLDENS` -- the strict check fails a missing file -- until a set of those classes is recorded on that runtime.  On
a tree that carries the four classes, the strict command of the cc 10.3 record below reports those 12 cases as failed: missing files,
not changed bytes.  The 9 cases of the first three classes compare against their unchanged files.

GPU coverage.  The first three classes: cc 10.3 (replay, batch invariance, goldens), cc 10.7 (replay and batch invariance; no golden
set is committed for a cc 10.7 runtime, so the golden comparison is not covered there) and, since 2026-11-13, cc 10.0 (replay, batch
invariance, goldens).  The four classes added on 2026-11-13: cc 10.0 only (replay, batch invariance, goldens); they were not run on
cc 10.3 or cc 10.7, where the entry admits them and the status page reads Unverified.  Until 2026-11-13 cc 10.0 had not been run and
the status page read "SM100: Unverified".  Every run of the 2026-11-11 records is one in-process pytest run per command (`-n 0`, or an
environment without pytest-xdist), one visible device, with `CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1`,
`CUDNN_FRONTEND_DISABLE_COMPILED_CACHE=1` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`, from `test/python` of a clean tree: the two runs
before and during recording on the tree before the 9 golden files and the cc 10.7 compiler gate were added, every other run on the
tree that carries them; `<suite>` = `gemm/frost/projection_decode/test_fp8_projection_decode_determinism.py`, `<module>` =
`gemm/frost/projection_decode/test_fp8_projection_decode.py`.

cc 10.3 record (2026-11-11 UTC): a GB300 (cc 10.3, 152 SMs; aarch64 host, Python 3.12) with torch 2.14.0+cu130, CUDA 13.0, driver API
13030, nvidia-cutlass-dsl 4.8.0 compiling with CUDA 13.4, in two environments.  (a) The goldens' runtime -- cuDNN 92400, tvm-ffi
0.1.15.post1 (the `runtime` block of the files in `6451c9d94a2d4210/`): `pytest <suite> -m L0 -q -rsP` before any set existed -> 218
passed, 9 skipped (the 9 `test_golden` cases, "no golden for ..."); `CUDNN_DETERMINISM_RECORD=<new directory> pytest <suite> -m L0 -k
golden` -> 9 passed, 218 deselected; the 9 records were inspected (the runtime block equals the directory's; shapes and dtypes; the
operand hashes equal those of the operands rebuilt on an x86 host from the suite's CPU generator in 9 of 9 files; the two splits of
`ProjectionGeometry(4608, 8192)` share their operand hashes and differ in the result hash at all three token tiles) and copied;
`CUDNN_DETERMINISM_GOLDENS=$PWD/determinism_goldens pytest <suite> -m L0 -v -rs` -> 227 passed, 0 skipped: `test_replay` 96,
`test_replay_on_a_fresh_entry_and_fresh_operands` 9, `test_batch_invariance` 93,
`test_inserting_or_removing_a_row_keeps_the_other_rows_bytes` 9, `test_duplicated_row_and_redrawn_neighbours` 3,
`test_pick_arms_cover_other_ring_depths_at_every_class` 1, `test_pick_is_byte_neutral` 3, `test_inputs_are_not_modified` 3,
`test_thirty_two_to_thirty_three_rows_is_the_typed_decline` 1, `test_golden` 9.  `test_pick_is_byte_neutral` compared 80 / 40 / 40 (row
count, form) cells at the three classes -- ring depths 1, 2, 3, 4, 7 and 11 (one and two waves) at `(4608, 8192)` split 4, 5 and 7 at
split 2, 3, 4 and 5 at `(8192, 2048)` -- all bitwise with the part's pick.  (b) cuDNN 92700, tvm-ffi 0.1.14.post0 (another runtime key:
no golden set): `pytest <suite> -m L0 -q -rs -n 0` -> 218 passed, 9 skipped (the 9 golden cases); `pytest <module>
core/frost/test_frost_kernel_name_prefix.py -m L0 -q -rs -n 0 -k "not performance_arms"` -> 198 passed, 45 deselected, and `pytest <module>
-m L0 -q -rs -n 0 -k performance_arms` -> 45 passed, 194 deselected: the module 239 passed, 0 skipped, the symbol-prefix test 4 passed.

The other compiler flavor of the DSL on cc 10.3 (2026-11-11 UTC): nvidia-cutlass-dsl 4.8.0 bound to its CUDA 12.9 compiler flavor
(`cutlass.base_dsl.version_info.CUDA_VERSION` = 12.9) inside environment (a), under the same CUDA 13 driver, torch and cuDNN -- a
statement about the DSL's compiler flavor, not about a CUDA 12 deployment.  `CUDNN_DETERMINISM_GOLDENS=$PWD/determinism_goldens pytest
<suite> -m L0 -q -rs` -> 227 passed, 0 skipped: the kernels compiled by the CUDA 12.9 flavor write the bytes of the goldens recorded
under the CUDA 13.4 flavor.  `pytest <module> core/frost/test_frost_kernel_name_prefix.py -m L0 -q -rs` -> 243 passed, 0 skipped (the
case that launches the kernels in a child interpreter, `test_launch_matrix_completes_in_a_child_process`, compiles there with the
default flavor).

cc 10.7 record (2026-11-11 UTC): a VR200-class part (cc 10.7, 212 SMs; aarch64 host, Python 3.12) with torch 2.14.0+cu130, CUDA 13.0,
driver API 13050, cuDNN 92700, nvidia-cutlass-dsl 4.8.0 compiling with CUDA 13.4, tvm-ffi 0.1.14.post0.  `pytest <suite> -m L0 -v -rs
-n 0` -> 218 passed, 9 skipped (the 9 golden cases, "no golden for ...": no set for this runtime); `pytest <module>
core/frost/test_frost_kernel_name_prefix.py -m L0 -q -rs -n 0 -k "not performance_arms"` -> 198 passed, 45 deselected (`<module>` 194,
the symbol-prefix test 4), and `pytest <module> -m L0 -q -rs -n 0 -k performance_arms` -> 45 passed, 194 deselected: the module 239
passed, 0 skipped.  With cuDNN 92400 and tvm-ffi 0.1.15.post1 on the same part, a candidate set was recorded into a new directory (9
passed, 218 deselected) and the suite was checked strictly against it (227 passed, 0 skipped); the operand and result hashes of its 9
records equal those of the cc 10.3 set.  That equality is an observation on these two runtimes, not a contract across architectures;
the candidate is not committed, so a cc 10.7 run of the suite skips its 9 golden cases.

The other compiler flavor of the DSL on cc 10.7 is a DECLINE, not a lane: the DSL's CUDA 12.9 compiler has no target for the part (the
DSL refuses it from inside the trace), so the entry declines it by name before any compile.  In the cuDNN 92400 environment bound to
that flavor, `pytest <suite> <module> core/frost/test_frost_kernel_name_prefix.py -m L0 -v -rs` -> 73 passed, 397 skipped: every case
that needs a compiled kernel skips with the entry's message ("... on compute capability 10.7 needs a CuTe DSL that compiles with CUDA
13.4 or newer ..."), `test_compile_follows_the_gate_on_this_device` passes on its declined branch (`compile` raises that message, nothing
is traced), the host-only cases and the symbol-prefix test pass.  Before the gate existed the same command failed 394 cases inside the
DSL; replay and batch invariance on cc 10.7 are therefore evidence for the CUDA 13.4 flavor only.

cc 10.0 record (2026-11-13 UTC), all seven classes: a cc 10.0 part with 148 SMs (GPU name reported as `NVIDIA Graphics Device`; x86
host, Python 3.10) with torch 2.13.0+cu130, CUDA 13.0, driver API 13010, cuDNN 92700, nvidia-cutlass-dsl 4.8.0 compiling with CUDA 13.4,
tvm-ffi 0.1.13.post3 -- the `runtime` block of the files in `c84dd9abaee769b4/`.  The GPU is shared with other users: these are
functional runs, and no timing was taken on it.  One in-process pytest run per command (`-n 0`), one visible device,
`CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`, from `test/python` of a clean tree; from the recording
on also `CUDNN_FRONTEND_DISABLE_COMPILED_CACHE=1` and empty compile caches per process.  `<suite>` and `<module>` as above.  `pytest
<suite> core/frost/test_frost_kernel_name_prefix.py -m L0 -q -rs -n 0` before a set existed for this runtime -> 510 passed, 21 skipped
(the 21 `test_golden` cases, "no golden for ..."); `CUDNN_DETERMINISM_RECORD=<new directory> pytest <suite> -m L0 -q -rs -n 0 -k
golden` -> 21 passed, 506 deselected; the 21 records were inspected (one runtime block for the set; shapes and dtypes; the operand
hashes equal those of the operands rebuilt on an x86 host from the suite's CPU generator in 21 of 21 files; the two splits of
`ProjectionGeometry(4608, 8192)` and of `ProjectionGeometry(5120, 6144)` share their operand hashes and differ in the result hash at
all three token tiles; 21 distinct result hashes) and copied; on the tree that carries them,
`CUDNN_DETERMINISM_GOLDENS=$PWD/determinism_goldens pytest <suite> core/frost/test_frost_kernel_name_prefix.py -m L0 -q -rs -n 0` ->
531 passed, 0 skipped: the suite 527 (`test_replay` 224, `test_replay_on_a_fresh_entry_and_fresh_operands` 21, `test_batch_invariance`
217, `test_inserting_or_removing_a_row_keeps_the_other_rows_bytes` 21, `test_duplicated_row_and_redrawn_neighbours` 7,
`test_pick_arms_cover_other_ring_depths_at_every_class` 1, `test_pick_is_byte_neutral` 7, `test_inputs_are_not_modified` 7,
`test_thirty_two_to_thirty_three_rows_is_the_typed_decline` 1, `test_golden` 21) and the symbol-prefix test 4.
`test_pick_is_byte_neutral` compared 80 / 40 / 40 / 144 / 160 / 144 / 144 (row count, form) cells at the seven classes in the order of
the first paragraph, all bitwise with the part's pick: ring depths 1, 2, 3, 4, 7 and 11 (one and two waves), 5 and 7, and 3, 4 and 5
at the first three classes; 2, 4, 6 and 7 at `ProjectionGeometry(5120, 6144)` split 4 and 2, 4, 7, 9, 10 and 11 at its split 2 (one
and two waves each), 2, 4, 5 and 8 at the two split-1 classes (one wave), with both weight tiles and both tensor-memory release
orders at every row count of these four.  `pytest <module> -m L0 -q -rs -n 0 -k "not performance_arms"` -> 417 passed, 1 skipped
(the case that needs two CUDA devices), 105 deselected, and `pytest <module> -m L0 -q -rs -n 0 -k performance_arms` -> 105 passed,
418 deselected: the module 522 passed, 1 skipped; nine launch-fact cells compared equal with the profiler's record of a launch, and
the warm stress of the four classes that run their table row as written read 0 wrong launches in 16 of 16 cases behind a control that
reproduced in 200 of 200 launches (11 of the 16 shapes run fewer than two thread blocks per SM, where a zero bounds little).  The nine
cc 10.0 result hashes of the first three classes equal those of the cc 10.3 set -- an observation on these two runtimes, not a
contract across architectures.

The other compiler flavor of the DSL on cc 10.0 (2026-11-13 UTC): nvidia-cutlass-dsl 4.8.0 bound to its CUDA 12.9 compiler flavor
(`cutlass.base_dsl.version_info.CUDA_VERSION` = 12.9) on the same part, under the same CUDA 13 driver, torch and cuDNN -- a statement
about the DSL's compiler flavor, not about a CUDA 12 deployment.  `CUDNN_DETERMINISM_GOLDENS=$PWD/determinism_goldens pytest <suite>
core/frost/test_frost_kernel_name_prefix.py -m L0 -q -rs -n 0` -> 531 passed, 0 skipped: at all seven classes the kernels compiled by the
CUDA 12.9 flavor write the bytes of the goldens recorded under the CUDA 13.4 flavor.  `pytest <module> -m L0 -q -rs -n 0 -k "not
performance_arms"` -> 417 passed, 1 skipped, and `pytest <module> -m L0 -q -rs -n 0 -k performance_arms` -> 105 passed: the module 522
passed, 1 skipped.  The cases that launch the kernels in child interpreters (`test_launch_matrix_completes_in_a_child_process`) compile
there with the default flavor.

A second DSL version, compile only (2026-11-13 UTC): on an x86 host without a cc 10.x device, nvidia-cutlass-dsl 4.8.1 with its CUDA
13.4 compiler rendered 147 forms for the `sm_100a`, `sm_103a` and `sm_107a` targets -- 120 picks of the four classes added on
2026-11-13 (rows 1, 6, 9, 17 and 32, with and without the dependent-launch attribute) and 27 picks of the first three classes -- and
every kernel binary and PTX is byte-identical to the 4.8.0 rendering of the same form (147 of 147).

A second DSL version on cc 10.0 (2026-11-13 UTC): nvidia-cutlass-dsl 4.8.1 on the same part and tree, torch, driver and cuDNN as
above -- another runtime key, for which no golden set is committed.  With its CUDA 13.4 compiler: `CUDNN_DETERMINISM_RECORD=<new
directory> pytest <suite> core/frost/test_frost_kernel_name_prefix.py -m L0 -q -rs -n 0` -> 531 passed (the 21 golden cases record
instead of comparing); the 21 records hold the input and output hashes of the committed 4.8.0 set in 21 of 21 files and are not
committed; `pytest <module> -m L0 -q -rs -n 0 -k "not performance_arms"` -> 417 passed, 1 skipped, and `-k performance_arms` -> 105
passed.  With its CUDA 12.9 compiler flavor: the same determinism command -> 531 passed, and its 21 records hold the committed
hashes too; the functional module was not run in that combination.

### Add + RMSNorm + quantize decode step: replay, batch invariance and goldens per numerical path and residual mode

`norm/frost/test_add_norm_quant_decode_determinism.py` pins `cudnn.AddNormQuantDecode` (the residual add, the RMSNorm and the quantization
of the next projection's input in one launch: bf16 `x` / `residual` `[rows, 5120]`, a bf16 weight, one fp32 scale; bf16 `out`, FP8 E4M3
or packed E2M1 `codes`, E4M3 block `scales`; rows 1 .. 1024).  Forward only: the entry is an inference step with no backward entry point,
no gradient and no state, so there is no backward case.  Every comparison is raw bytes over every buffer a launch writes: the residual
(in / out under the add, out under the store), `out`, `codes` and the NVFP4 `scales`.  The suite runs six of the entry's fifteen forms,
one per numerical path and per residual mode: the add with bf16 and E4M3 codes, with bf16 and NVFP4 codes, and with bf16 alone; the
residual store with E4M3 codes; no residual with E4M3 codes and with bf16 and NVFP4 codes.  The other nine forms select which buffers
are written, not what is computed: `test_add_norm_quant_decode.py` runs all fifteen against the references and compares each form that
does not write the bf16 value with the form that does, byte for byte.

What is held fixed: the row's sum of squares is one tree whose shape depends on the width alone, so no split, pick or knob selects
another association and there is nothing to pin across batch layouts besides the operands.  Operands come from a CPU generator (an
operand set has the same bytes on every device) and are built once per case; the cases slice and permute them.  The residual is
in / out: every execution runs on a fresh copy.  Every output is a fresh caller-owned buffer filled with a sentinel (the NaN pattern of
bf16 and of E4M3, `0xA5` for the packed codes; the scale buffer zero-filled, as its owner must).  A block scale's position in the 128 x 4
scale layout follows its row index while its value does not, so batch layouts are compared on the scale bytes per row, read through the
layout rule; the replay and golden cases compare the whole scale buffer.  The independent references are a separate check
(`add_norm_quant_utils.assert_reference`: every `out` element one of the two bf16 neighbours of an fp64 evaluation; the residual, the
E4M3 codes and the NVFP4 block scales bitwise the host model on the kernel's own bf16 value; every NVFP4 code inside the bracket of the
two approximate reciprocals) and gate the goldens.

| Test method | Level | Parameters and purpose | Cases |
| --- | --- | --- | ---: |
| `test_replay` | L0 | form x rows 1 / 17 / 160 / 1024; eight executions into fresh buffers under side-stream contention; then the references.  1024 rows are 1,024 thread blocks of 640 threads, three per SM: more than one scheduling wave | 24 |
| `test_replay_on_a_fresh_entry_and_fresh_operands` | L0 | form; two entries, two identical 24-row operand sets drawn afresh | 6 |
| `test_launch_on_a_side_stream_writes_the_same_bytes` | L0 | launch arm (plain, wait, wait + release); the add with bf16 and NVFP4 codes at 17 rows under a side stream (ordered after the operands' producer stream) vs the current stream | 3 |
| `test_batch_invariance` | L0 | form; one 1024-row operand set: the 1024-row call replayed three times and checked against the references; every row alone (a one-row call) vs inside that call, vs the first 2 / 3 / 8 / 16 / 17 / 24 / 32 / 64 / 127 / 128 / 129 / 160 rows as one call (both sides of the 128-row scale-tile edge), vs the reversed batch | 6 |
| `test_insertion_removal_and_reordering` | L0 | form; a 130-row call (two scale tiles), then a row inserted at position 5 (row 127 moves into the second tile), rows 3 / 11 / 129 removed (rows 128 / 129 move into the first tile), a permutation: every row keeps its bytes, the inserted row writes its alone bytes | 6 |
| `test_duplicated_row_and_neighbour_isolation` | L0 | form; a row twice in one call (both copies = alone); the row at position 5 among 63 and at position 128 among 159 redrawn neighbours | 6 |
| `test_inputs_are_unmodified` | L0 | every form x launch arm x rows 1 / 17 / 1024, caller-owned and entry-owned outputs (108 launches): `x`, `weight` and both scales hold their bytes; the residual is bitwise `bf16(x + residual)` under the add and `x` under the store | 1 |
| `test_crosses_a_scheduling_wave` | L0 | the add with bf16 and E4M3 / NVFP4 codes; the premise (1024 rows exceed three thread blocks per SM on this part) is asserted first; eight executions at 1024 rows; the rows on both sides of every wave edge alone vs inside | 2 |
| `test_row_ceiling_is_the_typed_decline` | L0 | 1025 rows is a `NotImplementedError`, zero rows a `ValueError`; never a launch (the residual is untouched), never another provider | 1 |
| `test_golden` | L0 | six cases: three executions, the references, the wait and the wait + release arms on the same operands, the saved hashes of the residual and of every output | 6 |

Collected: 61 cases, all on a device.  Replay executions: `test_replay` 24 x 8 = 192; `test_batch_invariance` 6 x 3 of the 1024-row call
and 6 x 1,024 of a row alone; `test_crosses_a_scheduling_wave` 2 x 8; `test_golden` 6 x 3.

Golden sets: `add-norm-quant-decode-h5120-<case>`, six files in `c84dd9abaee769b4/`, the first goldens of this API -- `add-bf16-e4m3-rows3`,
`store-e4m3-rows3` and `none-e4m3-rows3` (the three residual modes with static E4M3 codes), `add-nvfp4-rows3` and `add-nvfp4-rows130`
(NVFP4 codes in one and in two scale tiles) and `add-bf16-nvfp4-ties-rows2` (rows of `x = +-1` whose weight places the normalized value
on chosen bf16 values: two blocks get the block scales 1.5 and 1.75 and fourteen elements each on an exact E2M1 rounding tie).  The
launch arms write the default's bytes inside `test_golden` and, on more shapes, in `test_add_norm_quant_decode.py`
(`test_bit_neutral_arms_write_the_default_bytes`; behind a predecessor that released its dependents before it wrote the operands,
`test_wait_holds_every_read_behind_a_predecessor_that_released_early`), so the one set holds for every arm.  The tie case's
codes follow the part's approximate reciprocal: its hashes are a record of this runtime, not a claim across GPU architectures.  A new
set is owed with a stated reason when the sum tree, the rounding chain or the scale layout changes.

GPU coverage: cc 10.0 (replay, batch invariance and goldens).  cc 10.3 and cc 10.7 have not been run on a device: the entry admits
compute capability 10.x and the status page reads "SM103: Unverified" and, for SM107, "Unverified"; a run there skips the six golden
cases (no set for its runtime) -- a coverage gap, not a pass.  cc 9.0 and cc 12.x are declined by type.  Every run below is one
pytest run per command on one visible device, from `test/python`; `<suite>` = `norm/frost/test_add_norm_quant_decode_determinism.py`,
`<module>` = `norm/frost/test_add_norm_quant_decode.py`.  A command on `<suite>` alone is an in-process run
(`CUDNN_TEST_NO_ISOLATION=1`); a command that names `<module>` runs under the suite's default per-test isolation.  The device is a
shared development part: these are correctness runs, and no timing was taken.

cc 10.0 record (2026-11-13 UTC): a cc 10.0 part with 148 SMs (the device reports the name `NVIDIA Graphics Device`; x86 host, Python
3.10) with torch 2.13.0+cu130, CUDA 13.0, driver API 13010, cuDNN 92700, nvidia-cutlass-dsl 4.8.0 (compiling with CUDA 13.4), tvm-ffi
0.1.13.post3 (the `runtime` block of the goldens in `c84dd9abaee769b4/`).  Before the set existed: `pytest <suite> -m L0 -q -ra` -> 55
passed, 6 skipped (the six `test_golden` cases, "no golden for ...").  The set was then recorded from a clean tree at the commit that
adds the suite, into a new directory: `CUDNN_DETERMINISM_RECORD=<new directory> pytest <suite> -m L0 -q -ra -k golden` -> 6 passed, 55
deselected.  The six records were inspected before the copy: one runtime block whose key is the directory's name; shapes and dtypes;
the operand hashes equal those of the operands rebuilt on an x86 host from the suite's CPU generator in 6 of 6 files; the updated
residual's hash equals that of the host's `bf16(x + residual)` in the four add cases and that of `x` in the store case; the tie case's
`out` hash equals the host construction's and its scale-buffer hash the host model's laid out by the layout rule; the two 3-row add
cases share their operands and their updated residual; the store and no-residual cases share `x` and differ in their codes (scales 0.02
and 1.0).

Strict check and the functional module, from a clean tree with the package, test and golden files of the branch's last commit that
changes any of them (2026-11-14 UTC, the runtime above; the six golden files are unchanged since they were recorded).
`CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" pytest <suite> -m L0 -q -ra` -> 61 passed, 0 skipped: `test_replay` 24,
`test_replay_on_a_fresh_entry_and_fresh_operands` 6, `test_launch_on_a_side_stream_writes_the_same_bytes` 3, `test_batch_invariance` 6,
`test_insertion_removal_and_reordering` 6, `test_duplicated_row_and_neighbour_isolation` 6, `test_inputs_are_unmodified` 1,
`test_crosses_a_scheduling_wave` 2, `test_row_ceiling_is_the_typed_decline` 1, `test_golden` 6.  `pytest <module>
core/frost/test_frost_kernel_name_prefix.py -m L0 -q -ra` -> 190 passed, 1 skipped: the
module 186 passed and 1 skipped (`test_entry_on_another_device_launches_there_and_keeps_the_callers_device_current`: "two CUDA devices
are required" -- one device is visible, so an entry on a device that is not the caller's current one is not covered on cc 10.0), the
symbol-prefix test 4 passed.  The module's launch-facts case compared the profiler's record of a launch with the entry's facts in 12 of
12 cells (one kernel per call; grid `(rows, 1, 1)`, block `(640, 1, 1)`, 80 B of shared memory).  The module's case behind a
predecessor that released early (`test_wait_holds_every_read_behind_a_predecessor_that_released_early`) compared 16 of 16 launches of the
step equal to the plain step, behind its positive control: a dependent without a wait read other bytes than the written ones in 4 of 4
eager launches and 4 of 4 replays.  On a part where that control does not reproduce the case skips -- a coverage gap, not a pass.
The module's overflow case (`test_rows_whose_sum_of_squares_overflows_normalize_to_zeros`: a finite row whose sum of squares exceeds
the largest fp32 value writes signed zeros, the signed zero's codes and zero block scales, and its neighbours keep their bytes) is one
of the 186.  So are the eleven cases of an operand that overlaps an entry-owned output (an output the call does not pass): ten on the
host with the launch intercepted (`test_declines_an_operand_that_overlaps_an_entry_owned_output`: six declines -- among them a tensor
an earlier call returned, passed back as the residual -- and four controls that stay served) and one on the device
(`test_returned_output_as_the_residual_is_declined_before_the_launch`: the declined call leaves the tensor's bytes as they were, and
with a caller-owned `out` the same residual is served and `out` holds the bytes of a call on separate tensors).  The same command with
`<suite>` added, on an x86 host with a cc 8.0 device (every device case skips): 112 passed, 140 skipped.

The records of the next three paragraphs (the CUDA 12.9 flavour, CuTe DSL 4.8.1, the cold compile) are from an earlier tree: the
functional module had twelve cases fewer there (the overflow case and the eleven overlap cases did not exist), it did not reset its
captured CUDA graphs, and the package differed from this tree's in docstrings, in the text of two decline messages, in the entry's
overlap check of the outputs a call does not pass, and in one line of the kernel module that binds a name (the kernel's renderings
are byte-identical) only.

CUDA 12.9 compiler flavour on the same part and runtime (2026-11-13 UTC; the CuTe DSL 4.8.0 install without its CUDA 13 toolchain
libraries, so the DSL binds its CUDA 12 flavour: `CUDA_VERSION` 12.9; torch and the driver stay CUDA 13 -- this isolates DSL
compatibility and is not a deployment that uses CUDA 12 throughout): the strict determinism command -> 61 passed, 0 skipped -- the six
golden cases pass against the SAME set, whose runtime key does not name the compiler flavour, so the two flavours write the same bytes
for those cases; the functional command -> 177 passed, 2 skipped (`test_rendering_pins_the_convention` for `sm_107a`: that compiler
rejects the target, and the case skips with the entry's own decline; the two-device case).

CuTe DSL 4.8.1 on the same part (2026-11-13 UTC; the 4.8.1 wheels put first on the path of the same runtime; the same tree as the
CUDA 12.9 runs above).  Compiling with CUDA 13.4: the functional command -> 178
passed, 1 skipped (the two-device case); the determinism command WITHOUT the goldens variable -> 55 passed, 6 skipped (the six
`test_golden` cases: the DSL version is part of the runtime key and no set is committed for that key, so the golden comparison is a
coverage gap there, not a pass).  With only the CUDA 12 toolchain libraries of 4.8.1 (`CUDA_VERSION` 12.9): 177 passed, 2 skipped (the
`sm_107a` rendering case and the two-device case); 55 passed, 6 skipped (the golden cases, same reason).  A record into a new directory
under each flavour (`-k golden`: 6 passed, 55 deselected) holds the input and output hashes of the committed 4.8.0 set in 6 of 6 files;
it is not committed.

Cold compile on an x86 host (2026-11-13 UTC; a cc 8.0 device: every device case skips; fake tensors).  The kernel's 45 renderings per
target (fifteen forms x three launch arms) under 4.8.1 are byte-identical cubins to 4.8.0's for `sm_100a`, `sm_103a` and `sm_107a` with
the CUDA 13.4 compiler (135 of 135) and for `sm_100a` / `sm_103a` with the CUDA 12.9 compiler (90 of 90); `sm_107a` does not compile
under CUDA 12.9 in either version (the entry's typed decline).  `pytest <module> <suite> core/frost/test_frost_kernel_name_prefix.py -m
L0 -q -ra` on that host: 102 passed, 138 skipped under 4.8.0 and under 4.8.1 with CUDA 13.4; 101 passed, 139 skipped under either
version with CUDA 12.9 (the `sm_107a` rendering case skips with the entry's decline).

### Attention preparation decode step: replay, batch invariance and goldens per position form

`attention_prepare/frost/test_attention_prepare_decode_determinism.py` pins `cudnn.AttentionPrepareDecode` (one launch per decode
step: the RMSNorm of the query and key rows with their `(1 + weight)` gain, the rotary rotation of their first 64 elements, the copy
of the gate rows, and the scatter of the prepared key rows and of the value rows into a paged cache through the slot vector; geometry
24 query heads, 4 key/value heads, head width 256; tokens 1 .. 1024).  Forward only: the entry is an inference step with no backward
entry point, no gradient and no state, so there is no backward case.  Every comparison is raw bytes over every buffer a launch writes:
`q_out`, `gate_out` and the two pools, compared as the whole packed cache (canary rows included) or, across batch layouts, per token
as the K and V rows its slot names.  The suite runs one form per position form -- three position rows on a 64-row page with int64
integers, one position row on a 16-row page with int64 positions and int32 slots -- and, in the replay case, three more forms (the
other page size of each position form and the int32 positions).  The page size and the integer dtypes select addresses, not
arithmetic; `test_attention_prepare_decode.py` runs all five forms against the reference.

What is held fixed: a head row's sum of squares is one tree whose shape depends on the head width alone, so no split, pick or knob
selects another association and there is nothing to pin across batch layouts besides the operands.  Operands come from a CPU
generator (an operand set has the same bytes on every device) and are built once per case; the cases slice and permute them, and a
token's head rows, positions and slot move together.  The operands are views, as a decode step holds them: the four row operands are
views of one projection slab and the two pools are the halves of one packed cache.  Every output is a fresh caller-owned buffer
filled with a sentinel and every launch writes a fresh cache filled with a canary (both bf16 NaN patterns no written row holds), so a
reused buffer cannot make a comparison pass.  Two tokens never share a slot: that is the caller's contract, and a case that
duplicates a token gives the copy a slot of its own.  The independent reference is a separate check
(`attention_prepare_utils.assert_twin`: a PyTorch chain with the kernel's rounding points and none of its approximate instructions;
every prepared element inside the stated band of bf16 units; the gate rows, the value rows and every pool row no live slot names
bitwise) and gates the goldens.

| Test method | Level | Parameters and purpose | Cases |
| --- | --- | --- | ---: |
| `test_replay` | L0 | form (five) x tokens 1 / 17 / 160 / 1024; eight executions into fresh buffers under side-stream contention; then the reference.  1024 tokens are 28,672 thread blocks of 128 threads: more than twelve scheduling waves on a 148-SM part | 20 |
| `test_replay_on_a_fresh_entry_and_fresh_operands` | L0 | position form; two entries, two identical 24-token operand sets drawn afresh | 2 |
| `test_launch_on_a_side_stream_writes_the_same_bytes` | L0 | launch arm (plain, wait, release + wait); 17 tokens under a side stream (ordered after the operands' producer stream) vs the current stream | 3 |
| `test_batch_invariance` | L0 | position form; one 1024-token operand set: the 1024-token call replayed three times and checked against the reference; every token alone (a one-token call) vs inside that call, vs the first 2 / 3 / 8 / 16 / 17 / 32 / 64 / 160 tokens as one call, vs the reversed call | 2 |
| `test_reordered_and_duplicated_tokens_and_neighbour_isolation` | L0 | position form; a 130-token call, then a token inserted at position 5, tokens 3 / 11 / 129 removed, a permutation: every token keeps its bytes; a token twice in one call with distinct slots (both copies = alone); the token at position 5 among 63 and at position 128 among 159 redrawn neighbours | 2 |
| `test_slots_decide_pool_rows_and_nothing_else` | L0 | position form; nine tokens under six slot vectors -- all live (both sides of a page edge, the pool's last row), with negative slots, with a slot at the pool's end and one past it, 4 entries, 1 entry, empty; each replayed three times: `q_out` and `gate_out` are byte-identical in all six, a live token's pool rows are those of the all-live call, every other pool row holds its canary | 2 |
| `test_inputs_are_unmodified` | L0 | position form x launch arm x tokens 1 / 17 / 1024, caller-owned and entry-owned outputs (36 launches): the slab, the weights, the table, the positions and the slot vector hold their bytes | 1 |
| `test_crosses_scheduling_waves` | L0 | position form; the premise (the 1024-token grid exceeds one scheduling wave on this part) is asserted first; eight executions at 1024 tokens; the tokens on both sides of every wave edge alone vs inside | 2 |
| `test_row_ceiling_and_the_empty_call_are_typed_declines` | L0 | 1025 tokens is a `NotImplementedError`, zero tokens a `ValueError`; never a launch (the outputs and the pools are untouched), never another provider | 1 |
| `test_golden` | L0 | four cases: three executions, the reference, the wait and the release + wait arms on the same operands, the saved hashes of `q_out`, `gate_out` and the whole packed cache | 4 |

Collected: 39 cases, all on a device.  Replay executions: `test_replay` 20 x 8 = 160; `test_batch_invariance` 2 x 3 of the 1024-token
call and 2 x 1,024 of a token alone; `test_slots_decide_pool_rows_and_nothing_else` 2 x 6 x 3; `test_crosses_scheduling_waves` 2 x 8;
`test_golden` 4 x 3.

Golden sets: `attention-prepare-decode-<form>-<slots>-rows9`, four files in `c84dd9abaee769b4/`, the first goldens of this API --
`3d-p64-i64-i64-full-rows9` and `1d-p16-i64-i32-full-rows9` (nine tokens, one slot per token) and `3d-p64-i64-i64-short-rows9` and
`1d-p16-i64-i32-short-rows9` (the same operands with a slot vector of four entries, one of them negative: three tokens write pool
rows, all nine write `q_out` and `gate_out`).  The operands hold large entries, zero stretches, alternating-sign runs, negative zeros
and weights of -1 and below.  The launch arms write the default's bytes inside `test_golden` and, on more shapes, in
`test_attention_prepare_decode.py` (`test_bit_neutral_arms_write_the_default_bytes`; behind a predecessor that released its
dependents before it wrote the operands, `test_wait_holds_every_read_behind_a_predecessor_that_released_early`), so the one set holds
for every arm.  The prepared values follow the part's approximate division and reciprocal square root: the hashes are a record of
this runtime, not a claim across GPU architectures.  A new set is owed with a stated reason when the sum tree or the rounding chain
changes.

GPU coverage: cc 10.0 (replay, batch invariance and goldens).  cc 10.3 and cc 10.7 have not been run on a device: the entry admits
compute capability 10.x and the status page reads "SM103: Unverified" and, for SM107, "Unverified"; a run there skips the four golden
cases (no set for its runtime) -- a coverage gap, not a pass.  cc 9.0 and cc 12.x are declined by type.  Every run below is one
pytest run per command on one visible device, from `test/python`; `<suite>` =
`attention_prepare/frost/test_attention_prepare_decode_determinism.py`, `<module>` =
`attention_prepare/frost/test_attention_prepare_decode.py`.  The device is a shared development part: these are correctness runs,
and no timing was taken.

cc 10.0 record (2026-11-18 UTC): a cc 10.0 part with 148 SMs (the device reports the name `NVIDIA Graphics Device`; x86 host, Python
3.10) with torch 2.13.0+cu130, CUDA 13.0, driver API 13010, cuDNN 92700, nvidia-cutlass-dsl 4.8.0 (compiling with CUDA 13.4), tvm-ffi
0.1.13.post3 (the `runtime` block of the goldens in `c84dd9abaee769b4/`).  Before the set existed: `pytest <suite> -m L0 -q -ra` -> 35
passed, 4 skipped (the four `test_golden` cases, "no golden for ...").  The set was then recorded from a clean tree with the package
and test files of this change, into a new directory: `CUDNN_DETERMINISM_RECORD=<new directory> pytest <suite> -m L0 -q -ra -k golden`
-> 4 passed, 35 deselected.  The four records were inspected before the copy: one runtime block whose key is the directory's name;
shapes and dtypes; the operand hashes equal those of the operands rebuilt on an x86 host from the suite's CPU generator in 4 of 4
files; the `gate_out` hash equals that of the host's gate rows in 4 of 4; the two cases of a position form share every operand but
the slot vector and share `q_out` and `gate_out`, and their caches differ; the two position forms share the slab and `gate_out` and
differ in `q_out`.

Strict check and the functional module, from a clean tree with the package, test and golden files of the branch's last commit that
changes any of them (2026-11-18 UTC, the runtime above; the four golden files are unchanged since they were recorded).
`CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" pytest <suite> -m L0 -q -ra` -> 39 passed, 0 skipped: `test_replay` 20,
`test_replay_on_a_fresh_entry_and_fresh_operands` 2, `test_launch_on_a_side_stream_writes_the_same_bytes` 3, `test_batch_invariance` 2,
`test_reordered_and_duplicated_tokens_and_neighbour_isolation` 2, `test_slots_decide_pool_rows_and_nothing_else` 2,
`test_inputs_are_unmodified` 1, `test_crosses_scheduling_waves` 2, `test_row_ceiling_and_the_empty_call_are_typed_declines` 1,
`test_golden` 4.  `pytest <module> core/frost/test_frost_kernel_name_prefix.py -m L0 -q -ra` -> 157 passed, 1 skipped: the module 152
passed and 1 skipped (`test_entry_on_another_device_launches_there_and_keeps_the_callers_device_current`: "two CUDA devices are
required" -- one device is visible, so an entry on a device that is not the caller's current one is not covered on cc 10.0), the
symbol-prefix test 5 passed.  The module's launch-facts case compared the profiler's record of a launch with the entry's facts in 12
of 12 cells (one kernel per call; grid `(tokens, 28, 1)`, block `(128, 1, 1)`, 16 B of shared memory).  The module's case on rows
whose rotary products underflow (`test_fused_product_of_a_rotation_is_the_forms_on_underflowing_rows`: head rows and table rows of
bf16 subnormals, where the two product orders of the second rotary half differ in the sign of a zero) compared the rotated
elements of `q_out` and of the written K pool rows bitwise with the zero of each form's fused product at 1, 9 and 160 tokens in 5 of
5 forms; the bits of the other order differ from them in about a quarter of those elements.  The module's case behind a
predecessor that released early (`test_wait_holds_every_read_behind_a_predecessor_that_released_early`) compared 8 of 8 launches of
the step equal to the plain step, behind its positive control: a dependent without a wait read other bytes than the written ones in
4 of 4 eager launches and 4 of 4 replays.  On a part where that control does not reproduce the case skips -- a coverage gap, not a
pass.  The same command with `<suite>` added, on an x86 host with a cc 8.0 device (every device case skips): 116 passed, 81 skipped.

CUDA 12.9 compiler flavour on the same part and runtime (2026-11-18 UTC; the CuTe DSL 4.8.0 install without its CUDA 13 toolchain
libraries, so the DSL binds its CUDA 12 flavour: `CUDA_VERSION` 12.9; torch and the driver stay CUDA 13), from the same clean tree.
The strict command -> 39 passed, 0 skipped against the same four golden files: the two flavours write the same bytes.  The module
command -> 156 passed, 2 skipped: the module 151 passed and 2 skipped (the two-device case; the `sm_107a` rendering case, which
skips with the entry's decline -- that flavour does not assemble the target), the symbol-prefix test 5 passed.  The case behind a
predecessor that released early: 8 of 8 launches equal, its control 4 of 4 and 4 of 4.  The case on rows whose rotary products
underflow: 5 of 5 forms.

Renderings (host, no device; one fresh process per target and flavour, CuTe DSL 4.8.0): all 48 forms of the served set (two position
forms x two page sizes x two position dtypes x two slot dtypes x three launch arms) trace-compile for `sm_100a`, `sm_103a` and
`sm_107a` with the CUDA 13.4 compiler and for `sm_100a` and `sm_103a` with the CUDA 12.9 compiler: 240 renderings, 48 distinct
images per target and flavour.  The PTX text of the two flavours is identical apart from its `.version` line in 48 of 48 forms for
`sm_100a` and in 48 of 48 for `sm_103a`.

### SwiGLU FP4 gate / up product for decode rows: replay and batch invariance at every row count, goldens per token tile

`gemm/frost/swiglu_decode/test_swiglu_fp4_decode_determinism.py` pins `cudnn.SwigluFp4Decode` (the gate / up product of a block-scaled
FP4 MLP with SiLU(gate) x up and the FP4 re-quantization of the result: FP4 E2M1 weight and activation with E4M3 block scales, two fp32
scalars, FP4 E2M1 `out` with E4M3 block scales; 1 .. 64 rows, one launch) at its two served geometries, `SwigluGeometry(5120, 17408)`
and `SwigluGeometry(5120, 128)`.  Forward only: the entry is an inference step with no backward entry point, so there is no backward
case.  Every comparison is raw bytes over the entry's two results: the packed codes `out`, the live rows' block scales gathered from
their positions in the 128 x 4 scale layout and -- where one call is compared with itself -- the whole caller-owned scale band, whose
bytes outside the live rows must keep the sentinel the buffer was filled with.  The independent fp64 reference of the dequantized
operands is a separate check under a derived bound (an interval of admissible bytes per block scale and a bound per code; the
utilities module beside the suite), and operands whose every intermediate is exact are compared byte for byte with a host-built
expectation in `test_swiglu_fp4_decode.py`.  The kernel has no split and no reduction across rows.  One compiled kernel per token
tile (8 / 16 / 32 / 64 tokens) serves every row count the tile holds, so replay has one case per token tile and launch arm, and batch
invariance one case per served row count.  Operands come from a CPU generator, so an operand set has the same bytes on every device;
the activation set alternates ordinary rows with an all-zero row, a row with one large value per block, a row of tiny values, a
saturating row and a row with all-zero blocks, so every batch holds rows that take the zero rule and rows that saturate.

| Test method | Level | Parameters and purpose | Cases |
| --- | --- | --- | ---: |
| `test_replay` | L0 | geometry x rows 1 / 9 / 17 / 33 / 64 (the first row count of each token tile, and the ceiling) x launch attribute off / on; eight executions into fresh buffers under side-stream contention; the sentinel outside the live rows; then the fp64 reference under its bound | 20 |
| `test_replay_across_scheduling_boundaries` | L0 | default geometry x rows 20 / 64 x grid cap 1 (one thread block runs all 272 work tiles), 8 (34 work tiles per block: the staging ring is refilled inside the launch) and twice the SM count (one block per work tile: a second wave); eight executions, then the default entry's bytes | 6 |
| `test_replay_on_a_fresh_entry_and_fresh_operands` | L0 | geometry x rows 3 / 12 / 20 / 40 (one per token tile); two entries, two identical operand sets drawn afresh | 8 |
| `test_launch_on_a_side_stream_writes_the_same_bytes` | L0 | geometry; rows 1 / 9 / 17 / 33 / 64 under a side stream ordered behind the operands' stream against the default stream | 2 |
| `test_batch_invariance` | L0 | geometry x R 2 .. 64; the R-row call replayed three times; every row equals the same row launched alone (one row: the 8-token tile, token position 0); the reversed batch and a shuffled batch | 126 |
| `test_inserting_or_removing_a_row_keeps_the_other_rows_bytes` | L0 | geometry x 8 / 16 / 32 / 63 rows; one new row inserted at the head, in the middle and at the tail (8 -> 9, 16 -> 17 and 32 -> 33 change the token tile, 63 -> 64 reaches the ceiling): the original rows keep their bytes, the inserted row writes its alone bytes; read from the larger batch it is the removal; then a permutation of the grown batch | 8 |
| `test_duplicated_row_and_neighbour_isolation` | L0 | geometry; a row twice in one batch (both copies = alone); the other rows redrawn around it -- another draw, raw random codes and scales, rows whose scale bytes hold NaN codes -- at the head, middle and tail of 8-, 16-, 32- and 64-row batches | 2 |
| `test_token_tile_pick_is_byte_neutral` | L0 | geometry; every row count 1 .. 64 on every token tile that holds it, against the narrowest (56 comparisons over the six tile pairs) | 2 |
| `test_inputs_are_unmodified` | L0 | geometry; every operand holds its bytes after 24 launches (both sides of each token-tile boundary; the launch attribute; a grid cap that refills the staging ring) | 2 |
| `test_sixty_four_to_sixty_five_rows_is_the_typed_decline` | L0 | the row ceiling is a `NotImplementedError` at the launch and at the plan-time compile, never another provider; the declined call writes nothing | 1 |
| `test_golden` | L0 | default geometry x rows 3 / 12 / 20 / 40 (one per token tile) and the small geometry x rows 20; three executions on a fresh entry, the reference, the saved hashes (`swiglu-fp4-decode-h<HIDDEN>-i<INTERMEDIATE>-tile<T>`) | 5 |

Collected: 182 cases, all on a device.  Replay executions: `test_replay` 20 x 8, `test_replay_across_scheduling_boundaries` 6 x 8,
`test_batch_invariance` 126 x 3 of the batch, 126 x 2 of the permuted batches and 2 x 64 of a row alone; `test_golden` 5 x 3.

The remaining performance knobs (no pre-wait stages, the default L2 policy of the weight stream, a grid cap of one block, of a partial
wave, of the part's SM count and of twice that, the launch attribute) are bitwise with the default in `test_swiglu_fp4_decode.py`
(`test_form_and_knob_arms_write_the_default_bytes`, 12 arms at 14 row counts each), which also holds the CUDA-graph replay against
eager, the chain with a dependent launch that waits behind the entry and copies its outputs
(`test_dependent_that_waits_reads_the_eager_bytes`, eager and captured) and the warm-launch stress with byte references on three shapes
that refill the staging ring inside a launch (600 warm and 600 poisoned launches per shape).

Golden sets: 5 files -- one per token tile at the default geometry and one at the small geometry -- recorded on cc 10.0 into
`c84dd9abaee769b4/`, after the replay, batch-invariance and reference cases passed.  A new set is owed with a stated reason when the
rounding chain changes (the K order of the accumulation, either rounding to bf16, the SiLU's instructions, the block-scale chain).

GPU coverage: cc 10.0 (replay, batch invariance, goldens).  cc 10.3 and cc 10.7 were not run: the entry admits them, the kernel's
rendering is pinned for their targets on a host, and the status page reads "SM103: Unverified" and, for SM107, "Unverified".  Every
device run below is one in-process pytest run per command (an environment without pytest-xdist), one visible device, with
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, from `test/python` of a clean tree: the recording run on the tree before the 5 golden files were
added, every other run on a tree that carries them; `<suite>` = `gemm/frost/swiglu_decode/test_swiglu_fp4_decode_determinism.py`,
`<module>` = `gemm/frost/swiglu_decode/test_swiglu_fp4_decode.py`.  The device is a shared development part: these are correctness
runs.

cc 10.0 record (2026-11-13 UTC): a cc 10.0 part with 148 SMs (x86 host, Python 3.10) with torch 2.13.0+cu130, CUDA 13.0, driver API
13010, cuDNN 92700, nvidia-cutlass-dsl 4.8.0 compiling with CUDA 13.4, tvm-ffi 0.1.13.post3 (the `runtime` block of the files in
`c84dd9abaee769b4/`).  `CUDNN_DETERMINISM_RECORD=<new directory> pytest <suite> -m L0 -x -q -ra` -> 182 passed: the whole suite in one
run that stops at its first failure, so every replay, batch-invariance and reference case had passed before the first record was
written (the golden cases are the suite's last).  The 5 records were inspected (the runtime block equals the directory's key; names,
shapes and dtypes; the hashes of the six operands equal those of the operands rebuilt on an x86 host from the suite's CPU generator in
5 of 5 files; the four default-geometry cases share the weight, its scales and the two scalars and differ in the activation and in
every result) and copied.  `CUDNN_DETERMINISM_GOLDENS=$PWD/determinism_goldens pytest <suite> -m L0 -q -ra` -> 182 passed, 0 skipped:
`test_replay` 20, `test_replay_across_scheduling_boundaries` 6, `test_replay_on_a_fresh_entry_and_fresh_operands` 8,
`test_launch_on_a_side_stream_writes_the_same_bytes` 2, `test_batch_invariance` 126,
`test_inserting_or_removing_a_row_keeps_the_other_rows_bytes` 8, `test_duplicated_row_and_neighbour_isolation` 2,
`test_token_tile_pick_is_byte_neutral` 2, `test_inputs_are_unmodified` 2, `test_sixty_four_to_sixty_five_rows_is_the_typed_decline` 1,
`test_golden` 5.  `pytest <module> core/frost/test_frost_kernel_name_prefix.py -m L0 -q -ra` -> 206 passed, 0 skipped: the module 202,
the symbol-prefix test 4.  After the record the kernel's work tiles went from the tile scheduler in the DSL's wheel to the repository's
vendored one (the same static order; the rendering differs in parameter loads).  The set was NOT recorded again: the strict run and the
module run above are from the tree with the vendored scheduler, so the committed hashes are that kernel's bytes too.

Negative control (same part, same day): the suite run once with ONE launch per case given a changed operand, patched in memory -- the
second launch of a case receives a copy of the activation whose first row has the sign of every code flipped -> 179 of the 182 cases
fail.  The three that pass compare no launch result: `test_inputs_are_unmodified` (2) and the typed-decline case.

The other compiler flavor of the DSL is a DECLINE, not a lane.  The entry's bytes follow the compiler's lowering of three
approximate instructions and were qualified with the DSL compiling with CUDA 13.4; an older compiler is declined by name before any
compile.  With nvidia-cutlass-dsl 4.8.0 bound to its CUDA 12.9 flavor on the same part and runtime
(`cutlass.base_dsl.version_info.CUDA_VERSION` = 12.9; torch and the driver stay CUDA 13 -- a statement about the DSL's compiler
flavor, not about a CUDA 12 deployment): `pytest <module> <suite> core/frost/test_frost_kernel_name_prefix.py -m L0 -q -rs` -> 86
passed, 302 skipped.  Every case that needs a compiled kernel (298) skips with the entry's message ("... is qualified with the CuTe
DSL compiling with CUDA 13.4 or later; the DSL in this process compiles with CUDA 12.9 ..."), the three rendering cases and the scan
of the kernel module's DSL names skip with the same message, nothing is traced; the other host-only cases and the symbol-prefix test
pass.  Replay and batch invariance are therefore evidence for the CUDA 13.4 flavor only.  A probe, not a claim: with that gate lifted in the test process alone, the CUDA 12.9 flavor
passes `<suite>` strictly against the committed set on this part (182 passed, 0 skipped); the decline stays until that flavor is
qualified in full.

Cold compile on an x86 host (a cc 8.0 device: every device case skips; fake tensors, no launch), `pytest <module> <suite>
core/frost/test_frost_kernel_name_prefix.py -m L0 -q -rs`: 90 passed, 298 skipped under CuTe DSL 4.8.0 and under 4.8.1 (the 4.8.1
wheels put first on the path), both compiling with CUDA 13.4 -- the three rendering cases (`sm_100a`, `sm_103a`, `sm_107a`) among the
passes, each of which also fails on a `DeprecationWarning` that a trace attributes to a file of the cudnn package (none is raised);
86 passed, 302 skipped under 4.8.0 and under 4.8.1 with CUDA 12.9 (the rendering cases and the scan of the kernel module's DSL names
skip with the entry's decline).  The four token-tile forms, without and with the launch attribute, render byte-identical cubins under
4.8.1 and under 4.8.0 for the three targets (24 of 24).

CuTe DSL 4.8.1 on the cc 10.0 part (2026-11-13 UTC; the 4.8.1 wheels put first on the path of the same runtime, compiling with CUDA
13.4): `pytest <module> core/frost/test_frost_kernel_name_prefix.py -m L0 -q -ra` -> 206 passed, 0 skipped.
`CUDNN_DETERMINISM_RECORD=<new directory> pytest <suite> -m L0 -x -q -ra` -> 182 passed; its 5 records hold the input and output
hashes of the committed 4.8.0 set and are not committed.  The DSL version is part of the runtime key, so against the committed
directory the 5 golden cases skip under 4.8.1 -- a coverage gap there, not a pass.

### FP4 row quantization decode step: replay, batch invariance, locality and goldens

`row_quantize/frost/test_row_quantize_fp4_decode_determinism.py` pins `cudnn.RowQuantizeFp4Decode` (the block-scaled FP4 quantization of
bf16 rows under a global scale in one launch: bf16 `x` `[rows, 5120]`, one fp32 scale; packed E2M1 `codes`, E4M3 block `scales` in the
128 x 4 layout; rows 1 .. 1024).  Forward only: the entry is an inference step with no backward entry point, no gradient and no state,
so there is no backward case.  Every comparison is raw bytes over both buffers a launch writes.  The entry has ONE kernel body; its
three launch arms write the same bytes and are compared where a case says so.

What is held fixed: a block's bytes are a function of its sixteen values and of the global scale -- one thread owns the whole block,
and the kernel has no reduction across threads, no barrier and no shared memory -- so no split, pick or knob selects another
association and there is nothing to pin across batch layouts besides the operands.  Operands are built on the host (an operand set
has the same bytes on every device), once per case; the cases slice and permute them.  Every output is a fresh caller-owned buffer:
the codes filled with the sentinel `0xA5`, the scale buffer zero-filled, as its owner must (the kernel writes the live rows' scale
bytes only).  A block scale's position in the 128 x 4 layout follows its row index while its value does not, so batch layouts are
compared on the scale bytes per row, read through the layout written out in the test helpers; the replay and golden cases compare the
whole scale buffer.  The independent references are a separate check (`row_quantize_fp4_decode_utils.assert_reference`: every block
scale bitwise the definition's `e4m3(gs x (amax x fp32(1/6)))`, every code inside the bracket of the two approximate reciprocals) and
gate the goldens.

| Test method | Level | Parameters and purpose | Cases |
| --- | --- | --- | ---: |
| `test_replay` | L0 | rows 1 / 17 / 160 / 1024; eight executions into fresh buffers under side-stream contention; then the references | 4 |
| `test_replay_on_a_fresh_entry_and_fresh_operands` | L0 | two entries, two identical 24-row operand sets built afresh | 1 |
| `test_launch_on_a_side_stream_writes_the_same_bytes` | L0 | launch arm (plain, wait, wait + release); 17 rows under a side stream (ordered after the operands' producer stream) vs the current stream | 3 |
| `test_batch_invariance` | L0 | two global scales; one 1024-row operand set: the 1024-row call replayed three times and checked against the references; every row alone (a one-row call) vs inside that call, vs the first 2 / 3 / 8 / 16 / 17 / 31 / 32 / 33 / 63 / 64 / 65 / 95 / 96 / 97 / 127 / 128 / 129 / 160 rows as one call (both sides of the 32-row lines of a scale atom and of the 128-row band edge), vs the reversed batch | 2 |
| `test_insertion_removal_and_reordering` | L0 | a 130-row call (two scale bands), then a row inserted at position 5 (row 127 moves into the second band; rows 31, 63 and 95 move to the next quarter of their atoms), rows 3 / 11 / 129 removed (rows 128 / 129 move into the first band), a permutation: every row keeps its bytes, the inserted row writes its alone bytes | 1 |
| `test_duplicated_row_and_neighbour_isolation` | L0 | a row twice in one call (both copies = alone); the row at position 5 among 63 and at position 128 among 159 redrawn neighbours | 1 |
| `test_inputs_are_unmodified` | L0 | launch arm x rows 1 / 17 / 1024, caller-owned and entry-owned outputs (18 launches): `x` and `scale` hold their bytes | 1 |
| `test_one_element_moves_one_block` | L0 | one element changed (the first and the last element of the first, a middle and the last block; the first, a middle and the last row: 18 changes): the eight code bytes and the one scale byte of its block move, and no other byte of the call | 1 |
| `test_the_row_ceiling_call` | L0 | eight executions at 1024 rows; rows 0 / 511 / 512 / 1023 and the rows on both sides of every scheduling-wave edge below 1024 (where the part has one) alone vs inside; whether the call crossed a wave is reported in the run's `measured` section, not asserted | 1 |
| `test_row_ceiling_is_the_typed_decline` | L0 | 1025 rows is a `NotImplementedError`, zero rows a `ValueError`; never a launch (sentinel-filled outputs keep their bytes), never another provider | 1 |
| `test_golden` | L0 | four cases: three executions, the references, the wait and the wait + release arms on the same operands, the saved hashes of the codes and of the whole scale buffer | 4 |

Collected: 20 cases, all on a device.  Replay executions: `test_replay` 4 x 8 = 32; `test_batch_invariance` 2 x 3 of the 1024-row call
and 2 x 1,024 of a row alone; `test_the_row_ceiling_call` 8; `test_golden` 4 x 3.

Golden sets: `row-quantize-fp4-decode-w5120-<case>`, four files in `c84dd9abaee769b4/`, the first goldens of this API -- `random-rows1`
(one row under a non-power-of-two global scale), `random-rows8` (eight rows under 0.25), `structured-rows130` (two scale bands, under a
global scale that separates the associations of the block-scale product) and `code-domain-rows854`: the ENUMERATED code domain of the
non-power-of-two scale -- for every scale byte 0 .. `0x7E` every positive and every negative bf16 element a block of that byte can
hold.  A code is a function of its block's scale byte, the element and the global scale, so that case holds every code the kernel can
write under that scale, among them the elements whose code follows the last bit of the approximate reciprocals; random operands meet
those only by chance.  The launch arms write the default's bytes inside `test_golden` and, on more shapes, in
`test_row_quantize_fp4_decode.py` (`test_bit_neutral_arms_write_the_default_bytes`; behind a predecessor that released its dependents
before it wrote the operand, `test_wait_holds_every_read_behind_a_predecessor_that_released_early`), so the one set holds for every
arm.  The codes follow the part's approximate reciprocal: the hashes are a record of this runtime, not a claim across GPU
architectures.  A new set is owed with a stated reason when the rounding chain or the scale layout changes.

GPU coverage: cc 10.0 (replay, batch invariance and goldens).  cc 10.3 and cc 10.7 have not been run on a device: the entry admits
compute capability 10.x and the status page reads "SM103: Unverified" and, for SM107, "Unverified"; a run there skips the four golden
cases (no set for its runtime) -- a coverage gap, not a pass.  cc 9.0 and cc 12.x are declined by type.  Every run below is one
pytest run per command on one visible device, from `test/python`; `<suite>` = `row_quantize/frost/test_row_quantize_fp4_decode_determinism.py`,
`<module>` = `row_quantize/frost/test_row_quantize_fp4_decode.py`.  The device is a shared development part: these are correctness runs,
and no timing was taken.

cc 10.0 record (2026-11-18 UTC): a cc 10.0 part with 148 SMs (the device reports the name `NVIDIA Graphics Device`; x86 host, Python
3.10) with torch 2.13.0+cu130, CUDA 13.0, driver API 13010, cuDNN 92700, nvidia-cutlass-dsl 4.8.0 (compiling with CUDA 13.4), tvm-ffi
0.1.13.post3 (the `runtime` block of the goldens in `c84dd9abaee769b4/`).  The set was recorded into a new directory
(`CUDNN_DETERMINISM_RECORD=<new directory> pytest <suite> -m L0 -q -ra` -> 20 passed) after the functional module's device cases
had passed on the same part; the four files were copied unchanged.  Strict check and the functional module, from a clean tree at the commit that adds
the set: `CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" pytest <suite> -m L0 -q -ra` -> 20 passed, 0 skipped: `test_replay` 4,
`test_replay_on_a_fresh_entry_and_fresh_operands` 1, `test_launch_on_a_side_stream_writes_the_same_bytes` 3, `test_batch_invariance` 2,
`test_insertion_removal_and_reordering` 1, `test_duplicated_row_and_neighbour_isolation` 1, `test_inputs_are_unmodified` 1,
`test_one_element_moves_one_block` 1, `test_the_row_ceiling_call` 1 (measured: the 1024-row call crossed a scheduling wave; one wave
holds 888 thread blocks on 148 SMs; 7 rows compared alone), `test_row_ceiling_is_the_typed_decline` 1, `test_golden` 4.
`pytest <module> core/frost/test_frost_kernel_name_prefix.py -m L0 -q -ra` -> 127 passed, 1 skipped: the module 122 passed and 1
skipped (`test_entry_on_another_device_launches_there_and_keeps_the_callers_device_current`: "two CUDA devices are required" -- one
device is visible, so an entry on a device that is not the caller's current one is not covered on cc 10.0), the symbol-prefix test 5
passed.  The module's launch-facts case compared the profiler's record of a launch with the entry's facts in 9 of 9 cells (one kernel
per call; grid `(rows, 1, 1)`, block `(320, 1, 1)`, 0 B of shared memory).  The module's case behind a predecessor that released
early (`test_wait_holds_every_read_behind_a_predecessor_that_released_early`) compared 8 of 8 launches of the step equal to the plain
step, behind its positive control: a dependent without a wait read other bytes than the written ones in 4 of 4 eager launches and 4
of 4 replays.  On a part where that control does not reproduce the case skips -- a coverage gap, not a pass.  The reference cases
report, per global scale, how many elements have a two-code bracket (the elements within the reciprocal bound of a rounding
midpoint): between 0 and 72,804 of 6,333,440 per scale, and 100 of 4,372,480 on the enumerated code domain; no code was outside its
bracket and every block scale was bitwise the definition's.  The helpers' cases of the same change
(`pytest sdpa/frost/test_tile_dsl_rowwise.py sdpa/frost/test_tile_dsl_sf_layout.py -m L0 -q -ra`): 43 passed, 0 skipped.

CUDA 12.9 compiler flavour on the same part and runtime (the CuTe DSL 4.8.0 install without its CUDA 13 toolchain libraries, so the
DSL binds its CUDA 12 flavour: `CUDA_VERSION` 12.9; torch and the driver stay CUDA 13 -- this isolates DSL compatibility and is not a
deployment that uses CUDA 12 throughout): the strict determinism command -> 20 passed, 0 skipped -- the four golden cases, the
enumerated code domain among them, pass against the SAME set, whose runtime key does not name the compiler flavour, so the two
flavours write the same bytes for those cases; the functional command -> 126 passed, 2 skipped (`test_rendering_pins_the_convention`
for `sm_107a`: that compiler rejects the target, and the case skips with the entry's own decline; the two-device case); the helpers'
cases 42 passed, 1 skipped (the `sm_107a` rendering).

CuTe DSL 4.8.1 on the same part (the 4.8.1 wheels put first on the path of the same runtime).  Compiling with CUDA 13.4: the
functional command -> 127 passed, 1 skipped (the two-device case); the determinism suite recording into a new directory -> 20 passed
(the DSL version is part of the runtime key and no set is committed for that key).  That record holds the input and output hashes
of the committed 4.8.0 set in 4 of 4 files; it is not committed.  With only the CUDA 12 toolchain libraries of 4.8.1 (`CUDA_VERSION`
12.9): the functional command -> 126 passed, 2 skipped (the `sm_107a` rendering case and the two-device case); the determinism suite
checked strictly against the 4.8.1 record of the other flavour -> 20 passed, 0 skipped.

Cold compile on an x86 host (a cc 8.0 device: every device case skips; fake tensors).  The kernel's three renderings per target (the
launch arms) under 4.8.1 are byte-identical cubins to 4.8.0's for `sm_100a`, `sm_103a`, `sm_107a` and `sm_120a` with the CUDA 13.4
compiler (12 of 12) and for `sm_100a` / `sm_103a` with the CUDA 12.9 compiler (6 of 6); the three `sm_100a` cubins the device compiled
under each of 4.8.0 / CUDA 13.4, 4.8.0 / CUDA 12.9 and 4.8.1 / CUDA 13.4 are those host renderings (9 of 9 by hash).
`pytest <module> <suite> sdpa/frost/test_tile_dsl_rowwise.py sdpa/frost/test_tile_dsl_sf_layout.py
sdpa/frost/test_tile_dsl_quant_helpers.py norm/frost core/frost/test_frost_kernel_name_prefix.py core/test_import_boundaries.py -m L0
-q -ra` on that host under 4.8.0 with CUDA 13.4: 253 passed, 231 skipped (the device cases).

Re-run on the same part and runtimes (2026-11-19 UTC), from a clean tree with the files of this change as they stand before this
text.  Since the record above the
module gained three cases that need no device (the `sm_103a` and the `sm_107a` rendering are the `sm_100a` text; the scale positions
against the layout a block-scaled FP4 multiply reads), its rendering cases evaluate the byte every global access addresses, and its
reference holds a block whose global scale and stored scale are powers of two to ONE code.  The kernel did not change: its 36
renderings (four targets with the CUDA 13.4 compiler and two with CUDA 12.9, three launch arms, CuTe DSL 4.8.0 and 4.8.1) are
byte-identical cubins to those of the record above (18 distinct).  CuTe DSL 4.8.0 with CUDA 13.4: the strict determinism command ->
20 passed, 0 skipped; the functional command -> 130 passed, 1 skipped (the two-device case): the module 125 passed, the symbol-prefix
test 5 passed; the helpers' cases 43 passed.  CuTe DSL 4.8.0 with CUDA 12.9: strict -> 20 passed, 0 skipped against the SAME set;
functional -> 128 passed, 3 skipped (the `sm_107a` rendering case and the `sm_107a` text case, each with the entry's own decline; the
two-device case); the helpers' cases 42 passed, 1 skipped.  CuTe DSL 4.8.1 with CUDA 13.4: functional -> 130 passed, 1 skipped; the
determinism suite checked strictly against the 4.8.1 record of 2026-11-18 -> 20 passed, 0 skipped.  CuTe DSL 4.8.1 with CUDA 12.9:
functional -> 128 passed, 3 skipped; strict against the same 4.8.1 record -> 20 passed, 0 skipped; the helpers' cases
42 passed, 1 skipped.  The measured lines are those of the record above (launch facts 9 of 9 cells equal; the wait
case 8 of 8 launches behind a control seen in 4 of 4 eager launches and 4 of 4 replays; the same two-code counts per global scale; no
code outside its bracket and none in an exact block that is not the one code); the three `sm_100a` cubins each run compiled are the
host's renderings.  The host command above: 256 passed, 231 skipped.

Second re-run on the same part (2026-11-19 UTC), CuTe DSL 4.8.0 with CUDA 13.4, from a clean tree with the files of this change as
they stand before this text.  Since the record above the entry orders a launch behind the initialization of its entry-owned outputs
(an event recorded on the allocating stream behind the allocation; on another stream a stream-side wait for it; under a stream
capture a host wait for that one event), and the module gained nine cases: one that needs no device (the decision on stand-ins for
the event, the streams and the driver) and eight on a device (`compile(rows)` on a side stream behind a delay kernel and the call on
the original stream, per launch arm; the streams exchanged; a raw capture replayed twice while unrelated work is pending on a third
stream; the warm path after the initialization completed; two row counts initialized on two streams; entry-owned codes on a recycled
block).  The kernel and its config did not change: the 36 renderings are byte-identical, cubin and PTX text, to those of the record
above.  The strict determinism command -> 20 passed, 0 skipped; the functional command -> 139 passed, 1 skipped (the two-device
case): the module 134 passed, the symbol-prefix test 5 passed, and none of the nine new cases skipped; the three `sm_100a` cubins
the run compiled are the host's renderings.  Against the entry as it stood before, all nine fail: with the pair of seven rows
initialized on another stream the kernel's block scales were written first and the pending zero-fill erased them (2,240 of 2,240
live scale bytes zero, 0 code bytes differing), and on a recycled block a pending write of the block's previous owner replaced the
codes.  The other three compilers of the record above were not run again (the kernel is unchanged).  The host command above:
257 passed, 239 skipped.

Third re-run on the same part (2026-11-19 UTC), CuTe DSL 4.8.0 with CUDA 13.4, from a clean tree with the files of this change as
they stand before this text.  Since the record above the ordering follows the allocation: a launch is ordered behind the pending
initialization of every entry-owned pair whose bytes it writes, however the tensor reached the call.  The record above ordered only
a call that took an output by default; a pair taken with `outputs(rows)` and passed back as explicit outputs was launched unordered.
The module gained nine cases: two that need no device (which pending pairs a launch is ordered behind, by address; the place of the
record behind the allocation and the fill) and seven on a device (the pair taken with `outputs(rows)` on a side stream behind a
delay kernel and passed back whole, as its scales beside caller-owned codes and as a view of its scales; slices of another row
count's pair; entry-owned codes on a recycled block, passed back; two streams with disjoint entry-owned outputs; the documented
capture flow with entry-owned outputs).  The kernel and its config did not change: the 36 renderings are byte-identical, cubin and
PTX text, to those of the records above.  The strict determinism command -> 20 passed, 0 skipped; the functional command -> 148
passed, 1 skipped (the two-device case): the module 143 passed, the symbol-prefix test 5 passed, and none of the eighteen cases of
the ordering skipped; the three `sm_100a` cubins the run compiled are the host's renderings.  Against the entry of the record above
the five device cases that pass a tensor of a pair back fail: with the pair of seven rows taken on another stream and passed back,
the kernel's block scales were written first and the pending zero-fill erased them (2,240 of 2,240 live scale bytes zero, 0 code
bytes differing), and on a recycled block a pending write of the block's previous owner replaced the passed-back codes.  CuTe DSL
4.8.1 with CUDA 12.9 ran the functional command again -> 146 passed, 3 skipped (the two `sm_107a` cases, each with the entry's own
decline; the two-device case); the other two compilers of the records above were not run again (the kernel is unchanged).  The host
command above: 259 passed, 246 skipped.

### FP8 row quantization decode step: replay, batch invariance, locality and goldens

`row_quantize/frost/test_row_quantize_fp8_decode_determinism.py` pins `cudnn.RowQuantizeFp8Decode` (the dynamic FP8 quantization of
bf16 rows in one launch, each row under its own scale: bf16 `x` `[rows, 17408]`; E4M3 `codes`, one fp32 scale per row; rows
1 .. 1024).  Forward only: the entry is an inference step with no backward entry point, no gradient and no state, so there is no
backward case.  Every comparison is raw bytes over both buffers a launch writes.  The entry has ONE kernel body; its three launch
arms write the same bytes and are compared where a case says so.

What is held fixed: a row's bytes are a function of its values -- the one reduction, the row maximum of magnitudes, never leaves the
row's thread block and is exact in any order, and every step behind it is exactly specified -- so no split, pick or knob selects
another association and there is nothing to pin across batch layouts besides the operands.  Operands are built on the host (an
operand set has the same bytes on every device), once per case; the cases slice and permute them.  Every output is a fresh
caller-owned buffer filled with a sentinel.  The independent reference is a separate check
(`row_quantize_fp8_decode_utils.assert_reference`: every code byte and every scale bit against the operation's definition, with each
fp32 rounding spelled) and gates the goldens.

| Test method | Level | Parameters and purpose | Cases |
| --- | --- | --- | ---: |
| `test_replay` | L0 | rows 1 / 17 / 297 / 1024; eight executions into fresh buffers under side-stream contention; then the reference | 4 |
| `test_replay_on_a_fresh_entry_and_fresh_operands` | L0 | two entries, two identical 24-row operand sets built afresh | 1 |
| `test_launch_on_a_side_stream_writes_the_same_bytes` | L0 | launch arm (plain, wait, release + wait); 17 rows under a side stream (ordered after the operands' producer stream) vs the current stream | 3 |
| `test_batch_invariance` | L0 | one 1024-row operand set: the 1024-row call replayed three times and checked against the reference; every row alone (a one-row call) vs inside that call, vs the first 2 / 3 / 8 / 16 / 17 / 64 / 129 / 160 rows and the rows around the part's first two scheduling-wave edges (one and two times its SM count) as one call, vs the reversed batch | 1 |
| `test_insertion_removal_and_reordering` | L0 | a 40-row call, then a row inserted at position 5, rows 3 / 11 / 39 removed, a permutation: every row keeps its bytes, the inserted row writes its alone bytes | 1 |
| `test_duplicated_row_and_neighbour_isolation` | L0 | a row twice in one call (both copies = alone); the row at position 5 among 63 and at position 299 among 600 redrawn neighbours; the row between two rows of the largest finite magnitude | 1 |
| `test_inputs_are_unmodified` | L0 | launch arm x rows 1 / 17 / 1024, caller-owned and entry-owned outputs (18 launches): `x` holds its bytes | 1 |
| `test_one_element_moves_only_its_own_row` | L0 | one element changed (the first and the last column of each of the kernel's three passes; the first, a middle and the last row: 18 elements, two changes each): below the row maximum its own code byte moves and no other byte of the call; as the new row maximum its row's scale and codes move and no byte of another row | 1 |
| `test_the_row_ceiling_call` | L0 | eight executions at 1024 rows; rows 0 / 511 / 512 / 1023 and the rows on both sides of every scheduling-wave edge below 1024 (every multiple of the part's SM count: an edge whether one thread block occupies an SM or two share it) alone vs inside; the number of rows compared is reported in the run's `measured` section, not asserted | 1 |
| `test_row_ceiling_is_the_typed_decline` | L0 | 1025 rows is a `NotImplementedError`, zero rows a `ValueError`; never a launch (sentinel-filled outputs keep their bytes), never another provider | 1 |
| `test_golden` | L0 | two cases: three executions, the reference, the wait and the release + wait arms on the same operands, the saved hashes of the codes and the scales | 2 |

Collected: 17 cases, all on a device.  Replay executions: `test_replay` 4 x 8 = 32; `test_batch_invariance` 3 of the 1024-row call and
1,024 of a row alone; `test_the_row_ceiling_call` 8; `test_golden` 2 x 3.

Golden sets: `row-quantize-fp8-decode-w17408-<case>`, two files in `c84dd9abaee769b4/`, the first goldens of this API --
`random-rows1` (one random row) and `structured-rows8` (a zero row, a row of negative zeros, a row under the floor scale, rows whose
one dominant element sits in the last column, in the first column, in the last vector of the kernel's second pass and in the first
vector of its third, and a row of two values of opposite sign).  The launch arms write the default's bytes inside `test_golden` and,
on more shapes, in `test_row_quantize_fp8_decode.py` (`test_bit_neutral_arms_write_the_default_bytes`; behind a predecessor that
released its dependents before it wrote the operand, `test_wait_holds_every_read_behind_a_predecessor_that_released_early`), so the
one set holds for every arm.  Every step of the rounding chain is exactly specified; the hashes are still a record of this runtime
(the runtime is part of a golden's key).  A new set is owed with a stated reason when the rounding chain changes.

GPU coverage: cc 10.0 (replay, batch invariance and goldens).  cc 10.3 and cc 10.7 have not been run on a device: the entry admits
compute capability 10.x and the status page reads "SM103: Unverified" and, for SM107, "Unverified"; a run there skips the two golden
cases (no set for its runtime) -- a coverage gap, not a pass.  cc 9.0 and cc 12.x are declined by type.  Every run below is one
pytest run per command on one visible device, from `test/python`; `<suite>` = `row_quantize/frost/test_row_quantize_fp8_decode_determinism.py`,
`<module>` = `row_quantize/frost/test_row_quantize_fp8_decode.py`.  The device is a shared development part: these are correctness runs,
and no timing was taken.

cc 10.0 record (2026-11-22 UTC): a cc 10.0 part with 148 SMs (the device reports the name `NVIDIA Graphics Device`; x86 host, Python
3.10) with torch 2.13.0+cu130, CUDA 13.0, driver API 13010, cuDNN 92700, nvidia-cutlass-dsl 4.8.0 (compiling with CUDA 13.4), tvm-ffi
0.1.13.post3 (the `runtime` block of the goldens in `c84dd9abaee769b4/`).  The set was recorded into a new directory
(`CUDNN_DETERMINISM_RECORD=<new directory> pytest <suite> -m L0 -q -ra` -> 17 passed) after the functional module's device cases had
passed on the same part; the two files were copied unchanged, and the sha256 of the independent reference's codes and scales for the
two operand sets, computed on the host, are the output hashes the files hold.  Strict check and the functional module, from a clean
tree that holds the set: `CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" pytest <suite> -m L0 -q -ra` -> 17 passed, 0 skipped:
`test_replay` 4, `test_replay_on_a_fresh_entry_and_fresh_operands` 1, `test_launch_on_a_side_stream_writes_the_same_bytes` 3,
`test_batch_invariance` 1, `test_insertion_removal_and_reordering` 1, `test_duplicated_row_and_neighbour_isolation` 1,
`test_inputs_are_unmodified` 1, `test_one_element_moves_only_its_own_row` 1, `test_the_row_ceiling_call` 1 (measured: 22 rows
compared alone against the 1024-row call: rows 0, 511, 512, 1023 and both sides of every multiple of 148 SMs),
`test_row_ceiling_is_the_typed_decline` 1, `test_golden` 2.
`pytest <module> core/frost/test_frost_kernel_name_prefix.py -m L0 -q -ra` -> 144 passed, 1 skipped: the module 139 passed and 1
skipped (`test_entry_on_another_device_launches_there_and_keeps_the_callers_device_current`: "two CUDA devices are required" -- one
device is visible, so a launch on a device that is not the caller's current one is not covered on cc 10.0; a host case holds, on
stand-ins, that the compile and the launch run inside the scope of the entry's device), the symbol-prefix test 5 passed.  The
module's launch-facts case compared the profiler's record of a launch with the entry's facts in 9 of 9 cells (one kernel per call;
grid `(rows, 1, 1)`, block `(1024, 1, 1)`, 128 B of shared memory).  The module's case behind a predecessor that released early
(`test_wait_holds_every_read_behind_a_predecessor_that_released_early`) compared 8 of 8 launches of the step equal to the plain
step, behind its positive control: a dependent without a wait read other bytes than the written ones in 4 of 4 eager launches and 4
of 4 replays.  The module's case for the release in front of the wait
(`test_release_reaches_a_dependent_while_the_predecessor_still_runs`) read the step's own dependent starting while the step's
predecessor still ran in 4 of 4 eager launches and 4 of 4 replays, behind its control (4 and 4).  On a part where such a control
does not reproduce the case skips -- a coverage gap, not a pass.  The reference cases
found no differing byte: every code and every scale bit of the random and structured rows at 1 .. 1024 rows, and the scale and the
codes of all 32,640 finite bf16 magnitudes as a row maximum.  Rows with a NaN or an infinity are outside the status; what they held
on this run is reported by the module, not asserted: a row with an infinity had an infinite scale, the codes of signed zeros and the
code `0x7E` at the infinity; a row of NaN had the floor scale and the code `0x7E` in every element; the rows beside them were
unchanged.

CUDA 12.9 compiler flavour on the same part and runtime (the CuTe DSL 4.8.0 install without its CUDA 13 toolchain libraries, so the
DSL binds its CUDA 12 flavour: `CUDA_VERSION` 12.9; torch and the driver stay CUDA 13 -- this isolates DSL compatibility and is not a
CUDA 12 runtime): the strict check -> 17 passed against the SAME golden set (the compiler flavour is not part of the runtime key, and
the bytes are equal); the functional module with the symbol-prefix test -> 143 passed, 2 skipped (the two-device case, and the
`sm_107a` rendering case: that flavour does not assemble the target, which is the entry's own typed decline on cc 10.7); the two
dependent-launch cases compared as under CUDA 13.4 (8 of 8 launches equal; the release read in 4 of 4 eager launches and 4 of 4
replays, each behind its control).

### Row argmax decode step: replay, batch invariance, locality and goldens

`row_argmax/frost/test_row_argmax_decode_determinism.py` pins `cudnn.RowArgmaxDecode` (per row of a bf16 `[rows, 65536]` slab the
lowest column of the row maximum, or the entry of an int32 / int64 id table at that column, in one launch; rows 1 .. 1024).  Forward
only: the entry is an inference step with no backward entry point, no gradient and no state, so there is no backward case.  Every
comparison is raw bytes over the one vector a launch writes.  The entry has ONE kernel body; the table kind (none, int32, int64)
selects the load and the store width of the result and is one compiled kernel each.

What is held fixed: a row's result is a function of the row's values and of the table.  The op is exact -- compares, selects and bit
copies -- so no association and no rounding can differ, and no split, pick or knob exists: there is nothing to pin across batch
layouts besides the operands.  Operands are built on the host (an operand set has the same bytes on every device), once per case;
the cases slice and permute them.  Every output is a fresh caller-owned vector filled with a sentinel.  The independent reference
is a separate check (`row_argmax_decode_utils.assert_rule`: every byte of the result against the rule written out on the host --
the lowest column whose value is greater than -inf and not below any other, column 0 when there is none, a NaN never chosen) and
gates the goldens.

| Test method | Level | Parameters and purpose | Cases |
| --- | --- | --- | ---: |
| `test_replay` | L0 | (table, rows) = (none, 1) / (int32, 17) / (int64, 300) / (int64, 1024); eight executions into fresh vectors under side-stream contention; then the reference | 4 |
| `test_replay_on_a_fresh_entry_and_fresh_operands` | L0 | two entries, two identical 24-row operand sets built afresh | 1 |
| `test_launch_on_a_side_stream_writes_the_same_result` | L0 | 17 rows under a side stream (ordered after the operands' producer stream) vs the current stream | 1 |
| `test_batch_invariance` | L0 | table int64 / none; one 1024-row operand set: the 1024-row call replayed three times and checked against the reference; every row alone (a one-row call) vs inside that call, vs the first 2 / 3 / 8 / 17 / 64 rows and the rows around the part's first four scheduling-wave edges (one to four times its SM count) as one call, vs the reversed batch | 2 |
| `test_insertion_removal_and_reordering` | L0 | a 130-row call, then a row inserted at position 5, rows 3 / 11 / 129 removed, a permutation: every row keeps its result, the inserted row writes its alone result | 1 |
| `test_duplicated_row_and_neighbour_isolation` | L0 | a row twice in one call (both copies = alone); the row at position 5 among 63 redrawn neighbours and, in a call 104 rows taller than twice the part's SM count, at the first position behind that edge | 1 |
| `test_inputs_are_unmodified` | L0 | table kind x rows 1 / 17 / 1024, caller-owned and entry-owned outputs (18 launches): `x` and the tables hold their bytes | 1 |
| `test_one_element_moves_one_row` | L0 | one element raised above its row's maximum (the first, a middle and the last column; the first, a middle and the last row: 9 changes): its row's result becomes that column and no other row's result moves | 1 |
| `test_the_row_ceiling_call` | L0 | eight executions at 1024 rows; rows 0 / 511 / 512 / 1023 and the rows on both sides of every scheduling-wave edge below 1024 (every multiple of the part's SM count: an edge whether one thread block occupies an SM or two share it) alone vs inside; the number of rows compared is reported in the run's `measured` section, not asserted | 1 |
| `test_row_ceiling_and_the_empty_call_are_typed_declines` | L0 | 1025 rows is a `NotImplementedError`, zero rows a `ValueError`; never a launch (a sentinel-filled output keeps its bytes), never another provider | 1 |
| `test_golden` | L0 | four cases: three executions, the reference, the saved hashes of the inputs and of the result | 4 |

Collected: 18 cases, all on a device.  Replay executions: `test_replay` 4 x 8 = 32; `test_batch_invariance` 2 x 3 of the 1024-row call
and 2 x 1,024 of a row alone; `test_the_row_ceiling_call` 8; `test_golden` 4 x 3.

Golden sets: `row-argmax-decode-w65536-<case>-rows<n>-<table kind>`, four files in `c84dd9abaee769b4/`, the first goldens of this
API -- `random-rows1-none` (one random row, the column as int32), `special-rows8-int32` (rows with NaN, both infinities and zeros
of both signs, a row of -inf, through an int32 table), `rule-rows27-int64` (the rows that state the rule one sentence at a time:
ties inside a vector, across a thread's passes, across threads and across warps, the maximum in the first and in the last column,
zeros of both signs, infinities, the most negative and the largest finite value, a subnormal, NaN beside values, rows of NaN and of
-inf, through an int64 table) and `ties-rows300-int64` (300 rows that hold their maximum thousands of times, more than twice
the SM count of a 148-SM part).  The op is exact: the result hashes are the same on every part, and a change of one is a
change of the rule.  The runtime key still names one runtime, so a run on another runtime has no set to check against.

GPU coverage: cc 10.0 (replay, batch invariance and goldens).  cc 10.3 and cc 10.7 have not been run on a device: the entry admits
compute capability 10.x and the status page reads "SM103: Unverified" and, for SM107, "Unverified"; a run there skips the four golden
cases (no set for its runtime) -- a coverage gap, not a pass.  cc 9.0 and cc 12.x are declined by type.  Every run below is one
pytest run per command on one visible device, from `test/python`; `<suite>` = `row_argmax/frost/test_row_argmax_decode_determinism.py`,
`<module>` = `row_argmax/frost/test_row_argmax_decode.py`.  The device is a shared development part: these are correctness runs,
and no timing was taken.

cc 10.0 record (2026-11-22 UTC): a cc 10.0 part with 148 SMs (the device reports the name `NVIDIA Graphics Device`; x86 host, Python
3.10) with torch 2.13.0+cu130, CUDA 13.0, driver API 13010, cuDNN 92700, nvidia-cutlass-dsl 4.8.0 (compiling with CUDA 13.4), tvm-ffi
0.1.13.post3 (the `runtime` block of the goldens in `c84dd9abaee769b4/`).  The set was recorded into a new directory
(`CUDNN_DETERMINISM_RECORD=<new directory> pytest <suite> -m L0 -q -ra` -> 18 passed) after the functional module's device cases
had passed on the same part; the four files were copied unchanged.  Strict check and the functional module, from a clean tree at
the commit that carries this record: `CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" pytest <suite> -m L0 -q -ra` -> 18 passed, 0
skipped: `test_replay` 4, `test_replay_on_a_fresh_entry_and_fresh_operands` 1, `test_launch_on_a_side_stream_writes_the_same_result`
1, `test_batch_invariance` 2, `test_insertion_removal_and_reordering` 1, `test_duplicated_row_and_neighbour_isolation` 1,
`test_inputs_are_unmodified` 1, `test_one_element_moves_one_row` 1, `test_the_row_ceiling_call` 1 (measured: 22 rows compared
alone against the 1024-row call: rows 0, 511, 512, 1023 and both sides of every multiple of 148 SMs),
`test_row_ceiling_and_the_empty_call_are_typed_declines` 1, `test_golden` 4.
`pytest <module> core/frost/test_frost_kernel_name_prefix.py -m L0 -q -ra` -> 109 passed, 1 skipped: the module 104 passed and 1
skipped (`test_entry_on_another_device_launches_there_and_keeps_the_callers_device_current`: "needs two CUDA devices" -- one device
is visible, so an entry on a device that is not the caller's current one is not covered on cc 10.0), the symbol-prefix test 5
passed.  The module's launch-facts case compared the profiler's record of a launch with the entry's facts in 6 of 6 cells (the
three table kinds at 3 and at 17 rows: one kernel per call; grid `(rows, 1, 1)`, block `(1024, 1, 1)`, 256 B of shared memory).
The cases of a warm launch behind an allocation that is pending on another stream passed with the parked work still pending when
the call returned (4 + 1 + 1 cases, none skipped).  Asked on the same part on the same day: the driver's occupancy answer for each
of the three compiled kernels at the launch's 1024 threads and 256 B of shared memory is two thread blocks per SM, and each kernel
holds 30 registers per thread, under the CUDA 13.4 and under the CUDA 12.9 compiler flavour -- one scheduling wave holds 296 rows
on this part.

CUDA 12.9 compiler flavour on the same part and runtime (the CuTe DSL 4.8.0 install without its CUDA 13 toolchain libraries, so the
DSL binds its CUDA 12 flavour: `CUDA_VERSION` 12.9; torch and the driver stay CUDA 13 -- this isolates DSL compatibility and is not a
deployment that uses CUDA 12 throughout): the strict determinism command -> 18 passed, 0 skipped against the SAME set, whose runtime
key does not name the compiler flavour; the functional command -> 108 passed, 2 skipped (`test_rendering_holds_the_rule` for
`sm_107a`: that compiler rejects the target, and the case skips with the entry's own decline; the two-device case).

CuTe DSL 4.8.1 on the same part (the 4.8.1 wheels put first on the path of the same runtime), compiling with CUDA 13.4: the
functional command -> 109 passed, 1 skipped (the two-device case); the determinism suite recording into a new directory -> 18 passed
(the DSL version is part of the runtime key and no set is committed for that key).  That record holds the input and output hashes
of the committed 4.8.0 set in 4 of 4 files; it is not committed.

Cold compile on an x86 host (a cc 8.0 device: every device case skips).  The kernel's three renderings per target (the table kinds)
assemble for `sm_100a`, `sm_103a`, `sm_107a` and `sm_120a` with the CUDA 13.4 compiler and for `sm_100a` / `sm_103a` with the CUDA
12.9 compiler; the three `sm_100a` cubins the device compiled under each of 4.8.0 / CUDA 13.4, 4.8.0 / CUDA 12.9 and 4.8.1 / CUDA
13.4 are those host renderings of the same compiler flavour, by hash.  `pytest <module> <suite> row_quantize/frost
sdpa/frost/test_tile_dsl_rowwise.py core/frost/test_init_order.py core/frost/test_frost_kernel_name_prefix.py
core/test_import_boundaries.py -m L0 -q -ra` on that host under 4.8.0 with CUDA 13.4: 209 passed, 148 skipped (the device cases).

### QSA: 42 cases

QSA uses BF16 Q/K/V with head dimension 256. The saved FP32 output and LSE
must match even when the BF16 output rounds to the same value.

| Test method | Level | Parameters and purpose | Cases |
| --- | --- | --- | ---: |
| `test_replay` | L0 | fprop/bprop × Q:KV heads 12:1, 24:2, 6:1; eight executions; input mutation check | 6 |
| `test_batch_invariance` | L0 | Same six combinations; lengths 65, 3, 17; removal/reorder with masks preserved | 6 |
| `test_golden` | L0 | fprop/bprop; lengths 17, 3; three executions, reference, saved hashes; the backward runs its default path | 2 |
| `test_sparse_batch_invariance` | L0 | fprop/bprop; lengths 2057, 3; four executions then standalone comparison | 2 |
| `test_dispatch_boundary` | L0 | fprop/bprop × total tokens 512/513; a fixed 3-token target before/after a neighbor | 4 |
| `test_mask_replay` | L0 | fprop/bprop × all selected, every seventh row empty, all masked; reference and eight executions | 6 |
| `test_mask_batch_invariance` | L0 | Same six combinations; each sequence alone, reversed batch, duplicate sequence | 6 |
| `test_neighbor_isolation` | L0 | fprop/bprop; change neighboring values, masks, and upstream gradients; the backward keeps the packed layout | 2 |
| `test_zero_upstream_gradient_bprop` | L0 | Eight executions with zero dO; all gradients must be zero | 1 |
| `TestQSABackwardEngine::test_replay` | L0 | bitmap and indexed entries with `deterministic=True`; multi-tile dense, sparse record windows, packed sparse; four executions and reference | 7 |

Head parameters have explicit IDs such as `q24-kv2`. A failure such as
`test_dispatch_boundary[total513-fprop]` identifies the routing boundary and
direction without a separate lookup table. A test stops at its first mismatch;
later checks in that case are not claimed as passes.

### DSA: 101 cases

The suite lives in `deepseek_sparse_attention/cutedsl/test_DSA_determinism.py`:
56 `TestDSA` cases and 45 `TestDSADeterministicDKV` cases.
Each `TestDSA` case checks all returned forward tensors. Backward uses deterministic
execution and checks dQ, the shared K/V latent gradient dKV, and dSink.
The three `(heads, QK dimension, Top-K)` layouts are `(64,512,64)`,
`(64,576,64)`, and `(128,512,128)`; `(96,576,64)` and `(128,576,64)` run
forward only. SM100 H96 forward launches the H64 kernel on heads [0, 64) and
[32, 96), and H128 D576 forward on [0, 64) and [64, 128); their backward is
not in this suite. The 576-wide path uses all QK dimensions
but only the first 512 KV dimensions for values.

| Test method | Level | Parameters and purpose | Cases |
| --- | --- | --- | ---: |
| `test_replay` | L0 | fprop/bprop × FP16/BF16 × three layouts, and fprop × FP16/BF16 × H96 and H128 D576; eight executions, input mutation check, numerical reference | 16 |
| `test_batch_invariance` | L0 | Same sixteen combinations; each sequence alone, packed, reversed, and duplicated | 16 |
| `test_selection` | L0 | fprop/bprop × ragged lengths, invalid indices, empty rows, Top-K 65 tail, indexer prefix 512; replay, reference, batch checks | 10 |
| `test_query_boundary` | L0 | fprop/bprop × H64 with 128/129/257 queries and H128 with 74/75 queries; four replays then batch checks | 10 |
| `test_neighbor_isolation` | L0 | fprop/bprop; change neighboring Q/KV values and Top-K selection while keeping the target fixed | 2 |
| `test_golden` | L0 | fprop/bprop; query lengths 3, 2; Top-K 16; three executions, reference, saved hashes | 2 |

Packing shifts valid Top-K indices by the sequence's KV offset. Negative and
out-of-bounds indices stay invalid; they must not name a neighboring KV row.
`TestDSA` rows have unique valid indices, as the shard and relay layouts of
deterministic DSA backward require. Duplicating a sequence creates a separate
KV region.

dSink is one shared per-head vector, so it cannot be sliced by sequence.
Backward batch checks isolate one target's loss: its dO stays fixed and all
other sequences receive zero dO. The complete dSink vector is then comparable
to the target alone. Replay/reference cases also cover nonzero dO for every
sequence. Empty-row sentinels retain their actual float dtype and raw bits;
NaNs and non-finite outputs/gradients still fail.

### DSA deterministic dKV layouts: 45 cases

`TestDSADeterministicDKV` runs on SM100/SM103, where deterministic backward
stores dKV in conflict-rank storage, and skips elsewhere: SM107 plans keep the
shard layouts that `TestDSA` covers. Every case runs the public wrappers with
`deterministic=True` and `deterministic_dkv_mode`, and asserts the backend and
dKV layout each batch selects. Each layout packs two sequences; the packed
batch spans several waves, and the KV extents keep fewer dense planes than a
wave has queries, so shared rows also take overflow records.

| Layout ID | Heads, QK dim, Top-K, dtype | Query lengths | KV lengths | Mode | Backend/layout |
| --- | --- | --- | --- | --- | --- |
| `records-h64-d512-k64-bf16` | 64, 512, 64, BF16 | 17, 300 | 1000, 1000 | `records` | generic M64, records |
| `records-h64-d512-k64-fp16` | 64, 512, 64, FP16 | 17, 300 | 1000, 1000 | `records` | generic M64, records |
| `records-h64-d576-k64-bf16` | 64, 576, 64, BF16 | 17, 300 | 1000, 1000 | `records` | generic M64, records |
| `records-strided-h64-d512-k64-bf16` | 64, 512, 64, BF16 | 17, 1183 | 1000, 1000 | `records` | generic M64, records, strided waves |
| `chain-h64-d512-k128-bf16` | 64, 512, 128, BF16 | 17, 300 | 2048, 2048 | `chain` | generic M64, chain relay |
| `shards-h64-d512-k64-bf16` | 64, 512, 64, BF16 | 17, 300 | 1000, 1000 | `shards` | generic M64, shards |
| `rank-fused-h128-d512-k128-bf16` | 128, 512, 128, BF16 | 40, 60 | 8192, 16384 | `records` | H128 D512 two-CTA, fused conflict-rank |
| `rank-setup-h128-d512-k1024-bf16` | 128, 512, 1024, BF16 | 40, 60 | 8192, 16384 | `records` | H128 D512 two-CTA, per-wave-setup conflict-rank |
| `rank-h128-d576-k128-bf16` | 128, 576, 128, BF16 | 17, 60 | 8192, 16384 | `records` | H128 D576 two-CTA, one-round conflict-rank |
| `rank-sorted-h128-d576-k1024-bf16` | 128, 576, 1024, BF16 | 17, 60 | 2200, 2201 | `records` | H128 D576 two-CTA, one-round conflict-rank, sorted rows |

| Test method | Level | Parameters and purpose | Cases |
| --- | --- | --- | ---: |
| `test_dkv_replay` | L0 | Every layout with full selections; the records BF16, chain, both H128 D512 and the D576 Top-K 128 layouts also with ragged and duplicate selections; shards also with ragged selections. Eight executions with fresh outputs, caller workspaces filled with 0x00, 0xA5 and 0xFF bytes, and side-stream work; input mutation check; numerical reference | 21 |
| `test_dkv_neighbor_isolation` | L0 | Every layout, ragged selections; the other sequence's values, Top-K selection and lengths change while the shapes stay; every result | 10 |
| `test_batch_invariance` | L0 | Every layout except the sorted D576 one, ragged selections; each sequence alone, packed, reversed and duplicated; forward results, dQ and, for the generic M64 layouts, dSink | 9 |
| `test_independent_of_dkv_storage` | L0 | Same inputs under two modes: records and shards (H64; forward results, dQ, dSink), chain and shards (forward results, dSink), conflict-rank and private planes (H128 D512 Top-K 128 and 1024, D576 Top-K 128; forward results and dQ) | 5 |

Ragged selections add per-query lengths with empty rows and negative and
out-of-range indices. Duplicate selections repeat a quarter of every row's slots
and make every query select its sequence's KV row 0 Top-K / 32 times; they
replay the larger sequence alone, so that row is shared by every query of a
wave. Duplicates are inside the contract of every conflict-rank layout; no
shard case uses them. H128 D576 has no public forward kernel: its sequences
carry reference forward results as fixed backward inputs, so these cases
compare no forward results.

Known gaps, stated rather than tolerated:

- dKV is not batch invariant once a batch spans several waves, on any layout.
  A conflict-rank layout ranks a KV row's writers by their wave position and
  sizes its dense planes from the batch's KV extent; the shard layouts take
  their shard count from the batch's query extent and fold shards in groups;
  the default mode picks the layout from the batch's extents. `develop`
  changes dKV in the same multi-wave comparisons (validation record below).
  `test_batch_invariance` therefore does not compare dKV.
- Generic M64 plans reduce dSink with one warp per head (lane `t` sums queries
  `t, t + 32, ...`, then an xor butterfly); a uniform query shift only rotates
  the lanes, so dSink is batch invariant under the measured packings and
  records and chain plans give the shard plans' bits. H128 two-CTA plans sum
  dSink in an order that follows query positions, in both layouts, as on
  `develop`; their batch cases do not compare dSink.
- H128 D576 plans with Top-K 1024 and wider sort each query's rows by ranges of
  the batch's KV extent and alternate the walk direction between waves, so
  their dQ is not batch invariant either and differs in bits from the
  private-plane plans; the sorted layout has replay and neighbor cases only.
- Not exercised: records plans with split pairs (their dQ adds two partials
  once the batch's KV extent selects them), chain plans whose rows are shared
  by every active query of a wave (their tile order follows those rows),
  H128 D576 one-round plans wider than Top-K 128 that stride their waves (the
  wave order follows the batch's indices), the few-plane H128 D512 variant,
  H96 D576 plans and the cluster relay. The L1 bitwise tests in
  `test_DSA_sparse_attention_backward.py` cover several of these paths.
- These layouts have no golden records; their bits depend on the plan, and
  for D576 on the SM count.

### SM90 DSA conflict-rank backward: H20 and H200 validation, 2026-11-17

The SM90 default deterministic backward now uses four-plane conflict-rank
storage; an explicit logical relay keeps the shard layout. Validation used an
NVIDIA H20-3e (SM90), PyTorch 2.8.0+cu129, CUDA 12.9, cuDNN 9.10.2,
CuTe DSL 4.8.1 and `cuda-bindings` 12.9.10. Imports resolved to the isolated
`icavan/sm90-dsa-conflict-rank` checkout based on upstream `3b7cd6cab`.

New BF16 evidence:

- H64/D512 and D576, 65 queries, KV 256 and Top-K 64 with every query selecting
  the same rows: conflict ranks 0 through 64 exercise all four planes and the
  column-phase overflow tickets. Four fresh poisoned-workspace executions per
  shape are raw-byte equal for dQ, dKV and dSink and pass the independent FP32
  reference.
- H96/D576, 129 queries, KV 256 and Top-K 128: eager execution and three CUDA
  Graph replays are raw-byte equal after poisoning outputs/workspace; replay
  also matches eager execution after changing `topk_idxs` and `topk_length`.
- The complete SM90 L0 selection in
  `test_DSA_sparse_attention_backward.py` has 67 passing expanded cases. It
  covers ordinary execution,
  both `topk_length` variants, invalid/padded rows, workspace boundaries,
  compile-cache reuse, the default records route and every explicit shard
  relay size. The H20 C16 logical relay selects physical C8.

The long-context performance case is BF16 H64, Q=16384, KV=262144 and Top-K
2048. Five warmups and twenty allocation-free calls measure 119.223 ms for
D512 (92.22 TFLOPS, 3.008 GiB workspace) and 117.632 ms for D576 (100.48
TFLOPS, 3.258 GiB). The upstream + physical-cluster-fix D512 C16
logical/C8 physical baseline measures 484.915 ms and 4.508 GiB: 4.07x speedup
and 33.3% less workspace.

The AIStudio H200 allocation (runtime name `NVIDIA L20X`) used PyTorch
2.12.1+cu130, CUDA 13.0, cuDNN 9.19.0, CuTe DSL 4.8.1 with the CUDA 13 extra
(CUDA 13.4 lowering), and `cuda-bindings` 13.0.3. The prepared plan selected
one 128-CTA wave instead of two 64-CTA phases. The same high-conflict reference
case and all three CUDA Graph variants passed, including poisoned workspace and
changed sparse inputs. Those 65 native/host cases predate the later cooperative
launch contract; the two added small-device policy cases expand the current
selection to 67. Records and shard-relay compile-cache reuse across changing
runtime extents was covered.
One-wave overflow writers use four independent column-phase tickets for D512
and two for D576; the tickets preserve rank order within every disjoint column
group while allowing adjacent ranks to pipeline. Long-context performance with
five warmups and 20 calls is 69.147--69.217 ms for D512 and 83.164--83.285 ms
for D576. The previous single-chain records implementation measured
75.056--75.151 and 85.201--85.257 ms; upstream C16 baselines were 84.643 and
100.126 ms. Thus the final records path is 1.224x and 1.204x faster than
upstream, respectively. In a high-conflict stress where every query shares the
same 2048 KV rows, two warmups and five calls reduced D512 from 151.586 to
87.678 ms (1.73x) and D576 from 195.635 to 139.779 ms (1.40x) relative to the
previous records implementation.

H20-3e retained the two ordered cooperative 64-CTA phases and a single overflow ticket per
row; extra intra-launch column barriers regress this split-wave schedule and
are deliberately disabled. Its phase-policy, prepared-plan cache,
records/shards compile-cache, D512/D576 high-conflict references, and all three
changed-input CUDA Graph cases passed after the H200 scheduling change. The
complete SM90 L0 selection passed 67 cases on H20-3e (CUDA 12.9 lowering); the
earlier 65-case revision passed on the AIStudio H200 allocation (CUDA 13.4
lowering) before the two host-only small-device policy cases were added.

No support status is upgraded: multi-wave dKV is still not batch invariant,
so SM90 BF16 remains **Replay only**. Duplicate active indices within one
query are outside the SM90 contract. FP16, other SM90 head-count/dimension
combinations, SM100/SM103 and SM107 are unverified for this change. No golden
record is added because these plan-dependent backward bits already have no
golden contract; replay and the independent reference are the qualification.

### Standalone DSA Top-K with per-row starts

`deepseek_sparse_attention/cutedsl/test_DSA_indexer_top_k.py` covers FP32,
FP16 and BF16 against a per-interval reference, including unaligned starts,
empty rows, cutoff ties and speculative groups. Deterministic output indices
and values are byte-identical after isolating, reordering and duplicating rows.
Standalone index selection has no backward operator.

`pytest -q deepseek_sparse_attention/cutedsl/test_DSA_indexer_top_k.py -k
'starts or accepts_strided_inputs'` passed 17 cases on SM90 (PyTorch 2.8.0
cu128, DSL 4.8.1) and SM103 (PyTorch 2.12.1 cu130, DSL 4.8.0 and 4.8.1).
The SM103 checks exercised both the CUDA 12.9 and 13.4 DSL compiler flavors.
On SM100 with DSL 4.8.0, the same 17 cases passed. Additional bitmap checks
at K=512 passed for all three dtypes and both tie directions. Memcheck reported
driver API errors during compilation, so the SM100 sanitizer result is inconclusive.
SM107 and SM12x interval coverage remains unverified.

### DSA BF16 indexer: 46 cases

The suite lives in `deepseek_sparse_attention/cutedsl/test_DSA_indexer_determinism.py`.
Q/K are BF16. Forward weights use BF16 or FP32; scores are FP32. Backward
weights, dQ and dWeights are BF16; dK uses BF16 or FP32. Score operands,
normalizers and the loss gradient are FP32. This BF16-input suite is separate from the FP8/MXFP8 cases below.

| Test method | API path | Parameters and purpose | Cases |
| --- | --- | --- | ---: |
| `test_replay` | Forward | H32/H64 × BF16/FP32 weights × ratio 1/4; eight executions and FP64 reference | 8 |
| `test_batch_invariance` | Forward | Same eight combinations; standalone, packed, reversed, duplicate | 8 |
| `test_boundary` | Forward | Q/K lengths 127/128/129; all-masked rows; four executions, reference, batch checks | 4 |
| `test_ragged_batch` | Forward | BF16/FP32 weights; packed query/key lengths (3,7)/(17,129), reorder/duplicate, mask padding | 2 |
| `test_neighbor_isolation` | Forward | Fixed target with changed neighboring Q/K/weights | 1 |
| `test_golden` | Forward | BF16 weights, three queries; reference, three executions, saved hash | 1 |
| `test_replay` | Backward | Sparse H32/H64 and dense H64 × BF16/FP32 dK; reference then eight executions | 6 |
| `test_batch_invariance` | Backward | Same six combinations; standalone, packed, reversed, duplicate | 6 |
| `test_sparse_invalid_slots` | Sparse backward | Local/global indices with negative padding; reference, replay, batch checks | 2 |
| `test_sparse_persistent_rows` | Sparse backward | Top-K 896/2048, 193 queries over 96 producer planes; reference, four executions, batch checks | 2 |
| `test_zero_loss_gradient` | Both backward APIs | Zero loss gradient; eight executions and exact zero gradients | 2 |
| `test_neighbor_isolation` | Both backward APIs | Fixed target and loss scale with changed neighbors | 2 |
| `test_golden` | Both backward APIs | One query, FP32 dK; reference, three executions, saved hashes | 2 |

All 46 cases are L0. Sparse backward sets `deterministic=True`; dense backward
has no determinism switch. Each call restores mutable score inputs and checks
immutable inputs. Gradient buffers start at NaN; sparse deterministic workspace
starts at `0xA5`. Results include all three gradients and each mutated score
buffer. Numerical references use the existing autograd checks with cosine
similarity at least 0.99 and RMS relative error at most 0.02; bitwise checks
still allow no difference.

Backward sets `loss_coeff = B * S_q` and fixes `grad_loss = 0.75`, so adding or
removing a batch member does not change its gradient scale. Local Top-K IDs
stay local. Global IDs are rebuilt for each batch order; negative padding stays
invalid. The 193-row cases force a persistent producer to process a third row. Golden backward cases have one query; their passes do not
establish determinism for multi-query reductions.

See the [API-by-dtype results](#api-and-datatype-results) and
[SM100 indexer failures](#indexer-failures-sm100) below.

### DSA FP8/MXFP8 indexer: 34 cases

`deepseek_sparse_attention/cutedsl/test_DSA_indexer_fp8_determinism.py` adds
17 L0 cases for SM100/SM103/SM107 MXFP8 and the same 17 for SM90 FP8.
Each parameter ID names the architecture group and precision. The fixture
skips other architectures before launching a kernel and checks the DSL target.

| Method | Parameters | Cases per precision |
| --- | --- | ---: |
| `test_replay` | H32/H64 × ratio 1/4; eight executions, numerical reference | 4 |
| `test_batch_invariance` | Same combinations; standalone, packed, reordered, each sequence duplicated | 4 |
| `test_boundary` | Q/K lengths 127/128/129, all-masked rows; four executions, reference, batch checks | 4 |
| `test_ragged_batch` | H32/H64; lengths (3,7)/(17,129); replay, reference, reorder/duplicate with repacked scales | 2 |
| `test_neighbor_isolation` | Fixed target payload/scales with changed neighbors | 1 |
| `test_scale_change` | Fixed FP8 payload; doubled Q scales must change scores and still match the reference/replay | 1 |
| `test_golden` | Three queries; reference, three executions, payload/physical-scale/output hashes | 1 |

Both paths use E4M3 Q/K and BF16 weights. MXFP8 uses nonuniform E8M0 scales
per 32 values, packed with the independent scale-layout reference. SM90 uses
FP32 token/head descales. Quantization happens once per sequence using CPU-seeded
inputs; repacking never requantizes. Input mutation checks compare raw payload
and scale bytes. Both paths compare FP32 scores; SM90 also compares FP32 LSE.
The score reference uses dequantized inputs and FP64 computation. Existing
correctness tolerances apply only to references; replay and batch comparisons
remain byte-exact.

SM100/SM103/SM107 MXFP8 and SM90 FP8 each had **17 passed, 0 failed**,
including their own saved golden. Each device skips the other precision group.
The [API-by-dtype results](#api-and-datatype-results) keep these gaps separate
from measured results. FP8/MXFP8 backward is not
supported by these DSA attention/indexer APIs and has no positive test cases.

### DSA NVFP4 indexer: 17 cases

`deepseek_sparse_attention/cutedsl/test_DSA_indexer_fp8_determinism.py` runs the
17 cases of the table above for SM100/SM103/SM107 NVFP4 as well (parameter ID
`sm100-sm103-sm107-nvfp4`). Q/K are packed E2M1 with non-negative E4M3 scales,
one per 16 values, drawn with mantissa bits in [2^-2, 2^3] and quantized once
from CPU-seeded inputs. Their bytes travel as `uint8` and are viewed as
`float4_e2m1fn_x2` only for the launch, so input mutation checks compare raw
payload and scale bytes. The reference decodes the stored E2M1 codes and scales
independently of the quantizer and computes in FP64; replay and batch
comparisons remain byte-exact. `test_scale_change` doubles the Q E4M3 scales.

| GPU | Runtime | Result |
| --- | --- | --- |
| SM100 B200, 148 SMs | Torch `2.13.0+cu130`, CUDA `13.0`, driver 590.47, cuDNN `92700`, CuTe DSL `4.8.1` | 16 passed, 1 skipped |
| SM103 B300, 72 SMs | Torch `2.13.0+cu130`, CUDA `13.0`, driver 595.84.01, cuDNN `92700`, CuTe DSL `4.8.0` | 16 passed, 1 skipped |
| SM107 Rubin, 216 SMs | Torch `2.13.0+cu130`, CUDA `13.0`, driver 620.62, cuDNN `92700`, CuTe DSL `4.8.0` | 16 passed, 1 skipped |

The skip on each GPU is `test_golden`: it passed replay and the FP64 reference,
then found no committed record for that runtime (MXFP8 has none for these
runtimes either). No NVFP4 golden is committed yet; until one is recorded on a
runtime that CI runs, NVFP4 byte stability is checked within a process only
(replay, batch, boundary, ragged and neighbor cases). Command, from
`test/python`:

```bash
python -m pytest deepseek_sparse_attention/cutedsl/test_DSA_indexer_fp8_determinism.py -q -rs
```

### Block-sparse attention blk64: 30 cases

The suite lives in `block_sparse_attention/cutedsl/test_BSA_determinism.py`
and is collected separately from the 330-case attention set above
(`pytest -m L0 block_sparse_attention/cutedsl/test_BSA_determinism.py --collect-only -q`
lists 30 node IDs). It drives the public `block_sparse_attention_forward`,
`block_sparse_attention_backward` and `block_sparse_attention_backward_metadata`
entry points with `sparse_block_size=64`: the SM90 blk64 kernels, and the
SM100/SM103 blk64 path through the same API. Head dimension 128, MHA, BHSD.
A sequence is one batch element with its own KV block lists and counts;
`block_sizes` is shared by the batch, inactive list slots hold a poison index.
Every call runs on clones, checks the output names, that no input byte changed,
that `-inf` LSE appears exactly on the empty query rows (whose O and dQ must be
zero), and writes gradients into fresh NaN-poisoned caller buffers.

Four forward configurations run in both datatypes (BF16 and FP16), each for
replay and for batch invariance, so every configuration the support page
names carries both kinds of evidence per datatype: `fixed-full` (fixed counts;
8 x 8 blocks, four-block lists, three batch elements, two heads),
`variable-partial` (variable counts with empty rows, random partial KV block
sizes), `fixed-partial-split2` (fixed counts, partial blocks, `kv_splits=2`)
and `variable-kvtail-512tiles` (512 query tiles: B2 H4 x 64 tiles, 40 KV
blocks, ten-block lists, a 47-token KV tail marked by `block_sizes`).

| Test method | Level | Parameters and purpose | Cases |
| --- | --- | --- | ---: |
| `TestBlk64Forward::test_replay` | L0 | The four configurations in BF16 and FP16: eight executions with fresh buffers and side-stream work, input mutation check, FP32 reference | 8 |
| `TestBlk64Forward::test_batch_invariance` | L0 | The four configurations in BF16 and FP16: a batch element alone against its slice of the packed, reversed, duplicated and neighbor-duplicated batches, with reversed heads, and with changed neighbor data and masks (shared block sizes) | 8 |
| `TestBlk64Forward::test_golden` | L0 | BF16, FP16: B1 H2, three query blocks (one empty), four KV blocks (sizes 64/40/64/23), two-block lists; three executions, reference, saved hashes | 2 |
| `TestBlk64Backward::test_replay` | L0 | Routes `regular` (one query group: direct dK/dV stores), `grouped` (`bucket_size_blocks=5`: four query groups, FP32 dK/dV workspace), `paired` (paired-KV kernel, task gate forced to 0, four query groups), `prepared-regular` and `prepared-paired` (SM90: `block_sparse_attention_backward_metadata` + `metadata=`); B3 H2, 16 x 16 blocks, six-block lists with empty rows and partial blocks; eight executions, dispatcher decisions asserted, FP32 reference, empty-row dQ zero; dQ/dK/dV bytes measured, asserted only for the prepared-regular dK/dV | 5 |
| `TestBlk64Backward::test_batch_invariance` | L0 | `regular` and `paired`: a batch element alone against the packed, reversed, duplicated and changed-neighbor batches; forward results asserted bitwise, gradients measured, route asserted | 2 |
| `TestBlk64Backward::test_prepared_parity` | L0 | SM90: prepared against default route on the regular and paired kernels, within the FP32 reduce-add noise (atol/rtol 1e-2), reused object, reference | 2 |
| `TestBlk64Backward::test_zero_upstream_gradient` | L0 | Eight executions with zero dO on the regular route; every gradient exactly zero, bitwise | 1 |
| `TestBlk64Backward::test_golden` | L0 | `regular` (fixed counts: B1 H2, two query blocks attending disjoint KV block pairs) and `paired` (variable counts, a third empty query block, head 1 attends one block) on order-free patterns; three executions bitwise, reference, SM90 prepared route bitwise equal, saved hashes | 2 |

On SM100/SM103 the nine FP16 forward cases (four replay, four batch, the
golden) and the four prepared-route cases skip by contract: the blk64 forward
there serves BF16 only, and `block_sparse_attention_backward_metadata` serves
SM90 only. Each skip is gated on the API itself (`skip_by_contract`): the case
first makes a tiny FP16 forward call or prepare call and requires the
documented rejection (`NotImplementedError` naming BF16, or the SM90-only
clause); a call the API accepts fails the case, so a widened contract cannot
keep skipping silently while the support page goes stale.

Known gap, stated rather than tolerated: the backward reduce-adds dQ in FP32
across the CTAs of a query block's KV blocks (`cp.reduce.async.bulk.add.f32`),
the FP32 dK/dV workspace and the paired kernel's later arrivers reduce-add
dK/dV the same way, and the K-to-Q CSR build orders each KV block's query list
with an atomic cursor, so even the direct dK/dV stores of the single-group
route accumulate in a call-dependent order. The backward replay and batch
cases record the differing byte counts of dQ, dK and dV (`record_property`,
the captured output, and the run's `measured` terminal summary through
`frost_routing.measured`, which reaches the CI job log of a passing run) and
assert bytes only where the arithmetic is order-free: the prepared regular
route's dK/dV (one CSR build, one owner CTA per KV block, direct stores) and
the golden patterns (one query block per KV block, at most two reduce-adds
onto zero per dQ tile). The measured counts are in the dated record above; a
small golden establishes bytes for its pattern and nothing more.

### SDPA MXFP8 completed-block causal: 15 cases

`sdpa/frost/test_sdpa_mxfp8_own_block_determinism.py` covers `sdpa_mxfp8` with `use_causal_mask=True, causal_exclude_own_block=32`: each query reads only the keys of completed 32-token blocks before its own (NVIDIA/TransformerEngine#3248). The plan is pinned to the SM100 MXFP8 engine. PackGQA keeps each row's key order, so packed and unpacked plans write the same bytes. Split-KV adds per-split partials in a fixed order: it replays bit for bit, but its bytes follow the split count, which the heuristics pick from the launch grid (on B200, only a single sequence at one or two KV heads splits); the batch-invariance cases run unsplit plans. Run it with `requires_pre_rubin_blackwell`; other architectures skip, and SM90/SM107 decline the option by contract.

| Method | Parameters | Cases |
| --- | --- | ---: |
| `test_replay` | E4M3/E5M2 × MHA 8:8 / GQA 8:2; B4 S2048 (512 work tiles); eight executions with side-stream contention, fresh NaN-poisoned outputs, input-byte mutation check | 4 |
| `test_batch_invariance` | Same combinations, S512; four sequences alone, packed, reordered, each duplicated, and repeated into a 16-sequence batch that crosses one wave | 4 |
| `test_neighbor_isolation` | E4M3/E5M2; fixed target payload and scales beside a requantized neighbour, then a scale-only neighbour change (every V E8M0 scale one exponent up); the neighbour's bytes must change | 2 |
| `test_pack_gqa_keeps_bytes` | E4M3/E5M2, B2 GQA 8:2 S1024; pinned PackGQA and unpacked plans on the same packed bytes | 2 |
| `test_split_kv_replay` | E4M3, B1 H1 S2048, pinned split-KV plan (4 splits on B200); eight executions with side-stream contention | 1 |
| `test_pack_reuses_quantized_bytes` | Harness control: per-sequence quantized payloads and swizzled scales, packed, equal the joint quantization's bytes and strides | 1 |
| `test_golden` | E4M3, B2 H4:2 S256; FP32 reference, three executions, payload/scale/output hashes | 1 |

Each sequence is quantized once, alone, from CPU-seeded FP32 inputs; batches reuse those payload and E8M0 scale bytes and never requantize. Comparisons cover O and Stats by raw bytes. Rows 0-31 keep no key: their O is zero and their Stats is the declared `-inf` sentinel, checked by location in `test_replay`. The numerical reference for the full shape grid is `test_sdpa_mxfp8_causal_exclude_own_block.py::test_numerics_against_reference`; `test_golden` re-checks its own inputs against the same reference before hashing.

Validation, 2026-11-14 (UTC): `NVIDIA B200`, SM100, 148 SMs, driver 580.178.04 (API `13040`); Torch `2.14.0a0+4fdf77b940.nv26.08`, CUDA `13.4`, cuDNN `92500`, CuTe DSL `4.8.1`, TVM FFI `0.1.13.post3`. Source: `develop` at `f0a0a1b1c43f2338193c10201a5dc467e917ec93` plus this change; imports resolved to that checkout (editable install). The golden was recorded on 2026-11-10 (`CUDNN_DETERMINISM_RECORD`, `-k golden`, into a new directory, after replay and the reference passed) and is unchanged. Steps: a normal run of both own-block suites at L0-L4 (**76 passed, 0 skipped**); a strict run of the 15 cases against the committed [golden](3851e622ca8e3564/sdpa-mxfp8-own-block32-fprop.json) with `CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens"`: **15 passed, 0 failed, 0 skipped**. SM103 is unverified: the kernel compiles for `sm_103a` with the option on and off, but no SM103 GPU ran it.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest -q -m L0 sdpa/frost/test_sdpa_mxfp8_own_block_determinism.py
```

## Golden checks

Each golden records SHA-256 hashes of all input and result bytes, with their
shapes and dtypes. This retains full precision without large tensor files.
The runtime key includes the GPU, SM count, Torch, CUDA, driver API version,
cuDNN, CuTe DSL, and TVM FFI.
The frontend revision is deliberately not part of the key: the check must
detect changes across frontend revisions on the same runtime.

Golden cases first pass replay and an independent numerical reference.
Linear attention uses the existing FP64 recurrent references on the CPU.
QSA uses its existing FP64 attention and gradient reference. DSA uses the
existing FP32 gather-based forward and autograd gradient references. Reference
tolerances test mathematical correctness; golden and replay checks use no
tolerance. A first execution is only a replay baseline, never a saved golden.

A normal run reads this directory. An unrecorded runtime skips only the golden
comparison; replay and batch checks still run. For CI, require a complete
runtime baseline with an explicit directory. Missing goldens then fail:

```bash
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  pytest linear_attention/frost/test_*_determinism.py QSA/cutedsl/test_qsa_determinism.py \
  deepseek_sparse_attention/cutedsl/test_DSA*determinism.py -m L0
```

Record candidates into a **new** directory on a known-good revision:

```bash
CUDNN_DETERMINISM_RECORD=/tmp/attention-goldens-candidate \
  pytest linear_attention/frost/test_*_determinism.py QSA/cutedsl/test_qsa_determinism.py \
  deepseek_sparse_attention/cutedsl/test_DSA*determinism.py -m L0 -k golden
```

Recording refuses to overwrite an existing file. Check every golden supported on the target architecture, inspect
the input/output hashes, and explain any numerical change before replacing
committed files. Never enable recording in a PR check. A new runtime needs its
own reviewed baseline; matching results across architectures is not assumed.

## Agent and review rules

Follow [the root agent rules](../../../AGENTS.md#kernel-determinism-and-batch-invariance)
and [the test guide](../../AGENTS.md). The PR template requires an accompanying
determinism test change for kernel changes, or an explicit unsupported-path
explanation. Extend this table and the suites when adding an operation or a
new numerical path. Keep hardware/dependency skips specific; a missing FROST
plan on a supported runtime is a failure. Explicitly requested temporary skips
must name the defect, retain the recorded failures, and be removed with its fix.

The earlier suites cover FROST and public QSA/DSA APIs on SM100/SM103/SM107.
The quantized indexer suite adds SM90 FP8 and SM100/SM103/SM107 MXFP8.
The set excludes cuTile/CAKE, other Hopper paths, QSA indexed/two-pass APIs,
checkpoint/summary/decode paths, shared gate parameters, fused/separate Top-K,
paged/MXFP4 indexer forward, THD dense indexer backward, score recompute,
pooled DSA, and distributed recipes. Existing tests cover some paths separately.
Recipe CI and an automatic source-to-test gate remain integration work;
the checklist is a review safeguard.

## Temporary skips

The 21 cases below use `skip until fix` reasons, as requested. They remain L0.
The set is the union of failures recorded on SM100, SM103, and SM107. A pass
on one run does not clear a nondeterministic case that failed on another GPU.
Passing parameters, golden checks, and all FP8/MXFP8 cases remain active.

| Suite | Temporarily skipped parameters | Cases |
| --- | --- | ---: |
| QSA | Backward batch checks, all three head layouts | 3 |
| QSA | Sparse batch checks, both directions | 2 |
| QSA | Dispatch boundary at total513, both directions | 2 |
| QSA | Backward mask batch checks, all-selected and empty-rows | 2 |
| DSA attention | H128/D512/Top-K128 BF16 backward batch check | 1 |
| DSA attention | Backward query boundaries H64 Q129/Q257 and H128 Q74/Q75 | 4 |
| DSA indexer | Dense H64 replay/batch checks, BF16 and FP32 dK | 4 |
| DSA indexer | Sparse persistent rows, Top-K896/2048 | 2 |
| DSA indexer | Dense backward neighbor isolation | 1 |

These skips do not change API support status. The baseline measurements and
failure diagnostics below were recorded before these skips. Remove each skip
with the kernel fix and verify the affected cases on the supported GPUs.

### Skip validation: 2026-11-05 (UTC)

The three changed suites ran on SM100: **103 passed, 26 skipped, 0 failed**
in 131.55 seconds. Runtime key: `5f8a825586430e85`, as in the SM100 baseline
below. Imports resolved to this checkout. Golden recording was disabled and
all seven golden cases in these suites passed.

| Suite | Passed | Skipped |
| --- | ---: | ---: |
| QSA | 21 | 14 |
| DSA sparse attention | 43 | 5 |
| DSA indexer forward | 24 | 0 |
| DSA indexer backward | 15 | 7 |

The skipped case IDs exactly match the union of the recorded SM100/SM103/SM107
failures. All 285 original case IDs still collect with `-m L0`. Pre-commit and
`git diff --check` passed. Other suites and GPUs were not rerun for this change.

Run from `test/python`:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest QSA/cutedsl/test_qsa_determinism.py \
  deepseek_sparse_attention/cutedsl/test_DSA_determinism.py \
  deepseek_sparse_attention/cutedsl/test_DSA_indexer_determinism.py -m L0 -q -rs
```

## SM100 H64 persistent sparse attention forward validation: 2026-11-18 (UTC)

Change under test: the gather4 path of the SM100 H64 sparse attention forward
kernel is persistent. CTAs claim queries through cluster launch control and
run several queries each, so which CTA runs a query now depends on the launch;
kernel outputs are unchanged. `TestDSA` is unchanged.
`test_DSA_sparse_attention_forward.py` adds an L0 check that a persistent
launch equals, byte for byte, launches in which every CTA runs one query (D512,
D576, and D512 with `indexer_topk=512`). Source: this change on `develop`
`07c1b377a`. Only the DSA suite was rerun.

| GPU | Runtime | Result |
| --- | --- | --- |
| SM100 (cc 10.0, `NVIDIA Graphics Device`), 148 SMs | Torch `2.13.0+cu130`, CUDA `13.0`, driver API `13010`, cuDNN `92700`, CuTe DSL `4.7.0a0` (development build), TVM FFI `0.1.13.post3`; key `86bcefdfad455cd7` | **94 passed, 2 failed, 5 skipped** in 188 s: five temporary skips; both goldens failed with `no golden` (no baseline for this runtime) |

SM103 and SM107 were not run for this change.

From `test/python`:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest deepseek_sparse_attention/cutedsl/test_DSA_determinism.py -m L0 -q -rs
```

## SM100 H128 windowed sparse attention forward validation: 2026-11-11 (UTC)

Change under test: `DSA.sparse_attention_forward_wrapper` accepts H128 D576,
and H128 D512 with `indexer_topk=2048`, on SM100 by launching the H64 kernel
on heads [0, 64) and [64, 128); `TestDSA` adds the forward-only
`h128-d576-k64` layout (four cases). Source: this change on the H96 change
above. Only the DSA suite was rerun.

| GPU | Runtime | Result |
| --- | --- | --- |
| B300, SM103 (cc 10.3), 148 SMs | Torch `2.14.1+cu130`, CUDA `13.0`, driver API `13030`, cuDNN `92700`, CuTe DSL `4.8.1`, TVM FFI `0.1.15.post0`; key `55097120e8121c53` | **94 passed, 2 failed, 5 skipped**: the four H128 D576 and four H96 cases passed; five temporary skips; both goldens failed with `no golden` (no baseline for this runtime) |

H128 D576 backward is outside this change and is not in this suite.

## SM100 H96 sparse attention forward validation: 2026-11-11 (UTC)

Change under test: `DSA.sparse_attention_forward_wrapper` accepts H96 with
D512/D576 on SM100 by launching the H64 kernel on heads [0, 64) and [32, 96),
and `TestDSA` adds the forward-only `h96-d576-k64` layout (four cases). Source:
this change on `develop` `17e10b09e`. `cudnn` resolved to a package whose
Python modules link to this checkout, with the installed pybind module. Only
the DSA suite was rerun.

| GPU | Runtime | Result |
| --- | --- | --- |
| B200, SM100, 148 SMs | Torch `2.14.0a0+b2c75dd062.nv26.09`, CUDA `13.4`, driver API `13040`, cuDNN `92600`, CuTe DSL `4.8.0` (CUDA 13.4 compiler), TVM FFI `0.1.14`; key `5c945a33b1699e7e` | **90 passed, 2 failed, 5 skipped** in 177 s: the four H96 cases passed; five temporary skips; both goldens failed with `no golden` (no baseline for this runtime) |
| B300, SM103 (cc 10.3), 148 SMs | Same software runtime; key `b28819ca225d3a0b` | **90 passed, 2 failed, 5 skipped** in 289 s: the four H96 cases passed; five temporary skips; both goldens failed with `no golden` (no baseline for this runtime) |
| SM107 (cc 10.7, `NVIDIA Graphics Device`), 208 SMs | Same software runtime; key `1c7a6611734bfd74` | **45 passed, 2 failed, 50 skipped** in 23 s: the four H96 cases passed; five temporary skips and 45 `TestDSADeterministicDKV` cases that run only on SM100/SM103; both goldens failed with `no golden` (no baseline for this runtime) |

H96 backward is outside this change and is not in this suite.

From `test/python`:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest deepseek_sparse_attention/cutedsl/test_DSA_determinism.py -m L0 -q -rs
```

## Deterministic dKV storage validation: 2026-11-06 (UTC)

Change under test: conflict-rank dKV storage for deterministic SM100/SM103 DSA
backward, the `deterministic_dkv_mode` option, conflict-rank plans that reduce
dSink like the shard plans, and `TestDSADeterministicDKV`. Source revision:
<RESULT revision>. The DSA suite was rerun; the other suites were not rerun
for this change and keep their earlier results.

The default plans of these `TestDSA` cases now select conflict-rank storage,
so their dKV bits change: `test_replay[h128-d512-k128-bf16-bprop]` and the
skipped H128 BF16 batch and query-boundary cases (fused H128 D512 layout), and
the packed batch of the skipped `test_query_boundary[h64-q257-bprop]`
(records). The golden cases (H64, D512, Top-K 16, query lengths 3 and 2) keep
the 32-shard layout; no golden record changes.

Run from `test/python`:

```bash
python -c 'import cudnn; print(cudnn.__file__)'
python -m pytest deepseek_sparse_attention/cutedsl/test_DSA_determinism.py -m L0 --collect-only -q
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest deepseek_sparse_attention/cutedsl/test_DSA_determinism.py -m L0 -q -rs
```

The B200 runs below omitted `CUDNN_DETERMINISM_GOLDENS` because their runtime
has no recorded baseline; the two golden cases then skip.

| GPU | Runtime | Result |
| --- | --- | --- |
| B200, SM100, 148 SMs | Torch `2.14.0a0+4fdf77b940.nv26.08`, CUDA `13.4`, driver API `13040`, cuDNN `92500`, CuTe DSL `4.8.0` (CUDA 13.4 compiler), TVM FFI `0.1.13.post3`; key `3c93e700c060c612` | **86 passed, 0 failed, 7 skipped** in 192 s: 45 of 45 `TestDSADeterministicDKV`, 41 `TestDSA`; five temporary skips, two goldens without a baseline |
| B200, SM100, 148 SMs | Same, CuTe DSL `4.8.0` with the CUDA 12.9 compiler | Earlier revision of this change, before the dSink reduction change: **85 passed, 0 failed, 7 skipped** in 186 s |
| SM100, recorded runtime `5f8a825586430e85` | <RESULT runtime> | <RESULT pytest summary, including both goldens> |
| B300, SM103, 148 SMs | <RESULT runtime> | <RESULT pytest summary> |

The new cases take about 140 s of a B200 run with empty compile caches,
mostly first-use compilation of their plans.

With the temporary skips removed locally (not committed), four of the five
`TestDSA` cases fail on B200 on dKV, as on `develop`; 42 passed and the two
goldens skipped. The H128 cases now fail at the first packed batch, because the
conflict-rank plane count follows the batch's KV extent.
`test_query_boundary[h64-q257-bprop]` passes on B200: the target alone selects
shards and the packed batch selects records, whose dQ and dSink match the shard
plans and whose dKV for this target adds the same contributions in the same
order. The case keeps its skip: SM107 plans keep the shard layouts, where it
failed on dKV ([SM107 validation](#sm107-validation)), and the skip set is the
union of the recorded failures.

| Skipped case | First mismatch with conflict-rank storage (B200) |
| --- | --- |
| `TestDSA::test_batch_invariance[h128-d512-k128-bf16-bprop]` | batch (0, 1), sequence 0 dkv: 8/164864 bytes differ |
| `TestDSA::test_query_boundary[h64-q129-bprop]` | batch (0, 1), sequence 0 dkv: 2/99328 bytes differ |
| `TestDSA::test_query_boundary[h64-q257-bprop]` | passes |
| `TestDSA::test_query_boundary[h128-q74-bprop]` | batch (0, 1), sequence 0 dkv: 8/164864 bytes differ |
| `TestDSA::test_query_boundary[h128-q75-bprop]` | batch (0, 1), sequence 0 dkv: 9/164864 bytes differ |

#### Batch layouts on `develop` and with this change (B200)

The same comparisons ran on `develop` (no `deterministic_dkv_mode`) and with
this change under `'shards'`, `'records'` and `'chain'`: each sequence alone
against its slice of the packed, reversed and duplicated batches (six
comparisons per configuration, other sequences' dO zero), neighbor isolation,
and replay, on FP16 and BF16. Configurations: the `TestDSA` sequences (lengths
17 and 3) for H64 D512, H64 D576 and H128 D512; H64 D512 Top-K 64 with lengths
17 and 300 and KV 1000 + 1000 (three records waves), and lengths 17 and 1183
(strided records waves), full and ragged selections; the same with KV
2048 + 2048 (Top-K 128 and 64) for the chain relay. Counts are comparisons out
of six with any differing byte.

| Configuration | `develop` | `'shards'` | `'records'` | `'chain'` |
| --- | --- | --- | --- | --- |
| `TestDSA` sequences, H64 D512/D576 and FP16 H128 | none differ | none differ | none differ (shard layout selected) | none differ (shard layout selected) |
| `TestDSA` sequences, BF16 H128 D512 (two-CTA) | dKV 1, dSink 2 | dKV 1, dSink 2 | dKV 3, dSink 2 | rejected (two-CTA backend) |
| Three waves, FP16 and BF16, full / ragged | dKV 3 / 1 | dKV 3 / 1 | dKV 3 / 3 | dKV 5 / 2-3 |
| Strided waves, FP16 and BF16, full / ragged | dKV 3 / 1 | dKV 3 / 1 | dKV 4-5 / 3 | dKV 4-5 / 3 |
| KV 2048 + 2048, three and strided waves, FP16 and BF16, full / ragged | not run | dKV 1-3 / 1-2 | dKV 3 / 3 | dKV 3-5 / 2-3 |

Forward results, dQ, replay and neighbor isolation matched in every
configuration. Every generic configuration kept dSink under all three modes;
records and chain dSink is bitwise the shard plans' dSink for the same inputs.
`develop` and `'shards'` gave the same per-comparison counts. FP16 DSA backward
on SM100/SM103 is listed as Replay only because `develop`'s shard layouts
already change dKV at these multi-wave shapes; the earlier Supported status
reflected the single-wave `TestDSA` shapes only.

SM107 was not run for this change: its plans keep the shard layouts that
`TestDSA` measured on SM107, and `TestDSADeterministicDKV` skips there.

## Baseline validation: 2026-11-04

Kernel base: `da88207aa5b9197b0c84514a7321d5bc3d2683b4` (`develop`).
GPU: SM100, 148 SMs. Runtime: Torch `2.15.0.dev20260820+cu134`, CUDA `13.4`,
cuDNN `92500`, CuTe DSL `4.8.0.dev0`, TVM FFI `0.1.12`.
The complete runtime record is in [the goldens](5f8a825586430e85/dsa-fprop.json).
The test banners confirmed imports from this checkout.

Three disjoint default-L0 runs on SM100 covered all 285 cases. Run from `test/python`:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4
export CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens"
python -m pytest core/test_determinism_utils.py \
  deepseek_sparse_attention/cutedsl/test_DSA_determinism.py \
  deepseek_sparse_attention/cutedsl/test_DSA_indexer_determinism.py -q
python -m pytest linear_attention/frost/test_*_determinism.py \
  QSA/cutedsl/test_qsa_determinism.py -q
python -m pytest deepseek_sparse_attention/cutedsl/test_DSA_indexer_fp8_determinism.py -q
```

- DSA attention, indexer, and harness: **89 passed, 11 failed, 0 skipped** in 128.77 seconds.
- Existing five operation suites: **137 passed, 14 failed, 0 skipped** in 489.44 seconds.
- New FP8/MXFP8 module: **17 passed, 0 failed, 17 skipped** in 20.17 seconds.
- Combined: **243 passed, 25 failed, 17 skipped**. Node IDs match the
  285 cases selected by default L0 collection (0.29 seconds). The five existing
  operation suites retain their earlier results; their code is unchanged by
  the indexer addition.

| Suite | fprop | bprop |
| --- | --- | --- |
| KDA | 13 passed | 15 passed |
| GDP | 15 passed | 17 passed |
| GDN | 13 passed | 15 passed |
| GDN-2 | 13 passed | 15 passed |
| QSA | 15 passed, 2 failed | 6 passed, 12 failed |
| DSA sparse attention | 24 passed | 19 passed, 5 failed |
| DSA BF16 indexer forward | 24 passed | — |
| DSA sparse indexer backward | — | 13 passed, 2 failed |
| DSA dense indexer backward | — | 3 passed, 4 failed |
| DSA MXFP8 indexer (SM100) | 17 passed | Unsupported |
| DSA FP8 indexer (SM90) | 17 passed on SM90; 17 skipped on SM100 | Unsupported |

All six harness controls and all sixteen goldens runnable on SM100 passed,
including replay and independent references. SM90 also passed its FP8 golden
on Hopper. MXFP8 and Hopper FP8 each add one record; the fifteen earlier
records are unchanged. Golden recording was disabled in validation. Sparse indexer large-Top-K cases pass replay and
reference, then fail dK after reorder. Dense indexer backward fails dK replay,
batch, or neighbor checks. The [API-by-dtype results](#api-and-datatype-results)
and [SM100 indexer failure list](#indexer-failures-sm100) record the measurements.

Pre-commit (Black, Black Jupyter, SPDX), Python syntax, and `git diff --check`
passed. Runtime warnings were Torch context/debug notices and existing CuTe
DSL scheduler/allocator deprecations. `fern check --warnings` found zero
errors; the remote redirect check was skipped after an FDR HTTP 403 response.

### API and datatype results

Each cell is **passed / failed / skipped** for the 2026-11-04 runs.
Counts are expanded pytest cases, each assigned once to its direction.
Backward cases also compare forward results where applicable. A case stops at
its first mismatch; later comparisons do not count as passes.

| API | Direction | Datatype | SM90 Hopper | SM100 | SM103 | SM107 Rubin |
| --- | --- | --- | --- | --- | --- | --- |
| `kimi_delta_attention` | fprop | BF16 | 0 / 0 / 10 | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |
| `kimi_delta_attention` | fprop | FP16 | 0 / 0 / 3 | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 / 0 |
| `kimi_delta_attention` | bprop | BF16 | 0 / 0 / 12 | 12 / 0 / 0 | 12 / 0 / 0 | 12 / 0 / 0 |
| `kimi_delta_attention` | bprop | FP16 | 0 / 0 / 3 | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 / 0 |
| `gated_delta_product` | fprop | BF16 | 0 / 0 / 12 | 12 / 0 / 0 | 12 / 0 / 0 | 12 / 0 / 0 |
| `gated_delta_product` | fprop | FP16 | 0 / 0 / 3 | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 / 0 |
| `gated_delta_product` | bprop | BF16 | 0 / 0 / 14 | 14 / 0 / 0 | 14 / 0 / 0 | 14 / 0 / 0 |
| `gated_delta_product` | bprop | FP16 | 0 / 0 / 3 | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 / 0 |
| `gated_delta_net` | fprop | BF16 | 0 / 0 / 10 | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |
| `gated_delta_net` | fprop | FP16 | 0 / 0 / 3 | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 / 0 |
| `gated_delta_net` | bprop | BF16 | 0 / 0 / 12 | 12 / 0 / 0 | 12 / 0 / 0 | 12 / 0 / 0 |
| `gated_delta_net` | bprop | FP16 | 0 / 0 / 3 | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 / 0 |
| `gated_delta_net_v2` | fprop | BF16 | 0 / 0 / 10 | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |
| `gated_delta_net_v2` | fprop | FP16 | 0 / 0 / 3 | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 / 0 |
| `gated_delta_net_v2` | bprop | BF16 | 0 / 0 / 12 | 12 / 0 / 0 | 12 / 0 / 0 | 12 / 0 / 0 |
| `gated_delta_net_v2` | bprop | FP16 | 0 / 0 / 3 | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 / 0 |
| `qsa_forward` | fprop | BF16 | 0 / 0 / 17 | 15 / 2 / 0 | 15 / 2 / 0 | 15 / 2 / 0 |
| `qsa_backward` | bprop | BF16 | 0 / 0 / 18 | 6 / 12 / 0 | 7 / 11 / 0 | 6 / 12 / 0 |
| `DSA.sparse_attention_forward_wrapper` | fprop | BF16 | 0 / 0 / 18 | 18 / 0 / 0 | 18 / 0 / 0 | 18 / 0 / 0 |
| `DSA.sparse_attention_forward_wrapper` | fprop | FP16 | 0 / 0 / 6 | 6 / 0 / 0 | 6 / 0 / 0 | 6 / 0 / 0 |
| `DSA.sparse_attention_backward_wrapper` | bprop | BF16 | 0 / 0 / 18 | 13 / 5 / 0 | 13 / 5 / 0 | 14 / 4 / 0 |
| `DSA.sparse_attention_backward_wrapper` | bprop | FP16 | 0 / 0 / 6 | 6 / 0 / 0 | 6 / 0 / 0 | 6 / 0 / 0 |
| `DSA.indexer_forward_wrapper` | fprop | BF16 Q/K; BF16 weights | 0 / 0 / 15 | 15 / 0 / 0 | 15 / 0 / 0 | 15 / 0 / 0 |
| `DSA.indexer_forward_wrapper` | fprop | BF16 Q/K; FP32 weights | 0 / 0 / 9 | 9 / 0 / 0 | 9 / 0 / 0 | 9 / 0 / 0 |
| `DSA.indexer_backward_wrapper` | bprop | BF16 Q/K/W; BF16 dK | 0 / 0 / 4 | 4 / 0 / 0 | 4 / 0 / 0 | 4 / 0 / 0 |
| `DSA.indexer_backward_wrapper` | bprop | BF16 Q/K/W; FP32 dK | 0 / 0 / 11 | 9 / 2 / 0 | 9 / 2 / 0 | 9 / 2 / 0 |
| `DSA.dense_indexer_backward_wrapper` | bprop | BF16 Q/K/W; BF16 dK | 0 / 0 / 2 | 1 / 1 / 0 | 1 / 1 / 0 | 1 / 1 / 0 |
| `DSA.dense_indexer_backward_wrapper` | bprop | BF16 Q/K/W; FP32 dK | 0 / 0 / 5 | 2 / 3 / 0 | 2 / 3 / 0 | 2 / 3 / 0 |
| `DSA.indexer_forward_wrapper` | fprop | MXFP8 E4M3 Q/K; E8M0 scales; BF16 weights | 0 / 0 / 17 | 17 / 0 / 0 | 17 / 0 / 0 | 17 / 0 / 0 |
| `DSA.indexer_forward_wrapper` | fprop | SM90 FP8 E4M3 Q/K; FP32 scales; BF16 weights | 17 / 0 / 0 | 0 / 0 / 17 | 0 / 0 / 17 | 0 / 0 / 17 |

The table covers 279 operation cases per GPU. Six harness controls also passed
on each GPU. Hopper attention and BF16 indexer skips are coverage gaps; they
do not establish that the APIs are unsupported. The other precision group's
FP8/MXFP8 cases skip on each architecture. Runtime details follow below.

Run the full selection from `test/python`, with imports from this checkout:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4
export CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens"
python -m pytest core/test_determinism_utils.py \
  linear_attention/frost/test_*_determinism.py \
  QSA/cutedsl/test_qsa_determinism.py \
  deepseek_sparse_attention/cutedsl/test_DSA*determinism.py -q
```

Add `--collect-only` to list the 285 L0 cases. The explicit golden directory
makes missing runtime baselines fail. Keep golden recording disabled.

### SM90 Hopper validation

The same 285-case default-L0 selection had **23 passed, 0 failed, 262 skipped**
in 13.71 seconds. All 17 FP8 indexer cases and six harness controls passed.
The other cases skip at their architecture gates. This does not measure
attention or BF16 indexer behavior on Hopper.

Runtime: SM90, 124 SMs; Torch `2.14.0a0+b2c75dd062.nv26.09`, CUDA `13.4`,
driver API `13040`, cuDNN `92700`, CuTe DSL `4.8.0`, TVM FFI `0.1.14.post1`.
The [Hopper golden](ffe6ff35e0022211/dsa-indexer-fp8-fprop.json) was recorded
only after replay and the FP64 reference passed, then checked again with
recording disabled. It includes E4M3 input bytes, FP32 scales, weights, offsets,
scores, and LSE. Test source came from commit
`35cf3082e5f3e505b066097f29bf232a848ea269`; imports resolved to that snapshot.

### SM103 validation

All 285 L0 cases ran: **244 passed, 24 failed, 17 skipped** in 482.93 seconds.
The 17 skips are SM90-only FP8 cases. All 17 MXFP8 indexer cases, six harness
controls, and 16 operation goldens passed. Failures: 13 QSA, five DSA attention
backward, and six BF16 indexer backward cases.

GPU: `NVIDIA B300 SXM6 AC`, SM103, 148 SMs. Torch
`2.14.0a0+b2c75dd062.nv26.09`, CUDA `13.4`, driver API `13040`, cuDNN `92700`,
CuTe DSL `4.8.0`, TVM FFI `0.1.14.post1`. Source snapshot:
`8671c79c2a1e7073bf1efe0cce1939f093a0d840`; imports resolved to that snapshot.
The [SM103 goldens](7a727a3958771fbc) passed replay and independent references
before a separate run with recording disabled. The same full-suite command
listed under [API and datatype results](#api-and-datatype-results) selected all 285 cases.

These 16 records add a new GPU/runtime baseline. Existing records are unchanged.
Compared with SM100, linear-attention gate input hashes differ; the other input
hashes match. The numerical reference checks use each run's actual inputs.
Software versions differ between SM100 and SM103, so these are separate
measurements, not a hardware-only comparison.

| SM103 failed case | First mismatch |
| --- | --- |
| `TestQSA::test_replay[bprop-q12-kv1]` | replay 1 dk: 2/43520 bytes differ |
| `TestQSA::test_replay[bprop-q24-kv2]` | replay 1 dk: 3/87040 bytes differ |
| `TestQSA::test_batch_invariance[bprop-q12-kv1]` | batch (1, 0), sequence 0 dq: 1/399360 bytes differ |
| `TestQSA::test_batch_invariance[bprop-q24-kv2]` | batch (0,), sequence 0 dk: 2/66560 bytes differ |
| `TestQSA::test_batch_invariance[bprop-q6-kv1]` | batch (1, 0), sequence 0 dq: 1/199680 bytes differ |
| `TestQSA::test_sparse_batch_invariance[fprop]` | sparse sequence 1 out_fp32: 237/36864 bytes differ |
| `TestQSA::test_sparse_batch_invariance[bprop]` | replay 1 dk: 142/1054720 bytes differ |
| `TestQSA::test_dispatch_boundary[total513-fprop]` | total 513, target position 0 out_fp32: 196/36864 bytes differ |
| `TestQSA::test_dispatch_boundary[total513-bprop]` | total 513, target position 0 out_fp32: 196/36864 bytes differ |
| `TestQSA::test_mask_replay[all-selected-bprop]` | replay 1 dv: 1/43520 bytes differ |
| `TestQSA::test_mask_replay[empty-rows-bprop]` | replay 3 dk: 1/43520 bytes differ |
| `TestQSA::test_mask_batch_invariance[all-selected-bprop]` | mask all-selected, batch (0,), sequence 0 dv: 2/33280 bytes differ |
| `TestQSA::test_mask_batch_invariance[empty-rows-bprop]` | mask empty-rows, batch (0,), sequence 0 dk: 1/33280 bytes differ |
| `TestDSA::test_batch_invariance[h128-d512-k128-bf16-bprop]` | batch (1, 0), sequence 0 dkv: 4/164864 bytes differ |
| `TestDSA::test_query_boundary[h64-q129-bprop]` | batch (0, 1), sequence 0 dkv: 2/99328 bytes differ |
| `TestDSA::test_query_boundary[h64-q257-bprop]` | batch (0, 1), sequence 0 dkv: 2/99328 bytes differ |
| `TestDSA::test_query_boundary[h128-q74-bprop]` | batch (1, 0), sequence 0 dkv: 6/164864 bytes differ |
| `TestDSA::test_query_boundary[h128-q75-bprop]` | batch (1, 0), sequence 0 dkv: 6/164864 bytes differ |
| `TestDSAIndexerBackward::test_replay[dk-bf16-dense-h64]` | replay 5 d_index_k: 1/131072 bytes differ |
| `TestDSAIndexerBackward::test_replay[dk-fp32-dense-h64]` | replay 1 d_index_k: 2038/262144 bytes differ |
| `TestDSAIndexerBackward::test_batch_invariance[dk-fp32-dense-h64]` | batch (0, 1), sequence 0 d_index_k: 989/131072 bytes differ |
| `TestDSAIndexerBackward::test_sparse_persistent_rows[topk896]` | batch (1, 0), sequence 0 d_index_k: 132237/1048576 bytes differ |
| `TestDSAIndexerBackward::test_sparse_persistent_rows[topk2048]` | batch (1, 0), sequence 0 d_index_k: 202412/1048576 bytes differ |
| `TestDSAIndexerBackward::test_neighbor_isolation[dense]` | changed neighbor d_index_k: 513/131072 bytes differ |

### SM107 validation

All 285 L0 cases ran: **244 passed, 24 failed, 17 skipped** in 399.55 seconds.
The 17 skips are SM90-only FP8 cases. All 17 MXFP8 indexer cases, six harness
controls, and 16 operation goldens passed. Failures: 14 QSA, four DSA attention
backward, and six BF16 indexer backward cases.

GPU: `NVIDIA VR NVL72`, SM107, 212 SMs, on ARM64. Torch
`2.14.0a0+b2c75dd062.nv26.09`, CUDA `13.4`, driver API `13050`, cuDNN `92700`,
CuTe DSL `4.8.0`, TVM FFI `0.1.14.post1`. Source snapshot:
`35cf3082e5f3e505b066097f29bf232a848ea269`; imports resolved to that snapshot.
Test and kernel source match the SM103 run. The [SM107 goldens](72e27159dc117014)
passed replay and independent references before a separate full-suite run
with recording disabled. The same command selected all 285 L0 cases.

These 16 records add a new GPU/runtime baseline. Existing records are unchanged.
Some CPU-generated input hashes differ from the x86 runs; references use the
actual inputs. No cross-device byte equality is inferred from separate passes.

| SM107 failed case | First mismatch |
| --- | --- |
| `TestQSA::test_replay[bprop-q12-kv1]` | replay 1 dk: 1/43520 bytes differ |
| `TestQSA::test_replay[bprop-q24-kv2]` | replay 1 dk: 1/87040 bytes differ |
| `TestQSA::test_replay[bprop-q6-kv1]` | replay 1 dv: 2/43520 bytes differ |
| `TestQSA::test_batch_invariance[bprop-q12-kv1]` | batch (0,), sequence 0 dk: 2/33280 bytes differ |
| `TestQSA::test_batch_invariance[bprop-q24-kv2]` | batch (0,), sequence 0 dk: 1/66560 bytes differ |
| `TestQSA::test_batch_invariance[bprop-q6-kv1]` | batch (0,), sequence 0 dk: 1/33280 bytes differ |
| `TestQSA::test_sparse_batch_invariance[fprop]` | sparse sequence 1 out_fp32: 237/36864 bytes differ |
| `TestQSA::test_sparse_batch_invariance[bprop]` | replay 1 dk: 140/1054720 bytes differ |
| `TestQSA::test_dispatch_boundary[total513-fprop]` | total 513, target position 0 out_fp32: 196/36864 bytes differ |
| `TestQSA::test_dispatch_boundary[total513-bprop]` | total 513, target position 0 out_fp32: 196/36864 bytes differ |
| `TestQSA::test_mask_replay[all-selected-bprop]` | replay 4 dv: 1/43520 bytes differ |
| `TestQSA::test_mask_replay[empty-rows-bprop]` | replay 1 dv: 1/43520 bytes differ |
| `TestQSA::test_mask_batch_invariance[all-selected-bprop]` | mask all-selected, batch (0,), sequence 0 dk: 1/33280 bytes differ |
| `TestQSA::test_mask_batch_invariance[empty-rows-bprop]` | mask empty-rows, batch (0,), sequence 0 dk: 1/33280 bytes differ |
| `TestDSA::test_query_boundary[h64-q129-bprop]` | batch (0, 1), sequence 0 dkv: 2/99328 bytes differ |
| `TestDSA::test_query_boundary[h64-q257-bprop]` | batch (0, 1), sequence 0 dkv: 2/99328 bytes differ |
| `TestDSA::test_query_boundary[h128-q74-bprop]` | batch (1, 0), sequence 0 dkv: 9/164864 bytes differ |
| `TestDSA::test_query_boundary[h128-q75-bprop]` | batch (1, 0), sequence 0 dkv: 12/164864 bytes differ |
| `TestDSAIndexerBackward::test_replay[dk-bf16-dense-h64]` | replay 1 d_index_k: 1/131072 bytes differ |
| `TestDSAIndexerBackward::test_replay[dk-fp32-dense-h64]` | replay 1 d_index_k: 1843/262144 bytes differ |
| `TestDSAIndexerBackward::test_batch_invariance[dk-fp32-dense-h64]` | batch (0, 1), sequence 0 d_index_k: 1083/131072 bytes differ |
| `TestDSAIndexerBackward::test_sparse_persistent_rows[topk896]` | batch (1, 0), sequence 0 d_index_k: 132358/1048576 bytes differ |
| `TestDSAIndexerBackward::test_sparse_persistent_rows[topk2048]` | batch (1, 0), sequence 0 d_index_k: 202644/1048576 bytes differ |
| `TestDSAIndexerBackward::test_neighbor_isolation[dense]` | changed neighbor d_index_k: 1035/131072 bytes differ |

### Indexer failures (SM100)

**DSA BF16 indexer: 40 passed, 6 failed, 0 skipped.** The first mismatch in each failed case is dK.
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

### DSA failures (SM100)

All 24 fprop cases passed. Bprop had 19 passes and five failures, all in batch
invariance. Every replay and numerical-reference check reached in this suite
passed. The first mismatch in each failed case was dKV; forward results and
dQ matched. dSink comparison was not reached after those failures.

| Test | First mismatch |
| --- | --- |
| `test_batch_invariance[h128-d512-k128-bf16-bprop]` | batch (1, 0), sequence 0 dkv: 4/164864 bytes differ |
| `test_query_boundary[h64-q129-bprop]` | batch (0, 1), sequence 0 dkv: 2/99328 bytes differ |
| `test_query_boundary[h64-q257-bprop]` | batch (0, 1), sequence 0 dkv: 2/99328 bytes differ |
| `test_query_boundary[h128-q74-bprop]` | batch (1, 0), sequence 0 dkv: 6/164864 bytes differ |
| `test_query_boundary[h128-q75-bprop]` | batch (1, 0), sequence 0 dkv: 6/164864 bytes differ |

The H64 failures cross the documented 32/64/128-shard query-count thresholds.
The H128 BF16 failures move the target within the two-CTA path. Same-shape
replay passed: `deterministic=True` did not establish batch invariance for
these cases. No kernel changes, xfails, or relaxed comparisons were added.

### QSA failures (SM100)

The same 14 case IDs failed in the earlier run. First mismatch counts and
tensors can vary with the order of atomic reductions.

| Test | First mismatch |
| --- | --- |
| `test_replay[bprop-q12-kv1]` | replay 1 dk: 1/43520 bytes differ |
| `test_replay[bprop-q24-kv2]` | replay 1 dk: 1/87040 bytes differ |
| `test_replay[bprop-q6-kv1]` | replay 1 dk: 1/43520 bytes differ |
| `test_batch_invariance[bprop-q12-kv1]` | batch (0,), sequence 0 dk: 3/33280 bytes differ |
| `test_batch_invariance[bprop-q24-kv2]` | batch (0,), sequence 0 dk: 1/66560 bytes differ |
| `test_batch_invariance[bprop-q6-kv1]` | batch (0,), sequence 0 dk: 1/33280 bytes differ |
| `test_sparse_batch_invariance[fprop]` | sparse sequence 1 out_fp32: 237/36864 bytes differ |
| `test_sparse_batch_invariance[bprop]` | replay 1 dk: 136/1054720 bytes differ |
| `test_dispatch_boundary[total513-fprop]` | total 513, target position 0 out_fp32: 196/36864 bytes differ |
| `test_dispatch_boundary[total513-bprop]` | total 513, target position 0 out_fp32: 196/36864 bytes differ |
| `test_mask_replay[all-selected-bprop]` | replay 1 dv: 3/43520 bytes differ |
| `test_mask_replay[empty-rows-bprop]` | replay 1 dk: 1/43520 bytes differ |
| `test_mask_batch_invariance[all-selected-bprop]` | mask all-selected, batch (0,), sequence 0 dv: 3/33280 bytes differ |
| `test_mask_batch_invariance[empty-rows-bprop]` | mask empty-rows, batch (0,), sequence 0 dk: 1/33280 bytes differ |

QSA backward [documents floating-point atomic reductions](../../../docs/operations/sparse_attention/qsa.md).
The dK/dV replay changes are consistent with changing reduction order. Sparse
bprop fails replay before the batch comparison. Both 513-token cases first
fail on saved FP32 output, before gradient comparison; BF16 output matches.
Both 512-token controls pass. All-masked rows, changed-neighbor isolation,
and zero-upstream-gradient QSA cases pass.

### SM90 compact IndexerForward + Top-K validation (2026-11-09)

The new packed-THD ratio-4, D128, 64Q/1KV combined path ran **9 passed, 0
failed, 0 skipped** for BF16 and E4M3 FP8 on SM90 with Torch `2.12.1+cu130`,
CUDA `13.0`, cuDNN `92000`, and CuTe DSL `4.8.1`. Imports resolved to the
tested checkout. The suite covers raw-byte replay under side-stream pressure,
batch reorder/duplication and changed-neighbor isolation, ragged/empty/short
segments, tail rows, exact ties, compile counts, buffer-alias rejection,
nonzero causal offsets, explicit-side-stream internal sizing, and changed-input
CUDA Graph replay.
Independent FP32 references validate selected indices, logits, softmax, and
LSE, including caller-owned LSE-only output. The two existing SM90 Indexer
Forward suites plus this suite ran **41 passed, 66 skipped**. Backward is
outside this combined forward API.

The focused suite command was:

```bash
cd test/python
pytest -q -s deepseek_sparse_attention/cutedsl/test_DSA_indexer_compressed_sm90_determinism.py
```

### SM100 IndexerForward regression (2026-11-09)

The existing compressed IndexerForward + Top-K suite ran **96 passed, 0
failed** on SM100, including BF16/block-scaled numerics, deterministic ties,
compile-cache reuse, preallocated buffers, and CUDA Graph cases. The general
IndexerForward suite ran **95 passed, 3 skipped**. The tested checkout used
Torch `2.14.1+cu130`, CUDA `13.0`, cuDNN `92400`, and CuTe DSL `4.8.1`.

## Method

- [Megatron module tests](https://github.com/NVIDIA/Megatron-LM/tree/main/tests/unit_tests/determinism): fixed inputs, restored state, outputs and gradients.
- [Megatron kernel replay PR](https://github.com/NVIDIA/Megatron-LM/pull/7148): raw-byte comparison, side-stream contention, negative controls, and coverage review.
- [Nemotron functional tests](https://github.com/NVIDIA/Megatron-LM/tree/main/tests/functional_tests/test_cases/nemotron): deterministic execution and saved regression values.
- Internal recipe CI (not linked from this public repository): follow-up integration target; its contents were not accessible during this change.

### Hopper compact H32 sparse indexer backward

Verified after the rebase on 2026-11-09 with NVIDIA L20X SM90 (132 SMs),
PyTorch 2.12.1+cu130, cuDNN 9.20.0, and CuTe DSL 4.8.0 compiling with the
CUDA 13.4 flavor. The checkout was the exact pushed commit `f4540d88` on
develop parent `7a68feff`. `cudnn`, the API module, and the SM90 kernel all
resolved from `/tmp/cudnn-fe-h32-f4540d88/python`, reporting frontend version
1.33.0; the host's installed frontend 1.31.0 was not used.

The L0 `test_sm90_deterministic_h32_topk512` cases cover BF16 and FP32 dK,
sparse negative IDs, replay, short five-query batch reordering with global
IDs, and an independent gradient reference. Strict replay against the
committed runtime key `70c200b13182bbc2` passed 2/2 cases in 7.63 seconds. The
L0 support-envelope cells passed 24/24 in 0.38 seconds. Four H32 L0 stress
cases passed in 15.25 seconds, covering Top-K 384/512/2048, both deterministic
dW folds, BF16 and FP32 dK, local and global IDs, and 194 query rows over 96
producer planes. Those stress cases check fixed-layout replay and the
independent numerical reference; they do not compare batch reordering across
the 96-plane producer boundary. Because the producer-plane assignment depends
on the flattened batch row after that boundary, SM90 H32 is documented as
Replay only rather than batch-invariant. The complete indexer-backward L0 file
finished with 43 passed, 20 expected SM100-only skips, and 63 deselected in
32.64 seconds.

The RED control used parent `7a68feff` with the new test file but without the
implementation. Both H32 core cases failed at admission with
`SM90 IndexerBackward requires H=64, got H=32`, confirming that the green run
exercised this change. Commands run with `PYTHONPATH` pinned to the checkout:

```bash
python -m pytest deepseek_sparse_attention/cutedsl/test_DSA_indexer_backward.py \
  -m L0 -k sm90_deterministic_h32_topk512 -q -rs
python -m pytest deepseek_sparse_attention/cutedsl/test_DSA_indexer_backward.py \
  -m L0 -k check_support_envelope -q -rs
python -m pytest deepseek_sparse_attention/cutedsl/test_DSA_indexer_backward.py \
  -m L0 -k 'deterministic_reproducible_and_correct and h32' -q -rs
python -m pytest deepseek_sparse_attention/cutedsl/test_DSA_indexer_backward.py \
  -m L0 -q -rs
```

The CUDA 12 compiler flavor was unavailable and remains unverified.

### 2026-11-09: native SM107 per-tensor FP8 paged D512

`TestNativeD512Fp8Paged` in `sdpa/frost/test_sdpa_fp8_sm107.py` exercises
the native SM107 role-split kernel and the shared prepared binding. The
implementation was based on `fc4490c88557` (the SM100/SM103 paging addition),
with matching FE 1.33.0 bindings rebuilt for Python 3.12.12. Runtime:
Torch 2.14.1+cu130, cuDNN 9.30.0.2, CuTe DSL 4.8.0 (CUDA 13.4 compiler).
The imported package was the isolated qualification source, not an editable
installation of another checkout.

Coverage includes E4M3/E5M2 inputs, BF16/FP16/scalar-FP8 outputs, FLOAT and
HALF softmax, page8/16/32/64/128/256, independent K/V page tables and pool
layouts, runtime strides, fresh buffers, changed-page/scale capture replay,
empty and partial pages, causal/windows/sinks, split1/4, whole-group packing,
and the aligned D384/V320 envelope with poisoned storage padding. Capability
checks retain the smaller FP8 flavors', half/MXFP8 D512 and THD-Q declines.

Six golden records in `0972294c7bd4a660/` cover four FLOAT and two HALF
softmax configurations. Quantized payloads and scale bytes stay fixed across
sequences alone, reordered, duplicated and mixed with changed neighbors,
including 129-request multi-wave batches. O/Stats are compared as raw bytes
per sequence; whole-batch Amax is compared on identical repeated batches and
against the independent numerical reference, not misclassified as a per-row
output. Inputs and prepared owners are checked for unintended mutation.

Commands from the isolated `test/python` directory:

```bash
# After independent-reference/replay qualification, record into a fresh directory:
CUDNN_DETERMINISM_RECORD=/path/to/new-records python -m pytest \
  sdpa/frost/test_sdpa_fp8_sm107.py::TestNativeD512Fp8Paged -k golden -q
# Inspect/copy the six JSON records, then leave RECORD unset:
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" python -m pytest \
  sdpa/frost/test_sdpa_fp8_sm107.py::TestNativeD512Fp8Paged -q
```

Measured: 31 feature/strict-golden cases passed, with no failures or skips.
The existing native D512 Pack/Split suites additionally passed 32 selected
FLOAT-softmax controls (`-k 'not rebind_and_replay and not half_adapter'`).
One earlier HALF-adapter control reached its reference's FP64 `torch.exp2`
and failed because the installed NVRTC rejected the SM107 target; this
reference-environment failure is retained, not counted as a kernel pass.
The new HALF paging, replay and batch-invariance checks pass independently.

Compute Sanitizer 13.4.92: eight boundary/head-envelope cases under memcheck,
the same eight under synccheck, and two page8/page128 persistent cases under
racecheck; no memory/synchronization errors or race hazards. The older
13.2 sanitizer reported an unsupported device and is not a qualifying run.
No backward capability or other architecture's status is inferred from this
SM107 execution. No heuristics ranking changed.

The paged launch's generated SM107 cubin was also checked with nvdisasm
13.5.37: exactly one `USETMAXREG.DEALLOC` at 40 registers and one
`USETMAXREG.TRY_ALLOC` at 240. All four service warps release before role
branching. Dense launch/register behavior is unchanged.

### Native SM107 half D128 dense-Q paging (2026-11-10)

The architecture entry `test_sdpa_fwd_native_paged_sm107.py::TestNativeD128DensePaged`
collects the shared `native_d128_dense_paged_determinism.py` checks. Six
standalone cases pin native D128, CGA1/2, N64/N128 and split1/4, with FP16/BF16
Q/O and matching half or E4M3/E5M2 K/V. Each case checks fixed CPU-seeded inputs,
fresh buffers, queued independent work, reordered/duplicated batches and129
requests, then records O and FP32 Stats only after the independent CPU-double
reference and byte comparisons pass. Changed-input CUDA capture mutates Q,
Q/KV lengths and independent K/V table contents. Poisoned output/workspace,
output-padding canaries and input-byte preservation are checked. K/V physical
layouts differ; table strides match but are non-contiguous. Equal pool
capacities and matching table strides retain the existing half binder contract.

Graph tests separately cover partial GQA, page8–1024, short and prefill-shaped
queries, all-empty KV, head envelopes, sinks/windows and changed-input replay.
The new mixed-KV graph checks use the same independent double reference on CPU:
this runtime's Torch eager outer-product specialization invokes an older Triton
ptxas that does not recognize SM107. The native kernel remains on the GPU; no
kernel output, test tolerance or admission is substituted to accommodate it.

Runtime: SM107216SM, driver630.10, Python3.12.12, Torch2.14.1+cu130,
cuDNN9.30.0.2, CuTeDSL4.8.0/CUDA13.4 compiler, matching rebuilt FE1.33 bindings.
Final strict suite:61 passed, zero failures/skips, including all six goldens.
Related controls:23 passed (metadata and existing THD N64), plus24 L1 native
dense packing/split replay cases. Compute Sanitizer13.4.92: memcheck7,
synccheck7 and racecheck4 passed, zero diagnostics. The final published tests
also pass the smallest D32 head envelope boundary separately (1 case).
Other GPU results are not inferred from this native SM107 qualification.

### Native SM107 half D64 dense/THD paging (2026-11-10)

`test_sdpa_fwd_dsl_sm107.py::TestNativeD64Paged` qualifies physical D64
operands/MMA/output on the native D128-family kernel file. Parent D64 dense
paging selects the D128 envelope; parent THD D64 paging is declined. The new
route supports matching FP16/BF16 Q/K/V/O, CGA1/2, packed/unpacked execution,
N64/N128 and dense split-KV. Paged THD remains unsplit; mixed FP8 KV stays D128.

Eight fixed-CPU-payload goldens cover four dense and four compact-THD plans.
The dense checks reuse the D128 fixture at D64; THD additionally changes the
target sequence's packed origin across unrelated neighbors. Both verify
independent CPU-double references, O/Stats byte identity with fresh buffers,
queued stream work, repeated/reordered/duplicated batches through129 requests,
input preservation and untouched capacity/padding. Separate numerical graph
checks change page tables, lengths and tensor bindings under CUDA capture.

The current-parent suite passes105 cases without failures/skips:56 new D64
checks (32 dense,8 THD origins,8 goldens,7 boundaries,1 admission),46 D128
controls with unchanged golden bytes and3 architecture metadata controls.
Pages8–1024, Q1/2/17/129/257, partial head envelopes, empty/tail KV, asymmetric
pool layouts, partial GQA, windows and sinks are included. The existing THD
stage assertion is corrected for D64's fixed four-stage K/V ring.

Runtime: SM107216SM, driver630.10, cuDNN9.30.0.2, Torch2.14.1+cu130,
DSL4.8.0/CUDA13.4 and matching FE1.33 bindings. Goldens use runtime key
`0972294c7bd4a660`. No qualification of another architecture is implied.

Compute Sanitizer13.4.92:7 page-boundary memcheck cases,8 THD synccheck cases
and8 THD racecheck cases pass with zero diagnostics. No backward path changes.

Additional nonpaged D64 controls pass30 cases:22 dense split/replay and8
independent THD-origin cases.


### Native SM107 half D512 dense-Q paging (2026-11-10)

`test_sdpa_fwd_native_paged_sm107.py::TestNativeD512HalfPaged` pins the native SM107
2x2 kernel file, two-CTA MMA, twin-multicast K/V column shares and descriptor
version1. The new page-box loader preserves three-stage K/V rings and pair
transaction credits. Dense FP16/BF16 Q/K/V/O supports partial PackGQA and
split-KV; THD paging and mixed FP8 KV remain declined.

Four goldens cover FP16/BF16 and split1/4 at fixed CGA2/N128. The shared paging
fixture at D512 verifies independent CPU-double O/Stats references, raw bytes
under replay and queued stream work, fresh buffers, reordered/duplicated
batches through129 requests, input preservation, output-padding canaries and
changed Q/length/table contents under capture. Separate graph checks use
independent K/V table contents and HND/NHD layouts, including mixed pools.

The strict feature suite passes21 cases with no failures/skips:8 numerical
rebind/capture cases,8 boundaries,4 goldens and1 graph/config admission check.
Boundaries include page8–1024, Q1/2/4/17/129/257, empty/tail KV, windows/sinks,
partial GQA and D504/V480 and D264/V256 envelopes. The first boundary attempt
requested LPT for a plan whose existing domain is NATURAL-only; the final
suite pins the supported scheduler without widening its domain.

Runtime: SM107216SM, driver630.10, cuDNN9.30.0.2, Torch2.14.1+cu130,
DSL4.8.0/CUDA13.4 and matching FE1.33 bindings. Goldens use runtime key
`0972294c7bd4a660`. No new backward or other-GPU qualification is implied.

Existing nonpaged native 2x2 controls pass16 cases:8 dense graph cases,4
sink/Stats cases and4 THD cases. Compute Sanitizer13.4.92 passes8 page-boundary
memcheck cases,8 dense synccheck cases and8 dense racecheck cases, with zero
errors/hazards. Commands from `test/python`: `python -m pytest
sdpa/frost/test_sdpa_fwd_native_paged_sm107.py::TestNativeD512HalfPaged -q -x`; sanitizer
runs select `test_boundaries` (memcheck) or `test_dense` (synccheck/racecheck)
from the same class with `--error-exitcode 99`.


### Native SM107 half D576/V512 dense-Q paging (2026-11-10)

`test_sdpa_fwd_native_paged_sm107.py::TestNativeD576Half` pins the native
2x2 body with three192-column QK chunks, one CTA pair and three-stage rings.
The complete P ring requires descriptor version1 even though its first slot
starts below256KiB. Exact FP16/BF16 D576/V512 serves dense or paged K/V,
partial PackGQA and dense split-KV; THD and mixed FP8 KV remain declined.

Four new strict goldens cover paged FP16/BF16 and split1/4 at CGA2/N128.
The shared fixture gains an independent V width; prior defaults and golden
names are unchanged. Independent CPU-double O/Stats references, fresh-buffer
raw-byte replay with queued work, changed table/length/Q graph replay,
reordered and duplicated batches through129 requests, input preservation and
output-padding canaries are checked. Nonpaged coverage separately checks
pre-folded scale, sink/window semantics and changed-input capture.

Runtime: SM107208SM, driver630.10, cuDNN9.30.0.2, Torch2.14.1+cu130,
CuTeDSL4.8.0/CUDA13.4, matching FE1.33 bindings. Runtime key2916acaa3402817b.
The initial28-case suite passes with no failures/skips, alongside39 existing
2x2 structural checks and16 nonpaged D512 dense/THD/sink controls. Additional
shared-record MLA checks and final sanitizer results are recorded below.

Initial qualification caught a too-narrow descriptor guard checking only
the P-ring start; it now agrees with the config's full-ring bound. Two harness
errors were corrected without loosening numerical/byte comparators: record
and comparison directories were initially set together, and the D576 metadata
probe initially omitted the required operand descriptors. Original failures
remain in the private qualification report. No backward or other-GPU claim.


Final qualification passes31 cases, including three actual absorbed-MLA
shared-record checks (V aliases the first512 columns of each576-wide K row),
G96/G128 and ordinary Q1/Q2/Q16 candidates. Compute Sanitizer13.4.92 reports
zero diagnostics:7 page-boundary memcheck cases,3 shared-record memcheck cases,
8 paging synccheck cases and8 paging racecheck cases. Run from `test/python`:
`python -m pytest sdpa/frost/test_sdpa_fwd_native_paged_sm107.py::TestNativeD576Half -q -x`.
Sanitizers select `test_boundaries`, `test_shared_mla_record` or `test_paged`
from that same class with `--error-exitcode 99`. Pre-commit passes; Fern has
zero errors and its existing unauthenticated-redirect warning.

### D512 per-tensor FP8 backward — 2026-11-10

`sdpa/frost/test_sdpa_bwd_d512_fp8.py` covers graph numerical references and
raw-byte replay/batch invariance for E4M3, FP16 and BF16 gradients. Dense
BSHD E4M3 inputs use independent Q/K/V/O/dO tensor scales, left windows,
GQA and tail tiles. Three golden cases per runtime preserve the quantized
inputs while running alone, reordered, duplicated and in batches through129.
All gradients and four Amax outputs are compared; changed-input capture and
sync-debug execution run on the same plans. Nine numerical cases and three
recording cases pass on SM100(148SM, cuDNN9.28.0.38, Torch2.14.1+cu130,
DSL4.8.0/CUDA12.9) and SM107(208SM, cuDNN9.30.0.2, same Torch,
DSL4.8.0/CUDA13.4). The 19-case strict suite passes on each runtime,
including all six committed goldens. SM100 additionally passes ten existing
D192 controls with the shared delta/prepared host. On both SM100 and SM107,
all nine numerical cases pass memcheck, synccheck and racecheck with zero
errors or hazards. SM103 compiles all three gradient types but is unverified
at execution. The existing SM107 CI entry exposes the same shared checks as
`test_sdpa_fp8_sm107.py::TestD512Fp8Backward`; the new D512 entry runs on
SM100/SM103 only, so directory sweeps do not duplicate the Rubin cases. Original16k-SWA geometry has a full structured oracle
on SM107; it is not a full random-matrix oracle or a performance result.

Review closeout: the SM107 D512 FP8 class moved into the actual selected
`test_sdpa_fp8_sm107.py` entry; direct collection of the former backward file
was not CI inclusion. The new kernel now carries its required cuDNN prefix.

### D512 MXFP8 backward — 2026-11-10

`TestD512Mxfp8Backward` is collected by the SM100 MXFP8 entry and the
explicitly selected SM107 `test_sdpa_fp8_sm107.py` entry. The shared `d512_mxfp8_bwd_checks.py` has six independent
reference cases, seven graph declines and two raw-byte golden/batch cases.
CPU-seeded inputs are quantized once; SF bytes are reordered in their canonical
rowwise/columnwise layout with the corresponding payloads. Singletons,
reordered/duplicated batches through129, fresh buffers, concurrent work,
repeat launches, sync-debug execution and captured dO/SF mutations are checked.
Neither the MXFP8 reference nor its cosine tolerance is changed.

Both SM100 and SM107 pass all15 recording cases. Two goldens per runtime cover
FP16/BF16 gradients under the same runtime keys as the D512 per-tensor FP8
qualification above. SM103 execution is unverified. ExactD512 MHA/GQA/SWA uses
only FP32 row-delta scratch and reads canonical scales without repacking.

Final strict checks:15 pass on SM107;39 pass on SM100 comprising15 MXFP8,
19 per-tensor D512 FP8 (all prior golden bytes unchanged) and5 existing
D128/D192 MXFP8 controls. On each architecture, all6 new numerical cases
pass memcheck, synccheck and racecheck without errors or hazards. SM103
compiles both gradient types; no execution result is inferred. Original
B2/H64:1/S16384/window128 passes a full structured output oracle and changed-
input capture on SM107 with8,388,848bytes of scratch. This is not a full
random-matrix oracle or a performance result.

Review closeout on SM107: the selected `test_sdpa_fp8_sm107.py` entry passes
all 15 MXFP8 and 19 FP8 strict cases, plus the three kernel-prefix guards
(37 total). Both new backward kernels have cuDNN symbol prefixes. Arithmetic
and all existing golden output bytes remain unchanged.

### SM107 D64 mixed FP8 K/V — 2026-11-10

`test_sdpa_fp8_sm107.py::TestMixedFp8Kv` now exposes the exact-D64 mixed
storage checks in the selected architecture entry. FP16/BF16 Q/O use E4M3/E5M2
K/V directly. The native physical-64 prefill specialization serves dense and
paged KV, CGA1/2, dense splits and unsplit THD; mixed THD splits decline.

On SM107 with cuDNN 9.30.0.2, Torch 2.14.1+cu130 and DSL 4.8.0/CUDA 13.4,
90 strict feature/resource/control cases pass, including eight new dense/THD
runtime goldens and 16 existing D128 mixed-paged controls. Eight additional
Q=1 dense/paged graph cases pass for all four Q/KV dtype combinations. Memcheck,
synccheck and racecheck each pass 13 selected cases with zero errors/hazards.
The public architecture entry collects all new cases. Other architectures keep
their existing mixed-storage support; no additional GPU execution is claimed.

CPU-fixed, once-quantized payloads are retained across alone/reordered/duplicated
batches through 129 requests. Each case checks all O/Stats bytes, fresh-buffer
replays with queued concurrent work, input preservation and an independent
CPU-f64 numerical reference. THD changes packed origins and checks untouched
capacity tails. Graph tests separately check independent physical offsets,
poisoned gaps, independent HND/NHD tables and changed-input capture. Dense
standalone tests also exercise the native split path. The initial admission
fixture retained dense Stats while synthesizing THD facts and therefore hit an
earlier padded-Stats decline; correcting that fixture required no kernel or
numerical-tolerance change. All eight goldens pass strict comparison.

The shared standalone driver now forwards and asserts its requested split count.
All eight D64 standalone split cases pass with the actual split3 plan; the earlier
run had exercised unsplit standalone execution despite its parameter labels.
Dense graph splits and the eight standalone replay goldens were already passed
with their requested split counts and remain unchanged.


CI-route correction: the pinned SM107 pipeline selects `test_sdpa_fp8_sm107.py`,
not `test_sdpa_fwd_dsl_sm107.py`. The shared mixed-KV class now lives in the
selected file, with explicit L0/DSL/SM107 guards and no duplicate collector.
The GPU counts above are direct qualification, not completed CI results.

### SM80 native THD sink forward

The native adapter passes sink pointers and head strides through the existing
THD host for dense and paged K/V. `test_sdpa_fwd_paged_sm80.py` covers FP16/BF16,
strided sink declarations with flat graph carriers, HND/NHD independent pools,
new storage, dynamic lengths/origins and changed-sink CUDA Graph replay. Invalid
sink bindings are checked before output or metadata writes.

The determinism cases use CPU-generated fixed operands and an independent CPU
FP64 sink reference, repeat under concurrent scheduling work, compare target
sequences alone, with neighbors, reordered and duplicated, and mutate neighbor
pages under capture. On the A100 qualification environment (exact runtime
identifiers retained in private evidence),
the original96cases pass:64existing controls,24finite-sink cases covering
D64/D256/D576x512 and eight strict dense/paged runtime goldens. An additional12endpoint cases cover disabled(-inf) and dominating(+inf) sinks
with live Q but zero KV iterations, graph/standalone and captured replay. The
new regression fails before the kernel fix; all108cases pass afterward, with
unchanged finite-input golden files. The four initial paged and eight expanded
dense/paged golden-recording cases passed before the original strict run. These are SM80 execution results only; no other GPU is claimed.

The initial finite-sink qualification passed eight D64 cases per sanitizer.
After the endpoint fix, memcheck, racecheck and synccheck each pass four
endpoint cases across D64/D256/D576 and graph/standalone with zero findings. These runs use
driver-compatible task-local CUDA bindings. The initial environment
reported API-version query errors before attention; an
exact-source create/destroy-handle control reproduced them. That initial run
is not counted as a sanitizer pass. No suppression is used in the clean runs.

Commands from `test/python`: `python -m pytest sdpa/frost/test_sdpa_fwd_paged_sm80.py -m L0 -q -x`
with `CUDNN_DETERMINISM_GOLDENS` pointing to the committed golden directory.
The endpoint sanitizer selection is `sink_endpoints and dtype0 and (False or dims0)`
on the same file, with `--error-exitcode 91`. Pre-commit passes; Fern has zero errors and
one existing redirect warning.

### SM120 small-head paged forward — 2026-11-10

`sdpa/frost/test_sdpa_fwd_paged_sm120.py::test_sm120_frost_paged_width_determinism`
records eight FP16/BF16 goldens for D64/V64 and D120/V88, packed and unpacked.
CPU-seeded operands and a separate CPU-double oracle cover O and Stats, repeated
raw bytes with queued work, reordered/duplicated batches through129 requests,
and changed-neighbor CUDA Graph replay. Padded head columns and pool rows are
poisoned; output gaps and strided Stats retain their canaries.

Runtime key42742425a3d29d7c identifies the RTX5080(SM120,84SM)
qualification environment; exact toolchain and driver details are retained
in private qualification evidence.
The full paging file passes143 cases with strict goldens; the pre-existing
40GiB physical-wide-pool case is skipped on this16GiB GPU. This does not
qualify wide-address execution, SM121 or RTXPRO6000 performance.

Twelve cases spanning D24/V40,D64/V64,D120/V128, both half types, Q1,
packed GQA, sinks and windows pass each of memcheck, racecheck and synccheck
with zero findings. These runs use driver-compatible task-local CUDA bindings.
The original environment and a handle-only control report the same14
API-query errors; their exact compatibility diagnosis is retained privately.
Those initial runs are not counted as sanitizer passes.
Run from `test/python`: `python -m pytest
sdpa/frost/test_sdpa_fwd_paged_sm120.py -m L0 -q -x`; the sanitizer selection
is `-k 'widths and (dims6 or dims2 or dims10) and (decode or packed_sink)'`.

### SM120 paged decode and split-KV — 2026-11-10

Eight new records at key42742425a3d29d7c cover FP16/BF16 D64/V64 on M16
and D120/V88 on M32, packed, split1/3. The preceding small-head paging runtime
applies. CPU-seeded inputs, a CPU-double O/Stats oracle, raw-byte repeats with
concurrent work, reordered/duplicated batches through129 requests and
changed-neighbor capture all pass before recording. All eight compare strictly
alongside the eight original prefill records; the existing records are unchanged.
The split count and tile shape remain fixed for each batch-invariance check.

Feature coverage includes direct K/V pages, all16/32/48/64-key stage widths,
D8 through128, independent HND/NHD layouts, poison/canaries, GQA6 partial
packing, sink endpoints, device Q/KV lengths and strided natural/log2 Stats.
Twenty existing nonpaged numerical/capture controls pass, including the newly
admitted48-key dense stage for unequal QK/V widths. Target execution is SM120;
SM121 remains unverified. Sanitizer and multi-wrap completion are recorded below.

Final qualification:54 feature cases,17 strict original/new golden and scope
checks,20 nonpaged numerical/capture controls and7 ring/domain checks pass.
Each sanitizer reports zero findings on8 compact cases plus2 cases whose
per-split stage count exceeds the physical ring size. The latter force slot
reuse for both M16 and M32 consumer groups; four larger numerical cases also
change page contents/lengths under capture after several ring wraps.
Original GPT-OSS Q1/KV131072/D64 full and SWA128 graphs execute through ordinary
OPENSOURCE candidates (M16/N32, split8 and split1 respectively), without a
candidate or ranking patch. The bounded timing run is not performance qualification.

### Native SM120 wide paged decode — 2026-11-10

`test_sm120_frost_paged_wide_decode_determinism` adds eight records for
D256/V256 on M16 and D192/V128 on M32, both half dtypes and split1/3.
The existing independent CPU-double oracle, CPU-seeded inputs, eager/captured
byte checks, queued work and reordered/duplicated batches through129 all pass
before recording. Runtime key42742425a3d29d7c identifies the same RTX5080
qualification environment as the small-head records; exact runtime details
remain in private evidence. Existing goldens are unchanged.

Fifty-six feature cases cover independent eight-aligned QK/V widths through256,
including D136/V8 and D8/V192, plus asymmetric D248/V256 and D256/V248 tails.
Two compact multi-wrap cases verify that every producer reuses ring slots;
one host admission detector retains the prefill128 and decode256 boundaries.
The wide tests explicitly request legal native plans because ordinary
candidate generation does not propose decode tiles for every longer Q.
This qualifies native capability, not that candidate ranking or performance.
SM121 remains unexecuted. Memcheck, racecheck and synccheck each pass10cases
with zero findings, including the two multi-wrap cases. Reproduce with
`-k 'wide_decode_ring_pipeline or (wide_decode and not determinism and
(dims0 or dims6) and dtype0)'` on `test_sdpa_fwd_paged_sm120.py`.

### Native SM107 D256 mixed FP8 KV — 2026-11-10

`sdpa/frost/test_sdpa_fp8_sm107.py::TestMixedFp8Kv` exposes D256 checks
through the explicitly selected architecture CI entry. FP16/BF16 Q/O with
E4M3/E5M2 KV use the native two-CTA D256 prefill pipeline, dense Q and no
packing. Dense/paged KV, Q=1 and split1/3/4 are covered. Four committed
goldens preserve CPU-fixed quantized inputs across fresh-buffer replay,
queued work, reordered/duplicated batches through129, independent HND/NHD
pools, zero-length sequences, sinks, bottom-right/window masks and changed
capture replay. Independent CPU-double O/Stats references and input/output
padding canaries accompany raw-byte comparisons.

SM107208SM, driver630.10, cuDNN9.30.0.2, Torch2.14.1+cu130,
CuTeDSL4.8.0/CUDA13.4, runtime key2916acaa3402817b:44 feature/resource/golden
checks pass strictly, with no failures/skips; six existing uniform-half D256
gate/THD controls also pass. Compute Sanitizer13.4.92 memcheck, racecheck and
synccheck each pass eight dense/paged split/unsplit cases with zero diagnostics.
Run from `test/python`: `python -m pytest
sdpa/frost/test_sdpa_fp8_sm107.py::TestMixedFp8Kv -m L0 -q -x
-k 'd256 or native_resources'`. Sanitizer runs select
`test_d256_dense_paged and (qdtype0 or qdtype3)` with `--error-exitcode 91`.
Two additional host admission checks cover unsupported combinations.

Qualification found a paged split ABI mismatch: the adapter excluded paging
from its FP32-partial predicate while the native host wrote FP32. The fix
retains the native partial format for both layouts; no numerical comparator
or tolerance changed. Earlier failing probes remain in the private report.
Forward only; other GPU mixed-D256 routes are not admitted.


### Native SM107 D512 mixed FP8 KV — 2026-11-10

The selected `test_sdpa_fp8_sm107.py::TestMixedFp8Kv` entry exposes exact
D512/V512 mixed-storage checks. FP16/BF16 Q/O and E4M3/E5M2 K/V use one
native CTA pair, with dense/paged KV, packed/unpacked groups and splits.
Four fixed-input goldens cover both Q and KV types, split1/4 and packing;
raw O/Stats bytes are compared under fresh-buffer replay, queued work,
reordered/duplicated batches through129 and changed-input graph replay.
Independent CPU-double references, empty rows, bottom-right/window/sink
semantics, independent HND/NHD pools and capacity canaries are retained.
Q1/Q16 at H128:1 additionally checks partial packing of a full MQA group.

SM107216SM, driver630.10, cuDNN9.30.0.2, Torch2.14.1+cu130,
CuTeDSL4.8.0/CUDA13.4, runtime key0972294c7bd4a660:46 feature/admission/golden
recording cases pass, with no failures/skips. The initial probe stopped at a
configuration guard because the generic record retained d_flavor128; the
loader now normalizes exact mixed D512 before selecting its native2x2 body.
No numerical comparator changed. Strict comparison, original uniform-half
controls and sanitizer results are recorded below after completion.

Run from `test/python`: `python -m pytest
sdpa/frost/test_sdpa_fp8_sm107.py::TestMixedFp8Kv -m L0 -q -x -k d512`.
Forward only; mixed D512 admission on other GPUs is unchanged.


Final qualification repeats46 strict cases against all four recorded goldens.
Twenty existing uniform-half D512/D576 paging and shared-MLA controls pass.
The structural suite passes40 cases; its one SM100-only config case is skipped
by that file's architecture marker and checked separately as host-only assertions.
Compute Sanitizer13.4.92 memcheck, racecheck and synccheck each pass eight
representative dense/paged split/unsplit cases with zero errors/hazards.
Sanitizer selection is `test_d512_dense_paged and (qdtype0 or qdtype3)` with
`--error-exitcode 91`. No SM100 kernel execution or mixed-D576 support is claimed.


### Native SM80 split-KV forward — 2026-11-10

`sdpa/frost/test_sdpa_fwd_split_sm80_determinism.py` qualifies FP16/BF16
D128 with split 3 and D512 with split 17 on A100. Four new fixed-input goldens
compare O and FP32 Stats byte for byte under fresh-buffer replay with concurrent
work, reordered/duplicated batches through 129, empty neighbors and
changed-neighbor CUDA Graph replay. Independent CPU-double references and
input-preservation checks pass before recording. Split count is fixed within
each comparison; equality across split counts is not claimed.

Runtime key `a29ca37b88513cb6` identifies the A100 qualification environment;
complete runtime details are retained in private evidence. Four recording
cases pass, followed by 112 strict cases (the 108 existing SM80 paging/sink
cases plus the four new goldens). The final source then passes 36 numerical
split cases plus the four strict goldens, with no failures or skips. These
cover dense/paged KV, both half types, D64 through D576/V512, graph and
standalone entry points, strided outputs, sink endpoints, padding, optional
Stats, both Stats log bases, poisoned partials and changed bindings. Two
additional split cases reload compiled artifacts with compilation forbidden.
No existing golden was regenerated.

Compute Sanitizer memcheck, racecheck and synccheck each pass eight cases
with zero errors/hazards: paged D64/D256/D576 in graph and standalone modes,
and dense D512 with absent or base-two Stats. The selection is
`-k '(paged_split_rebind and dtype0 and (dims0 or dims3 or dims5)) or
(dense_split_partial_initialization and dtype1 and dims1)'`, with
`--error-exitcode 91`. The unchanged common combine separately passes 12
infinite-sink endpoint controls on SM100; its production source is unchanged.

Run from `test/python`: `CUBLAS_WORKSPACE_CONFIG=:4096:8
CUDNN_DETERMINISM_GOLDENS=$PWD/determinism_goldens python -m pytest
sdpa/frost/test_sdpa_fwd_paged_sm80.py
sdpa/frost/test_sdpa_fwd_split_sm80.py
sdpa/frost/test_sdpa_fwd_split_sm80_determinism.py -m L0 -n 0 -q`.
Forward only; THD split and packing remain unsupported. This does not
qualify another GPU, default plan ranking, or serving performance.


### Native SM80 small query/KV tiles — 2026-11-10

`sdpa/frost/test_sdpa_fwd_small_tiles_sm80.py` exercises explicit M16/M32
query tiles and N16/N32 KV stages without changing default candidate policy.
Eight new runtime goldens cover D64 M16/N16, D128 M16/N32 and M32/N16,
and D256 M16/N16 in both half types, all at split 3. The reused split
qualification preserves CPU-fixed inputs through fresh-buffer replay with
concurrent work, reordered/duplicated batches through 129 and changed-neighbor
CUDA Graph replay. O and FP32 Stats pass independent CPU-double references
before their raw bytes are recorded. No old golden is regenerated.

A100, runtime key `a29ca37b88513cb6`: 64 numerical feature cases and eight
recording cases pass. Features include dense/THD queries, independent paged
K/V, dense-Q splitting, eight-aligned head envelopes, sinks including infinity
endpoints, padded/empty rows, strided outputs, pointer rebinding and capture.
The host geometry suite passes 96 cases while leaving default candidate
choices implicit. These are forward-only results on this GPU/runtime.

The new D192/M128/N16 case exposed a copy-wave divisibility restriction.
The SM80 loader now chooses a whole subset of CTA threads for each operand,
so dense and paged vector copies still fill complete tiles. Existing tile
geometries retain all their copy threads. The first 64-case attempt stopped
after five passes at this compile-time rejection. A subsequent DSL-control-flow
error was resolved by keeping the static configuration diagnostic outside
the dynamic lane guard; both failed attempts remain in private evidence.
The final 64 cases pass after that repair.

Run from `test/python`: `python -m pytest
sdpa/frost/test_sdpa_fwd_small_tiles_sm80.py -m L0 -n 0 -q` with
`CUDNN_DETERMINISM_GOLDENS=$PWD/determinism_goldens` for strict comparison.
The standard golden workflow remains required on a different runtime.


The strict follow-up passes 120 cases: 108 existing paging/sink cases,
four original split goldens and eight new small-tile goldens. Memcheck,
racecheck and synccheck each pass 12 cases with zero errors/hazards,
covering standalone/graph and dense/THD query forms over paged KV at
D48/V32 M16/N16, D192/V128 M128/N16 and D256 M16/N16.
Sanitizer selection is `-k 'paged_rebind and dtype0 and
(48-32-16-16 or 192-128-128-16 or 256-256-16-16)'` with
`--error-exitcode 91`. Runtime details and initial failures remain in
private evidence; no unsupported GPU or default performance claim is made.

Two additional dense D192/V128 M16/N16 and M128/N16 graph cases pass
numerical, fresh-binding and replay checks. Each also passes memcheck,
racecheck and synccheck with zero diagnostics, bringing the sanitizer scope
to 14 cases per tool. These controls exercise the nonpaged copy path with
all CTA threads and with the reduced copy-thread subset.


### Native SM80 PackGQA forward — 2026-11-10

`sdpa/frost/test_sdpa_fwd_pack_gqa_sm80.py` qualifies native FP16/BF16
packed-head forward on A100. Ten new CPU-seeded goldens cover D64/group3
unsplit M16/N16, D128/group12 split3 M32/N32, D128/group128 split3 M16/N16,
D256/group8 unsplit M16/N32 and D512/group3 split17 M32/N16. These include
head groups crossing tile boundaries. Independent CPU FP64 references, fresh
buffers and concurrent-work replay pass before recording. O and FP32 Stats
are byte-compared alone, in reordered/duplicated batches through129 and under
changed-neighbor CUDA Graph replay. Runtime key `a29ca37b88513cb6` identifies
the qualification environment; exact runtime details remain in private evidence.
No old golden was regenerated. This is forward-only, with fixed knobs and
runtime; THD PackGQA remains unsupported.

The native feature run passes64paged graph/standalone cases across both half
types, split/unsplit, independent HND/NHD pools, masks, sink endpoints,
strided canaries, fresh pointers and changed device lengths under capture.
The route assertion observes the packed head count in the compiled template.
Eight dense split partial-initialization cases and six dense bias/sink replay
cases pass. The first launch attempt lacked the isolated worktree's compiled
Python extension and stopped during import; no kernel executed in that attempt.

From `test/python`, run `python -m pytest -n 0 -m L0 -q
sdpa/frost/test_sdpa_fwd_pack_gqa_sm80.py`, with
`CUDNN_DETERMINISM_GOLDENS=$PWD/determinism_goldens` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` for strict comparisons. Other runtimes
require the usual separate golden-recording workflow.


The strict follow-up passes132cases:108existing paging/sink cases,
four original split goldens, eight small-tile goldens, ten new PackGQA goldens,
and two packed split artifact-reload cases with compilation forbidden on the
second plan. All108host geometry/admission cases pass, plus the direct THD
PackGQA rejection. Memcheck passes18cases with zero errors. The initial unfiltered racecheck was interrupted after five passes because
instrumenting the independent Torch reference dominated runtime; its log is
retained. Final racecheck and synccheck each pass22cases with zero errors or
warnings, retaining all18original cases and adding four G128 single-query
cases. These runs use `--kernel-name kns=cudnn_kernel_` to check every native
producer/combine and exclude reference kernels. A full launched-name audit
confirms the producer and combine are included; the first audit hit the
100-message printing limit, so the completed audit uses `--print-limit 0`.
The standalone audit passes one additional graph/capture case. No sanitizer
finding was suppressed or converted into a pass.

### SM90 D512 dense split and partial PackGQA forward — 2026-11-10

`sdpa/frost/test_sdpa_fwd_split_sm90_determinism.py` adds eight L0
forward-only cases: FP16/BF16, split3 unpacked and split17 full PackGQA,
plus split3 partial groups G12/G128. CPU-seeded inputs retain an independent
FP64 numerical reference. Fixed-plan checks cover fresh-buffer replay,
concurrent work, B129 reorder/duplicates/empty neighbors and changed-neighbor
CUDA Graph replay. Eight H100 records are appended under runtime key
`6521230c64ce3d35`; existing goldens are unchanged. No equality across split
counts, packing choices or runtimes is claimed.

The final combined run covers 449 cases: all 57 split/partial-packing
features, eight strict goldens, existing D512 paths and the newly landed D64
route. Initially 434 pass; one old error-message expectation and 14 child
compilation probes fail. The probes imported an unrelated editable install;
they now remove that finder, bind the parent's package directory and assert
the source path. Twelve D64 cross-compilation probes pass on A100 (compiler
checks only), then two D512 probes plus twelve template-validation cases pass
on H100. Thus all 15 initial failures are addressed without repeating the
unchanged native numerical suite. Eighteen existing singleton-stride tests
also exposed a recorder/API/spec reference cycle; retaining only frame
metadata removes all eighteen shutdown leak diagnostics, with eighteen
focused passes before and after the test-only correction.

On H100, unfiltered memcheck and native-producer/combine-filtered racecheck
and synccheck each pass 21 cases with zero diagnostics. The selection is
`(dtype0 and (multiple_kv or (dense_split_partial and dims0) or
partial_pack_multiple)) or partial_pack_graph`, with `--error-exitcode 91`.
The latter tools use `--kernel-name kns=cudnn_kernel_`; independent framework
reference kernels are excluded, while the attention producer and shared
combine are included. An earlier racecheck failed on both the original
unsplit parent and split prototype. Direct per-thread arrivals with matching
counts now order the shared alpha/inverse-factor handoff; intermediate
named-barrier experiments were discarded. Numerical success alone was not
used to waive the initial sanitizer diagnostics. This is fixed-GPU/runtime
forward qualification, not backward or default-dispatch performance.


### SM120 D512 paged forward — 2026-11-10

`sdpa/frost/test_sdpa_fwd_paged_sm120.py::test_sm120_frost_paged_d512_determinism`
adds eight L0 forward cases: FP16/BF16, unpacked/PackGQA and split1/3 at
D512/V512, M64/N32. CPU-seeded inputs and an independent FP64 O/Stats reference
precede byte checks and golden recording. Tests include fresh storage,
concurrent work, B129 reorder/duplicates/empty neighbors, changed-neighbor
CUDA Graph replay, input immutability and output canaries. Split and packing
choices stay fixed within each comparison; cross-configuration identity is
not claimed.

RTX5080 (84 SMs), CUDA driver13.5, Torch2.14.1+cu130, cuDNN9.30.0.2,
CuTe DSL4.8.0 and tvm-ffi0.1.14.post1 use runtime key `1858cfcbf6876f83`.
Eight new records are appended without replacing existing baselines.
Eight existing D64/D120-V88 paging cases also pass and gain records for this
runtime: their input and output hashes exactly match the older driver13.4
records under `42742425a3d29d7c`. This observed equality does not establish
a cross-runtime guarantee. SM121 and backward remain unverified by these runs.

From `test/python`, run `python -m pytest -n 0 -m L0 -q
sdpa/frost/test_sdpa_fwd_paged_sm120.py -k "d512_determinism or paged_width_determinism"`
with `CUDNN_DETERMINISM_GOLDENS=$PWD/determinism_goldens` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. Recording used separate new directories;
normal comparison leaves `CUDNN_DETERMINISM_RECORD` unset.

The complete paging regression passes 144 cases, including all sixteen strict
goldens on this runtime. Final D512 qualification passes 49 cases: the 25 new
feature/admission/golden cases, existing dense/THD controls and two new dense
infinite-sink endpoint checks. After the last paged-only score-buffer barrier
fix, all 25 new cases pass again (279 deselected, no failures or skips).
Ten BF16 feature/endpoint cases pass each of Compute Sanitizer 13.4 memcheck,
racecheck and synccheck with zero reported errors; racecheck also reports zero
warnings. Memcheck is unfiltered; racecheck/synccheck select the native
`cudnn_kernel_` kernels. These are forward-path checks, not backward coverage.

Retained controls distinguish numerical success from synchronization safety.
The redundant paged V-tail stores passed numerics but produced 288 racecheck
errors; skipping those stores on paged input makes the identical case clean.
A broader ten-case run then passed numerically while reporting 80 grouped
score-exchange races. An explicit warp-pair read-completion barrier fixes the
single-buffer reuse: its failing split/packed control and the full ten-case
racecheck now report zero errors. The dense TMA path retains its prior handoff.

### SM120 half Q/O with paged FP8 K/V — 2026-11-10

`sdpa/frost/test_sdpa_fwd_paged_sm120.py::test_sm120_frost_paged_mixed_fp8_determinism`
adds eight L0 forward goldens on RTX5080 (84 SMs): D64 prefill and D128/D256
paged decode plus D512, each with FP16/E4M3 and BF16/E5M2 Q/KV types. Packing
is fixed; split is one for D64 and three for the other shapes. Raw quantized
cache payloads stay fixed across fresh buffers, eager/captured replay,
reordered/duplicated batches up to B129, empty neighbors and changed neighbor
contents. Independent FP64 O/Stats references and input/output canaries are
checked before byte comparisons. This is plain `sdpa` cache storage, with no
scalar descales or quantized Q/O contract.

Torch2.14.1+cu130, cuDNN9.30.0.2, CuTeDSL4.8.0 (CUDA12.9 compiler),
CUDA driver13.5 and tvm-ffi0.1.14.post1 use runtime key `1858cfcbf6876f83`.
Eight records were appended from a new directory after successful recording;
the separate strict run passes all eight (331 deselected, no failures/skips).
No old golden was regenerated. The 27 feature/conversion/candidate checks
also pass: both half types and both FP8 formats, independent HND/NHD pools,
head envelopes, masked dead page/head tails, pointer/length rebinding, packed
splits, persistent reuse, normal graph candidates and all 256 FP8 encodings.
Finite conversion results, including signed zeros, match their references
byte-for-byte. Direct packed FP8-to-BF16 conversion failed with the CUDA12.9
compiler; the global loader now widens through FP16 exactly, then converts to
BF16. Existing shared-memory FP8 conversion code generation is preserved.

Compute Sanitizer13.4 runs ten E5M2 FP16/BF16 feature cases each under
memcheck, racecheck and synccheck: all pass with zero errors (racecheck zero
warnings). Memcheck is unfiltered; racecheck/synccheck select native
`cudnn_kernel_` kernels. Full host graph analyzer:628 passed. Earlier failures
were retained: one incorrect test page-size argument; one BF16 assembly
failure after a FP16 pass; three longer-Q cases blocked because the explicit
plan helper first required a heuristics proposal. The helper now reaches the
explicit plan directly; three separate ordinary-candidate cases remain active.

From `test/python`, run `python -m pytest -n0 -m L0 -q
sdpa/frost/test_sdpa_fwd_paged_sm120.py -k mixed` with
`CUDNN_DETERMINISM_GOLDENS=$PWD/determinism_goldens` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`; leave recording unset. The sanitizer
selection is `-k "mixed_fp8 and not determinism and kv_dtype1"`.
SM121, other runtimes and backward remain unverified; mixed dense KV and
THD queries remain unsupported on this new SM120 path. Cross-split or
cross-packing byte identity is not claimed.

Final shared-loader regression:36 passed,303 deselected, no failures/skips in
168.73s, including24 strict goldens (16 existing half plus8 mixed), native half
rebinding/replay and decode ring reuse. The old half baselines are unchanged.
The owned SM120 allocation completed normally after these checks.

## SM120 THD D256/GQA backward qualification

`test_sdpa_bwd_thd_sm120_gqa.py` compares dQ/dK/dV with independent CPU FP64
autograd, including per-port origins, GQA, bottom-right quiet rows and empty
sequences. Its L0 single-KV-tile byte checks preserve fixed inputs/upstream
gradients across fresh storage, concurrent work, B1/B3/B129, duplicates,
reordering and changed-neighbor capture. On GB10/SM121 (driver 580.142,
Torch 2.9.0a0+50eac811a6.nv25.09/cu130, cuDNN 9.13.0, DSL 4.8.0/compiler 12.9),
both FP16/BF16 cases pass recording and fresh-process strict checks, using
runtime key `d2d1094fc872ddba`. All three gradients match their independent
reference. SM120 remains unverified.
Final combined execution passes 73 cases (40 new and 33 existing D128),
including the two strict goldens; 123 host checks pass. Five cases each pass
memcheck, racecheck and synccheck using CUDA 13.4 sanitizer 2026.3.0 with
API-query reporting disabled (CUDA Python 13.2 probes newer APIs than the
CUDA 13.0 driver). Racecheck reports zero errors and zero warnings.
General deterministic backward remains unsupported because multiple KV tiles
accumulate dQ with unordered FP32 atomics; admission tests retain that decline.

SM120-family wide half backward (`sdpa-sm120-wide-bwd-d512-gqa-swa-*`):
D512, GQA 6:2, Q17/KV19 crossing two native KV tiles, bottom-right SWA,
FP16/BF16. The deterministic relay is exercised under fresh allocations,
repeated calls, CUDA Graph replay, changed neighbors, duplicate/reordered
sequences and batches 1/3/129. dQ/dK/dV are checked against independent CPU
FP64 autograd before strict raw-byte comparison. Recorded on SM121 GB10;
this fingerprint does not qualify SM120 hardware or low-precision backward.

### SM120 native per-tensor FP8 backward

`sdpa/frost/test_sdpa_bwd_fp8_sm120.py::test_replay_batch_and_golden` adds six
SM121/GB10 runtime-keyed records for D192/D512 and E4M3/FP16/BF16 gradients.
Each checks dQ/dK/dV and amax_dQ/dK/dV/dP with fixed scalar scales and payloads
across batch reorder/duplication, fresh pointers, changed-dO capture replay,
and repeated calls. The independent FP8 numerical oracle is checked first.
SM120 hardware and cross-runtime bit identity are not covered.


### SM120 native MXFP8 backward

`test_sdpa_bwd_mxfp8_sm120.py::test_replay_batch_and_golden` adds four
SM121/GB10 records for D128/D512 with FP16/BF16 gradients. Input payloads
and all seven canonical SF tensors are reordered together, including the
D-plane-major transposed atoms. The checks cover B1/3/4/129, fresh storage,
repeated calls, changed-payload capture and live SF mutation. The independent
MXFP8 numerical oracle passes before byte comparisons. SM120 hardware and
cross-runtime identity remain unverified.

Qualification on 2026-11-10: GB10 SM121, driver 580.126.20, Torch
2.9.0a0+50eac811a6.nv25.09/cu130, cuDNN 9.13, DSL 4.8/compiler 12.9.
The full new suite passes **55 tests, zero skips**, including the four goldens
in a fresh strict-check process; the existing per-tensor FP8 suite passes
**79 tests**, with all six committed hashes unchanged. The numerical oracle
also independently requires finite dQ/dK/dV and relative L2 error below 0.05.
Eleven representative cases pass each of memcheck, racecheck and synccheck.
From `test/python`, with the isolated checkout verified:

```bash
CUDNN_TEST_NO_ISOLATION=1 CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
python -m pytest -q -n0 sdpa/frost/test_sdpa_bwd_mxfp8_sm120.py
```

This change adds no forward kernel; existing forward coverage is unchanged.

#### Optional MXFP8 gradient amax outputs — 2026-11-10

`test_sdpa_bwd_mxfp8_sm120_amax.py` adds fourteen L0 cases: all eight
subsets of amax_dQ/dK/dV, two standalone binding cases, and four D128/D512
FP16/BF16 batch/replay goldens. Each requested scalar is the global maximum
absolute FP32 gradient accumulator before conversion to the gradient dtype.
The independent MXFP8 numerical reference passes before byte comparisons;
the amax comparison bounds only the final FP16/BF16 rounding.

The four records under runtime key `d2d1094fc872ddba` preserve all parent
MXFP8 input and gradient hashes, adding the three FP32 maxima. Coverage
includes B1/3/4/129, reorder/duplication, fresh pointers, NaN/Inf output
poisoning, workspace poisoning and changed-dO capture replay. Global amax
must compose as the maximum of singleton values; it is not a per-sequence
output invariant to adding arbitrary neighbors. Standalone checks also
require invalid optional bindings to leave outputs and workspace untouched,
followed by successful execution on the same plan.

Recorded on GB10 SM121, driver 580.126.20, Torch
2.9.0a0+50eac811a6.nv25.09/cu130, cuDNN9.13, DSL4.8/compiler12.9 and
tvm-ffi0.1.14.post1. SM120 hardware and cross-runtime bit identity remain
unverified. This extension changes no forward kernel or existing golden.
Fresh-process strict qualification passes all 148 cases: 14 in this extension,
55 existing MXFP8 cases and 79 existing per-tensor FP8 cases, with no skips.
All four new records retain the parent's input and gradient hashes. Host
admission/specialization checks pass 95 cases; 12 native SM100 D128/D192/D512
controls also pass with the shared prepared ABI. The amax-enabled D128 and
D512 kernels cross-compile for SM120; this is not hardware qualification.
Twelve representative cases pass each of memcheck, racecheck and synccheck,
with zero device errors or race warnings. Compute Sanitizer 2026.3.0 instruments
all kernels; API-query reporting is disabled for the recorded CUDA Python
13.2/driver13.0 startup mismatch. The first standalone fixture incorrectly
equated graph `left_bound=9` with `window_size_left=9`; it now uses eight
positions to the left plus the diagonal. No kernel change or golden rerecording
was needed for that fixture correction.

From `test/python`, after verifying the isolated import path:

```bash
CUDNN_TEST_NO_ISOLATION=1 CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
python -m pytest -q -n0 sdpa/frost/test_sdpa_bwd_mxfp8_sm120_amax.py
```

### SM120-family D576/V512 forward — SM121 qualification, 2026-11-10

`sdpa/frost/test_sdpa_fwd_d576_sm120.py` adds eighteen L0 replay/batch cases:
eight uniform-half paged, four mixed-FP8 paged, and six dense/split/THD.
They hold split and packing fixed, preserve CPU-seeded inputs, compare O/Stats
against FP64, and check duplicated/reordered batches up to B129 and changed
neighbors under capture. Compact THD repacks inputs and declared offsets as
a neighbor changes between empty and live; independent physical gaps are not
part of this existing path's contract. Backward is unsupported by this addition.

GB10 (SM121,48SMs), driver580.142/CUDA13.0, Torch2.9.0a0+50eac811a6.nv25.09,
cuDNN9.13.0, CuTeDSL4.8.0/compiler12.9, tvm-ffi0.1.14.post1, matching native
FE extension and verified isolated import: **18 passed,35 deselected** when
recording into a fresh directory (178.53s), followed by **18 passed,35
deselected** in a new strict-check process (5.15s). The18 new JSON baselines
live under runtime key `d2d1094fc872ddba`; all16 hashes recorded before the THD
fixture correction are unchanged. The independent reference and batch checks
run before each baseline is recorded. Command from `test/python`:

```bash
CUDNN_TEST_NO_ISOLATION=1 CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
python -m pytest -q -n0 sdpa/frost/test_sdpa_fwd_d576_sm120.py -k determinism
```

The first recording passed16 and failed2 THD cases because the fixture changed
lengths while leaving later sequences at old physical addresses. That failure
is retained separately from the successful compact replay. An earlier expanded
feature run stopped at4pass/3fail because the native binder rejected asymmetric
widths before launch; after its envelope and extension-gate correction, all
**35 feature cases pass** (275.33s). The full host analyzer passes629checks.
SM120 D576 hardware execution, other runtimes and performance remain unverified.
Targeted sanitizer and existing D512 regression results are recorded separately
when complete; no completed result is implied by this golden qualification.


### SM90 native D128/D256 forward — 2026-11-10

`sdpa/frost/test_sdpa_fwd_native_heads_sm90.py` adds eight L0 forward
bytewise cases at D128 and D256, FP16/BF16, unsplit unpacked and split3
PackGQA. A fixed CPU payload and FP64 O/Stats reference precede recording.
The checks cover fresh storage, concurrent work, B1 versus B129, duplicated
and reordered sequences, changed-neighbor isolation, empty neighbors and
CUDA Graph replay, with padding, a bottom-right window and sinks. Eight new
JSON records under `6521230c64ce3d35` were recorded into a new directory,
then copied and strictly checked with `CUDNN_DETERMINISM_RECORD` unset.

H100 NVL cc9.0, driver595.71.05, Torch2.14.1+cu130, cuDNN9.28.0.38 and
DSL4.8/compiler12.9: all 88 new feature/strict cases pass without skips.
The initial 64 cover numerical/mask/layout/split contracts; the final 24
add the eight strict goldens, partial groups G12/G128, all three schedulers,
independent THD offsets with both packed Stats layouts, and offset replay.
Forty existing controls pass, including all eight unchanged D512 goldens,
D64 prepared execution and standalone D64 feature envelopes.

Memcheck, native-attention/combine racecheck and synccheck each pass the same
22 cases with zero errors or race warnings. The selection is
`thd_origin_replay or thd_physical_origins or partial_pack_replay or
multiple_stages or (split_replay and bf16)`, under Compute Sanitizer13.4.
Memcheck is unfiltered; the other tools use `--kernel-name kns=cudnn_kernel_`.
The first memcheck run passed all22 tests but reported40 startup API-query
errors: CUDA Python13.3 requested newer symbols than the13.2 driver API.
The clean run uses cached CUDA Python13.2 with API-error reporting retained;
no device diagnostic was suppressed. This is H100 forward qualification,
not a backward, H200 performance, cross-runtime or cross-plan byte guarantee.

Commands run from `test/python` with the isolated checkout verified:

```bash
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  pytest -q -n0 sdpa/frost/test_sdpa_fwd_native_heads_sm90.py -m L0
```

The measured execution split this command into the initial64 feature cases
and the final24 cases; together they cover the complete88-case collection.

### Native SM90 paged KV forward — 2026-11-10

`sdpa/frost/test_sdpa_fwd_paged_sm90.py` adds 31 L0 cases for dense-query
FP16/BF16 paging through the native D128/D256/D512 bodies. Coverage includes
page sizes 8/16/32/64/128/192/256, independent K/V pool layouts and table
pitches, fresh pointers, changed-table CUDA Graph replay, poisoned padding,
base-2 Stats, sinks, split3, all three schedulers and partial GQA groups
G12/G128. Standalone head boundaries include D8/V512 and D512/V8; graph
head envelopes retain their existing limits. THD paging and Hopper backward
remain unsupported.

Six new golden records under `6521230c64ce3d35` cover D128, D192/V128 and
D512 in both dtypes. Each compares O and Stats with an independent FP64
reference, then checks raw bytes across fresh storage, concurrent work,
B1/B129, reordered/duplicated sequences, changed neighbors and captured
replay. Recording into a new directory passed all six cases; the subsequent
strict run left recording unset and used the committed files.

H100 NVL cc9.0, driver 595.71.05, Torch 2.14.1+cu130, cuDNN 9.28.0.38 and
DSL 4.8/compiler 12.9: the strict run passed 172 cases, 68 deselected,
zero failures/skips, in 266.05 seconds. These are 23 new paging cases and
149 existing native binding/head controls. The six standalone head-boundary
cases and two partial-PackGQA cases also passed under each sanitizer,
completing all 31 new cases.

Memcheck, racecheck and synccheck each passed 24 selected cases with zero
errors or hazards: eight BF16 page-replay cases, six standalone boundaries,
two partial groups and eight existing D64 graph controls. The runs were
split into 22 cases and the final two partial groups. Memcheck was unfiltered;
racecheck and synccheck used `--kernel-name kns=cudnn_kernel_`. CUDA Python
13.2 matched the driver API and API-error reporting stayed enabled.

The first two control runs exposed a prototype nonpaged host-ABI mismatch:
the common frame binds K table strides as `(0, 0)` and V table strides as
`None`. The final host specialization preserves that distinction. A first
recording command incorrectly combined recording and checking modes and
stopped before writing; the clean recording and strict run above supersede it.
These results qualify H100 forward correctness and per-plan determinism;
H200 execution, other runtimes and performance remain unmeasured.

Reproduction commands from `test/python` (the measured runs split the suite
between strict and sanitizer selections):

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  pytest -q -n0 sdpa/frost/test_sdpa_fwd_paged_sm90.py -m L0

compute-sanitizer --tool memcheck --error-exitcode 91 \
  python -m pytest -q -n0 \
    sdpa/frost/test_sdpa_fwd_paged_sm90.py \
    sdpa/frost/test_sdpa_native_d64_sm90.py \
    -k '(paged_replay and bf16) or standalone_page_head_boundaries or paged_partial_pack_groups or graph_uses_prepared_native_body'
```

### SM107 single-sequence physical-origin setup — 2026-11-10

`sdpa/frost/test_sdpa_single_sequence_setup_sm107.py` adds14 L0 cases for
uniform FP16/BF16 D64/D128, nonpaged and unsplit. Eight physical-origin cases
exercise both CTA group sizes, packing, independent INT32/INT64 element
origins and multipliers, token/head-major Stats, changed token strides,
rebound addresses, retained CUDA Graph replay, keyless rows and empty work.
Four byte tests use fixed CPU payloads and independent CPU FP64 O/Stats
references before recording. They compare fresh storage/concurrent work,
poisoned-workspace replay, B1 against B3/B5/B129, duplicate/reordered/empty
sequences and changed-neighbor isolation. Each byte test now reuses one
capacity-4096 plan through public shape overrides, then returns to B1 after
multi-sequence bindings. Two additional tests use runtime B4 and check the
generic setup branch without forcing a specialization.
Stats negative infinity is permitted only as the keyless-row sentinel; the
reference and canaries still check its location and all other output values.

SM107 / driver620.62 / Torch2.14.1+cu130 / cuDNN9.30.0.2 /
DSL4.8 compiler13.4:14new cases passed, four new runtime goldens under
`aa97de087283f297` were recorded, then all42 strict/control cases passed
(14new+20dead-subtile+8existing D64 physical-origin controls), without skips.
The strict run had recording unset and used the copied golden directory.
The final capacity-4096 qualification repeated all42strict/control cases
(63.83s). Memcheck/racecheck/synccheck each passed all14new cases
with zero errors or hazards. Memcheck was unfiltered; race/sync used
`--kernel-name kns=cudnn_kernel_`; API-error reporting remained enabled.
The generic multi-sequence setup body is AST-identical to the prior source.
This qualifies the setup path, not new arithmetic, other GPUs, split/paged
specializations or cross-runtime/cross-plan byte invariance.

From `test/python`, after asserting the isolated import path:

```bash
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  pytest -q sdpa/frost/test_sdpa_single_sequence_setup_sm107.py \
  sdpa/frost/test_sdpa_thd_dead_subtile_sm107.py \
  sdpa/frost/test_sdpa_fp8_sm107.py::TestNativeD64Half::test_independent_origins
```

The initial collection attempt used an unavailable xdist option and performed
no tests. The first byte-test attempt rejected the legitimate keyless Stats
sentinel under the name `stats`; explicitly declaring its allowed negative
infinity fixed the harness, with no kernel or tolerance change. All14 cases
then completed and were checked again in the42-case strict run.

## Simplified sparse attention — 2026-11-09

The new suite is
`simplified_sparse_attention/cutedsl/test_simplified_sparse_attention_determinism.py`.
It tests the public `cudnn.ops.simplified_sparse_attention_forward` call. There is no
mode that guarantees determinism. All cases are L0 and use BF16 input,
BF16 output/gradients, and FP32 loss. Tests check output, indexer loss,
dQ, dK, dV, dIndexerQ, and dIndexerK. They do not expose or compare saved
internal statistics. There are no gate or recurrent-state gradients.

The initial SM100 run expanded to 16 cases: **9 passed, 7 failed, 0 skipped**.
Forward passed all eight cases. Backward passed one small golden case and
failed the other seven cases. Each failure stopped at dK, so it does not
establish the status of later gradient comparisons. No backward golden is
approved from that run. The three forward baselines passed the independent
FP64 reference and eight fresh-allocation replays before recording. Golden
names include the loaded DSL compiler's CUDA flavor.

| Check | Shapes / variation | Result |
| --- | --- | --- |
| Replay | Packed lengths 65/3/17; Hq16/top-k512, Hq32/top-k1024, Hq96/top-k2048 | Forward passes; backward dK differs by 3, 1, and 1 bytes in 43,520 bytes on replay 1. |
| Scheduling boundary | 297 queries, Hq16, top-k512; 148 SMs | Forward passes; backward dK differs by 13 of 152,064 bytes. |
| Batch changes | Hq16/top-k512; target length65; neighbor length17; insertion, removal, reordering, duplication, changed neighbor | Forward passes; backward stops at first insertion, with 3 of 33,280 dK bytes changed. Later layouts are unmeasured for backward. |
| Golden | Length17; same three Hq/top-k pairs | All forward cases pass. Hq16 backward passes locally; Hq32/Hq96 dK each differ by 1 of 8,704 bytes during replay. |

Inputs and upstream gradients are generated once on CPU, then packed without
new draws. Invalid rows include empty key intervals. The scalar loss uses a
fixed denominator of 257, and batch checks isolate the target's loss and
upstream gradient. Full empty Q/K streams are unsupported. Concurrent GPU
work and fresh allocations vary scheduling without changing the inputs.

Runtime: SM100, 148 SMs, driver 610.11 (CUDA driver API 13.3), PyTorch
2.13.0+cu130, cuDNN 9.27.0, CuTe DSL 4.8.0 with the CUDA 13.4 native compiler,
TVM FFI 0.1.13.post2. The integration starts from Frontend revision
`cb5f9e54bbb1f64b9b3af2992dc5dbe95f831888`. The printed import path was the
integration checkout, verified before testing. SM103 and long filtered-path
determinism are unverified.

Initial measurement, from `test/python`, with the runtime library path set:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDNN_DETERMINISM_RECORD=/tmp/ssa-new-goldens \
  python -m pytest simplified_sparse_attention/cutedsl/test_simplified_sparse_attention_determinism.py -m L0 -q -s
```

Use a new recording directory. Inspect hashes and reference results before
copying baselines. Normal checks leave `CUDNN_DETERMINISM_RECORD` unset:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest simplified_sparse_attention/cutedsl/test_simplified_sparse_attention_determinism.py -m L0 -q
```

The expanded suite has 20 cases. Adding Hq32/top-k1024 and Hq96/top-k2048
batch checks exposed two forward failures: each changes 1 of 4 scalar-loss
bytes when the target moves to packed slot 1. The output bytes match before
the loss comparison fails. Loss accumulation groups physical rows into warps,
so moving a sequence can change FP32 rounding. The expanded strict forward
run was **8 passed, 2 failed, 10 deselected**.

At the requester's direction, retain kernel behavior and scope expected
failures on SM100 to the measured dK byte mismatches and those two packed-loss
cases. Each replay passes its numerical reference before byte comparison. Shape/dtype changes,
non-finite results, other tensors, and other loss mismatches still fail.
Backward golden cases remain expected failures until replay is stable and
a baseline can be approved; an intermittent replay pass is insufficient.
Remove these exceptions with the corresponding reduction fixes.

The final installed-wheel L0 run on the same SM100 runtime reported
**106 passed, 12 xfailed, 4 deselected, 0 failed, 0 skipped**. Of these,
the determinism suite accounts for **8 passed and 12 xfailed**. Nine xfails
stop at a dK byte mismatch, two at the packed scalar loss, and one at the
unapproved backward golden after its small replay passes. Later tensor and
layout comparisons after each mismatch remain unmeasured. Forward status
is **Replay only**; backward is **Not guaranteed**. Expected failures do
not improve those support claims.

A final 20-case rerun with the FP64 reference checked on every replay also
reported **8 passed, 12 xfailed**. Eight cases stopped at dK, two at packed
loss, and two at unapproved backward goldens. Hq32's small backward replay
passed in this run despite failing earlier; no new baseline was recorded.

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest simplified_sparse_attention/cutedsl -m L0 -q -s -rxX
```

The installed wheel contains all 32 Python operation modules and its source
notice. Separate CuTe DSL 4.8.0 CUDA 12.9 compiler testing passed 17 public API
cases, including all nine Hq/top-k forward/backward reference combinations,
selected-support numerics, tails, invalid configurations, and capture rejection.
This used PyTorch/cuDNN built for CUDA 13; it qualifies the DSL compiler
flavor, not a deployment using CUDA 12 throughout. The loaded native libraries
were the cu12 and cu13 compiler extensions under `cutlass/_mlir/_mlir_libs/`
in separate environments. SM103, long-path CUDA 12 compilation, and newer DSL
versions remain unverified. The actual DSL 4.7.1 runtime rejects this API before
kernel import. A missing PyTorch conditional-graph entry also rejects before
filtered-state allocation; bypassing that guard makes its regression test fail.

The installed-wheel L1 run passed **4 cases, 118 deselected, 0 skipped**:
cold exact repair through the public API, changed addresses/bounds/extents,
reserved capacity at its maxima, sampler diagnostics, and a 16,384-query
structured-score stress case. These are numerical and storage-reuse checks,
not bytewise determinism qualification of the long path.

```sh
python -m pytest simplified_sparse_attention/cutedsl -m L1 -q -s
```

Compute Sanitizer `synccheck` passed all three Hq16/top-k512,
Hq32/top-k1024, and Hq96/top-k2048 public forward/backward reference cases:
**3 passed, 15 deselected; 0 sanitizer errors**. This covers those short
packed numerical paths, not the long filtered repair graph.

```sh
compute-sanitizer --tool synccheck --error-exitcode 86 python -m pytest \
  simplified_sparse_attention/cutedsl/test_simplified_sparse_attention.py -m L0 -q -s \
  -k 'public_forward_backward_reference and ((k512 and bf16-h16) or (k1024 and bf16-h32) or (k2048 and bf16-h96))'
```

## Simplified sparse attention JAX — 2026-11-09

The JAX suite is
`simplified_sparse_attention/cutedsl/test_simplified_sparse_attention_determinism_jax.py`.
It exercises the public native-FFI API under `jax.jit` and `jax.vjp`, using
JAX/JAXlib 0.11.0 on the SM100 runtime above. The source base remains
`cb5f9e54bbb1f64b9b3af2992dc5dbe95f831888`, with this integration applied.

| Suite | Forward results | Backward results | Cases |
| --- | --- | --- | --- |
| PyTorch determinism | BF16 output, FP32 indexer loss | BF16 dQ/dK/dV/dIndexerQ/dIndexerK | 20 |
| JAX determinism | BF16 output, FP32 indexer loss | BF16 dQ/dK/dV/dIndexerQ/dIndexerK | 20 |

Both suites check the independent FP64 reference on every replay, then compare
raw bytes with fresh result buffers and concurrent GPU work. The JAX suite
uses the same fixed packed data, ragged tails, insertion/removal/reordering,
duplication, changed neighbors, and 297-query scheduling boundary. Internal
LSE, teacher distributions, and long filtered-path determinism remain unverified.

The JAX recording run reported **8 passed, 12 xfailed, 0 skipped**. Nine
cases stopped at atomic dK, two at the known packed loss, and one at the
unapproved backward golden. The three new forward JSON records passed their
references and replay checks. All input and output hashes match the existing
PyTorch goldens. Their filenames also pin JAX 0.11.0 and compiler CUDA 13.4.
No backward baseline was recorded.

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDNN_DETERMINISM_RECORD=/tmp/ssa-jax-new-goldens \
  python -m pytest simplified_sparse_attention/cutedsl/test_simplified_sparse_attention_determinism_jax.py -m L0 -q -s -rxX
```

A combined installed-wheel run reached three additional byte mismatches after
dK happened to match: PyTorch H96/top-k2048 packed backward dV (1/33280 bytes),
its backward golden replay dV (1/8704 bytes), and JAX H32/top-k1024 replay dV
(2/43520 bytes). Each passed its independent numerical reference first. These
are the same already-excepted backward cases; both dK and dV use FP32 atomic
reductions. The scoped SM100 backward exception now names both gradients.
Other tensors, non-finite values, shapes, dtypes, and reference failures still
fail. The interrupted diagnostic run reported **104 passed, 3 failed,
13 xfailed, 6 deselected**; it is not a qualification pass.

The two JAX L1 numerical cases passed at top-k512 and top-k2048 with 129
queries and 131,329 keys. All 129 tied rows force two exact-repair chunks;
a second call changes indexer inputs and key bounds and includes invalid
rows. The tests check selected scores against full FP32 top-k and compare
output, loss, and all five gradients with an independent FP64 reference
using the selection from that same public call. Tied selections may differ
between calls, so a separate selector execution cannot supply its reference.

Near-uniform KL is about 2.5e-6 in the tied top-k2048 case. A pure 1% relative
loss check exposed about 2.6e-8 absolute error. A shared native attention
probe also showed FP32 teacher-mass rounding (up to 5.3e-7), contributing
about 1.5e-8 of loss error. The new long-path loss check uses 1% relative
error plus an absolute budget of `16 * FP32_epsilon * loss_coeff * valid_rows /
loss_denominator`; output and gradient tolerances remain unchanged. This
budget applies equally to both compiler flavors. No kernel arithmetic changed.

The final combined installed-wheel L0 run reported **135 passed, 24 xfailed,
6 deselected, 0 failed, 0 skipped**. Each framework's determinism suite
contributes 8 passes and 12 expected failures. Across both suites, 16 cases
stop at atomic gradient differences, four at packed scalar loss, and four at
unapproved backward goldens after small replays pass. Later comparisons after
a mismatch remain unmeasured. The 21 JAX public API cases cover all nine
head/top-k forward/backward combinations, selected support, tails, triangular
metadata, zero loss coefficient, input ownership, invalid inputs, captured-input
JIT, rejection of mixed GPU architectures, and the dense workspace index limit.

```sh
CUTE_DSL_CACHE_DIR=/tmp/ssa-cu13-cache CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest simplified_sparse_attention/cutedsl -m L0 -q -s -rxX
```

In a separate installed-wheel environment, CuTe DSL 4.8.0's CUDA 12.9 compiler
passed **23 JAX cases, 0 failed, 0 skipped**: all 21 public cases plus both long
Hq16 filtered-repair cases. The loaded native libraries under `cutlass/_mlir/_mlir_libs/` used the cu12
and cu13 flavors, respectively. Each flavor used a fresh
process and its own initially empty DSL cache. PyTorch, JAX, and cuDNN still
use CUDA 13 packages, so this is compiler compatibility, not a full CUDA 12
deployment. CUDA 12 bytewise determinism and SM103 execution remain unverified.

```sh
CUTE_DSL_CACHE_DIR=/tmp/ssa-cu12-cache python -m pytest \
  simplified_sparse_attention/cutedsl/test_simplified_sparse_attention_jax.py \
  simplified_sparse_attention/cutedsl/test_filtered_jax.py -m 'L0 or L1' -q -s
```

The final wheel includes 34 Python operation modules, its Apache-2.0 notice,
and `contributors.md`. The inventory was compared byte for byte with the
checkout. The final rebuild changes only the JAX docstring's description of
the XLA repair loop relative to the full L0 run; the CUDA 12 run uses that
rebuild. Source copyright notices remain in the released files.

Compute Sanitizer **memcheck and synccheck each passed 1 JAX L1 case,
1 deselected, with 0 reported errors** on the final rebuilt wheel. The case
covers top-k512 cold repair in two chunks, changed inputs/bounds, selected
scores, and all five gradients. This does not qualify every head/top-k route.

```sh
compute-sanitizer --tool memcheck --report-api-errors explicit \
  --suppressions /tmp/cuda-binding-probe-suppression.xml --error-exitcode 86 \
  python -m pytest simplified_sparse_attention/cutedsl/test_filtered_jax.py -m L1 -q -s -k 512
# Repeat with --tool synccheck.
```

The initial memcheck run completed its numerical test and reported only seven
CUDA-binding feature-probe failures, plus seven extended diagnostics: the
bindings request CUDA API version 13041 against driver API 13030. A separate
lazy-import test reproduced the API errors without running these kernels.
The final run uses explicit API-error reporting and suppresses only
`cuGetProcAddress_v2` returning `CUDA_ERROR_INVALID_VALUE` (result 1) from
`cuda/bindings/_internal/driver*.so`. The XML record came from that import
control and retains its binding-module stack constraint. All memory-access
checks and other explicit API errors remain enabled; no device-access or
barrier error was suppressed.

JAX installed-wheel benchmarks (three warmups, 20 timed calls, synchronized
wall time) measured 2.951 ms forward and 3.441 ms training at
Tq1024/Tk4096/H16/top-k512, and 6.914 ms filtered forward at
Tq1024/Tk131072/H32/top-k2048. These shared-GPU samples include host dispatch
and completion waits; they are not direct speed comparisons with the earlier
PyTorch CUDA-event measurements.

### CodeQL follow-up — 2026-11-09

The follow-up to `40d2377` adds explicit lazy-export declarations, rejects
invalid GEMM layouts before allocation, and initializes top-k settings in
the base constructor. Removing unused names preserves the Hq16 kernel's
shared-memory reservation and arithmetic. No golden or expected-failure
rule changed.

The source-checkout L0 run on SM100 reported **148 passed, 24 xfailed,
6 deselected, 0 failed, 0 skipped**. This includes 13 new host cases for
invalid layouts and explicit/automatic top-k settings. The lazy-import test
also checks all four package exports against `cudnn.ops`. Both frameworks'
forward/backward reference, replay, batch-invariance, and golden checks ran
with the existing scoped exceptions. Support classifications remain unchanged;
SM103 and long-path bytewise determinism remain unverified.

Runtime: Python 3.12, PyTorch 2.13.0+cu130, JAX 0.11.0, cuDNN 9.27,
and CuTe DSL 4.8.0 with its CUDA 13.4 compiler. Imports resolved to the
edited checkout. From `test/python`:

```sh
XLA_PYTHON_CLIENT_PREALLOCATE=false CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest simplified_sparse_attention/cutedsl -m L0 -q -s -rxX
```

CodeQL 2.27.4 with `codeql/python-queries@1.9.0` scanned all 34 operation
modules using `python-security-and-quality.qls`. None of the 14 inline
findings on PR #2370 remains. The scan still reports 239 other package
diagnostics; no query or path was excluded by this fix. Pre-commit passed.
Fern reported zero errors and one existing warning.

The subsequent L1 run reported **6 passed, 172 deselected, 0 failed,
0 skipped**. It covers PyTorch repair-state reuse, reservation, sampling,
and diagnostics, plus both JAX long-key repair cases and their gradients.

```sh
XLA_PYTHON_CLIENT_PREALLOCATE=false CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python -m pytest simplified_sparse_attention/cutedsl -m L1 -q -s
```

### Host concurrency and export follow-up — 2026-11-09

The follow-up to `80de1aa` serializes each stream's shared-scratch enqueue
sequence for dense indexing, filtered indexing/reservation, and H16 backward.
Filtered-state retirement also protects its shared host list. No kernel
arithmetic, golden, or expected-failure rule changed.

The six new public-API cases pause a host thread after writing shared scores
or filtered inputs. Another thread then enters with opposite indexer queries.
They cover default, shared side, and separate streams at Tq8/Hq16/top-k512,
with Tk1024 and Tk131072. The independent expected attention outputs are all
ones and all zeros. Concurrent output and scalar-loss bytes must match the
sequential results; separate-stream calls must complete while the first is
paused.

With the original `80de1aa` kernels, these cases reported **4 failed,
2 passed, 31 deselected**: both shared-stream modes corrupted all 32,768
output elements in the first call, on both indexer paths. With the fix,
the six cases and the Torch-only import check reported **7 passed,
30 deselected, 0 failed, 0 skipped**. The import check blocks JAX in a fresh
interpreter; the JAX export assertion now lives in the optional JAX suite.

```sh
python -m pytest simplified_sparse_attention/cutedsl/test_simplified_sparse_attention.py \
  -m L0 -q -s -k 'host_threads_share_stream or lazy_imports'
```

The rebuilt `nvidia_cudnn_frontend-1.33.0-cp312-cp312-linux_x86_64.whl`
has SHA256 `55db69c2370f8bdc0fdd294c7a8352790011c0d54cd8ee1c71c6ada153a8eade`.
All 35 operation Python modules, `NOTICE`, and `contributors.md` match the
edited checkout byte for byte. The installed-wheel API-index check reported
**11 passed**, including the runtime baseline with six intentional additions
and no removals.

The installed-wheel L0 and L1 run reported **161 passed, 24 xfailed,
0 failed, 0 skipped, 0 deselected** in 750.39 seconds. It includes all six
concurrency cases, both frameworks' numerical and determinism suites, and
all six long-sequence cases. Existing expected failures and goldens are
unchanged. SM100 forward remains replay-only and backward remains not
guaranteed; SM103, long-path batch invariance, and saved internal statistics
remain unverified.

Runtime: SM100 (148 SMs), driver 610.11, Python 3.12.12,
PyTorch 2.13.0+cu130, JAX/JAXlib 0.11.0, cuDNN 9.27.0, CuTe DSL 4.8.0
with the CUDA 13.4 compiler, and TVM FFI 0.1.13.post2. Imports resolved to
the installed wheel. From `test/python`:

```sh
XLA_PYTHON_CLIENT_PREALLOCATE=false CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest simplified_sparse_attention/cutedsl -m 'L0 or L1' -q -s -rxX
```

## Phase 1 dense indexer KL public exports (2026-11-09)

Revision `b1a9c4a`, based on `55b9180`. The installed wheel exposes
`dense_indexer_kl_forward` through both `cudnn.ops` and
`cudnn.simplified_sparse_attention`. The API retains the scalar loss and
indexer-only autograd contract. `use_kf=False` selects chunked PyTorch;
`use_kf=True` selects the supplied-LSE CuTe kernel. Both reject external
CUDA graph capture. No kernel arithmetic or deterministic mode was added.

The new suites collect 64 L0 cases: 32 numerical/contract cases and 32
forward/backward determinism cases. Numerical coverage uses an independent
FP64 full-softmax teacher and autograd for the student. It checks all three
head counts, query/key tails (3/17, 65/257), and split-key work (33/4097).
The loss uses coefficient 0.17 and fixed denominator 71; backward uses a
fixed upstream gradient of 0.7. Loss tolerance is `rtol=2e-4, atol=2e-7`;
BF16 gradient tolerance is `rtol=0.02, atol=2e-6`.

Replay compares raw bytes of the loss, BF16 indexer gradients, and saved
FP32 gradients with fresh allocations and concurrent GPU work. The stress
case has `32 * (SM_count + 1)` query rows (4,768 here), crossing CTA scheduling
waves. Packed checks preserve inputs, scaling, and upstream gradients across
neighbor insertion/removal, reordering, duplication, and changed neighbors.
Only the target contributes to the scalar loss. Inputs are checked for
mutation, outputs for finiteness, and teacher Q/K for absent gradients.
Contract tests cover invalid metadata, chunk sizes, fused dtype/head/scale/LSE
requirements, old-DSL rejection before kernel import, capture rejection, and
side-stream forward/backward without host synchronization.

Before exceptions, the initial runs reported **40 passed, 10 failed,
8 deselected**, followed by **6 passed, 2 failed, 24 deselected** for stress
and golden recording. All numerical checks passed. The 12 bytewise failures
were approved as scoped expected failures:

- Fused H32 tail forward/backward replay: scalar loss differs by one byte.
- Fused H96 split-key backward replay: saved FP32 indexer gradients differ.
- Fused scheduling stress, both directions: scalar loss differs by one byte.
- Packed PyTorch backward at H16/H32/H96 and fused backward at H32/H96:
  loss or saved gradient bytes differ.
- Packed H96 forward, both paths: scalar loss differs by one byte.

Exceptions match only these named cases on SM100 and raw-byte mismatch
messages for loss/indexer gradients. Numerical, non-finite, mutation,
shape/dtype, and unrelated failures remain strict. A case stops at its first
byte mismatch; later comparisons in that case are unmeasured. Remove each
exception when its replay and batch checks are fixed. Intermittent passes do
not establish determinism.

Runtime: SM100, 148 SMs, driver 610.11 / CUDA driver API 13.3; Python 3.12.12,
frontend 1.33.0, PyTorch 2.13.0+cu130, cuDNN 9.27.0, CuTe DSL 4.8.0,
TVM FFI 0.1.13.post2. Separate environments loaded native compiler extensions from
`cutlass/_mlir/_mlir_libs/`: the cu13 flavor reported compiler CUDA 13.4
and the cu12 flavor reported compiler CUDA 12.9.
Each installed-wheel lane used a fresh process and separate empty DSL/driver
cache directories. Imports resolved to the corresponding environment's
`site-packages/cudnn`, not the editable checkout.

| Check | Result |
| --- | --- |
| Installed wheel, CUDA 13.4, Phase 1 plus nine Phase 2 numerical cases and both framework export checks | 63 passed, 12 xfailed; 0 failed/skipped/deselected, 102.49 s |
| Installed wheel, CUDA 12.9, all Phase 1 cases | 54 passed, 10 xfailed; 0 failed/skipped/deselected, 43.82 s |
| Editable source, Phase 1, lazy exports, and three kernel-prefix checks | 56 passed, 12 xfailed; 0 failed/skipped/deselected, 54.12 s |
| Installed-wheel API inventory | 11 passed; one added name, no removals |
| Wheel contents | All 35 operation Python modules, NOTICE, and contributors.md match the checkout bytes |
| Formatting / docs | Pre-commit passed; Fern: 0 errors, 1 existing warning |

The two H32 fused replay cases passed in the CUDA 12 run; their known
exceptions remain. Four one-row forward/backward goldens were recorded in a
new directory for each compiler flavor after reference and replay checks.
The eight committed files have identical input and output hashes across
flavors for the same backend/direction. No existing golden was replaced.
These tiny cases cover the scalar oracle and PyTorch path; the fused fast
path has no stable golden because larger reductions fail replay.

Wheel: `nvidia_cudnn_frontend-1.33.0-cp312-cp312-linux_x86_64.whl`.
SHA256: `4a3bb8627750be8121c8436647dd4ff5d9b8dd10803f6603cf6fe3e745871208`.
The CUDA 12 compiler lane uses CUDA 13 framework/backend packages; it does
not qualify a full CUDA 12 deployment. SM103, other GPUs, newer DSL releases,
and multi-device execution remain unverified. Phase 1 has no JAX entry point.
Phase 2's earlier full-suite results above remain historical measurements.

Commands from `test/python`, with the installed cuDNN library directory on
`LD_LIBRARY_PATH` and separate fresh `CUTE_DSL_CACHE_DIR`/`CUDA_CACHE_PATH`
values per compiler lane:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 XLA_PYTHON_CLIENT_PREALLOCATE=false \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
python -m pytest \
  simplified_sparse_attention/cutedsl/test_dense_indexer_kl.py \
  simplified_sparse_attention/cutedsl/test_dense_indexer_kl_determinism.py \
  simplified_sparse_attention/cutedsl/test_simplified_sparse_attention.py::test_lazy_imports \
  simplified_sparse_attention/cutedsl/test_simplified_sparse_attention.py::test_public_forward_backward_reference \
  simplified_sparse_attention/cutedsl/test_simplified_sparse_attention_jax.py::test_public_jax_export \
  -m L0 -q -s -rxX --tb=short

CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
python -m pytest \
  simplified_sparse_attention/cutedsl/test_dense_indexer_kl.py \
  simplified_sparse_attention/cutedsl/test_dense_indexer_kl_determinism.py \
  -m L0 -q -s -rxX --tb=short
```

## Simplified sparse attention: offset and CI regressions (2026-11-10)

The triangular score producer and top-k consumer now widen operands before
workspace products. The tile decoder derives its upper bound from the query
extent. `test_triangular_offsets.py` extracts the production expressions and
checks them with fixed-width host scalars and native CuTe probes. Rows span
0 through 262143, including the 64K overflow boundary; decoder cases cover
1, 512, 1024, 1025 and 2048 blocks. The probes write offsets to small output
arrays and never dereference the large addresses.

The unfixed code produced **13 failed, 12 passed** in the 25-case regression
selection. The fixed code produced **25 passed** on SM100 with the CUDA 13.4
compiler. This selection also checks that public JAX exports remain inspectable
with a modeled DSL 4.7.1 while execution still rejects before kernel import,
and that all Torch/JAX test transfers enter the tensor's device context.
Three public triangular cases additionally passed numerical forward/backward
checks and forward replay across two tiles, for H16/K512, H32/K1024 and
H96/K2048. These checks do not establish backward determinism.

CI also measured a one-byte scalar-loss mismatch for packed H16/K512 forward
on SM100 with PyTorch 2.14.0a0 and CUDA 13.4. It passed numerical checks before
failing byte comparison. The earlier local H16 pass does not establish batch
invariance; the support page retains the replay-only status.

The installed-wheel API-index suite passed all 11 tests with actual DSL 4.7.1
and all 11 with DSL 4.8.0. Direct execution under 4.7.1 still raises the
4.8 requirement before loading SSA kernel modules. No API baseline names were
removed. The guard-word scan passed all 214 rules on the tracked source tree.
The rebuilt wheel contains all 35 operation Python modules, NOTICE and
contributors.md with bytes identical to the checkout.

The full installed-wheel CUDA 13.4 run reported **242 passed, 1 failed,
1 skipped, 34 xfailed** in 833.79 s. Its sole failure was a test-side stream
race in `test_reserved_state_at_its_maxima`: the side stream consumed zero
queries before the caller stream finished producing them. The test now waits
on the producer and delays that producer to expose a missing dependency.
It passed in 23.69 s; removing only the wait failed again in 24.52 s. No
kernel, tolerance or expected-failure rule changed for this test repair.
The skipped case needs two GPUs; its device-context control passed on this
single-GPU host. All existing goldens remain unchanged. The H16 CI packed-loss
case remained strict in this run; the approved exception is recorded below.

Runtime: SM100, 148 SMs, driver 610.11, frontend 1.33.0, Python 3.12.12,
PyTorch 2.13.0+cu130, JAX/JAXlib 0.11.0, cuDNN 9.27.0, CuTe DSL 4.8.0
and TVM FFI 0.1.13.post2. Imports resolved to the installed wheel. Its SHA256
is `52b1d3f198e09e6ece095ceb454b9469666643c870249becf01f676137c03a86`.
SM103 and actual multi-GPU execution remain unverified. External CUDA graph
capture remains unsupported; the existing rejection tests still pass.

Code revision: `a8300fe`. The separate installed-wheel CUDA 12.9 compiler run
passed **62 tests, with 1 two-GPU skip**, in 336.19 s: all address/decoder
probes, nine public Torch numerical cases, three triangular numerical/replay
cases, the JAX version guard, all public JAX tests, both long-key JAX cases,
and the delayed-producer reservation test. This lane uses CUDA 13 framework
and backend packages; it does not qualify a full CUDA 12 deployment or CUDA
12 bytewise batch invariance. Pre-commit passed. Fern reported 0 errors and
1 existing warning.

Commands from `test/python`, with the installed wheel environment active,
`LD_LIBRARY_PATH` pointing to cuDNN 9.27, and separate DSL/driver cache paths:

```sh
# CUDA 13.4: full suite, before correcting the test's stream dependency.
CUBLAS_WORKSPACE_CONFIG=:4096:8 XLA_PYTHON_CLIENT_PREALLOCATE=false \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
python -m pytest simplified_sparse_attention/cutedsl \
  -m 'L0 or L1' -q -s -rxX --tb=short

# CUDA 13.4: corrected delayed-producer test and its missing-wait control.
python -m pytest \
  simplified_sparse_attention/cutedsl/test_filtered_contract_cuda.py::test_reserved_state_at_its_maxima \
  -m L1 -q -s --tb=short

# CUDA 12.9: changed paths and numerical compatibility.
CUBLAS_WORKSPACE_CONFIG=:4096:8 XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m pytest \
  simplified_sparse_attention/cutedsl/test_triangular_offsets.py \
  simplified_sparse_attention/cutedsl/test_simplified_sparse_attention.py::test_public_forward_backward_reference \
  simplified_sparse_attention/cutedsl/test_simplified_sparse_attention.py::test_triangular_tiles_reference_and_replay \
  simplified_sparse_attention/cutedsl/test_simplified_sparse_attention.py::test_jax_old_dsl_allows_inspection_but_rejects_execution \
  simplified_sparse_attention/cutedsl/test_simplified_sparse_attention_jax.py \
  simplified_sparse_attention/cutedsl/test_filtered_jax.py \
  simplified_sparse_attention/cutedsl/test_filtered_contract_cuda.py::test_reserved_state_at_its_maxima \
  -m 'L0 or L1' -q -s --tb=short
```

From the repository root, in each DSL 4.7.1/4.8.0 environment, with
`CUDNN_API_INDEX_PACKAGE_ROOT` set to the installed `cudnn` directory:

```sh
CUDA_VISIBLE_DEVICES='' python -m unittest discover \
  -s test/python/api_index -p test_api_index.py -v
```

### Approved H16 packed-loss exception (2026-11-10)

Extend the existing byte-mismatch exception to PyTorch BF16 forward at
H16/top-k512 on SM100, as approved. It matches only
`packed slot 1 indexer_loss: 1/4 bytes differ`. Numerical checks still run
before byte comparison. H16 backward loss, replay, other tensors, other byte
counts, SM103, and the JAX H16 case remain outside this new exception. Remove
it when the scalar-loss reduction is batch invariant. The replay-only support
status is unchanged.

On the same installed wheel and CUDA 13.4 runtime recorded above, the six
Torch batch cases reported **1 passed, 5 xfailed** in 33.32 s. H16 forward
passed locally; its CI failure remains recorded above. Seven injected-failure
controls confirmed the previous code rejects the observed diagnostic, the
approved forward case reports an expected failure, and backward loss, SM103,
two-byte loss, other-output, and numerical failures still raise assertions.
No kernel, numerical tolerance or golden changed.

```sh
# From test/python, with the installed wheel environment active.
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m pytest \
  simplified_sparse_attention/cutedsl/test_simplified_sparse_attention_determinism.py::test_batch_invariance \
  -m L0 -q -s -rxX --tb=short
```

### SM107 D512/D576 FP32 split partial stores — 2026-11-10

Aligned, D-contiguous partial rows now use 16-byte global stores. The scalar
fallback retains per-element bounds and dead-row zero selection. Two added
L0 paged boundary cases exercise V480 and empty V264 inside the D512 envelope.
The independent references, rebinding and changed-input replay stay unchanged.

SM107,216SM, driver620.62, Torch2.14.1+cu130, cuDNN9.30.0.3, DSL4.8.0
with compiler13.4:98 native cases passed in158.35s, without skips. These cover
dense/paged KV, FP16/BF16, D512/D576, packed/unpacked splits, masks/sinks,
optional/natural/log2 Stats, physical layouts and D512 mixed E4M3/E5M2 KV.
The12 half/mixed runtime goldens under `d634113bbc56110d` were recorded from
the unchanged parent after its independent references and byte checks passed
(12cases,31.06s). The candidate passed strict comparison against all12 with
recording unset. No existing golden was replaced. An initial candidate run
passed34 cases before stopping on the missing runtime-specific golden; it
reported no numerical mismatch. Other GPU/runtime execution is unverified.

From `test/python`, after asserting the isolated import path:

```bash
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  pytest -q -m 'L0 or L1' \
  sdpa/frost/test_sdpa_d512_pack_split_sm107.py \
  sdpa/frost/test_sdpa_fwd_native_paged_sm107.py::TestNativeD512HalfPaged \
  sdpa/frost/test_sdpa_fwd_native_paged_sm107.py::TestNativeD576Half \
  sdpa/frost/test_sdpa_fp8_sm107.py::TestMixedFp8Kv::test_d512_dense_paged \
  sdpa/frost/test_sdpa_fp8_sm107.py::TestMixedFp8Kv::test_d512_standalone \
  sdpa/frost/test_sdpa_fp8_sm107.py::TestMixedFp8Kv::test_d512_replay_batch_golden
```

### Cake DSA key bounds (2026-11-10)

PR #2435, based on `17e10b09e`, adds logical key-count checks inside the
register-MMA attention forward/backward and indexer-backward kernels. Invalid
indices become negative holes before both initial and prefetched KV loads.
The public API, caller workspace, and launch count are unchanged. Local source
hashes and rebuilt cache keys are recorded in `_cake/VENDORED_DSA_DET.json`.

On `cudnn-dev-equator-22-04` (SM100, 148 SMs), CUDA Toolkit 13.4.92,
driver 610.11, Torch 2.11.0+cu128, and TVM-FFI 0.1.13.post3:

- All four Cake attention suites: 46 passed, no skips, in 96.22 seconds.
- Eight BF16 boundary cases cover H=8/16/32/64 and D=512/576, K=128,
  the logical end index, oversized positive indices, int32 extremes, negative
  holes, and a valid key in the second tile. A larger backing allocation holds
  canary rows outside the logical KV view.
- Independent FP64 references check O, LSE, dQ, and dKV. The cases check input
  and guard-row preservation, all-hole rows, supplied scratch, no allocation or
  host synchronization after warmup, and changed-index graph replay on a side
  stream with refreshed grouping. Repeated results and an isolated live row
  with fresh buffers match byte for byte. These small analytic cases do not
  establish general batch invariance.
- Compute Sanitizer 13.2 memcheck: eight boundary cases passed; zero errors.
- Eight new golden files were recorded only after reference and replay checks.
  They record the corrected invalid-index behavior; no existing golden changed.
- The existing three indexer-gradient cases now include invalid positive IDs
  in later tiles and compare all gradients against the independent reference.
  SM103 and SM107 were not available for this qualification.

The pre-fix H16/D576 probe failed with an illegal memory access in its isolated
process. The patched case passed, followed by the complete suite and memcheck.

Run from the repository root with the matching Python environment and CUDA
Toolkit configured. `PYTHONPATH=python:test/python` must resolve this checkout:

```bash
python -m pytest --noconftest --import-mode=importlib -o addopts='' \
  test/python/sdpa/cake/test_gqa_training.py \
  test/python/deepseek_sparse_attention/cake/test_indexer_topk.py \
  test/python/deepseek_sparse_attention/cake/test_sparse.py \
  test/python/deepseek_sparse_attention/cake/test_deterministic.py -ra
compute-sanitizer --tool memcheck --error-exitcode 99 python -m pytest \
  --noconftest --import-mode=importlib -o addopts='' \
  test/python/deepseek_sparse_attention/cake/test_deterministic.py -k key_bounds -x
CUDNN_DETERMINISM_GOLDENS="$PWD/test/python/determinism_goldens" python -m pytest \
  --noconftest --import-mode=importlib -o addopts='' \
  test/python/deepseek_sparse_attention/cake/test_deterministic.py -k key_bounds -x
```

### SM120 half THD physical origins — 2026-11-10

`test_sdpa_fwd_thd_origins_sm120.py` adds 24 L0 functional cases covering
FP16/BF16, token/head Stats, D64/D128/D192-V128/D256/D512/D576-V512.
Each compares against an independent FP64 reference, protects padding
canaries and checks mixed-width/multiplier origins, fresh pitches/pointers,
empty and keyless transitions under retained capture. Exact-byte checks
cover repeat execution, batch reorder/duplication, a sequence alone and
changed neighbors. Twelve separate golden cases finish those checks before
recording. No prior golden or tolerance is replaced.

RTX PRO 6000 SM120 (188 SMs), driver 595.84.01, Torch 2.14.0+cu130,
cuDNN 9.24.0.43, DSL 4.8.0 / compiler 13.4, cuda-bindings 13.0.3:
51 functional cases passed, including6 native/reference adapter-fallback,
3 bounded-batch and18 dense half/per-tensor-FP8 host controls. All 12 new
goldens under `812f98809d9324c5` recorded and strictly checked. Six memcheck
cases passed with 0 errors. Racecheck then reported 628 errors in V-tail
sanitization; synccheck did not run. The follow-up gives one warp ownership and synchronizes the compute group.
Complete compact D64/D256 controls report 88/226 race errors on the original
source and zero on the repair; their reference and retained replay checks
pass in both arms. The final run passes 73 functional/golden cases in 183.14 s
and 15 memcheck cases in 6.97 s with zero errors, preserving all 12 golden
hashes. Racecheck passes the same 15 cases in 62.93 s with zero errors/warnings;
synccheck passes 15 in 3.79 s with zero errors. The original FlashInfer issue
regression also passes all three default FROST cases and all three backend
controls after the synchronization repair. Earlier script quoting, fixture
multiplier, cache-directory and node-local-reuse failures happened outside
this successful final run; they are not counted as GPU passes.
SM121 and other runtimes remain unverified for independent physical gaps.

```bash
# From test/python after checking the isolated import path.
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  pytest -m L0 sdpa/frost/test_sdpa_fwd_thd_origins_sm120.py
```


### SM107 coalesced packed split partials — 2026-11-11

`test_coalesced_split_replay_batch_golden` adds four L0 cases for FP16/BF16
and packing groups32/64 at split4. It checks standalone/reordered/duplicated
and changed-neighbor batches through33members, repeated raw bytes, numerical
references and changed-input CUDA Graph replay. Existing graph/standalone
boundary coverage now includes groups16and32, alongside64/128, partial groups,
keyless rows, masks and sinks.

On SM107 with driver620.62, Torch2.14.1+cu130, cuDNN9.24.0.43 and
DSL4.8.0/compiler13.4,106native functional/strict-golden cases passed without
skips.20cases each passed Compute Sanitizer2026.3 memcheck, racecheck and
synccheck, with zero errors and zero race warnings. Memcheck was unfiltered;
race/sync used the attention-kernel name filter with API errors enabled.
All16records under `de01ba4cf30a4c9b` were recorded from unchanged parent
`8e31a5725` after its references and replay/batch checks passed, then checked
strictly by the candidate. No previous baseline or tolerance changed.

From `test/python`, use the preceding D512/D576 qualification command and add
`sdpa/frost/test_sdpa_fp8_sm107.py::test_coalesced_split_replay_batch_golden`.
The four coalesced cases live in that explicit SM107 CI entry, so the architecture
lane retains the new coverage without duplicate collection in directory sweeps.
The sanitizer subset is
`test_sdpa_fwd_native_paged_sm107.py::TestNativeD512HalfPaged::test_boundaries`
plus `test_sdpa_d512_pack_split_sm107.py`, selected with
`test_boundaries or g64_split or g128_split or g32_split or g16_split or keyless`.
Earlier runs stopped before tests on a missing interpreter target and a DSL
package path; the final execution used the explicitly verified imports above.

### SM120 FP8/MXFP8 shared-memory synchronization — 2026-11-11

`test_sdpa_fp8_v_tail_sync_sm120.py` adds eight replay/batch cases, eight
runtime goldens and four CPU scale-layout controls. Coverage is per-tensor
FP8 and MXFP8, D128/D512, E4M3/E5M2, BF16 output, Q33 with KV capacity80
and live length17, and poisoned padding. Exact split1/unpacked plans retain
quantized payloads and logical scales across standalone, reordered, duplicated
and changed-neighbor batches. O/Stats are compared bytewise; global Amax is
checked for identical-input replay and against each whole-batch reference.

RTX PRO 6000 Blackwell Server Edition (SM120,188SM), driver595.84.01,
Torch2.14.0+cu130, cuDNN9.24.0.43, DSL4.8.0/compiler13.4 and
cuda-bindings13.0.3: **45 functional/strict checks passed** in76.38s without
skips. The existing dense/THD cases include multiple KV tiles and persistent
work units. **19 cases each passed memcheck, racecheck and synccheck**
in22.48s/16.95s/4.05s, with zero errors and zero race warnings. Memcheck
was unfiltered; race/sync used the `cudnn_kernel_` name filter with API errors
enabled, under Compute Sanitizer2026.3.0.

All eight goldens under `812f98809d9324c5` were recorded from unchanged parent
`8e31a5725` after independent references and byte checks passed, then checked
strictly by the candidate with recording unset. No existing baseline or
numerical tolerance changed. The parent passes four selected numerical cases
but reports924race errors; this negative control prevents numerical success
from hiding the synchronization defect. Earlier fixture and sanitizer failures
were retained during development. The supplemental performance script stopped
after qualification on a missing test-helper import; its failure did not
produce a performance result or bypass any qualification check.

From `test/python`, after checking that `cudnn.__file__` names the intended
checkout, the functional qualification command is:

```bash
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  pytest -v -m 'L0 or L1' \
  sdpa/frost/test_sdpa_fp8_v_tail_sync_sm120.py \
  sdpa/frost/test_sdpa_fwd_fp8_sm120.py \
  sdpa/frost/test_sdpa_fwd_mxfp8_sm120.py \
  -k 'v_tail or thd_poisoned_pad or d512_thd or thd_multi_unit_per_cta or padded_kv_poisoned_tail or d512_persistent_units or thd_head_dims'
```

For the sanitizer subset, replace the selection with
`(replay_and_batch and e4m3) or thd_poisoned_pad or d512_thd or thd_multi_unit_per_cta`.
SM121 execution, cross-runtime identity and independent physical THD origins
for quantized inputs remain unverified by this addition.

### SM120 per-tensor FP8 THD physical origins — 2026-11-11

`test_sdpa_fwd_thd_origins_fp8_sm120.py` adds 28 L0 functional cases and
eight golden cases. E4M3/E5M2 inputs and BF16 output cover independent
Q/K/V/O/Stats physical origins, mixed INT32/INT64 offsets and multipliers,
token/head-major Stats, changed pitches/pointers, empty/keyless transitions,
and retained capture with changed lengths, origins and scalar scales. Head
shapes are D64, D128, D192/V128, D256, D512 and D384/V512. Numerical
checks use an independent CPU FP64 reference; raw-byte checks cover repeat
execution, batch reorder/duplication, a sequence alone and changed neighbors.
The prepared and standalone fallback routes are checked within their existing
contracts. The hot-path check rejects host synchronization and CUDA allocations.

RTX 5090 SM120 (170 SMs), driver 620.62, Torch 2.14.0+cu130, cuDNN
9.24.0.43, DSL 4.8.0/compiler 13.4: **30 functional cases passed**, including
the existing D512 dense-split and interleaved standalone THD controls. The
candidate strictly preserves **12 unchanged-parent goldens** (eight FP8/MXFP8
V-tail and four FP16/BF16 origin controls). **Eight new origin goldens** under
`dee7499a23a43ab6` were recorded only after reference and byte checks passed,
then strictly checked with recording unset. No existing baseline was replaced.
The qualified runtime/test source is `b971af997`; the subsequent commit adds
only these records and documentation.

The unchanged combined parent `5196c9c35` fails all four D128/D512 E4M3/E5M2
poisoned-gap detectors while the candidate passes; all eight compact controls
pass. A separate E5M2 attribution run compares parent compact, candidate
compact and candidate gapped layouts: all 48 input/output/reference/error
records match exactly across the three arms, including nine identical failures
of the initially proposed 0.03 budget (maximum absolute error 0.05463). The new
E5M2 cases therefore use the existing SM120 E5M2 absolute 0.1 budget, with
zero relative tolerance, accounting for the two-mantissa-bit P cast. Existing
tests, E4M3/Stats tolerances and strict byte checks are unchanged.

**Four cases each passed memcheck, racecheck and synccheck**, with zero
errors and zero race warnings under Compute Sanitizer 2026.3.0. No kernel
filter was used. API reporting was `--report-api-errors explicit`, retaining
errors returned by explicitly called CUDA APIs. The default extended mode
first exited 91 after a `Selective device code recompilation in progress`
API warning; an independent `torch.full()` control, without importing FE,
reproduced that same warning and passed under explicit reporting. The original
warning is retained as a tool/runtime diagnostic, not counted as a clean run.
Earlier missing-dependency, fallback-override, unsupported-packing and
provisional E5M2-budget failures were resolved before these passing results.

From `test/python`, after checking the isolated import path:

```bash
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  pytest -v -m 'L0 or L1' \
  sdpa/frost/test_sdpa_fwd_thd_origins_fp8_sm120.py \
  sdpa/frost/test_sdpa_fwd_fp8_sm120.py::test_fp8_sm120_d512_split_kv \
  sdpa/frost/test_sdpa_fwd_fp8_sm120.py::test_fp8_sm120_thd_fused_record_slices
```

The sanitizer subset selects independent-origin token/E4M3/D128 and
head/E5M2/D512, plus omitted-total-bound E5M2/D128 and E4M3/D512. Each
uses fixed split1/unpacked execution. SM121, MXFP8 physical gaps, backward
and cross-runtime byte identity remain outside this qualification.

### SM120 MXFP8 THD physical origins — 2026-11-11

`test_sdpa_fwd_thd_origins_mxfp8_sm120.py` adds28 L0 functional cases and
eight golden cases. E4M3/E5M2 inputs with BF16 output cover independent
Q/K/V/O/Stats origins, mixed INT32/INT64 offsets and multipliers,
token/head-major Stats, fresh pointers/pitches, empty/keyless transitions
and retained capture. Head shapes are D64,D128,D192/V128,D256,D512 and
D384/V512, with fixed split1/unpacked plans. An independent CPU FP64
reference checks values; exact-byte comparisons cover replay, batch
reordering/duplication, a sequence alone and changed neighbors. Batch
changes preserve raw payloads and repack existing logical scale bytes.
The fallback-adapter and hot-path checks retain their existing contracts.

RTX5090 SM120 (170SMs), driver620.62, Torch2.14.0+cu130, cuDNN9.24.0.43,
DSL4.8.0/compiler13.4:28 functional cases and12 unchanged-parent strict
controls pass on source288043149. Runtime sources are unchanged since
49c628af6. Four shared default MXFP8 cases and four native-route twins
also pass; the FP8-fixed/MXFP8-old parent fails all four origin detectors.
The combined shared suite passes10 native and11 public cases, including
retained half/FP8 and explicit BF16 backend controls.

The shared D512 V-scale fixture previously reshaped plane-major scales.
Both fixtures now use a byte permutation checked against independent
per-head/128-token quantization:8 controls pass, with4 old D512 failures
and4 old D128 passes. The production scale-load arithmetic is unchanged.
An earlier replay test doubled V without scaling its absolute reference
budget and failed15 cases with1–2 mismatches each. The final test first
checks the exact replay inputs at the original scale, then requires doubled
O and unchanged Stats/padding bytewise. It compares the doubled result in
original V units using the same reference error constants. All28 cases
pass; no existing tolerance or golden is replaced.

All eight new records under `dee7499a23a43ab6` were recorded after the
reference/replay/batch checks, then strictly verified with recording unset.
Four cases each pass memcheck, racecheck and synccheck in20.98s/285.20s/13.63s,
with zero errors and zero race warnings. These include D128/E4M3 and
D512/E5M2 full replay plus D128/E5M2 and D512/E4M3 without total hints.
No kernel-name filter is used. API diagnostics use `--report-api-errors explicit`;
the extended-mode runtime warning was independently reproduced by a Torch-only
control in the preceding per-tensor qualification. The first recording attempt
set both record/check variables and failed their mutual-exclusion precondition;
it produced no records and was corrected before the successful strict run.
SM121, backward, other runtimes and performance remain unverified here.

```bash
# From test/python with the isolated import path verified.
CUBLAS_WORKSPACE_CONFIG=:4096:8 pytest -m L0 \
  sdpa/frost/test_sdpa_fwd_thd_origins_mxfp8_sm120.py \
  -k 'not golden'
# Record only after reference/replay checks; then run with RECORD unset.
CUDNN_DETERMINISM_RECORD=/tmp/new-mxfp8-goldens pytest -m L0 \
  sdpa/frost/test_sdpa_fwd_thd_origins_mxfp8_sm120.py -k golden
CUDNN_DETERMINISM_GOLDENS=/tmp/new-mxfp8-goldens pytest -m L0 \
  sdpa/frost/test_sdpa_fwd_thd_origins_mxfp8_sm120.py -k golden
```

### SM100 FP8 D128 THD physical origins — 2026-11-11

`test_sdpa_fwd_fp8_sm100.py::TestD128Fp8PhysicalOrigins` collects the
shared `test_mhas_v2` E4M3/E5M2 D128 cases with a strict native engine pin,
four token/head-major reference/rebind/replay cases, two no-total-hint
cases, two half D128/D256 forwarding controls and four new strict goldens.
All use the existing unsplit, unpacked CGA2 nonpaged THD contract.

The four replay cases check mixed INT32/INT64 offsets and multipliers,
independent Q/K/V/O/Stats gaps, NaN-filled input padding, output canaries,
changed pitches and storage, retained captures with changed live lengths
and offsets, and all-empty queries. Exact-byte checks cover repetitions,
batch permutations/duplicates, a sequence alone and changed neighbors.
The prepared graph and standalone graph routes produce the same bytes;
execute retains its allocation-free, asynchronous behavior. The numerical
reference and tolerances are unchanged.

Measured on SM100 (148 SMs), driver 595.71.05, Torch 2.14.1+cu130,
cuDNN 9.24.0.43, DSL 4.8.0 / compiler CUDA 12.9, tvm-ffi 0.1.14.post1:
the 10 functional cases and four strict goldens pass. New hashes live under
`361d6da48de8fd55`; no earlier golden is replaced. Twelve existing compact
D128 / paged D128-D256 FP8 controls also pass. The original shared D128
FP8 cases fail on the parent and pass on the repair. An exploratory CGA1
selection was rejected by the existing graph contract before kernel launch;
it is outside this unsplit nonpaged test domain.

Four cases each pass memcheck, racecheck and synccheck with zero errors
and zero race warnings. The initial memcheck run reported 20
`cuGetProcAddress_v2` API errors; a Torch-initialization-only control with
no FE import reproduced all 20. Initializing cuda-python before Torch in
the private sanitizer launcher eliminates those diagnostics while retaining
`--report-api-errors explicit`, all kernel instrumentation and error exits.
SM103 E4M3/E5M2 physical-origin lowerings compile offline; GPU execution
there is unverified. Wider heads, MXFP8, SM107 and backward are not qualified
by these D128 results.

```bash
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  pytest -q -m L0 \
  sdpa/frost/test_sdpa_fwd_fp8_sm100.py::TestD128Fp8PhysicalOrigins
```

### SM100 MXFP8 D128/D512 THD physical origins — 2026-11-11

`test_sdpa_fwd_mxfp8_sm100.py::TestPhysicalOrigins` collects 24 L0 cases:
4 shared `test_mhas_v2` regressions, 8 reference/rebind/replay/batch cases,
4 no-total-hint cases and 8 new goldens. Both E4M3/E5M2 types run the
existing public THD configurations: D128 CGA2 and D512 CGA1. The initial
expanded probe also requested D128 CGA1 and D512 CGA2; those twelve cases
were existing planning declines, not numerical failures. The suite now
states the admitted pairs explicitly, with no capability or ranking change.
Every expanded case asserts its actual native kernel template.

The shared fixture retains quantized payload and scale bytes across batch
changes, poisons physical data gaps with NaNs, and checks output canaries.
It covers independent mixed-width offsets and multipliers, FP64 references,
new pointers/strides, live/empty retained replay, scale-only replay changes,
batch reorder/duplication/isolation and prepared/standalone graph parity.
The D512 maps use token dimension 1 for Q/K/V and dimension 2 for O;
D128 maps retain dimension 2 for every port. Scale map coordinates continue
to use compact logical prefixes.

Current measured runtime: SM100, 148 SMs; driver 595.71.05;
Torch 2.14.1+cu130; cuDNN 9.24.0.43; DSL 4.8.0 with its CUDA 13.4
compiler; cuda-bindings 13.4.4; tvm-ffi 0.1.14.post1. The runner asserts
`cutlass.CUDA_VERSION`, uses a fresh compiler cache, and checks the imported
checkout and native-extension hashes. The native C++ extension is rebuilt
against toolkit 13.4.92.

On this runtime all four shared cases fail on the immediate parent and pass
after repair. The full strict matrix passes all 24 cases, including eight
unchanged goldens; 30 neighboring MXFP8 cases also pass. No case is skipped.
The new baselines use `361d6da48de8fd55`; older goldens and all tolerances
are unchanged. The historical CUDA 12.9 run passed the 16 functional cases
and eight new golden-record cases with independent references and exact-byte
checks before recording. Its later strict run also passed all 24 cases.
Compiler-version equality is asserted separately because the current golden
runtime key records the DSL package version, not its CUDA compiler flavor.

CUDA 13.4 memcheck additionally reproduced pre-existing single-CTA THD
scheduler errors on the unchanged D256/D512 parent. Local publication fixes
those errors; D256 broadcasts all predecoded fields before an elected warp
returns slot credit, while D512 retains its per-lane credits and matching count.
The six directed memcheck cases then pass with zero errors.

Racecheck also reproduced the existing D128 CGA2 V-scale tail helper's access
to an exited peer: a compact causal-tail parent control reports 2400 errors.
The tail path now keeps both CTAs alive through a cluster exit barrier. Both
origin replay dtype cases pass racecheck with zero hazards, errors or warnings;
the remaining four D256/D512 racecheck cases also pass in a separate run,
and the six-case synccheck suite passes with zero errors. The final repaired
source passes strict24 with unchanged goldens and all30 MXFP8 neighbors. The earlier failed
sanitizer runs remain in the private qualification report.

All six SM103 D128/D256/D512 x E4M3/E5M2 lowerings compile with CUDA 13.4.
The shared origin helper also compiles in the SM107 half D128 consumer,
including its single-batch fast path. This is compile coverage only;
SM103 execution and SM107 MXFP8 execution remain unverified.

```bash
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  pytest -q -m L0 \
  sdpa/frost/test_sdpa_fwd_mxfp8_sm100.py::TestPhysicalOrigins
```


### SM100 D192/D256 quantized THD origins — 2026-11-11

`TestMidwidthPhysicalOrigins` in both SM100 FP8/MXFP8 architecture suites
collects60 L0 cases. It pins the native D192/V128 body at CGA1/CGA2 and the
D256 body at CGA1; both E4M3/E5M2 inputs write BF16 O and token/head-major
Stats. Tests cover independent mixed-width/multiplier offsets, poisoned gaps,
changed pointers/pitches, omitted total hints, empty/keyless transitions,
retained changed-input capture, raw-byte replay during concurrent work,
batch reordering/duplication/neighbor isolation and graph fallback parity.
Two persistent clusters force sequence and scheduler-slot reuse. MXFP8 data
moves independently of the original packed logical scale tiles.

The recorded runtime is SM100/148SM, driver595.71.05, Torch2.14.1+cu130,
cuDNN9.24.0.43, DSL4.8.0 with actual CUDA13.4 compiler, bindings13.4.4 and
toolkit13.4.92. Import root and native-extension hash are asserted. The
unchanged parent extension is reused; this change does not edit its ABI.
There are24 new baselines under `361d6da48de8fd55`; the earlier16 remain
byte-identical after adding CGA1 and concurrent-work coverage. Existing
repository goldens and numerical tolerances are unchanged.

The immediate parent fails four independent-origin reference controls.
Sanitizers also reproduce its dedicated D192 CGA1 schedulers' invalid DSMEM
publication and its CGA2 MXFP8 V-scale peer lifetime. Persistent compact
D192 FP8 CGA2 additionally reports scheduler field reuse hazards while its
numerical reference passes. The repair publishes locally at CGA1, broadcasts
every decoded field before returning slot credit, and retains the MXFP8 peer
until V-scale tail accesses finish. Earlier positive logs are preserved.

The final source passes all 60 cases with strict golden comparison and 38
existing neighboring-path controls (98 passed, no skips). Directed memcheck,
racecheck and synccheck each pass all 12 family/width/CGA/dtype cases, with
zero sanitizer errors; racecheck also reports zero hazards or warnings.
These sanitizer cases retain reference, rebinding and capture/replay checks;
concurrent work and batch invariance are covered by the full 60-case suite.
All 12 final SM103 lowerings compile with CUDA 13.4. SM103 execution, SM107
and backward remain unverified by this suite.

```bash
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  pytest -q -m L0 \
  sdpa/frost/test_sdpa_fwd_fp8_sm100.py::TestMidwidthPhysicalOrigins \
  sdpa/frost/test_sdpa_fwd_mxfp8_sm100.py::TestMidwidthPhysicalOrigins
```

### Native SM107 D128 FP8 THD origins — 2026-11-11

`test_sdpa_fp8_sm107.py::TestD128Fp8PhysicalOrigins` collects 12 L0 cases:
two shared public-graph regressions, four independent-origin replay cases,
two missing-total-hint cases and four golden cases. Both E4M3/E5M2 inputs
write BF16 O and token/head-major Stats. The expanded cases pin two persistent
clusters, the native SM107 module and the prepared binder, then check references,
independent address/stride rebinding, retained live/empty capture, batch
reordering/duplication/neighbor isolation and raw-byte replay with concurrent work.

All ten offline CUDA 13.4 variants compile: two encodings with token, head,
padded or omitted Stats under physical origins, plus their dense controls.
Native SM107 under actual CuTe DSL CUDA13.4, DSL4.8.0, Torch2.14.1+cu130
and cuDNN9.24.0.43 passes all12 strict cases with zero failures/skips and
four new byte goldens. Functional checks pass before recording; recording
and strict comparison run in separate processes. Directed memcheck4,
racecheck4 and synccheck4 pass with zero errors/hazards/warnings. The native
binder is rebuilt with CUDA13.4.92 and Python3.12.12. Production source is
identical to the tested snapshot; the parent stack's later test-only optional
reference initialization does not change numerical operations.

The two shared regressions fail against the unfixed parent at the Stats
comparison, then pass with this change. The four existing `test_fp8_thd`
neighbors report three passes and one existing causal E5M2 numerical failure
on this SM107 part (`0.043079853 > 0.04`). A same-device parent/candidate
comparison records identical quantized input payloads/scales, O/reference
bytes and Amax for all four cases, including that failure. Its tolerance and
existing device-specific edge marker are unchanged; it is not counted as a
passing case. Other native SM107 widths and backward remain unverified here.

```bash
cd test/python
CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  pytest -q -m L0 sdpa/frost/test_sdpa_fp8_sm107.py::TestD128Fp8PhysicalOrigins
```

#### Refresh after the #2220 develop merge

The integrated native D128 implementation passes12 strict cases plus all four
compact THD neighbors, with zero failures/skips. Its memcheck4, racecheck4
and synccheck4 runs are clean; all ten offline CUDA13.4 variants still compile.
The comparison control contains the same develop schedule with only this
D128 physical-origin change reverted. Its four compact neighbor cases also
pass, with byte-identical inputs/scales, O/reference and Amax.

The incoming #2220 `_E11` schedule normalizes O with the unquantized FP32
row sum instead of the earlier quantized-P ones-MMA denominator. Replaying
the original source reproduces its four old golden files exactly; comparing
against the integrated source preserves input/runtime and Stats hashes while
changing O. Maximum O changes are0.01171875 for E4M3 and0.0234375 for E5M2,
identical for token/head Stats layouts. Independent references, byte replay,
concurrent work and batch invariance pass before recording the four updated
O hashes, then strict comparison passes in another process. No tolerance or
skip was changed. The earlier pre-#2220 E5M2 edge and old golden mismatch
remain historical evidence; neither is a failure on this integrated head.

### Native SM107 D192/D256/D512 FP8 THD origins — 2026-11-11

The selected architecture entry `test_sdpa_fp8_sm107.py` exposes
`TestD192Fp8PhysicalOrigins` and `TestWideFp8PhysicalOrigins`: 30 L0 cases
cover E4M3/E5M2 inputs with BF16 O and token/head-major Stats. The native
D192/V128 CGA2, D256 CGA1 and D512 CTA-pair role-split bodies use independent
Q/K/V/O/Stats physical offsets. Tests pin two persistent clusters, require
the native template and prepared binder, compare independent references,
and check retained live/empty captures, fresh addresses and strides, missing
total hints, batch reorder/duplication/neighbor isolation and concurrent-work
raw-byte replay. Twelve golden cases preserve fixed quantized inputs/scales.

Initial D192 qualification under native SM107, actual CUDA13.4, DSL4.8.0,
Torch2.14.1+cu130 and cuDNN9.24.0.43 passed10 strict cases,8 existing THD
neighbors and four cases under each of memcheck/racecheck/synccheck. Two
old-source controls fail because the prepared plan omits physical origins;
this detector stops before GPU execution, so it is not an old-source numerical
failure. All eight compact controls pass on that parent.

Final D192/D256/D512 qualification, including the Q acquire fold, passes
30 strict new cases plus22 existing compact THD neighbors,0fail/skip. The
parent passes the same22 compact controls; six new-origin controls fail at
the missing-binding assertion before GPU execution. Memcheck12, racecheck12
and synccheck12 are clean with zero errors/hazards/warnings. All30 CUDA13.4
offline variants compile: three widths, two encodings and token/head/padded/
omitted Stats plus dense controls. Twelve new golden files pass a separate
strict process; the four D192 files exactly match the initial qualification.
Other GPUs, datatypes, backward, split, packing and paged routes are outside
this added coverage.

```bash
cd test/python
CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  pytest -q -m L0 \
  sdpa/frost/test_sdpa_fp8_sm107.py::TestD192Fp8PhysicalOrigins \
  sdpa/frost/test_sdpa_fp8_sm107.py::TestWideFp8PhysicalOrigins
```

### Native SM107 D256 MXFP8 THD origins — 2026-11-11

`test_sdpa_fp8_sm107.py::TestD256Mxfp8PhysicalOrigins` exposes10 L0 cases
for E4M3/E5M2 inputs, E8M0 block scales, BF16 O and token/head-major Stats.
CGA1 nonpaged, unsplit, unpacked plans bind independent Q/K/V/O/Stats
origins. The data maps select each physical sequence while scale descriptors
retain packed batch zero and logical per-sequence tile prefixes. Fixed
quantized values and scale tiles survive batch reorder/duplication/isolation,
changed addresses/pitches and retained live/empty CUDA Graph capture. The
suite independently checks numerical references and concurrent-work raw bytes.

Native SM107 with actual CUDA13.4, DSL4.8.0, Torch2.14.1+cu130 and
cuDNN9.24.0.43 passes all10 strict cases,0fail/skip, after four golden cases
pass recording and separate strict processes. The two old-source controls
stop at the missing-origin-binding assertion before GPU execution. All ten
offline variants compile: both formats with token/head/padded/omitted Stats
and dense controls. All24 compact prepared neighbors pass on both parent
and candidate, including changed scale bindings, all four output dtypes,
optional Stats/Amax, SF carriers and no-allocation/no-sync checks.
Memcheck4, racecheck4 and synccheck4 pass with zero errors/hazards/warnings.
Other widths, GPUs, backward, split, packing and paging remain outside this
new suite; their existing admission is unchanged.

```bash
cd test/python
CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  pytest -q -m L0 sdpa/frost/test_sdpa_fp8_sm107.py::TestD256Mxfp8PhysicalOrigins
```

### SM100 half / per-tensor FP8 D512 THD physical origins — 2026-11-11

`test_sdpa_fwd_fp8_sm100.py::TestD512PhysicalOrigins` collects 42 L0 cases:
4 shared `test_mhas_v2` regressions, 12 reference/rebind/replay/batch cases,
6 no-total-hint cases, 8 half head-envelope cases and 12 goldens. FP16/BF16
each run both the role-split and 2x2 half implementations; E4M3/E5M2 run
the FP8 implementation. Every
expanded case asserts the actual kernel template, so a sibling implementation
cannot satisfy its check. All plans retain the existing unsplit/unpacked
contract and cover token/head-major Stats.

Independent Q/K/V/O/Stats origins use mixed INT32/INT64 buffers and element
multipliers. The fixture poisons input gaps with NaNs and checks output
canaries, independent FP64 references, fresh pointer/stride bindings and
retained capture after changing live lengths, origins and scalar scales.
All-empty query work must leave output padding untouched. Exact-byte checks
cover repeats, batch reorder/duplication, an isolated sequence and changed
neighbors. Prepared and standalone graph execution must agree. D512 uses
16-element Q/K multipliers so its +16 head-padding variant still represents
whole physical token rows exactly. Eight additional half cases cover
D384/V448 and D264/V504 in both implementations and dtypes; the latter
uses an 8-element origin unit that divides its declared token stride.

Measured runtime: SM100, 148 SMs; driver 595.71.05; Torch 2.14.1+cu130;
cuDNN 9.24.0.43; DSL 4.8.0/compiler CUDA 12.9; tvm-ffi 0.1.14.post1.
The final suite passes all 42 cases in 52.62 seconds, with zero failures or
skips, including strict checks of twelve new goldens under
`361d6da48de8fd55`. These were recorded after their references and byte
comparisons passed; earlier goldens and tolerances are unchanged. The
paired-host ABI detector passes separately. All 23 neighbor controls pass:
compact D512 FP8 THD, dense output/amax, existing half dense/THD and D576
physical-origin packed/split paths.
The immediate parent fails all three shared D512 dtype cases. The first
expanded candidate exposed seven 2x2 declines caused by an obsolete D576-only
origin compile guard; removing it enables the existing D512 origin body.
All six SM103 lowerings compile offline; execution there remains unverified.

Compute Sanitizer memcheck, racecheck and synccheck each pass eight cases:
the six token-major replay legs and both BF16 D264/V504 implementations.
All report zero errors; racecheck reports zero hazards and warnings. Runtimes
are 6.64, 341.42 and 4.67 seconds respectively. The private launcher initializes
cuda-python before Torch for the procedure-lookup issue documented above;
API reporting remains explicit, with all kernels instrumented and error exit
code 86. No sanitizer diagnostics are suppressed.

```bash
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  pytest -q -m L0 \
  sdpa/frost/test_sdpa_fwd_fp8_sm100.py::TestD512PhysicalOrigins
```

### SM100 D256 paged THD packed split — 2026-11-11

`sdpa/frost/test_sdpa_fwd_dsl_sm100.py::TestPagedD256PackedSplit` is collected
by the existing SM100 CI entry, with all six cases marked L0. Four public graph
cases cover FP16/BF16, CGA2, split2/3/5/9, unpacked and packed heads (including
partial GQA6), HND/NHD/mixed page pools, page8/16/128/256, independent physical
Q/O/Stats offsets and token/head Stats. They compare an independent CPU FP64
reference, preserve poisoned gaps and inputs, rebind fresh storage, and replay
changed data/offsets/tables/lengths through retained CUDA Graphs. Route assertions
pin the D256 native body and native binder.

The two standalone dtype cases hold split5 and packing fixed while comparing
raw O/FP32 Stats bytes across fresh-buffer replay with concurrent work, sequence
alone/batched/reordered/duplicated, changed-neighbor Q and129 requests. Lengths
include zero and fully masked query rows. Independent references check both
outputs and signed infinity sentinels before two small runtime-keyed goldens
are recorded. These tests cover forward only; no backward capability is added.

New measurement: six recording cases pass; the subsequent strict run passes all
50 cases (six public/golden cases,36 native binding cases, eight existing
paged controls), zero failures/skips. Both goldens match unchanged. The four
public origin cases pass memcheck, racecheck and synccheck with zero sanitizer
errors/hazards/warnings. Four SM103 half dtype/packing variants compile offline;
this is not execution evidence. Runtime:
SM100/148SM, driver595.71.05, cuDNN9.24.0.43, Torch2.14.1+cu130,
CuTe DSL4.8.0 with actual compiler CUDA13.4, cuda-bindings13.4.4 and toolkit13.4.92.
The Torch wheel's CUDA13.0 tag is separate from the asserted DSL compiler.
The two JSON records were inspected after reference/replay passed. Strict
comparison must leave `CUDNN_DETERMINISM_RECORD` unset. SM103 execution is
unverified; this evidence does not establish cross-runtime bitwise stability.

```bash
cd test/python
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest -m L0 sdpa/frost/test_sdpa_fwd_dsl_sm100.py::TestPagedD256PackedSplit
```

### SM100 D512 TMEM reader lifetime — 2026-11-12

`TestD512PhysicalOrigins.test_tmem_release_waits_for_all_readers` adds three
L0 cases (BF16, E4M3, E5M2). A temporary template delays one output warp and
records completion of each warp's final TMEM load and the deallocation.
It uses the existing synthetic physical-origin fixture, including reference,
batch-invariance, changed-binding and retained live/empty graph checks. The
assertion compares event order; it has no duration threshold. No production
debug argument is added.

Before the terminal reader join, all three cases fail the lifetime assertion
in 10.96 seconds, although their numerical checks pass. With the join, the
full D512 class passes all 48 cases in 94.30 seconds, with no failures/skips.
This includes the twelve unchanged committed goldens and the existing Stats
reader checks. Runtime: SM100/148SM, Torch 2.14.1+cu130, cuDNN 9.24.0.43,
DSL 4.8.0 with actual compiler CUDA 13.4; all three artifact caches disabled.
The same native extension is reused because no native source changed.

The three new cases also pass Compute Sanitizer 13.4 memcheck and synccheck
(12.46 and 12.29 seconds), with zero errors. The retained four-case selection
passes racecheck 13.2 in 179.45 seconds, with zero hazards/errors/warnings;
kernel compilation remains CUDA 13.4. These are local candidate results.

This establishes and repairs a TMEM lifetime violation. It does not by itself
attribute or clear the separately retained instrumented numerical failure.
SM103 execution remains unverified; no golden or tolerance is changed.

```bash
cd test/python
python -m pytest -q -m L0 \
  sdpa/frost/test_sdpa_fwd_fp8_sm100.py::TestD512PhysicalOrigins
```

### D512 paired-CTA reader completion — 2026-11-12 follow-up

The terminal four-warp join above still allowed an MMA warp to proceed
before its own CTA's readers finished: its barrier counted only the peer
CTA's completion. The role-split half/FP8 paths now send one elected local
credit and one elected peer credit after each four-warp join. Each MMA warp
waits for both reader groups. This retains the existing 2x2 path's
local-plus-peer completion requirement and adds no mainloop synchronization.

The lifetime regression now has nine cases: BF16/E4M3/E5M2 crossed with a
delay in both output CTAs, only the even CTA, or only the odd CTA. It checks
both CTAs' final TMEM loads against each deallocation. On unchanged
`b70a558b4ea2`, the three symmetric cases pass but all six one-sided cases
fail the event-order assertion (28.34s); output numerics still pass there.
With the two-credit protocol, all nine pass (29.28s). A separate synthetic
racecheck control removes only the local credit and its count from the
candidate's temporary template: the even-CTA BF16 case fails the same
ordering assertion, with zero reported hazards. That control is not a
reproduction of the reviewer's numerical failure.

The current-head reviewer still observed numerical/repeatability failures on
`b70a558b4ea2` after the earlier join. Those results remain valid; this new
local evidence does not by itself clear the reviewer's original setup.
No seed, reference tolerance or golden is changed.

The expanded full D512 class passes 54 cases, zero failures/skips, in 101.85s
on the same SM100/148SM runtime, including all 12 unchanged committed
goldens. All artifact caches remain disabled. Four changed SM103 variants
(FP16/BF16 half and E4M3/E5M2 per-tensor FP8) compile offline with CUDA 13.4;
SM103 execution remains unverified. The 2x2 and native binding sources are
unchanged.

CUDA 13.4 memcheck and synccheck each pass all nine lifetime variants with
zero errors (34.04s and 30.95s). The separate racecheck runs pass the
original four-case selection/order (182.94s) and an even-CTA BF16 lifetime
case (56.87s), with zero hazards/errors/warnings. Kernel compilation remains
CUDA 13.4 when using the available 13.2 racecheck tool. These local passes
do not replace requalification in the original reviewer environment.

### SM100 single-sequence physical-origin setup — 2026-11-12 (UTC)

`sdpa/frost/test_sdpa_fwd_dsl_sm100.py::TestSingleSequenceSetup` collects
18 new L0 cases through the existing CI entry. The helper is intentionally
not named `test_*`, so a directory sweep does not collect a second copy.
The four dense D128 physical-origin cases cover FP16/BF16, packed/unpacked
CGA2, independent element offsets, changed addresses/strides, token/head
Stats, retained capture and empty/keyless replay. Four paged B1 cases cover
D64/D128, CGA1/CGA2, HND/NHD/mixed KV layouts and independent Q/O/Stats
origins. Ten FP16/BF16 byte cases cover dense D128 packed/unpacked, paged
D64 CGA1/CGA2 and paged packed D128 CGA2, all with split count one.

Each byte case uses fixed CPU-generated sequence payloads and independent
CPU FP64 O/Stats references. It compares B1 with reordered, duplicated and
changed-neighbor B3/B5/B129 bindings through one capacity-4096 plan, then
returns to B1. Fresh buffers, concurrent GPU work, poisoned workspace,
empty/keyless sequences and retained CUDA Graph replay remain checked.
Paged payloads are copied once into finite-poisoned pools; unused page tails
cannot change the reference. Input bytes, O bytes and FP32 Stats bytes are
checked separately. Negative-infinity Stats is accepted only at keyless rows.
There is no backward setup change or new backward qualification.

On local SM100 (148 SM), actual CUDA 13.4 compiler/sanitizer, DSL4.8,
Torch2.14.1+cu130, cuDNN9.24.0.43, driver API13020 and TVM FFI0.1.14.post1,
the unchanged kernel parent `cb4c4e759873` passed all25 selected L0 cases
(18 new, six existing D64 paged-origin controls, one compact B4 control),
with one L1 case deselected, in68.41s. Ten new runtime goldens were recorded
under `361d6da48de8fd55`; no existing golden was changed. The final candidate
passed the same25 cases, zero failures/skips and one deselection, in70.12s
with recording unset and every new golden checked against that baseline.
Actual imports and native compatibility were asserted; all three compiled
cache controls were disabled. The generic multi-sequence setup body and
existing native source are unchanged.

CUDA13.4 memcheck and synccheck each passed the packed BF16 dense-origin
case and D64 CGA1 paged-origin case, with zero errors (10.93s and8.38s).
Offline compilation passed six SM103 half variants (FP16/BF16 times paged
D64 CGA1, dense D128 CGA2 and packed paged D128 CGA2), plus two SM107 dense
D128 CGA2 controls of the shared helper's unchanged default. GPU execution
on SM103 and new SM107 execution remain unverified; compilation is not a
runtime pass. This setup-only optimization changes neither the attention
arithmetic nor eligibility/ranking, and does not promise cross-plan bytes.

From `test/python`, after checking that `cudnn` imports this checkout:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_FRONTEND_DISABLE_COMPILED_CACHE=1 CUTE_DSL_NO_CACHE=1 CUTE_EXPERIMENTAL_DSL_NO_CACHE=1 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest -q -m L0 \
  sdpa/frost/test_sdpa_fwd_dsl_sm100.py::TestSingleSequenceSetup \
  sdpa/frost/test_sdpa_fwd_dsl_sm100.py::TestPagedD64PhysicalOrigins \
  sdpa/frost/test_sdpa_fwd_thd_origins.py::test_compact_thd_origins_use_single_load_maps
```

The first baseline attempt copied unsupported SM107 configurations and
failed10 cases before candidate testing: dense D64 origins, dense D128 GQA
CGA1 and unsplit packed D64 are outside these SM100 routes. The final suite
uses supported dense D128 and paged D64/D128 routes; reference tolerances
and byte assertions were not weakened. The earlier failed run is not counted
as successful coverage.

### SM100 unified D64/D128 split setup — 2026-11-12 (UTC)

`sdpa/frost/test_sdpa_fwd_dsl_sm100.py::TestUnifiedSplitSetup` adds18 L0
cases through the existing SM100/SM103 CI entry, with a helper that is not
independently collected. Eight B1 paged-origin cases cover D64/D128,
FP16/BF16, packed split5 and unpacked split3, HND/NHD pools and token/head
Stats. Two nonpaged D128 split3 cases cover both half types. Two D128
mixed-KV cases cover FP16/E4M3 paged and BF16/E5M2 nonpaged split3.
They retain independent origins, changed storage/strides, numerical O/Stats
checks and changed-input capture replay, including empty/keyless queries.

Six split5 byte cases cover packed paged D64/D128 and unpacked nonpaged
D128 for FP16/BF16. One declared-capacity256 plan alternates through runtime
B1/B3/B5/B129 and back to B1, including reordered/duplicated sequences,
changed neighbors, concurrent work, independent physical origins, poisoned
workspace and CUDA Graph replay. Fixed CPU-generated payloads and CPU FP64
references remain unchanged from the unsplit fixture. Split count and packing
stay fixed; there is no assertion of cross-plan equality. The appended
`splits=1` helper default preserves the existing unsplit tests and golden names.

On local SM100 (148 SM), actual CUDA13.4 compiler, DSL4.8,
Torch2.14.1+cu130 and cuDNN9.24.0.43, parent `21abdcc482a1` passed all18
cases in50.89s and recorded six new goldens under `361d6da48de8fd55`.
The candidate passed the same18 in50.09s, zero failures/skips, with recording
unset and all six O/Stats hashes checked against the parent. No existing
golden changed; the new input hashes also match the existing unsplit fixtures.
All three compiled cache controls were disabled and source imports asserted.
Earlier candidate qualification on local SM100 (68 SM), with the same compiler
and library versions, passed13 existing physical-origin cases in35.75s and
24 mixed-KV cases in70.21s, including unsplit and multisequence controls.

CUDA13.4 memcheck and synccheck each passed three focused cases in12.43s
and10.83s, with zero errors: packed B1 BF16 D64 paging, B1 FP16/E4M3 D128
paging, and BF16 nonpaged D128 B1/B3/B5/B129 byte comparisons. Offline
SM103 compilation passed ten split-host variants: both half types times
paged D64, paged/nonpaged D128 and paged E4M3/nonpaged E5M2 KV.
This is compilation evidence only; SM103 execution remains unverified.
The change affects setup only and adds no backward qualification.
Four existing unsplit controls also passed in13.60s, retaining their original
goldens. CI-style collection with `-m L0 -k frost` finds all18 new cases.

From `test/python`, after checking the actual imported checkout:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_FRONTEND_DISABLE_COMPILED_CACHE=1 CUTE_DSL_NO_CACHE=1 CUTE_EXPERIMENTAL_DSL_NO_CACHE=1 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest -q -m L0 -k frost \
  sdpa/frost/test_sdpa_fwd_dsl_sm100.py::TestUnifiedSplitSetup
```

The first parent-only attempt failed15 cases before kernel launch: the paged
B1 fixture retained a packed-total hint above `B*S_q`, and the shape-override
byte fixture omitted its required split packed-total bound. The corrected
fixtures declare valid bounds; no numerical tolerance or byte assertion was
relaxed. Those failures are retained separately from passing qualification.

### Native SM107 D64 paged THD split forward — 2026-11-12

`TestNativeD64Paged` in the CI-selected `test_sdpa_fwd_dsl_sm107.py` covers
matching FP16/BF16 paged THD at CGA1/N128,
with fixed split counts and packed or unpacked GQA. The shared physical-D64
single-Q body is compiled for SM107 and writes FP32 partials; the final
combine retains independent O/Stats origins. Mixed FP8 KV, split sinks and
padded Stats retain their existing declines. No backward path is added.

On SM107, cuDNN 9.30.0.2, CuTe DSL 4.8.0 compiled with CUDA 13.4 and
Torch 2.14.0+cu130, 20 distinct focused cases pass: eight split origin/replay
cases, eight existing unsplit controls, one admission case, one partial-GQA
multi-wrap case and two replay/batch/golden cases. The two new goldens also
pass strict comparison in a fresh process. No existing golden is replaced.

The batch cases preserve fixed Q/K/V payloads across sequence insertion,
reordering and duplication, including empty sequences, fresh output/workspace
allocations and concurrent GPU work. Both O and Stats retain identical raw
bytes under the same plan; independent FP64 checks remain separate. The graph
cases check HND/NHD/mixed pools, physical origins and strides, changed-input
capture replay and empty replay. The six-head sharing group exercises partial
packing with more KV iterations than ring slots. The initial three-head-group
fixture was correctly declined by the existing packing contract; the final
case uses a supported group without widening that contract.

```bash
cd test/python
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest -q -m L0 -k frost \
  sdpa/frost/test_sdpa_fwd_dsl_sm107.py::TestNativeD64Paged::test_admission \
  sdpa/frost/test_sdpa_fwd_dsl_sm107.py::TestNativeD64Paged::test_thd \
  sdpa/frost/test_sdpa_fwd_dsl_sm107.py::TestNativeD64Paged::test_thd_split_partial_pack \
  sdpa/frost/test_sdpa_fwd_dsl_sm107.py::TestNativeD64Paged::test_thd_split_determinism
```

These new measurements qualify the SM107 extension only. Existing SM100/103
paged-split support is unchanged and receives no new hardware coverage here.

### SM107 D128 per-tensor FP8 single-CTA dense splits — 2026-11-12 (UTC)

`test_sdpa_fp8_sm107.py::TestNativeD128Fp8Cga1Split` exposes12 L0 cases
through the selected Rubin CI entry: one graph-domain check, six functional
rebind/capture cases, two replay/batch cases, two golden cases and one
refilled-ring stress. E4M3/E5M2 input, BF16/FP16/E4M3/E5M2 output, packed
and unpacked groups, splits2/3/9, dense/causal/bottom-right/SWA/sink,
optional Stats and D112/V96 plus D80/V112 envelopes are exercised.
Every functional check uses the shared FP64 FP8 probability-cast reference
and checks Amax_O separately. Native template, single-CTA geometry,
four-stage KV ring, K64 MMA and FP32 combine are asserted.

On SM107216SMs, driver620.62, fullcuDNN9.30.0.4, CuTeDSL4.8.0 compiled
with CUDA13.4 and Torch2.14.0+cu130, all12 cases passed in25.88s.
The two new goldens also passed strict comparison in a fresh process
(2 cases,3.21s). No previous golden was changed. The stress recorded zero
wrong bytes in512 warm and512 poisoned launches; its unsafe control failed
200/200 launches. The stress shape launches512 blocks and refills its ring
four times per launch; numerical correctness is checked separately first.

Fixed-payload batch checks compare the target alone, reordered, duplicated,
with changed neighbors and among129 sequences. O/Stats are per-sequence;
Amax_O is a whole-batch maximum, checked numerically after rebatching and
bitwise during replay. New byte guarantees cover E4M3/BF16 packed split3
and E5M2/FP16 unpacked split2. Other GPU lines are unmeasured by this suite;
this change adds no backward or THD path.

From `test/python`, with the correct checkout imported:

```bash
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest -q -m L0 -k frost \
  sdpa/frost/test_sdpa_fp8_sm107.py::TestNativeD128Fp8Cga1Split
```

### SM100/103 FP8 coalesced paged loads — validation, 2026-11-12 (UTC)

`sdpa/frost/test_sdpa_fp8_d512_paged_determinism.py` now also covers D64/D128
paging, preserving its existing D512 defaults. Five new configurations each
have a repeatability/batch-invariance test and a strict golden test:

| D | Page | CGA | Split | Layout | Input / output |
| --- | --- | --- | --- | --- | --- |
| 64 | 8 | 1 | 1 | NHD | E4M3 / BF16 |
| 128 | 16 | 1 | 2 | HND | E4M3 / E4M3 |
| 128 | 8 | 2 | 1 | HND | E5M2 / FP16 |
| 64 | 64 | 1 | 4 | NHD | E5M2 / E5M2 |
| 128 | 128 | 1 | 4 | NHD | E4M3 / BF16 |

The suite fixes quantized payloads and scales, uses independent K/V page
orders, and checks O, FP32 Stats and Amax numerically. Replay compares all
three outputs bytewise with fresh buffers and concurrent work. The per-row
O/Stats checks cover isolated, reordered, duplicated and changed-neighbor
sequences plus a 129-sequence batch; Amax is a whole-batch reduction.

Both B200 (SM100, 148 SMs) and B300 SXM6 AC (SM103, 148 SMs) pass
**37 cases, 16 deselected, no failures or skips**: the ten new byte cases,
one refilled-ring stress, twelve existing page-boundary cases extended to
page8, five two-KV-stream paged cases, eight D128 prepared-stride/capture
cases and one existing D512 helper control. Runtime was 100.58s on B200 and
127.37s on B300. The single warning is Torch's synchronization-debug mode.

Five goldens per GPU were recorded from the unchanged parent kernel only
after independent reference/replay checks, then strictly matched by the
candidate in a fresh process with recording disabled. The ten new JSON
records live under `c8aefc970ef3af15` and `98a778ce33bb9402`; no existing
record is replaced. The initial fixture requested an unsupported D64/CGA2
combination and was corrected to the existing CGA1 domain before this run;
no production admission or numerical tolerance was changed.

The stress case traverses eight KV tiles and refills its four-stage ring,
checking 256 warm and 256 shared-memory-poisoned launches. Both GPUs record
zero wrong bytes; their unsafe controls fail 200/200 launches. The tested
kernel grid has only four blocks (0.03 per SM), so this is a low-occupancy
rate bound, not a saturated-scheduling race qualification. An independent
numerical check precedes the stability stress. GPU-exclusive scheduling
markers were added after serial qualification; test bodies are identical.

B200 also passes Compute Sanitizer 2026.3 memcheck and synccheck on the
page8/HND/E4M3 mixed-length case, one case each, with 0 errors. No new racecheck
measurement is claimed. The sanitizer processes report FE cuDNN 93000 while
Torch's cuDNN query reports 92400; this explicit FROST case does not execute
backend attention. The normal qualification uses full cuDNN 9.30.0.4,
Torch 2.14.1+cu130, CuTe DSL 4.8.0 and actual compiler/NVRTC 13.4; driver API 13040.
Source/import hashes were verified against parent
`32d4554e93f2564be0a005d305e458e72551e1c9` plus the loader/test patch.
The relevant kernel/test files are unchanged on subsequent develop
`f013a9d1e`. SM90/SM107/SM120 are unmeasured by this suite. This modifies
forward paging only; no paged-KV backward API is added.

From `test/python`, with the qualified checkout imported and recording unset:

```bash
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest -n0 -m L0 -q -rs \
  sdpa/frost/test_sdpa_fp8_d512_paged_determinism.py \
  -k 'small_paged or paged_loader_warm_stress'
python -m pytest -n0 -m L0 -q \
  sdpa/frost/test_sdpa_fwd_paged_sm100.py::test_paged_graph_fp8_matches_reference \
  sdpa/frost/test_sdpa_fwd_fp8_dual_kv_sm100.py::test_paged_two_kv_streams_match_the_reference
python -m pytest -n0 -m L0 -q \
  sdpa/frost/test_sdpa_prepared_fp8_paged.py::test_prepared_fp8_paged_strides_and_capture -k128
python -m pytest -n0 -m L0 -q \
  'sdpa/frost/test_sdpa_fp8_d512_paged_determinism.py::test_fp8_d512_paged_replay_and_batch_invariance[fwd-d512-e4m3-bf16-split1-NHD]'
```


### SM100 FP8 interleaved two-KV-stream schedule — validation, 2026-11-12 (UTC)

The SMEM-P dual-stream loop issues QK(A), PV(previous A), QK(B), then
PV(previous B). Each stream retains its previous arithmetic order, all waits,
arrivals, credits and ring phases. SM103 uses the unchanged TMEM-P branch.
No eligibility, tuning domain, ranking, or backward path changes.

B200 and B300 SXM6 AC (148 SMs each) both pass **62 cases, 16 deselected,
no failures or skips**. This reuses the preceding coalesced-page qualification
and runs the entire 30-case `test_sdpa_fwd_fp8_dual_kv_sm100.py` suite in place
of its five paged cases. Coverage includes dense and paged KV, odd tails,
empty splits, persistent scheduler waves, deliberately delayed MMA,
independent K/V strides, changed-input capture, replay and batch invariance.
Candidate runtimes were 132.38s on B200 and 95.96s on B300.

Each GPU also passes five parent golden checks (38.64s and 39.82s), recorded
in an isolated directory from the coalesced parent and strictly matched by
the candidate in a fresh process with recording disabled. All ten parent
JSON records are byte-identical to the committed baselines; none is updated.
The existing refilled-ring stress reports zero mismatches in 256 warm and
256 poisoned launches on each GPU, with unsafe controls failing 200/200.
Its four-block geometry remains a limited-occupancy measurement; separate
numerical tests provide the reference checks.

B200 Compute Sanitizer 2026.3 synccheck and racecheck each pass the existing
`page16-unaligned` dual-stream test: two sequences, Q1/H64:8/D128, live KV
lengths 1024/777, eight/seven KV tiles, and repeated ring reuse. Synccheck
reports 0 errors; racecheck reports 0 hazards (0 errors, 0 warnings). Each
sanitizer invocation runs one case, four deselected. The sanitizer processes
report FE cuDNN 93000 and Torch cuDNN 92400; the tested explicit FROST path does
not execute backend attention. Normal tests use full cuDNN 9.30.0.4.

Both GPUs use Torch 2.14.1+cu130, CuTe DSL 4.8.0 and actual compiler/NVRTC 13.4.
Imports and source/library hashes are checked. The runner disables the
editable-install finder and gives child interpreters the same source path.
The shared runner has no pytest-xdist; tests run serially with
`CUDNN_TEST_NO_ISOLATION=1`. Its two warnings are the unavailable
`xdist_group` marker and Torch synchronization-debug mode.
The kernel and test files on landed parent `9be74f4d7` match those qualified
here. After qualification, only the phantom-step explanatory docstrings
changed; the executable kernel AST is identical. SM90/SM107/SM120 are not
measured by this follow-up.

Reproduce the preceding qualification commands, replacing the selected
`test_paged_two_kv_streams_match_the_reference` node with the complete
`sdpa/frost/test_sdpa_fwd_fp8_dual_kv_sm100.py` file. Omit `-n0` when xdist is
not installed. Run the sanitizer on the following case, then repeat with
`--tool racecheck`:

```bash
compute-sanitizer --tool synccheck --error-exitcode 91 --report-api-errors no \
  python -m pytest -m L0 -q \
  'sdpa/frost/test_sdpa_fwd_fp8_dual_kv_sm100.py::test_paged_two_kv_streams_match_the_reference[page16-unaligned]'
```


### SM100 paged FP8 separate Q/K and V producers — validation, 2026-11-12 (UTC)

The existing SM100 D128 paged dual-stream body gives V production to the
O-store warp while the loader produces Q/K. V retains its own ring index and
phase across persistent work items, using the existing MMA empty credits.
Arithmetic, barrier counts, plan knobs and graph eligibility are unchanged.
SM103 retains the existing producer path; no backward path changes.

B200 and B300 SXM6 AC (148 SMs each) pass **43 cases, six deselected, no
failures or skips**. Two D128/CGA1 parent golden cases per GPU are recorded
into a fresh directory and strictly compared by the candidate with recording
disabled. All four parent records equal the committed baselines. Two existing
fixed-plan batch-invariance cases retain O/Stats bytes through reordered,
duplicated, changed-neighbor and multiple-wave batches.

The suite also runs all 30 existing dual-stream cases and eight new capture
cases with an assertion on the actual producer flag (enabled on SM100,
disabled on SM103). The new cases cover HND/NHD, independent V page tables,
packing on/off, split 1/4, scheduling policies 0/1/2, causal bounds, odd tails,
empty splits and persistent work. Each retained graph replays shorter, empty
and restored live lengths 16 times each; restored O and Amax compare raw bytes
with snapshots from the original execution. Independent numerical checks keep
the original quantization bounds. No committed golden or tolerance changes.

The existing four-block refilled-ring stress is included, with 256 warm and
256 poisoned launches and the unsafe control. Both GPUs report zero mismatches
and their unsafe controls fail 200/200. Its four-block geometry remains a
limited-occupancy measurement, not a full-device stress claim.

B200 Compute Sanitizer 2026.3 racecheck, memcheck and synccheck each pass all
eight new capture cases. Racecheck reports zero hazards/errors/warnings;
memcheck and synccheck report zero errors. Racecheck filters
`kns=cudnn_kernel_`; the other tools are unfiltered. CUDA API error reporting
is disabled. These B200 processes report FE cuDNN 93000 and Torch cuDNN 92400;
the explicit FROST attention path does not execute backend attention.
B300 has no new sanitizer result in this follow-up.

Actual compiler/NVRTC 13.4 and CuTe DSL 4.8 are checked separately from the
Torch 2.14.1+cu130 build label. Explicit FROST plans are required; source/import
and library hashes are checked. Other architectures are unmeasured by this
follow-up. The only postqualification kernel change removes experiment
wording from a comment; the executable AST is identical.

Reproduce from `test/python`, with the candidate checking the isolated parent
golden directory through `CUDNN_DETERMINISM_GOLDENS` and recording disabled:

```bash
python -m pytest -m L0 -q sdpa/frost/test_sdpa_fwd_fp8_dual_kv_sm100.py
python -m pytest -m L0 -q -k 'd128 and cga1' \
  sdpa/frost/test_sdpa_fp8_d512_paged_determinism.py::test_fp8_small_paged_golden \
  sdpa/frost/test_sdpa_fp8_d512_paged_determinism.py::test_fp8_small_paged_replay_and_batch_invariance
python -m pytest -m L0 -q \
  sdpa/frost/test_sdpa_fp8_d512_paged_determinism.py::test_fp8_paged_loader_warm_stress
compute-sanitizer --tool memcheck --error-exitcode 91 --report-api-errors no \
  python -m pytest -m L0 -q \
  sdpa/frost/test_sdpa_fwd_fp8_dual_kv_sm100.py::test_paged_split_producer_lengths_replay
```

Repeat the final command with `--tool synccheck`, and with
`--tool racecheck --kernel-name kns=cudnn_kernel_`.

### 2026-11-12 — service-register CI follow-up and FP8 scale lifetime

On parent `e2fbd4f69`, converging the service register releases exposed an
extra output-scale spill in native SM107 D128 FP8. The follow-up computes
that scale in the correction role, retaining the multiply order and the
common service release. Existing spill ceilings and numerical tolerances
are unchanged. No output golden is replaced.

Native SM107, CUDA compiler/NVRTC 13.4, CuTeDSL 4.8.0, cuDNN 9.30.0.4:
the unchanged parent records four E4M3/E5M2 token/head-Stats origin goldens
after their references and replay checks pass (**4 passed, 36 deselected**).
The candidate then passes **40 cases, no skips**, including those four strict
parent records, all D128 origin checks, D64/D128 masks, D64 GPT-OSS shapes
and the D128 output-dtype cases. O/Stats batch invariance and changed-storage
replay retain their existing scope; Amax_O is checked numerically by the
dense cases. Backward, new sanitizer coverage and performance are unmeasured
by this follow-up.

Run the same selected L0 suites with `CUBLAS_WORKSPACE_CONFIG=:4096:8`:

```bash
python -m pytest -m L0 -q \
  sdpa/frost/test_sdpa_fp8_sm107.py::TestD128Fp8PhysicalOrigins \
  sdpa/frost/test_sdpa_fwd_fp8_sm100.py::test_fp8_masks \
  sdpa/frost/test_sdpa_fwd_fp8_sm100.py::test_fp8_d64_masks \
  sdpa/frost/test_sdpa_fwd_fp8_sm100.py::test_fp8_d64_gptoss_shape \
  sdpa/frost/test_sdpa_fwd_fp8_sm100.py::test_fp8_output_dtypes
```

For the parent phase add `-k origin_golden` and record into a fresh directory.
The candidate uses that directory through `CUDNN_DETERMINISM_GOLDENS`, with
recording disabled. Both phases load Python and matching native bindings
from the isolated source package; the reused bindings have identical native
build inputs to this source.

B200 with CUDA compiler/NVRTC 13.4 and CuTeDSL 4.8.0 passes **61 cases, no
skips**: the complete dual-KV suite (including the delayed-MMA subprocess),
six D576 SASS cases, the THD/paged gate cubin records, both masked-softmax
pins, all four epilogue/scheduler pins, six score-off records, both score-on
store checks and the SM107 D128 service-release guard. Cross-compiling
SM103/SM107 instructions on B200 is not native execution on those targets.
The fresh child receives no inherited test-root path; an isolated CPU
control fails its import before the fix and passes afterward.

All 19 old cubin records are reproduced with the pre-convergence kernel
sources and the same DSL 4.8.0/CUDA 13.4 compiler. Sixteen change with the
common-release fix and are refreshed; the three SM107 score-off records stay
byte-identical. PTX and SASS are retained for attribution, and the existing
spill and score-store detectors remain active. CUDA 13.5 disassembles the
SM107 cubins. Native SM103 execution is unverified by this follow-up.

## SM100/SM103 D64 packed THD prefill (2026-11-13)

The new L0 `TestD64ThdPackedUnsplit` entry checks exact D64 with whole-group
PackGQA, N128, CGA1/2 and SPLIT_KV=1. Both graph and standalone calls assert
the direct native prefill body, native binding, independent origin metadata
and absent split workspace. Coverage includes FP16/BF16, HND/NHD/mixed pools,
GQA2/4/8/16, pages8/16/64/256, short-query routing, window/sink/Stats handling,
fresh strides, changed device lengths/tables, empty replay and tail canaries.
The graph batch tests hold sequence payloads fixed across reorder, duplication,
changed neighbors and B1/B3/B5/B129 under one capacity4096 plan. The refill
checks validate both cold O/Stats baselines with CPU FP64 before stressing
the four-stage KV ring with exact byte comparisons.
No backward kernel is changed or qualified by these tests.

```bash
cd test/python
python -m pytest sdpa/frost/test_sdpa_fwd_dsl_sm100.py::TestD64ThdPackedUnsplit \
  -o addopts= -m L0 -q -rs
```

B200 (148 SMs) and B300 (148 SMs), with CuTeDSL4.8.0, compiler/NVRTC13.4,
FE backend9.30.0 and Torch2.14.1+cu130, each pass all **22 unique cases**:
16 contract/functional cases, four bytewise batch/golden cases and two warm
refill cases. Each run additionally passes four strict golden rechecks and
two cases under each of memcheck, synccheck and racecheck: **32 successful
test executions per GPU**, no failures or skips. Sanitizers report zero
errors/hazards. Complete allocations took 2m52s and 2m53s respectively.
Four new golden records per GPU are committed; existing records are unchanged.
Both precisions pass256 warm and256 poisoned launches; the unsafe ring
control fails200/200. The actual persistent grid has one CTA per SM, below
the control's three-CTA geometry, so the rate bound is specific to that grid.
Forty-two existing host config/capability regressions and eight new contract
checks also pass locally. Changed-file pre-commit and fern check pass.

The original qualification snapshot's production and test bytes match
`e4e1e46c53567673b1c728e388e890f6f0e04083`, based on
`fed09b0fa691273435e993676f4bfb28c28a0461`. The review follow-up adds the FP64
checks to the two actual cold launches; it changes no production code or
golden bytes. Its two updated stress cases pass on each B200 and B300, with
no skips, using the same runtime and unchanged 256 warm/256 poisoned counts.
The qualified helper SHA256 is `ceef49f454fc3ff386680d00eb728b78a862e7b3b0320ab149c688ea9713847f`.
The phase commands are below (from `test/python`, using the runtime above). The recording test writes only after its numerical and
bytewise batch checks pass. Strict rechecks always disable recording.

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDNN_TEST_NO_ISOLATION=1
entry=sdpa/frost/test_sdpa_fwd_dsl_sm100.py::TestD64ThdPackedUnsplit
golden_dir=$(mktemp -d)
env -u CUDNN_DETERMINISM_RECORD -u CUDNN_DETERMINISM_GOLDENS \
  python -m pytest "$entry" -o addopts= -m L0 -q -rs -x --tb=short \
  -k 'not batch_and_golden and not warm_refill'
env -u CUDNN_DETERMINISM_GOLDENS CUDNN_DETERMINISM_RECORD="$golden_dir" \
  python -m pytest "$entry" -o addopts= -m L0 -q -rs -x --tb=short -k batch_and_golden
env -u CUDNN_DETERMINISM_RECORD CUDNN_DETERMINISM_GOLDENS="$golden_dir" \
  python -m pytest "$entry" -o addopts= -m L0 -q -rs -x --tb=short -k batch_and_golden
env -u CUDNN_DETERMINISM_RECORD -u CUDNN_DETERMINISM_GOLDENS \
  python -m pytest "$entry" -o addopts= -m L0 -q -rs -x --tb=short -k warm_refill
for sanitizer in memcheck synccheck racecheck; do
  env -u CUDNN_DETERMINISM_RECORD -u CUDNN_DETERMINISM_GOLDENS \
    compute-sanitizer --tool "$sanitizer" --error-exitcode 91 --report-api-errors no \
    python -m pytest "$entry" -o addopts= -m L0 -q -rs -x --tb=short \
    -k 'test_paged_origins and (g2 or g4)'
done
```

The retained development smoke passed two N128 paged origin/replay cases;
it also exposed the existing N64 dense-only factory boundary, which is now
rejected by capability and standalone checks before compilation. An earlier
config admission failure and a synthetic fixture missing its padding flag
are retained as development failures; neither is a successful GPU test.

### Linear attention cuTile launch autotune: fixed-configuration replay and host selection (2026-11-16)

The cuTile GDN and KDA engines tune each kernel's launch hints, `occupancy`
in {1, 2, 3, 4} times `num_worker_warps` in {4, 8}, by timing. The worker-warp
count selects the cross-warp summation tree of every kernel that reduces a
tile with `ct.sum`, so two processes that time their own candidates can write
different bytes. The claim qualified here is narrower than candidate
agreement: **a fixed launch configuration replays its own bytes**, and a host
can fix the configuration through two hooks in
`cutile/kernels/common.py`. `select_hints(kernel, cache_key, configs)` returns
the candidate to launch and skips the timing; `record_hint(kernel, cache_key,
winner)` is told each timed winner. The hooks are read only when a launch
misses the in-process hint cache. A failed or timed-out search launches the
first candidate (`occupancy=1`, `num_worker_warps=4`) and records nothing.
Independently timed candidates, batch invariance and cross-architecture
equality are not claimed.

`test_determinism_cutile_autotune_configs` in `linear_attention/test_la.py`
replaces the timed search with a stand-in that forces candidate *i* on every
launch, runs the forward and backward twice per *i* for every *i* the sweep
has, and compares output, final state and every tensor-input gradient
bitwise; a failing stand-in covers the tuning-failure fallback. A candidate
the compiler rejects is excluded for that kernel and reported.
`test_determinism_cutile_autotune_host_hooks` runs a timed round under
`record_hint`, asserts that every recorded key holds only ints and strings
(no device id, so one table serves every rank) and that the recorded winner
is the first candidate, then asserts that a `record_hint` that raises reaches
the caller. It then pins every launch to the second candidate through
`select_hints` with the timed search replaced by a mock that fails if called,
and compares the pinned bytes with a repeat and with the second candidate
forced directly. Both tests run for GDN and KDA on a ragged batch of six
sequences.

One cuTile sweep stays outside the hooks: `kda.autotuned_launch_bv` times the
V-tile width `BV` (16, 8, 32, 64) of the KDA state kernel
`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`. `BV` is a compile-time
constant that partitions the V axis across CTAs; the K axis is reduced in
fixed 64-wide blocks whatever `BV` is, so a different width must not move a
bit, and then two processes that time it differently still agree.
`test_determinism_cutile_kda_bv_candidates_agree` checks that claim: with
every hooked launch held at its first candidate it forces each `BV` in turn
and compares output, final state and every gradient bitwise against `BV`
candidate 0 (not only against a repeat). On GB300 (job 1995834, source as
below plus this test) the four widths agree byte for byte, so the sweep
needs no host pin. From `test/python`:

```bash
python -m pytest linear_attention/test_la.py -k "autotune or determinism" -rA
```

The two stream-contention tests, `test_determinism_two_streams` (forward,
every variant) and `test_bwd_same_graph_two_streams` (backward, now every
variant, fp16 and bf16; it covered KDA and mamba2 before), were FROST-only
and now run on the cuTile engine as well: two instances on two side streams, each repeat compared bitwise with
its own single-stream baseline, with the serial baselines warming the hint
cache so the concurrent repeats never tune. The first run of the backward
case on cuTile KDA found a stream defect: `chunk_kda_grad` launched its five
gradient casts without a stream, so they ran on the legacy default stream
while the rest of the backward ran on the caller's, and on a non-blocking
torch stream the first backward after another stream had tuned returned
`dq` and `dk` as zeros (GB300 job 1990036; a probe that logged every launch
saw stream 0 beside the side stream, and a device sync after every launch
made the zeros disappear). The casts now take the caller's stream; develop
has the same defect. GDN's backward does not.

Source `a06ec348` ran on one GB300 (cc 10.3, 152 SMs) with cuDNN backend
9.26.0, cuDNN frontend 1.33.0 from the checkout, CuTe DSL 4.8.0, CUDA 13.4
and driver API 13040, Torch 2.14.0a0 (NGC 26.09), selecting
`-k "autotune or determinism or two_streams"`:
**65 passed, 93 skipped, 0 failed.** The passes are `test_determinism_fwd`,
`_bwd`, `_multi_tile_fwd` and `_multi_tile_bwd` (7 each), `_bwd_all_grads`
(11), `test_determinism_two_streams` (7: five FROST variants, cuTile GDN and
KDA), `test_bwd_same_graph_two_streams` (14: the five FROST variants and cuTile
GDN and KDA, two dtypes each), `test_beta_guard_bwd_determinism` (1), plus the two
autotune tests for GDN and KDA (2 each); `test_determinism_cutile_kda_bv_candidates_agree`
(1) was added after that run and passed in job 1995834 (5 passed with the
four autotune cases, 1:49). The skips are the `hopper` and
`hopper_cuda` rows and cuTile's gdn2, gdp and mamba2 rows, which have no
plan on this GPU; `test_multi_graph_stress` is not selected. After the cast
fix the launch log of the two-stream backward shows every cuTile launch and
memset on the caller's stream. Forcing each candidate on this GPU showed
that occupancy never moves a bit and that `num_worker_warps=8` moves the
bits of the kkt solve, dqkwg and prepare_wy_repr_bwd kernels, which is why
the host pins the pick instead of relying on the timing.

CI pipeline 82134950 ran the same two autotune tests at source `f080e1fe`
(identical kernel and host code) on SM80, SM90 and SM100 with cuDNN 9.28.1
and CuTe DSL 4.8.0: **12 of 12 cases passed, no skips**. SM107 was not
run.

Performance was measured on the same GB300 against a develop worktree
(`9bdf9a8a`) with the same compiled binding: cuTile GDN and KDA forward plus
backward, bf16, 16 heads, K = V = 128, median of 40 iterations after warmup,
each mode in a fresh plan and launch-hint cache. The timed sweep is develop's
and costs the same: GDN 0.968 ms on develop and 0.985 ms here on the ragged
six-sequence batch, 2.160 and 2.158 ms at 2 x 4096 tokens; KDA 1.243 and
1.247 ms, 3.340 and 3.344 ms. A host pin of the recorded winners runs the
same kernels at the same time (GDN 0.987 and 2.174 ms, KDA 1.270 and
3.338 ms) and cuts the first-call warmup from 24 s to 4 s because nothing is
timed. A pin of the first candidate, the table-miss fallback, costs GDN 6%
on the ragged batch and 28% at 2 x 4096 (2.775 ms), KDA 1% and 4%, which
is why a host records a table before pinning. The per-call host overhead on
the cache-hit path is unchanged: 200 us per GDN forward and 242 us per KDA
forward on a 64-token shape, develop and this tree alike.

End to end, the Megatron-Bridge recipe
`qwen35_text_9b_pretrain_8gpu_gb200_bf16_deterministic_config` (Qwen3.5-9B,
Megatron-Core deterministic mode, `gdn_kernel_backend=transformer_engine`,
2 nodes x 4 GB300, mock data, 20 steps) ran three times with this tree's
`python/cudnn` mounted over the container's cudnn 1.29.0 package, on the
nemo 26.10.rc5 image: **the three arms are identical on every logged step
(loss, grad norm, num zeros), 3.66 s per iteration**. On this GPU and head
dim 128 cuDNN serves the FROST GDN engine, which this revision does not
change, so this run is a regression check of the tree in the training
container, not a cuTile exercise; TE forwards no plan name, so the cuTile
engine cannot be selected from the recipe on GB300.

QSA is not changed by this revision; its deterministic backward is develop's
(#2417). For the record on the same GB300 and source `20493fe8` (the cuTile
cast fix touches no QSA file), the QSA
L0/L1 determinism and replay selection, goldens deselected, gives
**75 passed, 9 skipped, 0 failed**. The skips are the batch-invariance
cases waived under PR #2266 (batch layout changes the saved FP32 output or the
gradients); no replay case is skipped.
From `test/python`:

```bash
python -m pytest QSA/cutedsl/test_qsa.py QSA/cutedsl/test_qsa_backward_sm100.py \
  QSA/cutedsl/test_qsa_indexed_training.py QSA/cutedsl/test_qsa_determinism.py \
  -m "L0 or L1" -k "(determinism or deterministic or replay) and not golden" -rA
```

## 2026-11-10 — JAX grouped SiTU integration

Added `gemm/jax/test_grouped_gemm_situ_determinism.py`: eight L0 cases covering
forward/backward, 256/1024/32768 active rows, fresh buffers, concurrent work,
empty experts, poisoned capacity tails, expert packing/reordering/duplication
and changed neighbors. Active data and scale outputs use raw-byte comparisons.
Atomic dprob is excluded from bitwise claims; canonical parity checks it
against the Torch result numerically. General dprob determinism is not claimed.

New GB200 SM100 measurements: native cuDNN builds and imports passed;
73 canonical parity/validation cases passed, two new golden recordings passed,
and all eight strict replay/invariance/golden cases passed (zero skips).

Runtime: JAX 0.12.0.dev20261014, Torch 2.16.0.dev20261030+cu130, CUDA toolkit
13.5, CuTeDSL 4.8.0/compiler CUDA13.4; cuDNN backend 9.26 (Torch reports 9.27).
New goldens are under runtime key `ec5f7cd6b51a0ce3`; no existing golden changed.
They were recorded into a new candidate directory after numerical parity,
then checked in a separate run with recording disabled. FP8 E5M2, SM103,
cu12 and other runtime combinations remain unverified.

Commands (from `test/python`, with `CUDA_VISIBLE_DEVICES=0` and
`XLA_PYTHON_CLIENT_PREALLOCATE=false`):

```bash
python -m pytest gemm/jax/test_grouped_gemm_canonical_jax.py -m L0 -q -x \
  -k 'canonical_jax_glu_parity or canonical_jax_dglu_parity or situ or jax_glu_rejects_unsupported_options or jax_dglu_rejects_unsupported_options'
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest gemm/jax/test_grouped_gemm_situ_determinism.py -m L0 -q -x
```

### 2026-11-13 follow-up

Tested source revision: `f9619cedc5d8631654ee789d01d1eb8e1fdd04cb`.
All 85 selected canonical cases and eight strict replay/invariance/golden
cases passed, with zero failures or skips. The canonical total includes four
independent Torch-reference cases and eight changed-data/routing cases.
Both small golden cases also check their actual inputs and outputs against
the shared numerical reference before checking or recording hashes.

Runtime: GB200 SM100 (152 SMs), driver CUDA API 13050,
JAX 0.12.0.dev20261014, Torch 2.16.0.dev20261030+cu130 (CUDA 13.0),
cuDNN 9.26, CuTeDSL 4.8.0/compiler CUDA 13.4 and apache-tvm-ffi 0.1.15.post2.
Use the commands above with `CUBLAS_WORKSPACE_CONFIG=:4096:8` and
`CUDNN_DETERMINISM_RECORD` unset. Both golden JSONs are byte-identical to
`5593a9f3816b918896f5ba526c0097ccdffa7f75`; no baseline was regenerated.

Two earlier reference-harness failures involved dprob singleton dimensions
and inputs beyond the active row range. These were fixed without changing
numerical tolerances or golden bytes; the complete suites then passed.

Atomic dprob remains outside bytewise/golden guarantees. SM103, E5M2 and other
runtime combinations remain unverified. No shared-memory kernel body changed;
dedicated warm-stress qualification of the underlying GEMM is not claimed.

### 2026-11-13 CI-environment reference correction

Tested source revision: `97993e6879182a6693f9307c9b8bf96a02f06447`.
The original CI runtime reproduced 124 failing raw FP8 elements in the
256-row backward math-reference check. Replay and capacity isolation passed.
Torch's default `high` FP32 matmul mode permitted TF32 in the independent
reference GEMMs. Four reference scale blocks crossed an MXFP8 exponent
boundary: raw -224 versus -448 used scales 1/256 versus 1/512, respectively,
and both dequantized to -87.5 with normalization 0.01. With the same native
output, `highest` reference precision removed all 124 mismatches and all
four scale mismatches. Relative L2 error against the quantized reference
after dequantization fell from 0.005990 to 0.000127.

The helper now uses `highest` only for reference GEMMs and restores the
caller's precision in `finally`. Kernel code, numerical tolerances and
previous golden records are unchanged. Explicit `high` and `highest`
small-case coverage increases the dedicated L0 suite from eight to ten
cases and checks precision restoration.

GB200 SM100 (152 SMs), driver CUDA API 13040, Torch
2.15.0a0+875d815502.nv26.10 (CUDA 13.4), JAX 0.12.0.1, cuDNN 9.28.1,
CuTeDSL 4.8.0/compiler CUDA 13.4 and apache-tvm-ffi 0.1.16:
two candidate recordings passed, ten strict dedicated cases passed with
recording disabled, and all 335 full JAX L0 cases passed, with two pre-existing packed-FP4 xfails
and zero skips. The original CI-built native ARM
extension was used. All six implementation/test source hashes matched.
The tests retained the image's default `high` matmul mode outside the
reference guard.

The two new records under `7685f6b6976b350d` cover this CI runtime.
They were recorded into a new candidate directory after the independent
reference passed, then inspected before a separate strict check. Every
input/output hash, shape and dtype equals the corresponding original
`ec5f7cd6b51a0ce3` record; only runtime metadata differs. The old records
remain untouched.

On a second SM100 device (148 SMs), the same Torch/CUDA/JAX/cuDNN/DSL
versions and apache-tvm-ffi 0.1.15.post2 passed two candidate recordings,
ten strict dedicated cases and 54 selected canonical SiTU cases. This
run used a native extension built from the matching source. Its private
candidate records are not included. Both devices ran the dedicated suite
with recording disabled after candidate generation.

Commands (from `test/python`, with `CUDA_VISIBLE_DEVICES=0`,
`XLA_PYTHON_CLIENT_PREALLOCATE=false` and `CUDNN_DETERMINISM_RECORD` unset):

```bash
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest gemm/jax/test_grouped_gemm_situ_determinism.py -m L0 -q -x
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest gemm/jax -m L0 -q -x
python -m pytest gemm/jax/test_grouped_gemm_canonical_jax.py -m L0 -q -x -k situ
```

This corrects the test reference; it does not establish broader kernel
coverage. SM103, E5M2 and atomic dprob determinism remain unverified.


### SM100/SM103 D256 engine isolation pilot — 2026-11-14

The exact half D256 graph domain now has explicit lookahead and swap-AB
entries. Kernel bodies/arithmetic are unchanged; no golden is added or changed
and no general determinism/batch-invariance status is upgraded.

On both B200 (SM100) and B300 (SM103), CuTeDSL 4.8 compiling with CUDA 13.4,
cuDNN 9.30.0 and Torch 2.14.1+cu130, the selection below passed **59 cases**
with **4 skips** (nvdisasm unavailable) per GPU. It includes twelve explicit
FP16/BF16 dense/paged cases selecting N16, N32 or lookahead128, exporting the
public record into a fresh graph, and checking changed-V CUDA Graph replay
and exact unchanged Stats. These checks do not establish general batch
invariance. Backward was not changed or requalified.

```bash
cd test/python
pytest -q -x -rs -m 'L0 or L1' \
  sdpa/frost/test_sdpa_fwd_d256_implementation_contract.py \
  sdpa/frost/test_sdpa_fwd_decode_d256_sm100.py \
  sdpa/frost/test_sdpa_fwd_d256_fused_correction.py \
  sdpa/frost/test_sdpa_prepared_thd.py \
  sdpa/frost/test_sdpa_fwd_thd_origins.py \
  -k 'explicit_d256 or d256_nonpaged or fused or implementation_contract or routing_boundary or wide_tile or compact_thd_origins'
```

A separate B200 migration run passed 148 checks, with 3 skips: the existing
D256 paged packed-split float16/bfloat16 goldens were absent for this exact
runtime, and one case required SM107. Those golden comparisons remain
**unverified**, not passes. Five B200 BF16 forced-FROST parent/candidate
comparisons preserved input/output hashes, actual body/config and workspace
against parent `ae3d4e4425c563a159af2acb5f873a5a0dcc6022`; independent FP64
references and changed-input replay also passed. This bounded comparison is
not a replacement for the full batch-invariance or golden contract.

### SM100/SM103 wide half explicit-engine migration — 2026-11-14

The D512 role-split and D512/D576 2x2 implementations now have explicit graph
owners. The existing physical-origin suite pins each owner and its physical
cluster width, retaining the original numerical, raw-byte, batch and retained
capture checks. The kernel algorithms and golden files are unchanged.

On B200/SM100 and B300/SM103, CUDA 13.4 compiler, CuTe DSL 4.8.0,
cuDNN 9.30 and Torch 2.14.1+cu130, all eight cases in
`sdpa/frost/test_sdpa_fwd_d512_explicit_engines.py` pass: FP16/BF16,
role-split unsplit/split4, D512 2x2 and D576 2x2. They export the public
record, rebuild a fresh graph, retain the intended body despite the legacy
A/B switch, and check changed-input CUDA Graph replay against a reference.
These cases do not establish general batch invariance.

An additional B300 run passes all eight half cases in
`sdpa/frost/sdpa_fwd_thd_origins_d512_sm100.py::test_independent_origins_rebind_replay`
(`-m L0 -k 'fp16 or bf16'`). Both implementations and token/head-major Stats
retain O/Stats bytes through replay, reordered/duplicated batches, an isolated
sequence and changed neighbors, with independent numerical checks and physical
origin changes. Production source is `615798010c6378b22c92c141614e2ec01d24d67a`;
the follow-up fixture changes only migrate other test owners and the split
capability inventory. FP8 on SM103, strict runtime-golden comparisons, other
architectures and backward are not qualified by this migration. No support
status is upgraded from these bounded identity/replay checks.


## 2026-11-14 — SM120 dense D128 additive-bias integration (#2193)

The integration retains the paged prepared host and append-only API argument
order. It qualifies `SdpaFwdDslSm120` forward only: FP16/BF16 D128, FP32 or
matching-input bias, unpaged/unpacked/unsplit, without physical origins.
Bias backward and SM121 hardware were not qualified.

`test_frost_sdpa_bias_sm120_determinism.py` adds eight L0 cases: both input
dtypes, both bias dtypes, M64 dense and M128 causal. CPU-seeded B3/H64/Hkv16
inputs use Q129/K257 tails, padded bias storage, and an all-minus-infinity
row. Tests compare O and FP32 LSE bytes over fresh-buffer/concurrent replays,
isolated/reordered/duplicated batches, changed neighbors and captured replay,
separately check an FP64 numerical reference, and retain runtime-keyed goldens.
The golden case name also includes the DSL compiler version; the common
runtime key alone does not distinguish CUDA 12 and CUDA 13 DSL wheels.

`test_frost_sdpa_bias_sm120_stress.py` adds two L0 cases, one per input dtype,
with FP32 bias, B3/H64/Hkv16/Q257/K513/D128, M128/N128, 576 thread blocks and
four K/V refills per launch. Each case independently checks its FP64 reference,
then 3,000 warm and 3,000 poisoned launches against a cold byte reference.
The unsafe control failed 200/200 launches on this device. Cold agreement is
only a stability check, separate from the FP64 reference.

On the initial f0a0a1b1c43f integration, ordinary bias/contract/analyzer/geometry
checks passed 1,575 cases; paged/prepared/origin controls passed 271 with 12
missing-runtime-golden skips and eight deselections. Those skips supply no
strict golden qualification. The eight new replay/batch checks passed, but
the new stress failed on warm launch 13 (FP16) and 55 (BF16). The true develop
f0a0a1b1c43f no-bias control with the same Q/K/V passed both stress cases.
Enabling the existing result-dependent consumption stores for dense bias
retains K/V use before the named reuse barriers, just as for paged half-storage.

After that bounded fix, native RTX 6000D (cc 12.0, 156 SMs), driver API 13040,
Torch 2.14.1+cu130, cuDNN 9.27.0 and CuTe DSL 4.8.0 produced:

| DSL compiler | Scope | Result |
|---|---|---|
| CUDA 13.4 | bias + eight strict replay/batch goldens + API contract + paged native rebind | 88 passed, zero skips |
| CUDA 13.4 | two warm/poisoned stress cases | 2 passed; zero byte mismatches |
| CUDA 12.9 | bias + API contract | 72 passed, zero skips |
| CUDA 12.9 | eight new golden candidates, then a separate strict replay/batch run | 8 passed + 8 passed, zero skips |
| CUDA 12.9 | two warm/poisoned stress cases | 2 passed; zero byte mismatches |

The eight CUDA 13.4 candidates recorded before the fix also passed strict
checks after it; no existing golden was replaced. All sixteen reviewed new
JSON files are committed under runtime key `0c261da063500aab`. Fixed-runtime
identity is asserted, not cross-runtime identity. Source/native-extension and
loaded-template hashes were checked for each run. The CUDA 13.4 cache objects
for all 32 bias renderings exercised by the regression contain the consumption
store before each of 204 K/V `BAR.ARV` sites, without an intervening branch.

Reproduce from `test/python` with the desired installed DSL compiler flavor:

```bash
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
pytest -o addopts= -m L0 sdpa/frost/test_frost_sdpa_bias_sm120_determinism.py \
  sdpa/frost/test_frost_sdpa_bias_sm120_stress.py
```

The ordinary numerical and integration controls are
`test_frost_sdpa_bias_sm120.py`, `test_sdpa_fwd_api_contract.py`,
`test_sdpa_graph_analyzer.py`, `test_heuristic_geometry.py` and
`test_sdpa_fwd_paged_sm120.py::test_sm120_frost_paged_native_rebind_replay`.

## SM90 explicit D64 forward owners (2026-11-14)

This implementation-isolation batch moves exact D64/V64 graph records to
`sdpa_fwd_sm90_d64_one_wg` (20539, native nonpaged body) and
`sdpa_fwd_sm90_f16_three_wg` (20540, existing paged body). Wider three-WG
shapes and dedicated D128 are outside this batch. Backward is unchanged.
The base is `5913f355354bd35d5be54494b91b1227b01917bb`.

New native evidence uses H100 NVL (cc 9.0, 132 SMs), Python 3.12,
Torch 2.14.1+cu130, cuDNN 9.30.0 and CuTe DSL 4.8.0 compiling CUDA 13.4.
The import path was asserted to be the isolated source checkout. The initial run reused a native extension, but a later capability check found
that it lacked the claim-counter property present in current C++ source.
A matching extension was then built from the frozen source with cuDNN 9.30
headers; `supports_claim_counter=True` was verified. The final native result
below uses that fresh extension.

`test_sdpa_fwd_sm90_implementation_determinism.py` adds 16 L0 cases:

- Twelve FP16/BF16 x unmasked/top-left-causal x three-scheduler cases check
  O and FP32 Stats bytes across four replays, alone/reordered/duplicated
  sequences and changed neighbors. The B3/Hq4/Hkv1/Q65/K257 inputs cross
  Q/K tile boundaries; FP64 attention provides a separate numerical reference.
- Two dtype cases use B8/Hq16/Hkv1/Q129/K257, 384 CTAs (2.91 per SM), four
  K/V stage refills, and actual physical output storage. Each passed 3,000
  warm and 3,000 poisoned launches without a byte mismatch. The unsafe ring
  control failed 200/200 launches. This bounds the observed per-launch error
  rate below 0.0998% at 95% confidence for each tested arm; it is not a
  hardware-refinement proof. FP64 reference checks precede the helper; the
  helper's cold byte reference measures launch stability.
- Two runtime-golden cases passed their replay/reference preconditions but
  skipped for missing baselines. No golden was generated or replaced.

The new public-contract suite separately checks explicit record rebuilds,
fixed body selection with legacy selection poisoned, captured changed input,
and the retired 20517 exact-D64 recipe rejection. Existing paged tests retain
independent table pitches, refreshed storage, padded O/Stats sentinels and
capture checks. They do not establish new batch-invariance claims for the
paged owner. Only the native nonpaged D64 row is added to the support page;
other GPUs are outside these SM90 owners' architecture contract.

Reproduce on Hopper with the stated dependencies, the source import verified,
and the matching cuDNN library loaded before Torch:

```bash
cd test/python
CUDNN_TEST_NO_ISOLATION=1 CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  pytest -o addopts= -m L0 -q -rs \
  sdpa/frost/test_sdpa_fwd_sm90_implementation_determinism.py \
  sdpa/frost/test_sdpa_fwd_sm90_implementation_contract.py
```

The initial combined native D64/paged/determinism run gave 142 passed,
8 missing-runtime-golden skips and 14 cross-compile deselections. The skips
were six existing paged baselines plus the two new D64 baselines. A later
identity-fix run gave 69 passed, 2 missing-runtime-golden skips and 6 existing
paged-golden deselections. These runs overlap and are not additive coverage.
Two negative compiled-parameter identity tests first failed with shared
legacy records and passed after checking implementation-local record types.
The final 3,000-launch stress result is newer than those initial 256-launch
runs. No missing golden is counted as a strict-golden pass.

With the freshly built native extension, source `6985f58c1` passed **85 tests
with 2 missing-runtime-golden skips** in the new contract/determinism suites
and the complete `test_sdpa_prepared_sm90.py` suite. This includes the D128
claim-counter control and all fresh-process artifact reloads. An earlier
frozen-source sweep on the reused binary gave **2129 passed, 10 skipped,
15 deselected**; its skips were eight missing goldens and two SM100-only cases,
and the deselections were 14 cross-compile cases plus the then-unqualified
claim-counter control. That control is qualified by the fresh-extension run.

Six omitted-scheduler host controls first gave four failures and two passes:
causal plans had been incorrectly assigned NATURAL. Both new hosts now resolve
an omitted scheduler through the unchanged adapter rule. All six controls,
plus native/paged public-record rebuilds with omitted knobs, pass. Ranked
candidate schedules and thresholds are unchanged.


### SM90 wide and dedicated D128 implementation isolation — 2026-11-14

H100 NVL, cc9.0, 132 SMs; Torch2.14.1+cu130, cuDNN9.30, CuTe DSL4.8.0
with CUDA13.4 compiler, driver API13020. The extension was rebuilt from the
matching source and cuDNN9.30 headers; `supports_claim_counter` is present.
The device code is unchanged: all 13 dedicated-D128 top-level function/class
ASTs match the retained original module, including its pipeline and scheduler.

The combined fixed-owner suite and repaired fixture controls gave **110 passed
and 12 missing-runtime-golden skips**. This comprises 92 contract cases, four
existing fixture controls, and 26 owned determinism cases. These counts overlap
the earlier 89-case development run; do not add them together.

- Four D128 batch cells in both FP16 and BF16 passed independent FP64 reference,
  replay/fresh-storage, sequence-alone, reordered/duplicated batches and
  changed-neighbor checks. Shapes `(Hq,Hkv,Q,KV;B)` are `(8,8,500,500;40)`,
  `(8,2,1280,1500;8)`, `(8,8,448,1000;16)`, and `(32,8,256,256;256)`.
  Measured records include the persistent loop, causal walk, NATURAL stride
  and claim counter. Eight separate golden cases skip for missing baselines.
- Four three-WG D512 split cases (split3/unpacked and split17/packed, both
  dtypes) pass their original FP64/reference, replay and changed-batch checks,
  then skip at the missing golden. They include padding, sinks, poisoned
  scratch, empty rows and changed-neighbor retained capture. No numerical,
  storage or byte assertion was relaxed, and no golden was created.
- Six stress cases (D128 K/V ring with claim off/on, and three-WG D256, both
  dtypes) each passed 3,000 warm and 3,000 poisoned launches with zero byte
  errors. The unsafe ring control failed 200/200. D128 uses
  B16/Hq8/Hkv2/Q512/KV2048: 132 persistent CTAs, one per SM, and 7,928 aggregate
  K/V-stage refills per launch. This is below the control's three-CTAs-per-SM
  geometry and is explicitly limited evidence. Three-WG uses
  B8/Hq16/Hkv2/Q129/KV513/D256, 384 CTAs and eight K/V refills. FP64 reference
  checks precede stress; the helper's cold repeated-output reference measures
  stability. Zero failures bound each tested per-launch rate below 0.0998%
  at 95% confidence, not correctness of every schedule or shape.
- Fourteen native contract cases compare complete old/new parameter records,
  O/Stats bytes, fresh bindings and changed-input capture. Explicit records
  build with the automatic selectors poisoned. Retired engine20517 rejects
  explicitly and requires replanning. Partial tile records keep the fixed
  owner's tile, and omitted scheduling preserves the existing adapter rule.

A broader selected L0 native run gave **636 passed, 2 failed, 15 skipped and
162 deselected**. Both failures were stale fixture assumptions: a blanket
paged refusal also fails on the frozen parent, and a D128 path assertion named
the old module. The corrected dense-Q-paged/THD-paged boundary and local-module
assertion pass in the four controls above. The broad skips are 14 existing
runtime goldens and one D128 SASS-inspection case; none counts as a strict
baseline or rendering pass. Existing THD physical-origin, paging, prepared
binding and fresh-process artifact-reload tests remain intact.

Shared integration first found a real placement regression when the retired
row disappeared from registration. Its classifier now reads the same private
historical policy; 725 affected host cases pass. A separate comparison of
17,280 graphs, 34,560 heuristic-mode rows and 49,738 candidates has zero changes
in body, ordered knobs, placement or placement class, including partial GQA.
No threshold was retuned. The old broad-row name remains a private study key.

Reproduce the final new-owner group on Hopper with the runtime above:

```bash
cd test/python
CUDNN_TEST_NO_ISOLATION=1 pytest -o addopts= -q -rs \
  sdpa/frost/test_sdpa_fwd_dsl_sm90.py::test_sm90_row_declines_what_it_never_claimed \
  sdpa/frost/test_sdpa_fwd_d128_sm90.py::test_d128_graph_compiles_the_d128_template \
  sdpa/frost/test_sdpa_fwd_sm90_wide_implementation.py \
  sdpa/frost/test_sdpa_fwd_sm90_owned_determinism.py
```

The support page updates only the dedicated D128 and three-WG D512 fixed-body
rows. This does not add THD batch-invariance, other-GPU or backward evidence.

Bounded BF16 warm CUDA Graph replay checks used 32 executions per capture and
seven parent/head/head/parent rounds. Parent/owned latency in microseconds was
8.9284/8.9148 (D128 dense), 22.2600/22.2216 (D128 forced claim),
8.7796/8.7815 (three-WG D256), and 12.3298/12.3297 (three-WG D512).
Complete parameter records and O/Stats bytes matched; independent numerical
checks also passed after timed replay. This is component evidence for these
four configurations, not serving performance or a speedup claim.

The final backend/FROST build-all controls pass (2 cases). One additional
owned-D128 rendering case reuses the original claim-ring guards with only the
module path changed: **1 passed**. With CUDA13.4 compiler output and the cached
CUDA13.2 disassembler, all claim-release dependencies are preserved, STL/LDL
are zero, MAXREG is236, the register split is240/24, and the two narrow compute
KV loops contain no FP64 instructions. The standalone ptxas13.2 executable is
too old for this PTX version; its serialization-advisory check remains a gap.
The earlier missing-PATH skip is retained as history, not counted as a pass.
Local `fern check` cannot run because the command is not installed; changed-file
pre-commit passes, and documentation validation awaits normal PR CI.

Fresh-process artifact qualification now explicitly selects the fixed owners.
The earlier prepared-cache tests instantiated the retained standalone API, so
those results alone did not qualify the moved modules. Four new cases cover
F04 D64 dense, F05 D512 dense/THD, and dedicated D128 dense: each verifies the
owned module and local parameter type, independent O/Stats references, poisoned
output capture replay, identical O/Stats bytes across interpreters, and cache
hits with `cute.compile` forbidden in the second interpreter.

The initial run gave **6 passed, 1 failed** (four owned cases and three existing
standalone controls). The THD failure used an invalid Stats fixture whose head
stride did not cover the packed token total. The same fixture failed on the
original and owned bodies with identical parameters and output bytes; a legal
packed token-major view passed on both. Only that fixture was corrected, and
the targeted rerun gave **1 passed**. All four owned cases and the three existing
controls are therefore qualified; this is not a claim that the initial run was
green. No production source or reference/byte/capture assertion changed.

```bash
CUDNN_TEST_NO_ISOLATION=1 pytest -o addopts= -q -rs \
  sdpa/frost/test_sdpa_fwd_sm90_wide_implementation.py::test_owned_artifact_reloads_without_jit_in_fresh_process \
  sdpa/frost/test_sdpa_prepared_sm90.py::test_prepared_artifact_reloads_in_fresh_process \
  sdpa/frost/test_sdpa_prepared_sm90.py::test_prepared_d128_artifact_reloads_in_fresh_process
```

Public documentation CI subsequently passed on `00a232ea0`; the local `fern`
command remains unavailable. This test-only follow-up preserves production
source from that qualified head.


### SM107 small/mid half explicit records — registration checkpoint, 2026-11-15

The new fixed native and shared-body contracts are exercised by
`sdpa/frost/test_sdpa_fwd_sm107_half_selection.py`: 37 host tests pass,
including nine registered-host public-knob round trips, preserved physical
width envelopes, and N64 split ownership. These are metadata/host checks;
they do not launch a GPU or compare output bytes. The configuration audit
compares the real registered admission and host adapters with the legacy
loader, including complete configuration/TMA records.

At that registration checkpoint the new owners' native output/replay and
batch-invariance checks were pending. Existing body tests and cross-compiled
PTX equality alone do not qualify new graph records.

The integrated implementation revision subsequently passed 53 focused checks
on SM107, CUDA 13.4, CuTe DSL 4.8 and cuDNN 9.30. These cover public-record
reconstruction, fresh bindings, retained capture, both output-stride binders,
and score/gate admission. Two strict expected failures retain the existing
shared D256 nonpaged split independent-origin gap reproduced on the parent;
compact split is covered positively. A preceding focused revision passed 38
paged lookup, physical-origin, stride and dead-row checks, with 15 architecture
scope skips. The runs are partly overlapping and are not a full-pool claim.

No numerical tolerance or historical golden changed. These tests establish
only their explicit output and lifecycle checks; they do not add a general
batch-invariance or Stats-stride-override declaration.

### SM100 explicit MXFP8 implementation records — 2026-11-15

The existing D64/D128 multi-Q, D192/V128 multi-Q and D256 single-Q bodies
now have independent implementation identities. The new L0
`test_sdpa_fwd_small_mxfp8_explicit_engines.py` and
`test_sdpa_fwd_d256_mxfp8_explicit_engine.py` check public-record reconstruction,
the actual compiled body, fresh bindings and retained CUDA Graph replay.
Existing prepared, independent-origin and paged suites use the current owners.
No numerical tolerance or golden file changed. This qualification does not
expand the API's general batch-invariance claim.

Bounded native validation used B200 (SM100) and B300 (SM103), CUDA 13.4,
CuTeDSL 4.8 and cuDNN 9.30, with the loaded source checkout asserted:

- D64/D128/D192: 65 B200 and 40 B300 cases passed without skips or failures
  in the final focused run. An additional B300 paged subgroup passed all
  27 cases. Twelve cases already passed on each GPU in the preceding run
  were deliberately deselected in the final focused run.
- D256: the first B200 run passed 47 native cases; a separate stale host
  analyzer fixture failed and was migrated. B300 passed 14 bounded native
  cases. After correcting explicit-owner THD-origin binding, B200 passed
  31 checks (12 historical-golden cases deselected), and B300 passed six
  independent-origin checks (24 other cases deselected).
- Eight parent/head comparison scenarios for the small/mid bodies and four
  for D256 preserve the complete configuration, parameters, TMA descriptors,
  workspace, input/output bytes and changed-input capture results.
- On the combined implementation tree and its landed parent, 1,062 host
  contract/policy tests passed; nine tests needing a CUDA backend handle
  were deselected in the GPU-free run. The 3,840-graph comparison preserved
  all 7,320 candidate records and placement decisions.

These are separate, partly overlapping checks, not an aggregate count of
unique shapes or a full architecture sweep. The initial missing template
identity, off-contract THD fixture and missing origin binding failures were
retained and corrected before the reported passing reruns. Device evidence
comes from the frozen implementation revisions; subsequent parent integration
preserved those device bodies and repeated the affected host checks. General
batch invariance, unmeasured layouts and other GPU targets remain unverified
by this refactor qualification.

## Shared backward delta grid boundaries (2026-11-16)

`sdpa/frost/test_sdpa_bwd_dot_grid_determinism.py` directly exercises the shared
backward delta helper. Its contract is FP32 `rowsum(O * dO)`, optionally scaled
by the two device descales, plus any declared dQ accumulator/relay zeroing.
It has no forward operation and does not produce final attention gradients.
These results do not establish full-chain batch invariance or expand graph
engine eligibility.

The suite checks complete delta outputs against an independent FP64 formula,
zero/sentinel tails, fresh bindings, eight executions with concurrent work,
changed-input CUDA Graph replay, and a sequence alone, reordered and duplicated.
Dense FP16/BF16/E4M3 cases include batch/head 65535 and 65536 and a live second
query tile. Half THD cases include empty sequences, independent poisoned O/dO
origins, compact output packing and a live second query tile. The optional
accumulator and relay regions must both be cleared.

Runtime: PyTorch 2.14.1+cu130, cuDNN 9.28.0.38, CuTe DSL 4.8.0 with CUDA 13.4
compiler. Every process verified its isolated checkout import.

| Device / scope | Passed | Skipped | Interpretation |
|---|---:|---:|---|
| A100 / SM80, native L0 suite | 23 | 6 | FP8 conversion unavailable on SM80; those cases are unverified here |
| A100 / SM80, large-index L1 detector | 1 | 0 | Delta crosses the signed Int32 element boundary; owned prefix/tail canaries survive |
| L40S / SM89, native L0 suite | 29 | 0 | Portable helper execution, not an SM89 SDPA engine claim |
| P2021 / SM100, native L0 suite | 29 | 0 | Native shared helper at all new boundaries, with strict goldens |
| P2021 / SM100, existing integration controls | 14 | 0 | Five physical-stride helper cases plus nine D512 THD graph/standalone numerical, rebind/replay and deterministic-repeat cases; regular-sized complete chains |
| Host contract and cross-compilation, run on L40S | 14 | 1 | SM80 FP8 conversion skipped; half/FP8 compiled for SM89/90/100/103/107/120; no native claim for those other targets |

Before the fix, the original helper at `68853830a1b520f91dd77220acfe44b1579a9379`
failed the initial 21-case L40S matrix **11 failed / 10 passed**, all failures
being CUDA launch-configuration rejection above 65535. The same A100 FP16
subset had **2 failed / 2 passed**. Keeping the flat launch while restoring
only the Int32 delta multiplication makes the large-index detector fail on
untouched live output rows; widening that multiplication passes. Its whole
wrapped-address interval belongs to the test's prefix allocation. This L1
case requires 28 GiB free and remains outside the smoke suite.

Fourteen new small goldens were recorded only after the reference/replay checks,
then compared with recording disabled. No previous golden was replaced.
FP16/BF16 delta output hashes agree on both tested devices; FP8 includes its
explicit descale product. The new records are in runtime directories
`a29ca37b88513cb6` (A100), `98a9396f557b5d7c` (L40S) and
`3bada6ccd1cb05e1` (SM100). The SM100 combined packet was 43 passed / 46
L1 cases deselected; those deselections are not additional hardware coverage.

Commands from `test/python`, with the stated GPU visible and source import
verified:

```bash
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  pytest sdpa/frost/test_sdpa_bwd_dot_grid_determinism.py -m L0
pytest sdpa/frost/test_sdpa_bwd_dot_grid_contract.py -m L0
pytest sdpa/frost/test_sdpa_bwd_dot_large_index.py -m L1
```

Native SM103/107/120 execution at the new boundary and full SDPA-chain
qualification at that boundary are pending. Existing API support rows retain
their prior limits; the SM100 integration controls above cover regular shapes.

### Parallel-memory follow-up (2026-11-16)

The large live query-tail and THD head-boundary fixtures now use D32 storage;
batch/head/query extents, complete comparisons, retained old dO, eight repeats,
and changed-input capture are preserved. Ordinary dense grid boundaries and
THD batch boundaries still use D64. The large accumulator uses D32 with an
additional D64 small control. The L1 address-span detector and production
helpers are unchanged; small golden inputs and outputs are unchanged.

With the same runtime listed above and a 2-GiB Torch allocator limit per
worker, the final L40S suite passes all 30 cases. On SM100, four workers run
the native suite, five existing stride controls and the host/compiler suite:
49 pass and one SM80-FP8 compile case skips. Reserving unrelated memory leaves
16 GiB for the entire process group, including contexts/compiler allocations;
the sampled group peak is 9.14 GiB. Four additional synchronized THD rebind
runs hold their largest live inputs together and all pass, with a sampled
group peak of 9.35 GiB. Both runs stay below 1.880 GiB allocated / 1.917 GiB
reserved per worker. Sampling is not an exact instantaneous peak measurement.

The pre-follow-up D64 query-tail fixture fails its initial input allocation
under the same 2-GiB cap. Original-helper RED controls must run in separate
processes: retained failed-test tracebacks otherwise exhaust that cap and
mask later launch failures. Ten isolated L40S controls against the exact
original helper produce seven invalid-launch failures (FP16/BF16/FP8 dense,
half THD and accumulator initialization) and three legal-boundary passes,
without OOM. These bounded SM100/L40S results do not establish
native SM120 qualification or change the API support status above.

## Shared THD backward fold grid boundaries (2026-11-16)

The direct `dkv_reduce_thd_origins_host` regression covers FP16 and BF16 outputs
from FP32 or matching half partials, MHA and GQA=3, and Dk/Dv=64/128 and 128/64.
It checks batch and KV-head counts 65535/65536, a live second block, empty
sequences, independent physical output origins, and the complete output
allocations including gap canaries. An independent FP64 sum of exactly
representable values checks every live gradient. Eight repeats include concurrent
GPU work. Fresh storage, singleton/reordered/duplicated sequences, changed
neighbors, and changed-input CUDA Graph replay preserve the expected bytes.
All input partials and metadata are checked for unintended mutation.

The base reducer at `b837a4eae4f4` produces **24 failed / 16 passed** on A100
with small golden cases deselected: all failures are `CUDA_LAUNCH_INVALID_CONFIG`
above grid.y/z limits; the legal edge controls pass. The fix retains ordinary
three-dimensional grids and each row's reduction order. Only overflowing batch
or head dimensions use a flat grid. The maximum sequence length stays a runtime
host argument. D128 single/two-CTA callers pass their existing compile-time
batch fact without first converting it into a runtime scalar.

Qualification uses Torch 2.14.1+cu130, cuDNN 9.28.0.38, CuTe DSL 4.8.0 and
CUDA compiler 13.4, with the isolated source import verified:

| Device | Direct helper | Other checks |
| --- | --- | --- |
| A100 / SM80 | 48 passed | None |
| L40S / SM89 | 48 passed | 15 host/cross-compilation checks passed |
| P2021 / SM100 | 48 passed | 2 existing D128 single/two-CTA numerical comparisons and 9 D512 THD graph/standalone cases passed |

The SM100 command also deselects 14 existing L1 cases; these are not coverage.
The two D128 cases compare both native half partials and FP32 accumulation
against the existing reference. The D512 checks include independent origins,
rebinding, changed-input replay and deterministic repeatability at ordinary
shapes. The 15 non-native checks comprise one launch-bound test and half/FP32
partial cross-compilation for SM80/89/90/100/103/107/120, retaining a runtime
maximum length. They do not establish numerical behavior on absent GPUs.

Twenty-four new small golden records (eight per measured GPU) were recorded
only after the reference and replay checks, inspected, and checked with recording
disabled. No previous golden was replaced. The eight output pairs match across
these three measured runtimes; this is not a cross-runtime compatibility promise.

Reproduction, after installing this checkout and verifying `cudnn.__file__`:

```bash
cd test/python
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" pytest -o addopts= -m L0 \
  sdpa/frost/test_sdpa_bwd_thd_fold_grid_determinism.py \
  sdpa/frost/test_sdpa_bwd_thd_fold_grid_contract.py
# On SM100, additionally run the caller integration cases:
pytest -o addopts= -m L0 \
  sdpa/frost/test_sdpa_bwd_d128_dkv_accum_sm100.py::test_dkv_accum_fp32_dq_bitwise_vs_default_on_one_kv_block_and_dk_dv_recorded \
  sdpa/frost/test_sdpa_bwd_thd_origins_sm100.py::test_origins_rebind_and_replay \
  sdpa/frost/test_sdpa_bwd_thd_origins_sm100.py::test_graph_deterministic_origins_repeat_bitwise \
  sdpa/frost/test_sdpa_bwd_thd_origins_sm100.py::test_graph_independent_origins_2x2
```

This helper has no forward or FP8-output contract. Forward kernels and the
other backward reducers are not changed. SM90/103/107/120 native execution,
complete attention chains at B/H=65536, and metadata addressing at billion-entry
batch sizes remain unverified. No FROST capability or full-API determinism row
is widened by this helper qualification. Other main/auxiliary launch chains
and the native backend remain within the open #1573 audit.


### Full-product correction after independent review

The initial flattened revision (`756febfd5942`) bounded B*H but omitted the
runtime tile factor. On H100, the new compact-allocation regression at B=65536,
Hkv=32, Dk/Dv=512/256 and max_skv=8193 is **1 failed**: the launch and sync
succeed, but live gradients retain 123 instead of 1. This original numerical
failure is distinct from the earlier explicit invalid-grid failures.

The correction bounds each physical grid.x to 65535 and traverses owners and
live vectors with Int64 indices. Empty sequences do not scan the declared
max_skv tile range. Ordinary layouts retain their batch/head axes and stride
through local vectors; reduction order and output conversion are unchanged.
The existing B*H admission bound remains. Eight added native cases cover
logical work above 2^31-1, above 2^32, a 2097184-token sequence, and ordinary
grid-stride tails, with both output dtypes, independent origins and changed
input capture replay. The helper's max_skv is now runtime in the native test
wrapper as well as offline compilation.

On the frozen correction, Torch2.14.1/cuDNN9.28.0.38/DSL4.8/compiler12.9:

| Device | Full helper + host suite | Follow-up |
| --- | --- | --- |
| A100 SM80 | 69 passed, 2 compiler-target skips | Existing 8 small goldens match |
| L40S SM89 | 69 passed, 2 compiler-target skips | Existing 8 small goldens match |
| H100 NVL SM90 | 61 passed, 10 skips | 8 missing small goldens added; 8 strict checks passed |
| SM100, 148 SMs | 61 passed, 10 skips | 8 missing small goldens added; 8 strict checks passed |

The two remaining skips are the installed CUDA12.9 compiler's explicit
unsupported-SM107 diagnostic. All other compiler errors still fail the test.
Sixteen new runtime records were first written to a separate directory; their
input and output hashes match every existing corresponding three-device record.
No golden was overwritten, and recording was disabled for strict rechecks.
These results are direct-helper qualification, including all eight new extreme
cases on each GPU; they do not qualify complete attention at those dimensions.
On the same frozen production source, DSL4.8/compiler13.4 compiles all seven
architecture/precision pairs (15 host/offline checks passed). The actual SM100
D128 single/two-CTA and D512 THD caller packet passes 11 L0 cases; 14 L1 cases
are deselected. Pre-commit passes and Fern check reports zero errors.


### SM100/SM103 FP8 swap-AB wide output addressing — 2026-11-16

The D128 body now widens all coordinates before an Array row slice accumulates
runtime stride products. Original compact output controls pass on B200 and
B300; an output batch stride of `2**31 - 65536` elements at B4 leaves half the
original output unwritten. The fix changes addressing only, with no arithmetic,
barrier, capability or plan-selection change. This forward body adds no backward
implementation.

Native qualification used cuDNN 9.24.0.43, Torch 2.11.0+cu130, CuTeDSL 4.8.0
with compiler/NVRTC 13.4, and verified checkout imports. Parent kernel source
is `f1b7ee075`; final candidate adds the coordinate-width fix and maintained
prepared-output regression. B200/SM100 and B300/SM103 both have 148 SMs in these
runs. CUDA 12 compiler compatibility was not measured in this qualification.

- B200: **74 passed, 0 skipped** for the 48 new output-layout cases and existing
  dense/paged replay, batch, golden and warm-refill cases.
- B300: **58 passed, 0 skipped** for the same 48 output-layout cases and existing
  dense replay, batch, golden and warm-refill cases. Paged SwapAB is not claimed
  on SM103.
- Separate B200 D64/D128 paged, split, optional-output and route controls:
  **60 passed, 0 skipped** with the identical kernel patch.

The 48-case regression crosses E4M3/E5M2 input, FP16/BF16/E4M3/E5M2 output,
packed/unpacked heads, compact storage, wide batch products and wide strides.
It checks O, Stats and Amax against the independent reference, byte equality
for unchanged replay, changed V/scales, and restored-input replay. Snapshots
are cloned before mutation. Existing batch tests preserve quantized payloads
through reordering, duplication and changed neighbors; whole-batch Amax retains
its separate comparison rule. Warm-refill controls on both GPUs detect the
unsafe ring; the candidate has zero mismatches in 256 warm plus 256 poisoned
launches per stress case.

For the exact old-stack environments, record the **unmodified parent** into a
new private golden directory, then compare the candidate with recording unset.
All 10 B200 and 4 B300 parent records pass strict candidate comparison; no
committed baseline is replaced. The parent recording runs are separate from
the candidate counts above.

From `test/python`, run the maintained wide-output detector plus the affected
existing CI entries:

```bash
pytest sdpa/frost/test_sdpa_prepared_fp8.py::test_decode_fp8_physical_output_strides_replay -m L0 -q
CUDNN_DETERMINISM_GOLDENS=/path/to/same-runtime-parent-goldens \
  pytest sdpa/frost/test_sdpa_fwd_fp8_sm100.py -m L0 -q \
  -k 'TestDenseFp8SwapAB and (test_replay_batch or test_golden or test_warm_refill)'
# SM100 additionally:
CUDNN_DETERMINISM_GOLDENS=/path/to/same-runtime-parent-goldens \
  pytest sdpa/frost/test_sdpa_fwd_fp8_sm100.py::TestNativeFp8DecodeDeterminism -m L0 -q
pytest sdpa/frost/test_sdpa_fwd_fp8_sm100.py::TestNativeFp8Decode -m L0 -q
```

### JAX SDPA Python frontend route (2026-11-16)

Suite: `sdpa/jax/test_sdpa_jax.py`. Test environment: SM100, cuDNN 9.27,
CUDA 13.4, JAX source with jaxlib/CUDA plugin `0.12.1.dev20261116`.
The paired suite uses identical inputs/cotangents with the frontend setting
unset, `0`, and `1`, and compares with JAX's XLA reference. A separate BF16
case compares raw output/gradient bytes across fresh allocations and B=1
isolation. Golden hashes and concurrent-load stress are not yet qualified.

Command from `test/python`, with the companion JAX checkout on `PYTHONPATH`:
`python -m pytest sdpa/jax/test_sdpa_jax.py -m L0 -x -q`.

Result: **22 passed**, no failures or skips (38.90 s). This includes 20
paired numerical cases, one native-lowering case, and one replay/isolation
case. Four companion JAX dispatch tests also passed.

### Explicit JAX SDPA operations (2026-11-17)

`cudnn.ops.sdpa_forward_jax` returns output and FP32 stats;
`cudnn.ops.sdpa_backward_jax` consumes both and the fixed upstream gradient.
The BF16 `(B=2, S=128, H=4, D=64)` test compares raw output, stats and dQ/dK/dV
bytes across three fresh-allocation replays and each sequence isolated at B=1.
Backward uses `deterministic=True`. There is no bias in this case; dBias,
FP16 byte determinism, concurrent-load stress and golden hashes are unqualified.

With the companion JAX and frontend source on `PYTHONPATH`, run from `test/python`:
`python -m pytest sdpa/jax/test_sdpa_jax.py -q -ra`.

On cuDNN 9.27/CUDA 13.4, the suite passed **29 tests** on SM100 (47.57 s)
and SM90 (31.10 s), without failures or skips. This includes 22 dtype/feature
cases comparing the explicit operations, all three environment settings and
an XLA reference; one native-lowering check for each direction in one test;
one replay/isolation test; four saved-tensor validation cases; and one missing
FFI-header check. Stats remain FP32 between forward and backward. The native
backward lowering has one FFI call, with no forward recomputation.


### Shared backward delta: native SM120 memory qualification — 2026-11-17

The unchanged shared-helper source at `54b464d2bd233db1535caec4628fae6decf9456f`
was run on an actual RTX 5080 (SM120, 84 SMs, 16-GB device), with Torch
2.14.1+cu130, cuDNN 9.28.0.38, CuTe DSL 4.8.0 and its CUDA 13.4 compiler.
The frozen checkout and every runtime dependency were hash-checked; imports
resolved to that checkout. No unrelated memory reservation simulated a smaller
GPU. Each pytest worker had a 2-GiB Torch allocator limit.

- Four workers: **49 passed, 1 skipped**, including all **30 native** delta
  cases, five stride controls and the host/compiler suite. The only skip is
  the existing SM80-FP8 compiler case; no native SM120 case skipped.
- Four synchronized replicas of the largest THD rebind case: **4 passed**.
  All four workers reached the live-storage barrier while retaining their
  old and fresh inputs.
- A separate process with recording disabled and strict golden checking:
  **6 passed, 0 skipped**. Five small golden records cover dense FP16/BF16/FP8
  and THD FP16/BF16; the sixth case is the small accumulator control.

The sampled device-memory increments were 8.54 GiB for the four-worker
suite and 9.26 GiB for synchronized replicas. Maximum measured per-worker
Torch allocation/reservation was 1.880/1.916 GiB. These samples do not establish
an instantaneous device peak. The run completed normally. An earlier runner
attempt stopped before pytest because two helper plugins were not staged;
that failure is retained separately and contributes no qualification.

The five records under `45e8f430a9d818a2/` were recorded only after reference,
replay and batch checks, then compared strictly in the separate process.
Their input/output records also match the existing runtime-specific records
where present. Only the runtime fingerprint differs. Re-run the strict small
selection on the matching environment with:

```bash
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  pytest sdpa/frost/test_sdpa_bwd_dot_grid_determinism.py -m L0 -k small -q
```

This closes the native SM120 qualification gap for these direct-helper cases.
It does not qualify full attention chains at the launch boundary, native
SM103/SM107 boundary cases, or a new public API support row. Current-head
review and integration CI remain separate from this bounded packet.

### Canonical NVFP4 Hadamard paired-CTA SFB handoff — 2026-11-19

The nonquant `grouped_gemm_glu_hadamard_wrapper_sm100` forward path now
observes each CTA's own SFB TMA completion before in-place scale permutation.
A per-stage, two-arrival ready barrier and cluster-scope acquire precede
collective S2T. Stage reuse remains owned by the original AB pipeline.
This changes synchronization, not the numerical formula or output scale
format. The API has no backward entry point. The separate Hadamard+quant
kernel is unchanged.

Fresh validation used NVIDIA GB200 (SM100, 152 SMs), Torch
2.13.0a0+8145d630e8.nv26.06, CUDA 13.3, cuDNN 9.25.1, and CuTe DSL 4.8.0
(compiling with CUDA 13.4), job 9182480. The worktree was
`glu_canonical_validation/pr_publication_20261110/worktrees/cudnn-frontend`
at `ca3f85b6` with the local synchronization fix. Kernel source SHA256 was
`15e2088c1539992fd0906f7daff550abc8575faa9dfc8dfec962555ac89c844f`;
new determinism suite SHA256 was
`3ecabd0f3b84ab54e294752704809fd5f082fd2a99160143163ce12fbab01143`.
The launcher asserted the Python package/API paths and unchanged kernel
hashes during each run; it reused the existing compiled cuDNN binding.

Run from `test/python`, with `CUBLAS_WORKSPACE_CONFIG=:4096:8` and the
worktree package selected explicitly (equivalent pytest arguments):

```bash
python -m pytest -v -m L0 gemm/cutedsl/test_grouped_gemm_glu_hadamard_layout.py
python -m pytest -s -v -m L0 gemm/cutedsl/test_grouped_gemm_glu_hadamard_determinism.py
python -m pytest -v -m L0 gemm/cutedsl/test_grouped_gemm_glu_hadamard_quant_gate_then_up.py gemm/cutedsl/test_grouped_gemm_glu_hadamard_quant_determinism.py
```

- Existing Hadamard layout/reference/replay/batch-invariance suite: **13 passed**.
  Two cases check the existing rejection of canonical discrete-weight TMEM RHT.
- New stage reuse/synchronization suite: **19 passed**. Twelve cases cover
  K=1792/4096, static/dynamic scheduling, dense/discrete SMEM RHT and dense
  TMEM RHT. Four inject a 20,000 ns nanosleep before either CTA's scale
  permutation, for dense/discrete weights. Emitted PTX confirms all four
  instrumented kernels contain the delay and the cluster acquire. The
  instrumented subset was rerun separately: **4 passed, 15 deselected**;
  this overlaps the 19-case result.
- Three warm-stress cases use grouped M=[1024,0,2048], N=2048, K=4096,
  five input stages and persistent scheduling. Each completed 3,000 warm
  and 3,000 poisoned launches with zero differing bytes. The helper's unsafe
  ring control failed 200/200 launches. The target uses 152 resident CTAs
  (one per SM), versus 456 for the control; these bounded measurements do
  not exhaust possible schedules. Cold-launch byte references supplement
  the independent FP64 GEMM/SwiGLU and explicit H16 numerical checks.
- Separate unchanged Hadamard+quant suites: **33 passed**, including row/column
  payloads/scales, changed-input graphs, batch invariance and warm stress.
- No failures or skips in those completed runs. An initial launcher attempt
  failed argument parsing because pytest-xdist was absent; removing `-n0`
  allowed the serial runs above. No numerical tolerance or golden changed.

The independent numerical checks retain C rtol/atol=0.01/0.005,
D=0.015/0.005, exact stored-D amax, and post-RHT amax=1e-4/1e-4.
All replay, input-preservation and legacy-layout comparisons use exact bytes.
The six pre-existing modified files were verified unchanged by SHA256.
For dense/discrete SMEM and dense TMEM, legacy interleave32 PTX is identical
before/after after normalizing entry-symbol names and removing source comments.
Canonical PTX confirms group-one SFB multicast completion, permutation stores
before release, and cluster acquire before group-two S2T.

No old-kernel timing failure was reproduced; the fix closes the synchronization
contract gap. The generic unsafe-ring control is not a reproduction of that
specific defect. SM103 and untested shapes remain unverified.
Evidence (logs, XML, provenance, before/after sources and PTX) is under
`glu_canonical_validation/hadamard_sync_fix_20261119/` in the validation workspace.

### SM107 D256 per-tensor FP8 two-warpgroup body — 2026-11-19

The two-warpgroup D256 per-tensor FP8 body (Q resident in TMEM, one softmax
warpgroup per KV-step parity) is selected through the standalone adapter
on an explicit `cga=2` with `softmax_precision=HALF`; the public record keeps
`cga=1`. `TestMidFp8Implementation` in `sdpa/frost/test_sdpa_fp8_sm107.py`
gained the admission test (route assertion, declines, default width), the
fp64 reference + replay cases (odd and even step counts, a zero-Q/K run that
never rescales O, causal, bottom-right causal with keyless Q tiles handed
over under the cluster launch control, and window, per-batch length and sink
masks), batch-isolation cases (a sequence alone, reordered within the batch,
next to changed neighbors; dense, causal, and bottom-right with a window,
per-batch lengths and a sink), a PTX pin of the four shared-memory slot
credits that depend on their loaded words through the release-zero argument
(eight rendering families covering every input type, mask form, output type
and Stats / Amax_O setting), and a warm-stress case over the refilled alpha /
running-max / final-stat slots (2048 CTAs, eight KV steps per tile).

Run on native cc 10.7 (208 SMs) with Torch 2.12.1+cu130, cuDNN 9.30.0 and
CuTe DSL 4.8.0a0 (CUDA 13.4 compiler): all cases pass; the warm-stress control
reproduced the unsafe twin's failure on the same device. No goldens were
recorded.

## Simplified sparse attention CI failures — 2026-11-20

Audited all 18 `python/misc/cutedsl:be-*-cutedsl-*` jobs in
pipeline 84114932,
source `6d739b867aa5ca1006767296a94f6687d9b71786`. The three version columns
below use cuDNN 9.28.1 / DSL 4.8.0, cuDNN 9.29i / DSL 4.8.0, and
cuDNN 9.30i / DSL 4.9.0a0, respectively. Results are retained CI evidence,
not reruns of the skip change.

| Lane | 9.28.1 / 4.8.0 | 9.29i / 4.8.0 | 9.30i / 4.9.0a0 |
| --- | --- | --- | --- |
| SM80 | Canceled before start (job 563000189) | Canceled before start (job 563000177) | Canceled before start (job 563000201) |
| SM90 | 4098 passed (job 563000190) | 4098 passed (job 563000179) | 4098 passed (job 563000203) |
| SM100 | 4 failed (job 563000193) | 4 failed (job 563000181) | 4 failed (job 563000206) |
| SM103 | 30 failed (job 563000195) | 31 failed (job 563000183) | 30 failed (job 563000209) |
| SM107 | Resource-wait timeout (job 563000196) | Resource-wait timeout (job 563000185) | Resource-wait timeout (job 563000211) |
| SM120 | 3 failed (job 563000198) | 18 failed (job 563000187) | 9 failed (job 563000213) |

### Failure classification

- SM100: each lane has three raw-byte determinism failures (dense KL indexer
  H96 forward replay, dense KL indexer H16 Torch forward batch invariance,
  JAX attention H16 forward batch invariance) and one JAX invalid-input
  `BufferError`: expected CUDA device 0, current device 3. The invalid-input
  parameters differ by lane (`denominator`, `empty`, `bounds`). That device
  mismatch is not a determinism failure and remains active.
- SM103: all 30/31/30 failures are raw-byte replay or batch-invariance
  mismatches in the three simplified sparse attention determinism suites.
  The middle lane also fails the PyTorch H96 backward golden's replay check.
  Across the lanes, the union is 31 failing case IDs. Golden failures occur
  before golden comparison; golden hashes were not changed.
- SM120-tagged lanes: all 3/18/9 failures are capability or compilation
  failures. NSA Top-K requires tcgen05, which is unavailable on SM12.x.
  CSA backward exceeds the device's 101376-byte shared-memory limit; DSL 4.9
  also fails to compile six configurations for this reason. The 9.29i lane
  has 15 paged QSA score failures because deterministic Top-K limits FP32
  columns to 16384 while those cases request 32768. These are not measured
  raw-byte failures, including cases whose names contain `deterministic`,
  and remain active. The NSA log identifies SM121 in the middle lane and
  SM120 in the other two; the CI lane tag alone does not identify the GPU.
- SM107: all three jobs exhaust the one-hour CI limit while Slurm reports
  `queued and waiting for resources`. No tests ran. SM80 jobs were canceled
  before starting. Neither result establishes a determinism status.

Job 563000195 (job 563000195)
ran on `theia0118` (NVIDIA GB300, SM103), driver 620.78 (CUDA 13.5).
Its result was **30 failed, 5719 passed, 1938 skipped, 1 xfailed** in
1310.01 seconds. All 30 failures were raw-byte mismatches, with no JUnit
errors: dense KL indexer 4 replay / 7 batch-invariance failures, PyTorch
attention 5 / 4, and JAX attention 6 / 4. Each case stops at its first
mismatch; later comparisons are unmeasured. These logs establish the failure
types, not their kernel or test-setup cause.

### Temporary skip scope

At the user's request, only the measured determinism failures are temporarily
skipped: three case IDs on SM100 and 31 on SM103. The architecture-specific
lists are in `simplified_sparse_attention/conftest.py`; the reason starts with
`skip until fix: known determinism failure`. Other case IDs, GPUs, independent
correctness suites, and golden files remain active. Remove these skips with
the fixes and rerun replay, batch invariance, and independent reference checks
before upgrading support status.

Host validation compares the configured IDs with the union of raw-byte
failures in all six SM100/SM103 JUnit artifacts. With mocked GPU capabilities,
the fixture skips exactly 3 IDs on SM100 and 31 on SM103, and none on SM90,
SM107, or SM120. JAX invalid-input and passing golden controls remain active.
This is a fixture check, not a GPU test run. The local Python environment has
no Torch, so the GPU suites and collection were not rerun. Pre-commit and
`git diff --check` passed. `fern check --warnings` reported no errors; its
redirect check could not reach FDR (HTTP 403).

## Misc CuTeDSL CI device and capacity fixes — 2026-11-20

Follow-up to pipeline 84114932:

- The JAX invalid-input tests call `jax_inputs` directly. Its first DLPack
  conversion could initialize XLA after entering a tensor's CUDA device guard,
  changing the current device before export. Initialize XLA in a separate
  restoring guard before conversion; exercise direct conversion and execution,
  including initialization and execution errors.
- The NSA THD Top-K test now uses the existing tcgen05 capability gate.
  Paged QSA's FP32 deterministic Top-K capacity gate now covers SM121 as well
  as SM120, matching the API's architecture check.
- CSA ratio-128 backward support checks one compressed-row schedule, but
  compilation previously built both small and default schedules. The unused
  default could exceed SM12.x shared memory even when the declared shape fit.
  Compile only the validated shape's schedule; empty plans compile no kernel.
  Batch-invariance checks apply the existing shared-memory gate separately
  to each packed backward layout. Unsupported shapes are skipped as unsupported,
  without claiming byte determinism or extending the supported envelope.

Validation used SM100, driver 610.11, Torch `2.14.0a0+b2c75dd062.nv26.09`,
CUDA 13.4, cuDNN 92700, JAX 0.11.2, and CuTe DSL 4.8.0, in the
cuDNN 9.27.0.42 / CUDA 13.4.1 amd64 CI image.
The test runner prepended `/workspace/python/cudnn` to the installed package's
search path and loaded the checkout's JAX version constants. Changed Python
APIs and kernels resolved to `/workspace`; the container retained its installed
1.28.0 native extension. These checks do not validate a new C++ build or the
CI backend/DSL combinations. Base source was `fdae16db7dd55dc69ba8da4026bca272177948d5`
plus this change.

Run from `test/python`, with `CUDNN_TEST_NO_ISOLATION=1`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, and `XLA_PYTHON_CLIENT_PREALLOCATE=false`:

| Selection | Result |
| --- | --- |
| Targeted JAX device, NSA capability, QSA column-limit, CSA compilation and packed-layout memory regressions (`-m L0`) | 34 passed |
| Complete `simplified_sparse_attention/cutedsl/test_simplified_sparse_attention_jax.py` (`-m L0`) | 27 passed, 1 skipped in 156.96 s |
| CSA backward batch-invariance cases and packed-layout memory regression (`-m L0 -k test_batch_invariance`) | 18 passed, 1 xpassed in 10.69 s |
| CSA ratio-128 replay and CUDA Graph capture (`-m L0 -k 'replay_determinism_r128 or cuda_graph_capture_r128'`) | 2 passed in 2.30 s |
| CSA ratio-128 schedule envelope (`-m 'L0 or L1' -k r128_envelope_execution`) | 10 passed in 10.55 s |
| NSA THD noninterference, CSA compile-capacity and shared-memory checks, and CSA batch invariance (earlier bounded run) | 29 passed, 1 xpassed in 11.98 s |

The one JAX skip requires a second visible CUDA device. The known CSA forward
schedule-boundary xfail happened to pass; it remains a known failure and was
not removed. The envelope checks retain independent FP32/FP64 references and
existing replay comparisons. No golden files or numerical tolerances changed.

A negative control restored the previous JAX conversion helper, CSA compile
method, and paged QSA capacity helper while retaining the new tests: **10 failed,
10 passed** in 3.52 s. The corrected versions passed the targeted regression run.
Mocked capability cases cover SM120 and SM121, but no SM12.x GPU or DSL 4.9
run was available. Those execution scopes remain unverified.

## MoE MXFP8 scale-row offsets — 2026-11-21

The fused Gate/Up scale loader shifted a blocked scale layout by a signed
scalar row offset. For CTA 1 of an even FC1 tile, that offset is -128.
Decomposing it inside the 128-row scale atom under CuTe DSL 4.9 corrupted
alternate Up tiles. The loader now applies whole-tile offsets to the outer
row mode, retaining zero coordinates inside the atom.

On one SM100 GPU with 148 SMs, driver 610.11, PyTorch 2.14.0a0,
CUDA/compiler 13.4 and cuDNN 9.27.0, the original DSL 4.9 forward had relative L1 error 0.367806
against the independent PyTorch reference. The fixed forward has error
0.0346925, within the unchanged 0.1 bound. Its output, quantized weights,
and every saved forward-context tensor match the original DSL 4.8 run
byte for byte for the fixed seed-40, 1024-token, four-expert case.

`moe_ep/cutemok/test_cutemok.py::test_cuda_graph[mxfp8-False]` now uses
`determinism_utils.assert_replays`: eight executions with fresh outputs,
fixed inputs and gradients, and concurrent GPU work. It compares output,
input and router gradients, and all six routed/shared weight gradients as
raw bytes. Changed-input and changed-routing graph replay also compares
raw bytes. The separate `test_forward_backward[mxfp8]` retains its numerical
reference and error bounds.

Commands from `test/python`, with `CUDNN_TEST_NO_ISOLATION=1`:

```bash
python -m pytest moe_ep/cutemok/test_cutemok.py -q -rs --tb=short
python -m pytest attention_prepare/frost/test_attention_prepare_decode.py -k rendering_pins -q --tb=short
```

The replay scope is one rank. Batch insertion, removal and reordering,
multiple ranks, warm-stress qualification, and SM103 execution remain
unverified. The API support row remains unverified.

Measured from `ad170f52` plus this fix:

| CuTe DSL | Full MoE suite | Cross-target rendering checks |
| --- | --- | --- |
| 4.8.0 | 30 passed, 1 skipped; 80.13 s | 3 passed; 18.67 s |
| 4.9.0a0+20261120230000.ecb3b50 | 30 passed, 1 skipped; 75.60 s | 3 passed; 19.21 s |

The skip is the existing multi-rank barrier-wrap case: one rank launches
no communication barrier. Each rendering check covers 15 forms/launch arms
for one of SM100, SM103 and SM107, including instruction and SASS checks;
rendering another target does not establish GPU execution on that target.
The model report suite on current develop also passes: 51 passed, 35.16 s.

### Grouped BF16 dGLU deterministic dprob — SM100, 2026-11-23 UTC

`grouped_gemm_dglu_wrapper_sm100(..., deterministic=True)` now covers the
SM100-family BF16 kernel as well as the block-scaled one. Each
(token, N-tile) dprob partial gets its own FP32 slot, and the slots are
summed in a fixed order on the launch stream. The new
`gemm/cutedsl/test_grouped_gemm_dglu_bf16_determinism.py` (18 cases: 16 L0,
plus two L1 cases at scale) checks the following:

- Replay: eight runs on fresh outputs with concurrent work on another
  stream compare dprob and `D_row` as raw bytes. This covers both
  activations, dense and discrete weights, and the static and dynamic
  schedulers. The shape is N=1024 (four N-tiles), K=256 and experts of
  512/0/256/768 tokens plus 256 rows of capacity.
- Expert reorder, solo, duplicate and changed-neighbor runs: they keep each
  expert's dprob and `D_row` bytes, with natural-rank tensors in both GLU
  orders.
- `D_row` is byte-equal to the atomic path.
- dprob matches the atomic path within 1e-4 and the independent Torch
  reference within the BF16 tolerance. Rows past the last offset stay zero.
- Slot zero-fill and reduction run on the launch stream, both when the
  current stream is busy and when the launch stream is busy.
- dbias and JAX inputs are rejected.

The one skip is dGeGLU with `gate_then_up`, which the BF16 kernel does not
support.

Runtime: NVIDIA GB200 (cc 10.0, 152 SMs, aarch64), driver 595.71.05,
PyTorch 2.14.1+cu130, CuTe DSL 4.8.0 (CUDA 13.4 compiler), tvm-ffi 0.1.16,
at `2b98a25d` plus this change. The source tree was imported with a stubbed
compiled core, so no cuDNN backend was involved.

```
pytest -m "L0 or L1" gemm/cutedsl/test_grouped_gemm_dglu_bf16_determinism.py  # 17 passed, 1 skipped
pytest -m L0 <the dGLU suites below>   # new tree: 602 passed, 426 skipped, 2 failed of 1030
```

The dGLU suites are `test_grouped_gemm_dglu{,_bf16_determinism,_bf16_training,_activation,_gate_then_up,_workspace}.py`,
`test_grouped_dgeglu_autograd.py`, `test_grouped_gemm_backward_canonical.py`
and `gemm/jax/test_grouped_gemm_dglu_jax.py`. The two failures,
`test_dglu_activation_caller_workspace_and_first_capture[False/True]`, fail
the same way on develop in this environment: the stubbed core's
`make_operand_buffer` has no `__dlpack__`.

The suite fails against develop: 17 failed, 1 skipped (BF16 rejects the
flag). It also fails against three mutations:

- All N-tiles write slot 0: replay and batch-invariance cases fail with
  156-373 differing bytes.
- Reduction on torch's current stream: the busy-launch-stream case fails.
- Zero-fill on torch's current stream: the busy-current-stream case fails.

At the suite's shape, the atomic dprob changed in 29 of 29 replays for each
of the eight activation/weight/scheduler cases, in up to 645 of 1792 rows.
The deterministic dprob changed in 0 of 29.

At the FC2 shapes, with 20 runs, fixed inputs and EP4/EP8, the atomic dprob
changed in 19 of 19 replays. That was up to 32152/65536 rows for DeepSeek-V3
(N=2048) and up to 55320/98304 for Qwen3-2.4T. The deterministic dprob
changed in 0 of 19, and `D_row` never changed on either path. Non-default
tiles and clusters ((256, 256)/(2, 2) at N=768, (128, 256)/(1, 1) at N=800
and (256, 128)/(2, 1) at N=544) replay bitwise and stay within 4e-7 of the
atomic path.

compute-sanitizer 2026.1.1 reports 0 errors under memcheck and 0 hazards
under racecheck. Memcheck ran three suite cases, including `gate_then_up`,
and four direct launches: two at the default tile, one with the (2, 2)
cluster and one with the (128, 256) tile. Racecheck ran the two default-tile
launches. SM103 and SM107 were not run, only compiled: ptxas 13.4.92 reports
0 spills for the deterministic configurations, 26 on sm_100a and 24 on sm_107a.

### Phase 1 SM100/SM103 loss-byte expectations — 2026-11-24

Pipeline 85038338, job 569171688, reports `replay 2 loss: 1/4 bytes differ`
for `test_replay[bf16-h32-tail-fprop-cutedsl]` on SM103 with cuDNN 9.28.1.4
and CuTe DSL 4.8.0. The independent numerical reference passed before the
byte comparison failed. The fused loss uses unordered FP32 atomic additions;
its determinism status remains **Not guaranteed**.

The test treats only this case and one-byte loss replay mismatch as an
expected failure on SM103. On SM100, the H32 fused forward batch case also
reports `packed slot 0 loss: 1/4 bytes differ` under CUDA 13.4. Restoring the
old expectation handler reproduces that failure on the second isolated run.
That case has a separate one-byte scalar-loss exception. Numerical,
non-finite, gradient and other-case checks remain strict. No kernel,
tolerance or golden changes.

On SM100 with cuDNN 9.30.0.12, Torch 2.11.0+cu128 and CuTe DSL
4.9.0a0+20261121230000.ecb3b50, the harness suite, eight expectation checks
and H32 replay/batch cases report 37 passed, 1 skipped and 2 xfailed in
31.45 s. The skip requires two CUDA devices; both xfails are known
SM100 byte differences. Command from the repository root:

```bash
python -m pytest test/python/test_conftest_parallel.py \
  test/python/simplified_sparse_attention/cutedsl/test_dense_indexer_kl_determinism.py \
  -k 'test_conftest_parallel or test_known_byte_failure_scope or bf16-h32-tail' -q --tb=short
```

Disabling the protocol lock and restoring the old SM103 expectation makes
five regression checks fail. The same 13 checks pass with the changes.

With Torch 2.14.0a0+b2c75dd062.nv26.09 / CUDA 13.4 and the same SM100,
backend and DSL, both MoK suites pass with four workers sharing one GPU:
37 passed, 1 skipped. The wider initial run exposed the SM100 loss-byte
case above. After its scoped expectation change, the full harness and
Phase 1 suites report 50 passed, 7 skipped and 11 xfailed in 58.40 s:

```bash
python -m pytest test/python/test_conftest_parallel.py \
  test/python/simplified_sparse_attention/cutedsl/test_dense_indexer_kl_determinism.py \
  -n 4 --dist loadfile -q -rsx --tb=short
```

The seven skips are two existing determinism exclusions, four missing
runtime goldens and the two-device guard. The 12 expectation-policy checks
run within this result. All xfails remain byte-comparison failures.

Native SM103 pipeline 85191475 stopped before tests because Enroot pulled
the private image anonymously and received HTTP 403. Pipelines 85192768
(cuDNN 9.28.1.4 / CuTe DSL 4.8.0) and 85192784 (9.30.0.9 / 4.9.0a0)
retry the affected suites with the public PyTorch 26.09 base image. Both
4.8.0 jobs import the image, then fail before pytest: Enroot cannot mount
the workspace (`Operation not permitted`), and `scontrol` cannot read
`/etc/slurm/slurm.conf`. The 4.9.0a0 attempt fails at the same mount step.
Native SM103 execution remains unverified. The version overrides differ
from the default job names.
