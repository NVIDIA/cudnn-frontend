# SM90 KDA FP32 P: H20 precision and cost

Baseline: `8348887bd580770ec2dd2339ec919f06dbaf8702` (`develop`). The unchanged baseline host and CUDA source were SHA256-checked against that commit. Candidate source hashes and raw samples are in [the accompanying JSON records](results/kda/h20_fp32_p/).

Tested on H20-3e, driver 570.133.20, PyTorch 2.9.1+cu128, CUDA Toolkit 12.8, cuDNN 9.10.2, and CuTe DSL 4.8.1. Inputs are generated on CPU from fixed seeds. SM100 is unchanged and was not retested.

## What changes

The reverse recurrence adds `P = Q_g^T dO` into an FP32 state gradient. Its Tensor Core accumulator was FP32, but storing P in BF16 discarded low bits before that addition. P now stays FP32 in global scratch and is loaded into registers early in the scan. BF16 Tensor Core input operands, the inverse, saved state operands, and other BF16 intermediates retain their existing precision.

The scan no longer stages P in shared memory. Dynamic shared memory falls from 198,144 to 148,992 bytes per CTA (48 KiB); the unchanged 1 KiB static allocation is separate. The seed is staged into its existing MMA-operand buffer before the aliased seed storage is reused by prefetch. Its recurrent accumulator remains FP32.

## Precision

The focused case has unit keys on one coordinate, queries on an orthogonal coordinate, `g=0`, and exactly representable BF16 upstream gradients. It produces `P=1+2^-10` and an incoming FP32 gradient of `-1`. The correct residual is `2^-10 = 0.0009765625`. Baseline returns zero; FP32 P returns the exact residual. All 10 shape/scan variants fail on the baseline and pass after the change.

The following are relative L2 errors against an independent FP64 token recurrence, in percent; each cell is **baseline → FP32 P**. H=2 and K=V=128. All eight tensors are finite and byte-identical across fresh-allocation replays.

| Input | O | Final state | dQ | dK | dV | dG | dBeta | dInitialState |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ragged | 0.4164 → 0.4164 | 0.3213 → 0.3213 | 0.4011 → 0.4011 | 0.4076 → 0.4075 | 0.3630 → 0.3627 | 0.3577 → 0.3572 | 0.3355 → 0.3371 | 0.1235 → 0.1028 |
| weak | 0.4221 → 0.4221 | 0.3245 → 0.3245 | 0.4010 → 0.4010 | 0.4753 → 0.4623 | 0.3665 → 0.3505 | 0.4360 → 0.4202 | 0.3584 → 0.3217 | 0.3027 → 0.2568 |
| steep | 0.4178 → 0.4178 | 0.2029 → 0.2029 | 0.2193 → 0.2193 | 0.2517 → 0.2517 | 0.3181 → 0.3181 | 0.6891 → 0.6891 | 0.2225 → 0.2225 | 0.2394 → 0.1646 |
| correlated | 0.6296 → 0.6296 | 0.4497 → 0.4497 | 1.1711 → 1.1711 | 0.3322 → 0.3322 | 1.0410 → 1.0410 | 4.3598 → 4.3598 | 1.1784 → 1.1784 | 0.1417 → 0.0957 |
| long | 0.4495 → 0.4495 | 0.3577 → 0.3577 | 0.4298 → 0.4298 | 0.4678 → 0.4536 | 0.4271 → 0.4116 | 0.4675 → 0.4553 | 0.4111 → 0.3966 | 0.2865 → 0.2244 |

Ragged lengths are `[0,1,15,16,17,63,64,65]`; weak decay is T=257 with g=-1e-6; steep decay is T=65 with g=-5; correlated keys use T=65 with two repeated nonzero coordinates and g=0; the long random case has T=1024. These measurements isolate one rounding removal. The correlated case still has a larger dG error: this PR does not claim to fix all SM90 precision differences, establish loss parity, or replace a distributed training test.

## Performance and memory

CUDA-event latency for prepared launches captured in CUDA graphs; compilation, allocations, FP64 references, and Python autograd dispatch are outside timing. Each sample averages 20 replays; the table reports the median of 9 groups with alternating baseline/candidate order. Both implementations use the same inputs and output addresses, with separate scratch. F+B includes the unchanged forward kernel and the complete backward chain.

| THD workload | Backward ms, baseline → fix | F+B ms, baseline → fix | F+B change | Workspace MiB, baseline → fix | Public F+B peak GiB, baseline → fix |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1k-h8 | 0.122 → 0.132 | 0.427 → 0.437 | +2.27% | 25.06 → 29.06 | — |
| 16k-h8 | 1.570 → 1.713 | 6.459 → 6.603 | +2.23% | 401.00 → 465.00 | 0.928 → 0.990 |
| 128k-h8 | 12.196 → 13.372 | 52.558 → 53.737 | +2.24% | 3208.02 → 3720.02 | 7.397 → 7.897 |
| 128k-h32 | 43.420 → 47.688 | 86.479 → 90.733 | +4.92% | 12832.02 → 14880.02 | 29.588 → 31.588 |
| 128k-thd22-h8 | 11.550 → 13.233 | 21.478 → 23.161 | +7.83% | 3225.16 → 3742.41 | 7.466 → 7.970 |
| ragged-h8 | 0.458 → 0.514 | 1.192 → 1.246 | +4.55% | 112.21 → 130.71 | — |

128K is 131,072 tokens. The 22-document workload is 20 × 256-token documents plus 2 × 62,976-token documents. The small ragged case is `[0,1,65,257,4097]`.

A separate five-replay CUDA profiler sample at 16K/H8 attributes the increase to prep (428.98 → 509.59 µs) and scan (538.51 → 599.50 µs); the final backward kernel is unchanged within noise (609.65 → 608.96 µs). This is consistent with the larger P stores/loads and additional live scan registers. The measured fix is a precision tradeoff, not a throughput improvement.

Workspace increases by `max(ceil(T/64)+N-1,1) * H * 128 * 128 * 2` bytes. The shape-only chunk-slot upper bound reserves space for ragged tails. At 128K/H8/N1 this is exactly 512 MiB; H32 adds 2 GiB. Public peak allocation is measured separately in fresh baseline/fix processes with `torch.cuda.max_memory_allocated()` around the warmed public forward + autograd call, including fixed inputs, cloned differentiable leaves, outputs, gradients, and transient workspace. It excludes allocator-reserved free blocks, driver memory, and CUDA graph pools; it is not a full training/model-memory figure.

## Reproduce

Use an unchanged baseline checkout and this branch with its built `cudnn` Python binding. Set `PYTHONPATH` to this branch, and verify `cudnn.__file__` before running. In the commands below `$BASELINE` names the unchanged checkout.

```bash
python benchmark/linear_attention/benchmark_sm90_kda_precision.py --baseline "$BASELINE" --mode baseline --output baseline-precision.json
python benchmark/linear_attention/benchmark_sm90_kda_precision.py --baseline "$BASELINE" --mode fp32 --output fp32-precision.json
python benchmark/linear_attention/benchmark_sm90_kda_precision.py --baseline "$BASELINE" --mode performance --output performance-final.json
python benchmark/linear_attention/benchmark_sm90_kda_precision.py --baseline "$BASELINE" --mode baseline --memory --output baseline-memory.json
python benchmark/linear_attention/benchmark_sm90_kda_precision.py --baseline "$BASELINE" --mode fp32 --memory --output fp32-memory.json
```

The sanitizer harness uses the shipped prepared backward launcher, five geometries covering every scan specialization, exact-sized scratch surrounded by 256-byte canaries, all six gradients, and ten CUDA graph replays per geometry:

```bash
for check in memcheck racecheck synccheck; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False compute-sanitizer --tool "$check" --error-exitcode 99 \
    python benchmark/linear_attention/check_sm90_kda_workspace.py
done
```

Public autograd tests remain separate: the CUDA 12.8 sanitizer reports handled `cuCtxGetDevice` probes on autograd worker threads as API errors, and its default expandable allocator fails `cuMemCreate` on this host. No API-error suppression is used in the prepared-launch run.

## Review and coverage

See [the determinism coverage record](../../test/python/determinism_goldens/README.md#sm90-cuda-kda-fp32-p--h20-validation-2026-11-11) for pytest counts, strict golden checks, and the 6,000-launch warm/poisoned stress test. The new golden records all forward outputs and all six gradients, after the independent FP64 reference check. FP16 and grouped-head backward are outside this CUDA provider’s contract; safe-gate backward and composed CP summaries remain unsupported. Nonzero FP32 boundary gradients are exercised, but no multi-rank communication or end-to-end model loss is measured here.

Compatibility audit: passed for the changed packed BF16 CUDA path, both scan widths, full/partial chunks, empty documents, nonzero state and final-state gradient, graph replay, and existing routing/layout/serving tests. No public signature or dispatch gate changed. Docstring audit: passed for the KDA wrapper, workspace sizing, and raw launcher.

Fence audit: P has one writer per element in prep and is read only after that kernel completes on the same stream; the graph preserves this dependency. P uses ordinary global accesses and has no shared ring to release. The retained W/KG/A/gn stages keep their cp.async commit/wait and CTA publication/reuse barriers. Generated SASS was inspected for all five scan specializations, including seed consumption before aliased prefetch and staged operand consumption before refill. The sanitizer/stress results supplement this ownership review. Memcheck and synccheck each report **0 errors**; racecheck reports **0 errors and 0 warnings** across all five geometries. The source-to-SASS seed audit is recorded in `results/kda/h20_fp32_p/seed_release_audit.json`.
