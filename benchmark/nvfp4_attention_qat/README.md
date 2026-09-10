# NVFP4 QAT backward backend comparison

Run from an installed FE checkout with CUDA-enabled PyTorch, Triton >=3.7
and CuTe DSL >=4.7.0 on SM100:

```bash
python benchmark/nvfp4_attention_qat/benchmark_backends.py --output results.json
python benchmark/nvfp4_attention_qat/benchmark_backends.py --head-chunk 1 --output chunked.json
```

Both arms use `cudnn.Nvfp4AttentionQatBackward` with the same Q/K/V/dO and
matching forward auxiliaries. The default Triton backend is the reference.
The candidate explicitly selects `backend="cutedsl"`. Correctness checks
must pass before timing: finite values, pointwise atol/rtol 0.005, and
relative L2 below 0.01 for each gradient. Changed-dO graph replay poisons
both arms' outputs/workspaces before replay to detect stale intermediates.
Failure writes `status="fail"`, omits subsequent timing, and exits nonzero.
The output path must not already exist.

Each CUDA Graph contains three complete backwards. Five alternating
ABBA/BAAB rounds yield ten timing samples per arm. Reported speedup is the
ratio of per-arm medians, not a sum of independently timed kernels.
Compilation, allocations, reference forward construction and validation
are excluded; preprocessing, dV/dS, and dQ/dK GEMMs are included.
Source hashes, versions, hardware, workspace sizes and raw samples are saved.

## Measured snapshot: 2026-09-10

NVIDIA B200, 148 SMs; B1/H3/D128 BF16, dense noncausal, equal sequence
lengths, all-head dS. FE base `8059fdf490edb4de838bb2df7096d0b39b4c8336`
(merged PR #778); exact modified-source hashes are in
[the raw artifact](results/b200_20260910.json).

| Sequence | Triton ms | CuTe DSL ms | Speedup |
| --- | --- | --- | --- |
| 8192 | 1.02237 | 0.42394 | 2.412x |
| 32768 | 13.86795 | 5.09393 | 2.722x |

These are **complete backward component measurements, not model E2E**.
dV was bitwise equal for these inputs; dQ/dK relative L2 was about 0.0032.
The latter fold scale before the BF16 dS store, while Triton applies scale
after gradient accumulation. Gates are unchanged; results are not a promise
of bitwise agreement for arbitrary data.

Memory tradeoff: at 32K/H3, the candidate's total explicit scratch is
6,518,341,632 bytes versus 75,890,688 bytes for Triton. dS alone is 6 GiB.
`head_chunk=1` reduces dS to 2 GiB, not total process memory. This table
does not measure that chunked configuration.

[Memcheck artifact](results/b200_memcheck_20260910.json) and
[log](results/b200_memcheck_20260910.log): H3/head_chunk1 at S256 and S8192,
eager plus changed-input graph replay, zero errors. S256 is correctness
coverage, not a DiT performance target. This is not racecheck/synccheck.

Additional local coverage: 23 passing focused tests (15 existing Triton,
8 new CuTe DSL) including numerical reference, explicit nondefault stream,
runtime scale, rejected tails/contracts, workspace size/alignment, no
execute-time compilation/JIT dispatch, allocation/copy detector, and CUDA
Graph replay. The allocation detector is exercised with a positive failure
control. No cross-device/multi-GPU or built-wheel test is claimed.

Initial CuTe DSL support intentionally declines B>1, causal, GQA, unequal
lengths and non-256-aligned lengths. Native tails, broader adversarial
coverage, multi-device testing and further workspace reduction are follow-ups;
the Triton default and its broader support remain unchanged.
