# NVFP4 QAT backward backend comparison

Run from an installed FE checkout with CUDA-enabled PyTorch, Triton >=3.7
and CuTe DSL >=4.7.0 on SM100:

```bash
python benchmark/nvfp4_attention_qat/benchmark_backends.py --output results.json
python benchmark/nvfp4_attention_qat/benchmark_backends.py --head-chunk 1 --output chunked.json
```

Both arms use `cudnn.Nvfp4AttentionQatBackward` with the same Q/K/V/dO and
matching forward auxiliaries. Explicit `backend="triton"` is the reference.
The candidate defaults to `backend="auto"` and must resolve to FROST; use
`--candidate-backend frost` to force it. Both requested and selected routes
are recorded. This controlled benchmark fails if auto selects Triton instead
of timing Triton against itself. Correctness checks
must pass before timing: finite values, pointwise atol/rtol 0.005, and
relative L2 below 0.01 for each gradient. Changed-dO graph replay poisons
both arms' outputs/workspaces before replay to detect stale intermediates.
Failure writes `status="fail"`, omits subsequent timing, and exits nonzero.
The output path must not already exist.

Each CUDA Graph contains three complete backwards. Five alternating
ABBA/BAAB rounds yield ten timing samples per arm. Reported speedup is the
ratio of per-arm medians, not a sum of independently timed kernels.
Compilation, allocations, reference forward construction and validation
are excluded; preprocessing and both FROST kernels (dK/dV, dQ) are included.
Source hashes, versions, hardware, workspace sizes and raw samples are saved.

## Current implementation: dK/dV kernel + dQ recompute kernel (2026-09-10)

Measured on the two-kernel implementation (`db863604a` plus later
workspace-plumbing and comment-only changes; exact source hashes are in the
raw artifacts) on a pre-production SM100 board: 148 SMs, reported as
"NVIDIA Graphics Device", not a B200 product board. torch 2.13.0+cu130,
Triton 3.7.1, CuTe DSL 4.7.0. B1/D128 BF16, dense noncausal, equal sequence
lengths, `head_chunk=0`. Raw artifacts:
[H3](results/sm100_two_kernel_20260910_h3.json),
[H12](results/sm100_two_kernel_20260910_h12.json).

| Heads | Sequence | Triton ms | FROST ms | Speedup | Workspace bytes (both) |
| --- | --- | --- | --- | --- | --- |
| 3 | 8192 | 1.03287 | 0.54076 | 1.910x | 18,972,672 |
| 3 | 32768 | 16.94992 | 6.88940 | 2.460x | 75,890,688 |
| 12 | 8192 | 4.12633 | 1.82194 | 2.265x | 75,890,688 |
| 12 | 32768 | 62.74596 | 28.12887 | 2.231x | 303,562,752 |

dV was bitwise equal to Triton in every case; dQ/dK relative L2 was about
0.0032 (scale folded before the BF16 dS store, as before). The FROST
workspace equals Triton's, `3*H*S*128*2 + H*S*4` bytes, and FROST results
are bitwise reproducible run to run (fixed accumulation order, no atomics).
This board is power-limited under sustained load, so the 32K samples spread
more (Triton H3: 15.3 to 21.5 ms) than the 8K samples; treat the 32K ratios
as indicative. These are complete backward component measurements, not
model E2E.

## Historical snapshot: 2026-09-10, original single-kernel design

The raw JSON is preserved from before the public backend was named FROST.
Its candidate key `cutedsl` refers to the same implementation now selected
with `backend="frost"`; it is not a separate backend or a new measurement.
New runs use `frost` for the candidate key. The top-level `cutedsl` version
field still identifies the implementation dependency.

This snapshot measured implementation `c042524b1`: one FROST dV/dS kernel
that stored BF16 dS to a workspace, followed by dQ/dK GEMMs. That design
was replaced by the two-kernel implementation measured above.
NVIDIA B200, 148 SMs; B1/H3/D128 BF16, dense noncausal, equal sequence
lengths, all-head dS. FE base `8059fdf490edb4de838bb2df7096d0b39b4c8336`
(merged PR #778); exact source hashes are in
[the raw artifact](results/b200_20260910.json).

| Sequence | Triton ms | FROST ms | Speedup |
| --- | --- | --- | --- |
| 8192 | 1.02237 | 0.42394 | 2.412x |
| 32768 | 13.86795 | 5.09393 | 2.722x |

These are **complete backward component measurements, not model E2E**.
dV was bitwise equal for these inputs; dQ/dK relative L2 was about 0.0032.
The latter fold scale before the BF16 dS store, while Triton applies scale
after gradient accumulation. Gates are unchanged; results are not a promise
of bitwise agreement for arbitrary data.

That design needed 6,518,341,632 bytes of explicit scratch at 32K/H3 (dS
alone 6 GiB) versus 75,890,688 bytes for Triton. The current implementation
has no dS workspace; its scratch equals Triton's.

[Memcheck artifact](results/b200_memcheck_20260910.json) and
[log](results/b200_memcheck_20260910.log): H3/head_chunk1 at S256 and S8192,
eager plus changed-input graph replay, zero errors. S256 is correctness
coverage, not a DiT performance target. This is not racecheck/synccheck.

Additional local coverage: 23 passing focused tests (15 existing Triton,
8 new FROST) including numerical reference, explicit nondefault stream,
runtime scale, rejected tails/contracts, workspace size/alignment, no
execute-time compilation/JIT dispatch, allocation/copy detector, and CUDA
Graph replay. The allocation detector is exercised with a positive failure
control. No cross-device/multi-GPU or built-wheel test is claimed.

An expanded run including the repository-wide symbol-prefix audit reported
24 passed / 1 failed: the failing global test lists 19 definitions that are
byte-identical to the merged base, outside this change. The new QAT kernel
passes that AST guard after moving its unchanged name-registration call
immediately after the kernel definition (a red/green guard check was run).
That naming-placement-only cleanup follows measured commit `c042524b1`;
it does not change the kernel arithmetic or launch configuration.

At rename commit `eee52baf0`, all 24 focused B200 tests passed
(the same 23 cases plus a backend-name regression). That run verified the
then-current Triton default and rejection of the draft-only `cutedsl` selector
in both the class and wrapper. The FROST numerical tests also verified that
the prepared implementation came from the FROST module.

Initial FROST support intentionally declines B>1, causal, GQA, unequal
lengths and non-256-aligned lengths. Native tails, broader adversarial
coverage and multi-device testing are follow-ups;
Triton's implementation and broader support remain unchanged. The public
default is now `auto`, preferring FROST and selecting Triton when FROST cannot
serve the declaration. The historical timings above used explicit backends;
changing dispatch does not create a new measurement.

## Automatic dispatch validation

At the automatic-dispatch commit `20238857a`, all 45 focused B200 tests
passed (15 original, 12 FROST, 18 dispatch cases). Coverage includes forced-route isolation,
old-DSL fallback before kernel import, supported-architecture selection
(SM103/SM120/SM121 probes are metadata-only, not execution on those GPUs),
batch/tail/causal rejection, wrapper cache separation, workspace-budget
chunk selection and execution, and propagation of compiler/launch errors.
The default-selection regression was first observed failing against
`eee52baf0` before implementing the new policy.

A separate process using an actual CuTe DSL **4.6.2** installation reported
**27 passed, 18 skipped**: FROST-specific cases were not executed, while the
Triton/default-fallback cases passed. No QAT FROST kernel module was imported.
Base package dependencies are still required; complete absence of `cutlass`
is not covered by this compatibility claim.

[Auto route evidence](results/b200_auto_20260910.json) covers H3/D128 at
8K and 32K: requested `auto`, selected `frost`, all-head workspace, eager and
changed-input graph comparisons passed. These are check-only runs, not new
timings. Historical performance and memcheck artifacts are unchanged.
The controlled benchmark's route guard was exercised on real DSL 4.6.2:
it exited nonzero with `status="fail"`, reported the Triton fallback reason,
and emitted no timing fields rather than comparing Triton against itself.
