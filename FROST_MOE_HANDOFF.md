# Frost / FlashInfer MoE pathfinding handoff

## September 17 followup: formal paired FC2 projection engine

Small-row BF16 grouped down projections can now explicitly select engine20402
(`frost_moe_fc2_pair`). It computes both output-channel halves in the paired
MMA kernel and stores both directly. The supported graph is one untransformed
grouped projection with BF16 input/output, FP32 accumulation,1..8 routed rows,
output width divisible by128 and K divisible by64 on SM100. Canonical expert
weights may have nonoverlapping16-byte-aligned pitches. Unsupported transforms,
epilogues, dynamic graph dimensions and knob combinations decline at planning.
Ordinary engine20400 and paired SwiGLU engine20401 retain their meanings.

FI selects the new engine through the existing `fc2_tactic` interface. Execute
uses current input/weight/offset pointers and caller workspace/stream; it does
not repack, allocate, synchronize or compile. This followup needs no FI runtime
override or API change. The engine is an explicit candidate, not a heuristic
ranking claim.

The public FE suite passes31 collected tests under each of normal execution,
memcheck and racecheck on B200, zero skips:93 executions,360 independently
audited raw outputs,108 changed-reference pairs and36 actual FC2 routes. This
includes the retained FC1/metadata regressions, pitched storage, two captures,
live weights/tokens/offsets and allocation/compile/synchronization guards.

Actual full FI on B2001000W, BF16 T1/E128/top8/H2048/I768, now measures:

| Precomputed routing | Paired FC1 + ordinary FC2 | Paired FC1 + engine20402 | Latency reduction |
|---|---:|---:|---:|
| Unpacked | 32.888000 us | 32.192000 us | 2.116% |
| Packed | 32.884125 us | 32.208000 us | 2.056% |

These are same-source, same-card cold-L2 CUDA-Graph/CUPTI full-MoE spans from
fresh ABBA processes. Both routes pass normal/memcheck/racecheck, with296 raw
output checks,120 independently changed-reference controls and768 spans.
Two retained parent captures, live X/IDs/scales/FC1 and FC2 weights, and
independent legacy weights are covered. PDL intervals overlap; stage durations
must not be added. This is a synthetic model-shaped operator result, not model
E2E or a new competitor victory. It supersedes the earlier private FC2 override
as evidence for the real public graph path.

Evidence: local `artifacts/analysis1883/public.json` and
`artifacts/analysis1887/fc2.json` under `.fi-cudnn-work/20260914`.
The tested formal-engine archive SHA256 is
`814dee6cdeaea97e01f468e5fbf1979e892335bc66cbb5fba6efd806ac5e2047`.

The preceding published-source TRT comparison is now independently audited:
Frost/TRT T1=33.276/28.424125us and T64=214.043625/197.01975us,
or17.07%/8.64% Frost latency overhead. TRT kernels match the prior352-joint-
tactic winners by BMM symbols/cubin hashes. Both preparations are outside
timing; TRT BF16 includes gate/up row interleave, MMA row shuffle and BlockMajorK.
These measurements precede engine20402 and must not be combined with the table
above to infer a new TRT gap. Current parent-binding integration also passes
on RTX PRO6000 Blackwell Server600W:72 raw checks,36 poisoned-output controls,
36 changed-reference pairs and36 routes under all three modes. That is SM120
compatibility evidence; the earlier RTX5090 performance result remains separate.

The FC2 implementation retains credit to NVIDIA Frost, Kernel Factory624,
Yanqin Zhai's PR1090, NVIDIA CUTLASS example113, and canonical rank5 pairing/
early-PDL work. TRT layout motivated exploration; no TRT kernel body is copied.
New KF compact-resource and single-pass scheduler experiments remain outside
this publication until complete-FI validation. Full repository CI and a
performance roof are not claimed. Yanqin/Yihua can continue distilling these
same consolidated drafts.

---

## September 17: real paired MoE graph engine and FI parent binding

This followup carries the small-token path into the real graph API. Engine
20401 (`frost_moe_swiglu_pair`) recognizes two BF16 grouped projections and
FP32 SwiGLU over explicit slices of one canonical gate/up weight parent.
Engine20400 remains available. The specialization is limited to SM100,
1..8 routed rows, and positive N/K divisible by64; unsupported graphs and
knobs decline. Compilation occurs while building the plan. Execute binds the
current parent and caller workspace/stream without repacking, allocating,
synchronizing or compiling.

FI now declares the parent relationship only after checking the actual view
pointers, dtype, device, dimensions and strides. Its cache key distinguishes
true shared storage from independent tensors with identical strides. A
planning-only parent-graph decline retries the existing separate-input fused
FC1 graph, preserving weight views and explicit tactics. This retry does not
catch compiler or runtime errors. Actual old/new FE compatibility passed on
B200 under normal execution, memcheck and racecheck: 144 raw output checks,
72 poisoned-output negative controls, 72 independently changed-reference pairs
and 72 captured routes. The new FE binds the shared parent; the old FE declines
that graph and preserves the existing fused separate-input path.

The FE changes include Yanqin Zhai's merged
[PR #1090](https://github.com/NVIDIA/cudnn-frontend/pull/1090) and its regression
coverage, with the prior SM120 scheduler/shared-A path retained. The new paired
kernel derives from Kernel Factory optimizer624 plus the validated early-PDL
change, using NVIDIA CUTLASS example113 layout concepts and canonical rank5
pairing guidance. TRT-LLM's gated-row interleave motivated layout exploration;
no TRT-LLM kernel body is copied. The larger-token KF candidate and subsequent
ballot-prefix experiment are not included in this engine.

On a full B2001000W, BF16 T1/E128/top8/H2048/I768 with precomputed routing,
cold-L2 CUDA-Graph/CUPTI full MoE timing measured:

| Routing | Tuned ordinary Frost | Early-PDL prototype | Real FI graph engine20401 |
|---|---:|---:|---:|
| Unpacked |33.915625us|33.127875us|33.123625us|
| Packed |33.991375us|33.147250us|33.159500us|

The real integration is2.34–2.45% lower latency than ordinary Frost and within
0.04% of the prototype. These numbers are not an additional gain over the
previously reported KF/PDL improvement. No fresh TRT/CUTLASS or model comparison
is claimed. All arms use the same FI/FE source; preprocessing is outside replay.
The full-MoE audit passes414 raw BF16 comparisons,150 changed-input negative
controls and1152 timing spans, after normal/memcheck/racecheck for both routing
modes. Two retained captures, independent legacy weights, and live X/IDs/scales/
weight updates are covered. Overlapping PDL stage durations must not be summed.

The standalone paired graph engine separately passed180 raw outputs and54
negative controls under normal/memcheck/racecheck, including16-byte-aligned
pitched storage and six geometries. The actual collected public pytest port
now also passes all9 cases (3 metadata plus6 GPU) in each of normal execution,
memcheck and racecheck on B200:27 test executions, independently audited180 raw
outputs and54 changed-reference controls, with no skips or tolerance changes.

The earlier public-port failures are retained. Its first run called the DSL
version helper without a required argument; that test-only call was repaired.
A subsequent run hit `cudaErrorStreamCaptureInvalidated` (901) after correct
eager output. GC callbacks located collection inside MagicMock construction
during capture; original/force-GC diagnostic processes were1/0/0/0 exits.
The test now constructs simple allocation/JIT guards before capture and resets
retained graphs after use. Runtime kernels, default GC behavior, input mutations,
route assertions and numerical thresholds are unchanged. All three public-port
validation modes then passed. This fixes the test harness, not a kernel speedup.
The explicit-parent CPU regression is confirmed RED against the preceding FE
source (`no lowering for node type SLICE`).

Timing above used frozen pair1820/fi_pair1825. This publication additionally
carries the planning-only compatibility retry and SM120 source reconciliation,
validated separately; the final assembled head has not been retimed.

The earlier pending SM120 parent-binding gate and published-source TRT refresh
are completed in the followup above. An initial SM120 integration run failed before kernel execution
because its source copy lacked the already validated static scheduler; the
combined source restores it without relaxing the gate. Full repository CI is
not claimed. These remain consolidated pathfinding drafts for Yanqin/Yihua to
distill; new validation and improvements will be stacked on these branches.

---

Updated September 16, 2026. Companion drafts:
[cuDNN Frontend #1080](https://github.com/NVIDIA/cudnn-frontend/pull/1080) and
[FlashInfer #5250](https://github.com/flashinfer-ai/flashinfer/pull/5250).
Yanqin and Yihua can distill these drafts asynchronously. Validated followups
continue on the same branches; the owners can choose the eventual split.

This uses open-source Frost/CuTeDSL engine **20400**, with FlashInfer routing,
permutation and weighted finalization. It does not benchmark closed-source
cuDNN GEMM kernels. Frost needs explicit per-workload tuning; no heuristic
quality claim is made.

## Latest source changes

The September 16 followup carries the runtime/test source from frozen Frontend
`ea4b86dd9823ed5849d3d6f2994f26906b2f0d8f` and FlashInfer
`0fb8ffad58680b1dac03317d75f417f1aa5fd298`. Publication commits add this updated
handoff; these frozen IDs identify the tested source, not the published heads.

| Repository | Change | Purpose |
|---|---|---|
| Frontend | Async shared-memory lifetime fence before releasing an A/B stage | Fixes the ordinary-execution FC1 corruption found with the large SM120 tile; prevents the next TMA load from reusing a stage too early. |
| Frontend | Explicit SM120 static scheduling alongside dynamic scheduling | Allows tuning away atomic work distribution when a regular workload benefits; does not assume static always wins. |
| Frontend | Shared-A paired grouped GEMM on SM120, with epilogue and shared-memory capacity accounting | Enables gate/up projection and activation fusion, saving the FP32 intermediate roundtrip and separate activation launch. |
| Frontend | Multi-output, pitched-weight, multiwave and scheduler regression coverage | Checks the new fusion and scheduler contracts under live-input capture replay. |
| FlashInfer | Explicit fused/unfused BF16 FC1 route choice | Makes the fusion route selectable and replayable rather than silently conflating different graphs. |
| FlashInfer | Independent FC1/FC2 configuration domains, joint tactic replay, cache version 7 | Allows each stage to use an appropriate tactic and prevents reuse of stale tactic records. |
| FlashInfer | Public fused-path, capture and live-weight/interleave tests | Exercises the actual interface and changing inputs without local architecture overrides. |

Earlier work remains included: SM100 scheduler/reset improvements, absolute-A
addressing, shared-A fusion, aligned independent expert strides, explicit
`blocked_128x128_v1` weight layout, native FI helper integration, and capture-safe
prepared plans. The packed layout is an operation attribute, not a freely
selectable tuning knob. Packed E4M3 support is scoped to eligible SM100 tiles;
SM120 uses ordinary BF16 weights. SM121 and SM120 FP8 support are not added.

## Validation of the followup

Correctness precedes timing; no tolerance was relaxed. Source import paths and
hashes were recorded. Tests check live inputs/routing/scales, poisoned buffers,
captured replay and actual Frost routing.

| Target | Completed checks |
|---|---|
| RTX PRO 6000 Blackwell Server Edition, 188 SMs, 600 W | 8 shared-A FE cases, 42 FI public cases, 30 single-GEMM scheduler cases in normal execution; all 8 shared-A cases under unfiltered memcheck and racecheck; 2 mandatory-fusion public cases under each sanitizer. |
| RTX 5090, 600 W | 44 public cases in normal execution (42 existing plus 2 joint-tactic cases); 2 joint-tactic cases under unfiltered memcheck. |

Audit receipts: `analysis1055/result.json` and `analysis1188/result.json` under
the local experiment root
`/home/scratch.yanxu_libs/cudnn_frontend/.fi-cudnn-work/20260914`.

The formerly failing SM120 BF16 `256x256x64/warps2x4` FC1 case is repaired by
the included lifetime fence. On full RTX PRO 6000, T4096/E32/top2/H2048/I1024
with the retained skewed inputs, old source fails 1,643 FC1 elements while the
fence candidate passes 15 full-output and 75 stage checks at unchanged 0.02
tolerances (audit960). The repair additionally passed 12 ring and 40 public
cases in normal execution and both unfiltered sanitizers (audit993). Those
earlier fence-only gates are distinct from the expanded followup coverage above.

The original full public racecheck gate remains open: instrumented runs ran
out of host memory, including a 512-GiB allocation. A diagnostic that explicitly
synchronizes after each of 40 replays passes both modes, but changes execution
concurrency and therefore does not close the original gate. Earlier failures
and logs are retained. Full repository CI is not claimed green.

An earlier B200 captured native-sort / FP32 reference failure also remains in
the record. A standalone cooperative CUDA Graph kernel, importing neither
FlashInfer nor Frost, reproduces the subsequent `CUBLAS_STATUS_INTERNAL_ERROR`
under racecheck while its own output is correct. This narrows the interaction;
it does not turn the original full-MoE failure into a pass.

Run the focused FE cases from `test/python` and the FI cases from its root:

```bash
pytest -m L1 gemm/frost/test_moe_shared_a_sm120.py gemm/frost/test_moe_scheduler_sm120.py
pytest -q tests/moe/test_moe_cudnn_sm120.py tests/moe/test_moe_cudnn_bf16_joint.py
```

Install both matching branches, CuTe DSL >= 4.7, and verify `cudnn.__file__` /
`flashinfer.__file__`. Enable `CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1`; SM120 uses
`enable_pdl=False`. An editable install can silently select another checkout.

## Performance evidence and next experiments

Historical full-B200 FP8 results, E128/top8/H4096/I2048, cold-L2 CUDA Graph/CUPTI:
at T256, transferred tactics 526.522 us -> per-shape tactics 500.219 us
(5.00% lower latency), versus the measured TRT-LLM control at 519.523 us
(3.72% lower). T16 is near parity: Frost 225.129 us versus TRT-LLM 224.433 us.
These are earlier validated snapshots, not measurements of this assembled head;
Frost helper PDL was off and TRT-LLM PDL on. Preparation, logits/top-k, expert
communication and full model execution are excluded.

Separate latest RTX5090 research, BF16 T64/E128/top8/H2048/I768, precomputed
uniform routing, TP1/EP1, PDL off: four fresh same-card ABBA processes and 768
complete-MoE CUPTI spans measured 797.650 -> 775.778 us (-2.742%) for M64 compact
FC1; a later matched comparison measured 777.960 -> 773.547 us (-0.567%) for M32
padless FC1. The comparisons used different physical cards. Both include
additional experimental resource hooks that are **not in these product
commits**; do not attribute their timings to this PR head or compound the raw
latencies. FC1 accounts for roughly 67% and FC2 32% of the first candidate span.
Actual memory/tensor counters are being collected; a hardware roof is not proven.

Primary next direction: combine Yanqin's
[SM100 swap-AB #1090](https://github.com/NVIDIA/cudnn-frontend/pull/1090) with the
FI integration and independently choose FC1/FC2 orientation. An isolated merged
prototype resolves both textual conflicts and the compiled scheduler-reset
handoff. Its CPU render/replay checks and the first target-GPU screen pass.
The first screen covers all four orientations at one fixed geometry, not a
complete tile sweep. The experimental merge is not included in these commits.

#1090 currently declines SM120 MoE swap. SM120 benefit remains an independent
question; an existing FI MXFP8 swap-AB implementation offers a design reference,
but does not establish a speedup for Frost BF16. Continue promising ideas and
stack validated changes onto these drafts so reviewers can consume them
asynchronously. Negative results should retain shapes, routes and timings.

KF found a remotely faster component candidate, but a finite-BF16 stress case
exposed NaNs. A minimal parenthesization repair passes the targeted diagnostic
and the original normal/memcheck checks. The full racecheck reached its 3,300-s
bound (exit 124) without a completed artifact, so its gate remains incomplete;
fresh fixed-candidate timing is also required. No KF speedup is attributed to
the FI path.

## Acknowledgements and provenance

The grouped-GEMM extensions build on the existing NVIDIA Frost/CuTeDSL kernels
and FlashInfer MoE infrastructure; their original copyright/license headers
and implementation provenance are retained.

Yanqin Zhai (@yanqinz2) authored [Frontend #1090](https://github.com/NVIDIA/cudnn-frontend/pull/1090),
head `068ffd87f053543a63ddf929aeb87fcc8b231380`. Credit for its MoE swap-AB
lowering, separate plain/block-scaled SM100 templates, public knob replay and
coverage belongs to that work, with thanks to Yanqin and Yihua for the parallel
distillation effort. Our isolated combination adds FI wiring, conflict/reset
integration and matched measurements. #1090's code is in that experimental
combination, not this published runtime followup. The bounded B200 result below credits
that implementation; a later runtime publication must retain this attribution.

For SM120, the design reference is the NVIDIA CuTeDSL MegaMoE kernel team's
implementation in the `bangyus/cutedsl_megamoe` fork, vendored by FI at
`d19d30a748f9e402b8a2a33083fdf530231cf647`; see
`flashinfer/moe_ep/kernel_src/sm120/swapab_cutedsl_megakernel/VENDOR.md` in the
companion FlashInfer tree for the dirty-snapshot exceptions and original source.
A private SM120 BF16 prototype now references its accumulator-coordinate
reasoning, together with #1090's ragged-N design. No MegaMoE source body was
copied. Our adaptation adds BF16 operand binding and a direct FI-layout
epilogue. This prototype is outside the published runtime; component numerical
and sanitizer checks pass, but no SM120 speedup is established yet.

Kernel Factory assisted separate candidate exploration. Its exported candidate
and local numerical repair remain experimental and are not promoted by this
followup. Future gains will distinguish upstream contribution, adaptation,
integration and configuration search, with workload-specific evidence.


## Combined prototype: first B200 result (September 16)

BF16 T64/E128/top8/H2048/I768, synthetic Qwen-shaped inputs, precomputed uniform
routing, TP1/EP1, FI API `enable_pdl=False` (see qualification below),
full 148-SM/1000-W B200. At one fixed 128x64 geometry,
complete-MoE cold-L2 CUPTI spans are:

| FC1 swap | FC2 swap | Complete MoE, us |
|---|---|---:|
| No | No | 276.481 |
| No | Yes | **226.992** |
| Yes | No | 286.817 |
| Yes | Yes | 238.288 |

FC2-only swap lowers complete latency **17.899%** at this geometry. FC2 changes
117.680 -> 68.160 us; FC1 remains about142.6 us. Credit for the swap lowering
and templates belongs to Yanqin's #1090; this pathfinding work adds FI stage
selection, compiled-reset integration and measurement. This is evidence that
the two efforts combine usefully on this fixture, not a comparison between
independently tuned best configurations.

All four arms pass20 strict full-output checks and2 skipped-replay negative
controls before384 raw spans. Inputs/routing change and buffers are poisoned.
Independent audit1453 verifies exact source hashes, all spans and actual stage
routes. Synchronization validation, fresh-process repetition, tile tuning and a
tuned competing-backend comparison remain open. This experimental
merge is not included in this PR's runtime changes; no model-E2E gain is claimed.

## Qualification of experimental observations

The combined prototype's initial timing observations above do not establish a
production-ready performance gain. Successful synchronization validation and
confirmation against separately tuned configurations remain required before
promotion. No SM120 orientation-specific or complete-MoE benefit is claimed.

The PDL label refers to the public FI API flag; Frost's generated launchers
control their own PDL setting. It should not be read as all kernels disabling
PDL. These qualifications do not change the previously reported arithmetic
or the acknowledgements above.
