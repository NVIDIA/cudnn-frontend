# GPU verification approaches relevant to F06

Research snapshot: 2026-11-15 UTC. These are primary-source descriptions,
not results from running each external tool on F06. The prototype in this
folder was run locally; no external verifier or GPU sanitizer was run on F06
as part of this change.

## What people are building

| Effort | Method and useful guarantee | Relevance and boundary for FROST |
|---|---|---|
| [FlashInfer batch-invariant kernels](https://github.com/flashinfer-ai/flashinfer/blob/9c50996794564e1e3b2a4b0683e9aa2bcc528df8/flashinfer/batch_invariant/README.md) | Fixed ordered arithmetic terms, Lean models, configuration classes and device conformance; fingerprints identify the numerical contract | Strong precedent for numerical attributes. The latest MegaMoE tier reuses its existing kernel. Source/lowering/device conformance still bridges the proof to execution; its model explicitly excludes CUDA synchronization and remote communication |
| [ProofWright](https://arxiv.org/abs/2511.12294) (2025; revised March 2026) | Agents construct VerCors safety annotations and Rocq semantic proofs; reports safety verification for 74% of generated KernelBench L1 kernels, with functional equivalence for a class of elementwise kernels | Let agents produce proof obligations, but use independent specifications and mechanical checking. Its benchmark success rate is not a prediction for asynchronous attention |
| [Logos CUDA/PyTorch translation](https://www.logosresearch.ai/blog/verified-gpu-kernels/) and [Lean/Velvet + VerCors paper](https://arxiv.org/abs/2610.24548) (August/October 2026) | Deterministic frontend translation, floating-point semantic reasoning in Lean, opaque transcendental functions, separate safety verification in VerCors | Similar to our ordered-expression bridge, with a stronger translation infrastructure. The supported CUDA subset and safety-to-sequentialization assumptions must be checked before adapting to CuTe/TMEM |
| [Racer](https://arxiv.org/html/2610.28429v1) (October 2026) | PTX symbolic execution and output-equivalence checking, shape reduction, synchronization diagnostics, and numeric refutation of unresolved expressions | Useful direction for detecting indexing/masking mistakes that tolerance tests hide. Its paper restricts equivalence to checked configurations/output assignments; unsupported operations and Unknown results remain explicit. It is not an unrestricted bitwise or all-schedules F06 proof |
| [Formalizing PTX in Lean](https://arxiv.org/html/2611.04447v1) (November 2026) | PTX execution candidates, memory-model predicates and machine-checked partial correctness with constructive non-vacuity; includes a pipelined Hopper WGMMA example | Closest direction for proving async protocol semantics. Examples use handwritten PTX and real arithmetic. Blackwell `tcgen05` and cluster extensions remain future work; termination and ISA undefined-behavior freedom are not proved. Its FlashAttention example is scalar, not a proof of optimized FlashAttention |
| [WarpDRF](https://arxiv.org/html/2610.07541v2) (October 2026) | Formalizes warp participation and memory-race obligations in Rocq, with a Faial-based analysis | A useful model for separating participation from ordering. Its language explicitly excludes async copies, atomics and cross-block synchronization, so it cannot directly certify this pipeline |
| [GPUMC](https://arxiv.org/abs/2505.20207) (2025) | Stateless exploration under scoped-RC11 for races, barrier divergence and assertions | Good pattern for small counterexample-producing protocol models. A scoped-RC11 model is not automatically the PTX async/proxy/tensor-memory model |
| [Triton ConSan](https://github.com/triton-lang/triton/blob/1c53e566ef8f719ff9b69132a72ad943e5409492/include/triton/Dialect/TritonInstrument/IR/TritonInstrument.md) (current compiler implementation) | Instruments IR with runtime visibility tracking for shared/TMEM accesses, mbarriers, async peers, commit groups and proxy effects | The most directly relevant engineering example for coverage of TMA/TCGen5 and cluster recipients. It is dynamic checking in Triton's IR; CuTe DSL needs an appropriate lowering/instrumentation integration, not just a runtime flag |
| [GPUVerify](https://github.com/mc-imperial/gpuverify) | Established static CUDA/OpenCL race- and divergence-freedom analysis | Useful architecture and invariants background; do not assume modern async instructions are covered |
| [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) | Dynamic bounds, synchronization, initialization and shared-memory hazard checks; compiler-assisted TMEM access guardrails | Keep as independent runtime evidence. A clean sampled execution is not universal race freedom; TMEM allocation/bounds guardrails are not a proof of the TMEM reuse protocol |

[FlashProve](https://assets.claude.com/a492b1fea80e6dcfe4e908ed77bd44961f8c02c2.pdf)
(November 2026) solves a related but different problem: cryptographic
verification of claimed model inference, including post-hoc proof generation.
That does not establish that the original serving kernel had no memory race.
The distinction matters when comparing products described as "provable
inference."

## Recommendation for existing kernels

Keep existing implementations and attach small independent specifications.
Start with a finite, explicit profile and an honest list of excluded paths.
There is no need to rewrite an attention kernel just to add verification;
rewriting also creates a new implementation that needs validation.

Use three complementary evidence paths:

1. **Numerical computation:** ordered primitive traces, exact term comparison,
   supported configuration equivalence and raw-bit device tests. Avoid
   silently substituting real arithmetic for the requested floating-point
   property.
2. **Ownership and synchronization:** allocation/alias maps, aggregate reader
   sets, generation-indexed credits, async completion and proxy visibility.
   Prove reusable protocol invariants, then connect actual lowered operations
   to them. Dynamic instrumentation and bounded counterexamples help test
   the abstraction before investing in full proofs.
3. **Function and indexing:** independent attention/masking specifications,
   bounds and unique-output ownership. A correct reduction trace can still
   read the wrong Q/K/V addresses or store to the wrong output row.

The important engineering investment is the **bridge**, not a larger count of
Lean theorems. An agent should propose annotations, invariants and proofs;
the reviewable specification, fail-closed extraction, proof checker and
independent negative controls should decide what evidence is accepted.

The current [lifecycle pilot](LIFECYCLE.md) deliberately stops before claiming
that bridge is verified. The abstract protocol theorem and the source
regression checks are reported separately. Extending this into a useful
kernel qualifier should preserve that distinction until an actual refinement
proof closes it.

## Priorities after this pilot

- First extract the O staging protocol from emitted IR/PTX. It has a small
  ownership boundary and an explicit TMA read drain.
- Add empty-work, phase-wrap and exit schedules, plus source-to-IR mutation
  tests. A missing gate should produce a concrete unordered-conflict trace.
- Extend the same event vocabulary to Stats/S/P aliasing and MMA-backed K/V
  credits, including all CTA recipients and consumer counts.
- Reuse protocol lemmas across FROST engines. A change to a buffer layout,
  synchronization primitive or specialization invalidates only the affected
  obligations, while numerical equivalence remains a separate qualification.
- Keep proof execution optional for runtime users and reproducible for
  development. Promote public numerical/safety attributes only when the
  complete selected-plan claim has evidence, rather than treating a local
  helper proof as an engine-wide property.
