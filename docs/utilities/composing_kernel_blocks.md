# Composing multi-kernel blocks in Python

**The FE-OSS block APIs are experimental and subject to change.**

## What this is

A *block* is a model sub-layer served by several kernels behind ONE Python class, ONE workspace and ONE
`execute()` call — for example the [gated attention block](../fe-oss-apis/gated_attention_block.md):
projection GEMM, QK-RMSNorm + RoPE, SDPA, sigmoid gate, out projection. Some of its stages are FROST engines
reached through the graph API (`cudnn.pygraph` GEMM plans), some are CuTe-DSL kernels the block owns, and the
fusions between stages are compile-time specializations behind the same signature. This page records the rules
that make such a block composable, using the gated attention block as the worked example. It is the pattern to
copy for the next block.

## The shape of a block

```text
class <Block>Fwd(APIBase)
    __init__(sample tensors..., geometry, *, knobs)   -> check_support() per stage, compile() per stage
    get_workspace_size() -> int                       -> honest total of every scratch byte the launch touches
    execute(tensors..., out, workspace, ...)          -> launches only: no allocation, no host read, no conversion
```

- **Geometry, not model names.** The class takes a frozen geometry dataclass (`d_model`, `h_q`, `h_kv`, `d_head`,
  `rope_dim`, mask fields, ...) whose `validate()` raises `ValueError`. Model provenance lives in comments.
- **Sample tensors at build time** fix dtypes, layouts and shapes; `check_support()` runs every stage's typed
  decline (`NotImplementedError` for "not served", `ValueError` for a contract violation) so an unserved
  combination fails at construction, never at the first launch. A `Capabilities`-style row per stage is the
  source of truth for what is served; the standalone adapters mirror the same declines.
- **`execute()` allocates nothing.** Every intermediate is a strided view of the caller's workspace, carved
  into aligned regions at build time (`cudnn.frost.workspace.WorkspaceLayout`), and `get_workspace_size()` is the
  exact byte count. Nothing is read back to the host and nothing is converted, which is what makes a block
  CUDA-graph friendly with stable pointers.
- **Append-only signatures.** New parameters go at the end with defaults (the gated block appended `quant=`,
  then the MXFP8 scale-factor blobs, then `qk_norm` on the geometry, then the fp4 `W_o` scale blob
  `sample_w_o_sf` / `w_o_sf`); positional callers never break. The same rule applies to the spec dataclasses: a new
  mode is a new field with a default (`MxQuantSpec.w_qkvg_dtype`, `MxQuantSpec.o_fp4`), never a new positional
  argument or a second class.

## Mixing graph-API engines and DSL kernels

A block may drive a FROST engine through the graph API (build a `cudnn.pygraph` GEMM, pin the FROST plan by
name — plan names carry knobs, `frost_gemm[...]`, so match on the prefix — and execute it) next to CuTe-DSL
kernels it owns. Two contracts follow:

1. **One launch stream through both routes.** A graph-API plan launched through `plan.jit(vp, stream=)` runs on
   its `stream` argument; through `plan.graph.execute(vp, workspace, handle)` it runs on the handle's stream. DSL
   stages run on torch's current stream. A block resolves the stream once in `execute()` and passes it to every
   stage: the JIT route gets it directly, the graph route gets a per-`(device, stream)` cached `cudnn.Handle`
   with `set_stream()` (`gated_attention_block/kernels/proj_gemm.py::handle_for_stream`). Refusing an explicit
   stream argument is not enough — the ambient non-default stream of `with torch.cuda.stream(s)` must be honoured
   too, or the projections race the norm/SDPA stages. The block's test parks the default stream and runs on a
   side stream, comparing bitwise.
2. **The engine's workspace is part of the block's.** The GEMM plans' and the SDPA adapter's own scratch is
   carved from the same caller buffer (`engine_scratch` region), so the block still has one workspace.

## Fusions as knobs, not forks of the API

A fusion changes the partitioning behind the signature, never the signature. In the gated block:

- `fuse_norm_rope=True` moves stages (2)+(3) into the projection GEMM's epilogue (a fork of the shipped GEMM
  template with the norm/RoPE/quantize math on the fp32 accumulator);
- `fuse_gate=True` moves the sigmoid gate into the SDPA epilogue (a production feature of the Rubin d256 SDPA
  kernels, selected through `TemplateParams.epilogue_gate`).

Each knob value is a distinct compiled specialization; the default is the honest unfused pipeline, and the fused
paths are **feature-detected**: when the fork a knob needs does not exist for the requested dtype or shape, the
block declines with a typed `NotImplementedError` naming the missing piece instead of silently running the
unfused pipeline under a fused label.

Two structural facts decide where fusions can go: an out projection that contracts over all heads can never share
a kernel with an SDPA whose CTA owns one head (it needs a cross-CTA reduction), and any pointwise on the SDPA
output must run **after** the epilogue's dead-row substitution (a fully masked row is zeroed by a select; multiplying
accumulator residue by `sigmoid(gate)` before that select propagates NaN).

## Precision modes share the signature

bf16 / fp16, per-tensor FP8 (a `QuantSpec` of static scales) and MXFP8 (an `MxQuantSpec` plus scale-factor blobs)
select different stage sets — the quantized pipelines add quantize passes in the unfused form and fold them into
the projection epilogue in the fused form — but the caller sees one class and one `execute()`. The workspace
layout appends regions for new modes; existing offsets stay byte-identical.

Two further modes ride the same signature and show the pattern at its smallest:

- **fp4 weights.** `MxQuantSpec(w_qkvg_dtype=torch.float4_e2m1fn_x2)` swaps stage (1) onto the FROST GEMM's mixed
  MXFP8 x MXFP4 block-scale row. Nothing else moves: the weight arrives packed (`[N, K // 2]`, two e2m1 codes per
  byte) with its existing E8M0 scale blob, the stage list and the workspace are identical to MXFP8, and the one
  fusion the mode cannot serve (the fused projection fork is rendered for an e4m3 B) is a feature-detected typed
  `NotImplementedError`, so the test inverts the day the fork arm lands.
- **fp4 output.** `MxQuantSpec(o_fp4=Fp4Format.NVFP4 | MXFP4)` replaces the per-tensor tail (quantize `O`, FP8 out
  projection) with a block quantize of the gated `O` and the fp4 x fp4 block-scale out projection against an e2m1
  `W_o` whose scale blob is one appended argument (`sample_w_o_sf` / `w_o_sf`, required iff the mode is on). The
  format is an enum whose member IS the (codes, scale dtype, block) triple, so an illegal pairing cannot be spelled,
  and the two per-tensor scales that no longer have a consumer are pinned to `1.0` by a typed `ValueError` rather
  than silently dropped. The workspace appends the two new slots (`o4`, `sf_o`) at the end of the arm and drops the
  e4m3 `o8` it no longer writes; a frozen offset snapshot pins every other layout byte-identical.

Both are fields on `MxQuantSpec`, not constructor flags: a mode that only makes sense inside one precision pipeline
lives on that pipeline's spec, so it is unrepresentable on the others instead of being one more decline to test.

## Measuring a block honestly

- **Baseline first, op by op.** Time the framework chain the block replaces (cuBLAS GEMMs, torch pointwise, torch
  attention) per op, cold (L2 flushed), with the achieved rate against the part's pinned peaks (MMA FLOP/s from
  the locked SM clock, HBM bandwidth from the bus width), and the share of the whole. That table is what the block
  is judged against and it is where the fusion order comes from.
- **A/B the thing the user runs.** Whole-block time, arms interleaved per round, a control pair (the baseline
  twice) in every table so noise has a number; resolved implementation printed per slot so two arms cannot
  silently resolve to the same kernel.
- **Perf numbers come from a perf node with a locked clock**, functional nodes are for correctness: a fixed
  in-kernel cost is a different fraction of a slower kernel, so a percentage taken on a dev node can have the
  wrong sign.

## Tests a block needs

- a **layout contract** test (column offsets, alignments, workspace regions, append-only positions);
- a pure-torch **reference oracle** with its own self-checks, and an **adversarial** test against a sequential
  emulation of the kernel's rounding points;
- **end-to-end** tests per precision and per fusion knob, including the padded / masked corner rows and the
  bitwise equality of the fused and unfused pipelines where the math is identical;
- **typed-decline** tests for every unserved combination (`pytest.raises(NotImplementedError | ValueError,
  match=...)`), never skips;
- a **stream-ordering** test on a side stream with the default stream parked.

Module basenames under `test/python/fe_api/<pkg>/` must be unique across the test tree (pytest's default import
mode has no packages there), so name the oracle `<pkg>_reference.py` and the tests `test_<pkg>_*.py`.
