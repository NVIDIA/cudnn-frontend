# FROST SDPA support

This is the current support-documentation entry point for FROST engines behind
the FE Python graph API. Start with your architecture below. Each page describes
the graph declarations its engines serve, including restrictions on combinations
of features. Backend cuDNN support and standalone APIs have their own contracts.

## Architecture pages

Migration is incremental: each refactor updates its own support section from
current code. **Pending documentation does not mean unsupported hardware or
features.** It means this new reference has not yet been checked for that scope.

| Architecture | Current reference | Documentation coverage |
|---|---|---|
| SM80 (cc 8.0) | [SM80 support](sm80.md) | FP16/BF16 forward documented; backward pending migration |
| SM90 (cc 9.0) | [SM90 support](sm90.md) | Forward documented; no FROST SDPA backward engine |
| SM100 / SM103 (cc 10.0 / 10.3) | [SM100 / SM103 support](sm100.md) | Forward and D192/V128 FP16/BF16 accumulation backward documented; other backward paths pending migration |
| SM107 (cc 10.7) | [SM107 support](sm107.md) | FP16/BF16, per-tensor FP8 and MXFP8 forward documented, including supported head-width envelopes; backward pending migration |
| SM120 / SM121 (cc 12.0, 12.1) | [SM120 support](sm120.md) | FP16/BF16, FP8 and MXFP8 forward documented; backward pending migration |

For a pending section, consult the registered [forward](../../fwd/engines.py) and
[backward](../../bwd/engines.py) `ENGINE_SPECS`, their declaration checks and the
[manifest](../../../engines/manifest.py). The
[legacy tracker](../SUPPORT_MATRIX_TRACKER.md) remains a historical reference;
its claims may be stale or conflict with later entries.

## Reading a support page

- **Supported** means an engine serves the stated graph domain, subject to all
  listed restrictions. Feature columns are not independent promises: paging,
  THD, masks, packing and split-KV may have combination limits.
- **Native** means the implementation has storage and contraction for that head
  geometry. **Envelope-served** means a larger native geometry covers the heads
  with padding, at that larger geometry's cost. Neither is a speed guarantee.
- **Discovery** describes whether the engine participates in ordinary candidate
  generation or requires opt-in. Eligibility does not promise the engine ranks
  first, or that every legal configuration is proposed automatically.
- **Qualification** describes what was tested on specific hardware and runtimes.
  It does not follow from registration or successful cross-compilation. See the
  [determinism reference](../../../../../docs/operations/determinism.md) for
  repeatability and batch-invariance scope.

The [attention API](../../../../../docs/operations/Attention.md) and
[graph execution contract](../../../../../docs/utilities/python_graph_and_execution_backends.md)
define the public interface. These pages describe the FROST subset of that
interface; callers use the same graph preparation and execution APIs.

## Updating or adding a section

1. Copy [TEMPLATE.md](TEMPLATE.md) to `sm<architecture>.md`, or update the existing
   architecture page. [SM90](sm90.md) is a completed example. Create only the
   sections you have checked; mark the remaining pass or precision sections
   **Pending migration**. Link the page and its exact coverage in the table above.
2. Read the registered capabilities, declaration checks, lowering and binding
   constraints together. Write the current supported domain and its important
   exclusions. Link the enforcing source symbols and relevant accept/reject tests.
   Do not infer support from an old tracker row or copy its chronology.
3. Update the affected section in the same commit as an eligibility change or an
   engine addition/retirement, as required by [SDPA Rule S2](../../AGENTS.md#hard-rules).
   A refactor that preserves support still updates engine identities and source
   links when they move. Keep multiple contributors' changes in separate pass or
   precision sections of the shared architecture page.
4. Keep this a reference to current behavior. Development notes, PR histories,
   temporary commit IDs, test pass counts and benchmark results belong in their
   existing reports or work logs. Link qualification evidence instead of copying
   it. Stop appending updates to the legacy tracker.

No architecture page is required to migrate unrelated sections in one change.
Remove a pending label only after that section has been checked against its
implementation and tests.
