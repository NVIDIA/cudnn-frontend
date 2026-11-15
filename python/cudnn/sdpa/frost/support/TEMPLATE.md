# Architecture support-page template

Copy the section below into `sm<architecture>.md`, replace the bracketed text,
and remove these instructions. Use [sm90.md](sm90.md) as a filled example.
Keep headings for unreviewed scopes and label them **Pending migration**.
Use **Unsupported** only when the current implementation actually declines that
scope. A page can be useful before every section is migrated.

Keep tables small: split forward/backward and FP16/BF16, FP8 or MXFP8 when their
contracts differ. Name exact compute capabilities, dtype combinations, head
domains and feature interactions. Avoid a checkbox for a feature whose limits
are hidden elsewhere. Link repository files and name the enforcing symbols;
avoid line-number links on a moving branch. Do not add a development diary.

---

# [Architecture] SDPA support

[Back to the support index](README.md)

Scope: [exact compute capabilities; graph API operations covered].
Documentation coverage: [which passes/precisions are checked; which are pending].
[Name any environment requirement stricter than the shared package requirement,
with a link to its enforcing check. Do not duplicate the package's version floor.]

## Forward

### [Precision, if more than one contract]

| Engine name / public ID | Head domain and native geometry | Layouts and feature limits | Discovery |
|---|---|---|---|
| [Registered identity] | [Exact pair(s), or bounded aligned envelope and the larger geometry it uses] | [Dense/THD/paged; decode; mask/length/packing/split combinations] | [Ordinary discovery or opt-in; relevant selection limits] |

[State Q/K/V/O types, accumulator/softmax restrictions and Stats contract. For
quantized paths, include scale formats and allowed mixed types. Define D_QK/D_V
and layout abbreviations when used. Keep a row's restrictions adjacent to it.]

### Combination and storage limits

- [Unsupported combinations, with exact boundaries; distinguish an unsupported
  graph from a supported graph that needs another engine.]
- [Mask/length requirements, ragged offsets and Stats layouts, page constraints,
  alignment and aliasing obligations that affect callers.]
- [Any difference in direct/standalone API scope; this page's table is for graph
  engines and must not silently inherit a wider standalone domain.]

## Backward

[Use the same compact table and restrictions for each reviewed precision.
If no implementation exists, say so and link the registered source.
If not reviewed yet, write **Pending migration**. Forward Stats alone never
imply backward support.]

## Plan selection and execution

[Explain discovery versus explicit public records, opt-in requirements and any
retired identities that callers must replan. Link the shared graph contract for
common preparation/execution semantics. State selected-plan behavior notes only
where that engine/configuration actually advertises them. Leave tuning recipes,
heuristic thresholds and performance comparisons in their own documentation.]

## Sources and validation

| Contract | Enforcing source / existing detector |
|---|---|
| Eligibility and native/envelope bounds | [Capabilities, mismatch/declaration checks and local config; source files + symbol names] |
| Identity and discovery | [Manifest and owner contract; identity/candidate tests] |
| Layouts, combinations and binding | [Lowering/binder; accept and reject tests] |
| Numerical/replay qualification | [Dedicated determinism/coverage document and relevant tests] |

[State material qualification limits without copying results: admitted hardware
not executed, cross-compilation only, fixed-plan scope, missing goldens, or
unverified combinations. An existing test is a detector, not proof that every
configuration passed on every device.]
