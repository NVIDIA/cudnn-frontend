# python/cudnn/sdpa — Agent Guide

SDPA-specific hard rules, on top of the package-wide Rules 1-5 in
[../AGENTS.md](../AGENTS.md). Numbered `S1, S2, ...` so reviews can cite them
without colliding with the package-wide numbering; the list grows — append,
never renumber.

## Hard rules

**Rule S1 — THD/packed Stats (LSE) must stay packed: token-major or
head-major, never dense-padded.**

- Consumers read Stats through the same `cu_seqlen` packing as Q/O: TE and
  Megatron take token-major `(T, H)` (cuDNN's TH1 recipe) natively, FA-style
  callers take head-major `(H, head_stride)`. A dense-padded declaration
  (per-sequence stride) mis-addresses under that packing — it must be
  rejected at validation time, not silently accepted and mis-read.
- Validate by stride, not by a `thd`/`packed` flag: token-major is
  `stride_h == 1 and stride_s == H`; head-major is `stride_s == 1` with
  `stride_h` the declared head stride, which must cover the **packed token
  count `T`** — `stride_h >= H` is not the bound; at `T > stride_h` the
  per-head slices alias. `T` is a runtime total, so plan-time
  classification can only check `stride_s == 1 and stride_h >= 1`. In the
  THD path the packed total is a *device* value — Rule 3 bans reading it
  back, so `stride_h >= T` is **caller contract** (stated in
  `_thd_lse_view`'s docstring), not something the adapter verifies:
  `as_strided` bounds-checks storage capacity, never overlap. Do not "fix"
  this with a host-side length read; an in-kernel assert is the only
  legal detector. See the THD classification sites in `fwd/api_dsl.py`
  (duplicated at the two THD compile-key call sites — extract rather than
  re-copy if you touch it, per Rule 3's "suspect duplicated logic first").
- Covered by `test_fwd_probe_rejects_invalid_stats_metadata` and the
  `stats_layout`-parametrized THD tests (`test_dsl_sm100_thd_stats` and
  siblings) in `test/python/sdpa/frost/`.
