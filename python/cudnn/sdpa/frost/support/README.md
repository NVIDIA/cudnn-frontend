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
| SM100 / SM103 (cc 10.0 / 10.3) | [SM100 / SM103 support](sm100.md) | Forward exact D32/D64/D128 half and exact D512 MXFP8 owners documented; remaining forward and backward pending |
| SM107 (cc 10.7) | [SM107 support](sm107.md) | Wide-head half and per-tensor FP8 forward documented; other forward domains and backward pending migration |
