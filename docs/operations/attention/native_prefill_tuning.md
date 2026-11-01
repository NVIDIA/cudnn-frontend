<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: MIT -->

# Explicit native prefill topology

SDPA forward plans use the shared `cudnn.knob_type` vocabulary. An explicit
`WARP_SPEC_CFG=1` selects a single full Q tile per CTA in the native SM107
D128 half-precision THD kernel. This is a performance option with the same
attention semantics. It is not selected automatically by this change.

```python
graph.create_execution_plan(
    engine_id,
    {
        cudnn.knob_type.WARP_SPEC_CFG: 1,
        cudnn.knob_type.TILE_CGA_M: 1,
        cudnn.knob_type.TILE_M: 128,
        cudnn.knob_type.TILE_N: 128,
    },
)
```

The option requires exact QK/V head dimensions of 128, fp16 or bf16 Q/K/V,
THD query storage, one CTA, no head packing, and no KV split. Both ragged
and paged K/V are supported. Bias, per-key masks, mixed Q/K/V dtypes, and
epilogue gates are not supported by
this topology. Requests outside this domain are rejected. Independent
physical origins, padded tensor strides, supported diagonal bands, natural
or LPT scheduling, and optional Stats use the existing graph contract.

`WARP_SPEC_CFG=0`, or omission of the knob, retains the existing topology.
The selected value is returned by `get_engine_and_knobs_at_index`; preserve
it when caching and recreating explicit plan choices. The standalone DSL
adapter exposes the same topology through its existing `softmax_wgs=1`
argument, with the same geometry restrictions.

For paged K/V, this option accepts pages that are multiples of 8 dividing 128,
or multiples of 128. K and V can independently use HND or NHD pools and
different block-table strides. Q, O and Stats retain their declared physical
origins; page tables address K/V independently. Packing and KV split remain
outside this explicit topology.
