# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Fixed multi-Q entry; shared services resolve its legal width envelope."""

PACKED_HN_STATS_OVERRIDE = True


def lower(spec, facts, knobs, *, lower_impl, knob_type, selected_shape, resolve_geometry):
    knobs = knobs or knob_type()
    flavor = selected_shape(spec.capabilities, facts, knobs)
    if flavor is None:
        raise ValueError("half multi-Q has no legal head-dimension envelope")
    geometry = resolve_geometry(facts, knobs, d_flavor=flavor[0])
    return lower_impl(spec, facts, geometry.canonical_knobs(knobs), selected_kernel=geometry)
