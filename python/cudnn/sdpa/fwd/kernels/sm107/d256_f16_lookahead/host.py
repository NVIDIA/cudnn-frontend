# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Fixed native implementation using injected common graph lowering."""

from .contract import NAME
from ..mid_half_contract import explicit_geometry

PACKED_HN_STATS_OVERRIDE = True


def lower(spec, facts, knobs, *, lower_impl, knob_type):
    knobs = knobs or knob_type()
    geometry = explicit_geometry(NAME, facts, knobs)
    return lower_impl(spec, facts, geometry.canonical_knobs(knobs), selected_kernel=geometry)
