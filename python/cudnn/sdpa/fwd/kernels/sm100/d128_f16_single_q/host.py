# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Fixed implementation lowered through common graph services."""

from .contract import Geometry, small_geometry


def lower(spec, facts, knobs, *, lower_impl, knob_type):
    knobs = knobs or knob_type()
    geometry = Geometry() if (facts.d_qk, facts.d_v) == (192, 128) else small_geometry(facts, knobs)
    return lower_impl(spec, facts, geometry.canonical_knobs(knobs), selected_kernel=geometry)
