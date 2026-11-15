# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Lower a fixed work item through the injected shared adapter."""

from .contract import NAME
from ..small_half_contract import explicit_geometry


def lower(spec, facts, knobs, *, lower_impl, knob_type):
    knobs = knobs or knob_type()
    geometry = explicit_geometry(NAME, facts, knobs)
    return lower_impl(spec, facts, geometry.canonical_knobs(knobs), selected_kernel=geometry)
