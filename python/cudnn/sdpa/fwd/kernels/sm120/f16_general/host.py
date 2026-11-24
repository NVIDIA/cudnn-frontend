# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed general half body lowered through the common graph binding and launch code."""

from ..half_exact_contract import explicit_geometry
from .contract import NAME

PACKED_HN_STATS_OVERRIDE = True


def lower(spec, facts, knobs, *, lower_impl, knob_type):
    # The registry supplies common framework services; this implementation
    # never imports the registry that owns it.
    geometry = explicit_geometry(NAME, knobs)
    return lower_impl(spec, facts, geometry.canonical_knobs(knobs or knob_type()), selected_kernel=geometry)
