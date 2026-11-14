# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Fixed FP8 body lowered through the common graph binding and launch code."""

from .contract import Geometry


def lower(spec, facts, knobs, *, lower_impl, knob_type):
    # The registry supplies common framework services; this implementation
    # never imports the registry that owns it.
    geometry = Geometry(cga=2 if knobs is None or knobs.cga is None else knobs.cga)
    return lower_impl(spec, facts, geometry.canonical_knobs(knobs or knob_type()), selected_kernel=geometry)
