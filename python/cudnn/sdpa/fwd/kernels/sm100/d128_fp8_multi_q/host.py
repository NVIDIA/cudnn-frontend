# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Fixed multi-Q entry using shared graph lowering services."""

from .contract import Geometry
from ..fp8_flavor_contract import padded_shape


def lower(spec, facts, knobs, *, lower_impl, knob_type):
    knobs = knobs or knob_type()
    width = padded_shape(facts)[0]
    cga = knobs.cga if knobs.cga is not None else 1 if width == 64 else 2
    geometry = Geometry(d_flavor=width, cga=cga)
    return lower_impl(spec, facts, geometry.canonical_knobs(knobs), selected_kernel=geometry)
