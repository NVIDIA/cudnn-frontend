# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Fixed swap-AB entry using shared graph lowering services."""

from .contract import Geometry
from ..fp8_flavor_contract import padded_shape


def lower(spec, facts, knobs, *, lower_impl, knob_type):
    geometry = Geometry(d_flavor=padded_shape(facts)[0])
    return lower_impl(spec, facts, geometry.canonical_knobs(knobs or knob_type()), selected_kernel=geometry)
