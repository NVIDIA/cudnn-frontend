# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Fixed main implementation using injected common lowering and binding services."""

from .contract import NAME
from ..d512_half_contract import explicit_geometry


def lower(spec, facts, knobs, *, lower_impl, knob_type):
    geometry = explicit_geometry(NAME, facts)
    return lower_impl(spec, facts, geometry.canonical_knobs(knobs or knob_type()), selected_kernel=geometry)
