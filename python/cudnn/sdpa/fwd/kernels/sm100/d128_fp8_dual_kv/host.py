# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Fixed dual-KV entry using shared graph lowering services."""

from .contract import Geometry


def lower(spec, facts, knobs, *, lower_impl, knob_type):
    geometry = Geometry()
    return lower_impl(spec, facts, geometry.canonical_knobs(knobs or knob_type()), selected_kernel=geometry)
