# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed D128 pipeline over the common graph binder and launch services."""

from .contract import Geometry


def lower(spec, facts, knobs, *, lower_impl, knob_type):
    # Preserve an omitted scheduler for the shared adapter's persistent/walk
    # defaults. Only geometry is fixed here; no implementation selection runs.
    geometry = Geometry()
    return lower_impl(spec, facts, geometry.canonical_knobs(knobs or knob_type()), selected_kernel=geometry)
