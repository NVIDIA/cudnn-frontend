# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit lookahead entry, sharing the existing graph binder/launch builder."""

from ..d256_f16_contract import LOOKAHEAD, explicit_geometry

PACKED_HN_STATS_OVERRIDE = True


def lower(spec, facts, knobs, *, lower_impl, knob_type):
    geometry = explicit_geometry(LOOKAHEAD, knobs)
    return lower_impl(spec, facts, geometry.canonical_knobs(knobs or knob_type()), selected_kernel=geometry)
