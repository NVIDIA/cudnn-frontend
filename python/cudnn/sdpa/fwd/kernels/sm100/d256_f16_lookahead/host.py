# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit lookahead entry, sharing the existing graph binder/launch builder."""

from ..d256_f16_contract import LOOKAHEAD, explicit_geometry


def lower(spec, facts, knobs):
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs, lower_dsl_prefill

    geometry = explicit_geometry(LOOKAHEAD, knobs)
    return lower_dsl_prefill(spec, facts, geometry.canonical_knobs(knobs or SdpaFwdKnobs()), selected_kernel=geometry)
