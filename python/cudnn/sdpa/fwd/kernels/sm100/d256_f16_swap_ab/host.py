# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit swap-AB entry, sharing the existing graph binder/launch builder."""

from ..d256_f16_contract import SWAP_AB, explicit_geometry


def lower(spec, facts, knobs):
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs, lower_dsl_prefill

    geometry = explicit_geometry(SWAP_AB, knobs)
    return lower_dsl_prefill(spec, facts, geometry.canonical_knobs(knobs or SdpaFwdKnobs()), selected_kernel=geometry)
