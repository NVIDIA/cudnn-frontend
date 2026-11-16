# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed implementation entry using the shared graph binder and launch builder."""

from ..d512_f16_contract import TWO_BY_TWO, explicit_geometry


def lower(spec, facts, knobs):
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs, lower_dsl_prefill

    geometry = explicit_geometry(TWO_BY_TWO, facts)
    return lower_dsl_prefill(spec, facts, geometry.canonical_knobs(knobs or SdpaFwdKnobs()), selected_kernel=geometry)
