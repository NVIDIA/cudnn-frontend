# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Fixed FP8 body lowered through the common graph binding and launch code."""

from .contract import Geometry


def lower(spec, facts, knobs):
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs, lower_dsl_prefill

    geometry = Geometry()
    return lower_dsl_prefill(spec, facts, geometry.canonical_knobs(knobs or SdpaFwdKnobs()), selected_kernel=geometry)
