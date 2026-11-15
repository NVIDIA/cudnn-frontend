# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed fused half body lowered through the common graph binding and launch code."""

from ..half_contract import FUSED, select_implementation


def lower(spec, facts, knobs, *, lower_impl):
    # The registry supplies common framework services; this implementation
    # never imports the registry that owns it.
    if select_implementation(facts) != FUSED:
        raise ValueError(f"{FUSED} lowers dense graphs with D_QK <= 256 only")
    return lower_impl(spec, facts, knobs)
