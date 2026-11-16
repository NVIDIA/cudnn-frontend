# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed SM80 half body lowered through the common graph binding and launch code."""

from ..half_contract import D64, select_implementation


def lower(spec, facts, knobs, *, lower_impl):
    # The registry supplies common framework services; this implementation
    # never imports the registry that owns it.
    if select_implementation(facts) != D64:
        raise ValueError(f"{D64} lowers plain dense MHA D64 graphs on whole tiles with packed Stats only")
    return lower_impl(spec, facts, knobs)
