# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The D256 MXFP8 chain lowered through the common quantized graph binding and launch code."""


def lower(spec, facts, knobs, *, lower_impl):
    # The registry supplies common framework services; this implementation
    # never imports the registry that owns it.
    return lower_impl(spec, facts, knobs)
