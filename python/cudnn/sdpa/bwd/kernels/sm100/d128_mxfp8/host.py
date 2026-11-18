# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The D128 MXFP8 prep -> dK/dV -> persistent dQ chain lowered through the common MXFP8 graph binding."""


def lower(spec, facts, knobs, *, lower_impl):
    # The registry supplies common framework services; this implementation
    # never imports the registry that owns it.
    return lower_impl(spec, facts, knobs)
