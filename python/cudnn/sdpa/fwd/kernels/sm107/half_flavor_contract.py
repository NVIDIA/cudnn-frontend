# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Existing Rubin half width envelopes, independently of body-selection policy.

The shared capability walk still promotes feature-limited small tiles. This
function only classifies logical widths; D576/V512 remains exact-only.
"""


def padded_shape(facts):
    dq, dv = facts.d_qk, facts.d_v
    if dq <= 0 or dv <= 0 or dq % 8 or dv % 8:
        return None
    if (dq, dv) == (576, 512):
        return (576, 512)
    for shape in ((32, 32), (64, 64), (128, 128), (192, 128), (256, 256), (512, 512)):
        if dq <= shape[0] and dv <= shape[1]:
            return shape
    return None
