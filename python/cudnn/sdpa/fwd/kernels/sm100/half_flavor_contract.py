# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Logical half widths and their existing covering geometry.

This is width legality, not main-body selection. Feature-limited D32 geometry
can still advance to D64/D128 through the shared capability walk. D576 retains
its exact-only boundary; it must not absorb an unsupported wider envelope.
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
