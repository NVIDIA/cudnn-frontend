# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Per-tensor FP8 logical widths and their existing Rubin covering geometry."""


def padded_shape(facts):
    dq, dv = facts.d_qk, facts.d_v
    if dq <= 0 or dv <= 0 or dq % 16 or dv % 16:
        return None
    for shape in ((64, 64), (128, 128), (192, 128), (256, 256)):
        if dq <= shape[0] and dv <= shape[1]:
            return shape
    # The wide pipeline retains the existing floor on both logical axes.
    if 256 < dq <= 512 and 256 < dv <= 512:
        return (512, 512)
    return None
