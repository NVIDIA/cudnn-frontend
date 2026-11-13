# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Shared compiler and architecture admission for the D512 FP8 pair."""

import re


def environment_error(cc, version):
    if tuple(cc) not in ((10, 0), (10, 3), (10, 7)):
        return f"D512 FP8 backward requires SM100, SM103 or SM107; got {cc}"
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version[1]) if version and version[0] == "nvidia-cutlass-dsl" else None
    if match is None or tuple(map(int, match.groups())) < (4, 8, 0):
        return f"D512 FP8 backward requires nvidia-cutlass-dsl >= 4.8.0; found {version}"
    return None
