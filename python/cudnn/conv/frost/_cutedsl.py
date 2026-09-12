# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CuTeDSL requirements for the Frost convolution kernel variants."""

from cudnn.frost import buffers

BLOCK_SCALE_CUTEDSL_MIN_VERSION = (4, 8)
DENSE_CUTEDSL_MIN_VERSION = (4, 9)


def requirement_error(what: str, minimum_version: tuple[int, int]) -> str | None:
    """Return why a convolution variant cannot use the installed CuTeDSL."""
    installed, version = buffers.cutedsl_state()
    if not installed:
        return f"{what} requires the cutedsl extra (nvidia-cutlass-dsl), which is not installed"
    if not version:
        return None

    distribution, raw_version = version
    if distribution != "nvidia-cutlass-dsl":
        return None
    try:
        major_minor = tuple(int(part) for part in raw_version.split("+", 1)[0].split(".")[:2])
    except ValueError:
        return None
    if len(major_minor) != 2 or major_minor >= minimum_version:
        return None

    floor = ".".join(str(part) for part in minimum_version)
    return f"{what} requires nvidia-cutlass-dsl >= {floor}; found {raw_version}. " f"Upgrade with: pip install -U 'nvidia-cutlass-dsl>={floor}'"


__all__ = ["BLOCK_SCALE_CUTEDSL_MIN_VERSION", "DENSE_CUTEDSL_MIN_VERSION", "requirement_error"]
