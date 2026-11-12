# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework-independent format and storage names for expert-parallel MoE."""

from enum import Enum


class MoeFormat(str, Enum):
    """Data formats supported by the MoE+EP interface."""

    BF16 = "bf16"
    MXFP8 = "mxfp8"
    NVFP4 = "nvfp4"


def parse_format(value: MoeFormat | str) -> MoeFormat:
    """Normalize a public format value."""
    if isinstance(value, MoeFormat):
        return value
    try:
        return MoeFormat(value.lower())
    except (AttributeError, ValueError) as exc:
        choices = ", ".join(item.value for item in MoeFormat)
        raise ValueError(f"unsupported format {value!r}; expected one of: {choices}") from exc


class MoeEpNativeWeightStorageMode(str, Enum):
    """Addressing mode used by kernel-native training weights."""

    CONTIGUOUS = "contiguous"
    DISCRETE = "discrete"


def parse_native_weight_storage_mode(value: MoeEpNativeWeightStorageMode | str) -> MoeEpNativeWeightStorageMode:
    """Normalize a native training-weight storage mode."""
    if isinstance(value, MoeEpNativeWeightStorageMode):
        return value
    try:
        return MoeEpNativeWeightStorageMode(value.lower())
    except (AttributeError, ValueError) as exc:
        choices = ", ".join(mode.value for mode in MoeEpNativeWeightStorageMode)
        raise ValueError(f"unsupported native weight storage mode {value!r}; expected one of: {choices}") from exc
