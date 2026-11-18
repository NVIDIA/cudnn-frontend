# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The dense Qwen 3.5-architecture family: ``QwenDense`` (geometry, layer order, block wiring) and its provider table;
``QwenDenseDrafter``, the loader of a checkpoint's drafter."""

from .drafter import QwenDenseDrafter
from .model import Geometry, QwenDense

__all__ = ["Geometry", "QwenDense", "QwenDenseDrafter"]
