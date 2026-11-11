# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The dense Qwen 3.5-architecture family: ``QwenDense`` (geometry, layer order, block wiring) and its provider table."""

from .model import Geometry, QwenDense

__all__ = ["Geometry", "QwenDense"]
