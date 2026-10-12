# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Overlay this checkout's Python modules onto a prebuilt frontend package."""

from pathlib import Path

import cudnn

_SOURCE_CUDNN = Path(__file__).resolve().parents[4] / "python" / "cudnn"
if str(_SOURCE_CUDNN) not in cudnn.__path__:
    cudnn.__path__.insert(0, str(_SOURCE_CUDNN))
