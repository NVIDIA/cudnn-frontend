# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make ``cudnn.linear_attention.frost`` importable: append the source ``python/cudnn`` dir
to the installed ``cudnn`` package's ``__path__`` (the wheel lacks the ``FROST``
subtree). Unnecessary once the engine ships in the built frontend package."""

from __future__ import annotations

import sys
from pathlib import Path

import cudnn

_SRC_CUDNN = Path(__file__).resolve().parents[4] / "python" / "cudnn"
if _SRC_CUDNN.is_dir() and str(_SRC_CUDNN) not in cudnn.__path__:
    cudnn.__path__.append(str(_SRC_CUDNN))

# The shared helpers live in the linear_attention test package root
# (test/python/linear_attention); pytest only prepends this file's own directory.
_TEST_PY = Path(__file__).resolve().parents[2]
if str(_TEST_PY) not in sys.path:
    sys.path.insert(0, str(_TEST_PY))
