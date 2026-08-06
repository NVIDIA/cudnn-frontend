# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make ``cudnn.gemm.frost`` importable: overlay the source ``python/cudnn``
dir onto the installed ``cudnn.__path__`` and the nested ``cudnn.gemm.__path__``.
Unnecessary once the engine ships in the built frontend package."""

from __future__ import annotations

from pathlib import Path

import pytest

import cudnn

_SRC_CUDNN = Path(__file__).resolve().parents[4] / "python" / "cudnn"
if _SRC_CUDNN.is_dir():
    if str(_SRC_CUDNN) not in cudnn.__path__:
        cudnn.__path__.append(str(_SRC_CUDNN))
    import cudnn.gemm

    _src_gemm = str(_SRC_CUDNN / "gemm")
    if _src_gemm not in cudnn.gemm.__path__:
        cudnn.gemm.__path__.append(_src_gemm)


@pytest.fixture(autouse=True)
def _frost_opt_in(monkeypatch):
    """The FROST manifest rows are opt-in; this suite exercises them.

    Per test rather than at import: a module-level ``os.environ[...] = "1"``
    runs during COLLECTION and would leave the flag set for every other test in
    the same pytest process, quietly turning a default-path run into an opt-in
    one."""
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
