# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Route assertions shared by the native binder migration tests."""

import pytest


def forbid_python_forward_binding(monkeypatch):
    from cudnn.sdpa.fwd import prepared

    def forbidden(*args, **kwargs):
        pytest.fail("native template entered the Python core binder")

    for name in ("bind_dense", "bind_dense_split", "execute_quantized", "_bind_block_output", "_bind_mxfp8_scales"):
        monkeypatch.setattr(prepared, name, forbidden)
