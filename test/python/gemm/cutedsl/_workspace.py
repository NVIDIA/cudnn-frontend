# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Caller-owned workspace for the direct grouped-GEMM ``execute()`` calls in these tests (recipe R2).

The APIs never allocate scratch; a test that calls ``api.execute(...)`` directly is a caller and
sizes the buffer from ``scratch_workspace_bytes()`` exactly as the ``*_wrapper_sm100`` functions do.
"""

import torch


def ws(api, device="cuda"):
    """``torch.empty(api.scratch_workspace_bytes(), uint8)`` on ``device``, or None when the API needs no scratch."""
    nbytes = api.scratch_workspace_bytes()
    return torch.empty(nbytes, dtype=torch.uint8, device=device) if nbytes else None
