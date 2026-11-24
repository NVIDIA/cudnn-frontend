# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only alias validation for caller-owned dGLU scratch.

``validate_workspace_aliases`` lives in ``..backend_utils`` with the other caller-scratch helpers of
the grouped GEMM backends; it is re-exported here so existing importers keep working.
"""

from ..backend_utils import validate_workspace_aliases

__all__ = ["validate_workspace_aliases"]
