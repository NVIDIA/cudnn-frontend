# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cudnn.frost.buffers import cutedsl_requirement_error

_error = cutedsl_requirement_error("simplified_sparse_attention", floor=(4, 8, 0))
if _error:
    pytest.skip(_error, allow_module_level=True)
