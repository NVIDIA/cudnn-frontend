# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private CuTe DSL kernels."""

from cudnn.frost.buffers import cutedsl_requirement_error

_error = cutedsl_requirement_error("simplified_sparse_attention", floor=(4, 8, 0))
if _error:
    raise ImportError(_error)
