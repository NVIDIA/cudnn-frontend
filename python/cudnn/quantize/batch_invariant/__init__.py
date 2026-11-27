# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch-invariant block-scale quantizers (MXFP8, MXFP4). Light imports only: the CuTeDSL kernels load on first call."""

from .api import mxfp4_quantize_forward, mxfp8_quantize_forward

__all__ = ["mxfp8_quantize_forward", "mxfp4_quantize_forward"]
