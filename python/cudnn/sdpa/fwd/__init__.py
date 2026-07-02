# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Lazy API exports for d=256 SDPA forward."""

from ..._operation_api import make_operation_api

_API_EXPORTS = (
    "SdpafwdSm100D256",
    "sdpa_fwd_wrapper_sm100_d256",
)

__all__, __getattr__ = make_operation_api(
    globals(),
    exports=_API_EXPORTS,
)
