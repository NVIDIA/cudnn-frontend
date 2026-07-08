# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Lazy Torch SDPA-forward API exports."""

from ..._operation_api import make_operation_api

__all__, __getattr__, __dir__ = make_operation_api(
    globals(),
    exports={"api": ("SdpafwdSm100D256", "sdpa_fwd_wrapper_sm100_d256")},
    submodules=("api", "jax"),
)
