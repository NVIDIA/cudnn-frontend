# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Lazy Torch API and framework-neutral sparse-attention backward exports."""

from ..._operation_api import make_operation_api

__all__, __getattr__, __dir__ = make_operation_api(
    globals(),
    exports={
        "op": ("SparseAttentionBackwardOp",),
        "dsa_bwd_sm90": ("FlashAttentionDSABackwardSm90",),
        "dsa_bwd_sm100": ("FlashAttentionDSABackwardSm100",),
        "api": (
            "SparseAttentionBackward",
            "sparse_attention_backward_wrapper",
        ),
    },
    submodules=("api", "dsa_bwd_sm90", "dsa_bwd_sm100", "op"),
)
