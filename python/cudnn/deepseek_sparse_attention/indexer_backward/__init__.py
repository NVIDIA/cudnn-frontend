# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Lazy Torch API and framework-neutral indexer-backward exports."""

from ...common.operation_api import make_operation_api

__all__, __getattr__, __dir__ = make_operation_api(
    globals(),
    exports={
        "op": ("DenseIndexerBackwardOp", "IndexerBackwardOp"),
        "api": (
            "DenseIndexerBackward",
            "IndexerBackward",
            "dense_indexer_backward_wrapper",
            "indexer_backward_wrapper",
        ),
    },
    submodules=("api", "op"),
)
