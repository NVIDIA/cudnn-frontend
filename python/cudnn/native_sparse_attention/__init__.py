# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module

_SYMBOLS = {
    "SelectionAttention": ".selection",
    "selection_attention_wrapper": ".selection",
    "SlidingWindowAttention": ".sliding_window_attention",
    "sliding_window_attention_wrapper": ".sliding_window_attention",
    "CompressionAttention": ".compression",
    "compression_attention_wrapper": ".compression",
    "TopKReduction": ".top_k",
    "topk_reduction_wrapper": ".top_k",
}


# Resolved on first access so the kernel modules (used by ``jax_api``) import without torch.
class NSANamespace:
    def __getattr__(self, name):
        if name in _SYMBOLS:
            return getattr(import_module(_SYMBOLS[name], __name__), name)
        raise AttributeError(f"NSA has no attribute {name!r}")


NSA = NSANamespace()

__all__ = [
    "NSA",
]
