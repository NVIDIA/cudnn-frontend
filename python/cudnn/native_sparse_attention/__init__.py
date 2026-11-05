# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# NSA is built on first access, so importing the kernel modules (as ``jax_api`` does) does not import torch.
# It imports every PyTorch API at once, so ``from cudnn import NSA`` still fails when a dependency is missing.
def __getattr__(name):
    if name != "NSA":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from . import compression, selection, sliding_window_attention, top_k

    class NSANamespace:
        SelectionAttention = staticmethod(selection.SelectionAttention)
        selection_attention_wrapper = staticmethod(selection.selection_attention_wrapper)

        SlidingWindowAttention = staticmethod(sliding_window_attention.SlidingWindowAttention)
        sliding_window_attention_wrapper = staticmethod(sliding_window_attention.sliding_window_attention_wrapper)

        CompressionAttention = staticmethod(compression.CompressionAttention)
        compression_attention_wrapper = staticmethod(compression.compression_attention_wrapper)

        TopKReduction = staticmethod(top_k.TopKReduction)
        topk_reduction_wrapper = staticmethod(top_k.topk_reduction_wrapper)

    globals()["NSA"] = NSANamespace()
    return globals()["NSA"]


__all__ = [
    "NSA",
]
