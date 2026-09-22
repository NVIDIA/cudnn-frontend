# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from .qdq import RopeQDQInplace, rope_qdq_inplace

__all__ = ["RopeQDQInplace", "rope_qdq_inplace", "TailRoPEForward", "tail_rope"]


def __getattr__(name):
    # Accessing the Triton QDQ API must not impose the tail kernel's DSL floor.
    if name in ("TailRoPEForward", "tail_rope"):
        from .tail import TailRoPEForward, tail_rope

        globals().update(TailRoPEForward=TailRoPEForward, tail_rope=tail_rope)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
