# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scratch layout shared by adapters and prepared launches."""


def ws_align(nbytes: int) -> int:
    """Round a scratch-chunk size up to the carve alignment (128 B)."""
    return -(-int(nbytes) // 128) * 128
