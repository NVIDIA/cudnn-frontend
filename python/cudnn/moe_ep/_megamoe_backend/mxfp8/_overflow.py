# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared post-kernel overflow policy for MXFP8 inference and training."""

from __future__ import annotations

import torch


def apply_overflow_policy(
    overflow_flag: torch.Tensor,
    *,
    drop_on_overflow: bool,
    overflow_ok: torch.Tensor,
    message: str,
) -> None:
    """Drop silently or surface a completed kernel's global overflow bit."""

    if drop_on_overflow:
        return

    assert_async = getattr(torch, "_assert_async", None)
    if callable(assert_async):
        torch.eq(overflow_flag, 0, out=overflow_ok)
        assert_async(overflow_ok, message)
        return

    if torch.cuda.is_current_stream_capturing():
        raise NotImplementedError("CUDA graph capture requires torch._assert_async to surface " "Rubin MegaMoE overflow")

    # Preserve the existing eager-only compatibility behavior for PyTorch
    # builds without a device-side asynchronous assertion.
    value = int(overflow_flag.item())
    if value != 0:
        raise RuntimeError(f"{message} (overflow_flag={value})")


__all__ = ["apply_overflow_policy"]
