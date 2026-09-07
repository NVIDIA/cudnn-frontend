# SPDX-License-Identifier: BSD-3-Clause
"""Minimal fake-tensor mode detection used by compile-cache paths."""

from torch._guards import active_fake_mode


def is_fake_mode() -> bool:
    """Return whether PyTorch fake-tensor mode is currently active."""

    return active_fake_mode() is not None


__all__ = ["is_fake_mode"]
