# SPDX-License-Identifier: Apache-2.0

"""Workspace layout helpers for NVFP4 QAT attention backward."""

from __future__ import annotations

from functools import reduce
from operator import mul

import torch

_WORKSPACE_ALIGNMENT = 16

WorkspaceEntry = tuple[int, tuple[int, ...], torch.dtype]


def _numel(shape: tuple[int, ...]) -> int:
    """Return the number of elements in ``shape``."""
    return reduce(mul, shape, 1)


def _align_up(value: int, alignment: int = _WORKSPACE_ALIGNMENT) -> int:
    """Round ``value`` up to the requested byte alignment."""
    return (value + alignment - 1) // alignment * alignment


def nvfp4_workspace_layout(
    q_shape: tuple[int, ...],
    k_shape: tuple[int, ...],
    v_shape: tuple[int, ...],
    lse_shape: tuple[int, ...],
) -> tuple[tuple[WorkspaceEntry, ...], int]:
    """Return aligned workspace entries and the total required byte count."""
    entries = []
    offset = 0
    for shape, dtype in (
        (q_shape, torch.bfloat16),
        (k_shape, torch.bfloat16),
        (v_shape, torch.bfloat16),
        (lse_shape, torch.float32),
    ):
        normalized_shape = tuple(shape)
        offset = _align_up(offset)
        entries.append((offset, normalized_shape, dtype))
        offset += _numel(normalized_shape) * dtype.itemsize
    return tuple(entries), _align_up(offset)
