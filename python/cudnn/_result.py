# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Framework-neutral result containers shared by frontend APIs."""

from __future__ import annotations

from typing import Any


class TupleDict(dict):
    """Dictionary result that also unpacks and indexes like a tuple.

    Values follow insertion order, matching the result order documented by
    the operation wrapper. String-key access remains ordinary dictionary
    access.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._keys = tuple(self.keys())

    def __iter__(self):
        return (dict.__getitem__(self, key) for key in self._keys)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            if key < 0 or key >= len(self._keys):
                raise IndexError(f"index {key} out of range for TupleDict with {len(self._keys)} items")
            key = self._keys[key]
        return dict.__getitem__(self, key)


__all__ = ["TupleDict"]
