# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX pytree registration for framework-neutral frontend results."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import jax

from ..common.result import TupleDict


def _flatten(value: TupleDict) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    keys = tuple(dict.keys(value))
    return tuple(dict.__getitem__(value, key) for key in keys), keys


def _flatten_with_keys(value: TupleDict):
    children, keys = _flatten(value)
    return tuple((jax.tree_util.DictKey(key), child) for key, child in zip(keys, children)), keys


def _unflatten(keys: tuple[Any, ...], children: Iterable[Any]) -> TupleDict:
    return TupleDict(zip(keys, children))


if not getattr(TupleDict, "_jax_pytree_registered", False):
    jax.tree_util.register_pytree_with_keys(
        TupleDict,
        _flatten_with_keys,
        _unflatten,
        _flatten,
    )
    TupleDict._jax_pytree_registered = True


__all__ = ["TupleDict"]
