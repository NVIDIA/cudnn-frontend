# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manifold-constrained hyper-connection operations."""

from importlib import import_module

_LAZY_EXPORTS = {
    "workspace_size": ("cudnn.mhc.api", "workspace_size"),
    "pre_mix_forward": ("cudnn.mhc.api", "pre_mix_forward"),
    "pre_mix_backward": ("cudnn.mhc.api", "pre_mix_backward"),
    "post_mix_forward": ("cudnn.mhc.api", "post_mix_forward"),
    "post_mix_backward": ("cudnn.mhc.api", "post_mix_backward"),
    "head_forward": ("cudnn.mhc.api", "head_forward"),
    "head_backward": ("cudnn.mhc.api", "head_backward"),
    "block_forward_jax": ("cudnn.mhc.jax_api", "block_forward_jax"),
    "pre_mix_forward_jax": ("cudnn.mhc.jax_api", "pre_mix_forward_jax"),
    "pre_mix_backward_jax": ("cudnn.mhc.jax_api", "pre_mix_backward_jax"),
    "post_mix_forward_jax": ("cudnn.mhc.jax_api", "post_mix_forward_jax"),
    "post_mix_backward_jax": ("cudnn.mhc.jax_api", "post_mix_backward_jax"),
    "head_forward_jax": ("cudnn.mhc.jax_api", "head_forward_jax"),
    "head_backward_jax": ("cudnn.mhc.jax_api", "head_backward_jax"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _LAZY_EXPORTS[name]
    try:
        value = getattr(import_module(module), symbol)
    except ImportError as error:
        from cudnn import _optional_dependency_message

        raise ImportError(_optional_dependency_message(name, error)) from error
    globals()[name] = value
    return value
