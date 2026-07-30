# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch dependency boundary for cuDNN Frontend.

This module is the sole location in ``python/cudnn`` that performs a runtime
``import torch``. Every other module reaches PyTorch through :func:`require` or
probes for it with :func:`is_available`, so that importing ``cudnn`` and its
submodules never requires PyTorch to be installed.

The import is attempted lazily, on the first call to :func:`is_available` or
:func:`require` -- importing this module must not trigger it. That keeps
``import cudnn`` free of PyTorch's import cost and side effects for callers that
only use the framework-neutral ``cudnn.pygraph``/DLPack paths.
"""

from typing import Any, Optional

__all__ = ["TorchNotAvailableError", "is_available", "require"]


class TorchNotAvailableError(ImportError):
    """Raised when a PyTorch-dependent entry point is used without PyTorch.

    Subclasses :class:`ImportError` so existing ``except ImportError`` handlers
    keep working, while allowing callers to catch the missing-framework case
    specifically rather than catching unrelated runtime failures.
    """


_MESSAGE = (
    "{feature} requires PyTorch, but PyTorch is not installed. Install a "
    "compatible PyTorch distribution separately; nvidia-cudnn-frontend[cutedsl] "
    "does not install it."
)

# Sentinel distinguishing "import not yet attempted" from "attempted and failed"
# (which caches ``None``). Never leaks outside this module.
_UNPROBED = object()

_torch_module: Any = _UNPROBED
_import_error: Optional[BaseException] = None


def _probe() -> Any:
    """Attempt the ``import torch`` once and cache the outcome.

    Returns the module on success or ``None`` on failure. The failure is cached
    alongside, so a broken or absent PyTorch is never re-imported on subsequent
    calls.
    """
    global _torch_module, _import_error

    if _torch_module is _UNPROBED:
        try:
            import torch
        except Exception as e:
            # Catch broadly, not just ImportError: a PyTorch installation that is
            # present but unusable (e.g. mismatched CUDA runtime) raises OSError
            # and friends. Such a build is equally unavailable to us, and the
            # captured cause is chained onto the error raised by require().
            _torch_module = None
            _import_error = e
        else:
            _torch_module = torch
            _import_error = None

    return _torch_module


def is_available() -> bool:
    """Return whether PyTorch can be imported. Never raises."""
    return _probe() is not None


def require(feature: str) -> Any:
    """Return the imported ``torch`` module, or raise :class:`TorchNotAvailableError`.

    :param feature: Human-readable name of the entry point requiring PyTorch;
        used verbatim at the start of the error message.
    """
    torch = _probe()
    if torch is None:
        raise TorchNotAvailableError(_MESSAGE.format(feature=feature)) from _import_error
    return torch
