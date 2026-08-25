# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuDNN drop-in acceleration for flash-linear-attention (FLA).

``accelerate_fla()`` monkeypatches FLA entry points cuDNN can serve and
transparently calls the original implementation for unsupported configurations.
The backward-compatible no-argument call enables the linear-attention targets
(``gated_delta_rule`` and ``kda``).  The dense MLP adapter is intentionally
opt-in because it has a narrower FLA 0.5.2/local-module contract::

    import cudnn.fla

    cudnn.fla.accelerate_fla()                    # GDN + KDA, as before
    cudnn.fla.accelerate_fla(targets="gated_mlp") # incremental MLP opt-in

Targets can be enabled and restored independently.  Every adapter is
fail-closed: an incompatible installed FLA target rejects explicit activation,
while a runtime configuration outside an activated adapter's validated contract
executes the exact original FLA callable.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import sys
from typing import Callable, Iterable

from .gated_delta_rule import make_chunk_gated_delta_rule, last_path
from .gated_mlp import last_path as mlp_last_path
from .gated_mlp import make_gated_mlp_forward
from .gated_mlp import _supports_installed_fla
from .kda import make_chunk_kda

__all__ = ["accelerate_fla", "is_accelerated", "last_path", "mlp_last_path", "restore_fla"]


@dataclass(frozen=True)
class _PatchSpec:
    module_path: str
    attribute: str
    make_replacement: Callable
    owner_attribute: str | None = None
    default: bool = True


@dataclass(frozen=True)
class _AppliedPatch:
    spec: _PatchSpec
    owner: object
    original: object
    replacement: object


def _function_replacement(factory):
    def make_replacement(module, owner, original):
        del module, owner
        return factory(original)

    return make_replacement


def _gated_mlp_replacement(module, owner, original):
    if not _supports_installed_fla():
        raise ImportError("the cuDNN GatedMLP shim requires flash-linear-attention==0.5.2")
    if owner.__module__ != module.__name__ or owner.__dict__.get("forward") is not original:
        raise ImportError("FLA GatedMLP.forward does not match the expected owning class")
    if original.__module__ != module.__name__ or original.__qualname__ != "GatedMLP.forward" or hasattr(original, "__wrapped__"):
        raise ImportError("FLA GatedMLP.forward was replaced before cuDNN acceleration")
    swiglu_linear_cls = getattr(module, "SwiGLULinear", None)
    if swiglu_linear_cls is None or swiglu_linear_cls.__module__ != module.__name__:
        raise ImportError("FLA GatedMLP does not expose the expected SwiGLULinear helper")
    return make_gated_mlp_forward(original, owner, swiglu_linear_cls)


_TARGETS = {
    "gated_delta_rule": _PatchSpec(
        "fla.ops.gated_delta_rule",
        "chunk_gated_delta_rule",
        _function_replacement(make_chunk_gated_delta_rule),
    ),
    "kda": _PatchSpec(
        "fla.ops.kda",
        "chunk_kda",
        _function_replacement(make_chunk_kda),
    ),
    "gated_mlp": _PatchSpec(
        "fla.modules.mlp",
        "forward",
        _gated_mlp_replacement,
        owner_attribute="GatedMLP",
        default=False,
    ),
}
_ALIASES = {"gdn": "gated_delta_rule", "mlp": "gated_mlp"}
_DEFAULT_TARGETS = tuple(name for name, spec in _TARGETS.items() if spec.default)
_ORIGINALS: dict[str, _AppliedPatch] = {}


def _canonical_target(target: str) -> str:
    if not isinstance(target, str):
        raise TypeError(f"FLA acceleration target must be a string, got {type(target).__name__}")
    target = _ALIASES.get(target, target)
    if target not in _TARGETS:
        choices = ", ".join(_TARGETS)
        raise ValueError(f"unknown FLA acceleration target {target!r}; expected one of: {choices}")
    return target


def _normalize_targets(targets: str | Iterable[str] | None, *, default: Iterable[str]) -> tuple[str, ...]:
    if targets is None:
        requested = tuple(default)
    elif isinstance(targets, str):
        requested = (targets,)
    else:
        requested = tuple(targets)
    if not requested:
        raise ValueError("at least one FLA acceleration target is required")
    canonical = {_canonical_target(target) for target in requested}
    # Registry order makes logging/restoration deterministic even if a set was
    # supplied by the caller.
    return tuple(target for target in _TARGETS if target in canonical)


def is_accelerated(target: str | None = None) -> bool:
    """Whether any target, or one named target, is currently patched."""
    if target is None:
        return any(getattr(applied.owner, applied.spec.attribute, None) is applied.replacement for applied in _ORIGINALS.values())
    applied = _ORIGINALS.get(_canonical_target(target))
    return applied is not None and getattr(applied.owner, applied.spec.attribute, None) is applied.replacement


def _drop_displaced_patch(target: str) -> None:
    """Forget a patch whose owner was replaced without clobbering the new owner."""
    applied = _ORIGINALS.pop(target, None)
    if applied is None:
        return
    if applied.spec.owner_attribute is None:
        _rebind_everywhere(applied.spec.attribute, applied.replacement, applied.original)


def _rebind_everywhere(fn_name: str, original, replacement) -> None:
    """Rebind imports that captured a patched module-level function by reference."""
    for module in list(sys.modules.values()):
        if module is None:
            continue
        try:
            if getattr(module, fn_name, None) is original:
                setattr(module, fn_name, replacement)
        except Exception:
            # Some modules raise on getattr of arbitrary names; skip them.
            continue


def accelerate_fla(verbose: bool = True, *, targets: str | Iterable[str] | None = None) -> None:
    """Patch selected FLA targets, incrementally and idempotently.

    ``targets=None`` preserves the original behavior and enables only GDN/KDA.
    Use ``targets="gated_mlp"`` (or the ``"mlp"`` alias) for the opt-in dense
    MLP adapter.  A string or iterable of strings is accepted.
    """
    requested = _normalize_targets(targets, default=_DEFAULT_TARGETS)
    available = {target for target in requested if is_accelerated(target)}
    resolved = []
    missing = []
    rejection_reasons = {}

    for target in requested:
        if is_accelerated(target):
            continue
        _drop_displaced_patch(target)
        spec = _TARGETS[target]
        try:
            module = importlib.import_module(spec.module_path)
        except ImportError as error:
            missing.append(target)
            rejection_reasons[target] = str(error) or f"cannot import {spec.module_path}"
            continue
        owner = getattr(module, spec.owner_attribute, None) if spec.owner_attribute is not None else module
        if owner is None:
            missing.append(target)
            rejection_reasons[target] = f"{spec.module_path} has no {spec.owner_attribute} owner"
            continue
        original = getattr(owner, spec.attribute, None)
        if original is None:
            missing.append(target)
            rejection_reasons[target] = f"the target owner has no {spec.attribute} attribute"
            continue
        try:
            replacement = spec.make_replacement(module, owner, original)
        except ImportError as error:
            missing.append(target)
            rejection_reasons[target] = str(error) or "the installed target does not match the supported contract"
            continue
        resolved.append((target, spec, owner, original, replacement))

    # An explicit selection is a contract: never silently apply only a subset.
    # The legacy no-target call retains its best-effort GDN/KDA behavior.
    if targets is not None and missing:
        details = ", ".join(f"{target} ({rejection_reasons[target]})" if target in rejection_reasons else target for target in missing)
        raise ImportError(f"accelerate_fla() could not enable FLA target(s): {details}")

    newly_patched = []
    for target, spec, owner, original, replacement in resolved:
        if spec.owner_attribute is None:
            _rebind_everywhere(spec.attribute, original, replacement)
        else:
            setattr(owner, spec.attribute, replacement)
        _ORIGINALS[target] = _AppliedPatch(spec, owner, original, replacement)
        newly_patched.append(target)
        available.add(target)

    if not available and not newly_patched:
        names = ", ".join(requested)
        raise ImportError(f"accelerate_fla() could not find supported FLA target(s): {names}")
    if verbose and newly_patched:
        print(f"[cudnn.fla] accelerated FLA {', '.join(newly_patched)} with cuDNN (SM100); " "unsupported configs fall back to FLA.")


def restore_fla(*, targets: str | Iterable[str] | None = None) -> None:
    """Undo all active patches, or only the selected targets."""
    if targets is None:
        requested = tuple(target for target in _TARGETS if target in _ORIGINALS)
    else:
        requested = _normalize_targets(targets, default=())
    for target in requested:
        applied = _ORIGINALS.pop(target, None)
        if applied is None:
            continue
        spec, owner, original, replacement = (
            applied.spec,
            applied.owner,
            applied.original,
            applied.replacement,
        )
        if spec.owner_attribute is None:
            _rebind_everywhere(spec.attribute, replacement, original)
        if getattr(owner, spec.attribute, None) is replacement:
            setattr(owner, spec.attribute, original)
