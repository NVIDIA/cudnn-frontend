# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helper layer of the model samples: text in and out, weights, state, the provider plan, the decode loop's session,
token choice, timing and the tie-out.  Sample code: not installed and not part of the cudnn-frontend wheel.  A sample puts
``samples/models`` on ``sys.path`` and imports from here; every name below is imported on first use.

``check_environment()`` is the first call of a sample: it compares the installed distribution of cuDNN Frontend with the
``cudnn`` module that was imported and says whether that module is an editable checkout or a wheel.  A mismatch is an error:
an editable install whose metadata is stale lets any later ``pip install`` that requires cuDNN Frontend replace the checkout
with a released wheel, and the sample would then run other kernels than the tree it sits in.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

DISTRIBUTION = "nvidia-cudnn-frontend"

_LAZY = {
    "TextIO": (".text_io", "TextIO"),
    "TextStream": (".text_io", "TextStream"),
    "load_messages": (".text_io", "load_messages"),
    "Checkpoint": (".weights", "Checkpoint"),
    "ScaledTensor": (".weights", "ScaledTensor"),
    "BlockScaledTensor": (".weights", "BlockScaledTensor"),
    "merge_scaled": (".weights", "merge_scaled"),
    "dequantize": (".weights", "dequantize"),
    "ProviderRow": (".blocks", "ProviderRow"),
    "Block": (".blocks", "Block"),
    "DeviceFacts": (".blocks", "DeviceFacts"),
    "plan": (".blocks", "plan"),
    "Session": (".loop", "Session"),
    "StopRule": (".loop", "StopRule"),
    "Report": (".timing", "Report"),
    "Trace": (".timing", "Trace"),
    "GREEDY": (".sampling", "GREEDY"),
    "run_checks": (".tie_out", "run_checks"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(target[0], __name__), target[1])
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


__all__ = [*_LAZY, "check_environment", "Environment", "EnvironmentMismatch"]


class EnvironmentMismatch(RuntimeError):
    """The imported ``cudnn`` module and the installed distribution disagree."""


@dataclass(frozen=True)
class Environment:
    cudnn_version: str
    distribution_version: str
    editable: bool
    source_hash: str  # of the package files the sample's cuDNN Frontend provider runs; never a path
    torch_version: str
    dsl_version: str  # "" when the CuTe DSL is not installed

    def header(self) -> str:
        install = "editable checkout" if self.editable else "wheel"
        dsl = self.dsl_version or "not installed"
        return (
            f"cuDNN Frontend {self.cudnn_version} ({install}, source hash {self.source_hash}; distribution {self.distribution_version}) | "
            f"CuTe DSL {dsl} | PyTorch {self.torch_version}"
        )


def _probe(sources: Sequence[str]) -> dict:
    """The facts of this process: versions, whether ``cudnn`` is an editable checkout, and a hash of ``sources``."""
    import hashlib
    import importlib.metadata as metadata
    import json
    from pathlib import Path

    import cudnn  # before torch: the first cuDNN library loaded wins
    import torch

    package = Path(cudnn.__file__).resolve().parent
    digest = hashlib.sha256()
    for name in sources:
        path = package / name
        digest.update(name.encode() + b"\0" + (path.read_bytes() if path.is_file() else b"<missing>"))
    facts = {"cudnn_version": cudnn.__version__, "torch_version": torch.__version__, "source_hash": digest.hexdigest()[:12]}
    try:
        dist = metadata.distribution(DISTRIBUTION)
    except metadata.PackageNotFoundError:
        dist = None
    facts["distribution_version"] = None if dist is None else dist.version
    editable = False
    if dist is not None:
        direct = dist.read_text("direct_url.json")
        editable = bool(direct and json.loads(direct).get("dir_info", {}).get("editable"))
        try:
            editable = editable or not package.is_relative_to(Path(str(dist.locate_file(""))).resolve())
        except (OSError, ValueError):
            pass  # the distribution's location cannot be resolved: keep what direct_url.json said
    facts["editable"] = editable
    try:
        facts["dsl_version"] = metadata.version("nvidia-cutlass-dsl")
    except metadata.PackageNotFoundError:
        facts["dsl_version"] = ""
    return facts


def check_environment(sources: Optional[Sequence[str]] = None, probe: Optional[Callable[[Sequence[str]], dict]] = None) -> Environment:
    """Compare the installed distribution with the imported ``cudnn`` module; raise :class:`EnvironmentMismatch` naming both
    when they disagree or when no distribution is installed.  ``sources`` are package files to hash (default: the files of the
    fused linear-attention decode step); ``probe`` replaces the facts of this process (tests)."""
    if sources is None:
        from .providers.fe import SOURCE_FILES as sources
    facts = (probe or _probe)(sources)
    version, dist = facts["cudnn_version"], facts["distribution_version"]
    where = "an editable checkout" if facts["editable"] else "an installed wheel"
    if dist is None:
        raise EnvironmentMismatch(
            f'cudnn {version} was imported from a source tree, but no {DISTRIBUTION} distribution is installed: run "pip install -e ." in the checkout'
        )
    if dist != version:
        fix = 'run "pip install -e ." again' if facts["editable"] else f"reinstall {DISTRIBUTION}"
        raise EnvironmentMismatch(f"cudnn {version} was imported from {where}, but the installed distribution says {dist}: {fix}")
    return Environment(version, dist, bool(facts["editable"]), facts["source_hash"], facts["torch_version"], facts["dsl_version"])
