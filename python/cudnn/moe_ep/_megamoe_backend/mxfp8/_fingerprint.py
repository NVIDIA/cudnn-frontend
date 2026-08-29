# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Stable, machine-readable fingerprints for compiled MXFP8 kernels."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

FINGERPRINT_SCHEMA_VERSION = 1
_KERNEL_IDENTITY_FIELDS = (
    "kernel_name",
    "kernel_source",
    "effective_config",
    "cutlass_version",
    "source_tree_sha256",
    "source_git_revision",
    "launch_geometry",
)


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def kernel_identity_sha256(fingerprint: dict[str, Any]) -> str:
    """Hash fields that must match across AOT and in-process JIT paths."""

    return canonical_json_sha256({field: fingerprint.get(field) for field in _KERNEL_IDENTITY_FIELDS})


def source_tree_sha256(root: Path) -> str:
    """Hash Python source content and relative paths in deterministic order."""

    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        raise RuntimeError(f"kernel source tree does not exist: {resolved}")
    digest = hashlib.sha256()
    sources = sorted(path for path in resolved.rglob("*.py") if path.is_file())
    if not sources:
        raise RuntimeError(f"kernel source tree contains no Python files: {resolved}")
    for path in sources:
        relative = path.relative_to(resolved).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _cutlass_version() -> str:
    for distribution in (
        "nvidia-cutlass-dsl",
        "nvidia-cutlass-dsl-internal",
    ):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    import cutlass

    return str(getattr(cutlass, "__version__", "unknown"))


def _json_layout_signature(signature: tuple) -> list[object]:
    return [
        (
            None
            if entry is None
            else {
                "shape": list(entry[0]),
                "stride": list(entry[1]),
                "dtype": str(entry[2]),
            }
        )
        for entry in signature
    ]


def build_kernel_fingerprint(
    prepared: Any,
    layout_signature: tuple,
    *,
    compiled_binary_sha256: str | None = None,
) -> dict[str, Any]:
    """Describe exactly what was compiled and how the main kernel launches."""

    kernel = prepared.kernel
    source_root = Path(__file__).resolve().parents[1] / "cutedsl_src"
    effective_config = prepared.config.effective_config(prepared.launch_cluster_count)
    launch_geometry = {
        "grid": [
            prepared.config.cluster_shape_mnk[0],
            prepared.config.cluster_shape_mnk[1],
            prepared.launch_cluster_count,
        ],
        "block": [int(kernel.threads_per_cta), 1, 1],
        "cluster": list(prepared.config.cluster_shape_mnk),
        "min_blocks_per_mp": int(getattr(kernel, "occupancy", 1)),
        "dynamic_shared_memory_bytes": int(getattr(kernel, "smem_capacity", 0)),
    }
    layout = _json_layout_signature(layout_signature)
    fingerprint = {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "kernel_name": str(kernel.name()),
        "kernel_source": "vendored-training-mega",
        "effective_config": effective_config,
        "cutlass_version": _cutlass_version(),
        "source_tree_sha256": source_tree_sha256(source_root),
        "source_git_revision": os.environ.get("MOE_EP_SOURCE_GIT_REVISION"),
        "launch_geometry": launch_geometry,
        "layout_signature_sha256": canonical_json_sha256(layout),
        "compiled_binary_sha256": compiled_binary_sha256,
    }
    fingerprint["kernel_identity_sha256"] = kernel_identity_sha256(fingerprint)
    fingerprint["fingerprint_sha256"] = canonical_json_sha256(fingerprint)
    return fingerprint


__all__ = [
    "FINGERPRINT_SCHEMA_VERSION",
    "build_kernel_fingerprint",
    "canonical_json_sha256",
    "kernel_identity_sha256",
    "source_tree_sha256",
]
